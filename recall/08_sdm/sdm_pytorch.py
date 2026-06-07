"""
SDM 召回模型 —— PyTorch 实现
（Session-based Deep Matching, Alibaba KDD 2019）

【这一节的产出】
1. 第一次显式建模「短期 session」vs「长期历史」
2. 实现 LSTM + Multi-Head Self-Attention + UserAttention 三层短期塔
3. 实现 Gated Fusion 门控融合（SDM 的灵魂）
4. 理解「同一用户向量空间」下的 Sampled Softmax 召回

【与 MIND 的关键区别】
┌────────────────┬──────────────────────────────┬───────────────────────────────┐
│      维度       │            MIND              │            SDM                │
├────────────────┼──────────────────────────────┼───────────────────────────────┤
│ 核心问题        │  兴趣「广度」（K 个并行兴趣）  │  兴趣「时效」（长 vs 短）      │
│ 序列输入        │  一条 history                │  short_history + long_history │
│ 序列聚合        │  Capsule Dynamic Routing     │  LSTM → MHA → UserAttention   │
│ 融合方式        │  Label-Aware Attn（训练塌缩） │  Gated Fusion（可解释门控）    │
│ 推理召回        │  K 路 ANN 合并               │  1 个用户向量 ANN              │
└────────────────┴──────────────────────────────┴───────────────────────────────┘

【任务拆解】
- TODO ①：UserAttention（用户画像当 Query 的注意力）
- TODO ②：GatedFusion（长短期门控融合）
- TODO ③：ShortTermInterest（LSTM + MHA + UserAttention）
- TODO ④：LongTermInterest（多特征 UserAttention + Dense）
- TODO ⑤：SDM 主模型 forward

运行：
    python sdm_pytorch.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict


# ============================================================
# 【0】配置
# ============================================================
DATA_PATH = "/Users/qiruihou/Desktop/学习/推荐算法/dataset/ml-latest-small/ratings.csv"
MOVIES_PATH = "/Users/qiruihou/Desktop/学习/推荐算法/dataset/ml-latest-small/movies.csv"

EMBEDDING_DIM = 32
SHORT_LEN = 5           # 短期 session 长度（FunRec ml-1m 配置 = 5）
LONG_LEN = 20           # 长期历史长度（FunRec 用 50，小数据缩短）
NUM_HEADS = 2           # 短期 Multi-Head Attention 头数
LR = 0.01
BATCH_SIZE = 256
N_EPOCHS = 15
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 【1】加载数据
# ============================================================
print("=" * 60)
print("【1】加载数据")
print("=" * 60)

ratings = pd.read_csv(DATA_PATH)
movies = pd.read_csv(MOVIES_PATH)
ratings = ratings[ratings['rating'] >= 3.0].copy()

movies['genres_list'] = movies['genres'].str.split('|')
all_genres = set()
for genres in movies['genres_list']:
    all_genres.update(genres)
genre2idx = {g: i + 1 for i, g in enumerate(sorted(all_genres))}  # 0 = PAD
n_genres = len(genre2idx) + 1

unique_users = ratings['userId'].unique()
unique_items = ratings['movieId'].unique()
user2idx = {u: i for i, u in enumerate(unique_users)}
item2idx = {m: i + 1 for i, m in enumerate(unique_items)}
idx2item = {i: m for m, i in item2idx.items()}

movie_genre = {}
for _, row in movies.iterrows():
    mid = item2idx.get(row['movieId'])
    if mid is None:
        continue
    glist = row['genres_list']
    movie_genre[mid] = genre2idx.get(glist[0], 0) if glist else 0

ratings['user_idx'] = ratings['userId'].map(user2idx)
ratings['item_idx'] = ratings['movieId'].map(item2idx)
n_users = len(user2idx)
n_items = len(item2idx) + 1

print(f"用户数: {n_users}, 物品数: {n_items}, 类型数: {n_genres}")


def pad_left(seq, max_len, pad_val=0):
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad_val] * (max_len - len(seq)) + seq


# ============================================================
# 【2】构造样本：short / long 双序列
# ============================================================
print("\n" + "=" * 60)
print("【2】构造 (user, short, long, long_genre, target) 样本")
print("=" * 60)

ratings_sorted = ratings.sort_values(['user_idx', 'timestamp']).reset_index(drop=True)
user_seqs = defaultdict(list)
for _, row in ratings_sorted.iterrows():
    user_seqs[row['user_idx']].append((int(row['item_idx']), int(row['timestamp'])))

samples = []
for u, seq in user_seqs.items():
    for t in range(1, len(seq)):
        target = seq[t][0]
        target_ts = seq[t][1]
        past = [item for item, _ in seq[:t]]
        short = pad_left(past, SHORT_LEN)
        long_items = pad_left(past, LONG_LEN)
        long_genres = pad_left([movie_genre.get(i, 0) for i in past], LONG_LEN)
        samples.append((u, short, long_items, long_genres, target, target_ts))

print(f"总样本数：{len(samples)}")
print(f"示例 short={samples[0][1]}, long 末5={samples[0][2][-5:]}")


class SDMDataset(Dataset):
    def __init__(self, samples):
        self.users = torch.LongTensor([s[0] for s in samples])
        self.short = torch.LongTensor([s[1] for s in samples])
        self.long_items = torch.LongTensor([s[2] for s in samples])
        self.long_genres = torch.LongTensor([s[3] for s in samples])
        self.targets = torch.LongTensor([s[4] for s in samples])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (self.users[idx], self.short[idx], self.long_items[idx],
                self.long_genres[idx], self.targets[idx])


samples_sorted = sorted(samples, key=lambda x: x[5])
split = int(len(samples_sorted) * 0.8)
train_loader = DataLoader(SDMDataset(samples_sorted[:split]), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(SDMDataset(samples_sorted[split:]), batch_size=BATCH_SIZE, shuffle=False)
print(f"训练集：{split}, 测试集：{len(samples_sorted) - split}")


# ============================================================
# 【3】TODO ①：UserAttention
# ============================================================
# 公式：α_k = softmax(h_k^T · e_u)，输出 = Σ α_k · h_k
#
# query:  [B, 1, D]  用户画像
# keys:   [B, L, D]  序列各位置（LSTM/MHA 输出 或 长期 embedding）
# return: [B, 1, D]
#
# 提示：
#   scores = query @ keys.transpose(1, 2)     → [B, 1, L]
#   weights = softmax(scores, dim=-1)
#   context = weights @ keys                  → [B, 1, D]

class UserAttention(nn.Module):
    def forward(self, query, keys, mask=None):
        """
        mask: [B, L] bool，True=有效位置。padding 位置应在 softmax 前置 -inf。
        """
        scores = query @ keys.transpose(1, 2)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        weights = F.softmax(scores, dim=-1)
        return weights @ keys


# ============================================================
# 【4】TODO ②：GatedFusion
# ============================================================
# G = σ(e_u·W1 + s_t·W2 + p_u·W3 + b)
# o = (1 - G) ⊙ p_u + G ⊙ s_t
#
# 三个输入都是 [B, 1, D]

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W1 = nn.Linear(dim, dim, bias=False)
        self.W2 = nn.Linear(dim, dim, bias=False)
        self.W3 = nn.Linear(dim, dim, bias=False)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, user_emb, short_term, long_term):
        # gate 由三者线性变换决定；融合仍用原始 s_t / p_u（与 FunRec 一致）
        gate = torch.sigmoid(
            self.W1(user_emb) + self.W2(short_term) + self.W3(long_term) + self.b
        )
        return (1 - gate) * long_term + gate * short_term


# ============================================================
# 【5】TODO ③：ShortTermInterest
# ============================================================
# 三层：LSTM → LayerNorm + MultiHeadAttention → UserAttention
#
# 输入 short_items: [B, SHORT_LEN]
# 输出 short_interest: [B, 1, D]

class ShortTermInterest(nn.Module):
    def __init__(self, n_items, dim, num_heads):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.user_attention = UserAttention()

    def forward(self, short_items, user_emb, item_embedding):
        mask = short_items != 0
        seq_emb = item_embedding(short_items) * mask.unsqueeze(-1).float()

        seq_out, _ = self.lstm(seq_emb)
        seq_out = self.norm1(seq_out)

        mha_out, _ = self.mha(
            seq_out, seq_out, seq_out,
            key_padding_mask=~mask,
        )
        seq_out = self.norm2(seq_out + mha_out)

        return self.user_attention(user_emb, seq_out, mask)

# ============================================================
# 【6】TODO ④：LongTermInterest
# ============================================================
# FunRec：对 movie_id / genres 等各做 UserAttention，concat 后 Dense
# 我们两路：long_items + long_genres

class LongTermInterest(nn.Module):
    def __init__(self, n_items, n_genres, dim):
        super().__init__()
        self.genre_emb = nn.Embedding(n_genres, dim, padding_idx=0)
        self.attn_item = UserAttention()
        self.attn_genre = UserAttention()
        self.fuse = nn.Linear(dim * 2, dim)

    def forward(self, long_items, long_genres, user_emb, item_embedding):
        item_emb = item_embedding(long_items)
        genre_emb = self.genre_emb(long_genres)
        item_mask = long_items != 0
        genre_mask = long_genres != 0

        item_out = self.attn_item(user_emb, item_emb, item_mask)
        genre_out = self.attn_genre(user_emb, genre_emb, genre_mask)
        return torch.tanh(self.fuse(torch.cat([item_out, genre_out], dim=-1)))


# ============================================================
# 【7】TODO ⑤：SDM 主模型
# ============================================================

class SDM(nn.Module):
    def __init__(self, n_users, n_items, n_genres, dim, num_heads):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.user_proj = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.item_embedding = nn.Embedding(n_items, dim, padding_idx=0)

        self.short_term = ShortTermInterest(n_items, dim, num_heads)
        self.long_term = LongTermInterest(n_items, n_genres, dim)
        self.gated_fusion = GatedFusion(dim)

    def encode_user(self, users, short_items, long_items, long_genres, return_gate=False):
        """推理时：得到最终用户向量 [B, D]；return_gate=True 时额外返回门控值"""
        user_e = self.user_proj(self.user_emb(users)).unsqueeze(1)
        s_t = self.short_term(short_items, user_e, self.item_embedding)
        p_u = self.long_term(long_items, long_genres, user_e, self.item_embedding)
        gate = torch.sigmoid(
            self.gated_fusion.W1(user_e)
            + self.gated_fusion.W2(s_t)
            + self.gated_fusion.W3(p_u)
            + self.gated_fusion.b
        )
        fused = self.gated_fusion(user_e, s_t, p_u).squeeze(1)
        user_vec = F.normalize(fused, p=2, dim=-1)
        if return_gate:
            return user_vec, gate.squeeze(1).mean(dim=-1)
        return user_vec

    def forward(self, users, short_items, long_items, long_genres, targets):
        user_vec = self.encode_user(users, short_items, long_items, long_genres)
        logits = user_vec @ self.item_embedding.weight.T
        return F.cross_entropy(logits, targets)


def build_user_features(user_seq, short_len=SHORT_LEN, long_len=LONG_LEN):
    """从用户完整历史构造 SDM 所需的 short / long 输入"""
    past = [item for item, _ in user_seq]
    short = pad_left(past, short_len)
    long_items = pad_left(past, long_len)
    long_genres = pad_left([movie_genre.get(i, 0) for i in past], long_len)
    return short, long_items, long_genres


# ============================================================
# 【8】训练
# ============================================================
print("\n" + "=" * 60)
print("【8】训练 SDM")
print("=" * 60)

model = SDM(
    n_users=n_users,
    n_items=n_items,
    n_genres=n_genres,
    dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
)
optimizer = optim.Adam(model.parameters(), lr=LR)

run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_sdm"
writer = SummaryWriter(log_dir=f"runs/{run_name}")
print(f"📊 TensorBoard 日志：runs/{run_name}")
print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

best_loss = float('inf')
patience = 3
patience_cnt = 0
best_state = None
best_epoch = 0

for epoch in range(N_EPOCHS):
    model.train()
    total_loss, n = 0.0, 0
    for users, short, long_items, long_genres, targets in train_loader:
        loss = model(users, short, long_items, long_genres, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    train_loss = total_loss / n

    model.eval()
    with torch.no_grad():
        total, m = 0.0, 0
        for users, short, long_items, long_genres, targets in test_loader:
            loss = model(users, short, long_items, long_genres, targets)
            total += loss.item()
            m += 1
        test_loss = total / m

    print(f"Epoch {epoch+1:2d}/{N_EPOCHS} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f}")
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)

    if test_loss < best_loss:
        best_loss = test_loss
        best_epoch = epoch + 1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= patience:
            print(f"\n⏸ Early stopping at epoch {epoch+1}（最佳：epoch {best_epoch}, loss={best_loss:.4f}）")
            break

if best_state:
    model.load_state_dict(best_state)
    torch.save(best_state, 'best_sdm.pt')
    print(f"✅ 已加载最佳模型（epoch {best_epoch}）")


# ============================================================
# 【9】召回示例
# ============================================================
print("\n" + "=" * 60)
print("【9】SDM 召回示例")
print("=" * 60)

FINAL_TOPK = 10

model.eval()
with torch.no_grad():
    all_item_embs = F.normalize(model.item_embedding.weight, p=2, dim=-1)

    for user_idx in range(3):
        u_seq = user_seqs.get(user_idx, [])
        if len(u_seq) < 2:
            continue

        short, long_items, long_genres = build_user_features(u_seq)
        seen = {item for item, _ in u_seq}

        users_t = torch.LongTensor([user_idx])
        short_t = torch.LongTensor([short])
        long_items_t = torch.LongTensor([long_items])
        long_genres_t = torch.LongTensor([long_genres])

        user_vec, gate_mean = model.encode_user(
            users_t, short_t, long_items_t, long_genres_t, return_gate=True
        )
        scores = (user_vec @ all_item_embs.T).squeeze(0)
        scores = scores.clone()
        scores[0] = -1e9
        for s in seen:
            scores[s] = -1e9

        top_k = torch.topk(scores, FINAL_TOPK).indices.tolist()

        print(f"\n用户 {user_idx} | gate均值={gate_mean.item():.3f}（越接近1越偏短期）")
        print(f"  最近短期序列: {short}")
        print(f"  Top-{FINAL_TOPK} 推荐：")
        for rank, idx in enumerate(top_k):
            item_id = idx2item.get(idx)
            if item_id is None:
                continue
            row = movies[movies['movieId'] == item_id]
            if len(row) == 0:
                continue
            title = row['title'].iloc[0]
            print(f"  {rank+1:2d}. {title:<60s} 分数={scores[idx]:.4f}")

writer.close()
