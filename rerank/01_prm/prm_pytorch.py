"""
PRM 个性化重排模型 —— PyTorch 实现
（Personalized Re-Ranking Model, Alibaba RecSys 2019）

【这一节的产出】
1. 你将亲手实现重排序阶段的经典模型——用 Transformer 建模"物品间相互影响"
2. 第一次理解"个性化向量 PV"——用预训练 CTR 模型的隐层输出做个性化信号
3. 学会 Listwise 建模思想——loss 从 pointwise 变成列表级 softmax
4. 理解"排序位置"对用户行为的影响（位置编码 PE）

【与精排模型的关键区别】
┌──────────────────┬──────────────────────────┬───────────────────────────────────┐
│      维度         │     精排（DeepFM等）      │           重排（PRM）              │
├──────────────────┼──────────────────────────┼───────────────────────────────────┤
│ 输入粒度          │  单个 (user, item)        │  一整个列表 [item1, ..., itemN]    │
│ 物品间交互        │  无（独立打分）            │  Transformer 自注意力建模          │
│ loss 级别         │  pointwise               │  listwise（列表级 softmax）        │
│ 位置信息          │  不考虑                   │  位置编码（位置影响用户行为）       │
│ 上游信号          │  原始特征                 │  精排模型隐层 PV（个性化向量）      │
└──────────────────┴──────────────────────────┴───────────────────────────────────┘

【核心建模思路】
   输入层：
     E_j = [item_feature(x_j) ; personalization_vector(pv_j)] + position_embedding(pe_j)

   编码层（Transformer Encoder × N_layers）：
     Multi-Head Self-Attention → 捕捉列表中物品间相互影响
     Feed-Forward Network → 非线性变换

   输出层：
     logit_j = W · F_j + b    （线性映射到标量）
     P = softmax(logits)      （列表级概率分布）

【任务拆解】
- TODO ①：实现 Positional Encoding（位置编码）
- TODO ②：实现 Multi-Head Self-Attention
- TODO ③：实现 Transformer Encoder Block（含残差 + LayerNorm + FFN）
- TODO ④：实现 PRM 主模型 forward（输入层 + 编码层 + 输出层）
- TODO ⑤：实现训练循环中的 listwise loss 计算

运行：
    python prm_pytorch.py
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
import math


# ============================================================
# 【0】配置
# ============================================================
DATA_PATH = "/Users/qiruihou/Desktop/学习/推荐算法/dataset/ml-latest-small/ratings.csv"
MOVIES_PATH = "/Users/qiruihou/Desktop/学习/推荐算法/dataset/ml-latest-small/movies.csv"

LIST_SIZE = 20            # 重排列表长度（一次重排多少个物品）
D_MODEL = 64             # Transformer 隐层维度
N_HEADS = 4              # 多头注意力头数
N_LAYERS = 2             # Transformer 堆叠层数
D_FF = 128               # FFN 中间层维度
DROPOUT = 0.1            # Dropout 率
PV_DIM = 32              # 个性化向量维度（模拟预训练模型输出）
ITEM_FEAT_DIM = 32       # 物品特征维度

LR = 0.001
BATCH_SIZE = 128
N_EPOCHS = 15
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 【1】加载数据 + 构造重排训练样本
# ============================================================
# 重排的训练数据和召回/精排不同：
#   - 输入：一个候选列表（LIST_SIZE 个物品）
#   - 标签：列表中哪些物品被用户点击了（0/1 向量）
#
# 构造方法：
#   1. 对每个用户，把他评过分的物品按时间排序
#   2. rating >= 4 视为"点击"（正样本），< 4 视为"曝光未点击"（负样本）
#   3. 用滑动窗口从用户的曝光列表中取 LIST_SIZE 个物品作为一个训练样本
#   4. 标签就是这 LIST_SIZE 个物品中哪些被"点击"了

print("=" * 60)
print("【1】加载数据，构造重排训练样本")
print("=" * 60)

ratings = pd.read_csv(DATA_PATH)
movies = pd.read_csv(MOVIES_PATH)

# ID 映射
unique_users = ratings['userId'].unique()
unique_items = ratings['movieId'].unique()
user2idx = {u: i for i, u in enumerate(unique_users)}
item2idx = {m: i for i, m in enumerate(unique_items)}
idx2item = {i: m for m, i in item2idx.items()}
n_users = len(user2idx)
n_items = len(item2idx)

ratings['user_idx'] = ratings['userId'].map(user2idx)
ratings['item_idx'] = ratings['movieId'].map(item2idx)
ratings['clicked'] = (ratings['rating'] >= 4.0).astype(int)

print(f"用户数: {n_users}, 物品数: {n_items}")
print(f"正样本（rating>=4）占比: {ratings['clicked'].mean():.2%}")

# 物品类型 multi-hot（作为物品特征）
movies['genres_list'] = movies['genres'].str.split('|')
all_genres = set()
for genres in movies['genres_list']:
    all_genres.update(genres)
genre2idx = {g: i for i, g in enumerate(sorted(all_genres))}
n_genres = len(genre2idx)

def genres_to_multihot(genres):
    vec = np.zeros(n_genres, dtype=np.float32)
    for g in genres:
        if g in genre2idx:
            vec[genre2idx[g]] = 1.0
    return vec

movies['genre_multihot'] = movies['genres_list'].apply(genres_to_multihot)

# 构建物品特征表（genre multi-hot）
item_genre_table = np.zeros((n_items, n_genres), dtype=np.float32)
for item_id, idx in item2idx.items():
    row = movies[movies['movieId'] == item_id]
    if len(row) > 0:
        item_genre_table[idx] = row.iloc[0]['genre_multihot']

print(f"电影类型数: {n_genres}")


# ============================================================
# 【2】构造重排样本 (item_list, click_labels, position_in_original_rank)
# ============================================================
print("\n" + "=" * 60)
print("【2】构造重排训练样本")
print("=" * 60)

# 按用户和时间排序，模拟用户看到的候选列表
ratings_sorted = ratings.sort_values(['user_idx', 'timestamp']).reset_index(drop=True)

samples = []  # [(item_indices_list, click_labels_list, user_idx), ...]

for u_idx in range(n_users):
    user_df = ratings_sorted[ratings_sorted['user_idx'] == u_idx]
    if len(user_df) < LIST_SIZE:
        continue
    item_indices = user_df['item_idx'].values
    click_labels = user_df['clicked'].values

    # 滑动窗口取样本
    for start in range(0, len(item_indices) - LIST_SIZE + 1, LIST_SIZE // 2):
        end = start + LIST_SIZE
        if end > len(item_indices):
            break
        items_in_list = item_indices[start:end]
        labels_in_list = click_labels[start:end]

        # 至少有 1 个正样本和 1 个负样本才有意义
        if labels_in_list.sum() == 0 or labels_in_list.sum() == LIST_SIZE:
            continue
        samples.append((items_in_list.tolist(), labels_in_list.tolist(), u_idx))

print(f"总重排样本数: {len(samples)}")
print(f"每个样本: {LIST_SIZE} 个物品的列表 + 对应点击标签")


# ============================================================
# 【3】Dataset + DataLoader
# ============================================================
class RerankDataset(Dataset):
    def __init__(self, samples):
        self.items = torch.LongTensor([s[0] for s in samples])      # [N, LIST_SIZE]
        self.labels = torch.FloatTensor([s[1] for s in samples])    # [N, LIST_SIZE]
        self.users = torch.LongTensor([s[2] for s in samples])      # [N]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.items[idx], self.labels[idx], self.users[idx]


# 时序分割
split = int(len(samples) * 0.8)
train_samples = samples[:split]
test_samples = samples[split:]

train_loader = DataLoader(RerankDataset(train_samples),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(RerankDataset(test_samples),
                         batch_size=BATCH_SIZE, shuffle=False)
print(f"训练集: {len(train_samples)}, 测试集: {len(test_samples)}")


# ============================================================
# 【4】TODO ①：Positional Encoding（位置编码）
# ============================================================
# 重排中位置编码的意义：
#   - 用户是否点击物品，受到它在列表中位置的影响（越靠前越容易被点击）
#   - 位置编码告诉 Transformer "这个物品在列表的第几个位置"
#
# 两种方案（这里用可学习的 Learned PE，更简单也够用）：
#   - Sinusoidal PE（固定公式，不可训练）
#   - Learned PE（nn.Embedding，可训练）← 我们用这个
#
# 输入：position_ids [B, LIST_SIZE]（值为 0, 1, 2, ..., LIST_SIZE-1）
# 输出：[B, LIST_SIZE, D_MODEL]
#
# 提示：
#   self.pos_embedding = nn.Embedding(max_len, d_model)
#   forward:
#       positions = torch.arange(seq_len, device=x.device)  # [LIST_SIZE]
#       return self.pos_embedding(positions)                 # [LIST_SIZE, D_MODEL]

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, seq_len, device):
        """
        返回: [seq_len, d_model] 的位置编码
        """
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        pass
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【5】TODO ②：Multi-Head Self-Attention
# ============================================================
# 这是 Transformer 的核心组件。在重排场景中，
# 自注意力让"列表中的每个物品都能看到其他所有物品"，
# 从而建模物品间的相互影响（互补、替代、同类冗余等）。
#
# 公式：
#   Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
#
# Multi-Head 版本：
#   1. Q/K/V 各做线性变换 → 切分成 n_heads 个头
#   2. 每个头独立做 Attention
#   3. 拼接所有头 → 线性变换
#
# 输入/输出：[B, LIST_SIZE, D_MODEL]
#
# 提示：
#   self.W_q = nn.Linear(d_model, d_model)
#   self.W_k = nn.Linear(d_model, d_model)
#   self.W_v = nn.Linear(d_model, d_model)
#   self.W_o = nn.Linear(d_model, d_model)
#
#   forward:
#       B, L, D = x.shape
#       d_k = D // n_heads
#       Q = self.W_q(x).view(B, L, n_heads, d_k).transpose(1, 2)  # [B, H, L, d_k]
#       K = self.W_k(x).view(B, L, n_heads, d_k).transpose(1, 2)
#       V = self.W_v(x).view(B, L, n_heads, d_k).transpose(1, 2)
#       attn = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)         # [B, H, L, L]
#       attn = F.softmax(attn, dim=-1)
#       attn = self.dropout(attn)
#       out = (attn @ V).transpose(1, 2).contiguous().view(B, L, D)
#       return self.W_o(out)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, x):
        """
        x: [B, L, D]
        return: [B, L, D]
        """
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        pass
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【6】TODO ③：Transformer Encoder Block
# ============================================================
# 标准结构：
#   1. Self-Attention + Residual + LayerNorm
#   2. FFN (Linear → ReLU → Linear) + Residual + LayerNorm
#
# 在重排中，堆叠多层 Block 可以建模更高阶的物品间依赖关系：
#   - 第 1 层：直接的两两影响（A 和 B 是替代品）
#   - 第 2 层：间接影响（A 通过 B 影响 C）
#
# 提示：
#   self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
#   self.norm1 = nn.LayerNorm(d_model)
#   self.norm2 = nn.LayerNorm(d_model)
#   self.ffn = nn.Sequential(
#       nn.Linear(d_model, d_ff),
#       nn.ReLU(),
#       nn.Dropout(dropout),
#       nn.Linear(d_ff, d_model),
#       nn.Dropout(dropout)
#   )
#
#   forward:
#       # Sub-layer 1: Self-Attention + Residual + LayerNorm
#       attn_out = self.self_attn(x)
#       x = self.norm1(x + attn_out)
#       # Sub-layer 2: FFN + Residual + LayerNorm
#       ffn_out = self.ffn(x)
#       x = self.norm2(x + ffn_out)
#       return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, x):
        """
        x: [B, L, D]
        return: [B, L, D]
        """
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        pass
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【7】TODO ④：PRM 主模型
# ============================================================
# PRM 的完整 pipeline：
#
#   [物品特征 x_j] + [个性化向量 pv_j] → Linear → [D_MODEL]
#                                             ↓
#                                     + Position Embedding
#                                             ↓
#                                   Transformer Encoder × N
#                                             ↓
#                                     Linear → 标量 score
#                                             ↓
#                                         Softmax
#                                             ↓
#                                    列表级概率 P(y_j)
#
# 关于"个性化向量 PV"的模拟：
#   工业上：用预训练好的 CTR 模型，提取它最后一层隐层的输出作为 PV
#   我们简化：用 user_embedding 和 item_embedding 的拼接过一个小 MLP 来模拟 PV
#   （本质相同：都是编码"这个用户对这个物品的偏好程度"的向量）
#
# 提示：
#   self.item_embedding = nn.Embedding(n_items, item_feat_dim)
#   self.user_embedding = nn.Embedding(n_users, pv_dim)
#   self.pv_net = nn.Sequential(
#       nn.Linear(item_feat_dim + pv_dim, pv_dim),
#       nn.ReLU()
#   )   # 模拟预训练 CTR 模型生成个性化向量
#   self.input_proj = nn.Linear(item_feat_dim + pv_dim, d_model)
#   self.pos_encoding = PositionalEncoding(list_size, d_model)
#   self.encoder_layers = nn.ModuleList([
#       TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
#       for _ in range(n_layers)
#   ])
#   self.output_layer = nn.Linear(d_model, 1)
#
#   forward(item_ids, user_ids):
#       # 1. 物品特征
#       item_emb = self.item_embedding(item_ids)            # [B, L, item_feat_dim]
#       # 2. 个性化向量 PV
#       user_emb = self.user_embedding(user_ids)            # [B, pv_dim]
#       user_emb = user_emb.unsqueeze(1).expand(-1, L, -1) # [B, L, pv_dim]
#       pv = self.pv_net(torch.cat([item_emb, user_emb], dim=-1))  # [B, L, pv_dim]
#       # 3. 拼接 + 投影
#       x = self.input_proj(torch.cat([item_emb, pv], dim=-1))     # [B, L, d_model]
#       # 4. 加位置编码
#       x = x + self.pos_encoding(L, x.device)
#       # 5. Transformer 编码
#       for layer in self.encoder_layers:
#           x = layer(x)
#       # 6. 输出打分
#       logits = self.output_layer(x).squeeze(-1)           # [B, L]
#       return logits

class PRM(nn.Module):
    def __init__(self, n_users, n_items, item_feat_dim, pv_dim,
                 d_model, n_heads, n_layers, d_ff, list_size, dropout=0.1):
        super().__init__()
        self.list_size = list_size

        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, item_ids, user_ids):
        """
        item_ids: [B, LIST_SIZE]   列表中物品的 id
        user_ids: [B]              用户 id
        return:   [B, LIST_SIZE]   每个物品的 logit（未经 softmax）
        """
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        pass
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【8】TODO ⑤：训练（Listwise Loss）
# ============================================================
# PRM 的 loss 和精排不同：
#   - 精排（pointwise）：每个物品独立算 BCE loss
#   - PRM（listwise）：把整个列表看成一个"分类问题"
#
# Listwise Softmax Loss：
#   P_pred = softmax(logits)             # 模型预测的列表概率分布
#   P_true = labels / labels.sum()       # 真实标签归一化为概率分布
#   loss = -Σ P_true * log(P_pred)       # KL 散度（= cross_entropy when P_true is one-hot）
#
# 但因为标签可能有多个正样本（non-one-hot），我们用 soft cross-entropy：
#   loss = -(labels_normalized * log_softmax(logits)).sum(dim=-1).mean()
#
# 也可以简化为 pointwise BCE：
#   loss = F.binary_cross_entropy_with_logits(logits, labels)
#
# 这里我们实现 listwise 版本（和论文一致）。
#
# 提示：
#   logits: [B, LIST_SIZE]
#   labels: [B, LIST_SIZE]  (0/1)
#
#   # 归一化标签为概率分布
#   label_probs = labels / (labels.sum(dim=-1, keepdim=True) + 1e-9)
#   # 计算 log_softmax
#   log_probs = F.log_softmax(logits, dim=-1)
#   # soft cross-entropy
#   loss = -(label_probs * log_probs).sum(dim=-1).mean()

print("\n" + "=" * 60)
print("【8】训练 PRM")
print("=" * 60)

model = PRM(
    n_users=n_users,
    n_items=n_items,
    item_feat_dim=ITEM_FEAT_DIM,
    pv_dim=PV_DIM,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    list_size=LIST_SIZE,
    dropout=DROPOUT,
)

optimizer = optim.Adam(model.parameters(), lr=LR)
run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_prm"
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
    for item_ids, labels, user_ids in train_loader:
        logits = model(item_ids, user_ids)   # [B, LIST_SIZE]

        # ↓↓↓↓↓ 你的代码：计算 listwise loss ↓↓↓↓↓
        loss = None   # TODO：实现 listwise softmax loss
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    train_loss = total_loss / n

    model.eval()
    with torch.no_grad():
        total, m = 0.0, 0
        for item_ids, labels, user_ids in test_loader:
            logits = model(item_ids, user_ids)

            # 同样的 loss 计算
            # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
            loss = None   # TODO：同上
            # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

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
    torch.save(best_state, 'best_prm.pt')
    print(f"✅ 已加载最佳模型（epoch {best_epoch}）")


# ============================================================
# 【9】评估：重排前 vs 重排后的 NDCG
# ============================================================
# 重排的效果用"排序质量"衡量：重排之后，点击物品是否被排到更前面了？
# 指标：NDCG@K（Normalized Discounted Cumulative Gain）

print("\n" + "=" * 60)
print("【9】评估：重排前 vs 重排后的 NDCG")
print("=" * 60)


def calc_ndcg(ranked_labels, k=5):
    """计算单个列表的 NDCG@K"""
    # DCG
    dcg = 0.0
    for i in range(min(k, len(ranked_labels))):
        dcg += ranked_labels[i] / math.log2(i + 2)
    # IDCG（理想排序）
    ideal = sorted(ranked_labels, reverse=True)
    idcg = 0.0
    for i in range(min(k, len(ideal))):
        idcg += ideal[i] / math.log2(i + 2)
    return dcg / (idcg + 1e-9)


model.eval()
ndcg_before_list = []
ndcg_after_list = []

with torch.no_grad():
    for item_ids, labels, user_ids in test_loader:
        logits = model(item_ids, user_ids)    # [B, LIST_SIZE]
        B = logits.size(0)

        for b in range(B):
            # 原始顺序的 NDCG（就是输入顺序）
            original_labels = labels[b].tolist()
            ndcg_before = calc_ndcg(original_labels, k=5)
            ndcg_before_list.append(ndcg_before)

            # 重排后的 NDCG（按模型 logits 降序排列）
            scores = logits[b].tolist()
            # 按分数排序，看排序后的标签顺序
            sorted_pairs = sorted(zip(scores, original_labels), reverse=True)
            reranked_labels = [lab for _, lab in sorted_pairs]
            ndcg_after = calc_ndcg(reranked_labels, k=5)
            ndcg_after_list.append(ndcg_after)

avg_ndcg_before = np.mean(ndcg_before_list)
avg_ndcg_after = np.mean(ndcg_after_list)
print(f"\n重排前平均 NDCG@5: {avg_ndcg_before:.4f}")
print(f"重排后平均 NDCG@5: {avg_ndcg_after:.4f}")
print(f"提升: {(avg_ndcg_after - avg_ndcg_before) / (avg_ndcg_before + 1e-9) * 100:.2f}%")

writer.close()


# ============================================================
# 【10】可视化：Attention 权重（看物品间的"互相影响"）
# ============================================================
print("\n" + "=" * 60)
print("【10】可视化示例：查看物品间的注意力关系")
print("=" * 60)
print("（完成 TODO ② 后，可以在这里 hook 出 attention weights 来可视化）")
print("Tips: 注意力权重矩阵 [LIST_SIZE, LIST_SIZE] 中，")
print("      entry (i, j) 表示物品 i 在决策时关注了物品 j 的程度。")
print("      如果 i 和 j 是同类别电影，权重通常偏高（互相影响）。")


# ============================================================
# 🤔 思考题（写完代码再答）
# ============================================================
"""
1. PRM 的 listwise loss 和 pointwise BCE loss 有什么区别？
   如果列表中只有 1 个正样本，两者退化为什么？
   （提示：1 个正样本时 listwise softmax loss 就是标准 cross_entropy）

2. 位置编码用 Learned PE 还是 Sinusoidal PE 更好？
   在什么场景下 Sinusoidal 更合适？
   （提示：Learned PE 适合固定长度；Sinusoidal 可以泛化到训练中没见过的长度）

3. PRM 的输入中，"个性化向量 PV" 和 "物品特征 X" 是分开编码再拼接的。
   如果把它们一起过同一个 DNN 再送入 Transformer，效果会更好还是更差？
   （提示：分开编码保留了"这是用户偏好 vs 这是物品属性"的语义边界）

4. 多头注意力中，不同的头学到了什么？
   能否设计实验验证：某些头关注"同类别物品"，某些头关注"位置接近的物品"？
   （提示：可以统计注意力权重和物品类别/位置的相关性）

5. PRM 假设列表长度固定为 LIST_SIZE。工业中列表长度不固定怎么办？
   （提示：padding + attention mask；或者用 padding 后的 mask 屏蔽无效位置）

6. PRS 论文认为 PRM 忽略了"排列顺序的影响"。你怎么理解这个差异？
   PRM 的位置编码是否已经部分解决了这个问题？
   （提示：PRM 的 PE 告诉模型"物品在第几位"，但没有建模"如果换个顺序会怎样"）
"""
