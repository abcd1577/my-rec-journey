"""
YouTubeDNN 召回模型 —— PyTorch 实现

【这一节的产出】
1. 你将亲手实现召回算法的"祖师爷"——YouTubeDNN（2016 年 RecSys）
2. 第一次实现"用户历史行为序列建模"（Average Pooling）
3. 学会 Sampled Softmax / 全库 Softmax 的训练方式
4. 学会 Embedding 共享技巧（历史 Embedding ↔ 候选 Embedding）

【与 DSSM 的关键区别】
- DSSM 是双塔（用户塔 + 物品塔），把 (user, item) 当作"二分类"问题
- YouTubeDNN 是单塔（只有用户塔，物品向量直接来自 Embedding 表）
  把"下一个看哪部电影"建模成对全库 N 部电影的"多分类"问题
  
【与 Item2Vec 的区别】
- Item2Vec：每个样本是 (中心物品, 上下文物品)，没有用户概念
- YouTubeDNN：每个样本是 (user_id, history_seq, target_item)
  历史序列 → 用户兴趣表征 → 预测下一个物品

【核心建模思路（论文精髓）】
   把"用户下一秒看哪个视频"建模为对全库 V 个视频的多分类
   
        P(w_t = i | U, C) = exp(v_i · u) / Σ_j exp(v_j · u)
                              ↑               ↑
                          (正样本得分)  (全库所有视频得分之和)

【任务拆解】
- TODO ①：构造 (user, history_seq, target) 三元组数据集
- TODO ②：UserTower（含历史序列 Average Pooling）
- TODO ③：YouTubeDNN 主模型（共享 Embedding + Softmax）
- TODO ④：训练 + 召回示例

运行：
    python youtubednn_pytorch.py
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

EMBEDDING_DIM = 32        # 用户 / 物品 Embedding 维度（共享！）
HISTORY_LEN = 20          # 用户最多保留最近 N 个历史观看（论文里是 50，小数据缩短）
TOWER_HIDDEN = [128, 64]  # 用户塔 DNN 隐层
OUTPUT_DIM = 32           # 用户向量 / 物品向量最终维度（必须等于 EMBEDDING_DIM，因为共享！）
LR = 0.001
BATCH_SIZE = 256
N_EPOCHS = 15
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 【1】加载数据
# ============================================================
print("=" * 60)
print("【1】加载数据，构造用户/物品特征")
print("=" * 60)

ratings = pd.read_csv(DATA_PATH)
movies = pd.read_csv(MOVIES_PATH)

# 只保留高分（≥4）作为正样本（与 DSSM 保持一致，方便对比）
ratings = ratings[ratings['rating'] >= 3.0].copy()

# === 物品类型特征 ===
movies['genres_list'] = movies['genres'].str.split('|')
all_genres = set()
for genres in movies['genres_list']:
    all_genres.update(genres)
genre2idx = {g: i for i, g in enumerate(sorted(all_genres))}
n_genres = len(genre2idx)

# === ID 重映射 ===
# 注意：物品索引从 1 开始（0 留给 "padding"，因为历史序列不足时要补 0）
unique_users = ratings['userId'].unique()
unique_items = ratings['movieId'].unique()
user2idx = {u: i for i, u in enumerate(unique_users)}
item2idx = {m: i + 1 for i, m in enumerate(unique_items)}   # ← 从 1 开始！0 = PAD
idx2item = {i: m for m, i in item2idx.items()}

ratings['user_idx'] = ratings['userId'].map(user2idx)
ratings['item_idx'] = ratings['movieId'].map(item2idx)
n_users = len(user2idx)
n_items = len(item2idx) + 1   # +1 是因为 0 索引被 padding 占用

print(f"用户数: {n_users}, 物品数（含PAD）: {n_items}")
print(f"💡 0 号索引保留给 padding（历史不足 {HISTORY_LEN} 个时填充）")


# ============================================================
# 【2】TODO ①：构造 (user, history_seq, target) 三元组
# ============================================================
# 这是 YouTubeDNN 与 DSSM 最大的不同！
#
# DSSM 的样本：(user_id, item_id) → 一对正样本
# YouTubeDNN 的样本：(user_id, [历史 N 个物品], target_item) → 序列 + 目标
#
# 构造逻辑（按时间排序）：
#   假设用户 Alice 按时间顺序看了 [v1, v2, v3, v4, v5, v6, v7]
#   生成的样本（滑窗）：
#     (user, history=[PAD,PAD,...,v1,v2,v3], target=v4)   ← 用前 3 个预测第 4 个
#     (user, history=[PAD,...,v1,v2,v3,v4],  target=v5)
#     (user, history=[PAD,...,v2,v3,v4,v5],  target=v6)
#     (user, history=[v3,v4,v5,v6],          target=v7)
#
# ⚠️ 重点：history 永远不能包含 target 之后的物品！（防止时间穿越）
#
# 提示：
#   1. 按 timestamp 排序每个用户的观看记录
#   2. 对每个用户，从第 2 次观看开始，把"之前看过的最近 HISTORY_LEN 个"作为 history
#   3. 不足 HISTORY_LEN 的用 0 填充（注意要左 padding，让真实物品贴在末尾）

print("\n" + "=" * 60)
print("【2】构造 (user, history_seq, target) 三元组")
print("=" * 60)

# 按 (user, timestamp) 排序
ratings_sorted = ratings.sort_values(['user_idx', 'timestamp']).reset_index(drop=True)

# 收集每个用户的观看序列
user_seqs = defaultdict(list)
for _, row in ratings_sorted.iterrows():
    user_seqs[row['user_idx']].append((int(row['item_idx']), int(row['timestamp'])))

# 生成 (user, history, target, timestamp) 样本
samples = []
for u, seq in user_seqs.items():
    # seq 已经按时间排好序
    for t in range(1, len(seq)):
        target = seq[t][0]
        target_ts = seq[t][1]
        # 取 target 之前的最近 HISTORY_LEN 个
        history = [item for item, _ in seq[max(0, t - HISTORY_LEN):t]]
        # 左侧补 0（PAD）让长度恰好 = HISTORY_LEN
        if len(history) < HISTORY_LEN:
            history = [0] * (HISTORY_LEN - len(history)) + history
        samples.append((u, history, target, target_ts))

print(f"总样本数：{len(samples)}")
print(f"\n前 3 个样本示例：")
for i in range(3):
    u, h, t, ts = samples[i]
    print(f"  用户 {u} | history={h[-5:]}... | target={t}")


# ============================================================
# 【3】Dataset
# ============================================================
class YoutubeDNNDataset(Dataset):
    def __init__(self, samples):
        self.users = torch.LongTensor([s[0] for s in samples])
        self.histories = torch.LongTensor([s[1] for s in samples])
        self.targets = torch.LongTensor([s[2] for s in samples])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.histories[idx], self.targets[idx]


# 按时间分训练/测试（用最后 20% 时间的样本作为测试集）
samples_sorted = sorted(samples, key=lambda x: x[3])
split = int(len(samples_sorted) * 0.8)
train_samples = samples_sorted[:split]
test_samples = samples_sorted[split:]

train_loader = DataLoader(YoutubeDNNDataset(train_samples),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(YoutubeDNNDataset(test_samples),
                         batch_size=BATCH_SIZE, shuffle=False)
print(f"\n训练集：{len(train_samples)}, 测试集：{len(test_samples)}")


# ============================================================
# 【4】TODO ②：UserTower（带历史序列建模）
# ============================================================
# 这是 YouTubeDNN 的灵魂！相比 DSSM 的 UserTower，多了一个"历史序列"输入。
#
# 输入：
#   user_idx [B]           ：用户 ID
#   history  [B, HIST_LEN] ：用户最近 N 个观看物品的 ID（含 padding=0）
# 输出：
#   user_emb [B, OUTPUT_DIM] ：用户向量（已 L2 归一化）
#
# 提示（__init__）：
#   - self.user_emb_layer 已经给你创建好了（user_id → D 维向量）
#   - self.item_embedding 是从外部传进来的"共享 Embedding 表"，不要再造一个！
#   - 你只需要构造 self.dnn，结构参考 DSSM：Linear → ReLU → Linear → ReLU → Linear
#   - DNN 输入维度 = user_id_emb (D) + history_pooled (D) = 2 * embedding_dim
#
# 提示（forward）：
#   - user_id 走 self.user_emb_layer 得到 [B, D]
#   - history 走 self.item_embedding 得到 [B, HIST_LEN, D]
#   - Masked Average Pooling（核心难点，照下面公式抄即可）：
#       mask = (history != 0).float().unsqueeze(-1)             # [B, HIST_LEN, 1]
#       pooled = (history_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
#     ↑ 除以"真实物品数量"才是真平均；clamp(min=1) 防止全是 PAD 时除以 0
#   - torch.cat([user_emb, pooled], dim=1) 拼起来过 DNN
#   - 最后 F.normalize(x, p=2, dim=1) 做 L2 归一化

class UserTower(nn.Module):
    def __init__(self, n_users, embedding_dim, output_dim, item_embedding):
        """
        item_embedding: nn.Embedding(n_items, embedding_dim)
                        ← 从外部传进来，与候选物品 Embedding 共享！
        """
        super().__init__()
        self.user_emb_layer = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = item_embedding   # 引用，不复制（共享 Embedding）

        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.dnn = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim)
        )
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, user_idx, history):
        """
        user_idx: [B]
        history:  [B, HIST_LEN]
        return:   [B, OUTPUT_DIM]，已 L2 归一化
        """
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        x = self.user_emb_layer(user_idx)
        history_emb = self.item_embedding(history)
        mask = (history != 0).float().unsqueeze(-1)
        pooled = (history_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        x = torch.cat([x, pooled], dim=1)
        x = self.dnn(x)
        x =  F.normalize(x, p=2, dim=1)
        return x 
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【5】TODO ③：YouTubeDNN 主模型（共享 Embedding + 全库 Softmax）
# ============================================================
# 与 DSSM 最大差异：
#   - DSSM：用户塔 + 物品塔，两个塔独立编码
#   - YouTubeDNN：只有用户塔，物品向量 = item_embedding 表直接查询
#
# 提示（forward）：
#   - 用户塔输出 user_emb，shape [B, D]
#   - 物品塔不存在！直接拿 self.item_embedding.weight 当"分类器权重"，shape [n_items, D]
#   - logits = user_emb @ item_embedding.weight.T   →  [B, n_items]
#   - loss = F.cross_entropy(logits, targets)
#
# 这就是论文里的 "softmax over V"！
# 教学版能直接全库 Softmax 是因为 n_items 只有 6298（算得动）
# 工业级亿级物品要换成 Sampled Softmax（见文末参考实现）

class YoutubeDNN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, output_dim):
        super().__init__()
        # 物品 Embedding（共享给用户塔的历史 + 最终 Softmax）
        # padding_idx=0 让 0 号索引的向量恒为 0，不参与梯度更新
        self.item_embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)

        # 用户塔（注意把 item_embedding 传进去共享）
        self.user_tower = UserTower(n_users, embedding_dim, output_dim,
                                    self.item_embedding)

        # 注意：output_dim 必须等于 embedding_dim
        # 因为 user_emb 要和 item_embedding 表做矩阵乘法（维度必须一致）
        assert output_dim == embedding_dim, \
            "OUTPUT_DIM 必须等于 EMBEDDING_DIM，因为 user_emb 要和 item_embedding 算相似度"

    def forward(self, users, histories, targets):
        """
        users:     [B]
        histories: [B, HIST_LEN]
        targets:   [B]            ← 真实"下一个看的物品"索引
        return:    loss (标量)
        """
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        x = self.user_tower(users, histories)
        logits = x @ self.item_embedding.weight.T
        loss = F.cross_entropy(logits, targets)
        return loss
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【6】训练（你不用改）
# ============================================================
print("\n" + "=" * 60)
print("【6】训练 YouTubeDNN")
print("=" * 60)

model = YoutubeDNN(n_users, n_items, EMBEDDING_DIM, OUTPUT_DIM)
optimizer = optim.Adam(model.parameters(), lr=LR)

run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_youtubednn_K{OUTPUT_DIM}"
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
    for users, histories, targets in train_loader:
        loss = model(users, histories, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    train_loss = total_loss / n

    model.eval()
    with torch.no_grad():
        total, m = 0.0, 0
        for users, histories, targets in test_loader:
            loss = model(users, histories, targets)
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
    torch.save(best_state, 'best_youtubednn.pt')
    print(f"✅ 已加载最佳模型（epoch {best_epoch}）")


# ============================================================
# 【7】召回示例（给前 3 个用户推荐）
# ============================================================
print("\n" + "=" * 60)
print("【7】YouTubeDNN 召回示例")
print("=" * 60)

model.eval()
with torch.no_grad():
    # 工业上线流程模拟：
    #   1. 离线：直接导出物品 Embedding 表（不需要跑物品塔！）
    all_item_embs = model.item_embedding.weight   # [n_items, D]

    # 给前 3 个用户分别推荐
    for user_idx in range(3):
        # 找该用户的最近 HISTORY_LEN 次观看（用整个数据集，模拟"线上"信息）
        u_seq = user_seqs.get(user_idx, [])
        if len(u_seq) < 2:
            continue
        history = [item for item, _ in u_seq[-HISTORY_LEN:]]
        if len(history) < HISTORY_LEN:
            history = [0] * (HISTORY_LEN - len(history)) + history

        # 已经看过的物品（避免重复推荐）
        seen = set([item for item, _ in u_seq])

        user_t = torch.LongTensor([user_idx])
        history_t = torch.LongTensor([history])
        user_emb = model.user_tower(user_t, history_t)  # [1, D]

        # 算这个用户和所有物品的相似度
        scores = (user_emb @ all_item_embs.T).squeeze(0).numpy()
        # 屏蔽已看过的 + PAD
        scores[0] = -1e9
        for s in seen:
            scores[s] = -1e9
        top10 = scores.argsort()[::-1][:10]

        print(f"\n用户 {user_idx} 的 Top-10 推荐：")
        for rank, idx in enumerate(top10):
            item_id = idx2item.get(idx, None)
            if item_id is None:
                continue
            row = movies[movies['movieId'] == item_id]
            if len(row) == 0:
                continue
            title = row['title'].iloc[0]
            print(f"  {rank+1:2d}. {title:<60s} 分数={scores[idx]:.4f}")

writer.close()


# ============================================================
# 🤔 思考题
# ============================================================
"""
1. 为什么 YouTubeDNN 的 Loss 是 cross_entropy（多分类），而 DSSM 是 in-batch softmax（二分类视角）？
   两者本质有什么数学联系？
   （提示：In-batch 其实是 batch 内 B 分类，YouTubeDNN 是 全库 N 分类）

2. 为什么物品 Embedding 要"共享"给历史序列和最终 Softmax？
   （提示：让历史看过的视频和待预测视频处于同一个语义空间）

3. 为什么用 Average Pooling 而不是 RNN/LSTM？
   论文原话："Surprisingly, simple averaging worked best."（出乎意料地，简单平均效果最好）
   你能想到哪些可能的原因？

4. 工业级亿级物品时全库 Softmax 算不动，怎么改成 Sampled Softmax？
   （提示：torch.nn.functional 没有现成的 sampled_softmax，
    通常自己实现：每个正样本配 K 个采样负样本 → cross_entropy 在 K+1 个类别上算）

5. 教学版 padding_idx=0 起到了什么作用？
   如果不设 padding_idx，会发生什么？
   （提示：PAD 的 Embedding 会被反向传播误更新，污染整个表）
"""


# ============================================================
# 📚 工业级 Sampled Softmax 参考实现（仅供阅读）
# ============================================================
"""
# 核心思想：每次只采样 K 个负样本，而不是全库
def sampled_softmax_loss(user_emb, target_idx, item_embedding, num_negatives=100):
    B = user_emb.size(0)
    n_items = item_embedding.weight.size(0)
    
    # 正样本得分
    pos_emb = item_embedding(target_idx)         # [B, D]
    pos_score = (user_emb * pos_emb).sum(dim=1, keepdim=True)  # [B, 1]
    
    # 随机采样负样本
    neg_idx = torch.randint(1, n_items, (B, num_negatives))    # [B, K]，避开 PAD
    neg_emb = item_embedding(neg_idx)            # [B, K, D]
    neg_score = (user_emb.unsqueeze(1) * neg_emb).sum(dim=2)   # [B, K]
    
    # 拼接 [pos, neg1, neg2, ...] → [B, 1+K]
    logits = torch.cat([pos_score, neg_score], dim=1)
    labels = torch.zeros(B, dtype=torch.long)    # 正样本永远在第 0 位
    return F.cross_entropy(logits, labels)
"""
