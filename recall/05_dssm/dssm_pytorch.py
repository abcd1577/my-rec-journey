"""
DSSM 双塔召回模型 —— PyTorch 实现

【这一节的产出】
1. 你将亲手实现工业界最主流的召回模型
2. 第一次接触"In-batch negatives"（batch 内负采样）
3. 学会 L2 归一化 + 温度系数

【与 Item2Vec 的区别】
- Item2Vec：每个样本是 (中心物品, 上下文物品)，没有用户概念
- DSSM：每个样本是 (user_features, item_features)，**用户和物品分别建塔**

【与 FunkSVD 的区别】
- FunkSVD：用户/物品各 1 个 embedding，内积预测评分
- DSSM：用户/物品各**多个特征 → DNN → 向量**，余弦预测匹配概率

【任务拆解】
- TODO ①：构造特征：用户的电影类型偏好、物品的类型 onehot
- TODO ②：UserTower（用户塔）
- TODO ③：ItemTower（物品塔）
- TODO ④：DSSM 整体（带 L2 归一化 + 温度系数 + in-batch loss）

运行：
    python dssm_pytorch.py
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


# ============================================================
# 【0】配置
# ============================================================
DATA_PATH = "/Users/qiruihou/Desktop/学习/推荐算法/dataset/ml-latest-small/ratings.csv"
MOVIES_PATH = "/Users/qiruihou/Desktop/学习/推荐算法/dataset/ml-latest-small/movies.csv"

EMBEDDING_DIM = 32        # ID embedding 维度
TOWER_HIDDEN = [128, 64]  # 塔的隐层
OUTPUT_DIM = 32           # 最终向量维度
TEMPERATURE = 0.1         # 温度系数 τ
LR = 0.001
BATCH_SIZE = 512
N_EPOCHS = 10
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 【1】加载数据 + 准备特征
# ============================================================
print("=" * 60)
print("【1】加载数据，构造用户/物品特征")
print("=" * 60)

ratings = pd.read_csv(DATA_PATH)
movies = pd.read_csv(MOVIES_PATH)

# 只保留高分（≥4）作为正样本
ratings = ratings[ratings['rating'] >= 4.0].copy()

# === 物品特征：电影类型（多值类别特征）===
# 把 "Action|Adventure|Sci-Fi" 拆成 list
movies['genres_list'] = movies['genres'].str.split('|')
all_genres = set()
for genres in movies['genres_list']:
    all_genres.update(genres)
genre2idx = {g: i for i, g in enumerate(sorted(all_genres))}
n_genres = len(genre2idx)
print(f"电影类型数：{n_genres}（{list(genre2idx.keys())[:5]}...）")

# 把每部电影的类型转成 multi-hot 向量
def genres_to_multihot(genres):
    vec = np.zeros(n_genres, dtype=np.float32)
    for g in genres:
        if g in genre2idx:
            vec[genre2idx[g]] = 1.0
    return vec

movies['genre_multihot'] = movies['genres_list'].apply(genres_to_multihot)

# === ID 重映射 ===
unique_users = ratings['userId'].unique()
unique_items = ratings['movieId'].unique()
user2idx = {u: i for i, u in enumerate(unique_users)}
item2idx = {m: i for i, m in enumerate(unique_items)}
idx2item = {i: m for m, i in item2idx.items()}

ratings['user_idx'] = ratings['userId'].map(user2idx)
ratings['item_idx'] = ratings['movieId'].map(item2idx)
n_users = len(user2idx)
n_items = len(item2idx)

# === 物品的类型 multi-hot 表 ===
item_genre_table = np.zeros((n_items, n_genres), dtype=np.float32)
for item_id, idx in item2idx.items():
    row = movies[movies['movieId'] == item_id]
    if len(row) > 0:
        item_genre_table[idx] = row.iloc[0]['genre_multihot']
item_genre_tensor = torch.FloatTensor(item_genre_table)

print(f"\n用户数: {n_users}, 物品数: {n_items}, 类型数: {n_genres}")
print(f"item_genre_table.shape: {item_genre_table.shape}")
print(f"前 3 部电影的类型：")
for i in range(3):
    item_id = idx2item[i]
    title = movies[movies['movieId'] == item_id]['title'].iloc[0]
    genres = movies[movies['movieId'] == item_id]['genres'].iloc[0]
    print(f"  [{i}] {title}  类型：{genres}")


# ============================================================
# 【2】Dataset
# ============================================================
# 每个样本：(user_idx, item_idx)
# 注意：DSSM 训练只需要"正样本"，负样本由 in-batch 自动产生（看下面 loss 计算）
class DSSMDataset(Dataset):
    def __init__(self, df):
        self.users = torch.LongTensor(df['user_idx'].values)
        self.items = torch.LongTensor(df['item_idx'].values)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]


# 简单按时间分训练/测试
ratings_sorted = ratings.sort_values('timestamp')
split = int(len(ratings_sorted) * 0.8)
train_df = ratings_sorted.iloc[:split].reset_index(drop=True)
test_df = ratings_sorted.iloc[split:].reset_index(drop=True)

train_loader = DataLoader(DSSMDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(DSSMDataset(test_df), batch_size=BATCH_SIZE, shuffle=False)
print(f"\n训练集: {len(train_df)}, 测试集: {len(test_df)}")


# ============================================================
# 【3】TODO ②：UserTower
# ============================================================
# 设计：
#   输入：user_idx [B]
#   输出：user_emb [B, OUTPUT_DIM]（已 L2 归一化）
#
# 步骤：
#   1. user_id → embedding [B, EMBEDDING_DIM]
#   2. 过 DNN（多层 Linear + ReLU）
#   3. L2 归一化
#
# 提示：
#   self.user_emb_layer = nn.Embedding(n_users, EMBEDDING_DIM)
#   self.dnn = nn.Sequential(
#       nn.Linear(EMBEDDING_DIM, 128), nn.ReLU(),
#       nn.Linear(128, 64), nn.ReLU(),
#       nn.Linear(64, OUTPUT_DIM)
#   )
#   forward:
#       x = self.user_emb_layer(user_idx)
#       x = self.dnn(x)
#       x = F.normalize(x, p=2, dim=1)   # ← L2 归一化
#       return x

class UserTower(nn.Module):
    def __init__(self, n_users, embedding_dim, output_dim):
        super().__init__()
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.user_emb_layer = nn.Embedding(n_users, embedding_dim)
        self.dnn = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, user_idx):
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        x = self.user_emb_layer(user_idx)
        x = self.dnn(x)
        x = F.normalize(x, p=2, dim=1)
        return x
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【4】TODO ③：ItemTower
# ============================================================
# 设计：
#   输入：item_idx [B]
#   输出：item_emb [B, OUTPUT_DIM]（已 L2 归一化）
#
# 关键：物品特征 = item_id embedding + 类型 multi-hot
#
# 步骤：
#   1. 通过 item_genre_tensor 查找该 item 的类型 multi-hot 向量
#      （注意：item_genre_tensor 是 [n_items, n_genres] 的常量）
#   2. item_id → embedding [B, EMBEDDING_DIM]
#   3. 类型 multi-hot 直接拼上 [B, n_genres]
#   4. concat 成 [B, EMBEDDING_DIM + n_genres]
#   5. 过 DNN
#   6. L2 归一化
#
# 提示：
#   self.item_emb_layer = nn.Embedding(n_items, embedding_dim)
#   self.register_buffer('genre_table', item_genre_tensor)   # 注册为 buffer，模型移动设备时会跟着走
#   self.dnn = nn.Sequential(
#       nn.Linear(embedding_dim + n_genres, 128), nn.ReLU(),
#       nn.Linear(128, 64), nn.ReLU(),
#       nn.Linear(64, output_dim)
#   )
#   forward:
#       id_emb = self.item_emb_layer(item_idx)         # [B, embedding_dim]
#       genre_emb = self.genre_table[item_idx]          # [B, n_genres]
#       x = torch.cat([id_emb, genre_emb], dim=1)       # [B, embedding_dim + n_genres]
#       x = self.dnn(x)
#       x = F.normalize(x, p=2, dim=1)
#       return x

class ItemTower(nn.Module):
    def __init__(self, n_items, embedding_dim, n_genres, output_dim, genre_table):
        super().__init__()
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.register_buffer('genre_table', item_genre_tensor)
        self.dnn = nn.Sequential(
            nn.Linear(embedding_dim + n_genres, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, item_idx):
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        x1 = self.item_emb(item_idx)
        x2 = self.genre_table[item_idx]
        x = torch.cat([x1, x2], dim=1)
        x = self.dnn(x)
        x = F.normalize(x, p=2, dim=1)
        return x
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【5】TODO ④：DSSM 主模型 + In-batch Negatives Loss
# ============================================================
# 关键：In-batch Negatives（batch 内负采样）
# 思想：在一个 batch 内，
#   - 第 i 个样本的"用户 i"和"物品 i"是正样本对
#   - "用户 i"和"物品 j"（j≠i）就是负样本对
#
# 这样 1 个 batch 的 B 个正样本，自动产生 B*(B-1) 个负样本！效率超高。
#
# 计算：
#   user_emb = user_tower(users)      # [B, D]
#   item_emb = item_tower(items)      # [B, D]
#
#   # 算 batch 内所有 user-item 对的相似度矩阵
#   logits = user_emb @ item_emb.T / temperature   # [B, B]
#
#   # label：每个用户的"正样本"是同行的物品（对角线）
#   labels = torch.arange(B)   # [0, 1, 2, ..., B-1]
#
#   # 用 cross_entropy 自动算 softmax + 负对数似然
#   loss = F.cross_entropy(logits, labels)

class DSSM(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, n_genres,
                 output_dim, genre_table, temperature):
        super().__init__()
        self.user_tower = UserTower(n_users, embedding_dim, output_dim)
        self.item_tower = ItemTower(n_items, embedding_dim, n_genres,
                                    output_dim, genre_table)
        self.temperature = temperature

    def forward(self, users, items):
        """
        users: [B]
        items: [B]
        return: loss (标量)
        """
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        B = len(users)
        loss = None
        user_emb = self.user_tower(users)
        item_emb = self.item_tower(items)
        logits = user_emb @ item_emb.T / self.temperature
        labels = torch.arange(B)
        loss = F.cross_entropy(logits, labels)
        
        return loss
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【6】训练（你不用改）
# ============================================================
print("\n" + "=" * 60)
print("【6】训练 DSSM")
print("=" * 60)

model = DSSM(n_users, n_items, EMBEDDING_DIM, n_genres,
             OUTPUT_DIM, item_genre_tensor, TEMPERATURE)
optimizer = optim.Adam(model.parameters(), lr=LR)

run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_dssm_K{OUTPUT_DIM}"
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
    total_loss = 0.0
    n = 0
    for users, items in train_loader:
        loss = model(users, items)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    train_loss = total_loss / n

    model.eval()
    with torch.no_grad():
        total = 0.0
        m = 0
        for users, items in test_loader:
            loss = model(users, items)
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
    torch.save(best_state, 'best_dssm.pt')
    print(f"✅ 已加载最佳模型（epoch {best_epoch}）")


# ============================================================
# 【7】用学到的模型给用户推荐
# ============================================================
print("\n" + "=" * 60)
print("【7】DSSM 召回示例")
print("=" * 60)

model.eval()
with torch.no_grad():
    # 把所有物品都过物品塔（这就是工业上"离线预计算"做的事）
    all_items_idx = torch.arange(n_items)
    all_item_embs = model.item_tower(all_items_idx)   # [n_items, OUTPUT_DIM]

    # 给前 3 个用户分别推荐
    for user_idx in range(3):
        user_t = torch.LongTensor([user_idx])
        user_emb = model.user_tower(user_t)            # [1, OUTPUT_DIM]

        # 算这个用户和所有物品的相似度
        scores = (user_emb @ all_item_embs.T).squeeze(0).numpy()  # [n_items]
        top10 = scores.argsort()[::-1][:10]

        print(f"\n用户 {user_idx} 的 Top-10 推荐：")
        for rank, idx in enumerate(top10):
            item_id = idx2item[idx]
            title = movies[movies['movieId'] == item_id]['title'].iloc[0]
            print(f"  {rank+1:2d}. {title:<60s} 分数={scores[idx]:.4f}")

writer.close()


# ============================================================
# 🤔 思考题
# ============================================================
"""
1. 为什么 DSSM 的 loss 用 cross_entropy 而不是 MSE？
   （提示：预测的是"匹配概率分布"，不是"评分"）

2. 如果不做 L2 归一化，会发生什么？
   （提示：训练-检索不一致，线上 ANN 拿不到对的 Top-K）

3. 温度系数 τ 调到 0.01 / 0.1 / 1.0 三个值，分别会怎样？
   （建议你跑 3 次对比，TensorBoard 一目了然）

4. 工业上召回时，"全量打分"是不可能的（1 亿物品）。
   DSSM 怎么解决？（提示：FAISS / HNSW 等 ANN 索引）

5. In-batch negatives 有个问题：随机一个 batch 里可能没有
   高质量的"硬负样本"。怎么改？
   （提示：YouTube DNN 用了 hard negative mining）
"""
