"""
BiasSVD 的 PyTorch 实现 —— FunkSVD 的升级版

【与 FunkSVD 的区别】
公式：    r̂ = μ + b_u + b_i + p_u · q_i
              ↑    ↑     ↑    ↑
            全局   用户   物品   交互（同 FunkSVD）

【新增 3 个东西】
- μ：全局偏置（标量，整个评分系统的"基础水位"）
- b_u：用户偏置（每个用户一个标量，反映打分习惯）
- b_i：物品偏置（每个物品一个标量，反映普世受欢迎度）

【你这一节的任务】
- TODO ①：定义 BiasSVD 模型
- 其他所有代码（DataLoader / 训练循环 / 早停 / TensorBoard）直接复用 FunkSVD

运行：
    python biassvd_pytorch.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from datetime import datetime


# ============================================================
# 【0】配置
# ============================================================
DATA_PATH = "/Users/qiruihou/Desktop/学习/推荐算法/dataset/ml-latest-small/ratings.csv"
EMBEDDING_DIM = 32
LR = 0.01
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1024
N_EPOCHS = 2000
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 【1】加载数据 + ID 映射（与 FunkSVD 完全一样）
# ============================================================
df = pd.read_csv(DATA_PATH)

user_ids = df['userId'].unique()
movie_ids = df['movieId'].unique()
user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
movie2idx = {mid: idx for idx, mid in enumerate(movie_ids)}
df['user_idx'] = df['userId'].map(user2idx)
df['movie_idx'] = df['movieId'].map(movie2idx)
n_users = len(user2idx)
n_movies = len(movie2idx)

df_sorted = df.sort_values('timestamp')
train_size = int(len(df_sorted) * 0.8)
train_df = df_sorted.iloc[:train_size].reset_index(drop=True)
test_df = df_sorted.iloc[train_size:].reset_index(drop=True)


class RatingDataset(Dataset):
    def __init__(self, df):
        self.users = torch.LongTensor(df['user_idx'].values)
        self.items = torch.LongTensor(df['movie_idx'].values)
        self.ratings = torch.FloatTensor(df['rating'].values)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


train_loader = DataLoader(RatingDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(RatingDataset(test_df), batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 【2】TODO ①：定义 BiasSVD 模型
# ============================================================
# 公式：r̂ = μ + b_u + b_i + p_u · q_i
#
# 提示（4 个组件）：
#   self.user_emb    = nn.Embedding(n_users, K)          # p_u
#   self.item_emb    = nn.Embedding(n_items, K)          # q_i
#   self.user_bias   = nn.Embedding(n_users, 1)          # b_u
#   self.item_bias   = nn.Embedding(n_items, 1)          # b_i
#   self.global_bias = nn.Parameter(torch.zeros(1))      # μ
#
# 偏置初始化为 0（约定，让训练过程学出修正值）：
#   nn.init.zeros_(self.user_bias.weight)
#   nn.init.zeros_(self.item_bias.weight)
#
# forward 里注意 squeeze（不然 shape 会爆）：
#   b_u = self.user_bias(user_idx).squeeze()   # [B, 1] → [B]
#   b_i = self.item_bias(item_idx).squeeze()
#   prediction = self.global_bias + b_u + b_i + (p * q).sum(dim=1)

class BiasSVD(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, user_idx, item_idx):
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        prediction = None
        p_u = self.user_emb(user_idx).squeeze()
        q_i = self.item_emb(item_idx).squeeze()
        b_u = self.user_bias(user_idx).squeeze()
        b_i = self.item_bias(item_idx).squeeze()
        prediction = self.global_bias + b_u + b_i + (p_u * q_i).sum(dim=1)
        return prediction
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【3】实例化模型 + 损失 + 优化器
# ============================================================
model = BiasSVD(n_users, n_movies, EMBEDDING_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_biassvd_K{EMBEDDING_DIM}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")
print(f"📊 TensorBoard 日志目录：runs/{run_name}")

total_params = sum(p.numel() for p in model.parameters())
print("\n模型结构：")
print(model)
print(f"总参数量：{total_params:,}（FunkSVD 的 ~{total_params/331328:.2f} 倍）")


# ============================================================
# 【4】训练循环（与 FunkSVD 完全一样）
# ============================================================
train_losses = []
best_loss = float('inf')
patience = 3
patience_cnt = 0
best_state = None
best_epoch = 0

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        users, items, ratings = batch
        preds = model(users, items)
        loss = criterion(preds, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / max(n_batches, 1)
    train_losses.append(avg_loss)

    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        n = 0
        for users, items, ratings in test_loader:
            preds = model(users, items)
            test_loss += criterion(preds, ratings).item()
            n += 1
        test_loss /= max(n, 1)

    print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | train_loss={avg_loss:.4f} | test_loss={test_loss:.4f}")
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    # 顺便记录全局偏置 μ 的学习过程（看它有没有收敛到约 3.5）
    writer.add_scalar("Global_bias_mu", model.global_bias.item(), epoch)

    if test_loss < best_loss:
        best_loss = test_loss
        best_epoch = epoch + 1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
        print(f"  ⭐ 新的最佳 test_loss={best_loss:.4f}")
    else:
        patience_cnt += 1
        print(f"  ⏳ test_loss 没改善（{patience_cnt}/{patience}）")
        if patience_cnt >= patience:
            print(f"\n⏸ Early stopping at epoch {epoch+1}")
            print(f"   最佳模型：epoch {best_epoch}，test_loss={best_loss:.4f}")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    torch.save(best_state, 'best_biassvd.pt')
    print(f"\n✅ 已恢复到最佳模型并保存到 best_biassvd.pt")


# ============================================================
# 【5】观察学到的全局偏置 μ
# ============================================================
print("\n" + "=" * 60)
print("【观察学到的偏置】")
print("=" * 60)
print(f"全局偏置 μ = {model.global_bias.item():.4f}")
print(f"   （MovieLens 训练集真实平均评分 = {train_df['rating'].mean():.4f}）")
print(f"   如果 μ 接近真实均值，说明 BiasSVD 成功捕捉了'整体水位'")

# 看 5 个用户的偏置
print(f"\n5 个用户的 b_u（个人打分习惯）：")
for i in range(5):
    b = model.user_bias.weight[i].item()
    print(f"  用户 {i}: b_u = {b:+.4f}")

writer.close()
