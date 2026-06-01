"""
FunkSVD (Matrix Factorization) 的 PyTorch 实现

【这一节的核心】
你的第一个"现代意义上的"推荐模型 —— 终于用上 PyTorch 训练循环了！

【任务拆解】
- TODO ①：定义 FunkSVD 模型（继承 nn.Module）
- TODO ②：准备 MovieLens 训练数据
- TODO ③：写训练循环（5 步法，与 03_nn_module.py 一致）
- TODO ④：用学好的模型做预测，看效果

【关键参数】
- K (embedding_dim)：隐向量维度，常用 16 / 32 / 64
- lr：学习率，常用 0.01
- weight_decay：L2 正则系数，常用 1e-4
- n_epochs：训练轮数，30-100 都可以

运行：
    python funksvd_pytorch.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter   # ← 新增：TensorBoard 写入器
import pandas as pd
import numpy as np
from datetime import datetime


# ============================================================
# 【0】配置
# ============================================================
DATA_PATH = "/Users/qiruihou/Desktop/学习/推荐算法/dataset/ml-latest-small/ratings.csv"
EMBEDDING_DIM = 8       # 隐向量维度 K
LR = 0.05                # 学习率
WEIGHT_DECAY = 1e-4      # L2 正则系数 λ
BATCH_SIZE = 1024
N_EPOCHS = 2000
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 【1】加载并预处理数据
# ============================================================
print("=" * 60)
print("【1】加载数据")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"原始数据 shape: {df.shape}")
print(df.head())

# 关键步骤：把原始的 userId / movieId 重新映射到 [0, n) 的连续整数
# 因为 nn.Embedding 要求 ID 是 [0, num_embeddings) 的整数
user_ids = df['userId'].unique()
movie_ids = df['movieId'].unique()

user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
movie2idx = {mid: idx for idx, mid in enumerate(movie_ids)}

df['user_idx'] = df['userId'].map(user2idx)
df['movie_idx'] = df['movieId'].map(movie2idx)

n_users = len(user2idx)
n_movies = len(movie2idx)
print(f"\n用户数：{n_users}, 电影数：{n_movies}")
print(f"映射后前 3 行：")
print(df[['user_idx', 'movie_idx', 'rating']].head(3))


# ============================================================
# 【2】简单划分训练集 / 测试集（按时间，与 fun-rec 思路类似）
# ============================================================
df_sorted = df.sort_values('timestamp')
train_size = int(len(df_sorted) * 0.8)
train_df = df_sorted.iloc[:train_size].reset_index(drop=True)
test_df = df_sorted.iloc[train_size:].reset_index(drop=True)
print(f"\n训练集：{len(train_df)}, 测试集：{len(test_df)}")


# ============================================================
# 【3】PyTorch Dataset：把 DataFrame 变成可迭代的训练数据
# ============================================================
class RatingDataset(Dataset):
    """
    PyTorch 的 Dataset 套路：
    - __init__：保存数据
    - __len__：返回总样本数
    - __getitem__：返回第 idx 个样本（user_idx, movie_idx, rating）
    """
    def __init__(self, df):
        self.users = torch.LongTensor(df['user_idx'].values)
        self.items = torch.LongTensor(df['movie_idx'].values)
        self.ratings = torch.FloatTensor(df['rating'].values)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


train_loader = DataLoader(
    RatingDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    RatingDataset(test_df),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 【4】TODO ①：定义 FunkSVD 模型
# ============================================================
# 提示：
#   - 继承 nn.Module
#   - __init__ 里定义两个 Embedding 层：user_emb 和 item_emb
#       nn.Embedding(num_embeddings, embedding_dim)
#   - forward 接收 (user_idx, item_idx)，返回预测评分
#       预测评分 = (user_emb(user_idx) * item_emb(item_idx)).sum(dim=1)
#       注意 sum(dim=1) 是按"行内"求和，得到每个样本一个分数

class FunkSVD(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        # ↓↓↓↓↓ 你的代码写在这里 ↓↓↓↓↓
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

        # ↑↑↑↑↑ 你的代码写在这里 ↑↑↑↑↑

    def forward(self, user_idx, item_idx):
        # ↓↓↓↓↓ 你的代码写在这里 ↓↓↓↓↓
        prediction = (self.user_emb(user_idx) * self.item_emb(item_idx)).sum(dim=1)

        return prediction
        # ↑↑↑↑↑ 你的代码写在这里 ↑↑↑↑↑


# ============================================================
# 【5】实例化模型 + 损失 + 优化器（三件套，和 03_nn_module.py 一样！）
# ============================================================
model = FunkSVD(n_users, n_movies, EMBEDDING_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ===== TensorBoard 设置 =====
# 每次跑都用时间戳建一个新目录，方便对比不同实验
run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_K{EMBEDDING_DIM}_lr{LR}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")
print(f"\n📊 TensorBoard 日志目录：runs/{run_name}")
print(f"   启动命令：tensorboard --logdir=runs")

# 检查模型参数量
total_params = sum(p.numel() for p in model.parameters())
print("\n" + "=" * 60)
print("【4】模型结构")
print("=" * 60)
print(model)
print(f"总参数量：{total_params:,}")
print(f"  user_emb: {n_users} × {EMBEDDING_DIM} = {n_users * EMBEDDING_DIM:,}")
print(f"  item_emb: {n_movies} × {EMBEDDING_DIM} = {n_movies * EMBEDDING_DIM:,}")


# ============================================================
# 【6】TODO ②：训练循环（5 步法）
# ============================================================
# 这部分你已经在 03_nn_module.py 写过了，回顾一下：
#   for epoch in range(N_EPOCHS):
#       for batch in train_loader:
#           users, items, ratings = batch          # 解包
#           predictions = model(users, items)      # ① forward
#           loss = criterion(predictions, ratings) # ② loss
#           optimizer.zero_grad()                  # ③ 清零梯度
#           loss.backward()                        # ④ 反向传播
#           optimizer.step()                       # ⑤ 更新参数

print("\n" + "=" * 60)
print("【5】训练")
print("=" * 60)

train_losses = []

# ===== Early Stopping 相关变量 =====
best_loss = float('inf')   # 历史最佳 test_loss
patience = 3               # 容忍多少轮不进步
patience_cnt = 0           # 当前已经多少轮没进步
best_state = None          # 保存最佳模型权重（dict 形式）
best_epoch = 0             # 最佳模型对应的 epoch

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    # ───── 训练一轮 ─────
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

    # ───── 评估测试集 loss ─────
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        n = 0
        for users, items, ratings in test_loader:
            preds = model(users, items)
            test_loss += criterion(preds, ratings).item()
            n += 1
        test_loss /= max(n, 1)

    # ───── 打印 + TensorBoard 记录（先记录，再判断早停！）─────
    print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | train_loss={avg_loss:.4f} | test_loss={test_loss:.4f}")
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_histogram("user_emb", model.user_emb.weight, epoch)
    writer.add_histogram("item_emb", model.item_emb.weight, epoch)

    # ───── Early Stopping ─────
    if test_loss < best_loss:
        # 进步了：更新最佳，保存权重
        best_loss = test_loss
        best_epoch = epoch + 1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
        print(f"  ⭐ 新的最佳 test_loss={best_loss:.4f}，已保存权重")
    else:
        patience_cnt += 1
        print(f"  ⏳ test_loss 没改善（{patience_cnt}/{patience}）")
        if patience_cnt >= patience:
            print(f"\n⏸ Early stopping at epoch {epoch+1}")
            print(f"   最佳模型：epoch {best_epoch}，test_loss={best_loss:.4f}")
            break

# ===== 训练结束：恢复最佳模型权重 =====
if best_state is not None:
    model.load_state_dict(best_state)
    print(f"\n✅ 已恢复到最佳模型（epoch {best_epoch}，test_loss={best_loss:.4f}）")
    torch.save(best_state, 'best_funksvd.pt')
    print(f"💾 最佳模型已保存到 best_funksvd.pt")

# 训练结束，记录学到的 embedding 投影（最酷的功能）
writer.add_embedding(
    model.item_emb.weight,
    metadata=[f"movie_{i}" for i in range(n_movies)],
    tag="item_embeddings",
)
writer.close()
print(f"\n✅ TensorBoard 日志已保存到 runs/{run_name}")


# ============================================================
# 【7】用训练好的模型做预测（看看学的怎么样）
# ============================================================
print("\n" + "=" * 60)
print("【6】预测示例")
print("=" * 60)

model.eval()
with torch.no_grad():
    # 拿测试集前 5 条看看
    for i in range(5):
        u_idx = int(test_df.iloc[i]['user_idx'])     # pandas 单值会变 float，转回 int
        m_idx = int(test_df.iloc[i]['movie_idx'])
        true_rating = float(test_df.iloc[i]['rating'])

        user_t = torch.LongTensor([u_idx])
        item_t = torch.LongTensor([m_idx])
        pred = model(user_t, item_t).item()

        print(f"  user={u_idx:3d}, movie={m_idx:5d} | 真实={true_rating:.1f}, 预测={pred:.2f}")


# ============================================================
# 【8】观察学到的 embedding（最有趣的部分！）
# ============================================================
print("\n" + "=" * 60)
print("【7】观察学到的电影向量（前 5 部，前 8 维）")
print("=" * 60)
item_emb_weight = model.item_emb.weight.detach().numpy()
print(f"item_emb 矩阵 shape: {item_emb_weight.shape}")
print(f"前 5 部电影的前 8 个隐因子值：")
print(item_emb_weight[:5, :8].round(3))


# ============================================================
# 🤔 思考题
# ============================================================
"""
1. 为什么要把 userId 重映射到 [0, n_users) 这个连续整数空间？
   原始的 userId 不是已经是整数了吗？
   （提示：MovieLens 的 userId 从 1 开始，电影数 9742 但 movieId 远超 10000）

2. nn.Embedding 和你之前用的 nn.Linear 的本质区别是什么？
   （提示：输入类型不同——一个是连续值，一个是离散 ID）

3. 我们这里训练时只用了"用户实际评过的电影"，
   为什么不用"用户没评过的电影"作为负样本？
   （这是个深刻问题，矩阵分解的局限——下下节 DSSM 会解决它）

4. 如果不做 weight_decay（L2 正则），会发生什么？
   （提示：embedding 数值会无限增长，过拟合）
"""
