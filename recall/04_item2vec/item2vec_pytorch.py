"""
Item2Vec 的 PyTorch 实现（从零写 SkipGram + 负采样）

【这一节的产出】
你将亲手实现 NLP 历史上最重要的算法之一，并把它用到推荐场景。
完成这一节后，你能用一句话讲清楚：
    1. SkipGram 是怎么把"序列"变成"训练样本"的？
    2. 为什么需要负采样？它解决了什么问题？
    3. Item2Vec 学出的 embedding 怎么用于推荐？

【任务拆解】
- TODO ①：构造训练样本（中心物品, 上下文物品）对
- TODO ②：实现 SkipGramNS 模型（负采样版 SkipGram）
- TODO ③：训练 + 用余弦相似度找相似电影

运行：
    python item2vec_pytorch.py
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
MOVIES_PATH = "/Users/qiruihou/Desktop/学习/推荐算法/dataset/ml-latest-small/movies.csv"
EMBEDDING_DIM = 64
WINDOW_SIZE = 5         # 上下文窗口大小（左右各 5 个）
N_NEGATIVES = 5         # 每个正样本配几个负样本
LR = 0.005
BATCH_SIZE = 1024
N_EPOCHS = 5            # Item2Vec 通常不用太多 epoch
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 【1】加载数据 + 构造"行为序列"
# ============================================================
print("=" * 60)
print("【1】构造行为序列")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
movies = pd.read_csv(MOVIES_PATH)

# 只保留高分（≥4）的交互——把"看过"等价于"喜欢"
df = df[df['rating'] >= 4.0].copy()

# ID 重映射
movie_ids = df['movieId'].unique()
movie2idx = {mid: idx for idx, mid in enumerate(movie_ids)}
idx2movie = {idx: mid for mid, idx in movie2idx.items()}
df['movie_idx'] = df['movieId'].map(movie2idx)
n_movies = len(movie2idx)

# 按用户分组、按时间排序，得到每个用户的"行为序列"
df_sorted = df.sort_values(['userId', 'timestamp'])
sequences = df_sorted.groupby('userId')['movie_idx'].apply(list).tolist()

print(f"用户数：{len(sequences)}")
print(f"电影数：{n_movies}")
print(f"序列长度分布（前 5 个用户）：{[len(s) for s in sequences[:5]]}")
print(f"用户 0 的序列前 10 个：{sequences[0][:10]}")


# ============================================================
# 【2】TODO ①：从序列生成 (center, context) 训练对
# ============================================================
# 例如，序列 [a, b, c, d, e]，window=2：
#   中心 a：上下文 [b, c]
#   中心 b：上下文 [a, c, d]
#   中心 c：上下文 [a, b, d, e]
#   中心 d：上下文 [b, c, e]
#   中心 e：上下文 [c, d]
#
# 每个 (中心, 上下文) 都是一个训练样本
#
# 提示：
#   pairs = []
#   for seq in sequences:
#       for i, center in enumerate(seq):
#           # 窗口范围：[max(0, i-window), min(len, i+window+1)]
#           for j in range(max(0, i-WINDOW_SIZE), min(len(seq), i+WINDOW_SIZE+1)):
#               if j != i:  # 跳过自己
#                   pairs.append((center, seq[j]))

pairs = []
# ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
for seq in sequences:
    for i,center in enumerate(seq):
        for j in range(max(0,i-WINDOW_SIZE),min(len(seq),i+WINDOW_SIZE+1)):
            if j!=i:
                pairs.append((center,seq[j]))

# ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

print(f"\n生成的 (center, context) 训练对数量：{len(pairs):,}")
if len(pairs) > 0:
    print(f"前 5 对：{pairs[:5]}")


# ============================================================
# 【3】统计物品频率（用于负采样的"频率^0.75 分布"）
# ============================================================
item_counts = np.zeros(n_movies)
for seq in sequences:
    for item in seq:
        item_counts[item] += 1

# 0.75 次方，再归一化
neg_sample_probs = item_counts ** 0.75
neg_sample_probs = neg_sample_probs / neg_sample_probs.sum()

print(f"\n用于负采样的频率分布（前 5）：{neg_sample_probs[:5]}")


# ============================================================
# 【4】PyTorch Dataset
# ============================================================
class SkipGramDataset(Dataset):
    """每条样本：(center, context) 二元组"""
    def __init__(self, pairs):
        self.centers = torch.LongTensor([p[0] for p in pairs])
        self.contexts = torch.LongTensor([p[1] for p in pairs])

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]


# ============================================================
# 【5】TODO ②：定义 SkipGramNS 模型
# ============================================================
# 关键设计：用两套 embedding 表（这是 Word2Vec 的标准做法）
#   - in_embedding ：作为"中心物品"时用
#   - out_embedding：作为"上下文物品"时用
# （也可以共享一套，但分开学习效果更好）
#
# forward 接收 3 个东西：
#   - centers:   [B]            一批中心物品 ID
#   - contexts:  [B]            一批正样本上下文 ID
#   - negatives: [B, K]         每个中心配 K 个负样本
#
# 计算：
#   v_c = in_embedding(centers)              # [B, D]
#   v_o = out_embedding(contexts)            # [B, D]
#   v_n = out_embedding(negatives)           # [B, K, D]
#
#   pos_score = (v_c * v_o).sum(dim=1)       # [B]
#   neg_score = torch.bmm(v_n, v_c.unsqueeze(2)).squeeze()  # [B, K]
#
# loss（负采样目标）：
#   pos_loss = -F.logsigmoid(pos_score).mean()
#   neg_loss = -F.logsigmoid(-neg_score).sum(dim=1).mean()
#   loss = pos_loss + neg_loss
#
# 提示：用 import torch.nn.functional as F
#       sigmoid 的对数 = F.logsigmoid

import torch.nn.functional as F

class SkipGramNS(nn.Module):
    def __init__(self, n_items, embedding_dim):
        super().__init__()
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.in_embedding = nn.Embedding(n_items,embedding_dim)
        self.out_embedding = nn.Embedding(n_items,embedding_dim)

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, centers, contexts, negatives):
        """
        centers:   [B]
        contexts:  [B]
        negatives: [B, K]
        返回：标量 loss
        """
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        loss = None
        v_c = self.in_embedding(centers)
        v_o = self.out_embedding(contexts)
        v_n = self.out_embedding(negatives)

        pos_score = (v_c * v_o).sum(dim=1)
        neg_score = torch.bmm(v_n, v_c.unsqueeze(2)).squeeze()

        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1).mean()
        loss = pos_loss + neg_loss
        return loss
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【6】训练（你不用改）
# ============================================================
print("\n" + "=" * 60)
print("【6】训练 SkipGramNS")
print("=" * 60)

dataset = SkipGramDataset(pairs)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SkipGramNS(n_movies, EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters(), lr=LR)

run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_item2vec_K{EMBEDDING_DIM}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")
print(f"📊 TensorBoard 日志目录：runs/{run_name}")

print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
print(f"训练样本数：{len(dataset):,}")
print(f"每个 epoch batch 数：{len(loader)}")

step = 0
for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    n = 0
    for centers, contexts in loader:
        # 为这一批的每个中心，按 neg_sample_probs 采 N_NEGATIVES 个负样本
        batch_size = centers.size(0)
        negatives = np.random.choice(
            n_movies,
            size=(batch_size, N_NEGATIVES),
            p=neg_sample_probs
        )
        negatives = torch.LongTensor(negatives)

        loss = model(centers, contexts, negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n += 1
        step += 1

        if step % 200 == 0:
            writer.add_scalar("Loss/train", loss.item(), step)

    avg = epoch_loss / max(n, 1)
    print(f"Epoch {epoch+1}/{N_EPOCHS}: avg loss = {avg:.4f}")

writer.close()


# ============================================================
# 【7】TODO ③：用学到的 embedding 做"找相似电影"
# ============================================================
print("\n" + "=" * 60)
print("【7】检验 embedding 质量：找相似电影")
print("=" * 60)

# 取出学到的 item embedding（用 in_embedding 即可）
item_embs = model.in_embedding.weight.detach()   # [n_movies, D]

# L2 归一化（让内积变成余弦相似度）
item_embs_norm = item_embs / item_embs.norm(dim=1, keepdim=True)

def find_similar_movies(movie_title_keyword, top_k=10):
    """根据电影标题关键词，找最相似的 top_k 部电影"""
    # 找包含关键词的电影
    matched = movies[movies['title'].str.contains(movie_title_keyword, case=False, na=False)]
    if len(matched) == 0:
        print(f"找不到包含 '{movie_title_keyword}' 的电影")
        return

    target = matched.iloc[0]
    target_movie_id = target['movieId']
    if target_movie_id not in movie2idx:
        print(f"《{target['title']}》在我们的训练集里没出现过（可能评分不够）")
        return

    target_idx = movie2idx[target_movie_id]
    target_emb = item_embs_norm[target_idx]

    # 算余弦相似度
    sims = (item_embs_norm @ target_emb).numpy()
    sim_indices = sims.argsort()[::-1][:top_k+1]

    print(f"\n查询：《{target['title']}》")
    print(f"最相似的 {top_k} 部电影：")
    for rank, idx in enumerate(sim_indices):
        if idx == target_idx:
            continue
        mid = idx2movie[idx]
        title = movies[movies['movieId'] == mid]['title'].iloc[0]
        print(f"  {rank+1:2d}. {title:<60s}  similarity={sims[idx]:.4f}")


# 试几个经典电影
find_similar_movies("Toy Story")
find_similar_movies("Lord of the Rings")
find_similar_movies("Star Wars")


# ============================================================
# 🤔 思考题
# ============================================================
"""
1. Item2Vec 完全不用评分信息（你这里只过滤了 rating>=4），
   它怎么还能学出"相似电影"？

2. 为什么用两套 embedding（in/out）而不是一套？
   （提示：实际工业版很多用共享一套，效果接近）

3. 如果窗口 WINDOW_SIZE = 100（极大），会发生什么？
   （提示：所有物品都互为上下文，等价于... ItemCF？）

4. 如果数据按时间排序变成"按字母排序"，效果会怎样？
   （提示：序列的"局部性"是 Item2Vec 的核心假设）

5. 我们这里用了"用户全部观影序列"，但忽略了电影类型/演员等信息。
   下一节 EGES 会怎么改进？
"""
