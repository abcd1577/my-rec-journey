"""
MIND 召回模型 —— PyTorch 实现
（Multi-Interest Network with Dynamic Routing, Alibaba KDD 2019）

【这一节的产出】
1. 你将亲手实现"多兴趣"召回——把用户从 1 个向量升级为 K 个向量
2. 第一次实现"胶囊网络的动态路由"（Capsule Routing）
3. 学会 Label-Aware Attention（训练时把 K 个向量塌缩成 1 个）
4. 理解"训练 vs 推理"不同的用法（训练塌缩、推理 K 路并发召回）

【与 YouTubeDNN 的关键区别】
┌────────────────┬──────────────────────┬───────────────────────────────┐
│      维度       │     YouTubeDNN       │            MIND               │
├────────────────┼──────────────────────┼───────────────────────────────┤
│ 用户向量个数    │  1                   │  K（自适应，最多 max_k）       │
│ 序列聚合方式    │  Average Pooling     │  Capsule Dynamic Routing      │
│ 训练 loss      │  Softmax / Sampled   │  同 YouTubeDNN（但先做 Attn）  │
│ 推理召回次数    │  1 次 ANN            │  K 次 ANN 并发合并              │
│ 兴趣表达能力    │  单一兴趣（被平均）   │  多元兴趣（程序+运动+生活）     │
└────────────────┴──────────────────────┴───────────────────────────────┘

【核心建模思路（B2I 动态路由）】
   把"用户历史 N 个行为"软聚类到 K 个"兴趣胶囊"：

       b_ij         ← 行为 i 对兴趣 j 的"投票分数"（初始随机）
       ↓ softmax
       w_ij = softmax_j(b_ij)
       ↓ 加权聚合
       z_j  = Σ_i w_ij · S · e_i              （S：共享变换矩阵）
       ↓ 非线性压缩（保方向，长度压到[0,1)）
       u_j  = squash(z_j)
       ↓ 根据新画像更新投票
       b_ij += u_j^T · S · e_i
       ↓
       回到第一步，迭代 3 次

【任务拆解】
- TODO ①：实现 squash 函数（非线性压缩）
- TODO ②：实现 CapsuleLayer（动态路由的灵魂）
- TODO ③：实现 LabelAwareAttention（训练时塌缩 K → 1）
- TODO ④：实现 MIND 主模型 forward
- TODO ⑤：实现"推理时 K 路并发召回"

运行：
    python mind_pytorch.py
"""

import math
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

EMBEDDING_DIM = 32        # 物品 Embedding 维度
HISTORY_LEN = 20          # 用户最多保留最近 N 个历史观看
MAX_K = 4                 # 最大兴趣胶囊数（论文里通常取 4~8）
ROUTING_ITERS = 3         # 动态路由迭代次数（论文经验值 3）
POW_P = 1.0               # LabelAwareAttention 的温度系数（>=100 退化为 argmax）
OUTPUT_DIM = 32           # 用户向量 / 物品向量最终维度（必须等于 EMBEDDING_DIM）
LR = 0.001
BATCH_SIZE = 256
N_EPOCHS = 15
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 【1】加载数据（和 YouTubeDNN 完全一致，便于对比效果）
# ============================================================
print("=" * 60)
print("【1】加载数据，构造用户/物品特征")
print("=" * 60)

ratings = pd.read_csv(DATA_PATH)
movies = pd.read_csv(MOVIES_PATH)
ratings = ratings[ratings['rating'] >= 3.0].copy()

unique_users = ratings['userId'].unique()
unique_items = ratings['movieId'].unique()
user2idx = {u: i for i, u in enumerate(unique_users)}
item2idx = {m: i + 1 for i, m in enumerate(unique_items)}   # 0 = PAD
idx2item = {i: m for m, i in item2idx.items()}

ratings['user_idx'] = ratings['userId'].map(user2idx)
ratings['item_idx'] = ratings['movieId'].map(item2idx)
n_users = len(user2idx)
n_items = len(item2idx) + 1

print(f"用户数: {n_users}, 物品数（含PAD）: {n_items}")


# ============================================================
# 【2】构造 (user, history_seq, target) 三元组（同 YouTubeDNN）
# ============================================================
print("\n" + "=" * 60)
print("【2】构造 (user, history_seq, target) 三元组")
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
        history = [item for item, _ in seq[max(0, t - HISTORY_LEN):t]]
        # 注意：MIND 同样用"左 padding"，让真实物品贴在右侧
        if len(history) < HISTORY_LEN:
            history = [0] * (HISTORY_LEN - len(history)) + history
        samples.append((u, history, target, target_ts))

print(f"总样本数：{len(samples)}")


class MINDDataset(Dataset):
    def __init__(self, samples):
        self.users = torch.LongTensor([s[0] for s in samples])
        self.histories = torch.LongTensor([s[1] for s in samples])
        self.targets = torch.LongTensor([s[2] for s in samples])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.histories[idx], self.targets[idx]


samples_sorted = sorted(samples, key=lambda x: x[3])
split = int(len(samples_sorted) * 0.8)
train_samples = samples_sorted[:split]
test_samples = samples_sorted[split:]

train_loader = DataLoader(MINDDataset(train_samples),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(MINDDataset(test_samples),
                         batch_size=BATCH_SIZE, shuffle=False)
print(f"训练集：{len(train_samples)}, 测试集：{len(test_samples)}")


# ============================================================
# 【3】TODO ①：squash 函数（非线性压缩）
# ============================================================
# 公式：
#   squash(z) = (||z||² / (1 + ||z||²)) * (z / ||z||)
#
# 它做了两件事：
#   1. 把向量"长度"压到 [0, 1)（前半部分 ||z||²/(1+||z||²) 是个 sigmoid 形）
#   2. 保持向量"方向"不变（后半部分 z/||z|| 是单位向量）
#
# 为什么要这样？胶囊网络的核心哲学：
#   - 长度 = "这个兴趣存在的概率"
#   - 方向 = "兴趣的具体内容（指向哪类物品）"
# 所以长度必须在 [0,1) 当概率用。
#
# 实现提示：
#   - z: [B, K, D]
#   - 用 z.pow(2).sum(dim=-1, keepdim=True) 算 ‖z‖²，shape [B, K, 1]
#   - 加 1e-9 防除零

def squash(z, eps=1e-9):
    """
    输入: z  [..., D]
    输出: 同 shape，长度压到 [0,1)，方向不变
    """
    # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
    pow_z = z.pow(2).sum(dim=-1, keepdim=True)
    return (pow_z / (1 + pow_z)) * (z / (torch.norm(z, dim=-1, keepdim=True) + eps))
    # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【4】TODO ②：CapsuleLayer（动态路由）
# ============================================================
# 这是 MIND 的灵魂。把 [B, N, D] 的行为序列 → [B, K, D] 的兴趣胶囊。
#
# 数据流（迭代 ROUTING_ITERS 次）：
#   每轮：
#     1. w_ij = softmax_j(b_ij)              # 投票归一化  [B, K, N]
#     2. z_j  = Σ_i w_ij · S · e_i           # 加权聚合    [B, K, D]
#     3. u_j  = squash(z_j)                  # 非线性压缩  [B, K, D]
#     4. b_ij += u_j^T · S · e_i             # 更新投票
#
# 🔑 关键参数：
#   - bilinear_mapping_matrix S: [D_in, D_out]  共享变换矩阵（可训练）
#   - routing_logits b: [1, K, N]              路由系数（不可训练，前向时高斯初始化）
#
# 🔑 Mask 处理：
#   - 历史里有 padding (item=0)，要把这些位置的投票分数置为 -inf
#   - 这样 softmax 后这些位置权重 → 0，padding 不参与聚合
#
# ⚠️ 注意：原版 TF 代码用 self.routing_logits.assign_add(...) 把 b 当成"状态"跨 batch 累加，
#   但这并不是论文本意。论文的动态路由是"每个样本独立做 3 次迭代"。
#   我们用更标准的写法：每次 forward 重新初始化 b，每个样本独立路由。

class CapsuleLayer(nn.Module):
    def __init__(self, input_units, out_units, max_len, k_max,
                 iteration_times=3, init_std=1.0):
        super().__init__()
        self.input_units = input_units    # D_in
        self.out_units = out_units        # D_out
        self.max_len = max_len            # N (历史最大长度)
        self.k_max = k_max                # K (最大兴趣数)
        self.iteration_times = iteration_times
        self.init_std = init_std

        # 共享双线性变换矩阵 S，行为空间 → 兴趣空间
        self.S = nn.Parameter(
            torch.randn(input_units, out_units) * init_std
        )

    def forward(self, behavior_embeddings, history_mask, capsule_num=None):
        """
        参数:
          behavior_embeddings: [B, N, D_in]  历史行为 embedding
          history_mask:        [B, N]        True/1 表示真实物品，False/0 表示 PAD
          capsule_num:         [B] 或 None   每个用户实际兴趣数（自适应 K_u'）
                               为简化，第一版我们先不实现自适应，传 None 表示统一用 K
        返回:
          interest_capsules:   [B, K, D_out] 多兴趣向量
        """
        B = behavior_embeddings.size(0)
        device = behavior_embeddings.device

        # ↓↓↓↓↓ 你的代码（核心 30 行）↓↓↓↓↓

        # ① 初始化路由系数 b_ij：[B, K, N]
        #    每次 forward 独立初始化（高斯随机）
        #    提示：torch.randn(B, self.k_max, self.max_len, device=device) * self.init_std
        b = torch.randn(B, self.k_max, self.max_len, device=device) * self.init_std

        # ② 准备 mask：[B, K, N]
        #    把 history_mask [B, N] 扩成 [B, K, N]
        #    提示：history_mask.unsqueeze(1).expand(-1, self.k_max, -1)
        mask = history_mask.unsqueeze(1).expand(-1, self.k_max, -1)

        # ③ 提前算好 S·e_i（每轮迭代不变，可以缓存在循环外）
        #    behavior_embeddings [B, N, D_in] @ S [D_in, D_out] → [B, N, D_out]
        behavior_mapped = behavior_embeddings @ self.S

        # ④ 动态路由迭代
        for it in range(self.iteration_times):
            # Step 1: 把 b 中 padding 位置置为 -inf，再 softmax
            #   提示：b.masked_fill(~mask, -1e9)  （bool mask）
            #         然后在 dim=-1（也就是 N 这个维度）上 softmax
            b_masked = b.masked_fill(~mask, -1e9)
            w = F.softmax(b_masked, dim=-1)        # TODO  [B, K, N]

            # Step 2: 加权聚合 z_j = Σ_i w_ij · (S · e_i)
            #   w [B, K, N]  @  behavior_mapped [B, N, D_out]  →  [B, K, D_out]
            z = w @ behavior_mapped # TODO

            # Step 3: squash 压缩
            u = squash(z)   # TODO  [B, K, D_out]

            # Step 4: 更新 b （除了最后一轮可以不更新，但更新也无害）
            #   delta_b_ij = u_j · (S · e_i)
            #   u [B, K, D_out]  @  behavior_mapped.transpose(1,2) [B, D_out, N]  →  [B, K, N]
            if it < self.iteration_times - 1:
                delta_b = u @ behavior_mapped.transpose(1, 2)# TODO
                b = b + delta_b

        # ⑤ 返回最终兴趣胶囊
        return u
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【5】TODO ③：LabelAwareAttention（训练时塌缩 K → 1）
# ============================================================
# 训练时我们有 K 个用户向量，但只有 1 个 target，要算 loss。
# 解决方案：让模型自己挑出"最贴近 target 的那个兴趣向量"。
#
# 公式：
#   scores_k = (u_k · target)              # 每个兴趣和 target 的相似度
#   scores_k = pow(scores_k, p)            # 极化（p 越大，越接近 argmax）
#   weights  = softmax(scores)             # 归一化成权重
#   v_u      = Σ_k weights_k · u_k         # 加权求和，得到最终用户向量
#
# 注意：
#   - 训练时用这个；推理时不用，K 个向量分开召回
#   - p 一般取 1（柔和加权）或 >=100（硬 argmax）

class LabelAwareAttention(nn.Module):
    def __init__(self, pow_p=1.0):
        super().__init__()
        self.pow_p = pow_p

    def forward(self, keys, query):
        """
        keys:  [B, K, D]   K 个兴趣向量
        query: [B, D]      target 物品向量
        return: [B, D]     塌缩后的用户向量
        """
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓

        # 提示：
        # 1. 算相似度： (keys * query.unsqueeze(1)).sum(dim=-1)  → [B, K]
        # 2. 极化： .pow(self.pow_p)
        # 3. softmax over K → 得到 [B, K] 的权重
        # 4. 加权求和：(keys * weights.unsqueeze(-1)).sum(dim=1)  → [B, D]

        attn = (keys * query.unsqueeze(1)).sum(dim=-1).pow(self.pow_p)
        attn = F.softmax(attn, dim=-1)
        return (keys * attn.unsqueeze(-1)).sum(dim=1)
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【6】TODO ④：MIND 主模型
# ============================================================
# 整体结构（对照理论图）：
#
#   user_id ─→ user_emb ─┐
#                        ├─→ DNN ─→ V_u [B, K, D]
#   history ─→ item_emb ─┴─→ CapsuleLayer ─→ [B, K, D]
#                                                │
#                                  Label-Aware Attention（训练时用）
#                                                │
#                                          v_u [B, D]
#                                                │
#                                  全库 Softmax (== YouTubeDNN)
#
# 与 fun-rec 官方 TF 实现的差异：
#   - 官方在胶囊层 *之后* 还会"把 user_dnn_emb 拼到每个胶囊上 → 再过 MLP 降维"
#     （让兴趣向量带上用户基础画像）
#   - 我们第一版先简化掉这一步，直接把胶囊输出作为 V_u
#   - 等你跑通后，思考题 Q5 会让你思考要不要加回来

class MIND(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, output_dim,
                 max_len, k_max, routing_iters=3, pow_p=1.0):
        super().__init__()
        self.k_max = k_max

        # 物品 Embedding（共享给历史序列 + 最终 Softmax）
        self.item_embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)

        # 胶囊层
        self.capsule = CapsuleLayer(
            input_units=embedding_dim,
            out_units=output_dim,
            max_len=max_len,
            k_max=k_max,
            iteration_times=routing_iters,
        )

        # 标签感知注意力
        self.label_attention = LabelAwareAttention(pow_p=pow_p)

        assert output_dim == embedding_dim, \
            "OUTPUT_DIM 必须等于 EMBEDDING_DIM（要和 item_embedding 算相似度）"

    def get_user_interests(self, histories):
        """
        推理 / 评估时调用：返回 K 个兴趣向量（不做 attention 塌缩）

        histories: [B, N]
        return:    [B, K, D]  L2 归一化后的兴趣向量
        """
        history_emb = self.item_embedding(histories)        # [B, N, D]
        mask = (histories != 0)                              # [B, N] bool
        interests = self.capsule(history_emb, mask)         # [B, K, D]
        return F.normalize(interests, p=2, dim=-1)

    def forward(self, users, histories, targets):
        """
        users:     [B]            （本第一版没用 user_id，留接口给后续升级）
        histories: [B, N]
        targets:   [B]
        return:    loss (标量)
        """
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓

        # Step 1: 拿到 K 个兴趣向量（已 L2 归一化）
        interests = self.get_user_interests(histories)      # [B, K, D]   TODO

        # Step 2: 拿到 target embedding
        target_emb = self.item_embedding(targets)     # [B, D]      TODO   提示：self.item_embedding(targets)

        # Step 3: Label-Aware Attention 塌缩成 1 个向量
        user_emb = self.label_attention(interests, target_emb)      # [B, D]      TODO   提示：self.label_attention(interests, target_emb)

        # Step 4: L2 归一化 user_emb（可选，但和 YouTubeDNN 对齐）
        user_emb = F.normalize(user_emb, p=2, dim=-1)

        # Step 5: 全库 Softmax loss（同 YouTubeDNN）
        logits = user_emb @ self.item_embedding.weight.T        # [B, n_items]  TODO   提示：user_emb @ self.item_embedding.weight.T
        loss = F.cross_entropy(logits, targets)
        return loss
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


# ============================================================
# 【7】训练（你不用改）
# ============================================================
print("\n" + "=" * 60)
print("【7】训练 MIND")
print("=" * 60)

model = MIND(
    n_users=n_users,
    n_items=n_items,
    embedding_dim=EMBEDDING_DIM,
    output_dim=OUTPUT_DIM,
    max_len=HISTORY_LEN,
    k_max=MAX_K,
    routing_iters=ROUTING_ITERS,
    pow_p=POW_P,
)
optimizer = optim.Adam(model.parameters(), lr=LR)

run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_mind_K{MAX_K}"
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
    torch.save(best_state, 'best_mind.pt')
    print(f"✅ 已加载最佳模型（epoch {best_epoch}）")


# ============================================================
# 【8】TODO ⑤：推理 —— K 路并发召回 + 合并去重
# ============================================================
# 🔑 这是 MIND 真正发力的地方！
#   - 训练时把 K 个向量塌缩成 1 个是为了能算 loss
#   - 推理时 K 个向量分开用，每个独立召回 Top-N，合并 → 体现"多兴趣"价值
#
# 召回流程：
#   1. 用 model.get_user_interests(history) 得到 [1, K, D] 共 K 个向量
#   2. 对每个兴趣向量 u_k，算 scores_k = u_k @ all_items.T → 取 Top-N
#   3. 合并 K 路结果（可以用每个兴趣的 max score 作为最终排序依据）
#   4. 去重 + 屏蔽已看过的

print("\n" + "=" * 60)
print("【8】MIND 召回示例（K 路并发）")
print("=" * 60)

TOPN_PER_INTEREST = 20    # 每个兴趣召回多少
FINAL_TOPK = 10           # 最终展示几个

model.eval()
with torch.no_grad():
    all_item_embs = model.item_embedding.weight    # [n_items, D]

    for user_idx in range(3):
        u_seq = user_seqs.get(user_idx, [])
        if len(u_seq) < 2:
            continue
        history = [item for item, _ in u_seq[-HISTORY_LEN:]]
        if len(history) < HISTORY_LEN:
            history = [0] * (HISTORY_LEN - len(history)) + history

        seen = set([item for item, _ in u_seq])
        history_t = torch.LongTensor([history])

        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓

        # Step 1: 拿到 K 个兴趣向量
        interests = model.get_user_interests(history_t)   # [1, K, D]   TODO   提示：model.get_user_interests(history_t)
        interests = interests.squeeze(0)   # [K, D]

        # Step 2: 每个兴趣独立算分
        #   interests [K, D]  @  all_item_embs.T [D, n_items]  →  [K, n_items]
        scores_per_interest = interests @ all_item_embs.T # TODO

        # Step 3: 取每个 item 在 K 个兴趣中的最高分作为最终分（max pooling 合并）
        #   scores_per_interest.max(dim=0).values  →  [n_items]
        final_scores = scores_per_interest.max(dim=0).values   # TODO

        # 屏蔽
        final_scores = final_scores.clone()
        final_scores[0] = -1e9
        for s in seen:
            final_scores[s] = -1e9

        top_k = torch.topk(final_scores, FINAL_TOPK).indices.tolist()

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        print(f"\n用户 {user_idx} 的 Top-{FINAL_TOPK} 推荐（融合 {MAX_K} 个兴趣）：")
        for rank, idx in enumerate(top_k):
            item_id = idx2item.get(idx, None)
            if item_id is None:
                continue
            row = movies[movies['movieId'] == item_id]
            if len(row) == 0:
                continue
            title = row['title'].iloc[0]
            print(f"  {rank+1:2d}. {title:<60s} 分数={final_scores[idx]:.4f}")

        # 进阶：打印每个兴趣分别召回了什么（看看兴趣有没有"分化"）
        print(f"\n  🔍 看看 {MAX_K} 个兴趣分别长什么样（各取 Top-3）：")
        for k in range(MAX_K):
            s_k = scores_per_interest[k].clone()
            s_k[0] = -1e9
            for s in seen:
                s_k[s] = -1e9
            top3 = torch.topk(s_k, 3).indices.tolist()
            titles = []
            for idx in top3:
                item_id = idx2item.get(idx, None)
                if item_id is None:
                    continue
                row = movies[movies['movieId'] == item_id]
                if len(row) > 0:
                    titles.append(row['title'].iloc[0][:30])
            print(f"     兴趣 {k}: {' | '.join(titles)}")

writer.close()


# ============================================================
# 🤔 思考题（写完代码再答）
# ============================================================
"""
1. 我们的 CapsuleLayer 每次 forward 都重新随机初始化 b_ij。
   fun-rec 官方 TF 代码却用 self.routing_logits.assign_add(...) 把 b 跨 batch 累加。
   哪种更符合论文原意？为什么？
   （提示：动态路由是"对每个用户独立做软聚类"，应该 batch 间独立）

2. 训练时用 Label-Aware Attention 塌缩成 1 个向量，本质上是在"鼓励 K 个兴趣分化"还是"鼓励 K 个兴趣同质化"？
   （提示：被 attention 选中的那个兴趣会被 loss 直接监督；没被选中的兴趣几乎不被惩罚 → ?）

3. 看你召回出来的 K 个兴趣的 Top-3，它们之间是真的"分化"了，还是高度重复？
   如果重复严重，可能的原因有哪些？怎么改进？
   （提示：可以加 diversity loss、可以增大 init_std、可以调 pow_p）

4. 推理时我们用 max 聚合 K 路召回。如果改用 sum、avg、或加权（weight = ||u_k||）会怎样？
   （提示：胶囊向量的"长度"代表该兴趣的强度，弱兴趣的召回该不该被压低？）

5. fun-rec 官方实现里，胶囊层输出后还会"拼接 user_dnn_emb → MLP 降维"再当兴趣向量。
   这一步是必要的吗？什么场景下加上更有帮助？
   （提示：让兴趣向量带上"用户基础画像"——年龄/性别/地域等）

6. 自适应兴趣数 K_u' = max(1, min(K, log2(N))) 为什么对工业系统重要？
   如果一个用户只有 3 次历史行为，强行给他 8 个兴趣胶囊会怎样？
"""
