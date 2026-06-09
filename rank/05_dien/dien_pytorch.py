"""
DIEN 精排模型 —— PyTorch 实现
（Deep Interest Evolution Network, Alibaba 2019）

【与 fun-rec 源码对齐】
- 数据集：KuaiRand-1K（和 DIN 一致，复用历史序列）
- 核心创新：不仅根据候选商品做 Attention，还引入 GRU 建模兴趣的"演化"
- 核心组件：
  1. InterestExtractor（兴趣提取层）：GRU 从行为序列提取兴趣状态
  2. Auxiliary Loss（辅助损失）：用下一个行为监督 GRU 学习
  3. InterestEvolution（兴趣演化层）：双线性注意力 + AUGRU
- 损失：BCEWithLogitsLoss + auxiliary_loss_weight * aux_loss
- 评估：AUC

【DIEN 解决了 DIN 什么问题？】
  DIN 的问题：直接把历史行为的 embedding 拼起来做 Attention
    → 忽略了用户兴趣是"随时间演化"的
    → 最近点击的"手机"和半年前点击的"手机"，对当前候选"手机壳"的影响应该不同

  DIEN 的答案：
    Step 1: GRU 逐时间步读取历史行为 → 得到每个时刻的"兴趣状态"
    Step 2: 辅助损失监督 GRU → 让兴趣状态能预测下一个真实行为
    Step 3: AUGRU（注意力更新 GRU）→ 根据候选商品，对不同历史时刻的兴趣加权演化

【核心结构对比】
  DIN:  history_emb ──→ Attention ──→ 加权求和 ──→ DNN
                         ↑
                      candidate_emb

  DIEN: history_emb ──→ GRU ──→ interest_states ──→ 双线性 Attention ──→ AUGRU ──→ DNN
          ↓                              ↑                                ↑
      aux_loss(辅助)              candidate_emb                    candidate_emb

【AUGRU 是什么？】
  标准 GRU 的更新门：z_t ∈ [0,1]^d
  AUGRU 的更新门：z_t' = a_t * z_t   （a_t 是候选商品对该时刻的注意力分数）

  含义：
    - a_t 高（候选商品和历史行为相关）→ z_t' 大 → 隐藏状态更新多
    - a_t 低（不相关）→ z_t' 小 → 隐藏状态几乎不更新（遗忘掉这段历史）

【辅助损失 Auxiliary Loss】
  GRU 容易过拟合或学不到真正的兴趣表示。
  辅助损失的做法：
    - 正样本：用时刻 t 的兴趣状态 h_t，预测时刻 t+1 的真实行为 → 应该匹配
    - 负样本：用时刻 t 的兴趣状态 h_t，预测时刻 t+1 的随机行为 → 不应该匹配
  这样强迫 h_t 编码"用户接下来会做什么"的信息。

【和 DIN 的代码差异】
  - 新增 InterestExtractor（GRU + Aux MLP）
  - 新增 AUGRUCell（自定义 GRU cell）
  - 新增 InterestEvolution（双线性注意力 + AUGRU）
  - 训练循环需要同时优化 BCE 和 aux_loss

运行：
    python dien_pytorch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


# ============================================================
# 【0】配置
# ============================================================
KUAIRAND_PATH = Path("/Users/qiruihou/Desktop/学习/推荐算法/dataset/kuairand/KuaiRand-1K")

EMB_DIM = 8
GRU_HIDDEN = 64                # DIEN 兴趣提取 GRU 的隐层维度（fun-rec 默认 64）
DNN_UNITS = [128, 64]
DROPOUT_RATE = 0.1
LR = 0.001
BATCH_SIZE = 256
N_EPOCHS = 2
SUBSAMPLE_SIZE = 100000
VALIDATION_SPLIT = 0.2
HISTORY_LEN = 20
AUX_LOSS_WEIGHT = 0.1          # 辅助损失权重

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 【1】加载 & 预处理（复用 DIN，注意提前算 vocab 再采样）
# ============================================================
print("=" * 60)
print("【1】加载 KuaiRand-1K 数据（含历史序列，用于 DIEN）")
print("=" * 60)

# ── 1.1 读 CSV ──
log_df = pd.read_csv(KUAIRAND_PATH / "data" / "log_standard_4_22_to_5_08_1k.csv")
user_feat = pd.read_csv(KUAIRAND_PATH / "data" / "user_features_1k.csv")
video_basic = pd.read_csv(KUAIRAND_PATH / "data" / "video_features_basic_1k.csv")

# ── 1.2 用户特征处理 ──
user_cols = ["user_id", "user_active_degree", "is_live_streamer", "is_video_author",
             "follow_user_num_range", "fans_user_num_range",
             "friend_user_num_range", "register_days_range"]
user_clean = user_feat[user_cols].copy()
user_clean["is_live_streamer"] = user_clean["is_live_streamer"].apply(
    lambda x: 0 if x == -124 else x
)
str_feats = ["user_id", "user_active_degree", "follow_user_num_range",
             "fans_user_num_range", "friend_user_num_range", "register_days_range"]
for feat in str_feats:
    le = LabelEncoder()
    user_clean[feat + "_enc"] = le.fit_transform(user_clean[feat].astype(str)) + 1
    if feat != "user_id":
        user_clean[feat] = user_clean[feat + "_enc"]
        del user_clean[feat + "_enc"]

# ── 1.3 视频特征处理 ──
video_cols = ["video_id", "author_id", "video_type", "upload_type",
              "visible_status", "music_id", "music_type", "tag"]
video_basic_clean = video_basic[video_cols].copy()
for feat in ["visible_status", "music_type"]:
    max_val = video_basic_clean[feat].max()
    video_basic_clean[feat] = video_basic_clean[feat].fillna(
        max_val + 1 if pd.notna(max_val) else 0
    ).astype(int)
for feat in ["video_id", "author_id", "video_type", "upload_type",
             "visible_status", "music_id", "music_type"]:
    le = LabelEncoder()
    video_basic_clean[feat + "_enc"] = le.fit_transform(
        video_basic_clean[feat].astype(str)
    ) + 1
    if feat != "video_id":
        video_basic_clean[feat] = video_basic_clean[feat + "_enc"]
        del video_basic_clean[feat + "_enc"]
video_basic_clean["tag"] = video_basic_clean["tag"].fillna("-1")
tag_set = set()
for x in video_basic_clean["tag"].values:
    for t in str(x).split(","):
        tag_set.add(t)
tag_map = {t: i+1 for i, t in enumerate(sorted(tag_set))}
video_basic_clean["tag"] = video_basic_clean["tag"].apply(
    lambda x: tag_map.get(str(x).split(",")[0], 0)
)

# ── 1.4 合并基础特征 ──
cols = ["user_id", "video_id", "date", "time_ms", "is_click", "tab"]
log_clean = log_df[cols].copy()

merged = log_clean.merge(user_clean, on="user_id", how="left")
merged = merged.merge(video_basic_clean, on="video_id", how="left")
merged["user_id"] = merged["user_id_enc"]
merged["video_id"] = merged["video_id_enc"]
for col in ["user_id_enc", "video_id_enc"]:
    if col in merged.columns:
        del merged[col]
merged["tag"] = merged["tag"].fillna(0).astype(int)

SELECT_FEATURES = [
    "user_id", "video_id", "user_active_degree", "is_live_streamer",
    "is_video_author", "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range", "author_id",
    "video_type", "upload_type", "visible_status", "music_id",
    "music_type", "tag",
]
for feat in SELECT_FEATURES:
    if feat not in merged.columns:
        merged[feat] = 0
    merged[feat] = merged[feat].fillna(0).astype(int)

main_tabs = set([1, 0, 4, 2, 6])
merged = merged[merged["tab"].isin(main_tabs)]
print(f"基础数据: {len(merged)} 行, 正样本率: {merged['is_click'].mean()*100:.2f}%")

# ── 1.5 TODO ①：构造历史序列 ──
# 和 DIN 完全一致。注意：历史序列在采样前构建，vocab 也要在采样前算。
# 提示：
#   ① 按 user_id + time_ms 排序
#   ② 用 defaultdict(list) 累积每个用户的点击历史
#   ③ 对每个样本，取该用户在此之前点击过的 video_id（最多 HISTORY_LEN 个）
#   ④ 左填充到 HISTORY_LEN，pad=0
#   ⑤ mask：有效位置=1，pad=0
#   ⑥ 存到 merged_sorted["history_seq"] 和 merged_sorted["history_mask"]
# ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
merged_sorted = merged.sort_values(by=["user_id", "time_ms"]).reset_index(drop=True)
user_history = defaultdict(list)
user_seq_feats = []
for _, row in merged_sorted.iterrows():
    uid = int(row["user_id"])
    vid = int(row["video_id"])
    past = user_history[uid][-(HISTORY_LEN-1):] if user_history[uid] else []
    hist = [0]*(HISTORY_LEN-len(past)) + past if len(past) < HISTORY_LEN else past
    mask = [1 if x!=0 else 0 for x in hist]
    user_seq_feats.append([hist, mask])
    user_history[uid].append(vid)

merged_sorted["history_seq"] = [x[0] for x in user_seq_feats]
merged_sorted["history_mask"] = [x[1] for x in user_seq_feats]
# ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

# ── 1.6 采样 & 特征字典 ──
# 【关键修复】先算 vocab（基于全部数据），再采样，防止 history_seq 里的视频ID越界
feature_vocabs = {}
for feat in SELECT_FEATURES:
    feature_vocabs[feat] = int(merged_sorted[feat].max()) + 1
feature_vocabs["video_id_for_hist"] = feature_vocabs["video_id"]

if len(merged_sorted) > SUBSAMPLE_SIZE:
    merged_sorted = merged_sorted.sample(n=SUBSAMPLE_SIZE, random_state=SEED)

labels = torch.FloatTensor(merged_sorted["is_click"].values)
feature_tensors = {}
for feat in SELECT_FEATURES:
    feature_tensors[feat] = torch.LongTensor(merged_sorted[feat].values)
feature_tensors["history_seq"] = torch.LongTensor(
    np.stack(merged_sorted["history_seq"].values)
)
feature_tensors["history_mask"] = torch.BoolTensor(
    np.stack(merged_sorted["history_mask"].values)
)

n_total = len(labels)
n_val = int(n_total * VALIDATION_SPLIT)
indices = torch.randperm(n_total)


# ============================================================
# 【2】Dataset
# ============================================================
class KuaiRandSeqDataset(Dataset):
    def __init__(self, features, labels, indices):
        self.labels = labels[indices]
        self.features = {k: v[indices] for k, v in features.items()}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feats = {k: v[idx] for k, v in self.features.items()}
        return feats, self.labels[idx]


train_dataset = KuaiRandSeqDataset(feature_tensors, labels, indices[n_val:])
val_dataset = KuaiRandSeqDataset(feature_tensors, labels, indices[:n_val])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")


# ============================================================
# 【3】DIEN 核心模块
# ============================================================

class InterestExtractor(nn.Module):
    """
    兴趣提取层：GRU 读取行为序列，输出每个时刻的兴趣状态。

    输入：
      behavior_emb: [B, L, D]   ← 历史行为的 embedding
      mask:         [B, L]      ← 填充掩码
      neg_behavior_emb: [B, L, D]（可选，训练时传入，用于辅助损失）

    输出：
      interest_states: [B, L, H]  ← 每个时刻的 GRU 隐藏状态
      aux_loss: 标量（如果 neg_behavior_emb 不为 None，否则为 None）

    辅助损失原理：
      - 取 interest_states[:, :-1, :]（当前兴趣）
      - 取 behavior_emb[:, 1:, :]（下一个真实行为，正样本）
      - 取 neg_behavior_emb[:, 1:, :]（下一个随机行为，负样本）
      - MLP 输入 = concat([interest, next_behavior])
      - 正样本预测 → 应该接近 1；负样本预测 → 应该接近 0
      - 损失 = -[log(pos) + log(1 - neg)]，只算有效位置（mask）
    """

    def __init__(self, emb_dim, hidden_dim, use_aux_loss=True, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_aux_loss = use_aux_loss

        # ── GRU：从行为 embedding 提取兴趣状态 ──
        # 输入 [B, L, emb_dim] → 输出 [B, L, hidden_dim]
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)

        # ── 辅助损失 MLP（只在 use_aux_loss=True 时创建）──
        # 输入 = concat([interest_state, next_behavior_emb])
        #     形状 [B, L-1, hidden_dim + emb_dim]
        # 输出 1 个 logits（用 sigmoid 转概率）
        if use_aux_loss:
            self.aux_mlp = nn.Sequential(
                nn.Linear(hidden_dim + emb_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )
        else:
            self.aux_mlp = None

    def forward(self, behavior_emb, mask, neg_behavior_emb=None):
        """
        提示：
          ① behavior_emb 过 self.gru → interest_states, _
          ② 如果 training 且 neg_behavior_emb 不为 None：
             - current = interest_states[:, :-1, :]      # [B, L-1, H]
             - next_pos = behavior_emb[:, 1:, :]         # [B, L-1, D]
             - next_neg = neg_behavior_emb[:, 1:, :]     # [B, L-1, D]
             - pos_input = concat([current, next_pos], dim=-1)  # [B, L-1, H+D]
             - neg_input = concat([current, next_neg], dim=-1)  # [B, L-1, H+D]
             - pos_prob = sigmoid(self.aux_mlp(pos_input))      # [B, L-1, 1]
             - neg_prob = sigmoid(self.aux_mlp(neg_input))      # [B, L-1, 1]
             - mask_slice = mask[:, 1:].unsqueeze(-1).float()   # [B, L-1, 1]
             - aux_loss = -(log(pos_prob + 1e-8) + log(1 - neg_prob + 1e-8)) * mask_slice
             - aux_loss = aux_loss.sum() / mask_slice.sum()     # 平均
             - return interest_states, aux_loss
          ③ 否则 return interest_states, None
        """
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        interest_states, _ = self.gru(behavior_emb)
        if self.training and neg_behavior_emb is not None:
            current = interest_states[:, :-1, :]      # [B, L-1, H]
            next_pos = behavior_emb[:, 1:, :]         # [B, L-1, D]
            next_neg = neg_behavior_emb[:, 1:, :]     # [B, L-1, D]
            pos_input = torch.cat([current, next_pos], dim=-1)  # [B, L-1, H+D]
            neg_input = torch.cat([current, next_neg], dim=-1)  # [B, L-1, H+D]
            pos_prob = torch.sigmoid(self.aux_mlp(pos_input))      # [B, L-1, 1]
            neg_prob = torch.sigmoid(self.aux_mlp(neg_input))      # [B, L-1, 1]
            mask_slice = mask[:, 1:].unsqueeze(-1).float()   # [B, L-1, 1]
            aux_loss = -(torch.log(pos_prob + 1e-8) + torch.log(1 - neg_prob + 1e-8)) * mask_slice
            aux_loss = aux_loss.sum() / (mask_slice.sum() + 1e-8)  # 标量
            return interest_states, aux_loss
        else: 
            return interest_states, None
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


class AUGRUCell(nn.Module):
    """
    AUGRU 单步计算单元（Attentional Update GRU Cell）

    和标准 GRUCell 的区别：
      更新门 z_t 会被 attention score a_t 缩放：z_t' = a_t * z_t

    公式：
      z = sigmoid(W_z @ [x, h_prev] + b_z)
      r = sigmoid(W_r @ [x, h_prev] + b_r)
      h_tilde = tanh(W_h @ [x, r * h_prev] + b_h)
      z' = a * z                           ← 关键！attention 缩放更新门
      h = (1 - z') * h_prev + z' * h_tilde

    输入：
      x:          [B, input_dim]   ← 当前时刻的输入
      h_prev:     [B, hidden_dim]  ← 上一时刻的隐藏状态
      attn_score: [B, 1]           ← 当前时刻的注意力分数（来自候选商品）

    输出：
      h_new: [B, hidden_dim]
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── 三个门控的线性层 ──
        # 输入都是 concat([x, h_prev])，所以输入维度 = input_dim + hidden_dim
        # 提示：分别创建 update_gate, reset_gate, candidate 三个 nn.Linear
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, x, h_prev, attn_score):
        """
        提示：
          ① torch.cat([x, h_prev], dim=-1) 得到 [B, input_dim + hidden_dim]
          ② update_gate = sigmoid(self.update_gate(concat))  # [B, hidden_dim]
          ③ reset_gate  = sigmoid(self.reset_gate(concat))   # [B, hidden_dim]
          ④ candidate_input = cat([x, reset_gate * h_prev], dim=-1)
          ⑤ candidate = tanh(self.candidate(candidate_input))  # [B, hidden_dim]
          ⑥ 【AUGRU 核心】update_gate = attn_score * update_gate  # [B, hidden_dim]
          ⑦ h_new = (1 - update_gate) * h_prev + update_gate * candidate
        """
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        concat = torch.cat([x, h_prev], dim=-1)
        update_gate = torch.sigmoid(self.update_gate(concat))  # [B, hidden_dim]
        reset_gate = torch.sigmoid(self.reset_gate(concat))   # [B, hidden_dim]
        candidate_input = torch.cat([x, reset_gate * h_prev], dim=-1)
        candidate = torch.tanh(self.candidate(candidate_input))  # [B, hidden_dim]
        update_gate = attn_score * update_gate  # [B, hidden_dim]
        h_new = (1 - update_gate) * h_prev + update_gate * candidate 

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑
        return h_new


class InterestEvolution(nn.Module):
    """
    兴趣演化层：双线性注意力 + AUGRU

    输入：
      interest_states: [B, L, hidden_dim]  ← 来自 InterestExtractor
      target_emb:      [B, emb_dim]        ← 候选商品的 embedding
      mask:            [B, L]               ← 填充掩码

    计算：
      1. 双线性注意力：
         h_W = interest_states @ W              # [B, L, emb_dim]
         scores = (h_W * target_emb.unsqueeze(1)).sum(-1)  # [B, L]
         attn = softmax(scores.masked_fill(~mask, -1e9), dim=1)  # [B, L]

      2. AUGRU 逐时间步演化：
         h = zeros([B, hidden_dim])
         for t in range(L):
             h = augru_cell(interest_states[:, t, :], h, attn[:, t].unsqueeze(1))
         return h  # [B, hidden_dim]
    """

    def __init__(self, hidden_dim, emb_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim

        # ── 双线性注意力权重 W：[hidden_dim, emb_dim] ──
        # 用于计算 scores = interest_states @ W @ target_emb^T
        self.bilinear = nn.Parameter(torch.randn(hidden_dim, emb_dim) * 0.01)

        # ── AUGRU Cell ──
        self.augru_cell = AUGRUCell(hidden_dim, hidden_dim)

    def forward(self, interest_states, target_emb, mask):
        """
        提示：
          ① 双线性注意力计算
             h_W = torch.matmul(interest_states, self.bilinear)  # [B, L, emb_dim]
             target_expanded = target_emb.unsqueeze(1)             # [B, 1, emb_dim]
             scores = (h_W * target_expanded).sum(dim=-1)          # [B, L]
             scores = scores.masked_fill(~mask, -1e9)              # pad 设极小值
             attn = F.softmax(scores, dim=1)                       # [B, L]

          ② AUGRU 逐时间步演化
             h = torch.zeros(B, self.hidden_dim, device=...)
             for t in range(L):
                 x_t = interest_states[:, t, :]          # [B, hidden_dim]
                 a_t = attn[:, t].unsqueeze(1)           # [B, 1]
                 h = self.augru_cell(x_t, h, a_t)
             return h  # [B, hidden_dim]
        """
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        h_W = torch.matmul(interest_states, self.bilinear)  # [B, L, emb_dim]
        target_expanded = target_emb.unsqueeze(1)             # [B, 1, emb_dim]
        scores = (h_W * target_expanded).sum(dim=-1)          # [B, L]
        scores = scores.masked_fill(~mask, -1e9)              # pad 设极小值
        attn = F.softmax(scores, dim=1)                       # [B, L]

        B, L, _ = interest_states.shape
        h = torch.zeros(B, self.hidden_dim, device=interest_states.device)
        for t in range(L):
            x_t = interest_states[:, t, :]          # [B, hidden_dim]
            a_t = attn[:, t].unsqueeze(1)           # [B, 1]
            h = self.augru_cell(x_t, h, a_t)
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑
        return h


class DIEN(nn.Module):
    """
    DIEN = 线性部分 + 其他特征 + DIEN 演化兴趣 → concat → DNN

    结构：
      ① behavior_emb = hist_embedding(history_seq)        # [B, L, D]
      ② interest_states, aux_loss = Extractor(behavior_emb, mask, neg_behavior_emb)  # [B, L, H]
      ③ candidate_emb = embedding("video_id")              # [B, D]
      ④ evolved_interest = Evolution(interest_states, candidate_emb, mask)  # [B, H]
      ⑤ concat([静态特征flatten, evolved_interest]) → DNN → logits
    """

    def __init__(self, feature_vocabs, emb_dim, hidden_dim, dnn_units,
                 use_aux_loss=True, dropout=0.1):
        super().__init__()
        self.feature_names = [f for f in feature_vocabs.keys()
                              if f not in ("history_seq", "history_mask", "video_id_for_hist")]
        self.n_features = len(self.feature_names)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_aux_loss = use_aux_loss

        # ── Embedding 表 ──
        self.embeddings = nn.ModuleDict()
        for feat_name in self.feature_names:
            self.embeddings[feat_name] = nn.Embedding(
                feature_vocabs[feat_name], emb_dim, padding_idx=0
            )

        # ── 历史序列用单独的 Embedding 表 ──
        self.hist_embedding = nn.Embedding(
            feature_vocabs["video_id_for_hist"], emb_dim, padding_idx=0
        )

        # ── 线性偏置 ──
        self.linear_weights = nn.ModuleDict()
        for feat_name in self.feature_names:
            self.linear_weights[feat_name] = nn.Embedding(
                feature_vocabs[feat_name], 1, padding_idx=0
            )
        self.global_bias = nn.Parameter(torch.zeros(1))

        # ── TODO ②：兴趣提取层 ──
        # 创建 InterestExtractor(emb_dim, hidden_dim, use_aux_loss, dropout)
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.interest_extractor = InterestExtractor(emb_dim, hidden_dim, use_aux_loss, dropout)
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ── TODO ④：兴趣演化层 ──
        # 创建 InterestEvolution(hidden_dim, emb_dim)
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.interest_evolution = InterestEvolution(hidden_dim, emb_dim)

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ── TODO ⑤：DNN 部分 ──
        # 输入维度 = 静态特征展平 (n_features * emb_dim) + 演化兴趣 (hidden_dim)
        # 结构：Linear → BN → PReLU(units) → Dropout，循环 dnn_units
        # 最后 Linear(dnn_units[-1] → 1)
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        dnn_input_dim = self.n_features * emb_dim + hidden_dim
        dnn_layers = []
        for uint in dnn_units:
            dnn_layers.extend([
                nn.Linear(dnn_input_dim, uint),
                nn.BatchNorm1d(uint),
                nn.PReLU(uint),
                nn.Dropout(dropout)
            ])
            dnn_input_dim = uint
        self.dnn = nn.Sequential(*dnn_layers)
        self.dnn_output = nn.Linear(dnn_units[-1], 1)
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, features):
        """
        输入：features 字典，包含所有 SELECT_FEATURES + history_seq + history_mask
        输出：logits [B], aux_loss（训练时）或 None（验证时）
        """
        # ── 1. 静态特征 → Embedding ──
        embs = []
        for feat_name in self.feature_names:
            embs.append(self.embeddings[feat_name](features[feat_name]))
        static_emb = torch.stack(embs, dim=1)   # [B, N, D]

        # ── 2. 线性部分 ──
        linear_logit = self.global_bias.expand(static_emb.size(0))
        for feat_name in self.feature_names:
            lin = self.linear_weights[feat_name](features[feat_name])
            linear_logit = linear_logit + lin.squeeze(1)

        # ── 3. 候选商品 embedding（query） ──
        candidate_emb = self.embeddings["video_id"](features["video_id"])  # [B, D]

        # ── 4. 历史行为 embedding ──
        behavior_emb = self.hist_embedding(features["history_seq"])  # [B, L, D]
        mask = features["history_mask"]                               # [B, L]

        # ── 5. TODO ⑤：组装 DIEN 核心逻辑 ──
        # 提示：
        #   ① 如果 training 且 use_aux_loss：
        #        neg_behavior_emb = behavior_emb[torch.randperm(behavior_emb.size(0))]
        #        interest_states, aux_loss = self.extractor(behavior_emb, mask, neg_behavior_emb)
        #     否则：
        #        interest_states, aux_loss = self.extractor(behavior_emb, mask)
        #   ② evolved_interest = self.evolution(interest_states, candidate_emb, mask)  # [B, H]
        #   ③ flatten = static_emb.view(B, -1)  # [B, N*D]
        #   ④ dnn_input = concat([flatten, evolved_interest], dim=-1)  # [B, N*D + H]
        #   ⑤ dnn_output = self.dnn(dnn_input)  # [B, dnn_units[-1]]
        #   ⑥ dnn_logit = self.dnn_output(dnn_output).squeeze(-1)  # [B]
        #   ⑦ logits = linear_logit + dnn_logit  # [B]
        #   ⑧ return logits, aux_loss
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        if self.training and self.use_aux_loss:
            neg_behavior_emb = behavior_emb[torch.randperm(behavior_emb.size(0))]
            interest_states, aux_loss = self.interest_extractor(behavior_emb, mask, neg_behavior_emb)
        else:
            interest_states, aux_loss = self.interest_extractor(behavior_emb, mask)

        evolved_interest = self.interest_evolution(interest_states, candidate_emb, mask)
        flat = static_emb.view(static_emb.size(0), -1)
        dnn_input = torch.cat([flat, evolved_interest], dim=-1)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_output(dnn_output).squeeze(-1)
        logits = linear_logit + dnn_logit
        return logits, aux_loss
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

model = DIEN(
    feature_vocabs=feature_vocabs,
    emb_dim=EMB_DIM,
    hidden_dim=GRU_HIDDEN,
    dnn_units=DNN_UNITS,
    use_aux_loss=True,
    dropout=DROPOUT_RATE,
)
print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# 【4】TODO ⑥：训练
# ============================================================
print("\n" + "=" * 60)
print("【4】训练 DIEN")
print("=" * 60)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_dien"
writer = SummaryWriter(log_dir=f"runs/{run_name}")

best_auc = 0.0
best_state = None

for epoch in range(N_EPOCHS):
    model.train()
    train_loss, n_batch = 0.0, 0
    train_preds, train_labels_list = [], []

    for features, labels in train_loader:
        # ── TODO ⑥：训练循环 ──
        # 提示：
        #   ① logits, aux_loss = model(features)
        #   ② loss = criterion(logits, labels)
        #   ③ 如果 aux_loss 不为 None：loss = loss + AUX_LOSS_WEIGHT * aux_loss
        #   ④ optimizer.zero_grad(); loss.backward(); optimizer.step()
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        logits, aux_loss = model(features)
        loss = criterion(logits, labels)
        if aux_loss is not None:
            loss = loss + AUX_LOSS_WEIGHT * aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        train_loss += loss.item()
        n_batch += 1
        train_preds.append(torch.sigmoid(logits).detach())
        train_labels_list.append(labels)

    train_loss /= n_batch
    train_preds = torch.cat(train_preds).numpy()
    train_labels_arr = torch.cat(train_labels_list).numpy()
    train_auc = roc_auc_score(train_labels_arr, train_preds)

    model.eval()
    val_loss, v_batch = 0.0, 0
    val_preds, val_labels_list = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            logits, _ = model(features)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            v_batch += 1
            val_preds.append(torch.sigmoid(logits))
            val_labels_list.append(labels)

    val_loss /= v_batch
    val_preds = torch.cat(val_preds).numpy()
    val_labels_arr = torch.cat(val_labels_list).numpy()
    val_auc = roc_auc_score(val_labels_arr, val_preds)

    print(f"Epoch {epoch+1:2d}/{N_EPOCHS} | "
          f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} | "
          f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("AUC/train", train_auc, epoch)
    writer.add_scalar("AUC/val", val_auc, epoch)

    if val_auc > best_auc:
        best_auc = val_auc
        best_epoch = epoch + 1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

if best_state:
    model.load_state_dict(best_state)
    torch.save(best_state, "best_dien.pt")
    print(f"最佳模型（epoch {best_epoch}, AUC={best_auc:.4f}）")

writer.close()
print("\n训练完成！")


# ============================================================
# 【5】TODO ⑦：业务分析 —— 看看 DIEN 学到了什么
# ============================================================
print("\n" + "=" * 60)
print("【5】DIEN 业务案例分析（精排模型的可解释性）")
print("=" * 60)

"""
【为什么要做业务分析？】
  训练指标（AUC）只能说明模型整体好坏，但不知道它"为什么"对。
  精排模型的业务价值：告诉运营/产品——
    - 这个用户点击概率 90%，因为最近看了 3 个同类视频
    - 那个用户点击概率 10%，因为历史行为和候选商品完全不相关

【DIEN 特有的可解释性】
  1. 双线性注意力权重：候选视频对用户历史每个行为的关注度
  2. AUGRU 演化后的兴趣向量：模型最终提炼出的"用户兴趣"长什么样
  3. 辅助损失概率：GRU 认为"下一个行为是正的"的信心程度

【你需要做的前置修改】
  当前 DIEN.forward 只返回 logits 和 aux_loss。
  为了拿到中间结果，你需要在 DIEN 类里加一个方法，比如：

    def analyze(self, features):
        # 和 forward 一样的逻辑，但额外返回中间结果
        behavior_emb = self.hist_embedding(features["history_seq"])   # [B, L, D]
        mask = features["history_mask"]                                # [B, L]
        candidate_emb = self.embeddings["video_id"](features["video_id"])  # [B, D]
        interest_states, _ = self.extractor(behavior_emb, mask)      # [B, L, H]
        evolved_interest, attn_weights = self.evolution.analyze(...)  # 需要修改 InterestEvolution
        return logits, evolved_interest, attn_weights, interest_states

  或者简单点：修改 InterestEvolution.forward，让它也返回 attention 权重。
"""

model.eval()
with torch.no_grad():
    # 先拿一整批验证数据做预测
    all_val_features, all_val_labels = [], []
    for feats, labs in val_loader:
        all_val_features.append({k: v.clone() for k, v in feats.items()})
        all_val_labels.append(labs.clone())
        if sum(len(v) for v in all_val_labels) >= 5000:
            break

    # 合并成一个大 batch
    merged_feats = {}
    for k in all_val_features[0].keys():
        merged_feats[k] = torch.cat([b[k] for b in all_val_features], dim=0)
    merged_labels = torch.cat(all_val_labels, dim=0)

    # TODO ⑦：选 3 个有代表性的样本做深度分析
    # 提示：
    #   ① 用 model(merged_feats) 或 model.analyze(merged_feats) 拿到预测
    #   ② probs = torch.sigmoid(logits)  # [N]
    #   ③ 选样本：
    #        - 正样本里预测最高的（模型"很有信心"）
    #        - 负样本里预测最低的（模型"很有信心不点"）
    #        - 正样本里预测最低的（模型"判断错了"，难样本）
    #   ④ 对每个样本，打印：
    #        user_id, video_id, true_label, predicted_ctr
    #        history_seq（最近 HISTORY_LEN 个点击的视频ID）
    #        attention_weights（候选视频对每个历史行为的关注度）
    #        evolved_interest[:5]（最终兴趣向量的前5维）
    #   ⑤ 业务解读：
    #        - "attention 集中在第 3、7 个历史行为，都是同类视频"
    #        - "attention 很分散，说明候选视频和用户历史兴趣不匹配"
    # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
    logits, _ = model(merged_feats)
    probs = torch.sigmoid(logits).squeeze()  # [N]

    pos_mask = (merged_labels == 1)
    neg_mask = (merged_labels == 0)

    pos_probs = probs[pos_mask]
    pos_indices = torch.where(pos_mask)[0]
    best_pos = pos_indices[torch.argmax(pos_probs)]
    worst_pos = pos_indices[torch.argmin(pos_probs)]

    neg_probs = probs[neg_mask]
    neg_indices = torch.where(neg_mask)[0]
    best_neg = neg_indices[torch.argmin(neg_probs)]

    sample_indices = [best_pos, worst_pos, best_neg]
    sample_labels = ["正样本-高置信", "正样本-难样本(误判)", "负样本-高置信"]

    for sample_idx, label_desc in zip(sample_indices, sample_labels):
        i = sample_idx.item()
        print(f"\n{'='*60}")
        print(f"【{label_desc}】")
        print(f"{'='*60}")

        user_id = merged_feats["user_id"][i].item()
        video_id = merged_feats["video_id"][i].item()
        true_label = int(merged_labels[i].item())
        pred_ctr = probs[i].item()

        print(f"user_id={user_id}, candidate_video_id={video_id}")
        print(f"真实标签={true_label}, DIEN预测CTR={pred_ctr:.4f}")

        hist_seq = merged_feats["history_seq"][i].tolist()
        hist_mask = merged_feats["history_mask"][i].tolist()
        real_hist = [v for v, m in zip(hist_seq, hist_mask) if m == 1]
        print(f"历史序列（有效{len(real_hist)}个）: {hist_seq}")
        print(f"  其中有效历史 (mask=1): {real_hist}")

        single = {k: v[i:i+1] for k, v in merged_feats.items()}
        behavior_emb = model.hist_embedding(single["history_seq"])    # [1, L, D]
        mask_1 = single["history_mask"]                                # [1, L]
        candidate_emb = model.embeddings["video_id"](single["video_id"])  # [1, D]

        interest_states, _ = model.interest_extractor(behavior_emb, mask_1)  # [1, L, H]

        h_W = torch.matmul(interest_states, model.interest_evolution.bilinear)  # [1, L, emb_dim]
        scores = (h_W * candidate_emb.unsqueeze(1)).sum(dim=-1)            # [1, L]
        scores = scores.masked_fill(~mask_1, -1e9)
        attn_weights = F.softmax(scores, dim=1).squeeze(0)                 # [L]

        print(f"\n双线性注意力权重（候选商品对每个历史行为的关注度）:")
        for t in range(HISTORY_LEN):
            vid = hist_seq[t]
            w = attn_weights[t].item()
            bar = "█" * int(w * 40)
            tag = "(有效)" if hist_mask[t] else "(pad)"
            print(f"  t={t:2d} video_id={vid:4d} {tag:<6s} attn={w:.4f} {bar}")

        max_t = torch.argmax(attn_weights).item()
        print(f"\n业务解读:")
        if hist_mask[max_t]:
            print(f"  → DIEN最关注第{max_t}个历史行为(video_id={hist_seq[max_t]})，和候选最相似")
        mask_t = torch.tensor(hist_mask, dtype=torch.bool)
        mean_attn = attn_weights[mask_t].mean().item() if mask_t.sum() > 0 else 0.0
        if mean_attn > 0.1:
            print(f"  → 注意力集中在有效历史上(均值={mean_attn:.4f})，候选与用户兴趣匹配")
        else:
            print(f"  → 注意力分散(均值={mean_attn:.4f})，候选可能不匹配用户兴趣")
        if pred_ctr > 0.5 and true_label == 1:
            print(f"  → 模型正确预测高CTR，排序时可排到前排")
        elif pred_ctr < 0.3 and true_label == 0:
            print(f"  → 模型正确识别了不感兴趣的内容")
        elif true_label == 1 and pred_ctr < 0.3:
            print(f"  → ⚠ 误判样本！实际点击但DIEN给了低分，值得排查")
    # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

