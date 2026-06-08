"""
AFM 精排模型 —— PyTorch 实现
（Attentional Factorization Machines, 2017）

【与 fun-rec 源码对齐】
- 数据集：KuaiRand-1K（和 Wide&Deep 一致）
- 核心思想：FM 对所有特征交叉一视同仁 → AFM 用注意力区分不同交叉的重要性
- 损失：BCEWithLogitsLoss（二分类 CTR）
- 评估：AUC

【AFM 解决了 FM 的什么问题？】
  FM 公式：y = linear + Σ_{i<j} <v_i, v_j> x_i x_j
                    ↑
              所有特征交叉"权重相等"

  但这不合理！
  - "年龄=25" × "视频类别=体育" → 强信号（年轻人爱看体育）
  - "年龄=25" × "上传时间=周三" → 弱信号（没有相关性）

  AFM 的改进：给每个交叉对学一个权重 a_ij，让模型自己决定"哪个交叉更重要"

【核心公式】
  y = w_0 + Σ w_i x_i   +   p^T · Σ_{i<j} a_ij · (v_i ⊙ v_j) · x_i x_j
    └── 线性部分 ──┘     └──── 注意力加权的二阶交叉 ──────┘
  Wide & Deep 的 Wide      FM 的二阶 + 注意力

  其中：
    v_i ⊙ v_j : 特征 i 和 j 的 embedding 做元素积（element-wise product）
    a_ij      : 注意力分数，由小 MLP 算出 → softmax 归一化
    p         : 投影向量，把 D 维的加权和压成 1 个数

【模型结构】
  ┌────────────────────────────────────────────────────────────┐
  │                                                            │
  │   user_id ──┐                                              │
  │   video_id ─┤                                              │
  │   author_id ┤                                              │
  │   ...       ├─→ Embedding ─→ [B, N, D]                    │
  │   tag      ─┘                            │                  │
  │                                           ▼                 │
  │              ┌──────────────────────────────────┐            │
  │              │  Pairwise Interaction Layer      │            │
  │              │  对每对 (i,j) 算 v_i ⊙ v_j       │            │
  │              │  输出 [B, num_pairs, D]          │            │
  │              └──────────────┬───────────────────┘            │
  │                             ▼                                │
  │              ┌──────────────────────────────────┐            │
  │              │  Attention Network               │            │
  │              │  每对 → Linear → ReLU → Linear   │            │
  │              │  → softmax → a_ij 注意力权重     │            │
  │              └──────────────┬───────────────────┘            │
  │                             ▼                                │
  │              Σ a_ij · (v_i ⊙ v_j) → [B, D]                  │
  │                     │                                        │
  │                     ▼ p^T · [D] → [1]                       │
  │    ┌────────────────┼────────────────┐                       │
  │    │                │                │                       │
  │  global_bias    linear_part    attention_part                │
  │    │                │                │                       │
  │    └────────────────┼────────────────┘                       │
  │                     ▼                                        │
  │               y = bias + linear + p^T · attn                 │
  │                     │                                        │
  │                     ▼ sigmoid → CTR                          │
  └──────────────────────────────────────────────────────────────┘

【任务拆解】
- TODO ①：Pairwise Interaction Layer（所有特征对元素积）
- TODO ②：Attention Network（每对 → 注意力分数）
- TODO ③：Attention-based Pooling（加权求和 + p^T 投影）
- TODO ④：完整的 AFM forward（线性 + 注意力交叉）
- TODO ⑤：训练 + AUC 评估 + 注意力可视化

运行：
    python afm_pytorch.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


# ============================================================
# 【0】配置
# ============================================================
KUAIRAND_PATH = Path("/Users/qiruihou/Desktop/学习/推荐算法/dataset/kuairand/KuaiRand-1K")

EMB_DIM = 8                    # FM 隐向量维度
ATTENTION_SIZE = 4             # 注意力隐藏层大小（fun-rec 默认）
DROPOUT_RATE = 0.1
LR = 0.001
BATCH_SIZE = 1024
N_EPOCHS = 1
SUBSAMPLE_SIZE = 300000
VALIDATION_SPLIT = 0.2

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 【1】加载 & 预处理（和 Wide&Deep 完全一样的 KuaiRand 流程）
# ============================================================
print("=" * 60)
print("【1】加载 KuaiRand-1K 数据")
print("=" * 60)

log_df = pd.read_csv(KUAIRAND_PATH / "data" / "log_standard_4_22_to_5_08_1k.csv")
user_feat = pd.read_csv(KUAIRAND_PATH / "data" / "user_features_1k.csv")
video_basic = pd.read_csv(KUAIRAND_PATH / "data" / "video_features_basic_1k.csv")
print(f"log: {len(log_df)}, user_feat: {len(user_feat)}, video_basic: {len(video_basic)}")

cols = ["user_id", "video_id", "date", "time_ms", "is_click",
        "is_like", "is_follow", "is_comment", "is_forward",
        "is_hate", "long_view", "is_profile_enter", "tab"]
log_clean = log_df[cols].copy()
print(f"正样本率: {log_clean['is_click'].mean()*100:.2f}%")

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
print(f"最终数据: {len(merged)} 行, 正样本率: {merged['is_click'].mean()*100:.2f}%")

feature_vocabs = {}
for feat in SELECT_FEATURES:
    feature_vocabs[feat] = int(merged[feat].max()) + 1

if len(merged) > SUBSAMPLE_SIZE:
    merged = merged.sample(n=SUBSAMPLE_SIZE, random_state=SEED)
    print(f"采样至 {SUBSAMPLE_SIZE} 条")

labels = torch.FloatTensor(merged["is_click"].values)
feature_tensors = {}
for feat in SELECT_FEATURES:
    feature_tensors[feat] = torch.LongTensor(merged[feat].values)

n_total = len(labels)
n_val = int(n_total * VALIDATION_SPLIT)
indices = torch.randperm(n_total)


# ============================================================
# 【2】Dataset
# ============================================================
class KuaiRandDataset(Dataset):
    def __init__(self, features, labels, indices):
        self.labels = labels[indices]
        self.features = {k: v[indices] for k, v in features.items()}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feats = {k: v[idx] for k, v in self.features.items()}
        return feats, self.labels[idx]


train_dataset = KuaiRandDataset(feature_tensors, labels, indices[n_val:])
val_dataset = KuaiRandDataset(feature_tensors, labels, indices[:n_val])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")


# ============================================================
# 【3】AFM 模型
# ============================================================
class AFM(nn.Module):
    """
    AFM = Linear Part + Attention-based Pairwise Interaction

    对照 fun-rec 源码的三个核心组件：
      - pairwise_feature_interactions → TODO ①
      - AttentionPoolingLayer        → TODO ② + ③
      - linear_logits + afm_logits   → TODO ④
    """

    def __init__(self, feature_vocabs, emb_dim, attention_size, dropout=0.1):
        super().__init__()
        self.feature_names = list(feature_vocabs.keys())
        n_features = len(self.feature_names)
        self.emb_dim = emb_dim

        # ────────── Embedding 表 ──────────
        self.embeddings = nn.ModuleDict()
        for feat_name, vocab_size in feature_vocabs.items():
            self.embeddings[feat_name] = nn.Embedding(
                vocab_size, emb_dim, padding_idx=0
            )

        # ────────── 线性偏置（和 Wide&Deep 一样） ──────────
        self.linear_weights = nn.ModuleDict()
        for feat_name, vocab_size in feature_vocabs.items():
            self.linear_weights[feat_name] = nn.Embedding(vocab_size, 1, padding_idx=0)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # ────────── 注意力相关 ──────────
        # 用于 Pairwise Interaction 的 Dropout
        self.pair_dropout = nn.Dropout(dropout)

        # ───── TODO ②：Attention Network ─────
        # 输入：每对交互向量 (v_i ⊙ v_j)，shape [B, num_pairs, D]
        # 第一层：Linear(D → attention_size)
        # 激活：ReLU
        # 第二层：Linear(attention_size → 1, bias=False)
        #
        # 提示：
        #   对照 fun-rec 源码 AttentionPoolingLayer：
        #     weighted_inputs = matmul(inputs, W) + b   ← Linear(D, A)
        #     activation = ReLU(weighted_inputs)
        #     projected = matmul(activation, h)          ← Linear(A, 1, no_bias)
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.attention_network = nn.Sequential(
            nn.Linear(emb_dim, attention_size),
            nn.ReLU(),
            nn.Linear(attention_size, 1, bias=False)
        )

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ───── TODO ③：投影层 ─────
        # 将注意力池化后的 D 维向量映射为标量
        # fun-rec 对应：Dense(1, activation=None)(attention_output)
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.projection_layer = nn.Linear(emb_dim, 1)

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, features):
        """
        输入：features = {name: [B]}
        输出：logits [B]
        """

        # ── 1. Lookup Embedding ──
        # 把所有特征的 embedding 堆成 [B, N, D]
        embs = []
        for feat_name in self.feature_names:
            embs.append(self.embeddings[feat_name](features[feat_name]))
        emb_concat = torch.stack(embs, dim=1)   # [B, N, D]

        # ── 2. 线性部分 ──
        linear_logit = self.global_bias.expand(emb_concat.size(0))
        for feat_name in self.feature_names:
            lin = self.linear_weights[feat_name](features[feat_name])
            linear_logit = linear_logit + lin.squeeze(1)

        # ── 3. TODO ①：Pairwise Interaction Layer ──
        # 对所有 i<j 计算 v_i ⊙ v_j（元素积）
        #
        # 公式：f_PI(E) = { (v_i ⊙ v_j) }_{i<j}
        #
        # 【为什么用元素积而不是内积？】
        #   内积 <v_i, v_j> = Σ_k v_i_k · v_j_k → 标量
        #   元素积 v_i ⊙ v_j = [v_i_1·v_j_1, ..., v_i_D·v_j_D] → D 维向量
        #
        #   AFM 需要向量不是标量，因为后续注意力网络需要"完整向量信息"
        #   标量信息太少了，注意力网络无法区分"这对交叉在哪些维度上重要"
        #
        # 提示：
        #   N = emb_concat.size(1)
        #   两层 for 循环 i in range(N), j in range(i+1, N)
        #   每对：emb_concat[:, i, :] * emb_concat[:, j, :]  # [B, D]
        #   用 torch.stack(pair_list, dim=1) → [B, num_pairs, D]
        #   最后 self.pair_dropout(..)
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        N = emb_concat.size(1)
        pair_list = []
        for i in range(N):
            for j in range(i+1, N):
                pair = emb_concat[:, i, :] * emb_concat[:, j, :]
                pair_list.append(pair)
        pairwise = torch.stack(pair_list, dim=1)
        pairwise = self.pair_dropout(pairwise)
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ── 4. TODO ②：Attention Network ──
        # 对每个交互对算注意力分数
        #
        # 公式：
        #   a_ij' = h^T · ReLU(W · (v_i ⊙ v_j) + b)   ← 标量分数
        #   a_ij  = softmax_j(a_ij')                    ← 沿 pairs 维归一化
        #
        # 输入：pairwise [B, num_pairs, D]
        # 输出：attn_weights [B, num_pairs, 1]
        #
        # 提示：
        #   ① self.attention_W(pairwise) → [B, num_pairs, A]
        #   ② ReLU
        #   ③ self.attention_h(..) → [B, num_pairs, 1]
        #   ④ F.softmax(.., dim=1) → [B, num_pairs, 1]
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        attn_weights = self.attention_network(pairwise)
        attn_weights = F.softmax(attn_weights, dim=1)

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ── 5. TODO ③：Attention-based Pooling + 投影 ──
        # 加权池化：Σ a_ij · (v_i ⊙ v_j) → [B, D]
        #   逐元素乘后按 pair 维求和
        #
        # 投影：p^T · pooled → 1 个标量
        #   用 self.projection
        #   最后 squeeze(-1) 去掉最后一维
        #
        # 提示：
        #   attn_weights * pairwise → [B, num_pairs, D]
        #   .sum(dim=1) → [B, D]
        #   self.projection(pooled) → [B, 1]
        #   .squeeze(1) → [B]
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        pooled = (attn_weights * pairwise).sum(dim=1)
        logits = self.projection_layer(pooled).squeeze(-1)


        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ── 6. TODO ④：融合 ──
        # y = global_bias + y_linear + p^T · Σ a_ij · (v_i ⊙ v_j)
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        logits = linear_logit + logits
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        return logits


model = AFM(
    feature_vocabs=feature_vocabs,
    emb_dim=EMB_DIM,
    attention_size=ATTENTION_SIZE,
    dropout=DROPOUT_RATE,
)
print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# 【4】TODO ⑤：训练
# ============================================================
print("\n" + "=" * 60)
print("【4】训练 AFM")
print("=" * 60)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_afm"
writer = SummaryWriter(log_dir=f"runs/{run_name}")

best_auc = 0.0
best_state = None

for epoch in range(N_EPOCHS):
    model.train()
    train_loss, n_batch = 0.0, 0
    train_preds, train_labels_list = [], []

    for features, labels in train_loader:
        # ↓↓↓↓↓ TODO ⑤：前向 → loss → 反向 → 更新 ↓↓↓↓↓
        logits = model(features)
        loss = criterion(logits, labels)
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
            logits = model(features)
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
    torch.save(best_state, "best_afm.pt")
    print(f"✅ 最佳模型（epoch {best_epoch}, AUC={best_auc:.4f}）")
    print(f"   fun-rec 官方 AFM AUC≈0.5867")


# ============================================================
# 【5】注意力权重可视化（AFM 独有的可解释性）
# ============================================================
print("\n" + "=" * 60)
print("【5】注意力权重可视化（验证 AFM 学到了有意义的注意力分布）")
print("=" * 60)

model.eval()
with torch.no_grad():
    sample_features, sample_labels = next(iter(val_loader))

    # 重新走一遍前向到 attention 那步，拿到注意力权重
    embs = []
    for feat_name in model.feature_names:
        embs.append(model.embeddings[feat_name](sample_features[feat_name]))
    emb_concat = torch.stack(embs, dim=1)

    N = emb_concat.size(1)
    pair_list = []
    for i in range(N):
        for j in range(i + 1, N):
            pair_list.append(emb_concat[:, i, :] * emb_concat[:, j, :])
    pairwise = torch.stack(pair_list, dim=1)

    # 注意：如果 TODO ② 没写完，下面这行会报错
    try:
        attn_scores = torch.relu(model.attention_W(pairwise))
        attn_scores = model.attention_h(attn_scores)
        attn_weights = F.softmax(attn_scores, dim=1).squeeze(-1)

        sample_weights = attn_weights[0].numpy()
        top_pairs = np.argsort(sample_weights)[-5:][::-1]

        pair_name_list = []
        for i in range(N):
            for j in range(i + 1, N):
                pair_name_list.append(
                    f"({model.feature_names[i]}, {model.feature_names[j]})"
                )

        print("\n第一条样本注意力 Top-5 特征交叉：")
        for rank, idx in enumerate(top_pairs):
            print(f"  {rank+1}. {pair_name_list[idx]:<45s} α={sample_weights[idx]:.4f}")
    except AttributeError:
        print("  ⚠️ 注意力网络未定义，请先完成 TODO ②")

writer.close()
print("\n✅ 训练完成！")
