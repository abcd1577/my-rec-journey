"""
DIN 精排模型 —— PyTorch 实现
（Deep Interest Network, Alibaba 2018）

【与 fun-rec 源码对齐】
- 数据集：KuaiRand-1K（和之前一致，但新增序列特征）
- 核心创新：用户兴趣不是固定的，而是根据候选商品动态变化的
- 核心组件：DinAttentionLayer（注意力机制）
- 损失：BCEWithLogitsLoss（二分类 CTR）
- 评估：AUC

【DIN 解决了之前模型什么问题？】
  之前 DeepFM / AFM 处理用户历史行为的方式：
    mean([v_1, v_2, ..., v_N]) → 一个固定向量
                                ↑
                    无论推荐"跑鞋"还是"手机"都用同一个向量

  但这不合理！
  推荐"跑鞋"时 → 应该更关注历史上和运动相关的点击
  推荐"手机"时 → 应该更关注历史上和数码相关的点击

  DIN 的答案：让候选商品当"query"，历史行为当"keys"
  通过注意力机制动态算权重 → 不同的候选商品得到不同的用户表达

【核心公式】
  v_U(A) = Σ a(e_j, v_A) · e_j

  其中：
    v_A   = 候选商品 embedding（query）
    e_j   = 第 j 个历史行为的 embedding（key）
    a(·)  = 注意力网络，输入 concat(e_j, v_A, e_j-v_A, e_j*v_A)

【关键设计：不做 softmax】
  一般的注意力：softmax → 权重和为 1 → 丢失强度信息
  DIN 的注意力：只乘 mask，不做 softmax → 保留兴趣强度
  权重和可以 >1 或 <1，表达"用户对这个候选商品有多感兴趣"

【和 AFM 注意力的区别】
  ┌────────────┬────────────────────┬──────────────────────┐
  │            │  AFM               │  DIN                 │
  ├────────────┼────────────────────┼──────────────────────┤
  │ query      │ 无（直接对特征交叉）│ 候选商品 embedding   │
  │ keys       │ 特征 embedding 对   │ 历史行为 embedding   │
  │ attention │ 用于二阶交叉加权    │ 用于历史行为加权     │
  │ 归一化     │ softmax             │ 不做 softmax         │
  │ 输出       │ 加权交叉向量 → p^T  │ 加权历史向量 [B, D]  │
  └────────────┴────────────────────┴──────────────────────┘

【模型结构】
  输入特征:
    稀疏特征 (user_id, video_id, ...) → Embedding → concat → 其他特征向量
    历史序列 (last_k_clicked_items)   → Embedding → [B, L, D]
    候选视频 (video_id)               → Embedding → [B, D]  ← query
                                          │
                    ┌─────────────────────┘
                    ▼
            ┌───────────────┐
            │ DIN Attention │
            │ [query, keys,  │
            │  query-keys,   │
            │  query*keys]   │
            │ → MLP → scores │
            │ → sum(scores   │
            │     * keys)    │
            └───────┬───────┘
                    ▼
             DIN输出 [B, D]
                    │
      ┌─────────────┼─────────────┐
      │             │             │
     线性偏置     其他特征     DIN序列特征
      │             │             │
      └─────────────┼─────────────┘
                    ▼
            concat → DNN → sigmoid

【数据格式区别】
  之前：每条样本 = (user, video, 静态特征...)
  现在：每条样本 = (user, video, 静态特征..., history_seq, target)

  新增字段：
    history_seq: [20]  ← 用户最近 20 个点击的视频 ID
    mask:        [20]  ← padding 掩码（有历史的=1, pad=0）

【任务拆解】
- TODO ①：构建带序列特征的数据集（history_seq + mask）
- TODO ②：DIN Attention Layer（核心！query, keys, query-keys, query*keys）
- TODO ③：融合 DIN 输出 + 其他特征 → DNN
- TODO ④：训练循环

运行：
    python din_pytorch.py
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
from collections import defaultdict


# ============================================================
# 【0】配置
# ============================================================
KUAIRAND_PATH = Path("/Users/qiruihou/Desktop/学习/推荐算法/dataset/kuairand/KuaiRand-1K")

EMB_DIM = 8
DNN_UNITS = [128, 64]
ATTENTION_HIDDEN = [80, 40]    # DIN 注意力 MLP 隐藏层（fun-rec 默认）
DROPOUT_RATE = 0.1
LR = 0.001
BATCH_SIZE = 256
N_EPOCHS = 10
SUBSAMPLE_SIZE = 100000
VALIDATION_SPLIT = 0.2
HISTORY_LEN = 20               # 每个用户保留最近 20 条历史

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 【1】加载 & 预处理（包含序列特征）
# ============================================================
print("=" * 60)
print("【1】加载 KuaiRand-1K 数据（含历史序列）")
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
# 对每个用户，按时间排序，取他之前点击过的视频作为历史序列
# 
# 提示：
#   ① 按 user_id + 时间排序 merged.sort_values(["user_id", "time_ms"])
#   ② 对每个用户，对每个时间步 t，取前 t 个点击的 video_id
#   ③ 左填充/截断到 HISTORY_LEN=20
#   ④ 停用词 [PAD]=0（video_id 从 1 开始编码）
#   ⑤ 同时生成 mask：有效位置=1，padding=0
#   ⑥ 存到 merged_sorted["history_seq"] 和 merged_sorted["history_mask"]
#
# 可以用 defaultdict(list) 来累积每个用户的历史
# ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
merged_sorted = merged.sort_values(["user_id", "time_ms"]).reset_index(drop=True)
user_history = defaultdict(list)
user_seq_feats = []
for _ ,row in merged_sorted.iterrows():
    uid = int(row["user_id"])
    vid = int(row["video_id"])
    past = user_history[uid][-(HISTORY_LEN-1):] if user_history[uid] else []
    hist = [0]*(HISTORY_LEN-len(past)) + past if len(past) < HISTORY_LEN else past
    mask = [1 if x!=0 else 0 for x in hist]
    user_seq_feats.append([hist, mask])
    user_history[uid].append(vid)

merged_sorted["history_seq"] = [x[0] for x in user_seq_feats]
merged_sorted["history_mask"] = [x[1] for x in user_seq_feats]
print(f"序列构建完成，历史长度={HISTORY_LEN}")

# ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

# ── 1.6 采样 & 特征字典 ──
if len(merged_sorted) > SUBSAMPLE_SIZE:
    merged_sorted = merged_sorted.sample(n=SUBSAMPLE_SIZE, random_state=SEED)

feature_vocabs = {}
for feat in SELECT_FEATURES:
    feature_vocabs[feat] = int(merged_sorted[feat].max()) + 1
feature_vocabs["video_id_for_hist"] = feature_vocabs["video_id"]

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
    """
    和之前 Dataset 的区别：多了 history_seq 和 history_mask
    """
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
# 【3】DIN 模型
# ============================================================
class DINAttention(nn.Module):
    """
    DIN 注意力层（核心组件）

    输入：
      query: [B, D]   ← 候选商品 embedding（video_id 的向量）
      keys:  [B, L, D] ← 历史点击视频的 embedding
      mask:  [B, L]    ← 有效位置 True/False

    计算：
      att_inputs = concat(query广播, keys, query-keys, query*keys)
                                                      ← [B, L, 4D]
      scores = MLP(att_inputs)                        ← [B, L, 1]
      scores = scores * mask（不做 softmax）          ← 保留强度
      output = sum(scores * keys, dim=1)              ← [B, D]

    返回：
      torch.sum(attention_weight * keys, dim=1)       ← [B, D]
    """
    def __init__(self, emb_dim, hidden_units=[80, 40]):
        super().__init__()
        # 输入是 4D 维（query, key, query-key, query*key 拼接）
        input_dim = emb_dim * 4

        # ── MLP：4D → hidden[0] → hidden[1] → 1 ──
        layers = []
        for units in hidden_units:
            layers.extend([nn.Linear(input_dim, units), nn.PReLU(units)])
            input_dim = units
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, query, keys, mask):
        """
        query: [B, D]     ← 候选商品
        keys:  [B, L, D]   ← 历史行为
        mask:  [B, L]      ← True=有效
        返回:  [B, D]      ← 加权后的用户兴趣
        """
        B, L, D = keys.shape

        # ── TODO ②：构建注意力输入 ──
        # query 广播到 [B, L, D]
        # concat([query, keys, query-keys, query*keys], dim=-1)
        # → [B, L, 4D]
        #
        # 提示：
        #   query_exp = query.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]
        #   att_input = torch.cat([query_exp, keys, query_exp-keys, query_exp*keys], dim=-1)
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        query_exp = query.unsqueeze(1).expand(-1, L, -1)
        att_input = torch.cat([query_exp, keys, query_exp-keys, query_exp*keys], dim=-1)

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ── MLP → scores ──
        scores = self.mlp(att_input).squeeze(-1)   # [B, L]

        # ── Key：不做 softmax！只乘 mask ──
        scores = scores.masked_fill(~mask, 0.0)     # padding 位置设 0

        # ── 加权求和 ──
        output = (scores.unsqueeze(-1) * keys).sum(dim=1)  # [B, D]

        return output


class DIN(nn.Module):
    """
    DIN = 线性部分 + 其他特征 + DIN 注意力序列特征 → concat → DNN

    对照 fun-rec 源码：
      - linear_logits: get_linear_logits (和之前一样)
      - din_output: DinAttentionLayer
      - dnn_inputs: concat([其他特征, din输出])
      - dnn: DNNs(units=[128, 64, 1])
    """

    def __init__(self, feature_vocabs, emb_dim, dnn_units,
                 attn_hidden=[80, 40], dropout=0.1):
        super().__init__()
        self.feature_names = [f for f in feature_vocabs.keys()
                              if f not in ("history_seq", "history_mask", "video_id_for_hist")]
        n_features = len(self.feature_names)
        self.emb_dim = emb_dim

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

        # ── TODO ②：DIN Attention ──
        # 创建 DINAttention(emb_dim, attn_hidden) 实例
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.din_attention = DINAttention(emb_dim, attn_hidden)

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ── TODO ③：DNN 部分 ──
        # 输入维度 = 静态特征展平 (n_features * emb_dim) + DIN 输出 (emb_dim)
        # DNN 结构：用 dnn_units 循环创建
        #   Linear(input_dim → units) → BN → PReLU → Dropout
        # 最后 Linear(dnn_units[-1] → 1) 作为输出层
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        dnn_input_dim = n_features * emb_dim + emb_dim
        dnn_layers = []
        for units in dnn_units:
            dnn_layers.extend([
                nn.Linear(dnn_input_dim, units),
                nn.BatchNorm1d(units),
                nn.PReLU(),
                nn.Dropout(dropout)
            ])
            dnn_input_dim = units
        self.dnn = nn.Sequential(*dnn_layers)
        self.dnn_output = nn.Linear(dnn_units[-1], 1)

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, features):
        """
        输入：features 包含
          - 所有 SELECT_FEATURES（如 user_id, video_id, ...）
          - "history_seq": [B, L]
          - "history_mask": [B, L]
        输出：logits [B]
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

        # ── 3. 候选商品 embedding（作为 DIN 的 query） ──
        # 候选 = 当前要算 CTR 的 video_id
        candidate_emb = self.embeddings["video_id"](features["video_id"])  # [B, D]

        # ── 4. TODO ② forward：DIN 注意力 ──
        # 用候选商品当 query，历史序列当 keys
        # hist_emb = self.hist_embedding(features["history_seq"])   # [B, L, D]
        # self.din_attention(candidate_emb, hist_emb, features["history_mask"])  # [B, D]
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        hist_emb = self.hist_embedding(features["history_seq"])   # [B, L, D]
        din_output = self.din_attention(candidate_emb, hist_emb, features["history_mask"])

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ── 5. TODO ③ forward：融合 + DNN ──
        # 静态特征展平 + DIN 输出拼接 → DNN
        # flatten = static_emb.view(static_emb.size(0), -1)         # [B, N*D]
        # concat → dnn_input → self.dnn → self.dnn_output → squeeze
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        flatten = static_emb.view(static_emb.size(0), -1)       
        dnn_input = torch.concat([flatten, din_output], dim=-1)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_output(dnn_output).squeeze(-1)

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ── 6. TODO ④：融合 ──
        # logits = linear_logit + dnn_logit
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        logits = linear_logit + dnn_logit

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        return logits


model = DIN(
    feature_vocabs=feature_vocabs,
    emb_dim=EMB_DIM,
    dnn_units=DNN_UNITS,
    attn_hidden=ATTENTION_HIDDEN,
    dropout=DROPOUT_RATE,
)
print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# 【4】TODO ④：训练
# ============================================================
print("\n" + "=" * 60)
print("【4】训练 DIN")
print("=" * 60)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_din"
writer = SummaryWriter(log_dir=f"runs/{run_name}")

best_auc = 0.0
best_state = None

for epoch in range(N_EPOCHS):
    model.train()
    train_loss, n_batch = 0.0, 0
    train_preds, train_labels_list = [], []

    for features, labels in train_loader:
        # ↓↓↓↓↓ TODO ④：前向 → loss → 反向 → 更新 ↓↓↓↓↓
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
    torch.save(best_state, "best_din.pt")
    print(f"✅ 最佳模型（epoch {best_epoch}, AUC={best_auc:.4f}）")
    print(f"   fun-rec 官方 DIN AUC≈0.5999")

writer.close()
print("\n✅ 训练完成！")
