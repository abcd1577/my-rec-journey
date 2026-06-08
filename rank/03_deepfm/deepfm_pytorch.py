"""
DeepFM 精排模型 —— PyTorch 实现
（Deep Factorization Machines, Huawei 2017）

【与 fun-rec 源码对齐】
- 数据集：KuaiRand-1K（和 Wide&Deep 一致）
- 核心创新：FM 和 DNN 共享同一套 Embedding，端到端联合训练
- 损失：BCEWithLogitsLoss（二分类 CTR）
- 评估：AUC

【DeepFM 解决了 Wide&Deep 的什么问题？】
  Wide&Deep 的 Wide 部分需要大量人工交叉特征
  → DeepFM 用 FM 替代了 Wide 部分，FM 自动学二阶交叉，不用人工设计

  Wide&Deep 的 Wide 和 Deep 使用不同的权重
  → DeepFM 中 FM 和 DNN 共享同一套 Embedding，参数效率更高

【模型公式】
  y = sigmoid(y_FM + y_DNN)

  其中：
    y_FM   = w_0 + Σ w_i x_i + Σ_{i<j} <v_i, v_j> x_i x_j
    y_DNN  = MLP(flatten([v_1, v_2, ..., v_N]))

  关键：FM 和 DNN 用的是同一套 v_i（共享 Embedding！）

【和 Wide&Deep 的架构对比】

  ┌──────────────────┬──────────────────────────────────────┐
  │   Wide&Deep      │  DeepFM                              │
  ├──────────────────┼──────────────────────────────────────┤
  │  Wide: 1维权重   │  FM: 二阶交叉（内积）              │
  │  Deep: D维 emb   │  DNN: 高阶交叉（用同一套 emb）     │
  │  两套独立参数     │  一套共享参数                      │
  │  需要人工交叉特征  │  自动二阶交叉                      │
  └──────────────────┴──────────────────────────────────────┘

【模型结构】
  输入 -> 共享 Embedding [N, D] -> FM二阶 + DNN -> 融合 -> sigmoid -> CTR

【FM 二阶交叉的高效计算】
  朴素方法：Σ_{i<j} <v_i, v_j> 需要 O(k·n²)
  
  优化方法（利用数学恒等式）：
    Σ_{i<j} <v_i, v_j> = 0.5 · ( (Σ v_i)² - Σ (v_i²) )
    
    sum = Σ v_i                  # [B, D]
    square_of_sum = sum²         # [B, D]
    sum_of_square = Σ (v_i²)     # [B, D]
    result = 0.5 · (square_of_sum - sum_of_square).sum()  # [B]
    
    复杂度 O(k·n)！不管有多少特征，都和特征数 n 成线性关系

【任务拆解】
- TODO ①：FM 一阶线性部分（和 Wide&Deep 的 Wide 一样）
- TODO ②：FM 二阶交叉（高效版本：square_of_sum - sum_of_square）
- TODO ③：DNN 部分（Embedding 共享！不是独立参数）
- TODO ④：融合 y_FM + y_DNN
- TODO ⑤：训练循环

运行：
    python deepfm_pytorch.py
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

EMB_DIM = 8
DNN_UNITS = [64, 32]
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
# 【1】加载 & 预处理（KuaiRand-1K）
# ============================================================
print("=" * 60)
print("【1】加载 KuaiRand-1K 数据")
print("=" * 60)

log_df = pd.read_csv(KUAIRAND_PATH / "data" / "log_standard_4_22_to_5_08_1k.csv")
user_feat = pd.read_csv(KUAIRAND_PATH / "data" / "user_features_1k.csv")
video_basic = pd.read_csv(KUAIRAND_PATH / "data" / "video_features_basic_1k.csv")

cols = ["user_id", "video_id", "date", "time_ms", "is_click",
        "is_like", "is_follow", "is_comment", "is_forward",
        "is_hate", "long_view", "is_profile_enter", "tab"]
log_clean = log_df[cols].copy()

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

feature_vocabs = {}
for feat in SELECT_FEATURES:
    feature_vocabs[feat] = int(merged[feat].max()) + 1

if len(merged) > SUBSAMPLE_SIZE:
    merged = merged.sample(n=SUBSAMPLE_SIZE, random_state=SEED)

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


# ============================================================
# 【3】DeepFM 模型
# ============================================================
class DeepFM(nn.Module):
    """
    DeepFM = FM part + Deep part · 共享 Embedding

    对比 Wide&Deep 的核心改进：
      - Wide&Deep: Wide(1维权重) + Deep(D维emb) → 两组独立参数
      - DeepFM:    FM(用D维emb做二阶交叉) + Deep(用同样的D维emb) → 共享参数
    """

    def __init__(self, feature_vocabs, emb_dim, dnn_units, dropout=0.1):
        super().__init__()
        self.feature_names = list(feature_vocabs.keys())
        n_features = len(self.feature_names)
        self.emb_dim = emb_dim

        # ────────── ♻️ 共享 Embedding 表（FM 和 DNN 共用！）──────────
        self.embeddings = nn.ModuleDict()
        for feat_name, vocab_size in feature_vocabs.items():
            self.embeddings[feat_name] = nn.Embedding(
                vocab_size, emb_dim, padding_idx=0
            )

        # ────────── TODO ①：FM 一阶线性部分 ──────────
        # 公式：y_linear = w_0 + Σ w_i[feat_i]
        # 和 Wide&Deep 的 Wide 部分完全一样
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.linear_weights = nn.ModuleDict()
        for feat_name, vocab_size in feature_vocabs.items():
            self.linear_weights[feat_name] = nn.Embedding(vocab_size, 1, padding_idx=0)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ────────── TODO ②：FM 二阶交叉（无参数）──────────
        # 在 forward 里直接写 4 行公式
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓     
         

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ────────── TODO ③：DNN 部分 ──────────
        # 输入 = flatten([v_1, ..., v_N]) = [B, N*D]
        # 结构：Linear(128→64) → BN → ReLU → Dropout
        #       Linear(64→32)  → BN → ReLU → Dropout
        #       Linear(32→1)
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.dnn = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, features):
        """
        输入：features = {name: [B]}
        输出：logits [B]
        """

        # ── 1. 共享 Embedding Lookup ──
        embs = []
        for feat_name in self.feature_names:
            embs.append(self.embeddings[feat_name](features[feat_name]))
        emb_concat = torch.stack(embs, dim=1)   # [B, N, D]

        # ── 2. TODO ① forward：FM 一阶线性 ──
        # y_linear = w_0 + Σ w_i[feat_i]
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        linear_weights = []
        for feat_name in self.feature_names:
            linear_weights.append(self.linear_weights[feat_name](features[feat_name]))
        linear_weights = torch.stack(linear_weights, dim=1)  # [B, N, 1]
        y_linear = linear_weights.sum(dim=1).squeeze(-1) + self.global_bias

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ── 3. TODO ② forward：FM 二阶交叉（高效） ──
        # y_cross = 0.5 · Σ( (Σ v_i)² - Σ(v_i²) )
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        cross = 0.5 * (torch.sum(emb_concat, dim=1) ** 2 - torch.sum(emb_concat ** 2, dim=1))
        y_cross = cross.sum(dim=1)

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ── 4. TODO ③ forward：DNN 部分 ──
        # flatten emb → DNN → Dense(1)
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        y_dnn = self.dnn(emb_concat.view(-1, self.emb_dim * len(self.feature_names)))
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ── 5. TODO ④：融合 ──
        # y = y_linear + y_cross + y_dnn
        # 注意：三大块用了同一份 embedding！
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        y_dnn = y_dnn.squeeze(dim=1)
        logits = y_linear + y_cross + y_dnn

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        return logits


model = DeepFM(
    feature_vocabs=feature_vocabs,
    emb_dim=EMB_DIM,
    dnn_units=DNN_UNITS,
    dropout=DROPOUT_RATE,
)


# ============================================================
# 【4】TODO ⑤：训练
# ============================================================
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_deepfm"
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
    torch.save(best_state, "best_deepfm.pt")
    print(f"✅ 最佳模型（epoch {best_epoch}, AUC={best_auc:.4f}）")
    print(f"   fun-rec 官方 DeepFM AUC≈0.5953")


# ============================================================
# 【5】消融实验
# ============================================================
print("\n" + "=" * 60)
print("【5】消融：FM vs DNN vs DeepFM")
print("=" * 60)

model.eval()
with torch.no_grad():
    fm_preds, dnn_preds, all_labels = [], [], []
    for features, labels in val_loader:
        embs = []
        for feat_name in model.feature_names:
            embs.append(model.embeddings[feat_name](features[feat_name]))
        emb_concat = torch.stack(embs, dim=1)

        sum_v = emb_concat.sum(dim=1)
        square_of_sum = sum_v ** 2
        sum_of_square = (emb_concat ** 2).sum(dim=1)
        cross = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)

        linear_logit = model.global_bias.expand(labels.size(0))
        for feat_name, feat_vocab in model.linear_weights.items():
            lin = model.linear_weights[feat_name](features[feat_name])
            linear_logit = linear_logit + lin.squeeze(1)

        fm_pred = torch.sigmoid(linear_logit + cross)
        fm_preds.append(fm_pred)

        deep_in = emb_concat.view(emb_concat.size(0), -1)
        deep_out = model.dnn(deep_in)
        dnn_logit = model.dnn(deep_in).squeeze(1)
        dnn_preds.append(torch.sigmoid(dnn_logit))

        all_labels.append(labels)

    fm_p = torch.cat(fm_preds).numpy()
    dnn_p = torch.cat(dnn_preds).numpy()
    lab = torch.cat(all_labels).numpy()

    print(f"  FM only（线性+二阶）AUC: {roc_auc_score(lab, fm_p):.4f}")
    print(f"  DNN only AUC:             {roc_auc_score(lab, dnn_p):.4f}")
    print(f"  DeepFM（FM+DNN）AUC:      {best_auc:.4f}")

writer.close()
print("\n✅ 训练完成！")
