"""
Wide & Deep 精排模型 —— PyTorch 实现
（Google 2016, 精排模型的"开山鼻祖"）

【与 fun-rec 源码对齐】
- 数据集：KuaiRand-1K（快手短视频，10亿量级精排数据的 1K 子集）
- 特征：15 个稀疏特征（user_id, video_id, author_id, tag, ...）
- 结构：Wide（线性）+ Deep（Embedding → DNN）
- 训练：Adam + BinaryCrossentropy，batch_size=1024，subsample=300000
- 评估：AUC

【这一节的产出】
1. 第一次做「二分类」精排任务（CTR 预估）
2. 理解「记忆（Memorization）」与「泛化（Generalization）」的互补
3. 实现 Wide 部分：对每个稀疏特征学 1 维偏置（线性加权求和）
4. 实现 Deep 部分：Embedding → Flatten → Concat → MLP
5. 掌握联合训练（Joint Training），对比两部分的贡献

【与召回模型的区别】
┌────────────────┬──────────────────────────────┬──────────────────────────────┐
│      维度       │           召回模型             │        精排模型（Wide&Deep） │
├────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 目标            │  多分类（下一个商品是哪个）    │  二分类（点或不点）           │
│ 损失函数        │  CrossEntropy（SampledSoftmax）│  BCEWithLogitsLoss           │
│ 输出            │  所有物品的分数分布            │  单个物品的点击概率           │
│ 特征使用        │  只用 user_id + history        │  user + item + context 全特征 │
│ 训练范式        │  召回（负采样）                │  排序（曝光样本）             │
│ 评估指标        │  HitRate / NDCG               │  AUC                          │
└────────────────┴──────────────────────────────┴──────────────────────────────┘

【任务拆解】
- TODO ①：Wide 部分——Linear Logits（每个特征学 1 维权重，加权求和）
- TODO ②：Deep 部分——Embedding → Flatten → Concat → DNN
- TODO ③：Wide + Deep 融合输出（相加后过 sigmoid）
- TODO ④：训练循环 + AUC 评估 + 消融实验

运行：
    python widedeep_pytorch.py
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


# ============================================================
# 【0】配置（与 fun-rec config_wide_deep.py 对齐）
# ============================================================
KUAIRAND_PATH = Path("/Users/qiruihou/Desktop/学习/推荐算法/dataset/kuairand/KuaiRand-1K")

EMB_DIM = 8              # fun-rec 默认 emb_dim = 8
DNN_UNITS = [64, 32]     # fun-rec 默认 [64, 32]
DNN_DROPOUT = 0.1
LR = 0.01
BATCH_SIZE = 1024        # fun-rec 默认 batch_size = 1024
N_EPOCHS = 20             # fun-rec 默认 epochs = 1
SUBSAMPLE_SIZE = 300000  # fun-rec 默认 subsample = 300000
VALIDATION_SPLIT = 0.2

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 【1】加载 & 预处理数据（KuaiRand-1K）
# ============================================================
print("=" * 60)
print("【1】加载 KuaiRand-1K 数据")
print("=" * 60)

# --- 1.1 读 CSV ---
log_df = pd.read_csv(KUAIRAND_PATH / "data" / "log_standard_4_22_to_5_08_1k.csv")
user_feat = pd.read_csv(KUAIRAND_PATH / "data" / "user_features_1k.csv")
video_basic = pd.read_csv(KUAIRAND_PATH / "data" / "video_features_basic_1k.csv")

print(f"log 行数: {len(log_df)}, 用户特征行数: {len(user_feat)}, "
      f"视频特征行数: {len(video_basic)}")

# ═══════════════════════════════════════════════════════════════
# 1.2 从 log 表中选取需要的列
# ═══════════════════════════════════════════════════════════════
#
# 【原始 log 表长啥样？】
#   每条日志 = 用户刷快手时看到一条视频 + 他的反馈行为
#   列很多（19列），但精排只关心这些：
#
#   user_id     → 谁看到了视频（int）
#   video_id    → 什么视频（int）
#   date        → 哪天（20220422）
#   time_ms     → 精确时间戳（毫秒级，用于排序）
#   is_click    → ⭐⭐⭐ 我们今天要预测的**标签**（0/1）：点没点？
#   is_like     → 点赞没（辅助特征，我们暂时不用）
#   is_follow   → 关注作者没
#   is_comment  → 评论没
#   tab         → 哪个场景（推荐/关注/同城等），用于过滤
#
# 【精排 vs 召回的数据区别】
#   召回：样本 = (user, [history_seq], target) ← 一个样本是一个"看完了"
#   精排：样本 = (user, video, 其他特征..., is_click)
#         ↑ 每条曝光（点击或未点击）都是独立的一条样本
#         这就是"曝光日志"——包含正样本（点了）和负样本（没点）
cols = ["user_id", "video_id", "date", "time_ms", "is_click",
        "is_like", "is_follow", "is_comment", "is_forward",
        "is_hate", "long_view", "is_profile_enter", "tab"]
log_clean = log_df[cols].copy()
print(f"正样本(is_click=1)占比: {log_clean['is_click'].mean()*100:.2f}%")
# ↑ 通常精排数据正样本率在 2%~10% 之间（绝大多数曝光用户都不会点击）

# ═══════════════════════════════════════════════════════════════
# 1.3 用户特征处理
# ═══════════════════════════════════════════════════════════════
#
# 【用户特征表长啥样？】
#   每行 = 一个用户的画像信息（静态的，不随时间变化）
#
#   user_active_degree  → 活跃度："high_active" / "full_active" / "middle_active"
#                         ↑ 这是字符串，需要转成数字（embedding lookup 用）
#   is_live_streamer    → 是否主播（0/1，但有异常值 -124）
#   is_video_author     → 是否自己发过视频（0/1）
#   follow_user_num_range  → 关注人数范围："(0,10]" / "(10,50]" / "500+"
#   fans_user_num_range    → 粉丝数范围："[100,1k)" / "[1k,5k)" ...
#   friend_user_num_range  → 好友数范围
#   register_days_range    → 注册天数范围："730+" / "366-730" ...
#
# 【为什么要做 LabelEncoder？】
#   神经网络只能吃数字，不能吃字符串。
#   所以 "high_active" → 1, "full_active" → 2, "middle_active" → 3
#   这样 model 就可以 lookup embedding 表得到向量
from sklearn.preprocessing import LabelEncoder

user_cols = ["user_id", "user_active_degree", "is_live_streamer", "is_video_author",
             "follow_user_num_range", "fans_user_num_range",
             "friend_user_num_range", "register_days_range"]
user_clean = user_feat[user_cols].copy()

# 修复异常值：is_live_streamer 里有个 -124，含义不明，统一改成 0
user_clean["is_live_streamer"] = user_clean["is_live_streamer"].apply(
    lambda x: 0 if x == -124 else x
)

# LabelEncoder：对"字符串类型"的特征做编码
# user_id 单独处理（保留原列用于 merge，用编码列 merge）
str_feats = ["user_id", "user_active_degree", "follow_user_num_range",
             "fans_user_num_range", "friend_user_num_range", "register_days_range"]
for feat in str_feats:
    le = LabelEncoder()
    user_clean[feat + "_enc"] = le.fit_transform(user_clean[feat].astype(str)) + 1
    # +1 是因为 Embedding 层 padding_idx=0，有效索引从 1 开始
    if feat != "user_id":
        # 非 user_id 的特征：直接替换原列
        user_clean[feat] = user_clean[feat + "_enc"]
        del user_clean[feat + "_enc"]
    # user_id 保留 _enc 列，原始 user_id 列留着给 merge 用

# ═══════════════════════════════════════════════════════════════
# 1.4 视频特征处理
# ═══════════════════════════════════════════════════════════════
#
# 【视频基础特征表长啥样？】
#   每行 = 一个视频的元信息
#
#   video_id      → 视频ID（和 log 里的 video_id 关联）
#   author_id     → 作者ID（同一个作者可能发很多视频）
#   video_type    → "NORMAL"（普通）或 "AD"（广告）
#   upload_type   → 上传方式
#   visible_status → 可见状态（数字，但有 NaN）
#   music_id      → 背景音乐ID
#   music_type    → 背景音乐类型（数字，也有 NaN）
#   tag           → ⭐ 视频标签列表，如 "12,65"（字符串，逗号分隔）
#
# 【为什么精排需要这么多特征？】
#   召回模型只用了 user_id + video_id → 靠 ID embedding 学相似度
#   精排要预测 CTR → 需要上下文特征
#   比如 "这视频是广告" → CTR 通常低 → video_type="AD" 帮助模型学这个规律

video_cols = ["video_id", "author_id", "video_type", "upload_type",
              "visible_status", "music_id", "music_type", "tag"]
video_basic_clean = video_basic[video_cols].copy()

# 填充 NaN：visible_status 和 music_type 有缺失值
# 策略：用当前列的最大值+1来填充（当作一个新的未知类别）
for feat in ["visible_status", "music_type"]:
    max_val = video_basic_clean[feat].max()
    video_basic_clean[feat] = video_basic_clean[feat].fillna(
        max_val + 1 if pd.notna(max_val) else 0
    ).astype(int)

# 对视频的稀疏特征做 LabelEncoder（同上理由：神经网络要数字）
for feat in ["video_id", "author_id", "video_type", "upload_type",
             "visible_status", "music_id", "music_type"]:
    le = LabelEncoder()
    video_basic_clean[feat + "_enc"] = le.fit_transform(
        video_basic_clean[feat].astype(str)
    ) + 1
    if feat != "video_id":
        video_basic_clean[feat] = video_basic_clean[feat + "_enc"]
        del video_basic_clean[feat + "_enc"]

# tag 处理
# tag 原始格式是 "12,65"（一个视频可能有多个标签）
# fun-rec 的做法：只取第一个标签（简化）
video_basic_clean["tag"] = video_basic_clean["tag"].fillna("-1")
tag_set = set()
for x in video_basic_clean["tag"].values:
    for t in str(x).split(","):
        tag_set.add(t)
# 所有不重复的 tag → 映射为 1,2,3,...
tag_map = {t: i+1 for i, t in enumerate(sorted(tag_set))}
video_basic_clean["tag"] = video_basic_clean["tag"].apply(
    lambda x: tag_map.get(str(x).split(",")[0], 0)
)

# ═══════════════════════════════════════════════════════════════
# 1.5 三表合并：log + user_features + video_features
# ═══════════════════════════════════════════════════════════════
#
# 【为什么需要合并？】
#   现在我们的数据分三张表：
#     log表：       (user_id, video_id, is_click, date, tab, ...)
#     user特征表：   (user_id, user_active_degree, ...)
#     video特征表：  (video_id, author_id, video_type, ...)
#
#   但模型一次 forward 需要拿到所有特征！
#   所以需要 on user_id 和 on video_id 做两次 merge（SQL 里的 left join）
#
# 【合并完的一条完整样本长啥样？】
#   user_id=17, video_id=342, is_click=1,        ← 标签
#   user_active_degree=2, is_live_streamer=0,     ← 用户特征
#   author_id=88, video_type=1, music_id=45,      ← 视频特征
#   tag=12, date=20220422, tab=1                  ← 上下文特征
#
#   模型要学：(user_id, user_active_degree, ..., video_id, author_id, ..., tab)
#              └────────── 15 个特征 ──────────┘
#    → 预测 is_click（点不点）
merged = log_clean.merge(user_clean, on="user_id", how="left")
merged = merged.merge(video_basic_clean, on="video_id", how="left")

# 用编码后的 ID 替换原始 ID（Embedding lookup 需要连续整数）
merged["user_id"] = merged["user_id_enc"]
merged["video_id"] = merged["video_id_enc"]
for col in ["user_id_enc", "video_id_enc"]:
    if col in merged.columns:
        del merged[col]

# tag 可能有 NaN（merge 时左边有 log 但右边没有 video 特征）
merged["tag"] = merged["tag"].fillna(0).astype(int)

# ─────────────────────────────────────────────────────────────
# 只保留 fun-rec config 中定义的 15 + tag 个特征
# ─────────────────────────────────────────────────────────────
# 【为什么选这 15 个特征？】
#   这是 fun-rec 官方配置里对 Wide&Deep 的特征定义
#   group=["wide_deep", "linear"] → 既参与 Wide 的线性加权，也参与 Deep 的 Embedding
#
#   这 15 个特征覆盖了：
#   - 用户身份：user_id（谁看了）
#   - 视频身份：video_id、author_id（看了什么）
#   - 用户画像：活跃度、是否主播、是否作者、关注数、粉丝数、注册天数
#   - 视频属性：类型、上传方式、可见状态、音乐、标签
SELECT_FEATURES = [
    "user_id", "video_id", "user_active_degree", "is_live_streamer",
    "is_video_author", "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range", "author_id",
    "video_type", "upload_type", "visible_status", "music_id",
    "music_type", "tag",
]
for feat in SELECT_FEATURES:
    if feat not in merged.columns:
        # 极少数情况 merge 丢了某个特征 → 补 0
        merged[feat] = 0
    merged[feat] = merged[feat].fillna(0).astype(int)

# 只保留主场景（tab=0,1,2,4,6）
# 其他 tab（如直播、小游戏等）场景不同、用户行为模式不同，混在一起会干扰模型
main_tabs = set([1, 0, 4, 2, 6])
merged = merged[merged["tab"].isin(main_tabs)]

print(f"\n最终数据行数: {len(merged)}")
print(f"正样本率: {merged['is_click'].mean()*100:.2f}%")

# 为每个特征统计 vocab_size（Embedding 表需要知道查表范围）
# vocab_size = max_id + 1（因为 ID 从 1 开始，0 留给 padding）
feature_vocabs = {}
for feat in SELECT_FEATURES:
    feature_vocabs[feat] = int(merged[feat].max()) + 1
    print(f"  {feat:<22s}: vocab = {feature_vocabs[feat]}")

# ─────────────────────────────────────────────────────────────
# 1.6 采样 & 切分训练/验证集
# ─────────────────────────────────────────────────────────────
# fun-rec 默认 subsample_size=300000（原始数据大于此值则随机采样）
if len(merged) > SUBSAMPLE_SIZE:
    merged = merged.sample(n=SUBSAMPLE_SIZE, random_state=SEED)
    print(f"已采样至 {SUBSAMPLE_SIZE} 条")

# 转成 PyTorch 张量
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
# 【3】Wide & Deep 模型
# ============================================================
class WideAndDeep(nn.Module):
    """
    【Wide 部分】
        y_wide = bias + Σ w_i^T · x_i
        每个稀疏特征 x_i 学一个 1 维权重 w_i → 直接"记住"共现模式

        fun-rec 对应：get_linear_logits（对每个 group="linear" 的特征做线性加权）

    【Deep 部分】
        y_deep = DNN(concat(E_1, E_2, ..., E_N))
        每个特征学一个 D 维 embedding → 展平拼接 → MLP
        → 通过 embedding 语义去"泛化"没见过的组合

        fun-rec 对应：concat_group_embedding + DNNs

    【融合】
        y = σ(y_wide + y_deep)
    """
    def __init__(self,feature_vocabs, emb_dim, dnn_units, dropout=0.1):
        super().__init__()


    def __init__(self, feature_vocabs, emb_dim, dnn_units, dropout=0.1):
        super().__init__()
        self.feature_names = list(feature_vocabs.keys())
        n_features = len(self.feature_names)
        self.emb_dim = emb_dim

        # ────────── Embedding 表（Deep 部分用）──────────
        self.embeddings = nn.ModuleDict()
        for feat_name, vocab_size in feature_vocabs.items():
            self.embeddings[feat_name] = nn.Embedding(
                vocab_size, emb_dim, padding_idx=0
            )

        # ────────── TODO ①：Wide 部分（Linear Logits）──────────
        # 对每个特征学一个 1 维权重（即 nn.Embedding(..., 1)）
        # 对应 fun-rec 的 get_linear_logits
        #
        # 提示：
        #   linear_weights 是 nn.ModuleDict
        #   每个元素是 nn.Embedding(vocab_size, 1)
        #   forward 时对 batch 中每个样本做 lookup 后 squeeze + sum
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓


        self.linear_weights = nn.ModuleDict()
        for feat_name, vocab_size in feature_vocabs.items():
            self.linear_weights[feat_name] = nn.Embedding(vocab_size, 1, padding_idx=0)
        

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑
        self.global_bias = nn.Parameter(torch.zeros(1))

        # ────────── TODO ②：Deep 部分（DNN）──────────
        # 输入 = n_features * emb_dim（所有特征 embedding 展平拼接）
        # fun-rec 用 DNNs(units=[64, 32], activation='relu', dropout=0.1)
        # 最后接 Dense(1) 输出 deep_logit
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓

        dnn_input_dim = n_features * emb_dim
        dnn_layers = []
        for units in dnn_units:
            dnn_layers.extend([
                nn.Linear(dnn_input_dim, units),
                nn.BatchNorm1d(units),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            dnn_input_dim = units
        self.dnn = nn.Sequential(*dnn_layers)
        self.deep_output = nn.Linear(dnn_units[-1], 1)

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, features):
        """
        输入：features = {name: [B], ...}
        输出：logits [B] — 未过 sigmoid 的分数
        """
        # ────────── TODO ①：Wide 部分 ──────────
        # 对每个特征 linear_weights[name](features[name]) → [B, 1]
        # squeeze(1) → [B]，然后全部加起来
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        linear_logit = self.global_bias.expand(features[self.feature_names[0]].size(0))
        for feat_name in self.feature_names:
            lin = self.linear_weights[feat_name](features[feat_name])
            linear_logit = linear_logit + lin.squeeze(1)

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ────────── TODO ②：Deep 部分 ──────────
        # 每个特征 lookup embedding → concat([B, N, D]) → flatten → DNN → Dense(1)
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        embs = []
        for feat_name in self.feature_names:
            embs.append(self.embeddings[feat_name](features[feat_name]))
        deep_in = torch.cat(embs, dim=-1)
        deep_out = self.dnn(deep_in)
        deep_logit = self.deep_output(deep_out).squeeze(1)

        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ────────── TODO ③：融合 ──────────
        # y_wide + y_deep（fun-rec 中的 add_tensor_func）
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        logits = linear_logit + deep_logit
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        return logits


model = WideAndDeep(
    feature_vocabs=feature_vocabs,
    emb_dim=EMB_DIM,
    dnn_units=DNN_UNITS,
    dropout=DNN_DROPOUT,
)
print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# 【4】TODO ④：训练
# ============================================================
print("\n" + "=" * 60)
print("【4】训练 Wide & Deep")
print("=" * 60)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_widedeep"
writer = SummaryWriter(log_dir=f"runs/{run_name}")

best_auc = 0.0
best_state = None

for epoch in range(N_EPOCHS):
    # ── 训练 ──
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

    # ── 验证 ──
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
    torch.save(best_state, "best_widedeep.pt")
    print(f"✅ 最佳模型（epoch {best_epoch}, AUC={best_auc:.4f}）")
    print(f"   fun-rec 官方 AUC≈0.5902（TF 实现，当前≈{best_auc:.4f}（PyTorch）")


# ============================================================
# 【5】消融：只看 Wide 或只看 Deep
# ============================================================
print("\n" + "=" * 60)
print("【5】消融实验（对比 Wide 和 Deep 各自 AUC）")
print("=" * 60)

model.eval()
with torch.no_grad():
    wide_preds, deep_preds, all_labels = [], [], []
    for features, labels in val_loader:
        # 只走 Wide 路径
        linear_logit = model.global_bias.expand(labels.size(0))
        for feat_name in model.feature_names:
            lin = model.linear_weights[feat_name](features[feat_name])
            linear_logit = linear_logit + lin.squeeze(1)

        # 只走 Deep 路径
        embs = []
        for feat_name in model.feature_names:
            embs.append(model.embeddings[feat_name](features[feat_name]))
        deep_in = torch.cat(embs, dim=-1)
        deep_out = model.dnn(deep_in)
        deep_logit = model.deep_output(deep_out).squeeze(1)

        wide_preds.append(torch.sigmoid(linear_logit))
        deep_preds.append(torch.sigmoid(deep_logit))
        all_labels.append(labels)

    wide_p = torch.cat(wide_preds).numpy()
    deep_p = torch.cat(deep_preds).numpy()
    lab = torch.cat(all_labels).numpy()

    print(f"  Wide only AUC: {roc_auc_score(lab, wide_p):.4f}")
    print(f"  Deep only AUC: {roc_auc_score(lab, deep_p):.4f}")
    print(f"  Wide+Deep AUC: {best_auc:.4f}")

writer.close()
print("\n✅ 训练完成！")
