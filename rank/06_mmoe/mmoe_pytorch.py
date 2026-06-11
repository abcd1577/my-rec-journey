"""
MMoE 精排模型 —— PyTorch 实现
（Multi-gate Mixture-of-Experts, Google 2018）

【与 fun-rec 源码对齐】
- 数据集：KuaiRand-1K
- 核心创新：多个 Expert 共享学习，每个任务有独立的 Gate 选择 Expert
- 评估：AUC（每个任务单独算）

【MMoE 解决了什么问题？】
  Shared-Bottom 的问题：所有任务共享同一个底层 DNN
    → 任务之间互相干扰（一个任务梯度大，会淹没另一个任务）
    → 任务关系不明确时，共享层学偏

  MMoE 的答案：
    - 不共享一个底层，而是共享多个 Expert
    - 每个任务用独立的 Gate 决定"听哪些 Expert"
    - 相关任务 → Gate 权重相似 → 共享相同的 Expert
    - 不相关任务 → Gate 权重不同 → 各自用各自的 Expert

【核心结构】
  input ──→ [Expert_0, Expert_1, Expert_2, Expert_3]  ← 共享
               ↑      ↑      ↑      ↑
            Gate_A  Gate_B  Gate_A  Gate_B
               ↓      ↓
            Tower_A Tower_B
               ↓      ↓
            Task_A  Task_B

【Gate 怎么工作？】
  Gate(x) = softmax(DNN(x))   # [B, num_experts]
  Task_input = sum(Gate[i] * Expert[i])   # 加权融合

【损失函数】
  loss = BCE(Task_A_pred, label_A) + BCE(Task_B_pred, label_B)
  （可以加权，也可以做 Uncertainty Weighting，这里先简化）

【和之前代码的差异】
  - 数据：需要构造多目标标签（is_click + is_like）
  - 模型：输出从 1 个 logits 变成 2 个 logits 列表
  - 训练：每个任务单独算 loss，再求和
  - 评估：每个任务单独算 AUC

运行：
    python mmoe_pytorch.py
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


# ============================================================
# 【0】配置
# ============================================================
KUAIRAND_PATH = Path("/Users/qiruihou/Desktop/学习/推荐算法/dataset/kuairand/KuaiRand-1K")

EMB_DIM = 8
BATCH_SIZE = 256
LR = 1e-3
N_EPOCHS = 5
DROPOUT_RATE = 0.1
VALIDATION_SPLIT = 0.2
SUBSAMPLE_SIZE = 50000
SEED = 42

# MMoE 特有配置
NUM_EXPERTS = 4           # Expert 数量
EXPERT_UNITS = [64, 32]   # 每个 Expert 的 DNN 结构
GATE_UNITS = [32]         # Gate 网络的 DNN 结构（最后接 softmax）
TOWER_UNITS = [32, 16]    # 每个任务塔的 DNN 结构

# 多目标标签
TASK_NAMES = ["is_click", "is_like"]   # 任务 A 和任务 B

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 【1】数据预处理
# ============================================================
print("=" * 60)
print("【1】数据加载 & 预处理")
print("=" * 60)

user_df = pd.read_csv(KUAIRAND_PATH / "data/user_features_1k.csv")
video_df = pd.read_csv(KUAIRAND_PATH / "data/video_features_basic_1k.csv")
log_df = pd.read_csv(KUAIRAND_PATH / "data/log_standard_4_22_to_5_08_1k.csv")

# ── 1.1 用户特征清洗 ──
user_clean = user_df.copy()
user_clean["user_id"] = user_clean["user_id"].astype(int)
TEXT_FEATS = ["user_active_degree", "is_live_streamer", "is_video_author",
              "follow_user_num_range", "fans_user_num_range",
              "friend_user_num_range", "register_days_range"]
for feat in TEXT_FEATS:
    le = LabelEncoder()
    user_clean[feat] = user_clean[feat].astype(str).fillna("__nan__")
    non_na = user_clean[feat] != "__nan__"
    encoded = np.zeros(len(user_clean), dtype=int)
    if non_na.sum() > 0:
        encoded[non_na] = le.fit_transform(user_clean[feat][non_na]) + 1  # padding=0
    user_clean[feat] = encoded

# ── 1.2 视频特征清洗 ──
video_basic_clean = video_df.copy()
video_basic_clean["video_id"] = video_basic_clean["video_id"].astype(int)
for feat in ["video_id", "author_id", "video_type", "upload_type",
             "visible_status", "music_id", "music_type"]:
    le = LabelEncoder()
    video_basic_clean[feat + "_enc"] = le.fit_transform(video_basic_clean[feat].astype(str)) + 1
    if feat != "video_id":
        video_basic_clean[feat] = video_basic_clean[feat + "_enc"]
        del video_basic_clean[feat + "_enc"]
video_basic_clean["tag"] = video_basic_clean["tag"].fillna("-1")
tag_set = set()
for x in video_basic_clean["tag"].values:
    for t in str(x).split(","):
        tag_set.add(t)
tag_map = {t: i+1 for i, t in enumerate(sorted(tag_set))}
video_basic_clean["tag"] = video_basic_clean["tag"].apply(lambda x: tag_map.get(str(x).split(",")[0], 0))

# ── 1.3 合并基础特征 ──
cols = ["user_id", "video_id", "date", "time_ms", "is_click", "is_like", "tab"]
log_clean = log_df[cols].copy()

merged = log_clean.merge(user_clean, on="user_id", how="left")
merged = merged.merge(video_basic_clean, on="video_id", how="left")
merged["user_id"] = merged["user_id_enc"] if "user_id_enc" in merged.columns else merged["user_id"]
merged["video_id"] = merged["video_id_enc"] if "video_id_enc" in merged.columns else merged["video_id"]
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
print(f"基础数据: {len(merged)} 行")
print(f"  is_click 正样本率: {merged['is_click'].mean()*100:.2f}%")
print(f"  is_like  正样本率: {merged['is_like'].mean()*100:.2f}%")

# ── 1.4 采样 & 特征字典 ──
feature_vocabs = {}
for feat in SELECT_FEATURES:
    feature_vocabs[feat] = int(merged[feat].max()) + 1

if len(merged) > SUBSAMPLE_SIZE:
    merged = merged.sample(n=SUBSAMPLE_SIZE, random_state=SEED)

# 多目标标签
labels_dict = {}
for task in TASK_NAMES:
    labels_dict[task] = torch.FloatTensor(merged[task].values)

feature_tensors = {}
for feat in SELECT_FEATURES:
    feature_tensors[feat] = torch.LongTensor(merged[feat].values)

n_total = len(merged)
n_val = int(n_total * VALIDATION_SPLIT)
indices = torch.randperm(n_total)


# ============================================================
# 【2】Dataset
# ============================================================
class KuaiRandMultiTaskDataset(Dataset):
    """
    和之前 Dataset 的区别：标签从 1 个变成多个任务的标签字典
    """
    def __init__(self, features, labels_dict, indices):
        self.features = {k: v[indices] for k, v in features.items()}
        self.labels = {k: v[indices] for k, v in labels_dict.items()}

    def __len__(self):
        return len(next(iter(self.labels.values())))

    def __getitem__(self, idx):
        feats = {k: v[idx] for k, v in self.features.items()}
        labs = {k: v[idx] for k, v in self.labels.items()}
        return feats, labs


train_dataset = KuaiRandMultiTaskDataset(feature_tensors, labels_dict, indices[n_val:])
val_dataset = KuaiRandMultiTaskDataset(feature_tensors, labels_dict, indices[:n_val])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")


# ============================================================
# 【3】MMoE 核心模块
# ============================================================

class Expert(nn.Module):
    """
    单个 Expert：就是一个普通的 DNN

    输入：dnn_input  [B, input_dim]
    输出：expert_out [B, output_dim]
    """
    def __init__(self, input_dim, units, dropout=0.1):
        super().__init__()
        layers = []
        for u in units:
            layers.extend([
                nn.Linear(input_dim, u),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = u
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Gate(nn.Module):
    """
    单个 Gate：DNN + softmax，输出对 Expert 的权重

    输入：dnn_input  [B, input_dim]
    输出：gate_weights [B, num_experts]  （softmax，和为1）
    """
    def __init__(self, input_dim, gate_units, num_experts):
        super().__init__()
        layers = []
        for u in gate_units:
            layers.extend([nn.Linear(input_dim, u), nn.ReLU()])
            input_dim = u
        layers.append(nn.Linear(input_dim, num_experts))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class MMoE(nn.Module):
    """
    MMoE = Shared Embedding → Multiple Experts → Task-specific Gates → Task Towers

    结构：
      ① static_emb = concat([emb(feat) for feat in features])  # [B, N*D]
      ② experts = [Expert_i(static_emb) for i in range(num_experts)]  # 每个 [B, expert_dim]
         expert_stack = torch.stack(experts, dim=1)  # [B, num_experts, expert_dim]
      ③ 对每个任务 t：
           gate_t = Gate_t(static_emb)  # [B, num_experts]
           fusion_t = sum(gate_t[:, i] * expert_stack[:, i, :])  # [B, expert_dim]
      ④ tower_t = Tower_t(fusion_t)  # [B, 1]
      ⑤ logits_t = linear_logit + tower_t
    """

    def __init__(self, feature_vocabs, emb_dim, num_experts, expert_units,
                 gate_units, tower_units, task_names, dropout=0.1):
        super().__init__()
        self.feature_names = list(feature_vocabs.keys())
        self.n_features = len(self.feature_names)
        self.num_experts = num_experts
        self.task_names = task_names
        self.n_tasks = len(task_names)

        # ── Embedding 表 ──
        self.embeddings = nn.ModuleDict()
        for feat_name in self.feature_names:
            self.embeddings[feat_name] = nn.Embedding(
                feature_vocabs[feat_name], emb_dim, padding_idx=0
            )

        # ── 线性偏置 ──
        self.linear_weights = nn.ModuleDict()
        for feat_name in self.feature_names:
            self.linear_weights[feat_name] = nn.Embedding(
                feature_vocabs[feat_name], 1, padding_idx=0
            )
        self.global_bias = nn.Parameter(torch.zeros(1))

        input_dim = self.n_features * emb_dim

        # ── TODO ②：Expert 网络 ──
        # 创建 num_experts 个 Expert，每个输入维度=input_dim，结构=expert_units
        # 提示：self.experts = nn.ModuleList([Expert(...) for _ in range(num_experts)])
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.experts = nn.ModuleList(
            [
                Expert(input_dim,expert_units,dropout=dropout) for _ in range(num_experts)
            ]
        )
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑
        expert_out_dim = expert_units[-1]

        # ── TODO ③：Gate 网络 ──
        # 每个任务有一个 Gate，输入=input_dim，输出=num_experts 的 softmax 权重
        # 提示：self.gates = nn.ModuleList([Gate(...) for _ in range(n_tasks)])
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.gates = nn.ModuleList(
            [
                Gate(input_dim, gate_units, num_experts) for _ in range(self.n_tasks)
            ]
        )
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ── TODO ④：任务塔（Task Tower）──
        # 每个任务有一个 Tower，输入=expert_out_dim，输出=1 个 logit
        # 注意：和 Expert 不同，Tower 最后要接 Linear(output_dim, 1)，不要 ReLU
        # 提示：self.towers = nn.ModuleList([nn.Sequential(...) for _ in range(n_tasks)])
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.towers = nn.ModuleList()
        for _ in range(self.n_tasks):
            layers = []
            dim = expert_out_dim
            for u in tower_units:
                layers.extend([nn.Linear(dim, u), nn.ReLU(), nn.Dropout(dropout)])
                dim = u
            layers.append(nn.Linear(dim, 1))  # 最后输出 1 个 logit
            self.towers.append(nn.Sequential(*layers))
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, features):
        # ── 1. 静态特征 → Embedding & 展平 ──
        embs = []
        for feat_name in self.feature_names:
            embs.append(self.embeddings[feat_name](features[feat_name]))
        static_emb = torch.stack(embs, dim=1)   # [B, N, D]
        flat = static_emb.view(static_emb.size(0), -1)  # [B, N*D]

        # ── 2. 线性部分 ──
        linear_logit = self.global_bias.expand(static_emb.size(0))
        for feat_name in self.feature_names:
            lin = self.linear_weights[feat_name](features[feat_name])
            linear_logit = linear_logit + lin.squeeze(1)

        # ── 3. TODO ④：MMoE 核心逻辑 ──
        # 提示：
        #   ① expert_outs = [expert(flat) for expert in self.experts]
        #      expert_stack = torch.stack(expert_outs, dim=1)   # [B, num_experts, expert_dim]
        #   ② 对每个任务 t：
        #        gate_weights = self.gates[t](flat)              # [B, num_experts]
        #        gate_expanded = gate_weights.unsqueeze(-1)      # [B, num_experts, 1]
        #        fusion = (gate_expanded * expert_stack).sum(dim=1)  # [B, expert_dim]
        #        tower_out = self.towers[t](fusion).squeeze(-1)      # [B]
        #        logits_t = linear_logit + tower_out
        #   ③ 返回 logits 列表：[logits_A, logits_B]
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        expert_outs = [expert(flat) for expert in self.experts]
        expert_stack = torch.stack(expert_outs, dim=1)
        logits_list = []
        for i in range(self.n_tasks):
            gate_weights = self.gates[i](flat)
            gate_expanded = gate_weights.unsqueeze(-1)
            fusion = (gate_expanded * expert_stack).sum(dim=1)
            tower_out = self.towers[i](fusion).squeeze(-1)
            logits_t = linear_logit + tower_out
            logits_list.append(logits_t)
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑
        return logits_list


model = MMoE(
    feature_vocabs=feature_vocabs,
    emb_dim=EMB_DIM,
    num_experts=NUM_EXPERTS,
    expert_units=EXPERT_UNITS,
    gate_units=GATE_UNITS,
    tower_units=TOWER_UNITS,
    task_names=TASK_NAMES,
    dropout=DROPOUT_RATE,
)
print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# 【4】TODO ⑤：训练
# ============================================================
print("\n" + "=" * 60)
print("【4】训练 MMoE")
print("=" * 60)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_mmoe"
writer = SummaryWriter(log_dir=f"runs/{run_name}")

best_auc = {task: 0.0 for task in TASK_NAMES}
best_state = None

for epoch in range(N_EPOCHS):
    model.train()
    train_loss, n_batch = 0.0, 0
    train_preds = {task: [] for task in TASK_NAMES}
    train_labels = {task: [] for task in TASK_NAMES}

    for features, labels in train_loader:
        # ── TODO ⑤：训练循环 ──
        # 提示：
        #   ① logits_list = model(features)   # [logits_click, logits_like]
        #   ② 对每个任务算 BCE loss，求和
        #      loss = sum(criterion(logits_list[i], labels[TASK_NAMES[i]]) for i in range(n_tasks))
        #   ③ optimizer.zero_grad(); loss.backward(); optimizer.step()
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        logits_list = model(features)
        loss = sum(criterion(logits_list[i], labels[TASK_NAMES[i]]) for i in range(len(TASK_NAMES)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        train_loss += loss.item()
        n_batch += 1
        for i, task in enumerate(TASK_NAMES):
            train_preds[task].append(torch.sigmoid(logits_list[i]).detach())
            train_labels[task].append(labels[task])

    train_loss /= n_batch
    train_aucs = {}
    for task in TASK_NAMES:
        p = torch.cat(train_preds[task]).numpy()
        y = torch.cat(train_labels[task]).numpy()
        train_aucs[task] = roc_auc_score(y, p)

    model.eval()
    val_loss, v_batch = 0.0, 0
    val_preds = {task: [] for task in TASK_NAMES}
    val_labels = {task: [] for task in TASK_NAMES}
    with torch.no_grad():
        for features, labels in val_loader:
            logits_list = model(features)
            loss = sum(criterion(logits_list[i], labels[TASK_NAMES[i]]) for i in range(len(TASK_NAMES)))
            val_loss += loss.item()
            v_batch += 1
            for i, task in enumerate(TASK_NAMES):
                val_preds[task].append(torch.sigmoid(logits_list[i]))
                val_labels[task].append(labels[task])

    val_loss /= v_batch
    val_aucs = {}
    for task in TASK_NAMES:
        p = torch.cat(val_preds[task]).numpy()
        y = torch.cat(val_labels[task]).numpy()
        val_aucs[task] = roc_auc_score(y, p)

    print(f"Epoch {epoch+1:2d}/{N_EPOCHS} | "
          f"train_loss={train_loss:.4f} | "
          + " ".join([f"train_{task}_auc={train_aucs[task]:.4f}" for task in TASK_NAMES])
          + " | "
          + " ".join([f"val_{task}_auc={val_aucs[task]:.4f}" for task in TASK_NAMES]))

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    for task in TASK_NAMES:
        writer.add_scalar(f"AUC/train_{task}", train_aucs[task], epoch)
        writer.add_scalar(f"AUC/val_{task}", val_aucs[task], epoch)

    # 保存最佳模型（以两个任务 AUC 平均值为准）
    avg_auc = sum(val_aucs.values()) / len(val_aucs)
    if avg_auc > sum(best_auc.values()) / len(best_auc):
        best_auc = val_aucs.copy()
        best_epoch = epoch + 1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

if best_state:
    model.load_state_dict(best_state)
    torch.save(best_state, "best_mmoe.pt")
    print(f"最佳模型（epoch {best_epoch}, "
          + ", ".join([f"{task}_AUC={best_auc[task]:.4f}" for task in TASK_NAMES])
          + ")")

writer.close()
print("\n训练完成！")


# ============================================================
# 【5】TODO ⑥：业务分析 —— 看看 MMoE 学到了什么
# ============================================================
print("\n" + "=" * 60)
print("【5】MMoE 业务案例分析（多目标模型的可解释性）")
print("=" * 60)

"""
【为什么要做业务分析？】
  多目标模型不只是看 AUC，还要知道：
    - 两个任务的关系是什么？（正相关还是负相关？）
    - Gate 权重长什么样？Task A 和 Task B 偏好哪些 Expert？
    - 有没有 CTR 高但 Like 低的样本？（用户点了但不点赞）

【MMoE 特有的可解释性】
  1. Gate 权重：每个任务对每个 Expert 的偏好程度
  2. 任务相关性：Gate 权重越相似 → 任务越相关
  3. 样本级分析：某个用户/视频，两个任务的预测差异

【你需要做的前置修改】
  当前 MMoE.forward 返回 logits 列表。
  为了拿到 gate 权重，你可以在 forward 里额外返回 gate_weights 列表，
  或者加一个 analyze 方法。
"""

model.eval()
with torch.no_grad():
    # 先拿一整批验证数据做预测
    all_val_features, all_val_labels = [], []
    for feats, labs in val_loader:
        all_val_features.append({k: v.clone() for k, v in feats.items()})
        all_val_labels.append({k: v.clone() for k, v in labs.items()})
        if sum(len(next(iter(v.values()))) for v in all_val_labels) >= 5000:
            break

    # 合并成一个大 batch
    merged_feats = {}
    for k in all_val_features[0].keys():
        merged_feats[k] = torch.cat([b[k] for b in all_val_features], dim=0)
    merged_labels = {}
    for k in all_val_labels[0].keys():
        merged_labels[k] = torch.cat([b[k] for b in all_val_labels], dim=0)

    # TODO ⑥：选代表性样本做深度分析
    # 提示：
    #   ① logits_list = model(merged_feats)
    #   ② probs = {task: torch.sigmoid(logits_list[i]).squeeze() for i, task in enumerate(TASK_NAMES)}
    #   ③ 选样本：
    #        - 两个任务都预测对的（高 CTR + 高 Like，正样本）
    #        - CTR 高但 Like 低的（点了但不点赞）
    #        - CTR 低但 Like 高的（没点但点了赞——比较少见）
    #   ④ 对每个样本，打印：
    #        user_id, video_id, true_label_click, true_label_like, pred_ctr, pred_like
    #   ⑤ 手动过一遍 Gate，打印两个任务的 Gate 权重，看偏好哪个 Expert
    #   ⑥ 业务解读：
    #        - "Task Click 的 Gate 集中在 Expert 0 和 2，说明点击行为偏好某些特征组合"
    #        - "Task Like 的 Gate 集中在 Expert 1，说明点赞是另一种特征逻辑"
    #        - "两个任务 Gate 很相似 → 说明点击和点赞高度相关"
    # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
    logits_list = model(merged_feats)
    probs = {task: torch.sigmoid(logits_list[i]).squeeze() for i, task in enumerate(TASK_NAMES)}

    click_prob = probs["is_click"]
    like_prob = probs["is_like"]
    click_label = merged_labels["is_click"]
    like_label = merged_labels["is_like"]

    # 重建 flat embedding（和 forward 里第一步一致），用于计算 Gate 权重
    embs = []
    for feat_name in model.feature_names:
        embs.append(model.embeddings[feat_name](merged_feats[feat_name]))
    static_emb = torch.stack(embs, dim=1)           # [N, n_features, emb_dim]
    flat = static_emb.view(static_emb.size(0), -1)  # [N, n_features*emb_dim]

    # 定义三类代表性样本的筛选条件
    sample_cases = [
        ("两个任务都预测对的正样本", (click_label == 1) & (like_label == 1) & (click_prob > 0.7) & (like_prob > 0.7)),
        ("CTR高但Like低（点了不赞/标题党嫌疑）", (click_label == 1) & (like_label == 0) & (click_prob > 0.7) & (like_prob < 0.3)),
        ("CTR低但Like高（没点但赞——罕见）", (click_label == 0) & (like_label == 1) & (click_prob < 0.3) & (like_prob > 0.7)),
    ]

    def first_index(mask_tensor):
        idx = torch.where(mask_tensor)[0]
        return idx[0].item() if len(idx) > 0 else None

    for case_name, mask in sample_cases:
        idx = first_index(mask)
        if idx is None:
            print(f"\n{'='*60}")
            print(f"【{case_name}】未找到符合条件的样本（放宽阈值后可重现）")
            continue

        i = idx
        print(f"\n{'='*60}")
        print(f"【{case_name}】")
        print(f"{'='*60}")
        print(f"user_id={merged_feats['user_id'][i].item()}, video_id={merged_feats['video_id'][i].item()}")
        print(f"真实标签: click={int(click_label[i].item())}, like={int(like_label[i].item())}")
        print(f"MMoE预测: CTR={click_prob[i].item():.4f}, Like={like_prob[i].item():.4f}")

        # 打印两个任务的 Gate 权重（每个 Expert 的偏好程度）
        print(f"\n各任务的 Gate 权重（对 {NUM_EXPERTS} 个 Expert 的偏好）：")
        gate_weights_all = []
        for t_idx, task in enumerate(TASK_NAMES):
            gw = model.gates[t_idx](flat[i:i+1]).squeeze(0)  # [num_experts]
            gate_weights_all.append(gw)
            print(f"  Task [{task:8s}]: ", end="")
            for e_idx in range(NUM_EXPERTS):
                w = gw[e_idx].item()
                bar = "█" * int(w * 30)
                print(f"E{e_idx}={w:.3f} [{bar:<30s}]", end="  ")
            print()

        # 业务解读
        print(f"\n业务解读:")
        for t_idx, task in enumerate(TASK_NAMES):
            gw = gate_weights_all[t_idx]
            max_e = torch.argmax(gw).item()
            print(f"  → Task [{task}] 最偏好 Expert {max_e} (权重={gw[max_e].item():.3f})")

        # 对比两个任务的 Gate 相似度
        if model.n_tasks >= 2:
            gw0 = gate_weights_all[0].unsqueeze(0)
            gw1 = gate_weights_all[1].unsqueeze(0)
            sim = F.cosine_similarity(gw0, gw1, dim=1).item()
            print(f"  → 两个任务 Gate 余弦相似度: {sim:.4f}", end="")
            if sim > 0.9:
                print(" → 高度相关，两个任务共享大部分 Expert")
            elif sim < 0.3:
                print(" → 差异大，两个任务各自偏好不同 Expert（MMoE 发挥了分离作用）")
            else:
                print(" → 中度相关，部分 Expert 被共享")

        # 根据预测概率做业务判断
        cp = click_prob[i].item()
        lp = like_prob[i].item()
        if cp > 0.7 and lp > 0.7:
            print(f"  → 该视频既有吸引力(CTR高)又有质量(Like高)，应该前排推荐")
        elif cp > 0.7 and lp < 0.3:
            print(f"  → ⚠ 标题党/封面党嫌疑！用户会点但不会赞，短期 CTR 高但长期伤害用户体验")
            print(f'    建议：降低此类内容权重，或单独建一个"低质高点击"识别任务')
        elif cp < 0.3 and lp > 0.7:
            print(f"  → 罕见模式：模型认为不会点但会赞，可能样本标注异常或存在数据穿越")
        else:
            print(f"  → 普通样本，模型预测较为保守，可结合其他特征进一步排序")
    # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑
