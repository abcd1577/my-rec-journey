"""
多目标损失融合 —— PyTorch 学习实现
（3.4.3 节的理论 + 代码对照）

【三大方法对比】
┌─────────────────────┬──────────────────────┬──────────────────────────┐
│    Uncertainty Weight│        GradNorm      │    Pareto Optimization   │
├─────────────────────┼──────────────────────┼──────────────────────────┤
│ 核心思想: 不确定性大 │ 核心思想: 梯度量级   │ 核心思想: 帕累托最优     │
│ 的任务权重小        │ + 收敛速度平衡       │ 解集通过 KKT 条件求解    │
│ 机制: 可学习 σ      │ 机制: 额外梯度 loss  │ 机制: 二次规划 + 投影    │
│ 更新: SGD 随模型训练 │ 更新: 梯度下降调权   │ 更新: 闭式解交替优化     │
│ 实现复杂度: ★☆☆     │ 实现复杂度: ★★☆     │ 实现复杂度: ★★★         │
└─────────────────────┴──────────────────────┴──────────────────────────┘

【和 MMoE 代码的差异】
  - 模型结构复用 MMoE（Shared Embedding → Experts → Gates → Towers）
  - 唯一差别是 loss 融合策略（等权重 vs UWL vs GradNorm vs Pareto）
  - 本文件包含全部四种策略的训练，让你直观对比效果

运行：
    python multi_loss_fusion.py
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

import copy


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

# MMoE 结构
NUM_EXPERTS = 4
EXPERT_UNITS = [64, 32]
GATE_UNITS = [32]
TOWER_UNITS = [32, 16]

# 多目标
TASK_NAMES = ["is_click", "is_like"]

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ============================================================
# 【1】数据预处理（复用 MMoE 代码）
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
        encoded[non_na] = le.fit_transform(user_clean[feat][non_na]) + 1
    user_clean[feat] = encoded

# ── 1.2 视频特征清洗 ──
video_basic_clean = video_df.copy()
video_basic_clean["video_id"] = video_basic_clean["video_id"].astype(int)
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
tag_map = {t: i + 1 for i, t in enumerate(sorted(tag_set))}
video_basic_clean["tag"] = video_basic_clean["tag"].apply(
    lambda x: tag_map.get(str(x).split(",")[0], 0)
)

# ── 1.3 合并 ──
cols = ["user_id", "video_id", "date", "time_ms", "is_click", "is_like", "tab"]
log_clean = log_df[cols].copy()
merged = log_clean.merge(user_clean, on="user_id", how="left")
merged = merged.merge(video_basic_clean, on="video_id", how="left")
merged["user_id"] = (
    merged["user_id_enc"] if "user_id_enc" in merged.columns else merged["user_id"]
)
merged["video_id"] = (
    merged["video_id_enc"] if "video_id_enc" in merged.columns else merged["video_id"]
)
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
# 【2】Dataset（复用）
# ============================================================
class KuaiRandMultiTaskDataset(Dataset):
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
# 【3】MMoE 模型
# ============================================================
class Expert(nn.Module):
    def __init__(self, input_dim, units, dropout=0.1):
        super().__init__()
        layers = []
        for u in units:
            layers.extend([nn.Linear(input_dim, u), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = u
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Gate(nn.Module):
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
    """MMoE 模型，多任务共享 Expert + 独立 Gate + 独立 Tower"""

    def __init__(self, feature_vocabs, emb_dim, num_experts, expert_units,
                 gate_units, tower_units, task_names, dropout=0.1):
        super().__init__()
        self.feature_names = list(feature_vocabs.keys())
        self.n_features = len(self.feature_names)
        self.num_experts = num_experts
        self.task_names = task_names
        self.n_tasks = len(task_names)

        self.embeddings = nn.ModuleDict()
        for feat_name in self.feature_names:
            self.embeddings[feat_name] = nn.Embedding(
                feature_vocabs[feat_name], emb_dim, padding_idx=0
            )

        self.linear_weights = nn.ModuleDict()
        for feat_name in self.feature_names:
            self.linear_weights[feat_name] = nn.Embedding(
                feature_vocabs[feat_name], 1, padding_idx=0
            )
        self.global_bias = nn.Parameter(torch.zeros(1))

        input_dim = self.n_features * emb_dim

        # ── TODO ②：Expert ──
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.experts = nn.ModuleList(
            [Expert(input_dim, expert_units, dropout=dropout) for _ in range(num_experts)]
        )
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑
        expert_out_dim = expert_units[-1]

        # ── TODO ③：Gate ──
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.gates = nn.ModuleList(
            [Gate(input_dim, gate_units, num_experts) for _ in range(self.n_tasks)]
        )
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

        # ── TODO ④：Tower ──
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.towers = nn.ModuleList()
        for _ in range(self.n_tasks):
            layers = []
            dim = expert_out_dim
            for u in tower_units:
                layers.extend([nn.Linear(dim, u), nn.ReLU(), nn.Dropout(dropout)])
                dim = u
            layers.append(nn.Linear(dim, 1))
            self.towers.append(nn.Sequential(*layers))
        # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑

    def forward(self, features):
        embs = []
        for feat_name in self.feature_names:
            embs.append(self.embeddings[feat_name](features[feat_name]))
        static_emb = torch.stack(embs, dim=1)
        flat = static_emb.view(static_emb.size(0), -1)

        linear_logit = self.global_bias.expand(static_emb.size(0))
        for feat_name in self.feature_names:
            lin = self.linear_weights[feat_name](features[feat_name])
            linear_logit = linear_logit + lin.squeeze(1)

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

        return logits_list


# ============================================================
# 【4】评估工具
# ============================================================
criterion = nn.BCEWithLogitsLoss()


def evaluate(model, loader):
    """在验证集上计算每个任务的平均 loss 和 AUC"""
    model.eval()
    total_loss = 0.0
    n_batch = 0
    preds = {task: [] for task in TASK_NAMES}
    trues = {task: [] for task in TASK_NAMES}

    with torch.no_grad():
        for features, labels in loader:
            logits_list = model(features)
            loss = sum(
                criterion(logits_list[i], labels[TASK_NAMES[i]])
                for i in range(len(TASK_NAMES))
            )
            total_loss += loss.item()
            n_batch += 1
            for i, task in enumerate(TASK_NAMES):
                preds[task].append(torch.sigmoid(logits_list[i]))
                trues[task].append(labels[task])

    avg_loss = total_loss / n_batch
    aucs = {}
    for task in TASK_NAMES:
        p = torch.cat(preds[task]).numpy()
        y = torch.cat(trues[task]).numpy()
        aucs[task] = roc_auc_score(y, p)
    return avg_loss, aucs


# ============================================================
# 【5】训练函数工厂
# ============================================================
def build_model():
    """创建一个新的 MMoE 模型"""
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
    return model


# ============================================================
# 【5.1】等权重融合 —— 基线（之前 MMoE 的做法）
# ============================================================
def train_equal_weight(model, train_loader, val_loader, n_epochs=N_EPOCHS):
    """
    等权重融合: L = L_click + L_like
    """
    print(f"\n{'='*60}")
    print("【5.1】等权重融合（基线）")
    print(f"{'='*60}")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(n_epochs):
        train_loss, n_batch = 0.0, 0
        for features, labels in train_loader:
            logits_list = model(features)
            loss = sum(
                criterion(logits_list[i], labels[TASK_NAMES[i]])
                for i in range(len(TASK_NAMES))
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batch += 1
        train_loss /= n_batch

        val_loss, val_aucs = evaluate(model, val_loader)
        print(f"  Epoch {epoch+1:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              + " | ".join([f"{t}_auc={val_aucs[t]:.4f}" for t in TASK_NAMES]))

    _, final_aucs = evaluate(model, val_loader)
    return final_aucs


# ============================================================
# 【5.2】Uncertainty Weight —— 基于不确定性的自适应加权
# ============================================================
def train_uncertainty_weight(model, train_loader, val_loader, n_epochs=N_EPOCHS):
    """
    Uncertainty Weighted Loss (UWL)

    理论（来自 Kendall et al. 2018）:
      对于多分类任务（如 BCE），总损失为:
        L_total = (1/(2*σ₁²)) * L₁ + (1/(2*σ₂²)) * L₂ + log(σ₁) + log(σ₂)

      其中 σ 是可学习的噪声参数，表示任务的不确定性。

    直观理解:
      - 任务不确定性越大 (σ 大)，(1/σ²) 越小 → 该任务权重越小
      - log(σ) 是正则项，防止 σ 无限增大
    
    TODO ⑤：实现 UWL 训练循环
      提示：
        ① 创建 nn.Parameter 存储 log σ（初始化为 0，保证σ=1）
        ② 前向：logits = model(features)
        ③ loss_i = BCE(logits[i], labels[TASK_NAMES[i]])  # 先不求和
        ④ L_uwl = (1/(2*σ²)) * loss_i_sum + log σ 求和
        ⑤ 反向传播，同时更新模型参数和 σ
    """
    print(f"\n{'='*60}")
    print("【5.2】Uncertainty Weight 训练")
    print(f"{'='*60}")

    # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓

    # ── 可学习的 log_sigma（初始化 log_sigma=0 → sigma=1） ──
    log_sigma = nn.ParameterDict()
    for task in TASK_NAMES:
        log_sigma[task] = nn.Parameter(torch.zeros(1))

    optimizer = optim.Adam(
        list(model.parameters()) + list(log_sigma.parameters()),
        lr=LR,
    )

    model.train()
    for epoch in range(n_epochs):
        train_loss, n_batch = 0.0, 0
        for features, labels in train_loader:
            logits_list = model(features)

            # 每个任务单独算 loss
            task_losses = {}
            for i, task in enumerate(TASK_NAMES):
                task_losses[task] = criterion(logits_list[i], labels[task])

            # UWL 融合: L = sum(1/(2*σ²) * L_i) + sum(log σ)
            # 对 BCE 任务，通常使用 1/(2σ²) 系数
            uwl_loss = torch.tensor(0.0)
            for task in TASK_NAMES:
                sigma = torch.exp(log_sigma[task])  # σ = exp(log_sigma) > 0
                uwl_loss = uwl_loss + (
                    task_losses[task] / (2.0 * sigma ** 2) + log_sigma[task]
                )

            optimizer.zero_grad()
            uwl_loss.backward()
            optimizer.step()
            train_loss += uwl_loss.item()
            n_batch += 1

        train_loss /= n_batch
        # 打印当前 sigma
        sigma_str = " | ".join(
            [f"{t}_σ={torch.exp(log_sigma[t]).item():.4f}" for t in TASK_NAMES]
        )
        val_loss, val_aucs = evaluate(model, val_loader)
        print(f"  Epoch {epoch+1:2d} | train_loss={train_loss:.4f} | {sigma_str} | "
              + " | ".join([f"{t}_auc={val_aucs[t]:.4f}" for t in TASK_NAMES]))

    print(f"  最终学习到的 σ: " + ", ".join(
        [f"{t}={torch.exp(log_sigma[t]).item():.4f}" for t in TASK_NAMES]
    ))
    # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑
    _, final_aucs = evaluate(model, val_loader)
    return final_aucs


# ============================================================
# 【5.3】GradNorm —— 梯度标准化动态调权
# ============================================================
def train_gradnorm(model, train_loader, val_loader, n_epochs=N_EPOCHS, alpha=1.5):
    """
    GradNorm（Chen et al. 2018）

    理论:
      传统多任务训练的问题是不同任务 loss 量级不同，导致大 loss 主导梯度。
      GradNorm 在训练中额外优化 loss 权重 w_i，使所有任务的梯度量级接近。
    
    核心变量:
      w_i(t): 任务 i 在第 t 步的 loss 权重（可学习）
      G_W^(i)(t) = ||∇_W (w_i(t) * L_i(t))||₂
        任务 i 的加权 loss 对共享参数 W 的梯度二范数
      Ḡ_W(t) = mean(G_W^(i)(t))
        所有任务的平均梯度范数
      r_i(t) = L̃_i(t) / mean(L̃_i(t))
        任务 i 的相对训练速度
        其中 L̃_i(t) = L_i(t) / L_i(0) 是 loss 相对于初始值的比例
      α: 超参数，控制牵引强度（越大越鼓励任务训练速度一致）

    梯度损失:
      L_grad = Σ_i |G_W^(i)(t) - Ḡ_W(t) * [r_i(t)]^α|₁

    关键实现细节:
      ① 共享参数 W 取最后一层 shared layer 的参数
         在 MMoE 中，可以是所有 Expert 的参数（所有任务共享的层）
      ② w_i 初始化为 1/N (所有任务等权)
      ③ w_i 和模型参数用不同的优化器
      ④ 更新 w_i 后再做归一化: sum(w_i) = N

    TODO ⑥：实现 GradNorm 训练循环
      提示：
        ① 初始化 w_i = nn.Parameter(torch.ones(n_tasks))，初始为 1
           注：实际权重为 softmax(w) * n_tasks，保证和为 n_tasks
        ② 前向计算 logits
        ③ 对每个任务: weighted_loss_i = w_i_norm * L_i
        ④ total_loss = sum(weighted_loss_i)
        ⑤ 算梯度时只保留对共享参数 W 的梯度
           → 计算 G_W^(i)(t) = ||∇_W w_i_norm * L_i||₂
        ⑥ 计算 L_grad，更新 w_i
        ⑦ 重新归一化 w_i
        ⑧ 最后用 total_loss 更新模型参数（用另一个优化器）
    """
    print(f"\n{'='*60}")
    print("【5.3】GradNorm 训练")
    print(f"{'='*60}")

    # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓

    # ── 可学习的 loss 权重 ──
    # 直接优化 w（不是 raw logit），每次 step 后手动归一化 sum(w)=n_tasks
    w = nn.Parameter(torch.ones(model.n_tasks, dtype=torch.float))

    # ── 共享参数 W：取所有 Expert 的参数 ──
    shared_params = []
    for expert in model.experts:
        for p in expert.parameters():
            shared_params.append(p)

    # ── 两个优化器：一个给模型，一个给权重 ──
    optimizer_model = optim.Adam(model.parameters(), lr=LR)
    optimizer_w = optim.Adam([w], lr=LR * 10)  # w 的学习率可以稍大

    # ── 记录初始 loss，用于计算 r_i(t) ──
    model.eval()
    with torch.no_grad():
        init_losses = []
        for features, labels in train_loader:
            logits_list = model(features)
            for i in range(model.n_tasks):
                init_losses.append(criterion(logits_list[i], labels[TASK_NAMES[i]]).item())
            break  # 只用 1 个 batch
    init_losses = np.array(init_losses)
    print(f"  初始任务 loss: {init_losses}")

    alpha = 1.5  # GradNorm 超参：控制恢复力度的指数

    model.train()
    for epoch in range(n_epochs):
        train_loss, n_batch = 0.0, 0
        for features, labels in train_loader:
            # ── 第 1 步：前向 ──
            logits_list = model(features)

            # 当前规范化权重: sum(w_norm) = n_tasks
            w_sum = w.sum().clamp_min(1e-6)
            w_norm = w / w_sum * model.n_tasks  # [n_tasks]

            # ── 第 2 步：计算每个任务的 loss ──
            task_losses = []
            for i in range(model.n_tasks):
                task_losses.append(criterion(logits_list[i], labels[TASK_NAMES[i]]))
            total_weighted_loss = sum(
                w_norm[i] * task_losses[i] for i in range(model.n_tasks)
            )

            # ── 第 3 步：计算 C_i = ||∇_W L_i||（不加权梯度范数） ──
            # 注意：GradNorm 原始论文定义 G_W^(i) = ||∇_W (w_i * L_i)|| = w_i * ||∇_W L_i||
            # 所以这里先算 C_i = ||∇_W L_i||，再得 G_i = w_i * C_i
            C_list = []
            for i in range(model.n_tasks):
                optimizer_model.zero_grad()
                task_losses[i].backward(retain_graph=True)
                grad_sq_sum = 0.0
                for p in shared_params:
                    if p.grad is not None:
                        grad_sq_sum += p.grad.data.norm(2).item() ** 2
                C_list.append(np.sqrt(grad_sq_sum))

            # 清零模型梯度，准备后续更新
            optimizer_model.zero_grad()

            # G_i = w_i * C_i
            G_list = [w_norm[i].item() * C_list[i] for i in range(model.n_tasks)]
            mean_G = np.mean(G_list)

            # ── 第 4 步：计算 r_i(t)（相对训练速度） ──
            L_vals = np.array([task_losses[i].item() for i in range(model.n_tasks)])
            r_vals = L_vals / init_losses  # 当前 loss / 初始 loss
            r_mean = r_vals.mean()
            r_norm = r_vals / r_mean  # r_i(t)

            # ── 第 5 步：手动计算 L_grad 对 w_i 的梯度 ──
            # L_grad = Σ |G_i - target_i|，其中 target_i = mean_G * (r_i)^α
            # ∂L_grad/∂w_i = sign(G_i - target_i) * C_i
            #
            # 由于有约束 sum(w)=K，梯度需要投影到正交于约束的方向：
            # grad_proj = grad - mean(grad)
            target_list = [mean_G * (r_norm[i] ** alpha) for i in range(model.n_tasks)]
            grad_w = np.array([
                np.sign(G_list[i] - target_list[i]) * C_list[i]
                for i in range(model.n_tasks)
            ])
            grad_w = grad_w - grad_w.mean()  # 投影到 sum=0 的约束空间

            # 手动赋值梯度
            if w.grad is None:
                w.grad = torch.tensor(grad_w, dtype=torch.float)
            else:
                w.grad.data = torch.tensor(grad_w, dtype=torch.float)

            optimizer_w.step()

            # 重新归一化，保证 sum(w) = n_tasks
            with torch.no_grad():
                w.data = w.data.clamp(min=0.1)  # 防止权重过小
                w.data = w.data / w.data.sum() * model.n_tasks
                w_new = w.data.clone()

            # ── 第 6 步：用更新后的 w_norm 重新计算加权 loss，更新模型参数 ──
            # 注意：w 已被 in-place 修改，之前创建的 total_weighted_loss 计算图已失效
            # 需要用新的 w_norm 重新构建加权 loss
            with torch.no_grad():
                w_sum = w.sum().clamp_min(1e-6)
                w_norm = w / w_sum * model.n_tasks
            total_weighted_loss = sum(
                w_norm[i] * task_losses[i] for i in range(model.n_tasks)
            )
            optimizer_model.zero_grad()
            total_weighted_loss.backward()
            optimizer_model.step()

            train_loss += total_weighted_loss.item()
            n_batch += 1

        train_loss /= n_batch
        print(f"  Epoch {epoch+1:2d} | train_loss={train_loss:.4f} | "
              + " | ".join([f"w_{TASK_NAMES[i]}={w_new[i].item():.4f}" for i in range(model.n_tasks)]))

    _, final_aucs = evaluate(model, val_loader)
    # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑
    return final_aucs


# ============================================================
# 【5.4】Pareto Optimization —— 帕累托优化框架
# ============================================================
def solve_pareto_weights(grad_norm_list, c_min=0.1):
    """
    求解帕累托最优权重 w_i

    理论（Lin et al. 2019 PE-LTR）:
      min_w  ||Σ_i(w_i * g_i)||₂²
      s.t.   Σ w_i = 1, w_i ≥ c_i

      其中 g_i = ∇θ L_i(θ) 是第 i 个任务的梯度向量。
      这个二次规划的解可以解析得到。

    实现步骤:
      ① g_i 是任务 i 对所有参数梯度的 flatten 拼接向量
      ② 构造 G = [g₁, g₂, ..., g_K]，K 是任务数
      ③ 目标函数: ||G @ w||₂² = w^T G^T G w
      ④ 通过拉格朗日乘子法 + 投影求解

    TODO ⑦：实现帕累托权重求解
      提示：
        ① 将每个任务的梯度 flatten 成向量
        ② 构造梯度矩阵 G: [n_params, n_tasks]
        ③ 计算 Gram 矩阵: M = G^T @ G  [n_tasks, n_tasks]
        ④ 无约束解: w* = M^{-1} @ 1 / (1^T @ M^{-1} @ 1)
        ⑤ 投影到 w_i ≥ c_min 的约束空间
        ⑥ 对投影后的结果做 softmax 确保和为 1
    """

    # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
    K = grad_norm_list.shape[-1]  # 任务数

    # 构造梯度矩阵 G: [n_params, K]
    grads_flat = []
    for i in range(K):
        grads_flat.append(grad_norm_list[:, i])
    G = torch.stack(grads_flat, dim=1)  # [n_params, K]

    # Gram 矩阵: M = G^T @ G  [K, K]
    M = G.T @ G

    try:
        # 无约束解: w = M^{-1} @ 1 / (1^T @ M^{-1} @ 1)
        ones = torch.ones(K, dtype=torch.float)
        M_inv = torch.linalg.inv(M + 1e-6 * torch.eye(K))  # 加正则防止奇异
        w_unconstrained = M_inv @ ones / (ones @ M_inv @ ones)
    except Exception:
        # 如果求逆失败，回退到等权
        return torch.ones(K) / K

    # 投影到 w_i ≥ c_min
    w = w_unconstrained.clone()
    w = torch.clamp(w, min=c_min)

    # 重新归一化
    w = w / w.sum()
    return w
    # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑


def train_pareto(model, train_loader, val_loader, n_epochs=N_EPOCHS, c_min=0.1):
    """
    Pareto Optimization 训练

    交替更新：
      第 1 步（固定 w）：用加权 loss 更新模型参数
      第 2 步（固定 θ）：求解二次规划更新 w

    实现细节：
      ① 每步训练都需要计算所有任务对共享参数的梯度
      ② 用这些梯度求解二次规划
      ③ 用求解出的 w 作为下一轮的损失权重
      ④ 权重不能太低（c_min 防止某个任务被完全忽略）
    """
    print(f"\n{'='*60}")
    print("【5.4】Pareto Optimization 训练")
    print(f"{'='*60}")

    # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓

    # ── 共享参数 W：所有 Expert 的参数 ──
    shared_params = []
    for expert in model.experts:
        for p in expert.parameters():
            shared_params.append(p)

    # ── 优化器 ──
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ── 初始化权重 ──
    w = torch.full((model.n_tasks,), 1.0 / model.n_tasks)

    model.train()
    for epoch in range(n_epochs):
        train_loss, n_batch = 0.0, 0
        for features, labels in train_loader:
            # ── 第 1 步：计算每个任务的梯度 g_i ──
            logits_list = model(features)

            # 存储所有任务的梯度
            task_grads = []

            for i in range(model.n_tasks):
                optimizer.zero_grad()
                loss_i = criterion(logits_list[i], labels[TASK_NAMES[i]])
                loss_i.backward(retain_graph=(i < model.n_tasks - 1))

                # 收集共享参数梯度并 flatten
                grads = []
                for p in shared_params:
                    if p.grad is not None:
                        grads.append(p.grad.data.view(-1))
                task_grads.append(torch.cat(grads))  # [n_params]

            # 清零梯度
            optimizer.zero_grad()

            # ── 第 2 步：用 Pareto 求解权重 ──
            grad_matrix = torch.stack(task_grads, dim=1)  # [n_params, n_tasks]
            w_new = solve_pareto_weights(grad_matrix, c_min=c_min)
            w = w_new.detach()  # 防止计算图继续累积

            # ── 第 3 步：用求解出的权重计算加权 loss，更新模型 ──
            logits_list = model(features)
            total_loss = sum(
                w[i] * criterion(logits_list[i], labels[TASK_NAMES[i]])
                for i in range(model.n_tasks)
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            n_batch += 1

        train_loss /= n_batch
        print(f"  Epoch {epoch+1:2d} | train_loss={train_loss:.4f} | "
              + " | ".join([f"w_{TASK_NAMES[i]}={w[i].item():.4f}" for i in range(model.n_tasks)]))

    _, final_aucs = evaluate(model, val_loader)
    # ↑↑↑↑↑ 你的代码 ↑↑↑↑↑
    return final_aucs


# ============================================================
# 【6】TODO ⑧：对比分析
# ============================================================
print("\n" + "=" * 60)
print("【6】四种 loss 融合策略对比")
print("=" * 60)

# ── 6.1 等权重 ──
model_equal = build_model()
auc_equal = train_equal_weight(model_equal, train_loader, val_loader)

# ── 6.2 Uncertainty Weight ──
model_uwl = build_model()
auc_uwl = train_uncertainty_weight(model_uwl, train_loader, val_loader)

# ── 6.3 GradNorm ──
model_gn = build_model()
auc_gn = train_gradnorm(model_gn, train_loader, val_loader)

# ── 6.4 Pareto ──
model_pareto = build_model()
auc_pareto = train_pareto(model_pareto, train_loader, val_loader)

# ── 6.5 对比总表 ──
print("\n" + "=" * 60)
print("对比总表")
print("=" * 60)
header = f"{'策略':<25s} | {'click AUC':<10s} | {'like AUC':<10s} | {'avg AUC':<10s}"
print(header)
print("-" * len(header))
for name, auc in [
    ("等权重（基线）", auc_equal),
    ("Uncertainty Weight", auc_uwl),
    ("GradNorm", auc_gn),
    ("Pareto Optimization", auc_pareto),
]:
    avg = (auc[TASK_NAMES[0]] + auc[TASK_NAMES[1]]) / 2
    print(f"{name:<25s} | {auc[TASK_NAMES[0]]:<10.4f} | {auc[TASK_NAMES[1]]:<10.4f} | {avg:<10.4f}")

# ── 6.6 TODO ⑧：业务解读 ──
print("\n" + "=" * 60)
print("【TODO ⑧】业务解读")
print("=" * 60)
print("""
等权重（基线）: 两个任务完全平等，如果任务量级不同，大 loss 主导优化。
Uncertainty Weight: 任务不确定性越大权重越小，适合有噪声标注的场景。
GradNorm: 自动平衡梯度量级和收敛速度，适合各任务收敛速度不一的场景。
Pareto Optimization: 在梯度方向冲突时找帕累托解，适合任务严重冲突的场景。

【选型建议】
  - 任务间无明显冲突 → 等权重即可
  - 标注质量有差异 → Uncertainty Weight
  - 收敛速度不均衡 → GradNorm
  - 梯度方向严重冲突 → Pareto Optimization

【你的观察】
  从对比结果看，哪种方法在你的数据集上表现最好？
  两个任务的 AUC 差距是缩小了还是扩大了？为什么？
  尝试调参 (α, c_min) 看效果变化。
""")

print("\n训练全部完成！")
