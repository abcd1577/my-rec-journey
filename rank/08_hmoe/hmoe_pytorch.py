"""
HMoE 精排模型 —— PyTorch 实现
（Hierarchical Mixture-of-Experts for Multi-Domain CTR）

【与 fun-rec 源码对齐】
- 数据集：KuaiRand-1K
- 核心创新：多个 Expert 共享学习，每个场景（Domain）有独立的 Gate + Tower
- 评估：AUC（按 Domain 分组计算）

【HMoE 解决了什么问题？】
  多场景推荐的问题：
    - 首页推荐、关注页、搜索页 用户行为差异大
    - 用一个模型拟合所有场景 → 大场景主导、小场景被淹没
    - 每个场景独立建模 → 小场景数据稀疏、无法迁移大场景知识

  HMoE 的答案：
    - 共享多个 Expert：所有场景共享底层知识（大场景数据丰富，学好共性）
    - 每个场景有独立的 Gate：选择最适合自己的 Expert 组合
    - 每个场景有独立的 Tower：在 Expert 融合后，做场景特有的映射
    - 样本按 domain_id 走对应的路径：只更新自己场景的 Gate + Tower

【核心结构】
  input ──→ [Expert_0, Expert_1, Expert_2, Expert_3]  ← 所有场景共享
               ↑      ↑      ↑      ↑
           Gate_0  Gate_1  Gate_2  Gate_3   ← 每个场景(domain)一个 Gate
               ↓      ↓      ↓      ↓
           Tower_0 Tower_1 Tower_2  Tower_3  ← 每个场景一个 Tower
               ↓      ↓      ↓      ↓
            Domain  Domain  Domain  Domain
              0       1       2       3

【与 MMoE 的区别】
  MMoE：多任务（点击、点赞）→ 每个任务一个 Gate
  HMoE：多场景（首页、关注页）→ 每个场景一个 Gate + Tower
        但任务只有一个（CTR），只是不同场景的数据分布不同

【Domain 怎么定义？】
  本代码中，用 video_type 作为 domain indicator（快手视频类型）
  把不同的 video_type 映射到 0~N-1 的 domain_id

【参考 fun-rec 章节】
  - 3.5.1.1 HMoE（多场景建模 → 多塔结构）
  - fun-rec 源码：funrec/models/hmoe.py
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from pathlib import Path

# ─────────────────────────────────────────────────
# 【0】全局超参数 & 设备
# ─────────────────────────────────────────────────
SEED = 42
EMB_DIM = 8
NUM_EXPERTS = 4
EXPERT_UNITS = [64, 32]
GATE_UNITS = [32]
TOWER_UNITS = [32, 16]

BATCH_SIZE = 256
LR = 1e-3
N_EPOCHS = 5

KUAIRAND_PATH = Path("/Users/qiruihou/Desktop/学习/推荐算法/dataset/kuairand/KuaiRand-1K")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ============================================================
# 【1】数据加载 & 特征工程
# ============================================================
print("=" * 60)
print("【1】数据加载 & 特征工程")
print("=" * 60)

user_df = pd.read_csv(KUAIRAND_PATH / "data/user_features_1k.csv")
video_df = pd.read_csv(KUAIRAND_PATH / "data/video_features_basic_1k.csv")
log_df = pd.read_csv(KUAIRAND_PATH / "data/log_standard_4_22_to_5_08_1k.csv")

print(f"  用户表: {user_df.shape}, 视频表: {video_df.shape}, 行为表: {log_df.shape}")

# ── 1.1 用户特征清洗 ──
user_clean = user_df.copy()
user_clean["user_id"] = user_clean["user_id"].astype(int)
TEXT_FEATS = [
    "user_active_degree", "is_live_streamer", "is_video_author",
    "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
]
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
    video_basic_clean[feat + "_enc"] = le.fit_transform(video_basic_clean[feat].astype(str)) + 1
    if feat != "video_id":
        video_basic_clean[feat] = video_basic_clean[feat + "_enc"]
        del video_basic_clean[feat + "_enc"]

# tag 列单独处理：先 fillna("-1")，合并后再 astype(int)
video_basic_clean["tag"] = video_basic_clean["tag"].fillna("-1")

# ── 1.3 合并 ──
merged = log_df.merge(user_clean, on="user_id", how="left")
merged = merged.merge(video_basic_clean, on="video_id", how="left")
merged["tag"] = merged["tag"].astype(str)
merged["tag"] = merged["tag"].str.split(",").str[0]
merged["tag"] = merged["tag"].replace({"nan": "0", "-1": "0"})
merged["tag"] = merged["tag"].astype(int)

# ── TODO ①：构造 Domain 特征 ──
# HMoE 是多场景模型，需要 domain_indicator 告诉模型"这个样本属于哪个场景"
# 提示：
#   ① 用 video_type 作为 domain（不同视频类型 = 不同场景）
#   ② 把 video_type 映射到 0~num_domains-1 的整数
#   ③ 注意：domain_id 从 0 开始，和 Gate/Tower 的索引对应
#   ④ 如果 domain 样本太少，可以合并相似的 domain
# ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
NUM_DOMAINS = merged["video_type"].nunique()
domain_map = {v: i for i, v in enumerate(sorted(merged["video_type"].unique()))}
merged["domain_id"] = merged["video_type"].map(domain_map).astype(int)
print(f"  Domain 数量: {NUM_DOMAINS}, 各domain样本数:\n{merged['domain_id'].value_counts().sort_index()}")
# ↑↑↑↑ 你的代码 ↑↑↑↑

# ── 1.4 确定特征 ──
FEATURE_NAMES = [
    "user_id", "video_id", "author_id", "video_type", "upload_type",
    "visible_status", "music_id", "music_type", "tag",
    "user_active_degree", "is_live_streamer", "is_video_author",
    "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
]
# 确保所有特征都存在
FEATURE_NAMES = [f for f in FEATURE_NAMES if f in merged.columns]
feature_vocabs = {f: int(merged[f].max()) + 2 for f in FEATURE_NAMES}

# ── 1.5 标签 & 划分 ──
merged["is_click"] = merged["is_click"].astype(np.float32)
train_df = merged[merged["date"] <= 20220427].copy()
val_df = merged[merged["date"] > 20220427].copy()
print(f"  训练集: {len(train_df)}, 验证集: {len(val_df)}")


# ============================================================
# 【2】PyTorch Dataset & DataLoader
# ============================================================
class KuaiRandDataset(Dataset):
    def __init__(self, df, feature_names):
        self.features = {f: torch.LongTensor(df[f].fillna(0).astype(int).values) for f in feature_names}
        self.labels = torch.FloatTensor(df["is_click"].values)
        self.domains = torch.LongTensor(df["domain_id"].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {f: self.features[f][idx] for f in self.features}
        item["domain_id"] = self.domains[idx]
        return item, self.labels[idx]

train_ds = KuaiRandDataset(train_df, FEATURE_NAMES)
val_ds = KuaiRandDataset(val_df, FEATURE_NAMES)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 【3】HMoE 核心模块
# ============================================================
class Expert(nn.Module):
    """单个 Expert：普通 DNN"""
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
    """单个 Gate：DNN + softmax，输出对 Expert 的权重"""
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


class DomainTower(nn.Module):
    """Domain Tower：将 Expert 融合结果映射到 1 个 logit"""
    def __init__(self, input_dim, tower_units, dropout=0.1):
        super().__init__()
        layers = []
        dim = input_dim
        for u in tower_units:
            layers.extend([nn.Linear(dim, u), nn.ReLU(), nn.Dropout(dropout)])
            dim = u
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HMoE(nn.Module):
    """
    HMoE = Shared Embedding → Multiple Experts → Domain-specific Gates → Domain Towers

    结构：
      ① static_emb = concat([emb(feat) for feat in features])  # [B, N*D]
      ② experts = [Expert_i(static_emb) for i in range(num_experts)]  # 每个 [B, expert_dim]
         expert_stack = torch.stack(experts, dim=1)  # [B, num_experts, expert_dim]
      ③ 对每个 domain d：
           gate_d = Gate_d(static_emb)  # [B, num_experts]
           fusion_d = sum(gate_d[:, i] * expert_stack[:, i, :])  # [B, expert_dim]
      ④ tower_d = Tower_d(fusion_d)  # [B, 1]
      ⑤ 根据 domain_id 选择对应的 tower_out：
           logits[b] = tower_{domain_id[b]}[b]
    """

    def __init__(self, feature_vocabs, emb_dim, num_experts, expert_units,
                 gate_units, tower_units, num_domains, dropout=0.1):
        super().__init__()
        self.feature_names = list(feature_vocabs.keys())
        self.n_features = len(self.feature_names)
        self.num_experts = num_experts
        self.num_domains = num_domains

        # Embedding
        self.embeddings = nn.ModuleDict()
        for feat_name in self.feature_names:
            self.embeddings[feat_name] = nn.Embedding(
                feature_vocabs[feat_name], emb_dim, padding_idx=0
            )

        # 线性部分（和 MMoE 一样）
        self.linear_weights = nn.ModuleDict()
        for feat_name in self.feature_names:
            self.linear_weights[feat_name] = nn.Embedding(
                feature_vocabs[feat_name], 1, padding_idx=0
            )
        self.global_bias = nn.Parameter(torch.zeros(1))

        input_dim = self.n_features * emb_dim

        # ── TODO ②：共享 Expert ──
        # 提示：
        #   ① self.experts = nn.ModuleList([Expert(...) for _ in range(num_experts)])
        #   ② 所有 domain 共享同一组 Expert
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.experts = nn.ModuleList([Expert(input_dim, expert_units, dropout) for _ in range(num_experts)])
        # ↑↑↑↑ 你的代码 ↑↑↑↑
        expert_out_dim = expert_units[-1]

        # ── TODO ③：Domain Gate ──
        # 每个 domain 有独立的 Gate，选择 Expert
        # 提示：self.gates = nn.ModuleList([Gate(...) for _ in range(num_domains)])
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.gates = nn.ModuleList([Gate(input_dim, gate_units, num_experts) for _ in range(num_domains)])
        # ↑↑↑↑ 你的代码 ↑↑↑↑

        # ── TODO ④：Domain Tower ──
        # 每个 domain 有独立的 Tower，输入=expert_out_dim，输出=1 个 logit
        # 注意：最后要加 Linear(..., 1)，不要 ReLU
        # 提示：self.towers = nn.ModuleList([DomainTower(...) for _ in range(num_domains)])
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        self.towers = nn.ModuleList([DomainTower(expert_out_dim, tower_units, dropout) for _ in range(num_domains)])
        # ↑↑↑↑ 你的代码 ↑↑↑↑

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

        # ── 3. TODO ⑤：HMoE 核心逻辑 ──
        # 提示：
        #   ① expert_outs = [expert(flat) for expert in self.experts]
        #      expert_stack = torch.stack(expert_outs, dim=1)   # [B, num_experts, expert_dim]
        #   ② 对每个 domain d，计算 gate 权重和融合输出：
        #        gate_weights = self.gates[d](flat)              # [B, num_experts]
        #        gate_expanded = gate_weights.unsqueeze(-1)      # [B, num_experts, 1]
        #        fusion_d = (gate_expanded * expert_stack).sum(dim=1)  # [B, expert_dim]
        #        tower_out_d = self.towers[d](fusion_d)          # [B]
        #   ③ 收集所有 domain 的 tower_out：domain_logits[d] = tower_out_d
        #   ④ 根据 domain_id 选择：
        #        domain_ids = features["domain_id"]              # [B]
        #        logits[b] = domain_logits[domain_ids[b]][b]
        #
        # 关键：同一个 batch 里可能有多个 domain 的样本，
        #       需要为每个 domain 都算一遍，然后按 domain_id mask 选择
        # ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
        domain_logits = []
        expert_outs = [expert(flat) for expert in self.experts]
        expert_stack = torch.stack(expert_outs, dim=1)  # [B, num_experts, expert_dim]
        for d in range(self.num_domains):
            gate_weights = self.gates[d](flat)
            gate_expanded = gate_weights.unsqueeze(-1)
            fusion_d = (gate_expanded * expert_stack).sum(dim=1)
            tower_out_d = self.towers[d](fusion_d)
            domain_logits.append(tower_out_d)
        domain_ids = features["domain_id"]
        logits = torch.stack(domain_logits, dim=1)  # [B, num_domains]
        logits = logits[torch.arange(len(domain_ids)), domain_ids]  # [B]
        # ↑↑↑↑ 你的代码 ↑↑↑↑

        return logits + linear_logit


# ============================================================
# 【4】训练 & 评估（参考代码，可直接使用）
# ============================================================
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_domains = [], [], []
    with torch.no_grad():
        for features, labels in loader:
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_domains.extend(features["domain_id"].cpu().numpy())

    # 整体 AUC
    overall_auc = roc_auc_score(all_labels, all_preds)

    # 各 domain 的 AUC
    domain_aucs = {}
    for d in range(NUM_DOMAINS):
        mask = np.array(all_domains) == d
        if mask.sum() > 10 and len(np.unique(np.array(all_labels)[mask])) > 1:
            domain_aucs[d] = roc_auc_score(
                np.array(all_labels)[mask], np.array(all_preds)[mask]
            )
        else:
            domain_aucs[d] = None

    return overall_auc, domain_aucs


def train(model, train_loader, val_loader, n_epochs=N_EPOCHS, save_path="best_hmoe.pt"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(n_epochs):
        model.train()
        train_loss, n_batch = 0.0, 0
        for features, labels in train_loader:
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batch += 1

        train_loss /= n_batch
        overall_auc, domain_aucs = evaluate(model, val_loader)
        auc_str = " | ".join([f"D{d}={domain_aucs.get(d, 0):.4f}" for d in range(NUM_DOMAINS) if domain_aucs.get(d) is not None])
        print(f"  Epoch {epoch+1:2d} | loss={train_loss:.4f} | overall_AUC={overall_auc:.4f} | {auc_str}")

        if overall_auc > best_auc:
            best_auc = overall_auc
            torch.save(model.state_dict(), save_path)

    print(f"  最佳 overall AUC: {best_auc:.4f}")
    return best_auc


# ============================================================
# 【5】训练 HMoE（需要先完成 TODO ②~⑤）
# ============================================================
print("\n" + "=" * 60)
print("【5】HMoE 训练")
print("=" * 60)

model = HMoE(
    feature_vocabs, emb_dim=EMB_DIM, num_experts=NUM_EXPERTS,
    expert_units=EXPERT_UNITS, gate_units=GATE_UNITS,
    tower_units=TOWER_UNITS, num_domains=NUM_DOMAINS, dropout=0.1,
)
print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

train(model, train_loader, val_loader)


# ============================================================
# 【6】TODO ⑥：深度分析 —— 各 Domain 的 Gate 权重差异
# ============================================================
print("\n" + "=" * 60)
print("【6】TODO ⑥：各 Domain 的 Gate 权重差异分析")
print("=" * 60)
"""
目标：
  ① 加载训练好的 HMoE 模型
  ② 取验证集中的样本，按 domain 分组
  ③ 对每个 domain，计算该 domain 的 Gate 在该 domain 样本上的平均权重
  ④ 对比不同 domain 的 Expert 偏好差异
  ⑤ 业务解读：
      - "Domain 0 的 Gate 集中在 Expert 0，说明这个场景偏好某种特征"
      - "Domain 1 和 Domain 2 的 Gate 很相似，说明这两个场景数据分布接近"
      - "某个 Expert 被所有 domain 共享 → 它是通用知识 Expert"
      - "某个 Expert 只被特定 domain 使用 → 它是场景专属 Expert"

提示：
  ① 和 MMoE 的 TODO ⑥ 类似，但这里每个 domain 只有一个 Gate
  ② 用 model.gates[d](flat) 可以获取 domain d 的 Gate 权重
  ③ 按 domain_id mask 分组，计算平均权重
  ④ 打印成表格或热力图
"""
# ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
model = HMoE(
    feature_vocabs, emb_dim=EMB_DIM, num_experts=NUM_EXPERTS,
    expert_units=EXPERT_UNITS, gate_units=GATE_UNITS,
    tower_units=TOWER_UNITS, num_domains=NUM_DOMAINS, dropout=0.1
)
model.load_state_dict(torch.load("best_hmoe.pt", map_location="cpu"))
model.eval()

gate_weights_by_domain = {d: [] for d in range(NUM_DOMAINS)}
with torch.no_grad():
    for features, labels in val_loader:
        # 构造 flat embedding（和 HMoE.forward 里的逻辑完全一致）
        embs = []
        for feat_name in model.feature_names:
            embs.append(model.embeddings[feat_name](features[feat_name]))
        static_emb = torch.stack(embs, dim=1)           # [B, N, D]
        flat = static_emb.view(static_emb.size(0), -1)  # [B, N*D]

        domain_ids = features["domain_id"].numpy()

        for d in range(NUM_DOMAINS):
            mask = domain_ids == d
            if mask.sum() == 0:
                continue
            flat_d = flat[mask]                  # [n_d, N*D]
            gate_w = model.gates[d](flat_d)      # [n_d, num_experts]
            gate_weights_by_domain[d].append(gate_w.numpy())

print("\n各 Domain 的平均 Gate 权重分布：")
print("-" * 60)

avg_weights = {}
for d in range(NUM_DOMAINS):
    if len(gate_weights_by_domain[d]) == 0:
        avg_weights[d] = np.zeros(NUM_EXPERTS)
        continue
    all_w = np.concatenate(gate_weights_by_domain[d], axis=0)  # [n, num_experts]
    avg_w = all_w.mean(axis=0)
    avg_weights[d] = avg_w

    # 打印每个 Expert 的平均权重
    expert_str = "  |  ".join([f"E{i}: {avg_w[i]:.3f}" for i in range(NUM_EXPERTS)])
    print(f"Domain {d:2d}: {expert_str}")
    top_expert = avg_w.argmax()
    print(f"  → 最偏好 Expert {top_expert} (权重 {avg_w[top_expert]:.3f})")

print("\n" + "-" * 60)
print("各 Domain Gate 权重的集中度（Entropy，越低越集中）：")
for d in range(NUM_DOMAINS):
    w = avg_weights[d] + 1e-9
    entropy = -(w * np.log(w)).sum()
    print(f"  Domain {d}: Entropy = {entropy:.3f}")

print("\n" + "-" * 60)
print("Domain 间 Gate 权重相似度（余弦相似度矩阵）：")

sim_matrix = np.zeros((NUM_DOMAINS, NUM_DOMAINS))
for i in range(NUM_DOMAINS):
    for j in range(NUM_DOMAINS):
        wi, wj = avg_weights[i], avg_weights[j]
        sim = np.dot(wi, wj) / (np.linalg.norm(wi) * np.linalg.norm(wj) + 1e-9)
        sim_matrix[i][j] = sim

# 打印矩阵
header = "     " + "  ".join([f"D{j:2d}" for j in range(NUM_DOMAINS)])
print(header)
for i in range(NUM_DOMAINS):
    row = f"D{i:2d}  " + "  ".join([f"{sim_matrix[i][j]:.2f}" for j in range(NUM_DOMAINS)])
    print(row)

# 找出最相似/最不相似的 domain 对
max_sim, min_sim = -1, 2
max_pair, min_pair = None, None
for i in range(NUM_DOMAINS):
    for j in range(i + 1, NUM_DOMAINS):
        if sim_matrix[i][j] > max_sim:
            max_sim, max_pair = sim_matrix[i][j], (i, j)
        if sim_matrix[i][j] < min_sim:
            min_sim, min_pair = sim_matrix[i][j], (i, j)

print(f"\n  → 最相似的 Domain 对: D{max_pair[0]} & D{max_pair[1]} (相似度={max_sim:.3f})")
print(f"  → 差异最大的 Domain 对: D{min_pair[0]} & D{min_pair[1]} (相似度={min_sim:.3f})")

print("\n" + "-" * 60)
print("Expert 角色分析：")
for e in range(NUM_EXPERTS):
    avg_across_domains = np.mean([avg_weights[d][e] for d in range(NUM_DOMAINS)])
    max_domain = max(range(NUM_DOMAINS), key=lambda d: avg_weights[d][e])
    max_weight = avg_weights[max_domain][e]

    if avg_across_domains > 0.5 / NUM_EXPERTS and max_weight < 0.6:
        role = "通用 Expert（被多个 domain 共享）"
    elif max_weight > 0.5:
        role = f"Domain {max_domain} 的专属 Expert"
    else:
        role = "边缘 Expert（权重较低）"

    print(f"  Expert {e}: 平均权重={avg_across_domains:.3f}, "
          f"在 Domain {max_domain} 最高={max_weight:.3f} → {role}")

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.imshow(sim_matrix, cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(range(NUM_DOMAINS), [f'D{i}' for i in range(NUM_DOMAINS)])
    plt.yticks(range(NUM_DOMAINS), [f'D{i}' for i in range(NUM_DOMAINS)])
    plt.title('Domain Gate 权重余弦相似度')
    plt.tight_layout()
    plt.savefig('hmoe_domain_gate_similarity.png', dpi=150)
    print("\n热力图已保存: hmoe_domain_gate_similarity.png")
except ImportError:
    pass





# ↑↑↑↑ 你的代码 ↑↑↑↑


# ============================================================
# 【7】TODO ⑦：对比实验 —— Shared-Bottom vs HMoE
# ============================================================
print("\n" + "=" * 60)
print("【7】TODO ⑦：Shared-Bottom vs HMoE 对比")
print("=" * 60)
"""
目标：实现一个 Shared-Bottom 多域模型，和 HMoE 对比

Shared-Bottom 多域模型结构：
  input → Embedding → Shared DNN → [Domain_0_Tower, Domain_1_Tower, ...]
  所有场景共享同一个底层 DNN，只有 Tower 是独立的

和 HMoE 的区别：
  Shared-Bottom：只有一个共享 DNN，没有 Expert + Gate 的机制
  HMoE：多个 Expert + 每个 domain 独立 Gate 选择 Expert

TODO：
  ① 实现 SharedBottomMultiDomain 模型
  ② 训练并计算各 domain 的 AUC
  ③ 对比 HMoE 和 Shared-Bottom 在每个 domain 上的 AUC
  ④ 分析：
      - HMoE 在哪个 domain 上提升最大？为什么？
      - Shared-Bottom 在数据量大的 domain 上是否效果更好？
      - HMoE 的参数量比 Shared-Bottom 大多少？值不值？
"""
# ↓↓↓↓↓ 你的代码 ↓↓↓↓↓
class SharedBottomMultiDomain(nn.Module):
    def __init__(self,features_vocabs,emb_dim,shared_units,tower_units,num_domains,dropout=0.1):
        super().__init__()
        self.feature_names = list(features_vocabs.keys())
        self.num_features = len(self.feature_names)
        self.num_domains = num_domains

        # embedding 
        self.embeddings = nn.ModuleDict()
        for feat_name in self.feature_names:
            self.embeddings[feat_name] = nn.Embedding(features_vocabs[feat_name],emb_dim,padding_idx=0)
        # 线性部分
        self.linear_weights = nn.ModuleDict()
        for feat_name in self.feature_names:
            self.linear_weights[feat_name] = nn.Embedding(features_vocabs[feat_name],1,padding_idx=0)
        self.global_bias = nn.Parameter(torch.zeros(1))
        # shared bottom
        input_dim = emb_dim * self.num_features
        layers = []
        for u in shared_units:
            layers.extend([
                nn.Linear(input_dim,u),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = u
        self.shared_bottom = nn.Sequential(*layers)

        # towers
        self.towers = nn.ModuleList([
            DomainTower(input_dim,tower_units,dropout)
            for _ in range(num_domains)
        ])
    def forward(self,features):
        embs = []
        for feat_name in self.feature_names:
            embs.append(self.embeddings[feat_name](features[feat_name]))
        static_emb = torch.stack(embs,dim=1) # [B,N,D]
        flat = static_emb.view(static_emb.size(0),-1) # [B,N*D]

        linear_logit = self.global_bias.expand(static_emb.size(0))
        for feat_name in self.feature_names:
            lin = self.linear_weights[feat_name](features[feat_name])
            linear_logit = linear_logit + lin.squeeze(1)

        shared_out = self.shared_bottom(flat)
        domain_logits = []
        for d in range(self.num_domains):
            tower_out = self.towers[d](shared_out)
            domain_logits.append(tower_out)

        domain_ids = features["domain_id"]
        logits = torch.stack(domain_logits,dim=1) # [B,num_domains]
        logits = logits[torch.arange(len(domain_ids)),domain_ids]

        return logits + linear_logit

print("\n" + "-" * 60)
print("【训练 Shared-Bottom 模型】")

sb_model = SharedBottomMultiDomain(
    feature_vocabs, emb_dim=EMB_DIM, shared_units=EXPERT_UNITS,
    tower_units=TOWER_UNITS, num_domains=NUM_DOMAINS, dropout=0.1,
)
print(f"  Shared-Bottom 参数量: {sum(p.numel() for p in sb_model.parameters()):,}")
train(sb_model, train_loader, val_loader, save_path="best_sb.pt")

print("\n" + "-" * 60)
print("【加载 HMoE 最佳模型】")

hmoe_model = HMoE(
    feature_vocabs, emb_dim=EMB_DIM, num_experts=NUM_EXPERTS,
    expert_units=EXPERT_UNITS, gate_units=GATE_UNITS,
    tower_units=TOWER_UNITS, num_domains=NUM_DOMAINS, dropout=0.1,
)
hmoe_model.load_state_dict(torch.load("best_hmoe.pt", map_location="cpu"))

# 用同一个 evaluate 函数评估
hmoe_overall_auc, hmoe_domain_aucs = evaluate(hmoe_model, val_loader)
sb_overall_auc, sb_domain_aucs = evaluate(sb_model, val_loader)

print("\n" + "=" * 60)
print("【对比结果：Shared-Bottom vs HMoE】")
print("=" * 60)

# 表头
print(f"{'Domain':<10} {'SB AUC':>10} {'HMoE AUC':>10} {'Δ':>10} {'提升':>10}")
print("-" * 60)

# 各 domain 对比
for d in range(NUM_DOMAINS):
    sb_auc = sb_domain_aucs.get(d)
    hmoe_auc = hmoe_domain_aucs.get(d)
    
    if sb_auc is None or hmoe_auc is None:
        continue
    
    delta = hmoe_auc - sb_auc
    pct = delta / (sb_auc + 1e-9) * 100
    marker = "↑" if delta > 0 else "↓" if delta < 0 else "="
    print(f"Domain {d:<4} {sb_auc:>10.4f} {hmoe_auc:>10.4f} {delta:>+9.4f} {marker} {pct:>+7.2f}%")

# 整体对比
print("-" * 60)
delta_overall = hmoe_overall_auc - sb_overall_auc
pct_overall = delta_overall / (sb_overall_auc + 1e-9) * 100
marker = "↑" if delta_overall > 0 else "↓"
print(f"{'Overall':<10} {sb_overall_auc:>10.4f} {hmoe_overall_auc:>10.4f} "
      f"{delta_overall:>+9.4f} {marker} {pct_overall:>+7.2f}%")

print("\n" + "-" * 60)
print("【参数量对比】")

sb_params = sum(p.numel() for p in sb_model.parameters())
hmoe_params = sum(p.numel() for p in hmoe_model.parameters())

print(f"Shared-Bottom 参数: {sb_params:,}")
print(f"HMoE 参数:          {hmoe_params:,}")
print(f"参数增长:           {(hmoe_params - sb_params) / sb_params * 100:.1f}%")

print("\n参数量增长来源：")
print(f"  - Expert 网络:     {NUM_EXPERTS} 个 Expert × {EXPERT_UNITS} 每层")
print(f"  - Gate 网络:       {NUM_DOMAINS} 个 domain × 各 {GATE_UNITS} 每层")
print(f"  - 对比: Shared-Bottom 只有 1 个 DNN，HMoE 有 {NUM_EXPERTS} 个 Expert + {NUM_DOMAINS} 个 Gate")

print("\n" + "=" * 60)
print("【分析结论】")
print("=" * 60)

# 找出 HMoE 提升最大的 domain
best_improvement = -float('inf')
best_domain = -1
for d in range(NUM_DOMAINS):
    sb_auc = sb_domain_aucs.get(d)
    hmoe_auc = hmoe_domain_aucs.get(d)
    if sb_auc is None or hmoe_auc is None:
        continue
    delta = hmoe_auc - sb_auc
    if delta > best_improvement:
        best_improvement = delta
        best_domain = d

print(f"1. HMoE 提升最大的 Domain: {best_domain} (ΔAUC = {best_improvement:+.4f})")
print(f"   原因推测: 该 domain 数据分布与其他 domain 差异较大，")
print(f"   HMoE 的 domain-specific Gate 能更好地为其选择合适的 Expert。")

print(f"\n2. 参数量增长: {(hmoe_params - sb_params) / sb_params * 100:.1f}%")
if hmoe_overall_auc > sb_overall_auc:
    print(f"   但 overall AUC 从 {sb_overall_auc:.4f} → {hmoe_overall_auc:.4f}")
    print(f"   结论: 参数量增长带来的收益 {'值得' if delta_overall > 0.005 else '一般'}")
else:
    print(f"   overall AUC 反而下降，说明在这个数据集上 HMoE 的优势不明显。")

# ↑↑↑↑ 你的代码 ↑↑↑↑

print("\n训练全部完成！")
