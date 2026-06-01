# 03. Matrix Factorization (FunkSVD / BiasSVD)

> 对应教材：fun-rec 第 2.1.4 节
> 论文：Funk (2006) / Koren et al. (2009)

## 🎯 这一节的里程碑意义

| 之前（ItemCF / Swing） | 现在（MF） |
|---|---|
| 统计共现 | **学习参数** |
| 没有训练过程 | 用梯度下降训练 |
| 相似度矩阵 | **Embedding 表** |
| 不需要 PyTorch | **第一次真正用上 PyTorch** |
| 离散数学 | 进入"深度学习" |

## 核心公式

**FunkSVD 预测**：
$$ \hat{r}_{ui} = p_u^T q_i $$

**BiasSVD 预测**（加偏置项）：
$$ \hat{r}_{ui} = \mu + b_u + b_i + p_u^T q_i $$

**损失函数**：
$$ L = \text{MSE}(r, \hat{r}) + \lambda(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2) $$

PyTorch 实现中，L2 正则由 `optim.Adam(..., weight_decay=λ)` 自动处理，等价。

## 📊 实验结果与反思

### FunkSVD 表现（baseline）
- N_EPOCHS=20, K=32, lr=0.01, weight_decay=1e-4
- train_loss → 0.5（过拟合）
- **test_loss → 13-14**（严重过拟合，差距巨大）

### BiasSVD 表现（升级版）
- 同样超参，早停在 epoch 87 触发
- train_loss → 0.19
- **test_loss = 1.02**
- RMSE ≈ 1.01（接近业界 baseline 0.87-0.95）

### 学到的偏置（部分用户）
- 全局偏置 μ = 3.36（接近训练集均值 3.51）
- 用户 0: b_u = +0.81（佛系好评党）
- 用户 2: b_u = -0.64（毒舌评论家）
- 用户 3: b_u = -0.02（极度中庸）

### 🔥 关键认知

**BiasSVD 把 test_loss 从 13 降到 1.02，下降 13 倍！**

为什么效果这么夸张？
1. FunkSVD 浪费向量容量去"硬学系统偏置"
2. BiasSVD 把"系统水位""用户习惯""电影质量"显式建模，让 $p_u \cdot q_i$ 专注学习真正的偏好匹配
3. **简单的 3 个偏置项，效果碾压扩大 K 的"暴力"做法**

### 工程认知
- **模型架构的精细化改动 > 盲目扩大参数**
- 训练 loss 低 ≠ 模型好（要看 test loss，过拟合是头号敌人）
- 早停（Early Stopping）是必备技术，配合大 N_EPOCHS 用
- PyTorch 的 `weight_decay` 等价于书里的 L2 正则项

## 文件说明

- `funksvd_pytorch.py`：FunkSVD 基础版（过拟合严重）
- `biassvd_pytorch.py`：BiasSVD 升级版（test_loss 显著下降）
- `best_funksvd.pt` / `best_biassvd.pt`：保存的最佳模型权重（gitignore 忽略）

## 学习产出

- [x] 理解 nn.Embedding 的本质（查找表 / 可学习参数）
- [x] 掌握 PyTorch 训练循环 5 步法
- [x] 学会用 TensorBoard 监控训练
- [x] 实现并对比 FunkSVD vs BiasSVD
- [x] 亲眼看到"过拟合"现象
- [x] 用 Early Stopping 治过拟合
- [x] 理解 weight_decay = L2 正则
