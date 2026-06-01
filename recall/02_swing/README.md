# 02. Swing：用图结构修正 ItemCF 的相似度

> 对应教材：fun-rec 第 2.1.2 节
> 论文：阿里 2020，*Large scale product graph construction for recommendation in e-commerce*

## 核心公式

**带用户权重的 Swing 分数**：

$$ s(i, j) = \sum_{u, v \in U_i \cap U_j} w_u \cdot w_v \cdot \frac{1}{\alpha + |I_u \cap I_v|} $$

其中：
- $U_i, U_j$：买过物品 i / j 的用户集合
- $I_u, I_v$：用户 u / v 买过的物品集合
- $w_u = 1/\sqrt{|I_u|}$：用户权重，活跃用户被降权
- $\alpha$：平滑系数（通常取 1）

## 与 ItemCF 的对比

| 维度 | ItemCF | Swing |
|---|---|---|
| 抗噪声 | ❌ | ✅ 用共同物品数自动降权 |
| 抑制活跃用户 | ❌ | ✅ 用 $w_u$ 降权 |
| 适用场景 | 教学、小规模 | 工业（电商 i2i 主流方案） |

## 📊 实验结果与反思（2026-06-01）

在 MovieLens-latest-small 上对比：

|         | hit_rate@10 | hit_rate@5 | precision@10 | precision@5 |
|---------|-------------|------------|--------------|-------------|
| ItemCF  | 0.6594      | 0.5459     | 0.1444       | 0.1826      |
| Swing   | 0.6194      | 0.5042     | 0.1282       | 0.1629      |

**反直觉的发现**：Swing 全部指标都不如 ItemCF。

**原因分析**：
1. Swing 的设计目标是抗噪声、抑制热门偏置。MovieLens 是高质量评分数据，
   "无病可治"，Swing 的复杂机制反成累赘。
2. 数据稀疏（98%+），Swing 的"双重交集"条件（u, v 都买过 i 和 j）
   导致大量物品对直接为 0，召回的候选池更小。
3. Swing 把 5 分评分简化成 0/1，丢弃了评分信息。

**工程认知**：算法选型必须看数据特征，论文 SOTA ≠ 你的场景最优。
Swing 在阿里电商海量噪声数据上比 ItemCF 提升 24%，但在干净小数据上反而吃亏。

## 文件说明

- `swing_numpy.py`：核心函数实现 + 玩具数据验证
