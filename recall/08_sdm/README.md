# SDM 召回模型学习笔记

## 一句话定位

**SDM（Session-based Deep Matching）在序列召回里显式区分「短期即时兴趣」和「长期稳定偏好」，再用门控机制动态融合。**

论文：[SDM: Session-based Deep Matching Model for Search and Recommendation (KDD 2019)](https://dl.acm.org/doi/10.1145/3292500.3330657)

FunRec 章节：[2.4.2 SDM：融合长短期兴趣，捕捉动态变化](https://funrec.github.io/chapter_1_retrieval/4.sequence/2.sdm.html)

---

## 你在召回路上的位置

```
01 ItemCF → 02 Swing → 03 MF → 04 Item2Vec
    → 05 DSSM → 06 YouTubeDNN → 07 MIND → 【08 SDM】→ 09 Trinity ...
         双塔          单塔+序列        多兴趣        长短期+门控
```

---

## MIND 之后，为什么还需要 SDM？

| 问题 | MIND | SDM |
|---|---|---|
| 兴趣是多元的吗？ | ✅ K 个胶囊 | ✅ 短期侧用 Multi-Head Attention |
| 区分「刚发生的」和「很久以前的」？ | ❌ 一条历史序列一视同仁 | ✅ **short seq + long seq** |
| 融合策略 | 训练时 Label-Aware Attn 塌缩 | **Gated Fusion** 按维度选长/短 |

直觉例子：你最近 5 次点击都是「跑鞋、运动袜、护膝」→ **短期**强烈指向运动；但你过去半年也大量看「科幻电影」→ **长期**偏好仍在。SDM 让模型自己学：此刻该听短期还是长期。

---

## 架构三大模块

```
                    ┌─────────────────────────────────────┐
                    │         用户画像 e_u               │
                    │    (user_id → Embedding → Dense)   │
                    └──────────┬──────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │  短期序列     │      │  长期序列     │      │  (同上 e_u)   │
  │  最近 5 个    │      │  最近 50 个   │      │              │
  └──────┬───────┘      └──────┬───────┘      └──────────────┘
         │ LSTM                  │ UserAttention × 多特征
         │ MultiHeadAttn         │ (movie_id / genres)
         │ UserAttention         │
         ▼                     ▼
      s_t (短期)            p_u (长期)
         └──────────┬──────────┘
                    ▼
              Gated Fusion
         G = σ(W1·e_u + W2·s_t + W3·p_u + b)
         o = (1-G)⊙p_u + G⊙s_t
                    │
                    ▼
            全库 Softmax 召回
```

---

## 核心公式（必背 3 个）

### 1. 个性化注意力（UserAttention）

\[
\alpha_k = \frac{\exp(\hat{h}_k^T e_u)}{\sum_j \exp(\hat{h}_j^T e_u)}, \quad s_t = \sum_k \alpha_k \hat{h}_k
\]

用户画像 \(e_u\) 当 **Query**，序列各位置当 **Key/Value**——「这个用户更该关注序列里哪几步？」

### 2. 门控融合（Gated Fusion）

\[
G = \sigma(W_1 e_u + W_2 s_t + W_3 p_u + b), \quad o = (1-G)\odot p_u + G\odot s_t
\]

每个维度独立决定：更信长期还是短期（\(G\) 接近 1 → 短期主导）。

### 3. 训练目标（同 YouTubeDNN）

Sampled Softmax / 全库 Softmax，物品向量来自 **共享 Embedding 表**。

---

## 与前几章对比

| 维度 | YouTubeDNN | MIND | SDM |
|---|---|---|---|
| 历史怎么用 | Average Pooling | Capsule 路由 → K 兴趣 | **LSTM + MHA + Attn** |
| 时间结构 | 无 | 无 | **short / long 双序列** |
| 用户向量个数 | 1 | K（推理） | 1 |
| 融合 | — | Label-Aware Attn | **Gated Fusion** |

---

## 文件说明

- `sdm_pytorch.py`：主模型（填空式 TODO）
- `README.md`：本笔记

---

## 学习顺序（建议 4 步）

| 步骤 | 内容 | TODO |
|---|---|---|
| **Step 1** | 理解 UserAttention（和 DIN 的 Attention 同源） | TODO ① |
| **Step 2** | 实现 GatedFusion | TODO ② |
| **Step 3** | 短期模块 LSTM → MHA → UserAttention | TODO ③ |
| **Step 4** | 长期模块 + SDM 主模型 + 训练 | TODO ④⑤ |

```bash
cd my-rec-journey/recall/08_sdm
python sdm_pytorch.py
```

脚本会自动：加载数据 → 训练（early stopping）→ 保存 `best_sdm.pt` → 打印 3 个用户的 Top-10 召回 + gate 均值。

TensorBoard：

```bash
tensorboard --logdir runs/
```

---

## FunRec 官方配置要点（ml-1m）

- `hist_movie_id_short`：最近 **5** 个（短期 session）
- `hist_movie_id_long`：最近 **50** 个（长期）
- 长期侧还有 `hist_genres_long` 等**多特征维度**，各做一遍 UserAttention 再 concat

我们用小数据集 `ml-latest-small` 简化：短期 5、长期 20，长期两路特征 = **movie_id 序列 + genre 序列**。
