# 04. Item2Vec —— 把推荐当 NLP 来做

> 对应教材：fun-rec 第 2.2.2 节
> 论文：Barkan & Koenigstein (2016)
> 根源：Word2Vec (Mikolov, 2013)

## 🎯 范式跃迁

| 之前（FunkSVD/BiasSVD） | 现在（Item2Vec） |
|---|---|
| 输入：(user, item) 二元组 | 输入：用户行为**序列** |
| 目标：预测评分 | 目标：让共现物品向量相似 |
| loss：MSE | loss：**负采样 + 二分类** |
| 输出：评分 | **输出：item embedding** |

## 核心公式

**SkipGram 目标**（带负采样）：
$$ L = -\log \sigma(v_c^T v_o) - \sum_{k=1}^K \log \sigma(-v_c^T v_{neg_k}) $$

含义：
- 让"中心物品 c 和真实上下文 o"的内积尽量大
- 让"中心物品 c 和 K 个负采样物品"的内积尽量小

## 关键技术

| 技术 | 解决什么 |
|---|---|
| **滑动窗口** | 把序列切成 (中心, 上下文) 对 |
| **负采样** | softmax 太慢，转成二分类 |
| **频率^0.75 采样** | 平衡热门/冷门负样本 |
| **共享或分离 embedding** | center 用 in_embedding，context 用 out_embedding |

## 文件说明

- `item2vec_pytorch.py`：从零实现 SkipGram + 负采样
