# 01. ItemCF：基于物品的协同过滤

> 对应教材：fun-rec 第 2.1.1 节
> 学习目标：用 Python 复现书里"用户 1 对物品 5 的评分预测"

## 核心公式

**物品相似度（皮尔逊相关系数）**：

$$ w_{ij} = \frac{\sum_{u}(r_{ui}-\bar{r}_i)(r_{uj}-\bar{r}_j)}{\sqrt{\sum_u(r_{ui}-\bar{r}_i)^2}\sqrt{\sum_u(r_{uj}-\bar{r}_j)^2}} $$

**评分预测**：

$$ \hat{r}_{u,j} = \bar{r}_j + \frac{\sum_{k \in S_j} w_{jk}(r_{u,k}-\bar{r}_k)}{\sum_{k \in S_j} w_{jk}} $$

## 文件说明

- `itemcf_numpy.py`：用 numpy + pandas 复现书里的例子（基础版，**先写这个**）
- `itemcf_pytorch.py`：用 PyTorch 实现（进阶版，下一节再写）

## 学习产出

- [ ] 跑通 `itemcf_numpy.py`，得到与教材一致的预测分数 ~4.6
- [ ] 自己能口头讲清楚"为什么要做中心化"
- [ ] 用 git 提交本节代码
