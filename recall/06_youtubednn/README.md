# YouTubeDNN 召回模型学习笔记

## 📜 一句话定位

**YouTubeDNN 是 2016 年 YouTube 提出的深度学习召回奠基之作，至今仍是工业界的主流召回方案之一。**

论文：[Deep Neural Networks for YouTube Recommendations (RecSys 2016)](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)

---

## 🆚 和 DSSM 的核心区别

| 维度 | DSSM | YouTubeDNN |
|---|---|---|
| 架构 | 双塔（用户塔 + 物品塔） | **单塔**（只有用户塔） |
| 物品表征 | 物品塔 DNN 输出 | **直接来自 Embedding 表** |
| 历史行为 | ❌ 无 | ✅ **Average Pooling** |
| 建模视角 | 二分类 / batch 内多分类 | **全库 N 分类** |
| 负样本 | In-batch Negatives | 全库 Softmax / Sampled Softmax |

---

## 🎯 三大核心创新（必须吃透）

### ① 把召回建模成"超大规模多分类"
$$P(w_t = i \mid U, C) = \frac{e^{v_i \cdot u}}{\sum_{j \in V} e^{v_j \cdot u}}$$

把"用户下一秒看哪个视频"建模成对全库 V 个视频的多分类。

### ② 历史行为的 Average Pooling
$$h = \frac{1}{N}\sum_{i=1}^{N} e_i$$

> "Surprisingly, simple averaging worked best." —— 论文原话

### ③ Embedding 共享
- 历史观看视频的 Embedding 表
- 最终 Softmax 分类层的 Embedding 表
- **就是同一张表！**（让历史和未来在同一语义空间）

---

## 📁 文件说明

- `youtubednn_pytorch.py`：主模型（填空式 TODO，跟着提示填）
- `1.py`：关键代码片段实验场（验证每一步理解）
- `README.md`：本笔记

---

## 🚀 跑通步骤

```bash
# 1. 先跑实验场，理解 Masked Pooling / Embedding 共享 / Softmax Loss
python 1.py

# 2. 再填 youtubednn_pytorch.py 里的 TODO 部分

# 3. 训练 + 召回
python youtubednn_pytorch.py
```

---

## 🔑 你需要填的 TODO 一览

| 文件 | 位置 | 内容 |
|---|---|---|
| `youtubednn_pytorch.py` | UserTower.__init__ | 构造 DNN |
| `youtubednn_pytorch.py` | UserTower.forward | 用户塔前向（含 Masked Pooling） |
| `youtubednn_pytorch.py` | YoutubeDNN.forward | 全库 Softmax Loss |

---

## 🎨 与 DSSM 互动对比训练

跑完 YouTubeDNN 后建议做的实验：
1. 对比相同 3 个用户，DSSM vs YouTubeDNN 的 Top-10 推荐
2. YouTubeDNN 由于引入了**历史行为**，推荐应该更"理解用户"
3. DSSM 推荐更倾向于"全局热门相似"，YouTubeDNN 更倾向于"个性化"

---

## 💡 工程化扩展思路

1. **Sampled Softmax**：当物品数 > 10 万时，用采样近似（代码末尾有参考实现）
2. **Hard Negative Mining**：增加难负样本（与正样本相似但用户不喜欢的）
3. **更多用户特征**：地域、年龄、活跃度等
4. **更多上下文特征**：当前时间、设备、network 等
5. **YouTube 论文的 "Example Age" 特征**：解决新视频冷启动的关键技巧
