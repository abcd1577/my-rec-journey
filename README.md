# my-rec-journey

我的推荐算法 + PyTorch 学习记录。跟着 fun-rec 官方教程，先理论后用 PyTorch 手写实现。

## 目录结构

```
my-rec-journey/
├── pytorch_basics/        # PyTorch 基础练习
│   ├── 01_tensor.py       # Tensor 基本操作
│   ├── 02_autograd.py     # 自动微分 + 手写梯度下降
│   └── 03_nn_module.py    # 用 nn.Module 重构线性回归（工程化写法）
│
├── recall/                # 📌 召回模型
│   ├── 01_item2vec/       # Item2Vec：Word2Vec 思想应用到物品
│   ├── 02_mf/             # Matrix Factorization：矩阵分解
│   ├── 03_dssm/           # DSSM：双塔模型
│   ├── 04_youtubednn/     # YouTubeDNN：序列建模召回
│   ├── 05_youtubednn_sample/ # YouTubeDNN（负采样版）
│   ├── 06_mind/           # MIND：多兴趣召回
│   ├── 07_sdm/            # SDM：长短期兴趣融合
│   └── 08_gsd/            # GSD：图序列深度匹配
│
├── rank/                  # 📌 精排模型
│   ├── 01_widedeep/       # Wide & Deep：记忆 + 泛化
│   ├── 02_afm/            # AFM：注意力因子分解机
│   ├── 03_deepfm/         # DeepFM：FM + DNN 共享 Embedding
│   └── 04_din/            # DIN：深度兴趣网络（序列建模）
│
└── README.md
```

## 学习路径

### 第一阶段：召回（Retrieval）

从海量物品中快速筛选出几十个候选。

| 模型 | 核心思想 | 状态 |
|------|---------|------|
| Item2Vec | 把物品序列当句子，Word2Vec 学 embedding | ✅ |
| MF | 用户 × 物品 = 隐向量内积 | ✅ |
| DSSM | 双塔结构，user 和 item 各自编码 | ✅ |
| YouTubeDNN | 平均 pooling 序列 + Sampled Softmax | ✅ |
| MIND | 动态路由 → K 个兴趣胶囊 → K 路召回 | ✅ |
| SDM | 长短期双塔 + 门控融合 | ✅ |

### 第二阶段：精排（Ranking）

对几十个候选逐一算 CTR，排序输出。

| 模型 | 核心思想 | 状态 |
|------|---------|------|
| Wide & Deep | Wide 记忆 + Deep 泛化 | ✅ |
| AFM | FM 二阶交叉 + 注意力加权 | ✅ |
| DeepFM | FM 二阶 + DNN 高阶（共享 Embedding） | ✅ |
| DIN | 候选商品当 query，动态加权历史序列 | ✅（写作中） |

## 数据集

- **KuaiRand-1K**（精排）：快手推荐数据集，1000 用户，437 万视频，12 种反馈信号
- **MovieLens 1M / latest-small**（召回）：电影评分数据

## 学习日志

- 2026-05-31：搞定 git 全流程，能 push 到工蜂
- 2026-06-07：进入精排阶段，完成 Wide&Deep → AFM → DeepFM
- 2026-06-08：开始 DIN（序列建模精排）
