"""
MovieLens-latest-small 数据集探索性分析（EDA）

【目的】
工业界拿到任何新数据集，第一件事不是写模型，而是 EDA（Exploratory Data Analysis）：
    - 数据有多大？
    - 有没有缺失？
    - 分布长什么样？
    - 是否有异常？
    - 是否符合"长尾分布"（推荐系统的典型现象）？

跑这个脚本，你就能从"调包党"升级成"懂数据的算法工程师"。

运行：
    cd /Users/qiruihou/Desktop/学习/推荐算法/my-rec-journey/recall/01_itemcf
    python data_explore.py
"""

import pandas as pd
import numpy as np

# ============================================================
# 【1】读数据
# ============================================================
DATA_DIR = "/Users/qiruihou/Desktop/学习/推荐算法/dataset/ml-latest-small"

ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
movies = pd.read_csv(f"{DATA_DIR}/movies.csv")

print("=" * 70)
print("【1】数据规模")
print("=" * 70)
print(f"ratings.csv: {ratings.shape}  ← (行数, 列数)")
print(f"movies.csv : {movies.shape}")
print(f"\nratings 字段：{ratings.columns.tolist()}")
print(f"\nratings 前 5 行：")
print(ratings.head())


# ============================================================
# 【2】基本统计
# ============================================================
print("\n" + "=" * 70)
print("【2】基本统计")
print("=" * 70)
print(f"  用户数      : {ratings['userId'].nunique():,}")
print(f"  电影数(被评)  : {ratings['movieId'].nunique():,}")
print(f"  电影数(总)   : {movies['movieId'].nunique():,}")
print(f"  评分总条数   : {len(ratings):,}")
print(f"  人均评分数   : {len(ratings) / ratings['userId'].nunique():.1f}")
print(f"  人均看过比例 : {(len(ratings) / ratings['userId'].nunique()) / movies['movieId'].nunique() * 100:.2f}%")


# ============================================================
# 【3】稀疏度（推荐系统的关键概念！）
# ============================================================
n_users = ratings['userId'].nunique()
n_movies = movies['movieId'].nunique()
total_possible = n_users * n_movies
actual = len(ratings)
sparsity = 1 - actual / total_possible

print("\n" + "=" * 70)
print("【3】稀疏度（推荐系统的核心痛点！）")
print("=" * 70)
print(f"  理论上的总评分数 = {n_users} 用户 × {n_movies} 电影 = {total_possible:,}")
print(f"  实际评分数       = {actual:,}")
print(f"  稀疏度           = {sparsity * 100:.2f}%")
print(f"  → 意思：用户-物品矩阵里 {sparsity*100:.1f}% 的格子是空的")
print(f"  → 这就是推荐系统的根本难题：从极稀疏数据里学规律")


# ============================================================
# 【4】评分分布
# ============================================================
print("\n" + "=" * 70)
print("【4】评分值分布")
print("=" * 70)
print(ratings['rating'].value_counts().sort_index())
print(f"\n  平均分：{ratings['rating'].mean():.2f}")
print(f"  中位数：{ratings['rating'].median()}")
print(f"  → 观察：用户更倾向于打高分（评分偏置）")
print(f"  → 这种偏置如果不处理会让相似度计算失真——这就是为什么需要中心化")


# ============================================================
# 【5】每个用户评了多少部？（长尾分布观察）
# ============================================================
user_counts = ratings.groupby('userId').size()

print("\n" + "=" * 70)
print("【5】每个用户的评分数分布（长尾！）")
print("=" * 70)
print(f"  最少：{user_counts.min()} 部")
print(f"  最多：{user_counts.max()} 部")
print(f"  中位数：{user_counts.median():.0f}")
print(f"  平均：{user_counts.mean():.1f}")
print(f"\n  分位数：")
for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"    {int(q*100):>3}% 用户评分数 ≤ {user_counts.quantile(q):.0f}")


# ============================================================
# 【6】每部电影被多少人评了？（也是长尾！）
# ============================================================
movie_counts = ratings.groupby('movieId').size()

print("\n" + "=" * 70)
print("【6】每部电影被评分的次数（典型长尾分布）")
print("=" * 70)
print(f"  最少：{movie_counts.min()} 次（可能是冷门小众）")
print(f"  最多：{movie_counts.max()} 次（爆款）")
print(f"  中位数：{movie_counts.median():.0f}")
print(f"\n  → 80/20 法则验证：")
top_20_pct_movies = int(len(movie_counts) * 0.2)
top_20_count_sum = movie_counts.sort_values(ascending=False).head(top_20_pct_movies).sum()
print(f"    前 20% 的热门电影（{top_20_pct_movies} 部）贡献了 "
      f"{top_20_count_sum / len(ratings) * 100:.1f}% 的评分")
print(f"    → 推荐系统要解决的核心问题之一：把流量分给长尾物品")

print(f"\n  评分次数最多的 10 部电影：")
top10 = movie_counts.sort_values(ascending=False).head(10)
top10_with_title = top10.to_frame('count').merge(movies, on='movieId')
print(top10_with_title[['title', 'count']].to_string(index=False))


# ============================================================
# 【7】时间跨度
# ============================================================
print("\n" + "=" * 70)
print("【7】时间跨度")
print("=" * 70)
ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
print(f"  最早评分：{ratings['date'].min()}")
print(f"  最晚评分：{ratings['date'].max()}")
print(f"  跨度    ：{(ratings['date'].max() - ratings['date'].min()).days} 天")


# ============================================================
# 【8】关键洞察总结
# ============================================================
print("\n" + "=" * 70)
print("【8】算法工程师的关键洞察")
print("=" * 70)
print("""
1. 数据极稀疏（>98%）→ 协同过滤要面对的核心挑战
2. 评分偏置严重（高分多）→ 必须做归一化（皮尔逊 vs 余弦）
3. 长尾分布严重 → 热门物品天然占优，冷启动是难题
4. 时间跨度长（22年）→ 真实场景需要考虑兴趣漂移（DIN 等模型解决）

下一步建议：
- 用这些观察反推：ItemCF 在这份数据上**应该**表现如何？
- 比如：稀疏度 98% → ItemCF 计算相似度时大量物品对没有共同用户
- 这正是 fun-rec 工程版用"倒排表"优化的根本原因
""")
