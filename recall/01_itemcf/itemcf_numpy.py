"""
ItemCF 的最小可运行实现（numpy + pandas 版）

目标：复现 fun-rec 2.1.1 节里那张表的例子
- 5 个用户对 5 个物品打过分
- 用户 1 没评过物品 5
- 预测用户 1 对物品 5 的评分（教材答案：约 4.6）

【这一份代码我已经帮你写好了"框架 + 数据"，但留了 3 个 TODO 让你自己实现】
按照 TODO 的顺序写，每写完一个 TODO 跑一下，看输出。
"""

import numpy as np
import pandas as pd


# ============================================================
# 【0】准备数据：原封不动来自 fun-rec 教材表 2.1.1
# ============================================================
# 数据结构：{物品名: {用户名: 评分}}
# 注意 item5 缺了 user1 的评分（这正是我们要预测的）
item_data = {
    "item1": {"user1": 5, "user2": 3, "user3": 4, "user4": 3, "user5": 1},
    "item2": {"user1": 3, "user2": 1, "user3": 3, "user4": 3, "user5": 5},
    "item3": {"user1": 4, "user2": 2, "user3": 4, "user4": 1, "user5": 5},
    "item4": {"user1": 4, "user2": 3, "user3": 3, "user4": 5, "user5": 2},
    "item5": {           "user2": 3, "user3": 5, "user4": 4, "user5": 1},  # user1 未知
}

print("=" * 60)
print("原始数据（行=物品，列=用户）")
print("=" * 60)
print(pd.DataFrame(item_data).T)  # .T 转置成"物品行 × 用户列"，方便看


# ============================================================
# 【1】构造一个空的相似度矩阵 5x5，对角线为 1（物品和自己 100% 相似）
# ============================================================
similarity_matrix = pd.DataFrame(
    np.identity(len(item_data)),  # np.identity(5) 生成 5x5 单位矩阵
    index=item_data.keys(),
    columns=item_data.keys(),
)

print("\n初始相似度矩阵（对角线 = 1）：")
print(similarity_matrix)


# ============================================================
# 【2】TODO ①：算每两个物品之间的皮尔逊相关系数，填进 similarity_matrix
# ============================================================
# 提示：
#   - 用两层 for 循环遍历所有 (i1, i2) 物品对（跳过自己和自己）
#   - 对于一对物品，找出"同时评分过这两个物品的用户"，构造 vec1 和 vec2
#   - 用 np.corrcoef(vec1, vec2)[0][1] 算皮尔逊相关系数
#   - 把结果填到 similarity_matrix[i1][i2]
#
# 写法参考（伪代码）：
#   for i1, users1 in item_data.items():
#       for i2, users2 in item_data.items():
#           if i1 == i2:
#               continue
#           vec1, vec2 = [], []
#           # 遍历 users1 里的每个用户，看 users2 里有没有这个用户
#           # 都有的话就把两个评分都加到 vec1 / vec2
#           ...
#           similarity_matrix[i1][i2] = np.corrcoef(vec1, vec2)[0][1]

# ↓↓↓↓↓ 你的代码写在这里 ↓↓↓↓↓

for i1, users1 in item_data.items():
    for i2,users2 in item_data.items():
        if i1 == i2:
            continue
        vec1,vec2=[],[]
        for user,rating1 in users1.items():
            rating2 = users2.get(user,-1)
            if rating2 == -1:
                continue
            vec1.append(rating1)
            vec2.append(rating2)

        similarity_matrix[i1][i2] = np.corrcoef(vec1,vec2)[0][1]
        


# ↑↑↑↑↑ 你的代码写在这里 ↑↑↑↑↑

print("\n" + "=" * 60)
print("【TODO ① 完成后】物品相似度矩阵：")
print("=" * 60)
print(similarity_matrix.round(3))
print("\n✅ 验证：similarity_matrix['item5']['item1'] 应该 ≈ 0.969")
print(f"   你算出来的是：{similarity_matrix['item5']['item1']:.3f}")


# ============================================================
# 【3】TODO ②：找出与 item5 最相似的 2 个物品（且 user1 评过）
# ============================================================
target_user = "user1"
target_item = "item5"
top_k = 2

# 提示：
#   - 取 similarity_matrix[target_item]（一列，包含 item5 与所有物品的相似度）
#   - 按相似度从高到低排序（.sort_values(ascending=False)）
#   - 遍历这个排序结果，跳过 item5 自己，跳过 user1 没评过的物品
#   - 取前 top_k 个

sim_items = []  # 最终要装 2 个最相似物品的名字

# ↓↓↓↓↓ 你的代码写在这里 ↓↓↓↓↓
sim_items_list = similarity_matrix[target_item].sort_values(ascending=False).index.tolist()
for item in sim_items_list:
    # 如果target_user对物品item评分过
    if target_user in item_data[item]:
        sim_items.append(item)
    if len(sim_items) == top_k:
        break
print(f'与物品{target_item}最相似的{top_k}个物品为：{sim_items}')

print(sim_items)


# ↑↑↑↑↑ 你的代码写在这里 ↑↑↑↑↑

print("\n" + "=" * 60)
print(f"【TODO ② 完成后】与 {target_item} 最相似且 {target_user} 评过的 {top_k} 个物品：")
print("=" * 60)
print(sim_items)
print("✅ 验证：应该是 ['item1', 'item4']")


# ============================================================
# 【4】TODO ③：套预测公式，算 user1 对 item5 的预测评分
# ============================================================
# 公式： r_hat = r_bar_target + Σ(w_jk · (r_uk - r_bar_k)) / Σ(w_jk)
#
# 提示：
#   - target_item_mean = item5 的平均评分（只用已知评分，list(item_data[target_item].values())）
#   - 遍历 sim_items 里的每个相似物品 k：
#       w_jk = similarity_matrix[target_item][k]
#       r_uk = item_data[k][target_user]
#       r_bar_k = item_data[k] 的平均评分
#       累加分子 / 分母
#   - 最终：预测分 = target_item_mean + 分子 / 分母

target_item_mean = 0.0   # ← TODO
weighted_scores = 0.0    # 分子（加权偏差累加）
corr_values_sum = 0.0    # 分母（相似度累加）

# ↓↓↓↓↓ 你的代码写在这里 ↓↓↓↓↓


# ↑↑↑↑↑ 你的代码写在这里 ↑↑↑↑↑

# 注意：如果 corr_values_sum 是 0 会除 0 报错；正常情况下不会
predicted_rating = target_item_mean + weighted_scores / corr_values_sum

print("\n" + "=" * 60)
print(f"【TODO ③ 完成后】预测 {target_user} 对 {target_item} 的评分：")
print("=" * 60)
print(f"  预测分数 = {predicted_rating:.3f}")
print(f"  教材答案 ≈ 4.6")
print(f"  误差     = {abs(predicted_rating - 4.6):.3f}")
print("=" * 60)


# ============================================================
# 🤔 思考题（写完代码后想一想）
# ============================================================
"""
1. 为什么 similarity_matrix 对角线要设成 1？把它设成 0 会怎样？

2. 在算物品相似度时，我们只用"同时评过两个物品的用户"。
   如果两个物品共同的用户特别少（比如只有 1 个），相似度可信吗？
   工业上怎么解决这个问题？（提示：加一个最小共同用户数阈值）

3. 这种"暴力两两算相似度"的复杂度是 O(N²)，N 是物品数。
   假设有 1000 万物品，能跑得动吗？工业上怎么优化？
   （提示：教材里提到了"用户-物品倒排表"）
"""
