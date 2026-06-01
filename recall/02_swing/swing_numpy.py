"""
Swing 算法的最小可运行实现

【目的】
对照 ItemCF 学 Swing。两者都是计算"物品-物品"相似度，但 Swing 更鲁棒。

【核心数据结构】
和 fun-rec 工程版一样，我们用两个倒排表：
    user_items: {user_id: set([item1, item2, ...])}   # 每个用户买过哪些东西
    item_users: {item_id: set([user1, user2, ...])}   # 每个物品被哪些用户买过

【任务】
- TODO ①：计算用户权重 w_u = 1/sqrt(|I_u|)
- TODO ②：实现核心函数 swing_score(i, j)
- TODO ③：在玩具数据上验证（与手算对比）

运行：
    python swing_numpy.py
"""

import numpy as np
from collections import defaultdict


# ============================================================
# 【0】构造书里 2.1.2 节图 2.1.3 那个例子
# ============================================================
# 用户 A：买过 h, t, r, p（4 个）
# 用户 B：买过 h, t, r, p（和 A 一模一样）
# 用户 C：买过 h, p（只 2 个）
#
# 我们关心两个相似度：
#   s(h, p)：直觉上应该高（用户 C 提供"特异性证据"）
#   s(h, t)：直觉上较低（只有 A、B 提供证据，但他俩什么都共同买）

interactions = [
    # (用户, 物品)
    ("A", "h"), ("A", "t"), ("A", "r"), ("A", "p"),
    ("B", "h"), ("B", "t"), ("B", "r"), ("B", "p"),
    ("C", "h"),                            ("C", "p"),
]


# ============================================================
# 【1】构建两个倒排表（核心数据结构！工业代码也这么干）
# ============================================================
user_items = defaultdict(set)   # 用户 -> 买过的物品集合
item_users = defaultdict(set)   # 物品 -> 被哪些用户买过

for user, item in interactions:
    user_items[user].add(item)
    item_users[item].add(user)

print("=" * 60)
print("【1】倒排表（Swing 算法的核心数据结构）")
print("=" * 60)
print("user_items:")
for u, items in user_items.items():
    print(f"  {u} -> {sorted(items)}")
print("\nitem_users:")
for i, users in item_users.items():
    print(f"  {i} -> {sorted(users)}")


# ============================================================
# 【2】TODO ①：计算每个用户的权重 w_u = 1/sqrt(|I_u|)
# ============================================================
# 提示：
#   - 用一个 dict: user_weights = {user: w_u}
#   - |I_u| 就是 len(user_items[u])
#   - w_u = 1.0 / np.sqrt(len(...))

user_weights = {}

# ↓↓↓↓↓ 你的代码写在这里 ↓↓↓↓↓
for u,items in user_items.items():
    user_weights[u] = 1.0 / np.sqrt(len(items))

# ↑↑↑↑↑ 你的代码写在这里 ↑↑↑↑↑

print("\n" + "=" * 60)
print("【2】用户权重 w_u = 1/sqrt(|I_u|)")
print("=" * 60)
for u, w in user_weights.items():
    print(f"  w_{u} = {w:.4f}  (|I_{u}|={len(user_items[u])})")
print("\n✅ 验证：")
print(f"  w_A 应 ≈ {1/np.sqrt(4):.4f}（你算的：{user_weights.get('A', 0):.4f}）")
print(f"  w_C 应 ≈ {1/np.sqrt(2):.4f}（你算的：{user_weights.get('C', 0):.4f}）")


# ============================================================
# 【3】TODO ②：实现核心函数 swing_score(item_i, item_j)
# ============================================================
# 公式（重要！）：
#   s(i, j) = ΣΣ w_u · w_v · 1/(α + |I_u ∩ I_v|)
#
#   外层 Σ：u 遍历"同时买过 i 和 j 的用户"
#   内层 Σ：v 遍历"同时买过 i 和 j 的用户"，u != v
#
# 提示：
#   1. 找共同用户：common_users = item_users[i] ∩ item_users[j]
#      （集合 intersection，Python 写法是 a & b 或 a.intersection(b)）
#   2. 至少要有 2 个共同用户才能算（少于 2 直接 return 0）
#   3. 双层 for u in common_users: for v in common_users: 跳过 u==v
#   4. 每对 (u, v)：算 |I_u ∩ I_v| = len(user_items[u] & user_items[v])
#   5. 累加贡献 w_u * w_v / (alpha + |I_u ∩ I_v|)

def swing_score(item_i, item_j, alpha=1.0):
    """计算 item_i 和 item_j 的 Swing 相似度分数"""

    # ↓↓↓↓↓ 你的代码写在这里 ↓↓↓↓↓
    score = 0.0
    common_users = item_users[item_i] & item_users[item_j]
    if len(common_users) < 2:
        return 0.0
    for u in common_users:
        for v in common_users:
            if u==v :
                continue
            common_items = user_items[u] & user_items[v]
            score += user_weights[u] * user_weights[v] / (alpha + len(common_items))

    return score
    # ↑↑↑↑↑ 你的代码写在这里 ↑↑↑↑↑


# ============================================================
# 【4】TODO ③：在玩具数据上验证
# ============================================================
print("\n" + "=" * 60)
print("【3】Swing 分数计算结果（α=1）")
print("=" * 60)

s_hp = swing_score("h", "p")
s_ht = swing_score("h", "t")
s_hr = swing_score("h", "r")
s_tr = swing_score("t", "r")

print(f"  s(h, p) = {s_hp:.4f}")
print(f"  s(h, t) = {s_ht:.4f}")
print(f"  s(h, r) = {s_hr:.4f}")
print(f"  s(t, r) = {s_tr:.4f}")

print(f"\n💡 关键观察：")
print(f"   s(h, p) 应该 > s(h, t)，因为用户 C 给 (h,p) 提供了特异性证据")
print(f"   你算出 s(h,p) > s(h,t) 吗？{'✅ 是' if s_hp > s_ht else '❌ 否，去检查代码'}")


# ============================================================
# 🤔 思考题
# ============================================================
"""
1. 为什么用户 A 和 B 给 (h, p) 的贡献分别只有 1/5？
   而用户 C 给 (h, p) 的贡献是多少？
   （手算验证一下你的理解）

2. 如果有个超级活跃用户 D，买过 100 个商品，他对相似度的影响为什么会被压制？
   （从 w_u 公式看：w_D = 1/sqrt(100) = 0.1，权重只有 A/B 的 0.2 倍）

3. ItemCF 的复杂度大致是 O(用户数 × 人均物品数²)
   Swing 的复杂度比它**慢多少**？为什么工业上还要用？
   （提示：3 层嵌套循环 vs 2 层；但相似度质量更高）

4. 教材里的图 2.1.3 算 s(h, p) 得到 13/15，我们算出来不一样，
   到底谁对？有没有可能两种实现都"对"，只是定义不同？
   （提示：是否双向枚举 (u, v) 和 (v, u)）
"""

from funrec import run_experiment

run_experiment('swing')