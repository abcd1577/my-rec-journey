"""
PyTorch 入门第 1 课：Tensor（张量）

Tensor = PyTorch 里的"多维数组"，是所有运算的基本单位。
你可以把它理解为"能在 GPU 上跑的 NumPy 数组"。

运行方式：
    python 01_tensor.py
"""

import torch
import numpy as np

print("=" * 50)
print("【1】创建 Tensor 的几种方式")
print("=" * 50)

# 1.1 从 Python 列表创建
a = torch.tensor([1.0, 2.0, 3.0])
print(f"从列表创建: {a}, 形状: {a.shape}, 数据类型: {a.dtype}")

# 1.2 从 NumPy 数组创建（推荐系统里 pandas → numpy → tensor 是常见流程）
np_arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = torch.from_numpy(np_arr)
print(f"\n从NumPy创建:\n{b}\n形状: {b.shape}")

# 1.3 创建特殊 Tensor
zeros = torch.zeros(2, 3)        # 全 0
ones = torch.ones(2, 3)          # 全 1
rand = torch.randn(2, 3)         # 标准正态分布随机数（深度学习常用！）
print(f"\n全0: \n{zeros}")
print(f"\n随机正态分布:\n{rand}")


print("\n" + "=" * 50)
print("【2】Tensor 的基本运算")
print("=" * 50)

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(f"加法 x + y     = {x + y}")
print(f"点乘 x * y     = {x * y}        # 逐元素相乘")
print(f"内积 x.dot(y)  = {x.dot(y)}     # 1*4 + 2*5 + 3*6 = 32")


print("\n" + "=" * 50)
print("【3】矩阵乘法（深度学习的核心运算！）")
print("=" * 50)

# 推荐系统中：用户向量 × 物品矩阵 = 用户对每个物品的偏好分数
user_vec = torch.randn(1, 4)        # 1 个用户，embedding 维度 4
item_mat = torch.randn(4, 10)       # 10 个物品，每个 embedding 也是 4 维
scores = user_vec @ item_mat        # @ 是矩阵乘法
print(f"用户向量 shape: {user_vec.shape}")
print(f"物品矩阵 shape: {item_mat.shape}")
print(f"打分结果 shape: {scores.shape}  # 1个用户对10个物品的分数")
print(f"打分结果: {scores}")


print("\n" + "=" * 50)
print("【4】Tensor 形状操作（推荐系统最常用！）")
print("=" * 50)

t = torch.arange(12)  # 0,1,2,...,11
print(f"原始 t: {t}, shape={t.shape}")

t2 = t.reshape(3, 4)   # 变成 3 行 4 列
print(f"\nreshape(3,4):\n{t2}")

t3 = t2.unsqueeze(0)   # 在最前面加一维（常用于"加 batch 维度"）
print(f"\nunsqueeze(0) 后 shape: {t3.shape}  # 多了 batch 维度")

t4 = t3.squeeze(0)     # 把大小为 1 的维度去掉
print(f"squeeze(0)   后 shape: {t4.shape}")


print("\n" + "=" * 50)
print("【5】MPS 加速（Apple 芯片专属）")
print("=" * 50)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✅ MPS 可用，使用设备: {device}")

    # 把 tensor 放到 GPU 上
    big_mat = torch.randn(1000, 1000, device=device)
    result = big_mat @ big_mat
    print(f"在 MPS 上跑了 1000x1000 矩阵乘法，结果 shape: {result.shape}")
else:
    print("❌ MPS 不可用，会用 CPU")


print("\n" + "=" * 50)
print("🎉 恭喜！你完成了 PyTorch 第 1 课")
print("=" * 50)
print("""
🤔 思考题（不用回答，自己想想）：
1. 为什么深度学习要用 Tensor 而不是 Python list？（提示：GPU + 自动微分）
2. unsqueeze 在什么场景下会用到？（提示：batch 维度）
3. @ 和 * 有什么区别？
""")
