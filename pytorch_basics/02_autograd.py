"""
PyTorch 入门第 2 课：自动微分（autograd）

PyTorch 最厉害的能力：自动算梯度。
你只要写 forward（前向计算），它能自动帮你算 backward（反向传播）。

运行方式：
    python 02_autograd.py
"""

import torch

print("=" * 50)
print("【1】requires_grad：开启梯度追踪")
print("=" * 50)

# 创建一个需要算梯度的 Tensor（深度学习里所有"参数"都要 requires_grad=True）
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x}, requires_grad = {x.requires_grad}")

# 定义一个函数 y = x^2 + 2x + 1
y = x ** 2 + 2 * x + 1
print(f"y = x² + 2x + 1 = {y.item()}")

# 反向传播：自动算 dy/dx
y.backward()
print(f"dy/dx 在 x=3 时 = {x.grad.item()}  # 数学上 = 2x+2 = 8 ✅")


print("\n" + "=" * 50)
print("【2】多变量的梯度（神经网络的雏形）")
print("=" * 50)

# 模拟"一个最简单的线性模型"：y_pred = w * x + b
# 我们要学的"参数"是 w 和 b
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# 训练数据：x=2 时真实 y=5（也就是说 w=2, b=1 是最优解）
x_train = torch.tensor(2.0)
y_true  = torch.tensor(5.0)

# Forward：计算预测值
y_pred = w * x_train + b
print(f"初始 w={w.item()}, b={b.item()}, y_pred={y_pred.item()}, y_true={y_true.item()}")

# 计算 loss（用最常见的均方误差 MSE）
loss = (y_pred - y_true) ** 2
print(f"loss = (y_pred - y_true)² = {loss.item()}")

# Backward：自动算每个参数的梯度
loss.backward()
print(f"dloss/dw = {w.grad.item()}  # 这告诉我们 w 该往哪个方向调")
print(f"dloss/db = {b.grad.item()}")


print("\n" + "=" * 50)
print("【3】手动跑一次"梯度下降"（揭秘训练的本质）")
print("=" * 50)

# 重新初始化
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.01   # 学习率

x_train = torch.tensor(2.0)
y_true  = torch.tensor(5.0)

print(f"目标：让 y_pred 接近 y_true={y_true.item()}")
print(f"初始: w={w.item():.4f}, b={b.item():.4f}\n")

# 跑 100 轮，看看 w 和 b 怎么自动逼近最优解
for step in range(100):
    # Forward
    y_pred = w * x_train + b
    loss = (y_pred - y_true) ** 2

    # Backward（算梯度）
    loss.backward()

    # 手动更新参数（这就是"训练"的本质！）
    with torch.no_grad():        # 这一段不要追踪梯度
        w -= lr * w.grad
        b -= lr * b.grad

    # 清零梯度（重要！否则梯度会累加）
    w.grad.zero_()
    b.grad.zero_()

    if step % 10 == 0:
        print(f"Step {step:3d}: loss={loss.item():.6f}, w={w.item():.4f}, b={b.item():.4f}, y_pred={y_pred.item():.4f}")

print(f"\n✅ 训练完成！最终 w={w.item():.4f}, b={b.item():.4f}")
print(f"   y_pred = w*2 + b = {(w * 2 + b).item():.4f}，接近目标 5 ✅")


print("\n" + "=" * 50)
print("🎉 你刚刚亲手写了一次"机器学习训练"！")
print("=" * 50)
print("""
💡 核心循环（所有神经网络训练都是这个套路）：
   1. forward  →  得到预测值
   2. 算 loss
   3. backward →  自动算梯度
   4. 用梯度更新参数
   5. 清零梯度，进入下一轮

🤔 思考题：
1. 如果 lr 改成 0.5 会怎样？（试试看，会发散！）
2. 如果不调用 zero_grad() 会怎样？（梯度累加，训练废了）
3. 为什么参数更新要在 with torch.no_grad() 里？
""")
