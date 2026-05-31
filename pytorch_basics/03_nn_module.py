"""
PyTorch 入门第 3 课：nn.Module —— 工程化的训练写法

【这一课的核心】
对比 02_autograd.py，回答一个问题：
    为什么不直接用 02 那种"手动管 w 和 b"的写法？

答：推荐模型动辄几百万参数（比如一个 user_embedding 表就是 [用户数, 64]），
   手动管根本不可能。PyTorch 提供了一套"工程化"的工具：
     - nn.Module     ：把模型封装成一个类，参数自动管理
     - nn.Linear     ：现成的"线性层"，等价于 02 课的 w*x + b
     - 优化器 optim   ：自动帮你做 w -= lr * w.grad
     - loss 函数      ：现成的 MSE / CrossEntropy 等

之后所有的推荐模型（DSSM/DeepFM/DIN）都长这个样子，所以这一课必须吃透。

运行：
    python 03_nn_module.py
"""

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 【1】定义模型：继承 nn.Module
# ============================================================
# 套路：__init__ 里声明"有哪些层"，forward 里写"数据怎么流过这些层"
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(in_features, out_features) = 一个线性变换 y = W·x + b
        # 这里 in=1, out=1，就等价于 02 课里的 w*x + b（一个标量乘一个标量）
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        # x 的 shape 必须是 [batch_size, in_features]
        return self.linear(x)


# ============================================================
# 【2】准备训练数据
# ============================================================
# 我们造一个"真实规律"：y = 2x + 1，加一点点噪声
# 让模型自己从数据里把 w=2, b=1 学出来
torch.manual_seed(42)  # 固定随机种子，保证每次结果一样，方便对照
n_samples = 100
x_train = torch.linspace(-1, 1, n_samples).unsqueeze(1)   # shape [100, 1]
y_train = 2 * x_train + 1 + 0.1 * torch.randn_like(x_train)  # 加噪声

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"前 3 条数据：")
for i in range(3):
    print(f"  x={x_train[i].item():.3f}, y={y_train[i].item():.3f}")


# ============================================================
# 【3】实例化"模型 / 损失函数 / 优化器" —— 三件套
# ============================================================
model = LinearModel()
criterion = nn.MSELoss()                         # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.1)  # SGD = 随机梯度下降

# 看看模型里有哪些可学习的参数（PyTorch 自动帮我们追踪了！）
print("\n模型的可学习参数：")
for name, param in model.named_parameters():
    print(f"  {name}: shape={tuple(param.shape)}, 初始值={param.data.flatten().tolist()}")


# ============================================================
# 【4】训练循环 —— 5 步法（请背下来！所有 PyTorch 训练都是这 5 步）
# ============================================================
print("\n开始训练...")
n_epochs = 200

for epoch in range(n_epochs):
    # ① Forward：算预测值
    y_pred = model(x_train)

    # ② 算 loss
    loss = criterion(y_pred, y_train)

    # ③ 清零上一轮的梯度（必须！）
    optimizer.zero_grad()

    # ④ Backward：自动算梯度
    loss.backward()

    # ⑤ 用梯度更新参数（这一行替代了 02 课里手写的 w -= lr * w.grad）
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f"Epoch {epoch+1:3d} | loss={loss.item():.6f} | w={w:.4f}, b={b:.4f}")


# ============================================================
# 【5】查看最终学到的参数
# ============================================================
print("\n" + "=" * 50)
print("训练完成！")
print(f"  真实参数:  w=2.0000, b=1.0000")
print(f"  学到参数:  w={model.linear.weight.item():.4f}, b={model.linear.bias.item():.4f}")
print("=" * 50)


# ============================================================
# 【6】对比表：02 课 vs 03 课
# ============================================================
print("""
┌──────────────┬─────────────────────────┬─────────────────────────────┐
│   步骤       │   02 课（手动）         │   03 课（工程化）           │
├──────────────┼─────────────────────────┼─────────────────────────────┤
│ 定义参数     │ w=tensor(...,            │ nn.Linear(1, 1)             │
│              │ requires_grad=True)     │ （自动管理 w 和 b）         │
│ 算 loss      │ (y_pred-y_true)**2      │ nn.MSELoss()                │
│ 清零梯度     │ w.grad.zero_()          │ optimizer.zero_grad()       │
│ 更新参数     │ w -= lr * w.grad        │ optimizer.step()            │
│              │ （手动写）              │ （一行搞定全部参数）        │
└──────────────┴─────────────────────────┴─────────────────────────────┘

🤔 思考题：
1. 为什么 model(x_train) 能直接调用？它没有定义 __call__ 啊。
   （提示：nn.Module 父类帮你定义了，它会自动调你写的 forward）

2. optimizer.zero_grad() 为什么要每轮都调？不调会怎样？
   （提示：PyTorch 的梯度默认是"累加"的，不是"覆盖"的）

3. 如果我现在要把模型从 y = wx+b 改成 y = w2·x² + w1·x + b（二次函数），
   __init__ 和 forward 各要怎么改？
""")
