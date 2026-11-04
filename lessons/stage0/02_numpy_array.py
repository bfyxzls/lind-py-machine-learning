"""阶段 0 · 第 2 节：NumPy 数组直觉。"""

import numpy as np

# 1. 用列表创建一维 / 二维数组
a = np.array([1, 2, 3, 4])
b = np.array([[1, 2], [3, 4], [5, 6]])

print("一维 a:", a)
print("二维 b:\n", b)
print("a.shape =", a.shape, "  b.shape =", b.shape)
print("a.dtype =", a.dtype)

# 2. 向量化运算：整组一起算，不用 for 循环
print("a * 2 =", a * 2)
print("a + 10 =", a + 10)
print("a 的均值 =", a.mean(), "  总和 =", a.sum())

# 3. 切片：取一部分（机器学习里常用来取特征列 / 样本行）
print("a 前两个元素:", a[:2])
print("b 第 0 行:", b[0])
print("b 第 1 列:", b[:, 1])
