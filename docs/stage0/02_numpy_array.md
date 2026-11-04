# 阶段 0 · 第 2 节：NumPy 数组直觉

NumPy 是 Python 里做数值计算的基础库，核心是提供了一个高性能的多维数组对象 ndarray，以及大量对数组进行数学运算的函数。你可以把它理解成“Python 版的 MATLAB 数组”，或者“没有标签、只有数字的 Pandas DataFrame”。在机器学习里，NumPy 是底层计算引擎：Pandas 的列、scikit-learn 的模型输入、PyTorch/TensorFlow 的张量，底层都和 NumPy 数组高度兼容。

## 学习目标

- 理解 `ndarray`、`shape`、`dtype`
- 会做向量化运算与基础切片

## 直觉

1. ML 数据本质是数字表；NumPy 提供高效的数字表。
2. `shape` 描述形状，如 `(150, 4)` = 150 样本 × 4 特征。
3. `a * 2`、`a.mean()` 对整组数据一起算，少写 `for`。

## 运行

```powershell
.\.venv\Scripts\Activate.ps1
python lessons\stage0\02_numpy_array.py
```

## 预期输出

```
一维 a: [1 2 3 4]
二维 b:
 [[1 2]
 [3 4]
 [5 6]]
a.shape = (4,)   b.shape = (3, 2)
...
a 前两个元素: [1 2]
b 第 0 行: [1 2]
b 第 1 列: [2 4 6]
```

## 常见报错

| 现象 | 处理 |
|------|------|
| `ModuleNotFoundError: numpy` | 未激活虚拟环境 |
| 二维数组创建失败 | 每行长度必须一致（矩形） |

## 练习

1. `x = np.array([10,20,30,40,50])`，打印 `shape`、最大值、最小值。
2. 建一个 2×3 数组，打印第 1 行（`arr[0]`）和第 2 列（`arr[:, 1]`）。

## 下一节

→ [03_pandas_dataframe.md](./03_pandas_dataframe.md)
