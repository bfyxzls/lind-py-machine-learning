# 阶段 0 · 第 4 节：sklearn 内置数据集

## 学习目标

- 用 `load_breast_cancer` 拿到 `X`、`y`
- 理解特征矩阵与标签向量
- 把数据转成 DataFrame 便于分析

## 直觉

1. 监督学习 = 用已知标签的历史数据，学「特征 → 标签」的映射。
2. `X` 是特征，`y` 是要预测的目标（这里：良性/恶性）。
3. 内置小数据集无需下载，适合练手；正式项目再换公开数据。

## 运行

```powershell
python lessons\stage0\04_sklearn_builtin_data.py
```

## 预期输出（摘要）

```
特征矩阵形状: (569, 30)
标签形状: (569,)
类别名: ['malignant' 'benign']
...
```

## 常见报错

| 现象 | 处理 |
|------|------|
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn`（包名与 import 名不同） |

## 练习

1. 打印第 0 个样本的全部 30 个特征值：`X[0]`。
2. 计算特征 `mean radius` 的均值（DataFrame 列名里找）。

## 阶段验收

能加载乳腺癌数据并说明：多少样本、多少特征、两类各多少。

## 下一阶段

→ [../stage1/01_matplotlib_basics.md](../stage1/01_matplotlib_basics.md)
