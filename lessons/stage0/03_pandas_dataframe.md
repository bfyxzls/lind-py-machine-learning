# 阶段 0 · 第 3 节：Pandas 表格（DataFrame）

## 学习目标

- 会创建、查看 `DataFrame`
- 会选列、按条件筛行
- 知道缺失值怎么检查和简单填充

## 直觉

1. `DataFrame` ≈ Excel 表：行是样本，列是特征/标签。
2. EDA（探索性分析）几乎都在 Pandas 上完成。
3. 真实数据常有缺失；先 `isna()` 看清楚，再决定填充或删除。

## 运行

```powershell
python lessons\stage0\03_pandas_dataframe.py
```

## 预期输出（摘要）

- 打印 4 行 3 列的表
- `shape` 为 `(4, 3)`
- `describe()` 给出均值、标准差等
- 筛选出 `income > 5000` 的行
- 演示缺失值填充

## 常见报错

| 现象 | 处理 |
|------|------|
| `KeyError: 'age'` | 列名写错，用 `df.columns` 检查 |
| `SettingWithCopyWarning` | 先 `df = df.copy()` 再改，或用 `.loc` |

## 练习

1. 给 `df` 增加一列 `income_k = income / 1000`，打印前 3 行。
2. 统计 `label` 各类有多少个：`df["label"].value_counts()`。

## 下一节

→ [04_sklearn_builtin_data.md](./04_sklearn_builtin_data.md)
