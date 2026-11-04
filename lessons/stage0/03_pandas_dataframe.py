"""阶段 0 · 第 3 节：Pandas 表格（DataFrame）。"""

import pandas as pd

# 用字典创建一张小表：每列一个特征，每行一个样本
df = pd.DataFrame(
    {
        "age": [25, 32, 47, 51],
        "income": [4000, 5200, 7800, 8100],
        "label": [0, 0, 1, 1],
    }
)

print("整张表:\n", df)
print("\n形状 (行, 列):", df.shape)
print("列名:", list(df.columns))
print("\n描述统计:\n", df.describe())

# 选列、筛选行 —— EDA 和特征工程天天用
print("\n只要 age 列:\n", df["age"])
print("\nincome > 5000 的行:\n", df[df["income"] > 5000])

# 缺失值处理直觉（真实数据经常有 NaN）
df2 = df.copy()
df2.loc[1, "income"] = None
print("\n含缺失:\n", df2)
print("每列缺失数:\n", df2.isna().sum())
print("用中位数填充后:\n", df2.fillna({"income": df2["income"].median()}))
