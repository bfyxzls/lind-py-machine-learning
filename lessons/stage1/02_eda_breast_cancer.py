"""阶段 1 · 第 2 节：乳腺癌数据 EDA（探索性分析）。"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print("=== 基本信息 ===")
print("形状:", df.shape)
print("缺失值总数:", int(df.isna().sum().sum()))
print("类别分布:\n", df["target"].value_counts().sort_index())

print("\n=== 按类别看 mean radius 的均值 ===")
print(df.groupby("target")["mean radius"].mean())

# 与 target 相关性最高的若干特征（绝对值）
corr = df.corr(numeric_only=True)["target"].drop("target").abs().sort_values(ascending=False)
print("\n=== 与 target 相关度 Top 8 ===")
print(corr.head(8))

top_feats = list(corr.head(4).index) + ["target"]
plt.figure(figsize=(6, 5))
sns.heatmap(df[top_feats].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Top features correlation")
plt.tight_layout()
plt.savefig("lessons/stage1/corr_heatmap.png", dpi=120)
print("\n已保存: lessons/stage1/corr_heatmap.png")

print("\n=== EDA 小结模板（请自己改写成文字）===")
print("1. 样本量与特征数:", df.shape)
print("2. 有无缺失:", df.isna().sum().sum() == 0)
print("3. 类别是否平衡: 见 value_counts")
print("4. 可能有用的特征: ", list(corr.head(5).index))
