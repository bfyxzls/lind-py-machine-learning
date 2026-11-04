"""阶段 0 · 第 4 节：加载 sklearn 内置数据集。"""

from sklearn.datasets import load_breast_cancer
import pandas as pd

# 乳腺癌二分类：后续项目主线都用它
data = load_breast_cancer()
X = data.data          # 特征矩阵 (n_samples, n_features)
y = data.target        # 标签 0/1
feature_names = data.feature_names
target_names = data.target_names

print("特征矩阵形状:", X.shape)
print("标签形状:", y.shape)
print("类别名:", target_names)
print("前 5 个特征名:", list(feature_names[:5]))
print("标签分布 (0/1 各多少):", dict(pd.Series(y).value_counts().sort_index()))

# 转成 DataFrame，方便 EDA
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
print("\n前 3 行:\n", df.head(3))
print("\n目标列含义: 0 =", target_names[0], ", 1 =", target_names[1])
