"""阶段 1 · 第 1 节：Matplotlib 基础画图。"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target

# 取两个特征做散点图，看两类是否可分
feat_a, feat_b = 0, 1  # mean radius, mean texture
plt.figure(figsize=(6, 4))
plt.scatter(X[y == 0, feat_a], X[y == 0, feat_b], alpha=0.5, label=data.target_names[0])
plt.scatter(X[y == 1, feat_a], X[y == 1, feat_b], alpha=0.5, label=data.target_names[1])
plt.xlabel(data.feature_names[feat_a])
plt.ylabel(data.feature_names[feat_b])
plt.title("Breast Cancer: 2 features scatter")
plt.legend()
plt.tight_layout()
plt.savefig("lessons/stage1/scatter_two_features.png", dpi=120)
print("已保存图: lessons/stage1/scatter_two_features.png")

# 标签分布柱状图
plt.figure(figsize=(4, 3))
counts = [ (y == 0).sum(), (y == 1).sum() ]
plt.bar(data.target_names, counts, color=["salmon", "seagreen"])
plt.ylabel("count")
plt.title("Class balance")
plt.tight_layout()
plt.savefig("lessons/stage1/class_balance.png", dpi=120)
print("已保存图: lessons/stage1/class_balance.png")
