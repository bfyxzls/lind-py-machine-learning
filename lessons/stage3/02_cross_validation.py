"""阶段 3 · 第 2 节：交叉验证（更稳的评估）。"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=42)),
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

print("5 折准确率:", np.round(scores, 4))
print("均值 ± 标准差:", round(scores.mean(), 4), "±", round(scores.std(), 4))
print("说明: 单次 train/test 有偶然性；CV 看平均表现更稳。")
