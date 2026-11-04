"""阶段 4 · 第 1 节：多模型对比。"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    "logreg": LogisticRegression(max_iter=5000, random_state=42),
    "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    "svc": SVC(kernel="rbf", random_state=42),
}

print(f"{'model':10s} {'mean':>8s} {'std':>8s}")
print("-" * 30)
for name, model in candidates.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    # 树模型不强制缩放，但统一 Pipeline 便于公平对比与部署
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"{name:10s} {scores.mean():8.4f} {scores.std():8.4f}")

print("\n选择建议: 优先看均值，兼顾标准差与可解释性/推理速度。")
