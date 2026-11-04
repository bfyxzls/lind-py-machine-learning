"""阶段 5：完整项目脚本 —— 问题定义到保存模型。"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS / "breast_cancer_pipeline.joblib"

# ---------- 1. 问题定义 ----------
# 二分类：根据细胞核特征预测肿瘤为 malignant(0) / benign(1)
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)

# ---------- 2–3. 划分（EDA/特征工程在前面课程已做；此处聚焦可复现训练） ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 4–6. 基线 + 调参（Pipeline 内完成，防泄漏） ----------
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=42)),
    ]
)
param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
search.fit(X_train, y_train)
best = search.best_estimator_

# ---------- 7. 最终评估（测试集只用一次） ----------
y_pred = best.predict(X_test)
y_proba = best.predict_proba(X_test)[:, 1]
print("最佳参数:", search.best_params_)
print("CV 准确率:", round(search.best_score_, 4))
print("\n测试集报告:\n", classification_report(y_test, y_pred, target_names=data.target_names))
print("测试集 ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

# ---------- 8. 保存模型 + 元数据（部署需要特征顺序） ----------
bundle = {
    "pipeline": best,
    "feature_names": feature_names,
    "target_names": list(data.target_names),
    "best_params": search.best_params_,
}
joblib.dump(bundle, MODEL_PATH)
print("\n模型已保存:", MODEL_PATH.resolve())

# 冒烟：加载后再预测一行
loaded = joblib.load(MODEL_PATH)
sample = X_test[0:1]
print("冒烟预测:", loaded["target_names"][int(loaded["pipeline"].predict(sample)[0])])
