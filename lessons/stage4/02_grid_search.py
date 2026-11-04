"""阶段 4 · 第 2 节：网格搜索调参（在 Pipeline 内完成）。"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=42)),
    ]
)

# 参数名 = 步骤名__超参数名
param_grid = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l2"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
search.fit(X_train, y_train)

print("最佳参数:", search.best_params_)
print("CV 最佳准确率:", round(search.best_score_, 4))

best = search.best_estimator_
test_acc = accuracy_score(y_test, best.predict(X_test))
print("独立测试集准确率:", round(test_acc, 4))
print("提醒: GridSearch 只应看到 train；test 仅最终评估一次。")
