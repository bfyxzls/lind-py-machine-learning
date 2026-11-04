"""阶段 2 · 第 2 节：逻辑回归基线模型。"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 基线也要做最基本的缩放：逻辑回归对特征尺度敏感
# 注意：scaler 只在 train 上 fit，再 transform train/test —— 防止泄漏
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(max_iter=5000, random_state=42)
clf.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)

print("准确率:", round(accuracy_score(y_test, y_pred), 4))
print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred, digits=4))
