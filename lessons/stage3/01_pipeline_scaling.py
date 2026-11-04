"""阶段 3 · 第 1 节：Pipeline 把预处理和模型绑在一起。"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline：fit 时按顺序；预测时自动先 transform 再 predict
pipe = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=42)),
    ]
)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print("Pipeline 准确率:", round(accuracy_score(y_test, pred), 4))
print("步骤:", list(pipe.named_steps.keys()))
print("好处: 交叉验证/调参时不会在错误的数据上 fit 缩放器。")
