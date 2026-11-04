"""阶段 2 · 第 1 节：训练 / 验证 / 测试划分（防泄漏第一步）。"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

# 先分出测试集（只在最终评估用一次）
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 再从剩余里分出训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)
# 0.8 * 0.25 = 0.2 → 约 train 60% / val 20% / test 20%

print("全集:", X.shape[0])
print("train:", X_train.shape[0], "val:", X_val.shape[0], "test:", X_test.shape[0])
print("train 正类比例:", round(y_train.mean(), 3))
print("val   正类比例:", round(y_val.mean(), 3))
print("test  正类比例:", round(y_test.mean(), 3))
print("\n规则: 调参看 val；最终报告只看 test；test 不能参与选模型。")
