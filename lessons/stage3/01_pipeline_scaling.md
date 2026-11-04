# 阶段 3 · 第 1 节：Pipeline 与标准化

## 学习目标

- 用 `Pipeline` 串联 `StandardScaler` + 分类器
- 明白 Pipeline 如何帮助防止预处理泄漏

## 直觉

1. 预处理和模型应视为**同一条流水线**，不要手写两段容易漏接的代码。
2. 交叉验证每一折都会重新 `fit` Pipeline → 缩放统计量不会偷看验证折。
3. 部署时也是保存整条 Pipeline，而不是只存模型。

## 运行

```powershell
python lessons\stage3\01_pipeline_scaling.py
```

## 练习

1. 把分类器换成 `from sklearn.svm import SVC`，仍放进 Pipeline，跑一次。
2. 说出：为什么 `pipe.fit(X_train, y_train)` 比手动 `scaler.fit` + `clf.fit` 更安全？

## 下一节

→ [02_cross_validation.md](./02_cross_validation.md)
