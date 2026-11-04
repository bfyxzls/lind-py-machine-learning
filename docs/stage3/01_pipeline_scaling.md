# 阶段 3 · 第 1 节：Pipeline 与标准化

Pipeline（流水线）把「预处理 → 模型」绑成一个对象，`fit` / `predict` 时按固定顺序自动执行。标准化（如 `StandardScaler`）则把不同量纲的特征拉到相近尺度，让逻辑回归、SVM 这类对尺度敏感的模型更稳。你可以把它理解成「封装好的生产流水线」：原料进站后，先清洗再加工，步骤不会漏接，也不会在交叉验证时误用验证折的信息来做缩放。部署时通常保存整条 Pipeline，而不是只存最后的分类器。

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
