# 阶段 4 · 第 1 节：多模型对比

## 学习目标

- 在同一 CV 设定下对比 ≥3 个模型
- 学会用表格思维记录 mean/std

## 直觉

1. 「最好的模型」依赖数据；要用同一套评估协议对比。
2. 分数接近时，选更简单、更快、更好解释的。
3. 缩放对 SVM/逻辑回归重要；对随机森林影响较小，但统一流程更稳妥。

## 运行

```powershell
python lessons\stage4\01_model_compare.py
```

## 练习

1. 增加 `KNeighborsClassifier`，加入对比表。
2. 写下你准备带入调参的 Top1 模型及理由（1–2 句）。

## 下一节

→ [02_grid_search.md](./02_grid_search.md)
