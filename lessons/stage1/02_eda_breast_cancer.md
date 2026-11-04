# 阶段 1 · 第 2 节：EDA（探索性分析）

## 学习目标

- 完成一份最小 EDA：形状、缺失、类别分布、相关特征
- 会读简单相关热力图

## 直觉

1. EDA 回答：数据长什么样？有没有坑？哪些特征可能有用？
2. **相关 ≠ 因果**，但高相关特征常是建模优先候选。
3. 没有缺失、类别不太偏，是入门项目的理想起点。

## 运行

```powershell
python lessons\stage1\02_eda_breast_cancer.py
```

## 验收标准

能用 3–5 句话写出：样本量、类别分布、是否缺失、你认为重要的 2–3 个特征。

## 练习

1. 根据脚本输出，写一段「EDA 结论」（放进笔记即可）。
2. 对 `mean radius` 按 `target` 分组画箱线图（可用 seaborn `boxplot`）。

## 下一阶段

→ [../stage2/01_train_val_test_split.md](../stage2/01_train_val_test_split.md)
