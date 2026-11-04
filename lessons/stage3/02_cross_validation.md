# 阶段 3 · 第 2 节：交叉验证

## 学习目标

- 会用 `cross_val_score` + `StratifiedKFold`
- 会报告「均值 ± 标准差」

## 直觉

1. 一次划分的分数像抽签；**K 折交叉验证**把数据轮流当验证集。
2. 分类用 `StratifiedKFold`，保持每折类别比例。
3. CV 分数用于比较模型；最终仍建议留一份独立 test。

## 运行

```powershell
python lessons\stage3\02_cross_validation.py
```

## 练习

1. 把 `scoring` 改成 `"f1"`，对比结果。
2. `n_splits=10` 时均值是否接近？标准差如何变？

## 下一阶段

→ [../stage4/01_model_compare.md](../stage4/01_model_compare.md)
