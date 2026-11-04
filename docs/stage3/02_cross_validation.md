# 阶段 3 · 第 2 节：交叉验证

交叉验证（Cross-Validation，CV）把数据轮流切成多折：每一折轮流当验证集，其余当训练集，最后汇总多折分数（常用均值 ± 标准差）。你可以把它理解成「多次模拟考取平均分」——比单次 train/test 划分更稳，能减少运气成分。分类任务优先用 `StratifiedKFold`，保证每折里各类比例接近；sklearn 的 `cross_val_score` 可以一行代码完成这件事。注意：CV 用于比较与选模型，最终仍建议留一份独立测试集做终评。

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
