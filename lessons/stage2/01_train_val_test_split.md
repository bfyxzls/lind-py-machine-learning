# 阶段 2 · 第 1 节：训练 / 验证 / 测试划分

## 学习目标

- 理解 train / val / test 各自职责
- 会用 `train_test_split`，并使用 `stratify`
- 建立「测试集只最终用一次」的习惯

## 直觉

1. **训练集**：学参数；**验证集**：选模型/调参；**测试集**：最终打分。
2. 用测试集反复调参 = 泄漏，分数会虚高。
3. `stratify=y` 让各子集里正负类比例接近，分类任务几乎总要开。

## 运行

```powershell
python lessons\stage2\01_train_val_test_split.py
```

## 预期输出（数量约）

```
全集: 569
train: 341 val: 114 test: 114
...
```

## 常见报错

| 现象 | 处理 |
|------|------|
| 某类样本过少导致 stratify 失败 | 减小划分份数或合并稀有类 |

## 练习

1. 改 `random_state`，看划分数量是否稳定、比例是否仍接近。
2. 用一句话解释：为什么不能把测试集也拿去 GridSearch？

## 下一节

→ [02_baseline_logistic.md](./02_baseline_logistic.md)
