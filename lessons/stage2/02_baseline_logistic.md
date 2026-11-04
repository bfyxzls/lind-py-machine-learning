# 阶段 2 · 第 2 节：逻辑回归基线

## 学习目标

- 训练一个可运行的分类基线
- 读懂准确率、混淆矩阵、precision/recall
- 理解：缩放器必须 `fit` 在训练集上

## 直觉

1. **基线** = 先有一个够简单、够稳的分数，后面一切改进都跟它比。
2. 逻辑回归输出「属于正类的概率」，阈值默认 0.5 转成类别。
3. 若在**全部数据**上 `fit` 缩放再划分，测试信息会泄漏进训练。

## 运行

```powershell
python lessons\stage2\02_baseline_logistic.py
```

## 预期输出

准确率通常在 **0.95+**（该数据集较「友好」）。关注分类报告里的 precision/recall。

## 常见报错

| 现象 | 处理 |
|------|------|
| `ConvergenceWarning` | 增大 `max_iter`，或先标准化 |
| 准确率低得离谱 | 检查是否忘了 `stratify` / 标签弄反 |

## 练习

1. 去掉 `StandardScaler`，对比准确率变化。
2. 解释混淆矩阵四个格子：TP/TN/FP/FN 哪个对应「把恶性判成良性」（结合 `target_names`）。

## 下一阶段

→ [../stage3/01_pipeline_scaling.md](../stage3/01_pipeline_scaling.md)
