# 阶段 4 · 第 2 节：网格搜索调参

## 学习目标

- 用 `GridSearchCV` 在 Pipeline 上搜超参数
- 遵守：搜索只在训练集，测试集最后评估一次

## 直觉

1. 超参数（如 `C`）不是数据学出来的，需要搜索或经验设定。
2. 网格搜索 = 穷举候选组合，用 CV 打分。
3. **数据泄漏高发点**：在全数据上调参，或把测试集卷入搜索。

## 运行

```powershell
python lessons\stage4\02_grid_search.py
```

## 练习

1. 为随机森林写一个小 `param_grid`（如 `n_estimators`, `max_depth`）并搜索。
2. 解释 `clf__C` 里双下划线的含义。

## 下一阶段

→ [../stage5/01_full_project.md](../stage5/01_full_project.md)
