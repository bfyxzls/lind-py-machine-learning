# 阶段 1 · 第 1 节：Matplotlib 基础画图

## 学习目标

- 画散点图观察两类样本
- 画柱状图看类别是否平衡
- 会把图保存成 PNG

## 直觉

1. **先画图再建模**：很多问题靠眼睛就能发现异常或可分性。
2. 两类在特征空间里若明显分开，简单模型往往就够用。
3. 类别极不平衡时，准确率会骗人，后面要用更合适的指标。

## 运行

在项目根目录：

```powershell
python lessons\stage1\01_matplotlib_basics.py
```

会生成：

- `lessons/stage1/scatter_two_features.png`
- `lessons/stage1/class_balance.png`

## 常见报错

| 现象 | 处理 |
|------|------|
| 找不到保存路径 | 务必在**项目根目录**运行 |
| 中文乱码 | 本课标题用英文；中文需配置字体 |

## 练习

1. 改 `feat_a, feat_b` 为另外两个特征索引，重新出图，观察可分性变化。
2. 查两类数量，写一句：数据是否大致平衡？

## 下一节

→ [02_eda_breast_cancer.md](./02_eda_breast_cancer.md)
