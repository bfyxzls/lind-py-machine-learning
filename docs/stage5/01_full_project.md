# 阶段 5：完整项目（训练 → 评估 → 保存）

完整机器学习项目不是单点技巧，而是一条可复现流水线：问题定义 → 数据与划分 → 预处理与模型（Pipeline）→ 调参 → 最终评估 → 保存产物。你可以把它理解成「从原材料到出厂检验的整条产线」——前面各课是零件，本课把它们装成能交付的版本。保存时不只存模型权重，还要带上特征名、类别名等元数据，否则部署端无法保证输入顺序与训练一致。本课用 `joblib` 导出到 `artifacts/`。

## 学习目标

- 把前面所有步骤串成一条可复现流水线
- 保存 **Pipeline + 特征名 + 类别名**，方便部署
- 测试集只做最终评估

## 问题定义（本项目）

| 项 | 内容 |
|----|------|
| 任务 | 二分类 |
| 数据 | sklearn `load_breast_cancer` |
| 输入 | 30 维细胞核特征 |
| 输出 | `malignant` / `benign` |
| 成功标准 | 测试集准确率与 ROC-AUC 达到可用水平，并成功导出模型 |

## 运行

在**项目根目录**：

```powershell
python lessons\stage5\01_full_project.py
```

成功后生成：`artifacts/breast_cancer_pipeline.joblib`

## 你应先自己改的地方（教学约定）

按 readme 约定，综合项目建议你：

1. 先自己写一版训练脚本
2. 再对照本文件查漏（划分、Pipeline、只 fit 训练集、保存元数据）

## 常见报错

| 现象 | 处理 |
|------|------|
| 找不到 `artifacts` | 在项目根目录运行 |
| `joblib` 缺失 | `pip install -r requirements.txt` |

## 验收

- [ ] 能打印测试集 classification_report
- [ ] `artifacts/` 下有模型文件
- [ ] 加载后能对单样本预测

## 下一阶段

→ [../stage6/app.md](../stage6/app.md)
