# 学习路线（从零到 FastAPI 部署）

按阶段顺序学习。每个主题有：

- `*.py`：可直接运行的最小示例
- `*.md`：直觉讲解、预期输出、常见报错、练习题

## 使用方式

```powershell
cd e:\github\lind-py-machine-learning
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 示例：运行某一课
python lessons\stage0\02_numpy_array.py
```

建议：**先读同名 `.md`，再跑 `.py`，再做 md 里的练习**。

## 阶段一览

| 阶段 | 目录 | 目标 | 验收 |
|------|------|------|------|
| 0 环境与数据基础 | `stage0/` | Python 环境、NumPy、Pandas、sklearn 数据 | 能加载内置数据集并做基础统计 |
| 1 可视化与 EDA | `stage1/` | 画图、探索性分析 | 对乳腺癌数据写出简短 EDA 结论 |
| 2 监督学习入门 | `stage2/` | 划分数据集、基线分类器 | 理解 train/val/test，跑通逻辑回归 |
| 3 特征与防泄漏 | `stage3/` | Pipeline、交叉验证 | 标准化放进 Pipeline，CV 评估 |
| 4 对比与调参 | `stage4/` | 多模型对比、网格搜索 | 选出较优模型并解释为何 |
| 5 完整项目 | `stage5/` | 端到端训练并保存模型 | 产出 `artifacts/model.joblib` |
| 6 API 部署 | `stage6/` | FastAPI 预测接口 | 本地 `POST /predict` 可调用 |

## 项目主线

统一使用 **sklearn 乳腺癌数据集（二分类）**：

问题定义 → EDA → 特征处理 → 基线 → 模型对比 → 调参 → 最终评估 → 保存模型 → FastAPI → README

## 学习建议

- 每天 1–1.5 小时，大约 10–12 周学完并做完项目
- 不要跳过划分数据与 Pipeline：这是防止「数据泄漏」的关键习惯
- 阶段 5、6 是综合演练；前面每课的练习都是为它们做准备
