# 学习路线（从零到 FastAPI 部署）

文档与代码分离管理：

| 类型 | 位置 | 说明 |
|------|------|------|
| 课件文档 `.md` | `docs/stage*/` | 直觉讲解、预期输出、常见报错、练习题 |
| 可运行代码 `.py` | `lessons/stage*/` | 与文档同名（如 `02_numpy_array`） |

## 使用方式

```powershell
cd e:\github\lind-py-machine-learning
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 先读文档，再跑代码
# 文档: docs\stage0\02_numpy_array.md
python lessons\stage0\02_numpy_array.py
```

建议：**先读 `docs/` 里对应课件，再跑 `lessons/` 里的脚本，再做 md 中的练习**。

命令速查见 [basic.md](./basic.md)。

## 阶段一览

| 阶段 | 文档目录 | 代码目录 | 目标 | 验收 |
|------|----------|----------|------|------|
| 0 环境与数据基础 | [stage0/](./stage0/) | `lessons/stage0/` | Python 环境、NumPy、Pandas、sklearn 数据 | 能加载内置数据集并做基础统计 |
| 1 可视化与 EDA | [stage1/](./stage1/) | `lessons/stage1/` | 画图、探索性分析 | 对乳腺癌数据写出简短 EDA 结论 |
| 2 监督学习入门 | [stage2/](./stage2/) | `lessons/stage2/` | 划分数据集、基线分类器 | 理解 train/val/test，跑通逻辑回归 |
| 3 特征与防泄漏 | [stage3/](./stage3/) | `lessons/stage3/` | Pipeline、交叉验证 | 标准化放进 Pipeline，CV 评估 |
| 4 对比与调参 | [stage4/](./stage4/) | `lessons/stage4/` | 多模型对比、网格搜索 | 选出较优模型并解释为何 |
| 5 完整项目 | [stage5/](./stage5/) | `lessons/stage5/` | 端到端训练并保存模型 | 产出 `artifacts/*.joblib` |
| 6 API 部署 | [stage6/](./stage6/) | `lessons/stage6/` | FastAPI 预测接口 | 本地 `POST /predict` 可调用 |

## 从第 1 课开始

→ [stage0/01_check_env.md](./stage0/01_check_env.md)

## 项目主线

统一使用 **sklearn 乳腺癌数据集（二分类）**：

问题定义 → EDA → 特征处理 → 基线 → 模型对比 → 调参 → 最终评估 → 保存模型 → FastAPI → README

## 学习建议

- 每天 1–1.5 小时，大约 10–12 周学完并做完项目
- 不要跳过划分数据与 Pipeline：这是防止「数据泄漏」的关键习惯
- 阶段 5、6 是综合演练；前面每课的练习都是为它们做准备
