# python machine learning

从零用 Python 学习机器学习，并完成：数据分析 → 建模 → FastAPI 部署。

## 快速开始

```powershell
cd e:\github\lind-py-machine-learning
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**学习路线与课件：** 见 [docs/README.md](docs/README.md)  
**命令速查：** 见 [docs/basic.md](docs/basic.md)

文档在 `docs/`，可运行代码在 `lessons/`（同名对应，如 `docs/stage0/02_numpy_array.md` ↔ `lessons/stage0/02_numpy_array.py`）。

## 最终项目（乳腺癌二分类）

```powershell
python lessons\stage5\01_full_project.py
uvicorn lessons.stage6.app:app --reload --host 127.0.0.1 --port 8000
```

- 模型文件：`artifacts/breast_cancer_pipeline.joblib`
- API 文档：http://127.0.0.1:8000/docs

## 教学约定（自学时请遵守）

1. 按阶段学习；每阶段有目标、知识点、练习、验收标准（写在各课 `.md`）。
2. 先读直觉，再跑最小示例，再做练习。
3. 数据集优先用 sklearn 内置小数据。
4. 调参与评估时坚持 train/val/test（或 CV）划分，防止数据泄漏。
5. 综合项目建议：自己先写一版，再对照 `stage5` / `stage6` 查漏。

## 背景（可自行填写）

- Python 水平：
- 数学水平：
- 每天可投入时间：
- 目标：
