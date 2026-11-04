# 常用命令速查

## 环境

- 升级 pip：`python -m pip install --upgrade pip`
- 安装依赖：`python -m pip install -r requirements.txt`
- 激活虚拟环境：`.\.venv\Scripts\Activate.ps1`

## 学习入口

- 路线总览：`lessons/README.md`
- 每课先读同名 `.md`，再运行对应 `.py`

## 示例运行

```powershell
python lessons\stage0\01_check_env.py
python lessons\stage0\02_numpy_array.py
python lessons\stage5\01_full_project.py
uvicorn lessons.stage6.app:app --reload --host 127.0.0.1 --port 8000
```

## 最终项目

1. 训练并保存模型：`python lessons\stage5\01_full_project.py`
2. 启动 API：`uvicorn lessons.stage6.app:app --reload --port 8000`
3. 浏览器打开：http://127.0.0.1:8000/docs
