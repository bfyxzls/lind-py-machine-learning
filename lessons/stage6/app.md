# 阶段 6：FastAPI 预测接口

## 学习目标

- 加载 `joblib` 模型并提供 HTTP 预测
- 会用 `/docs` 交互调试
- 理解请求必须与训练特征**顺序一致**

## 直觉

1. 训练产出「模型文件」；服务负责「接收特征 → 返回预测」。
2. API 契约（几个特征、什么顺序）必须与训练一致，否则结果无意义。
3. 先 `/health`，再 `/features`，最后 `/predict`。

## 前置

```powershell
python lessons\stage5\01_full_project.py
pip install -r requirements.txt
```

## 启动

在项目根目录：

```powershell
uvicorn lessons.stage6.app:app --reload --host 127.0.0.1 --port 8000
```

打开：http://127.0.0.1:8000/docs

## 调用示例

```powershell
# 健康检查
curl http://127.0.0.1:8000/health

# 预测（示例：30 个数字，请换成真实特征值）
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"features\": [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}"
```

PowerShell 也可用：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## 常见报错

| 现象 | 处理 |
|------|------|
| 启动时报模型不存在 | 先跑阶段 5 脚本 |
| `features` 长度错误 | 必须正好 30 维，顺序见 `GET /features` |
| 端口占用 | 换 `--port 8001` |

## 练习

1. 用 `/docs` 成功调用一次 `/predict`。
2. 为项目根目录写/完善 README：如何安装、训练、启动 API（可参考仓库 `readme.md` 学习约定，另写「项目使用说明」）。

## 项目收尾检查清单

- [ ] 阶段 0–4 课程能跑通
- [ ] 阶段 5 产出模型文件
- [ ] 阶段 6 `/predict` 返回 label 与概率
- [ ] README 写清复现步骤
