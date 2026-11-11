# 飞桨 UIE 实战项目 · 电商评论观点抽取

基于仓库内 `readme.md`（飞桨 UIE 介绍与训推流程）落地的**可训练、可推理**工程。

## 自定义领域

| 项 | 内容 |
|----|------|
| 领域 | **电商商品评论 · 观点抽取** |
| Schema（UIE） | `商品属性` / `评价词` / `情感极性` |
| 情感极性 | **原规则**：文本中带 `【情感极性：正向/负向】`，由 UIE 直接抽 span |
| 用途 | 从评论抽出「评价什么、怎么说、正/负向」 |

与文档中的「案件五字段」同构，只是换成更容易理解的业务例子。

## 目录结构

```textpython -m venv .venv
projects/baidu_uie/
  readme.md                 # 原 UIE 平台/GPU 部署长文档
  PROJECT.md                # 本项目使用说明（你在看的这份）
  requirements.txt
  configs/                  # schema + 训练/服务配置
  scripts/
    00_gen_synthetic_data.py
    02_train_uie.py
    05_infer_checkpoint.py
    06_zero_shot_infer.py
    check_env.py
  vendor/uie/finetune.py    # 微调实现
  mvp_worker/               # FastAPI 抽取服务
  datasets/                 # 运行后生成
  models/                   # 运行后生成
```

环境变量 `UIE_ROOT` 可改数据/模型根目录（默认=本项目目录），对齐文档约定。

## 1. 安装（建议独立环境）

```powershell
cd e:\github\lind-py-machine-learning\projects\baidu_uie
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip

# 1) Paddle CPU（勿用 packages/stable/cpu 源，Windows 常报 No matching distribution）
python -m pip install "paddlepaddle==2.6.2" "numpy>=1.21,<2" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) Windows：不要直接 pip install -r requirements.txt
#    paddlenlp 依赖的 tool-helpers 只有 Linux 轮子，会失败。请两步装：
python -m pip install -r requirements-windows.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install "paddlenlp==2.8.1" --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或一键脚本：
# powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1

python scripts\check_env.py
```

Linux/GPU 仍可用 `requirements.txt` + 文档中的 `paddlepaddle-gpu`。  
要求：Python **3.8–3.12**（64 位）。本项目实测 **3.11.9**。

**Windows 常见坑**

| 报错 | 处理 |
|------|------|
| `No matching distribution for paddlepaddle`（官方 cpu 源） | 改清华/PyPI 源 |
| `No matching distribution for tool-helpers` | 用 `requirements-windows.txt` + `paddlenlp --no-deps` |
| `cannot import name 'repo_type_and_id_from_hf_id'` | 钉死 `huggingface-hub<0.24`（已写入 windows 依赖） |

GPU（CUDA 11.8）请按 `readme.md` §4 安装 `paddlepaddle-gpu==2.6.2`，并用 `configs/train_gpu_base.json`。

## 2. 推荐流程

```powershell
cd e:\github\lind-py-machine-learning\projects\baidu_uie
.\.venv\Scripts\Activate.ps1

# A. 合成数据（无需 Paddle 也可跑）
python scripts\00_gen_synthetic_data.py --dataset-version review/smoke-v0.1.0

# B. 零样本试抽（需 PaddleNLP；不训练）
python scripts\06_zero_shot_infer.py --model uie-mini

# C. 少样本微调
python scripts\02_train_uie.py --config configs\train_cpu_smoke.json

# D. 加载 checkpoint 推理
python scripts\05_infer_checkpoint.py

# E. HTTP 服务（可选）
$env:UIE_MODEL_PATH = "models\review\<run_id>\checkpoint\export"
python -m mvp_worker --config configs\mvp_worker.json
```

### 如何测试 HTTP 接口

服务保持运行时，另开一个终端：

1. **健康检查**（你已经测通了）——只说明进程活着、模型已加载：

```powershell
curl http://127.0.0.1:8088/health
```

期望类似：`{"status":"ok","model":"...","schema":["商品属性","评价词","情感极性"]}`

2. **真正抽字段**——调用 `POST /extract`（PowerShell 推荐）：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8088/extract -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body '{"text":"这款手机屏幕很清晰，整体满意。【情感极性：正向】"}'
```

说明：当前方案下，请求文本需带 `【情感极性：正向】` / `【情感极性：负向】`，模型才能抽出「情感极性」字段（与训练标注一致）。

或用 curl（Git Bash / 新版 Windows curl）：

```bash
curl -s http://127.0.0.1:8088/extract ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"这款手机屏幕很清晰，整体满意。【情感极性：正向】\"}"
```

3. **浏览器可视化调试**：打开 http://127.0.0.1:8088/docs  
   点开 `POST /extract` → Try it out → 填 `text` → Execute。

成功标志：`DATA_OK` → `TRAIN_OK` → `INFER_OK` / `ZERO_SHOT_OK`；HTTP 能返回 `result` 里的字段 span。

## 3. 数据格式

`datasets/dataset_versions/review/smoke-v0.1.0/train.txt` 每行 JSON：

```json
{
  "content": "这款手机屏幕很清晰，整体满意。【情感极性：正向】",
  "prompt": "商品属性",
  "result_list": [{"text": "屏幕", "start": 4, "end": 6}]
}
```

换真实业务时：按同样格式造 `train.txt` / `dev.txt`，改 `configs` 里的 `dataset_version` 与 `schema` 即可（见 `readme.md` §9）。

## 4. 与原文档脚本对照

| 文档 | 本项目 |
|------|--------|
| `00_gen_synthetic_data.py` | ✅ 电商评论版 |
| `02_train_uie.py` | ✅ |
| `05_infer_checkpoint.py` | ✅ |
| `mvp_worker` | ✅ FastAPI |
| 案件 schema | 改为评论 schema |
| GPU ldlibs / Miniconda | 仍以 `readme.md` 为准 |

## 5. 注意

- 首次运行会从飞桨源站拉取 `uie-mini`/`uie-base` 权重，需可访问外网或已有缓存。
- CPU 冒烟用 `uie-mini`；显存足够再上 `uie-base`/`uie-medium`。
- 合成数据仅用于**链路验收**；上线请换真实标注。
