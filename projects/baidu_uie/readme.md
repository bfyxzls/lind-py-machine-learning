# 飞桨 UIE介绍

> **本仓库实战工程**：自定义领域「电商评论观点抽取」的训练 / 推理 / HTTP 服务，见同目录 **[PROJECT.md](./PROJECT.md)**。  
> 下文保留平台级 GPU 部署与操作记录，供对照环境与训推约定。

飞桨 UIE（Universal Information Extraction）是百度飞桨（PaddleNLP）基于文心大模型 ERNIE 3.0 训练并开源的**通用信息抽取模型**。它的核心用途是：让用户**用自然语言自定义需要抽取的目标**，无需训练或仅需极少样本，就能从文本中提取出实体、关系、事件等结构化信息。

### 🎯 它能做什么

UIE 将多种信息抽取任务统一到了一个框架下，主要能力包括：

*   **实体抽取**：识别文本中的特定实体，如人名、地名、公司名、时间等。你可以自己定义要抽取的类别，比如“赛事名称”“药物名称”。
*   **关系抽取**：提取两个实体之间的语义关系，例如从文本中抽取出“电影《XX》的导演是YY”这样的三元组信息。
*   **事件抽取**：从文本中识别事件的发生及其关键要素。例如，从一段地震报道中抽取出“时间”“震中位置”“震级”等信息。
*   **情感与观点抽取**：抽取评价的维度、观点词以及情感倾向（正向/负向）。例如，从商品评论中提取“屏幕：清晰（正向）”。

### 💡 核心优势

UIE 最大的特点是**灵活与低成本**。

传统的信息抽取模型通常需要为每个特定任务（如抽人名、抽公司名）准备大量标注数据来训练。而 UIE 借助 Prompt 思想，实现了**零样本（Zero-shot）** 和**少样本（Few-shot）** 抽取。
*   **零样本**：对于简单的场景，你只需要告诉模型你要抽什么（比如设定 schema 为 `['时间', '选手', '赛事名称']`），即使不提供任何训练数据，模型也能直接进行抽取。
*   **少样本微调**：对于复杂的专业领域（如医疗病历、金融公告），只需标注少量数据（例如 5 条）对模型进行微调，效果就能显著提升，大幅降低了数据标注的成本和门槛。

### 📦 系列模型

除了基础的文本 UIE 模型，飞桨还推出了针对不同场景的版本：
*   **UIE-X**：支持**跨模态文档**抽取，能同时处理文本和图像信息，适合从扫描件、PDF 等文档中直接抽取信息。
*   **UIE-senta**：专门针对**情感分析**任务进行了增强。
*   **UIE-M**：支持**多语言**（如中英文混合）的抽取。

总的来说，飞桨 UIE 是一个开箱即用的“万能”信息抽取工具，特别适合需要快速从各种文本（如合同、公告、报告、评论）中提取关键信息，但又缺乏充足标注数据的场景。

# 飞桨 UIE · GPU 训练环境部署操作（再装指导）

| 项目 | 说明 |
| ---- | ---- |
| 版本 | v1.0 |
| 日期 | 2026-09-03 |
| 状态 | **已在 GPU 上冒烟通过**；`/tmp` 清理后按 **§8.0**（Windows 备份环境包 → 上传解压 → 再传代码 → 训练）恢复 |
| 适用场景 | 无 sudo、无 Docker、有 NVIDIA GPU；Miniconda 装在 `/tmp`；**home 留给模型服务，环境包备份到 Windows** |
| 代码 | `search-assist/ml/uie-train/` |
| 对照 | CPU（60149 Docker）见 `飞桨UIE-训练平台-Docker部署操作.md`、`UIE训练环境-60149实验记录.md` |
| 概念 | `UIE训练与推理-知识记录.md`；**日常训推与 MVP 一站式**见 `UIE模型操作手册.md`；推理部署见 `飞桨UIE-生产推理部署方案.md` |


> **补丁合并说明（2026-10-17）**：已并入 transcript 回放时未命中的 StrReplace（写检测→`$HOME/uie`、§8.0.2 恢复顺序、§11 开训清单含五字段/ls300）。§4 安装步骤里若仍见 `/tmp/uie`，属 09-03 首次装机实录；**日常请以 `$HOME/uie` 为准**。

> 本文记录 **可复现的安装步骤与踩坑**，用于同一类 GPU 环境再次搭建或新同事上手。  
> **不要**把 60149 的 CPU Docker 流程原样搬到本环境（本环境通常无 Docker、无 root）。

---

## 1. 实测环境画像（2026-09-03）

| 项 | 值 |
| -- | -- |
| 入口 | 浏览器 **code-server**（远程 VS Code） |
| 形态 | K8s/平台上的 **GPU 开发 Pod**（主机名形如容器 ID） |
| OS | Ubuntu 24.04（容器内） |
| GPU | NVIDIA GeForce **GTX 1080 Ti**（11GB，Compute Capability **6.1** / Pascal） |
| 驱动 | 主机 Driver 约 580.x；`nvidia-smi` 显示 CUDA Version **13.0**（驱动上限） |
| 用户 | `abc`，**无免密 sudo**；有 `apt-get` 但不能装系统包 |
| Docker | **无** |
| 系统 Python | **无** |
| 可写大盘 | `/` 根目录通常不可写；**`$HOME`=`/config`** 挂大盘（实测约 3.6T，可用约 2T 级） |
| Home | `HOME=/config`；**环境与训练统一放 `$HOME/uie`**；重连后标记文件 **PERSIST_OK**（2026-09-08） |
| `/tmp` | 可写但 **code-server 重连可能清空** → 禁止再作为 UIE_ROOT |
| 持久性 | 以 `$HOME/uie` 为准；重要模型仍建议再备份到 Windows 或服务目录 |

验收时 `nvidia-smi` 正常、`docker` / `python3` 不存在，即与本文同一类环境。

---

## 2. 软件与版本清单（锁定）

| 组件 | 版本 / 说明 |
| ---- | ----------- |
| Miniconda | 用户目录安装；实测 `conda 26.7.1` |
| Python 环境名 | `uie` |
| Python | **3.10** |
| PaddlePaddle | **`paddlepaddle-gpu==2.6.2`**（官方 cu118 源） |
| Paddle Runtime CUDA | **11.8**（与 wheel 一致；勿用 CUDA 12 的 cublas 硬凑） |
| cuDNN | **8.9.x**（conda-forge / nvidia 频道） |
| PaddleNLP | **2.8.1**（实测显示 `2.8.1.post` 可接受） |
| 其它 pip | `aistudio-sdk==0.1.7`、`numpy&lt;2`、`pandas==2.0.3`、`pyarrow==12.0.1`、`datasets==2.18.0`、`tqdm`、`PyMySQL` |
| 易坏依赖钉死 | `charset-normalizer==3.3.2`（过新易 zlib 报错） |
| 冒烟基座 | **`uie-medium`**（比 CPU 的 mini/base 更大；OOM 可改 `uie-base`） |
| 经典 UIE 栈 | paddle **2.6** + paddlenlp **2.8**；**不是** PP-UIE-0.5B / paddle 3 |

---

## 3. 目录约定

一律使用大盘路径（示例）：

```text
/tmp/uie/                          # UIE_ROOT
  miniconda/                       # Miniconda 根
  ldlibs/                          # 仅软链 gomp/cudnn/cublas11/cudart11（见 §5）
  repo/uie-train/                  # 训练代码（从本机同步）
  datasets/                        # 训练数据
  models/                          # checkpoint / export
  cache/                           # Paddle / HF 缓存
  logs/
```

```bash
export UIE_ROOT=/tmp/uie
```

IDE 可：`ln -sfn /tmp/uie ~/workspace/uie`，但**实体文件不要写在 `~/`**。

---

## 4. 一键再装：环境部分

### 4.1 前置检查

```bash
nvidia-smi
nvidia-smi -L
cat /etc/os-release | head -5
whoami
sudo -n true && echo HAS_SUDO || echo NO_SUDO
df -h / /tmp /config ~
mkdir -p $HOME/uie && touch $HOME/uie/.write_test && echo HOME_UIE_OK && rm $HOME/uie/.write_test
which docker python python3 2>/dev/null || true
```

期望：GPU 可见、`HOME_UIE_OK`、`NO_SUDO`、无系统 Python/Docker。
（早期文档曾写 `/tmp` 可写检测；持久根迁到 `$HOME/uie` 后以本检查为准。）

### 4.2 安装 Miniconda + 建环境

```bash
export UIE_ROOT=/tmp/uie
mkdir -p "$UIE_ROOT"/{miniconda,repo,models,datasets,cache,logs,ldlibs}

curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
# 若目录已存在但不完整：rm -rf "$UIE_ROOT/miniconda"
bash /tmp/miniconda.sh -b -p "$UIE_ROOT/miniconda"

source "$UIE_ROOT/miniconda/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -y -n uie python=3.10
conda activate uie
python -V   # 期望 3.10.x
```

写入 `~/.bashrc`（可选，home 很小只写几行配置即可）：

```bash
cat >> ~/.bashrc <<'EOF'
export UIE_ROOT=/tmp/uie
[ -f "$UIE_ROOT/miniconda/etc/profile.d/conda.sh" ] && source "$UIE_ROOT/miniconda/etc/profile.d/conda.sh"
# conda activate uie   # 需要时再开，避免非交互脚本被打扰
EOF
```

### 4.3 安装 Paddle GPU + 依赖

```bash
source "$UIE_ROOT/miniconda/etc/profile.d/conda.sh"
conda activate uie
pip install -U pip

# ★ GPU 版，走飞桨 cu118 源（勿装 CPU 的 paddlepaddle）
python -m pip install paddlepaddle-gpu==2.6.2 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

pip install paddlenlp==2.8.1 aistudio-sdk==0.1.7 \
  "numpy>=1.21,<2" pandas==2.0.3 pyarrow==12.0.1 tqdm \
  "PyMySQL>=1.0" datasets==2.18.0 \
  "charset-normalizer==3.3.2"
```

### 4.4 系统库替代品（无 sudo）

```bash
conda activate uie
conda install -y -c conda-forge libgomp
conda install -y -c conda-forge cudnn=8.9
# 与 Paddle cu118 对齐的 CUDA 11 运行库（不要只用 CUDA 12 的 cublas）
conda install -y -c nvidia "cuda-version=11.8" "cuda-cudart=11.8.*" "libcublas=11.11.*"
```

若 conda 解依赖困难，可改用：

```bash
pip install --no-cache-dir nvidia-cublas-cu11 nvidia-cuda-runtime-cu11
# 再把 site-packages/nvidia/*/lib 下的 .so 链到 ldlibs（见下节）
```

---

## 5. LD_LIBRARY_PATH 约定（必读）

### 5.1 原则

| 做法 | 结果 |
| ---- | ---- |
| `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib`（整库） | 易触发 `charset_normalizer` / `yarl` 的 **`zlib.error: inconsistent stream state`** |
| 不设任何路径 | 缺 `libgomp` / `libcudnn` / `libcublas` |
| **只链需要的 .so 到 `/tmp/uie/ldlibs`** | 实测可行 |

**禁止**在本环境使用 60149 Docker 那套 `LD_PRELOAD=.../libz.so.1`。

### 5.2 重建 ldlibs

```bash
source "$UIE_ROOT/miniconda/etc/profile.d/conda.sh"
conda activate uie

rm -rf /tmp/uie/ldlibs
mkdir -p /tmp/uie/ldlibs

ln -sfn "$CONDA_PREFIX/lib"/libgomp.so* /tmp/uie/ldlibs/
ln -sfn "$CONDA_PREFIX/lib"/libcudnn*.so* /tmp/uie/ldlibs/

# 只要 .11，不要链 .12
ln -sfn "$CONDA_PREFIX/lib"/libcublas.so.11* /tmp/uie/ldlibs/
ln -sfn "$CONDA_PREFIX/lib"/libcublasLt.so.11* /tmp/uie/ldlibs/ 2>/dev/null || true
ln -sfn "$CONDA_PREFIX/lib"/libcudart.so.11* /tmp/uie/ldlibs/

# 若装在 targets 下
T="$CONDA_PREFIX/targets/x86_64-linux/lib"
if [ -d "$T" ]; then
  ln -sfn "$T"/libcublas.so.11* /tmp/uie/ldlibs/ 2>/dev/null || true
  ln -sfn "$T"/libcublasLt.so.11* /tmp/uie/ldlibs/ 2>/dev/null || true
  ln -sfn "$T"/libcudart.so.11* /tmp/uie/ldlibs/ 2>/dev/null || true
fi

cd /tmp/uie/ldlibs
ln -sfn "$(ls libcublas.so.11* | head -1)" libcublas.so
ln -sfn "$(ls libcudart.so.11* | head -1)" libcudart.so
ln -sfn "$(ls libcublasLt.so.11* 2>/dev/null | head -1)" libcublasLt.so 2>/dev/null || true

export LD_LIBRARY_PATH=/tmp/uie/ldlibs
unset LD_PRELOAD
ls -la /tmp/uie/ldlibs/
```

激活环境时自动带上：

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" <<'EOF'
export LD_LIBRARY_PATH=/tmp/uie/ldlibs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
unset LD_PRELOAD
EOF
```

### 5.3 环境自检（再装后必跑）

```bash
source "$UIE_ROOT/miniconda/etc/profile.d/conda.sh"
conda activate uie
export LD_LIBRARY_PATH=/tmp/uie/ldlibs
unset LD_PRELOAD

python -c "import charset_normalizer, yarl, requests, paddlenlp; print('IMPORT_OK', paddlenlp.__version__)"

python - <<'PY'
import paddle
import paddle.nn.functional as F
print("paddle", paddle.__version__, "cuda", paddle.is_compiled_with_cuda())
paddle.device.set_device("gpu:0")
x = paddle.randn([4, 8]); w = paddle.randn([8, 3])
y = F.linear(x, w)
print(y.shape, y.place)
print("PADDLE_GPU_OK")
PY
```

期望：`IMPORT_OK`、`compiled_cuda True`、`PADDLE_GPU_OK`；日志中 Runtime API **11.8**、cuDNN **8.9**、Capability **6.1**。

---

## 6. 同步代码与 GPU 训练配置

### 6.1 代码

将本机 `search-assist/ml/uie-train` 同步到：

```text
/tmp/uie/repo/uie-train
```

方式：code-server 拖拽到 `/tmp/uie/repo`、先打 `tar.gz` 再上传，或 scp/git。  
**不要**「打开本地 D: 盘」——code-server 只能打开**远程**目录。

确认：

```bash
ls /tmp/uie/repo/uie-train/scripts/02_train_uie.py
ls /tmp/uie/repo/uie-train/vendor/uie/finetune.py
```

### 6.2 冒烟配置示例（`uie-medium`）

```bash
cat > /tmp/uie/repo/uie-train/configs/train_gpu_medium.json <<'EOF'
{
  "data_type": "case",
  "dataset_version": "case/smoke-v0.1.0",
  "pretrained_model": "uie-medium",
  "device": "gpu",
  "max_seq_length": 512,
  "learning_rate": 1e-5,
  "num_epochs": 2,
  "batch_size": 2,
  "seed": 42,
  "logging_steps": 5,
  "save_steps": 50,
  "eval_steps": 50,
  "schema": ["审理法院", "案件类型", "审理法官"],
  "notes": [
    "GPU Pod / GTX 1080 Ti 11GB",
    "OOM 则改 uie-base 且 batch_size=1"
  ]
}
EOF
```

---

## 7. 冒烟训练与推理（合成数据）

每次新开终端：

```bash
export UIE_ROOT=/tmp/uie
source "$UIE_ROOT/miniconda/etc/profile.d/conda.sh"
conda activate uie
export LD_LIBRARY_PATH=/tmp/uie/ldlibs
unset LD_PRELOAD
export PADDLE_HOME=/tmp/uie/cache

cd /tmp/uie/repo/uie-train
mkdir -p "$UIE_ROOT"/{datasets,models,cache,logs}

python scripts/00_gen_synthetic_data.py --dataset-version case/smoke-v0.1.0
python scripts/02_train_uie.py --config configs/train_gpu_medium.json
python scripts/05_infer_checkpoint.py
```

成功标志：

- `DATA_OK` → `TRAIN_OK` → `INFER_OK`
- 产物：`$UIE_ROOT/models/case/case-smoke-*/checkpoint/export/`
- 推理样例能抽出「审理法院 / 案件类型 / 审理法官」

实测参考（2026-09-03）：

- run：`case-smoke-20260903-191702`
- 训练约 **16s** / 2 epoch（合成短文本）；约 **14.9 samples/s**
- eval F1 ≈ **0.89**（合成数据，仅链路验收）

`WARN: 60149 推荐 uie-mini` 可忽略（脚本旧提示）。

> **有真实标注数据时：可跳过本节合成冒烟，直接做 §7.1。**  
> 第 6 步「上传代码」仍需要；第 6 步里生成 `train_gpu_base.json`、第 7 步合成数据**不必再跑**。

---

## 7.1 真实五字段训练（case/real-v0.1.0）→ 推理 → MVP 挂载

适用：环境已按 §0.1 装到 `$HOME/uie`，代码已在 `$HOME/uie/repo/uie-train`，**跳过合成第 7 步**，用 `sample-data` 规则标注产物训练。

### Schema

`审理法院` · `审理程序` · `审理法官` · `当事人` · `代理人`

| 配置 / 脚本 | 说明 |
| ----------- | ---- |
| `configs/schema_case_real.json` | 推理 schema |
| `configs/train_gpu_real.json` | `uie-base` / `device=gpu` / `dataset_version=case/real-v0.1.0` |
| `scripts/01_annotate_sample_cases.py` | 本机规则标注 → `train.txt`/`dev.txt`（弱监督初稿） |
| `scripts/02_train_uie.py` | 微调 |
| `scripts/05_infer_checkpoint.py` | 训后单条推理验收 |
| `mvp_worker/` | 对外 MVP HTTP（Taskflow + export） |

### A. 本机准备数据（Windows 工作区）

```text
# 样例正文
D:\fabao\label\sample-data\*.txt

# 生成（已跑过可跳过）
cd search-assist/ml/uie-train
python scripts/01_annotate_sample_cases.py --sample-dir D:\fabao\label\sample-data --dataset-version case/real-v0.1.0
```

产物：

```text
ml/uie-train/data/dataset_versions/case/real-v0.1.0/
  train.txt
  dev.txt
  test.txt
  manifest.json
```

代理人约定：有律所 → `姓名#律师事务所`；非律所单位 → **只保留人名**。

### B. 拷到 GPU 机（路径必须对齐）

把本机 **`dataset_versions` 整目录** 放到 `$UIE_ROOT/datasets/` 下：

```text
本机:  .../uie-train/data/dataset_versions/case/real-v0.1.0/
服务器:
  $UIE_ROOT/datasets/dataset_versions/case/real-v0.1.0/train.txt
  $UIE_ROOT/datasets/dataset_versions/case/real-v0.1.0/dev.txt
```

```bash
# 自检
ls $HOME/uie/datasets/dataset_versions/case/real-v0.1.0/train.txt
ls $HOME/uie/datasets/dataset_versions/case/real-v0.1.0/dev.txt
```

> 不要拷成 `$UIE_ROOT/datasets/case/...`（少一层 `dataset_versions`），`02_train_uie.py` 会找不到数据。

### C. 开训前检查

```bash
export UIE_ROOT=$HOME/uie
source "$UIE_ROOT/miniconda/etc/profile.d/conda.sh"
conda activate uie
export LD_LIBRARY_PATH=$HOME/uie/ldlibs
unset LD_PRELOAD
export PADDLE_HOME=$HOME/uie/cache

nvidia-smi
# 空闲显存建议 ≥ 8～10GB；若两卡几乎满且 Processes 为空，见 §10「残留显存」
python -c "import paddle; print(paddle.__version__, paddle.is_compiled_with_cuda(), paddle.device.get_device())"
```

MVP Worker 若已用 `--no-load` 启动，一般不占大块显存；若已加载模型，开训前先停掉或保持不加载。

显存紧张时：把 `configs/train_gpu_real.json` 的 `batch_size` 改为 `1` 或 `2` 再训。

### D. 训练

```bash
cd $HOME/uie/repo/uie-train
python scripts/02_train_uie.py --config configs/train_gpu_real.json
```

成功标志：`TRAIN_OK`；产物在：

```text
$UIE_ROOT/models/case/<run_id>/checkpoint/export/
```

### E. 训后推理验收（替代合成第 7 步里的 infer）

```bash
cd $HOME/uie/repo/uie-train
python scripts/05_infer_checkpoint.py \
  --schema-config configs/schema_case_real.json \
  --text "河北省保定市易县人民法院执行裁定书，审判长贾晓东。"
# 期望：INFER_OK，并能看到字段 span
```

### F. 挂到 MVP Worker

```bash
mkdir -p $HOME/uie/serving/models/current
rm -rf $HOME/uie/serving/models/current/export
cp -a $HOME/uie/models/case/<run_id>/checkpoint/export \
      $HOME/uie/serving/models/current/export

# 停掉旧的 --no-load 进程后：
cd $HOME/uie/repo/uie-train
export UIE_MODEL_PATH=$HOME/uie/serving/models/current/export
python -m mvp_worker --config configs/mvp_worker.json
```

```bash
curl -s http://127.0.0.1:8088/health
curl -s http://127.0.0.1:8088/extract -H 'Content-Type: application/json' \
  -d '{"text":"河北省保定市易县人民法院执行裁定书，审判长贾晓东。"}'
```

**与第 6/7 步关系：**

| 步骤 | 真实训练路径 |
| ---- | ------------ |
| 第 6 步上传代码 | 需要（含 `train_gpu_real.json` / `mvp_worker`） |
| 第 6 步写 `train_gpu_base.json` | **不必** |
| 第 7 步合成数据+冒烟训 | **不必**；用本节 D/E 代替 |

---

## 8. 备份与持久化

> **默认**：环境与训练在 **`$HOME/uie`**（重连后一般仍在）。  
> **无外网或 `$HOME/uie` 被误删**时：用 Windows 上的环境 tar 按 §8.0 恢复到 **`$HOME/uie`**。  
> 模型服务可用 `$HOME` 下其它目录存放 export 副本；与「环境是否备份到 Windows」分开。

### 8.0 运行环境：打包到 Windows → 恢复到 `$HOME/uie`（无外网备用）

**标准流程：**

```text
上传环境包 → 解压到 $HOME/uie / 激活 / 自检 OK → 上传 uie-train 代码 → 训练
```

（有外网日常优先 §0.1 七步；本节给无外网或快速还原用。）

环境包与代码包、模型包**分开**存放：

| 包（建议文件名） | 内容 | 是否含 uie-train |
| ---------------- | ---- | ---------------- |
| `uie-miniconda.tar.gz` | `/tmp/uie/miniconda`（Python + Paddle 等） | 否 |
| `uie-ldlibs.tar.gz` | GPU 相关 `.so` 实体 | 否 |
| `uie-train-code.tar.gz`（可选） | `repo/uie-train` 代码 | 是（单独打） |
| `uie-*-smoke-*.tgz` 等 | 某次训练 `models/case/...` | 否（模型产物） |

#### 8.0.1 在 GPU 上打包（环境仍在时，纯 tar，不额外 pip 装工具）

```bash
export UIE_ROOT=/tmp/uie

# 1) Miniconda 整目录（含 uie 环境）
tar czf /tmp/uie-miniconda.tar.gz -C /tmp/uie miniconda

# 2) ldlibs：拷贝真实文件后再打包（不要只存断掉的软链）
mkdir -p /tmp/uie-ldlibs-backup
cd /tmp/uie/ldlibs
for f in *; do
  real=$(readlink -f "$f" 2>/dev/null || true)
  if [ -n "$real" ] && [ -f "$real" ]; then
    cp -a "$real" /tmp/uie-ldlibs-backup/
  elif [ -f "$f" ]; then
    cp -a "$f" /tmp/uie-ldlibs-backup/
  fi
done
tar czf /tmp/uie-ldlibs.tar.gz -C /tmp uie-ldlibs-backup

# 3) 可选：代码一并备份
# tar czf /tmp/uie-train-code.tar.gz -C /tmp/uie/repo uie-train

ls -lh /tmp/uie-miniconda.tar.gz /tmp/uie-ldlibs.tar.gz
```

用 code-server 将上述文件**下载到 Windows** 固定目录保存。  
**不要**占用 home 存这套环境包。

#### 8.0.2 无外网或环境损坏时：恢复顺序

> 早期（2026-09-03）环境曾在 `/tmp/uie`；**现正式根为 `$HOME/uie`（多为 `/config/uie`）**。下列按现约定恢复。

**① 上传环境包**到 GPU（可放到 `/tmp` 中转或 `$HOME/`）：

- `uie-miniconda.tar.gz`
- `uie-ldlibs.tar.gz`

**② 解压到 `$HOME/uie` / 激活 / 自检**（须出现 `OK` 再往下）：

```bash
export UIE_ROOT=$HOME/uie
mkdir -p "$UIE_ROOT"
# 若包在 /tmp：
tar -xzf /tmp/uie-miniconda.tar.gz -C "$UIE_ROOT"
tar -xzf /tmp/uie-ldlibs.tar.gz -C /tmp
cp -a /tmp/uie-ldlibs-backup/. "$UIE_ROOT/ldlibs/" 2>/dev/null || \
  cp -a $HOME/uie-ldlibs-backup/. "$UIE_ROOT/ldlibs/"

# 若包在 $HOME：把上面 /tmp/uie-*.tar.gz 改成 $HOME/uie-*.tar.gz
mkdir -p "$UIE_ROOT/ldlibs"

source "$UIE_ROOT/miniconda/etc/profile.d/conda.sh"
conda activate uie
export LD_LIBRARY_PATH=$HOME/uie/ldlibs
unset LD_PRELOAD

python -c "import paddle, paddlenlp; print('OK', paddle.__version__, paddle.is_compiled_with_cuda())"
```

**③ 上传 uie-train 代码**到 `$HOME/uie/repo/uie-train`：

```bash
mkdir -p $HOME/uie/repo
# tar -xzf /tmp/uie-train-code.tar.gz -C $HOME/uie/repo
ls $HOME/uie/repo/uie-train/scripts/02_train_uie.py
ls $HOME/uie/repo/uie-train/vendor/uie/finetune.py
```

**④ 训练**：见 §11，`UIE_ROOT=$HOME/uie`。

#### 8.0.3 时间线（备忘）

```text
【日常】环境在 $HOME/uie，重连后一般仍在；有外网装/重装用 §0.1
【无外网 / 目录被删】上传 tar → 解压到 $HOME/uie → 自检 → 上传代码 → 训练
【模型服务】export 可另拷到 $HOME 下服务目录
```

---

### 8.1 训练产物压缩包（模型 run，与环境包分开）

```bash
cd /tmp/uie/models/case
tar czf /tmp/uie-base-smoke-$(date +%Y%m%d).tgz case-smoke-*
# 下载到本机；home 若用于服务，可将 export 部署到 home 约定路径
# 实测包名示例：uie-medium-smoke-20260903.tgz（约 2.5G，含一次完整 run）
```

若日后另有专用数据盘，可将 `UIE_ROOT` 改为该盘路径。

### 8.2 压缩包是什么、里面目录结构

`uie-medium-smoke-20260903.tgz` / `uie-base-smoke-*.tgz` = 一次训练 run 的目录打包，对应例如：

`case-smoke-20260903-191702/`（或当日 `uie-base` 冒烟 run）。

**用途**：备份、换机恢复、联调加载 `export` 做推理。  
**不是**：正式长文本/新字段生产模型（需真实标注重训后再打一包）。

典型结构（`tar -tzf xxx.tgz | head` 可见）：

```text
case-smoke-YYYYMMDD-HHMMSS/
├── train_config.json              # 本次训练配置快照
├── train.log                      # 训练全过程日志
├── RUN_OK                         # 成功标记（多为 run_id 文本）
└── checkpoint/                    # ★ 主产物根
    ├── model_state.pdparams       # 微调结束后的完整权重（最大之一）
    ├── config.json                # 网络结构配置
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── vocab.txt                  # 分词词表
    ├── trainer_state.json         # step / loss 等状态（若有）
    ├── training_args.bin          # 训练超参快照（若有）
    ├── train_results.json / all_results.json   # 指标摘要（若有）
    ├── checkpoint-<step>/         # 训练过程中按 save_steps 存的中间快照（可选）
    │   └── model_state.pdparams   # 某一 step 的权重副本
    └── export/                    # ★ 推理优先加载这里（Taskflow）
        ├── model_state.pdparams   # 导出用的权重（常与上面同量级）
        ├── config.json
        ├── tokenizer_* / vocab.txt / special_tokens_map.json
        ├── model.pdmodel 等       # 导出图相关（随版本略有差异）
        └── static/                # 跑过 05_infer 后可能生成的静态推理图
            └── inference.*
```

查看本地已下载包：

```bash
tar -tzf uie-medium-smoke-20260903.tgz | head -80
```

### 8.3 `.pdparams` 是什么？为什么有好几个、都很大？

| 问题 | 说明 |
| ---- | ---- |
| **后缀含义** | Paddle 的 **模型参数文件**（Paddle Parameters）：保存网络里所有可训练权重（浮点参数矩阵等） |
| **为什么几百兆** | `uie-medium` / `uie-base` 参数量大；权重以二进制浮点存放，单份完整权重通常就 **数百 MB～1GB+** 量级 |
| **为什么不止一个** | 同一套参数会在不同阶段各存一份，**内容同类、角色不同**，不是多种「不同算法」 |

常见几份分别是：

| 路径 | 角色 | 日常要不要留 |
| ---- | ---- | ------------ |
| `checkpoint/model_state.pdparams` | 训练结束时的**最终权重**（继续训 / Trainer 加载） | 建议保留 |
| `checkpoint/export/model_state.pdparams` | **导出给推理**用的权重（Taskflow / 生产加载优先看 `export/`） | **必须保留** |
| `checkpoint/checkpoint-<N>/model_state.pdparams` | 训练中途按 step 存的**中间备份** | 冒烟后可删以减小 tgz |
| （若有）`static/` 旁其它大文件 | 静态图/推理引擎相关，不是 `.pdparams` 但也会占空间 | 推理需要则保留 |

记忆口诀：

- **`.pdparams` ≈「模型的大脑参数快照」**（不是日志、不是配置 JSON）  
- **抽字段** → 用 `export/` 目录（里面的 pdparams + config + tokenizer）  
- **包体积大** → 多半是多份权重叠在一起；精简备份可只打 `export/` + `train_config.json` + `RUN_OK`

精简打包示例（仅推理必需）：

```bash
RUN=case-smoke-20260903-191702
tar czf /tmp/${RUN}-export-only.tgz \
  -C /tmp/uie/models/case \
  $RUN/train_config.json $RUN/RUN_OK $RUN/checkpoint/export
```

---

## 9. 换真实长文本 / 其它字段（概要）

五字段真实训练的完整步骤见 **§7.1**。一般换字段时：

`uie-train` **可复用**；不要再跑 `00_gen_synthetic_data.py`。

1. 数据做成 UIE JSONL：`content` + `prompt`（字段名）+ `result_list[{text,start,end}]`  
2. 放到 `$UIE_ROOT/datasets/dataset_versions/<version>/{train,dev}.txt`  
3. 新建配置：改 `dataset_version`、`schema`、`max_seq_length`（长文先 512）、`batch_size`（先 1）  
4. 仍执行 `02_train_uie.py` → `05_infer_checkpoint.py`  

超长正文由官方 `reader` 按 `max_seq_length` 切段；生产批量推理另见《生产推理部署方案》滑窗合并。

---

## 10. 故障速查

| 现象 | 处理 |
| ---- | ---- |
| `libgomp.so.1: cannot open...` | `conda install -c conda-forge libgomp`，并链入 `ldlibs` |
| `Cannot load cudnn` / `libcudnn.so` | `conda install cudnn=8.9`，链入 `ldlibs` |
| `libcublas.so: cannot open...` | 安装 **CUDA 11** 的 `libcublas`/`cudart`，链 **`.11`**；仅有 `.12` 不够（Paddle cu118） |
| `zlib.error: inconsistent stream state`（charset/yarl） | **不要**把整个 `$CONDA_PREFIX/lib` 放进 `LD_LIBRARY_PATH`；钉 `charset-normalizer==3.3.2`；必要时 `pip install --force-reinstall yarl multidict aiohttp` |
| Miniconda 目录已存在但不完整 | `rm -rf $UIE_ROOT/miniconda` 后重装；或 `bash miniconda.sh -u -b -p ...` |
| Conda ToS 报错 | `conda tos accept ... pkgs/main` 与 `pkgs/r` |
| OOM / 显存不足 | `uie-base`、`batch_size=1`、`max_seq_length=512` |
| **残留显存：Memory 很高、Util=0%、Processes 为空** | **不能“抢占”已占用显存**。空闲只有剩余 MiB；需宿主机查占用进程、重启 GPU Pod/机器释放。详见下方说明 |
| 推理日志下载 uie_base | Taskflow 可能拉缓存；确认随后从本地 `export` 加载且 `INFER_OK` |
| 环境消失 | 确认是否误用 `/tmp`；有外网走 §0.1，无外网走 §8.0 恢复到 `$HOME/uie` |
| `python: command not found` | 先 `source .../conda.sh && conda activate uie`；用环境内 python |

### 残留显存说明（为何 Processes 为空也不能硬训）

```text
nvidia-smi 常见矛盾现象：
  Memory-Usage ≈ 19～22GB / 24GB
  GPU-Util = 0%
  Processes = No running processes found
```

- 显存是**按分配占用**的，不是“没进程就算空闲、后来者可抢”。  
- 本容器里看不到 PID，常见原因：占用在**宿主机 / 其它容器**，或进程已退出但驱动上下文未释放。  
- 此时空闲往往只有 **2～4GB**，不够 `uie-base` 正常训练；强行开训多半 **OOM**，并不会把别人的 20GB “挤掉”。  
- 处理：运维在宿主机 `nvidia-smi` 查杀占用 → 或 **重启本 GPU Pod** → 再训。  

---

## 11. 每次开训最短命令清单

```bash
export UIE_ROOT=$HOME/uie
source "$UIE_ROOT/miniconda/etc/profile.d/conda.sh"
conda activate uie
export LD_LIBRARY_PATH=$HOME/uie/ldlibs
unset LD_PRELOAD
export PADDLE_HOME=$HOME/uie/cache
cd $HOME/uie/repo/uie-train

# —— 合成冒烟（可选，可跳过）——
# python scripts/00_gen_synthetic_data.py --dataset-version case/smoke-v0.1.0
# python scripts/02_train_uie.py --config configs/train_gpu_medium.json
# python scripts/05_infer_checkpoint.py

# —— 真实五字段（历史，见 §7.1）——
# python scripts/02_train_uie.py --config configs/train_gpu_real.json
# python scripts/05_infer_checkpoint.py --schema-config configs/schema_case_real.json

# —— 四字段 ls300（当前正式）——
# python scripts/02_train_uie.py --config configs/train_gpu_ls300.json
# python scripts/05_infer_checkpoint.py --schema-config configs/schema_case_ls300.json
```

---

## 12. 与 CPU（60149）对照

| | 60149 CPU | 本文 GPU Pod |
| -- | --------- | ------------ |
| 运行时 | Docker `uie-train:cpu` | Miniconda `uie` 环境 |
| 权限 | 常有 root | 无 sudo |
| 设备 | `device=cpu` | `device=gpu` |
| 默认基座冒烟 | `uie-mini` | `uie-medium` |
| 数据根 | `/data/uie`（持久） | 现用 `/tmp/uie`（需备份） |
| Paddle | 2.6.2 CPU | 2.6.2 **GPU cu118** |
| LD_PRELOAD libz | Docker 内常用 | **禁止** |

同一套 `ml/uie-train` 代码；差异在**安装形态与配置**。

---

## 12.1 Harbor 镜像（推荐长期方案）

内网 Harbor（项目 **`paddle`**）：

| 项 | 值 |
| -- | -- |
| Web / Registry | `http://192.168.60.137:8888`（HTTP，按需配 insecure-registry） |
| 项目 | `paddle`（地址形如 `http://192.168.60.137:8888/paddle`） |
| 建议镜像名 | `192.168.60.137:8888/paddle/uie-train:gpu-cu118-<日期>` |
| 已有基础镜像参考 | `192.168.60.137:8888/library/python:3.10-slim-bullseye`（CPU 构建曾用） |

**镜像里放什么 / 不放什么：**

| 放进镜像 | 挂到 PVC（如 `/data/uie`） |
| -------- | -------------------------- |
| Python + Paddle GPU cu118 + PaddleNLP + 系统库 | `datasets/`、`models/`、`cache/`、各次 run |
| `uie-train` 代码（或启动时挂载最新代码） | 正式/冒烟 checkpoint、export |

**在有 Docker 的机器上构建并推送：**

```bash
cd /path/to/uie-train

# 1) 登录（用户名密码问运维）
docker login 192.168.60.137:8888

# 2) 构建（BASE_IMAGE 优先用 Harbor 里已有的 CUDA 11.8+cudnn 镜像；没有再用 Docker Hub）
docker build -f Dockerfile.gpu \
  --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 \
  -t 192.168.60.137:8888/paddle/uie-train:gpu-cu118-20260908 \
  .

# 3) 推送
docker push 192.168.60.137:8888/paddle/uie-train:gpu-cu118-20260908
```

**GPU 节点 / Job 使用示例：**

```bash
docker pull 192.168.60.137:8888/paddle/uie-train:gpu-cu118-20260908

docker run --rm --gpus all \
  -e UIE_ROOT=/data/uie \
  -e UIE_DEVICE=gpu \
  -v /data/uie:/data/uie \
  192.168.60.137:8888/paddle/uie-train:gpu-cu118-20260908 \
  python scripts/02_train_uie.py --config configs/train_gpu_base.json
```

**code-server 开发 Pod：** 请运维把工作负载镜像换成上述 tag，并挂载 PVC 到 `/data/uie`；之后 **`export UIE_ROOT=/data/uie`**，不再往 `/tmp` 装 Miniconda。

相关文件：`ml/uie-train/Dockerfile.gpu`、`requirements-gpu.txt`。

---

## 13. 文档关系

| 文档 | 职责 |
| ---- | ---- |
| **本文** | GPU Pod 再装与冒烟操作 |
| `飞桨UIE-训练平台-Docker部署操作.md` | 60149 CPU Docker |
| `UIE训练平台-CPU冒烟实施方案.md` | CPU 冒烟验收边界 |
| `飞桨UIE-生产推理部署方案.md` | 训后批量推理部署 |
| `UIE训练与推理-知识记录.md` | 概念与产物说明 |

---

**再装顺序**：§4 环境 → §5 ldlibs 自检 → §6 代码 → §7 冒烟 → §8 备份。  
正式业务数据训练见 §9，不替代 §7 的链路验收（建议先冒烟再上真实数据）。
