# Windows 一键安装 PaddleNLP（绕过 tool-helpers）
# 用法（在 projects/baidu_uie 且已激活 .venv）:
#   powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==> install runtime deps (skip Linux-only tool-helpers)"
python -m pip install -r requirements-windows.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host "==> pin huggingface-hub for paddlenlp 2.8.1"
python -m pip install "huggingface-hub>=0.19.2,<0.24" -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host "==> install paddlenlp==2.8.1 --no-deps"
python -m pip install "paddlenlp==2.8.1" --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple

Write-Host "==> verify"
python -c "import paddle, paddlenlp; from paddlenlp import Taskflow; print('OK', paddle.__version__, paddlenlp.__version__)"
python scripts\check_env.py
