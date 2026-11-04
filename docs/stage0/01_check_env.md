# 阶段 0 · 第 1 节：确认 Python 环境

Python 环境是跑通一切机器学习代码的前提：解释器负责执行脚本，虚拟环境（venv）负责把本项目的依赖与系统 Python 隔离开。你可以把它理解成「给这个学习项目单独开一间干净的工作台」——装哪些库、用哪个版本，都只影响这间工作台。环境没配好，后面 NumPy、Pandas、scikit-learn 再简单也会反复卡在 `命令找不到` 或 `ModuleNotFoundError`。

## 学习目标

- 确认本机 Python ≥ 3.10
- 会创建并激活虚拟环境
- 能安装并验证 `numpy` / `pandas` / `scikit-learn`

## 直觉（3 点）

1. 机器学习代码跑在 **Python 解释器** 里。
2. **虚拟环境** 把本项目依赖隔离，避免污染系统 Python。
3. 先确认环境，再学库，能少踩很多「装了但 import 不到」的坑。

## 依赖安装

```powershell
cd e:\github\lind-py-machine-learning
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

若激活报错「禁止运行脚本」：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 运行

```powershell
python lessons\stage0\01_check_env.py
```

## 预期输出（版本号可变）

```
Python 版本: 3.11.x ...
可执行文件: ...\.venv\Scripts\python.exe
numpy: ...
pandas: ...
scikit-learn: ...
环境检查通过 OK
```

注意：可执行文件路径应包含 `.venv`，说明用的是虚拟环境。

## 常见报错

| 现象 | 处理 |
|------|------|
| `python` 不是内部命令 | 安装 Python 并勾选 Add to PATH |
| `ModuleNotFoundError` | 激活 `.venv` 后再 `pip install -r requirements.txt` |
| 激活策略禁止 | 见上方 `Set-ExecutionPolicy` |

## 练习

1. 运行本脚本，确认输出里的路径是 `.venv` 下的 Python。
2. 执行：`python -c "print(3+5); print('hello ML'); print(len([1,2,3]))"`，理解交互式运行。

## 下一节

→ [02_numpy_array.md](./02_numpy_array.md)
