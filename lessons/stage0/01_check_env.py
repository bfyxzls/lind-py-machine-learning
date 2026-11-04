"""阶段 0 · 第 1 节：确认 Python 环境可用。"""

import sys

print("Python 版本:", sys.version)
print("可执行文件:", sys.executable)

# 若已安装依赖，下面会打印版本；未安装会报 ModuleNotFoundError
try:
    import numpy as np
    import pandas as pd
    import sklearn

    print("numpy:", np.__version__)
    print("pandas:", pd.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("环境检查通过 OK")
except ModuleNotFoundError as e:
    print("缺少依赖:", e.name)
    print("请先运行: pip install -r requirements.txt")
