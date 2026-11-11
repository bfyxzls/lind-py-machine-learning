"""销售诊断项目共用：读数、中文字体、低毛利标签。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "supermarket_sales.xlsx"
OUT_DIR = ROOT / "output"

# 毛利率低于该阈值 → 判定为「低毛利风险单」（分类标签）
LOW_MARGIN_THRESHOLD = 0.15


def setup_chinese_font() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def load_sales() -> pd.DataFrame:
    if not DATA_FILE.exists():
        from mock_sales_excel import main as gen

        print("未找到 Excel，先自动生成 mock 数据...")
        gen()
    df = pd.read_excel(DATA_FILE)
    df["销售日期"] = pd.to_datetime(df["销售日期"])
    df["毛利率"] = df["毛利润"] / df["销售额"].replace(0, np.nan)
    df = df.dropna(subset=["毛利率"]).copy()
    # 1 = 低毛利风险，0 = 正常/健康毛利
    df["低毛利"] = (df["毛利率"] < LOW_MARGIN_THRESHOLD).astype(int)
    return df


FEATURE_CAT = ["品类", "渠道", "门店"]
FEATURE_NUM = ["折扣率", "进货成本单价", "成交单价", "销量"]
