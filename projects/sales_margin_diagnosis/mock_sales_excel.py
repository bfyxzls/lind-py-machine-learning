"""生成一份「销售额上升、毛利润下降」的超市销售 mock Excel。

刻意埋入三类问题，供诊断脚本发现：
1. 品类结构恶化：低毛利「促销品」销量暴增，高毛利「生鲜」占比下降
2. 折扣加深：整体折扣率逐月上升
3. 成本上涨：部分品类进货成本上升，售价未同步上调
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_FILE = DATA_DIR / "supermarket_sales.xlsx"

RNG = np.random.default_rng(42)

# 品类基准：售价、成本、初始月销量权重
CATEGORIES = {
    "生鲜": {"price": 28.0, "cost": 16.0, "base_qty": 800, "trend": -0.04},  # 高毛利，份额下滑
    "粮油": {"price": 45.0, "cost": 36.0, "base_qty": 500, "trend": 0.01},
    "日化": {"price": 32.0, "cost": 22.0, "base_qty": 420, "trend": 0.02},
    "促销特价": {"price": 19.0, "cost": 17.5, "base_qty": 350, "trend": 0.14},  # 低毛利，暴增
    "烟酒": {"price": 120.0, "cost": 90.0, "base_qty": 180, "trend": -0.02},
}

STORES = ["城东店", "城西店", "城南店"]
CHANNELS = ["到店", "外卖", "团购"]


def build_records() -> pd.DataFrame:
    rows = []
    months = pd.period_range("2025-01", periods=12, freq="M")

    for mi, period in enumerate(months):
        month_idx = mi  # 0..11
        # 整体折扣：年初约 2%，年末约 12%
        base_discount = 0.02 + month_idx * 0.009

        for cat, meta in CATEGORIES.items():
            # 成本通胀：粮油/日化每月 +0.8%，生鲜 +0.3%，促销更狠 +1.2%
            cost_inflate = {
                "生鲜": 0.003,
                "粮油": 0.008,
                "日化": 0.008,
                "促销特价": 0.012,
                "烟酒": 0.004,
            }[cat]
            unit_cost = meta["cost"] * ((1 + cost_inflate) ** month_idx)
            # 售价几乎不动（只微调），导致毛利率被成本侵蚀
            unit_price = meta["price"] * (1 + month_idx * 0.001)

            # 整体客流/订单规模缓慢扩张，保证「销售额大体逐月上升」
            market_growth = (1.018) ** month_idx
            qty = meta["base_qty"] * ((1 + meta["trend"]) ** month_idx) * market_growth
            qty = max(50, int(RNG.normal(qty, qty * 0.03)))

            for store in STORES:
                for channel in CHANNELS:
                    # 团购折扣更深；外卖略深
                    channel_extra = {"到店": 0.0, "外卖": 0.02, "团购": 0.05}[channel]
                    discount = np.clip(base_discount + channel_extra + RNG.normal(0, 0.01), 0, 0.35)

                    # 门店销量拆分
                    share = 1 / len(STORES) / len(CHANNELS)
                    line_qty = max(1, int(qty * share * RNG.uniform(0.85, 1.15)))

                    list_amount = line_qty * unit_price
                    sales_amount = list_amount * (1 - discount)
                    cost_amount = line_qty * unit_cost
                    gross_profit = sales_amount - cost_amount

                    rows.append(
                        {
                            "销售日期": period.to_timestamp() + pd.Timedelta(days=int(RNG.integers(0, 27))),
                            "年月": str(period),
                            "门店": store,
                            "渠道": channel,
                            "品类": cat,
                            "SKU": f"{cat}-{RNG.integers(100, 999)}",
                            "销量": line_qty,
                            "标价单价": round(unit_price, 2),
                            "成交单价": round(unit_price * (1 - discount), 2),
                            "进货成本单价": round(unit_cost, 2),
                            "折扣率": round(discount, 4),
                            "销售额": round(sales_amount, 2),
                            "销售成本": round(cost_amount, 2),
                            "毛利润": round(gross_profit, 2),
                        }
                    )

    return pd.DataFrame(rows).sort_values("销售日期").reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = build_records()
    df.to_excel(OUT_FILE, index=False)
    monthly = df.groupby("年月", as_index=False).agg(销售额=("销售额", "sum"), 毛利润=("毛利润", "sum"))
    monthly["毛利率"] = monthly["毛利润"] / monthly["销售额"]
    print(f"已生成: {OUT_FILE}")
    print(f"记录数: {len(df)}")
    print("\n月度汇总预览:")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
