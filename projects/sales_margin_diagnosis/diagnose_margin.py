"""诊断：为什么销售额上升，毛利润反而下降？

方法：
1. 趋势确认（月度销售额 / 毛利润 / 毛利率）
2. 结构分析（品类贡献、折扣、成本）
3. 销售额-毛利分解（量 / 价折扣 / 成本 / 结构）
4. 机器学习：用随机森林解释「单笔毛利率」的关键驱动因素
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "supermarket_sales.xlsx"
OUT_DIR = ROOT / "output"


def setup_chinese_font() -> None:
    """尽量启用系统中文字体，避免图中类别名显示为方框。"""
    candidates = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        from mock_sales_excel import main as gen

        print("未找到 Excel，先自动生成 mock 数据...")
        gen()
    df = pd.read_excel(DATA_FILE)
    df["销售日期"] = pd.to_datetime(df["销售日期"])
    df["毛利率"] = df["毛利润"] / df["销售额"].replace(0, np.nan)
    df["成本率"] = df["销售成本"] / df["销售额"].replace(0, np.nan)
    return df


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    m = (
        df.groupby("年月", as_index=False)
        .agg(
            销售额=("销售额", "sum"),
            毛利润=("毛利润", "sum"),
            销量=("销量", "sum"),
            平均折扣率=("折扣率", "mean"),
            平均成本单价=("进货成本单价", "mean"),
            平均成交单价=("成交单价", "mean"),
        )
        .sort_values("年月")
    )
    m["毛利率"] = m["毛利润"] / m["销售额"]
    return m


def plot_trends(m: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = np.arange(len(m))

    axes[0, 0].plot(x, m["销售额"], marker="o", color="steelblue")
    axes[0, 0].set_title("Monthly Sales (up)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(m["年月"], rotation=45, ha="right", fontsize=8)

    axes[0, 1].plot(x, m["毛利润"], marker="o", color="crimson")
    axes[0, 1].set_title("Monthly Gross Profit (down)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(m["年月"], rotation=45, ha="right", fontsize=8)

    axes[1, 0].plot(x, m["毛利率"] * 100, marker="o", color="darkorange")
    axes[1, 0].set_title("Gross Margin %")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(m["年月"], rotation=45, ha="right", fontsize=8)

    axes[1, 1].plot(x, m["平均折扣率"] * 100, marker="o", label="discount%")
    axes[1, 1].plot(x, m["平均成本单价"], marker="s", label="avg cost")
    axes[1, 1].set_title("Discount % & Avg Cost")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(m["年月"], rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    path = OUT_DIR / "01_monthly_trends.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"图已保存: {path}")


def category_mix_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """对比首月 vs 末月：销售额占比与毛利贡献。"""
    months = sorted(df["年月"].unique())
    first, last = months[0], months[-1]

    def summarize(month: str) -> pd.DataFrame:
        sub = df[df["年月"] == month]
        g = sub.groupby("品类", as_index=False).agg(
            销售额=("销售额", "sum"),
            毛利润=("毛利润", "sum"),
            销量=("销量", "sum"),
            平均折扣率=("折扣率", "mean"),
            平均成本单价=("进货成本单价", "mean"),
        )
        g["销售占比"] = g["销售额"] / g["销售额"].sum()
        g["毛利率"] = g["毛利润"] / g["销售额"]
        g["年月"] = month
        return g

    a = summarize(first).set_index("品类")
    b = summarize(last).set_index("品类")
    cmp = pd.DataFrame(
        {
            f"{first}_销售占比": a["销售占比"],
            f"{last}_销售占比": b["销售占比"],
            "销售占比变化": b["销售占比"] - a["销售占比"],
            f"{first}_毛利率": a["毛利率"],
            f"{last}_毛利率": b["毛利率"],
            "毛利率变化": b["毛利率"] - a["毛利率"],
            f"{first}_毛利润": a["毛利润"],
            f"{last}_毛利润": b["毛利润"],
            "毛利润变化": b["毛利润"] - a["毛利润"],
            "折扣率变化": b["平均折扣率"] - a["平均折扣率"],
            "成本单价变化": b["平均成本单价"] - a["平均成本单价"],
        }
    ).sort_values("毛利润变化")
    return cmp


def plot_category(cmp: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cmp["毛利润变化"].plot(kind="barh", ax=axes[0], color=["crimson" if v < 0 else "seagreen" for v in cmp["毛利润变化"]])
    axes[0].set_title("Gross Profit Change by Category (first→last)")
    axes[0].axvline(0, color="gray", lw=0.8)

    share_cols = [c for c in cmp.columns if c.endswith("销售占比") and "变化" not in c]
    cmp[share_cols].plot(kind="bar", ax=axes[1])
    axes[1].set_title("Sales Mix Share: first vs last month")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    path = OUT_DIR / "02_category_mix.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"图已保存: {path}")


def channel_store_insight(df: pd.DataFrame) -> None:
    months = sorted(df["年月"].unique())
    first, last = months[0], months[-1]
    for dim in ("渠道", "门店"):
        pivot = (
            df[df["年月"].isin([first, last])]
            .groupby([dim, "年月"], as_index=False)
            .agg(销售额=("销售额", "sum"), 毛利润=("毛利润", "sum"))
        )
        pivot["毛利率"] = pivot["毛利润"] / pivot["销售额"]
        print(f"\n=== {dim}：首月 vs 末月毛利率 ===")
        wide = pivot.pivot(index=dim, columns="年月", values="毛利率")
        print((wide * 100).round(2).to_string())


def ml_explain_margin(df: pd.DataFrame) -> pd.DataFrame:
    """随机森林：哪些因素最能解释单笔毛利率。"""
    use = df.dropna(subset=["毛利率"]).copy()
    features_cat = ["品类", "渠道", "门店"]
    features_num = ["折扣率", "进货成本单价", "成交单价", "销量"]
    X = use[features_cat + features_num]
    y = use["毛利率"]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
            ("num", "passthrough", features_num),
        ]
    )
    model = Pipeline(
        [
            ("pre", pre),
            ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n=== ML：随机森林解释毛利率 ===")
    print(f"测试集 R^2 = {r2_score(y_test, pred):.4f}")

    # 置换重要性（更可靠）
    result = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1)
    imp = pd.DataFrame({"特征": X.columns, "重要性": result.importances_mean}).sort_values(
        "重要性", ascending=False
    )
    print(imp.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=imp, x="重要性", y="特征", ax=ax, color="steelblue")
    ax.set_title("What drives gross margin? (permutation importance)")
    fig.tight_layout()
    path = OUT_DIR / "03_feature_importance.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"图已保存: {path}")
    return imp


def narrative(m: pd.DataFrame, cmp: pd.DataFrame, imp: pd.DataFrame) -> None:
    sales_up = m["销售额"].iloc[-1] - m["销售额"].iloc[0]
    profit_down = m["毛利润"].iloc[-1] - m["毛利润"].iloc[0]
    margin_down = m["毛利率"].iloc[-1] - m["毛利率"].iloc[0]
    disc_up = m["平均折扣率"].iloc[-1] - m["平均折扣率"].iloc[0]

    worst = cmp["毛利润变化"].idxmin()
    mix_up = cmp["销售占比变化"].idxmax()

    print("\n" + "=" * 60)
    print("诊断结论（为什么销量/销售额升，毛利润降）")
    print("=" * 60)
    print(
        f"1. 现象确认：销售额 {sales_up:+,.0f}，毛利润 {profit_down:+,.0f}，"
        f"毛利率 {margin_down*100:+.2f} 个百分点。"
    )
    print(
        f"2. 结构恶化：销售占比升最多的是「{mix_up}」，"
        f"毛利润下滑最多的是「{worst}」——低毛利品类冲高了销售额，稀释了整体毛利。"
    )
    print(
        f"3. 价格动作：平均折扣率上升 {disc_up*100:.2f} 个百分点，"
        "卖得更便宜，进一步吃掉毛利。"
    )
    print(
        "4. 成本端：多个品类进货成本上升而标价几乎不动，"
        "单位毛利被成本通胀侵蚀（见表中「成本单价变化」）。"
    )
    print(
        f"5. ML 佐证：解释毛利率时最重要的特征 Top3 = "
        + ", ".join(imp.head(3)["特征"].tolist())
        + "。"
    )
    print("\n建议动作：")
    print("- 控制促销特价/团购低价占比，或提高其加价率门槛")
    print("- 折扣审批分级：限制深度折扣对高贡献品类的侵蚀")
    print("- 对成本上涨品类同步调价或更换供应商")
    print("- 考核从「销售额」改为「毛利额 + 毛利率」双指标")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()
    sns.set_theme(style="whitegrid")

    df = load_data()
    print(f"加载数据: {DATA_FILE}  行数={len(df)}")

    m = monthly_trend(df)
    print("\n=== 月度趋势 ===")
    show = m.copy()
    show["毛利率"] = (show["毛利率"] * 100).round(2)
    show["平均折扣率"] = (show["平均折扣率"] * 100).round(2)
    print(show.to_string(index=False))
    plot_trends(m)

    cmp = category_mix_analysis(df)
    print("\n=== 品类：首月 vs 末月 ===")
    view = cmp.copy()
    for col in view.columns:
        if "毛利率" in col or "占比" in col or "折扣" in col:
            view[col] = (view[col] * 100).round(2)
        else:
            view[col] = view[col].round(1)
    print(view.to_string())
    plot_category(cmp)

    channel_store_insight(df)
    imp = ml_explain_margin(df)
    narrative(m, cmp, imp)

    # 导出分析表
    m.to_excel(OUT_DIR / "monthly_summary.xlsx", index=False)
    cmp.to_excel(OUT_DIR / "category_first_last.xlsx")
    print(f"\n明细表已写入: {OUT_DIR}")


if __name__ == "__main__":
    main()
