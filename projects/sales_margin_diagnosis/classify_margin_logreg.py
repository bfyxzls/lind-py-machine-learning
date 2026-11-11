"""实例 A：逻辑回归 —— 识别「低毛利风险订单」

与 diagnose_margin.py（随机森林回归：预测连续毛利率）不同：
本脚本把问题改成二分类：毛利率 < 15% → 低毛利(1)，否则正常(0)。

逻辑回归优点：系数可解释（哪个因素让「低毛利」概率升高）。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_utils import (
    FEATURE_CAT,
    FEATURE_NUM,
    LOW_MARGIN_THRESHOLD,
    OUT_DIR,
    load_sales,
    setup_chinese_font,
)


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_CAT),
            ("num", StandardScaler(), FEATURE_NUM),
        ]
    )
    # C 越小正则越强；balanced 缓解两类不均
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0, random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def explain_coefficients(pipe: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """系数 > 0：该特征增大时，更倾向预测为「低毛利」。"""
    coef = pipe.named_steps["clf"].coef_.ravel()
    table = pd.DataFrame({"特征": feature_names, "系数": coef})
    table["方向"] = np.where(table["系数"] > 0, "推高低毛利风险", "降低低毛利风险")
    table["|系数|"] = table["系数"].abs()
    return table.sort_values("|系数|", ascending=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()
    sns.set_theme(style="whitegrid")

    df = load_sales()
    X = df[FEATURE_CAT + FEATURE_NUM]
    y = df["低毛利"]

    print("=== 逻辑回归：低毛利订单分类 ===")
    print(f"阈值: 毛利率 < {LOW_MARGIN_THRESHOLD:.0%} 视为低毛利")
    print("标签分布:\n", y.value_counts().rename({0: "正常(0)", 1: "低毛利(1)"}))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    print(f"\n准确率: {accuracy_score(y_test, pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
    print("\n分类报告:\n", classification_report(y_test, pred, target_names=["正常", "低毛利"]))

    # 混淆矩阵
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["正常", "低毛利"], ax=ax)
    ax.set_title("Logistic Regression — Confusion Matrix")
    fig.tight_layout()
    cm_path = OUT_DIR / "04_logreg_confusion.png"
    fig.savefig(cm_path, dpi=130)
    plt.close(fig)
    print(f"图已保存: {cm_path}")

    # 系数解释
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(FEATURE_CAT))
    feature_names = cat_names + FEATURE_NUM
    coef_table = explain_coefficients(pipe, feature_names)
    print("\n=== 系数 Top 10（绝对值）===")
    print(coef_table.head(10).drop(columns="|系数|").to_string(index=False))
    coef_table.to_excel(OUT_DIR / "logreg_coefficients.xlsx", index=False)

    top = coef_table.head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["crimson" if v > 0 else "seagreen" for v in top["系数"]]
    ax.barh(top["特征"], top["系数"], color=colors)
    ax.axvline(0, color="gray", lw=0.8)
    ax.invert_yaxis()
    ax.set_title("Logistic Regression coefficients (red = higher low-margin risk)")
    fig.tight_layout()
    coef_path = OUT_DIR / "05_logreg_coefficients.png"
    fig.savefig(coef_path, dpi=130)
    plt.close(fig)
    print(f"图已保存: {coef_path}")

    print("\n结论提示：")
    print("- 红色/正系数特征 ↑ → 更像低毛利单（例如某促销品类、高折扣）")
    print("- 绿色/负系数特征 ↑ → 更像健康毛利单")
    print("- 这与「促销冲量、折扣加深导致毛利下滑」的业务故事一致")


if __name__ == "__main__":
    main()
