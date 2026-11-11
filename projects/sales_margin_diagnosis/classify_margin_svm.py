"""实例 B：SVM（支持向量机）—— 识别「低毛利风险订单」

任务与逻辑回归脚本相同（二分类），算法换成 SVC：
- 线性核：可看系数，适合对照逻辑回归
- RBF 核：捕捉非线性边界，通常精度更高但更难直接解释

本脚本同时训练 Linear / RBF，并对比结果。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from data_utils import (
    FEATURE_CAT,
    FEATURE_NUM,
    LOW_MARGIN_THRESHOLD,
    OUT_DIR,
    load_sales,
    setup_chinese_font,
)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_CAT),
            ("num", StandardScaler(), FEATURE_NUM),
        ]
    )


def eval_model(name: str, pipe: Pipeline, X_test, y_test, proba=None) -> dict:
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"\n=== {name} ===")
    print(f"准确率: {acc:.4f}")
    if proba is not None:
        auc = roc_auc_score(y_test, proba)
        print(f"ROC-AUC: {auc:.4f}")
    else:
        auc = None
    print(classification_report(y_test, pred, target_names=["正常", "低毛利"]))
    return {"name": name, "accuracy": acc, "auc": auc, "pred": pred}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()
    sns.set_theme(style="whitegrid")

    df = load_sales()
    X = df[FEATURE_CAT + FEATURE_NUM]
    y = df["低毛利"]

    print("=== SVM：低毛利订单分类 ===")
    print(f"阈值: 毛利率 < {LOW_MARGIN_THRESHOLD:.0%} 视为低毛利")
    print("标签分布:\n", y.value_counts().rename({0: "正常(0)", 1: "低毛利(1)"}))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- 线性 SVM（可解释系数；用 CalibratedClassifierCV 拿概率）---
    linear_base = Pipeline(
        [
            ("pre", make_preprocessor()),
            (
                "clf",
                LinearSVC(class_weight="balanced", max_iter=10000, dual="auto", random_state=42),
            ),
        ]
    )
    # 先 fit 线性模型拿系数；再单独做一个带概率的校准版做 AUC
    linear_base.fit(X_train, y_train)
    linear_cal = Pipeline(
        [
            ("pre", make_preprocessor()),
            (
                "clf",
                CalibratedClassifierCV(
                    LinearSVC(class_weight="balanced", max_iter=10000, dual="auto", random_state=42),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )
    linear_cal.fit(X_train, y_train)
    linear_proba = linear_cal.predict_proba(X_test)[:, 1]
    r_linear = eval_model("Linear SVM", linear_base, X_test, y_test, proba=linear_proba)

    # 线性系数
    pre = linear_base.named_steps["pre"]
    cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(FEATURE_CAT))
    feature_names = cat_names + FEATURE_NUM
    coef = linear_base.named_steps["clf"].coef_.ravel()
    coef_table = (
        pd.DataFrame({"特征": feature_names, "系数": coef})
        .assign(方向=lambda d: np.where(d["系数"] > 0, "推高低毛利风险", "降低低毛利风险"))
        .assign(abs_coef=lambda d: d["系数"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
    )
    print("\n=== Linear SVM 系数 Top 10 ===")
    print(coef_table.head(10).to_string(index=False))
    coef_table.to_excel(OUT_DIR / "svm_linear_coefficients.xlsx", index=False)

    # --- RBF SVM（非线性）---
    rbf = Pipeline(
        [
            ("pre", make_preprocessor()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )
    rbf.fit(X_train, y_train)
    rbf_proba = rbf.predict_proba(X_test)[:, 1]
    r_rbf = eval_model("RBF SVM", rbf, X_test, y_test, proba=rbf_proba)
    n_sv = int(np.sum(rbf.named_steps["clf"].n_support_))
    print(f"RBF 支持向量个数: {n_sv}")

    # RBF 用置换重要性解释（系数不可直接读）
    result = permutation_importance(rbf, X_test, y_test, n_repeats=6, random_state=42, n_jobs=-1)
    imp = pd.DataFrame({"特征": X.columns, "重要性": result.importances_mean}).sort_values(
        "重要性", ascending=False
    )
    print("\n=== RBF SVM 置换重要性 ===")
    print(imp.to_string(index=False))

    # 图：两个混淆矩阵 + 重要性
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, r_linear["pred"], display_labels=["正常", "低毛利"], ax=axes[0]
    )
    axes[0].set_title("Linear SVM")
    ConfusionMatrixDisplay.from_predictions(
        y_test, r_rbf["pred"], display_labels=["正常", "低毛利"], ax=axes[1]
    )
    axes[1].set_title("RBF SVM")
    fig.tight_layout()
    cm_path = OUT_DIR / "06_svm_confusion.png"
    fig.savefig(cm_path, dpi=130)
    plt.close(fig)
    print(f"图已保存: {cm_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=imp, x="重要性", y="特征", ax=ax, color="steelblue")
    ax.set_title("RBF SVM — permutation importance")
    fig.tight_layout()
    imp_path = OUT_DIR / "07_svm_rbf_importance.png"
    fig.savefig(imp_path, dpi=130)
    plt.close(fig)
    print(f"图已保存: {imp_path}")

    # 对比表
    compare = pd.DataFrame(
        [
            {"模型": "Linear SVM", "准确率": r_linear["accuracy"], "ROC-AUC": r_linear["auc"]},
            {"模型": "RBF SVM", "准确率": r_rbf["accuracy"], "ROC-AUC": r_rbf["auc"]},
        ]
    )
    print("\n=== SVM 对比 ===")
    print(compare.to_string(index=False))
    compare.to_excel(OUT_DIR / "svm_compare.xlsx", index=False)

    print("\n怎么选：")
    print("- 要解释「为什么像低毛利」→ Linear SVM / 逻辑回归（看系数）")
    print("- 要更高拟合、边界更弯 → RBF SVM（看置换重要性）")
    print("- 与随机森林回归脚本配合：RF 解释连续毛利率；这里做风险单分类预警")


if __name__ == "__main__":
    main()
