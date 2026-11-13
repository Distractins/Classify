"""复用阶段一患者名单，以 TME + 年龄/性别训练 XGBoost MSI 二分类。"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_common import (CSV_DIR, TME_COLUMNS, load_patient_split, normalize_id,
                             project_path, relative_path)

import joblib
import matplotlib
matplotlib.use("Agg")  # 无图形界面也能保存所有图片。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

FEATURES = TME_COLUMNS + ["age", "sex"]
FIGURES = Path("outputs/figures/stage2_msi")


def load_features(path):
    data = pd.read_csv(path, dtype={"patient_id": str})
    required = ["patient_id"] + FEATURES + ["msi_label"]
    if not set(required).issubset(data.columns):
        raise ValueError(f"特征表缺少字段: {sorted(set(required) - set(data.columns))}")
    data["patient_id"] = data.patient_id.map(normalize_id)
    if data.patient_id.duplicated().any():
        raise ValueError("特征表必须每位患者一行。")
    train_ids, val_ids = load_patient_split(data.patient_id)
    data[FEATURES + ["msi_label"]] = data[FEATURES + ["msi_label"]].apply(pd.to_numeric, errors="raise")
    if np.isinf(data[FEATURES].to_numpy()).any():
        raise ValueError("特征不能含无穷大。")
    proportions = data[TME_COLUMNS]
    if (proportions.isna().any().any() or (proportions < 0).any().any()
            or (proportions > 100).any().any()
            or not np.allclose(proportions.sum(axis=1), 100, atol=1e-4)):
        raise ValueError("TME 必须为 8 类百分比，各行合计 100。")
    if not set(data.sex.dropna()).issubset({0, 1}):
        raise ValueError("sex 必须为 Male=1/Female=0 或缺失。")
    if ((data.age < 0) | (data.age > 120)).any():
        raise ValueError("age 必须在 0~120 岁之间或缺失。")
    if not set(data.msi_label.dropna()).issubset({0, 1}):
        raise ValueError("msi_label 必须为 MSI-H=1/MSS=0 或缺失。")
    indexed = data.set_index("patient_id", drop=False)
    # loc 严格按保存名单选取，无 train_test_split、shuffle 或重新分层。
    train = indexed.loc[train_ids].copy()
    val = indexed.loc[val_ids].copy()
    for name, frame in (("train", train), ("validation", val)):
        missing = frame.msi_label.isna()
        if missing.any():
            warnings.warn(f"{name} 排除 {missing.sum()} 位无 MSI 二分类标签患者: {frame.loc[missing, 'patient_id'].tolist()}")
    train, val = train.dropna(subset=["msi_label"]), val.dropna(subset=["msi_label"])
    if set(train.msi_label) != {0, 1} or set(val.msi_label) != {0, 1}:
        raise ValueError("固定训练/验证集合都需要包含 MSI-H 和 MSS 才能训练和计算 ROC。禁止为此重分患者。")
    if train[FEATURES].isna().all().any():
        raise ValueError("训练集某个特征全部缺失，无法用训练集估算中位数。")
    return train, val, train_ids, val_ids


def fit_model(train, columns, args):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", XGBClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", subsample=1.0,
            colsample_bytree=1.0, random_state=42, n_jobs=args.n_jobs)),
    ])
    # 插补器和分类器仅拟合训练患者；验证集不参与调参或早停。
    model.fit(train[columns], train.msi_label.astype(int))
    return model


def save_plots(model, train, val, probabilities, predictions):
    folder = project_path(FIGURES)
    folder.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(val.msi_label, probabilities)
    score = roc_auc_score(val.msi_label, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Validation AUC = {score:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="MSI-H / MSS ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(folder / "msi_roc_curve.png", dpi=300)
    plt.close(fig)

    matrix = confusion_matrix(val.msi_label, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["MSS", "MSI-H"], yticklabels=["MSS", "MSI-H"], ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title="Validation confusion matrix")
    fig.tight_layout()
    fig.savefig(folder / "msi_confusion.png", dpi=300)
    plt.close(fig)

    importance = pd.Series(model.named_steps["classifier"].feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importance.index, importance.values)
    ax.set(xlabel="XGBoost feature importance (gain)", title="MSI model feature importance")
    fig.tight_layout()
    fig.savefig(folder / "feature_importance.png", dpi=300)
    plt.close(fig)

    # 仅在训练患者上做解释性相关分析，避免验证标签影响特征选择。
    corr = train[FEATURES + ["msi_label"]].corr().loc[FEATURES, ["msi_label"]]
    fig, ax = plt.subplots(figsize=(5, 7))
    sns.heatmap(corr.fillna(0), annot=corr.map(lambda x: "N/A" if pd.isna(x) else f"{x:.3f}")
                if hasattr(corr, "map") else corr.applymap(lambda x: "N/A" if pd.isna(x) else f"{x:.3f}"),
                fmt="", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set(title="Feature / MSI correlation (training patients)", xlabel="MSI-H=1, MSS=0")
    fig.tight_layout()
    fig.savefig(folder / "tme_msi_corr_heatmap.png", dpi=300)
    plt.close(fig)
    return float(score), matrix.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-csv", default=str(CSV_DIR / "patient_tme_feature.csv"))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--ablations", action="store_true", help="额外比较仅临床/仅 TME/融合三组固定划分 AUC")
    args = parser.parse_args()
    if args.n_estimators < 1 or args.max_depth < 1 or args.learning_rate <= 0 or args.n_jobs == 0:
        parser.error("树数量、深度和学习率必须为正，n-jobs 不能为 0。")
    train, val, train_ids, val_ids = load_features(project_path(args.feature_csv))
    model = fit_model(train, FEATURES, args)
    probabilities = model.predict_proba(val[FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    score, matrix = save_plots(model, train, val, probabilities, predictions)
    model_path = project_path("outputs/weights/stage2_msi/msi_xgb_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # 保存完整插补 + 分类流水线，附带训练来源，后续 predict_proba 可直接接收同名特征表。
    model.feature_columns_ = FEATURES
    model.train_patient_ids_ = train_ids
    model.val_patient_ids_ = val_ids
    joblib.dump(model, model_path)
    csv_folder = project_path(CSV_DIR)
    csv_folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id": val.patient_id.to_numpy(), "msi_label": val.msi_label.astype(int).to_numpy(),
                  "msi_probability": probabilities, "predicted_label": predictions}).to_csv(
                      csv_folder / "msi_validation_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = {"auc": score, "accuracy": float(accuracy_score(val.msi_label, predictions)),
               "train_patients_with_label": len(train), "validation_patients_with_label": len(val),
               "confusion_matrix": matrix}
    if args.ablations:
        rows = []
        for name, columns in (("clinical_only", ["age", "sex"]), ("tme_only", TME_COLUMNS), ("combined", FEATURES)):
            estimator = model if name == "combined" else fit_model(train, columns, args)
            rows.append({"experiment": name, "validation_auc": roc_auc_score(
                val.msi_label, estimator.predict_proba(val[columns])[:, 1])})
        pd.DataFrame(rows).to_csv(csv_folder / "msi_ablation_results.csv", index=False, encoding="utf-8-sig")
    report = classification_report(val.msi_label, predictions, labels=[0, 1],
                                   target_names=["MSS", "MSI-H"], digits=4, zero_division=0)
    log = project_path("logs/stage2_log.txt")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(metrics, indent=2) + "\n\n" + report, encoding="utf-8")
    print(f"验证 AUC={score:.4f}\n{report}")
    print(f"模型: {relative_path(model_path)}\n四张评估图: {FIGURES.as_posix()}/")


if __name__ == "__main__":
    main()
