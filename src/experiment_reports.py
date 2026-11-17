"""八分类实验曲线及评估表，不依赖患者或临床数据。"""
import csv
import json
import os
from pathlib import Path
import numpy as np


def plotting(directory):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(directory) / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from utils.plot_style import configure_chinese_font
    configure_chinese_font()
    import matplotlib.pyplot as plt
    return plt


def save_history(directory, history):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "history.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(directory / "history.csv")
    plt = plotting(directory)
    for metric in ("loss", "accuracy", "macro_f1"):
        fig, ax = plt.subplots(figsize=(7, 4))
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row["train_" + metric] for row in history], label="训练集")
        ax.plot(epochs, [row["val_" + metric] for row in history], label="验证集")
        titles = {"loss": "训练与验证损失曲线", "accuracy": "训练与验证准确率曲线", "macro_f1": "训练与验证 Macro-F1 曲线"}
        ax.set(xlabel="训练轮次（Epoch）", ylabel=metric, title=titles[metric])
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / (metric + "_curve.png"), dpi=180)
        plt.close(fig)


def evaluation_metrics(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    support, predicted = matrix.sum(1), matrix.sum(0)
    tp = matrix.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(len(tp)), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(len(tp)), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(len(tp)), where=(support + predicted) != 0)
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Cannot evaluate an empty confusion matrix")
    expected = float(np.dot(support.astype(float), predicted.astype(float)) / total ** 2)
    accuracy = float(tp.sum() / total)
    scores = {"accuracy": accuracy, "macro_precision": float(precision.mean()),
              "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
              "weighted_f1": float(np.dot(f1, support) / total),
              "balanced_accuracy": float(recall[support > 0].mean()),
              "cohen_kappa": (accuracy - expected) / (1 - expected) if expected < 1 else None,
              "sample_count": total}
    return scores, precision, recall, f1, support


def evaluation_title(name):
    return {"best_val": "最佳模型验证集", "test": "测试集",
            "legacy_best_val": "历史最佳模型验证集"}.get(name, "分类评估")


def save_evaluation(directory, name, scores, matrix, classes, targets=None, probabilities=None, image_names=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    additional, precision, recall, f1, support = evaluation_metrics(matrix)
    scores.update(additional)
    with (directory / (name + "_confusion_matrix.csv")).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred"] + list(classes))
        writer.writerows([cls] + list(row) for cls, row in zip(classes, matrix))
    with (directory / (name + "_per_class.csv")).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for index, cls in enumerate(classes):
            writer.writerow([cls, precision[index], recall[index], f1[index], support[index]])
    plt = plotting(directory)
    fig, ax = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(classes))
    for offset, metric, values in ((-1, "Precision", precision), (0, "Recall", recall), (1, "F1", f1)):
        bars = ax.bar(positions + offset * .25, values, width=.25, label=metric)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .01,
                    f"{bar.get_height():.2f}", ha="center", fontsize=7)
    ax.set_xticks(positions)
    ax.set_xticklabels(classes)
    ax.set(ylim=(0, 1.12), ylabel="Score", title=evaluation_title(name) + "：各组织类别 Precision、Recall 与 F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / (name + "_precision_recall_f1.png"), dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    normalized = [[value / max(sum(row), 1) for value in row] for row in matrix]
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    ax.set(xlabel="预测类别", ylabel="真实类别", title=evaluation_title(name) + "：归一化混淆矩阵")
    for i, row in enumerate(normalized):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if value > 0.5 else "black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(directory / (name + "_confusion_matrix.png"), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    ax.set(xlabel="预测类别", ylabel="真实类别", title=evaluation_title(name) + "：混淆矩阵（样本数）")
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            ax.text(j, i, str(count), ha="center", va="center",
                    color="white" if count > np.max(matrix) / 2 else "black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(directory / (name + "_confusion_matrix_counts.png"), dpi=180)
    plt.close(fig)
    errors = [(classes[i], classes[j], matrix[i][j], matrix[i][j] / max(sum(matrix[i]), 1))
              for i in range(len(classes)) for j in range(len(classes)) if i != j and matrix[i][j]]
    with (directory / (name + "_confused_pairs.csv")).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class", "predicted_class", "count", "fraction_of_true_class"])
        writer.writerows(sorted(errors, key=lambda row: -row[2]))
    if targets is not None and probabilities is not None:
        save_roc(directory, name, targets, probabilities, classes, scores, image_names, matrix)
    (directory / (name + "_metrics.json")).write_text(json.dumps(scores, indent=2, allow_nan=False), encoding="utf-8")


def save_roc(directory, name, targets, probabilities, classes, scores, image_names=None, matrix=None):
    from sklearn.metrics import roc_curve, auc, average_precision_score
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.shape != (len(targets), len(classes)) or not len(targets)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or np.any(probabilities > 1) or not np.allclose(probabilities.sum(1), 1, atol=1e-5)
            or np.any(targets < 0) or np.any(targets >= len(classes))):
        raise ValueError("Invalid targets or class probabilities for ROC")
    if image_names is not None and len(image_names) != len(targets):
        raise ValueError("Image names and prediction counts differ")
    predicted = probabilities.argmax(1)
    actual = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(actual, (targets, predicted), 1)
    if matrix is not None and not np.array_equal(actual, matrix):
        raise ValueError("Confusion matrix and ROC predictions disagree")
    plt = plotting(directory)
    fig, ax = plt.subplots(figsize=(9, 7))
    curves, records = [], []
    binary = targets[:, None] == np.arange(len(classes))
    for i, cls in enumerate(classes):
        if np.unique(binary[:, i]).size < 2:
            records.append([cls, None, None, "undefined: positive or negative class absent"])
            continue
        fpr, tpr, _ = roc_curve(binary[:, i], probabilities[:, i])
        score = float(auc(fpr, tpr))
        ap = float(average_precision_score(binary[:, i], probabilities[:, i]))
        curves.append((fpr, tpr, score))
        records.append([cls, score, ap, "ok"])
        ax.plot(fpr, tpr, label=f"{cls} AUC={score:.4f}")
    valid_aucs = [curve[2] for curve in curves]
    # Mean-AUC is the unweighted mean of all eight OvR AUCs, not a pooled score.
    scores["mean_auc_ovr"] = float(np.mean(valid_aucs)) if len(curves) == len(classes) else None
    scores["valid_auc_class_count"] = len(curves)
    scores["mean_average_precision_ovr"] = float(np.mean([row[2] for row in records if row[2] is not None])) if len(curves) == len(classes) else None
    if curves:
        grid = np.unique(np.concatenate([curve[0] for curve in curves]))
        mean_tpr = np.mean([np.interp(grid, fpr, tpr) for fpr, tpr, _ in curves], axis=0)
        ax.plot(grid, mean_tpr, "k--", linewidth=2, label="宏平均 ROC（插值）")
    micro_fpr, micro_tpr, _ = roc_curve(binary.ravel(), probabilities.ravel())
    scores["micro_auc_ovr"] = float(auc(micro_fpr, micro_tpr))
    mean_label = f"{scores['mean_auc_ovr']:.4f}" if scores["mean_auc_ovr"] is not None else "不可计算（缺少类别）"
    ax.plot(micro_fpr, micro_tpr, linestyle=":", label=f"Micro AUC={scores['micro_auc_ovr']:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set(xlabel="假阳性率（FPR）", ylabel="真阳性率（TPR）", title=f"{evaluation_title(name)}：各类别 OvR ROC 曲线\nMean-AUC={mean_label}", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / (name + "_roc_ovr.png"), dpi=200)
    plt.close(fig)
    with (directory / (name + "_auc_per_class.csv")).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "ovr_auc", "average_precision", "status"])
        writer.writerows(records)
    with (directory / (name + "_predictions.csv")).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["img_name", "true_class", "predicted_class"] + ["prob_" + cls for cls in classes])
        for index, target in enumerate(targets):
            writer.writerow([image_names[index] if image_names is not None else index,
                             classes[target], classes[predicted[index]]] + probabilities[index].tolist())
