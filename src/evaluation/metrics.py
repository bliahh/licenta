import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score, f1_score, \
    recall_score, precision_score, fbeta_score, confusion_matrix
from src.utils.helpers import json_to_csv


def compute_metrics_for_binary(probs, labels,threshold=0.5):
    probs_are_finite = np.isfinite(probs).all()

    if probs_are_finite:
        preds = (probs >= threshold).astype(int)
    else:
        preds = np.zeros_like(probs, dtype=int)

    if probs_are_finite and len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, probs)
        pr_auc = average_precision_score(labels, probs)
        brier_score = brier_score_loss(labels, probs)

    else:
        auc = float("nan")
        pr_auc = float("nan")
        brier_score = float("nan")

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    f2 = fbeta_score(labels, preds, beta=2)
    recall = recall_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)

    return {
        "acc": accuracy,
        "f1": f1,
        "f2": f2,
        "auc": auc,
        "pr_auc": pr_auc,
        "recall": recall,
        "precision": precision,
        "brier_score": brier_score,
        "labels": labels,
        "preds": preds,
    }


def confusion(scores, labels, threshold):
    """Patient-level confusion matrix plus sensitivity and specificity at the given threshold."""
    preds = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(np.asarray(labels), preds, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1))}


if __name__ == "__main__":
    json_to_csv("/home/bliahh/Desktop/FACULATATE/licenta/output/outputs_vesselmask/comparison_results_TEST_TUNED_ALL.json","comparison.csv")
