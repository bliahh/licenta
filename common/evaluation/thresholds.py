import numpy as np
from sklearn.metrics import fbeta_score, roc_curve


def find_th(scores, labels, beta=1):
    """Threshold (among the observed scores) that maximizes the F_beta score."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    best_thresh, best_metric = 0.5, -1.0
    for threshold in np.unique(scores):
        preds = (scores >= threshold).astype(int)
        score = fbeta_score(labels, preds, beta=beta, zero_division=0)
        if score > best_metric:
            best_metric = score
            best_thresh = float(threshold)
    print(f"[THRESHOLD] best={best_thresh:.4f} | F{beta}={best_metric:.3f}")
    return best_thresh, best_metric


def youden_th(scores, labels):
    """Threshold that maximizes sensitivity + specificity - 1 (Youden index)."""
    fpr, tpr, thr = roc_curve(np.asarray(labels), np.asarray(scores))
    i = int(np.argmax(tpr - fpr))
    return float(min(thr[i], 1.0)), float(tpr[i] - fpr[i])


def youden_mr_threshold(probs, labels, modalities):
    """Youden threshold computed on the MR patients only (CT has no negatives)."""
    mr = np.array([str(m).lower() == "mr" for m in modalities])
    fpr, tpr, thr = roc_curve(np.asarray(labels)[mr], np.asarray(probs)[mr])
    return float(thr[np.argmax(tpr - fpr)])
