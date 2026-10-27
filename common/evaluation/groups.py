import numpy as np

from common.evaluation.metrics import compute_metrics_for_binary, confusion


def group_metrics(probs, labels, groups, threshold):
    """Patient-level metrics for every group mask (see common.data.patient_info.patient_groups)."""
    out = {}
    for g, mask in groups.items():
        p, y = np.asarray(probs)[mask], np.asarray(labels)[mask]
        if len(y) == 0:
            continue
        cm = confusion(p, y, threshold)
        m = compute_metrics_for_binary(probs=p, labels=y, threshold=threshold) if len(np.unique(y)) == 2 else {}
        spec = cm["specificity"] if (1 - y).sum() else np.nan
        out[g] = {"n": int(len(y)), "pos": int(y.sum()), "neg": int((1 - y).sum()), "auc": m.get("auc", np.nan), "pr_auc": m.get("pr_auc", np.nan), "sensitivity": float(cm["sensitivity"]), "specificity": float(spec), "tp": int(cm["tp"]), "fn": int(cm["fn"]), "fp": int(cm["fp"]), "tn": int(cm["tn"])}
    return out


def print_groups(gm):
    """One line per group."""
    fmt = lambda v: f"{v:.3f}" if v is not None and np.isfinite(v) else "  n/a"
    for g, r in gm.items():
        print(f"  [{g:<10}] n={r['n']:3d} pos={r['pos']:3d} neg={r['neg']:3d} | auc={fmt(r['auc'])} pr_auc={fmt(r['pr_auc'])} sens={fmt(r['sensitivity'])} spec={fmt(r['specificity'])} | TP={r['tp']} FN={r['fn']} FP={r['fp']} TN={r['tn']}")
