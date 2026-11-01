import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.paths import PATIENTS_CSV
from common.data.patient_info import center_of, patient_groups
from common.evaluation.groups import group_metrics, print_groups
from common.evaluation.thresholds import youden_mr_threshold
from patch_sliding.config import RUNS_DIR, TABLES_DIR


def patch_auc_by_group(res, loader, samples, split_name="test"):
    """Patch-level AUC per patient group (ambiguous patches are left out); res is the output of run_sliding on loader."""
    probs = np.asarray(res["patch_probs"])
    idx = np.asarray(res["patch_indices"])
    ds = loader.dataset

    labels = np.asarray(ds.patch_labels())[idx]
    entries = [ds.patch_index[i] for i in idx]
    pid = np.array([int(e[0]) for e in entries])
    print(f"[check] first entries: {entries[:3]} | patients covered: {len(set(pid.tolist()))} (expected {len(samples)})")

    images = [samples[p]["image"] for p in pid]
    acq = [samples[p]["type_acq"] for p in pid]
    thick = pd.read_csv(PATIENTS_CSV).set_index("image")["thickest"].loc[images].to_numpy()
    valid = labels >= 0

    groups = {g: valid & m for g, m in patient_groups(images, acq, thick).items()}
    groups["other_mr"] = groups["mr"] & (np.array([center_of(p) for p in images]) != "center1")

    print(f"\n=== PATCH-LEVEL AUC PER GROUP ({split_name}) ===")
    out = {}
    for g, m in groups.items():
        y, p = (labels[m] == 1).astype(int), probs[m]
        auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
        out[g] = {"patches": int(m.sum()), "pos_patches": int(y.sum()), "auc": auc}
        print(f"  [{g:<10}] patches={m.sum():5d} pos={y.sum():4d} | patch AUC={auc:.3f}")
    return out


def patch_groups_report(run_name, checkpoint="best_model_patient_auc", score_col="top5", patients_csv=PATIENTS_CSV, desc=""):
    """Patient-level metrics per group on TEST for one patch run; threshold = Youden on VAL MR."""
    eval_dir = os.path.join(RUNS_DIR, run_name, "evaluation")
    sp = pd.read_csv(patients_csv)[["image", "thickest"]]
    val = pd.read_csv(os.path.join(eval_dir, f"patient_scores_val_{checkpoint}.csv"))
    test = pd.read_csv(os.path.join(eval_dir, f"patient_scores_test_{checkpoint}.csv")).merge(sp, on="image", how="left")
    assert test["thickest"].notna().all(), "some test patients are missing from patients.csv (image path mismatch)"

    threshold = youden_mr_threshold(val[score_col], val["label"], val["type_acq"])
    groups = patient_groups(test["image"], test["type_acq"], test["thickest"])

    print(f"\n=== {desc} [{checkpoint}, {score_col}] | threshold younden = {threshold:.4f} ===")
    gm = group_metrics(test[score_col].to_numpy(), test["label"].to_numpy().astype(int), groups, threshold)
    print_groups(gm)
    out = pd.DataFrame([{"run": run_name, "checkpoint": checkpoint, "score_col": score_col, "threshold": threshold, "group": g, **r} for g, r in gm.items()])
    out.to_csv(os.path.join(eval_dir, f"groups_{checkpoint}_{score_col}.csv"), index=False)
    return out


def patch_summary(run_names, out_csv=os.path.join(TABLES_DIR, "patch_summary.csv")):
    """Patch-level AUC / PR-AUC on TEST of every run, read from the final_eval_*.json files."""
    rows = []
    for run in run_names:
        eval_dir = os.path.join(RUNS_DIR, run, "evaluation")
        for f in sorted(os.listdir(eval_dir)):
            if not f.endswith(".json"):
                continue
            with open(os.path.join(eval_dir, f), "r") as fh:
                r = json.load(fh)
            if "test_patch_metrics" not in r:
                continue
            rows.append({"run": run, "checkpoint": os.path.basename(r["checkpoint"]), "patch_auc": r["test_patch_metrics"]["auc"], "patch_pr_auc": r["test_patch_metrics"]["pr_auc"], "file": f})
    df = pd.DataFrame(rows).drop_duplicates(subset=["run", "checkpoint"]).round(3)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    return df
