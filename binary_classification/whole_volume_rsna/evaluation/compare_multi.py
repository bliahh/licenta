import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "binary_classification", "whole_volume_rsna", "out")

RUNS = {"final": "binary 1mm", "final05": "binary 0.5mm", "final_multi": "5 groups", "final_multi13": "13 classes"}
MODELS = {"ct": "CT", "mr": "MR", "joint": "CT+MR"}
GROUPS = {
    "ICA": ["L_InfraICA", "R_InfraICA", "L_SupraICA", "R_SupraICA"],
    "MCA": ["L_MCA", "R_MCA"],
    "ACom_ACA": ["ACom", "L_ACA", "R_ACA"],
    "PCom": ["L_PCom", "R_PCom"],
    "Posterior": ["BasilarTip", "OtherPost"],
}
LOCATIONS = [name for names in GROUPS.values() for name in names]


def auc(y, score):
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, score)


def rsna_score(pred):
    loc_aucs = [auc(pred[f"label_{name}"], pred[f"logit_{name}"]) for name in LOCATIONS]
    loc_aucs = [a for a in loc_aucs if not np.isnan(a)]
    return (13 * auc(pred["label"], pred["logit"]) + sum(loc_aucs)) / (13 + len(loc_aucs))


def groups_from_13(pred):
    result = {}
    for group, names in GROUPS.items():
        y = pred[[f"label_{name}" for name in names]].max(axis=1)
        score = pred[[f"logit_{name}" for name in names]].max(axis=1)
        result[group] = auc(y, score)
    return result


def per_class(out):
    rows = []
    for model, model_name in MODELS.items():
        for run, names in [("final_multi", list(GROUPS)), ("final_multi13", LOCATIONS)]:
            for seed in [0, 1, 2]:
                path = os.path.join(out, f"{run}_{model}_seed{seed}", "metrics_test.json")
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    metrics = json.load(f)
                for name in names:
                    rows.append({"model": model_name, "run": RUNS[run], "class": name, "seed": seed,
                                 "auc": metrics.get(f"auc_{name}", np.nan),
                                 "positives": metrics.get(f"positives_{name}", np.nan)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    table = df.groupby(["run", "class", "model"], sort=False).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"), positives=("positives", "first")).reset_index()
    table["auc"] = [f"{m:.3f} +- {s:.3f} (n={p:.0f})" for m, s, p in zip(table["auc_mean"], table["auc_std"], table["positives"])]
    table = table.pivot_table(index=["run", "class"], columns="model", values="auc", aggfunc="first", sort=False)
    return table[[m for m in MODELS.values() if m in table]].reset_index()


def summarize(df, cols, keys=("model", "run")):
    out = []
    for values, part in df.groupby(list(keys), sort=False):
        row = dict(zip(keys, values))
        row["seeds"] = len(part)
        for col in cols:
            row[col] = "-" if part[col].isna().all() else f"{part[col].mean():.3f} +- {part[col].std():.3f}"
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    rows, group_rows = [], []
    for model, model_name in MODELS.items():
        for run, run_name in RUNS.items():
            for seed in [0, 1, 2]:
                run_dir = os.path.join(args.out, f"{run}_{model}_seed{seed}")
                if not os.path.exists(os.path.join(run_dir, "metrics_test.json")):
                    continue
                with open(os.path.join(run_dir, "metrics_test.json")) as f:
                    metrics = json.load(f)
                row = {"model": model_name, "run": run_name, "seed": seed, "auc": metrics["auc"], "hit": metrics.get("hit"),
                       "sens": metrics.get("sensitivity"), "spec": metrics.get("specificity")}

                if run == "final_multi":
                    group_rows.append({"model": model_name, "run": "5 groups", "seed": seed,
                                       **{g: metrics[f"auc_{g}"] for g in GROUPS}})
                if run == "final_multi13":
                    pred = pd.read_csv(os.path.join(run_dir, "predictions_test.csv"))
                    row["rsna"] = rsna_score(pred)
                    group_rows.append({"model": model_name, "run": "13 classes grouped", "seed": seed, **groups_from_13(pred)})
                rows.append(row)

    runs = pd.DataFrame(rows)
    groups = pd.DataFrame(group_rows)
    print(f"found {len(runs)} runs")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)

    binary_table = summarize(runs, [c for c in ["auc", "sens", "spec", "hit", "rsna"] if c in runs])
    print("\nAneurysm present, test set (mean +- std over seeds)")
    print(binary_table.to_string(index=False))

    group_table = None
    if len(groups):
        group_table = summarize(groups, list(GROUPS))
        print("\nGroup AUC, 5 group model vs 13 class model (max over the locations in each group)")
        print(group_table.to_string(index=False))

    classes = per_class(args.out)
    if len(classes):
        print("\nAUC per class (n = positive test series)")
        print(classes.to_string(index=False))

    if args.save:
        with pd.ExcelWriter(args.save) as writer:
            binary_table.to_excel(writer, sheet_name="binary", index=False)
            runs.round(4).to_excel(writer, sheet_name="binary per seed", index=False)
            if group_table is not None:
                group_table.to_excel(writer, sheet_name="groups", index=False)
                groups.round(4).to_excel(writer, sheet_name="groups per seed", index=False)
            if len(classes):
                classes.to_excel(writer, sheet_name="per class", index=False)
        print(f"\nsaved {args.save}")