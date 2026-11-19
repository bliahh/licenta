import os
import json
import argparse

import numpy as np
import pandas as pd

from binary_classification.whole_volume_rsna.evaluation.utils import (
    RESULTS_DIR, MODALITIES, SEPARATE_MODELS, JOINT_MODEL, METRICS, mean_std, safe_auc, paired_bootstrap)

ALL_METRICS = METRICS + ["hit", "hit_exact"]


def find_runs(results_dir):
    runs = {}
    for folder in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, folder)
        if "_seed" not in folder or not os.path.exists(os.path.join(run_dir, "metrics_test.json")):
            continue
        name, seed = folder.rsplit("_seed", 1)
        if not seed.isdigit():
            continue
        if name not in runs:
            runs[name] = {}
        runs[name][int(seed)] = run_dir
    return runs


def read_test_predictions(run_dir):
    path = os.path.join(run_dir, "predictions_test.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def summarize_metrics(runs):
    rows = []
    for name, seeds in runs.items():
        metrics = []
        for run_dir in seeds.values():
            with open(os.path.join(run_dir, "metrics_test.json")) as f:
                metrics.append(json.load(f))

        row = {"experiment": name, "seeds": len(metrics), "n_test": int(metrics[0].get("n_test", 0))}
        for metric in ALL_METRICS:
            row[metric] = mean_std([m.get(metric, np.nan) for m in metrics])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_per_modality(runs):
    rows = []
    for name, seeds in runs.items():
        aucs = {"CTA": [], "MRA": []}
        for run_dir in seeds.values():
            predictions = read_test_predictions(run_dir)
            if predictions is None:
                continue
            for modality in MODALITIES:
                subset = predictions[predictions["modality"] == modality]
                if len(subset) > 0:
                    aucs[modality].append(safe_auc(subset["label"], subset["logit"]))

        row = {"experiment": name}
        for modality in MODALITIES:
            row[f"auc_{modality}"] = mean_std(aucs[modality])
        rows.append(row)
    return pd.DataFrame(rows)


def compare_joint_with_separate(runs, repetitions):
    rows = []
    if JOINT_MODEL not in runs:
        return pd.DataFrame(rows)

    rng = np.random.default_rng(0)
    for modality, separate_model in SEPARATE_MODELS.items():
        if separate_model not in runs:
            continue
        seeds = sorted(set(runs[JOINT_MODEL]) & set(runs[separate_model]))
        for seed in seeds:
            joint = read_test_predictions(runs[JOINT_MODEL][seed])
            separate = read_test_predictions(runs[separate_model][seed])
            if joint is None or separate is None:
                continue

            joint = joint[joint["modality"] == modality]
            merged = joint.merge(separate[["uid", "logit"]], on="uid", suffixes=("_joint", "_separate"))
            labels = merged["label"].to_numpy()
            joint_scores = merged["logit_joint"].to_numpy()
            separate_scores = merged["logit_separate"].to_numpy()

            auc_joint = safe_auc(labels, joint_scores)
            auc_separate = safe_auc(labels, separate_scores)
            low, high = paired_bootstrap(labels, joint_scores, separate_scores, repetitions, rng)
            rows.append({
                "modality": modality,
                "seed": seed,
                "n": len(merged),
                "auc_joint": round(auc_joint, 3),
                "auc_separate": round(auc_separate, 3),
                "difference": round(auc_joint - auc_separate, 3),
                "ci_low": round(low, 3),
                "ci_high": round(high, 3),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    runs = find_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no runs with metrics_test.json in {args.results_dir}")

    metrics_table = summarize_metrics(runs)
    print("\nTest metrics (mean +- std over seeds)")
    print(metrics_table.to_string(index=False))

    modality_table = summarize_per_modality(runs)
    print("\nAUC per modality (mean +- std over seeds)")
    print(modality_table.to_string(index=False))

    comparison_table = compare_joint_with_separate(runs, args.bootstrap)
    if len(comparison_table) > 0:
        print("\nJoint vs separate models, AUC difference on the same test series (95% bootstrap CI)")
        print(comparison_table.to_string(index=False))
        print("\nMean difference over seeds")
        print(comparison_table.groupby("modality")["difference"].apply(mean_std).to_string())

    output_path = os.path.join(args.results_dir, "summary.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        metrics_table.to_excel(writer, sheet_name="test_metrics", index=False)
        modality_table.to_excel(writer, sheet_name="auc_per_modality", index=False)
        comparison_table.to_excel(writer, sheet_name="joint_vs_separate", index=False)
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()
