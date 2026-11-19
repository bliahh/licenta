import os
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from binary_classification.whole_volume_rsna.evaluation.utils import (
    RESULTS_DIR, MODALITIES, METRICS, load_predictions, mean_std, paired_bootstrap, threshold_metrics)


def platt(val_predictions):
    model = LogisticRegression(C=1e6)
    model.fit(val_predictions[["logit"]].to_numpy(), val_predictions["label"].to_numpy())
    return lambda predictions: model.predict_proba(predictions[["logit"]].to_numpy())[:, 1]


def calibrate_per_modality(val_predictions, test_predictions):
    val_scores = np.zeros(len(val_predictions))
    test_scores = np.zeros(len(test_predictions))
    for modality in MODALITIES:
        val_mask = (val_predictions["modality"] == modality).to_numpy()
        test_mask = (test_predictions["modality"] == modality).to_numpy()
        calibrate = platt(val_predictions[val_mask])
        val_scores[val_mask] = calibrate(val_predictions[val_mask])
        test_scores[test_mask] = calibrate(test_predictions[test_mask])
    return val_scores, test_scores


def evaluate_seed(results_dir, seed, repetitions, rng, prefix="final"):
    separate_names = [f"{prefix}_ct", f"{prefix}_mr"]
    joint_name = f"{prefix}_joint"

    separate_val = pd.concat([load_predictions(results_dir, name, seed, "val") for name in separate_names])
    separate_test = pd.concat([load_predictions(results_dir, name, seed, "test") for name in separate_names])
    joint_val = load_predictions(results_dir, joint_name, seed, "val")
    joint_test = load_predictions(results_dir, joint_name, seed, "test")
    separate_test = separate_test.set_index("uid").loc[joint_test["uid"]].reset_index()
    test_labels = joint_test["label"].to_numpy()

    scores = {}
    scores["aggregation_raw"] = (separate_val["logit"].to_numpy(), separate_test["logit"].to_numpy(),
                                 separate_val["label"].to_numpy())
    val_cal, test_cal = calibrate_per_modality(separate_val, separate_test)
    scores["aggregation_calibrated"] = (val_cal, test_cal, separate_val["label"].to_numpy())
    scores["joint_raw"] = (joint_val["logit"].to_numpy(), joint_test["logit"].to_numpy(),
                           joint_val["label"].to_numpy())
    val_cal, test_cal = calibrate_per_modality(joint_val, joint_test)
    scores["joint_calibrated"] = (val_cal, test_cal, joint_val["label"].to_numpy())

    rows = []
    for method, (val_scores, test_scores, val_labels) in scores.items():
        row = {"seed": seed, "method": method}
        row.update(threshold_metrics(val_labels, val_scores, test_labels, test_scores))
        rows.append(row)

    comparisons = []
    for joint_method in ["joint_raw", "joint_calibrated"]:
        joint_scores = scores[joint_method][1]
        aggregation_scores = scores["aggregation_calibrated"][1]
        low, high = paired_bootstrap(test_labels, joint_scores, aggregation_scores, repetitions, rng)
        difference = roc_auc_score(test_labels, joint_scores) - roc_auc_score(test_labels, aggregation_scores)
        comparisons.append({
            "seed": seed,
            "comparison": f"{joint_method} - aggregation_calibrated",
            "n": len(test_labels),
            "difference": round(difference, 3),
            "ci_low": round(low, 3),
            "ci_high": round(high, 3),
        })
    return rows, comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--prefix", default="final")
    args = parser.parse_args()
    pd.set_option("display.width", 200)

    rng = np.random.default_rng(0)
    rows, comparisons = [], []
    for seed in args.seeds:
        seed_rows, seed_comparisons = evaluate_seed(args.results_dir, seed, args.bootstrap, rng, args.prefix)
        rows += seed_rows
        comparisons += seed_comparisons
    per_seed = pd.DataFrame(rows)
    comparison_table = pd.DataFrame(comparisons)

    summary = per_seed.groupby("method", sort=False)[METRICS].agg(mean_std)
    print("\nCombined test set CTA + MRA (mean +- std over seeds)")
    print(summary.to_string())
    print("\nJoint vs aggregation, AUC difference on the same test series (95% bootstrap CI)")
    print(comparison_table.to_string(index=False))
    print("\nMean difference over seeds")
    print(comparison_table.groupby("comparison")["difference"].apply(mean_std).to_string())

    tag = "" if args.prefix == "final" else f"_{args.prefix}"
    per_seed.round(4).to_csv(os.path.join(args.results_dir, f"aggregation_per_seed{tag}.csv"), index=False)
    summary.to_csv(os.path.join(args.results_dir, f"aggregation_summary{tag}.csv"))
    comparison_table.to_csv(os.path.join(args.results_dir, f"joint_vs_aggregation{tag}.csv"), index=False)
    print(f"\nsaved results in {args.results_dir}")


if __name__ == "__main__":
    main()
