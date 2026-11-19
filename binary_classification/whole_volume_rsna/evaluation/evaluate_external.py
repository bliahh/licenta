import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from binary_classification.whole_volume_rsna.data.loaders import load_points, roi_start
from binary_classification.whole_volume_rsna.models.run import predict
from binary_classification.whole_volume_rsna.models.train import localization_hit_rate
from binary_classification.whole_volume_rsna.evaluation.utils import (
    RESULTS_DIR, EXTERNAL_DIR, get_device, load_model, make_loader, run_dir_of,
    mean_std, paired_bootstrap, threshold_metrics)


MODELS = {"largeia": ("CTA", ["ct", "joint"]), "topaneu": ("MRA", ["mr", "joint"])}
SUBGROUPS = {"largeia": {"all": None, "institution1": "institution1"}, "topaneu": {"all": None}}


def predict_external(run_dir, table, points_path):
    device, use_amp = get_device()
    model, roi_size, _ = load_model(run_dir, device)
    table = table.reset_index(drop=True)
    loader = make_loader(table, roi_size, use_amp)
    logits, labels, peaks = predict(model, loader, device, use_amp, return_peaks=True)
    if logits.ndim == 2:
        logits = logits[:, 0]
    labels = labels.reshape(len(labels), -1)[:, 0]

    predictions = table[["uid", "site", "modality"]].copy()
    predictions["label"] = labels.astype(int)
    predictions["logit"] = logits

    hits = {}
    if points_path is not None:
        points = load_points(points_path)
        hits["hit"] = localization_hit_rate(table, peaks, points, roi_size=roi_size)
        hits["hit_exact"] = localization_hit_rate(table, peaks, points, tolerance=0, roi_size=roi_size)
    return predictions, hits


def roi_coverage(points_path, modality, roi_size=96):
    points = pd.read_csv(points_path)
    shifted = points[["i", "j", "k"]].to_numpy() - roi_start(modality, roi_size)
    inside = ((shifted >= 0) & (shifted < roi_size)).all(axis=1)
    print(f"aneurysm points inside the roi (size {roi_size}): {inside.sum()}/{len(inside)} ({inside.mean():.3f})")


def get_predictions(run_dir, table, points_path, cache_path):
    hits_path = cache_path.replace(".csv", "_hits.json")
    if os.path.exists(cache_path):
        predictions = pd.read_csv(cache_path)
        hits = {}
        if os.path.exists(hits_path):
            with open(hits_path) as f:
                hits = json.load(f)
        return predictions, hits

    predictions, hits = predict_external(run_dir, table, points_path)
    predictions.to_csv(cache_path, index=False)
    with open(hits_path, "w") as f:
        json.dump(hits, f)
    return predictions, hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(MODELS), required=True)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--prefix", default="final")
    args = parser.parse_args()
    pd.set_option("display.width", 200)

    modality, suffixes = MODELS[args.dataset]
    models = [f"{args.prefix}_{suffix}" for suffix in suffixes]
    table = pd.read_csv(os.path.join(EXTERNAL_DIR, f"index_{args.dataset}.csv"))
    points_path = os.path.join(EXTERNAL_DIR, f"localizers_{args.dataset}.csv")
    if not os.path.exists(points_path):
        points_path = None

    print(f"{args.dataset} ({modality}): {len(table)} cases, {int(table['label'].sum())} positive")
    if points_path is not None:
        roi_coverage(points_path, modality)
    output_dir = os.path.join(args.results_dir, "external")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    scores = {}
    for model_name in models:
        for seed in args.seeds:
            run_dir = run_dir_of(args.results_dir, model_name, seed)
            cache_path = os.path.join(output_dir, f"{args.dataset}_{model_name}_seed{seed}.csv")
            predictions, hits = get_predictions(run_dir, table, points_path, cache_path)
            scores[(model_name, seed)] = predictions

            val = pd.read_csv(os.path.join(run_dir, "predictions_val.csv"))
            val = val[val["modality"] == modality]

            for subgroup, site in SUBGROUPS[args.dataset].items():
                subset = predictions if site is None else predictions[predictions["site"] == site]
                row = {"model": model_name, "seed": seed, "subgroup": subgroup, "n": len(subset),
                       "positives": int(subset["label"].sum())}
                row.update(threshold_metrics(val["label"].to_numpy(), val["logit"].to_numpy(),
                                             subset["label"].to_numpy(), subset["logit"].to_numpy()))
                if site is None:
                    row.update(hits)
                rows.append(row)

            auc = roc_auc_score(predictions["label"], predictions["logit"])
            print(f"{model_name} seed {seed}: auc {auc:.3f}", flush=True)

    per_seed = pd.DataFrame(rows)
    all_metrics = ["auc", "sensitivity", "specificity", "balanced_accuracy", "f1", "hit", "hit_exact"]
    metrics = [column for column in all_metrics if column in per_seed]
    aggregations = {column: (column, mean_std) for column in metrics}
    summary = per_seed.groupby(["subgroup", "model"], sort=False).agg(
        n=("n", "first"), positives=("positives", "first"), **aggregations).reset_index()
    print(f"\n{args.dataset} results (mean +- std over seeds)")
    print(summary.to_string(index=False))

    rng = np.random.default_rng(0)
    comparisons = []
    specific, joint = models
    for subgroup, site in SUBGROUPS[args.dataset].items():
        for seed in args.seeds:
            merged = scores[(joint, seed)].merge(scores[(specific, seed)][["uid", "logit"]], on="uid",
                                                 suffixes=("_joint", "_specific"))
            if site is not None:
                merged = merged[merged["site"] == site]
            labels = merged["label"].to_numpy()
            joint_scores = merged["logit_joint"].to_numpy()
            specific_scores = merged["logit_specific"].to_numpy()
            low, high = paired_bootstrap(labels, joint_scores, specific_scores, args.bootstrap, rng)
            difference = roc_auc_score(labels, joint_scores) - roc_auc_score(labels, specific_scores)
            comparisons.append({"subgroup": subgroup, "seed": seed, "n": len(merged),
                                "difference": round(difference, 3), "ci_low": round(low, 3), "ci_high": round(high, 3)})
    comparison_table = pd.DataFrame(comparisons)
    print(f"\n{joint} minus {specific}, AUC difference (95% bootstrap CI)")
    print(comparison_table.to_string(index=False))
    print(comparison_table.groupby("subgroup", sort=False)["difference"].apply(mean_std).to_string())

    name = args.dataset if args.prefix == "final" else f"{args.dataset}_{args.prefix}"
    per_seed.round(4).to_csv(os.path.join(output_dir, f"{name}_per_seed.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, f"{name}_summary.csv"), index=False)
    comparison_table.to_csv(os.path.join(output_dir, f"{name}_joint_vs_specific.csv"), index=False)
    print(f"\nsaved results in {output_dir}")


if __name__ == "__main__":
    main()
