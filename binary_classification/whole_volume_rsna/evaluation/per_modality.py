import os
import argparse

import pandas as pd

from binary_classification.whole_volume_rsna.data.loaders import load_split, load_points
from binary_classification.whole_volume_rsna.models.run import predict
from binary_classification.whole_volume_rsna.models.train import localization_hit_rate
from binary_classification.whole_volume_rsna.evaluation.utils import (
    RESULTS_DIR, MODALITIES, METRICS, get_device, split_options, load_model, make_loader, run_dir_of,
    load_predictions, mean_std, threshold_metrics)


def localization_per_modality(run_dir, seed, name):
    device, use_amp = get_device()
    model, roi_size, n_outputs = load_model(run_dir, device)
    _, _, test_table = load_split(seed, **split_options(name))
    loader = make_loader(test_table, roi_size, use_amp, n_outputs=n_outputs)
    _, _, peaks = predict(model, loader, device, use_amp, return_peaks=True)
    points = load_points()

    results = {}
    for modality in MODALITIES:
        mask = (test_table["modality"] == modality).to_numpy()
        if not mask.any():
            continue
        subset = test_table[mask].reset_index(drop=True)
        hit = localization_hit_rate(subset, peaks[mask], points, roi_size=roi_size)
        hit_exact = localization_hit_rate(subset, peaks[mask], points, tolerance=0, roi_size=roi_size)
        results[modality] = (hit, hit_exact)
        print(f"{name} seed {seed}, {modality}: hit {hit:.3f}, exact {hit_exact:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--localization", action="store_true")
    parser.add_argument("--prefix", default="final")
    args = parser.parse_args()
    pd.set_option("display.width", 200)

    rows = []
    for name in [f"{args.prefix}_ct", f"{args.prefix}_mr", f"{args.prefix}_joint"]:
        for seed in args.seeds:
            val = load_predictions(args.results_dir, name, seed, "val")
            test = load_predictions(args.results_dir, name, seed, "test")
            hits = {}
            if args.localization:
                hits = localization_per_modality(run_dir_of(args.results_dir, name, seed), seed, name)

            for modality in MODALITIES:
                val_subset = val[val["modality"] == modality]
                test_subset = test[test["modality"] == modality]
                if len(test_subset) == 0:
                    continue
                row = {"experiment": name, "test_set": modality, "seed": seed, "n_test": len(test_subset)}
                row.update(threshold_metrics(val_subset["label"].to_numpy(), val_subset["logit"].to_numpy(),
                                             test_subset["label"].to_numpy(), test_subset["logit"].to_numpy()))
                if modality in hits:
                    row["hit"], row["hit_exact"] = hits[modality]
                rows.append(row)

    per_seed = pd.DataFrame(rows)
    columns = METRICS + [column for column in ["hit", "hit_exact"] if column in per_seed.columns]
    aggregations = {column: (column, mean_std) for column in columns}
    summary = per_seed.groupby(["test_set", "experiment"], sort=False).agg(
        n_test=("n_test", "first"), **aggregations).reset_index()
    summary = summary.sort_values(["test_set", "experiment"]).reset_index(drop=True)
    print("\nPer modality, threshold from the validation series of the same modality (mean +- std over seeds)")
    print(summary.to_string(index=False))

    tag = "" if args.prefix == "final" else f"_{args.prefix}"
    per_seed.round(4).to_csv(os.path.join(args.results_dir, f"per_modality_per_seed{tag}.csv"), index=False)
    summary.to_csv(os.path.join(args.results_dir, f"per_modality_summary{tag}.csv"), index=False)
    print(f"\nsaved results in {args.results_dir}")


if __name__ == "__main__":
    main()
