import os
import json
import numpy as np
import pandas as pd
import torch

from common.paths import DEVICE
from common.data.split import build_train_val_test_split, print_split_sizes
from common.evaluation.metrics import compute_metrics_for_binary, confusion, clean_metrics
from common.evaluation.thresholds import find_th, youden_th
from patch_sliding.config import RUNS_DIR
from patch_sliding.data.loaders import sliding_window_loader
from patch_sliding.data.preprocess import preprocess_all_splits
from patch_sliding.evaluation.groups import patch_auc_by_group
from patch_sliding.models.networks import ResnetArchitecture
from patch_sliding.models.run import run_sliding


def load_run_config(run_dir):
    """config.json saved at training time (patch settings, model, hyperparameters)."""
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: only runs trained with patch_sliding.main.train can be evaluated")
    with open(path, "r") as f:
        return json.load(f)


def evaluate_sliding_model(run_name, checkpoint="best_model.pth", batch_size=8, pos_weight=1.5, beta=1, threshold_method="fbeta"):
    """
    Patient-level evaluation of one run: the aggregation and the threshold (F_beta or Youden) are chosen on VAL, then applied on TEST.
    Saves final_eval_<checkpoint>_<tag>.json and the per-patient scores of VAL and TEST in RUNS_DIR/<run_name>/evaluation/.
    """
    run_dir = os.path.join(RUNS_DIR, run_name)
    ckpt_path = os.path.join(run_dir, checkpoint)
    tag = f"F{beta}" if threshold_method == "fbeta" else "youden"
    out_path = os.path.join(run_dir, "evaluation", f"final_eval_{os.path.splitext(checkpoint)[0]}_{tag}.json")

    print("\n" + "=" * 60)
    print(f"EVALUATING: {ckpt_path}")
    print(f"THRESHOLD METHOD: {tag}")
    print("=" * 60)

    run_config = load_run_config(run_dir)
    patch = run_config["patch"]
    model_cfg = run_config["model"]

    labeled_sample = build_train_val_test_split()
    print_split_sizes(labeled_sample)
    preprocess_all_splits(labeled_sample)

    model = ResnetArchitecture(dropout=0.0, **model_cfg).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))

    print("VAL DATASET")
    val_loader = sliding_window_loader(samples=labeled_sample["val"], batch_size=batch_size, split="val", is_training=False, patch=patch)
    print("TEST DATASET")
    test_loader = sliding_window_loader(samples=labeled_sample["test"], batch_size=batch_size, split="test", is_training=False, patch=patch)

    va = run_sliding(model, val_loader, criterion, optimizer=None, device=DEVICE, is_train=False, threshold=0.5, desc="VAL")
    te = run_sliding(model, test_loader, criterion, optimizer=None, device=DEVICE, is_train=False, threshold=0.5, desc="TEST")
    patch_groups = patch_auc_by_group(te, test_loader, labeled_sample["test"], "test")

    n_val, n_test = len(va["labels"]), len(te["labels"])

    print("\n" + "=" * 60)
    print(f"PATIENT LEVEL - AGGREGATION SELECTION ON VAL ({n_val} patients, threshold = {tag} on val)")
    print("=" * 60)
    selection = {}
    for name, val_scores in va["agg_probs"].items():
        if threshold_method == "fbeta":
            thr, crit = find_th(val_scores, va["labels"], beta=beta)
        else:
            thr, crit = youden_th(val_scores, va["labels"])

        val_m = compute_metrics_for_binary(probs=np.asarray(val_scores), labels=va["labels"], threshold=thr)
        val_cm = confusion(val_scores, va["labels"], thr)
        selection[name] = {"threshold": thr, f"val_{tag}": crit, "val": {**clean_metrics(val_m), **val_cm}}
        print(f"[VAL {name:<6}] auc={val_m['auc']:.3f} thr={thr:.4f} {tag}={crit:.3f} sens={val_cm['sensitivity']:.3f} spec={val_cm['specificity']:.3f}")

    chosen = max(selection, key=lambda n: selection[n]["val"]["auc"])
    chosen_thr = selection[chosen]["threshold"]
    print(f"\n[CHOSEN ON VAL] aggregation={chosen} | threshold={chosen_thr:.4f}")

    print("\n" + "=" * 60)
    print(f"PATIENT LEVEL - FINAL TEST EVALUATION ({n_test} patients, aggregation + threshold chosen on VAL)")
    print("=" * 60)
    test_final = compute_metrics_for_binary(probs=np.asarray(te["agg_probs"][chosen]), labels=te["labels"],
                                            threshold=chosen_thr)
    test_cm = confusion(te["agg_probs"][chosen], te["labels"], chosen_thr)
    for k, v in clean_metrics(test_final).items():
        print(f"{k}: {v:.4f}")
    print(f"sensitivity: {test_cm['sensitivity']:.4f}")
    print(f"specificity: {test_cm['specificity']:.4f}")
    print("\nConfusion Matrix (patients):")
    print(f"TN: {test_cm['tn']}  FP: {test_cm['fp']}")
    print(f"FN: {test_cm['fn']}  TP: {test_cm['tp']}")

    print("\n" + "-" * 60)
    print("PATIENT LEVEL - ALL AGGREGATIONS ON TEST")
    print("-" * 60)
    test_all = {}
    for name, test_scores in te["agg_probs"].items():
        m = compute_metrics_for_binary(probs=np.asarray(test_scores), labels=te["labels"], threshold=selection[name]["threshold"])
        cm = confusion(test_scores, te["labels"], selection[name]["threshold"])
        test_all[name] = {**clean_metrics(m), **cm}
        print(
            f"[TEST {name:<6}] auc={m['auc']:.3f} sens={cm['sensitivity']:.3f} spec={cm['specificity']:.3f} TP={cm['tp']} FN={cm['fn']} FP={cm['fp']} TN={cm['tn']}")

    results = {"checkpoint": ckpt_path, "threshold_method": tag, "chosen_aggregation": chosen,
               "chosen_threshold": chosen_thr, "val_selection": selection,
               "test_final": {**clean_metrics(test_final), **test_cm}, "test_patch_metrics": clean_metrics(te["patch_metrics"]),
               "test_patch_groups": patch_groups, "test_all_aggregations": test_all}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    for split_name, res in (("val", va), ("test", te)):
        split_samples = labeled_sample[split_name]
        assert len(split_samples) == len(res["labels"]), f"{split_name}: {len(split_samples)} samples vs {len(res['labels'])} scored patients"
        assert all(int(s["label"]) == int(l) for s, l in zip(split_samples, res["labels"])), f"{split_name}: patient order differs"
        scores = pd.DataFrame({"image": [s["image"] for s in split_samples], "label": [int(l) for l in res["labels"]], "type_acq": [s["type_acq"] for s in split_samples]})
        for name, probs in res["agg_probs"].items():
            scores[name] = np.asarray(probs)
        scores["chosen"] = scores[chosen]
        csv_path = os.path.join(os.path.dirname(out_path), f"patient_scores_{split_name}_{os.path.splitext(checkpoint)[0]}.csv")
        scores.to_csv(csv_path, index=False)
        print(f"[SAVED] csv:{csv_path}")
    return results
