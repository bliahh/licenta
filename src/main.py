import os
import json
import numpy as np
import torch
from src.config import OUTPUT_DIR, LABELS_DIR, OUTPUT_SAVE_PATH
from src.data.loaders import  Sliding_window_loader
from src.data.preprocess import  preprocess_all_splits
from src.data.split import build_train_val_test_split
from src.evaluation.evaluator import evaluate_models
from src.evaluation.metrics import compute_metrics_for_binary, confusion
from src.models.networks import ResnetArchitecture
from src.models.run import run_sliding
from src.models.trainer import train_sliding_window
from src.models.tuning.hyperparams import find_th, youden_th


def sliding_window_pipeline():
    labeled_sample = build_train_val_test_split(labels_dir=LABELS_DIR, val_ratio=0.16, test_ratio=0.20, seed=42)

    print(f"Train: {len(labeled_sample['train'])}")
    print(f"Val: {len(labeled_sample['val'])}")
    print(f"Test: {len(labeled_sample['test'])}")

    preprocess_all_splits(labeled_sample)


    test_samples = labeled_sample["test"]

    batch_size = 8
    epochs = 30

    lr = 1e-4
    weight_decay = 1e-4
    dropout = 0.3662004056693042
    #pos_weight = 1.0649244065345442
    pos_weight = 1.5
    patience = 10

    # batch_size = 8
    # epochs = 50
    #
    # lr = 1.7432210823763044e-05
    # weight_decay = 1.01471589306398e-04
    # dropout = 0.3662004056693042
    # # pos_weight = 1.0649244065345442
    # pos_weight = 4
    # patience = 10


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResnetArchitecture(dropout=dropout).to(device)
    print({type(m).__name__ for m in model.modules() if "Norm" in type(m).__name__})
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None


    model = train_sliding_window(model=model, epochs=epochs, train_samples=labeled_sample["train"], val_samples=labeled_sample["val"], batch_size=batch_size, criterion=criterion, optimizer=optimizer, scaler=scaler, patience=patience)

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    print("TEST DATASET")

    test_loader = Sliding_window_loader(samples=labeled_sample["test"], batch_size=batch_size, split="test",is_training=False)

    test_metrics = run_sliding(model, test_loader, criterion, optimizer=None, device=device, is_train=False, threshold=0.5, desc="TEST")

    print(f"TEST: loss={test_metrics['loss']:.3f} acc={test_metrics['acc']:.3f} f1={test_metrics['f1']:.3f} auc={test_metrics['auc']:.3f}")


def _clean(metrics):
    return {k: float(v) for k, v in metrics.items() if k not in ("labels", "preds") and np.isscalar(v)}


def evaluate_sliding_model(run_name, checkpoint="best_model.pth", batch_size=8, pos_weight=1.5, beta=1, threshold_method="fbeta"):

    run_dir = os.path.join(OUTPUT_SAVE_PATH, run_name)
    ckpt_path = os.path.join(run_dir, checkpoint)
    tag = f"F{beta}" if threshold_method == "fbeta" else "youden"
    out_path = os.path.join(run_dir, "evaluation", f"final_eval_{os.path.splitext(checkpoint)[0]}_{tag}.json")

    print("\n" + "=" * 60)
    print(f"EVALUATING: {ckpt_path}")
    print(f"THRESHOLD METHOD: {tag}")
    print("=" * 60)

    labeled_sample = build_train_val_test_split(labels_dir=LABELS_DIR, val_ratio=0.16, test_ratio=0.20, seed=42)
    print(f"Train: {len(labeled_sample['train'])}")
    print(f"Val: {len(labeled_sample['val'])}")
    print(f"Test: {len(labeled_sample['test'])}")
    preprocess_all_splits(labeled_sample)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResnetArchitecture(dropout=0.0).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    print("VAL DATASET")
    val_loader = Sliding_window_loader(samples=labeled_sample["val"], batch_size=batch_size, split="val",is_training=False)
    print("TEST DATASET")
    test_loader = Sliding_window_loader(samples=labeled_sample["test"], batch_size=batch_size, split="test",is_training=False)

    va = run_sliding(model, val_loader, criterion, optimizer=None, device=device, is_train=False, threshold=0.5,desc="VAL")
    te = run_sliding(model, test_loader, criterion, optimizer=None, device=device, is_train=False, threshold=0.5,desc="TEST")
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
        selection[name] = {"threshold": thr, f"val_{tag}": crit, "val": {**_clean(val_m), **val_cm}}
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
    for k, v in _clean(test_final).items():
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
        test_all[name] = {**_clean(m), **cm}
        print(
            f"[TEST {name:<6}] auc={m['auc']:.3f} sens={cm['sensitivity']:.3f} spec={cm['specificity']:.3f} TP={cm['tp']} FN={cm['fn']} FP={cm['fp']} TN={cm['tn']}")

    results = {"checkpoint": ckpt_path, "threshold_method": tag, "chosen_aggregation": chosen,
               "chosen_threshold": chosen_thr, "val_selection": selection,
               "test_final": {**_clean(test_final), **test_cm}, "test_patch_metrics": _clean(te["patch_metrics"]),
               "test_all_aggregations": test_all}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {out_path}")
    return results




def main():

    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION")
    print("TEST + THRESHOLD TUNING")
    print("=" * 70)


    focal_results = evaluate_models("focal")


    bce_results = evaluate_models("bce")


    all_results = {
        "focal": focal_results,
        "bce": bce_results,
    }

    all_results_path = os.path.join(OUTPUT_DIR,"comparison_results_TEST_TUNED_ALL.json")

    with open(all_results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS FINISHED")
    print("=" * 70)

    print("\nAll results saved to:")
    print(all_results_path)


if __name__ == "__main__":
    for ckpt in ("best_model.pth", "best_model_patient_auc.pth"):
        evaluate_sliding_model("run_20260926_171449", checkpoint=ckpt, threshold_method="fbeta", beta=1)
        evaluate_sliding_model("run_20260926_171449", checkpoint=ckpt, threshold_method="youden")


