import json
import os

import torch
from src.config import DEVICE, OUTPUT_DIR, ARCHITECTURES
from src.data.loaders import build_loaders
from src.models.loss import get_loss_params, build_loss
from src.models.networks import build_model
from src.evaluation.plots import save_confusion_matrix, plot_hyst
from src.models.run import run, run_sliding
from src.models.tuning.hyperparams import find_threshold
from src.utils.helpers import get_eval_dir, get_model_dir, get_results_path


def evaluate(name, loss_name, save_path, params, test_loader, threshold=0.5, eval_dir=None,description=""):
    if eval_dir is None:
        eval_dir = get_eval_dir(loss_name)

    print(f"\n[INFO] Loading {name}/{loss_name}")
    model = build_model(name, params["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()

    pos_weight, focal_alpha, focal_gamma = get_loss_params(loss_name, params)
    criterion = build_loss(loss_name=loss_name, pos_weight=pos_weight, focal_alpha=focal_alpha, focal_gamma=focal_gamma, device=DEVICE)

    print(f"[INFO] Evaluating {name}/{loss_name} threshold={threshold:.3f}")

    test_metrics = run_sliding(model, test_loader, criterion, optimizer=None, device=DEVICE, threshold=threshold, is_train=False, desc=f"TEST {name} [{loss_name}]")


    print(f"\n[RESULTS {name}/{loss_name}]")
    for metric, value in test_metrics.items():
        if metric not in ("labels", "preds", "logits"):
            print(f"  {metric}: {float(value):.3f}")
    save_confusion_matrix(name, test_metrics, eval_dir,description=description)
    plot_hyst(name, test_metrics["logits"], test_metrics["labels"], finded_thresh=threshold, eval_dir=eval_dir)
    return test_metrics



def evaluate_models(loss_name):

    print("\n" + "=" * 70)
    print(f"EVALUATION + THRESHOLD TUNING - {loss_name.upper()}")
    print("=" * 70)


    params_path = get_results_path(loss_name)

    print(f"\n[PARAMS] Reading:")
    print(params_path)

    if not os.path.exists(params_path):
        print(f"\n[ERROR] Results file not found:")
        print(params_path)
        return {}

    with open(params_path, "r") as f:
        original_results = json.load(f)

    output_path = os.path.join(OUTPUT_DIR,f"comparison_{loss_name}_TEST_TUNED.json")

    results = {}

    for name in ARCHITECTURES:

        print("\n" + "=" * 70)
        print(f"EVALUATING {name} / {loss_name}")
        print("=" * 70)

        save_path = os.path.join(get_model_dir(loss_name),f"{name}.pth")

        print(f"\n[MODEL]")
        print(save_path)

        if not os.path.exists(save_path):
            print("[MISSING] Model not found.")
            continue

        if name not in original_results:
            print(f"[MISSING] No saved results for {name}.")
            continue

        model_results = original_results[name]

        if "best_params" not in model_results:
            print(f"[MISSING] best_params not found for {name}.")
            continue

        params = model_results["best_params"]

        print("\n[PARAMS]")
        print(json.dumps(params, indent=2))


        print("\n[DATA] Building loaders...")

        _, val_loader, test_loader = build_loaders(name)
        print("\n" + "-" * 60)
        print(f"FINDING F1 THRESHOLD - {name}/{loss_name}")
        print("-" * 60)

        threshold_f1, val_f1 = find_threshold(name,loss_name,params,val_loader,beta=1,)

        print(f"[THRESHOLD F1] ")
        print(f"{threshold_f1:.4f}")

        print(f"[VAL F1] ")
        print(f"{val_f1:.4f}")

        print("\n" + "-" * 60)
        print(f"FINDING F2 THRESHOLD - {name}/{loss_name}")
        print("-" * 60)

        threshold_f2, val_f2 = find_threshold(name,loss_name,params,val_loader,beta=2,)

        print(f"[THRESHOLD F2] ")
        print(f"{threshold_f2:.4f}")

        print(f"[VAL F2] ")
        print(f"{val_f2:.4f}")


        thresholds = [("normal", 0.5),("F1opt", threshold_f1),("F2opt", threshold_f2)]
        results[name] = {}

        for threshold_name, threshold in thresholds:

            print("\n" + "=" * 60)
            print(f"TEST {name}/{loss_name} ")
            print(f"[{threshold_name}]")
            print(f"threshold = {threshold:.4f}")
            print("=" * 60)


            test_metrics = evaluate(name,loss_name,save_path,params,test_loader,threshold=threshold,eval_dir=get_eval_dir(loss_name),description=threshold_name)


            results[name][threshold_name] = {
                "loss": loss_name,
                "threshold": threshold,
                "best_params": params,
                "val_f1": val_f1 if threshold_name == "F1opt" else None,
                "val_f2": val_f2 if threshold_name == "F2opt" else None,
                "test_auc": test_metrics["auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_recall": test_metrics["recall"],
                "test_precision": test_metrics["precision"],
                "test_f1": test_metrics["f1"],
                "test_f2": test_metrics["f2"],
                "test_acc": test_metrics["acc"],
                "test_brier_score": test_metrics["brier_score"],
            }

            print(
                f"\n[TEST RESULTS]\n"
                f"Threshold : {threshold_name:.4f}\n"
                f"AUC       : {test_metrics['auc']:.4f}\n"
                f"PR-AUC    : {test_metrics['pr_auc']:.4f}\n"
                f"Recall    : {test_metrics['recall']:.4f}\n"
                f"Precision : {test_metrics['precision']:.4f}\n"
                f"F1        : {test_metrics['f1']:.4f}\n"
                f"F2        : {test_metrics['f2']:.4f}\n"
                f"Accuracy  : {test_metrics['acc']:.4f}\n"
                f"Brier     : {test_metrics['brier_score']:.4f}"
            )


        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[DONE] {name}")


    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"RESULTS SAVED:")
    print(output_path)
    print("=" * 70)

    return results



