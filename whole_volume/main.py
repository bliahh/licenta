import os
import json

from whole_volume.config import ARCHITECTURES, OUT_DIR
from whole_volume.data.loaders import build_loaders
from whole_volume.evaluation.analysis import evaluate_volume_models, thick_slice_experiment
from whole_volume.evaluation.evaluator import evaluate_models
from whole_volume.models.trainer import train_final
from whole_volume.models.tuning.hyperparams import optimize
from whole_volume.utils.helpers import get_loss_names, get_results_path, json_to_csv

ALL_RESULTS_PATH = os.path.join(OUT_DIR, "comparison_results_TEST_TUNED_ALL.json")


def tune_and_train(architectures=ARCHITECTURES):
    """Optuna search + final training for every architecture and loss; best_params and val AUC go to comparison_<loss>.json."""
    for loss_name in get_loss_names():
        results_path = get_results_path(loss_name)
        results = {}
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)

        for name in architectures:
            train_loader, val_loader, _ = build_loaders(name)
            params = optimize(name, loss_name, train_loader, val_loader)
            _, best_val_auc = train_final(name, loss_name, params, train_loader, val_loader)
            results[name] = {**results.get(name, {}), "loss": loss_name, "best_params": params, "val_auc": best_val_auc}

            os.makedirs(OUT_DIR, exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[SAVED] {results_path}")


def evaluate_all():
    """TEST evaluation with thresholds 0.5 / F1 / F2 chosen on VAL, for every loss."""
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION")
    print("TEST + THRESHOLD TUNING")
    print("=" * 70)

    all_results = {loss_name: evaluate_models(loss_name) for loss_name in ("focal", "bce")}

    with open(ALL_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS FINISHED")
    print("=" * 70)

    print("\nAll results saved to:")
    print(ALL_RESULTS_PATH)


def evaluate_groups():
    """TEST metrics per patient group (CT / MR / thin / thick / center1), for every loss."""
    for loss_name in ("bce", "focal"):
        evaluate_volume_models(loss_name)


def thick_slice(loss_name="focal", name="resnet10", thick_mm=0.7):
    thick_slice_experiment(loss_name, name, thick_mm=thick_mm)


def export_csv():
    """comparison_results_TEST_TUNED_ALL.json -> comparison.csv"""
    json_to_csv(ALL_RESULTS_PATH, os.path.join(OUT_DIR, "comparison.csv"))


if __name__ == "__main__":
    thick_slice("focal", "resnet10", thick_mm=0.7)
