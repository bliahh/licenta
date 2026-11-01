import torch

from common.paths import DEVICE
from common.data.split import build_train_val_test_split, print_split_sizes
from patch_sliding.config import PRESETS, ACTIVE_PRESET, PATCH, SAMPLER, MODEL
from patch_sliding.data.loaders import sliding_window_loader
from patch_sliding.data.preprocess import preprocess_all_splits
from patch_sliding.evaluation.evaluator import evaluate_sliding_model
from patch_sliding.evaluation.groups import patch_groups_report, patch_summary
from patch_sliding.models.networks import ResnetArchitecture
from patch_sliding.models.run import run_sliding
from patch_sliding.models.trainer import train_sliding_window


def preprocess():
    """Saves every patient of every split in PREPROCESSED_DIR (patients already on disk are skipped)."""
    labeled_sample = build_train_val_test_split()
    print_split_sizes(labeled_sample)
    preprocess_all_splits(labeled_sample)
    return labeled_sample


def train(preset=ACTIVE_PRESET):
    """Trains a new run with the hyperparameters of PRESETS[preset] and evaluates it on TEST (threshold 0.5)."""
    hp = PRESETS[preset]
    labeled_sample = preprocess()

    model = ResnetArchitecture(dropout=hp["dropout"], **MODEL).to(DEVICE)
    print({type(m).__name__ for m in model.modules() if "Norm" in type(m).__name__})
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hp["pos_weight"], device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    run_config = {"preset": preset, "hyperparams": hp, "patch": PATCH, "sampler": SAMPLER, "model": MODEL}
    model = train_sliding_window(model=model, epochs=hp["epochs"], train_samples=labeled_sample["train"], val_samples=labeled_sample["val"], batch_size=hp["batch_size"], criterion=criterion, optimizer=optimizer, scaler=scaler, patience=hp["patience"], run_config=run_config)

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    print("TEST DATASET")
    test_loader = sliding_window_loader(samples=labeled_sample["test"], batch_size=hp["batch_size"], split="test", is_training=False)
    test_metrics = run_sliding(model, test_loader, criterion, optimizer=None, device=DEVICE, is_train=False, threshold=0.5, desc="TEST")

    print(f"TEST: loss={test_metrics['loss']:.3f} acc={test_metrics['acc']:.3f} f1={test_metrics['f1']:.3f} auc={test_metrics['auc']:.3f}")


def evaluate(run_name, checkpoints=("best_model.pth", "best_model_patient_auc.pth"), threshold_method="youden", beta=1):
    """Patient-level evaluation of one run, for every checkpoint (threshold_method: "youden" or "fbeta")."""
    for ckpt in checkpoints:
        evaluate_sliding_model(run_name, checkpoint=ckpt, threshold_method=threshold_method, beta=beta)


def groups(run_names, score_cols=("top5", "chosen"), checkpoints=("best_model", "best_model_patient_auc")):
    """Per-group TEST metrics of every run in run_names = {"description": "run_<date>"} (needs the patient_scores_*.csv written by evaluate)."""
    for score_col in score_cols:
        for ckpt in checkpoints:
            print("=" * 60, score_col, ckpt)
            for desc, run in run_names.items():
                patch_groups_report(run, checkpoint=ckpt, score_col=score_col, desc=desc)


def summary(run_names):
    """Patch-level TEST AUC of every run in run_names = {"description": "run_<date>"} -> out/tables/patch_summary.csv"""
    patch_summary(list(run_names.values()))


if __name__ == "__main__":
    train()
