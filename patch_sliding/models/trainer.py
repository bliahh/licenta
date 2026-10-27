import os
import json
import time
import numpy as np
import torch

from common.paths import DEVICE
from common.evaluation.plots import plot_metrics
from common.utils.io import save_metrics
from patch_sliding.config import RUNS_DIR
from patch_sliding.data.loaders import sliding_window_loader
from patch_sliding.models.run import run_sliding


def train_sliding_window(model, epochs, train_samples, val_samples, batch_size=8, criterion=None, optimizer=None, scaler=None, patience=10, run_name=None, run_config=None):
    """
    Trains the patch classifier and keeps two checkpoints in RUNS_DIR/<run_name>/:
    best_model.pth (best val patch PR-AUC, used for early stopping) and best_model_patient_auc.pth (best val patient AUC).
    run_config (hyperparameters, PATCH, SAMPLER, MODEL) is saved as config.json, so the run can be evaluated with the same settings.
    """
    print("\n" + "=" * 60)
    print("SLIDING WINDOW TRAINING")
    print("=" * 60)

    device = DEVICE
    model.to(device)

    run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    run_config = run_config or {}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    best_model_path = os.path.join(run_dir, "best_model.pth")
    best_patient_path = os.path.join(run_dir, "best_model_patient_auc.pth")
    eval_dir = os.path.join(run_dir, "evaluation")

    print(f"RUN DIR: {run_dir}")
    print(f"TRAIN SAMPLES: {len(train_samples)}")
    print(f"VAL SAMPLES: {len(val_samples)}")

    print("TRAIN DATASET")
    train_loader = sliding_window_loader(samples=train_samples, batch_size=batch_size, split="train", is_training=True, patch=run_config.get("patch"), sampler=run_config.get("sampler"))
    print("VAL DATASET")
    val_loader = sliding_window_loader(samples=val_samples, batch_size=batch_size, split="val", is_training=False, patch=run_config.get("patch"))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_pr_auc = -1.0
    best_pr_auc_epoch = None
    best_patient_auc = -1.0
    best_patient_auc_epoch = None
    wait = 0

    metrics_val = {"loss_val": [], "auc_val": [], "f1_val": [], "acc_val": []}
    metrics_train = {"loss_train": [], "auc_train": [], "f1_train": [], "acc_train": []}
    metrics_patch = {"pr_auc_val": [], "auc_val": [], "pr_auc_train": [], "auc_train": [], "lr": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr = run_sliding(model=model, loader=train_loader, criterion=criterion, optimizer=optimizer, device=device, is_train=True, threshold=0.5, desc=f"Epoch {epoch}/{epochs} [train]", scaler=scaler)
        vl = run_sliding(model=model, loader=val_loader, criterion=criterion, optimizer=None, device=device, is_train=False, threshold=0.5, desc=f"Epoch {epoch}/{epochs} [val]")

        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        dt = time.time() - t0

        val_pr_auc = vl["patch_metrics"]["pr_auc"]
        improved = np.isfinite(val_pr_auc) and val_pr_auc > best_pr_auc

        if improved:
            best_pr_auc = val_pr_auc
            best_pr_auc_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            wait = 0
        else:
            wait += 1

        if np.isfinite(vl["auc"]) and vl["auc"] > best_patient_auc:
            best_patient_auc = vl["auc"]
            best_patient_auc_epoch = epoch
            torch.save(model.state_dict(), best_patient_path)

        print(f"[{epoch:02d}/{epochs}] {dt:.0f}s | lr={lr:.2e} | train: loss={tr['loss']:.3f} patient_auc={tr['auc']:.3f} | val: loss={vl['loss']:.3f} patch_pr_auc={val_pr_auc:.3f} patch_auc={vl['patch_metrics']['auc']:.3f} patient_auc={vl['auc']:.3f} f1={vl['f1']:.3f}{' <-- BEST' if improved else ''}", flush=True)

        for key in ("loss", "auc", "f1", "acc"):
            metrics_val[f"{key}_val"].append(vl[key])
            metrics_train[f"{key}_train"].append(tr[key])

        metrics_patch["pr_auc_val"].append(val_pr_auc)
        metrics_patch["auc_val"].append(vl["patch_metrics"]["auc"])
        metrics_patch["pr_auc_train"].append(tr["patch_metrics"]["pr_auc"])
        metrics_patch["auc_train"].append(tr["patch_metrics"]["auc"])
        metrics_patch["lr"].append(lr)

        if patience is not None and wait >= patience:
            print(f"[EARLY STOP] no improvement in val patch PR-AUC for {patience} epochs.")
            break

    if not os.path.exists(best_model_path):
        raise RuntimeError("No best model was saved during training.")

    os.makedirs(eval_dir, exist_ok=True)
    plot_metrics(metrics_train, metrics_val, eval_dir)
    save_metrics(metrics_train, eval_dir, filename="metrics_train.json")
    save_metrics(metrics_val, eval_dir, filename="metrics_val.json")
    save_metrics(metrics_patch, eval_dir, filename="metrics_patch.json")

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print(f"[FINISHED] best val patch PR-AUC={best_pr_auc:.3f} (epoch {best_pr_auc_epoch}) | best val patient AUC={best_patient_auc:.3f} (epoch {best_patient_auc_epoch})")
    print(f"[MODEL SAVED] {best_model_path}")
    print(f"[MODEL SAVED] {best_patient_path}")
    return model
