import os
import time
import numpy as np
import torch
import optuna
from src.config import EVAL_DIR, N_POS_TRAIN, N_NEG_TRAIN, FINAL_EPOCHS, DEVICE, PATIENCE
from src.evaluation.plots import plot_metrics
from src.models.loss import build_loss, get_loss_params
from src.models.networks import build_model
from src.models.run import run
from src.utils.helpers import get_model_dir, get_eval_dir, save_metrics


def train_model(name, train_loader, val_loader, n_pos, n_neg, loss_name, pos_weight=None, focal_alpha=0.5, focal_gamma=2.0, lr=1e-4, weight_decay=1e-5, epochs=30, dropout=0.0, device=None, save_path="best_model.pth", trial=None, patience=None, eval_dir=None):

    if eval_dir is None:
        eval_dir = EVAL_DIR

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[TRAIN] model={name} | loss={loss_name}")

    if loss_name == "bce":
        print(f"[TRAIN] pos_weight={pos_weight}")
    elif loss_name == "focal":
        print(f"[TRAIN] alpha={focal_alpha} | gamma={focal_gamma}")

    print(f"[TRAIN] device={device}")

    model = build_model(name, dropout).to(device)
    criterion = build_loss(loss_name=loss_name, pos_weight=pos_weight, focal_alpha=focal_alpha, focal_gamma=focal_gamma, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    best_auc = -1.0
    best_val_loss = float("inf")
    wait = 0
    metrics_val = {"loss_val": [], "auc_val": [], "f1_val": [], "acc_val": []}
    metrics_train = {"loss_train": [], "auc_train": [], "f1_train": [], "acc_train": []}


    for epoch in range(1, epochs + 1):
        t0 = time.time()


        tr = run(model, train_loader, criterion, optimizer, device, is_train=True, threshold=0.5, desc=f"Epoch {epoch}/{epochs} [train]", scaler=scaler)


        vl = run(model, val_loader, criterion, optimizer=None, device=device, is_train=False, threshold=0.5, desc=f"Epoch {epoch}/{epochs} [val]")

        dt = time.time() - t0


        val_auc_is_valid = np.isfinite(vl["auc"])

        if val_auc_is_valid:
            improved = vl["auc"] > best_auc
        else:
            improved = vl["loss"] < best_val_loss

        print(f"[{epoch:02d}/{epochs}] {dt:.0f}s | train: loss={tr['loss']:.3f} auc={tr['auc']:.3f} | "
              f"val: loss={vl['loss']:.3f} acc={vl['acc']:.3f} f1={vl['f1']:.3f} auc={vl['auc']:.3f}"
              f"{' <-- BEST' if improved else ''}", flush=True)


        if improved:
            if val_auc_is_valid:
                best_auc = vl["auc"]

            best_val_loss = vl["loss"]

            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.save(model.state_dict(), save_path)

            wait = 0
        else:
            wait += 1


        if trial is not None:
            report_auc = vl["auc"] if np.isfinite(vl["auc"]) else 0.0

            trial.report(report_auc, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        metrics_val["loss_val"].append(vl["loss"])
        metrics_val["auc_val"].append(vl["auc"])
        metrics_val["f1_val"].append(vl["f1"])
        metrics_val["acc_val"].append(vl["acc"])

        metrics_train["loss_train"].append(tr["loss"])
        metrics_train["auc_train"].append(tr["auc"])
        metrics_train["f1_train"].append(tr["f1"])
        metrics_train["acc_train"].append(tr["acc"])


        if patience is not None and wait >= patience:
            print(f"[EARLY STOP] no improvement for {patience} epochs.")
            break


    arch_eval_dir = os.path.join(eval_dir, name)
    os.makedirs(arch_eval_dir, exist_ok=True)

    plot_metrics(metrics_train, metrics_val, arch_eval_dir)

    save_metrics(metrics_train, arch_eval_dir, filename="metrics_train.json")
    save_metrics(metrics_val, arch_eval_dir, filename="metrics_val.json")

    print(f"[FINISHED {name}] loss={loss_name} | best val AUC={best_auc:.3f}")

    return model, best_auc


def train_final(name, loss_name, params, train_loader, val_loader):
    print("\n" + "=" * 60)
    print(f"FINAL MODEL: {name} / {loss_name}")
    print("=" * 60)

    model_dir = get_model_dir(loss_name)
    eval_dir = get_eval_dir(loss_name)
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"{name}.pth")

    pos_weight, focal_alpha, focal_gamma = get_loss_params(loss_name, params)

    model, best_val_auc = train_model(name=name, train_loader=train_loader, val_loader=val_loader, n_pos=N_POS_TRAIN, n_neg=N_NEG_TRAIN, loss_name=loss_name, pos_weight=pos_weight, focal_alpha=focal_alpha, focal_gamma=focal_gamma, lr=params["lr"], weight_decay=params["weight_decay"], epochs=FINAL_EPOCHS, dropout=params["dropout"], device=DEVICE, save_path=save_path, patience=PATIENCE, eval_dir=eval_dir)
    return save_path, best_val_auc
