import os
import time
import numpy as np
import torch
import optuna
from src.config import EVAL_DIR, N_POS_TRAIN, N_NEG_TRAIN, FINAL_EPOCHS, DEVICE, PATIENCE,OUTPUT_SAVE_PATH
from src.data.loaders import Sliding_window_loader
from src.evaluation.plots import plot_metrics
from src.models.loss import build_loss, get_loss_params
from src.models.networks import build_model
from src.models.run import run, run_sliding
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

def train_sliding_window(model, epochs, train_samples, val_samples, batch_size=8, criterion=None, optimizer=None, scaler=None, patience=10, run_name=None):
    print("\n" + "=" * 60)
    print("SLIDING WINDOW TRAINING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_SAVE_PATH, run_name)
    os.makedirs(run_dir, exist_ok=True)
    best_model_path = os.path.join(run_dir, "best_model.pth")
    best_patient_path = os.path.join(run_dir, "best_model_patient_auc.pth")
    eval_dir = os.path.join(run_dir, "evaluation")

    print(f"RUN DIR: {run_dir}")
    print(f"TRAIN SAMPLES: {len(train_samples)}")
    print(f"VAL SAMPLES: {len(val_samples)}")

    print("TRAIN DATASET")
    train_loader = Sliding_window_loader(samples=train_samples, batch_size=batch_size, split="train", is_training=True)
    print("VAL DATASET")
    val_loader = Sliding_window_loader(samples=val_samples, batch_size=batch_size, split="val", is_training=False)

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

        metrics_val["loss_val"].append(vl["loss"])
        metrics_val["auc_val"].append(vl["auc"])
        metrics_val["f1_val"].append(vl["f1"])
        metrics_val["acc_val"].append(vl["acc"])

        metrics_train["loss_train"].append(tr["loss"])
        metrics_train["auc_train"].append(tr["auc"])
        metrics_train["f1_train"].append(tr["f1"])
        metrics_train["acc_train"].append(tr["acc"])

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


