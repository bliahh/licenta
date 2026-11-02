import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from binary_classification.whole_volume_rsna.data.loaders import build_loader, load_split, load_points, VOLUME_SIZE, GRID_SIZE, points_to_roi
from binary_classification.whole_volume_rsna.models.network import ResNet18, ResNet18MultiTask
from binary_classification.whole_volume_rsna.models.run import run, predict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def heatmap_loss(heat_map_logits, heat_map_target, has_target, pos_weight=50.0):

    per_cell = torch.nn.functional.binary_cross_entropy_with_logits( heat_map_logits, heat_map_target, reduction="none", pos_weight=torch.tensor(pos_weight, device=heat_map_logits.device))
    per_volume = per_cell.mean(dim=(1, 2, 3, 4))

    return (per_volume * has_target).sum() / has_target.sum().clamp(min=1)


def localization_hit_rate(table, peaks, points, tolerance=1, roi_size=None):
    hits = []
    for uid, label, modality, peak in zip(table["uid"], table["label"], table["modality"], peaks):
        if label != 1 or uid not in points:
            continue
        series_points = points[uid] if roi_size is None else points_to_roi(points[uid], modality, roi_size)
        if len(series_points) == 0:
            hits.append(False)
            continue
        cell_size = (roi_size or VOLUME_SIZE) // GRID_SIZE
        cell_distance = np.abs(series_points // cell_size - peak).max(axis=1)
        hits.append(cell_distance.min() <= tolerance)
    return float(np.mean(hits)) if hits else float("nan")

def train(seed, only_mr=False, only_ct=False, run_dir="runs/test", epochs=30, lr=1e-4, batch_size=1, patience=7, min_epochs=10, dropout=0.3, num_workers=4, multitask=True, lambd=0.0, sigma=1.0, top_n=4, pos_weight=50.0, augm_proba=0.5, roi_size = None):
    set_seed(seed)

    train_table, val_table, _ = load_split(seed, only_mr=only_mr, only_ct=only_ct)


    train_loader = build_loader(train_table, is_training=True, batch_size=batch_size, num_workers=num_workers, use_amp=USE_AMP, multitask=multitask, sigma=sigma, augm_proba=augm_proba, roi_size=roi_size)
    val_loader = build_loader(val_table, is_training=False, batch_size=batch_size, num_workers=num_workers,  use_amp=USE_AMP, multitask=multitask, sigma=sigma, roi_size=roi_size)


    points = load_points()

    if multitask:
        model = ResNet18MultiTask(dropout=dropout, top_n=top_n, conv1_stride=1 if roi_size else 2).to(DEVICE)
    else:
        model = ResNet18(dropout=dropout).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    os.makedirs(run_dir, exist_ok=True)
    config = {"seed": seed, "only_mr": only_mr, "only_ct": only_ct,
              "epochs": epochs, "lr": lr, "batch_size": batch_size,
              "patience": patience, "min_epochs": min_epochs,
              "dropout": dropout, "multitask": multitask,
              "lambd": lambd, "sigma": sigma, "top_n": top_n,
              "pos_weight": pos_weight, "augm_proba": augm_proba,
              "n_train": len(train_table), "n_val": len(val_table),
              "roi_size": roi_size,
              }

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[ TRAIN ] {run_dir} | train {len(train_table)} | val {len(val_table)} | {DEVICE}")

    best_auc = 0
    no_increment = 0
    history = []
    best_path = os.path.join(run_dir, "best.pt")

    for epoch in range(epochs):
        start = time.time()

        train_loss, train_loss_cls, train_loss_heat = run(model, train_loader, criterion, scaler, optimizer, DEVICE, USE_AMP, desc=f"Epoch {epoch + 1}/{epochs}", heatmap_loss=heatmap_loss if multitask else None, lambd=lambd, pos_weight=pos_weight)

        val_logits, val_labels, val_peaks = predict(model, val_loader, device=DEVICE, use_amp=USE_AMP, return_peaks=True)
        val_loss = criterion(torch.tensor(val_logits), torch.tensor(val_labels)).item()
        val_auc = roc_auc_score(val_labels, val_logits)
        val_hit = localization_hit_rate(val_table, val_peaks, points, roi_size=roi_size) if multitask else float("nan")
        val_hit_exact = localization_hit_rate(val_table, val_peaks, points, tolerance=0, roi_size=roi_size) if multitask else float("nan")

        lr_saved = scheduler.get_last_lr()[0]
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "train_loss_cls": train_loss_cls, "train_loss_heat": train_loss_heat, "val_loss": val_loss, "val_auc": val_auc, "val_hit": val_hit, "lr": lr_saved, "seconds": round(time.time() - start, 1), "val_hit_exact": val_hit_exact})
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        print(f"[ TRAIN ] epoch {epoch + 1} | train_loss_binary_classif {train_loss_cls:.4f} | train_loss_loc {train_loss_heat:.4f} | val loss {val_loss:.4f} | val AUC {val_auc:.4f} | val hit {val_hit:.3f} | val hit exact {val_hit_exact:.3f}| scale {scaler.get_scale():.0f} | {history[-1]['seconds']} s")

        torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))

        if val_auc > best_auc:
            best_auc = val_auc
            no_increment = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_increment += 1
            if epoch + 1 >= min_epochs and no_increment >= patience:
                print(f"[ TRAIN ] stopped at {epoch}")
                break

    print(f"[ TRAIN ] best val AUC: {best_auc:.4f}")
    return best_auc, best_path