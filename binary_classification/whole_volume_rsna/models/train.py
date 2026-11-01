import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from binary_classification.whole_volume_rsna.data.loaders import build_loader, load_split
from binary_classification.whole_volume_rsna.models.network import ResNet18
from binary_classification.whole_volume_rsna.models.run import run, predict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(seed, only_mr=False, only_ct=False, run_dir="runs/test", epochs=30, lr=1e-4, batch_size=1, patience=30, dropout=0.0, num_workers=4):

    set_seed(seed)

    train_table, val_table, _ = load_split(seed, only_mr=only_mr, only_ct=only_ct)
    train_loader = build_loader(train_table, is_training=True, batch_size=batch_size, num_workers=num_workers, use_amp=USE_AMP)
    val_loader = build_loader(val_table, is_training=False, batch_size=batch_size, num_workers=num_workers, use_amp=USE_AMP)




    model = ResNet18(dropout=dropout).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    os.makedirs(run_dir, exist_ok=True)

    os.makedirs(run_dir, exist_ok=True)
    config = {"seed": seed,
              "only_mr": only_mr,
              "only_ct": only_ct,
              "epochs": epochs,
              "lr": lr,
              "batch_size": batch_size,
              "patience": patience,
              "dropout": dropout,
              "n_train": len(train_table),
              "n_val": len(val_table)}

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[ TRAIN ] {run_dir} | train {len(train_table)} | val {len(val_table)} | {DEVICE}")


    best_auc = 0
    no_increment = 0
    history = []


    best_path = os.path.join(run_dir, "best.pt")

    for epoch in range(epochs):
        start = time.time()

        train_loss = run(model, train_loader, criterion, scaler, optimizer, DEVICE, USE_AMP,desc=f"Epoch {epoch + 1}/{epochs}")

        val_logits, val_labels = predict(model, val_loader, device=DEVICE, use_amp=USE_AMP)
        val_loss = criterion(torch.tensor(val_logits), torch.tensor(val_labels)).item()
        val_auc = roc_auc_score(val_labels, val_logits)

        lr_saved = scheduler.get_last_lr()[0]
        scheduler.step()

        history.append({"epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_auc": val_auc,
                        "lr": lr_saved,
                        "seconds": round(time.time() - start, 1)})

        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        print(f"[ TRAIN ] epoch: {epoch} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
              f"| val AUC {val_auc:.4f} | {history[-1]['seconds']} s")
        print(f"... | scale {scaler.get_scale():.0f}")

        torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))

        if val_auc > best_auc:
            best_auc = val_auc
            no_increment = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_increment += 1
            if no_increment >= patience:
                print(f"[ TRAIN ] stopped at {epoch}")
                break

    print(f"[ TRAIN ] best val AUC: {best_auc:.4f}")
    return best_auc, best_path