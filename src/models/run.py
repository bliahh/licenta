import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, average_precision_score, brier_score_loss
from tqdm import tqdm
import numpy as np

from src.evaluation.metrics import compute_metrics_for_binary
from src.utils.helpers import save_in_file, debug_ct_rm_prediction_separately

type_acq ={"ct":{"tn":0,"tp":0,"fn":0,"fp":0},"mr":{"tn":0,"tp":0,"fn":0,"fp":0}}

def run(model, loader, criterion, optimizer, device, is_train, threshold=0.5, desc="", scaler=None):

    save_in_file(desc, "/home/bliahh/Desktop/FACULATATE/licenta/output/outputs_vesselmask/fn_fp.txt")
    if is_train:
        model.train()
    else:
        model.eval()

    use_amp = device.type == "cuda"
    if is_train and use_amp and scaler is None:
        scaler = torch.amp.GradScaler("cuda")

    loss_sum = torch.zeros(1, device=device)

    n_batches = 0

    all_logits = []
    all_labels = []
    all_modalities = []
    all_paths = []
    pbar = tqdm(loader, desc=desc, leave=True)

    with torch.set_grad_enabled(is_train):
        for batch in pbar:
            #print(batch["image"].meta.get("filename_or_obj"))
            x = batch["image"]
            y = batch["label"]
            paths = batch["image"].meta["filename_or_obj"]
            modalities = ["ct" if "_ct_" in str(p).lower() else "mr" for p in paths]

            x = x.to(device, non_blocking=True)
            y = y.float().unsqueeze(1).to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            loss_sum += loss.detach()
            n_batches += 1

            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())
            all_modalities.extend(modalities)
            all_paths.extend(paths)

            pbar.set_postfix(loss=f"{loss.item():.3f}")

    logits_t = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy().ravel()
    probs = torch.sigmoid(logits_t).numpy().ravel()
    avg_loss = float(loss_sum / max(n_batches, 1))

    debug_ct_rm_prediction_separately(probs, labels, threshold,all_paths)


    metrics = compute_metrics_for_binary(
        probs=probs,
        labels=labels,
        threshold=threshold
    )
    return {
        "loss": avg_loss,
        **metrics,
        "logits": logits_t,
    }









