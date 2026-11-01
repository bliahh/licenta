import numpy as np
import torch
from tqdm import tqdm

from common.evaluation.metrics import compute_metrics_for_binary
from common.utils.io import save_in_file
from whole_volume.config import FN_FP_LOG
from whole_volume.utils.helpers import debug_ct_rm_prediction_separately


@torch.no_grad()
def predict(model, loader, device):
    """Sigmoid score and label of every patient, in loader order."""
    model.eval()
    probs, labels = [], []
    for batch in loader:
        logits = model(batch["image"].to(device))
        probs += torch.sigmoid(logits.float()).view(-1).cpu().tolist()
        labels += batch["label"].view(-1).tolist()
    return np.asarray(probs), np.asarray(labels).astype(int)


def run(model, loader, criterion, optimizer, device, is_train, threshold=0.5, desc="", scaler=None):

    save_in_file(desc, FN_FP_LOG)
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
