import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from common.evaluation.metrics import compute_metrics_for_binary
from patch_sliding.models.aggregation import AGGREGATORS, aggregate_patches


def run_sliding(model, loader, criterion, optimizer, device, is_train, threshold=0.5, desc="", scaler=None, max_grad_norm=1.0):
    model.train() if is_train else model.eval()
    use_amp = device.type == "cuda"
    if is_train and use_amp and scaler is None:
        scaler = torch.amp.GradScaler("cuda")

    loss_sum = 0.0
    n_batches = 0
    all_logits, all_labels = [], []
    all_patient_labels, all_sample_indices, all_indices = [], [], []
    pbar = tqdm(loader, desc=desc, leave=True)

    with torch.set_grad_enabled(is_train):
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].float().to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x).squeeze(1)

            valid = y >= 0
            loss = criterion(logits.float()[valid], y[valid]) if valid.any() else logits.sum() * 0.0

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            loss_sum += loss.item()
            n_batches += 1
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(y.detach().cpu())
            all_sample_indices.extend(batch["sample_index"].cpu().tolist())
            all_patient_labels.extend(batch["patient_label"].cpu().tolist())
            if "index" in batch:
                all_indices.extend(batch["index"].cpu().tolist())
            pbar.set_postfix(loss=f"{loss.item():.3f}")

    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels)
    probs_patch = torch.sigmoid(logits_t).numpy()
    avg_loss = loss_sum / max(n_batches, 1)
    valid = labels_t >= 0
    patch_metrics = metrics_for_patches(logits=logits_t[valid], y=labels_t[valid], threshold=threshold, batch_idx=None, loss=avg_loss)
    agg_results = {}
    agg_probs = {}
    for name, fn in AGGREGATORS.items():
        patient_probs, patient_labels = aggregate_patches(probs_patch=probs_patch, patient_labels=all_patient_labels, all_sample_index=all_sample_indices, fn=fn)
        patient_probs = np.asarray(patient_probs)
        agg_probs[name] = patient_probs
        agg_results[name] = compute_metrics_for_binary(probs=patient_probs, labels=patient_labels, threshold=threshold)
        print(f"[AGG {name:<6}] auc={agg_results[name]['auc']:.3f} | "
              f"f1={agg_results[name]['f1']:.3f} | "
              f"accuracy={agg_results[name]['acc']:.3f} | "
              f"f2={agg_results[name]['f2']:.3f} | "
              f"pr_auc = {agg_results[name]['pr_auc']:.3f} | "
              f"recall={agg_results[name]['recall']:.3f} | "
              f"precision={agg_results[name]['precision']:.3f} | "
              f"brier_score={agg_results[name]['brier_score']:.3f} ")


    patient_probs = agg_probs["max"]
    metrics = agg_results["max"]

    return {"loss": avg_loss, **metrics, "patch_metrics": patch_metrics, "patient_probs": patient_probs, "logits": torch.tensor(patient_probs), "labels": patient_labels, "preds": (patient_probs >= threshold).astype(int), "patch_probs": probs_patch, "patch_indices": np.asarray(all_indices), "agg_results": agg_results, "agg_probs": agg_probs}


def metrics_for_patches(logits, y, threshold, batch_idx=None, loss=None):
    print("\n" + "=" * 60)
    print("PATCH-LEVEL METRICS")
    print("=" * 60)

    patch_probs = torch.sigmoid(logits.detach()).cpu().numpy().ravel()
    patch_labels = y.detach().cpu().numpy().ravel()
    patch_metrics = compute_metrics_for_binary(probs=patch_probs, labels=patch_labels, threshold=threshold)
    tn, fp, fn, tp = confusion_matrix(patch_metrics["labels"], patch_metrics["preds"], labels=[0, 1]).ravel()

    print(f"\nTotal patches: {len(patch_labels)}")
    if loss is not None:
        print(f"Loss: {float(loss):.4f}")
    for metric, value in patch_metrics.items():
        if metric in ("labels", "preds"):
            continue
        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    print("=" * 60)
    return patch_metrics
