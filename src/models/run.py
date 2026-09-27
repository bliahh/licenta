from functools import partial

import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
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
    aggregators = {"max": arg_max_prob, "top2": partial(top_k_prob, k=2), "top3": partial(top_k_prob, k=3),
                   "top5": partial(top_k_prob, k=5), "top7": partial(top_k_prob, k=7),
                   "top10": partial(top_k_prob, k=10), "p90": partial(k_percentile_prob, k=0.9),
                   "p75": partial(k_percentile_prob, k=0.75), "median": median_prob, "mean": mean_prob}
    agg_results = {}
    agg_probs = {}
    for name, fn in aggregators.items():
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

        # "acc": accuracy,
        # "f1": f1,
        # "f2": f2,
        # "auc": auc,
        # "pr_auc": pr_auc,
        # "recall": recall,
        # "precision": precision,
        # "brier_score": brier_score,
        # "labels": labels,
        # "preds": preds,

    patient_probs = agg_probs["max"]
    metrics = agg_results["max"]

    return {"loss": avg_loss, **metrics, "patch_metrics": patch_metrics, "patient_probs": patient_probs, "logits": torch.tensor(patient_probs), "labels": patient_labels, "preds": (patient_probs >= threshold).astype(int), "patch_probs": probs_patch, "patch_indices": np.asarray(all_indices), "agg_results": agg_results, "agg_probs": agg_probs}
def aggregate_patches(probs_patch, patient_labels, all_sample_index,fn, threshold=0.5, ):
    patient_proba = {}
    patient_label = {}
    for proba, label, sample_index in zip(probs_patch, patient_labels, all_sample_index):
        patient_proba.setdefault(sample_index, []).append(proba)
        patient_label[sample_index] = label

    patient_proba = {index: fn(probas) for index, probas in patient_proba.items()}
    patient_idx = sorted(patient_proba.keys())
    probs = np.array([patient_proba[idx] for idx in patient_idx])
    labels = np.array([patient_label[idx] for idx in patient_idx])
    return probs, labels


def arg_max_prob(probas):
    return float(np.max(probas))

def top_k_prob(probas,k):
    return float(np.average(sorted(probas,reverse=True)[:k]))

def mean_prob(probas):
    return float(np.average(probas))

def median_prob(probas):
    return float(np.median(probas))

def k_percentile_prob(probas,k):
    return float(np.quantile(probas,q=k))


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

