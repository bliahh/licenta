import os
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from binary_classification.whole_volume_rsna.data.loaders import load_split, build_loader
from binary_classification.whole_volume_rsna.models.network import ResNet18MultiTask
from binary_classification.whole_volume_rsna.models.run import predict

RESULTS_DIR = os.path.expanduser("~/Desktop/FACULATATE/licenta/results")
EXTERNAL_DIR = os.environ.get("EXTERNAL_DIR", os.path.expanduser("~/Desktop/FACULATATE/licenta/external_test"))

MODALITIES = ["CTA", "MRA"]
SEPARATE_MODELS = {"CTA": "final_ct", "MRA": "final_mr"}
JOINT_MODEL = "final_joint"
METRICS = ["auc", "sensitivity", "specificity", "balanced_accuracy", "f1"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    return torch.device("cpu"), False


def split_options(name):
    if name.endswith("_ct"):
        return {"only_ct": True}
    if name.endswith("_mr"):
        return {"only_mr": True}
    return {}


def load_model(run_dir, device, load_weights=True):
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    roi_size = config.get("roi_size")
    n_outputs = config.get("n_outputs", 1)

    stride = 1 if roi_size else 2
    model = ResNet18MultiTask(dropout=config["dropout"], top_n=config["top_n"], conv1_stride=stride,
                              n_outputs=n_outputs)
    if load_weights:
        weights = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu")
        model.load_state_dict(weights)

    model = model.to(device)
    model.eval()
    return model, roi_size, n_outputs


def make_loader(table, roi_size, use_amp, batch_size=2, n_outputs=1):
    return build_loader(table, is_training=False, batch_size=batch_size, num_workers=4,
                        use_amp=use_amp, multitask=True, roi_size=roi_size, n_outputs=n_outputs)


def run_dir_of(results_dir, name, seed):
    return os.path.join(results_dir, f"{name}_seed{seed}")


def load_predictions(results_dir, name, seed, split):
    run_dir = run_dir_of(results_dir, name, seed)
    path = os.path.join(run_dir, f"predictions_{split}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    if split != "val":
        raise SystemExit(f"missing {path}")

    device, use_amp = get_device()
    model, roi_size, n_outputs = load_model(run_dir, device)
    _, val_table, _ = load_split(seed, **split_options(name))
    loader = make_loader(val_table, roi_size, use_amp, n_outputs=n_outputs)
    logits, labels = predict(model, loader, device, use_amp)
    if logits.ndim == 2:
        logits, labels = logits[:, 0], labels[:, 0]

    predictions = pd.DataFrame({
        "uid": val_table["uid"].values,
        "modality": val_table["modality"].values,
        "label": labels.astype(int),
        "logit": logits,
    })
    predictions.to_csv(path, index=False)
    print(f"computed val predictions for {name} seed {seed}: n={len(predictions)}, "
          f"auc={roc_auc_score(labels, logits):.3f}")
    return predictions


def mean_std(values):
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return ""
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{values.mean():.3f} +- {values.std(ddof=1):.3f}"


def safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, scores)


def paired_bootstrap(labels, scores_a, scores_b, repetitions, rng):
    n = len(labels)
    differences = []
    for _ in range(repetitions):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        auc_a = roc_auc_score(labels[idx], scores_a[idx])
        auc_b = roc_auc_score(labels[idx], scores_b[idx])
        differences.append(auc_a - auc_b)
    low, high = np.percentile(differences, [2.5, 97.5])
    return low, high


def threshold_metrics(val_labels, val_scores, test_labels, test_scores):
    fpr, tpr, thresholds = roc_curve(val_labels, val_scores)
    threshold = thresholds[np.argmax(tpr - fpr)]

    pred = test_scores >= threshold
    tp = np.sum(pred & (test_labels == 1))
    fp = np.sum(pred & (test_labels == 0))
    tn = np.sum(~pred & (test_labels == 0))

    sensitivity = tp / np.sum(test_labels == 1)
    specificity = tn / np.sum(test_labels == 0)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    if precision + sensitivity > 0:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    else:
        f1 = 0.0

    return {
        "auc": roc_auc_score(test_labels, test_scores),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "f1": f1,
    }
