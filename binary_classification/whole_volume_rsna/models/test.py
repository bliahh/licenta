import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score

from binary_classification.whole_volume_rsna.data.loaders import load_split, build_loader, load_points, location_scheme, build_2d_loader
from binary_classification.whole_volume_rsna.models.network import ResNet18, ResNet18MultiTask, ResNet18_2D, ConvNext_2D
from binary_classification.whole_volume_rsna.models.run import predict
from binary_classification.whole_volume_rsna.models.train import localization_hit_rate, load_2d_tables


def binary_metrics(val_labels, val_logits, test_labels, test_logits):
    fpr, tpr, thresholds = roc_curve(val_labels, val_logits)
    threshold = thresholds[np.argmax(tpr - fpr)]
    test_pred = (test_logits >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(test_labels, test_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    metrics = {"auc": roc_auc_score(test_labels, test_logits),
               "sensitivity": sensitivity,
               "specificity": specificity,
               "precision": tp / (tp + fp) if tp + fp > 0 else 0.0,
               "f1": f1_score(test_labels, test_pred),
               "accuracy": (tp + tn) / len(test_labels),
               "balanced_accuracy": (sensitivity + specificity) / 2,
               "threshold_logit": threshold,
               "n_test": len(test_labels)}
    return metrics, test_pred


def per_series(slices):
    return slices.groupby(["uid", "modality"], as_index=False)["logit"].max()


def test_2d(run_dir, config, seed, device, use_amp, only_mr=False, only_ct=False, batch_size=32, num_workers=4):
    if config.get("backbone", "resnet18") == "convnext":
        model = ConvNext_2D(in_channels=3 if config["use_context"] else 1, dropout=config["dropout"],
                            multitask=config.get("multitask", False), top_n=config.get("top_n", 4),
                            separe_head=config.get("separe_head", False)).to(device)
    else:
        model = ResNet18_2D(in_channels=3 if config["use_context"] else 1, dropout=config["dropout"],
                            multitask=config.get("multitask", False), top_n=config.get("top_n", 4),
                            separe_head=config.get("separe_head", False), fpn=config.get("fpn", False)).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))

    train_table, val_table, test_table = load_split(seed, only_mr=only_mr, only_ct=only_ct)
    _, val_slices, _ = load_2d_tables(train_table, val_table, config["slices_csv"], config["thickness_csv"])
    _, test_slices, _ = load_2d_tables(train_table, test_table, config["slices_csv"], config["thickness_csv"])

    tables = {}
    for name, slices in [("val", val_slices), ("test", test_slices)]:
        loader = build_2d_loader(slices, None, config["folder_2d"], is_training=False, batch_size=batch_size, num_workers=num_workers, use_context=config["use_context"])
        logits, labels, uids, slice_idx, _ = predict(model, loader, device, use_amp, return_ids=True)
        modality = dict(zip(slices["uid"], slices["modality"]))
        tables[name] = pd.DataFrame({"uid": uids, "slice": slice_idx, "modality": [modality[u] for u in uids],
                                     "soft_label": labels, "label": (labels >= 0.5).astype(int), "logit": logits})

    val, test = tables["val"], tables["test"]
    metrics, test["pred"] = binary_metrics(val["label"], val["logit"], test["label"], test["logit"])
    for modality in ["CTA", "MRA"]:
        v, t = val[val["modality"] == modality], test[test["modality"] == modality]
        if t["label"].nunique() == 2 and v["label"].nunique() == 2:
            m, _ = binary_metrics(v["label"], v["logit"], t["label"], t["logit"])
            metrics.update({f"{k}_{modality}": m[k] for k in ["auc", "sensitivity", "specificity"]})

    labels = pd.concat([val_table, test_table]).set_index("uid")["label"]
    val_series, test_series = per_series(val), per_series(test)
    val_series["label"], test_series["label"] = labels[val_series["uid"]].values, labels[test_series["uid"]].values
    series_metrics, test_series["pred"] = binary_metrics(val_series["label"], val_series["logit"],
                                                         test_series["label"], test_series["logit"])
    metrics.update({f"series_{k}": v for k, v in series_metrics.items()})

    val.to_csv(os.path.join(run_dir, "predictions_val_slices.csv"), index=False)
    test.to_csv(os.path.join(run_dir, "predictions_test_slices.csv"), index=False)
    val_series[["uid", "modality", "label", "logit"]].to_csv(os.path.join(run_dir, "predictions_val.csv"), index=False)
    test_series[["uid", "modality", "label", "logit", "pred"]].to_csv(os.path.join(run_dir, "predictions_test.csv"), index=False)

    with open(os.path.join(run_dir, "metrics_test.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    print(f"[ TEST 2D ] {run_dir} | " + " | ".join(f"{k} {float(v):.3f}" for k, v in metrics.items()))
    return metrics


def test(run_dir, seed, device, use_amp, only_mr=False, only_ct=False, batch_size=2, num_workers=4):
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    if config.get("use_2d", False):
        return test_2d(run_dir, config, seed, device, use_amp, only_mr=only_mr, only_ct=only_ct,
                       batch_size=max(batch_size, 32), num_workers=num_workers)
    multitask = config.get("multitask", False)
    roi_size = config.get("roi_size")
    n_outputs = config.get("n_outputs", 1)
    if multitask:
        model = ResNet18MultiTask(dropout=config["dropout"], top_n=config["top_n"], conv1_stride=1 if roi_size else 2,
                                  n_outputs=n_outputs).to(device)
    else:
        model = ResNet18(dropout=config["dropout"]).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))

    _, val_table, test_table = load_split(seed, only_mr=only_mr, only_ct=only_ct)
    loader_options = dict(is_training=False, batch_size=batch_size, num_workers=num_workers,
                          use_amp=use_amp, multitask=multitask, roi_size=roi_size, n_outputs=n_outputs,
                          vessel_dilation=config.get("vessel_dilation"))

    val_logits, val_labels = predict(model, build_loader(val_table, **loader_options), device, use_amp)
    test_logits, test_labels, test_peaks = predict(model, build_loader(test_table, **loader_options),
                                                   device, use_amp, return_peaks=True)

    group_logits, group_labels = None, None
    if test_logits.ndim == 2:
        group_logits, group_labels = test_logits[:, 1:], test_labels[:, 1:]
        val_logits, val_labels = val_logits[:, 0], val_labels[:, 0]
        test_logits, test_labels = test_logits[:, 0], test_labels[:, 0]

    metrics, test_pred = binary_metrics(val_labels, val_logits, test_labels, test_logits)

    if multitask:
        points = load_points()
        metrics["hit"] = localization_hit_rate(test_table, test_peaks, points, roi_size=roi_size)
        metrics["hit_exact"] = localization_hit_rate(test_table, test_peaks, points, tolerance=0, roi_size=roi_size)

    predictions = pd.DataFrame({"uid": test_table["uid"].values,
                                "modality": test_table["modality"].values,
                                "label": test_labels,
                                "logit": test_logits,
                                "pred": test_pred})

    if group_logits is not None:
        group_aucs = []
        for g, name in enumerate(location_scheme(n_outputs)[1][1:]):
            predictions[f"label_{name}"] = group_labels[:, g]
            predictions[f"logit_{name}"] = group_logits[:, g]
            positives = group_labels[:, g].sum()
            auc = roc_auc_score(group_labels[:, g], group_logits[:, g]) if 0 < positives < len(group_labels) else float("nan")
            metrics[f"auc_{name}"] = auc
            metrics[f"positives_{name}"] = positives
            group_aucs.append(auc)
        metrics["auc_groups_mean"] = float(np.nanmean(group_aucs))

    predictions.to_csv(os.path.join(run_dir, "predictions_test.csv"), index=False)

    with open(os.path.join(run_dir, "metrics_test.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    print(f"[ TEST ] {run_dir} | " + " | ".join(f"{k} {float(v):.3f}" for k, v in metrics.items()))
    return metrics