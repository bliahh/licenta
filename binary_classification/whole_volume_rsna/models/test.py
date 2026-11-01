import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score

from binary_classification.whole_volume_rsna.data.loaders import load_split, build_loader
from binary_classification.whole_volume_rsna.models.network import ResNet18
from binary_classification.whole_volume_rsna.models.run import predict


def test(run_dir, seed,device,use_amp, only_mr=False, only_ct=False, dropout=0.3, batch_size=2, num_workers=4,):
    model = ResNet18(dropout=dropout).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))

    _, val_table, test_table = load_split(seed, only_mr=only_mr, only_ct=only_ct)
    val_logits, val_labels = predict(model, build_loader(val_table, False, batch_size, num_workers, use_amp= use_amp), device, use_amp)
    test_logits, test_labels = predict(model, build_loader(test_table, False, batch_size, num_workers, use_amp= use_amp), device, use_amp)

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

    pd.DataFrame({"uid": test_table["uid"].values,
                  "modality": test_table["modality"].values,
                  "label": test_labels,
                  "logit": test_logits,
                  "pred": test_pred}).to_csv(os.path.join(run_dir, "predictions_test.csv"), index=False)


    with open(os.path.join(run_dir, "metrics_test.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    print(f"[ TEST ] {run_dir} | " + " | ".join(f"{k} {float(v):.3f}" for k, v in metrics.items()))
    return metrics