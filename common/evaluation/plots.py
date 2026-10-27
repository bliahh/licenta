import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_confusion_matrix(name, test_metrics, eval_dir, description=""):
    arch_eval_dir = os.path.join(eval_dir, name)
    os.makedirs(arch_eval_dir, exist_ok=True)
    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
    print(f"\n[CM {name}{' ' + description if description else ''}]\n{cm}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Blues")
    title = f"{name} Confusion Matrix" + (f" ({description})" if description else "")
    plt.title(title)
    plt.tight_layout()

    suffix = f"_{description}" if description else ""
    filename = f"{name}_confusion{suffix}.png"
    plt.savefig(os.path.join(arch_eval_dir, filename), dpi=150, bbox_inches="tight")
    plt.close()

def plot_hyst(name, logits, labels, finded_thresh=None, eval_dir=None):
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy().ravel()
    else:
        labels = np.asarray(labels).ravel()

    probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
    print(f"[PROBS {name}] min={probs.min():.6f}, max={probs.max():.6f}, mean={probs.mean():.6f}, std={probs.std():.6f}")

    neg_mask = (labels == 0)
    pos_mask = (labels == 1)
    if np.any(neg_mask):
        print(f"[NEG {name}] min={probs[neg_mask].min():.6f}, max={probs[neg_mask].max():.6f}, mean={probs[neg_mask].mean():.6f}")
    if np.any(pos_mask):
        print(f"[POS {name}] min={probs[pos_mask].min():.6f}, max={probs[pos_mask].max():.6f}, mean={probs[pos_mask].mean():.6f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    data_to_plot, colors_to_plot, labels_to_plot = [], [], []
    if len(probs[neg_mask]) > 0:
        data_to_plot.append(probs[neg_mask]); colors_to_plot.append("red"); labels_to_plot.append("negative")
    if len(probs[pos_mask]) > 0:
        data_to_plot.append(probs[pos_mask]); colors_to_plot.append("green"); labels_to_plot.append("positive")

    if data_to_plot:
        ax.hist(data_to_plot, bins=30, range=(0.0, 1.0), alpha=0.6, color=colors_to_plot, label=labels_to_plot, histtype="bar")

    ax.axvline(0.5, color="black", linestyle="--", label="threshold 0.5")
    if finded_thresh is not None and abs(finded_thresh - 0.5) > 1e-6:
        ax.axvline(finded_thresh, color="blue", linestyle=":", label=f"best threshold ({finded_thresh:.2f})")

    ax.legend(loc="upper right")
    ax.set_xlabel("prediction probability")
    ax.set_ylabel("number of samples")
    ax.set_title(f"{name} - Probability Distribution")
    ax.grid(True, alpha=0.3, linestyle=":")

    target_dir = os.path.join(eval_dir if eval_dir else "", name)
    os.makedirs(target_dir, exist_ok=True)
    plt.savefig(os.path.join(target_dir, f"{name}_prob_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_metrics(metrics_train, metrics_val, dir):
    os.makedirs(dir, exist_ok=True)

    metric_names = [k[:-len("_train")] for k in metrics_train if k.endswith("_train")]

    for metric in metric_names:
        train_key = f"{metric}_train"
        val_key = f"{metric}_val"

        train_vals = metrics_train.get(train_key)
        val_vals = metrics_val.get(val_key)

        x = range(1, len(train_vals) + 1)

        plt.figure()
        plt.plot(x, train_vals, label="train")
        if val_vals is not None:
            plt.plot(x, val_vals, label="val")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.title("Metrics " + metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(dir, f"Metrics_{metric}.png"))
        plt.close()


