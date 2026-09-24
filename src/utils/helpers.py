import json
import os
import json
import pandas as pd
from src.config import LOSS_NAME,SAVE_DIR,EVAL_DIR,TUNED_EVAL_DIR,OUTPUT_DIR
from prettytable import PrettyTable


def print_comparison(results, key="baseline", title="RESULTS"):

    print("\n" + "=" * 90)
    print(f"{title} [{key}]")
    print("=" * 90)

    table = PrettyTable()
    table.field_names = [
        "Model",
        "Loss",
        "Thr",
        "Val AUC",
        "Test AUC",
        "PR-AUC",
        "Recall",
        "Precision",
        "F1",
        "F2",
        "Acc",
        "Brier"
    ]

    for name, per_key in results.items():
        if key not in per_key:
            continue

        r = per_key[key]

        table.add_row([
            name,
            r.get("loss", "-"),
            r["threshold"],
            r["val_auc"],
            r["test_auc"],
            r["test_pr_auc"],
            r["test_recall"],
            r["test_precision"],
            r["test_f1"],
            r["test_f2"],
            r["test_acc"],
            r["test_brier_score"],
        ])

    print(table)


def get_loss_names():
    if LOSS_NAME == "both":
        return ["bce", "focal"]

    if LOSS_NAME in ("bce", "focal"):
        return [LOSS_NAME]

    raise ValueError(
        f"Invalid LOSS_NAME='{LOSS_NAME}'. "
        f"Use 'bce', 'focal' or 'both'."
    )


def get_model_dir(loss_name):
    return os.path.join(SAVE_DIR, loss_name)


def get_eval_dir(loss_name):
    return os.path.join(EVAL_DIR, loss_name)


def get_tuned_eval_dir(loss_name):
    return os.path.join(TUNED_EVAL_DIR, loss_name)


def get_results_path(loss_name):
    return os.path.join(
        OUTPUT_DIR,
        f"comparison_{loss_name}.json"
    )


def get_tuned_results_path(loss_name):
    return os.path.join(
        OUTPUT_DIR,
        f"comparison_tuned_{loss_name}.json"
    )


def save_labeled_samples(labeled_samples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(labeled_samples, f, indent=2)

    return True


def save_metrics(metrics, dir, filename="metrics.json"):
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, filename), "w") as f:
        json.dump(metrics, f, indent=2)



def json_to_csv(in_file, out_file):
    with open(in_file, "r") as f:
        metrics = json.load(f)

    rows = []

    for loss, models in metrics.items():
        for model, strategies in models.items():
            for optimize, values in strategies.items():

                row = {
                    "loss": loss,
                    "model": model,
                    "optimize": optimize,
                    "threshold": values.get("threshold"),

                    "val_f1": values.get("val_f1"),
                    "val_f2": values.get("val_f2"),

                    "test_auc": values.get("test_auc"),
                    "test_pr_auc": values.get("test_pr_auc"),
                    "test_recall": values.get("test_recall"),
                    "test_precision": values.get("test_precision"),
                    "test_f1": values.get("test_f1"),
                    "test_f2": values.get("test_f2"),
                    "test_acc": values.get("test_acc"),
                    "test_brier_score": values.get("test_brier_score"),
                }


                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)

    return df


def save_in_file(string, filename):
    with open(filename, "a") as f:
        f.write(string)



def debug_ct_rm_prediction_separately(probs, labels, threshold,paths):
    preds = (probs >= threshold).astype(int)
    type_acq = {"ct":
                    {"tn": 0, "tp": 0,
                     "fn": 0, "fp": 0},
                "mr": {"tn": 0, "tp": 0, "fn": 0, "fp": 0}}

    for i,p in enumerate(paths):
        modalities = "ct" if "_ct_" in str(p).lower() else "mr"
        true, predicted = int(labels[i]), int(preds[i])
        if true == 0 and predicted == 0:
            type_acq[modalities]["tn"] += 1
        if true == 1 and predicted == 1:
            type_acq[modalities]["tp"] += 1
        if true == 0 and predicted == 1:
            type_acq[modalities]["fp"] += 1
            print(f"{p} produced a FP")
            save_in_file(f"{p} produced a FP","/home/bliahh/Desktop/FACULATATE/licenta/output/outputs_vesselmask/fn_fp.txt")
        if true == 1 and predicted == 0:
            type_acq[modalities]["fn"] += 1
            print(f"{p} produced a FN")
            save_in_file(f"{p} produced a FN","/home/bliahh/Desktop/FACULATATE/licenta/output/outputs_vesselmask/fn_fp.txt")


    for modality in ["ct", "mr"]:
        m = type_acq[modality]
        total = sum(m.values())
        if total == 0:
            continue
        recall = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0
        precision = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0
        print(f"\n[{modality.upper()}] {total} cases")
        print(f"CM: [[TN={m['tn']}, FP={m['fp']}], [FN={m['fn']}, TP={m['tp']}]]")
        print(f"  recall={recall:.3f}, precision={precision:.3f}")


