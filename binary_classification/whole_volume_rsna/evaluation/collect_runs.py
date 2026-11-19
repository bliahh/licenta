import os
import re
import json
import argparse

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "binary_classification", "whole_volume_rsna", "out")
DEFAULT_SAVE = os.path.join(PROJECT_ROOT, "results", "runs_history.xlsx")
CONFIG_KEYS = ["modality", "roi_size", "lr", "epochs", "patience", "lambd", "sigma", "pos_weight",
               "top_n", "dropout", "jitter", "batch_size", "seed"]


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as file:
        return json.load(file)


def flatten(data, prefix=""):
    flat = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}_"))
        elif isinstance(value, (int, float, str, bool)) or value is None:
            flat[name] = value
    return flat


def find_column(history, words):
    for column in history.columns:
        if all(word in column.lower() for word in words):
            return column
    return None


def summarize_run(run_dir, out_dir):
    history = pd.read_csv(os.path.join(run_dir, "history.csv"))
    config = flatten(load_json(os.path.join(run_dir, "config.json")))
    metrics = flatten(load_json(os.path.join(run_dir, "metrics_test.json")))
    row = {"run": os.path.relpath(run_dir, out_dir), "epochs_done": len(history)}
    row.update({key: config[key] for key in CONFIG_KEYS if key in config})

    val_auc = find_column(history, ["val", "auc"])
    if val_auc is not None:
        best = history[val_auc].idxmax()
        row["best_val_auc"] = round(history[val_auc].max(), 3)
        row["best_epoch"] = int(history["epoch"][best]) if "epoch" in history else int(best) + 1
        row["val_auc_last10_mean"] = round(history[val_auc].tail(10).mean(), 3)
    for words, name in [(["val", "hit", "exact"], "best_val_hit_exact"), (["val", "hit"], "best_val_hit")]:
        column = find_column(history, words)
        if column is not None and name not in row:
            row[name] = round(history[column].max(), 3)
    row.update({f"test_{key}": round(value, 3) if isinstance(value, float) else value
                for key, value in metrics.items()})
    return row, history


def sheet_name(run, used):
    base = re.sub(r"[\[\]\*\?/\\:]", "_", run)[:28]
    name, n = base, 1
    while name in used:
        name, n = f"{base[:25]}_{n}", n + 1
    used.add(name)
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--save", default=DEFAULT_SAVE)
    args = parser.parse_args()

    run_dirs = sorted(root for root, _, files in os.walk(args.out) if "history.csv" in files)
    print(f"[ COLLECT ] runs with history.csv: {len(run_dirs)}")

    rows, histories = [], {}
    for run_dir in run_dirs:
        try:
            row, history = summarize_run(run_dir, args.out)
        except Exception as error:
            print(f"[ SKIP ] {run_dir}: {error}")
            continue
        rows.append(row)
        histories[row["run"]] = history
        print(f"[ RUN ] {row['run']} | epochs {row['epochs_done']} | best val auc {row.get('best_val_auc')} "
              f"| test auc {row.get('test_auc')}")

    summary = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    used = {"Rezumat"}
    with pd.ExcelWriter(args.save) as writer:
        summary.to_excel(writer, sheet_name="Rezumat", index=False)
        for run, history in histories.items():
            history.to_excel(writer, sheet_name=sheet_name(run, used), index=False)
    print(f"[ SAVED ] {args.save}")