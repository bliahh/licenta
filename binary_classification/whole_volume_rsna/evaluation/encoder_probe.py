import os
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from binary_classification.whole_volume_rsna.data.loaders import load_split, load_points, points_to_roi
from binary_classification.whole_volume_rsna.evaluation.utils import (
    RESULTS_DIR, get_device, split_options, load_model, make_loader, run_dir_of)

GRID_SIZE = 12
NEGATIVES_PER_POSITIVE = 3


def aneurysm_cells(points, modality, roi_size):
    if points is None or len(points) == 0:
        return set()
    cell_size = roi_size // GRID_SIZE
    cells = set()
    for point in points_to_roi(points, modality, roi_size):
        cell = point // cell_size
        cells.add((int(cell[0]), int(cell[1]), int(cell[2])))
    return cells


def extract_features(kind, run_dir, table, device, use_amp):
    torch.manual_seed(0)
    model, roi_size, _ = load_model(run_dir, device, load_weights=(kind == "trained"))
    backbone = model.backbone

    points = load_points()
    positive_cells = {}
    for row in table.itertuples():
        positive_cells[row.uid] = aneurysm_cells(points.get(row.uid), row.modality, roi_size)
    all_cells = set()
    for cells in positive_cells.values():
        all_cells |= cells

    table = table.reset_index(drop=True)
    loader = make_loader(table, roi_size, use_amp, batch_size=1)
    cell_features = {}
    series_features = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            volume = batch[0].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                x = backbone.act(backbone.bn1(backbone.conv1(volume)))
                x = backbone.layer1(x)
                x = backbone.layer2(x)
                x = backbone.layer3(x)
                x = backbone.layer4(x)
            grid = x[0].float().cpu().numpy()

            uid = table.loc[i, "uid"]
            if table.loc[i, "label"] == 1:
                cells_to_keep = positive_cells[uid]
            else:
                cells_to_keep = all_cells

            cell_features[uid] = {}
            for cell in cells_to_keep:
                cell_features[uid][cell] = grid[:, cell[0], cell[1], cell[2]].astype(np.float16)

            series_features.append(grid.reshape(grid.shape[0], -1).max(axis=1))
            if (i + 1) % 100 == 0:
                print(f"{kind} encoder: {i + 1}/{len(table)}", flush=True)

    return cell_features, positive_cells, np.stack(series_features)


def cell_dataset(cell_features, positive_cells, table, rng):
    negative_uids = table[table["label"] == 0]["uid"].tolist()
    positive_uids = table[table["label"] == 1]["uid"].tolist()
    x = []
    y = []
    for uid in positive_uids:
        for cell in positive_cells[uid]:
            x.append(cell_features[uid][cell])
            y.append(1)
            for negative_uid in rng.choice(negative_uids, NEGATIVES_PER_POSITIVE, replace=False):
                x.append(cell_features[negative_uid][cell])
                y.append(0)
    return np.array(x, dtype=np.float32), np.array(y)


def fit_probe(train_x, train_y, test_x, test_y):
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    probe.fit(train_x, train_y)
    return roc_auc_score(test_y, probe.predict_proba(test_x)[:, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="final_mr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--max_train", type=int, default=None)
    args = parser.parse_args()

    device, use_amp = get_device()
    run_dir = run_dir_of(args.results_dir, args.run, args.seed)
    train_table, _, test_table = load_split(args.seed, **split_options(args.run))
    if args.max_train:
        train_table = train_table.sample(n=min(args.max_train, len(train_table)), random_state=0)
    table = pd.concat([train_table, test_table]).reset_index(drop=True)
    labels = table["label"].to_numpy()
    is_train = np.arange(len(table)) < len(train_table)
    print(f"{args.run} seed {args.seed}: {len(train_table)} train, {len(test_table)} test")

    rows = []
    for kind in ["random", "trained"]:
        cell_features, positive_cells, series_features = extract_features(kind, run_dir, table, device, use_amp)

        rng = np.random.default_rng(0)
        train_x, train_y = cell_dataset(cell_features, positive_cells, train_table, rng)
        test_x, test_y = cell_dataset(cell_features, positive_cells, test_table, rng)
        cell_auc = fit_probe(train_x, train_y, test_x, test_y)
        series_auc = fit_probe(series_features[is_train], labels[is_train],
                               series_features[~is_train], labels[~is_train])

        rows.append({"encoder": kind, "cell_auc": round(cell_auc, 3), "series_auc": round(series_auc, 3),
                     "train_cells": int(train_y.sum()), "test_cells": int(test_y.sum())})
        print(f"{kind} encoder: cell auc {cell_auc:.3f}, series auc {series_auc:.3f}", flush=True)

    summary = pd.DataFrame(rows)
    print("\nLogistic regression on frozen layer4 features")
    print(summary.to_string(index=False))
    output_path = os.path.join(args.results_dir, f"encoder_probe_{args.run}_seed{args.seed}.csv")
    summary.to_csv(output_path, index=False)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
