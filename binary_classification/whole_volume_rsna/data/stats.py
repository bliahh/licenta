
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from binary_classification.whole_volume_rsna.data.loaders import DATA_DIR, load_split, load_points, roi_start, points_to_roi

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
ROI_SIZE = 96


def check_labels(seed=0):
    split = pd.read_csv(os.path.join(DATA_DIR, f"split_seed{seed}.csv"))
    competition = pd.read_csv(TRAIN_CSV).set_index("SeriesInstanceUID")
    merged = split.join(competition[["Aneurysm Present", "Modality"]], on="uid")
    print(f"[ LABELS ] missing in train.csv: {merged['Aneurysm Present'].isna().sum()} "
          f"| label mismatches: {(merged['label'] != merged['Aneurysm Present']).sum()} "
          f"| modality mismatches: {(merged['modality'] != merged['Modality']).sum()}")
    sets = {name: set(group["uid"]) for name, group in split.groupby("split")}
    print(f"[ SPLITS ] train&val {len(sets['train'] & sets['val'])} | train&test {len(sets['train'] & sets['test'])} "
          f"| val&test {len(sets['val'] & sets['test'])}")
    points = load_points()
    has_points = split["uid"].isin(list(points))
    print(f"[ POINTS ] positives without points: {((split['label'] == 1) & ~has_points).sum()} "
          f"| negatives with points: {((split['label'] == 0) & has_points).sum()}")


def save_roi_mips(seed=0, n_positives=8, n_negatives=4, out_dir=os.path.join(DATA_DIR, "roi_checks")):
    os.makedirs(out_dir, exist_ok=True)
    train_table, _, _ = load_split(seed)
    points = load_points()
    positives = train_table[(train_table["label"] == 1) & train_table["uid"].isin(list(points))]
    negatives = train_table[train_table["label"] == 0]
    sample = pd.concat([positives.sample(n_positives, random_state=seed), negatives.sample(n_negatives, random_state=seed)])
    for row in sample.itertuples():
        start = roi_start(row.modality, ROI_SIZE)
        end = start + ROI_SIZE
        volume = np.load(row.path, mmap_mode="r")[start[0]:end[0], start[1]:end[1], start[2]:end[2]].astype(np.float32)
        roi_points = points_to_roi(points[row.uid], row.modality, ROI_SIZE) if row.uid in points else np.zeros((0, 3))
        views = [("axial MIP", volume.max(axis=2).T, 0, 1, "upper"),
                 ("coronal MIP", volume.max(axis=1).T, 0, 2, "lower"),
                 ("sagittal MIP", volume.max(axis=0).T, 1, 2, "lower")]
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, (title, image, x_index, y_index, origin) in zip(axes, views):
            axis.imshow(image, cmap="gray", origin=origin)
            if len(roi_points):
                axis.scatter(roi_points[:, x_index], roi_points[:, y_index], s=150, facecolors="none", edgecolors="red")
            axis.set_title(title)
        label_name = "POS" if row.label == 1 else "NEG"
        figure.suptitle(f"{row.modality} | {label_name} | {row.uid[-12:]}")
        figure.tight_layout()
        figure.savefig(os.path.join(out_dir, f"{label_name}_{row.modality}_{row.uid[-8:]}.png"), dpi=100)
        plt.close(figure)
        print(f"[ ROI ] {label_name} {row.modality} | min {volume.min():.2f} max {volume.max():.2f} "
              f"mean {volume.mean():.2f} std {volume.std():.2f} | points in roi {len(roi_points)}")


if __name__ == "__main__":
    check_labels()
    save_roi_mips()