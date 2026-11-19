import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from binary_classification.whole_volume_rsna.data.loaders import (GRID_SIZE, VOLUME_SIZE, load_split, build_loader, load_points, points_to_roi)
from binary_classification.whole_volume_rsna.models.network import ResNet18MultiTask

RUN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "out", "mra_roi96_seed0"))
OUT_DIR = os.path.join(RUN_DIR, "heatmap_checks")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def load_model(run_dir):
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    roi_size = config.get("roi_size")
    model = ResNet18MultiTask(dropout=0.0, top_n=config["top_n"], conv1_stride=1 if roi_size else 2).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=DEVICE))
    return model.eval(), config


@torch.no_grad()
def predict_heat_maps(model, table, roi_size, keep_volumes=False):
    loader = build_loader(table, is_training=False, batch_size=1, num_workers=2, use_amp=USE_AMP, multitask=True, roi_size=roi_size)
    heat_maps, probabilities, volumes = [], [], []
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            logit, heat_map = model(batch[0].to(DEVICE))
        heat_maps.append(torch.sigmoid(heat_map.float())[0, 0].cpu().numpy())
        probabilities.append(torch.sigmoid(logit.float()).item())
        if keep_volumes:
            volumes.append(batch[0][0, 0].numpy())
    return np.stack(heat_maps), np.array(probabilities), volumes


def heat_map_statistics(heat_maps, table):
    flat = heat_maps.reshape(len(heat_maps), -1)
    peaks = [tuple(int(v) for v in np.unravel_index(row.argmax(), (GRID_SIZE,) * 3)) for row in flat]
    peak_counts = pd.Series(peaks).value_counts()
    mean_map = flat.mean(axis=0)
    correlations = np.array([np.corrcoef(row, mean_map)[0, 1] for row in flat])
    print(f"[ HEAT ] distinct peak cells {len(peak_counts)} / {len(peaks)} "
          f"| most common {peak_counts.index[0]} ({peak_counts.iloc[0] / len(peaks):.2f})")
    print(f"[ HEAT ] correlation with mean map | median {np.median(correlations):.3f} | min {correlations.min():.3f}")
    for label in (0, 1):
        print(f"[ HEAT ] label {label} | mean of max cell probability "
              f"{flat[table['label'].values == label].max(axis=1).mean():.3f}")


def plot_heat_map(volume, heat_map, probability, roi_points, title, save_path):
    volume_size = volume.shape[0]
    cell_size = volume_size // GRID_SIZE
    heat_volume = F.interpolate(torch.from_numpy(heat_map)[None, None], size=(volume_size,) * 3, mode="trilinear", align_corners=False)[0, 0].numpy()
    peak = np.array(np.unravel_index(heat_map.argmax(), heat_map.shape)) * cell_size + (cell_size - 1) / 2
    views = [("axial", 2, 0, 1, "upper"), ("coronal", 1, 0, 2, "lower"), ("sagittal", 0, 1, 2, "lower")]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for axis, (name, mip_axis, x_index, y_index, origin) in zip(axes, views):
        axis.imshow(volume.max(axis=mip_axis).T, cmap="gray", origin=origin)
        axis.imshow(heat_volume.max(axis=mip_axis).T, cmap="jet", alpha=0.35, origin=origin, vmin=0, vmax=1)
        if len(roi_points):
            axis.scatter(roi_points[:, x_index], roi_points[:, y_index], s=200, facecolors="none", edgecolors="red", linewidths=2)
        axis.scatter(peak[x_index], peak[y_index], s=120, marker="x", color="cyan", linewidths=2)
        axis.set_title(f"{name} MIP")
    figure.suptitle(f"{title} | p={probability:.2f}")
    figure.tight_layout()
    figure.savefig(save_path, dpi=100)
    plt.close(figure)


def main(n_positives=6, n_negatives=3):
    assert os.path.exists(os.path.join(RUN_DIR, "best.pt")), f"best.pt not found in {RUN_DIR}"
    os.makedirs(OUT_DIR, exist_ok=True)
    model, config = load_model(RUN_DIR)
    roi_size = config.get("roi_size")
    _, val_table, _ = load_split(config["seed"], only_mr=config["only_mr"], only_ct=config["only_ct"])

    heat_maps, _, _ = predict_heat_maps(model, val_table, roi_size)
    heat_map_statistics(heat_maps, val_table)

    points = load_points()
    sample = pd.concat([val_table[val_table["label"] == 1].sample(n_positives, random_state=0), val_table[val_table["label"] == 0].sample(n_negatives, random_state=0)])
    sample_heat_maps, sample_probabilities, sample_volumes = predict_heat_maps(model, sample, roi_size, keep_volumes=True)
    for row, heat_map, probability, volume in zip(sample.itertuples(), sample_heat_maps, sample_probabilities, sample_volumes):
        if row.uid not in points:
            roi_points = np.zeros((0, 3))
        elif roi_size is None:
            roi_points = points[row.uid]
        else:
            roi_points = points_to_roi(points[row.uid], row.modality, roi_size)
        label_name = "POS" if row.label == 1 else "NEG"
        plot_heat_map(volume, heat_map, probability, roi_points,
                      f"{row.modality} | {label_name} | {row.uid[-12:]}",
                      os.path.join(OUT_DIR, f"{label_name}_{row.uid[-8:]}.png"))
    print(f"[ HEAT ] saved {len(sample)} images to {OUT_DIR}")


if __name__ == "__main__":
    main()