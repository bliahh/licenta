import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "rsna_dataset")
VOLUME_SIZE = 192


def point_mm_to_voxel(point_ras_mm, volume_affine, crop_offset):
    point_homogeneous = np.append(point_ras_mm, 1.0)
    voxel_before_crop = np.linalg.inv(np.asarray(volume_affine)) @ point_homogeneous
    return voxel_before_crop[:3] + np.asarray(crop_offset)


def build_localizer_table():
    series_table = pd.read_csv(os.path.join(DATA_DIR, "index.csv"))
    geometry_table = pd.read_csv(os.path.join(DATA_DIR, "geometry.csv"))
    localizers = pd.read_csv(os.path.join(DATA_DIR, "localizers_world.csv"))

    shape_check = series_table.merge(geometry_table[["uid", "shape"]], on="uid", suffixes=("", "_geometry"))
    shape_mismatch = shape_check[shape_check["shape"] != shape_check["shape_geometry"]]
    print(f"[ LOCALIZERS ] series with geometry: {len(shape_check)} / {len(series_table)} | shape mismatches: {len(shape_mismatch)}")

    localizers = localizers[localizers["ok"] & localizers["uid"].isin(series_table["uid"])]
    localizers = localizers.merge(geometry_table[["uid", "affine", "offset"]], on="uid")
    voxel_coords = np.array([
        point_mm_to_voxel([row.x, row.y, row.z], json.loads(row.affine), json.loads(row.offset))
        for row in localizers.itertuples()
    ])
    localizers["i"] = voxel_coords[:, 0].round(1)
    localizers["j"] = voxel_coords[:, 1].round(1)
    localizers["k"] = voxel_coords[:, 2].round(1)
    localizers["inside"] = ((voxel_coords >= 0) & (voxel_coords <= VOLUME_SIZE - 1)).all(axis=1)
    localizers = localizers.merge(series_table[["uid", "modality", "path"]], on="uid")

    print(f"[ LOCALIZERS ] points: {len(localizers)} | inside cube: {localizers['inside'].sum()} "
          f"| multi-frame: {localizers['multiframe'].sum()}")
    print(localizers.groupby("modality")["inside"].mean().round(3))

    output_columns = ["uid", "modality", "location", "multiframe", "i", "j", "k", "inside", "path"]
    localizers[output_columns].to_csv(os.path.join(DATA_DIR, "localizers_npy.csv"), index=False)
    return localizers[output_columns]


def show_localizer(localizer_row, save_path=None, zoom_radius=24):
    volume = np.load(localizer_row["path"]).astype(np.float32)
    i, j, k = int(localizer_row["i"]), int(localizer_row["j"]), int(localizer_row["k"])
    views = [
        ("axial", volume[:, :, k].T, i, j, "upper"),
        ("coronal", volume[:, j, :].T, i, k, "lower"),
        ("sagittal", volume[i, :, :].T, j, k, "lower"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    for column, (title, image, point_x, point_y, origin) in enumerate(views):
        for row, zoomed in enumerate([False, True]):
            axis = axes[row, column]
            axis.imshow(image, cmap="gray", origin=origin)
            axis.scatter(point_x, point_y, s=600 if zoomed else 200, facecolors="none", edgecolors="red", linewidths=1.5)
            if zoomed:
                axis.set_xlim(point_x - zoom_radius, point_x + zoom_radius)
                if origin == "upper":
                    axis.set_ylim(point_y + zoom_radius, point_y - zoom_radius)
                else:
                    axis.set_ylim(point_y - zoom_radius, point_y + zoom_radius)
            axis.set_title(f"{title} zoom" if zoomed else title)
    figure.suptitle(f"{localizer_row['modality']} | {localizer_row['location']} | {localizer_row['uid'][-12:]}")
    figure.tight_layout()
    if save_path:
        figure.savefig(save_path, dpi=100)
    else:
        plt.show()
    plt.close(figure)



if __name__ == "__main__":
    localizers = build_localizer_table()

    outside = localizers[~localizers["inside"]]
    print(f"[ LOCALIZERS ] outside cube: {len(outside)}")
    print(outside.groupby("modality").size())

    plt.switch_backend("Agg")
    check_dir = os.path.join(DATA_DIR, "localizer_checks")
    os.makedirs(check_dir, exist_ok=True)

    inside = localizers[localizers["inside"]]
    single_frame = inside[~inside["multiframe"]]
    multi_frame = inside[inside["multiframe"]]
    sample = pd.concat([
        single_frame.sample(min(10, len(single_frame)), random_state=0),
        multi_frame.sample(min(5, len(multi_frame)), random_state=0),
    ])
    for n, row in enumerate(sample.to_dict("records")):
        frame_type = "multiframe" if row["multiframe"] else "single"
        show_localizer(row, save_path=os.path.join(check_dir, f"{n:02d}_{row['modality']}_{frame_type}.png"))
    print(f"[ LOCALIZERS ] saved {len(sample)} check images to {check_dir}")