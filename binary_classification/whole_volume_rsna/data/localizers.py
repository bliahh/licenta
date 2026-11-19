import os
import re
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "rsna_dataset")
VOLUME_SIZE = 192
HALF_DIR = os.path.join(DATA_DIR, "mra_05")


def point_mm_to_voxel(point_ras_mm, volume_affine, crop_offset):
    point_homogeneous = np.append(point_ras_mm, 1.0)
    voxel_before_crop = np.linalg.inv(np.asarray(volume_affine)) @ point_homogeneous
    return voxel_before_crop[:3] + np.asarray(crop_offset)


def parse_shape(shape_text):
    return tuple(int(value) for value in re.findall(r"\d+", str(shape_text)))


def crop_or_pad_offset(shape, size=VOLUME_SIZE):
    offset = []
    for axis, length_before in enumerate(shape):
        length = min(length_before, size)
        source_start = length_before - length if axis == 2 else (length_before - length) // 2
        offset.append((size - length) // 2 - source_start)
    return offset


def check_half_spacing(geometry_table, half_log):
    geometry = geometry_table.set_index("uid")
    formula_mismatch = sum(
        crop_or_pad_offset(parse_shape(row.shape)) != list(json.loads(row.offset))
        for row in geometry.itertuples()
    )
    shared = half_log[half_log["uid"].isin(geometry.index)]
    shape_difference = np.array([
        np.abs(np.array(parse_shape(shape_half)) - 2 * np.array(parse_shape(geometry.loc[uid, "shape"]))).max()
        for uid, shape_half in zip(shared["uid"], shared["shape"])
    ])
    print(f"[ LOCALIZERS ] offset formula vs geometry.csv: mismatches {formula_mismatch} / {len(geometry)}")
    print(f"[ LOCALIZERS ] half shape vs 2 x 1 mm shape: max difference {shape_difference.max()} voxels "
          f"| series compared {len(shape_difference)}")


def build_localizer_table(spacing_scale=1, log_path=None, volume_dir=None, modality=None,
                          output_name="localizers_npy.csv"):
    series_table = pd.read_csv(os.path.join(DATA_DIR, "index.csv"))
    geometry_table = pd.read_csv(os.path.join(DATA_DIR, "geometry.csv"))
    localizers = pd.read_csv(os.path.join(DATA_DIR, "localizers_world.csv"))
    if modality is not None:
        series_table = series_table[series_table["modality"] == modality]

    shape_check = series_table.merge(geometry_table[["uid", "shape"]], on="uid", suffixes=("", "_geometry"))
    shape_mismatch = shape_check[shape_check["shape"] != shape_check["shape_geometry"]]
    print(f"[ LOCALIZERS ] series with geometry: {len(shape_check)} / {len(series_table)} | shape mismatches: {len(shape_mismatch)}")

    localizers = localizers[localizers["ok"] & localizers["uid"].isin(series_table["uid"])]
    localizers = localizers.merge(geometry_table[["uid", "affine", "offset"]], on="uid")
    if log_path is not None:
        half_log = pd.read_csv(log_path)
        half_log = half_log[half_log["ok"]]
        check_half_spacing(geometry_table[geometry_table["uid"].isin(series_table["uid"])], half_log)
        half_offsets = {uid: crop_or_pad_offset(parse_shape(shape)) for uid, shape in zip(half_log["uid"], half_log["shape"])}
        localizers = localizers[localizers["uid"].isin(half_offsets)]
        localizers = localizers.assign(offset=localizers["uid"].map(lambda uid: json.dumps(half_offsets[uid])))

    voxel_coords = np.array([
        spacing_scale * point_mm_to_voxel([row.x, row.y, row.z], json.loads(row.affine), [0, 0, 0])
        + np.asarray(json.loads(row.offset))
        for row in localizers.itertuples()
    ])
    localizers["i"] = voxel_coords[:, 0].round(1)
    localizers["j"] = voxel_coords[:, 1].round(1)
    localizers["k"] = voxel_coords[:, 2].round(1)
    localizers["inside"] = ((voxel_coords >= 0) & (voxel_coords <= VOLUME_SIZE - 1)).all(axis=1)
    localizers = localizers.merge(series_table[["uid", "modality", "path"]], on="uid")
    if volume_dir is not None:
        localizers["path"] = localizers["uid"].map(lambda uid: os.path.join(volume_dir, f"{uid}.npy"))

    print(f"[ LOCALIZERS ] points: {len(localizers)} | inside cube: {localizers['inside'].sum()} "
          f"| multi-frame: {localizers['multiframe'].sum()}")
    print(localizers.groupby("modality")["inside"].mean().round(3))

    output_columns = ["uid", "modality", "location", "multiframe", "i", "j", "k", "inside", "path"]
    localizers[output_columns].to_csv(os.path.join(DATA_DIR, output_name), index=False)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    if args.half:
        localizers = build_localizer_table(spacing_scale=2, log_path=os.path.join(HALF_DIR, "log_0_1252.csv"),
                                           volume_dir=os.path.join(HALF_DIR, "preprocessed"), modality="MRA",
                                           output_name="localizers_mra05.csv")
        check_dir, zoom_radius = os.path.join(DATA_DIR, "localizer_checks_05"), 48
    else:
        localizers = build_localizer_table()
        check_dir, zoom_radius = os.path.join(DATA_DIR, "localizer_checks"), 24

    outside = localizers[~localizers["inside"]]
    print(f"[ LOCALIZERS ] outside cube: {len(outside)}")
    print(outside.groupby("modality").size())

    plt.switch_backend("Agg")
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
        show_localizer(row, save_path=os.path.join(check_dir, f"{n:02d}_{row['modality']}_{frame_type}.png"),
                       zoom_radius=zoom_radius)
    print(f"[ LOCALIZERS ] saved {len(sample)} check images to {check_dir}")