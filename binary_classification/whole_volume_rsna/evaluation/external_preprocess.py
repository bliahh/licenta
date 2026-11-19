import os
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import nibabel as nib
import monai.transforms as T
from scipy import ndimage

from binary_classification.whole_volume_rsna.data.localizers import crop_or_pad_offset
from binary_classification.whole_volume_rsna.evaluation.utils import EXTERNAL_DIR

TOPANEU_DIR = os.path.join(EXTERNAL_DIR, "topaneu_release", "topaneu_release")
TOPANEU_PREFIX = "topaneu_center1_mr_"
ORIENTATION_LABELS = (("L", "R"), ("P", "A"), ("I", "S"))
SPACING = (1.0, 1.0, 1.0)
SIZE = 192
CT_WINDOW = (0, 900)


def list_topaneu_cases():
    cases = []
    pattern = os.path.join(TOPANEU_DIR, "images", f"{TOPANEU_PREFIX}*_0000.nii.gz")
    for image_path in sorted(glob.glob(pattern)):
        uid = os.path.basename(image_path).replace("_0000.nii.gz", "")
        cases.append({
            "uid": uid,
            "image": image_path,
            "mask": os.path.join(TOPANEU_DIR, "location_masks", f"{uid}.nii.gz"),
            "modality": "MRA",
            "site": "center1",
            "label": None,
        })
    return cases


def list_largeia_cases():
    instances = pd.read_csv(os.path.join(EXTERNAL_DIR, "internal_instances.csv"))
    instances = instances[instances["subset"] == "test"].set_index("instance_id")
    cases = []
    for image_path in sorted(glob.glob(os.path.join(EXTERNAL_DIR, "internal_Ts0*", "Ts*.nii.gz"))):
        uid = os.path.basename(image_path).replace(".nii.gz", "")
        cases.append({
            "uid": uid,
            "image": image_path,
            "mask": None,
            "modality": "CTA",
            "site": f"institution{instances.loc[uid, 'institution_id']}",
            "label": int(instances.loc[uid, "num_IAs"] > 0),
        })
    return cases


def build_transforms(modality, hu_shift, with_mask):
    if with_mask:
        keys = ["image", "mask"]
        modes = ("bilinear", "nearest")
    else:
        keys = ["image"]
        modes = "bilinear"

    if modality == "CTA":
        intensity = T.ScaleIntensityRanged(keys="image", a_min=CT_WINDOW[0], a_max=CT_WINDOW[1],
                                           b_min=-1.0, b_max=1.0, clip=True)
    else:
        intensity = T.NormalizeIntensityd(keys="image", nonzero=True)

    return T.Compose([
        T.LoadImaged(keys=keys, reader="NibabelReader"),
        T.EnsureChannelFirstd(keys=keys),
        T.ShiftIntensityd(keys="image", offset=hu_shift),
        T.Orientationd(keys=keys, axcodes="LPS", labels=ORIENTATION_LABELS),
        T.CropForegroundd(keys=keys, source_key="image"),
        T.Spacingd(keys=keys, pixdim=SPACING, mode=modes),
        intensity,
        T.EnsureTyped(keys=keys, track_meta=False),
    ])


def crop_or_pad(volume, fill_value):
    output = np.full((SIZE, SIZE, SIZE), fill_value, dtype=np.float32)
    source = []
    target = []
    for axis in range(3):
        length = volume.shape[axis]
        kept = min(length, SIZE)
        if axis == 2:
            source_start = length - kept
        else:
            source_start = (length - kept) // 2
        target_start = (SIZE - kept) // 2
        source.append(slice(source_start, source_start + kept))
        target.append(slice(target_start, target_start + kept))
    output[tuple(target)] = volume[tuple(source)]
    return output


def find_hu_shift(image_path):
    sample = np.asarray(nib.load(image_path).dataobj[::4, ::4, ::4], dtype=np.float32)
    if np.percentile(sample, 90) < -500:
        return 1024
    return 0


def preprocess_case(case, output_dir):
    entry = {
        "uid": case["uid"], "modality": case["modality"], "site": case["site"], "label": case["label"],
        "ok": False, "shape": None, "hu_shift": 0, "points": [], "message": "",
    }
    try:
        with_mask = case["mask"] is not None and os.path.exists(case["mask"])
        if case["modality"] == "CTA":
            entry["hu_shift"] = find_hu_shift(case["image"])

        inputs = {"image": case["image"]}
        if with_mask:
            inputs["mask"] = case["mask"]
        data = build_transforms(case["modality"], entry["hu_shift"], with_mask)(inputs)

        volume = np.asarray(data["image"][0], dtype=np.float32)
        entry["shape"] = list(volume.shape)

        if with_mask:
            offset = np.asarray(crop_or_pad_offset(volume.shape))
            mask = np.asarray(data["mask"][0]) > 0
            components, count = ndimage.label(mask)
            for i in range(1, count + 1):
                center = np.asarray(ndimage.center_of_mass(mask, components, i)) + offset
                entry["points"].append([round(float(value), 1) for value in center])
            entry["label"] = int(count > 0)

        path = os.path.join(output_dir, f"{case['uid']}.npy")
        np.save(path, crop_or_pad(volume, volume.min()).astype(np.float16))
        entry["path"] = path
        entry["ok"] = True
    except Exception as error:
        entry["message"] = str(error)[:300]
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["topaneu", "largeia"], required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = os.path.join(EXTERNAL_DIR, "preprocessed", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    if args.dataset == "topaneu":
        cases = list_topaneu_cases()
    else:
        cases = list_largeia_cases()
    cases = cases[:args.limit]
    print(f"{args.dataset}: {len(cases)} cases, writing to {output_dir}", flush=True)

    entries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(preprocess_case, cases, [output_dir] * len(cases))
        for number, entry in enumerate(results, start=1):
            entries.append(entry)
            if not entry["ok"]:
                print(f"failed {entry['uid']}: {entry['message']}", flush=True)
            if number % 20 == 0 or number == len(cases):
                print(f"{number}/{len(cases)}", flush=True)

    table = pd.DataFrame(entries)
    table["dataset"] = args.dataset
    index = table[table["ok"]][["uid", "dataset", "site", "modality", "label", "path", "shape", "hu_shift"]]
    index.to_csv(os.path.join(EXTERNAL_DIR, f"index_{args.dataset}.csv"), index=False)

    points = []
    for entry in entries:
        if not entry["ok"]:
            continue
        for point in entry["points"]:
            inside = all(0 <= value <= SIZE - 1 for value in point)
            points.append({"uid": entry["uid"], "modality": entry["modality"], "location": "", "multiframe": False,
                           "i": point[0], "j": point[1], "k": point[2], "inside": inside, "path": entry["path"]})

    print(f"\nprocessed {int(table['ok'].sum())}/{len(table)}, hu shift applied on {int((table['hu_shift'] > 0).sum())}")
    print(pd.crosstab([index["site"], index["modality"]], index["label"]).to_string())
    if points:
        points = pd.DataFrame(points)
        points.to_csv(os.path.join(EXTERNAL_DIR, f"localizers_{args.dataset}.csv"), index=False)
        print(f"{len(points)} aneurysm points, {points['inside'].mean():.3f} inside the {SIZE} volume")


if __name__ == "__main__":
    main()
