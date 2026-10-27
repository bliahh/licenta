import glob
import os
import json
from common.paths import DATASET_ROOT, LABELS_DIR, BINARY_LABELS_PATH, SPLIT_PATH
from common.utils.io import save_labeled_samples


def build_bin_samples(data_root=DATASET_ROOT):
    """
    :param data_root: path to data root
    For each sample, the function extracts and adds the following fields:
    {
        "image": path to the image,
        "vessel_mask": path to the vessel mask,
        "aneurysm_mask": path to the aneurysm mask,
        "label": 0/1,
        "type_acq": type of acquisition (CT/MR),
        "spec_labels": list of aneurysm labels
    }

    :return labeled_samples: a list of dictionaries in JSON format containing all samples
    """

    img_dir = os.path.join(data_root, "images")
    json_dir = os.path.join(data_root, "location_jsons")
    vessel_dir = os.path.join(data_root, "vessel_masks")
    aneurysm_dir = os.path.join(data_root, "location_masks")

    labeled_samples = []

    for img_path in sorted(
        glob.glob(os.path.join(img_dir, "*_0000.nii.gz"))
    ):
        base = os.path.basename(img_path).replace("_0000.nii.gz", "")

        json_path = os.path.join(json_dir, base + ".json")
        vessel_path = os.path.join(vessel_dir, base + ".nii.gz")
        aneurysm_path = os.path.join(aneurysm_dir, base + ".nii.gz")
        if not os.path.exists(json_path):
            print(f"[ERROR] JSON missing for {img_path}")
            continue

        if not os.path.exists(vessel_path):
            print(
                f"[ERROR] Vessel mask missing for {img_path} "
                f"at {vessel_path}"
            )
            continue
        if not os.path.exists(aneurysm_path):
            print(f"[ERROR] Aneurysm mask missing for {img_path}")
            print(f"at {aneurysm_path}")
            continue

        with open(json_path) as f:
            data = json.load(f)

        has_aneurysm = int(len(data["locations"]) > 0)

        labeled_samples.append({
            "image": img_path,
            "vessel_mask": vessel_path,
            "aneurysm_mask": aneurysm_path,
            "label": has_aneurysm,
            "type_acq": "ct" if "_ct_" in img_path.lower() else "mr",
            "spec_labels": data["locations"]
        })

    os.makedirs(LABELS_DIR, exist_ok=True)

    save_labeled_samples(labeled_samples, BINARY_LABELS_PATH)

    print(f"[SUCCESS] Saved {len(labeled_samples)} ")
    print(f"    labels to {BINARY_LABELS_PATH}")

    return labeled_samples


def find_binary_distribution():
    """
    Print binary label distribution.
    """

    with open(BINARY_LABELS_PATH, "r") as f:
        data = json.load(f)

    total = len(data)
    n_pos = sum(s["label"] for s in data)
    n_neg = total - n_pos

    print(f"[INFO] TOTAL: {total} | "
        f"positive: {n_pos} | "
        f"negative: {n_neg}")


def print_distribution():
    """
    print information about each split.
    for each split, it prints the following fields:
    ct_positive, ct_negative, rm_positive, rm_negative distribution
    :return:
    """
    if not os.path.exists(SPLIT_PATH):
        print(f"[ERROR] Dataset split not found: {SPLIT_PATH}")
        return

    with open(SPLIT_PATH, "r") as f:
        dataset_split = json.load(f)

    for split, samples in dataset_split.items():
        print(f"[INFO] Split {split}: {len(samples)} samples")
        n_ct = sum( sample["type_acq"] == "ct" for sample in dataset_split[split])
        n_rm = len(samples)- n_ct
        print(f"CT: {n_ct} | RM: {n_rm}")
        n_ct_positive = sum( sample["type_acq"] == "ct" and sample["label"] == 1 for sample in dataset_split[split])
        print(f"CT_positive: {n_ct_positive} | CT_negative: {n_ct-n_ct_positive}")
        n_rm_positive = sum( sample["type_acq"] == "mr" and sample["label"] == 1 for sample in dataset_split[split])
        print(f"RM_positive: {n_rm_positive} | RM_negative: {n_rm-n_rm_positive}")
