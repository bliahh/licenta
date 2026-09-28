import os

import numpy as np
import torch

from src.config import PREPROCESSED_DIR
from src.data.transforms import base_transform_dict

SPLITS = ("train", "val", "test")


def preprocess_patient(sample, sample_idx, split):
    from src.data.loaders import find_bifurcation

    """
    Preprocesses one patient and saves the result to disk.

    Applies the deterministic transforms (load, orientation, resampling, crop, normalization)
    and saves image.pt, aneurysm_mask.pt and metadata.pt in PREPROCESSED_DIR/<split>/patient_<idx>/.

    :param sample: patient sample (paths to image and masks, label, type_acq)
    :param sample_idx: index of the patient inside the split (used as folder name)
    :param split: "train", "val" or "test"
    """
    transform = base_transform_dict()
    data = {"image": sample["image"], "vessel_mask": sample["vessel_mask"], "aneurysm_mask": sample["aneurysm_mask"]}

    #print(f"\n[PREPROCESS] {split.upper()} patient {sample_idx}")
    data = transform(data)
    image = data["image"]
    aneurysm_mask = data["aneurysm_mask"]

    vessel_mask = data["vessel_mask"][0].cpu().numpy()
    bifurcation = find_bifurcation(np.asarray(vessel_mask))

    print(f"bifurcation {bifurcation}")


    if image.ndim == 3:
        image = image.unsqueeze(0)
    if aneurysm_mask.ndim == 3:
        aneurysm_mask = aneurysm_mask.unsqueeze(0)

    patient_dir = os.path.join(PREPROCESSED_DIR, split, f"patient_{sample_idx:04d}")
    os.makedirs(patient_dir, exist_ok=True)

    torch.save(image.cpu(), os.path.join(patient_dir, "image.pt"))
    torch.save(aneurysm_mask.cpu(), os.path.join(patient_dir, "aneurysm_mask.pt"))
    torch.save(torch.from_numpy(bifurcation), os.path.join(patient_dir, "bifurcations.pt"))

    metadata = {"patient_label": int(sample["label"]), "type_acq": sample["type_acq"], "spec_labels": sample.get("spec_labels", []), "spatial_shape": tuple(image.shape[1:])}
    torch.save(metadata, os.path.join(patient_dir, "metadata.pt"))

    #print(f"[SAVED] {patient_dir}")
    #print(f"        image shape: {tuple(image.shape)}")


def preprocess_split(samples, split):
    """
    Preprocesses all patients of one split, skipping those already saved on disk.

    :param samples: list of patient samples of the split
    :param split: "train", "val" or "test"
    """
    if split not in SPLITS:
        raise ValueError(f"Invalid split: {split}")

    split_dir = os.path.join(PREPROCESSED_DIR, split)
    os.makedirs(split_dir, exist_ok=True)

    # print("\n" + "=" * 70)
    # print(f"PREPROCESSING {split.upper()} SPLIT")
    # print("=" * 70)

    for idx, sample in enumerate(samples):
        patient_dir = os.path.join(split_dir, f"patient_{idx:04d}")
        files = [os.path.join(patient_dir, name) for name in ("image.pt", "aneurysm_mask.pt", "metadata.pt","bifurcations.pt")]

        if all(os.path.exists(path) for path in files):
            #print(f"[SKIP] {split.upper()} {idx + 1}/{len(samples)}")
            continue

        preprocess_patient(sample=sample, sample_idx=idx, split=split)

    # print("\n" + "=" * 70)
    # print(f"{split.upper()} PREPROCESSING COMPLETE")
    # print("=" * 70)


def preprocess_all_splits(labeled_sample):
    """
    Preprocesses the train, validation and test splits into separate directories.

    :param labeled_sample: dict with keys "train", "val" and "test", each a list of patient samples
    """
    for split in SPLITS:
        preprocess_split(labeled_sample[split], split)

    # print("\n" + "=" * 70)
    # print("ALL SPLITS PREPROCESSED")
    # print("=" * 70)