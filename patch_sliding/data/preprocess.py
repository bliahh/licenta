import os

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize

from patch_sliding.config import PREPROCESSED_DIR
from patch_sliding.data.transforms import base_transform_dict

SPLITS = ("train", "val", "test")


def find_bifurcation(vessel_mask):
    """Bifurcation points of the vessel tree: skeleton voxels with 3 or more skeleton neighbours, one point (center of mass) per cluster."""
    skeleton_mask = skeletonize(vessel_mask > 0).astype(bool)
    kernel = np.ones((3, 3, 3), np.int16)
    neighbours = ndimage.convolve(skeleton_mask.astype(np.int16), kernel, mode="constant") - 1
    branch = skeleton_mask & (neighbours >= 3)
    labels, num_features = ndimage.label(branch, structure=np.ones((3, 3, 3)))
    if num_features == 0:
        return np.zeros((0, 3), dtype=np.int32)

    centers = ndimage.center_of_mass(branch, labels, range(1, num_features + 1))
    return np.round(np.asarray(centers)).astype(np.int32)


def preprocess_patient(sample, sample_idx, split):
    """
    Preprocesses one patient and saves the result to disk.

    Applies the deterministic transforms (load, orientation, resampling, crop, normalization)
    and saves image.pt, aneurysm_mask.pt, bifurcations.pt and metadata.pt in PREPROCESSED_DIR/<split>/patient_<idx>/.

    :param sample: patient sample (paths to image and masks, label, type_acq)
    :param sample_idx: index of the patient inside the split (used as folder name)
    :param split: "train", "val" or "test"
    """
    transform = base_transform_dict()
    data = {"image": sample["image"], "vessel_mask": sample["vessel_mask"], "aneurysm_mask": sample["aneurysm_mask"]}

    data = transform(data)
    image = data["image"]
    aneurysm_mask = data["aneurysm_mask"]

    vessel_mask = data["vessel_mask"][0].cpu().numpy()
    bifurcation = find_bifurcation(np.asarray(vessel_mask))

    print(f"[{split}] patient {sample_idx}: {len(bifurcation)} bifurcations")


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

    for idx, sample in enumerate(samples):
        patient_dir = os.path.join(split_dir, f"patient_{idx:04d}")
        files = [os.path.join(patient_dir, name) for name in ("image.pt", "aneurysm_mask.pt", "metadata.pt","bifurcations.pt")]

        if all(os.path.exists(path) for path in files):
            continue

        preprocess_patient(sample=sample, sample_idx=idx, split=split)


def preprocess_all_splits(labeled_sample):
    """
    Preprocesses the train, validation and test splits into separate directories.

    :param labeled_sample: dict with keys "train", "val" and "test", each a list of patient samples
    """
    for split in SPLITS:
        preprocess_split(labeled_sample[split], split)
