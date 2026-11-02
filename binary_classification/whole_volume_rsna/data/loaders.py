import os, ast, glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.data import DataLoader
from sklearn.model_selection import train_test_split

from binary_classification.whole_volume_rsna.data.transforms import eval_transforms, train_transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "rsna_dataset")
FOLDERS = ["mra", "ct", "ct_2"]
CROP = (128, 128, 128)

ROI_CENTER = {"CTA": (95, 69, 84), "MRA": (95, 75, 97)}
ROI_SIZE = 96

def build_dataset_table():
    tables = []
    for folder in FOLDERS:
        for log_path in glob.glob(os.path.join(DATA_DIR, folder, "log_*.csv")):
            table = pd.read_csv(log_path)
            table["path"] = os.path.join(DATA_DIR, folder, "preprocessed") + "/" + table["uid"] + ".npy"
            tables.append(table)
    index = pd.concat(tables, ignore_index=True)

    n_before = len(index)
    index = index[index["ok"] & (index["n_files"] <= 2)].reset_index(drop=True)
    print(f"[ BUILD DATASET TABLE ] dropped: {n_before - len(index)}")

    index[["x", "y", "z"]] = index["shape"].apply(ast.literal_eval).tolist()
    index["large_coverage"] = index["z"] > 192

    bad_scale = (index["modality"] == "CTA") & (index["p90"] < -3000)
    print(f"[ BUILD DATASET TABLE ] CTA BAD_SCALE: {bad_scale.sum()}")
    index = index[~bad_scale].reset_index(drop=True)

    missing = index[~index["path"].map(os.path.exists)]
    if len(missing) > 0:
        print("missing:", len(missing))
        print(missing["path"].str.split("/").str[-3].value_counts())
        print(missing["path"].head(3).tolist())
    assert len(missing) == 0, "missing .npys "
    assert index["uid"].is_unique, "uid is not unique:"

    print(pd.crosstab(index["modality"], index["label"]))

    index.to_csv(os.path.join(DATA_DIR, "index.csv"), index=False)
    return index


def build_split(seed):
    index = pd.read_csv(os.path.join(DATA_DIR, "index.csv"))
    strata = index["modality"] + "_" + index["label"].astype(str)
    train_val, test = train_test_split(index, test_size=0.15, stratify=strata, random_state=seed)
    strata_tv = train_val["modality"] + "_" + train_val["label"].astype(str)
    train, val = train_test_split(train_val, test_size=0.15 / 0.85, stratify=strata_tv, random_state=seed)

    index["split"] = "train"
    index.loc[val.index, "split"] = "val"
    index.loc[test.index, "split"] = "test"
    index.to_csv(os.path.join(DATA_DIR, f"split_seed{seed}.csv"), index=False)
    return index


def load_split(seed, only_mr=False, only_ct=False):
    split = pd.read_csv(os.path.join(DATA_DIR, f"split_seed{seed}.csv"))
    if only_mr:
        split = split[split["modality"] == "MRA"]
    elif only_ct:
        split = split[split["modality"] == "CTA"]
    return split[split["split"] == "train"], split[split["split"] == "val"], split[split["split"] == "test"]

def load_points(path=os.path.join(DATA_DIR, "localizers_npy.csv")):
    localizers = pd.read_csv(path)
    localizers = localizers[localizers["inside"]]
    return {uid: group[["i", "j", "k"]].to_numpy(np.float32) for uid, group in localizers.groupby("uid")}


VOLUME_SIZE = 192
GRID_SIZE = 12


def roi_start(modality, roi_size):
    return np.array(ROI_CENTER[modality]) - roi_size // 2


def points_to_roi(points, modality, roi_size):
    shifted_points = points - roi_start(modality, roi_size)
    return shifted_points[((shifted_points >= 0) & (shifted_points < roi_size)).all(axis=1)]

class VolumeDataset(Dataset):
    def __init__(self, table, transform):
        super().__init__()
        self.paths = table["path"].tolist()
        self.labels = table["label"].astype("float32").tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = torch.from_numpy(np.load(self.paths[i]).astype(np.float32))[None]
        return self.transform(x), torch.tensor(self.labels[i])

class MultiTaskVolumeDataset(Dataset):
    def __init__(self, table, transform, sigma, augm_proba, augment=False, roi_size=None):
        super().__init__()
        self.paths = table["path"].tolist()
        self.labels = table["label"].astype("float32").tolist()
        self.uids = table["uid"].tolist()
        self.modalities = table["modality"].tolist()
        self.transform = transform
        self.points = load_points()
        self.sigma = sigma
        self.augment = augment
        self.augm_proba = augm_proba
        self.roi_size = roi_size

    def __len__(self):
        return len(self.paths)

    def make_gaussian(self, points, sigma, cell_size):
        grid = np.indices((GRID_SIZE, GRID_SIZE, GRID_SIZE))
        target = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for cell in points // cell_size:
            squared_distance = ((grid - cell[:, None, None, None]) ** 2).sum(axis=0)
            target = np.maximum(target, np.exp(-squared_distance / (2 * sigma ** 2)))
        return torch.from_numpy(target.astype(np.float32))[None]

    def __getitem__(self, index):
        uid, modality = self.uids[index], self.modalities[index]
        volume = np.load(self.paths[index], mmap_mode="r")
        points = self.points.get(uid, np.zeros((0, 3), np.float32)).copy()

        if self.roi_size is not None:
            start = roi_start(modality, self.roi_size)
            end = start + self.roi_size
            volume = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            points = points_to_roi(points, modality, self.roi_size)

        volume = torch.from_numpy(volume.astype(np.float32))[None]
        if self.transform is not None:
            volume = self.transform(volume)

        if self.augment and torch.rand(1) < self.augm_proba:
            volume = torch.flip(volume, dims=[1])
            points[:, 0] = (volume.shape[1] - 1) - points[:, 0]

        heat_map = self.make_gaussian(points, self.sigma, volume.shape[1] // GRID_SIZE)
        has_target = 0.0 if self.labels[index] == 1 and len(points) == 0 else 1.0
        return volume, torch.tensor(self.labels[index]), heat_map, torch.tensor(has_target)


def build_loader(index, is_training, batch_size, num_workers, use_amp, augment=True, multitask=False, sigma=1.0, augm_proba=0.5, roi_size=None):
    use_augmentation = is_training and augment
    transform = train_transforms(spatial=not multitask) if use_augmentation else eval_transforms()
    if multitask:
        dataset = MultiTaskVolumeDataset(index, transform=transform, sigma=sigma, augm_proba=augm_proba, augment=use_augmentation, roi_size=roi_size)
    else:
        dataset = VolumeDataset(index, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_training, drop_last=is_training, num_workers=num_workers, pin_memory=use_amp)


if __name__ == "__main__":
    for i in range(3):
        build_split(i)
    summaries = {}

    build_dataset_table()