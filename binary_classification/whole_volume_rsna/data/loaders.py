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


def build_loader(index, is_training, batch_size, num_workers, use_amp, augment= False):
    transform = train_transforms() if (is_training and augment) else eval_transforms()
    dataset = VolumeDataset(index, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_training, drop_last=is_training, num_workers=num_workers, pin_memory=use_amp)



if __name__ == "__main__":
    for i in range(3):
        build_split(i)
    summaries = {}

    build_dataset_table()