import os, ast, glob

import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, RandFlip, RandRotate, RandZoom, RandScaleIntensity, RandShiftIntensity
from scipy.ndimage import binary_dilation
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from monai.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from binary_classification.whole_volume_rsna.data.transforms import eval_transforms, train_transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.environ.get("RSNA_DATA_DIR", os.path.join(PROJECT_ROOT, "rsna_dataset"))
FOLDERS = ["mra", "ct", "ct_2"]
CROP = (128, 128, 128)

ROI_CENTER = {"CTA": (95, 69, 84), "MRA": (95, 75, 97)}
ROI_SIZE = 96
MASK_DIR = os.path.join(DATA_DIR, "nnunet_masks")
SIGMA_PX = 8.0
HEAT_STRIDE = 32

def load_vessel_mask(uid):
    return np.load(os.path.join(MASK_DIR, f"{uid}.npz"))["mask"]


LOCATION_GROUPS = {
    "Left Infraclinoid Internal Carotid Artery": 1, "Right Infraclinoid Internal Carotid Artery": 1,
    "Left Supraclinoid Internal Carotid Artery": 1, "Right Supraclinoid Internal Carotid Artery": 1,
    "Left Middle Cerebral Artery": 2, "Right Middle Cerebral Artery": 2,
    "Anterior Communicating Artery": 3, "Left Anterior Cerebral Artery": 3, "Right Anterior Cerebral Artery": 3,
    "Left Posterior Communicating Artery": 4, "Right Posterior Communicating Artery": 4,
    "Basilar Tip": 5, "Other Posterior Circulation": 5,
}
GROUP_NAMES = ["any", "ICA", "MCA", "ACom_ACA", "PCom", "Posterior"]

LOCATIONS = list(LOCATION_GROUPS)
LOCATION_NAMES = ["any", "L_InfraICA", "R_InfraICA", "L_SupraICA", "R_SupraICA", "L_MCA", "R_MCA",
                  "ACom", "L_ACA", "R_ACA", "L_PCom", "R_PCom", "BasilarTip", "OtherPost"]


def location_scheme(n_outputs):
    if n_outputs == len(LOCATION_NAMES):
        return {name: i + 1 for i, name in enumerate(LOCATIONS)}, LOCATION_NAMES
    return LOCATION_GROUPS, GROUP_NAMES


def mirror_permutation(mapping, n_outputs):
    permutation = np.arange(n_outputs)
    for name, channel in mapping.items():
        mirrored = name.replace("Left", "#").replace("Right", "Left").replace("#", "Right")
        permutation[channel] = mapping[mirrored]
    return permutation

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


def load_points_with_groups(path=os.path.join(DATA_DIR, "localizers_npy.csv"), mapping=LOCATION_GROUPS):
    localizers = pd.read_csv(path)
    localizers = localizers[localizers["inside"]]
    unknown = set(localizers["location"]) - set(mapping)
    assert not unknown, f"unknown locations: {unknown}"
    localizers = localizers.assign(group=localizers["location"].map(mapping))
    return {uid: (group[["i", "j", "k"]].to_numpy(np.float32), group["group"].to_numpy(np.int64))
            for uid, group in localizers.groupby("uid")}


def load_location_labels(path=os.path.join(DATA_DIR, "train.csv"), mapping=LOCATION_GROUPS):
    table = pd.read_csv(path).set_index("SeriesInstanceUID")
    labels = np.zeros((len(table), max(mapping.values())), np.float32)
    for name, group in mapping.items():
        labels[:, group - 1] = np.maximum(labels[:, group - 1], table[name].to_numpy(np.float32))
    return dict(zip(table.index, labels))


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


SIGMA_MM = 2
def make_labels(slices, z_coords, aneurysm_slices, cutoff=0.05):
    labels = np.zeros(len(slices))
    for aneurysm in aneurysm_slices:
        if aneurysm not in slices:
            continue
        k = slices.index(aneurysm)
        distance = z_coords - z_coords[k]
        val = np.exp(-distance ** 2 / (2 * SIGMA_MM ** 2))
        labels = np.maximum(labels, val)
    labels[labels < cutoff] = 0
    return labels




CTA_RANGE = (-1.0, 1.0)    
MRA_RANGE = (-2.0, 8.0)


def read_slice(folder, uid, k, modality):
    value = np.asarray(Image.open(f"{folder}/{uid}/{k}.png"), dtype=np.float32)
    low, high = CTA_RANGE if modality == "CTA" else MRA_RANGE
    return value / 255 * (high - low) + low





def load_points_2d(csv_path):
    table = pd.read_csv(csv_path)
    return {uid: list(zip(g["slice"], g["x512"], g["y512"])) for uid, g in table.groupby("uid")}
def spatial_2d():
    return Compose([
        RandFlip(prob=0.5, spatial_axis=1),
        RandRotate(range_x=0.17, prob=0.5, padding_mode="border"),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
    ])


def intensity_2d():
    return Compose([
        RandScaleIntensity(factors=0.1, prob=0.5),
        RandShiftIntensity(offsets=0.1, prob=0.5),
    ])


class Dataset2D(Dataset):
    def __init__(self, table, folder, spatial=None, intensity=None, use_context=True,
                 multitask=False, points=None, series_labels=None):
        super().__init__()
        self.uids = table["uid"].tolist()
        self.slice_idx = table["slice"].tolist()
        self.labels = table["label"].astype("float32").tolist()
        self.modalities = table["modality"].tolist()
        self.n_slices = (table.groupby("uid")["slice"].max() + 1).to_dict()
        self.folder = folder
        self.spatial = spatial
        self.intensity = intensity
        self.use_context = use_context
        self.multitask = multitask
        self.points = points or {}
        self.series_labels = series_labels or {}
        self.yy, self.xx = torch.meshgrid(torch.arange(512.0), torch.arange(512.0), indexing="ij")

    def __len__(self):
        return len(self.uids)

    def make_heat(self, uid, k, soft_label):
        heat = torch.zeros(1, 512, 512)
        if soft_label > 0 and uid in self.points:
            s, x, y = min(self.points[uid], key=lambda p: abs(p[0] - k))
            heat[0] = soft_label * torch.exp(-((self.xx - x) ** 2 + (self.yy - y) ** 2) / (2 * SIGMA_PX ** 2))
        return heat

    def __getitem__(self, i):
        uid, k, modality, soft_label = self.uids[i], self.slice_idx[i], self.modalities[i], self.labels[i]
        n = self.n_slices[uid]
        neighbors = [max(k - 1, 0), k, min(k + 1, n - 1)] if self.use_context else [k]
        image = torch.from_numpy(np.stack([read_slice(self.folder, uid, j, modality) for j in neighbors]))

        heat = self.make_heat(uid, k, soft_label) if self.multitask else torch.zeros(1, 512, 512)
        if self.spatial:
            both = torch.as_tensor(self.spatial(torch.cat([image, heat])))
            image, heat = both[:-1], both[-1:]
        if self.intensity:
            image = torch.as_tensor(self.intensity(image))
        heat = F.max_pool2d(heat, HEAT_STRIDE)

        has_target = 0.0 if (self.series_labels.get(uid) == 1 and uid not in self.points) else 1.0
        if not self.multitask:
            has_target = 0.0
        return image, torch.tensor([soft_label]), heat, torch.tensor(has_target), uid, k





class MultiTaskVolumeDataset(Dataset):
    def __init__(self, table, transform, sigma, augm_proba, augment=False, roi_size=None, roi_jitter=0, n_outputs=1, vessel_dilation=None):
        super().__init__()
        self.paths = table["path"].tolist()
        self.labels = table["label"].astype("float32").tolist()
        self.uids = table["uid"].tolist()
        self.modalities = table["modality"].tolist()
        self.transform = transform
        self.n_outputs = n_outputs
        mapping, _ = location_scheme(n_outputs)
        self.mirror = mirror_permutation(mapping, n_outputs) if n_outputs > 1 else None
        self.points = load_points_with_groups(mapping=mapping)
        self.location_labels = load_location_labels(mapping=mapping) if n_outputs > 1 else None
        if self.location_labels is not None:
            missing = [uid for uid in self.uids if uid not in self.location_labels]
            assert not missing, f"series without location labels: {len(missing)}"
        self.sigma = sigma
        self.augment = augment
        self.augm_proba = augm_proba
        self.roi_size = roi_size
        self.roi_jitter = roi_jitter
        self.vessel_dilation = vessel_dilation

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
        mask = load_vessel_mask(uid) if self.vessel_dilation is not None else None
        points, groups = self.points.get(uid, (np.zeros((0, 3), np.float32), np.zeros(0, np.int64)))
        points = points.copy()

        if self.roi_size is not None:
            start = roi_start(modality, self.roi_size)
            if self.augment and self.roi_jitter > 0:
                start = start + torch.randint(-self.roi_jitter, self.roi_jitter + 1, (3,)).numpy()
            start = np.clip(start, 0, VOLUME_SIZE - self.roi_size)
            end = start + self.roi_size
            volume = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            if mask is not None:
                mask = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            points = points - start
            inside = ((points >= 0) & (points < self.roi_size)).all(axis=1)
            points, groups = points[inside], groups[inside]

        volume = torch.from_numpy(volume.astype(np.float32))[None]
        if self.transform is not None:
            volume = self.transform(volume)
        if mask is not None:
            keep = binary_dilation(mask > 0, iterations=self.vessel_dilation) if self.vessel_dilation > 0 else mask > 0
            volume = torch.where(torch.from_numpy(keep)[None], volume, volume.min())

        location_label = self.location_labels[uid] if self.n_outputs > 1 else None
        if self.augment and torch.rand(1) < self.augm_proba:
            volume = torch.flip(volume, dims=[1])
            points[:, 0] = (volume.shape[1] - 1) - points[:, 0]
            if self.mirror is not None:
                groups = self.mirror[groups]
                location_label = location_label[self.mirror[1:] - 1]

        cell_size = volume.shape[1] // GRID_SIZE
        heat_map = self.make_gaussian(points, self.sigma, cell_size)
        label = torch.tensor(self.labels[index])
        if self.n_outputs > 1:
            channels = [heat_map] + [self.make_gaussian(points[groups == g], self.sigma, cell_size) for g in range(1, self.n_outputs)]
            heat_map = torch.cat(channels)
            label = torch.tensor([self.labels[index], *location_label], dtype=torch.float32)
        has_target = 0.0 if self.labels[index] == 1 and len(points) == 0 else 1.0
        return volume, label, heat_map, torch.tensor(has_target)


def build_loader(index, is_training, batch_size, num_workers, use_amp, augment=True, multitask=False, sigma=1.0, augm_proba=0.5, roi_size=None, roi_jitter=0, n_outputs=1, vessel_dilation=None):
    use_augmentation = is_training and augment
    transform = train_transforms(spatial=not multitask) if use_augmentation else eval_transforms()
    if multitask:
        dataset = MultiTaskVolumeDataset(index, transform=transform, sigma=sigma, augm_proba=augm_proba, augment=use_augmentation, roi_size=roi_size, roi_jitter=roi_jitter, n_outputs=n_outputs, vessel_dilation=vessel_dilation)
    else:
        dataset = VolumeDataset(index, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_training, drop_last=is_training, num_workers=num_workers, pin_memory=use_amp)



def filter_training(index, series_info, max_thickness=2.0):
    info = series_info[series_info["uid"].isin(set(index["uid"]))]      # doar seriile din tabelul de felii
    thick = set(info.loc[info["slice_thickness"] > max_thickness, "uid"])

    max_label = index.groupby("uid")["label"].max()
    positive = info.loc[info["series_label"] == 1, "uid"]
    missing = {uid for uid in positive if max_label.get(uid, 0) == 0}

    print(f"excluded from train: {len(thick)} serii cu felii groase, {len(missing)} positives series w. no labels")
    return index[~index["uid"].isin(thick | missing)]


def transform_2d(is_training):
    if not is_training:
        return None
    return Compose([
        RandFlip(prob=0.5, spatial_axis=1),
        RandRotate(range_x=0.17, prob=0.5, padding_mode="border"),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        RandScaleIntensity(factors=0.1, prob=0.5),
        RandShiftIntensity(offsets=0.1, prob=0.5),
    ])

def build_2d_loader(index, series_info, folder, is_training, batch_size, num_workers, use_context=True, neg_ratio=3,multitask = False,points = None):
    transform = None
    if is_training:
        index = filter_training(index, series_info=series_info, max_thickness=2.0)
        transform = transform_2d(is_training)

    series_labels = dict(zip(series_info["uid"], series_info["series_label"])) if series_info is not None else {}
    dataset = Dataset2D(index, folder, spatial=spatial_2d() if is_training else None, intensity=intensity_2d() if is_training else None, use_context=use_context, multitask=multitask, points=points, series_labels=series_labels)
    if is_training:
        positive = (index["label"] > 0).to_numpy()
        n_pos, n_neg = positive.sum(), (~positive).sum()
        weights = np.where(positive, 1.0 / n_pos, neg_ratio / n_neg)
        num_samples = int(n_pos * (1 + neg_ratio))
        sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)




if __name__ == "__main__":
    for i in range(3):
        build_split(i)
    summaries = {}

    build_dataset_table()