import json
from monai.data import CacheDataset, DataLoader, Dataset

from common.paths import DATASET_ROOT, SPLIT_PATH
from common.data.labels import build_bin_samples, find_binary_distribution
from common.data.split import build_train_val_test_split
from whole_volume.config import BATCH_PER_ARCH, NUM_WORKERS
from whole_volume.data.datasets import preprocessing
from whole_volume.data.transforms import eval_transform_dict

_DATASETS = None


def get_datasets():
    """
    for the whole dataset, the function returns the training , validation and test datasets already preprocessed
    the dataset split is taken from the dataset_split.json contained in LABELS_DIR directory
    :return: a tuple containing train, validation and test datasets
    """
    global _DATASETS
    if _DATASETS is None:
        print("\n" + "=" * 60)
        print("BUILDING DATASET....")
        print("=" * 60)
        build_bin_samples(DATASET_ROOT)
        find_binary_distribution()
        build_train_val_test_split()
        _DATASETS = preprocessing(SPLIT_PATH)
    return _DATASETS


def build_loaders(name,batch_size = None):
    """
    for the whole dataset, the function returns the training and validation loaders already preprocessed
    :param name: name of the models / arhitecture (this is only for printing, the type of model doesen't affect this function)
    :param batch_size: batch size
    :return: a tuple containing train, validation and test loaders
    """
    train_ds, val_ds, test_ds = get_datasets()
    if batch_size is None:
        batch_size = BATCH_PER_ARCH.get(name, 2)
    print(f"[LOADERS] {name} -> batch_size={batch_size}")

    train_loader = DataLoader( train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0), prefetch_factor=(2 if NUM_WORKERS > 0 else None))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0), prefetch_factor=(2 if NUM_WORKERS > 0 else None))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,num_workers=NUM_WORKERS, pin_memory=True,persistent_workers=(NUM_WORKERS > 0),prefetch_factor=(2 if NUM_WORKERS > 0 else None))
    return train_loader, val_loader, test_loader


def _split_samples(name):
    """Samples of one split from dataset_split.json, with only the fields the volume pipeline needs."""
    with open(SPLIT_PATH, "r") as f:
        split = json.load(f)
    return [{k: s[k] for k in ("image", "vessel_mask", "label", "type_acq")} for s in split[name]]


def volume_loaders(batch_size=2, num_workers=2):
    """Val and test loaders of the whole-volume pipeline, read from dataset_split.json and cached once (train is not loaded)."""
    loaders = {}
    for name in ("val", "test"):
        data = _split_samples(name)
        ds = CacheDataset(data=data, transform=eval_transform_dict(), cache_rate=1.0, num_workers=num_workers)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"[{name}] {len(data)} patients")
    return loaders["val"], loaders["test"]


def volume_test_loader(thick_mm=None, batch_size=2):
    """Test loader of the whole-volume pipeline without caching: every patient is preprocessed when needed and released after its batch."""
    data = _split_samples("test")
    return DataLoader(Dataset(data=data, transform=eval_transform_dict(thick_mm)), batch_size=batch_size, shuffle=False, num_workers=0)
