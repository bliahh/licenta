from monai.data import DataLoader
from src.config import DATASET_ROOT, LABELS_DIR, SPLIT_PATH, BATCH_PER_ARCH, NUM_WORKERS
from src.data.labels import build_bin_samples, find_binary_distribution
from src.data.split import build_train_val_test_split
from src.data.datasets import preprocessing
from functools import lru_cache
from src.data.transforms import random_transform_dict, base_transform_dict, SlidingWindowPatchDataset

# @lru_cache(maxsize=None)
# def get_datasets():
#     print("\n" + "-" * 60)
#     print("BUILD DATASET")
#     print("-" * 60)
#     build_bin_samples(DATASET_ROOT)
#     find_binary_distribution(LABELS_DIR)
#     build_train_val_test_split(LABELS_DIR)
#     return preprocessing(SPLIT_PATH)

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
        build_train_val_test_split(LABELS_DIR)
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


def Sliding_window_loader(sample,batch_size,is_training = True):
    base_transform = base_transform_dict()
    sample = base_transform(sample)

    if is_training:
        patches = SlidingWindowPatchDataset(sample,random_transform_dict())
    else:
        patches = SlidingWindowPatchDataset(sample)
    loader = DataLoader( patches, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0), prefetch_factor=(2 if NUM_WORKERS > 0 else None))
    return loader


