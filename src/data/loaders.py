import torch
from monai.data import DataLoader
from torch.utils.data import default_collate, Sampler
from src.config import DATASET_ROOT, LABELS_DIR, SPLIT_PATH, BATCH_PER_ARCH, NUM_WORKERS
from src.data.labels import build_bin_samples, find_binary_distribution
from src.data.split import build_train_val_test_split
from src.data.datasets import preprocessing
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


def Sliding_window_loader(samples, batch_size, split, is_training=True):
    """
    Build a DataLoader of 3D sliding window for one data split
    the volumes are read from the preprocessed directory of the gicen split and divided into overlapping patches.

    if is_training is true, the function performs the following actions
        -applies random augmentations to every patch
        - a PosNegSampler draws, at every epoch, all positive patches and a random
          subset of negatives
        - the last incomplete batch is dropped.

    :param samples: list of patient samples of this split (as returned by build_train_val_test_split)
    :param batch_size: number of patches per batch
    :param split: name of the split ("train", "val" or "test"); selects the preprocessed subdirectory
    :param is_training: True for the training loader (augmentation + balanced sampling), False for val/test
    :return: torch DataLoader yielding dicts with "image", "label", "patient_label", "sample_index", "type_acq", "spec_labels", "index"
    """
    transform = random_transform_dict() if is_training else None

    patches = SlidingWindowPatchDataset(samples=samples, split=split, transform=transform, overlap=0.25)

    if is_training:
        labels = patches.patch_labels()
        sampler = PosNegSampler(labels, neg_ratio=3)
        print(f"[{split}] pos={len(sampler.pos)} neg={len(sampler.neg)} -> {len(sampler)} patches/epochs")
    else:
        sampler = None

    loader = DataLoader(
        patches,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=is_training,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=(4 if NUM_WORKERS > 0 else None),
        collate_fn=patch_collate,
    )
    return loader

def patch_collate(batch):
    """
    Collate function for patch batches. Stacks all fields with default_collate,
    except "spec_labels", which has variable length and is kept as a list of lists.

    :param batch: list of sample dicts from SlidingWindowPatchDataset
    :return: batched dict; "spec_labels" is a list with one list per patch
    """

    # extract "spec_labels"
    spec = [b.pop("spec_labels") for b in batch]
    # stack the remaining fields into tensors
    out = default_collate(batch)
    # add "spec_labels" back as a list of lists
    out["spec_labels"] = spec
    return out



class PosNegSampler(Sampler):
    """
    Sampler that balances positive and negative patches during training.

    At every epoch it takes all positive patches and a new random subset of
    negatives (neg_ratio negatives per positive), then shuffles them together.
    Over several epochs all negatives are eventually seen.

    :param labels: list of patch labels (1 = positive, 0 = negative; other values are ignored)
    :param neg_ratio: number of negative patches sampled per positive patch
    """

    def __init__(self, labels, neg_ratio=3):
        # indices of positive and negative patches
        self.pos = [i for i, l in enumerate(labels) if l == 1]
        self.neg = [i for i, l in enumerate(labels) if l == 0]
        # number of negatives per epoch (capped at the available negatives)
        self.n_neg = min(len(self.neg), neg_ratio * len(self.pos))

    def __iter__(self):
        # new random subset of negatives for this epoch
        neg = torch.randperm(len(self.neg))[:self.n_neg].tolist()
        idx = self.pos + [self.neg[i] for i in neg]
        # shuffle positives and negatives together
        return iter([idx[i] for i in torch.randperm(len(idx)).tolist()])

    def __len__(self):
        # patches per epoch
        return len(self.pos) + self.n_neg


