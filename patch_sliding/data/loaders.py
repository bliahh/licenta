from monai.data import DataLoader
from torch.utils.data import default_collate

from patch_sliding.config import NUM_WORKERS, PATCH, SAMPLER
from patch_sliding.data.datasets import SlidingWindowPatchDataset
from patch_sliding.data.samplers import build_sampler
from patch_sliding.data.transforms import random_transform_dict


def sliding_window_loader(samples, batch_size, split, is_training=True, patch=None, sampler=None):
    """
    Build a DataLoader of 3D sliding window for one data split
    the volumes are read from the preprocessed directory of the gicen split and divided into overlapping patches.

    if is_training is true, the function performs the following actions
        -applies random augmentations to every patch
        - the sampler (SAMPLER in config) draws, at every epoch, all positive patches and a random
          subset of negatives; without a sampler all patches are shuffled
        - the last incomplete batch is dropped.

    :param samples: list of patient samples of this split (as returned by build_train_val_test_split)
    :param batch_size: number of patches per batch
    :param split: name of the split ("train", "val" or "test"); selects the preprocessed subdirectory
    :param is_training: True for the training loader (augmentation + balanced sampling), False for val/test
    :param patch: patch settings (patch_size, overlap, min_bif, th_fraction); PATCH from config if None
    :param sampler: sampler settings (name, neg_ratio, hard_ratio); SAMPLER from config if None
    :return: torch DataLoader yielding dicts with "image", "label", "patient_label", "sample_index", "type_acq", "spec_labels", "index"
    """
    patch = patch or PATCH
    sampler_cfg = sampler or SAMPLER
    transform = random_transform_dict() if is_training else None

    patches = SlidingWindowPatchDataset(samples=samples, split=split, transform=transform, **patch)

    sampler = None
    if is_training:
        sampler = build_sampler(patches.patch_labels(), **sampler_cfg)
        if sampler is not None:
            print(f"[{split}] pos={len(sampler.pos)} neg={len(sampler.neg)} -> {len(sampler)} patches/epochs")

    loader = DataLoader(
        patches,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=is_training and sampler is None,
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
