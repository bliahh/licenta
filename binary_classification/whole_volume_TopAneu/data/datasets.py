from monai.data import CacheDataset, Dataset

from common.data.split import extract_split
from whole_volume.config import CACHE_RATE
from whole_volume.data.transforms import base_transform_dict, random_transform_dict, eval_transform_dict


def to_dict_list(image_files, vessel_masks, labels):
    """
       serialize all images, labels and masks into a list of dicts
       :param image_files: list of paths to the image files
       :param vessel_masks: list of paths to the corresponding vessel mask volumes
       :param labels: list of binary labels (0/1) aligned with image_files
       :return: list of dicts, one per sample: {"image": ..., "vessel_mask": ..., "label": ...}
       """
    return [ {"image": img, "vessel_mask": mask, "label": lbl} for img, mask, lbl in zip(image_files, vessel_masks, labels) ]


def preprocessing(file,cache_rate = CACHE_RATE,num_workers = 1):
    """
    does the preprocessing pipeline, performs the trasnformation on the dataset
    :param file: input file
    :return: train_dataset, validation_dataset, test_dataset
    """
    print("[INFO] preprocessing....")
    tr_imgs, tr_vmasks, _, tr_lbls, va_imgs, va_vmasks, _, va_lbls, te_imgs, te_vmasks, _, te_lbls = extract_split(file)

    train_data = to_dict_list(tr_imgs, tr_vmasks, tr_lbls)
    val_data = to_dict_list(va_imgs, va_vmasks, va_lbls)
    test_data = to_dict_list(te_imgs, te_vmasks, te_lbls)

    print(f"[CACHE] pre-processing {len(train_data)} training volumes ")
    train_cache = CacheDataset(data=train_data, transform=base_transform_dict(), cache_rate=cache_rate, num_workers=num_workers)
    train_dataset = Dataset(data=train_cache, transform=random_transform_dict())
    print("[CACHE] done.")

    print(f"[CACHE] pre-processing {len(val_data)} validation volumes...")
    validation_dataset = CacheDataset(data=val_data, transform=eval_transform_dict(), cache_rate=cache_rate, num_workers=num_workers)
    print("[CACHE] done.")

    print(f"[CACHE] pre-processing {len(test_data)} test volumes...")
    test_dataset = CacheDataset(data=test_data, transform=eval_transform_dict(), cache_rate=cache_rate, num_workers=num_workers)
    print("[CACHE] done.")

    print("[DONE] preprocessing ended, train, val and test datasets built and ready to use")

    return train_dataset, validation_dataset, test_dataset
