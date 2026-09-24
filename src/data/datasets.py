from monai.data import CacheDataset, Dataset
from src.data.split import extract_split
from src.data.transforms import base_transform_dict, random_transform_dict, eval_transform_dict


def to_dict_list(image_files, vessel_masks, labels):
    """
       serialize all images, labels and masks into a list of dicts
       :param image_files: list of paths to the image files
       :param vessel_masks: list of paths to the corresponding vessel mask volumes
       :param labels: list of binary labels (0/1) aligned with image_files
       :return: list of dicts, one per sample: {"image": ..., "vessel_mask": ..., "label": ...}
       """
    return [ {"image": img, "vessel_mask": mask, "label": lbl} for img, mask, lbl in zip(image_files, vessel_masks, labels) ]


def preprocessing(file,cache_rate = 0.3,num_workers = 1):
    """
    does the preprocessing pipeline, performs the trasnformation on the dataset
    :param file: input file
    :return: train_dataset, validation_dataset, test_dataset
    """
    print("[INFO] preprocessing....")
    tr_imgs, tr_masks, tr_lbls, va_imgs, va_masks, va_lbls, te_imgs, te_masks, te_lbls = extract_split(file)

    train_data = to_dict_list(tr_imgs, tr_masks, tr_lbls)
    val_data = to_dict_list(va_imgs, va_masks, va_lbls)
    test_data = to_dict_list(te_imgs, te_masks, te_lbls)

    print(f"[CACHE] pre-processing {len(train_data)} training volumes deterministic part")
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
