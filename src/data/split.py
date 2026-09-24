import os
import json
import random


def split_list(lst,val_ratio,test_ratio):
    """
    split the entire dataset into train,test and val set
    :param lst: list of the positive or negative samples
    :param val_ratio: percentage of validation samples
    :param test_ratio: percentage of test samples
    :return: train,test,val split for positive and negative samples
    """
    n = len(lst)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val = lst[:n_val]
    test = lst[n_val:n_val + n_test]
    train = lst[n_val + n_test:]
    return train, val, test


def build_train_val_test_split(labels_dir, val_ratio=0.16, test_ratio=0.20, seed=42):
    """
    build train, test, and val split for binary labels and saves it in .json format
    :param labels_dir: directory w labels
    :param val_ratio: percentage of validation samples
    :param test_ratio: percentage of test samples
    :param seed: 42
    :return: split_data: dictionary k = type_samples, v = list for each set
    """
    split_path = os.path.join(labels_dir, "dataset_split.json")
    labels_path = os.path.join(labels_dir, "binary_labels.json")

    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            data = json.load(f)
            return data

    print(f"[INFO] Gen. NEW split {labels_path}...")
    with open(labels_path, "r") as f:
        samples = json.load(f)

    pos_samples = [s for s in samples if s["label"] == 1]
    neg_samples = [s for s in samples if s["label"] == 0]

    random.seed(seed)
    random.shuffle(pos_samples)
    random.shuffle(neg_samples)

    pos_train, pos_val, pos_test = split_list(pos_samples,val_ratio,test_ratio)
    neg_train, neg_val, neg_test = split_list(neg_samples,val_ratio,test_ratio)

    train = pos_train + neg_train
    val = pos_val + neg_val
    test = pos_test + neg_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    split_data = {"train": train, "val": val, "test": test}

    with open(split_path, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"[SUCCESS] Split saved {split_path}")
    return split_data


def extract_split(split_file):
    """
    retreive the .json file that contains the train/test/val dataset
    :param split_file: file that contains the dataset splitted
    :return: for each set, returns image,mask,label of each samples
    """
    with open(split_file, "r") as f:
        data = json.load(f)

    tr_imgs = [s["image"] for s in data["train"]]
    tr_masks = [s["vessel_mask"] for s in data["train"]]
    tr_lbls = [s["label"] for s in data["train"]]

    va_imgs = [s["image"] for s in data["val"]]
    va_masks = [s["vessel_mask"] for s in data["val"]]
    va_lbls = [s["label"] for s in data["val"]]

    te_imgs = [s["image"] for s in data["test"]]
    te_masks = [s["vessel_mask"] for s in data["test"]]
    te_lbls = [s["label"] for s in data["test"]]

    return tr_imgs, tr_masks, tr_lbls, va_imgs, va_masks, va_lbls, te_imgs, te_masks, te_lbls
