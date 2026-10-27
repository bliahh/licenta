import os
import glob

from common.data.labels import build_bin_samples, find_binary_distribution, print_distribution
from common.data.split import build_train_val_test_split
from dataset_analysis.analysis.general import build_patient_table, dataset_report, group_counts, aneurysm_sizes
from dataset_analysis.analysis.patch import patch_distribution
from dataset_analysis.config import OUT_DIR, ANEURYSM_MASK_DIR, MIN_ANEURYSM_MM3


def prepare_labels():
    """binary_labels.json + dataset_split.json (the existing split is kept), then the distribution per split."""
    build_bin_samples()
    find_binary_distribution()
    build_train_val_test_split()
    print_distribution()


def report():
    """patients.csv + the dataset tables (composition, protocols, center1 protocols, shortcut AUC)."""
    df = build_patient_table(build_train_val_test_split())
    dataset_report(df)


def groups():
    """Patients per group (CT / MR / thin / thick / center1) and split; needs patients.csv (report)."""
    group_counts()


def sizes(mask_dir=ANEURYSM_MASK_DIR):
    """Aneurysm volumes and diameters per center -> aneurysm_sizes.csv and aneurysm_sizes_summary.csv"""
    paths = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
    print(len(paths), paths[:3])
    df = aneurysm_sizes(paths)
    df = df[df["volume_mm3"] >= MIN_ANEURYSM_MM3]
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "aneurysm_sizes.csv"), index=False)
    summary = df.groupby("center")[["volume_mm3", "diameter_mm"]].describe().round(1)
    summary.to_csv(os.path.join(OUT_DIR, "aneurysm_sizes_summary.csv"))
    print(summary)


def patches():
    """Patch categories and bifurcation counts per split, with PATCH from patch_sliding/config.py."""
    patch_distribution(build_train_val_test_split())


if __name__ == "__main__":
    report()
