import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from common.paths import PATIENTS_CSV
from common.data.patient_info import read_spacing, center_of, patient_groups
from dataset_analysis.config import OUT_DIR


def build_patient_table(labeled_sample, out_csv=PATIENTS_CSV):
    """One row per patient: split, center, modality, label, spacing and size, read from the image headers."""
    rows = []
    for split, samples in labeled_sample.items():
        for s in samples:
            (sx, sy, sz), (nx, ny, nz) = read_spacing(s["image"])
            rows.append({"split": split, "image": s["image"], "center": center_of(s["image"]), "modality": s["type_acq"].upper(), "label": "aneurysm" if int(s["label"]) == 1 else "healthy", "sx": sx, "sz": sz, "thickest": max(sx, sy, sz), "nx": nx, "ny": ny, "nz": nz})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv} ({len(df)} patients)")
    return df


def dataset_report(df, out_dir=OUT_DIR):
    """
    Tables for the report, printed and saved as CSV:
      composition        patients per center x modality x label, per split
      protocols          median in-plane spacing, median / min / max slice thickness per center x modality x label
      center1_protocols  scan protocols inside center1 (MR) with the number of healthy / aneurysm patients
      shortcut_auc       how well acquisition alone (no image) predicts the label
    """
    composition = pd.crosstab([df["center"], df["modality"], df["label"]], df["split"])[["train", "val", "test"]]
    composition["total"] = composition.sum(axis=1)

    protocols = df.groupby(["center", "modality", "label"]).agg(patients=("image", "count"), inplane_median=("sx", "median"), slice_median=("sz", "median"), slice_min=("sz", "min"), slice_max=("sz", "max")).round(2)

    c1 = df[(df["center"] == "center1") & (df["modality"] == "MR")].copy()
    c1["protocol"] = c1["sx"].round(2).astype(str) + " x " + c1["sz"].round(2).astype(str)
    c1_protocols = pd.crosstab(c1["protocol"], c1["label"])
    c1_protocols["total"] = c1_protocols.sum(axis=1)
    c1_protocols = c1_protocols.sort_values("total", ascending=False)

    mr = df[df["modality"] == "MR"]
    y_mr, y_c1 = (mr["label"] == "aneurysm").astype(int), (c1["label"] == "aneurysm").astype(int)
    shortcuts = pd.Series({"thickness alone": roc_auc_score(y_mr, mr["sz"]), "MR: 'not center1' alone": roc_auc_score(y_mr, (mr["center"] != "center1").astype(int)), "center1 MR: slice thickness alone": roc_auc_score(y_c1, c1["sz"]), "center1 MR: in-plane spacing alone": roc_auc_score(y_c1, -c1["sx"])}, name="auc").round(3)

    os.makedirs(out_dir, exist_ok=True)
    for name, table in (("composition", composition), ("protocols", protocols), ("center1_protocols", c1_protocols), ("AUC BASED", shortcuts)):
        print(f"\n=== {name.upper()} ===\n{table}")
        table.to_csv(os.path.join(out_dir, f"dataset_{name}.csv"))
    return composition, protocols, c1_protocols, shortcuts


def group_counts(patients_csv=PATIENTS_CSV, out_csv=os.path.join(OUT_DIR, "dataset_group_counts.csv")):
    """Number of patients (aneurysm / healthy) in every patient group, per split."""
    df = pd.read_csv(patients_csv)
    rows = []
    for split in ("train", "val", "test", "all"):
        d = df if split == "all" else df[df["split"] == split]
        y = (d["label"] == "aneurysm").to_numpy()
        for g, mask in patient_groups(d["image"], d["modality"], d["thickest"]).items():
            rows.append({"split": split, "group": g, "n": int(mask.sum()), "aneurysm": int(y[mask].sum()), "healthy": int((~y[mask]).sum())})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(out.pivot(index="group", columns="split", values=["n", "aneurysm", "healthy"]).reindex(["all", "ct", "mr", "mr_thin", "mr_thick", "center1_mr"]))
    print(f"\n[SAVED] {out_csv}")
    return out


def aneurysm_sizes(mask_paths):
    """Volume and equivalent sphere diameter of every aneurysm (connected component of the location masks)."""
    rows = []
    for path in mask_paths:
        img = sitk.ReadImage(path)
        mask = sitk.GetArrayFromImage(img) > 0
        voxel_mm3 = float(np.prod(img.GetSpacing()))
        comps, n = ndimage.label(mask)
        for i in range(1, n + 1):
            vol = (comps == i).sum() * voxel_mm3
            rows.append({"file": os.path.basename(path), "center": center_of(path), "volume_mm3": vol, "diameter_mm": (6 * vol / np.pi) ** (1 / 3)})
    df = pd.DataFrame(rows)
    print(df.groupby("center")[["volume_mm3", "diameter_mm"]].describe().round(1))
    return df
