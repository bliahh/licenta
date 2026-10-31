import os

import numpy as np
import pandas as pd
from monai.data.utils import nib

from common.paths import DATASET_ROOT


SP = ["spacing_x", "spacing_y", "spacing_z"]
DUPLICATES = ["topaneu_center4_ct_008_2_0000.nii.gz"]

def extract_headers(data_root=DATASET_ROOT):
    print("Extracting headers...")

    img_dir = os.path.join(data_root, "images")
    all_images = []
    for file in sorted(os.listdir(img_dir)):
        if not file.endswith((".nii", ".nii.gz")):
            continue
        img = nib.load(os.path.join(img_dir, file))
        print(img)
        all_images.append((file,img))
    return all_images

def attach_dim_info(files, data_root=DATASET_ROOT):
    img_dir = os.path.join(data_root, "images")
    rows = []
    for file in files:
        img = nib.load(os.path.join(img_dir, file))
        dim_x, dim_y, dim_z = img.shape[:3]
        rows.append({"file": file, "dim_x": dim_x, "dim_y": dim_y, "dim_z": dim_z})
    return pd.DataFrame(rows)


def attach_spacing_info(files, data_root=DATASET_ROOT):
    img_dir = os.path.join(data_root, "images")
    rows = []
    for file in files:
        img = nib.load(os.path.join(img_dir, file))
        sp_x, sp_y, sp_z = img.header.get_zooms()[:3]
        rows.append({"file": file, "spacing_x": sp_x, "spacing_y": sp_y, "spacing_z": sp_z})
    return pd.DataFrame(rows)

def attach_center_info(files):
    rows = []
    for file in files:
        center = int(file.split("_")[1][-1])
        rows.append({"file": file, "center": center})
    return pd.DataFrame(rows)

def attach_modality_info(files):
    rows = []
    for file in files:
        modality = file.split("_")[2]
        rows.append({"file": file, "modality": modality})
    return pd.DataFrame(rows)

def attach_label_info(files):
    rows = []
    for file in files:
        rows.append({"file": os.path.basename(file["image"]),
                     "label": int(file["label"])})
    return pd.DataFrame(rows)





def attach_fov_info(meta):
    out = pd.DataFrame({"file": meta["file"]})
    for ax in ["x", "y", "z"]:
        out[f"fov_{ax}"] = meta[f"dim_{ax}"] * meta[f"spacing_{ax}"]
    return out


def attach_voxel_info(meta):
    out = pd.DataFrame({"file": meta["file"]})
    out["voxel_volume"] = meta["spacing_x"] * meta["spacing_y"] * meta["spacing_z"]
    out["n_voxels"] = meta["dim_x"] * meta["dim_y"] * meta["dim_z"]
    return out


def attach_slice_info(meta):
    out = pd.DataFrame({"file": meta["file"]})
    out["thickest"] = meta[SP].max(axis=1)
    out["inplane"] = meta[SP].min(axis=1)
    return out


def attach_conversion_info(files, data_root=DATASET_ROOT):
    img_dir = os.path.join(data_root, "images")
    rows = []
    for file in files:
        img = nib.load(os.path.join(img_dir, file))
        hdr = img.header
        slope, inter = img.dataobj.slope, img.dataobj.inter
        descrip = hdr["descrip"].item().decode(errors="ignore").strip()
        rows.append({"file": file,
                     "datatype": str(hdr.get_data_dtype()),
                     "dim_info": int(hdr["dim_info"]),
                     "qform_code": int(hdr["qform_code"]),
                     "sform_code": int(hdr["sform_code"]),
                     "has_scaling": int(slope != 1 or inter != 0),
                     "xyzt_units": int(hdr["xyzt_units"]),
                     "has_descrip": int(descrip != ""),
                     "cal_min": float(hdr["cal_min"]),
                     "cal_max": float(hdr["cal_max"]),
                     "slice_code": int(hdr["slice_code"]),
                     "slice_duration": float(hdr["slice_duration"]),
                     "toffset": float(hdr["toffset"]),
                     "intent_code": int(hdr["intent_code"])})
    return pd.DataFrame(rows)




def attach_orientation_info(files, data_root=DATASET_ROOT):
    img_dir = os.path.join(data_root, "images")
    rows = []
    for file in files:
        img = nib.load(os.path.join(img_dir, file))
        rows.append({"file": file,
                     "orientation": "".join(nib.aff2axcodes(img.affine)),
                     "obliquity_deg": float(np.degrees(np.abs(nib.affines.obliquity(img.affine))).max())})
    return pd.DataFrame(rows)

def drop_constant(meta):
    const = [c for c in meta.columns if meta[c].nunique(dropna=False) <= 1]
    print("coloane ct:", const)
    return meta.drop(columns=const)

def attach_intensity_info(files, data_root=DATASET_ROOT, step=2):
    img_dir = os.path.join(data_root, "images")
    rows = []
    i = 0
    n = len(files)
    for file in files:

        print(f"[INTENSITY_INFO] preprocessing intensity table {i}/{n} ")
        i+=1
        img = nib.load(os.path.join(img_dir, file))
        data = np.asarray(img.dataobj[::step, ::step, ::step], dtype=np.float32)
        p1, p50, p99 = np.percentile(data, [1, 50, 99])
        rows.append({"file": file,
                     "int_min": float(data.min()), "int_max": float(data.max()),
                     "int_mean": float(data.mean()), "int_std": float(data.std()),
                     "int_p1": float(p1), "int_p50": float(p50), "int_p99": float(p99),
                     "int_range": float(p99 - p1),
                     "frac_min": float((data == data.min()).mean()),
                     "has_negative": int(data.min() < 0),
                     "n_unique": int(np.unique(data).size)})
        del data
    return pd.DataFrame(rows)



def attach_descrip_info(files, data_root=DATASET_ROOT):
    img_dir = os.path.join(data_root, "images")
    rows = []
    for file in files:
        img = nib.load(os.path.join(img_dir, file))
        descrip = img.header["descrip"].item().decode(errors="ignore").strip()
        parts = file.split("_")
        rows.append({"file": file,
                     "center": parts[1],
                     "patient": parts[1] + "_" + parts[3],
                     "descrip": descrip})
    return pd.DataFrame(rows)


def attach_patient_info(files):
    rows = []
    for file in files:
        parts = file.split("_")
        patient = parts[1] + "_" + parts[3]
        scan = int(parts[4]) if len(parts) == 6 else 0
        rows.append({"file": file, "patient": patient, "scan": scan})
    return pd.DataFrame(rows)


def build_dataset_table(samples, data_root=DATASET_ROOT, include_center=False, include_intensity=True, only_center=None, only_modality=None, intensity_csv="intensity.csv"):

    print("start building dataset table")

    img_dir = os.path.join(data_root, "images")
    files = sorted(f for f in os.listdir(img_dir) if f.endswith((".nii", ".nii.gz")))


    img_dir = os.path.join(data_root, "images")
    files = sorted(f for f in os.listdir(img_dir) if f.endswith((".nii", ".nii.gz")))
    files = [f for f in files if f not in DUPLICATES]


    # grupul 2: geometrie
    table = attach_dim_info(files, data_root).merge(attach_spacing_info(files, data_root), on="file")
    for attach in [attach_fov_info, attach_voxel_info, attach_slice_info]:
        table = table.merge(attach(table), on="file")

    for attach in [attach_conversion_info, attach_orientation_info]:
        table = table.merge(attach(files, data_root), on="file")

    for attach in [attach_modality_info, attach_center_info, attach_patient_info]:
        table = table.merge(attach(files), on="file")

    if include_intensity:
        if os.path.exists(intensity_csv):
            intensity = pd.read_csv(intensity_csv)
        else:
            intensity = attach_intensity_info(files, data_root)
            intensity.to_csv(intensity_csv, index=False)
        table = table.merge(intensity, on="file")

    table = table.merge(attach_label_info(samples), on="file")

    if only_center is not None:
        table = table[table["center"] == only_center]
    if only_modality is not None:
        table = table[table["modality"] == only_modality]
    if not include_center:
        table = table.drop(columns=["center"])

    table = drop_constant(table.reset_index(drop=True))
    for column in ["modality", "datatype", "orientation"]:
        if column in table:
            table[column] = table[column].astype("category")
    return table


