import os
import numpy as np
import SimpleITK as sitk

from common.paths import THIN_MM


def read_spacing(path):
    """Spacing and size of an image, read from the header only (the voxels are not loaded)."""
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()
    return r.GetSpacing(), r.GetSize()


def center_of(path):
    return os.path.basename(path).split("_")[1]


def patient_groups(images, modalities, thickest):
    """Boolean mask per patient group: all, ct, mr, mr_thin, mr_thick, center1_mr."""
    acq = np.array([str(m).lower() for m in modalities])
    center = np.array([center_of(p) for p in images])
    thick = np.asarray(thickest, dtype=float)

    is_ct = acq == "ct"
    is_mr = acq == "mr"
    is_thin = thick <= THIN_MM
    is_center1 = center == "center1"

    groups = {}
    groups["all"] = np.ones(len(acq), dtype=bool)
    groups["ct"] = is_ct
    groups["mr"] = is_mr
    groups["mr_thin"] = is_mr & is_thin
    groups["mr_thick"] = is_mr & ~is_thin
    groups["center1_mr"] = is_mr & is_center1
    return groups
