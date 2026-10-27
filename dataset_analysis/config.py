import os

from common.paths import DATASET_ROOT

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

ANEURYSM_MASK_DIR = os.path.join(DATASET_ROOT, "location_masks")

# connected components smaller than this are noise (1-2 voxels)
MIN_ANEURYSM_MM3 = 1.0
