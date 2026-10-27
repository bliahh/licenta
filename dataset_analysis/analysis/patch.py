import os
import numpy as np
import pandas as pd

from dataset_analysis.config import OUT_DIR
from patch_sliding.config import PATCH, POSITIVE, WEAK_POSITIVE, NEG_BIF, NEGATIVE
from patch_sliding.data.datasets import SlidingWindowPatchDataset

CATEGORY_NAMES = {POSITIVE: "pos", WEAK_POSITIVE: "ambiguous", NEG_BIF: "neg_bif", NEGATIVE: "neg"}


def patch_distribution(labeled_sample, patch=None, out_csv=os.path.join(OUT_DIR, "patch_distribution.csv")):
    """
    How the preprocessed patients split into patches with the given settings (PATCH from patch_sliding/config.py if None):
    number of patches per category and split, and the percentiles of the bifurcation count of the negative patches
    (useful to choose min_bif). Needs the patch preprocessing (patch_sliding.main.preprocess) to be done.
    """
    patch = patch or PATCH
    rows = []
    for split in ("train", "val", "test"):
        ds = SlidingWindowPatchDataset(samples=labeled_sample[split], split=split, **patch)
        cats = np.array(ds.patch_labels())
        n_bif = np.array(ds.patch_n_bif)
        neg_bif = n_bif[(cats == NEGATIVE) | (cats == NEG_BIF)]
        row = {"split": split, "patients": len(labeled_sample[split]), "patches": len(cats)}
        row.update({name: int((cats == c).sum()) for c, name in CATEGORY_NAMES.items()})
        if len(neg_bif):
            row.update({f"neg_bif_p{q}": float(v) for q, v in zip((25, 50, 60, 70, 80, 90), np.percentile(neg_bif, [25, 50, 60, 70, 80, 90]))})
        rows.append(row)
        print(f"[{split}] bifurcations of negative patches, percentiles 25/50/60/70/80/90: {[row.get(f'neg_bif_p{q}') for q in (25, 50, 60, 70, 80, 90)]}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n{df.to_string(index=False)}\n[SAVED] {out_csv}")
    return df
