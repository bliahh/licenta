import os
import re
import sys
import json
import glob
import shutil

import numpy as np
import nibabel as nib

SOURCE, TARGET = sys.argv[1], sys.argv[2]
ANEURYSM_LABEL = 37


def keep_case(case):
    match = re.match(r"topaneu_center(\d)_(ct|mr)_(\d+)", case)
    center, number = match.group(1), match.group(3)
    if center == "1":
        return False
    if center == "5" and number.startswith("4"):
        return False
    return True


if __name__ == "__main__":
    os.makedirs(f"{TARGET}/imagesTr", exist_ok=True)
    os.makedirs(f"{TARGET}/labelsTr", exist_ok=True)
    vessel_labels = json.load(open(f"{SOURCE}/vessel_mapping.json"))["labels"]
    assert sorted(vessel_labels.values()) == list(range(ANEURYSM_LABEL)), "vessel labels are not 0..36"

    cases = sorted(os.path.basename(p)[:-len("_0000.nii.gz")] for p in glob.glob(f"{SOURCE}/images/*_0000.nii.gz"))
    kept = [case for case in cases if keep_case(case)]
    print(f"[ CONVERT ] cases {len(cases)} | kept {len(kept)} | excluded {len(cases) - len(kept)}", flush=True)

    summary = []
    for n, case in enumerate(kept, 1):
        image_path = f"{SOURCE}/images/{case}_0000.nii.gz"
        vessel = nib.load(f"{SOURCE}/vessel_masks/{case}.nii.gz")
        aneurysm = nib.load(f"{SOURCE}/location_masks/{case}.nii.gz")
        image_shape = nib.load(image_path).shape
        label = np.asarray(vessel.dataobj).astype(np.uint8)
        aneurysm_voxels = np.asarray(aneurysm.dataobj) > 0
        assert label.shape == image_shape == aneurysm_voxels.shape, f"shape mismatch in {case}"
        label[aneurysm_voxels] = ANEURYSM_LABEL
        out = nib.Nifti1Image(label, vessel.affine, vessel.header)
        out.set_data_dtype(np.uint8)
        nib.save(out, f"{TARGET}/labelsTr/{case}.nii.gz")
        target_image = f"{TARGET}/imagesTr/{case}_0000.nii.gz"
        if not os.path.exists(target_image):
            try:
                os.link(image_path, target_image)
            except OSError:
                shutil.copy(image_path, target_image)
        summary.append({"case": case, "vessel_voxels": int((label > 0).sum()), "aneurysm_voxels": int(aneurysm_voxels.sum())})
        if n % 20 == 0:
            print(f"[ CONVERT ] {n} / {len(kept)}", flush=True)

    labels = dict(vessel_labels)
    labels["aneurysm"] = ANEURYSM_LABEL
    dataset = {"channel_names": {"0": "angio"}, "labels": labels, "numTraining": len(kept), "file_ending": ".nii.gz",
               "name": "TopAneuVessels", "description": "TopAneu centers 2, 4 and 5 (INSTED only); vessels + aneurysm"}
    json.dump(dataset, open(f"{TARGET}/dataset.json", "w"), indent=2)
    json.dump(summary, open(f"{TARGET}/conversion_summary.json", "w"), indent=1)
    with_aneurysm = sum(s["aneurysm_voxels"] > 0 for s in summary)
    print(f"[ DONE ] {len(kept)} cases | with aneurysm {with_aneurysm} | labels {len(labels)}", flush=True)