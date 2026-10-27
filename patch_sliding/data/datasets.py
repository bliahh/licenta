import os
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from patch_sliding.config import PREPROCESSED_DIR, WEAK_POSITIVE, POSITIVE, NEG_BIF, NEGATIVE


class SlidingWindowPatchDataset(Dataset):
    """
    Dataset of 3D sliding-window patches extracted from preprocessed patients.

    At initialization it builds an index of all patches (patient, patch position, category)
    using only the aneurysm masks and the precomputed bifurcation points. Images are opened
    with memory mapping, so only the requested patch is read from disk in __getitem__.

    Categories: 1 = positive, -1 = ambiguous (ignored), 2 = negative with bifurcations, 0 = negative.
    The sampler uses the categories; the loss receives a binary target (2 -> 0).

    :param samples: list of patient samples of the split
    :param split: "train", "val" or "test"; selects the preprocessed subdirectory
    :param transform: optional augmentation applied to each image patch
    :param patch_size: patch size in voxels
    :param overlap: fraction of overlap between neighbouring patches
    :param preprocessed_dir: root directory of the preprocessed data
    :param min_bif: minimum number of bifurcations for a negative patch to be category 2
    :param th_fraction: minimum fraction of an aneurysm inside a patch for it to be positive
    """

    def __init__(self, samples, split, transform=None, patch_size=(96, 96, 96), overlap=0.25, preprocessed_dir=PREPROCESSED_DIR, min_bif=1, th_fraction=0.5):
        self.samples = samples
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        self.overlap = overlap
        self.min_bif = min_bif
        self.th_fraction = th_fraction
        self.preprocessed_dir = os.path.join(preprocessed_dir, split)
        self.patch_index = []
        self.patch_n_bif = []
        self._patients = {}
        self._prepare_patch_index()

    def patch_labels(self):
        """Returns the category of every patch (1, -1, 2, 0), in dataset order. Used by the sampler."""
        return [category for _, _, category in self.patch_index]

    def __len__(self):
        """Returns the total number of patches."""
        return len(self.patch_index)

    def _patient_dir(self, sample_idx):
        """Returns the preprocessed directory of a patient."""
        return os.path.join(self.preprocessed_dir, f"patient_{sample_idx:04d}")

    def _load(self, path, mmap=False):
        """Loads a .pt file; with mmap=True the data is read from disk only when accessed."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing preprocessed file:\n{path}\nRun preprocessing for split '{self.split}' first.")
        return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)

    def _count_bifurcations(self, bifurcations, spatial_coords):
        """Number of bifurcation points inside the patch given by its three spatial slices."""
        sx, sy, sz = spatial_coords
        inside = (bifurcations[:, 0] >= sx.start) & (bifurcations[:, 0] < sx.stop) & (bifurcations[:, 1] >= sy.start) & (bifurcations[:, 1] < sy.stop) & (bifurcations[:, 2] >= sz.start) & (bifurcations[:, 2] < sz.stop)
        return int(inside.sum())

    def _assign_label(self, components, totals, spatial_coords, n_bif):
        """
        Category of one patch, computed separately for every aneurysm of the patient.

        :param components: 3D volume where every aneurysm has its own id, background = 0
        :param totals: totals[k] = number of voxels of aneurysm k in the whole volume
        :param spatial_coords: the three spatial slices of the patch (without channel)
        :param n_bif: number of bifurcations inside the patch
        """
        aneurysm_ids, voxels_in_patch = np.unique(components[spatial_coords], return_counts=True)
        is_aneurysm = aneurysm_ids != 0
        aneurysm_ids = aneurysm_ids[is_aneurysm]
        voxels_in_patch = voxels_in_patch[is_aneurysm]

        if len(aneurysm_ids) > 0:
            fractions = voxels_in_patch / totals[aneurysm_ids]
            return POSITIVE if (fractions >= self.th_fraction).any() else WEAK_POSITIVE

        return NEG_BIF if n_bif >= self.min_bif else NEGATIVE

    def _grid(self, spatial_shape):
        """Returns the number of patches and the stride along each axis for a volume of the given shape."""
        counts, strides = [], []
        for size, patch in zip(spatial_shape, self.patch_size):
            if size <= patch:
                counts.append(1)
                strides.append(0)
            else:
                stride = int(patch * (1 - self.overlap))
                counts.append((size - patch + stride - 1) // stride + 1)
                strides.append(stride)
        return counts, strides

    def _prepare_patch_index(self):
        """Builds the list of all patches as (patient index, patch index, category) and the bifurcation count of every patch."""
        for sample_idx in range(len(self.samples)):
            patient_dir = self._patient_dir(sample_idx)
            metadata = self._load(os.path.join(patient_dir, "metadata.pt"))
            mask = self._load(os.path.join(patient_dir, "aneurysm_mask.pt"), mmap=True)
            bifurcations = self._load(os.path.join(patient_dir, "bifurcations.pt")).numpy()
            spatial_shape = tuple(metadata["spatial_shape"])

            components, _ = ndimage.label(mask[0].numpy() > 0)
            totals = np.bincount(components.ravel())

            counts, _ = self._grid(spatial_shape)
            n_patches = counts[0] * counts[1] * counts[2]

            for patch_idx in range(n_patches):
                spatial_coords = self._get_patch_coords(spatial_shape, patch_idx)
                n_bif = self._count_bifurcations(bifurcations, spatial_coords)
                category = self._assign_label(components, totals, spatial_coords, n_bif)
                self.patch_index.append((sample_idx, patch_idx, category))
                self.patch_n_bif.append(n_bif)

        cats = np.array(self.patch_labels())
        print(f"[{self.split}] patients={len(self.samples)} | patches={len(cats)} | pos(1)={(cats == POSITIVE).sum()} | ambiguous(-1)={(cats == WEAK_POSITIVE).sum()} | neg_bif(2)={(cats == NEG_BIF).sum()} | neg(0)={(cats == NEGATIVE).sum()}")

    def _start_position(self, index, size, patch, stride):
        """Returns the start of a patch along one axis; the last patch is aligned to the volume border."""
        if size <= patch:
            return 0
        return min(index * stride, size - patch)

    def _get_patch_coords(self, spatial_shape, patch_idx):
        """Converts a flat patch index into the three spatial slices of that patch."""
        counts, strides = self._grid(spatial_shape)
        nx, ny, nz = counts
        iz = patch_idx % nz
        iy = (patch_idx // nz) % ny
        ix = patch_idx // (ny * nz)
        x = self._start_position(ix, spatial_shape[0], self.patch_size[0], strides[0])
        y = self._start_position(iy, spatial_shape[1], self.patch_size[1], strides[1])
        z = self._start_position(iz, spatial_shape[2], self.patch_size[2], strides[2])
        return (slice(x, x + self.patch_size[0]), slice(y, y + self.patch_size[1]), slice(z, z + self.patch_size[2]))

    def _load_patient(self, sample_idx):
        """Opens (once) and caches the memory-mapped image, mask and metadata of a patient."""
        if sample_idx not in self._patients:
            patient_dir = self._patient_dir(sample_idx)
            image = self._load(os.path.join(patient_dir, "image.pt"), mmap=True)
            aneurysm_mask = self._load(os.path.join(patient_dir, "aneurysm_mask.pt"), mmap=True)
            metadata = self._load(os.path.join(patient_dir, "metadata.pt"))
            self._patients[sample_idx] = (image, aneurysm_mask, metadata)
        return self._patients[sample_idx]

    def __getitem__(self, idx):
        """Extracts one patch, applies the optional augmentation and returns it with its target and metadata."""
        sample_idx, patch_idx, category = self.patch_index[idx]
        patch_label = NEGATIVE if category == NEG_BIF else category

        image, aneurysm_mask, metadata = self._load_patient(sample_idx)
        spatial_shape = tuple(image.shape[1:])
        coords = (slice(None),) + self._get_patch_coords(spatial_shape, patch_idx)
        image_patch = image[coords].clone().float()
        spec_labels = torch.unique(aneurysm_mask[coords])
        spec_labels = spec_labels[spec_labels != 0].tolist()

        if self.transform is not None:
            image_patch = self.transform({"image": image_patch})["image"]

        return {"image": image_patch.float(), "patient_label": int(metadata["patient_label"]), "label": patch_label, "category": category, "sample_index": sample_idx, "type_acq": metadata["type_acq"], "spec_labels": spec_labels, "index": idx}
