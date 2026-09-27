import os
import monai
import monai.transforms as T
import torch
from monai.data import GridPatchDataset, PatchIter, SlidingPatchWSIDataset
from monai.inferers import SlidingWindowSplitter
from torch.utils.data import Dataset
from src.config import SIZE, PREPROCESSED_DIR

def base_transform_dict():
    """
    Defines the common preprocessing pipeline applied to all samples.

    This pipeline:
    1. Loads the image and vessel mask.
    2. Standardizes their dimensions and anatomical orientation.
    3. Resamples them to a common voxel spacing.
    4. Uses the vessel mask to crop the image around the vessels.
    5. Normalizes the image intensity.
    6. Resizes or pads/crops the image to the target size.
    7. Removes the vessel mask because it is only used for preprocessing.
    8. Converts the image to a suitable tensor type.
    """

    return T.Compose([

        # Load the image and vessel mask from their file paths.
        # The input dictionary is expected to contain:
        # {"image": "...", "vessel_mask": "..."}
        T.LoadImaged(keys=["image", "vessel_mask","aneurysm_mask"]),

        # Ensure that the channel dimension is the first dimension.
        # For a single-channel 3D image, the shape becomes:
        # (1, H, W, D)
        T.EnsureChannelFirstd(keys=["image", "vessel_mask","aneurysm_mask"]),

        # Convert both the image and vessel mask to the same
        # anatomical orientation: Left-Posterior-Superior (LPS).
        # This ensures that all samples have a consistent orientation.
        T.Orientationd(keys=["image", "vessel_mask","aneurysm_mask"],axcodes="LPS"),

        # Resample both the image and vessel mask to a common
        # voxel spacing of pixdim(...)
        #
        # Bilinear interpolation is used for the image because
        # image intensities are continuous values.
        #
        # Nearest-neighbor interpolation is used for the vessel mask
        # because it preserves the discrete mask labels and avoids
        # creating intermediate values.
        T.Spacingd(keys=["image", "vessel_mask","aneurysm_mask"],pixdim=(0.5, 0.5, 0.5),mode=("bilinear", "nearest", "nearest")),

        # Convert the vessel mask into a binary mask.
        # Every voxel with a value greater than 0 becomes 1,
        # while all other voxels become 0.

        T.Lambdad(keys=["vessel_mask"],func=lambda x: (x > 0).astype(x.dtype)),
        T.Lambdad(keys=["aneurysm_mask"],func=lambda x: (x > 0).astype(x.dtype)),

        # Crop the image and vessel mask around the region containing
        # the vessels.
        #
        # The vessel mask is used to determine the foreground region.
        # This removes unnecessary background and focuses the input
        # on the anatomical region containing the vessels.
        T.CropForegroundd(keys=["image", "vessel_mask","aneurysm_mask"],source_key="vessel_mask",select_fn=lambda x: x > 0),

        # Normalize the image intensity using the 0.5th and 99.5th
        # percentiles as the lower and upper intensity limits.
        #
        # The resulting intensity range is mapped to [0, 1].
        # Values outside the percentile range are clipped because
        # clip=True.
        T.ScaleIntensityRangePercentilesd(keys=["image"],lower=0.5,upper=99.5,b_min=0.0,b_max=1.0,clip=True),

        T.Lambdad(
            keys=["image"],
            func=lambda x: print("Shape before final crop:", x.shape) or x
        ),

        # Make all images have the same spatial dimensions.
        #
        # If an image is larger than SIZE, it is cropped.
        # If an image is smaller than SIZE, it is padded.


        #T.ResizeWithPadOrCropd(keys=["image"],spatial_size=SIZE),

        # Remove the vessel mask from the sample.
        #
        # The mask was only needed to identify the vessel region
        # and perform the foreground crop. It is not used as a
        # model input afterward.
        T.DeleteItemsd(keys=["vessel_mask"]),

        # Ensure that the image is converted to a suitable tensor
        # representation for use with PyTorch/MONAI.
        T.EnsureTyped(keys=["image"]),
    ])


def random_transform_dict():
    """
    Defines the random data augmentation pipeline used during training. (AUGMENTATION STAGE)
    """

    return T.Compose([

        # With probability 0.5, randomly flip the image along
        # spatial axis 0.
        T.RandFlipd(keys=["image"],prob=0.5,spatial_axis=0),

        # With probability 0.5, randomly flip the image along
        # spatial axis 1.
        T.RandFlipd(keys=["image"],prob=0.5,spatial_axis=1),

        # With probability 0.5, randomly rotate the image by
        # a multiple of 90 degrees.
        #
        # max_k=3 allows rotations of 90, 180, or 270 degrees.
        T.RandRotate90d(keys=["image"],prob=0.5,max_k=3),

        # With probability 0.3, randomly zoom the image.
        #
        # The zoom factor is randomly selected between 0.9 and 1.1:
        #   0.9 -> slight zoom out
        #   1.0 -> no change
        #   1.1 -> slight zoom in
        #
        # Trilinear interpolation is used because the input is a
        # 3D continuous-valued medical image.
        T.RandZoomd(keys=["image"],prob=0.3,min_zoom=0.9,max_zoom=1.1,mode="trilinear"),

        # Ensure that the augmented image is stored as a suitable
        # tensor type for the model.
        T.EnsureTyped(keys=["image"]),
    ])


def eval_transform_dict():
    """
    Defines the preprocessing pipeline used during validation and testing.

    Validation and test data should not receive random augmentations,
    because their results need to be deterministic and directly
    comparable across different evaluations.

    The same deterministic preprocessing used in base_transform_dict()
    is therefore applied here.
    """

    # Use the common preprocessing pipeline without any
    # random data augmentation.
    return base_transform_dict()



def create_patch_dataset(samples):
    patch_iter = PatchIter(patch_size=(96,96,96), start_pos=(0, 0, 0), mode="empty")
    grid_patch_dataset = GridPatchDataset(
        data=samples,
        transform = eval_transform_dict(),
        patch_iter = patch_iter,

    )
    return grid_patch_dataset

def create_patches(data):
    splitter = SlidingWindowSplitter(
        patch_size=(96,96,96),
        overlap= 0.5,
        offset= 0,
        pad_mode="constant",
        pad_value= 0,
    )
    return splitter(data)
class SlidingWindowPatchDataset(Dataset):
    """
    Dataset of 3D sliding-window patches extracted from preprocessed patients.

    At initialization, it builds an index of all patches (patient, patch position, label)
    using only the aneurysm masks. Images are opened with memory mapping, so only the
    requested patch is read from disk in __getitem__.

    :param samples: list of patient samples of the split
    :param split: "train", "val" or "test"; selects the preprocessed subdirectory
    :param transform: optional augmentation applied to each image patch
    :param patch_size: patch size in voxels
    :param overlap: fraction of overlap between neighbor patches
    :param preprocessed_dir: root directory of the preprocessed data
    """

    def __init__(self, samples, split, transform=None, patch_size=(96, 96, 96), overlap=0.5, preprocessed_dir=PREPROCESSED_DIR):
        self.samples = samples
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        self.overlap = overlap
        self.preprocessed_dir = os.path.join(preprocessed_dir, split)
        self.patch_index = []
        self._patients = {}
        self._prepare_patch_index()

    def patch_labels(self):
        """Returns the label of every patch (1 = contains aneurysm, 0 = not), in dataset order."""
        return [label for _, _, label in self.patch_index]

    def _patient_dir(self, sample_idx):
        """Returns the preprocessed directory of a patient."""
        return os.path.join(self.preprocessed_dir, f"patient_{sample_idx:04d}")

    def _load(self, path, mmap=False):
        """Loads a .pt file; with mmap=True the data is read from disk only when accessed."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing preprocessed file:\n{path}\nRun preprocessing for split '{self.split}' first.")
        return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)

    def _prepare_patch_index(self):
        """Builds the list of all patches as (patient index, patch index, label), using only the aneurysm masks."""
        # print("=" * 70)
        # print(f"LOADING PREPROCESSED DATASET: {self.split.upper()}")
        # print("=" * 70)

        for sample_idx in range(len(self.samples)):
            patient_dir = self._patient_dir(sample_idx)
            metadata = self._load(os.path.join(patient_dir, "metadata.pt"))
            mask = self._load(os.path.join(patient_dir, "aneurysm_mask.pt"), mmap=True)
            spatial_shape = tuple(metadata["spatial_shape"])
            counts, _ = self._grid(spatial_shape)
            n_patches = counts[0] * counts[1] * counts[2]
            n_pos = 0

            for patch_idx in range(n_patches):
                coords = (slice(None),) + self._get_patch_coords(spatial_shape, patch_idx)
                label = int(torch.any(mask[coords] > 0))
                self.patch_index.append((sample_idx, patch_idx, label))
                n_pos += label

            #print(f"[LOAD] patient={sample_idx:04d} / shape={spatial_shape} / patches={n_patches} / pos={n_pos}")

        n_pos_total = sum(self.patch_labels())
        # print("=" * 70)
        # print(f"TOTAL PATIENTS: {len(self.samples)}")
        # print(f"TOTAL PATCHES: {len(self.patch_index)} / POS: {n_pos_total} / NEG: {len(self.patch_index) - n_pos_total}")
        # print("=" * 70)

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

    def __len__(self):
        """Returns the total number of patches."""
        return len(self.patch_index)

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
        """Extracts one patch, applies the optional augmentation and returns it with its labels and metadata."""
        sample_idx, patch_idx, patch_label = self.patch_index[idx]
        image, aneurysm_mask, metadata = self._load_patient(sample_idx)
        spatial_shape = tuple(image.shape[1:])
        coords = (slice(None),) + self._get_patch_coords(spatial_shape, patch_idx)
        image_patch = image[coords].clone().float()
        aneurysm_patch = aneurysm_mask[coords]
        labels = torch.unique(aneurysm_patch)
        labels = labels[labels != 0].tolist()

        if self.transform is not None:
            image_patch = self.transform({"image": image_patch})["image"]

        return {"image": image_patch.float(), "patient_label": int(metadata["patient_label"]), "label": patch_label, "sample_index": sample_idx, "type_acq": metadata["type_acq"], "spec_labels": labels, "index": idx}
if __name__ == "__main__":
 print(monai.__version__)


