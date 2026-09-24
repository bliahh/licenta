import monai
import monai.transforms as T
import torch
from monai.data import GridPatchDataset, PatchIter, SlidingPatchWSIDataset
from monai.inferers import SlidingWindowSplitter
from monai.transforms import GridPatch
from torch.utils.data import Dataset
from src.config import SIZE

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
        T.Spacingd(keys=["image", "vessel_mask","aneurysm_mask"],pixdim=(0.5, 0.5, 0.5),mode=("bilinear", "nearest")),

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
        device= "cuda",
    )
    return splitter(data)






class SlidingWindowPatchDataset(Dataset):

    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

        self.splitter = SlidingWindowSplitter(
            patch_size=(96, 96, 96),
            overlap=0.5,
            offset=0,
            pad_mode="constant",
            pad_value=0,
        )

        self.patches = []

        for sample_idx, sample in enumerate(samples):

            image = sample["image"]

            for patch, coords in self.splitter(image):

                self.patches.append({
                    "sample_index": sample_idx,
                    "patch": patch,
                    "coords": coords,
                    "label": self.assign_bin_label(sample,coords),
                    "type_acq": sample["type_acq"],
                    "spec_labels": self.assign_labels(sample,coords),
                })

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):

        item = self.patches[idx]

        patch = item["patch"]

        if self.transform is not None:
            patch = self.transform({"image": patch})["image"]

        return {
            "image": patch,
            "label": item["label"],
            "sample_index": item["sample_index"],
            "coords": item["coords"],
            "type_acq": item["type_acq"],
            "spec_labels": item["spec_labels"],
        }


    def assign_bin_label(self, sample,coords):
        #get the aneurysm mask
        aneurysm_mask = sample["aneurysm_mask"].squeeze(0)
        #extracts the corresponding mask patch
        mask_patch = aneurysm_mask[coords]
        return int(torch.any(mask_patch>0))

    def assign_labels(self, sample, coords):
        aneurysm_mask = sample["aneurysm_mask"].squeeze(0)
        mask_patch = aneurysm_mask[coords]

        labels = torch.unique(mask_patch)
        labels = labels[labels != 0]

        return labels.tolist()

if __name__ == "__main__":
 print(monai.__version__)

