import monai.transforms as T


def base_transform_dict():
    """
    Defines the deterministic preprocessing applied once to every patient before cutting it into patches.

    This pipeline:
    1. Loads the image, the vessel mask and the aneurysm mask.
    2. Standardizes their dimensions and anatomical orientation.
    3. Resamples them to a common voxel spacing.
    4. Binarizes the masks.
    5. Uses the vessel mask to crop the volume around the vessels.
    6. Normalizes the image intensity.
    7. Converts the image to a suitable tensor type.
    The volume keeps its size (no resize); the vessel mask is kept to find the bifurcations.
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
