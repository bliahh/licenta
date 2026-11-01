import monai.transforms as T


ORIENTATION = "LPS"
SPACING = (0.6, 0.6, 0.6)
CT_WINDOW = (0, 900)


def base_transforms(modality):
    if modality == "CTA":
        intensity = T.ScaleIntensityRanged(keys="image", a_min=CT_WINDOW[0], a_max=CT_WINDOW[1], b_min=-1.0, b_max=1.0, clip=True)
    else:
        intensity = T.NormalizeIntensityd(keys="image", nonzero=True)

    return T.Compose([
        T.LoadImaged(keys="image"),
        T.EnsureChannelFirstd(keys="image"),
        T.Orientationd(keys="image", axcodes=ORIENTATION),
        T.CropForegroundd(keys="image", source_key="image"),
        T.Spacingd(keys="image", pixdim=SPACING, mode="bilinear"),
        intensity,
        T.EnsureTyped(keys="image"),
    ])



def train_transforms():
    return T.Compose([
        T.RandFlip(prob=0.5, spatial_axis=0),
        T.RandAffine(prob=0.3, rotate_range=(0.17, 0.17, 0.17), scale_range=(0.1, 0.1, 0.1), padding_mode="border"),
        T.RandScaleIntensity(factors=0.1, prob=0.3),
        T.RandShiftIntensity(offsets=0.1, prob=0.3),
        T.EnsureType(data_type="tensor", track_meta=False),
    ])


def eval_transforms():
    return T.EnsureType(data_type="tensor", track_meta=False)