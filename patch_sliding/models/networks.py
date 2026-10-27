import torch.nn as nn
from monai.networks.nets import resnet18


def bn_to_gn(module, num_groups=8):
    """Recursively replaces every BatchNorm3d with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        else:
            bn_to_gn(child, num_groups)
    return module


class ResnetArchitecture(nn.Module):
    """
    3D ResNet18 patch classifier with a Dropout + Linear head.

    :param dropout: dropout before the final linear layer
    :param norm: "batch" or ("group", {"num_groups": n}); with group norm every BatchNorm3d is replaced by GroupNorm
    :param conv1_stride: temporal stride of the first convolution
    """

    def __init__(self, dropout=0.0, norm=("group", {"num_groups": 8}), conv1_stride=2):

        super().__init__()

        self.model = resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1,
            pretrained=False,
            norm="batch",
            conv1_t_stride=conv1_stride,
        )
        if norm != "batch":
            self.model = bn_to_gn(self.model, num_groups=norm[1].get("num_groups", 8))

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

        self.embedding = None

        self.model.fc.register_forward_pre_hook(
            self._save_embedding
        )

    def _save_embedding(self, module, inputs):
        self.embedding = inputs[0]

    def forward(self, x, return_embedding=False):
        logits = self.model(x)

        if return_embedding:
            return logits, self.embedding

        return logits
