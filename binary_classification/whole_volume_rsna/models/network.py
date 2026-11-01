import torch.nn as nn
from monai.networks.nets import resnet18


def batchnorm_to_groupnorm(module, num_groups=8):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        else:
            batchnorm_to_groupnorm(child, num_groups)
    return module


class ResNet18(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        backbone = resnet18(
            spatial_dims=3,
            n_input_channels=1,
            no_max_pool=True,
            conv1_t_size=3,
            shortcut_type="B",
            act="relu",
            conv1_t_stride=2,
            feed_forward=False,
            norm=("group", {"num_groups": 8}),
        )
        self.backbone = batchnorm_to_groupnorm(backbone, num_groups=8)
        self.backbone.avgpool = nn.AdaptiveMaxPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        logit = self.classifier(features)
        return logit.squeeze(1)

    def get_embeddings(self, x):
        return self.backbone(x)