import torch
import torch.nn as nn
from monai.networks.nets import resnet18
import torchvision.models as models
import torch.nn.functional as F
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

class ResNet18MultiTask(nn.Module):
    def __init__(self, dropout=0.3, top_n=4, conv1_stride=2, pretrained_path=None, n_outputs=1):
        super().__init__()
        backbone = resnet18(spatial_dims=3,
                            n_input_channels=1,
                            no_max_pool=True,
                            conv1_t_size=3,
                            shortcut_type="B",
                            act="relu",
                            conv1_t_stride=conv1_stride,
                            feed_forward=False,
                            norm=("group", {"num_groups": 8}))
        self.backbone = batchnorm_to_groupnorm(backbone, num_groups=8)
        self.dropout = nn.Dropout3d(dropout)
        self.heat_head = nn.Conv3d(512, n_outputs, kernel_size=1)
        self.top_n = top_n
        self.n_outputs = n_outputs

    def forward(self, x):
        b = self.backbone
        x = b.act(b.bn1(b.conv1(x)))
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)

        x = self.dropout(x)

        heat = self.heat_head(x)
        scores = heat.flatten(2)
        logit = scores.topk(self.top_n, dim=2).values.mean(dim=2)
        if self.n_outputs == 1:
            logit = logit.squeeze(1)
        return logit, heat

class ResNet18_2D(nn.Module):
    def __init__(self, num_classes=1, in_channels=1, stride=2, kernel_size=7, in_features=512, dropout=0.3,
                 use_groupnorm=False, groups=32, multitask=False, top_n=4, separe_head=False, fpn=False):
        super().__init__()
        norm = (lambda channels: nn.GroupNorm(groups, channels)) if use_groupnorm else None
        encoder = models.resnet18(weights=None, norm_layer=norm)
        encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        self.multitask = multitask
        self.separe_head = separe_head
        self.fpn = fpn
        if multitask:
            self.dropout = nn.Dropout2d(dropout)
            self.top_n = top_n
            if fpn:
                self.lat3 = nn.Conv2d(256, 256, kernel_size=1)
                self.lat4 = nn.Conv2d(in_features, 256, kernel_size=1)
                self.smooth = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.heat_head = nn.Conv2d(256 if fpn else in_features, num_classes, kernel_size=1)

    def features(self, x, return_layer3=False):
        e = self.encoder
        x = e.maxpool(e.relu(e.bn1(e.conv1(x))))
        f3 = e.layer3(e.layer2(e.layer1(x)))
        f4 = e.layer4(f3)
        return (f3, f4) if return_layer3 else f4

    def forward(self, x, return_embedding=False):
        f3, fmap = self.features(x, return_layer3=True)
        embedding = torch.flatten(self.encoder.avgpool(fmap), 1)
        if self.multitask:
            if self.fpn:
                p = self.smooth(self.lat3(f3) + F.interpolate(self.lat4(fmap), scale_factor=2, mode="nearest"))
                heat = self.heat_head(self.dropout(p))
            else:
                heat = self.heat_head(self.dropout(fmap))
            if self.separe_head:
                logit = self.head(embedding)
            else:
                logit = heat.flatten(2).topk(self.top_n, dim=2).values.mean(dim=2)
            out = (logit, heat)
        else:
            out = self.head(embedding)
        if return_embedding:
            return (*out, embedding) if self.multitask else (out, embedding)
        return out



class HeadsMixin:
    def build_heads(self, features, num_classes, dropout, multitask, top_n, separe_head):
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_classes))
        self.multitask = multitask
        self.separe_head = separe_head
        if multitask:
            self.dropout = nn.Dropout2d(dropout)
            self.heat_head = nn.Conv2d(features, num_classes, kernel_size=1)
            self.top_n = top_n

    def apply_heads(self, embedding, fmap, return_embedding):
        if self.multitask:
            heat = self.heat_head(self.dropout(fmap))
            if self.separe_head:
                logit = self.head(embedding)
            else:
                logit = heat.flatten(2).topk(self.top_n, dim=2).values.mean(dim=2)
            out = (logit, heat)
        else:
            out = self.head(embedding)
        if return_embedding:
            return (*out, embedding) if self.multitask else (out, embedding)
        return out



class ConvNext_2D(nn.Module, HeadsMixin):
    def __init__(self, num_classes=1, in_channels=1, dropout=0.3, multitask=False, top_n=4, separe_head=False):
        super().__init__()

        network = models.convnext_tiny(weight=None)

        network.features[0][0] = nn.Conv2d(in_channels,96,kernel_size=4,stride=4)

        self.features = network.features

        self.norm = nn.LayerNorm(768)

        self.build_heads(768, num_classes, dropout, multitask, top_n, separe_head)

    def forward(self, x, return_embedding=False):

        fmap = self.features(x)
        embedding = self.norm(fmap.mean(dim=(2, 3)))

        return self.apply_heads(embedding, fmap, return_embedding)