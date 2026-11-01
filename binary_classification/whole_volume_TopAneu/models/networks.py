import torch
import torch.nn as nn
from monai.networks.nets import (resnet18, resnet10, resnet34, resnet50, resnet101,DenseNet121, DenseNet169, DenseNet201, DenseNet264,EfficientNetBN)


def _with_dropout_head(neural_net, head, dropout):
    """Replaces the final linear layer `head` with Dropout + Linear when dropout > 0 (same layout as at training)."""
    if dropout > 0:
        in_feat = getattr(neural_net, head).in_features
        setattr(neural_net, head, nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, 1)))
    return neural_net


def _resnet(constructor):
    return lambda dropout: _with_dropout_head(constructor(spatial_dims=3, n_input_channels=1, num_classes=1), "fc", dropout)


def _densenet(constructor):
    return lambda dropout: constructor(spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=dropout)


def _efficientnet(variant):
    return lambda dropout: _with_dropout_head(EfficientNetBN(variant, pretrained=False, spatial_dims=3, in_channels=1, num_classes=1), "_fc", dropout)


MODELS = {
    'resnet10': _resnet(resnet10),
    'resnet18': _resnet(resnet18),
    'resnet34': _resnet(resnet34),
    'resnet50': _resnet(resnet50),
    'resnet101': _resnet(resnet101),
    'densenet121': _densenet(DenseNet121),
    'densenet169': _densenet(DenseNet169),
    'densenet201': _densenet(DenseNet201),
    'densenet264': _densenet(DenseNet264),
    'efficientnet-b0': _efficientnet("efficientnet-b0"),
    'efficientnet-b1': _efficientnet("efficientnet-b1"),
    'efficientnet-b2': _efficientnet("efficientnet-b2"),
    'efficientnet-b3': _efficientnet("efficientnet-b3"),
}


def build_model(model, dropout=0.0):
    return MODELS[model](dropout)


def load_model(name, params, save_path, device):
    """Builds the model exactly as at training time (build_model + best_params) and loads the saved weights."""
    model = build_model(name, dropout=params.get("dropout", 0.0))
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model.to(device).eval()
