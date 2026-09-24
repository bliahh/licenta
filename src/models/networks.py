import torch.nn as nn
from monai.networks.nets import (resnet18, resnet10, resnet34, resnet50, resnet101,DenseNet121, DenseNet169, DenseNet201, DenseNet264,EfficientNetBN)



def build_model_resnet18(dropout=0.0):
    neural_net = resnet18(spatial_dims=3, n_input_channels=1, num_classes=1)
    if dropout > 0:
        in_feat = neural_net.fc.in_features
        neural_net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, 1))
    return neural_net


def build_model_resnet10(dropout=0.0):
    neural_net = resnet10(spatial_dims=3, n_input_channels=1, num_classes=1)
    if dropout > 0:
        in_feat = neural_net.fc.in_features
        neural_net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, 1))
    return neural_net


def build_model_resnet34(dropout=0.0):
    neural_net = resnet34(spatial_dims=3, n_input_channels=1, num_classes=1)
    if dropout > 0:
        in_feat = neural_net.fc.in_features
        neural_net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, 1))
    return neural_net


def build_model_resnet50(dropout=0.0):
    neural_net = resnet50(spatial_dims=3, n_input_channels=1, num_classes=1)
    if dropout > 0:
        in_feat = neural_net.fc.in_features
        neural_net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, 1))
    return neural_net


def build_model_resnet101(dropout=0.0):
    neural_net = resnet101(spatial_dims=3, n_input_channels=1, num_classes=1)
    if dropout > 0:
        in_feat = neural_net.fc.in_features
        neural_net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, 1))
    return neural_net



def build_model_densenet121(dropout=0.0):
    neural_net = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=dropout)
    return neural_net


def build_model_densenet169(dropout=0.0):
    neural_net = DenseNet169(spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=dropout)
    return neural_net


def build_model_densenet201(dropout=0.0):
    neural_net = DenseNet201(spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=dropout)
    return neural_net


def build_model_densenet264(dropout=0.0):
    neural_net = DenseNet264(spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=dropout)
    return neural_net



def build_model_efficientnet_b0(dropout=0.0):
    neural_net = EfficientNetBN(
        "efficientnet-b0", pretrained=False,
        spatial_dims=3, in_channels=1, num_classes=1,
    )
    if dropout > 0:
        in_feat = neural_net._fc.in_features
        neural_net._fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, 1))
    return neural_net


def build_model_efficientnet_b1(dropout=0.0):
    neural_net = EfficientNetBN(
        "efficientnet-b1", pretrained=False,
        spatial_dims=3, in_channels=1, num_classes=1,
    )
    if dropout > 0:
        in_feat = neural_net._fc.in_features
        neural_net._fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, 1))
    return neural_net


def build_model_efficientnet_b3(dropout=0.0):
    neural_net = EfficientNetBN("efficientnet-b3", pretrained=False,spatial_dims=3, in_channels=1, num_classes=1)
    if dropout > 0:
        in_feat = neural_net._fc.in_features
        neural_net._fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, 1))
    return neural_net


def build_model_efficientnet_b2(dropout=0.0):
    neural_net = EfficientNetBN("efficientnet-b2", pretrained=False,spatial_dims=3, in_channels=1, num_classes=1)
    if dropout > 0:
        in_feat = neural_net._fc.in_features
        neural_net._fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, 1))
    return neural_net

MODELS = {
    'resnet10': build_model_resnet10,
    'resnet18': build_model_resnet18,
    'resnet34': build_model_resnet34,
    'resnet50': build_model_resnet50,
    'resnet101': build_model_resnet101,
    'densenet121': build_model_densenet121,
    'densenet169': build_model_densenet169,
    'densenet201': build_model_densenet201,
    'densenet264': build_model_densenet264,
    'efficientnet-b0': build_model_efficientnet_b0,
    'efficientnet-b1': build_model_efficientnet_b1,
    'efficientnet-b2': build_model_efficientnet_b2,
    'efficientnet-b3': build_model_efficientnet_b3,
}


def build_model(model, dropout=0.0):
    return MODELS[model](dropout)
