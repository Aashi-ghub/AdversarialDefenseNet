from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int = 10, use_pretrained: bool = False):
    """
    Build a CIFAR-10 friendly ResNet-18.

    `use_pretrained=False` avoids internet downloads in Colab by default.
    """

    weights = ResNet18_Weights.DEFAULT if use_pretrained else None
    model = resnet18(weights=weights)
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
