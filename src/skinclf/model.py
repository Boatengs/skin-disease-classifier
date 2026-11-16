"""EfficientNetB0 with the same head and unfrozen blocks as the original notebook."""
from __future__ import annotations

from torch import nn
from torchvision import models

from . import CLASSES


def build_model(pretrained: bool = True, trainable_blocks: int = 3) -> nn.Module:
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, len(CLASSES)),
    )
    if trainable_blocks:
        for p in model.features[-trainable_blocks:].parameters():
            p.requires_grad = True
    return model
