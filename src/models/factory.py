"""Model factory: torchvision backbones with a single-logit binary head.

Change the architecture with `--model resnet18` or `--set model.name=resnet18`.
To add a model, add one line to _REGISTRY: (constructor, weights enum, head attribute).
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models as tvm

# name: (constructor, default ImageNet weights, attribute holding the classifier head)
_REGISTRY = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.DEFAULT, "fc"),
    "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.DEFAULT, "fc"),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.DEFAULT, "fc"),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.DEFAULT, "classifier"),
    "efficientnet_b1": (tvm.efficientnet_b1, tvm.EfficientNet_B1_Weights.DEFAULT, "classifier"),
    "efficientnet_b2": (tvm.efficientnet_b2, tvm.EfficientNet_B2_Weights.DEFAULT, "classifier"),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.DEFAULT, "classifier"),
    "convnext_tiny": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.DEFAULT, "classifier"),
}
SUPPORTED_MODELS = tuple(_REGISTRY)


def _replace_last_linear(module: nn.Module) -> nn.Linear:
    """Swap the last nn.Linear inside `module` (or `module` itself) for a 1-output layer."""
    if isinstance(module, nn.Linear):
        raise TypeError("call on the parent so the layer can be replaced")
    last_name, last = None, None
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            last_name, last = name, child
    if last is None:
        raise ValueError("No nn.Linear found in classifier head")
    new = nn.Linear(last.in_features, 1)
    setattr(module, last_name, new)
    return new


def build_model(name: str, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """Backbone + 1-logit head (use with BCEWithLogitsLoss; sigmoid gives P(fake))."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Supported: {SUPPORTED_MODELS}")
    ctor, weights, head_attr = _REGISTRY[name]
    model = ctor(weights=weights if pretrained else None)
    head = getattr(model, head_attr)
    if isinstance(head, nn.Linear):  # e.g. ResNet .fc
        setattr(model, head_attr, nn.Linear(head.in_features, 1))
    else:  # Sequential heads (EfficientNet, MobileNet, ConvNeXt)
        _replace_last_linear(head)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in getattr(model, head_attr).parameters():
            p.requires_grad = True
    return model
