"""Image transforms, with a registry ready for Phase 3/4 perturbations.

Pipeline:  [perturbations on the PIL image at native resolution] -> Resize(224x224)
           -> ToTensor -> ImageNet Normalize

Baselines use NO perturbations (not even horizontal flips).

Adding a perturbation later (JPEG, resize, crop, blur, brightness, noise):

    @register("jpeg")
    class Jpeg(Perturbation):
        def __init__(self, quality=75, p=1.0): ...
        def apply(self, img): ...   # PIL.Image -> PIL.Image

Then use it from a config, as train-time augmentation or test-time condition:
    data.train_perturbations: [{name: jpeg, quality: [30, 95], p: 0.5}]
    eval.conditions:          [{name: jpeg, quality: 50}]

Perturbations are classes (not lambdas) so they pickle into DataLoader workers on
Windows/macOS (spawn start method).
"""
from __future__ import annotations

import random
from typing import Callable

from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PERTURBATIONS: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        PERTURBATIONS[name] = cls
        cls.registry_name = name
        return cls
    return deco


class Perturbation:
    """Base class. `p` = probability of applying (1.0 for test-time conditions)."""

    registry_name = "base"

    def __init__(self, p: float = 1.0, **params):
        self.p = p
        self.params = params

    def apply(self, img: Image.Image) -> Image.Image:  # override
        raise NotImplementedError

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p >= 1.0 or random.random() < self.p:
            return self.apply(img)
        return img

    def __repr__(self) -> str:
        return f"{self.registry_name}(p={self.p}, {self.params})"


@register("identity")
class Identity(Perturbation):
    """No-op; useful as the 'clean' test condition in robustness sweeps."""

    def apply(self, img: Image.Image) -> Image.Image:
        return img


def build_perturbations(specs: list[dict] | None) -> list[Perturbation]:
    out = []
    for spec in specs or []:
        spec = dict(spec)
        name = spec.pop("name")
        if name not in PERTURBATIONS:
            raise KeyError(f"Unknown perturbation '{name}'. Registered: {sorted(PERTURBATIONS)}")
        out.append(PERTURBATIONS[name](**spec))
    return out


def condition_name(spec: dict | None) -> str:
    """Filesystem-friendly name for a test condition, e.g. 'jpeg_quality-50'."""
    if not spec:
        return "clean"
    params = "_".join(f"{k}-{v}" for k, v in sorted(spec.items()) if k != "name")
    return f"{spec['name']}_{params}" if params else spec["name"]


def build_transform(image_size: int = 224, perturbations: list[dict] | None = None) -> T.Compose:
    steps: list = build_perturbations(perturbations)
    steps += [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return T.Compose(steps)
