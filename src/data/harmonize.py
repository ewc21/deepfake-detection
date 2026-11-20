"""'Harmonize' real images so they share the fakes' processing history.

Why: every DeepFakeFace fake was made by squashing the FULL wiki photo to 512x512, running
the manipulation, and saving with Pillow's default JPEG settings. The real images are the
untouched originals (thousands of sizes; some PNG/GIF/grayscale). Even after both are resized
to 224x224, a CNN can tell them apart from processing traces (resampling scale, JPEG block
grid) instead of the manipulation itself.

With `data.harmonize_real.enabled: true`, each REAL image is first given the same treatment
as the fakes minus the manipulation: bicubic resize of the full frame to 512x512, then an
in-memory JPEG round trip at quality 75 with 4:2:0 chroma subsampling. Fakes are left
untouched (re-encoding them would add a second compression only they have).

The defaults were measured on the dataset (2026-11-19, see CLAUDE.md):
  * 450/450 sampled fakes (150 per type) have the standard IJG quality-75 luminance table
    and 4:2:0 subsampling = Pillow's default JPEG save settings.
  * Re-creating face swaps from their target photo: bicubic resize matches best
    (median abs error outside the face 1.07 vs. 1.89-3.69 for other filters).
The step is deterministic, so it does not count as augmentation.
"""
from __future__ import annotations

import io

from PIL import Image

_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "box": Image.Resampling.BOX,
    "hamming": Image.Resampling.HAMMING,
}


class HarmonizeReal:
    """PIL RGB image -> resized to size x size -> JPEG-encoded -> decoded RGB image.

    A class (not a closure) so it pickles into DataLoader workers on Windows/macOS.
    """

    def __init__(self, size: int = 512, jpeg_quality: int = 75, subsampling: str = "4:2:0",
                 resample: str = "bicubic"):
        if resample not in _RESAMPLE:
            raise ValueError(f"resample must be one of {sorted(_RESAMPLE)}")
        self.size = int(size)
        self.jpeg_quality = int(jpeg_quality)
        self.subsampling = subsampling
        self.resample = resample

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB").resize((self.size, self.size), _RESAMPLE[self.resample])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality, subsampling=self.subsampling)
        buf.seek(0)
        with Image.open(buf) as out:
            return out.convert("RGB")

    def __repr__(self) -> str:
        return (f"HarmonizeReal(size={self.size}, jpeg_quality={self.jpeg_quality}, "
                f"subsampling={self.subsampling!r}, resample={self.resample!r})")


def build_harmonizer(cfg_block: dict | None) -> HarmonizeReal | None:
    """HarmonizeReal from the `data.harmonize_real` config block, or None if disabled."""
    if not cfg_block or not cfg_block.get("enabled", False):
        return None
    params = {k: v for k, v in cfg_block.items() if k != "enabled"}
    return HarmonizeReal(**params)
