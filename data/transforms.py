"""Train/eval transforms implementing the accepted spec's section 4 exactly.

torchvision is used when installed (the pinned environment declares it).  A
functional, dependency-free fallback with the *same class names and public
attributes* is used when it is not, so the transform contract remains
inspectable and executable in a minimal environment.  All parameters are read
from ``experiments.config.ExperimentConfig`` -- no literals.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.config import ExperimentConfig

try:  # optional in a minimal environment; pinned as a project dependency
    from torchvision import transforms as _tv

    Compose = _tv.Compose
    RandomAffine = _tv.RandomAffine
    RandomHorizontalFlip = _tv.RandomHorizontalFlip
    RandomVerticalFlip = _tv.RandomVerticalFlip
    ColorJitter = _tv.ColorJitter
    Resize = _tv.Resize
    ToTensor = _tv.ToTensor
    Normalize = _tv.Normalize
    InterpolationMode = _tv.InterpolationMode
    USING_TORCHVISION = True
except ImportError:  # pragma: no cover - depends on the installed environment
    _tv = None
    USING_TORCHVISION = False

    class InterpolationMode:  # minimal stand-in for torchvision's enum
        BILINEAR = "bilinear"

    class Compose:
        def __init__(self, transforms) -> None:
            self.transforms = list(transforms)

        def __call__(self, image):
            for transform in self.transforms:
                image = transform(image)
            return image

        def __repr__(self) -> str:
            return f"Compose({self.transforms!r})"

    class RandomAffine:
        """Affine rotation + zoom with a constant border fill (PIL -> PIL)."""

        def __init__(self, degrees, scale=(1.0, 1.0), interpolation=None, fill=0) -> None:
            self.degrees = (
                tuple(float(v) for v in degrees)
                if isinstance(degrees, (tuple, list))
                else (-float(degrees), float(degrees))
            )
            self.scale = tuple(float(v) for v in scale)
            self.interpolation = interpolation or InterpolationMode.BILINEAR
            self.fill = fill

        def __call__(self, image):
            from PIL import Image

            angle = float(torch.empty(1).uniform_(self.degrees[0], self.degrees[1]).item())
            zoom = float(torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())
            width, height = image.size
            new_w, new_h = max(1, round(width * zoom)), max(1, round(height * zoom))
            resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
            scaled = image.resize((new_w, new_h), resample=resample)
            canvas = Image.new(image.mode, (width, height), self.fill)
            if new_w >= width and new_h >= height:
                left, top = (new_w - width) // 2, (new_h - height) // 2
                scaled = scaled.crop((left, top, left + width, top + height))
                canvas.paste(scaled, (0, 0))
            else:
                canvas.paste(scaled, ((width - new_w) // 2, (height - new_h) // 2))
            rotate_resample = (
                Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
            )
            return canvas.rotate(angle, resample=rotate_resample, fillcolor=self.fill, expand=False)

    class RandomHorizontalFlip:
        def __init__(self, p: float = 0.5) -> None:
            if not 0.0 <= p <= 1.0:
                raise ValueError("p must be in [0, 1]")
            self.p = float(p)

        def __call__(self, image):
            from PIL import Image

            constant = getattr(Image, "FLIP_LEFT_RIGHT", 0)
            if float(torch.rand(1).item()) < self.p:
                return image.transpose(constant)
            return image

    class RandomVerticalFlip:
        def __init__(self, p: float = 0.5) -> None:
            if not 0.0 <= p <= 1.0:
                raise ValueError("p must be in [0, 1]")
            self.p = float(p)

        def __call__(self, image):
            from PIL import Image

            constant = getattr(Image, "FLIP_TOP_BOTTOM", 1)
            if float(torch.rand(1).item()) < self.p:
                return image.transpose(constant)
            return image

    class ColorJitter:
        def __init__(self, brightness=0, contrast=0, saturation=0, hue=0) -> None:
            self.brightness = brightness
            self.contrast = contrast
            self.saturation = saturation
            self.hue = hue

        def __call__(self, image):
            from PIL import ImageEnhance

            def factor(amount):
                if isinstance(amount, (tuple, list)):
                    low, high = float(amount[0]), float(amount[1])
                else:
                    low, high = max(0.0, 1.0 - float(amount)), 1.0 + float(amount)
                return float(torch.empty(1).uniform_(low, high).item())

            if self.brightness:
                image = ImageEnhance.Brightness(image).enhance(factor(self.brightness))
            if self.contrast:
                image = ImageEnhance.Contrast(image).enhance(factor(self.contrast))
            if self.saturation:
                image = ImageEnhance.Color(image).enhance(factor(self.saturation))
            return image

    class Resize:
        def __init__(self, size) -> None:
            self.size = tuple(int(v) for v in size) if isinstance(size, (tuple, list)) else int(size)

        def __call__(self, image):
            from PIL import Image

            resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
            return image.resize(self.size, resample=resample)

    class ToTensor:
        def __call__(self, image):
            array = np.asarray(image)
            if array.ndim == 2:
                array = array[:, :, None]
            tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
            if tensor.dtype == torch.uint8:
                tensor = tensor.float().div(255.0)
            return tensor

    class Normalize:
        def __init__(self, mean, std) -> None:
            self.mean = tuple(float(v) for v in mean)
            self.std = tuple(float(v) for v in std)

        def __call__(self, tensor):
            result = tensor.clone().float()
            for channel, (mean, std) in enumerate(zip(self.mean, self.std)):
                result[channel] = (result[channel] - mean) / std
            return result


__all__ = [
    "make_transforms",
    "Compose",
    "RandomAffine",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "ColorJitter",
    "Resize",
    "ToTensor",
    "Normalize",
    "InterpolationMode",
    "USING_TORCHVISION",
]


def make_transforms(cfg: ExperimentConfig) -> dict[str, object]:
    """Return ``{'train': ..., 'eval': ...}`` PIL -> tensor callables.

    Train order: RandomAffine -> HFlip -> VFlip -> ColorJitter -> Resize ->
    ToTensor -> Normalize.  Eval is deterministic: Resize -> ToTensor ->
    Normalize only.
    """
    fill = tuple(int(255 * mean) for mean in cfg.norm_mean)
    train = Compose(
        [
            RandomAffine(
                degrees=(-int(cfg.rotation_degrees), int(cfg.rotation_degrees)),
                scale=tuple(float(v) for v in cfg.scale_range),
                interpolation=InterpolationMode.BILINEAR,
                fill=fill,
            ),
            RandomHorizontalFlip(p=float(cfg.hflip_prob)),
            RandomVerticalFlip(p=float(cfg.vflip_prob)),
            ColorJitter(
                brightness=float(cfg.jitter),
                contrast=float(cfg.jitter),
                saturation=float(cfg.jitter),
                hue=0.0,
            ),
            Resize((int(cfg.image_size), int(cfg.image_size))),
            ToTensor(),
            Normalize(mean=tuple(cfg.norm_mean), std=tuple(cfg.norm_std)),
        ]
    )
    evaluation = Compose(
        [
            Resize((int(cfg.image_size), int(cfg.image_size))),
            ToTensor(),
            Normalize(mean=tuple(cfg.norm_mean), std=tuple(cfg.norm_std)),
        ]
    )
    return {"train": train, "eval": evaluation}
