"""Frozen PlantNet ResNet18 feature backbone for the QML pipeline (PLANT-D01).

Provenance/license: the pretrained weights are published on Hugging Face at
``cpoisson/plantnet300k-resnet18`` under the Apache-2.0 license (weights file
``plantnet_resnet18.pth``).  The backbone is always frozen and no target-dataset
labels were used in its pretraining; it is a fixed, label-agnostic feature
source, never a supervised model fitted to the benchmark's target labels.
Inputs are ImageNet-normalised 224x224 RGB tensors.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

# Provenance: cpoisson/plantnet300k-resnet18 (Hugging Face, Apache-2.0).
PLANTNET_REPO_ID = "cpoisson/plantnet300k-resnet18"
PLANTNET_WEIGHTS_FILE = "plantnet_resnet18.pth"
PLANTNET_FEATURE_DIM = 512
PLANTNET_CLASSES = 1081

# ImageNet normalisation, matching the upstream PlantNet ResNet18 recipe.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_EPS = 1e-8

_MISSING_HF = (
    "huggingface_hub is required to load the PlantNet ResNet18 weights "
    "(cpoisson/plantnet300k-resnet18, Apache-2.0). Install huggingface_hub "
    "or select backbone='imagenet'."
)
_MISSING_TORCHVISION = (
    "torchvision is required to build the PlantNet ResNet18 feature extractor "
    "(cpoisson/plantnet300k-resnet18, Apache-2.0)."
)
_MISSING_IMAGENET = (
    "The ImageNet backbone path scripts.run_real_benchmark.build_backbone is "
    "unavailable; install torch/torchvision and make the repository root "
    "importable."
)

__all__ = [
    "PLANTNET_REPO_ID",
    "PLANTNET_WEIGHTS_FILE",
    "PLANTNET_FEATURE_DIM",
    "PLANTNET_CLASSES",
    "build_plantnet_extractor",
    "plantnet_preprocess",
    "build_backbone_extractor",
]


def _unwrap_state_dict(obj: Any) -> dict:
    """Normalise a loaded checkpoint into a plain parameter-name -> tensor dict.

    Provenance/license: checkpoints originate from
    cpoisson/plantnet300k-resnet18 (Apache-2.0).  The backbone is frozen and no
    target-dataset labels were used in its pretraining.
    """
    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Unexpected PlantNet checkpoint type {type(obj).__name__}; expected "
            "a state-dict mapping (cpoisson/plantnet300k-resnet18, Apache-2.0)."
        )
    for key in ("state_dict", "model_state_dict", "model", "weights", "net", "network"):
        inner = obj.get(key)
        if isinstance(inner, dict):
            obj = inner
            break
    if obj and all(str(k).startswith("module.") for k in obj):
        obj = {str(k)[len("module."):]: v for k, v in obj.items()}
    return obj


class _FrozenPlantNet(nn.Module):
    """L2-normalising wrapper over the frozen PlantNet ResNet18 trunk.

    Provenance/license: trunk weights come from
    cpoisson/plantnet300k-resnet18 (Apache-2.0).  The backbone is frozen
    (``requires_grad=False`` + eval mode) and no target-dataset labels were
    used in its pretraining; only L2 normalisation is applied at inference.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(PLANTNET_FEATURE_DIM)
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.eval()  # recursively places BatchNorm in inference mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unit-L2 ``(B, 512)`` features for a ``(B, 3, 224, 224)`` batch.

        Provenance/license: cpoisson/plantnet300k-resnet18 (Apache-2.0); the
        backbone is frozen and no target-dataset labels were used in pretraining.
        """
        with torch.no_grad():
            feats = self.backbone(x)
            if feats.ndim != 2 or feats.shape[-1] != self.feature_dim:
                raise ValueError(
                    f"PlantNet trunk returned {tuple(feats.shape)}; expected "
                    f"(B, {self.feature_dim})."
                )
            feats = F.normalize(feats, p=2, dim=1, eps=_EPS)
        return feats


def build_plantnet_extractor(cfg: Any) -> nn.Module:
    """Build the frozen PlantNet ResNet18 ``(B, 3, 224, 224) -> (B, 512)`` extractor.

    Provenance/license: weights are fetched (cached on subsequent calls) from
    cpoisson/plantnet300k-resnet18 (Apache-2.0) with ``weights_only=True``.
    The backbone is frozen and no target-dataset labels were used in pretraining.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(_MISSING_HF) from exc
    try:
        from torchvision import models as tv_models
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(_MISSING_TORCHVISION) from exc

    weights_path = hf_hub_download(
        repo_id=PLANTNET_REPO_ID,
        filename=PLANTNET_WEIGHTS_FILE,
    )
    raw_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = _unwrap_state_dict(raw_state)

    backbone = tv_models.resnet18(weights=None, num_classes=PLANTNET_CLASSES)
    try:
        backbone.load_state_dict(state)
    except RuntimeError:  # pragma: no cover - upstream checkpoint variants
        model_keys = set(backbone.state_dict())
        cleaned = {k: v for k, v in state.items() if k in model_keys}
        if len(cleaned) < 0.9 * len(model_keys):
            raise
        backbone.load_state_dict(cleaned, strict=False)

    # Drop the 1081-way PlantNet classifier: only the 512-d trunk is reused.
    backbone.fc = nn.Identity()
    return _FrozenPlantNet(backbone)


def plantnet_preprocess():
    """Return the ImageNet-normalised 224x224 preprocessing pipeline.

    Provenance/license: matched to cpoisson/plantnet300k-resnet18 (Apache-2.0)
    Resize(256)/CenterCrop(224)/ToTensor/Normalize(ImageNet mean, std).  The
    corresponding backbone is frozen and no target-dataset labels were used in
    its pretraining.
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def build_backbone_extractor(name: str, cfg: Any):
    """Dispatch ``name`` to an ``(extractor, preprocess)`` pair.

    ``"plantnet"`` uses cpoisson/plantnet300k-resnet18 (Apache-2.0); ``"imagenet"``
    routes to the existing torchvision ImageNet backbone in
    ``scripts.run_real_benchmark.build_backbone``.  In both cases the backbone
    is frozen and no target-dataset labels were used in pretraining.
    """
    key = str(name).lower()
    if key == "plantnet":
        return build_plantnet_extractor(cfg), plantnet_preprocess()
    if key == "imagenet":
        try:
            from scripts.run_real_benchmark import build_backbone
        except Exception as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(_MISSING_IMAGENET) from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor, preprocess = build_backbone(device)
        return extractor, preprocess
    raise ValueError(
        f"Unknown backbone {name!r}; expected 'plantnet' or 'imagenet'."
    )
