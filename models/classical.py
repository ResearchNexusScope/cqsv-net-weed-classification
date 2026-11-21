"""Classical image models for the weed-classification rebuild.

Design contract (QC-D01 §4 / QC-D03 §1, rewritten by the mother worker for the
``models/`` package layout):

* ``build_classical_model(name, cfg)`` returns an ``nn.Module`` whose forward pass
  emits **raw logits of shape (batch, 2)** for the four logit-emitting classical
  pipelines named in the paper: ``Custom_Legacy_CNN``, ``ResNet18``,
  ``MobileNetV3`` (torchvision ``mobilenet_v3_small``) and ``EfficientNet-B0``.
* Torchvision backbones are always constructed with ``weights=None``.  **No
  pretrained checkpoint is ever downloaded** by this module.
* ``build_efficientnet_features(cfg)`` returns a single ``nn.Module`` bundling
  the frozen EfficientNet-B0 trunk, its average pool, and a trainable
  ``1280 -> cfg.feature_dim`` projection followed by explicit L2 normalisation.
  This is the feature source shared with the canonical transfer QNN.

Only :class:`experiments.config.ExperimentConfig` is used for configuration.
No PennyLane import occurs here, so this module imports cleanly in environments
without the quantum stack.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.config import ExperimentConfig

__all__ = [
    "CustomLegacyCNN",
    "EfficientNetFeatures",
    "build_classical_model",
    "build_efficientnet_features",
    "CLASSICAL_MODEL_NAMES",
]

# Canonical names (as they appear in the paper / experiment registry).  The
# aliases are accepted defensively so callers using snake_case do not break,
# but the four canonical names are the supported contract.
CLASSICAL_MODEL_NAMES = (
    "Custom_Legacy_CNN",
    "ResNet18",
    "MobileNetV3",
    "EfficientNet-B0",
)

_NAME_ALIASES = {
    "custom_legacy_cnn": "Custom_Legacy_CNN",
    "legacy_cnn": "Custom_Legacy_CNN",
    "resnet18": "ResNet18",
    "mobilenetv3": "MobileNetV3",
    "mobilenet_v3": "MobileNetV3",
    "mobilenet": "MobileNetV3",
    "efficientnet-b0": "EfficientNet-B0",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet": "EfficientNet-B0",
}

_EPS = 1e-8


class CustomLegacyCNN(nn.Module):
    """Faithful, weights-free reproduction of the paper's legacy CNN.

    Two conv+pool stages over an ``image_size`` square image, a single hidden
    layer and a 2-way linear head.  For the canonical 128x128 input this is the
    ``64*32*32 -> 128 -> 2`` network from the archived benchmark.
    """

    def __init__(self, image_size: int = 128) -> None:
        super().__init__()
        spatial = max(1, int(image_size) // 4)  # two 2x2 max-pools
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * spatial * spatial, 128)
        self.fc2 = nn.Linear(128, 2)
        self.image_size = int(image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # raw logits


class EfficientNetFeatures(nn.Module):
    """Frozen EfficientNet-B0 trunk + avgpool + trainable L2-normalised projection.

    Attributes
    ----------
    features, avgpool :
        The (always frozen) torchvision EfficientNet-B0 feature stages.  Their
        parameters have ``requires_grad is False`` and the trunk runs under
        ``no_grad`` for the whole lifetime of the module.
    projection :
        ``nn.Linear(1280, cfg.feature_dim)`` followed by explicit L2
        normalisation of each row.  This is the only trainable part.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        # Imported lazily so a torchvision problem cannot break the legacy CNN
        # or the classical module import, and so no weights are fetched.
        from torchvision import models as tv_models

        backbone = tv_models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.projection = nn.Linear(1280, int(cfg.feature_dim))
        self.feature_dim = int(cfg.feature_dim)

        # Freeze the trunk *permanently* and keep it in eval mode.
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.features(x)
            f = self.avgpool(f)
        f = torch.flatten(f, 1)  # (B, 1280)
        f = self.projection(f)
        return F.normalize(f, p=2, dim=1, eps=_EPS)  # (B, cfg.feature_dim)


def build_efficientnet_features(cfg: ExperimentConfig) -> EfficientNetFeatures:
    """Return the frozen EfficientNet-B0 feature source used by hybrid models.

    The returned module maps ``(B, 3, H, W)`` to ``(B, cfg.feature_dim)`` with
    unit-L2 rows.  ``cfg.feature_dim`` must equal ``2 ** cfg.n_qubits`` (the
    config validator enforces this) so the output is amplitude-embeddable.
    """
    return EfficientNetFeatures(cfg)


def _replace_head(model: nn.Module, attr: str, index: int | None, cfg: ExperimentConfig) -> None:
    """Replace a torchvision classification head with a 2-way linear layer."""
    # Kept local to this module; weights=None is guaranteed by callers.
    if index is None:
        head = getattr(model, attr)
        setattr(model, attr, nn.Linear(head.in_features, 2))
    else:
        head = getattr(model, attr)[index]
        getattr(model, attr)[index] = nn.Linear(head.in_features, 2)


def build_classical_model(name: str, cfg: ExperimentConfig) -> nn.Module:
    """Build one of the four logit-emitting classical models by name.

    Parameters
    ----------
    name:
        One of :data:`CLASSICAL_MODEL_NAMES` (case-insensitive aliases allowed).
    cfg:
        The project ``ExperimentConfig``.  ``cfg.image_size`` sizes the legacy
        CNN's first dense layer; all torchvision heads are replaced with a
        2-class linear layer.

    Returns
    -------
    nn.Module
        ``forward(x)`` maps ``(B, 3, H, W)`` to raw logits ``(B, 2)``.

    Raises
    ------
    ValueError
        If ``name`` is not a supported classical model.
    """
    key = str(name).strip()
    key = _NAME_ALIASES.get(key.lower(), key)
    if key not in CLASSICAL_MODEL_NAMES:
        raise ValueError(
            f"Unknown classical model {name!r}. Expected one of "
            f"{', '.join(CLASSICAL_MODEL_NAMES)}."
        )

    if key == "Custom_Legacy_CNN":
        return CustomLegacyCNN(image_size=int(cfg.image_size))

    from torchvision import models as tv_models

    if key == "ResNet18":
        model: Any = tv_models.resnet18(weights=None)
        _replace_head(model, "fc", None, cfg)
        return model

    if key == "MobileNetV3":
        model = tv_models.mobilenet_v3_small(weights=None)
        _replace_head(model, "classifier", 3, cfg)
        return model

    # EfficientNet-B0
    model = tv_models.efficientnet_b0(weights=None)
    _replace_head(model, "classifier", 1, cfg)
    return model
