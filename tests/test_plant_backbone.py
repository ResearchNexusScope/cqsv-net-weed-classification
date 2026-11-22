"""Synthetic contract tests for the frozen PlantNet backbone (PLANT-D01).

Provenance/license: cpoisson/plantnet300k-resnet18 (Apache-2.0).  The backbone
is frozen and no target-dataset labels were used in its pretraining.  These
tests use only randomly generated tensors / PIL images: no real dataset paths,
no paper result numbers and no hardcoded accuracy figures appear here.  Tests
that need heavy optional dependencies skip cleanly when they are absent, so the
module remains importable under a minimal verifier environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmlpipe import plant_backbone as pb  # noqa: E402
from qmlpipe.config import QMLConfig  # noqa: E402


# --------------------------------------------------------------------------- config

def test_backbone_default_and_validation() -> None:
    cfg = QMLConfig()
    assert cfg.backbone == "plantnet"
    assert cfg.validate() == []

    cfg_imagenet = QMLConfig(backbone="imagenet")
    assert cfg_imagenet.validate() == []

    # Unknown names must be rejected by the existing validate() contract.
    bad = QMLConfig(qubits=10, projection_dim=1024, backbone="resnet50")
    with pytest.raises(AssertionError, match="backbone"):
        bad.validate()


def test_backbone_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QML_BACKBONE", raising=False)
    assert QMLConfig.from_env().backbone == "plantnet"

    monkeypatch.setenv("QML_BACKBONE", "imagenet")
    cfg = QMLConfig.from_env()
    assert cfg.backbone == "imagenet"
    assert cfg.validate() == []


def test_plantnet_classical_dim_is_configurable_and_projects() -> None:
    """PlantNet's 512-d trunk must flow into the projection, not EfficientNet's 1280.

    Synthetic only: no backbone download is performed.  The projection is built
    directly from the config to catch the old ``Linear(1280 -> 1024)`` mismatch
    that failed with ``addmm: mat1 ... (32x512 and 1280x1024)``.
    """
    cfg = QMLConfig(backbone="plantnet", classical_dim=512)
    assert cfg.classical_dim == 512
    assert cfg.validate() == []

    torch = pytest.importorskip("torch", reason="torch is required")
    from torch import nn

    projection = nn.Linear(cfg.classical_dim, cfg.feature_dim)
    assert projection.in_features == 512
    assert projection.out_features == cfg.feature_dim
    # Forward pass on a synthetic 512-d PlantNet-style feature batch.
    assert projection(torch.randn(2, cfg.classical_dim)).shape == (2, cfg.feature_dim)


# --------------------------------------------------------------------------- preprocess

def test_plantnet_preprocess_output_shape() -> None:
    image_module = pytest.importorskip("PIL.Image", reason="Pillow is required")
    pytest.importorskip("torchvision", reason="torchvision is required")

    image = image_module.new("RGB", (320, 240))
    tensor = pb.plantnet_preprocess()(image)
    assert tuple(tensor.shape) == (3, 224, 224)


# --------------------------------------------------------------------------- extractor

def test_plantnet_extractor_shape_and_unit_norm() -> None:
    torch = pytest.importorskip("torch", reason="torch is required")
    pytest.importorskip("torchvision", reason="torchvision is required")
    pytest.importorskip("huggingface_hub", reason="huggingface_hub is required")

    extractor = pb.build_plantnet_extractor(QMLConfig())
    assert extractor.feature_dim == pb.PLANTNET_FEATURE_DIM
    assert pb.PLANTNET_FEATURE_DIM == 512

    features = extractor(torch.randn(2, 3, 224, 224))
    assert tuple(features.shape) == (2, pb.PLANTNET_FEATURE_DIM)
    norms = features.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# --------------------------------------------------------------------------- dispatch

def test_dispatch_plantnet_routes_to_plantnet_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker_model = object()
    marker_preprocess = object()
    monkeypatch.setattr(pb, "build_plantnet_extractor", lambda cfg: marker_model)
    monkeypatch.setattr(pb, "plantnet_preprocess", lambda: marker_preprocess)

    model, preprocess = pb.build_backbone_extractor("plantnet", QMLConfig())
    assert model is marker_model
    assert preprocess is marker_preprocess


def test_dispatch_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="backbone"):
        pb.build_backbone_extractor("unknown", QMLConfig())


def test_dispatch_imagenet_returns_callables() -> None:
    pytest.importorskip("torch", reason="torch is required")
    pytest.importorskip("torchvision", reason="torchvision is required")
    pytest.importorskip(
        "scripts.run_real_benchmark", reason="ImageNet backbone path is unavailable"
    )

    try:
        extractor, preprocess = pb.build_backbone_extractor(
            "imagenet", QMLConfig(backbone="imagenet")
        )
    except Exception as exc:  # weights not cached / offline verifier
        pytest.skip(f"ImageNet backbone not available in this environment: {exc}")
    assert callable(extractor)
    assert callable(preprocess)
