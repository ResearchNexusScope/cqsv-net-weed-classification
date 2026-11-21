"""Layer-2 model contract tests (QC-D01 §4 / QC-D03 §5.3).

Synthetic data only.  No dataset statistics, no legacy result values and no
hardcoded numbers from the paper appear here; the tests assert shapes, frozen
gradients, import safety, kernel algebra and seed propagation.  Tests that need
PennyLane skip cleanly when it (or lightning.qubit) is unavailable.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments.config import DEFAULT_CONFIG, ExperimentConfig  # noqa: E402
from models import classical as classical_mod  # noqa: E402
from models import quantum as quantum_mod  # noqa: E402
from models import qsvc as qsvc_mod  # noqa: E402

# Small synthetic image size keeps the legacy CNN's dense layer modest; the
# implementations accept any cfg.image_size, including the canonical 128.
CFG = replace(DEFAULT_CONFIG, image_size=64)
BATCH = 3


def _images(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(batch, 3, CFG.image_size, CFG.image_size)


def _unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim))
    return qsvc_mod._l2_normalize(x)


# --------------------------------------------------------------------------- classical

@pytest.mark.parametrize("name", classical_mod.CLASSICAL_MODEL_NAMES)
def test_classical_model_logits_shape(name: str) -> None:
    pytest.importorskip("torchvision", reason="torchvision backbones required")
    model = classical_mod.build_classical_model(name, CFG)
    logits = model(_images())
    assert isinstance(model, torch.nn.Module)
    assert logits.shape == (BATCH, 2)
    assert torch.isfinite(logits).all()


def test_unknown_classical_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown classical model"):
        classical_mod.build_classical_model("not_a_model", CFG)


def test_frozen_efficientnet_features() -> None:
    pytest.importorskip("torchvision", reason="torchvision required for EfficientNet")
    features = classical_mod.build_efficientnet_features(CFG)
    assert all(not p.requires_grad for p in features.features.parameters())
    assert all(p.requires_grad for p in features.projection.parameters())
    out = features(_images())
    assert out.shape == (BATCH, CFG.feature_dim)
    norms = torch.linalg.vector_norm(out, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# --------------------------------------------------------------------------- import guard

def test_quantum_import_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Modules import without PennyLane; only construction raises a clear error."""
    monkeypatch.setattr(quantum_mod, "qml", None)
    with pytest.raises(RuntimeError, match="PennyLane is required"):
        quantum_mod.build_legacy_qnn(CFG)

    monkeypatch.setattr(qsvc_mod, "qml", None)
    with pytest.raises(RuntimeError, match="PennyLane is required"):
        qsvc_mod.build_kernel_circuit(CFG)


def test_classical_models_do_not_need_pennylane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quantum_mod, "qml", None)
    model = classical_mod.build_classical_model("Custom_Legacy_CNN", CFG)
    assert model(_images()).shape == (BATCH, 2)


# --------------------------------------------------------------------------- quantum models

def _require_pl():
    pl = pytest.importorskip("pennylane", reason="PennyLane unavailable")
    return pl


def test_quantum_models_logits_and_introspection() -> None:
    _require_pl()
    builders = (
        quantum_mod.build_legacy_qnn,
        quantum_mod.build_hybrid_from_scratch,
        quantum_mod.build_hybrid_transfer,
    )
    for build in builders:
        model = build(CFG)
        logits = model(_images(batch=2))
        assert logits.shape == (2, 2)
        assert torch.isfinite(logits).all()
        assert quantum_mod.trainable_param_count(model) > 0
        assert quantum_mod.entangling_gate_count(model) > 0
        assert model.n_qubits == 10
        assert not model.ablation


def test_four_qubit_ablation_shape() -> None:
    _require_pl()
    model = quantum_mod.build_4qubit_ablation(CFG)
    assert model.ablation is True
    assert model.label == "q4_embamplitude_L2_ablation"
    assert model.n_qubits == 4
    assert model.feature_dim == 16
    assert model(_images(batch=2)).shape == (2, 2)


# --------------------------------------------------------------------------- kernels

def test_classical_kernel_symmetry_psd_unit_diagonal() -> None:
    A = _unit_vectors(16, CFG.feature_dim, seed=1)  # >=16 random L2-normalised vectors
    K = qsvc_mod.classical_kernel(A, A)
    assert K.shape == (16, 16)
    assert np.allclose(K, K.T, atol=1e-10)
    assert np.allclose(np.diag(K), np.ones(16), atol=1e-10)
    assert np.linalg.eigvalsh(K).min() >= -1e-8


def test_classical_kernel_rectangular_shape() -> None:
    A = _unit_vectors(5, CFG.feature_dim, seed=2)
    Bv = _unit_vectors(7, CFG.feature_dim, seed=3)
    assert qsvc_mod.classical_kernel(A, Bv).shape == (5, 7)


def test_kernel_equivalence_error_and_quantum_properties() -> None:
    _require_pl()
    try:
        qsvc_mod.build_kernel_circuit(CFG)
    except RuntimeError:
        pytest.skip("lightning.qubit device unavailable")

    A = _unit_vectors(4, CFG.feature_dim, seed=4)
    assert qsvc_mod.kernel_equivalence_error(A, A, CFG) < 1e-6

    circuit = qsvc_mod.build_kernel_circuit(CFG)
    assert quantum_mod.entangling_gate_count(circuit) == 0
    assert quantum_mod.trainable_param_count(circuit) == 0

    K = qsvc_mod.quantum_kernel(A, A, CFG)
    assert K.shape == (4, 4)
    assert np.allclose(K, K.T, atol=1e-10)
    assert np.allclose(np.diag(K), np.ones(4), atol=1e-10)
    assert np.linalg.eigvalsh(K).min() >= -1e-8


# --------------------------------------------------------------------------- QSVC wrapper

def test_svc_random_state_propagation_and_determinism() -> None:
    cfg: ExperimentConfig = CFG
    wrapper = qsvc_mod.QSVCWrapper(cfg)
    assert wrapper.random_state == cfg.seed
    assert wrapper.svc.random_state == cfg.seed
    assert wrapper.svc.kernel == "precomputed"
    assert wrapper.svc.probability is True

    X = _unit_vectors(8, 16, seed=5)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    K = qsvc_mod.classical_kernel(X, X)
    wrapper.fit(K, y)
    p1 = wrapper.predict(K)
    prob1 = wrapper.predict_proba(K)
    other = qsvc_mod.QSVCWrapper(cfg).fit(K, y)
    assert np.array_equal(p1, other.predict(K))
    assert np.allclose(prob1, other.predict_proba(K))
