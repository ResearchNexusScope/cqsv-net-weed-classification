"""Core QML pipeline contract tests (QML-D01).

Synthetic data only -- no dataset statistics, no legacy result values and no
hardcoded numbers from the paper appear here.  Tests assert config invariants,
classical kernel algebra/PSD/diagnostics, import safety, and (when the pinned
PennyLane stack is installed) QNN logit shapes and circuit introspection counts.
Quantum tests skip cleanly via ``pytest.importorskip`` when PennyLane is absent,
so the module is importable under the mother's Python 3.14 verifier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from qmlpipe import kernel as kernel_mod  # noqa: E402
from qmlpipe import models as models_mod  # noqa: E402
from qmlpipe.config import QMLConfig  # noqa: E402


# --------------------------------------------------------------------------- helpers

def _unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """``n`` random L2-normalised ``dim``-vectors from a private RNG."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _small_cfg(**overrides) -> QMLConfig:
    """A cheap 4-qubit / 16-feature config for circuit-level tests."""
    base = dict(qubits=4, projection_dim=2 ** 4, layers=2)
    base.update(overrides)
    return QMLConfig(**base)


def _require_pl():
    return pytest.importorskip("pennylane", reason="PennyLane unavailable")


# --------------------------------------------------------------------------- config

def test_default_config_is_self_consistent() -> None:
    cfg = QMLConfig()
    assert cfg.qubits == 10
    assert cfg.feature_dim == 2 ** cfg.qubits == 1024
    assert cfg.projection_dim == cfg.feature_dim
    assert cfg.validate() == []


@pytest.mark.parametrize("qubits", [4, 8, 10])
def test_qubit_ablation_dims(qubits: int) -> None:
    cfg = QMLConfig(qubits=qubits, projection_dim=2 ** qubits)
    assert cfg.feature_dim == 2 ** qubits
    assert cfg.qnn_input_dim in (qubits, 2 ** qubits)
    assert cfg.validate() == []


def test_validate_asserts_projection_dim_invariant() -> None:
    cfg = QMLConfig(qubits=10, projection_dim=512)
    with pytest.raises(AssertionError, match="projection_dim"):
        cfg.validate()


def test_config_dict_roundtrip() -> None:
    cfg = QMLConfig()
    d = cfg.as_dict()
    assert isinstance(d, dict)
    assert cfg.to_dict() == d
    assert d["qubits"] == cfg.qubits
    assert d["projection_dim"] == cfg.projection_dim
    # properties are not serialised as fields
    assert "feature_dim" not in d


def test_config_from_env_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = [
        "QML_SEED", "QML_QUBITS", "QML_LAYERS", "QML_EMBEDDING", "QML_ANSATZ",
        "QML_DEVICE", "QML_SHOTS", "QML_BATCH_SIZE", "QML_EPOCHS", "QML_LR",
        "QML_WEIGHT_DECAY", "QML_PATIENCE", "QML_OPTIMIZER", "QML_CLASSICAL_DIM",
        "QML_PROJECTION_DIM", "QML_NOISE_DEPOLARIZING", "QML_OUTPUT_DIR",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    default = QMLConfig.from_env()
    assert default.validate() == []
    assert default.seed == 42
    assert default.qubits == 10
    assert default.projection_dim == 1024

    monkeypatch.setenv("QML_QUBITS", "8")
    monkeypatch.setenv("QML_PROJECTION_DIM", "256")
    monkeypatch.setenv("QML_EMBEDDING", "ANGLE")
    monkeypatch.setenv("QML_NOISE_DEPOLARIZING", "0.01")
    monkeypatch.setenv("QML_SHOTS", "1024")
    overridden = QMLConfig.from_env()
    assert overridden.qubits == 8
    assert overridden.projection_dim == 256
    assert overridden.embedding == "angle"
    assert overridden.noise_depolarizing == 0.01
    assert overridden.shots == 1024
    assert overridden.validate() == []


# --------------------------------------------------------------------------- classical kernel

def test_classical_kernel_symmetry_psd_and_unit_diagonal() -> None:
    A = _unit_vectors(16, 32, seed=1)
    K = kernel_mod.classical_kernel_matrix(A, A)
    assert K.shape == (16, 16)
    assert np.allclose(K, K.T, atol=1e-10)
    assert np.allclose(np.diag(K), np.ones(16), atol=1e-10)
    assert np.linalg.eigvalsh(K).min() >= -1e-8


def test_classical_kernel_rectangular_shape_and_formula() -> None:
    A = _unit_vectors(5, 32, seed=2)
    B = _unit_vectors(7, 32, seed=3)
    K = kernel_mod.classical_kernel_matrix(A, B)
    assert K.shape == (5, 7)
    assert np.allclose(K, (A @ B.T) ** 2, atol=1e-12)


def test_kernel_diagnostics_on_16_random_vectors() -> None:
    A = _unit_vectors(16, 64, seed=4)
    K = kernel_mod.classical_kernel_matrix(A, A)
    diag = kernel_mod.kernel_diagnostics(K)
    assert set(diag) == {
        "symmetry_error", "min_diagonal", "min_eigenvalue", "psd"
    }
    assert diag["symmetry_error"] < 1e-10
    assert diag["min_diagonal"] > 1.0 - 1e-10
    assert diag["min_eigenvalue"] >= -1e-8
    assert diag["psd"] is True


# --------------------------------------------------------------------------- import guards

def test_quantum_import_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models_mod, "qml", None)
    with pytest.raises(RuntimeError, match="PennyLane is required"):
        models_mod.build_qnn(_small_cfg())

    monkeypatch.setattr(kernel_mod, "qml", None)
    A = _unit_vectors(2, 16, seed=5)
    with pytest.raises(RuntimeError, match="PennyLane is required"):
        kernel_mod.quantum_kernel_matrix(A, A, _small_cfg())


def test_classical_paths_work_without_pennylane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models_mod, "qml", None)
    monkeypatch.setattr(kernel_mod, "qml", None)
    A = _unit_vectors(4, 16, seed=6)
    K = kernel_mod.classical_kernel_matrix(A, A)
    assert kernel_mod.kernel_diagnostics(K)["psd"] is True


# --------------------------------------------------------------------------- QNN (PennyLane only)

def test_qnn_logits_shape_and_introspection() -> None:
    _require_pl()
    cfg = _small_cfg(ansatz="strongly_entangling")
    model = models_mod.build_qnn(cfg)
    logits = model(torch.randn(3, cfg.feature_dim))
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()

    info = models_mod.circuit_introspection(model)
    assert info["entangling_gate_count"] > 0
    # SEL weights (layers*qubits*3) + Linear head (2*qubits + 2 biases).
    expected = cfg.layers * cfg.qubits * 3 + 2 * cfg.qubits + 2
    assert info["trainable_param_count"] == expected


def test_qnn_angle_embedding_and_basic_entangler() -> None:
    _require_pl()
    cfg = _small_cfg(embedding="angle", ansatz="basic_entangler")
    model = models_mod.build_qnn(cfg)
    assert cfg.qnn_input_dim == cfg.qubits
    logits = model(torch.randn(2, cfg.qnn_input_dim))
    assert logits.shape == (2, 2)

    info = models_mod.circuit_introspection(model)
    assert info["entangling_gate_count"] > 0
    expected = cfg.layers * cfg.qubits + 2 * cfg.qubits + 2
    assert info["trainable_param_count"] == expected


@pytest.mark.parametrize("ansatz", ["strongly_entangling", "basic_entangler"])
def test_circuit_introspection_both_ansatze(ansatz: str) -> None:
    _require_pl()
    cfg = _small_cfg(ansatz=ansatz)
    model = models_mod.build_qnn(cfg)
    info = models_mod.circuit_introspection(model)
    assert set(info) == {"entangling_gate_count", "trainable_param_count"}
    assert info["entangling_gate_count"] > 0
    if ansatz == "basic_entangler":
        weights = cfg.layers * cfg.qubits
    else:
        weights = cfg.layers * cfg.qubits * 3
    assert info["trainable_param_count"] == weights + 2 * cfg.qubits + 2


def test_qnn_finite_shots_degrade_to_analytic() -> None:
    _require_pl()
    cfg = _small_cfg(shots=128)
    with pytest.warns(RuntimeWarning, match="shots"):
        model = models_mod.build_qnn(cfg)
    logits = model(torch.randn(2, cfg.feature_dim))
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_transfer_qnn_logits_shape() -> None:
    _require_pl()
    pytest.importorskip("torchvision", reason="torchvision required for EfficientNet")
    cfg = _small_cfg()
    model = models_mod.build_transfer_qnn(cfg)
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


# --------------------------------------------------------------------------- quantum kernel (PennyLane + Lightning only)

def test_quantum_kernel_equivalence_and_properties() -> None:
    _require_pl()
    cfg = _small_cfg()
    A = _unit_vectors(4, cfg.feature_dim, seed=7)
    try:
        K = kernel_mod.quantum_kernel_matrix(A, A, cfg)
    except RuntimeError as exc:
        pytest.skip(f"lightning.qubit device unavailable: {exc}")

    Kc = kernel_mod.classical_kernel_matrix(A, A)
    assert np.max(np.abs(K - Kc)) < 1e-6
    assert np.allclose(K, K.T, atol=1e-10)
    assert np.allclose(np.diag(K), np.ones(4), atol=1e-10)
    diag = kernel_mod.kernel_diagnostics(K)
    assert diag["psd"] is True

    # Structural confirmation of the no-advantage claim: the kernel circuit
    # has neither entangling gates nor trainable parameters.
    circuit = kernel_mod._build_kernel_circuit(cfg)
    info = models_mod.circuit_introspection(circuit)
    assert info == {"entangling_gate_count": 0, "trainable_param_count": 0}


# --------------------------------------------------------------------------- QSVC wrapper

def test_build_qsvc_precomputed_random_state() -> None:
    A = _unit_vectors(8, 16, seed=8)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    K = kernel_mod.classical_kernel_matrix(A, A)
    cfg = QMLConfig()
    svc = kernel_mod.build_qsvc(cfg, K, y)
    assert svc.kernel == "precomputed"
    assert svc.probability is True
    assert svc.random_state == cfg.seed

    svc2 = kernel_mod.build_qsvc(cfg, K, y)
    assert np.array_equal(svc.predict(K), svc2.predict(K))
    assert np.allclose(svc.predict_proba(K), svc2.predict_proba(K))
