"""QSVC kernels and the precomputed-kernel SVM wrapper.

Honesty statement (QC-D03 §3.5, QC-D01 C5 — retained verbatim in substance)
--------------------------------------------------------------------------
The canonical QSVC feature map contains **no entangling gates and no trainable
parameters**.  Its circuit value is exactly

    P(|0...0>) = ((x_i . x_j) / (||x_i|| ||x_j||))**2 = (A @ B.T)**2

for L2-normalised rows.  The PennyLane circuit only *reproduces a classical
polynomial kernel*; it provides **no quantum advantage**.  The quantum path is
retained solely as (a) a numerical cross-check that the simulator matches the
closed form and (b) a source of circuit diagrams.  Reported QSVC results should
be produced with the classical oracle (:func:`classical_kernel`) for speed and
determinism, accompanied by a passing :func:`kernel_equivalence_error` check.
The QNN models in ``models/quantum.py`` *do* add entanglement and trainable
weights, but at 10 simulated qubits they remain classically simulable and
likewise warrant no advantage claim.

Public API
----------
``quantum_kernel(A, B, cfg)``      -- exact analytic kernel via lightning.qubit
``classical_kernel(A, B)``         -- algebraic oracle (A @ B.T)**2
``kernel_equivalence_error(A, B)`` -- honest max abs difference between the two
``QSVCWrapper(cfg)``               -- SVC(kernel='precomputed', random_state=cfg.seed)
``build_kernel_circuit(cfg)``      -- QNode exposed for zero-entanglement introspection
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from sklearn.svm import SVC

from experiments.config import ExperimentConfig

try:  # pragma: no cover - depends on the environment
    import pennylane as qml
except Exception:  # pragma: no cover
    qml = None  # type: ignore[assignment]

__all__ = [
    "quantum_kernel",
    "classical_kernel",
    "kernel_equivalence_error",
    "QSVCWrapper",
    "build_kernel_circuit",
    "_l2_normalize",
]

_MISSING_PENNYLANE = (
    "PennyLane is required to evaluate the quantum kernel. Install the pinned "
    "stack (PennyLane==0.40.0 and PennyLane-Lightning==0.40.0). Use "
    "classical_kernel(A, B) for the equivalent classical oracle."
)

_EPS = 1e-12


def _require_pennylane() -> None:
    if qml is None:
        raise RuntimeError(_MISSING_PENNYLANE)


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    """Return L2-row-normalised float64 copy; reject zero rows.

    ``AmplitudeEmbedding(normalize=True)`` is undefined on the zero vector, so
    an explicit check is applied before either kernel is evaluated.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {arr.shape}.")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms <= _EPS):
        raise ValueError("Cannot kernelise an all-zero feature row (norm <= eps).")
    return arr / norms


def build_kernel_circuit(cfg: ExperimentConfig, device: Any = None):
    """Return the parameter-free amplitude-overlap QNode on ``lightning.qubit``.

    The circuit is ``AmplitudeEmbedding(x_i)`` followed by the adjoint of
    ``AmplitudeEmbedding(x_j)``, measuring ``qml.probs(wires=range(n))``.  Index
    0 of the probability vector is P(|0...0>), the squared state overlap.  With
    ``shots=None`` the result is exact, symmetric and PSD.  The QNode is exposed
    so callers can verify ``entangling_gate_count(circuit) == 0`` and
    ``trainable_param_count(circuit) == 0``.
    """
    _require_pennylane()
    n = int(cfg.n_qubits)
    if device is None:
        try:
            device = qml.device("lightning.qubit", wires=n, shots=None)
        except Exception as exc:  # pragma: no cover - plugin-dependent
            raise RuntimeError(
                "Could not create the 'lightning.qubit' device. Install "
                "PennyLane-Lightning==0.40.0 for the QSVC kernel. "
                f"Original error: {exc}"
            ) from exc

    @qml.qnode(device, interface=None)
    def kernel_circuit(x_i, x_j):
        qml.AmplitudeEmbedding(
            features=x_i, wires=range(n), normalize=True, pad_with=0.0
        )
        qml.adjoint(qml.AmplitudeEmbedding)(
            features=x_j, wires=range(n), normalize=True, pad_with=0.0
        )
        return qml.probs(wires=range(n))

    return kernel_circuit


def classical_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Exact classical oracle: ``clip((A @ B.T)**2, 0, 1)`` on L2-normalised rows.

    Returns a float64 matrix of shape ``(len(A), len(B))``.  This is the tested
    default for producing QSVC result matrices (see module docstring).
    """
    a = _l2_normalize(A)
    b = _l2_normalize(B)
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {a.shape[1]} vs {b.shape[1]}."
        )
    K = (a @ b.T) ** 2
    return np.clip(K, 0.0, 1.0).astype(np.float64)


def quantum_kernel(
    A: np.ndarray, B: np.ndarray, cfg: ExperimentConfig
) -> np.ndarray:
    """Exact analytic quantum kernel via ``lightning.qubit`` (``shots=None``).

    Each entry is ``P(|0...0>) = |<phi(x_i)|phi(x_j)>|^2`` from the
    amplitude-embedding overlap circuit.  When ``A`` and ``B`` are the same
    matrix, only the upper triangle is evaluated, mirrored, and the diagonal is
    set explicitly to exactly ``1.0`` (never estimated by a circuit).  Results
    are float64 of shape ``(len(A), len(B))``.

    See the module docstring: this is a *numerical cross-check* of a classical
    polynomial kernel, not a source of quantum advantage.
    """
    a = _l2_normalize(A)
    b = _l2_normalize(B)
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {a.shape[1]} vs {b.shape[1]}."
        )
    if a.shape[1] != int(cfg.feature_dim):
        raise ValueError(
            f"Feature dimension {a.shape[1]} != cfg.feature_dim "
            f"{cfg.feature_dim} (= 2**cfg.n_qubits)."
        )

    circuit = build_kernel_circuit(cfg)
    m, n = a.shape[0], b.shape[0]
    K = np.empty((m, n), dtype=np.float64)
    symmetric = a is b or (a.shape == b.shape and np.array_equal(a, b))

    for i in range(m):
        j_start = i if symmetric else 0
        for j in range(j_start, n):
            probs = np.asarray(circuit(a[i], b[j]), dtype=np.float64)
            value = float(np.clip(probs[0], 0.0, 1.0))
            K[i, j] = value
            if symmetric:
                K[j, i] = value

    if symmetric:
        np.fill_diagonal(K, 1.0)
    return K


def kernel_equivalence_error(
    A: np.ndarray, B: np.ndarray, cfg: ExperimentConfig | None = None
) -> float:
    """Honest ``max|K_quantum - K_classical|`` for the parameter-free map.

    A small value (< 1e-6 at ``shots=None``) confirms the simulator matches the
    closed form -- it is **not** evidence of quantum advantage.  If ``cfg`` is
    omitted it is derived from the feature dimension ``2**n_qubits``.
    """
    if cfg is None:
        d = int(np.asarray(A).shape[-1])
        n = int(round(np.log2(d)))
        cfg = replace(ExperimentConfig(), n_qubits=n, feature_dim=d)
    Kq = quantum_kernel(A, B, cfg)
    Kc = classical_kernel(A, B)
    return float(np.max(np.abs(Kq - Kc)))


class QSVCWrapper:
    """SVM over an explicitly precomputed kernel matrix.

    Uses ``sklearn.svm.SVC(kernel='precomputed', probability=True,
    random_state=cfg.seed)``.  The mandatory ``random_state`` fixes the legacy
    reproducibility defect where ``predict_proba`` depended on fold shuffling.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        C: float = 1.0,
        probability: bool = True,
    ) -> None:
        self.cfg = cfg
        self.random_state: int = int(cfg.seed)
        self.kernel: str = "precomputed"
        self.svc = SVC(
            kernel="precomputed",
            C=float(C),
            probability=bool(probability),
            random_state=self.random_state,
        )

    def fit(self, K_train: np.ndarray, y: np.ndarray) -> "QSVCWrapper":
        self.svc.fit(np.asarray(K_train), np.asarray(y))
        return self

    def predict(self, K_test: np.ndarray) -> np.ndarray:
        return self.svc.predict(np.asarray(K_test))

    def predict_proba(self, K_test: np.ndarray) -> np.ndarray:
        return self.svc.predict_proba(np.asarray(K_test))
