"""Quantum and classical kernels for the QSVC branch (QML-D01).

Public API
----------
``quantum_kernel_matrix(A, B, cfg)``  -- exact analytic amplitude-overlap kernel
``classical_kernel_matrix(A, B)``     -- algebraic oracle ``(A @ B.T) ** 2``
``kernel_diagnostics(K)``             -- symmetry / diagonal / PSD diagnostics
``build_qsvc(cfg, K_train, y_train)`` -- fitted ``SVC(kernel='precomputed')``

No quantum advantage (retained from QC-D03 §3.5)
------------------------------------------------
The circuit used by :func:`quantum_kernel_matrix` contains **no entangling
gates and no trainable parameters**.  It is ``AmplitudeEmbedding(x_i)``
followed by the adjoint of ``AmplitudeEmbedding(x_j)`` and returns
``P(|0...0>) = |<phi(x_i)|phi(x_j)>|^2 = (x_i . x_j)^2`` for L2-normalised real
rows -- exactly the classical polynomial kernel computed by
:func:`classical_kernel_matrix`.  Running the PennyLane circuit therefore
reproduces a classical kernel and provides **no quantum advantage**.  The
quantum path is kept only as (a) a numerical cross-check of the closed form and
(b) a source of circuit diagrams; reported QSVC results should be produced with
the classical oracle.  The trainable QNNs in ``qmlpipe.models`` *do* add
entanglement and weights, but at simulated qubit counts they remain classically
simulable and likewise warrant no advantage claim.

Import safety
-------------
PennyLane is imported defensively.  Importing this module never fails when the
quantum stack is absent; only :func:`quantum_kernel_matrix` raises a clear
``RuntimeError`` at call time.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from qmlpipe.config import QMLConfig

try:  # pragma: no cover - depends on the environment
    import pennylane as qml
except Exception:  # pragma: no cover
    qml = None  # type: ignore[assignment]

__all__ = [
    "quantum_kernel_matrix",
    "classical_kernel_matrix",
    "kernel_diagnostics",
    "build_qsvc",
    "_l2_normalize",
]

_MISSING_PENNYLANE = (
    "PennyLane is required to evaluate the quantum kernel. Install the pinned "
    "stack (PennyLane==0.40.0 and PennyLane-Lightning==0.40.0). Use "
    "classical_kernel_matrix(A, B) for the equivalent classical oracle."
)

_EPS = 1e-12
_PSD_TOL = 1e-8


def _require_pennylane() -> None:
    if qml is None:
        raise RuntimeError(_MISSING_PENNYLANE)


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    """Return an L2-row-normalised float64 array, rejecting all-zero rows.

    ``AmplitudeEmbedding(normalize=True)`` is undefined on the zero vector, so
    this guard is applied before either kernel is evaluated.
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


def classical_kernel_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Classical oracle ``(A @ B.T) ** 2`` for L2-normalised feature rows.

    Parameters
    ----------
    A, B:
        Feature matrices of shape ``(m, D)`` and ``(n, D)``.  Rows are expected
        to be L2-normalised (the feature extractor guarantees this); for such
        rows every entry lies in ``[0, 1]`` and the diagonal is exactly 1.

    Returns
    -------
    np.ndarray
        Float64 kernel matrix of shape ``(m, n)``.
    """
    a = np.asarray(A, dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D feature matrices, got {a.shape} and {b.shape}.")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {a.shape[1]} vs {b.shape[1]}."
        )
    K = (a @ b.T) ** 2
    return np.asarray(K, dtype=np.float64)


def _build_kernel_circuit(cfg: QMLConfig):
    """Construct (but do not cache) the parameter-free amplitude-overlap QNode."""
    _require_pennylane()
    n = int(cfg.qubits)
    try:
        dev = qml.device("lightning.qubit", wires=n, shots=None)
    except Exception as exc:  # pragma: no cover - plugin dependent
        raise RuntimeError(
            "Could not create the 'lightning.qubit' device for the quantum "
            "kernel. Install PennyLane-Lightning==0.40.0. "
            f"Original error: {exc}"
        ) from exc

    @qml.qnode(dev, interface=None)
    def kernel_circuit(x_i, x_j):
        qml.AmplitudeEmbedding(
            features=x_i, wires=range(n), normalize=True, pad_with=0.0
        )
        qml.adjoint(qml.AmplitudeEmbedding)(
            features=x_j, wires=range(n), normalize=True, pad_with=0.0
        )
        return qml.probs(wires=range(n))

    return kernel_circuit


def quantum_kernel_matrix(
    A: np.ndarray, B: np.ndarray, cfg: QMLConfig
) -> np.ndarray:
    """Exact analytic quantum kernel via ``lightning.qubit`` (``shots=None``).

    Each entry is ``P(|0...0>) = |<phi(x_i)|phi(x_j)>|^2`` from the
    amplitude-embedding overlap circuit.  When ``A`` and ``B`` are identical
    the upper triangle is evaluated, mirrored, and the diagonal is set to
    exactly ``1.0`` (never estimated by a circuit), guaranteeing symmetry, a
    unit diagonal and positive semidefiniteness in the analytic regime.

    This is a *numerical cross-check of a classical polynomial kernel*, not a
    source of quantum advantage (see module docstring).
    """
    _require_pennylane()
    a = _l2_normalize(A)
    b = _l2_normalize(B)
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {a.shape[1]} vs {b.shape[1]}."
        )
    if a.shape[1] != int(cfg.feature_dim):
        raise ValueError(
            f"Feature dimension {a.shape[1]} != cfg.feature_dim "
            f"{cfg.feature_dim} (= 2**cfg.qubits)."
        )

    circuit = _build_kernel_circuit(cfg)
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


def kernel_diagnostics(K: np.ndarray) -> dict:
    """Return symmetry, diagonal and PSD diagnostics for a kernel matrix.

    Keys
    ----
    ``symmetry_error`` : ``max|K - K.T|`` (``nan`` for a non-square matrix).
    ``min_diagonal``   : smallest diagonal entry.
    ``min_eigenvalue`` : smallest eigenvalue (square matrices only; else ``nan``).
    ``psd``            : ``min_eigenvalue >= -1e-8`` (``False`` if non-square).
    """
    mat = np.asarray(K, dtype=np.float64)
    result: dict[str, Any] = {
        "symmetry_error": float("nan"),
        "min_diagonal": float("nan"),
        "min_eigenvalue": float("nan"),
        "psd": False,
    }
    if mat.ndim != 2:
        return result

    min_diag = float(np.min(np.diag(mat))) if mat.shape[0] and mat.shape[1] else float("nan")
    result["min_diagonal"] = min_diag

    if mat.shape[0] != mat.shape[1]:
        return result

    result["symmetry_error"] = float(np.max(np.abs(mat - mat.T)))
    min_eig = float(np.min(np.linalg.eigvalsh(mat)))
    result["min_eigenvalue"] = min_eig
    result["psd"] = bool(min_eig >= -_PSD_TOL)
    return result


def build_qsvc(cfg: QMLConfig, K_train: np.ndarray, y_train: np.ndarray):
    """Fit and return ``SVC(kernel='precomputed', probability=True,
    random_state=cfg.seed)``.

    The mandatory ``random_state`` fixes the legacy defect where
    ``predict_proba`` depended on fold shuffling (QC-D03 §6.5).  ``K_train`` is
    an already-computed square kernel matrix; ``y_train`` its labels.
    """
    from sklearn.svm import SVC  # lazy: classical import must not require sklearn

    svc = SVC(
        kernel="precomputed",
        probability=True,
        random_state=int(cfg.seed),
    )
    svc.fit(np.asarray(K_train), np.asarray(y_train))
    return svc
