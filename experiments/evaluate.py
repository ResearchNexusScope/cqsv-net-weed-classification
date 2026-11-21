"""Model adapters and per-sample prediction primitives (QC-D04 §1.1 / §2.2).

The runner depends only on the small :class:`Predictor` protocol, so datasets,
model checkpoints and PennyLane are injected rather than discovered.  No class
in this module reads a dataset path or downloads a weight file.  When the
quantum kernel cannot be evaluated, the classical algebraic equivalent
``(A @ B.T)**2`` is used and the substitution is recorded as a warning; it is a
mathematical identity for the zero-entanglement feature map, not a fabricated
result.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np

__all__ = [
    "ModelPrediction",
    "Predictor",
    "PredictorError",
    "CallablePredictor",
    "TensorPredictor",
    "KernelSVMPredictor",
    "QSVC_MODEL_IDS",
    "build_qsvc_predictors",
    "predictor_warnings",
]

# The five QSVC arms required by QC-D04 §2.2.  Order is fixed for serialisation.
QSVC_MODEL_IDS: tuple[str, ...] = (
    "qsvc_quantum_kernel",
    "svc_rbf",
    "svc_linear",
    "svc_poly2_classical",
    "qsvc_no_mixing_control",
)


class PredictorError(RuntimeError):
    """A predictor was asked to predict without the information it needs."""


@dataclass(frozen=True)
class ModelPrediction:
    """Hard labels, optional positive-class scores and raw batch latency.

    ``inference_ms`` is wall-clock time for the whole call, measured with
    ``time.perf_counter``; the runner attributes it per sample when serialising.
    """

    y_pred: np.ndarray
    score_pos: np.ndarray | None
    inference_ms: float


class Predictor(Protocol):
    """Minimal interface the runner requires from every model arm."""

    def predict(
        self,
        X: np.ndarray,
        *,
        fit_X: np.ndarray | None = None,
        fit_y: np.ndarray | None = None,
        seed: int = 0,
    ) -> ModelPrediction:
        ...


@dataclass(frozen=True)
class CallablePredictor:
    """Adapt a plain function to :class:`Predictor` (used by test fixtures).

    The callable receives ``(X, fit_X, fit_y, seed)`` and returns either a
    :class:`ModelPrediction` or a ``(y_pred, score_pos, inference_ms)`` tuple.
    """

    fn: Callable[..., Any]

    def predict(
        self,
        X: np.ndarray,
        *,
        fit_X: np.ndarray | None = None,
        fit_y: np.ndarray | None = None,
        seed: int = 0,
    ) -> ModelPrediction:
        result = self.fn(X, fit_X, fit_y, seed)
        if isinstance(result, ModelPrediction):
            return result
        if isinstance(result, tuple) and len(result) == 3:
            y_pred, score_pos, inference_ms = result
            return ModelPrediction(
                np.asarray(y_pred),
                None if score_pos is None else np.asarray(score_pos),
                float(inference_ms),
            )
        raise PredictorError("CallablePredictor fn returned an unsupported object")


class TensorPredictor:
    """Inference adapter for an injected torch module emitting 2-way logits.

    ``module`` must already be trained and supplied by the caller; this adapter
    never constructs or downloads weights, so it cannot silently invent a
    benchmark number.  A missing module is an error at injection time.
    """

    def __init__(self, module: Any, *, batch_size: int = 32) -> None:
        if module is None:
            raise PredictorError("TensorPredictor requires a trained module")
        self.module = module
        self.batch_size = max(1, int(batch_size))

    def predict(
        self,
        X: np.ndarray,
        *,
        fit_X: np.ndarray | None = None,
        fit_y: np.ndarray | None = None,
        seed: int = 0,
    ) -> ModelPrediction:
        import torch  # lazy: keeps module import safe in minimal environments

        array = np.asarray(X, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if len(array) == 0:
            return ModelPrediction(
                np.asarray([], dtype=int), np.asarray([], dtype=float), 0.0
            )
        module = self.module
        module.eval()
        predictions: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        started = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(array), self.batch_size):
                batch = torch.from_numpy(array[start : start + self.batch_size])
                logits = module(batch)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                probabilities = torch.softmax(logits.float(), dim=1)
                predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
                scores.append(probabilities[:, 1].cpu().numpy())
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ModelPrediction(
            np.concatenate(predictions).astype(int),
            np.concatenate(scores).astype(float),
            elapsed_ms,
        )


class KernelSVMPredictor:
    """SVC over a precomputed kernel or a classical rbf/linear control.

    ``kernel`` is one of ``"quantum"``, ``"quantum_if_available"``,
    ``"classical"``, ``"rbf"`` or ``"linear"``.  The first three use
    :class:`models.qsvc.QSVCWrapper`; ``"quantum_if_available"`` falls back to
    the exact classical oracle when PennyLane is absent, the feature dimension
    is not amplitude-embeddable, or the simulator call fails.  Every fallback
    is appended to :attr:`warnings` so the manifest and metrics disclose it.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        kernel: str = "quantum_if_available",
        C: float = 1.0,
        probability: bool = True,
    ) -> None:
        allowed = {"quantum", "quantum_if_available", "classical", "rbf", "linear"}
        if kernel not in allowed:
            raise PredictorError(f"Unknown kernel {kernel!r}; expected one of {sorted(allowed)}")
        self.cfg = cfg
        self.kernel = kernel
        self.C = float(C)
        self.probability = bool(probability)
        self.warnings: list[str] = []

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def _kernel_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        from models import qsvc as qsvc_mod  # lazy: import-safe without PennyLane

        left = np.asarray(A, dtype=np.float64)
        right = np.asarray(B, dtype=np.float64)
        if self.kernel == "quantum":
            return qsvc_mod.quantum_kernel(left, right, self.cfg)
        if self.kernel == "classical":
            return qsvc_mod.classical_kernel(left, right)

        # quantum_if_available: faithful circuit when possible, exact classical
        # polynomial-kernel oracle otherwise.  Both are the same kernel.
        pennylane_present = getattr(qsvc_mod, "qml", None) is not None
        dimension_ok = left.shape[1] == int(self.cfg.feature_dim)
        if pennylane_present and dimension_ok:
            try:
                return qsvc_mod.quantum_kernel(left, right, self.cfg)
            except Exception as exc:  # simulator/plugin failures degrade explicitly
                self._warn(
                    "quantum kernel evaluation failed "
                    f"({type(exc).__name__}); used classical (A@B.T)^2 oracle"
                )
                return qsvc_mod.classical_kernel(left, right)
        reason = (
            "PennyLane is not installed"
            if not pennylane_present
            else f"feature_dim {left.shape[1]} != cfg.feature_dim {self.cfg.feature_dim}"
        )
        self._warn(f"{reason}; used classical (A@B.T)^2 oracle")
        return qsvc_mod.classical_kernel(left, right)

    def predict(
        self,
        X: np.ndarray,
        *,
        fit_X: np.ndarray | None = None,
        fit_y: np.ndarray | None = None,
        seed: int = 0,
    ) -> ModelPrediction:
        if fit_X is None or fit_y is None:
            raise PredictorError(
                f"{self.kernel} predictor needs fit_X/fit_y from the paired run"
            )
        features = np.asarray(X, dtype=np.float64)
        training = np.asarray(fit_X, dtype=np.float64)
        labels = np.asarray(fit_y)
        if features.ndim != 2 or training.ndim != 2:
            raise PredictorError("Kernel predictors require 2D feature matrices")
        if features.shape[1] != training.shape[1]:
            raise PredictorError(
                f"Feature dimension mismatch: {features.shape[1]} vs {training.shape[1]}"
            )

        started = time.perf_counter()
        if self.kernel in ("rbf", "linear"):
            from sklearn.svm import SVC

            classifier = SVC(
                kernel=self.kernel,
                C=self.C,
                probability=self.probability,
                random_state=int(seed),
            )
            classifier.fit(training, labels)
            y_pred = classifier.predict(features)
            score_pos = classifier.predict_proba(features)[:, 1]
        else:
            from models.qsvc import QSVCWrapper

            kernel_train = self._kernel_matrix(training, training)
            kernel_test = self._kernel_matrix(features, training)
            wrapper = QSVCWrapper(self.cfg, C=self.C, probability=self.probability)
            wrapper.random_state = int(seed)
            wrapper.svc.set_params(random_state=int(seed))
            wrapper.fit(kernel_train, labels)
            y_pred = wrapper.predict(kernel_test)
            score_pos = wrapper.predict_proba(kernel_test)[:, 1]
        return ModelPrediction(
            np.asarray(y_pred, dtype=int),
            np.asarray(score_pos, dtype=float),
            (time.perf_counter() - started) * 1000.0,
        )


def build_qsvc_predictors(
    cfg: Any, model_ids: Sequence[str]
) -> dict[str, KernelSVMPredictor]:
    """Build the injected QSVC/classical-control arms named in QC-D04 §2.2."""
    kernels = {
        "qsvc_quantum_kernel": "quantum_if_available",
        "qsvc_no_mixing_control": "quantum_if_available",
        "svc_poly2_classical": "classical",
        "svc_rbf": "rbf",
        "svc_linear": "linear",
    }
    return {
        model_id: KernelSVMPredictor(cfg, kernel=kernels[model_id])
        for model_id in model_ids
        if model_id in kernels
    }


def predictor_warnings(predictor: Any) -> tuple[str, ...]:
    """Return any disclosure strings a predictor accumulated during prediction."""
    values = getattr(predictor, "warnings", ()) or ()
    return tuple(str(value) for value in values)
