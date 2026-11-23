"""Configurable classification head and split-safe calibration (QHEAD-D01).

The wrapper keeps the legacy ``nn.Linear(n, 2)`` head numerically identical by
default (``temperature=1.0``, zero logit adjustment) while adding two optional,
explicit mechanisms that are fitted on **ID_VAL only**:

* temperature scaling: divide raw logits by a scalar ``T > 0``;
* logit adjustment: add ``-tau * log(train_prior_c)`` per class.

Class priors come from **training labels** (the direction of the adjustment),
and the adjustment strength ``tau`` together with ``T`` are fitted by grid
search on ID_VAL.  Neither routine ever sees validation/test/OOD labels during
training of the head, and :func:`fit_calibration` is never given the OOD loader
by the CLI.  All of this is recorded in the returned provenance dict.

Import safety mirrors ``qmlpipe.train``: torch is optional at import time, and
a clear ``RuntimeError`` is raised only when a torch-dependent function is used.
No dataset path, no project result number and no PennyLane import appear here.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

try:  # pragma: no cover - depends on the host environment
    import torch
    from torch import nn

    _TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

from qmlpipe.losses import class_priors_from_labels

__all__ = [
    "ClassificationHead",
    "build_classification_head",
    "collect_logits",
    "fit_temperature",
    "fit_logit_adjustment",
    "fit_calibration",
    "find_classification_head",
    "write_calibration_to_checkpoint",
]


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - depends on the host environment
        raise RuntimeError(
            "qmlpipe.heads requires PyTorch; "
            f"import failed with {type(_TORCH_IMPORT_ERROR).__name__}: {_TORCH_IMPORT_ERROR}"
        )


if nn is None:  # pragma: no cover - torch-free verifier placeholder

    class ClassificationHead:  # type: ignore[no-redef]
        """Placeholder that raises on use when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

else:

    class ClassificationHead(nn.Module):  # type: ignore[no-redef]
        """Wrap a logit-producing module with temperature + logit adjustment.

        The wrapped ``base`` module typically is ``nn.Linear(n_qubits, 2)``.
        ``temperature`` and ``logit_adjust`` are registered buffers (not trainable
        parameters), so they survive ``state_dict`` round-trips without changing
        trainable-parameter counts.  With the defaults the forward pass is bit-
        for-bit the legacy head: ``logits / 1.0 + 0``.
        """

        def __init__(
            self,
            base: "nn.Module",
            *,
            temperature: float = 1.0,
            logit_adjust: Any | None = None,
            n_classes: int = 2,
        ) -> None:
            super().__init__()
            self.base = base
            self.n_classes = int(n_classes)
            self.register_buffer(
                "temperature", torch.tensor(float(temperature), dtype=torch.float32)
            )
            if logit_adjust is None:
                adjust = torch.zeros(self.n_classes, dtype=torch.float32)
            else:
                adjust = torch.as_tensor(logit_adjust, dtype=torch.float32).clone().reshape(-1)
                if adjust.numel() != self.n_classes:
                    raise ValueError(
                        f"logit_adjust must have {self.n_classes} entries, got {adjust.numel()}"
                    )
            self.register_buffer("logit_adjust", adjust)

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            return self.apply_calibration(self.base(features))

        def apply_calibration(self, logits: "torch.Tensor") -> "torch.Tensor":
            """Return ``logits / temperature + logit_adjust`` with matching dtype."""
            temperature = self.temperature.to(dtype=logits.dtype).clamp_min(1e-6)
            adjust = self.logit_adjust.to(dtype=logits.dtype)
            return logits / temperature + adjust

        def set_calibration(
            self, *, temperature: float | None = None, logit_adjust: Any | None = None
        ) -> None:
            """Set fitted calibration values in place (used only at evaluation)."""
            if temperature is not None:
                value = float(temperature)
                if value <= 0.0:
                    raise ValueError("temperature must be positive")
                self.temperature.fill_(value)
            if logit_adjust is not None:
                adjust = torch.as_tensor(logit_adjust, dtype=self.logit_adjust.dtype).reshape(-1)
                if adjust.numel() != self.n_classes:
                    raise ValueError(
                        f"logit_adjust must have {self.n_classes} entries, got {adjust.numel()}"
                    )
                self.logit_adjust.copy_(adjust)

        def reset_calibration(self) -> None:
            """Restore legacy behaviour (T=1, zero adjustment)."""
            self.set_calibration(
                temperature=1.0, logit_adjust=np.zeros(self.n_classes, dtype=np.float32)
            )

        def calibration_state(self) -> dict:
            """JSON-native snapshot of the current calibration buffers."""
            return {
                "temperature": float(self.temperature.detach().cpu()),
                "logit_adjust": [float(v) for v in self.logit_adjust.detach().cpu().tolist()],
                "n_classes": int(self.n_classes),
            }


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping) and name in cfg:
        return cfg[name]
    value = getattr(cfg, name, None)
    if value is not None:
        return value
    as_dict = getattr(cfg, "as_dict", None)
    if callable(as_dict):
        try:
            return as_dict().get(name, default)
        except Exception:
            pass
    return default


def build_classification_head(
    in_features: int,
    cfg: Any | None = None,
    *,
    n_classes: int = 2,
    temperature: float | None = None,
    logit_adjust: Any | None = None,
) -> Any:
    """Build ``ClassificationHead(nn.Linear(in_features, n_classes))``.

    Defaults preserve the legacy ``Linear(10, 2)`` head exactly.  ``cfg`` may
    optionally supply ``head_temperature`` / ``head_logit_adjust``; explicit
    keyword arguments take precedence.  Calibration is normally applied later by
    :func:`fit_calibration`, not at construction time.
    """
    _require_torch()
    if temperature is None:
        temperature = float(_cfg_get(cfg, "head_temperature", 1.0))
    if logit_adjust is None:
        logit_adjust = _cfg_get(cfg, "head_logit_adjust", None)
    base = nn.Linear(int(in_features), int(n_classes))
    return ClassificationHead(
        base,
        temperature=float(temperature),
        logit_adjust=logit_adjust,
        n_classes=int(n_classes),
    )


def find_classification_head(model: Any) -> Any | None:
    """Return the first :class:`ClassificationHead` in ``model.modules()``."""
    _require_torch()
    for module in model.modules():
        if isinstance(module, ClassificationHead):
            return module
    return None


def write_calibration_to_checkpoint(model: Any, path: Any, calibration: dict) -> None:
    """Persist fitted calibration buffers into an existing best checkpoint.

    The model weights are left untouched; only ``temperature`` / ``logit_adjust``
    buffers and a ``calibration`` provenance block are updated, so evaluating
    from ``best.pt`` reproduces the calibrated head without re-fitting.
    """
    _require_torch()
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch without weights_only
        payload = torch.load(str(path), map_location="cpu")
    state = payload.get("state_dict", {})
    for key, value in model.state_dict().items():
        if "temperature" in key or "logit_adjust" in key:
            state[key] = value.detach().cpu()
    payload["state_dict"] = state
    payload["calibration"] = calibration
    torch.save(payload, str(path))


def _model_device_and_dtype(model: Any):
    try:
        parameter = next(model.parameters())
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu"), torch.float32
    return parameter.device, (
        parameter.dtype if parameter.is_floating_point() else torch.float32
    )


def collect_logits(model: Any, loader: Any) -> tuple[np.ndarray, np.ndarray]:
    """Collect **raw** logits ``(N, 2)`` and labels ``(N,)`` from ``loader``.

    The head is reset to legacy defaults before collection so the returned
    logits are uncalibrated.  The model is restored to its prior train/eval
    mode afterwards.  This helper is deliberately split-agnostic: the caller
    decides which split to pass (the CLI passes ID_VAL only).
    """
    _require_torch()
    head = find_classification_head(model)
    if head is not None:
        head.reset_calibration()
    was_training = bool(getattr(model, "training", False))
    model.eval()
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    device, dtype = _model_device_and_dtype(model)
    try:
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError("logit-collection batches must be (features, labels)")
                X, y = batch[0], batch[1]
                if not isinstance(X, torch.Tensor):
                    X = torch.as_tensor(X)
                if not isinstance(y, torch.Tensor):
                    y = torch.as_tensor(y)
                X = X.to(device=device, dtype=dtype)
                out = model(X)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                logits_list.append(out.detach().cpu().numpy().astype(np.float64).reshape(-1, 2))
                labels_list.append(y.detach().cpu().numpy().reshape(-1).astype(np.int64))
    finally:
        model.train(was_training)
    if not logits_list:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int64)
    return np.concatenate(logits_list, axis=0), np.concatenate(labels_list, axis=0)


def _nll(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of a softmax over logits (numpy, stable)."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0:
        raise ValueError("logits must be a non-empty (N, K) array")
    z = z - z.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if y.shape[0] != z.shape[0]:
        raise ValueError("labels and logits must have the same number of rows")
    return float(-log_p[np.arange(z.shape[0]), y].mean())


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    t_min: float = 0.05,
    t_max: float = 20.0,
    n_grid: int = 200,
) -> float:
    """Fit a scalar temperature by grid search minimising NLL.

    Uses only the supplied ``logits``/``labels`` (ID_VAL by contract).  The grid
    is log-spaced and deterministic; no randomness and no scipy dependency.
    """
    z = np.asarray(logits, dtype=np.float64)
    grid = np.geomspace(float(t_min), float(t_max), int(n_grid))
    scores = [_nll(z / t, labels) for t in grid]
    return float(grid[int(np.argmin(scores))])


def fit_logit_adjustment(
    logits: np.ndarray,
    labels: np.ndarray,
    train_priors: np.ndarray,
    temperature: float,
    *,
    tau_max: float = 3.0,
    n_grid: int = 61,
) -> tuple[np.ndarray, float]:
    """Fit the logit-adjustment strength ``tau`` by NLL grid search on ID_VAL.

    The adjustment direction is fixed to ``-log(train_prior_c)`` (priors from
    TRAIN labels).  Only ``tau`` is searched, on the supplied validation
    logits/labels.  Returns ``(adjust_vector, tau)``.
    """
    z = np.asarray(logits, dtype=np.float64)
    base = -np.log(np.clip(np.asarray(train_priors, dtype=np.float64), 1e-6, 1.0))
    taus = np.linspace(0.0, float(tau_max), int(n_grid))
    best_adjust = np.zeros_like(base)
    best_tau = 0.0
    best_score = _nll(z / float(temperature), labels)
    for tau in taus:
        adjust = tau * base
        score = _nll(z / float(temperature) + adjust, labels)
        if score < best_score:
            best_score, best_tau, best_adjust = score, float(tau), adjust
    return best_adjust.astype(np.float32), best_tau


def fit_calibration(
    model: Any,
    val_loader: Any,
    train_labels: Any,
    cfg: Any | None = None,
    *,
    n_classes: int = 2,
    fit_logit_adjust: bool = True,
) -> dict:
    """Fit temperature (and optional logit adjustment) on ID_VAL only.

    ``val_loader`` must be the in-distribution validation loader.  ``train_labels``
    must be the TRAINING labels (used only to compute the prior direction).
    This function never receives or reads the OOD loader; the returned
    provenance records ``fitted_on_split="id_val"`` and
    ``ood_labels_used=False`` explicitly.

    On success the fitted values are applied to the model's
    :class:`ClassificationHead` in place and a JSON-native provenance dict is
    returned.  If the model has no such head, calibration is skipped cleanly
    (``status="skipped"``) rather than faked.
    """
    head = find_classification_head(model)
    if head is None:
        return {
            "status": "skipped",
            "reason": "model has no ClassificationHead; calibration not applied",
            "fitted_on_split": "id_val",
            "ood_labels_used": False,
        }

    val_logits, val_labels = collect_logits(model, val_loader)
    if val_logits.shape[0] == 0:
        return {
            "status": "skipped",
            "reason": "empty ID_VAL loader; calibration not applied",
            "fitted_on_split": "id_val",
            "ood_labels_used": False,
        }

    priors = class_priors_from_labels(train_labels, n_classes=n_classes)
    temperature = fit_temperature(val_logits, val_labels)
    if fit_logit_adjust:
        adjust, tau = fit_logit_adjustment(val_logits, val_labels, priors, temperature)
    else:
        adjust, tau = np.zeros(int(n_classes), dtype=np.float32), 0.0

    head.set_calibration(temperature=temperature, logit_adjust=adjust)
    return {
        "status": "fitted",
        "method": "grid_nll_temperature" + ("_plus_logit_adjustment" if fit_logit_adjust else ""),
        "fitted_on_split": "id_val",
        "n_val_samples": int(val_logits.shape[0]),
        "class_prior_source": "train_labels",
        "class_priors": [float(v) for v in priors],
        "temperature": float(temperature),
        "logit_adjustment_tau": float(tau),
        "logit_adjust": [float(v) for v in adjust],
        "head_state": head.calibration_state(),
        "ood_labels_used": False,
        "note": (
            "Calibration was fitted on ID_VAL only. A validation-only gain is "
            "NOT evidence of improved OOD performance; OOD metrics are reported "
            "separately from a distinct, untouched split. Temperature scaling and "
            "a constant class logit shift preserve the within-class logit ordering "
            "and cannot manufacture class separation that the representation lacks."
        ),
    }
