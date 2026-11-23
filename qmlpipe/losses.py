"""Class-balance-aware loss functions for the hybrid QNN head (QHEAD-D01).

All statistics used here are computed from **training labels only**.  This module
contains no dataset path, no project result number and no PennyLane import.  It
is import-safe without PyTorch: the public functions raise a clear
``RuntimeError`` only when they are actually called in a torch-free environment,
so a minimal verifier can still import the module.

Public API
----------
``class_weights_from_labels`` -- balanced CE weights ``n / (K * count_c)``.
``class_priors_from_labels``  -- empirical prior ``count_c / n`` per class.
``weighted_cross_entropy``    -- cross entropy with an optional per-class weight.
``focal_loss``                -- focal loss (``(1-p_t)**gamma``) with weights.

Design notes
------------
* ``as_two_logits`` converts a single sigmoid logit ``z`` to ``[-z/2, z/2]`` so
  that ``softmax(...)[1] == sigmoid(z)`` and cross entropy is *exactly* equal to
  ``binary_cross_entropy_with_logits``.  This keeps the legacy sigmoid path
  numerically identical when no class weighting is requested.
* Focal loss follows Lin et al. (2017): ``FL(p_t) = -alpha_t (1-p_t)^gamma
  log(p_t)``.  ``gamma=0`` with ``weight`` reduces exactly to weighted cross
  entropy, which is what the synthetic tests check.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - depends on the host environment
    import torch
    import torch.nn.functional as F

    _TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

__all__ = [
    "class_weights_from_labels",
    "class_priors_from_labels",
    "weighted_cross_entropy",
    "focal_loss",
    "as_two_logits",
]


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - depends on the host environment
        raise RuntimeError(
            "qmlpipe.losses requires PyTorch; "
            f"import failed with {type(_TORCH_IMPORT_ERROR).__name__}: {_TORCH_IMPORT_ERROR}"
        )


def _as_numpy(labels: Any) -> np.ndarray:
    """Convert a torch tensor / numpy array / list of labels to a flat int array."""
    if torch is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels).reshape(-1).astype(np.int64)


def _counts(labels: Any, n_classes: int) -> tuple[np.ndarray, int]:
    y = _as_numpy(labels)
    if y.size == 0:
        raise ValueError("cannot derive class statistics from an empty label set")
    if int(y.min()) < 0 or int(y.max()) >= int(n_classes):
        raise ValueError(
            f"labels must be in [0, {int(n_classes) - 1}], got "
            f"[{int(y.min())}, {int(y.max())}]"
        )
    counts = np.bincount(y, minlength=int(n_classes))[: int(n_classes)].astype(np.float64)
    return counts, int(y.size)


def class_weights_from_labels(
    labels: Any, n_classes: int = 2, mode: str = "balanced"
) -> "torch.Tensor":
    """Return a ``(n_classes,)`` float tensor of per-class loss weights.

    ``mode="none"`` returns all ones (legacy unweighted cross entropy).
    ``mode="balanced"`` returns ``n / (K * count_c)`` for each observed class,
    the same convention as ``sklearn.utils.class_weight.compute_class_weight``
    with ``class_weight="balanced"``.  A class with zero training samples would
    produce an infinite weight, so it is assigned ``1.0`` and flagged by the
    caller's provenance instead of silently exploding the loss.

    Parameters
    ----------
    labels:
        Training labels only.  Never validation, test or OOD labels.
    n_classes:
        Number of classes (2 for the weed/crop head).
    mode:
        ``"none"`` or ``"balanced"``.
    """
    _require_torch()
    key = str(mode).strip().lower()
    if key not in {"none", "balanced"}:
        raise ValueError(f"unknown class-weight mode {mode!r}; expected 'none' or 'balanced'")
    if key == "none":
        return torch.ones(int(n_classes), dtype=torch.float32)
    counts, n = _counts(labels, n_classes)
    weights = np.ones(int(n_classes), dtype=np.float64)
    observed = counts > 0
    weights[observed] = n / (float(n_classes) * counts[observed])
    return torch.as_tensor(weights, dtype=torch.float32)


def class_priors_from_labels(
    labels: Any, n_classes: int = 2, eps: float = 1e-6
) -> np.ndarray:
    """Return the empirical class prior ``count_c / n`` from **training** labels.

    Used as the direction for logit adjustment in :mod:`qmlpipe.heads`.  Zero-
    count classes get ``eps`` rather than exactly zero so ``log(prior)`` stays
    finite.  Returns a plain ``numpy.ndarray`` of shape ``(n_classes,)``.
    """
    counts, n = _counts(labels, n_classes)
    priors = counts / float(n)
    return np.clip(priors, float(eps), 1.0)


def as_two_logits(logits: "torch.Tensor") -> "torch.Tensor":
    """Normalise binary logits to a ``(N, 2)`` tensor.

    * ``(N, 2+)`` -> first two columns (unchanged, the QNN/Linear head case).
    * ``(N,)`` or ``(N, 1)`` -> ``[-z/2, z/2]`` so the softmax probability
      equals ``sigmoid(z)`` and CE equals BCE-with-logits (legacy parity).
    """
    _require_torch()
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits, dtype=torch.float32)
    if logits.ndim == 1:
        return torch.stack((-0.5 * logits, 0.5 * logits), dim=1)
    if logits.ndim == 2 and logits.shape[1] == 1:
        z = logits[:, 0]
        return torch.stack((-0.5 * z, 0.5 * z), dim=1)
    if logits.ndim == 2 and logits.shape[1] >= 2:
        return logits[:, :2]
    raise ValueError(f"unsupported logits shape {tuple(logits.shape)} for binary training")


def weighted_cross_entropy(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    weight: "torch.Tensor | None" = None,
    reduction: str = "mean",
) -> "torch.Tensor":
    """Cross entropy on binary logits with optional per-class ``weight``.

    ``weight`` is a ``(2,)`` tensor from :func:`class_weights_from_labels`;
    ``None`` reproduces ``torch.nn.functional.cross_entropy`` exactly.
    """
    _require_torch()
    two = as_two_logits(logits)
    y = labels.long().reshape(-1)
    if weight is not None:
        weight = weight.to(device=two.device, dtype=two.dtype)
    return F.cross_entropy(two, y, weight=weight, reduction=reduction)


def focal_loss(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    weight: "torch.Tensor | None" = None,
    gamma: float = 2.0,
    alpha: "float | torch.Tensor | None" = None,
    reduction: str = "mean",
) -> "torch.Tensor":
    """Focal loss for binary logits.

    ``FL(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)``.  Correctly classified,
    high-confidence examples have ``p_t -> 1`` and are down-weighted by
    ``(1-p_t)**gamma``; hard/misclassified examples are not.  ``gamma == 0``
    reduces to :func:`weighted_cross_entropy`.

    ``alpha`` may be ``None`` (no extra class weighting), a float (the positive
    class gets ``alpha`` and the negative gets ``1-alpha``), or a ``(2,)``
    tensor indexed by the label.  ``weight`` is the per-class tensor returned by
    :func:`class_weights_from_labels` and multiplies the per-sample loss.
    """
    _require_torch()
    two = as_two_logits(logits)
    y = labels.long().reshape(-1)
    log_p = F.log_softmax(two.float(), dim=1)
    p = log_p.exp()
    idx = y.view(-1, 1)
    log_pt = log_p.gather(1, idx).squeeze(1)
    pt = p.gather(1, idx).squeeze(1)
    focal = (1.0 - pt).clamp(min=0.0).pow(float(gamma))

    per_sample = -focal * log_pt
    if weight is not None:
        per_sample = per_sample * weight.to(device=two.device, dtype=per_sample.dtype)[y]
    if alpha is not None:
        if isinstance(alpha, torch.Tensor):
            alpha_t = alpha.to(device=two.device, dtype=per_sample.dtype)[y]
        else:
            alpha_value = float(alpha)
            alpha_t = torch.where(
                y == 1,
                torch.as_tensor(alpha_value, dtype=per_sample.dtype, device=two.device),
                torch.as_tensor(1.0 - alpha_value, dtype=per_sample.dtype, device=two.device),
            )
        per_sample = per_sample * alpha_t

    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "none":
        return per_sample
    raise ValueError(f"unknown reduction {reduction!r}")
