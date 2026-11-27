"""Evaluation helpers for the rebuild's QSVC arm and cross-model comparisons (QML-D02).

Both public functions delegate every number to :mod:`experiments.metrics`:
``binary_metrics``, ``log_loss``, ``calibration_summary``, ``summarize_runs``
and ``paired_t_test``.  No significance is ever reported for fewer than two
paired runs -- the underlying ``paired_t_test`` itself rejects such inputs, and
this module records an explicit ``insufficient_samples`` status instead of
inventing a p-value.  The module is NumPy/scikit-learn only, so it imports
without PennyLane or torch.
"""
from __future__ import annotations

import itertools
from typing import Any, Mapping

import numpy as np


__all__ = ["evaluate_qsvc", "compare_models"]


def _as_vector(value: Any) -> list[float] | None:
    """Return a run vector from a plain sequence or a common metrics mapping."""
    if isinstance(value, (list, tuple, np.ndarray)):
        flat = np.asarray(value, dtype=float).ravel()
        if flat.size and np.issubdtype(flat.dtype, np.number):
            return [float(item) for item in flat]
        return None
    if isinstance(value, Mapping):
        for key in ("scores", "values", "run_scores", "accuracy_values"):
            if key in value:
                return _as_vector(value[key])
        accuracy = value.get("accuracy")
        if isinstance(accuracy, Mapping):
            return _as_vector(accuracy.get("values"))
        return _as_vector(accuracy)
    return None


def _positive_probabilities(svc: Any, K_test: Any) -> np.ndarray | None:
    """Return P(weed) if the fitted estimator exposes genuine probabilities."""
    predict_proba = getattr(svc, "predict_proba", None)
    if not callable(predict_proba):
        return None
    try:
        probability = np.asarray(predict_proba(K_test), dtype=float)
    except Exception:  # a predictor that cannot score degrades to N/A, never to fake scores
        return None
    if probability.ndim == 1:
        return probability
    if probability.ndim != 2:
        return None
    if probability.shape[1] == 1:
        return probability[:, 0]
    classes = getattr(svc, "classes_", None)
    if classes is not None:
        classes = np.asarray(classes)
        matches = np.flatnonzero(classes == 1)
        if matches.size:
            return probability[:, int(matches[0])]
    return probability[:, -1]


def evaluate_qsvc(K_test: Any, y_test: Any, svc: Any) -> dict:
    """Evaluate a fitted QSVC/SVC on a precomputed test kernel matrix.

    ``score_pos`` is only read from ``svc.predict_proba``; a hard-label-only
    estimator yields ``probability_available=False`` and a calibration status
    of ``N/A`` rather than synthesised probabilities.  All classification
    metrics come from ``experiments.metrics.binary_metrics``.
    """
    kernel = np.asarray(K_test, dtype=float)
    truth = np.asarray(y_test, dtype=int).ravel()
    predictor = getattr(svc, "predict", None)
    if not callable(predictor):
        raise TypeError("svc must expose a predict() method")
    predicted = np.asarray(predictor(kernel), dtype=int).ravel()
    if predicted.shape != truth.shape:
        raise ValueError(
            f"svc.predict returned {predicted.shape} for {truth.shape} labels"
        )

    probability = _positive_probabilities(svc, kernel)
    from experiments import metrics as _metrics

    result = _metrics.binary_metrics(
        truth, predicted, y_prob=None if probability is None else probability
    )
    payload: dict[str, Any] = {
        "n_samples": int(result.support),
        "support": int(result.support),
        "accuracy": float(result.accuracy),
        "precision": float(result.precision),
        "recall": float(result.recall),
        "f1": float(result.f1),
        "auc": None if result.auc is None else float(result.auc),
        "iou_class0": float(result.iou_class0),
        "iou_class1": float(result.iou_class1),
        "macro_iou": float(result.macro_iou),
        "confusion_matrix": result.confusion.tolist(),
        "probability_available": probability is not None,
        "log_loss": None,
        "calibration": {"status": "N/A", "reason": "no probability output"},
    }
    if probability is not None and probability.shape == truth.shape:
        payload["log_loss"] = _metrics.log_loss(truth, probability)
        calibration = _metrics.calibration_summary(truth, probability)
        payload["calibration"] = {
            "brier_score": calibration["brier_score"],
            "ece": calibration["ece"],
            "n_bins": calibration["n_bins"],
            "n_samples": calibration["n_samples"],
        }
    return payload


def compare_models(results: Mapping[str, Any]) -> dict:
    """Summarise run vectors and run paired t-tests where data permits.

    ``results`` maps a model name to a run vector (a sequence of per-run scores)
    or to a mapping containing one under ``scores``/``values``/``accuracy``.
    Each model is summarised with ``experiments.metrics.summarize_runs``.
    Every unordered model pair is compared only when both vectors have equal
    length ``n >= 2``; for ``n < 2`` (or mismatched lengths) the comparison is
    recorded with a status and a reason and contains **no** test statistic or
    p-value.
    """
    from experiments import metrics as _metrics

    vectors: dict[str, list[float]] = {}
    warnings: list[str] = []
    for name in sorted(results):
        vector = _as_vector(results[name])
        if vector is None:
            warnings.append(f"{name}: no run vector found; skipped")
            continue
        vectors[name] = vector

    summaries = {
        name: _metrics.summarize_runs(name, vector)
        for name, vector in sorted(vectors.items())
    }

    comparisons: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(vectors), 2):
        a, b = vectors[left], vectors[right]
        entry: dict[str, Any] = {"pair": [left, right], "n": min(len(a), len(b))}
        if len(a) != len(b):
            entry.update(
                {
                    "status": "unpaired",
                    "reason": f"unequal run counts ({len(a)} vs {len(b)})",
                }
            )
        elif len(a) < 2:
            entry.update(
                {
                    "status": "insufficient_samples",
                    "reason": "paired comparison requires n >= 2 runs",
                }
            )
        else:
            test = _metrics.paired_t_test(a, b)
            entry.update({"status": "tested", "df": int(len(a) - 1), **test})
        comparisons.append(entry)

    return {
        "summaries": summaries,
        "comparisons": comparisons,
        "warnings": warnings,
    }
