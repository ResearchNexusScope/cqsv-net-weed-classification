"""Correct, reusable classification metrics and run summaries.

The original paper's Table 5 labels precision/recall as IoU. These functions
compute IoU as TP/(TP+FP+FN) and return every number needed to reproduce it.

All public helpers return plain Python scalars, lists and dicts (or numpy
arrays only where the established API requires one, e.g. ``BinaryMetrics.confusion``);
every returned value is JSON-serialisable after at most a ``.tolist()`` call.
No dataset-specific numbers are embedded in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    get_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.model_selection import GroupKFold

# statsmodels is an optional diagnostics dependency (it is not pinned in
# requirements.txt). Import it lazily and degrade gracefully if unavailable.
try:  # pragma: no cover - depends on the installed environment
    from statsmodels.stats.multitest import multipletests as _sm_multipletests

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover - depends on the installed environment
    _sm_multipletests = None
    _HAS_STATSMODELS = False


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float | None
    iou_class0: float
    iou_class1: float
    macro_iou: float
    confusion: np.ndarray
    support: int


def iou_from_counts(tp: int, fp: int, fn: int) -> float:
    """Intersection over union for one class.

    Formula: ``IoU = TP / (TP + FP + FN)`` from integer confusion counts.
    Unitless and in [0, 1]; returns ``0.0`` when the denominator is zero.
    Returns a plain Python ``float``.
    """
    denom = tp + fp + fn
    return float(tp / denom) if denom else 0.0


def binary_metrics(y_true, y_pred, y_prob=None, positive_label: int = 1) -> BinaryMetrics:
    """Compute the binary classification metric suite from hard labels.

    Per-class IoU uses ``TP/(TP+FP+FN)`` (class 0 treats TN as its TP).
    ``accuracy``, ``precision``, ``recall``, ``f1`` and ``auc`` are unitless
    fractions in [0, 1]; ``auc`` is ``None`` when undefined. ``support`` is
    the integer sample count. ``confusion`` is a 2x2 numpy int array with
    layout ``[[TN, FP], [FN, TP]]``; call ``.tolist()`` for a JSON list.
    Returns a :class:`BinaryMetrics` of Python scalars plus that numpy array.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    tn, fp, fn, tp = cm.ravel()
    auc = None
    if y_prob is not None:
        prob = np.asarray(y_prob)
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, prob))

    iou0 = iou_from_counts(tn, fn, fp)  # class 0 treats negatives as positives
    iou1 = iou_from_counts(tp, fp, fn)
    return BinaryMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auc=auc,
        iou_class0=iou0,
        iou_class1=iou1,
        macro_iou=float(np.mean([iou0, iou1])),
        confusion=cm,
        support=int(len(y_true)),
    )


def mean_std(values) -> tuple[float, float]:
    """Sample mean and sample standard deviation (``ddof=1``).

    Formulas: ``mean = sum(x)/n`` and
    ``std = sqrt(sum((x-mean)**2)/(n-1))`` (Bessel-corrected). Units are
    those of the input. Returns two Python ``float`` values; for ``n == 1``
    the standard deviation is ``0.0``.
    """
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95% confidence interval for the mean.

    Resamples ``values`` with replacement ``n_boot`` times from a
    ``numpy`` Generator seeded by ``seed``, takes each resample mean and
    returns the 2.5th and 97.5th percentiles. Units match the input; the
    result is deterministic for a fixed seed. Returns two Python floats
    ``(ci95_low, ci95_high)``.
    """
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_t_test(a, b) -> dict:
    """Two-sided paired t-test plus Cohen's ``dz`` effect size.

    ``t_statistic`` and ``p_value`` come from ``scipy.stats.ttest_rel``.
    Effect size ``dz = mean(a-b) / std(a-b, ddof=1)`` is unitless;
    ``mean_difference`` is in the units of ``a``/``b``. Raises ``ValueError``
    unless the vectors are equal length with ``n >= 2``. Returns a dict of
    Python floats and an integer ``n`` (all JSON-serialisable).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("paired_t_test needs two equal-length vectors with n >= 2")
    diff = a - b
    t_stat, p_value = stats.ttest_rel(a, b)
    dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) else 0.0
    return {
        "mean_difference": float(diff.mean()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_dz": dz,
        "n": int(len(a)),
    }


def summarize_runs(name: str, values, seed: int = 42) -> dict:
    """Summarise repeated run scores without collapsing them to one number.

    Reports ``n`` (int), ``mean`` and ``std`` (sample, ``ddof=1``, in the
    input units) and a percentile-bootstrap 95% CI (``ci95_low``,
    ``ci95_high``, same units). Returns a JSON-serialisable dict of Python
    scalars.
    """
    mean, sd = mean_std(values)
    lo, hi = bootstrap_ci(values, seed=seed)
    return {"model": name, "n": len(values), "mean": mean, "std": sd, "ci95_low": lo, "ci95_high": hi}


def log_loss(y_true, y_prob) -> float:
    """Binary cross-entropy (log loss) in nats per sample.

    Formula: ``-(1/n) * sum_i [y_i*ln(p_i) + (1-y_i)*ln(1-p_i)]``, where
    ``p_i`` is the predicted probability of the positive class. Delegates to
    ``sklearn.metrics.log_loss`` with labels ``{0, 1}``; accepts a 1-D vector
    of positive-class probabilities or a 2-D ``(n, 2)`` probability array.
    Unitless and in [0, inf); lower is better. Returns a plain Python float.
    """
    y_true = np.asarray(y_true)
    prob = np.asarray(y_prob, dtype=float)
    if prob.ndim == 1:
        prob = np.column_stack([1.0 - prob, prob])
    return float(sk_log_loss(y_true, prob, labels=[0, 1]))


def grouped_kfold_scores(estimator_factory, X, y, groups, config) -> dict:
    """Leak-free k-fold scores using ``GroupKFold`` with disjoint groups.

    ``estimator_factory`` is a zero-argument callable returning a fresh
    scikit-learn-style estimator. ``groups`` labels experimental units (for
    example a source photograph); each group is held out in exactly one test
    fold, so no group is ever shared between folds. ``config`` may be a dict
    or an object and supplies ``n_splits`` (default 5) and, optionally,
    ``scoring`` (a sklearn scorer name; default is the estimator's own
    ``score`` method). Every fold fits on the training groups and is scored
    on the held-out test groups.

    Returns a JSON-serialisable dict: ``fold_scores`` (list of floats in the
    score's units), ``mean`` and ``std`` (sample, ``ddof=1``, same units),
    ``n_splits`` (int), and ``fold_test_groups`` (list of lists of group
    labels in each test fold) so a leakage assertion can verify disjointness.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    def _get(name, default):
        if isinstance(config, dict):
            return config.get(name, default)
        return getattr(config, name, default)

    n_splits = int(_get("n_splits", 5))
    scoring = _get("scoring", None)
    scorer = get_scorer(scoring) if scoring else None

    fold_scores: list[float] = []
    fold_test_groups: list[list] = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        estimator = estimator_factory()
        estimator.fit(X[train_idx], y[train_idx])
        score = scorer(estimator, X[test_idx], y[test_idx]) if scorer else estimator.score(X[test_idx], y[test_idx])
        fold_scores.append(float(score))
        fold_test_groups.append(
            [g.item() if hasattr(g, "item") else g for g in groups[test_idx]]
        )

    mean, std = mean_std(fold_scores)
    return {
        "fold_scores": fold_scores,
        "mean": mean,
        "std": std,
        "n_splits": n_splits,
        "fold_test_groups": fold_test_groups,
    }


def multipletests(pvals, alpha: float = 0.05, method: str = "holm") -> dict:
    """Convenience wrapper for statsmodels multiple-comparison correction.

    ``pvals`` is a sequence of raw p-values (unitless, in [0, 1]), ``alpha``
    the family-wise/FDR level and ``method`` any method accepted by
    ``statsmodels.stats.multitest.multipletests`` (e.g. ``"holm"``,
    ``"bonferroni"``, ``"fdr_bh"``). When statsmodels is importable the call
    is delegated to it and the result returned as Python lists of bools and
    floats. When it is not installed the function *skips cleanly*: it never
    raises and instead returns ``available=False`` with ``reject`` and
    ``pvals_corrected`` set to ``None`` plus a ``reason`` string. All fields
    are JSON-serialisable.
    """
    p = [float(v) for v in pvals]
    result = {
        "available": bool(_HAS_STATSMODELS),
        "method": str(method),
        "alpha": float(alpha),
        "pvals": p,
        "reject": None,
        "pvals_corrected": None,
        "reason": None,
    }
    if not _HAS_STATSMODELS:
        result["reason"] = "statsmodels is not installed; correction skipped"
        return result
    reject, pvals_corrected, _, _ = _sm_multipletests(p, alpha=alpha, method=method)
    result["reject"] = [bool(v) for v in reject]
    result["pvals_corrected"] = [float(v) for v in pvals_corrected]
    return result


def calibration_summary(y_true, y_prob, n_bins: int = 10) -> dict:
    """Brier score and equal-width reliability bins for binary probabilities.

    ``p`` is the predicted probability of the positive class (a 1-D vector,
    or the second column of a 2-D ``(n, 2)`` array). Brier score =
    ``mean((p - y)**2)`` in squared-probability units, in [0, 1] (lower is
    better). Expected calibration error (ECE) =
    ``sum_b (n_b/n) * |mean(y in b) - mean(p in b)|`` over ``n_bins``
    equal-width bins on [0, 1], in probability units. Each reliability bin
    carries its bounds, integer count, mean predicted probability and mean
    observed fraction of positives (``None`` for an empty bin). Uses numpy
    only, so it adds no hard dependency. Returns a JSON-serialisable dict.
    """
    y = np.asarray(y_true).astype(int).ravel()
    prob = np.asarray(y_prob, dtype=float)
    p = prob[:, 1].astype(float) if prob.ndim == 2 else prob.astype(float)
    p = p.ravel()
    if p.shape[0] != y.shape[0]:
        raise ValueError("y_true and y_prob must have the same number of samples")
    n = int(y.shape[0])

    brier = float(np.mean((p - y) ** 2)) if n else 0.0

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    reliability = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (p >= lo) & (p < hi)
        if hi == edges[-1]:  # include the exact upper endpoint in the last bin
            in_bin = in_bin | (p == 1.0)
        count = int(in_bin.sum())
        if count:
            mean_pred = float(p[in_bin].mean())
            mean_obs = float(y[in_bin].mean())
            ece += (count / n) * abs(mean_obs - mean_pred)
        else:
            mean_pred = None
            mean_obs = None
        reliability.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "count": count,
                "mean_predicted": mean_pred,
                "mean_observed": mean_obs,
            }
        )
    return {
        "brier_score": brier,
        "ece": float(ece),
        "n_bins": int(n_bins),
        "n_samples": n,
        "reliability": reliability,
    }


def _self_test() -> None:
    """Run the built-in synthetic smoke test for the legacy metric helpers.

    Uses the fixed 10-sample confusion fixture (6 TP, 1 FP, 2 FN, 1 TN);
    prints a one-line pass message and returns ``None``.
    """
    # Synthetic 10-sample case: 6 TP, 1 FP, 2 FN, 1 TN.
    y_true = np.array([1] * 8 + [0] * 2)
    y_pred = np.array([1] * 6 + [0] * 2 + [1] + [0])
    m = binary_metrics(y_true, y_pred)
    assert m.confusion.tolist() == [[1, 1], [2, 6]], m.confusion
    assert abs(m.iou_class1 - 6 / 9) < 1e-9, m.iou_class1
    assert abs(m.iou_class0 - 1 / 4) < 1e-9, m.iou_class0
    assert summarize_runs("demo", [0.8, 0.9, 1.0])["mean"] == 0.9
    print("metrics self-test passed:", m.confusion.tolist(), round(m.macro_iou, 4))


if __name__ == "__main__":
    _self_test()
