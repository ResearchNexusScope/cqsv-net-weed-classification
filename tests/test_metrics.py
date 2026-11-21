"""Synthetic, deterministic tests for :mod:`experiments.metrics` (QC-I03).

Every fixture is generated in memory from a fixed seed or from explicit small
arrays. No dataset-specific result numbers and no private paper data are used.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments import metrics


def test_iou_from_counts_synthetic():
    """IoU on synthetic counts equals the TP/(TP+FP+FN) formula."""
    assert metrics.iou_from_counts(6, 1, 2) == pytest.approx(6 / 9)
    assert metrics.iou_from_counts(0, 0, 0) == 0.0


def test_binary_metrics_confusion_fixture():
    """Confusion orientation and IoU on the mandated 10-sample fixture."""
    y_true = np.array([1] * 8 + [0] * 2)
    y_pred = np.array([1] * 6 + [0] * 2 + [1] + [0])
    m = metrics.binary_metrics(y_true, y_pred)
    assert m.confusion.tolist() == [[1, 1], [2, 6]]
    assert m.iou_class1 == pytest.approx(6 / 9)


def test_mean_std_uses_ddof1():
    """mean_std returns the sample mean and Bessel-corrected (ddof=1) std."""
    values = [1.0, 2.0, 3.0, 4.0]
    mean, sd = metrics.mean_std(values)
    assert mean == pytest.approx(2.5)
    assert sd == pytest.approx(np.std(values, ddof=1))
    assert sd != pytest.approx(np.std(values, ddof=0))
    assert metrics.mean_std([5.0]) == (5.0, 0.0)


def test_bootstrap_ci_deterministic_fixed_seed():
    """Identical seed and inputs give identical bootstrap CIs (low <= mean <= high)."""
    values = np.array([0.80, 0.85, 0.90, 0.95, 1.00])
    first = metrics.bootstrap_ci(values, n_boot=500, seed=123)
    second = metrics.bootstrap_ci(values, n_boot=500, seed=123)
    assert first == second
    lo, hi = first
    assert lo <= hi
    assert lo <= float(values.mean()) <= hi


def test_paired_t_test_errors_and_returns():
    """ValueError for n<2; p_value and cohens_dz returned for n>=2."""
    with pytest.raises(ValueError):
        metrics.paired_t_test([1.0], [0.5])
    with pytest.raises(ValueError):
        metrics.paired_t_test([1.0, 2.0], [1.0])

    a = np.array([0.90, 0.92, 0.88, 0.95, 0.91])
    b = np.array([0.80, 0.81, 0.79, 0.85, 0.82])
    result = metrics.paired_t_test(a, b)
    assert result["n"] == 5
    assert 0.0 <= result["p_value"] <= 1.0
    expected_dz = float(np.mean(a - b) / np.std(a - b, ddof=1))
    assert result["cohens_dz"] == pytest.approx(expected_dz)


def test_log_loss_known_value():
    """Log loss matches the hand-computed cross-entropy for a two-sample case."""
    y_true = [0, 1]
    expected = -(np.log(0.9) + np.log(0.8)) / 2
    assert metrics.log_loss(y_true, [[0.9, 0.1], [0.2, 0.8]]) == pytest.approx(expected)
    assert metrics.log_loss(y_true, [0.1, 0.8]) == pytest.approx(expected)


def test_grouped_kfold_never_shares_group_across_folds():
    """Each group lands in exactly one test fold, so folds never share a group."""
    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 3))
    y = rng.integers(0, 2, size=n)
    groups = np.repeat(np.arange(12), 5)  # 12 disjoint groups of 5 rows

    def factory():
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="prior")

    result = metrics.grouped_kfold_scores(
        factory, X, y, groups, {"n_splits": 4, "scoring": "accuracy"}
    )

    assert result["n_splits"] == 4
    assert len(result["fold_scores"]) == 4
    assert len(result["fold_test_groups"]) == 4

    seen: set = set()
    for fold_groups in result["fold_test_groups"]:
        current = set(fold_groups)
        assert seen.isdisjoint(current)  # no group appears in two folds
        seen |= current
    assert seen == set(groups.tolist())  # every group is a test group exactly once

    expected_mean, expected_std = metrics.mean_std(result["fold_scores"])
    assert result["mean"] == pytest.approx(expected_mean)
    assert result["std"] == pytest.approx(expected_std)


def test_calibration_summary_shape():
    """Calibration summary has the documented keys, bin count and Brier value."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=50)
    p = rng.random(50)
    out = metrics.calibration_summary(y, p, n_bins=10)

    assert {"brier_score", "ece", "n_bins", "n_samples", "reliability"} <= set(out)
    assert out["n_bins"] == 10
    assert out["n_samples"] == 50
    assert len(out["reliability"]) == 10
    assert sum(b["count"] for b in out["reliability"]) == 50
    assert out["brier_score"] == pytest.approx(float(np.mean((p - y) ** 2)))
    assert 0.0 <= out["ece"] <= 1.0
    for bin_info in out["reliability"]:
        assert set(bin_info) == {
            "bin_lower",
            "bin_upper",
            "count",
            "mean_predicted",
            "mean_observed",
        }
        assert 0.0 <= bin_info["bin_lower"] <= bin_info["bin_upper"] <= 1.0


def test_multipletests_skips_cleanly_without_statsmodels():
    """multipletests returns a serialisable result whether or not statsmodels exists."""
    out = metrics.multipletests([0.01, 0.04, 0.20], alpha=0.05, method="holm")
    assert out["pvals"] == [0.01, 0.04, 0.20]
    assert isinstance(out["available"], bool)
    if out["available"]:
        assert len(out["pvals_corrected"]) == 3
        assert all(isinstance(v, bool) for v in out["reject"])
    else:
        assert out["pvals_corrected"] is None
        assert out["reject"] is None
        assert out["reason"]
