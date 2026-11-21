"""Synthetic tests for run-vector statistics (QC-D04 §5).

The fixtures are small generated vectors; no legacy paper p-value, result table
or dataset count is used.  Assertions cover n/vector/mean/std(ddof=1)/bootstrap
CI, paired t-test sign and dz, Holm correction, decision rule, and effect
sizes.  Optional statsmodels power is accepted either as a probability or as
``None`` when the dependency is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import metrics, stats  # noqa: E402


def test_summarize_has_full_run_vector_and_ddof1_std() -> None:
    values = [0.80, 0.86, 0.91, 0.88, 0.95]
    result = stats.summarize(values, "toy", seed=17, n_boot=300)
    assert result["model"] == "toy"
    assert result["n"] == len(values)
    assert result["values"] == values
    assert result["mean"] == pytest.approx(float(np.mean(values)))
    assert result["std"] == pytest.approx(float(np.std(values, ddof=1)))
    assert result["std"] != pytest.approx(float(np.std(values, ddof=0)))
    assert result["ci95_low"] <= result["mean"] <= result["ci95_high"]
    assert result["boot_B"] == 300 and result["boot_seed"] == 17
    assert result["unit"] == "fraction"


def test_summarize_is_deterministic_for_a_fixed_seed() -> None:
    values = [0.71, 0.74, 0.68, 0.80, 0.79, 0.83]
    first = stats.summarize(values, "toy", seed=5, n_boot=200)
    second = stats.summarize(values, "toy", seed=5, n_boot=200)
    assert first == second
    assert stats.summarize(values, "toy", seed=6, n_boot=200)["ci95_low"] != first["ci95_low"]


def test_paired_comparison_fields_and_effect_size() -> None:
    rng = np.random.default_rng(3)
    baseline = rng.uniform(0.60, 0.80, size=10)
    improved = baseline + rng.uniform(0.02, 0.08, size=10)
    comparison = stats.paired_comparison(
        ("improved", "baseline"), improved, baseline, bootstrap_seed=9, n_boot=200
    )
    expected = metrics.paired_t_test(improved, baseline)
    expected_ci = metrics.bootstrap_ci(improved - baseline, n_boot=200, seed=9)
    assert comparison["n"] == len(baseline)
    assert comparison["df"] == len(baseline) - 1
    assert comparison["t_statistic"] == pytest.approx(expected["t_statistic"])
    assert comparison["p_raw"] == pytest.approx(expected["p_value"])
    assert comparison["mean_difference"] == pytest.approx(expected["mean_difference"])
    assert comparison["cohens_dz"] == pytest.approx(expected["cohens_dz"])
    assert (comparison["ci95_low"], comparison["ci95_high"]) == pytest.approx(expected_ci)
    assert comparison["p_holm"] is None  # assigned by compare_family
    assert comparison["decision"] == "inconclusive"
    power = comparison["power_at_observed_dz"]
    assert power is None or 0.0 <= power <= 1.0


def test_holm_family_assigns_adjusted_p_and_decision_rule() -> None:
    rng = np.random.default_rng(21)
    pairs = []
    for index in range(3):
        first = rng.uniform(0.55, 0.75, size=10)
        second = first + rng.uniform(0.03, 0.10, size=10)
        pairs.append(
            stats.paired_comparison(
                (f"a{index}", f"b{index}"), first, second, n_boot=200, bootstrap_seed=31
            )
        )
    family = stats.compare_family(pairs)
    assert len(family) == 3
    for comparison in family:
        assert comparison["p_holm"] is not None
        assert comparison["p_holm"] >= comparison["p_raw"]
        assert comparison["multiplicity"]["method"] == "holm"
        assert comparison["multiplicity"]["m"] == 3
        assert comparison["decision"] in ("significant", "inconclusive")
        if comparison["decision"] == "significant":
            power = comparison["power_at_observed_dz"]
            assert comparison["p_holm"] < 0.05
            assert comparison["ci95_low"] > 0 or comparison["ci95_high"] < 0
            assert power is not None and power >= 0.80
        else:
            assert comparison["decision_reason"]


def test_mcnemar_table_counts_discordant_cells() -> None:
    truth = np.array([1, 1, 0, 0, 1, 0])
    pred_a = np.array([1, 0, 0, 0, 1, 1])
    pred_b = np.array([1, 1, 1, 0, 1, 0])
    table = stats.mcnemar_table(truth, pred_a, pred_b)
    assert table.shape == (2, 2)
    assert table[0, 1] == 2  # b correct only (samples 1 and 5)
    assert table[1, 0] == 1  # a correct only (sample 2)
    assert int(table.sum()) == len(truth)
    with pytest.raises(ValueError):
        stats.mcnemar_table(truth, pred_a, pred_b[:2])


def test_paired_delta_summary_tracks_per_image_degradation() -> None:
    clean = [1, 1, 0, 1, 1, 0]
    noisy = [0, 1, 0, 0, 1, 0]
    delta = stats.paired_delta_summary(clean, noisy, seed=4, n_boot=100)
    assert delta["n"] == len(clean)
    assert delta["mean"] == pytest.approx(2 / len(clean))
    assert delta["ci95_low"] <= delta["mean"] <= delta["ci95_high"]
    with pytest.raises(ValueError):
        stats.paired_delta_summary(clean, noisy[:2])


def test_round_floats_is_stable_and_json_ready() -> None:
    payload = {"a": (0.1234567891234, np.float64(1.0)), "b": np.int64(7), "c": None}
    rounded = stats.round_floats(payload, digits=3)
    assert rounded == {"a": [0.123, 1.0], "b": 7, "c": None}
    assert stats.ROUND_DIGITS > 0
