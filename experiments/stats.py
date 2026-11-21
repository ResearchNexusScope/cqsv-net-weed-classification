"""Run-vector statistics built exclusively on :mod:`experiments.metrics` helpers.

Every summary carries n, the full run vector, mean, sample standard deviation
(ddof=1), a seeded percentile-bootstrap 95% CI and the bootstrap parameters.
Comparisons use the pre-registered paired t-test, report df = n - 1, the exact
p-value, effect size dz and a power estimate when statsmodels is installed;
Holm correction is applied to the whole pre-registered family in one call.
Legacy p-values from the archived paper are never read or reproduced here.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from experiments import metrics as _metrics

__all__ = [
    "ROUND_DIGITS",
    "round_floats",
    "summarize",
    "power_at_observed_dz",
    "mcnemar_table",
    "paired_comparison",
    "compare_family",
    "paired_delta_summary",
]

# Fixed rounding makes a seeded fixture produce byte-identical metrics.json.
ROUND_DIGITS = 10


def round_floats(value: Any, digits: int = ROUND_DIGITS) -> Any:
    """Recursively round floats and normalise numpy scalars for JSON output."""
    if isinstance(value, dict):
        return {str(key): round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(item, digits) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return round(number, digits) if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return round_floats(value.tolist(), digits)
    return value


def summarize(
    values: Sequence[float],
    name: str,
    *,
    seed: int = 42,
    n_boot: int = 2000,
    unit: str = "fraction",
) -> dict:
    """Summarise a run vector with n/vector/mean/std(ddof=1)/bootstrap CI.

    Delegates to ``experiments.metrics.summarize_runs`` for the core fields and
    re-derives mean, std and CI through ``mean_std`` / ``bootstrap_ci`` so the
    recorded ``boot_B`` and ``boot_seed`` always match the interval actually
    reported.
    """
    vector = [float(value) for value in values]
    if not vector:
        raise ValueError("summarize requires a non-empty run vector")
    base = _metrics.summarize_runs(name, vector, seed=int(seed))
    mean, std = _metrics.mean_std(vector)
    low, high = _metrics.bootstrap_ci(vector, n_boot=int(n_boot), seed=int(seed))
    base.update(
        {
            "values": vector,
            "n": len(vector),
            "mean": mean,
            "std": std,
            "ci95_low": low,
            "ci95_high": high,
            "boot_B": int(n_boot),
            "boot_seed": int(seed),
            "unit": unit,
        }
    )
    return base


def power_at_observed_dz(dz: float | None, n: int, alpha: float = 0.05) -> float | None:
    """Two-sided paired t-test power at the observed dz, or ``None`` if absent.

    ``statsmodels.stats.power.TTestPower`` is the only source of this number.
    When it is not installed the function returns ``None`` so that the caller
    can mark a significance decision inconclusive rather than invent power.
    """
    if dz is None or int(n) < 2:
        return None
    try:
        from statsmodels.stats.power import TTestPower
    except ImportError:
        return None
    try:
        value = float(
            TTestPower().power(
                effect_size=abs(float(dz)),
                nobs=int(n),
                alpha=float(alpha),
                alternative="two-sided",
            )
        )
    except Exception:  # pragma: no cover - depends on statsmodels version
        return None
    # statsmodels can return NaN for degenerate effect sizes; NaN would make
    # both JSON output and the decision rule ambiguous, so treat it as absent.
    return value if np.isfinite(value) else None


def mcnemar_table(
    y_true: Sequence[int], predictions_a: Sequence[int], predictions_b: Sequence[int]
) -> np.ndarray:
    """2x2 paired table ``[[both_correct, b_only], [a_only, both_wrong]]``.

    The off-diagonal cells are the discordant counts consumed by the exact
    McNemar test; ``n_discordant`` is their sum.
    """
    truth = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(predictions_a, dtype=int)
    pred_b = np.asarray(predictions_b, dtype=int)
    if not (len(truth) == len(pred_a) == len(pred_b)):
        raise ValueError("McNemar inputs must have equal length")
    a_correct = pred_a == truth
    b_correct = pred_b == truth
    both_correct = int(np.sum(a_correct & b_correct))
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(b_correct & ~a_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))
    return np.asarray([[both_correct, b_only], [a_only, both_wrong]], dtype=int)


def paired_comparison(
    pair: Sequence[str],
    a: Sequence[float],
    b: Sequence[float],
    *,
    bootstrap_seed: int = 42,
    n_boot: int = 2000,
    alpha: float = 0.05,
    mcnemar_table_data: Any | None = None,
    wilcoxon_differences: Sequence[float] | None = None,
) -> dict:
    """One pre-registered paired comparison (Holm applied later by the family).

    The primary test is ``experiments.metrics.paired_t_test`` (which wraps
    ``scipy.stats.ttest_rel`` two-sided).  Its CI is the bootstrap CI of the
    paired difference vector, not of either arm.  ``decision`` starts as
    ``"inconclusive"`` and is only upgraded by :func:`compare_family`.
    """
    result = _metrics.paired_t_test(a, b)
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    low, high = _metrics.bootstrap_ci(
        difference, n_boot=int(n_boot), seed=int(bootstrap_seed)
    )

    secondary: dict[str, dict] = {}
    if mcnemar_table_data is not None:
        table = np.asarray(mcnemar_table_data, dtype=int)
        p_value: float | None = None
        try:
            from scipy.stats import mcnemar

            p_value = float(mcnemar(table, exact=True).pvalue)
        except Exception:  # pragma: no cover - scipy/table dependent
            p_value = None
        secondary["mcnemar_exact"] = {
            "p": p_value,
            "n_discordant": int(table[0, 1] + table[1, 0]),
        }
    if wilcoxon_differences is not None:
        differences = np.asarray(wilcoxon_differences, dtype=float)
        p_value = None
        try:
            from scipy.stats import wilcoxon

            p_value = float(wilcoxon(differences, alternative="two-sided").pvalue)
        except Exception:  # e.g. all-zero differences; report as unavailable
            p_value = None
        secondary["wilcoxon_signed_rank"] = {"p": p_value}

    n = int(result["n"])
    dz = float(result["cohens_dz"])
    return {
        "pair": [str(name) for name in pair],
        "primary_test": "paired_t_test (scipy.stats.ttest_rel, two-sided)",
        "n": n,
        "t_statistic": float(result["t_statistic"]),
        "df": n - 1,
        "p_raw": float(result["p_value"]),
        "p_holm": None,
        "mean_difference": float(result["mean_difference"]),
        "ci95_low": low,
        "ci95_high": high,
        "cohens_dz": dz,
        "power_at_observed_dz": power_at_observed_dz(dz, n, alpha=alpha),
        "secondary_tests": secondary,
        "decision": "inconclusive",
        "_ci_excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def compare_family(comparisons: list[dict], *, alpha: float = 0.05) -> list[dict]:
    """Apply Holm-Bonferroni once to the whole family and set decisions.

    A difference is called significant only when all three QC-D04 §2.4
    conditions hold: ``p_holm < alpha``, the bootstrap CI of the difference
    excludes zero, and power at the observed dz is at least 0.80.  Otherwise
    the decision stays ``"inconclusive"`` with an explicit reason.
    """
    if not comparisons:
        return comparisons
    raw_p = [float(item.get("p_raw", 1.0)) for item in comparisons]
    corrected = _metrics.multipletests(raw_p, alpha=alpha, method="holm")
    adjusted: Sequence[float | None] = corrected["pvals_corrected"] or [None] * len(
        comparisons
    )
    for index, item in enumerate(comparisons):
        p_holm = adjusted[index]
        item["p_holm"] = None if p_holm is None else float(p_holm)
        item["multiplicity"] = {
            "method": "holm",
            "m": len(comparisons),
            "available": bool(corrected["available"]),
            "reason": corrected["reason"],
        }
        ci_excludes_zero = bool(item.pop("_ci_excludes_zero", False))
        power = item.get("power_at_observed_dz")
        power_ok = power is not None and float(power) >= 0.80
        p_ok = item["p_holm"] is not None and float(item["p_holm"]) < alpha
        if p_ok and ci_excludes_zero and power_ok:
            item["decision"] = "significant"
            item["decision_reason"] = (
                "p_holm<alpha, bootstrap CI excludes 0, power>=0.80"
            )
        else:
            failing = []
            if not p_ok:
                failing.append("p_holm")
            if not ci_excludes_zero:
                failing.append("bootstrap CI includes 0")
            if not power_ok:
                failing.append("power<0.80 or unavailable")
            item["decision"] = "inconclusive"
            item["decision_reason"] = "descriptive: " + ", ".join(failing)
    return comparisons


def paired_delta_summary(
    clean: Sequence[int | float],
    corrupted: Sequence[int | float],
    *,
    name: str = "delta",
    seed: int = 42,
    n_boot: int = 2000,
) -> dict:
    """Paired per-image degradation ``clean - corrupted`` with a bootstrap CI."""
    first = np.asarray(clean, dtype=float)
    second = np.asarray(corrupted, dtype=float)
    if first.shape != second.shape:
        raise ValueError("paired_delta_summary needs equally shaped vectors")
    return summarize(
        first - second, name=name, seed=seed, n_boot=n_boot, unit="fraction"
    )
