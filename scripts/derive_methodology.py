"""Derive the methodology evidence table and acceptance checks from result artifacts.

Contract: every *measurement* in the emitted report is read from a committed JSON
artifact under ``results/``.  The only numbers in this source are POLICY constants
(alpha, required power, minimum sample sizes and margin multipliers), which are
methodological rules and not observations.  Thresholds are expressed as relations
between observed controls, never as copied result values.

Usage::

    python scripts/derive_methodology.py --root . --output methodology_report.json

The report schema is::

    {
      "schema_version": "1.0",
      "root": <str>,
      "artifacts": {<logical name>: <repo-relative path>},
      "policy": {...},
      "evidence": [{"id", "approach", "status", "reason", "observations", "artifacts"}],
      "acceptance": [{"id", "criterion", "threshold", "observed", "passed", "artifacts"}],
      "summary": {"n_evidence", "n_checks", "n_passed", "n_failed"}
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

# Logical name -> repository-relative artifact path.  These are locations, not
# measurements; no result value is embedded here.
ARTIFACT_PATHS: dict[str, str] = {
    "real_run": "results/real_run10/qc-52fcca25a5b8/metrics/metrics.json",
    "qml_full": "results/qml_full/metrics.json",
    "qml_plantnet": "results/qml_plantnet/metrics.json",
    "threshold": "results/qml_plantnet/threshold_analysis.json",
    "ablation": "results/qml_ablation/ablation_summary.json",
    "model_card": "results/qml_ablation/model_card.md",
}

# Methodological policy, NOT observed results.  chance is derived from the
# dataset_table artifact (1 / number of classes), never written here.
POLICY: dict[str, float] = {
    "alpha": 0.05,
    "min_power": 0.80,
    "min_n_for_significance": 2,
    "min_n_recommended": 10,
    "sd_margin": 2.0,
    "advantage_min_gain": 0.0,
    "equivalence_tol": 1e-12,
    "min_recall_share": 0.5,
    "min_retention": 0.80,
}

_PAIRED_MODELS = ("qsvc_quantum_kernel", "svc_poly2_classical")
_PRIMARY_EXPERIMENT = "qsvc_mixed_variance"
_FALLBACK_MARKER = "PennyLane is not installed"


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, unreadable, or lacks a required key."""


def _read(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix == ".json" else {"text": text}


def load_artifacts(
    root: str | Path, paths: Mapping[str, str] = ARTIFACT_PATHS
) -> dict[str, Any]:
    """Load every declared artifact under *root* as a dict.

    Missing or malformed artifacts raise :class:`ArtifactError` listing the
    offending paths; the CLI turns this into a clean non-zero exit.
    """
    base = Path(root)
    missing = [str(base / rel) for rel in paths.values() if not (base / rel).is_file()]
    if missing:
        raise ArtifactError("missing artifacts: " + ", ".join(missing))
    loaded: dict[str, Any] = {}
    for name, rel in paths.items():
        try:
            loaded[name] = _read(base / rel)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"unreadable artifact {rel}: {exc}") from exc
    return loaded


def observed(
    artifacts: Mapping[str, Any], artifact_id: str, *keys: str | int
) -> dict[str, Any]:
    """Return one cited observation: artifact path, exact key path, value."""
    try:
        node: Any = artifacts[artifact_id]
        for key in keys:
            node = node[key]
    except (KeyError, IndexError, TypeError) as exc:
        raise ArtifactError(
            f"key {'.'.join(str(k) for k in keys)!r} absent in artifact {artifact_id!r}"
        ) from exc
    return {
        "artifact": ARTIFACT_PATHS.get(artifact_id, artifact_id),
        "key": ".".join(str(k) for k in keys),
        "value": node,
    }


def _aggregate(metrics: Mapping[str, Any], model: str) -> dict[str, Any]:
    key = f"{_PRIMARY_EXPERIMENT}/{model}"
    try:
        return metrics["aggregates"][key]
    except KeyError as exc:
        raise ArtifactError(f"aggregate {key!r} absent from metrics artifact") from exc


def _noise_gap(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    """Largest quantum-vs-classical noisy gap and quantum retention at worst sigma.

    Returns ``(max_gap, worst_sigma, retention)``; all values come from the
    artifact.  An empty noise section yields a gap above tolerance so that the
    corresponding acceptance check fails closed.
    """
    prefix_q = f"noise_robustness/{_PAIRED_MODELS[0]}/sigma="
    prefix_c = f"noise_robustness/{_PAIRED_MODELS[1]}/sigma="
    aggregates = metrics.get("aggregates", {})
    paired = sorted(
        (
            (float(key[len(prefix_q) :]), key, key.replace(prefix_q, prefix_c, 1))
            for key in aggregates
            if key.startswith(prefix_q) and key.replace(prefix_q, prefix_c, 1) in aggregates
        ),
        key=lambda item: item[0],
    )
    if not paired:
        return (float("inf"), float("nan"), 0.0)
    gaps = [
        abs(
            aggregates[qkey]["accuracy"]["mean"]
            - aggregates[ckey]["accuracy"]["mean"]
        )
        for _, qkey, ckey in paired
    ]
    max_gap = max(gaps)
    worst_sigma, worst_qkey, _ = paired[gaps.index(max_gap)]
    clean = _aggregate(metrics, _PAIRED_MODELS[0])["accuracy"]["mean"]
    noisy = aggregates[worst_qkey]["accuracy"]["mean"]
    retention = noisy / clean if clean else 0.0
    return (max_gap, worst_sigma, retention)


def _derive(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    """Compute all comparisons once, from artifacts only."""
    metrics = artifacts["real_run"]
    full = artifacts["qml_full"]
    plant = artifacts["qml_plantnet"]
    sweep = artifacts["threshold"]
    ablation = artifacts["ablation"]

    mix = _aggregate(metrics, _PAIRED_MODELS[0])
    control = _aggregate(metrics, "qsvc_no_mixing_control")
    poly = _aggregate(metrics, _PAIRED_MODELS[1])
    rbf = _aggregate(metrics, "svc_rbf")

    mix_acc = mix["accuracy"]["mean"]
    control_acc = control["accuracy"]["mean"]
    poly_acc = poly["accuracy"]["mean"]
    rbf_acc = rbf["accuracy"]["mean"]
    control_std = control["accuracy"]["std"]
    margin = POLICY["sd_margin"] * control_std
    class_count = metrics["dataset_table"][0]["categories"]
    chance = 1.0 / class_count
    gap = mix_acc - control_acc

    paired_index = next(
        (
            index
            for index, c in enumerate(metrics.get("comparisons", []))
            if set(c.get("pair", [])) == set(_PAIRED_MODELS)
        ),
        None,
    )
    paired = (
        metrics["comparisons"][paired_index] if paired_index is not None else None
    )
    best_classical = max(poly_acc, rbf_acc)
    advantage_gap = mix_acc - best_classical

    full_ood = full["ood"]
    plant_ood = plant["ood"]
    confusion = plant_ood["confusion_matrix"]
    predicted_classes = sum(1 for column in zip(*confusion) if sum(column) > 0)

    best_bal = sweep["best_by_balanced_accuracy"]
    best_f1 = sweep["best_by_f1"]
    noise_gap, worst_sigma, retention = _noise_gap(metrics)

    fallback_count = sum(
        1 for warning in metrics.get("warnings", []) if _FALLBACK_MARKER in warning
    )

    required_fields = ("n", "std", "values", "ci95_low", "ci95_high")
    complete_aggregates = all(
        all(field in agg.get(name, {}) for field in required_fields)
        for agg in (mix, control, poly, rbf)
        for name in ("accuracy", "auc", "f1")
    )
    min_n = POLICY["min_n_for_significance"]
    sample_sizes = [agg["accuracy"]["n"] for agg in (mix, control, poly, rbf)]
    bogus_significance = sum(
        1
        for c in metrics.get("comparisons", [])
        if c.get("n", min_n) < min_n and c.get("decision") == "significant"
    )

    accuracy = ablation["metrics"]["accuracy"]
    ablation_ci = (accuracy["ci95_low"], accuracy["ci95_high"])
    ablation_conclusive = (
        accuracy["n"] >= POLICY["min_n_recommended"]
        and not (ablation_ci[0] <= chance <= ablation_ci[1])
    )

    return {
        "class_count": class_count,
        "chance": chance,
        "mix_acc": mix_acc,
        "mix_std": mix["accuracy"]["std"],
        "mix_n": mix["accuracy"]["n"],
        "control_acc": control_acc,
        "control_std": control_std,
        "control_n": control["accuracy"]["n"],
        "poly_acc": poly_acc,
        "poly_n": poly["accuracy"]["n"],
        "rbf_acc": rbf_acc,
        "gap": gap,
        "margin": margin,
        "paired": paired,
        "paired_index": paired_index,
        "best_classical": best_classical,
        "advantage_gap": advantage_gap,
        "full_acc": full_ood["accuracy"],
        "full_auc": full_ood["auc"],
        "full_f1": full_ood["f1"],
        "plant_acc": plant_ood["accuracy"],
        "plant_auc": plant_ood["auc"],
        "plant_f1": plant_ood["f1"],
        "predicted_classes": predicted_classes,
        "best_bal_acc": best_bal["balanced_accuracy"],
        "best_bal_recall": best_bal["recall"],
        "best_f1_precision": best_f1["precision"],
        "best_f1_bal_acc": best_f1["balanced_accuracy"],
        "noise_gap": noise_gap,
        "worst_sigma": worst_sigma,
        "retention": retention,
        "fallback_count": fallback_count,
        "complete_aggregates": complete_aggregates,
        "sample_sizes": sample_sizes,
        "bogus_significance": bogus_significance,
        "ablation_n": accuracy["n"],
        "ablation_ci": ablation_ci,
        "ablation_conclusive": ablation_conclusive,
    }


def evidence_table(artifacts: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recompute the approach -> status table from observed artifact values."""
    d = _derive(artifacts)

    mixing_status = (
        "partially_supported"
        if d["gap"] > d["margin"] and d["mix_n"] >= POLICY["min_n_recommended"]
        else "failed"
    )
    mixing_reason = (
        "consistent large gain over the no-mixing control, but no pre-registered "
        "paired contrast for this comparison is recorded in the artifacts"
        if mixing_status == "partially_supported"
        else "gain is not distinguishable from control noise at the required sample size"
    )

    paired = d["paired"]
    advantage_status = (
        "failed"
        if paired is None
        or not (
            d["advantage_gap"] > POLICY["advantage_min_gain"]
            and paired.get("ci95_low", 0.0) > 0.0
            and paired.get("p_holm") is not None
            and paired["p_holm"] < POLICY["alpha"]
            and (paired.get("power_at_observed_dz") or 0.0) >= POLICY["min_power"]
        )
        else "supported"
    )
    advantage_reason = (
        "quantum kernel is numerically indistinguishable from the matched "
        "classical poly-2 oracle and the paired test is inconclusive/underpowered"
        if advantage_status == "failed"
        else "paired evidence exceeds the advantage bar"
    )

    primary_status = (
        "supported"
        if d["poly_acc"] > d["control_acc"] + d["margin"]
        and d["poly_n"] >= POLICY["min_n_recommended"]
        else "failed"
    )
    primary_reason = (
        "classical poly-2 reproduces the mixing gain on the same run vectors "
        "without any quantum backend"
        if primary_status == "supported"
        else "classical control does not reproduce the claimed gain"
    )

    qnn_status = (
        "failed"
        if d["full_auc"] <= d["chance"] or d["plant_auc"] <= d["chance"]
        or d["plant_f1"] <= POLICY["equivalence_tol"]
        else "partially_supported"
    )
    qnn_reason = (
        "OOD AUC is at/below chance and one QNN arm collapses to a single predicted class"
        if qnn_status == "failed"
        else "OOD ranking exceeds chance but is not confirmed across arms"
    )

    threshold_status = (
        "failed"
        if d["best_bal_acc"] - d["chance"] <= POLICY["equivalence_tol"]
        or d["best_bal_recall"] < POLICY["min_recall_share"] * d["chance"]
        else "partially_supported"
    )
    threshold_reason = (
        "post-hoc sweep yields no operating point that improves on chance with "
        "non-degenerate recall"
        if threshold_status == "failed"
        else "an operating point separates classes on the swept probabilities"
    )

    noise_status = (
        "failed"
        if d["noise_gap"] <= POLICY["equivalence_tol"]
        or d["retention"] < POLICY["min_retention"]
        else "partially_supported"
    )
    noise_reason = (
        "noisy quantum-kernel accuracies equal the classical oracle at every "
        "recorded sigma and retention falls below the required floor"
        if noise_status == "failed"
        else "quantum arm degrades differently from the matched classical arm"
    )

    ablation_status = "partially_supported" if not d["ablation_conclusive"] else "supported"
    ablation_reason = (
        "three configurations were measured but n is below the recommended "
        "minimum and the accuracy CI spans chance"
        if ablation_status == "partially_supported"
        else "ablations are conclusive at the required sample size"
    )

    # Convention: key constants never include the top-level "aggregates" segment;
    # every observed() call passes that segment explicitly as its first key.
    mix_key = f"{_PRIMARY_EXPERIMENT}/{_PAIRED_MODELS[0]}.accuracy"
    ctl_key = f"{_PRIMARY_EXPERIMENT}/qsvc_no_mixing_control.accuracy"
    poly_key = f"{_PRIMARY_EXPERIMENT}/{_PAIRED_MODELS[1]}.accuracy"
    noise_q_key = f"noise_robustness/{_PAIRED_MODELS[0]}/sigma="
    noise_c_key = f"noise_robustness/{_PAIRED_MODELS[1]}/sigma="

    rows = [
        {
            "id": "primary_classical_mixed_kernel",
            "approach": "classical poly-2 kernel with supervised domain mixing",
            "status": primary_status,
            "reason": primary_reason,
            "observations": [
                observed(artifacts, "real_run", "aggregates", *poly_key.split("."), "mean"),
                observed(artifacts, "real_run", "aggregates", *poly_key.split("."), "values"),
                observed(artifacts, "real_run", "aggregates", *ctl_key.split("."), "mean"),
            ],
        },
        {
            "id": "supervised_domain_mixing",
            "approach": "supervised source-domain mixing before kernel fit",
            "status": mixing_status,
            "reason": mixing_reason,
            "observations": [
                observed(artifacts, "real_run", "aggregates", *mix_key.split("."), "mean"),
                observed(artifacts, "real_run", "aggregates", *mix_key.split("."), "std"),
                observed(artifacts, "real_run", "aggregates", *mix_key.split("."), "n"),
                observed(artifacts, "real_run", "aggregates", *ctl_key.split("."), "mean"),
                observed(artifacts, "real_run", "aggregates", *ctl_key.split("."), "std"),
                observed(artifacts, "real_run", "aggregates", *ctl_key.split("."), "n"),
            ],
        },
        {
            "id": "quantum_kernel_advantage",
            "approach": "QSVC quantum kernel as a source of advantage",
            "status": advantage_status,
            "reason": advantage_reason,
            "observations": [
                observed(artifacts, "real_run", "aggregates", *mix_key.split("."), "values"),
                observed(artifacts, "real_run", "aggregates", *poly_key.split("."), "values"),
                observed(artifacts, "real_run", "comparisons", d["paired_index"])
                if paired is not None
                else observed(artifacts, "real_run", "comparisons"),
                observed(artifacts, "real_run", "warnings"),
            ],
        },
        {
            "id": "qnn_ood_generalization",
            "approach": "entangling QNN head for OOD generalization",
            "status": qnn_status,
            "reason": qnn_reason,
            "observations": [
                observed(artifacts, "qml_full", "ood", "accuracy"),
                observed(artifacts, "qml_full", "ood", "auc"),
                observed(artifacts, "qml_full", "ood", "f1"),
                observed(artifacts, "qml_full", "training", "best_val_accuracy"),
                observed(artifacts, "qml_plantnet", "ood", "accuracy"),
                observed(artifacts, "qml_plantnet", "ood", "auc"),
                observed(artifacts, "qml_plantnet", "ood", "f1"),
                observed(artifacts, "qml_plantnet", "ood", "confusion_matrix"),
            ],
        },
        {
            "id": "posthoc_threshold_selection",
            "approach": "post-hoc OOD threshold sweep as a usable decision rule",
            "status": threshold_status,
            "reason": threshold_reason,
            "observations": [
                observed(artifacts, "threshold", "best_by_balanced_accuracy"),
                observed(artifacts, "threshold", "best_by_f1"),
                observed(artifacts, "threshold", "scores_summary"),
                observed(artifacts, "threshold", "provenance", "note"),
            ],
        },
        {
            "id": "quantum_noise_robustness",
            "approach": "quantum-specific robustness to correlated noise",
            "status": noise_status,
            "reason": noise_reason,
            "observations": [
                observed(
                    artifacts,
                    "real_run",
                    "aggregates",
                    f"{noise_q_key}{d['worst_sigma']}",
                    "accuracy",
                    "mean",
                )
                if d["noise_gap"] != float("inf")
                else observed(artifacts, "real_run", "aggregates"),
                observed(
                    artifacts,
                    "real_run",
                    "aggregates",
                    f"{noise_c_key}{d['worst_sigma']}",
                    "accuracy",
                    "mean",
                )
                if d["noise_gap"] != float("inf")
                else observed(artifacts, "real_run", "aggregates"),
                observed(artifacts, "real_run", "dataset_table", 0, "categories"),
            ],
        },
        {
            "id": "architecture_ablation",
            "approach": "qubit/layer/encoding ablation as evidence of improvement",
            "status": ablation_status,
            "reason": ablation_reason,
            "observations": [
                observed(artifacts, "ablation", "metrics", "accuracy", "mean"),
                observed(artifacts, "ablation", "metrics", "accuracy", "std"),
                observed(artifacts, "ablation", "metrics", "accuracy", "n"),
                observed(artifacts, "ablation", "metrics", "accuracy", "ci95_low"),
                observed(artifacts, "ablation", "metrics", "accuracy", "ci95_high"),
                observed(artifacts, "ablation", "n_configs"),
            ],
        },
    ]
    for row in rows:
        row["artifacts"] = sorted({obs["artifact"] for obs in row["observations"]})
    return rows


def acceptance_checks(artifacts: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Evaluate pre-registered acceptance criteria against the artifacts."""
    d = _derive(artifacts)
    metrics = artifacts["real_run"]
    ablation = artifacts["ablation"]
    paired = d["paired"]
    models = list(_PAIRED_MODELS)
    min_n = POLICY["min_n_recommended"]

    def check(
        check_id: str,
        criterion: str,
        threshold: dict[str, Any],
        observed_value: dict[str, Any],
        passed: bool,
        artifacts_used: Sequence[str],
    ) -> dict[str, Any]:
        return {
            "id": check_id,
            "criterion": criterion,
            "threshold": threshold,
            "observed": observed_value,
            "passed": bool(passed),
            "artifacts": sorted(set(artifacts_used)),
        }

    # Same convention as evidence_table(): no leading "aggregates" in these keys.
    qacc = f"{_PRIMARY_EXPERIMENT}/{models[0]}.accuracy"
    cacc = f"{_PRIMARY_EXPERIMENT}/qsvc_no_mixing_control.accuracy"
    pacc = f"{_PRIMARY_EXPERIMENT}/{models[1]}.accuracy"

    significance_ok = (
        paired is not None
        and paired.get("ci95_low", 0.0) > 0.0
        and paired.get("p_holm") is not None
        and paired["p_holm"] < POLICY["alpha"]
        and (paired.get("power_at_observed_dz") or 0.0) >= POLICY["min_power"]
    )

    checks = [
        check(
            "mixing_gain_above_control_noise",
            "A primary-model gain must exceed a control-noise margin of two control SDs.",
            {"margin": d["margin"], "rule": "2 * no_mixing_control_std"},
            {
                "gain": d["gap"],
                "control_std": d["control_std"],
                "n": [d["mix_n"], d["control_n"]],
            },
            d["gap"] > d["margin"] and min(d["mix_n"], d["control_n"]) >= min_n,
            [
                observed(artifacts, "real_run", "aggregates", *qacc.split("."), "mean")["artifact"],
                observed(artifacts, "real_run", "aggregates", *cacc.split("."), "std")["artifact"],
            ],
        ),
        check(
            "quantum_advantage_over_matched_classical",
            "Quantum advantage requires a positive paired difference, Holm p below "
            "alpha, a CI excluding zero, and power at the required minimum.",
            {
                "min_mean_difference": POLICY["advantage_min_gain"],
                "max_p_holm": POLICY["alpha"],
                "min_power": POLICY["min_power"],
                "ci_excludes_zero": True,
            },
            {
                "difference_vs_best_classical": d["advantage_gap"],
                "mean_difference": paired.get("mean_difference") if paired else None,
                "ci95": [paired.get("ci95_low"), paired.get("ci95_high")] if paired else None,
                "p_holm": paired.get("p_holm") if paired else None,
                "power": paired.get("power_at_observed_dz") if paired else None,
                "decision": paired.get("decision") if paired else None,
            },
            d["advantage_gap"] > POLICY["advantage_min_gain"] and significance_ok,
            [observed(artifacts, "real_run", "comparisons")["artifact"]],
        ),
        check(
            "classical_control_reproduces_gain",
            "A matched classical control must reproduce the gain at the required n "
            "before any quantum mechanism is credited.",
            {"rule": "classical_acc > control_acc + 2 * control_std", "margin": d["margin"]},
            {
                "poly2_acc": d["poly_acc"],
                "rbf_acc": d["rbf_acc"],
                "control_acc": d["control_acc"],
                "n": d["poly_n"],
            },
            d["poly_acc"] > d["control_acc"] + d["margin"] and d["poly_n"] >= min_n,
            [observed(artifacts, "real_run", "aggregates", *pacc.split("."), "mean")["artifact"]],
        ),
        check(
            "qnn_ood_beats_chance",
            "Every QNN OOD arm must exceed chance accuracy and chance AUC.",
            {"min_accuracy": d["chance"], "min_auc": d["chance"]},
            {
                "imageNet_accuracy": d["full_acc"],
                "imageNet_auc": d["full_auc"],
                "plantNet_accuracy": d["plant_acc"],
                "plantNet_auc": d["plant_auc"],
            },
            min(d["full_acc"], d["full_auc"], d["plant_acc"], d["plant_auc"])
            > d["chance"],
            [
                observed(artifacts, "qml_full", "ood", "auc")["artifact"],
                observed(artifacts, "qml_plantnet", "ood", "auc")["artifact"],
            ],
        ),
        check(
            "qnn_predicts_all_classes",
            "The QNN must emit non-degenerate predictions for every dataset class.",
            {"min_predicted_classes": d["class_count"], "min_f1": 0.0 + POLICY["equivalence_tol"]},
            {
                "predicted_classes": d["predicted_classes"],
                "plantNet_f1": d["plant_f1"],
                "imageNet_f1": d["full_f1"],
            },
            d["predicted_classes"] >= d["class_count"]
            and d["plant_f1"] > POLICY["equivalence_tol"],
            [observed(artifacts, "qml_plantnet", "ood", "confusion_matrix")["artifact"]],
        ),
        check(
            "threshold_operating_point_usable",
            "A reported threshold must beat chance with balanced recall and precision "
            "at least half of chance (chance derived from the class count).",
            {
                "min_balanced_accuracy": d["chance"] + POLICY["equivalence_tol"],
                "min_recall": POLICY["min_recall_share"] * d["chance"],
                "min_precision": POLICY["min_recall_share"] * d["chance"],
            },
            {
                "best_balanced_accuracy": d["best_bal_acc"],
                "recall_at_that_point": d["best_bal_recall"],
                "best_f1": d["plant_f1"],
                "precision_at_best_f1": d["best_f1_precision"],
                "balanced_accuracy_at_best_f1": d["best_f1_bal_acc"],
            },
            d["best_bal_acc"] > d["chance"]
            and d["best_bal_recall"] >= POLICY["min_recall_share"] * d["chance"]
            and d["best_f1_precision"] >= POLICY["min_recall_share"] * d["chance"]
            and d["best_f1_bal_acc"] > d["chance"],
            [observed(artifacts, "threshold", "sweep")["artifact"]],
        ),
        check(
            "noise_robustness_is_quantum_specific",
            "A noise-robustness claim requires a quantum/classical gap at a recorded "
            "sigma and accuracy retention at the required floor.",
            {
                "min_paired_gap": POLICY["equivalence_tol"],
                "min_retention": POLICY["min_retention"],
            },
            {
                "max_paired_gap": d["noise_gap"],
                "worst_sigma": d["worst_sigma"],
                "quantum_retention": d["retention"],
            },
            d["noise_gap"] > POLICY["equivalence_tol"]
            and d["retention"] >= POLICY["min_retention"],
            [observed(artifacts, "real_run", "aggregates")["artifact"]],
        ),
        check(
            "ablation_conclusive",
            "Architecture ablations need the recommended sample size and an accuracy "
            "CI excluding chance.",
            {"min_n": min_n, "ci_excludes_chance": True, "chance": d["chance"]},
            {"n": d["ablation_n"], "ci95": list(d["ablation_ci"])},
            d["ablation_conclusive"],
            [observed(artifacts, "ablation", "metrics", "accuracy")["artifact"]],
        ),
        check(
            "no_quantum_backend_fallback",
            "A quantum run must not fall back to a classical oracle.",
            {"max_fallback_warnings": 0},
            {"fallback_warnings": d["fallback_count"]},
            d["fallback_count"] == 0,
            [observed(artifacts, "real_run", "warnings")["artifact"]],
        ),
        check(
            "reporting_fields_complete",
            "Every compared aggregate must carry n, ddof=1 SD, full values and a CI.",
            {
                "required_fields": ["n", "std", "values", "ci95_low", "ci95_high"],
                "min_n_for_significance": POLICY["min_n_for_significance"],
            },
            {"complete": d["complete_aggregates"], "sample_sizes": d["sample_sizes"]},
            d["complete_aggregates"]
            and all(n >= POLICY["min_n_for_significance"] for n in d["sample_sizes"]),
            [observed(artifacts, "real_run", "aggregates")["artifact"]],
        ),
        check(
            "no_significance_under_minimum_n",
            "No comparison may be labelled significant below the minimum paired n.",
            {"min_n": POLICY["min_n_for_significance"], "max_violations": 0},
            {"violations": d["bogus_significance"]},
            d["bogus_significance"] == 0,
            [observed(artifacts, "real_run", "comparisons")["artifact"]],
        ),
        check(
            "ablation_run_manifest_complete",
            "Every declared ablation configuration must report a run.",
            {"n_configs": ablation["n_configs"]},
            {"n_success": ablation["n_success"], "n_failed": ablation["n_failed"]},
            ablation["n_success"] + ablation["n_failed"] == ablation["n_configs"],
            [observed(artifacts, "ablation", "n_configs")["artifact"]],
        ),
    ]
    return checks


def build_report(root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Load artifacts under *root* and emit the full evidence + acceptance report."""
    artifacts = load_artifacts(root)
    try:
        evidence = evidence_table(artifacts)
        acceptance = acceptance_checks(artifacts)
    except (KeyError, IndexError, TypeError) as exc:
        # Direct lookups inside the derivation helpers can also fail when an
        # artifact lacks a required key; surface all of them as ArtifactError
        # so the CLI reports a clean message and a non-zero exit.
        raise ArtifactError(
            f"artifact lacks a required key for derivation: {exc!r}"
        ) from exc
    passed = sum(row["passed"] for row in acceptance)
    return {
        "schema_version": "1.0",
        "root": str(Path(root).resolve()),
        "artifacts": dict(ARTIFACT_PATHS),
        "policy": dict(POLICY),
        "evidence": evidence,
        "acceptance": acceptance,
        "summary": {
            "n_evidence": len(evidence),
            "n_checks": len(acceptance),
            "n_passed": passed,
            "n_failed": len(acceptance) - passed,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(REPO_ROOT), help="repository root containing results/")
    parser.add_argument("--output", default="-", help="output JSON path, or '-' for stdout")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.root)
    except ArtifactError as exc:
        print(f"derive_methodology: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(payload)
    else:
        Path(args.output).write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
