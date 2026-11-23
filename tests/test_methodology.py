"""Synthetic tests for the evidence-first methodology derivation.

All fixtures are generated dictionaries with arbitrary values; no real dataset,
result artifact, or manuscript is read.  The tests prove that the report values
and verdicts are computed from the supplied artifacts, that missing or malformed
artifacts fail cleanly, and that neither owned source file smuggles in result
literals as hardcoded claims.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "METHODOLOGY_V2.md"
SCRIPT = ROOT / "scripts" / "derive_methodology.py"

_spec = importlib.util.spec_from_file_location("derive_methodology", SCRIPT)
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)

# Arbitrary sentinel values, deliberately unlike any committed result.
MIX = 0.7319
MIX_SD = 0.0413
CONTROL = 0.2047
CONTROL_SD = 0.0117
POLY = 0.7319
RBF = 0.7101
N_RUNS = 12


def _summary(mean: float, sd: float, n: int) -> dict:
    return {
        "mean": mean,
        "std": sd,
        "n": n,
        "values": [mean] * n,
        "ci95_low": mean - sd,
        "ci95_high": mean + sd,
        "boot_B": 64,
        "boot_seed": 7,
        "unit": "fraction",
    }


def _agg(model: str, mean: float, sd: float = 0.01, n: int = N_RUNS) -> dict:
    return {
        "aggregates": {
            f"qsvc_mixed_variance/{model}": {
                "accuracy": _summary(mean, sd, n),
                "auc": _summary(mean, sd, n),
                "f1": _summary(mean, sd, n),
                "model_id": model,
            }
        }
    }


def make_artifacts(*, advantage: bool = False, healthy: bool = False) -> dict:
    """Build a complete synthetic artifact set with configurable verdicts."""
    poly_mean = 0.6203 if advantage else POLY
    rbf_mean = 0.6103 if advantage else RBF
    qnn_auc = 0.7023 if healthy else 0.3121
    qnn_acc = 0.6611 if healthy else 0.4123
    plant_auc = 0.6531 if healthy else 0.3317
    plant_acc = 0.6197 if healthy else 0.4733
    plant_f1 = 0.5123 if healthy else 0.0
    confusion = [[70, 30], [25, 75]] if healthy else [[300, 0], [300, 0]]
    best_bal = 0.7123 if healthy else 0.5037
    best_recall = 0.65 if healthy else 0.0071
    best_f1_precision = 0.68 if healthy else 0.5113
    best_f1_bal = 0.7101 if healthy else 0.5033
    noise_q = 0.6801 if healthy else 0.5203
    noise_c = 0.5203
    ablation_n = N_RUNS if healthy else 3
    ablation_ci = [0.6511, 0.7413] if healthy else [0.4901, 0.5503]
    warnings = [] if healthy else ["PennyLane is not installed; used classical oracle"]

    real = _agg("qsvc_quantum_kernel", MIX, MIX_SD)
    for model, mean in (
        ("qsvc_no_mixing_control", CONTROL),
        ("svc_poly2_classical", poly_mean),
        ("svc_rbf", rbf_mean),
    ):
        real["aggregates"].update(_agg(model, mean, CONTROL_SD)["aggregates"])
    for model in ("qsvc_quantum_kernel", "svc_poly2_classical"):
        for sigma in (0.05, 0.1):
            mean = noise_q if model == "qsvc_quantum_kernel" else noise_c
            real["aggregates"][f"noise_robustness/{model}/sigma={sigma}"] = {
                "accuracy": _summary(mean, 0.0123, N_RUNS)
            }
    real["dataset_table"] = [{"categories": 2, "training": 1111, "validation": 222, "test": 333}]
    real["comparisons"] = [
        {
            "pair": ["qsvc_quantum_kernel", "svc_poly2_classical"],
            "n": N_RUNS,
            "df": N_RUNS - 1,
            "mean_difference": 0.0311 if advantage else 0.0,
            "cohens_dz": 0.4123 if advantage else 0.0,
            "ci95_low": 0.0201 if advantage else 0.0,
            "ci95_high": 0.0911 if advantage else 0.0,
            "p_holm": 0.0113 if advantage else None,
            "power_at_observed_dz": 0.9131 if advantage else 0.0511,
            "decision": "significant" if advantage else "inconclusive",
            "primary_test": "paired_t_test (scipy.stats.ttest_rel, two-sided)",
        }
    ]
    real["warnings"] = warnings

    full = {
        "ood": {
            "accuracy": qnn_acc,
            "auc": qnn_auc,
            "f1": 0.5011,
            "confusion_matrix": [[61, 139], [91, 109]],
            "calibration": {"brier_score": 0.2211, "ece": 0.0811},
        },
        "training": {"best_val_accuracy": 0.8031},
    }
    plant = {
        "ood": {
            "accuracy": plant_acc,
            "auc": plant_auc,
            "f1": plant_f1,
            "confusion_matrix": confusion,
        },
        "training": {"best_val_accuracy": 0.6011},
    }
    threshold = {
        "best_by_balanced_accuracy": {
            "balanced_accuracy": best_bal,
            "recall": best_recall,
            "threshold": 0.4431,
        },
        "best_by_f1": {
            "f1": 0.6611,
            "precision": best_f1_precision,
            "balanced_accuracy": best_f1_bal,
            "threshold": 0.0,
        },
        "scores_summary": {"min": 0.4011, "max": 0.4491, "std": 0.0071},
        "provenance": {"note": "post-hoc analysis"},
        "sweep": [],
    }
    ablation = {
        "metrics": {"accuracy": {"mean": 0.5211, "std": 0.0311, "n": ablation_n,
                                 "ci95_low": ablation_ci[0], "ci95_high": ablation_ci[1],
                                 "values": [0.51, 0.52, 0.53]}},
        "n_configs": 3,
        "n_success": 3,
        "n_failed": 0,
    }
    return {
        "real_run": real,
        "qml_full": full,
        "qml_plantnet": plant,
        "threshold": threshold,
        "ablation": ablation,
        "model_card": {"text": "synthetic card"},
    }


def _check(checks: list[dict], check_id: str) -> dict:
    return next(item for item in checks if item["id"] == check_id)


def test_evidence_table_recomputes_observations_and_statuses() -> None:
    artifacts = make_artifacts()
    rows = {row["id"]: row for row in dm.evidence_table(artifacts)}

    mixing = rows["supervised_domain_mixing"]
    assert mixing["status"] == "partially_supported"
    means = [obs["value"] for obs in mixing["observations"] if obs["key"].endswith(".mean")]
    sample_sizes = [obs["value"] for obs in mixing["observations"] if obs["key"].endswith(".n")]
    assert MIX in means and CONTROL in means  # recomputed from the synthetic artifact
    assert N_RUNS in sample_sizes
    assert any(obs["value"] == CONTROL_SD for obs in mixing["observations"])

    assert rows["primary_classical_mixed_kernel"]["status"] == "supported"
    assert rows["quantum_kernel_advantage"]["status"] == "failed"
    assert rows["qnn_ood_generalization"]["status"] == "failed"
    assert rows["posthoc_threshold_selection"]["status"] == "failed"
    assert rows["quantum_noise_robustness"]["status"] == "failed"
    assert rows["architecture_ablation"]["status"] == "partially_supported"
    for row in rows.values():
        assert row["artifacts"]
        assert all(obs["artifact"].startswith("results/") for obs in row["observations"])


def test_noise_observation_keys_do_not_duplicate_aggregates_root() -> None:
    rows = {row["id"]: row for row in dm.evidence_table(make_artifacts(healthy=True))}
    noise = rows["quantum_noise_robustness"]
    keys = [obs["key"] for obs in noise["observations"]]
    assert not any("aggregates.aggregates" in key for key in keys)
    assert any(key.startswith("aggregates.noise_robustness/qsvc_quantum_kernel/sigma=") for key in keys)
    assert any(key.startswith("aggregates.noise_robustness/svc_poly2_classical/sigma=") for key in keys)
    means = [obs["value"] for obs in noise["observations"]]
    assert 0.6801 in means and 0.5203 in means  # synthetic quantum / classical noise means
    # Uniform convention: no row anywhere may repeat the top-level segment.
    for row in rows.values():
        for obs in row["observations"]:
            assert "aggregates.aggregates" not in obs["key"]


def test_acceptance_thresholds_are_derived_from_control_values() -> None:
    artifacts = make_artifacts()
    checks = dm.acceptance_checks(artifacts)
    mixing = _check(checks, "mixing_gain_above_control_noise")
    assert mixing["threshold"]["margin"] == pytest.approx(
        dm.POLICY["sd_margin"] * CONTROL_SD
    )
    assert mixing["observed"]["gain"] == pytest.approx(MIX - CONTROL)
    assert mixing["observed"]["control_std"] == CONTROL_SD
    assert mixing["passed"] is (MIX - CONTROL > dm.POLICY["sd_margin"] * CONTROL_SD)
    assert _check(checks, "quantum_advantage_over_matched_classical")["passed"] is False
    assert _check(checks, "no_quantum_backend_fallback")["observed"]["fallback_warnings"] == 1


def test_checks_flip_when_artifact_values_flip() -> None:
    failing = {item["id"]: item["passed"] for item in dm.acceptance_checks(make_artifacts())}
    healthy = {
        item["id"]: item["passed"]
        for item in dm.acceptance_checks(make_artifacts(advantage=True, healthy=True))
    }
    assert all(healthy.values())
    flipped = [key for key in failing if failing[key] != healthy[key]]
    assert set(flipped) >= {
        "quantum_advantage_over_matched_classical",
        "qnn_ood_beats_chance",
        "qnn_predicts_all_classes",
        "threshold_operating_point_usable",
        "noise_robustness_is_quantum_specific",
        "ablation_conclusive",
        "no_quantum_backend_fallback",
    }
    assert set(failing) == set(healthy)


def test_advantage_requires_paired_significance_not_just_a_gap() -> None:
    artifacts = make_artifacts(advantage=True, healthy=True)
    paired = artifacts["real_run"]["comparisons"][0]
    paired["p_holm"] = None
    paired["ci95_low"] = 0.0
    check = _check(dm.acceptance_checks(artifacts), "quantum_advantage_over_matched_classical")
    # The raw accuracy gap is still positive, but the paired bar is not met.
    assert check["observed"]["difference_vs_best_classical"] > 0
    assert check["passed"] is False


def test_source_contains_no_hardcoded_result_values() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    sentinels = (MIX, MIX_SD, CONTROL, CONTROL_SD, POLY, RBF, 0.7023, 0.6531)
    for value in sentinels:
        assert str(value) not in source
    # Legacy-style literal figures must never be baked into the derivation code.
    for literal in ("0.918", "0.496", "0.4275", "0.3755", "0.4413", "0.524"):
        assert literal not in source


def test_doc_numbers_are_always_cited_to_an_artifact() -> None:
    lines = DOC.read_text(encoding="utf-8").splitlines()
    metric = re.compile(r"\b0\.\d+\b")
    cited = [line for line in lines if metric.search(line)]
    assert cited
    for line in cited:
        assert "results/" in line, f"uncited metric literal: {line[:120]}"


def test_doc_explicitly_rejects_the_unsupported_claims() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "Explicitly rejected claims" in text
    for phrase in ("Quantum advantage — rejected", "Unsupported noise robustness — rejected",
                   "Hardcoded figures — rejected"):
        assert phrase in text


def test_report_end_to_end_from_synthetic_files(tmp_path: Path) -> None:
    artifacts = make_artifacts()
    for name, rel in dm.ARTIFACT_PATHS.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(artifacts[name]) if path.suffix == ".json" else artifacts[name]["text"],
            encoding="utf-8",
        )
    report = dm.build_report(tmp_path)
    assert report["schema_version"] == "1.0"
    assert report["summary"]["n_evidence"] == len(report["evidence"])
    assert report["summary"]["n_checks"] == len(report["acceptance"])
    assert report["summary"]["n_passed"] + report["summary"]["n_failed"] == len(
        report["acceptance"]
    )
    assert {row["id"] for row in report["evidence"]} >= {
        "primary_classical_mixed_kernel", "quantum_kernel_advantage", "qnn_ood_generalization"
    }
    checks = {row["id"] for row in report["acceptance"]}
    assert "mixing_gain_above_control_noise" in checks
    assert "ablation_manifest_complete" in checks


def test_missing_artifact_raises_with_path(tmp_path: Path) -> None:
    (tmp_path / "present.json").write_text("{}", encoding="utf-8")
    with pytest.raises(dm.ArtifactError, match="missing artifacts") as excinfo:
        dm.load_artifacts(tmp_path, {"a": "present.json", "b": "absent.json"})
    assert "absent.json" in str(excinfo.value)


def test_malformed_artifact_raises(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(dm.ArtifactError, match="unreadable artifact"):
        dm.load_artifacts(tmp_path, {"broken": "broken.json"})


def test_observed_missing_key_raises() -> None:
    with pytest.raises(dm.ArtifactError, match="absent"):
        dm.observed({"real_run": {}}, "real_run", "aggregates", "absent")


def test_cli_reports_missing_artifacts_cleanly(tmp_path: Path) -> None:
    exit_code = dm.main(["--root", str(tmp_path)])
    assert exit_code == 2
