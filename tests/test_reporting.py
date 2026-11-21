"""Synthetic, deterministic tests for the reporting layer (QC-I06).

All fixtures are small in-memory dictionaries and temporary files. The private
paper is never read or edited here; manuscript tests use invented tokens and
strings. No dataset-specific result numbers appear in source files under test.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from reporting import figures, manuscript_sync, tables


@pytest.fixture
def metrics() -> dict:
    """A synthetic metrics.json-like dict covering Tables 1-6 and Figs 1-5."""
    return {
        "schema_version": "1.0",
        "experiment_id": "synthetic_fixture",
        "provenance": {
            "config_hash": "cfg-synthetic",
            "master_seed": 123,
            "seed_map": {"model": 1, "noise": 2},
            "data_manifest_hash": "data-synthetic",
            "library_versions": {},
        },
        "dataset_table": [
            {
                "dataset": "synthetic_source",
                "categories": "corn, weed",
                "patch_size": "128x128x3",
                "dim_reduction": "none",
                "training": "100",
                "validation": "not independent",
                "test": "50",
            }
        ],
        "pipeline_table": [
            {"index": 1, "model": "synthetic_cnn", "pipeline_description": "conv + linear"}
        ],
        "performance_table": [
            {
                "model": "synthetic_cnn",
                "accuracy": {"mean": 0.812, "n": 3, "std": 0.011},
                "precision": {"mean": 0.801},
                "recall": {"mean": 0.790},
                "f1": {"mean": 0.795},
                "auc": {"mean": 0.880},
                "training_time_seconds": {"mean": 12.5},
            }
        ],
        "comparisons": [
            {
                "pair": ["hybrid_cnn_qnn", "cnn"],
                "t_statistic": 1.25,
                "df": 2,
                "mean_difference": 0.045,
                "ci95_low": -0.010,
                "ci95_high": 0.100,
                "cohens_dz": 0.700,
                "power_at_observed_dz": 0.310,
                "p_raw": 0.041,
                "p_holm": 0.082,
                "decision": "inconclusive",
            }
        ],
        "iou_table": [
            {
                "model": "synthetic_cnn",
                "class": 0,
                "precision": 0.75,
                "recall": 0.80,
                "tp": 6,
                "fp": 1,
                "fn": 2,
                "iou": 6 / 9,
            }
        ],
        "noise_table": [
            {
                "model": "synthetic_cnn",
                "sigma_clean": 0.0,
                "sigma_noisy": 0.15,
                "accuracy_clean": {"mean": 0.812},
                "accuracy_noisy": {"mean": 0.740, "ci95_low": 0.700, "ci95_high": 0.780},
                "delta": {"mean": 0.072, "ci95_low": 0.050, "ci95_high": 0.094, "n": 50},
            }
        ],
        "aggregates": {
            "synthetic_cnn": {
                "accuracy": {
                    "n": 3,
                    "values": [0.800, 0.812, 0.824],
                    "mean": 0.812,
                    "std": 0.012,
                    "ci95_low": 0.795,
                    "ci95_high": 0.829,
                }
            },
            "qsvc_quantum_kernel": {
                "accuracy": {
                    "n": 3,
                    "values": [0.870, 0.900, 0.930],
                    "mean": 0.900,
                    "std": 0.030,
                    "ci95_low": 0.870,
                    "ci95_high": 0.930,
                }
            },
        },
        "noise_headline": {
            "sigma_noisy": 0.15,
            "accuracy_noisy": {"mean": 0.740},
            "delta": {"mean": 0.072, "ci95_low": 0.050, "ci95_high": 0.094, "n": 50},
        },
        "calibration": {
            "ece": 0.031,
            "n_samples": 50,
            "reliability": [
                {"bin_lower": 0.0, "bin_upper": 0.5, "mean_predicted": 0.25,
                 "mean_observed": 0.20, "count": 20},
                {"bin_lower": 0.5, "bin_upper": 1.0, "mean_predicted": 0.75,
                 "mean_observed": 0.78, "count": 30},
            ],
        },
        "figures": [
            {
                "figure_id": "fig02_qsvc_run_variance",
                "caption_numbers": {"mean": 0.900, "std": 0.030, "n": 3,
                                    "lo": 0.870, "hi": 0.930},
                "caption": "Synthetic caption 0.900 +/- 0.030.",
            }
        ],
        "warnings": [],
    }


def test_no_literal_result_arrays_in_reporting_source():
    """Figure and table modules contain no hardcoded numeric result arrays."""
    tables.assert_no_result_literals(tables, figures)
    figures.assert_no_hardcoded_data()


def test_tables_render_exact_json_values(metrics):
    """Every numeric table cell equals the value stored in the metrics dict."""
    all_tables = tables.render_all_tables(metrics)
    assert tuple(all_tables) == tables.TABLE_IDS

    table3 = all_tables["table3_performance_comparison"].dataframe
    expected = metrics["performance_table"][0]["accuracy"]["mean"]
    assert table3.loc[0, "Accuracy"] == expected

    table4 = all_tables["table4_significance"].dataframe
    assert table4.loc[0, "p (Holm)"] == metrics["comparisons"][0]["p_holm"]
    assert table4.loc[0, "Model Pair"] == "hybrid_cnn_qnn vs cnn"

    table5 = all_tables["table5_iou"].dataframe
    assert table5.loc[0, "IoU"] == metrics["iou_table"][0]["iou"]
    assert table5.loc[0, "TP"] == metrics["iou_table"][0]["tp"]

    table6 = all_tables["table6_noise"].dataframe
    assert table6.loc[0, "Delta"] == metrics["noise_table"][0]["delta"]["mean"]
    assert table6.loc[0, "Sigma noisy"] == metrics["noise_table"][0]["sigma_noisy"]


def test_table_sidecar_metadata_present_and_deterministic(metrics, tmp_path):
    """Sidecars name source keys and repeat renders are byte-identical."""
    first = tables.render_all_tables(metrics)
    second = tables.render_all_tables(metrics)
    for table_id in tables.TABLE_IDS:
        assert first[table_id].csv == second[table_id].csv
        assert first[table_id].metadata == second[table_id].metadata
        assert first[table_id].source_keys == second[table_id].source_keys

    meta = first["table3_performance_comparison"].metadata
    assert "accuracy.mean" in meta["source_metric_keys"]
    assert meta["rows_key"] == "performance_table"
    assert meta["n_rows"] == 1
    assert meta["columns"][1] == "Accuracy"

    written = tables.write_tables(metrics, tmp_path / "tables")
    sidecar = json.loads(written["table5_iou"]["metadata"].read_text(encoding="utf-8"))
    assert "iou_table[0].iou" not in sidecar["source_metric_keys"]  # key paths, not values
    assert "iou" in sidecar["source_metric_keys"]
    assert (tmp_path / "tables" / "table5_iou.csv").exists()


def test_missing_table_keys_raise_clearly(metrics):
    """Missing row collections and missing nested cells raise MissingMetricKeys."""
    broken_rows = dict(metrics)
    broken_rows.pop("iou_table")
    with pytest.raises(tables.MissingMetricKeys, match="iou_table"):
        tables.render_table("table5_iou", broken_rows)

    broken_cell = json.loads(json.dumps(metrics))
    broken_cell["performance_table"][0]["accuracy"].pop("mean")
    with pytest.raises(tables.MissingMetricKeys, match="accuracy.mean"):
        tables.render_table("table3_performance_comparison", broken_cell)


def test_figures_sidecars_present_and_deterministic(metrics, tmp_path):
    """Rendered figures carry provenance sidecars; repeats are deterministic."""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    metrics_path = metrics_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")

    first_dir = tmp_path / "figures_a"
    second_dir = tmp_path / "figures_b"
    # An empty synthetic raw artefact file verifies the path is recorded.
    raw = tmp_path / "predictions.csv"
    raw.write_text("sample_id,y_true,y_pred\n", encoding="utf-8")

    first = figures.render_all_figures(metrics_path, [raw], first_dir, git_sha="deadbeef")
    second = figures.render_all_figures(metrics_path, [raw], second_dir, git_sha="deadbeef")

    assert [a.figure_id for a in first] == list(figures.FIGURE_IDS)
    core = {"fig01_main_ood_accuracy", "fig02_qsvc_run_variance",
            "fig03_noise_robustness", "fig04_significance_forest",
            "fig05_reliability"}
    rendered = {a.figure_id: a for a in first if a.status == "rendered"}

    if figures.matplotlib_available():
        assert core <= set(rendered)
        for figure_id, artifact in rendered.items():
            assert artifact.png_path is not None and artifact.png_path.exists()
            assert artifact.provenance_path is not None and artifact.provenance_path.exists()
            sidecar = json.loads(artifact.provenance_path.read_text(encoding="utf-8"))
            assert sidecar["config_hash"] == "cfg-synthetic"
            assert sidecar["seed"] == 123
            assert sidecar["git_sha"] == "deadbeef"
            assert sidecar["source_metric_keys"]
            assert str(raw) in sidecar["raw_artifacts"]
            assert sidecar["status"] == "rendered"

        for figure_id in core:
            left = (first_dir / f"{figure_id}.provenance.json").read_bytes()
            right = (second_dir / f"{figure_id}.provenance.json").read_bytes()
            assert left == right
    else:
        assert not rendered
        assert all(a.status == "skipped" for a in first)
        assert not list(first_dir.glob("*.png"))

    assert (first_dir / "figures_index.json").exists()


def test_figures_skip_when_inputs_missing(metrics, tmp_path):
    """Absent metric data skips a figure gracefully instead of fabricating data."""
    empty = {"provenance": {}}
    artifacts = figures.render_figure("fig01_main_ood_accuracy", empty, (), tmp_path)
    assert artifacts.status == "skipped"
    assert artifacts.png_path is None
    if figures.matplotlib_available():
        assert "no metric data" in artifacts.reason
    else:
        assert "matplotlib" in artifacts.reason


def test_manuscript_plan_is_read_only_and_dry_run_default(metrics, tmp_path):
    """plan_sync proposes edits; apply_sync without apply makes no edits."""
    md_text = (
        "QSVC accuracy is {{QSVC_ACC_MEAN}} over n={{QSVC_ACC_N}} runs. "
        "Legacy 0.1379 and {{UNREGISTERED}} must remain untouched."
    )
    mapping = {
        "{{QSVC_ACC_MEAN}}": "aggregates.qsvc_quantum_kernel.accuracy.mean",
        "{{QSVC_ACC_N}}": "aggregates.qsvc_quantum_kernel.accuracy.n",
    }
    plan = manuscript_sync.plan_sync(md_text, metrics, mapping)
    assert len(plan) == 2
    assert {d.token for d in plan} == set(mapping)
    assert plan[0].new == str(metrics["aggregates"]["qsvc_quantum_kernel"]["accuracy"]["mean"])
    assert plan[0].new == "0.9"
    assert plan[1].new == "3"

    assert manuscript_sync.apply_sync(md_text, metrics, mapping) == md_text
    md_path = tmp_path / "draft.md"
    md_path.write_text(md_text, encoding="utf-8")
    text, diffs = manuscript_sync.sync_file(md_path, metrics, mapping)
    assert text == md_text
    assert len(diffs) == 2
    assert md_path.read_text(encoding="utf-8") == md_text


def test_manuscript_strict_apply_only_registered_tokens(metrics):
    """With apply=True only mapped tokens change; unregistered text is stable."""
    md_text = "Mean {{QSVC_ACC_MEAN}}; keep 0.1379 and {{UNREGISTERED}}."
    mapping = {"{{QSVC_ACC_MEAN}}": "aggregates.qsvc_quantum_kernel.accuracy.mean"}
    updated = manuscript_sync.apply_sync(md_text, metrics, mapping, apply=True)
    assert "{{QSVC_ACC_MEAN}}" not in updated
    assert str(metrics["aggregates"]["qsvc_quantum_kernel"]["accuracy"]["mean"]) in updated
    assert "0.1379" in updated
    assert "{{UNREGISTERED}}" in updated

    with pytest.raises(manuscript_sync.UnsafeSyncError):
        manuscript_sync.apply_sync(md_text, metrics, apply=True)


def test_manuscript_missing_key_raises(metrics):
    """A registered placeholder with a missing/null source key never invents a value."""
    with pytest.raises(manuscript_sync.MissingResultKey):
        manuscript_sync.plan_sync("value {{X}}", metrics, {"{{X}}": "no.such.key"})
    with pytest.raises(manuscript_sync.MissingResultKey):
        manuscript_sync.plan_sync("value {{X}}", metrics, {"{{X}}": "schema_version.missing"})


def test_manuscript_claim_template_uses_json_number(metrics):
    """A registered claim template wraps a value that still comes from JSON."""
    from reporting.manuscript_sync import Placeholder

    mapping = (
        Placeholder(
            "{{CLAIM}}",
            "aggregates.qsvc_quantum_kernel.accuracy.mean",
            format_spec=":.2f",
            template="QSVC scored {value} on the OOD split.",
            claim_id="qsvc_claim",
        ),
    )
    updated = manuscript_sync.apply_sync("Before. {{CLAIM}} After.", metrics, mapping, apply=True)
    assert updated == "Before. QSVC scored 0.90 on the OOD split. After."


def test_sync_file_refuses_private_write_without_opt_in(metrics, tmp_path):
    """The file-level API never writes into a .paper path unless explicitly allowed."""
    paper_dir = tmp_path / ".paper"
    paper_dir.mkdir()
    md_path = paper_dir / "plant.md"
    md_path.write_text("mean {{X}}", encoding="utf-8")
    mapping = {"{{X}}": "aggregates.qsvc_quantum_kernel.accuracy.mean"}
    with pytest.raises(manuscript_sync.UnsafeSyncError):
        manuscript_sync.sync_file(md_path, metrics, mapping, apply=True)
    assert md_path.read_text(encoding="utf-8") == "mean {{X}}"


def test_render_table_unknown_id_raises_key_error(metrics):
    """Requesting an unregistered table fails loudly."""
    with pytest.raises(KeyError):
        tables.render_table("table7_does_not_exist", metrics)


@pytest.fixture
def runner_metrics() -> dict:
    """Synthetic payload shaped exactly like ``experiments.runner.compute_metrics``.

    Namespaced aggregate keys, ``fig_<experiment>`` descriptors, per-run rows
    and the runner's six reporting table keys are all reproduced. Values are
    invented; only the shape under test is taken from the runner.
    """
    def summary(values, name):
        return {
            "model": name,
            "n": len(values),
            "values": values,
            "mean": sum(values) / len(values),
            "std": 0.02,
            "ci95_low": min(values),
            "ci95_high": max(values),
            "boot_B": 2000,
            "boot_seed": 42,
            "unit": "fraction",
        }

    main_values = [0.811, 0.823, 0.835, 0.829]
    qsvc_values = [0.861, 0.883, 0.872, 0.880]
    main_acc = summary(main_values, "cnn.accuracy")
    qsvc_acc = summary(qsvc_values, "qsvc_quantum_kernel.accuracy")

    def descriptor(fig_id, accuracy, experiment):
        caption = (
            "{experiment}: accuracy {mean:.2%} +/- {std:.2%} "
            "(n={n} runs, bootstrap 95% CI {lo:.2%}-{hi:.2%})."
        ).format(
            experiment=experiment,
            mean=accuracy["mean"],
            std=accuracy["std"],
            n=accuracy["n"],
            lo=accuracy["ci95_low"],
            hi=accuracy["ci95_high"],
        )
        return {
            "figure_id": fig_id,
            "status": "not_rendered_by_runner (reporting layer owns figures)",
            "source_artifacts": [
                "raw/predictions.csv", "raw/run_summaries.csv", "metrics/metrics.json"
            ],
            "caption_numbers": {
                "mean": accuracy["mean"], "std": accuracy["std"], "n": accuracy["n"],
                "ci95": [accuracy["ci95_low"], accuracy["ci95_high"]],
            },
            "caption": caption,
        }

    return {
        "schema_version": "1.0",
        "experiment_id": "qsvc_mixed_variance",
        "run_id": "synthetic-run",
        "provenance": {
            "config_hash": "runner-config-hash",
            "master_seed": 42,
            "seed_map": {"master": 42, "bootstrap": 7},
            "source_artifact_hashes": {},
            "bootstrap_seed": 7,
            "data_manifest_hash": "runner-data-hash",
        },
        "aggregates": {
            "main_ood/cnn": {
                "experiment_id": "main_ood", "model_id": "cnn", "sigma": 0.0,
                "accuracy": main_acc, "f1": summary(main_values, "cnn.f1"),
            },
            "qsvc_mixed_variance/qsvc_quantum_kernel": {
                "experiment_id": "qsvc_mixed_variance",
                "model_id": "qsvc_quantum_kernel", "sigma": 0.0,
                "accuracy": qsvc_acc,
                "f1": summary(qsvc_values, "qsvc_quantum_kernel.f1"),
            },
            "noise_robustness/cnn/sigma=0.15": {
                "experiment_id": "noise_robustness", "model_id": "cnn", "sigma": 0.15,
                "accuracy": summary([0.771, 0.783], "cnn.noisy.accuracy"),
                "delta_accuracy_from_clean": summary(
                    [0.040, 0.052], "cnn.delta_sigma_0.15"
                ),
            },
        },
        "figures": [
            descriptor("fig_main_ood", main_acc, "main_ood"),
            descriptor("fig_qsvc_mixed_variance", qsvc_acc, "qsvc_mixed_variance"),
            descriptor(
                "fig_noise_robustness", summary([0.771, 0.783], "noise"), "noise_robustness"
            ),
        ],
        "comparisons": [{
            "pair": ["hybrid_cnn_qnn", "cnn"],
            "primary_test": "paired_t_test (scipy.stats.ttest_rel, two-sided)",
            "n": 4, "t_statistic": 1.5, "df": 3, "p_raw": 0.031,
            "p_holm": 0.093, "mean_difference": 0.041,
            "ci95_low": -0.005, "ci95_high": 0.087, "cohens_dz": 0.75,
            "power_at_observed_dz": 0.42, "decision": "inconclusive",
        }],
        "runs": [
            {"experiment_id": "main_ood", "run_id": "seed_00", "model_id": "cnn",
             "sigma": 0.0, "noise_draw": 0, "n_samples": 50, "accuracy": 0.811,
             "ece": 0.041, "seed": 11, "config_hash": "runner-config-hash"},
            {"experiment_id": "main_ood", "run_id": "seed_01", "model_id": "cnn",
             "sigma": 0.0, "noise_draw": 0, "n_samples": 50, "accuracy": 0.823,
             "ece": 0.053, "seed": 12, "config_hash": "runner-config-hash"},
        ],
        "dataset_table": [{
            "dataset": "synthetic", "categories": 2, "patch_size": 128,
            "dim_reduction": 1024, "training": 100, "validation": "", "test": 50,
        }],
        "pipeline_table": [{"index": 1, "model": "cnn", "pipeline_description": "conv"}],
        "performance_table": [{
            "model": "cnn", "accuracy": {"mean": main_acc["mean"]},
            "precision": {"mean": None}, "recall": {"mean": None}, "f1": {"mean": None},
            "auc": {"mean": None}, "training_time_seconds": {"mean": None},
        }],
        "iou_table": [{
            "model": "cnn", "class": 0, "precision": 0.75, "recall": 0.80,
            "tp": 30, "fp": 5, "fn": 8, "iou": 30 / 43,
        }],
        "noise_table": [{
            "model": "cnn", "sigma_clean": 0.0, "sigma_noisy": 0.15,
            "accuracy_clean": {"mean": main_acc["mean"]},
            "accuracy_noisy": {"mean": 0.777},
            "delta": {"mean": 0.046, "ci95_low": 0.040, "ci95_high": 0.052, "n": 50},
        }],
        "table_status": {},
        "warnings": [],
    }


def test_runner_figure_ids_map_to_paper_specs():
    """The runner's ``fig_<experiment>`` ids are explicitly registered."""
    assert figures.spec_id_for_runner_figure("fig_qsvc_mixed_variance") == (
        "fig02_qsvc_run_variance"
    )
    assert figures.spec_id_for_runner_figure("fig_main_ood") == "fig01_main_ood_accuracy"
    assert figures.spec_id_for_runner_figure("fig_noise_robustness") == "fig03_noise_robustness"


def test_runner_shaped_namespaced_keys_resolve(runner_metrics):
    """Flat registered keys resolve against namespaced runner aggregates."""
    expected = runner_metrics["aggregates"]["qsvc_mixed_variance/qsvc_quantum_kernel"]["accuracy"]["mean"]
    assert tables.resolve_metric_path(
        runner_metrics, "aggregates.qsvc_quantum_kernel.accuracy.mean"
    ) == expected
    plan = manuscript_sync.plan_sync("mean {{QSVC_ACC_MEAN}}", runner_metrics)
    assert plan
    assert float(plan[0].new) == pytest.approx(expected, abs=1e-3)  # defaults must not raise


def test_runner_shaped_fig02_renders_with_numeric_caption(runner_metrics, tmp_path):
    """fig02 resolves namespaced keys and its caption carries real numbers."""
    artifact = figures.render_figure(
        "fig02_qsvc_run_variance", runner_metrics, (), tmp_path, git_sha="synthetic"
    )
    if figures.matplotlib_available():
        assert artifact.status == "rendered"
        assert artifact.png_path is not None and artifact.png_path.exists()
    else:
        assert artifact.status == "skipped"
        assert "matplotlib" in artifact.reason
    assert artifact.caption != figures.get_figure_spec("fig02_qsvc_run_variance").title
    assert any(character.isdigit() for character in artifact.caption)


def test_runner_shaped_captions_all_contain_json_numbers(runner_metrics):
    """Every core paper figure caption derives numbers from the runner payload."""
    for figure_id in (
        "fig01_main_ood_accuracy", "fig02_qsvc_run_variance", "fig03_noise_robustness",
        "fig04_significance_forest", "fig05_reliability",
    ):
        spec = figures.get_figure_spec(figure_id)
        numbers = figures._caption_numbers(spec, runner_metrics)
        caption = figures._caption_text(spec, numbers, runner_metrics)
        assert numbers, f"{figure_id} has no derived caption numbers"
        assert caption != spec.title, f"{figure_id} fell back to a numberless title"
        assert any(character.isdigit() for character in caption)


def test_runner_shaped_fig01_only_main_ood_clean(runner_metrics):
    """The OOD bar chart excludes QSVC and noisy aggregates."""
    payload, reason = figures.figure_input("fig01_main_ood_accuracy", runner_metrics)
    assert reason is None
    assert set(payload) == {"main_ood/cnn"}
    for entry in payload.values():
        assert entry["experiment_id"] == "main_ood"
        assert float(entry["sigma"]) == 0.0


def test_runner_shaped_dry_run_validates_and_preserves_text(runner_metrics):
    """Dry-run apply_sync validates keys but never edits text."""
    md_text = "QSVC mean {{QSVC_ACC_MEAN}} remains."
    mapping = {"{{QSVC_ACC_MEAN}}": "aggregates.qsvc_quantum_kernel.accuracy.mean"}
    assert manuscript_sync.apply_sync(md_text, runner_metrics, mapping) == md_text

    broken = copy.deepcopy(runner_metrics)
    broken["aggregates"].pop("qsvc_mixed_variance/qsvc_quantum_kernel")
    with pytest.raises(manuscript_sync.MissingResultKey):
        manuscript_sync.apply_sync(md_text, broken, mapping)


def test_every_rendered_png_has_a_sidecar(runner_metrics, tmp_path):
    """No image file is left behind without its provenance sidecar."""
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(runner_metrics, sort_keys=True), encoding="utf-8")
    artifacts = figures.render_all_figures(metrics_path, (), tmp_path / "figures", git_sha="synthetic")
    for artifact in artifacts:
        if artifact.png_path is not None and artifact.png_path.exists():
            assert artifact.provenance_path is not None
            assert artifact.provenance_path.exists()
            sidecar = json.loads(artifact.provenance_path.read_text(encoding="utf-8"))
            assert sidecar["source_metric_keys"]
            assert sidecar["git_sha"] == "synthetic"
