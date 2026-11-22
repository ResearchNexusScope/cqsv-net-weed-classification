"""Synthetic, offline integration tests for the experiment runner (QC-D04).

Every fixture is an in-memory manifest/feature loader and a tiny injected
predictor: no dataset path, checkpoint, paper value or private paper content is
read.  The tests assert raw-before-metrics ordering, persisted schemas, seeded
re-runs, QSVC arm pairing/counts, the sigma grid and the reporting-facing table
contract (QC-I06).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.splits import QSVCProtocol, Split, sample_qsvc_pairs  # noqa: E402
from experiments import evaluate, noise, runner  # noqa: E402
from experiments.config import ExperimentConfig  # noqa: E402


def _sample(source: str, split: str, index: int, label: int) -> SimpleNamespace:
    name = f"g-{source}-{split}-{index:04d}"
    return SimpleNamespace(
        sample_id=f"{source}:{split}:{index:04d}",
        group_id=name,
        source=source,
        label=label,
        content_hash=hashlib.sha256(name.encode("utf-8")).hexdigest(),
        path=f"/synthetic/{source}/{split}/{index:04d}.png",
        bbox=None,
    )


def _manifest(cfg: ExperimentConfig) -> SimpleNamespace:
    """Synthetic split manifest large enough for the legacy QSVC protocol."""
    kaggle = [
        _sample("kaggle", "train", index, index % 2)
        for index in range(4 * cfg.qsvc_mixed_train_size)
    ]
    id_val = [_sample("kaggle", "id_val", index, index % 2) for index in range(20)]
    id_test = [_sample("kaggle", "id_test", index, index % 2) for index in range(20)]
    adaptation = [
        _sample("ninja", "adaptation", index, index % 2)
        for index in range(2 * cfg.qsvc_target_adaptation_per_run)
    ]
    ood = [
        _sample("ninja", "ood_test", index, index % 2)
        for index in range(2 * cfg.qsvc_test_per_run)
    ]
    record = lambda samples: SimpleNamespace(samples=tuple(samples))  # noqa: E731
    return SimpleNamespace(
        splits={
            Split.TRAIN: record(kaggle),
            Split.ID_VAL: record(id_val),
            Split.ID_TEST: record(id_test),
            Split.ADAPTATION: record(adaptation),
            Split.OOD_TEST: record(ood),
        }
    )


def _features(samples, cfg: runner.ResolvedConfig, split: str) -> np.ndarray:
    """Deterministic synthetic features; identical sample -> identical vector."""
    dimension = int(cfg.config.feature_dim)
    vectors = []
    for sample in samples:
        seed = int(hashlib.sha256(sample.sample_id.encode("utf-8")).hexdigest()[:8], 16)
        vectors.append(np.random.default_rng(seed).normal(size=dimension))
    return np.asarray(vectors, dtype=float)


@pytest.fixture
def base_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        seed=1234,
        n_qubits=3,
        feature_dim=2 ** 3,
        ninja_ann_dir="synthetic-ann",
        ninja_img_dir="synthetic-img",
        output_dir=str(tmp_path / "results"),
        qsvc_runs=3,
    )


def _qsvc_spec(base: ExperimentConfig) -> runner.ExperimentSpec:
    return runner.ExperimentSpec(
        experiment_id=runner.QSVC_EXPERIMENT,
        experiments=(runner.QSVC_EXPERIMENT,),
        manifest=_manifest(base),
        feature_loader=_features,
        data_provider=runner.manifest_data_provider,
        model_ids=("qsvc_quantum_kernel", "svc_poly2_classical", "qsvc_no_mixing_control"),
        comparison_pairs=(("qsvc_quantum_kernel", "svc_poly2_classical"),),
        bootstrap_n=100,
    )


def _resolve(base: ExperimentConfig, spec: runner.ExperimentSpec, *extra: str) -> runner.ResolvedConfig:
    argv = [
        "run",
        "--seed", str(base.seed),
        "--output-dir", base.output_dir,
        "--runs", str(base.qsvc_runs),
        "--noise-draws", "1",
        *extra,
    ]
    return runner.resolve_config(argv, config=base, spec=spec)


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- QSVC protocol
def test_qsvc_arms_are_paired_with_config_counts(base_config: ExperimentConfig) -> None:
    spec = _qsvc_spec(base_config)
    cfg = _resolve(base_config, spec)

    for run_index in range(base_config.qsvc_runs):
        mixed = sample_qsvc_pairs(spec.manifest, base_config, run_index, QSVCProtocol.MIXED)
        control = sample_qsvc_pairs(spec.manifest, base_config, run_index, QSVCProtocol.NO_MIXING)
        assert len(mixed.source_train) == base_config.qsvc_source_per_run
        assert len(mixed.target_train) == base_config.qsvc_target_adaptation_per_run
        assert len(mixed.train) == base_config.qsvc_mixed_train_size
        assert len(control.source_train) == base_config.qsvc_mixed_train_size
        assert control.target_train == ()
        assert len(control.train) == base_config.qsvc_mixed_train_size
        assert len(mixed.test) == len(control.test) == base_config.qsvc_test_per_run
        assert [s.sample_id for s in mixed.test] == [s.sample_id for s in control.test]

    data = runner.load_data(cfg, spec)
    for run_id in sorted({unit.run_id for unit in data.units}):
        units = [unit for unit in data.units if unit.run_id == run_id]
        assert {unit.model_id for unit in units} == set(spec.model_ids)
        test_ids = {unit.sample_ids for unit in units}
        assert len(test_ids) == 1  # every arm sees the same 50 test samples
        for unit in units:
            assert unit.X.shape[0] == base_config.qsvc_test_per_run
            assert len(unit.fit_X) == base_config.qsvc_mixed_train_size
            assert not set(unit.fit_group_ids) & set(unit.group_ids)
            assert unit.context["train_size"] == base_config.qsvc_mixed_train_size
            assert unit.context["test_size"] == base_config.qsvc_test_per_run


# --------------------------------------------------------------------------- runner pipeline
def test_raw_is_written_before_metrics_and_reporting_tables_are_consumable(
    base_config: ExperimentConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from models import qsvc as qsvc_mod

    monkeypatch.setattr(qsvc_mod, "qml", None)  # force the exact classical oracle
    spec = _qsvc_spec(base_config)
    cfg = _resolve(base_config, spec)

    original = runner.compute_metrics

    def guarded(target_cfg, target_dir):
        assert (Path(target_dir) / "raw" / "predictions.csv").is_file()
        return original(target_cfg, target_dir)

    monkeypatch.setattr(runner, "compute_metrics", guarded)
    manifest = runner.run_experiment(cfg, spec)

    assert manifest.stages.index("raw") < manifest.stages.index("metrics")
    predictions_path = cfg.output_dir / "raw" / "predictions.csv"
    summaries_path = cfg.output_dir / "raw" / "run_summaries.csv"
    assert tuple(_read_rows(predictions_path)[0]) == runner.PREDICTION_COLUMNS
    assert tuple(_read_rows(summaries_path)[0]) == runner.SUMMARY_COLUMNS

    rows = _read_rows(predictions_path)
    assert rows
    for column in ("y_true", "y_pred", "score_pos", "sample_id", "noise_seed", "model_seed"):
        assert column in rows[0]
    assert {row["experiment_id"] for row in rows} == {runner.QSVC_EXPERIMENT}
    assert {row["model_id"] for row in rows} == set(spec.model_ids)

    metrics_payload = json.loads((cfg.output_dir / "metrics" / "metrics.json").read_text())
    for key in runner.REPORTING_TABLE_KEYS:
        assert isinstance(metrics_payload[key], list)
    assert set(metrics_payload["table_status"]) == set(runner.REPORTING_TABLE_KEYS)
    assert metrics_payload["table_status"]["noise_table"]["status"] == "empty"
    assert metrics_payload["table_status"]["dataset_table"]["status"] == "populated"
    assert metrics_payload["comparisons"]
    comparison = metrics_payload["comparisons"][0]
    for key in (
        "pair", "t_statistic", "df", "mean_difference", "ci95_low", "ci95_high",
        "cohens_dz", "power_at_observed_dz", "p_raw", "p_holm", "decision",
    ):
        assert key in comparison
    accuracy_summary = next(
        iter(metrics_payload["aggregates"].values())
    )["accuracy"]
    for key in ("n", "values", "mean", "std", "ci95_low", "ci95_high", "boot_B", "boot_seed"):
        assert key in accuracy_summary
    assert accuracy_summary["n"] == base_config.qsvc_runs

    reporting = pytest.importorskip("reporting.tables")
    rendered = reporting.render_all_tables(metrics_payload)
    assert set(rendered) == set(reporting.TABLE_IDS)
    assert rendered["table4_significance"].n_rows == len(metrics_payload["comparisons"])
    assert rendered["table6_noise"].n_rows == 0  # empty table keeps its schema

    assert runner.verify_run(cfg.output_dir) == []


def test_metrics_subcommand_and_forced_rerun_are_byte_identical(
    base_config: ExperimentConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from models import qsvc as qsvc_mod

    monkeypatch.setattr(qsvc_mod, "qml", None)
    spec = _qsvc_spec(base_config)
    cfg = _resolve(base_config, spec)
    runner.run_experiment(cfg, spec)

    metrics_path = cfg.output_dir / "metrics" / "metrics.json"
    first_bytes = metrics_path.read_bytes()
    again = runner.compute_metrics(cfg, cfg.output_dir)
    assert again.path.read_bytes() == first_bytes

    rerun_cfg = _resolve(base_config, spec, "--force")
    runner.run_experiment(rerun_cfg, spec)
    assert metrics_path.read_bytes() == first_bytes

    with pytest.raises(runner.RunExistsError):
        runner.run_experiment(_resolve(base_config, spec), spec)


def test_no_data_provider_is_a_loud_error_not_a_fabricated_result(
    base_config: ExperimentConfig,
) -> None:
    spec = runner.ExperimentSpec(
        experiment_id=runner.QSVC_EXPERIMENT,
        experiments=(runner.QSVC_EXPERIMENT,),
        model_ids=("svc_poly2_classical",),
    )
    cfg = _resolve(base_config, spec)
    with pytest.raises(runner.RunnerError, match="data_provider"):
        runner.load_data(cfg, spec)
    assert not (cfg.output_dir / "raw" / "predictions.csv").exists()


def test_seed_pool_covers_every_model_at_full_run_count(
    base_config: ExperimentConfig, tmp_path: Path
) -> None:
    """Regression: 5 QSVC arms x 10 seeds needs 50 pool-size-stable seeds."""
    argv = [
        "run",
        "--seed", str(base_config.seed),
        "--output-dir", str(tmp_path / "full-seed-pool"),
        "--runs", str(max(base_config.qsvc_runs, 10)),
        "--dry-run",
    ]
    spec = runner.ExperimentSpec(
        experiment_id=runner.QSVC_EXPERIMENT,
        experiments=(runner.QSVC_EXPERIMENT,),
        model_ids=evaluate.QSVC_MODEL_IDS,
        seeds_per_model=10,
    )
    cfg = runner.resolve_config(argv, config=base_config, spec=spec)

    assert cfg.config.qsvc_runs == max(base_config.qsvc_runs, 10)
    all_seeds: list[int] = []
    for model_id in evaluate.QSVC_MODEL_IDS:
        for index in range(spec.seeds_per_model):
            key = f"model_{model_id}_seed_{index}"
            assert key in cfg.seed_map
            assert isinstance(cfg.seed_map[key], int)
            all_seeds.append(cfg.seed_map[key])
    assert len(set(all_seeds)) == len(all_seeds)
    assert cfg.seed_map["model_qsvc_quantum_kernel_seed_9"] in all_seeds

    # Pool-size independence: the SAME keys derived at 5 seeds/model must be
    # identical at 10 seeds/model (entropy is keyed by model index + seed index,
    # not by a pool position or spawn(n) child).
    smaller = runner.ExperimentSpec(
        experiment_id=runner.QSVC_EXPERIMENT,
        experiments=(runner.QSVC_EXPERIMENT,),
        model_ids=evaluate.QSVC_MODEL_IDS,
        seeds_per_model=5,
    )
    smaller_cfg = runner.resolve_config(argv, config=base_config, spec=smaller)
    compared = 0
    for key, value in smaller_cfg.seed_map.items():
        if key.startswith("model_"):
            assert cfg.seed_map[key] == value
            compared += 1
    assert compared == len(evaluate.QSVC_MODEL_IDS) * smaller.seeds_per_model


# --------------------------------------------------------------------------- noise experiment
def test_noise_rows_are_seeded_paired_and_reportable(
    base_config: ExperimentConfig, tmp_path: Path
) -> None:
    n_samples = 6
    generator = np.random.default_rng(base_config.seed)
    clean_X = generator.normal(scale=0.1, size=(n_samples, int(base_config.feature_dim)))
    y = np.asarray([index % 2 for index in range(n_samples)], dtype=int)
    sample_ids = tuple(f"synthetic:{index}" for index in range(n_samples))
    clean_seeds = noise.make_noise_seeds(
        base_config.seed, "run_00", 0.0, 0, n_samples, noise.SIGMA_GRID
    )

    def provider(cfg, spec):
        clean = runner.PredictionUnit(
            experiment_id=runner.MAIN_EXPERIMENT,
            run_id="run_00",
            model_id="toy",
            X=clean_X,
            y=y,
            sample_ids=sample_ids,
            group_ids=tuple(f"g{index}" for index in range(n_samples)),
            source_datasets=tuple("synthetic" for _ in sample_ids),
            splits=tuple("ood_test" for _ in sample_ids),
            model_seed=base_config.seed,
            noise_seeds=clean_seeds,
        )
        corrupted = runner.expand_noise_units(cfg, spec, (clean,))
        return runner.DataBundle(units=(clean,) + corrupted)

    def toy_predict(features, fit_X, fit_y, seed):
        return (
            (features[:, 0] > np.median(features[:, 0])).astype(int),
            1.0 / (1.0 + np.exp(-features[:, 0])),
            1.0,
        )

    spec = runner.ExperimentSpec(
        experiment_id=runner.NOISE_EXPERIMENT,
        experiments=(runner.NOISE_EXPERIMENT,),
        data_provider=provider,
        model_factories={"toy": lambda cfg: evaluate.CallablePredictor(toy_predict)},
        model_ids=("toy",),
        bootstrap_n=100,
    )
    cfg = _resolve(base_config, spec, "--output-dir", str(tmp_path / "noise-results"))
    runner.run_experiment(cfg, spec)

    rows = _read_rows(cfg.output_dir / "raw" / "predictions.csv")
    sigmas = {float(row["sigma"]) for row in rows}
    assert sigmas == set(noise.SIGMA_GRID)
    clean_rows = [row for row in rows if float(row["sigma"]) == 0.0]
    noisy_rows = [row for row in rows if float(row["sigma"]) > 0.0]
    assert {row["experiment_id"] for row in clean_rows} == {runner.MAIN_EXPERIMENT}
    assert {row["experiment_id"] for row in noisy_rows} == {runner.NOISE_EXPERIMENT}
    for sigma in noise.SIGMA_GRID[1:]:
        expected = noise.make_noise_seeds(
            base_config.seed, "run_00", sigma, 0, n_samples, noise.SIGMA_GRID
        )
        actual = [
            int(row["noise_seed"])
            for row in noisy_rows
            if float(row["sigma"]) == sigma
        ]
        assert sorted(actual) == sorted(expected)

    metrics_payload = json.loads((cfg.output_dir / "metrics" / "metrics.json").read_text())
    noise_rows = metrics_payload["noise_table"]
    assert len(noise_rows) == len(noise.SIGMA_GRID) - 1
    for row in noise_rows:
        assert row["sigma_clean"] == 0.0
        assert row["sigma_noisy"] in noise.SIGMA_GRID[1:]
        assert row["delta"]["n"] == n_samples
        assert row["delta"]["ci95_low"] <= row["delta"]["mean"] <= row["delta"]["ci95_high"]
        assert "mean" in row["accuracy_clean"] and "mean" in row["accuracy_noisy"]
    assert metrics_payload["table_status"]["noise_table"]["status"] == "populated"

    first_bytes = (cfg.output_dir / "metrics" / "metrics.json").read_bytes()
    runner.run_experiment(_resolve(base_config, spec, "--output-dir", str(tmp_path / "noise-results"), "--force"), spec)
    assert (cfg.output_dir / "metrics" / "metrics.json").read_bytes() == first_bytes


def test_dry_run_resolves_without_writing_and_figures_is_documented_noop(
    base_config: ExperimentConfig, tmp_path: Path
) -> None:
    spec = _qsvc_spec(base_config)
    cfg = runner.resolve_config(
        ["run", "--dry-run", "--seed", str(base_config.seed), "--output-dir", str(tmp_path / "dry")],
        config=base_config,
        spec=spec,
    )
    assert cfg.dry_run is True
    assert not (tmp_path / "dry").exists()
    assert cfg.seed_map["master"] == base_config.seed

    assert runner.render_figures(cfg, cfg.output_dir) == []
    assert runner.main(["figures", "--output-dir", str(tmp_path / "dry")]) == 0
