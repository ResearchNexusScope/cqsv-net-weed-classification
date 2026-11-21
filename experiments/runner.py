"""Seeded experiment runner for the Quantum_Classification rebuild (QC-D04).

Design contract
---------------
The runner is a thin orchestrator over pure stages::

    resolve_config -> load_data -> build_models -> predict_all
                   -> compute_metrics -> render_figures -> write_manifest

``predict_all`` persists one raw row per (experiment, run, model, sigma, draw,
sample) to ``results/<run_id>/raw/`` **before** ``compute_metrics`` reads those
artifacts back.  Metrics and figures never need the dataset, a checkpoint or an
in-memory prediction object: ``metrics`` and ``figures`` are independently
re-runnable from disk.

Everything external is injected through :class:`ExperimentSpec`: a data
manifest plus feature loader, and/or a ``data_provider`` callable, plus model
predictor factories.  No dataset path, checkpoint path, paper number or legacy
plot array appears in this module.  If data or a trained model is unavailable,
the runner reports the missing dependency and writes no numbers -- it never
fabricates a benchmark result.

CLI
---
``python -m experiments.runner {run,metrics,figures,verify} [options]``
supports ``--experiment``, ``--config``, ``--output-dir``, ``--seed``,
``--models``, ``--sigma-grid``, ``--runs``, ``--noise-draws``, ``--dry-run``,
``--force`` and ``--strict``.  ``figures`` is an intentional no-op owned by the
reporting layer; it is implemented, documented and returns an empty list rather
than a silent stub.  ``run`` requires an injected provider, so programmatic use
(:func:`run_experiment`) is the supported way to execute experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experiments import evaluate as _evaluate
from experiments import metrics as _metrics
from experiments import noise as _noise
from experiments import stats as _stats

__all__ = [
    "RunnerError",
    "ConfigValueError",
    "RunExistsError",
    "PredictionUnit",
    "DataBundle",
    "ExperimentSpec",
    "ResolvedConfig",
    "PredictionArtifact",
    "MetricsBundle",
    "FigureArtifact",
    "RunManifest",
    "PREDICTION_COLUMNS",
    "SUMMARY_COLUMNS",
    "METRICS_RUN_COLUMNS",
    "REPORTING_TABLE_KEYS",
    "resolve_config",
    "manifest_data_provider",
    "load_data",
    "build_models",
    "predict_all",
    "compute_metrics",
    "render_figures",
    "write_manifest",
    "verify_run",
    "run_experiment",
    "main",
]

MAIN_EXPERIMENT = "main_ood"
QSVC_EXPERIMENT = "qsvc_mixed_variance"
NOISE_EXPERIMENT = "noise_robustness"
KNOWN_EXPERIMENTS = (MAIN_EXPERIMENT, QSVC_EXPERIMENT, NOISE_EXPERIMENT, "significance", "calibration")

# Top-level keys consumed by reporting.tables.render_table (QC-I06 contract).
REPORTING_TABLE_KEYS: tuple[str, ...] = (
    "dataset_table",
    "pipeline_table",
    "performance_table",
    "comparisons",
    "iou_table",
    "noise_table",
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "run_id",
    "model_id",
    "sigma",
    "noise_draw",
    "sample_id",
    "group_id",
    "source_dataset",
    "split",
    "y_true",
    "y_pred",
    "score_pos",
    "correct",
    "inference_ms",
    "noise_seed",
    "model_seed",
    "checkpoint_hash",
    "created_utc",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "run_id",
    "model_id",
    "sigma",
    "noise_draw",
    "n_samples",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "iou_class0",
    "iou_class1",
    "macro_iou",
    "tn",
    "fp",
    "fn",
    "tp",
    "brier",
    "ece",
    "mean_inference_ms",
    "fps",
    "seed",
    "config_hash",
)

# Timing fields remain in raw/ but are excluded from metrics.json so a seeded
# fixture yields byte-identical aggregates even though wall-clock time changes.
METRICS_RUN_COLUMNS: tuple[str, ...] = tuple(
    name for name in SUMMARY_COLUMNS if name not in ("mean_inference_ms", "fps")
)


class RunnerError(RuntimeError):
    """A runner precondition or artifact invariant was violated."""


class ConfigValueError(RunnerError):
    """The resolved experiment configuration failed validation."""


class RunExistsError(RunnerError):
    """A non-empty run_id exists and ``--force`` was not supplied."""


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictionUnit:
    """One (experiment, run, model, sigma, draw) evaluation request.

    ``X`` is the already-normalised post-corruption test feature matrix;
    ``fit_X``/``fit_y`` are the paired run's training data for kernel arms.
    Metadata arrays are parallel to ``X`` and are persisted verbatim so a
    figure or verifier can re-derive every number.
    """

    experiment_id: str
    run_id: str
    model_id: str
    X: np.ndarray
    y: np.ndarray
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    source_datasets: tuple[str, ...]
    splits: tuple[str, ...]
    sigma: float = 0.0
    noise_draw: int = 0
    fit_X: np.ndarray | None = None
    fit_y: np.ndarray | None = None
    fit_group_ids: tuple[str, ...] = ()
    model_seed: int = 0
    noise_seeds: tuple[int, ...] = ()
    clipping_policy: str = "none"
    context: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DataBundle:
    """Injected data: evaluation units plus provenance for the manifest."""

    units: tuple[PredictionUnit, ...]
    manifest_hash: str = ""
    warnings: tuple[str, ...] = ()
    sample_counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    """Dependency-injection container for one runner invocation."""

    experiment_id: str = "all"
    experiments: tuple[str, ...] = (MAIN_EXPERIMENT,)
    manifest: Any | None = None
    feature_loader: Callable[[Sequence[Any], "ResolvedConfig", str], np.ndarray] | None = None
    data_provider: Callable[["ResolvedConfig", "ExperimentSpec"], DataBundle] | None = None
    model_factories: Mapping[str, Callable[["ResolvedConfig"], Any]] = field(
        default_factory=dict
    )
    model_ids: tuple[str, ...] = ()
    sigma_grid: tuple[float, ...] = _noise.SIGMA_GRID
    noise_draws: int = 3
    clip_bounds: tuple[Any, Any] | None = None
    seeds_per_model: int = 5
    bootstrap_n: int = 2000
    comparison_pairs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ResolvedConfig:
    """Fully resolved configuration: the only input the pure stages accept."""

    config: Any
    experiment_id: str
    run_id: str
    seed_map: dict[str, int]
    config_hash: str
    output_dir: Path
    command: str = "run"
    experiment_choice: str = "all"
    models: tuple[str, ...] | None = None
    sigma_grid: tuple[float, ...] = _noise.SIGMA_GRID
    runs: int | None = None
    noise_draws: int = 3
    seeds_per_model: int = 5
    bootstrap_n: int = 2000
    comparison_pairs: tuple[tuple[str, str], ...] = ()
    dry_run: bool = False
    force: bool = False
    strict: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredictionArtifact:
    """Raw rows and one run-summary row produced by a single unit."""

    rows: tuple[dict, ...]
    summary: dict | None
    warnings: tuple[str, ...] = ()
    skipped: bool = False


@dataclass(frozen=True)
class MetricsBundle:
    """The serialised ``metrics.json`` payload and its location."""

    payload: dict
    path: Path


@dataclass(frozen=True)
class FigureArtifact:
    """Descriptor returned by the reporting-owned figures stage."""

    figure_id: str
    source_artifacts: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class RunManifest:
    """Top-level record of a completed run."""

    run_id: str
    output_dir: Path
    stages: tuple[str, ...]
    outputs: Mapping[str, str]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict:
        """JSON-serialisable view of the manifest."""
        return {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "stages": list(self.stages),
            "outputs": dict(sorted(self.outputs.items())),
            "warnings": list(self.warnings),
            "status": "complete",
        }


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _canonical_json(payload: Mapping) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fmt(value: Any) -> str:
    """Stable scalar formatting for CSV; blanks represent missing values."""
    if value is None:
        return ""
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return "" if not np.isfinite(number) else repr(round(number, _stats.ROUND_DIGITS))
    return str(value)


def _row_key(row: Mapping) -> tuple:
    def flt(name: str) -> float:
        try:
            return float(row.get(name) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def integer(name: str) -> int:
        try:
            return int(float(row.get(name) or 0))
        except (TypeError, ValueError):
            return 0

    return (
        str(row.get("experiment_id") or ""),
        str(row.get("run_id") or ""),
        str(row.get("model_id") or ""),
        flt("sigma"),
        integer("noise_draw"),
        str(row.get("sample_id") or ""),
    )


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=_row_key):
            writer.writerow({name: _fmt(row.get(name)) for name in columns})


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise RunnerError(f"missing required raw artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_digest(rows: Sequence[dict], columns: Sequence[str]) -> str:
    """Hash only the scientific payload of raw rows, not wall-clock fields.

    This keeps ``metrics.json`` byte-identical across re-runs with the same
    seed and fixture even though ``created_utc`` and timing columns differ.
    """
    projection = [{name: row.get(name) for name in columns} for row in rows]
    return _sha256_text(_canonical_json(projection))


def _run_int(run_id: str) -> int:
    return _noise.run_index(run_id)


def _split_samples(manifest: Any, name: str) -> tuple[Any, ...]:
    """Read a split by enum key, string key or attribute (no path knowledge)."""
    splits = getattr(manifest, "splits", None)
    if splits is None:
        raise RunnerError("injected manifest has no 'splits' mapping")
    record = None
    try:
        if name in splits:
            record = splits[name]
    except (TypeError, KeyError):
        pass
    if record is None:
        try:
            from data.splits import Split

            if Split(name) in splits:
                record = splits[Split(name)]
        except Exception:
            pass
    if record is None:
        record = getattr(splits, name, None)
    if record is None:
        raise RunnerError(f"injected manifest has no split {name!r}")
    samples = getattr(record, "samples", record)
    return tuple(samples)


def _sample_meta(sample: Any, split: str) -> tuple[str, str, str, str, int]:
    sample_id = getattr(sample, "sample_id", None)
    if sample_id is None:
        sample_id = str(sample)
    group_id = getattr(sample, "group_id", None) or str(sample_id)
    source = getattr(sample, "source", "unknown")
    if hasattr(source, "value"):
        source = source.value
    label = int(getattr(sample, "label", 0))
    return str(sample_id), str(group_id), str(source), str(split), label


def _features(spec: ExperimentSpec, cfg: ResolvedConfig, samples: Sequence[Any], split: str) -> np.ndarray:
    if spec.feature_loader is None:
        raise RunnerError("ExperimentSpec.feature_loader is required to build feature units")
    array = np.asarray(spec.feature_loader(samples, cfg, split), dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] != len(samples):
        raise RunnerError(
            f"feature_loader returned shape {array.shape} for {len(samples)} samples"
        )
    return array


def _effective_models(cfg: ResolvedConfig, spec: ExperimentSpec) -> tuple[str, ...]:
    models = list(spec.model_ids) or list(cfg.models or ()) or list(spec.model_factories)
    if cfg.models:
        models = [model for model in models if model in cfg.models]
    return tuple(dict.fromkeys(models))


# ---------------------------------------------------------------------------
# Stage 1: configuration
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.runner",
        description="Seeded, artifact-first experiment runner (QC-D04).",
    )
    parser.add_argument("command", choices=("run", "metrics", "figures", "verify"))
    parser.add_argument(
        "--experiment",
        default="all",
        choices=("main_ood", "qsvc_mixed_variance", "noise_robustness", "significance", "calibration", "all"),
    )
    parser.add_argument("--config", help="JSON overrides of ExperimentConfig fields")
    parser.add_argument("--output-dir", help="base directory; <base>/<run_id> is used")
    parser.add_argument("--seed", type=int, help="override the master seed")
    parser.add_argument("--models", help="comma-separated model ids to restrict")
    parser.add_argument("--sigma-grid", help="comma-separated sigma levels")
    parser.add_argument("--runs", type=int, help="override qsvc_runs")
    parser.add_argument("--noise-draws", type=int, help="draws per sigma (default 3)")
    parser.add_argument("--run-id", help="explicit run id (default: derived from config hash)")
    parser.add_argument("--dry-run", action="store_true", help="resolve/validate; write nothing")
    parser.add_argument("--force", action="store_true", help="allow overwriting an existing run")
    parser.add_argument("--strict", action="store_true", help="non-zero exit on warnings")
    return parser


def resolve_config(
    argv: Sequence[str] | None = None,
    *,
    config: Any | None = None,
    spec: ExperimentSpec | None = None,
) -> ResolvedConfig:
    """Parse the CLI, resolve every seed/path/hyperparameter and validate.

    ``config`` and ``spec`` are optional dependency-injection hooks: callers may
    supply an :class:`~experiments.config.ExperimentConfig` and the settings
    normally sourced from a spec without touching the environment.  Validation
    is enforced before any ``run`` work starts; derived-only subcommands
    (``metrics``/``figures``/``verify``) read existing artifacts and therefore
    do not require the dataset environment variables.
    """
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    from experiments.config import ExperimentConfig

    base = config if config is not None else ExperimentConfig.from_env()
    warnings: list[str] = []

    if args.config:
        overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ConfigValueError("--config JSON must be an object")
        known = {item.name for item in __import__("dataclasses").fields(base)}
        for key, value in overrides.items():
            if key in known:
                base = replace(base, **{key: value})
            else:
                warnings.append(f"unknown --config key ignored: {key}")

    if args.seed is not None:
        base = replace(base, seed=int(args.seed))
    if args.runs is not None:
        base = replace(base, qsvc_runs=int(args.runs))
    if args.output_dir:
        base = replace(base, output_dir=args.output_dir)

    sigma_grid = tuple(
        float(value) for value in args.sigma_grid.split(",")
    ) if args.sigma_grid else (spec.sigma_grid if spec else _noise.SIGMA_GRID)
    noise_draws = (
        int(args.noise_draws)
        if args.noise_draws is not None
        else (spec.noise_draws if spec else 3)
    )
    models = tuple(m for m in args.models.split(",") if m) if args.models else None
    bootstrap_n = int(spec.bootstrap_n) if spec else 2000
    seeds_per_model = int(spec.seeds_per_model) if spec else 5
    comparison_pairs = tuple(spec.comparison_pairs) if spec else ()
    experiment_id = spec.experiment_id if spec else str(args.experiment)

    if args.command == "run":  # dry-run resolves AND validates; it only skips writes
        problems = base.validate()
        if problems:
            raise ConfigValueError("; ".join(problems))

    config_hash = _sha256_text(_canonical_json(base.as_dict()))
    run_id = args.run_id or f"qc-{config_hash[:12]}"
    base_output = Path(args.output_dir) if args.output_dir else Path(base.output_dir)
    if args.output_dir and base_output.name == run_id:
        # A derived subcommand may point directly at an existing <run_id> dir.
        output_dir = base_output.resolve()
    else:
        output_dir = (base_output / run_id).resolve()

    from numpy.random import SeedSequence

    seed_map: dict[str, int] = {"master": int(base.seed)}
    generic = SeedSequence([base.seed, 0x51554356]).spawn(32)
    for index, name in enumerate(
        ("bootstrap", "noise", "env", "figures", "verification")
    ):
        seed_map[name] = int(generic[index].generate_state(1)[0])
    for index, name in enumerate(KNOWN_EXPERIMENTS):
        seed_map[f"experiment_{name}"] = int(generic[5 + index].generate_state(1)[0])
    qsvc_seeds = SeedSequence([base.seed, base.seed_offset]).spawn(base.qsvc_runs)
    for index in range(int(base.qsvc_runs)):
        seed_map[f"qsvc_run_{index}"] = int(qsvc_seeds[index].generate_state(1)[0])
    model_seeds = SeedSequence([base.seed, 0x4D4F444C]).spawn(32)
    cursor = 0
    for model_id in _evaluate.QSVC_MODEL_IDS:
        for index in range(seeds_per_model):
            seed_map[f"model_{model_id}_seed_{index}"] = int(
                model_seeds[cursor].generate_state(1)[0]
            )
            cursor += 1

    return ResolvedConfig(
        config=base,
        experiment_id=experiment_id,
        run_id=run_id,
        seed_map=seed_map,
        config_hash=config_hash,
        output_dir=output_dir,
        command=args.command,
        experiment_choice=args.experiment,
        models=models,
        sigma_grid=sigma_grid,
        noise_draws=noise_draws,
        runs=args.runs,
        seeds_per_model=seeds_per_model,
        bootstrap_n=bootstrap_n,
        comparison_pairs=comparison_pairs,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        strict=bool(args.strict),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Stages 2-3: data and models
# ---------------------------------------------------------------------------
def manifest_data_provider(cfg: ResolvedConfig, spec: ExperimentSpec) -> DataBundle:
    """Build units from an injected manifest + feature loader (no I/O here).

    Dispatches to :func:`main_ood_provider` / :func:`qsvc_provider` according to
    ``spec.experiments`` and appends the noise units derived from clean ones.
    """
    units: list[PredictionUnit] = []
    warnings: list[str] = []
    if MAIN_EXPERIMENT in spec.experiments:
        units.extend(_main_ood_units(cfg, spec))
    if QSVC_EXPERIMENT in spec.experiments:
        units.extend(_qsvc_units(cfg, spec))
    if NOISE_EXPERIMENT in spec.experiments:
        units.extend(expand_noise_units(cfg, spec, tuple(units)))
    try:
        from data.manifest import compute_checksum

        manifest_hash = compute_checksum(getattr(spec.manifest, "_to_payload", lambda: {})())
    except Exception:
        manifest_hash = _sha256_text(_canonical_json({"run_id": cfg.run_id, "seed": cfg.config.seed}))
    return DataBundle(
        units=tuple(units),
        manifest_hash=manifest_hash,
        warnings=tuple(warnings),
        sample_counts={experiment: sum(1 for unit in units if unit.experiment_id == experiment)
                       for experiment in sorted({unit.experiment_id for unit in units})},
    )


def _main_ood_units(cfg: ResolvedConfig, spec: ExperimentSpec) -> list[PredictionUnit]:
    samples = _split_samples(spec.manifest, "ood_test")
    if not samples:
        raise RunnerError("main_ood requires a non-empty ood_test split")
    X = _features(spec, cfg, samples, "ood_test")
    y = np.asarray([meta[4] for meta in (_sample_meta(s, "ood_test") for s in samples)], dtype=int)
    metas = [_sample_meta(sample, "ood_test") for sample in samples]
    units: list[PredictionUnit] = []
    for index in range(int(spec.seeds_per_model)):
        run_id = f"seed_{index:02d}"
        for model_id in _effective_models(cfg, spec):
            if model_id in _evaluate.QSVC_MODEL_IDS and model_id not in spec.model_factories:
                continue  # QSVC arms get their paired train/test data in _qsvc_units
            unit_seed = cfg.seed_map.get(
                f"model_{model_id}_seed_{index}", int(cfg.config.seed) + index
            )
            noise_seeds = _noise.make_noise_seeds(
                cfg.config.seed, run_id, 0.0, 0, len(samples), cfg.sigma_grid
            )
            units.append(
                PredictionUnit(
                    experiment_id=MAIN_EXPERIMENT,
                    run_id=run_id,
                    model_id=model_id,
                    X=X,
                    y=y,
                    sample_ids=tuple(meta[0] for meta in metas),
                    group_ids=tuple(meta[1] for meta in metas),
                    source_datasets=tuple(meta[2] for meta in metas),
                    splits=tuple(meta[3] for meta in metas),
                    model_seed=int(unit_seed),
                    noise_seeds=noise_seeds,
                    context={"training_seed_index": index},
                )
            )
    return units


def _qsvc_units(cfg: ResolvedConfig, spec: ExperimentSpec) -> list[PredictionUnit]:
    from data.splits import QSVCProtocol, sample_qsvc_pairs

    if spec.manifest is None:
        raise RunnerError(
            "qsvc_mixed_variance requires an injected manifest "
            "(ExperimentSpec.manifest); no dataset path is discovered by the runner"
        )
    runs = int(cfg.runs if cfg.runs is not None else cfg.config.qsvc_runs)
    model_ids = [m for m in _effective_models(cfg, spec) if m in _evaluate.QSVC_MODEL_IDS]
    if not runs or not model_ids:
        return []
    arms = (
        (QSVCProtocol.MIXED, tuple(m for m in model_ids if m != "qsvc_no_mixing_control")),
        (QSVCProtocol.NO_MIXING, ("qsvc_no_mixing_control",)),
    )
    units: list[PredictionUnit] = []
    for run_index in range(runs):
        runs_by_protocol = {
            protocol: sample_qsvc_pairs(spec.manifest, cfg.config, run_index, protocol)
            for protocol, _ in arms
        }
        run_id = f"run_{run_index:02d}"
        for protocol, arm_ids in arms:
            paired = runs_by_protocol[protocol]
            if not arm_ids:
                continue
            fit_samples = tuple(paired.train)
            test_samples = tuple(paired.test)
            fit_metas = [_sample_meta(s, "train") for s in fit_samples]
            test_metas = [_sample_meta(s, "ood_test") for s in test_samples]
            X_fit = _features(spec, cfg, fit_samples, "train")
            X_test = _features(spec, cfg, test_samples, "ood_test")
            y_fit = np.asarray([meta[4] for meta in fit_metas], dtype=int)
            y_test = np.asarray([meta[4] for meta in test_metas], dtype=int)
            noise_seeds = _noise.make_noise_seeds(
                cfg.config.seed,
                run_id,
                0.0,
                0,
                len(test_samples),
                cfg.sigma_grid,
            )
            for model_id in arm_ids:
                units.append(
                    PredictionUnit(
                        experiment_id=QSVC_EXPERIMENT,
                        run_id=run_id,
                        model_id=model_id,
                        X=X_test,
                        y=y_test,
                        sample_ids=tuple(meta[0] for meta in test_metas),
                        group_ids=tuple(meta[1] for meta in test_metas),
                        source_datasets=tuple(meta[2] for meta in test_metas),
                        splits=tuple(meta[3] for meta in test_metas),
                        fit_X=X_fit,
                        fit_y=y_fit,
                        fit_group_ids=tuple(meta[1] for meta in fit_metas),
                        model_seed=int(paired.seed),
                        noise_seeds=noise_seeds,
                        context={
                            "protocol": protocol.value,
                            "run_idx": run_index,
                            "seed_offset": int(cfg.config.seed_offset),
                            "train_size": len(paired.train),
                            "test_size": len(paired.test),
                        },
                    )
                )
    return units


def expand_noise_units(
    cfg: ResolvedConfig, spec: ExperimentSpec, clean_units: Sequence[PredictionUnit]
) -> tuple[PredictionUnit, ...]:
    """Create paired test-only corruptions for every clean unit and draw.

    The ``sigma=0`` anchor is never duplicated: a clean tensor is identical for
    every noise draw, so the paired endpoint reuses the single clean prediction
    per (run, model, sample) and records that pairing in ``context``.
    """
    grid = tuple(cfg.sigma_grid)
    noise_spec = _noise.NoiseSpec(
        sigma_grid=grid,
        noise_draws=int(cfg.noise_draws),
        clip_bounds=spec.clip_bounds,
    )
    problems = noise_spec.validate()
    if problems:
        raise ConfigValueError("; ".join(problems))
    corrupted: list[PredictionUnit] = []
    for unit in clean_units:
        if float(unit.sigma) != 0.0:
            continue
        for draw in range(int(cfg.noise_draws)):
            for sigma in grid:
                if float(sigma) == 0.0:
                    continue
                seeds = _noise.make_noise_seeds(
                    cfg.config.seed,
                    _run_int(unit.run_id),
                    float(sigma),
                    draw,
                    len(unit.y),
                    grid,
                )
                noisy_X = _noise.add_gaussian_noise(
                    unit.X, float(sigma), seeds, spec.clip_bounds
                )
                context = dict(unit.context)
                context.update(
                    {
                        "sigma_index": _noise.sigma_index(float(sigma), grid),
                        "clipping_policy": noise_spec.clipping_policy,
                        "paired_clean": {
                            "experiment_id": unit.experiment_id,
                            "run_id": unit.run_id,
                        },
                    }
                )
                corrupted.append(
                    replace(
                        unit,
                        experiment_id=NOISE_EXPERIMENT,
                        X=noisy_X,
                        sigma=float(sigma),
                        noise_draw=int(draw),
                        noise_seeds=seeds,
                        clipping_policy=noise_spec.clipping_policy,
                        context=context,
                    )
                )
    return tuple(corrupted)


def load_data(cfg: ResolvedConfig, spec: ExperimentSpec) -> DataBundle:
    """Invoke the injected provider and validate the unit schema/leakage guard."""
    if spec.data_provider is None:
        raise RunnerError(
            "no data_provider injected; supply ExperimentSpec.data_provider "
            "(manifest_data_provider) or a custom callable -- the runner never "
            "discovers dataset paths on its own"
        )
    bundle = spec.data_provider(cfg, spec)
    if not isinstance(bundle, DataBundle) or not bundle.units:
        raise RunnerError("data_provider returned an empty or invalid DataBundle")
    for unit in bundle.units:
        n = len(unit.y)
        if unit.X.ndim != 2 or unit.X.shape[0] != n:
            raise RunnerError(f"unit {unit.run_id}/{unit.model_id}: X/y length mismatch")
        for name in ("sample_ids", "group_ids", "source_datasets", "splits"):
            if len(getattr(unit, name)) != n:
                raise RunnerError(f"unit {unit.run_id}/{unit.model_id}: {name} length mismatch")
        if unit.noise_seeds and len(unit.noise_seeds) != n:
            raise RunnerError(f"unit {unit.run_id}/{unit.model_id}: noise seed count mismatch")
        if not set(np.asarray(unit.y)).issubset({0, 1}):
            raise RunnerError(f"unit {unit.run_id}/{unit.model_id}: labels must be 0/1")
        if unit.fit_X is not None:
            # Group-disjointness is enforced upstream by data.sample_qsvc_pairs;
            # re-check it here against the persisted sample metadata so a bad
            # injected provider fails loudly instead of leaking test groups.
            if unit.fit_group_ids and set(unit.fit_group_ids) & set(unit.group_ids):
                raise RunnerError(f"leakage: group overlap in unit {unit.run_id}")
    return bundle


def build_models(cfg: ResolvedConfig, spec: ExperimentSpec) -> dict[str, Any]:
    """Assemble injected predictors plus the built-in QSVC/classical arms."""
    models: dict[str, Any] = {}
    for model_id, factory in spec.model_factories.items():
        models[model_id] = factory(cfg)
    qsvc_ids = [
        model_id
        for model_id in _effective_models(cfg, spec)
        if model_id in _evaluate.QSVC_MODEL_IDS and model_id not in models
    ]
    models.update(_evaluate.build_qsvc_predictors(cfg.config, qsvc_ids))
    return models


# ---------------------------------------------------------------------------
# Stage 4: raw predictions (written before any aggregation)
# ---------------------------------------------------------------------------
def _unit_summary(
    unit: PredictionUnit, rows: Sequence[dict], inference_ms: float, config_hash: str
) -> dict:
    y_true = np.asarray([int(row["y_true"]) for row in rows], dtype=int)
    y_pred = np.asarray([int(row["y_pred"]) for row in rows], dtype=int)
    scores = [
        float(row["score_pos"])
        for row in rows
        if row.get("score_pos") not in (None, "")
    ]
    result = _metrics.binary_metrics(
        y_true, y_pred, scores if len(scores) == len(rows) else None
    )
    tn, fp, fn, tp = (int(value) for value in result.confusion.ravel())
    brier = ece = None
    if scores and len(scores) == len(rows):
        calibration = _metrics.calibration_summary(y_true, np.asarray(scores, dtype=float))
        brier, ece = calibration["brier_score"], calibration["ece"]
    mean_ms = float(inference_ms) / max(1, len(rows))
    return {
        "experiment_id": unit.experiment_id,
        "run_id": unit.run_id,
        "model_id": unit.model_id,
        "sigma": float(unit.sigma),
        "noise_draw": int(unit.noise_draw),
        "n_samples": int(len(rows)),
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "auc": result.auc,
        "iou_class0": result.iou_class0,
        "iou_class1": result.iou_class1,
        "macro_iou": result.macro_iou,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "brier": brier,
        "ece": ece,
        "mean_inference_ms": mean_ms,
        "fps": (1000.0 / mean_ms) if mean_ms > 0.0 else None,
        "seed": int(unit.model_seed),
        "config_hash": config_hash,
    }


def _write_raw(output_dir: Path, artifacts: Sequence[PredictionArtifact]) -> None:
    raw_dir = output_dir / "raw"
    rows = [row for artifact in artifacts for row in artifact.rows]
    summaries = [artifact.summary for artifact in artifacts if artifact.summary is not None]
    _write_csv(raw_dir / "predictions.csv", PREDICTION_COLUMNS, rows)
    _write_csv(raw_dir / "run_summaries.csv", SUMMARY_COLUMNS, summaries)
    try:  # parquet is a convenience mirror; CSV is always authoritative
        import pandas as pd  # noqa: F401

        pd.DataFrame(rows, columns=list(PREDICTION_COLUMNS)).to_parquet(
            raw_dir / "predictions.parquet", index=False
        )
    except Exception:
        pass


def predict_all(
    cfg: ResolvedConfig,
    data: DataBundle,
    models: Mapping[str, Any],
    spec: ExperimentSpec,
) -> list[PredictionArtifact]:
    """Predict every unit and persist raw rows before returning."""
    artifacts: list[PredictionArtifact] = []
    created_utc = _utc_now()
    for unit in data.units:
        predictor = models.get(unit.model_id)
        if predictor is None:
            artifacts.append(
                PredictionArtifact(
                    (),
                    None,
                    (f"no predictor injected for {unit.model_id}; arm skipped",),
                    True,
                )
            )
            continue
        try:
            prediction = predictor.predict(
                unit.X,
                fit_X=unit.fit_X,
                fit_y=unit.fit_y,
                seed=unit.model_seed,
            )
        except Exception as exc:  # degrade explicitly; never invent numbers
            artifacts.append(
                PredictionArtifact(
                    (),
                    None,
                    (
                        f"{unit.model_id} prediction failed "
                        f"({type(exc).__name__}: {exc}); arm skipped",
                    ),
                    True,
                )
            )
            continue
        if len(prediction.y_pred) != len(unit.y):
            artifacts.append(
                PredictionArtifact(
                    (),
                    None,
                    (f"{unit.model_id} returned {len(prediction.y_pred)} predictions "
                     f"for {len(unit.y)} samples; arm skipped",),
                    True,
                )
            )
            continue
        rows: list[dict] = []
        per_sample_ms = float(prediction.inference_ms) / max(1, len(unit.y))
        for index in range(len(unit.y)):
            score = None
            if prediction.score_pos is not None and index < len(prediction.score_pos):
                score = float(prediction.score_pos[index])
            y_pred = int(prediction.y_pred[index])
            y_true = int(unit.y[index])
            rows.append(
                {
                    "experiment_id": unit.experiment_id,
                    "run_id": unit.run_id,
                    "model_id": unit.model_id,
                    "sigma": float(unit.sigma),
                    "noise_draw": int(unit.noise_draw),
                    "sample_id": unit.sample_ids[index],
                    "group_id": unit.group_ids[index],
                    "source_dataset": unit.source_datasets[index],
                    "split": unit.splits[index],
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "score_pos": score,
                    "correct": int(y_pred == y_true),
                    "inference_ms": per_sample_ms,
                    "noise_seed": int(unit.noise_seeds[index])
                    if unit.noise_seeds
                    else int(unit.model_seed),
                    "model_seed": int(unit.model_seed),
                    "checkpoint_hash": str(
                        getattr(predictor, "checkpoint_hash", "not_available")
                    ),
                    "created_utc": created_utc,
                    "_clipping_policy": unit.clipping_policy,
                }
            )
        summary = _unit_summary(unit, rows, float(prediction.inference_ms), cfg.config_hash)
        artifacts.append(
            PredictionArtifact(
                rows=tuple(rows),
                summary=summary,
                warnings=_evaluate.predictor_warnings(predictor),
            )
        )
    _write_raw(cfg.output_dir, artifacts)
    return artifacts


# ---------------------------------------------------------------------------
# Stage 5: metrics, computed from raw artifacts only
# ---------------------------------------------------------------------------
def _figure_descriptors(aggregates: Mapping[str, dict]) -> list[dict]:
    """Caption descriptors whose numbers come from loaded aggregates only."""
    descriptors: list[dict] = []
    by_experiment: dict[str, dict] = {}
    for value in aggregates.values():
        by_experiment.setdefault(value["experiment_id"], value)
    for experiment in sorted(by_experiment):
        accuracy = by_experiment[experiment]["accuracy"]
        descriptor = {
            "figure_id": f"fig_{experiment}",
            "status": "not_rendered_by_runner (reporting layer owns figures)",
            "source_artifacts": [
                "raw/predictions.csv",
                "raw/run_summaries.csv",
                "metrics/metrics.json",
            ],
            "caption_numbers": {
                "mean": accuracy["mean"],
                "std": accuracy["std"],
                "n": accuracy["n"],
                "ci95": [accuracy["ci95_low"], accuracy["ci95_high"]],
            },
        }
        descriptor["caption"] = (
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
        descriptors.append(descriptor)
    return descriptors


def dataset_table_from_manifest(cfg: ResolvedConfig, spec: ExperimentSpec) -> list[dict]:
    """Table 1 rows from the injected manifest and the resolved config only."""
    if spec.manifest is None:
        return []
    config = cfg.config.as_dict()
    rows: list[dict] = []
    for training, validation, test in (
        ("train", "id_val", "id_test"),
        ("adaptation", None, "ood_test"),
    ):
        try:
            train_samples = _split_samples(spec.manifest, training)
            test_samples = _split_samples(spec.manifest, test)
            val_samples = _split_samples(spec.manifest, validation) if validation else ()
        except RunnerError:
            continue
        all_samples = train_samples + test_samples + val_samples
        if not all_samples:
            continue
        labels = {int(getattr(sample, "label", 0)) for sample in all_samples}
        source = getattr(all_samples[0], "source", "unknown")
        if hasattr(source, "value"):
            source = source.value
        rows.append(
            {
                "dataset": str(source),
                "categories": len(labels),
                "patch_size": int(config["image_size"]),
                "dim_reduction": int(config["feature_dim"]),
                "training": len(train_samples),
                "validation": len(val_samples) if validation else "",
                "test": len(test_samples),
            }
        )
    return rows


def pipeline_table_from_models(models: Mapping[str, Any]) -> list[dict]:
    """Table 2 rows from injected adapters (no result numbers are stored here)."""
    rows: list[dict] = []
    for index, (model_id, predictor) in enumerate(sorted(models.items()), start=1):
        module = getattr(predictor, "module", predictor)
        parameters = None
        try:
            parameters = sum(
                int(parameter.numel())
                for parameter in module.parameters()
                if parameter.requires_grad
            )
        except Exception:
            parameters = None
        description = type(predictor).__name__
        if parameters is not None:
            description += f"; trainable_parameters={parameters}"
        description += (
            f"; checkpoint={getattr(predictor, 'checkpoint_hash', 'not_available')}"
        )
        rows.append(
            {"index": index, "model": model_id, "pipeline_description": description}
        )
    return rows


def performance_table(aggregates: Mapping[str, dict]) -> list[dict]:
    """Table 3 rows from clean-run aggregate means (unmeasured cells are null)."""
    rows: list[dict] = []
    for name in sorted(aggregates):
        entry = aggregates[name]
        if entry["experiment_id"] not in (MAIN_EXPERIMENT, QSVC_EXPERIMENT):
            continue
        if float(entry["sigma"]) != 0.0:
            continue
        row: dict[str, Any] = {"model": entry["model_id"]}
        for metric_name in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "training_time_seconds",
        ):
            summary = entry.get(metric_name)
            # reporting.tables resolves dotted columns by traversing mappings.
            row[metric_name] = {
                "mean": summary["mean"] if isinstance(summary, dict) else None
            }
        rows.append(row)
    return rows


def iou_table(prediction_rows: Sequence[dict], columns: Sequence[str]) -> list[dict]:
    """Table 5 rows from integer confusion counts on clean paired predictions."""
    by_model: dict[str, list[tuple[int, int]]] = {}
    for row in prediction_rows:
        if float(_float_or_none(row.get("sigma")) or 0.0) != 0.0:
            continue
        if int(_float_or_none(row.get("noise_draw")) or 0) != 0:
            continue
        if "y_true" not in columns or "y_pred" not in columns:
            continue
        by_model.setdefault(str(row.get("model_id")), []).append(
            (int(float(row.get("y_true") or 0)), int(float(row.get("y_pred") or 0)))
        )
    rows: list[dict] = []
    for model_id in sorted(by_model):
        pairs = by_model[model_id]
        truth = [value[0] for value in pairs]
        predicted = [value[1] for value in pairs]
        for class_label in sorted(set(truth) | set(predicted)):
            binary_truth = [int(value == class_label) for value in truth]
            binary_prediction = [int(value == class_label) for value in predicted]
            result = _metrics.binary_metrics(binary_truth, binary_prediction)
            tn, fp, fn, tp = (int(value) for value in result.confusion.ravel())
            rows.append(
                {
                    "model": model_id,
                    "class": class_label,
                    "precision": result.precision,
                    "recall": result.recall,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "iou": _metrics.iou_from_counts(tp, fp, fn),
                }
            )
    return rows


def noise_table(aggregates: Mapping[str, dict]) -> list[dict]:
    """Table 6 rows: paired clean/noisy accuracy and the degradation CI."""
    rows: list[dict] = []
    for name in sorted(aggregates):
        entry = aggregates[name]
        if entry["experiment_id"] != NOISE_EXPERIMENT or float(entry["sigma"]) == 0.0:
            continue
        delta = entry.get("delta_accuracy_from_clean")
        if not isinstance(delta, dict):
            continue
        clean = None
        for candidate in aggregates.values():
            if (
                candidate["model_id"] == entry["model_id"]
                and float(candidate["sigma"]) == 0.0
                and "accuracy" in candidate
            ):
                clean = candidate
                break
        noisy_accuracy = entry.get("accuracy", {})
        rows.append(
            {
                "model": entry["model_id"],
                "sigma_clean": 0.0,
                "sigma_noisy": float(entry["sigma"]),
                "accuracy_clean": {
                    "mean": clean["accuracy"]["mean"] if clean else None
                },
                "accuracy_noisy": {"mean": noisy_accuracy.get("mean")},
                "delta": {
                    "mean": delta["mean"],
                    "ci95_low": delta["ci95_low"],
                    "ci95_high": delta["ci95_high"],
                    "n": delta["n"],
                },
            }
        )
    return rows


def _table_status(tables: Mapping[str, Sequence[dict]]) -> dict:
    """Per-table status so an empty table is explicit rather than a silent gap."""
    reasons = {
        "dataset_table": "no injected data manifest was available for this run",
        "pipeline_table": "no model adapters were recorded for this run",
        "performance_table": "no clean benchmark run rows were available",
        "comparisons": "no pre-registered pair had two or more paired runs",
        "iou_table": "no clean per-sample predictions were available",
        "noise_table": "no paired clean/noisy corruption rows were available",
    }
    return {
        table_id: {
            "status": "populated" if tables.get(table_id) else "empty",
            "reason": "" if tables.get(table_id) else reasons[table_id],
        }
        for table_id in REPORTING_TABLE_KEYS
    }


def compute_metrics(cfg: ResolvedConfig, output_dir: Path) -> MetricsBundle:
    """Aggregate ``raw/`` into ``metrics/metrics.json`` and ``metrics.csv``.

    This stage reads only files written by :func:`predict_all` plus the
    resolved ``config.json`` (pre-registered pair family and bootstrap policy).
    It never imports a model or accesses the dataset.
    """
    output_dir = Path(output_dir)
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        raise RunnerError(f"missing resolved config artifact: {config_path}")
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    summary_rows = _read_csv(output_dir / "raw" / "run_summaries.csv")
    prediction_rows = _read_csv(output_dir / "raw" / "predictions.csv")

    boot_seed = int(
        config_payload.get("options", {}).get("bootstrap_seed")
        or cfg.seed_map.get("bootstrap", cfg.config.seed)
    )
    boot_n = int(config_payload.get("options", {}).get("bootstrap_n", cfg.bootstrap_n))
    pairs = [tuple(pair) for pair in config_payload.get("options", {}).get("comparison_pairs", ())]
    warnings = [str(item) for item in config_payload.get("warnings", ())]

    def group_key(row: Mapping) -> tuple[str, str, float]:
        return (
            str(row.get("experiment_id") or ""),
            str(row.get("model_id") or ""),
            float(_float_or_none(row.get("sigma")) or 0.0),
        )

    grouped: dict[tuple[str, str, float], list[dict]] = {}
    for row in summary_rows:
        grouped.setdefault(group_key(row), []).append(row)

    aggregates: dict[str, dict] = {}
    for key in sorted(grouped):
        experiment, model_id, sigma = key
        rows = sorted(
            grouped[key],
            key=lambda row: (str(row.get("run_id") or ""), int(_float_or_none(row.get("noise_draw")) or 0)),
        )
        run_rows = [row for row in rows if int(_float_or_none(row.get("noise_draw")) or 0) == 0]
        if not run_rows:
            run_rows = rows
        entry: dict[str, Any] = {
            "experiment_id": experiment,
            "model_id": model_id,
            "sigma": sigma,
        }
        for metric_name in ("accuracy", "precision", "recall", "f1", "macro_iou"):
            values = [_float_or_none(row.get(metric_name)) for row in run_rows]
            values = [float(value) for value in values if value is not None]
            if values:
                entry[metric_name] = _stats.summarize(
                    values,
                    f"{model_id}.{metric_name}",
                    seed=boot_seed,
                    n_boot=boot_n,
                )
        auc_values = [_float_or_none(row.get("auc")) for row in run_rows]
        auc_values = [float(value) for value in auc_values if value is not None]
        if len(auc_values) == len(run_rows) and auc_values:
            entry["auc"] = _stats.summarize(
                auc_values, f"{model_id}.auc", seed=boot_seed, n_boot=boot_n
            )
        entry["runs"] = [
            {
                "run_id": row.get("run_id"),
                "noise_draw": int(_float_or_none(row.get("noise_draw")) or 0),
                "accuracy": _float_or_none(row.get("accuracy")),
            }
            for row in rows
        ]
        name = f"{experiment}/{model_id}"
        if sigma != 0.0:
            name += f"/sigma={sigma}"
        aggregates[name] = entry

    # Paired per-image degradation: clean anchor is the same image/model/run.
    clean_lookup: dict[tuple[str, str, str], int] = {}
    for row in prediction_rows:
        if (
            str(row.get("experiment_id")) in (MAIN_EXPERIMENT, QSVC_EXPERIMENT)
            and float(_float_or_none(row.get("sigma")) or 0.0) == 0.0
        ):
            key = (str(row.get("model_id")), str(row.get("run_id")), str(row.get("sample_id")))
            clean_lookup[key] = int(float(row.get("correct") or 0))
    for name, entry in list(aggregates.items()):
        if entry["experiment_id"] != NOISE_EXPERIMENT or float(entry["sigma"]) == 0.0:
            continue
        model_id = entry["model_id"]
        sigma = float(entry["sigma"])
        deltas: list[float] = []
        for row in prediction_rows:
            if (
                str(row.get("experiment_id")) != NOISE_EXPERIMENT
                or str(row.get("model_id")) != model_id
                or float(_float_or_none(row.get("sigma")) or 0.0) != sigma
            ):
                continue
            key = (model_id, str(row.get("run_id")), str(row.get("sample_id")))
            if key in clean_lookup:
                deltas.append(float(clean_lookup[key] - int(float(row.get("correct") or 0))))
        if deltas:
            entry["delta_accuracy_from_clean"] = _stats.paired_delta_summary(
                [1.0] * len(deltas),
                [1.0 - delta for delta in deltas],
                name=f"{model_id}.delta_sigma_{sigma}",
                seed=boot_seed,
                n_boot=boot_n,
            )

    # Pre-registered paired comparisons, aligned by run id and sample id.
    vectors: dict[str, dict[str, float]] = {}
    for row in summary_rows:
        if (
            float(_float_or_none(row.get("sigma")) or 0.0) != 0.0
            or int(_float_or_none(row.get("noise_draw")) or 0) != 0
        ):
            continue
        accuracy = _float_or_none(row.get("accuracy"))
        if accuracy is None:
            continue
        vectors.setdefault(str(row.get("model_id")), {})[str(row.get("run_id"))] = float(accuracy)
    predictions_by_key: dict[tuple[str, str, str], dict] = {}
    for row in prediction_rows:
        if (
            float(_float_or_none(row.get("sigma")) or 0.0) != 0.0
            or int(_float_or_none(row.get("noise_draw")) or 0) != 0
        ):
            continue
        predictions_by_key[
            (str(row.get("model_id")), str(row.get("run_id")), str(row.get("sample_id")))
        ] = row
    comparisons: list[dict] = []
    for pair in pairs:
        if len(pair) != 2:
            warnings.append(f"invalid comparison pair {pair!r}; skipped")
            continue
        first, second = str(pair[0]), str(pair[1])
        common = sorted(set(vectors.get(first, {})) & set(vectors.get(second, {})))
        if len(common) < 2:
            warnings.append(f"comparison {first} vs {second} has fewer than 2 paired runs; skipped")
            continue
        sample_ids = sorted(
            {
                key[2]
                for key in predictions_by_key
                if key[0] in (first, second) and key[1] in common
            }
        )
        table_data = None
        if sample_ids:
            truth, pred_a, pred_b = [], [], []
            for run_id in common:
                for sample_id in sample_ids:
                    left = predictions_by_key.get((first, run_id, sample_id))
                    right = predictions_by_key.get((second, run_id, sample_id))
                    if left is None or right is None:
                        continue
                    truth.append(int(float(left.get("y_true") or 0)))
                    pred_a.append(int(float(left.get("y_pred") or 0)))
                    pred_b.append(int(float(right.get("y_pred") or 0)))
            if truth:
                table_data = _stats.mcnemar_table(truth, pred_a, pred_b)
        first_values = [vectors[first][run_id] for run_id in common]
        second_values = [vectors[second][run_id] for run_id in common]
        comparisons.append(
            _stats.paired_comparison(
                (first, second),
                first_values,
                second_values,
                bootstrap_seed=boot_seed,
                n_boot=boot_n,
                mcnemar_table_data=table_data,
                wilcoxon_differences=[a - b for a, b in zip(first_values, second_values)],
            )
        )
    comparisons = _stats.compare_family(comparisons)

    # Reporting-facing tables (QC-I06 contract). Every row is derived from the
    # manifest/config snapshot recorded at run time or from the raw artifacts.
    tables: dict[str, list[dict]] = {
        "dataset_table": [
            dict(row) for row in config_payload.get("dataset_table", ())
        ],
        "pipeline_table": [
            dict(row) for row in config_payload.get("pipeline_table", ())
        ],
        "performance_table": performance_table(aggregates),
        "comparisons": comparisons,
        "iou_table": iou_table(prediction_rows, PREDICTION_COLUMNS),
        "noise_table": noise_table(aggregates),
    }
    status = _table_status(tables)
    for table_id, rows in tables.items():
        if not rows:
            warnings.append(f"{table_id} is empty: {status[table_id]['reason']}")

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_id": str(config_payload.get("experiment_id", cfg.experiment_id)),
        "run_id": str(config_payload.get("run_id", cfg.run_id)),
        "provenance": {
            "config_hash": config_payload.get("config_hash", cfg.config_hash),
            "master_seed": int(cfg.config.seed),
            "seed_map": dict(sorted(cfg.seed_map.items())),
            "source_artifact_hashes": {
                "raw/predictions.csv": _stable_digest(
                    prediction_rows,
                    (
                        "experiment_id",
                        "run_id",
                        "model_id",
                        "sigma",
                        "noise_draw",
                        "sample_id",
                        "y_true",
                        "y_pred",
                        "score_pos",
                        "correct",
                        "noise_seed",
                        "model_seed",
                        "checkpoint_hash",
                    ),
                ),
                "raw/run_summaries.csv": _stable_digest(
                    summary_rows,
                    (
                        "experiment_id",
                        "run_id",
                        "model_id",
                        "sigma",
                        "noise_draw",
                        "n_samples",
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "auc",
                        "iou_class0",
                        "iou_class1",
                        "macro_iou",
                        "tn",
                        "fp",
                        "fn",
                        "tp",
                        "brier",
                        "ece",
                        "seed",
                        "config_hash",
                    ),
                ),
            },
            "bootstrap_seed": boot_seed,
            "data_manifest_hash": str(config_payload.get("data_manifest_hash", "")),
        },
        "runs": [
            {name: _value_for_json(row.get(name)) for name in METRICS_RUN_COLUMNS}
            for row in sorted(summary_rows, key=_row_key)
        ],
        "aggregates": aggregates,
        "figures": _figure_descriptors(aggregates),
        **tables,
        "table_status": status,
        "warnings": warnings,
    }
    payload = _stats.round_floats(payload)

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    target = metrics_dir / "metrics.json"
    target.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    flat_rows = []
    for name, entry in sorted(aggregates.items()):
        for metric_name in ("accuracy", "f1", "macro_iou", "auc", "delta_accuracy_from_clean"):
            summary = entry.get(metric_name)
            if not isinstance(summary, dict):
                continue
            flat_rows.append(
                {
                    "experiment_id": entry["experiment_id"],
                    "model_id": entry["model_id"],
                    "sigma": entry["sigma"],
                    "metric": metric_name,
                    "n": summary.get("n"),
                    "mean": summary.get("mean"),
                    "std": summary.get("std"),
                    "ci95_low": summary.get("ci95_low"),
                    "ci95_high": summary.get("ci95_high"),
                }
            )
    _write_csv(
        metrics_dir / "metrics.csv",
        ("experiment_id", "model_id", "sigma", "metric", "n", "mean", "std", "ci95_low", "ci95_high"),
        flat_rows,
    )
    return MetricsBundle(payload=payload, path=target)


def _value_for_json(value: Any) -> Any:
    """CSV cells arrive as strings; keep numeric fields typed in metrics.json."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if "." in str(value) or "e" in str(value).lower():
        return number
    return int(number) if number.is_integer() else number


# ---------------------------------------------------------------------------
# Stages 6-7: figures (reporting-owned no-op) and manifest
# ---------------------------------------------------------------------------
def render_figures(cfg: ResolvedConfig, output_dir: Path) -> list[FigureArtifact]:
    """Documented no-op: figures and captions are owned by ``reporting/``.

    The runner still writes ``caption_numbers`` into ``metrics.json`` so the
    reporting layer can render figures from that single source of truth.  No
    file is created here and no number is recomputed in plotting code.
    """
    return []


def write_manifest(
    cfg: ResolvedConfig,
    data: DataBundle,
    models: Mapping[str, Any],
    raw: Sequence[PredictionArtifact],
    metrics: MetricsBundle,
    figures: Sequence[FigureArtifact],
) -> RunManifest:
    """Write provenance, model/data indices, logs and the stage manifest."""
    from experiments.env import collect_env_manifest

    output_dir = cfg.output_dir
    stages = ["config", "data", "models", "raw", "metrics", "figures", "manifest"]
    env_manifest = collect_env_manifest(cfg.config)
    warnings = list(cfg.warnings) + list(data.warnings)
    warnings.extend(message for artifact in raw for message in artifact.warnings)

    provenance = {
        "schema_version": "1.0",
        "run_id": cfg.run_id,
        "experiment_id": cfg.experiment_id,
        "config_hash": cfg.config_hash,
        "master_seed": int(cfg.config.seed),
        "seed_map": dict(sorted(cfg.seed_map.items())),
        "python_version": env_manifest["python_version"],
        "packages": env_manifest["packages"],
        "git_commit_sha": env_manifest["git_commit_sha"],
        "env_checksum_sha256": env_manifest["checksum_sha256"],
        "data_manifest_hash": data.manifest_hash,
        "source_artifact_hashes": metrics.payload["provenance"]["source_artifact_hashes"],
        "legacy_qsvc_seed_mapping": {
            "legacy_random_state": ["run_idx*42", "run_idx*123", "run_idx*456"],
            "seed_offset": int(cfg.config.seed_offset),
        },
        "warnings": warnings,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_manifest_hash": data.manifest_hash,
                "unit_count": len(data.units),
                "sample_counts": dict(sorted(data.sample_counts.items())),
                "warnings": list(data.warnings),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "index.json").write_text(
        json.dumps(
            {
                model_id: {
                    "adapter": type(predictor).__name__,
                    "checkpoint_hash": str(
                        getattr(predictor, "checkpoint_hash", "not_available")
                    ),
                }
                for model_id, predictor in sorted(models.items())
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "runner.log").write_text(
        "\n".join(
            [
                f"run_id={cfg.run_id}",
                f"experiment_id={cfg.experiment_id}",
                f"config_hash={cfg.config_hash}",
                f"stages={','.join(stages)}",
                f"skipped_units={sum(1 for artifact in raw if artifact.skipped)}",
                *[f"WARNING {warning}" for warning in warnings],
                f"finished_utc={_utc_now()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    outputs: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            outputs[str(path.relative_to(output_dir)).replace("\\", "/")] = _hash_file(path)
    manifest = RunManifest(
        run_id=cfg.run_id,
        output_dir=output_dir,
        stages=tuple(stages),
        outputs=outputs,
        warnings=tuple(warnings),
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest.as_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_run(output_dir: str | Path, *, strict: bool = False) -> list[str]:
    """Read-only provenance/schema gate. Returns a list of violations."""
    root = Path(output_dir)
    violations: list[str] = []
    required = (
        "config.json",
        "provenance.json",
        "manifest.json",
        "raw/predictions.csv",
        "raw/run_summaries.csv",
        "metrics/metrics.json",
    )
    for name in required:
        if not (root / name).is_file():
            violations.append(f"missing:{name}")
    if violations:
        return violations
    config_payload = json.loads((root / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metrics_payload = json.loads((root / "metrics/metrics.json").read_text(encoding="utf-8"))

    for name in ("config_hash", "master_seed", "seed_map", "source_artifact_hashes"):
        if name not in provenance:
            violations.append(f"provenance_missing:{name}")
    if provenance.get("config_hash") != config_payload.get("config_hash"):
        violations.append("provenance_config_hash_mismatch")
    stages = manifest.get("stages", [])
    if {"raw", "metrics"}.issubset(stages):
        if stages.index("raw") > stages.index("metrics"):
            violations.append("raw_after_metrics")
    else:
        violations.append("stage_order_invalid")

    predictions = _read_csv(root / "raw" / "predictions.csv")
    summaries = _read_csv(root / "raw" / "run_summaries.csv")
    for column in ("y_true", "y_pred", "score_pos", "sample_id", "noise_seed"):
        if column not in (predictions[0] if predictions else {}):
            violations.append(f"prediction_column_missing:{column}")
    for column in ("n_samples", "accuracy", "sigma", "config_hash"):
        if column not in (summaries[0] if summaries else {}):
            violations.append(f"summary_column_missing:{column}")
    for row in predictions:
        if row.get("y_true") in (None, ""):
            violations.append("null_y_true")
            break
        if row.get("y_pred") in (None, ""):
            violations.append("null_y_pred")
            break
    for top_level in (
        "schema_version",
        "runs",
        "aggregates",
        "figures",
        "warnings",
        "table_status",
        *REPORTING_TABLE_KEYS,
    ):
        if top_level not in metrics_payload:
            violations.append(f"metrics_missing:{top_level}")
    for table_id in REPORTING_TABLE_KEYS:
        if not isinstance(metrics_payload.get(table_id), list):
            violations.append(f"reporting_table_not_list:{table_id}")
    if not isinstance(metrics_payload.get("table_status"), dict):
        violations.append("table_status_invalid")
    for name, entry in metrics_payload.get("aggregates", {}).items():
        accuracy = entry.get("accuracy")
        if not isinstance(accuracy, dict) or not {
            "n", "values", "mean", "std", "ci95_low", "ci95_high"
        }.issubset(accuracy):
            violations.append(f"aggregate_schema_invalid:{name}")
    for comparison in metrics_payload.get("comparisons", []):
        if "p_holm" not in comparison or "decision" not in comparison:
            violations.append("comparison_schema_invalid")
    if strict and (metrics_payload.get("warnings") or provenance.get("warnings")):
        violations.append("strict_warnings_present")
    return sorted(set(violations))


# ---------------------------------------------------------------------------
# Orchestration and CLI
# ---------------------------------------------------------------------------
def run_experiment(cfg: ResolvedConfig, spec: ExperimentSpec) -> RunManifest:
    """Execute the full stage chain, persisting raw predictions first."""
    if cfg.dry_run:
        raise RunnerError("run_experiment called with dry_run=True; use the CLI plan instead")
    raw_predictions = cfg.output_dir / "raw" / "predictions.csv"
    if raw_predictions.exists() and not cfg.force:
        raise RunExistsError(
            f"run_id {cfg.run_id} already exists at {cfg.output_dir}; pass --force"
        )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "schema_version": "1.0",
        "config": cfg.config.as_dict(),
        "experiment_id": cfg.experiment_id,
        "run_id": cfg.run_id,
        "config_hash": cfg.config_hash,
        "seed_map": dict(sorted(cfg.seed_map.items())),
        "warnings": list(cfg.warnings),
        "options": {
            "bootstrap_n": int(cfg.bootstrap_n),
            "bootstrap_seed": int(cfg.seed_map.get("bootstrap", cfg.config.seed)),
            "sigma_grid": list(cfg.sigma_grid),
            "noise_draws": int(cfg.noise_draws),
            "seeds_per_model": int(cfg.seeds_per_model),
            "comparison_pairs": [list(pair) for pair in cfg.comparison_pairs],
        },
    }
    (cfg.output_dir / "config.json").write_text(
        json.dumps(config_payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    data = load_data(cfg, spec)
    config_payload["data_manifest_hash"] = data.manifest_hash
    config_payload["dataset_table"] = dataset_table_from_manifest(cfg, spec)
    (cfg.output_dir / "config.json").write_text(
        json.dumps(config_payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    models = build_models(cfg, spec)
    config_payload["pipeline_table"] = pipeline_table_from_models(models)
    raw = predict_all(cfg, data, models, spec)
    if not (cfg.output_dir / "raw" / "predictions.csv").is_file():
        raise RunnerError("raw predictions were not persisted before metrics")
    config_payload["warnings"] = list(config_payload["warnings"]) + [
        message for artifact in raw for message in artifact.warnings
    ]
    (cfg.output_dir / "config.json").write_text(
        json.dumps(config_payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metrics = compute_metrics(cfg, cfg.output_dir)
    figures = render_figures(cfg, cfg.output_dir)
    return write_manifest(cfg, data, models, raw, metrics, figures)


def _print_plan(cfg: ResolvedConfig) -> None:
    plan = {
        "command": cfg.command,
        "experiment_id": cfg.experiment_id,
        "run_id": cfg.run_id,
        "output_dir": str(cfg.output_dir),
        "config_hash": cfg.config_hash,
        "seed_map": dict(sorted(cfg.seed_map.items())),
        "sigma_grid": list(cfg.sigma_grid),
        "noise_draws": cfg.noise_draws,
        "runs": cfg.runs if cfg.runs is not None else cfg.config.qsvc_runs,
        "warnings": list(cfg.warnings),
    }
    print(json.dumps(plan, sort_keys=True, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit status; writes nothing on dry-run."""
    try:
        cfg = resolve_config(argv)
        if cfg.command == "run":
            if cfg.dry_run:
                _print_plan(cfg)
                return 0
            spec = ExperimentSpec(experiment_id=cfg.experiment_id)
            manifest = run_experiment(cfg, spec)
            print(f"run {manifest.run_id}: {' '.join(manifest.stages)}")
            if cfg.strict and manifest.warnings:
                for warning in manifest.warnings:
                    print(f"WARNING {warning}", file=sys.stderr)
                return 1
            return 0
        if cfg.command == "metrics":
            bundle = compute_metrics(cfg, cfg.output_dir)
            print(f"wrote {bundle.path}")
            return 0
        if cfg.command == "figures":
            artifacts = render_figures(cfg, cfg.output_dir)
            print(f"figures stage owned by reporting layer; produced {len(artifacts)} artifacts")
            return 0
        violations = verify_run(cfg.output_dir, strict=cfg.strict)
        for violation in violations:
            print(f"VIOLATION {violation}", file=sys.stderr)
        print(f"verify: {len(violations)} violation(s)")
        return 1 if violations else 0
    except (RunnerError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
