"""Train the hybrid transfer QNN on leak-free real features (QML-D02).

All dataset locations come from ``KAGGLE_DATA_DIR`` / ``NINJA_ANN_DIR`` /
``NINJA_IMG_DIR``; no developer path is embedded.  Features are produced by the
same helpers as ``scripts/run_real_benchmark.py`` (``prepare_ninja_subset``,
``cap_ood_by_group``, ``make_feature_loader``), with the frozen feature backbone
selected through ``qmlpipe.plant_backbone.build_backbone_extractor``, so the OOD
crop cap is applied per source-image group and no crop leaks across splits.

The heavy dependencies (torch, torchvision, PennyLane, the dataset modules) are
imported only inside the real-run path.  ``--dry-run`` therefore resolves and
validates the plan and prints it without importing a simulator or writing a
single file, which keeps it usable in the central verifier's minimal
environment.

QHEAD-D02 -- matched labelled adaptation for the QNN
----------------------------------------------------
The matched classical "mixed" arm trains on 50 labelled target-domain (Ninja)
samples, while a bare QNN arm previously received zero.  ``--adaptation-samples``
mixes labelled samples from ``manifest.splits[Split.ADAPTATION]`` into the TRAIN
tensor *only*.  The OOD test split is never used as a source, never featurised,
and never used for checkpoint/threshold selection; a group (source-image) level
disjointness assertion is enforced before any feature extraction.  The default
is deliberately **0** so a bare invocation reproduces the previous QNN numbers;
the matched protocol is opt-in via ``--adaptation-samples 50``.  No improved OOD
performance is claimed by this recipe change -- it only makes the comparison
matched under Methodology V2 section 4.
"""
from __future__ import annotations

import argparse
from importlib import metadata as _metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Overridable QMLConfig fields.  ``feature_dim`` is intentionally absent: it is
# derived from ``qubits`` and any explicit override is rejected as a config
# error rather than silently ignored or honoured.
CONFIG_KEYS = frozenset(
    {
        "qubits",
        "layers",
        "embedding",
        "ansatz",
        "device_name",
        "shots",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "patience",
        "optimizer",
        "output_dir",
        "seed",
        "noise_depolarizing",
        "projection_dim",
        "classical_dim",
        "backbone",
        "loss",
        "class_weight",
        "focal_gamma",
        "focal_alpha",
        "early_metric",
        "early_min_delta",
        "temperature_calibration",
        "balanced_sampler",
        # QHEAD-D02 matched-adaptation knobs (not QMLConfig fields; consumed by
        # the data path and recorded in provenance, ignored by _build_config).
        "adaptation_samples",
        "adaptation_per_class_mix",
        "adaptation_seed",
    }
)

# Local mirrors of QMLConfig's allowed values so dry-run validation needs no
# heavy import (the config module itself is standard-library only).
ALLOWED_BACKBONES = frozenset({"plantnet", "imagenet"})
ALLOWED_LOSSES = frozenset({"ce", "focal"})
ALLOWED_CLASS_WEIGHTS = frozenset({"none", "balanced"})
ALLOWED_EARLY_METRICS = frozenset({"val_loss", "balanced_accuracy", "macro_f1"})

# QHEAD-D01 CLI defaults.  These deliberately differ from the QMLConfig
# dataclass defaults (which preserve legacy behaviour for reproducibility).
# Each choice is written to provenance.json with its rationale so no balanced
# setting is mistaken for a legacy result.
RECIPE_DEFAULTS = {
    "loss": "focal",
    "class_weight": "balanced",
    "focal_gamma": 2.0,
    "focal_alpha": None,
    "early_metric": "balanced_accuracy",
    "early_min_delta": 0.0,
    "temperature_calibration": True,
    "balanced_sampler": False,
}
RECIPE_DEFAULT_NOTES = {
    "loss": "focal down-weights easy examples; --loss ce --class-weight balanced gives standard balanced CE",
    "class_weight": "weights n/(K*count) are computed from TRAIN labels only",
    "early_metric": "balanced_accuracy prevents a collapsed single-class model from being checkpointed",
    "temperature_calibration": "temperature (+ logit adjustment) fitted on ID_VAL only; never on OOD",
    "balanced_sampler": "off by default to avoid triple correction; opt in with --balanced-sampler",
}

# QHEAD-D02: matched labelled-adaptation recipe.  ``Split.ADAPTATION`` is the
# ONLY permitted source.  String constants are kept here (stdlib-only module)
# and cross-checked against the real ``Split`` enum inside the data path.
ADAPTATION_SPLIT_NAME = "adaptation"
OOD_TEST_SPLIT_NAME = "ood_test"
ADAPTATION_LABEL_SOURCE = (
    "ground-truth class labels from DatasetNinja JSON box annotations, carried "
    "by manifest.splits[Split.ADAPTATION].samples (target domain, labelled)"
)
ADAPTATION_DEFAULT_NOTE = (
    "--adaptation-samples defaults to 0 so a bare invocation reproduces the prior "
    "QNN recipe exactly. The matched classical mixed protocol uses 50 labelled "
    "Ninja adaptation samples (qsvc_target_adaptation_per_run=50), so pass "
    "--adaptation-samples 50 to match it; this is recorded in provenance."
)
ADAPTATION_RECIPE_NOTE = (
    "QHEAD-D02: labelled ADAPTATION-split target-domain samples are mixed into "
    "TRAIN only, through the same frozen feature loader; ID_VAL and OOD_TEST "
    "loaders are untouched and OOD_TEST is never used as a mixing source. "
    "Source-image (group) disjointness from OOD_TEST is asserted. This makes the "
    "training budget matched to the classical mixed arm; it is NOT a claim of "
    "improved OOD generalisation."
)

# Raw (pre-projection) feature width emitted by each frozen backbone.  The
# feature-fed QNN projection consumes this many inputs, so ``classical_dim``
# defaults to the selected backbone's width (overridable via config-json).
BACKBONE_FEATURE_DIM = {"plantnet": 512, "imagenet": 1280}

OUTPUT_FILES = ("history.json", "metrics.json", "checkpoints/best.pt", "provenance.json")


# ---------------------------------------------------------------------------
# CLI / configuration
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/train_qml.py",
        description="Train and evaluate the hybrid transfer QNN on real OOD features.",
    )
    parser.add_argument("--config-json", help="JSON file overriding QMLConfig fields")
    parser.add_argument("--qubits", type=int, help="number of amplitude-embedding qubits")
    parser.add_argument("--layers", type=int, help="ansatz layers")
    parser.add_argument("--embedding", help="feature embedding name (e.g. amplitude)")
    parser.add_argument("--device", help="quantum device name (e.g. default.qubit)")
    parser.add_argument("--epochs", type=int, help="training epochs")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--output-dir", help="run output directory")
    parser.add_argument(
        "--backbone",
        choices=("plantnet", "imagenet"),
        default=None,
        help="frozen feature backbone (default plantnet; may also be set via --config-json)",
    )
    parser.add_argument(
        "--loss",
        choices=sorted(ALLOWED_LOSSES),
        default=None,
        help="training objective (QML CLI default: focal; legacy: ce)",
    )
    parser.add_argument(
        "--class-weight",
        choices=sorted(ALLOWED_CLASS_WEIGHTS),
        default=None,
        help="per-class loss weights from TRAIN labels (default: balanced; legacy: none)",
    )
    parser.add_argument(
        "--early-metric",
        choices=sorted(ALLOWED_EARLY_METRICS),
        default=None,
        help="checkpoint/early-stop metric (default: balanced_accuracy; legacy: val_loss)",
    )
    calibration_group = parser.add_mutually_exclusive_group()
    calibration_group.add_argument(
        "--temperature-calibration",
        dest="temperature_calibration",
        action="store_true",
        default=None,
        help="fit temperature/logit adjustment on ID_VAL only (default: on)",
    )
    calibration_group.add_argument(
        "--no-temperature-calibration",
        dest="temperature_calibration",
        action="store_false",
        default=None,
        help="disable ID_VAL calibration (legacy behaviour)",
    )
    sampler_group = parser.add_mutually_exclusive_group()
    sampler_group.add_argument(
        "--balanced-sampler",
        dest="balanced_sampler",
        action="store_true",
        default=None,
        help="use a class-balanced sampler built from TRAIN labels (default: off)",
    )
    sampler_group.add_argument(
        "--no-balanced-sampler",
        dest="balanced_sampler",
        action="store_false",
        default=None,
        help="disable the class-balanced sampler",
    )
    parser.add_argument(
        "--adaptation-samples",
        type=int,
        default=0,
        metavar="N",
        help=(
            "total labelled target-domain adaptation samples to mix into TRAIN "
            "(default 0 = no adaptation, reproduces prior QNN numbers; pass 50 "
            "to match the classical mixed protocol). Source is always "
            "Split.ADAPTATION, never Split.OOD_TEST."
        ),
    )
    parser.add_argument(
        "--adaptation-per-class",
        type=int,
        default=None,
        metavar="N",
        help=(
            "optional per-class adaptation quota; overrides --adaptation-samples "
            "with 2*N total (balanced across the two label classes)."
        ),
    )
    parser.add_argument(
        "--adaptation-seed",
        type=int,
        default=None,
        metavar="N",
        help="seed for deterministic adaptation subsampling (default: the run seed)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny, group-disjoint subset and a single epoch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and validate the plan but write nothing",
    )
    return parser


def _load_overrides(path: str | None) -> dict:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--config-json must contain a JSON object")
    return {str(key): value for key, value in payload.items()}


def _resolve_plan(args: argparse.Namespace) -> tuple[dict, list[str]]:
    """Merge CLI flags over --config-json over defaults; return plan and warnings."""
    overrides = _load_overrides(args.config_json)
    unknown = sorted(set(overrides) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(
            "unknown --config-json key(s): " + ", ".join(unknown)
            + "; derived fields such as feature_dim cannot be overridden"
        )
    warnings: list[str] = []

    def choose(cli_value: Any, key: str, default: Any) -> Any:
        if cli_value is not None:
            return cli_value
        if key in overrides:
            return overrides[key]
        return default

    qubits = int(choose(args.qubits, "qubits", 10))
    feature_dim = 2 ** qubits
    backbone = str(choose(args.backbone, "backbone", "plantnet"))
    plan: dict[str, Any] = {
        "qubits": qubits,
        "feature_dim": feature_dim,
        "projection_dim": int(choose(None, "projection_dim", feature_dim)),
        "classical_dim": int(
            choose(None, "classical_dim", BACKBONE_FEATURE_DIM.get(backbone, 1280))
        ),
        "layers": int(choose(args.layers, "layers", 2)),
        "embedding": str(choose(args.embedding, "embedding", "amplitude")),
        "ansatz": str(choose(None, "ansatz", "strongly_entangling")),
        "device_name": str(choose(args.device, "device_name", "default.qubit")),
        "shots": choose(None, "shots", None),
        "batch_size": int(choose(None, "batch_size", 2 if args.smoke else 32)),
        "epochs": 1 if args.smoke else int(choose(args.epochs, "epochs", 15)),
        "learning_rate": float(choose(args.lr, "learning_rate", 1e-3)),
        "weight_decay": float(choose(None, "weight_decay", 0.0)),
        "patience": int(choose(None, "patience", 3)),
        "optimizer": str(choose(None, "optimizer", "adamw")),
        "output_dir": str(
            choose(args.output_dir, "output_dir", os.getenv("QC_OUTPUT_DIR", "results/qml_run"))
        ),
        "seed": int(choose(None, "seed", os.getenv("QC_SEED", "42"))),
        "noise_depolarizing": choose(None, "noise_depolarizing", None),
        "backbone": backbone,
        "loss": str(choose(args.loss, "loss", RECIPE_DEFAULTS["loss"])).lower(),
        "class_weight": str(
            choose(args.class_weight, "class_weight", RECIPE_DEFAULTS["class_weight"])
        ).lower(),
        "focal_gamma": float(choose(None, "focal_gamma", RECIPE_DEFAULTS["focal_gamma"])),
        "focal_alpha": (
            None
            if choose(None, "focal_alpha", RECIPE_DEFAULTS["focal_alpha"]) is None
            else float(choose(None, "focal_alpha", RECIPE_DEFAULTS["focal_alpha"]))
        ),
        "early_metric": str(
            choose(args.early_metric, "early_metric", RECIPE_DEFAULTS["early_metric"])
        ).lower(),
        "early_min_delta": float(
            choose(None, "early_min_delta", RECIPE_DEFAULTS["early_min_delta"])
        ),
        "temperature_calibration": bool(
            choose(
                args.temperature_calibration,
                "temperature_calibration",
                RECIPE_DEFAULTS["temperature_calibration"],
            )
        ),
        "balanced_sampler": bool(
            choose(
                args.balanced_sampler,
                "balanced_sampler",
                RECIPE_DEFAULTS["balanced_sampler"],
            )
        ),
        # QHEAD-D02: default 0 keeps the no-adaptation recipe byte-identical.
        "adaptation_samples": int(choose(args.adaptation_samples, "adaptation_samples", 0)),
        "adaptation_per_class_mix": (
            None
            if choose(args.adaptation_per_class, "adaptation_per_class_mix", None) is None
            else int(choose(args.adaptation_per_class, "adaptation_per_class_mix", None))
        ),
        "adaptation_seed": (
            None
            if choose(args.adaptation_seed, "adaptation_seed", None) is None
            else int(choose(args.adaptation_seed, "adaptation_seed", None))
        ),
    }
    # A per-class quota implies the total budget (2*N) unless --adaptation-samples
    # already set one explicitly.  This keeps per-class flags self-contained.
    if plan["adaptation_per_class_mix"] is not None:
        implied_total = 2 * int(plan["adaptation_per_class_mix"])
        if plan["adaptation_samples"] == 0:
            plan["adaptation_samples"] = implied_total
    if args.smoke:
        warnings.append("smoke mode: subset capped, epochs forced to 1")
    warnings.append(
        "QHEAD-D01 CLI balance defaults: "
        + ", ".join(f"{key}={plan[key]}" for key in RECIPE_DEFAULT_NOTES)
        + ". Legacy recipe: --loss ce --class-weight none --early-metric val_loss "
        "--no-temperature-calibration --no-balanced-sampler."
    )
    warnings.append(
        "legacy-restore flags for reproducibility: --loss ce --class-weight none "
        "--early-metric val_loss --no-temperature-calibration --no-balanced-sampler"
    )
    for key, note in RECIPE_DEFAULT_NOTES.items():
        warnings.append(f"default rationale [{key}]: {note}")
    # QHEAD-D02 adaptation recipe: documented like every other default, and
    # explicit that no OOD improvement is implied by mixing labelled data.
    warnings.append(ADAPTATION_RECIPE_NOTE)
    warnings.append(f"default rationale [adaptation_samples]: {ADAPTATION_DEFAULT_NOTE}")
    return plan, warnings


def _validate_plan(plan: dict) -> list[str]:
    """Local, environment-free validation used by both dry-run and real runs."""
    problems: list[str] = []
    if int(plan["qubits"]) < 1:
        problems.append("qubits must be >= 1")
    if int(plan["feature_dim"]) != 2 ** int(plan["qubits"]):
        problems.append("feature_dim must equal 2**qubits for amplitude embedding")
    if int(plan["projection_dim"]) != 2 ** int(plan["qubits"]):
        problems.append("projection_dim must equal 2**qubits for amplitude embedding")
    if int(plan["classical_dim"]) < 1:
        problems.append("classical_dim must be positive")
    if int(plan["layers"]) < 1:
        problems.append("layers must be >= 1")
    if int(plan["epochs"]) < 1:
        problems.append("epochs must be >= 1")
    if int(plan["batch_size"]) < 1:
        problems.append("batch_size must be >= 1")
    if float(plan["learning_rate"]) <= 0:
        problems.append("learning_rate must be positive")
    if int(plan["patience"]) < 1:
        problems.append("patience must be >= 1")
    if str(plan["embedding"]).strip() == "":
        problems.append("embedding must be a non-empty string")
    if str(plan["backbone"]) not in ALLOWED_BACKBONES:
        problems.append(f"backbone must be one of {sorted(ALLOWED_BACKBONES)}")
    if str(plan["loss"]) not in ALLOWED_LOSSES:
        problems.append(f"loss must be one of {sorted(ALLOWED_LOSSES)}")
    if str(plan["class_weight"]) not in ALLOWED_CLASS_WEIGHTS:
        problems.append(f"class_weight must be one of {sorted(ALLOWED_CLASS_WEIGHTS)}")
    if str(plan["early_metric"]) not in ALLOWED_EARLY_METRICS:
        problems.append(f"early_metric must be one of {sorted(ALLOWED_EARLY_METRICS)}")
    if float(plan["focal_gamma"]) < 0:
        problems.append("focal_gamma must be non-negative")
    if plan["focal_alpha"] is not None and not (0 <= float(plan["focal_alpha"]) <= 1):
        problems.append("focal_alpha must be None or in [0, 1]")
    if float(plan["early_min_delta"]) < 0:
        problems.append("early_min_delta must be non-negative")

    # QHEAD-D02 adaptation knobs.
    total = int(plan["adaptation_samples"])
    per_class = plan["adaptation_per_class_mix"]
    if total < 0:
        problems.append("adaptation_samples must be non-negative")
    if per_class is not None and int(per_class) < 0:
        problems.append("adaptation_per_class_mix must be non-negative or None")
    if plan["adaptation_seed"] is not None:
        try:
            int(plan["adaptation_seed"])
        except (TypeError, ValueError):
            problems.append("adaptation_seed must be an integer or None")
    if per_class is not None:
        expected_total = 2 * int(per_class)
        if total not in (0, expected_total):
            problems.append(
                f"adaptation_samples={total} conflicts with "
                f"adaptation_per_class_mix={per_class} (expected 0 or {expected_total})"
            )
    return problems


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-learn",
        "pennylane",
        "pennylane-lightning",
    ):
        try:
            versions[name] = str(_metadata.version(name))
        except Exception:
            versions[name] = "not installed"
    return versions


def _git_commit_hash() -> str:
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    sha = proc.stdout.strip() if proc.returncode == 0 else ""
    return sha if sha else "unavailable"


# ---------------------------------------------------------------------------
# Real-run helpers (imported lazily)
# ---------------------------------------------------------------------------
def _build_config(plan: dict):
    """Instantiate QML-D01's QMLConfig from the resolved plan."""
    from dataclasses import fields

    from qmlpipe.config import QMLConfig

    try:
        accepted = {field.name for field in fields(QMLConfig)}
    except TypeError:  # not a dataclass: pass exactly the contracted keys
        accepted = set(CONFIG_KEYS)
    kwargs = {key: value for key, value in plan.items() if key in accepted}
    cfg = QMLConfig(**kwargs)
    validate = getattr(cfg, "validate", None)
    if callable(validate):
        try:
            problems = validate()
        except AssertionError as exc:  # QML-D01 hard-fails on invariant violations
            raise ValueError(f"QMLConfig validation failed: {exc}") from exc
        if problems:
            raise ValueError("QMLConfig validation failed: " + "; ".join(map(str, problems)))
    return cfg


def _feature_fed_transfer_model(base: Any, cfg: Any) -> Any:
    """Rewire D01's transfer QNN to consume cached feature-loader vectors.

    ``build_transfer_qnn`` embeds a frozen EfficientNet that expects images.
    The CLI already produced leak-free ``classical_dim``-wide features with
    ``make_feature_loader``, so re-extracting them inside the model would be
    redundant and shape-incompatible.  Replacing only that extractor with the
    same L2-normalised projection keeps D01's QNN, angle adapter and 2-logit
    head exactly as built; the substitution is disclosed in the provenance.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    classical_dim = int(getattr(cfg, "classical_dim", 1280))
    projection_dim = int(getattr(cfg, "projection_dim", 2 ** int(cfg.qubits)))
    extractor = getattr(base, "extractor", None)
    source_projection = getattr(extractor, "projection", None)
    if isinstance(source_projection, nn.Module):
        projection = source_projection
    else:  # pragma: no cover - D01's builders expose `.projection`
        projection = nn.Linear(classical_dim, projection_dim)
    for parameter in projection.parameters():
        parameter.requires_grad_(False)
    adapter = getattr(base, "adapter", None)
    if not isinstance(adapter, nn.Module):
        adapter = nn.Identity()
    qnn = base.qnn

    class _FeatureFedTransfer(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = projection
            self.adapter = adapter
            self.qnn = qnn

        def forward(self, x):
            features = F.normalize(self.projection(x), p=2, dim=1, eps=1e-8)
            return self.qnn(self.adapter(features))

    return _FeatureFedTransfer()


def _data_plan(smoke: bool) -> dict:
    """Resolve the real-data scope from the environment (no I/O; unit-testable).

    ``adaptation_per_class`` is kept equal to the OOD ``per_class`` cap so the
    smoke run never asks ``build_splits`` for the default 200/200 adaptation
    quota from a handful of linked source images.
    """
    subset_files = int(os.getenv("QC_NINJA_SUBSET_FILES", "8" if smoke else "1200"))
    per_class = int(os.getenv("QC_OOD_PER_CLASS", "2" if smoke else "200"))
    return {
        "subset_files": subset_files,
        "per_class": per_class,
        "adaptation_per_class": per_class,
    }


def _backbone_provenance(name: str, feature_dim: int) -> dict:
    """Build the provenance block for the selected frozen feature backbone.

    ``feature_dim`` is the *detected* width of the extracted vectors (read from
    the real feature tensor), not a hardcoded claim.  PlantNet-specific
    repository/weights/license metadata is attached only for that backbone; the
    heavy ``plant_backbone`` module is imported lazily here so ``--dry-run``
    never touches torch/HF.
    """
    info: dict[str, Any] = {
        "name": str(name),
        "feature_dim": int(feature_dim),
        "frozen": True,
        "note": "Feature backbone is frozen (no weights are updated during training).",
    }
    if str(name) == "plantnet":
        from qmlpipe import plant_backbone

        info["repo_id"] = plant_backbone.PLANTNET_REPO_ID
        info["weights_file"] = plant_backbone.PLANTNET_WEIGHTS_FILE
        info["license"] = "Apache-2.0"
    return info


# ---------------------------------------------------------------------------
# QHEAD-D02: matched labelled-adaptation mixing.
#
# The first three helpers are stdlib-only so they can be unit-tested with
# synthetic Sample-like objects without importing torch / the dataset layer.
# Only ``_mix_adaptation_into_train`` imports torch/numpy, and does so lazily.
# ---------------------------------------------------------------------------
def _split_samples_by_name(manifest: Any, name: str) -> tuple[Any, ...]:
    """Return samples of a split identified by enum value / string name.

    Dependency-free (plain string keys work) so synthetic manifests are
    testable.  The real path cross-checks these names against ``Split``.
    """
    found = None
    for key, record in manifest.splits.items():
        if str(getattr(key, "value", key)) == name:
            found = record
            break
    if found is None:
        available = sorted(str(getattr(k, "value", k)) for k in manifest.splits)
        raise KeyError(
            f"manifest has no split named {name!r} (available: {available}); "
            "refusing to guess an adaptation source"
        )
    return tuple(found.samples)


def _adaptation_pool(manifest: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Return (ADAPTATION samples, OOD_TEST samples) from the manifest.

    OOD_TEST is returned ONLY so the disjointness gate can inspect it; it is
    never a mixing source and is never featurised.
    """
    adaptation = _split_samples_by_name(manifest, ADAPTATION_SPLIT_NAME)
    ood = _split_samples_by_name(manifest, OOD_TEST_SPLIT_NAME)
    return adaptation, ood


def _source_image_id(sample: Any) -> str | None:
    """Source-image identity: explicit ``source_image_id`` else ``group_id``.

    In this repository ``Sample.group_id`` *is* the source-image key (raw-file
    hash for Kaggle, photo stem for Ninja), so both names resolve to the same
    physical photograph.
    """
    for attr in ("source_image_id", "group_id"):
        value = getattr(sample, attr, None)
        if value is not None and str(value) != "":
            return str(value)
    return None


def _select_adaptation_samples(
    samples: Sequence[Any],
    total: int,
    seed: int,
    per_class: int | None = None,
) -> tuple[list[Any], dict]:
    """Deterministically select labelled adaptation samples for TRAIN mixing.

    Selection is seeded and input-order independent: samples are put in
    canonical ``sample_id`` order, then shuffled with ``random.Random(seed)``.
    The default allocation is class-balanced (half per label); a class
    shortfall is reallocated to the other class up to availability.  Passing
    ``per_class`` requests exactly N per label (2N total).
    """
    import random  # stdlib only: importable without numpy/torch

    total = int(total)
    seed = int(seed)
    ordered = sorted(samples, key=lambda s: str(getattr(s, "sample_id", "")))
    by_class: dict[int, list[Any]] = {0: [], 1: []}
    for sample in ordered:
        label = int(getattr(sample, "label", -1))
        if label in by_class:
            by_class[label].append(sample)

    if per_class is not None:
        per_class = int(per_class)
        quotas = {0: per_class, 1: per_class}
        total = 2 * per_class
    else:
        base, extra = divmod(total, 2)
        quotas = {0: base + extra, 1: base}
        for label in (0, 1):
            shortfall = max(0, quotas[label] - len(by_class[label]))
            if shortfall:
                other = 1 - label
                capacity = max(0, len(by_class[other]) - quotas[other])
                move = min(shortfall, capacity)
                quotas[label] -= move
                quotas[other] += move

    rng = random.Random(seed)
    chosen: list[Any] = []
    for label in (0, 1):
        pool = list(by_class[label])
        rng.shuffle(pool)
        chosen.extend(pool[: quotas[label]])
    chosen.sort(key=lambda s: str(getattr(s, "sample_id", "")))
    selected_class_counts = {0: 0, 1: 0}
    for sample in chosen:
        label = int(getattr(sample, "label", -1))
        selected_class_counts[label] = selected_class_counts.get(label, 0) + 1
    info = {
        "requested_total": int(total),
        "per_class": None if per_class is None else int(per_class),
        "selection_seed": seed,
        "selection_method": (
            "seeded shuffle of canonical sample_id order, class-balanced; "
            "deterministic and input-order independent"
        ),
        "available_class_counts": {k: len(v) for k, v in by_class.items()},
        "selected_class_counts": {k: int(selected_class_counts[k]) for k in (0, 1)},
        "selected": len(chosen),
    }
    return chosen, info


def _assert_adaptation_ood_disjoint(
    adaptation_samples: Sequence[Any], ood_samples: Sequence[Any]
) -> dict:
    """Fail loudly if a selected adaptation sample shares a source image with OOD.

    The primary key is the source-image/group id; ``sample_id``,
    ``content_hash`` and ``path`` are defence in depth.  Returns a provenance
    block with ``passed=True`` on success; raises ``RuntimeError`` naming the
    offenders on failure (never fails silently).
    """
    checked = ("source_image_id/group_id", "sample_id", "content_hash", "path")
    overlap: dict[str, list[str]] = {}
    for field in checked:
        if field == "source_image_id/group_id":
            a = {
                value
                for value in (_source_image_id(s) for s in adaptation_samples)
                if value is not None
            }
            b = {
                value
                for value in (_source_image_id(s) for s in ood_samples)
                if value is not None
            }
        else:
            a = {
                str(v)
                for v in (getattr(s, field, None) for s in adaptation_samples)
                if v is not None and str(v) != ""
            }
            b = {
                str(v)
                for v in (getattr(s, field, None) for s in ood_samples)
                if v is not None and str(v) != ""
            }
        shared = sorted(a & b)
        if shared:
            overlap[field] = shared
    if overlap:
        raise RuntimeError(
            "QHEAD-D02 adaptation/OOD disjointness assertion FAILED: "
            + "; ".join(f"{key} overlap={values[:5]}" for key, values in overlap.items())
            + ". Refusing to mix labelled adaptation samples into TRAIN."
        )
    return {
        "passed": True,
        "checked_fields": list(checked),
        "overlap": {},
        "adaptation_sample_count": len(adaptation_samples),
        "ood_sample_count": len(ood_samples),
        "note": (
            "Selected ADAPTATION samples share no source-image/group, sample_id, "
            "content_hash or path with the capped OOD_TEST split."
        ),
    }


def _adaptation_common_provenance(adaptation_samples: Sequence[Any], seed: int) -> dict:
    """Fields shared by the mixed and no-op adaptation provenance blocks."""
    return {
        "adaptation_source_split": ADAPTATION_SPLIT_NAME,
        "adaptation_label_source": ADAPTATION_LABEL_SOURCE,
        "adaptation_source_image_ids_all_available": sorted(
            {v for v in (_source_image_id(s) for s in adaptation_samples) if v is not None}
        ),
        "adaptation_source_image_ids": [],  # filled with selected ids when mixing
        "adaptation_sample_ids": [],  # filled with selected ids when mixing
        "ood_labels_used": False,
        "default_choice": ADAPTATION_DEFAULT_NOTE,
        "recipe_note": ADAPTATION_RECIPE_NOTE,
        "class_weights_and_sampler_source": (
            "augmented TRAIN tensor (Kaggle TRAIN + selected labelled ADAPTATION)"
        ),
        "val_loader_unchanged": True,
        "ood_loader_unchanged": True,
        "selection_seed": int(seed),
    }


def _mix_adaptation_into_train(
    X_train,
    y_train,
    *,
    adaptation_samples: Sequence[Any],
    ood_samples: Sequence[Any],
    feature_loader,
    cfg,
    total: int,
    seed: int,
    per_class: int | None = None,
) -> tuple[Any, Any, dict]:
    """Select, verify, featurise and append labelled adaptation samples to TRAIN.

    Only the returned ``X_train`` / ``y_train`` change; ID_VAL and OOD tensors
    (built by the caller) are never touched and OOD labels are never read.
    ``total == 0`` returns the input tensors unchanged (a true no-op) so the
    prior QNN recipe is reproduced exactly.
    """
    import numpy as np
    import torch

    requested = int(total)
    if requested == 0 and per_class in (None, 0):
        prov = {
            "adaptation_samples_used": 0,
            "requested_adaptation_samples": 0,
            "adaptation_per_class": None,
            "selection_method": "no-op (--adaptation-samples 0)",
            "available_class_counts": {
                k: sum(
                    1
                    for s in adaptation_samples
                    if int(getattr(s, "label", -1)) == k
                )
                for k in (0, 1)
            },
            "selected_class_counts": {0: 0, 1: 0},
            "disjointness_assertion": {
                "passed": True,
                "checked_fields": [
                    "source_image_id/group_id", "sample_id", "content_hash", "path"
                ],
                "overlap": {},
                "adaptation_sample_count": 0,
                "ood_sample_count": len(ood_samples),
                "note": "No adaptation samples selected; mixing disabled.",
            },
            "warnings": [
                "adaptation mixing disabled (--adaptation-samples 0): TRAIN tensors "
                "are unchanged and reproduce the prior QNN recipe exactly; pass "
                "--adaptation-samples 50 for the matched mixed protocol"
            ],
        }
        prov.update(_adaptation_common_provenance(adaptation_samples, seed))
        return X_train, y_train, prov

    selected, selection_info = _select_adaptation_samples(
        adaptation_samples, requested, seed, per_class
    )
    disjointness = _assert_adaptation_ood_disjoint(selected, ood_samples)
    if not selected:
        raise RuntimeError(
            "QHEAD-D02: adaptation_samples > 0 but the ADAPTATION split yielded "
            "no samples; refusing to train an unmatched arm"
        )

    features = np.asarray(
        feature_loader(selected, cfg, ADAPTATION_SPLIT_NAME), dtype=np.float32
    )
    labels = np.asarray([int(s.label) for s in selected], dtype=np.int64)
    X_new = torch.cat([X_train, torch.from_numpy(np.ascontiguousarray(features))], dim=0)
    y_new = torch.cat([y_train, torch.as_tensor(labels, dtype=torch.long)], dim=0)

    source_ids = sorted(
        {v for v in (_source_image_id(s) for s in selected) if v is not None}
    )
    adaptation_warnings = [
        f"adaptation mixing applied: {len(selected)} labelled ADAPTATION-split "
        f"samples appended to TRAIN only (source-image ids: {source_ids}); "
        "OOD_TEST untouched, ood_labels_used=false"
    ]
    if len(selected) < int(selection_info["requested_total"]):
        adaptation_warnings.append(
            f"adaptation shortfall: requested {selection_info['requested_total']} but only "
            f"{len(selected)} labelled ADAPTATION samples were available; the arm is "
            "NOT budget-matched to the classical mixed protocol"
        )
    prov = {
        "adaptation_samples_used": int(len(selected)),
        "requested_adaptation_samples": int(selection_info["requested_total"]),
        "adaptation_per_class": selection_info["per_class"],
        "selection_method": selection_info["selection_method"],
        "available_class_counts": selection_info["available_class_counts"],
        "selected_class_counts": selection_info["selected_class_counts"],
        "disjointness_assertion": disjointness,
        "warnings": adaptation_warnings,
    }
    prov.update(_adaptation_common_provenance(adaptation_samples, seed))
    prov["adaptation_source_image_ids"] = source_ids
    prov["adaptation_sample_ids"] = [str(getattr(s, "sample_id", "")) for s in selected]
    return X_new, y_new, prov


def _prepare_data(plan: dict, smoke: bool) -> dict:
    """Build leak-free train/val/OOD loaders and record the data provenance."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from data import build_splits
    from data.splits import Split
    from experiments.config import ExperimentConfig
    from qmlpipe.plant_backbone import build_backbone_extractor
    from scripts.run_real_benchmark import (
        SubsetManifest,
        cap_ood_by_group,
        make_feature_loader,
        prepare_ninja_subset,
    )

    data_paths = {
        "kaggle_dir": os.getenv("KAGGLE_DATA_DIR", ""),
        "ninja_ann_dir": os.getenv("NINJA_ANN_DIR", ""),
        "ninja_img_dir": os.getenv("NINJA_IMG_DIR", ""),
    }
    missing = [name for name, value in data_paths.items() if not value]
    if missing:
        raise RuntimeError(
            "missing dataset environment variables: " + ", ".join(sorted(missing))
        )

    data_scope = _data_plan(smoke)
    subset_files = data_scope["subset_files"]
    per_class = data_scope["per_class"]
    prepare_ninja_subset(subset_files)

    from dataclasses import replace

    cfg_exp = replace(
        ExperimentConfig.from_env(),
        output_dir=plan["output_dir"],
        n_qubits=int(plan["qubits"]),
        feature_dim=int(plan["feature_dim"]),
        qnn_layers=int(plan["layers"]),
        epochs=int(plan["epochs"]),
        batch_size=int(plan["batch_size"]),
        learning_rate=float(plan["learning_rate"]),
        seed=int(plan["seed"]),
        adaptation_per_class=int(data_scope["adaptation_per_class"]),
    )
    base_manifest = build_splits(cfg_exp, kaggle_root=data_paths["kaggle_dir"])
    ood_samples = cap_ood_by_group(
        base_manifest.splits[Split.OOD_TEST].samples, per_class
    )
    manifest = SubsetManifest(base_manifest, ood_samples)

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_name = str(plan["backbone"])
    extractor, preprocess = build_backbone_extractor(backbone_name, _build_config(plan))
    feature_loader = make_feature_loader(extractor, preprocess, torch_device)

    def tensor_split(samples: Any, name: str):
        if not samples:
            return None, None
        features = np.asarray(feature_loader(samples, cfg_exp, name), dtype=np.float32)
        labels = np.asarray([int(sample.label) for sample in samples], dtype=np.int64)
        return (
            torch.from_numpy(np.ascontiguousarray(features)),
            torch.as_tensor(labels, dtype=torch.long),
        )

    X_train, y_train = tensor_split(manifest.splits[Split.TRAIN].samples, "train")
    X_val, y_val = tensor_split(manifest.splits[Split.ID_VAL].samples, "id_val")
    X_ood, y_ood = tensor_split(manifest.splits[Split.OOD_TEST].samples, "ood_test")
    if X_train is None or y_train is None:
        raise RuntimeError("no Kaggle training samples were available")

    # A2: declared before any code that appends to it (adaptation + sampler).
    warnings: list[str] = []

    # QHEAD-D02: mix labelled ADAPTATION (target-domain) samples into TRAIN only.
    # Cross-check the string names used by the dependency-free pool helper against
    # the real Split enum, so a renamed split cannot silently redirect the source.
    if (
        str(Split.ADAPTATION.value) != ADAPTATION_SPLIT_NAME
        or str(Split.OOD_TEST.value) != OOD_TEST_SPLIT_NAME
    ):
        raise RuntimeError(
            "Split enum values no longer match the QHEAD-D02 adaptation source "
            "contract; refusing to mix"
        )
    adaptation_samples, ood_pool = _adaptation_pool(manifest)
    adaptation_seed = (
        int(plan["seed"]) if plan["adaptation_seed"] is None else int(plan["adaptation_seed"])
    )
    X_train, y_train, adaptation_provenance = _mix_adaptation_into_train(
        X_train,
        y_train,
        adaptation_samples=adaptation_samples,
        ood_samples=ood_pool,
        feature_loader=feature_loader,
        cfg=cfg_exp,
        total=int(plan["adaptation_samples"]),
        seed=adaptation_seed,
        per_class=plan["adaptation_per_class_mix"],
    )
    warnings.extend(adaptation_provenance["warnings"])

    generator = torch.Generator().manual_seed(int(plan["seed"]))
    train_dataset = TensorDataset(X_train, y_train)
    sampler = None
    if bool(plan["balanced_sampler"]):
        from qmlpipe.train import build_balanced_sampler

        sampler = build_balanced_sampler(y_train.numpy(), seed=int(plan["seed"]))
        if sampler is None:
            warnings.append(
                "balanced sampler requested but the TRAIN label set was empty; disabled"
            )
        elif str(plan["class_weight"]).lower() == "balanced":
            # A4: class weighting and resampling both correct the same imbalance.
            warnings.append(
                "class_weight=balanced combined with balanced_sampler=True applies "
                "class reweighting twice (double correction); this may over-correct "
                "the minority class and is recorded for audit"
            )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(plan["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator if sampler is None else None,
    )
    train_counts = np.bincount(y_train.numpy().astype(np.int64), minlength=2)
    from qmlpipe.losses import class_weights_from_labels

    derived_weights = class_weights_from_labels(
        y_train.numpy(), n_classes=2, mode=str(plan["class_weight"])
    )
    val_loader = None
    if X_val is not None and len(X_val) > 0:
        val_loader = DataLoader(
            TensorDataset(X_val, y_val), batch_size=int(plan["batch_size"]), shuffle=False
        )
    ood_loader = None
    if X_ood is not None and len(X_ood) > 0:
        ood_loader = DataLoader(
            TensorDataset(X_ood, y_ood), batch_size=int(plan["batch_size"]), shuffle=False
        )

    if val_loader is None:
        warnings.append("no held-out validation split; early stopping monitors train loss")
    if ood_loader is None:
        warnings.append("no OOD crops were available after the group cap")
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "ood_loader": ood_loader,
        "counts": {
            "train": int(len(X_train)),
            "val": 0 if X_val is None else int(len(X_val)),
            "ood": 0 if X_ood is None else int(len(X_ood)),
        },
        "train_labels": y_train,
        "train_class_counts": [int(v) for v in train_counts],
        "train_class_weights": [float(v) for v in derived_weights],
        "sampler": "weighted_random_sampler" if sampler is not None else "none",
        "data_paths": data_paths,
        "subset_files": subset_files,
        "ood_per_class": per_class,
        "adaptation_per_class": int(data_scope["adaptation_per_class"]),
        # QHEAD-D02 matched-adaptation provenance (source split, label source,
        # selected ids/groups, disjointness gate, ood_labels_used=false).
        "adaptation": adaptation_provenance,
        "backbone_name": backbone_name,
        "backbone_feature_dim": int(X_train.shape[1]),
        "backbone_provenance": _backbone_provenance(
            backbone_name, int(X_train.shape[1])
        ),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        plan, warnings = _resolve_plan(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    problems = _validate_plan(plan)
    if problems:
        for problem in problems:
            print(f"invalid configuration: {problem}", file=sys.stderr)
        return 2

    output_dir = Path(plan["output_dir"]).resolve()
    if args.dry_run:
        # No imports of torch/PennyLane/data here and no mkdir: write nothing.
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "plan": plan,
                    "warnings": warnings,
                    "output_dir": str(output_dir),
                    "files_to_write": list(OUTPUT_FILES),
                },
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
        return 0

    checkpoint_path = output_dir / "checkpoints" / "best.pt"
    try:
        cfg = _build_config(plan)
        from qmlpipe.heads import fit_calibration, write_calibration_to_checkpoint
        from qmlpipe.models import build_transfer_qnn
        from qmlpipe.train import evaluate, load_checkpoint, seed_everything, train

        seed_everything(int(plan["seed"]))
        bundle = _prepare_data(plan, args.smoke)
        model = _feature_fed_transfer_model(build_transfer_qnn(cfg), cfg)
        warnings.append(
            "transfer QNN's image extractor replaced by its own projection so the "
            "model consumes cached feature-loader vectors (no double extraction)"
        )
        summary = train(
            model,
            bundle["train_loader"],
            bundle["val_loader"],
            cfg,
            output_dir=output_dir,
            history_path=output_dir / "history.json",
            ckpt_path=checkpoint_path,
        )
        # Evaluate the SELECTED checkpoint, not the last epoch's weights.
        selected = load_checkpoint(model, checkpoint_path)
        calibration = {
            "status": "skipped",
            "reason": "calibration not requested",
            "fitted_on_split": "id_val",
            "ood_labels_used": False,
        }
        if bool(plan["temperature_calibration"]):
            if bundle["val_loader"] is None:
                calibration = {
                    "status": "skipped",
                    "reason": "no ID_VAL split available; calibration not applied",
                    "fitted_on_split": "id_val",
                    "ood_labels_used": False,
                }
            else:
                # ID_VAL only. The OOD loader is deliberately never passed here.
                calibration = fit_calibration(
                    model,
                    bundle["val_loader"],
                    bundle["train_labels"],
                    cfg,
                    n_classes=2,
                )
                if calibration.get("status") == "fitted":
                    write_calibration_to_checkpoint(model, checkpoint_path, calibration)
        ood_metrics = (
            evaluate(model, bundle["ood_loader"], cfg)
            if bundle["ood_loader"] is not None
            else {"status": "N/A", "reason": "no OOD crops available"}
        )
    except Exception as exc:  # never fabricate a result for a failed run
        print(f"training run failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2

    all_warnings = list(warnings) + list(bundle["warnings"])
    all_warnings.append(
        f"feature backbone '{bundle['backbone_name']}' is frozen; its weights "
        "are not updated during training"
    )
    all_warnings.append(
        "Calibration (if fitted) used ID_VAL only; a validation-only gain is NOT "
        "evidence of improved OOD performance. OOD metrics come from an untouched split."
    )
    adaptation_block = bundle["adaptation"]
    all_warnings.append(
        f"QHEAD-D02 adaptation budget: {adaptation_block['adaptation_samples_used']} "
        f"labelled ADAPTATION-split samples mixed into TRAIN (requested "
        f"{adaptation_block['requested_adaptation_samples']}); "
        f"source split={adaptation_block['adaptation_source_split']}; "
        f"OOD disjointness passed="
        f"{adaptation_block['disjointness_assertion']['passed']}; "
        "ood_labels_used=false. This matches the training budget only; no OOD gain is claimed."
    )
    commit = _git_commit_hash()
    _write_json(
        output_dir / "metrics.json",
        {
            "schema_version": "1.0",
            "experiment_id": "qmlpipe_transfer_qnn",
            "config": cfg.as_dict(),
            "commit_hash": commit,
            "training": {
                "best_epoch": summary["best_epoch"],
                "best_val_loss": summary["best_val_loss"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "best_val_metric": summary["best_val_metric"],
                "early_metric": summary["early_metric"],
                "loss": summary["loss"],
                "class_weight": summary["class_weight"],
                "class_weights_source": summary["class_weights_source"],
                "class_weights": summary["class_weights"],
                "train_class_counts": summary["train_class_counts"],
                "balanced_sampler": summary["balanced_sampler"],
                "epochs_run": summary["epochs_run"],
                "early_stopped": summary["early_stopped"],
                "optimizer": summary["optimizer"],
                "selected_checkpoint_epoch": selected["epoch"],
            },
            "calibration": calibration,
            "ood": ood_metrics,
            "adaptation": adaptation_block,
            "claim_boundary": (
                "Calibration (if any) was fitted on ID_VAL only. A validation-only "
                "improvement is not evidence of improved OOD performance; OOD numbers "
                "are reported separately from an untouched split. Mixing labelled "
                "ADAPTATION-split samples into TRAIN matches the training budget of the "
                "classical mixed arm; by itself it is not a claim of improved OOD "
                "generalisation."
            ),
            "warnings": all_warnings,
        },
    )
    _write_json(
        output_dir / "provenance.json",
        {
            "schema_version": "1.0",
            "config": cfg.as_dict(),
            "plan": plan,
            "commit_hash": commit,
            "python_version": platform.python_version(),
            "packages": _package_versions(),
            "data": {
                "paths": bundle["data_paths"],
                "counts": bundle["counts"],
                "ninja_subset_files": bundle["subset_files"],
                "ood_per_class": bundle["ood_per_class"],
                "adaptation_per_class": bundle["adaptation_per_class"],
                "train_class_counts": bundle["train_class_counts"],
            },
            # QHEAD-D02: full matched-adaptation provenance (source split, label
            # source, selected sample/group ids, disjointness gate result, and
            # an explicit ood_labels_used=false for the adaptation step).
            "adaptation": adaptation_block,
            "training_recipe": {
                "loss": summary["loss"],
                "class_weight": summary["class_weight"],
                "class_weights_source": summary["class_weights_source"],
                "class_weights": summary["class_weights"],
                "train_class_counts": summary["train_class_counts"],
                "sampler": bundle["sampler"],
                "early_metric": summary["early_metric"],
                "early_min_delta": plan["early_min_delta"],
                "focal_gamma": plan["focal_gamma"],
                "focal_alpha": plan["focal_alpha"],
                "defaults": RECIPE_DEFAULTS,
                "default_rationale": RECIPE_DEFAULT_NOTES,
                "adaptation_default_note": ADAPTATION_DEFAULT_NOTE,
                "matched_mixed_protocol_flags": "--adaptation-samples 50",
                "legacy_restore_flags": (
                    "--loss ce --class-weight none --early-metric val_loss "
                    "--no-temperature-calibration --no-balanced-sampler"
                ),
            },
            "calibration": calibration,
            "claim_boundary": (
                "Any temperature/logit adjustment was fitted on ID_VAL only; OOD "
                "labels were never used for fitting or for checkpoint selection. "
                "No OOD improvement is inferred from a validation-only gain. "
                "The adaptation step mixed labelled ADAPTATION-split samples into "
                "TRAIN only with ood_labels_used=false; this matches the classical "
                "mixed training budget and is not an OOD-generalisation claim."
            ),
            "backbone": bundle["backbone_provenance"],
            "command": "python scripts/train_qml.py " + " ".join(argv or sys.argv[1:]),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "warnings": all_warnings,
        },
    )
    print(
        f"wrote {len(OUTPUT_FILES)} artifacts to {output_dir} "
        f"(best epoch {summary['best_epoch']}, {summary['epochs_run']} epochs run)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
