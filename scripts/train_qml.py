"""Train the hybrid transfer QNN on leak-free real features (QML-D02).

All dataset locations come from ``KAGGLE_DATA_DIR`` / ``NINJA_ANN_DIR`` /
``NINJA_IMG_DIR``; no developer path is embedded.  Features are produced by the
same helpers as ``scripts/run_real_benchmark.py`` (``prepare_ninja_subset``,
``cap_ood_by_group``, ``build_backbone``, ``make_feature_loader``), so the OOD
crop cap is applied per source-image group and no crop leaks across splits.

The heavy dependencies (torch, torchvision, PennyLane, the dataset modules) are
imported only inside the real-run path.  ``--dry-run`` therefore resolves and
validates the plan and prints it without importing a simulator or writing a
single file, which keeps it usable in the central verifier's minimal
environment.
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
from typing import Any

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
    }
)

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
    plan: dict[str, Any] = {
        "qubits": qubits,
        "feature_dim": feature_dim,
        "projection_dim": int(choose(None, "projection_dim", feature_dim)),
        "classical_dim": int(choose(None, "classical_dim", 1280)),
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
    }
    if args.smoke:
        warnings.append("smoke mode: subset capped, epochs forced to 1")
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


def _prepare_data(plan: dict, smoke: bool) -> dict:
    """Build leak-free train/val/OOD loaders and record the data provenance."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from data import build_splits
    from data.splits import Split
    from experiments.config import ExperimentConfig
    from scripts.run_real_benchmark import (
        SubsetManifest,
        build_backbone,
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
    backbone, preprocess = build_backbone(torch_device)
    feature_loader = make_feature_loader(backbone, preprocess, torch_device)

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

    generator = torch.Generator().manual_seed(int(plan["seed"]))
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=int(plan["batch_size"]),
        shuffle=True,
        generator=generator,
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

    warnings: list[str] = []
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
        "data_paths": data_paths,
        "subset_files": subset_files,
        "ood_per_class": per_class,
        "adaptation_per_class": int(data_scope["adaptation_per_class"]),
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

    try:
        cfg = _build_config(plan)
        from qmlpipe.models import build_transfer_qnn
        from qmlpipe.train import evaluate, seed_everything, train

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
            ckpt_path=output_dir / "checkpoints" / "best.pt",
        )
        ood_metrics = (
            evaluate(model, bundle["ood_loader"], cfg)
            if bundle["ood_loader"] is not None
            else {"status": "N/A", "reason": "no OOD crops available"}
        )
    except Exception as exc:  # never fabricate a result for a failed run
        print(f"training run failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2

    all_warnings = list(warnings) + list(bundle["warnings"])
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
                "epochs_run": summary["epochs_run"],
                "early_stopped": summary["early_stopped"],
                "optimizer": summary["optimizer"],
            },
            "ood": ood_metrics,
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
            },
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
