"""Leak-free logistic-regression baseline over frozen-backbone image features.

Given the real-data environment variables (``KAGGLE_DATA_DIR``, ``NINJA_ANN_DIR``,
``NINJA_IMG_DIR``) and a chosen frozen feature extractor, this script fits a
single classical logistic-regression baseline on **training features only** and
evaluates it on held-out validation and OOD-test features.  All dataset helpers
are the same group-disjoint, crop-capped helpers used by
``scripts/train_qml.py`` / ``scripts/run_real_benchmark.py``: the test split is
never used during ``fit`` and its crops cannot leak into training.

Every reported number is computed from the extracted features -- no accuracy,
AUC, sample count or other result is hardcoded.  Heavy dependencies (torch,
torchvision, sklearn, the dataset modules) are imported lazily, so ``--dry-run``
resolves and prints the plan without importing a model or writing any file.
"""
from __future__ import annotations

import argparse
from importlib import metadata as _metadata
import json
import os
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ALLOWED_BACKBONES = frozenset({"plantnet", "imagenet"})

# Fitting/evaluation hyperparameters only -- never result values.
DEFAULT_MAX_ITER = 2000
DEFAULT_OOR_PER_CLASS = 200
DEFAULT_SUBSET_FILES = 1200
DEFAULT_SEED = 42


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/compare_backbones.py",
        description=(
            "Fit a logistic-regression baseline on frozen-backbone features "
            "(train only) and evaluate on held-out validation/OOD sets."
        ),
    )
    parser.add_argument(
        "--backbone",
        choices=tuple(sorted(ALLOWED_BACKBONES)),
        default="plantnet",
        help="frozen feature backbone (default plantnet)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/compare_backbones",
        help="directory to write the comparison JSON",
    )
    parser.add_argument(
        "--oor-per-class",
        type=int,
        default=DEFAULT_OOR_PER_CLASS,
        help="max OOD crops per class after the group cap",
    )
    parser.add_argument(
        "--subset-files",
        type=int,
        default=DEFAULT_SUBSET_FILES,
        help="number of source-annotated Ninja images to link",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and print the plan but import nothing heavy and write nothing",
    )
    return parser


def _resolve_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the CLI plan without touching the environment or the filesystem."""
    backbone = str(args.backbone)
    if backbone not in ALLOWED_BACKBONES:
        raise ValueError(f"backbone must be one of {sorted(ALLOWED_BACKBONES)}")
    for name in ("subset_files", "oor_per_class", "seed"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be a positive integer")
    return {
        "backbone": backbone,
        "output_dir": str(args.output_dir),
        "subset_files": int(args.subset_files),
        "oor_per_class": int(args.oor_per_class),
        "seed": int(args.seed),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("torch", "torchvision", "numpy", "scikit-learn", "huggingface_hub"):
        try:
            versions[name] = str(_metadata.version(name))
        except Exception:
            versions[name] = "not installed"
    return versions


def fit_and_evaluate(
    X_train: Any,
    y_train: Any,
    eval_splits: dict[str, tuple[Any, Any]],
    seed: int,
    max_iter: int = DEFAULT_MAX_ITER,
) -> dict[str, dict[str, Any]]:
    """Fit one logistic baseline on train features only, then score held-out sets.

    Scaling lives inside the ``Pipeline`` so its statistics are fitted on the
    training split alone; the OOD/validation features are read only at scoring
    time.  This is the leakage guard mandated by QC-D04 §2.0.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=int(max_iter), random_state=int(seed)),
    )
    clf.fit(X_train, y_train)  # ONLY the training split is ever fitted.

    results: dict[str, dict[str, Any]] = {
        "train": _score_split(clf, X_train, y_train, accuracy_score, roc_auc_score)
    }
    for name, (X_eval, y_eval) in eval_splits.items():
        results[name] = _score_split(clf, X_eval, y_eval, accuracy_score, roc_auc_score)
    return results


def _score_split(clf: Any, X: Any, y: Any, accuracy_score: Any, roc_auc_score: Any):
    """Compute accuracy/AUC/n for one split from the fitted baseline."""
    y_pred = clf.predict(X)
    entry: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
    }
    proba = clf.predict_proba(X)
    if proba.shape[1] == 2 and len(set(map(int, y))) == 2:
        entry["auc"] = float(roc_auc_score(y, proba[:, 1]))
    else:  # AUC is undefined for a single present class: report honestly.
        entry["auc"] = None
        entry["auc_note"] = "undefined (fewer than two classes present in split)"
    return entry


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        plan = _resolve_plan(args)
    except (TypeError, ValueError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(plan["output_dir"]).resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "plan": plan,
                    "output_dir": str(output_dir),
                    "files_to_write": ["compare_backbones.json"],
                },
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    try:
        import numpy as np
        import torch

        from data import build_splits
        from data.splits import Split
        from experiments.config import ExperimentConfig
        from qmlpipe.config import QMLConfig
        from qmlpipe.plant_backbone import build_backbone_extractor
        from scripts.run_real_benchmark import (
            SubsetManifest,
            cap_ood_by_group,
            make_feature_loader,
            prepare_ninja_subset,
        )
        from scripts.train_qml import (
            _backbone_provenance,
            _git_commit_hash,
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

        prepare_ninja_subset(plan["subset_files"])
        cfg = replace(
            ExperimentConfig.from_env(),
            output_dir=plan["output_dir"],
            seed=plan["seed"],
        )
        base_manifest = build_splits(cfg, kaggle_root=data_paths["kaggle_dir"])
        ood_samples = cap_ood_by_group(
            base_manifest.splits[Split.OOD_TEST].samples, plan["oor_per_class"]
        )
        manifest = SubsetManifest(base_manifest, ood_samples)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor, preprocess = build_backbone_extractor(
            plan["backbone"], QMLConfig(backbone=plan["backbone"])
        )
        feature_loader = make_feature_loader(extractor, preprocess, device)

        def features(samples: Any, split: str):
            X = np.asarray(feature_loader(samples, cfg, split), dtype=np.float32)
            y = np.asarray([int(sample.label) for sample in samples], dtype=np.int64)
            return X, y

        X_train, y_train = features(manifest.splits[Split.TRAIN].samples, "train")
        if len(y_train) == 0:
            raise RuntimeError("no Kaggle training samples were available")

        eval_splits: dict[str, tuple[Any, Any]] = {}
        for split in (Split.ID_VAL, Split.OOD_TEST):
            samples = manifest.splits[split].samples
            if samples:
                eval_splits[split.value] = features(samples, split.value)

        scores = fit_and_evaluate(X_train, y_train, eval_splits, plan["seed"])

        # Headline numbers come from the held-out OOD test split when present;
        # otherwise from the validation split.  Never from the fitted train set.
        headline_split = Split.OOD_TEST.value if Split.OOD_TEST.value in scores else None
        if headline_split is None and eval_splits:
            headline_split = next(iter(eval_splits))
        headline = scores.get(headline_split, {}) if headline_split else {}

        command = "python scripts/compare_backbones.py " + " ".join(argv or sys.argv[1:])
        payload = {
            "schema_version": "1.0",
            "experiment_id": "compare_backbones_logreg",
            "plan": plan,
            "backbone": _backbone_provenance(plan["backbone"], int(X_train.shape[1])),
            "headline_split": headline_split,
            "accuracy": headline.get("accuracy"),
            "auc": headline.get("auc"),
            "n": headline.get("n"),
            "results": scores,
            "provenance": {
                "commit_hash": _git_commit_hash(),
                "python_version": platform.python_version(),
                "packages": _package_versions(),
                "data": {
                    "paths": data_paths,
                    "ninja_subset_files": plan["subset_files"],
                    "ood_per_class": plan["oor_per_class"],
                    "n_train": int(len(y_train)),
                },
                "feature_dim": int(X_train.shape[1]),
                "frozen_backbone": True,
                "command": command,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "leakage_note": (
                    "Logistic regression fitted on TRAIN features only; "
                    "validation/OOD features scored after fitting (QC-D04 §2.0)."
                ),
            },
        }
    except Exception as exc:  # never fabricate a result for a failed run
        print(
            f"backbone comparison failed ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        return 2

    _write_json(output_dir / "compare_backbones.json", payload)
    print(f"wrote backbone comparison to {output_dir / 'compare_backbones.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
