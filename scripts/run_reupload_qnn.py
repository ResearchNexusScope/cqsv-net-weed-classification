"""CLI for the genuine data-re-uploading QNN arm (REUPLOAD-D01).

Dry runs print the full plan and perform NO cache reads, training or writes.
Real runs produce additive ``metrics.json`` and ``provenance.json`` artifacts;
existing result trees are never edited.

The default 6 qubits / 4 uploads are chosen for CPU feasibility only -- not a
hardware, scaling or quantum-advantage claim.  The circuit uses shots=None
analytic simulation on CPU and the claim boundary forbids any advantage claim
without matched-budget, Holm-corrected paired power >= 0.8 evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS = ("deepweeds", "acre")
BACKBONES = ("plantnet", "imagenet")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "qc_datasets" / "features"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Data-re-uploading QNN on the fresh protocol (Perez-Salinas 2020). "
            "Default 6 qubits / 4 uploads are for CPU feasibility, not a hardware claim."
        )
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--backbone",
        choices=BACKBONES,
        default="plantnet",
        help="frozen feature backbone (default: plantnet)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="number of independent grouped-split seeds (default: 10)",
    )
    parser.add_argument(
        "--train-cap",
        type=int,
        default=2000,
        help="deterministic stratified TRAIN cap per seed (default: 2000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="bounded training epochs per seed (default: 3)",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=6,
        help="register size; 6 is for CPU feasibility only (default: 6)",
    )
    parser.add_argument(
        "--uploads",
        type=int,
        default=4,
        help="number of data re-uploads / independent V(omega_i) layers (default: 4)",
    )
    parser.add_argument(
        "--entangler",
        choices=("cnot", "cz"),
        default="cnot",
        help="ring entangler used in each V(omega_i) layer (default: cnot)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/fresh_reupload_qnn",
        help="output tree for additive metrics.json / provenance.json",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"frozen feature cache root (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--qnn-results",
        default=None,
        help=(
            "optional path to the old amplitude-QNN baseline: a metrics.json FILE or "
            "a DIRECTORY containing metrics.json (default: per-dataset committed artifact when present)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan only; no cache reads, training or writes",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    if args.train_cap < 0:
        parser.error("--train-cap must be >= 0")
    if args.epochs < 1 or args.qubits < 2 or args.uploads < 1:
        parser.error("--epochs>=1, --qubits>=2 and --uploads>=1 are required")

    from qmlpipe.reupload_qnn import build_plan, run_reupload_protocol

    plan = build_plan(
        args.dataset,
        args.backbone,
        args.seeds,
        args.output_dir,
        args.cache_dir,
        args.train_cap,
        args.epochs,
        args.qubits,
        args.uploads,
        args.entangler,
        args.qnn_results,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    try:
        result = run_reupload_protocol(
            args.dataset,
            args.backbone,
            args.seeds,
            args.output_dir,
            cache_dir=args.cache_dir,
            train_cap=args.train_cap,
            epochs=args.epochs,
            qubits=args.qubits,
            uploads=args.uploads,
            entangler=args.entangler,
            qnn_results=args.qnn_results,
        )
    except Exception as exc:  # graceful failure; never fabricate a result
        print(
            json.dumps(
                {"dry_run": False, "error": f"{type(exc).__name__}: {exc}"},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "dry_run": False,
                "metrics_path": result["metrics_path"],
                "provenance_path": result["provenance_path"],
                "n_runs_succeeded": result["metrics"]["n_runs_succeeded"],
                "n_runs_skipped": result["metrics"]["n_runs_skipped"],
                "warnings": result["warnings"],
                "claim_boundary": result["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
