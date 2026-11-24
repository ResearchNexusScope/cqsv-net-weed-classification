"""CLI for the genuine QCNN arm on the fresh grouped-split protocol (QCNN-D01).

Dry runs print the full plan (including the convolution/pooling circuit spec)
and perform NO cache reads, training or writes.  Real runs produce additive
``metrics.json`` and ``provenance.json`` artifacts; existing result trees are
never edited.

.. important::
   ``--qubits 8`` is a **CPU-feasibility** choice, not a hardware, scaling or
   quantum-advantage claim.  The circuit is a genuine convolution+pooling QCNN:
   shared-parameter local two-qubit unitaries in a brickwork pattern, followed
   by partial-trace pooling that halves the active register at every stage.
   Any difference versus the committed baselines is reported as paired evidence
   only; no advantage is asserted.
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
POOLING_MODES = ("single_ancilla", "skip")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "qc_datasets" / "features"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genuine QCNN arm on the fresh protocol (no quantum-advantage claim). "
            "Shared-parameter convolution + partial-trace pooling; 8 qubits by "
            "default for CPU feasibility only."
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
        default=8,
        help="power-of-two qubit register; CPU feasibility only, not a hardware claim (default: 8)",
    )
    parser.add_argument(
        "--pooling",
        choices=POOLING_MODES,
        default="single_ancilla",
        help="arbitrary-width strategy: single-ancilla padding or skip-pooling (default: single_ancilla)",
    )
    parser.add_argument(
        "--conv-reps",
        type=int,
        default=2,
        help="brickwork sweeps per pooling stage (even then odd offset; default: 2)",
    )
    parser.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default="balanced",
        help="TRAIN-only class weighting for cross-entropy (default: balanced)",
    )
    parser.add_argument("--output-dir", default="results/fresh_qcnn")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"frozen feature cache root (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--qnn-results",
        default=None,
        help=(
            "optional path to the old amplitude-QNN metrics.json used as an "
            "additional paired baseline; defaults to the committed artifact "
            "when one exists for this dataset"
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
    if args.epochs < 1 or args.qubits < 2:
        parser.error("--epochs must be >= 1 and --qubits must be >= 2")
    if args.qubits & (args.qubits - 1):
        parser.error("--qubits must be a power of two")
    if args.conv_reps < 1:
        parser.error("--conv-reps must be >= 1")

    from qmlpipe.fresh_qcnn import build_plan, run_qcnn_protocol

    plan = build_plan(
        args.dataset,
        args.backbone,
        args.seeds,
        args.output_dir,
        args.cache_dir,
        args.train_cap,
        args.epochs,
        args.qubits,
        args.pooling,
        conv_reps=args.conv_reps,
        class_weight=args.class_weight,
        qnn_results=args.qnn_results,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    try:
        result = run_qcnn_protocol(
            args.dataset,
            args.backbone,
            args.seeds,
            args.output_dir,
            cache_dir=args.cache_dir,
            train_cap=args.train_cap,
            epochs=args.epochs,
            qubits=args.qubits,
            pooling=args.pooling,
            conv_reps=args.conv_reps,
            class_weight=args.class_weight,
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
