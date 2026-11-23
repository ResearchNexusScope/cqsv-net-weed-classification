"""CLI for the pure variational-QNN arm on the fresh grouped-split protocol.

Dry runs print the full plan and perform NO cache reads, training or writes.
Real runs produce additive ``metrics.json`` and ``provenance.json`` artifacts;
existing result trees are never edited.

.. important::
   ``--qubits 4`` (the default) is chosen for **CPU feasibility only**.  It is
   not a hardware claim, not a scaling claim and not a quantum-advantage claim.
   A 4-qubit ``default.qubit`` simulation with 3 epochs is a tractable smoke test
   of the protocol wiring; no advantage is asserted from it.  A QNN win would
   additionally require the paired comparison against the committed classical
   poly-2 arm to pass with power >= 0.8 (see the claim boundary).
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
            "Pure QNN arm on the fresh protocol (no quantum-advantage claim). "
            "Default 4 qubits are for CPU feasibility, not a hardware claim."
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
        default=4,
        help="4 is for CPU feasibility only, not a hardware claim (default: 4)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="entangling ansatz layers (default: 2)",
    )
    parser.add_argument("--output-dir", default="results/fresh_qnn")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"frozen feature cache root (default: {DEFAULT_CACHE_DIR})",
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
    if args.epochs < 1 or args.qubits < 1 or args.layers < 1:
        parser.error("--epochs, --qubits and --layers must be >= 1")

    from qmlpipe.fresh_qnn import build_plan, run_qnn_protocol

    plan = build_plan(
        args.dataset,
        args.backbone,
        args.seeds,
        args.output_dir,
        args.cache_dir,
        args.train_cap,
        args.epochs,
        args.qubits,
        args.layers,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    try:
        result = run_qnn_protocol(
            args.dataset,
            args.backbone,
            args.seeds,
            args.output_dir,
            cache_dir=args.cache_dir,
            train_cap=args.train_cap,
            epochs=args.epochs,
            qubits=args.qubits,
            layers=args.layers,
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
