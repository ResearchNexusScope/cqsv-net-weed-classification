"""CLI for the HYBRID-D01 classical + variational-quantum fresh-protocol arm.

Dry runs print the full plan and perform NO cache loading, circuit execution,
training or writes.  Real runs produce only additive ``metrics.json`` and
``provenance.json`` artifacts; existing result trees are never edited.

CPU / qubit caveats
-------------------
The quantum branch is PennyLane ``default.qubit`` with torch backprop on CPU.
Memory and simulation time grow as ``2**qubits``; the default 4 qubits (16
amplitudes) and 2 layers are a CPU-feasibility choice, not a performance
optimum.  TRAIN is capped (default 2000 rows/seed) and each epoch is limited to
a bounded number of batches; TEST is always full and sealed.  This is a
simulator diagnostic only -- no quantum advantage, device claim or speed claim
is made.  The pure-QNN arm is a separate worker's module and is never run here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS = ("deepweeds", "acre", "maizeset")
BACKBONES = ("plantnet", "imagenet")

# Keep the claim boundary text in one place; the CLI prints it verbatim without
# importing the heavy model module during a dry run.
_DRY_RUN_CLAIM_BOUNDARY = (
    "HYBRID-D01 diagnostic arm on the frozen fresh 70:20:10 grouped splits. "
    "classical branch Linear(feature_dim -> hidden) -> ReLU; quantum branch "
    "Linear(feature_dim -> 2**qubits) adapter -> qmlpipe.models.build_qnn with "
    "reduced qubits, amplitude embedding and trainable StronglyEntanglingLayers "
    "emitting 2 logits; final Linear(concat([classical_hidden, quantum_logits]) -> 2). "
    "Both branches are trainable and both contribute to the fused classifier; this "
    "is a hybrid model, not a pure QNN and not a parameter-free quantum kernel. "
    "All quantum execution is CPU simulation (default.qubit, analytic); no quantum "
    "advantage is asserted and no speed or hardware claim is made. Paired comparisons "
    "are reported with Holm correction and decisions are computed from the data; "
    "they are evidence for or against a difference, not a licence to claim an "
    "advantage without the full matched-control and Critic gate. TEST is sealed and "
    "scored once; failures are recorded as warnings and never fabricated."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh-protocol hybrid (classical compression + variational quantum) arm. "
            "CPU simulator only; no quantum-advantage claim."
        )
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--backbone",
        choices=BACKBONES,
        default="plantnet",
        help="frozen feature backbone to use (default: plantnet)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="number of independent grouped-split seeds (default 10)",
    )
    parser.add_argument(
        "--train-cap",
        type=int,
        default=2000,
        help="deterministic stratified TRAIN row cap per seed (default 2000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="bounded training epochs per seed (default 3)",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=4,
        help="QNN qubits; state vector grows as 2**qubits (default 4)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="trainable entangling ansatz layers (default 2)",
    )
    parser.add_argument(
        "--qnn-results",
        default=None,
        help=(
            "optional metrics.json path (or directory) from the peer pure-QNN arm; "
            "adds a paired hybrid-vs-QNN comparison when present"
        ),
    )
    parser.add_argument("--output-dir", default="results/fresh_hybrid")
    parser.add_argument("--cache-dir", default=None, help="directory of frozen feature caches")
    parser.add_argument(
        "--classical-results",
        default=None,
        help="optional override for the committed classical poly-2 metrics.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan only; no cache load, circuit execution, training or writes",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    if args.train_cap < 0:
        parser.error("--train-cap must be >= 0")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.layers < 1:
        parser.error("--layers must be >= 1")
    if not 1 <= args.qubits <= 10:
        parser.error("--qubits must be in [1, 10] for CPU-feasible simulation")

    plan = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "seeds": list(range(args.seeds)),
        "train_cap": args.train_cap,
        "epochs": args.epochs,
        "qubits": args.qubits,
        "layers": args.layers,
        "qnn_results": args.qnn_results,
        "classical_results": args.classical_results,
        "output_dir": args.output_dir,
        "cache_dir": args.cache_dir,
        "runs_planned": args.seeds,
        "claim_boundary": _DRY_RUN_CLAIM_BOUNDARY,
        "architecture": (
            "Linear(D->hidden)->ReLU classical branch and Linear(D->2**qubits)->build_qnn "
            "quantum branch concatenated into a final Linear(->2); both branches trainable"
        ),
        "cpu_qubit_caveat": (
            "PennyLane default.qubit torch-backprop CPU simulation; memory/time scale "
            "as 2**qubits. No quantum advantage or speed claim is made."
        ),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2, sort_keys=True))
        return 0

    from qmlpipe.fresh_hybrid import run_hybrid_protocol

    result = run_hybrid_protocol(
        args.dataset,
        args.backbone,
        args.seeds,
        args.output_dir,
        cache_dir=args.cache_dir,
        train_cap=args.train_cap,
        epochs=args.epochs,
        qubits=args.qubits,
        layers=args.layers,
        qnn_results=args.qnn_results,
        classical_results=args.classical_results,
    )
    print(
        json.dumps(
            {
                "dry_run": False,
                "metrics_path": result["metrics_path"],
                "provenance_path": result["provenance_path"],
                "warnings": result["warnings"],
                "claim_boundary": result["claim_boundary"],
                "architecture": result["architecture"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
