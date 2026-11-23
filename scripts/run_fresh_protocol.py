"""CLI for the fresh classical control protocol (FRESH-D02).

Dry runs print the full plan and perform NO downloads, feature extraction,
training or writes.  Real runs only produce additive ``metrics.json`` and
``provenance.json`` artifacts; existing result trees are never edited.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS = ("maizeset", "deepweeds", "acre")
BACKBONES = ("plantnet", "imagenet")
MODELS = ("poly2", "rbf", "logreg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fresh grouped-split classical control protocol (no QNN, no quantum-advantage claim)."
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--backbone",
        action="append",
        choices=BACKBONES,
        default=None,
        help="repeatable; defaults to both plantnet and imagenet",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="number of independent grouped-split seeds (default 10)",
    )
    parser.add_argument(
        "--models",
        action="append",
        choices=MODELS,
        default=None,
        help="repeatable; defaults to poly2, rbf, logreg",
    )
    parser.add_argument("--output-dir", default="results/fresh_protocol")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--root", default=None, help="optional dataset root directory")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan only; no downloads, training or writes",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    backbones = list(dict.fromkeys(args.backbone or list(BACKBONES)))
    models = list(dict.fromkeys(args.models or list(MODELS)))
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    plan = {
        "dataset": args.dataset,
        "backbones": backbones,
        "seeds": list(range(args.seeds)),
        "models": models,
        "output_dir": args.output_dir,
        "cache_dir": args.cache_dir,
        "root": args.root,
        "alpha": float(args.alpha),
        "runs_planned": args.seeds * len(backbones) * len(models),
        "claim_boundary": (
            "Classical controls and paired statistics only; no QNN arm and no quantum "
            "advantage is asserted in this stage."
        ),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2, sort_keys=True))
        return 0

    from experiments.fresh_protocol import run_protocol

    result = run_protocol(
        args.dataset,
        backbones=backbones,
        seeds=args.seeds,
        models=models,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        root=args.root,
        alpha=float(args.alpha),
    )
    print(
        json.dumps(
            {
                "dry_run": False,
                "metrics_path": result["metrics_path"],
                "provenance_path": result["provenance_path"],
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
