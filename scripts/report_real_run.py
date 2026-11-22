"""Generate tables/figures from a real run's metrics.json (no hardcoded numbers)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reporting import figures, manuscript_sync, tables  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="results/real_run10")
    parser.add_argument("--metrics", default="")
    args = parser.parse_args()

    metrics_path = Path(args.metrics) if args.metrics else None
    if metrics_path is None:
        candidates = sorted(glob.glob(os.path.join(args.run_root, "*", "metrics", "metrics.json")))
        if not candidates:
            raise SystemExit(f"no metrics.json under {args.run_root}")
        metrics_path = Path(candidates[-1])
    run_dir = metrics_path.parents[1]
    results = json.loads(metrics_path.read_text(encoding="utf-8"))

    table_paths = tables.write_tables(results, run_dir / "tables")
    artifacts = figures.render_all_figures(
        metrics_path,
        raw_artifacts=[run_dir / "raw" / "predictions.parquet"],
        output_dir=run_dir / "figures",
    )
    figures.write_figures_index(artifacts, run_dir / "figures")

    print("metrics:", metrics_path)
    print("tables:", {k: str(v["csv"]) for k, v in table_paths.items()})
    print("figures:")
    for artifact in artifacts:
        print(" ", artifact.figure_id, artifact.status, getattr(artifact, "path", ""))
    print("warnings:", results.get("warnings", []))

    md_path = REPO_ROOT / ".paper" / "plant-1.md"
    if md_path.exists():
        try:
            diffs = manuscript_sync.plan_sync(md_path.read_text(encoding="utf-8"), results)
            print("manuscript sync plan (dry-run):", len(diffs), "planned replacements")
            for diff in diffs[:5]:
                print(" ", getattr(diff, "placeholder", diff))
        except Exception as exc:
            print("manuscript sync plan unavailable:", type(exc).__name__, str(exc)[:160])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
