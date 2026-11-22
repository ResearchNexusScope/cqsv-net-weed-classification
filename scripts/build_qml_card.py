"""Aggregate real QML ablation runs into an n/std/CI summary + model card."""
from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qmlpipe.ablate import summarize_ablation  # noqa: E402
from qmlpipe.config import QMLConfig  # noqa: E402
from qmlpipe.model_card import build_model_card  # noqa: E402


def main() -> int:
    paths = sorted(glob.glob(str(REPO_ROOT / "results" / "qml_ablate_*" / "metrics.json")))
    if not paths:
        raise SystemExit("no qml_ablate_*/metrics.json found")

    records = []
    for index, path in enumerate(paths):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        ood = payload.get("ood", {})
        training = payload.get("training", {})
        metrics = {
            key: ood.get(key)
            for key in ("accuracy", "auc", "f1", "precision", "recall", "macro_iou",
                        "log_loss", "n_samples")
            if isinstance(ood.get(key), (int, float))
        }
        if isinstance(training.get("best_val_accuracy"), (int, float)):
            metrics["val_accuracy"] = training["best_val_accuracy"]
        records.append({
            "run_index": index,
            "seed": int(cfg.get("seed", 42)),
            "status": "ok",
            "config": cfg,
            "metrics": metrics,
            "source": str(Path(path).relative_to(REPO_ROOT)),
        })

    summary = summarize_ablation(records)
    first_cfg = records[0]["config"]
    accepted = {item.name for item in fields(QMLConfig)}
    cfg = QMLConfig(**{k: v for k, v in first_cfg.items() if k in accepted})
    card = build_model_card(summary, cfg)

    out_dir = REPO_ROOT / "results" / "qml_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "model_card.md").write_text(card, encoding="utf-8")

    print("runs:", summary["n_configs"], "success:", summary["n_success"])
    for key, block in summary["metrics"].items():
        print(f"{key}: mean={block['mean']} std={block['std']} n={block['n']} CI={block['ci95_low']},{block['ci95_high']}")
    print("wrote:", out_dir / "ablation_summary.json", out_dir / "model_card.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
