"""Post-hoc threshold analysis for a saved QML checkpoint.

Reads the real metrics.json config, rebuilds the same data/feature pipeline and
model, loads checkpoints/best.pt, recomputes OOD probabilities, and sweeps decision
thresholds. Writes a JSON report. No number is hardcoded or copied from prose.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_metrics(run_dir: Path) -> tuple[dict, dict]:
    metrics_path = run_dir / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return payload, payload.get("config", {})


def _binary_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= float(threshold)).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float((tpr + (1 - fpr)) / 2),
        "confusion": [[tn, fp], [fn, tp]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="directory containing metrics.json and checkpoints/best.pt")
    parser.add_argument("--output", default=None, help="output JSON path (default: <run-dir>/threshold_analysis.json)")
    parser.add_argument("--steps", type=int, default=181, help="number of thresholds in [0,1]")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metrics, cfg_dict = _load_metrics(run_dir)

    import torch
    from qmlpipe.config import QMLConfig
    from qmlpipe.models import build_transfer_qnn
    from scripts.train_qml import _feature_fed_transfer_model, _prepare_data

    accepted = {item.name for item in fields(QMLConfig)}
    cfg = QMLConfig(**{key: value for key, value in cfg_dict.items() if key in accepted})
    # _prepare_data expects the CLI-plan spelling of QMLConfig fields.
    plan = dict(cfg_dict)
    plan.update({
        "backbone": cfg.backbone,
        "output_dir": str(run_dir),
        "feature_dim": int(cfg.feature_dim),
        "n_qubits": int(cfg.qubits),
        "qnn_layers": int(cfg.layers),
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "learning_rate": float(cfg.learning_rate),
        "seed": int(cfg.seed),
    })
    bundle = _prepare_data(plan, smoke=False)
    ood_loader = bundle.get("ood_loader")
    if ood_loader is None:
        raise SystemExit("no OOD loader available in saved-data plan")

    base = build_transfer_qnn(cfg)
    model = _feature_fed_transfer_model(base, cfg)
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    scores, labels = [], []
    with torch.no_grad():
        for features, targets in ood_loader:
            logits = model(features)
            probs = torch.softmax(logits, dim=1)[:, 1]
            scores.extend(probs.cpu().numpy().tolist())
            labels.extend(targets.cpu().numpy().tolist())
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(scores) == 0:
        raise SystemExit("no OOD samples scored")

    thresholds = np.linspace(0.0, 1.0, int(args.steps))
    sweep = [_binary_metrics(labels, scores, t) for t in thresholds]
    best_f1 = max(sweep, key=lambda item: (item["f1"], item["balanced_accuracy"]))
    best_ba = max(sweep, key=lambda item: (item["balanced_accuracy"], item["f1"]))
    baseline = metrics.get("ood", {})

    report = {
        "schema_version": "1.0",
        "source_run": str(run_dir),
        "n_samples": int(len(labels)),
        "scores_summary": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        },
        "saved_metrics_at_0_5": {
            key: baseline.get(key) for key in ("accuracy", "f1", "precision", "recall", "auc", "macro_iou")
        },
        "best_by_f1": best_f1,
        "best_by_balanced_accuracy": best_ba,
        "sweep": sweep,
        "provenance": {
            "config": cfg.as_dict(),
            "backbone": bundle.get("backbone_provenance"),
            "counts": bundle.get("counts"),
            "commit_hash": metrics.get("commit_hash"),
            "note": "Thresholds sweep the saved model's own OOD probabilities; no labels were used to fit weights, only to select a reported operating point. This is post-hoc analysis, not a claim of improved generalization.",
        },
    }
    out_path = Path(args.output).resolve() if args.output else run_dir / "threshold_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out_path)
    print("n", len(labels), "best_f1", best_f1)
    print("best_balanced_accuracy", best_ba)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
