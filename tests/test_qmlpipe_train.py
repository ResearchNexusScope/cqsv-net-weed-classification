"""Synthetic tests for the QML-D02 train/evaluate/CLI surface.

Every tensor is generated in memory from a fixed seed; no dataset path, legacy
result or paper number is used.  Tests that require PennyLane skip cleanly when
it is absent (the central verifier runs without it) and never fabricate a
quantum result.
"""
from __future__ import annotations

import dataclasses
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - depends on the host environment
    import torch
except Exception:  # pragma: no cover
    torch = None

from qmlpipe import evaluate as qml_evaluate  # noqa: E402
from qmlpipe.train import (  # noqa: E402
    HISTORY_KEYS,
    evaluate,
    seed_everything,
    train,
)

torch_required = pytest.mark.skipif(torch is None, reason="torch is not installed")


@dataclasses.dataclass
class _Cfg:
    """Minimal stand-in exposing the QMLConfig fields train/evaluate consume."""

    seed: int = 123
    epochs: int = 25
    patience: int = 2
    optimizer: str = "adamw"
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    batch_size: int = 4
    output_dir: str = "results"
    calibration_bins: int = 10
    qubits: int = 2
    layers: int = 1
    embedding: str = "amplitude"
    ansatz: str = "strongly_entangling"
    device_name: str = "default.qubit"
    shots: object = None
    noise_depolarizing: float = 0.0
    feature_dim: int = 4

    def as_dict(self) -> dict:
        return {item.name: getattr(self, item.name) for item in dataclasses.fields(self)}


def _loaders(n: int = 8, batch: int = 4, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n, 4, generator=generator)
    labels = torch.randint(0, 2, (n,), generator=generator)
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
@torch_required
def test_seed_everything_deterministic():
    """Re-seeding reproduces the Python, NumPy and torch RNG streams."""
    seed_everything(7)
    first = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
    seed_everything(7)
    second = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
    assert first == second


@torch_required
def test_train_is_deterministic_for_fixed_seed(tmp_path):
    """Two seeded runs of a synthetic linear model produce identical history."""
    cfg = _Cfg(seed=11, epochs=3, patience=10)

    def run(tag: str):
        seed_everything(cfg.seed)
        torch.manual_seed(cfg.seed)
        model = torch.nn.Linear(4, 2)
        loader = _loaders(n=12, batch=4, seed=3)
        return train(model, loader, loader, cfg, output_dir=tmp_path / tag)["history"]

    assert run("a") == run("b")


# ---------------------------------------------------------------------------
# Early stopping, checkpoint and history
# ---------------------------------------------------------------------------
if torch is not None:  # pragma: no branch - depends on the host environment

    class _ConstantModel(torch.nn.Module):
        """A model whose loss cannot improve, so early stopping fires exactly."""

        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x.new_zeros((x.shape[0], 2)) + self.dummy * 0.0


@torch_required
def test_early_stopping_history_and_checkpoint(tmp_path):
    """Early stop triggers; history JSON and best checkpoint carry the contract keys."""
    cfg = _Cfg(epochs=25, patience=2)
    result = train(_ConstantModel(), _loaders(), _loaders(), cfg, output_dir=tmp_path)

    assert result["early_stopped"] is True
    assert result["epochs_run"] <= cfg.epochs
    assert result["epochs_run"] >= 2

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert set(history) == set(HISTORY_KEYS)
    for values in history.values():
        assert len(values) == result["epochs_run"]

    checkpoint = torch.load(tmp_path / "checkpoints" / "best.pt", weights_only=False)
    assert {"state_dict", "epoch", "cfg", "commit_hash"} <= set(checkpoint)
    assert checkpoint["epoch"] == result["best_epoch"] == 0
    assert checkpoint["cfg"]["seed"] == cfg.seed
    assert isinstance(checkpoint["commit_hash"], str)
    assert set(checkpoint["state_dict"]) == {"dummy"}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch_required
def test_evaluate_returns_metric_keys():
    """evaluate() returns JSON-serialisable metrics from experiments.metrics."""
    result = evaluate(_ConstantModel(), _loaders(n=12), _Cfg())
    expected = {
        "n_samples",
        "support",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "iou_class0",
        "iou_class1",
        "macro_iou",
        "confusion_matrix",
        "log_loss",
        "calibration",
    }
    assert expected <= set(result)
    assert result["n_samples"] == 12
    assert {"brier_score", "ece"} <= set(result["calibration"])
    json.dumps(result)  # must be JSON-serialisable


def test_evaluate_qsvc_without_probability_is_na():
    """A hard-label-only SVC yields no fabricated probabilities or log loss."""

    class _HardSvc:
        def predict(self, K):
            return np.asarray([0, 1, 1, 0])

    result = qml_evaluate.evaluate_qsvc(np.eye(4), [0, 1, 0, 1], _HardSvc())
    assert result["probability_available"] is False
    assert result["log_loss"] is None
    assert result["calibration"]["status"] == "N/A"
    assert result["accuracy"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------
def test_compare_models_refuses_n_less_than_2():
    """n<2 pairs are reported without any test statistic or p-value."""
    result = qml_evaluate.compare_models({"a": [0.8], "b": [0.7]})
    assert len(result["comparisons"]) == 1
    pair = result["comparisons"][0]
    assert pair["status"] == "insufficient_samples"
    assert "p_value" not in pair
    assert "t_statistic" not in pair
    assert "cohens_dz" not in pair

    tested = qml_evaluate.compare_models(
        {"a": [0.80, 0.90, 0.85], "b": [0.70, 0.75, 0.72]}
    )["comparisons"][0]
    assert tested["status"] == "tested"
    assert tested["n"] == 3
    assert 0.0 <= tested["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_data_plan_scope_consistency(monkeypatch):
    """Smoke/full data scopes agree on adaptation and OOD caps (no real I/O)."""
    from scripts import train_qml

    monkeypatch.delenv("QC_NINJA_SUBSET_FILES", raising=False)
    monkeypatch.delenv("QC_OOD_PER_CLASS", raising=False)

    smoke = train_qml._data_plan(smoke=True)
    full = train_qml._data_plan(smoke=False)

    assert smoke["subset_files"] <= 8
    assert smoke["per_class"] == 2
    assert smoke["per_class"] == smoke["adaptation_per_class"]
    assert full["subset_files"] == 1200
    assert full["per_class"] == full["adaptation_per_class"] == 200


def test_cli_dry_run_does_not_write(tmp_path):
    """--dry-run resolves and validates without importing a simulator or writing."""
    from scripts import train_qml

    target = tmp_path / "dry_run_output"
    code = train_qml.main(
        ["--dry-run", "--output-dir", str(target), "--qubits", "2", "--layers", "1"]
    )
    assert code == 0
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_cli_rejects_invalid_plan(tmp_path, capsys):
    """An explicit derived-field override is rejected, not silently ignored."""
    from scripts import train_qml

    config_file = tmp_path / "bad.json"
    config_file.write_text(json.dumps({"qubits": 3, "feature_dim": 7}), encoding="utf-8")
    run_dir = tmp_path / "would_be_output"
    code = train_qml.main(
        [
            "--dry-run",
            "--config-json",
            str(config_file),
            "--output-dir",
            str(run_dir),
        ]
    )
    assert code == 2
    assert not run_dir.exists()
    error = capsys.readouterr().err
    assert "unknown --config-json key(s): feature_dim" in error
    assert "derived fields" in error


def test_resolve_plan_rejects_unknown_key(tmp_path):
    """_resolve_plan raises ValueError for any unknown --config-json key."""
    from scripts import train_qml

    config_file = tmp_path / "unknown.json"
    config_file.write_text(json.dumps({"not_a_field": 1}), encoding="utf-8")
    args = train_qml._build_parser().parse_args(
        ["--dry-run", "--config-json", str(config_file)]
    )
    with pytest.raises(ValueError, match="unknown --config-json key\\(s\\): not_a_field"):
        train_qml._resolve_plan(args)


# ---------------------------------------------------------------------------
# Quantum-dependent build (skips when PennyLane is absent)
# ---------------------------------------------------------------------------
def test_transfer_qnn_constructs_when_pennylane_available():
    """Build the real transfer QNN only when the pinned quantum stack exists."""
    pytest.importorskip("pennylane", reason="PennyLane unavailable")
    try:
        from qmlpipe.config import QMLConfig
        from qmlpipe.models import build_transfer_qnn
    except Exception as exc:  # e.g. the lightning plugin is missing
        pytest.skip(f"qmlpipe quantum builders unavailable: {exc}")

    requested = {
        "qubits": 2,
        "feature_dim": 4,
        "projection_dim": 4,
        "classical_dim": 1280,
        "layers": 1,
        "embedding": "amplitude",
        "device_name": "default.qubit",
        "shots": None,
        "seed": 7,
    }
    try:
        accepted = {item.name for item in dataclasses.fields(QMLConfig)}
        cfg = QMLConfig(**{k: v for k, v in requested.items() if k in accepted})
    except TypeError:  # not a dataclass: fall back to defaults
        cfg = QMLConfig()
    model = build_transfer_qnn(cfg)
    assert isinstance(model, torch.nn.Module)
    assert sum(int(p.numel()) for p in model.parameters() if p.requires_grad) > 0
