"""Synthetic tests for the QHEAD-D01 head / loss / training changes.

Every tensor, label vector and logit comes from a local fixture or a fixed RNG;
no dataset path, no legacy project result and no paper number appears here.
Heavy dependencies are optional: the module skips cleanly when torch is absent.

Covered contracts:
  * class weights and priors are derived from TRAIN labels only;
  * focal loss down-weights easy examples and reduces to weighted CE at gamma=0;
  * checkpoint/early-stop selection honours ``early_metric`` (val_loss vs a
    balanced metric) and records the recipe in the checkpoint;
  * temperature/logit adjustment are fitted on ID_VAL only and never on OOD;
  * head/logit shapes and legacy Linear parity;
  * balanced sampler derives from TRAIN labels;
  * CLI recipe defaults and flags.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmlpipe import heads as heads_mod  # noqa: E402
from qmlpipe import train as train_mod  # noqa: E402
from qmlpipe.heads import (  # noqa: E402
    ClassificationHead,
    build_classification_head,
    find_classification_head,
    fit_calibration,
)
from qmlpipe.losses import (  # noqa: E402
    class_priors_from_labels,
    class_weights_from_labels,
    focal_loss,
    weighted_cross_entropy,
)
from qmlpipe.train import (  # noqa: E402
    build_balanced_sampler,
    load_checkpoint,
    train,
)


@dataclasses.dataclass
class _Cfg:
    """Minimal stand-in exposing the QMLConfig fields train() consumes."""

    seed: int = 17
    epochs: int = 3
    patience: int = 10
    optimizer: str = "adamw"
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    batch_size: int = 2
    output_dir: str = "results"
    early_metric: str = "val_loss"
    early_min_delta: float = 0.0
    loss: str = "ce"
    class_weight: str = "none"
    balanced_sampler: bool = False
    focal_gamma: float = 2.0
    focal_alpha: object = None

    def as_dict(self) -> dict:
        return {item.name: getattr(self, item.name) for item in dataclasses.fields(self)}


def _loaders(n: int = 8, batch: int = 2, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(n, 4, generator=generator)
    labels = torch.randint(0, 2, (n,), generator=generator)
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch), dataset


class _Toy(torch.nn.Module):
    """Tiny model whose only module is a configurable classification head."""

    def __init__(self, in_features: int = 4):
        super().__init__()
        self.head = ClassificationHead(torch.nn.Linear(in_features, 2))

    def forward(self, x):
        return self.head(x)


# --------------------------------------------------------------------------- loss / weights


def test_class_weights_derived_from_train_labels_only():
    """Balanced weights use only the supplied labels; validation cannot change them."""
    train_labels = np.array([0] * 9 + [1])
    validation_labels = np.array([0, 1, 1, 1, 1])

    weights = class_weights_from_labels(train_labels, n_classes=2, mode="balanced")
    assert weights.shape == (2,)
    assert weights[1] > weights[0]  # minority class up-weighted
    assert float(weights[1]) == pytest.approx(10.0 / (2.0 * 1.0))
    assert float(weights[0]) == pytest.approx(10.0 / (2.0 * 9.0))

    weights_again = class_weights_from_labels(train_labels, n_classes=2, mode="balanced")
    assert torch.allclose(weights, weights_again)

    plain = class_weights_from_labels(validation_labels, n_classes=2, mode="none")
    assert torch.allclose(plain, torch.ones(2))

    priors = class_priors_from_labels(train_labels, n_classes=2)
    assert priors[0] == pytest.approx(0.9)
    assert priors[1] == pytest.approx(0.1)


def test_focal_loss_downweights_easy_examples():
    """Easy (high-confidence, correct) samples lose more weight than hard ones."""
    logits = torch.tensor([[4.0, -4.0], [0.1, -0.1]])
    labels = torch.tensor([0, 0])

    per_sample_focal = focal_loss(logits, labels, gamma=2.0, reduction="none")
    per_sample_ce = weighted_cross_entropy(logits, labels, reduction="none")

    ratio_focal = float(per_sample_focal[0] / per_sample_focal[1])
    ratio_ce = float(per_sample_ce[0] / per_sample_ce[1])
    assert ratio_focal < ratio_ce  # easy sample is relatively suppressed

    assert torch.allclose(
        focal_loss(logits, labels, gamma=0.0),
        weighted_cross_entropy(logits, labels),
        atol=1e-6,
    )

    weights = class_weights_from_labels(labels, n_classes=2, mode="balanced")
    assert not torch.allclose(
        weighted_cross_entropy(logits, labels, weight=weights),
        weighted_cross_entropy(logits, labels),
    )


# --------------------------------------------------------------------------- heads


def test_head_shapes_and_legacy_parity():
    """Default wrapper is numerically identical to Linear(n, 2); calibration changes it."""
    base = torch.nn.Linear(10, 2)
    head = ClassificationHead(base)
    x = torch.randn(4, 10)
    assert head(x).shape == (4, 2)
    assert torch.allclose(head(x), base(x))

    head.set_calibration(temperature=2.0, logit_adjust=[0.2, -0.2])
    assert not torch.allclose(head(x), base(x))
    state = head.calibration_state()
    assert state["temperature"] == pytest.approx(2.0)
    assert len(state["logit_adjust"]) == 2

    built = build_classification_head(10)
    assert built(x).shape == (4, 2)
    assert find_classification_head(_ToyWithHead(built)) is built


class _ToyWithHead(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x):
        return self.head(x)


def test_qnn_uses_calibratable_head_when_pennylane_available():
    """The real QNN head is a ClassificationHead so calibration can be applied."""
    pytest.importorskip("pennylane", reason="PennyLane unavailable")
    from qmlpipe.config import QMLConfig
    from qmlpipe.models import build_qnn

    cfg = QMLConfig(qubits=2, projection_dim=2**2, layers=1, seed=5)
    try:
        model = build_qnn(cfg)
    except Exception as exc:  # e.g. device/plugin issue
        pytest.skip(f"QNN construction unavailable: {exc}")
    assert find_classification_head(model) is not None
    logits = model(torch.randn(3, cfg.feature_dim))
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


# --------------------------------------------------------------------------- early stopping


def test_early_stopping_and_checkpoint_use_selected_metric(tmp_path, monkeypatch):
    """val_loss selects a collapsed checkpoint; balanced_accuracy selects the good one."""
    # (val_loss, val_acc, y_true, y_pred) per epoch. Epoch 1 has the lowest
    # val_loss but predicts one class only; epoch 2 is perfectly balanced.
    script = [
        (0.6, 0.5, [0, 1, 0, 1], [0, 0, 0, 0]),
        (0.3, 0.5, [0, 1, 0, 1], [0, 0, 0, 0]),
        (0.5, 1.0, [0, 1, 0, 1], [0, 1, 0, 1]),
        (0.5, 1.0, [0, 1, 0, 1], [0, 1, 0, 1]),
        (0.5, 1.0, [0, 1, 0, 1], [0, 1, 0, 1]),
    ]
    state = {"val_calls": 0}

    def fake_epoch(
        model, loader, cfg, optimizer=None, is_spsa=False, loss_fn=None, collect=False
    ):
        if optimizer is not None:  # training pass
            return 0.5, 0.5, 2
        loss, acc, y_true, y_pred = script[state["val_calls"]]
        state["val_calls"] += 1
        if collect:
            return loss, acc, len(y_true), y_true, y_pred
        return loss, acc, len(y_true)

    monkeypatch.setattr(train_mod, "_epoch_loss_acc", fake_epoch)
    train_loader, _ = _loaders()

    def run(metric: str, tag: str):
        state["val_calls"] = 0
        cfg = _Cfg(epochs=5, patience=2, early_metric=metric)
        model = torch.nn.Linear(4, 2)
        result = train(
            model, train_loader, train_loader, cfg, output_dir=tmp_path / tag
        )
        ckpt = torch.load(
            tmp_path / tag / "checkpoints" / "best.pt", map_location="cpu", weights_only=False
        )
        return result, ckpt

    by_loss, ckpt_loss = run("val_loss", "loss")
    assert by_loss["best_epoch"] == 1
    assert by_loss["best_val_metric"] == pytest.approx(0.3)
    assert ckpt_loss["early_metric"] == "val_loss"
    assert by_loss["epochs_run"] == 4  # two non-improving epochs after epoch 1

    by_bal, ckpt_bal = run("balanced_accuracy", "balanced")
    assert by_bal["best_epoch"] == 2
    assert by_bal["best_val_metric"] == pytest.approx(1.0)
    assert ckpt_bal["early_metric"] == "balanced_accuracy"
    assert ckpt_bal["best_val_metric"] == pytest.approx(1.0)
    assert by_bal["epochs_run"] == 5  # two non-improving epochs after epoch 2

    # Checkpoint carries the full recipe provenance.
    for key in ("state_dict", "epoch", "cfg", "commit_hash", "loss", "class_weight",
                "class_weights_source", "balanced_sampler"):
        assert key in ckpt_bal


def test_checkpoint_reload_restores_weights(tmp_path):
    """load_checkpoint restores state_dict and returns the selected epoch."""
    train_loader, _ = _loaders()
    cfg = _Cfg(epochs=1, patience=2)
    model = torch.nn.Linear(4, 2)
    result = train(model, train_loader, train_loader, cfg, output_dir=tmp_path)
    fresh = torch.nn.Linear(4, 2)
    meta = load_checkpoint(fresh, result["checkpoint_path"])
    assert meta["epoch"] == result["best_epoch"]
    for key in model.state_dict():
        assert torch.allclose(fresh.state_dict()[key], model.state_dict()[key])


# --------------------------------------------------------------------------- calibration safety


def test_temperature_fitted_only_on_validation(monkeypatch):
    """fit_calibration sees ID_VAL logits/labels and the documented split tag."""
    model = _Toy()
    train_loader, _ = _loaders(n=12, seed=1)
    val_loader, _ = _loaders(n=6, seed=2)
    val_labels = torch.cat([batch[1] for batch in val_loader]).numpy()
    train_labels = np.array([0, 1] * 6)

    seen: dict = {}

    def fake_fit_temperature(logits, labels, **kwargs):
        seen["labels"] = np.asarray(labels).copy()
        return 2.5

    monkeypatch.setattr(heads_mod, "fit_temperature", fake_fit_temperature)
    provenance = fit_calibration(
        model, val_loader, train_labels, _Cfg(), fit_logit_adjust=False
    )

    assert provenance["status"] == "fitted"
    assert provenance["temperature"] == pytest.approx(2.5)
    assert provenance["fitted_on_split"] == "id_val"
    assert provenance["ood_labels_used"] is False
    assert np.array_equal(seen["labels"], val_labels)
    assert float(model.head.temperature) == pytest.approx(2.5)


def test_calibration_does_not_touch_ood_labels(monkeypatch):
    """The OOD loader is never iterated during calibration; OOD labels stay intact."""
    model = _Toy()
    train_loader, _ = _loaders(n=8, seed=3)
    val_loader, _ = _loaders(n=4, seed=4)
    ood_loader, ood_dataset = _loaders(n=4, seed=5)
    ood_labels = ood_dataset.tensors[1].numpy().copy()
    before = ood_labels.copy()

    real_collect = heads_mod.collect_logits
    seen: dict = {}

    def spy_collect(model_arg, loader):
        seen["loader"] = loader
        return real_collect(model_arg, loader)

    monkeypatch.setattr(heads_mod, "collect_logits", spy_collect)
    provenance = fit_calibration(
        model, val_loader, np.array([0, 1] * 4), _Cfg(), fit_logit_adjust=False
    )

    assert seen["loader"] is val_loader
    assert seen["loader"] is not ood_loader
    assert np.array_equal(ood_labels, before)
    assert provenance["ood_labels_used"] is False


# --------------------------------------------------------------------------- sampler / CLI


def test_balanced_sampler_uses_train_labels():
    """The sampler oversamples the minority TRAIN class and handles empty input."""
    labels = np.array([0] * 9 + [1])
    sampler = build_balanced_sampler(labels, seed=0)
    assert sampler is not None
    draws: list[int] = []
    for _ in range(200):
        draws.extend(int(i) for i in sampler)
    sampled = np.asarray(labels)[draws]
    minority_fraction = float((sampled == 1).mean())
    assert minority_fraction > 0.3  # far above the 0.1 empirical prior
    assert build_balanced_sampler(np.array([]), seed=0) is None


def test_cli_recipe_defaults_and_legacy_flags():
    """The QML CLI exposes the balance flags and documents every default."""
    from scripts import train_qml

    parser = train_qml._build_parser()
    plan, warnings = train_qml._resolve_plan(parser.parse_args(["--dry-run"]))
    assert train_qml._validate_plan(plan) == []
    assert plan["loss"] == "focal"
    assert plan["class_weight"] == "balanced"
    assert plan["early_metric"] == "balanced_accuracy"
    assert plan["temperature_calibration"] is True
    assert plan["balanced_sampler"] is False
    assert any("legacy" in w for w in warnings)

    legacy_args = parser.parse_args(
        [
            "--dry-run",
            "--loss", "ce",
            "--class-weight", "none",
            "--early-metric", "val_loss",
            "--no-temperature-calibration",
            "--balanced-sampler",
        ]
    )
    legacy_plan, _ = train_qml._resolve_plan(legacy_args)
    assert legacy_plan["loss"] == "ce"
    assert legacy_plan["class_weight"] == "none"
    assert legacy_plan["early_metric"] == "val_loss"
    assert legacy_plan["temperature_calibration"] is False
    assert legacy_plan["balanced_sampler"] is True
    assert train_qml._validate_plan(legacy_plan) == []
    json.dumps(legacy_plan)  # provenance-serialisable
