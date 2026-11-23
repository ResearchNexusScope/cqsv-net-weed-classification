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
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmlpipe import heads as heads_mod  # noqa: E402

# ``qmlpipe.__init__`` re-exports a *function* named ``train``, so a plain
# ``from qmlpipe import train`` would shadow the module. Resolve explicitly.
train_mod = importlib.import_module("qmlpipe.train")  # noqa: E402
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
    # Mixed labels are required for the weight-effect assertion: PyTorch's
    # weighted cross entropy normalises by the sum of the sample weights, so a
    # single-class batch cancels the constant weight out.
    logits = torch.tensor([[4.0, -4.0], [0.1, -0.1], [-3.0, 3.0]])
    labels = torch.tensor([0, 0, 1])

    per_sample_focal = focal_loss(logits, labels, gamma=2.0, reduction="none")
    per_sample_ce = weighted_cross_entropy(logits, labels, reduction="none")

    # The first two samples share class 0; their ratio isolates the focusing
    # effect from any class-prior effect.
    ratio_focal = float(per_sample_focal[0] / per_sample_focal[1])
    ratio_ce = float(per_sample_ce[0] / per_sample_ce[1])
    assert ratio_focal < ratio_ce  # easy sample is relatively suppressed

    assert torch.allclose(
        focal_loss(logits, labels, gamma=0.0),
        weighted_cross_entropy(logits, labels),
        atol=1e-6,
    )

    weights = class_weights_from_labels(labels, n_classes=2, mode="balanced")
    assert float(weights[1]) > float(weights[0])
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
        result = train_mod.train(
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


# ---------------------------------------------------------------------------
# QHEAD-D02: matched labelled adaptation mixing (synthetic only)
#
# Every sample, tensor and feature vector below is built locally.  No project
# result, dataset path or paper number is used.  torch is already imported at
# module scope via importorskip, so heavy-dependent tests skip cleanly.
# ---------------------------------------------------------------------------
def _sample(sample_id: str, label: int, group: str) -> SimpleNamespace:
    """A minimal Sample-like object; group_id is the source-image identity."""
    return SimpleNamespace(
        sample_id=sample_id,
        label=label,
        group_id=group,
        source_image_id=group,
        content_hash="h:" + sample_id,
        path="p:" + sample_id,
        bbox=None,
    )


def _adaptation_pool(n_per_class: int = 6, prefix: str = "ad") -> list:
    samples = []
    for label in (0, 1):
        for i in range(n_per_class):
            samples.append(_sample(f"{prefix}-{label}-{i:02d}", label, f"{prefix}-g-{label}-{i:02d}"))
    return samples


def _fake_loader(feature_by_id: dict, calls: list):
    """Feature loader recording (split, ids); returns vectors keyed by sample id."""

    def loader(samples, cfg, split):
        calls.append({"split": split, "ids": tuple(s.sample_id for s in samples)})
        return np.stack([feature_by_id[s.sample_id] for s in samples]).astype(np.float32)

    return loader


def _feature_map(samples) -> dict:
    return {
        s.sample_id: np.asarray([float(i), float(s.label), 0.25], dtype=np.float32)
        for i, s in enumerate(samples)
    }


def test_adaptation_cli_flags_defaults_and_validation(tmp_path):
    """Flags exist, default to a no-op, and reject inconsistent quotas."""
    from scripts import train_qml

    parser = train_qml._build_parser()
    args = parser.parse_args(["--dry-run"])
    # Raw argparse default is None so choose() can consult --config-json;
    # the resolved plan default is 0.
    assert args.adaptation_samples is None
    assert args.adaptation_per_class is None
    assert args.adaptation_seed is None

    plan, warnings = train_qml._resolve_plan(args)
    assert train_qml._validate_plan(plan) == []
    assert plan["adaptation_samples"] == 0
    assert plan["adaptation_per_class_mix"] is None
    assert plan["adaptation_seed"] is None
    assert any("adaptation" in w.lower() for w in warnings)
    assert any("no OOD improvement" in w or "NOT a claim" in w for w in warnings)

    explicit, _ = train_qml._resolve_plan(
        parser.parse_args(["--adaptation-samples", "50", "--adaptation-seed", "11"])
    )
    assert explicit["adaptation_samples"] == 50
    assert explicit["adaptation_seed"] == 11
    assert train_qml._validate_plan(explicit) == []

    per_class, _ = train_qml._resolve_plan(
        parser.parse_args(["--adaptation-per-class", "25"])
    )
    assert per_class["adaptation_per_class_mix"] == 25
    assert per_class["adaptation_samples"] == 50  # implied total

    conflicting, _ = train_qml._resolve_plan(
        parser.parse_args(["--adaptation-samples", "50", "--adaptation-per-class", "20"])
    )
    assert any("conflicts" in p for p in train_qml._validate_plan(conflicting))

    negative, _ = train_qml._resolve_plan(
        parser.parse_args(["--adaptation-samples", "-1"])
    )
    assert any("non-negative" in p for p in train_qml._validate_plan(negative))

    cfg_file = tmp_path / "adapt.json"
    cfg_file.write_text(json.dumps({"adaptation_samples": 12}), encoding="utf-8")
    from_config, _ = train_qml._resolve_plan(
        parser.parse_args(["--config-json", str(cfg_file)])
    )
    assert from_config["adaptation_samples"] == 12

    # An explicit CLI value (including 0) wins over the config-json override.
    cli_zero, _ = train_qml._resolve_plan(
        parser.parse_args(["--config-json", str(cfg_file), "--adaptation-samples", "0"])
    )
    assert cli_zero["adaptation_samples"] == 0
    cli_seven, _ = train_qml._resolve_plan(
        parser.parse_args(["--config-json", str(cfg_file), "--adaptation-samples", "7"])
    )
    assert cli_seven["adaptation_samples"] == 7

    # Extra plan keys are provenance-only: QMLConfig ignores them.
    cfg = train_qml._build_config(explicit)
    assert "adaptation_samples" not in cfg.as_dict()


def test_adaptation_pool_reads_only_adaptation_split():
    """OOD_TEST is returned only for the disjointness gate, never as a source."""
    from scripts import train_qml

    adaptation = tuple(_adaptation_pool(2))
    ood = tuple(_sample(f"ood-{i}", i % 2, f"ood-g-{i}") for i in range(2))
    train = tuple(_sample(f"tr-{i}", i % 2, f"tr-g-{i}") for i in range(2))
    manifest = SimpleNamespace(
        splits={
            "train": SimpleNamespace(samples=train),
            "adaptation": SimpleNamespace(samples=adaptation),
            "ood_test": SimpleNamespace(samples=ood),
        }
    )
    pool, ood_pool = train_qml._adaptation_pool(manifest)
    assert pool == adaptation
    assert ood_pool == ood
    assert all(s not in pool for s in ood)
    assert all(s not in pool for s in train)
    # A missing ADAPTATION split fails loudly instead of falling back to OOD.
    bad = SimpleNamespace(splits={"ood_test": SimpleNamespace(samples=ood)})
    with pytest.raises(KeyError, match="adaptation"):
        train_qml._adaptation_pool(bad)


def test_adaptation_selection_is_deterministic_and_balanced():
    """Fixed seed gives a stable, order-independent, class-balanced selection."""
    from scripts import train_qml

    pool = _adaptation_pool(n_per_class=8)
    first, info = train_qml._select_adaptation_samples(pool, 6, seed=7)
    again, _ = train_qml._select_adaptation_samples(pool, 6, seed=7)
    reordered, _ = train_qml._select_adaptation_samples(list(reversed(pool)), 6, seed=7)

    assert [s.sample_id for s in first] == [s.sample_id for s in again]
    assert [s.sample_id for s in first] == [s.sample_id for s in reordered]
    assert info["selected"] == 6
    assert info["selected_class_counts"] == {0: 3, 1: 3}
    assert info["selection_seed"] == 7
    assert info["per_class"] is None

    per_class_sel, per_class_info = train_qml._select_adaptation_samples(
        pool, 0, seed=7, per_class=4
    )
    assert len(per_class_sel) == 8
    assert per_class_info["selected_class_counts"] == {0: 4, 1: 4}
    assert per_class_info["requested_total"] == 8

    # Different seeds change the subset for at least one nearby seed.
    variants = [train_qml._select_adaptation_samples(pool, 6, seed=s)[0] for s in (8, 9, 10)]
    assert any(
        [s.sample_id for s in v] != [s.sample_id for s in first] for v in variants
    )


def test_adaptation_appended_to_train_only_and_provenance_recorded():
    """Adaptation features/labels are appended to TRAIN; VAL and OOD are untouched."""
    from scripts import train_qml

    train = [_sample(f"tr-{i}", i % 2, f"tr-g-{i}") for i in range(4)]
    val = [_sample(f"va-{i}", i % 2, f"va-g-{i}") for i in range(2)]
    ood = [_sample(f"oo-{i}", i % 2, f"oo-g-{i}") for i in range(2)]
    adaptation = _adaptation_pool(n_per_class=4)
    fmap = _feature_map(train + val + ood + adaptation)
    calls: list = []
    loader = _fake_loader(fmap, calls)

    X_train = torch.stack([torch.from_numpy(fmap[s.sample_id]) for s in train])
    y_train = torch.tensor([s.label for s in train], dtype=torch.long)
    X_val = torch.stack([torch.from_numpy(fmap[s.sample_id]) for s in val])
    y_val = torch.tensor([s.label for s in val], dtype=torch.long)
    X_ood = torch.stack([torch.from_numpy(fmap[s.sample_id]) for s in ood])
    y_ood = torch.tensor([s.label for s in ood], dtype=torch.long)

    X_out, y_out, prov = train_qml._mix_adaptation_into_train(
        X_train,
        y_train,
        adaptation_samples=adaptation,
        ood_samples=ood,
        feature_loader=loader,
        cfg=SimpleNamespace(),
        total=4,
        seed=7,
    )

    assert tuple(X_out.shape) == (8, 3)
    assert tuple(y_out.shape) == (8,)
    assert torch.equal(X_out[:4], X_train)
    assert torch.equal(y_out[:4], y_train)
    selected_ids = prov["adaptation_sample_ids"]
    expected_features = torch.stack(
        [torch.from_numpy(fmap[sid]) for sid in selected_ids]
    )
    assert torch.equal(X_out[4:], expected_features)
    label_by_id = {s.sample_id: s.label for s in adaptation}
    selected_labels = [label_by_id[sid] for sid in selected_ids]
    assert torch.equal(y_out[4:], torch.tensor(selected_labels, dtype=torch.long))

    # The loader was called with split name "adaptation" and the same selected ids.
    assert calls == [{"split": "adaptation", "ids": tuple(selected_ids)}]

    # VAL/OOD tensors were never mutated.
    assert torch.equal(X_val, torch.stack([torch.from_numpy(fmap[s.sample_id]) for s in val]))
    assert torch.equal(y_val, torch.tensor([s.label for s in val], dtype=torch.long))
    assert torch.equal(X_ood, torch.stack([torch.from_numpy(fmap[s.sample_id]) for s in ood]))
    assert torch.equal(y_ood, torch.tensor([s.label for s in ood], dtype=torch.long))

    # Provenance: source, label source, ids/groups, disjointness, ood_labels_used.
    assert prov["adaptation_samples_used"] == 4
    assert prov["adaptation_source_split"] == train_qml.ADAPTATION_SPLIT_NAME == "adaptation"
    assert prov["adaptation_label_source"] == train_qml.ADAPTATION_LABEL_SOURCE
    assert prov["adaptation_source_image_ids"] == sorted(
        {s.group_id for s in adaptation if s.sample_id in selected_ids}
    )
    assert prov["disjointness_assertion"]["status"] == "passed"
    assert prov["disjointness_assertion"]["passed"] is True
    assert "source_image_id/group_id" in prov["disjointness_assertion"]["checked_fields"]
    assert prov["ood_labels_used"] is False
    assert prov["val_loader_unchanged"] is True
    assert prov["ood_loader_unchanged"] is True
    assert prov["selection_seed"] == 7
    # Provenance states class weights / sampler use the post-mix TRAIN labels.
    assert "augmented TRAIN" in prov["class_weights_and_sampler_source"]
    # Advisory 4: crop-level sampled set, not the classical whole-group draw.
    assert "crop-level" in prov["sample_selection_scope"]
    assert "whole source-image groups" in prov["sample_selection_scope"]
    assert any("crop-level" in w for w in prov["warnings"])
    json.dumps({k: v for k, v in prov.items() if k != "warnings"})


def test_adaptation_disjointness_fails_loudly_on_group_overlap():
    """A shared source-image group aborts mixing before any feature extraction."""
    from scripts import train_qml

    X_train = torch.zeros(2, 3)
    y_train = torch.tensor([0, 1])
    adaptation = [_sample("ad-0", 0, "shared-image")]
    ood = [_sample("oo-0", 1, "shared-image")]
    calls: list = []
    loader = _fake_loader(
        {"ad-0": np.zeros(3, dtype=np.float32), "oo-0": np.zeros(3, dtype=np.float32)},
        calls,
    )
    with pytest.raises(RuntimeError, match="disjointness assertion FAILED"):
        train_qml._mix_adaptation_into_train(
            X_train,
            y_train,
            adaptation_samples=adaptation,
            ood_samples=ood,
            feature_loader=loader,
            cfg=SimpleNamespace(),
            total=1,
            seed=7,
        )
    assert calls == []  # fail before the frozen feature loader ever runs


def test_adaptation_zero_is_a_true_noop():
    """--adaptation-samples 0 returns the original tensors without featurising."""
    from scripts import train_qml

    adaptation = _adaptation_pool()
    ood = [_sample("oo-0", 1, "oo-g-0")]
    X_train = torch.randn(3, 3)
    y_train = torch.tensor([0, 1, 0])
    calls: list = []
    loader = _fake_loader(_feature_map(adaptation + ood), calls)

    X_out, y_out, prov = train_qml._mix_adaptation_into_train(
        X_train,
        y_train,
        adaptation_samples=adaptation,
        ood_samples=ood,
        feature_loader=loader,
        cfg=SimpleNamespace(),
        total=0,
        seed=7,
    )
    assert X_out is X_train
    assert y_out is y_train
    assert calls == []
    assert prov["adaptation_samples_used"] == 0
    assert prov["requested_adaptation_samples"] == 0
    assert prov["ood_labels_used"] is False
    # No check was run, so do not claim it passed.
    disjointness = prov["disjointness_assertion"]
    assert disjointness["status"] == "not_applicable"
    assert disjointness["passed"] is None
    assert "No adaptation samples were selected" in disjointness["reason"]
    assert "original Kaggle TRAIN tensor" in prov["class_weights_and_sampler_source"]
    assert any("crop-level" in w for w in prov["warnings"])


def test_prepare_data_mixes_only_train_and_leaves_val_ood_untouched(monkeypatch):
    """End-to-end _prepare_data: only TRAIN grows; VAL/OOD loaders are identical."""
    import sys
    from enum import Enum

    from scripts import train_qml

    class FakeSplit(str, Enum):
        TRAIN = "train"
        ID_VAL = "id_val"
        ID_TEST = "id_test"
        ADAPTATION = "adaptation"
        OOD_TEST = "ood_test"

    train = [_sample(f"tr-{i}", i % 2, f"tr-g-{i}") for i in range(4)]
    val = [_sample(f"va-{i}", i % 2, f"va-g-{i}") for i in range(2)]
    ood = [_sample(f"oo-{i}", i % 2, f"oo-g-{i}") for i in range(2)]
    adaptation = _adaptation_pool(n_per_class=3)
    fmap = _feature_map(train + val + ood + adaptation)

    def rec(samples):
        return SimpleNamespace(samples=tuple(samples))

    manifest = SimpleNamespace(
        splits={
            FakeSplit.TRAIN: rec(train),
            FakeSplit.ID_VAL: rec(val),
            FakeSplit.ADAPTATION: rec(adaptation),
            FakeSplit.OOD_TEST: rec(ood),
        }
    )

    loader_calls: list = []
    weight_calls: list = []

    def fake_weights(labels, n_classes, mode):
        weight_calls.append(np.asarray(labels).copy())
        return np.ones(2)

    class FakeSubsetManifest:
        def __init__(self, base, ood_samples):
            self.splits = dict(base.splits)
            self.splits[FakeSplit.OOD_TEST] = rec(ood_samples)

    modules: dict[str, SimpleNamespace] = {
        "data": SimpleNamespace(build_splits=lambda *a, **k: manifest),
        "data.splits": SimpleNamespace(Split=FakeSplit),
        "qmlpipe.plant_backbone": SimpleNamespace(
            build_backbone_extractor=lambda name, cfg: (object(), object())
        ),
        "qmlpipe.losses": SimpleNamespace(
            class_weights_from_labels=fake_weights
        ),
        "qmlpipe.train": SimpleNamespace(
            build_balanced_sampler=lambda labels, seed: None
        ),
        "scripts.run_real_benchmark": SimpleNamespace(
            SubsetManifest=FakeSubsetManifest,
            cap_ood_by_group=lambda samples, per_class: list(samples),
            make_feature_loader=lambda extractor, preprocess, device: _fake_loader(
                fmap, loader_calls
            ),
            prepare_ninja_subset=lambda n: None,
        ),
    }

    @dataclasses.dataclass
    class FakeExpConfig:
        output_dir: str = "results"
        n_qubits: int = 10
        feature_dim: int = 1024
        qnn_layers: int = 2
        epochs: int = 1
        batch_size: int = 32
        learning_rate: float = 1e-3
        seed: int = 42
        adaptation_per_class: int = 200

        @classmethod
        def from_env(cls):
            return cls()

    modules["experiments.config"] = SimpleNamespace(ExperimentConfig=FakeExpConfig)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        train_qml,
        "_build_config",
        lambda plan: SimpleNamespace(qubits=3, classical_dim=3, projection_dim=8),
    )
    monkeypatch.setenv("KAGGLE_DATA_DIR", "synthetic/kaggle")
    monkeypatch.setenv("NINJA_ANN_DIR", "synthetic/ann")
    monkeypatch.setenv("NINJA_IMG_DIR", "synthetic/img")

    parser = train_qml._build_parser()
    plan, _ = train_qml._resolve_plan(
        parser.parse_args(["--adaptation-samples", "4", "--backbone", "imagenet"])
    )
    bundle = train_qml._prepare_data(plan, smoke=False)

    assert bundle["counts"] == {"train": 8, "val": 2, "ood": 2}
    assert bundle["adaptation"]["adaptation_samples_used"] == 4
    assert bundle["adaptation"]["ood_labels_used"] is False
    assert "augmented TRAIN" in bundle["adaptation"]["class_weights_and_sampler_source"]

    train_ds = bundle["train_loader"].dataset
    assert tuple(train_ds.tensors[0].shape) == (8, 3)
    assert torch.equal(
        train_ds.tensors[0][:4],
        torch.stack([torch.from_numpy(fmap[s.sample_id]) for s in train]),
    )
    selected_ids = bundle["adaptation"]["adaptation_sample_ids"]
    assert torch.equal(
        train_ds.tensors[0][4:],
        torch.stack([torch.from_numpy(fmap[sid]) for sid in selected_ids]),
    )
    label_by_id = {s.sample_id: s.label for s in adaptation}
    assert torch.equal(
        train_ds.tensors[1][4:],
        torch.tensor([label_by_id[sid] for sid in selected_ids]),
    )

    val_ds = bundle["val_loader"].dataset
    ood_ds = bundle["ood_loader"].dataset
    assert torch.equal(
        val_ds.tensors[0],
        torch.stack([torch.from_numpy(fmap[s.sample_id]) for s in val]),
    )
    assert torch.equal(
        ood_ds.tensors[0],
        torch.stack([torch.from_numpy(fmap[s.sample_id]) for s in ood]),
    )
    assert torch.equal(
        ood_ds.tensors[1], torch.tensor([s.label for s in ood])
    )

    splits_seen = [c["split"] for c in loader_calls]
    assert splits_seen == ["train", "id_val", "ood_test", "adaptation"]
    assert loader_calls[-1]["ids"] == tuple(selected_ids)

    # Advisory 2: class weights are computed from the POST-MIX label vector.
    assert len(weight_calls) == 1
    assert len(weight_calls[0]) == 8  # 4 original TRAIN + 4 adaptation
    assert np.array_equal(
        weight_calls[0], train_ds.tensors[1].numpy()
    )  # exactly the augmented TRAIN labels
    assert not np.array_equal(weight_calls[0], np.array([s.label for s in train]))
