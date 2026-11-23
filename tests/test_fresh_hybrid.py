"""Synthetic tests for the HYBRID-D01 classical + quantum arm.

No dataset, network, real result number or private-paper content is used.  Every
feature row, label and baseline value comes from a local fixture or a fixed RNG.
Heavy dependencies are optional: torch is required for this module (cleanly
skipped when absent), PennyLane is required only for the real-circuit test, and
scipy/sklearn are required only where metrics or paired tests are exercised.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="torch is not installed")
from qmlpipe import fresh_hybrid as fh  # noqa: E402
from qmlpipe import models as models_mod  # noqa: E402


class _FakeQNN(torch.nn.Module):
    """Stand-in for build_qnn: trainable, 2-logit, no PennyLane needed."""

    def __init__(self, input_dim: int = 16, out: int = 2) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = torch.nn.Linear(self.input_dim, out)

    def forward(self, x):
        return self.net(x)


def _use_fake_qnn(monkeypatch, input_dim: int = 16):
    monkeypatch.setattr(models_mod, "build_qnn", lambda cfg: _FakeQNN(input_dim))


def _cfg(**overrides):
    base = {
        "seed": 3,
        "qubits": 4,
        "layers": 2,
        "epochs": 1,
        "hybrid_hidden": 8,
        "batch_size": 4,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "class_weight": "balanced",
        "max_batches_per_epoch": 2,
    }
    base.update(overrides)
    return base


def _synthetic_arrays(n_train=24, n_val=8, n_test=8, width=6, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n_train, width)).astype(np.float32)
    y_train = np.asarray([i % 2 for i in range(n_train)], dtype=np.int64)
    X_val = rng.normal(size=(n_val, width)).astype(np.float32)
    y_val = np.asarray([i % 2 for i in range(n_val)], dtype=np.int64)
    X_test = rng.normal(size=(n_test, width)).astype(np.float32)
    y_test = np.asarray([i % 2 for i in range(n_test)], dtype=np.int64)
    return X_train, y_train, X_val, y_val, X_test, y_test


# --------------------------------------------------------------------------- #
# Architecture: provably hybrid                                                 #
# --------------------------------------------------------------------------- #
def test_hybrid_builds_with_both_trainable_branches_and_correct_shape(monkeypatch) -> None:
    _use_fake_qnn(monkeypatch)
    model = fh.build_hybrid_model(_cfg(), feature_dim=6)

    assert isinstance(model.classical, torch.nn.Sequential)
    assert isinstance(model.qnn, torch.nn.Module)
    assert isinstance(model.quantum_adapter, torch.nn.Linear)
    assert isinstance(model.classifier, torch.nn.Linear)

    counts = model.trainable_parameter_counts()
    assert counts["classical"] > 0
    assert counts["quantum"] > 0
    assert counts["fusion"] > 0
    assert counts["total"] == counts["classical"] + counts["quantum"] + counts["fusion"]
    assert "classical" in fh.ARCHITECTURE["description"].lower()
    assert "quantum" in fh.ARCHITECTURE["description"].lower()

    branches = model.branch_outputs(torch.zeros(5, 6))
    assert branches["classical"].shape == (5, 8)
    assert branches["quantum"].shape == (5, 2)
    assert branches["fused"].shape == (5, 10)
    assert model(torch.zeros(5, 6)).shape == (5, 2)
    with pytest.raises(ValueError, match="hybrid expects"):
        model(torch.zeros(5, 7))

    # Both branches are on the inference path: zeroing either one changes logits.
    x = torch.ones(2, 6)
    baseline = model(x).detach().clone()
    original_classical = model.classical.forward
    original_quantum = model.qnn.forward
    try:
        model.classical.forward = lambda _: torch.zeros(2, 8)
        without_classical = model(x).detach().clone()
        model.classical.forward = original_classical
        model.qnn.forward = lambda _: torch.zeros(2, 2)
        without_quantum = model(x).detach().clone()
    finally:
        model.classical.forward = original_classical
        model.qnn.forward = original_quantum
    assert not torch.allclose(baseline, without_classical)
    assert not torch.allclose(baseline, without_quantum)


def test_real_qnn_branch_is_entangled_and_trainable() -> None:
    pytest.importorskip("pennylane", reason="PennyLane is not installed")
    from qmlpipe.config import QMLConfig
    from qmlpipe.models import circuit_introspection

    cfg = QMLConfig(seed=5, qubits=4, layers=2, projection_dim=16)
    model = fh.build_hybrid_model(cfg, feature_dim=12)
    introspection = circuit_introspection(model.qnn)
    assert introspection["entangling_gate_count"] > 0
    assert introspection["trainable_param_count"] > 0
    assert model.trainable_parameter_counts()["quantum"] > 0
    assert model(torch.zeros(3, 12)).shape == (3, 2)


# --------------------------------------------------------------------------- #
# Training: bounded, seeded, both branches receive gradients                   #
# --------------------------------------------------------------------------- #
def test_train_from_features_is_bounded_deterministic_and_hybrid(monkeypatch) -> None:
    _use_fake_qnn(monkeypatch)
    Xtr, ytr, Xva, yva, _, _ = _synthetic_arrays()

    model_a, history_a = fh.train_hybrid_from_features(Xtr, ytr, Xva, yva, _cfg(), seed=5)
    model_b, history_b = fh.train_hybrid_from_features(Xtr, ytr, Xva, yva, _cfg(), seed=5)
    model_c, history_c = fh.train_hybrid_from_features(Xtr, ytr, Xva, yva, _cfg(), seed=6)

    # CPU-feasibility ceiling is respected.
    assert all(entry["n_batches"] <= 2 for entry in history_a["epochs"])
    assert len(history_a["epochs"]) == 1
    assert history_a["n_train"] == 24 and history_a["n_val"] == 8
    assert history_a["class_weight"] == "balanced"
    assert history_a["test_labels_used"] is False

    # Both paths are trainable and actually receive gradients.
    assert history_a["classical_grad_norm"] > 0
    assert history_a["quantum_grad_norm"] > 0
    assert model_a.trainable_parameter_counts()["classical"] > 0
    assert model_a.trainable_parameter_counts()["quantum"] > 0

    for key in model_a.state_dict():
        torch.testing.assert_close(model_a.state_dict()[key], model_b.state_dict()[key])
    assert history_a == history_b
    assert any(
        not torch.allclose(model_a.state_dict()[key], model_c.state_dict()[key])
        for key in model_a.state_dict()
    )


def test_train_rejects_single_class_train() -> None:
    X = np.zeros((6, 4), dtype=np.float32)
    y = np.zeros(6, dtype=np.int64)
    with pytest.raises(ValueError, match="both classes"):
        fh.train_hybrid_from_features(X, y, X[:2], y[:2], _cfg(), seed=1)


# --------------------------------------------------------------------------- #
# Sealed TEST                                                                   #
# --------------------------------------------------------------------------- #
def test_evaluate_once_scores_test_exactly_once(monkeypatch) -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    _use_fake_qnn(monkeypatch)
    Xtr, ytr, Xva, yva, Xte, yte = _synthetic_arrays()
    model, _ = fh.train_hybrid_from_features(Xtr, ytr, Xva, yva, _cfg(), seed=2)

    result = fh.evaluate_once(model, Xte, yte)
    assert {"accuracy", "auc", "f1", "precision", "recall", "confusion_matrix", "n"} <= set(result)
    assert result["n"] == len(yte) == result["n_test_rows"]
    assert result["test_evaluations"] == 1
    assert sum(map(sum, result["confusion_matrix"])) == len(yte)
    assert "no quantum advantage" not in result["architecture"].lower()  # architecture != claim boundary

    with pytest.raises(RuntimeError, match="already been scored once"):
        fh.evaluate_once(model, Xte, yte)
    assert model._test_eval_count == 1


# --------------------------------------------------------------------------- #
# Protocol orchestration / paired wiring                                        #
# --------------------------------------------------------------------------- #
def _install_fake_cache(monkeypatch, Xtr, ytr, Xva, yva, Xte, yte):
    def fake_cache(dataset, backbone, root=None, cache_dir=None, *, seed=42,
                   ratios=fh.fp.DEFAULT_RATIOS, force=False):
        return {
            "X": {"train": Xtr, "val": Xva, "test": Xte},
            "y": {"train": ytr, "val": yva, "test": yte},
            "groups": {"train": ["g0"], "val": ["g1"], "test": ["g2"]},
            "cache_path": "fake-cache.npz",
            "cache_hit": True,
            "provenance": {"split": {"seed": seed, "group_disjoint": True}},
        }

    monkeypatch.setattr(fh.fp, "cache_features", fake_cache)


def test_run_protocol_builds_paired_comparisons_and_writes_additive_artifacts(monkeypatch, tmp_path) -> None:
    pytest.importorskip("scipy")
    Xtr, ytr, Xva, yva, Xte, yte = _synthetic_arrays()
    _install_fake_cache(monkeypatch, Xtr, ytr, Xva, yva, Xte, yte)

    trained_seeds: list[int] = []

    def fake_train(X_train, y_train, X_val, y_val, cfg, seed):
        trained_seeds.append(int(seed))
        assert len(y_train) == 8  # train_cap applied before training
        history = {
            "parameter_counts": {"classical": 1, "quantum": 1, "fusion": 1, "total": 3},
            "classical_grad_norm": 0.1,
            "quantum_grad_norm": 0.1,
        }
        return types.SimpleNamespace(seed=int(seed), _test_eval_count=0), history

    def fake_evaluate(model, X_test, y_test):
        assert len(y_test) == len(yte)  # TEST never capped
        return {
            "accuracy": 0.6 + 0.01 * int(model.seed),
            "auc": 0.7,
            "f1": 0.6,
            "precision": 0.6,
            "recall": 0.6,
            "n": len(y_test),
        }

    def fake_loader(path, backbone, seeds, model_names):
        names = {str(name) for name in model_names}
        if "qnn" in names or "pure_qnn" in names:
            return {int(s): 0.55 + 0.01 * int(s) for s in seeds}
        return {int(s): 0.5 + 0.005 * int(s) for s in seeds}

    monkeypatch.setattr(fh, "train_hybrid_from_features", fake_train)
    monkeypatch.setattr(fh, "evaluate_once", fake_evaluate)
    monkeypatch.setattr(fh, "_load_per_seed_accuracy", fake_loader)

    output_dir = tmp_path / "out"
    result = fh.run_hybrid_protocol(
        "deepweeds", "plantnet", 4, str(output_dir),
        train_cap=8, epochs=1, qubits=4, layers=2,
        qnn_results=str(tmp_path / "peer_qnn_metrics.json"),
    )
    metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    provenance = json.loads(Path(result["provenance_path"]).read_text(encoding="utf-8"))

    assert trained_seeds == [0, 1, 2, 3]
    assert metrics["n_runs_expected"] == 4
    assert metrics["n_runs_succeeded"] == 4
    assert metrics["accounting_invariant"] is True
    assert metrics["aggregates"]["plantnet/hybrid"]["accuracy"]["n"] == 4
    for run in metrics["runs"]:
        assert run["train_cap"]["train_size_after"] == 8
        assert run["metrics"]["n"] == len(yte)

    comparisons = metrics["comparisons"]
    assert len(comparisons) == 2
    labels = [item["pair"][1] for item in comparisons]
    assert any("poly2" in label for label in labels)
    assert any("qnn" in label for label in labels)
    for item in comparisons:
        assert item["p_holm"] is not None
        assert item["multiplicity"]["method"] == "holm"
        assert item["decision"] == "inconclusive"  # n=4 < 10
        assert item["mean_difference"] is not None
        assert item["seeds"] == [0, 1, 2, 3]

    boundary = metrics["claim_boundary"].lower()
    assert "no quantum advantage is asserted" in boundary
    assert "linear" in boundary and "quantum" in boundary  # exact architecture description included
    assert provenance["claim_boundary"] == metrics["claim_boundary"]
    assert provenance["architecture"]["type"].startswith("parallel-concatenation hybrid")
    assert provenance["comparison_notes"]["no_advantage_claim"]
    assert metrics["config"]["qubits"] == 4 and metrics["config"]["layers"] == 2

    # Additive: a second run goes to a fresh child path and leaves the first intact.
    original = Path(result["metrics_path"]).read_text(encoding="utf-8")
    second = fh.run_hybrid_protocol(
        "deepweeds", "plantnet", 1, str(output_dir), train_cap=8, epochs=1
    )
    assert Path(second["metrics_path"]) != Path(result["metrics_path"])
    assert Path(result["metrics_path"]).read_text(encoding="utf-8") == original


def test_missing_cache_is_a_warning_not_a_fabricated_value(monkeypatch, tmp_path) -> None:
    def missing_cache(*args, **kwargs):
        raise FileNotFoundError("synthetic missing cache")

    def forbidden(*args, **kwargs):
        raise AssertionError("training must not run when the cache is missing")

    monkeypatch.setattr(fh.fp, "cache_features", missing_cache)
    monkeypatch.setattr(fh, "train_hybrid_from_features", forbidden)
    monkeypatch.setattr(fh, "evaluate_once", forbidden)
    monkeypatch.setattr(fh, "_load_per_seed_accuracy", lambda *a, **k: (_ for _ in ()).throw(
        FileNotFoundError("synthetic no baseline")
    ))

    result = fh.run_hybrid_protocol("acre", "plantnet", 2, str(tmp_path / "out"))
    metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    assert metrics["n_runs_expected"] == 2
    assert metrics["n_runs_succeeded"] == 0
    assert metrics["n_runs_skipped"] == 2
    assert metrics["accounting_invariant"] is True
    assert metrics["runs"] == []
    assert metrics["aggregates"] == {}
    assert metrics["comparisons"] == []
    assert [w["error_type"] for w in metrics["warnings"][:2]] == ["FileNotFoundError"] * 2


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #
def test_cli_dry_run_is_write_free_and_exposes_all_flags(monkeypatch, tmp_path, capsys) -> None:
    from scripts import run_fresh_hybrid as cli

    def forbidden(*args, **kwargs):
        raise AssertionError("run_hybrid_protocol must not be called during a dry run")

    monkeypatch.setattr(fh, "run_hybrid_protocol", forbidden)
    output_dir = tmp_path / "no-writes"
    code = cli.main(
        [
            "--dataset", "deepweeds",
            "--backbone", "imagenet",
            "--seeds", "4",
            "--train-cap", "100",
            "--epochs", "2",
            "--qubits", "3",
            "--layers", "1",
            "--qnn-results", str(tmp_path / "qnn.json"),
            "--output-dir", str(output_dir),
            "--cache-dir", str(tmp_path / "cache"),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not output_dir.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    plan = payload["plan"]
    assert plan["dataset"] == "deepweeds"
    assert plan["backbone"] == "imagenet"
    assert plan["seeds"] == [0, 1, 2, 3]
    assert plan["train_cap"] == 100
    assert plan["epochs"] == 2
    assert plan["qubits"] == 3
    assert plan["layers"] == 1
    assert plan["qnn_results"] == str(tmp_path / "qnn.json")
    assert plan["runs_planned"] == 4
    assert "no quantum advantage is asserted" in plan["claim_boundary"].lower()
    assert "cpu" in plan["cpu_qubit_caveat"].lower()

    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for flag in (
        "--dataset", "--backbone", "--seeds", "--train-cap", "--epochs",
        "--qubits", "--layers", "--qnn-results", "--output-dir", "--cache-dir", "--dry-run",
    ):
        assert flag in help_text


def test_no_advantage_claim_is_encoded_in_module_constants() -> None:
    assert "no quantum advantage is asserted" in fh.CLAIM_BOUNDARY.lower()
    assert fh.ARCHITECTURE["description"] in fh.CLAIM_BOUNDARY
    assert "not a pure qnn" in fh.CLAIM_BOUNDARY.lower()
