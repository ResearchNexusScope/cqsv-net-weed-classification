"""Synthetic tests for the pure QNN arm on the fresh protocol (QNN-D01).

No real datasets, no network and no committed result numbers are used.  Fake
feature matrices and fake classical accuracies are generated in-test.  Anything
requiring torch/PennyLane/scipy/sklearn skips cleanly when those are absent.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmlpipe import fresh_qnn as fq  # noqa: E402  (import-safe without torch/PennyLane)


# --------------------------------------------------------------------------- #
# Config + projection (no quantum stack required)                             #
# --------------------------------------------------------------------------- #
def test_qnn_config_and_projection_synthetic() -> None:
    cfg = fq.qnn_config(
        seed=7, qubits=3, layers=2, epochs=1, feature_dim=512, batch_size=8
    )
    assert cfg.qubits == 3
    assert cfg.projection_dim == 8 == cfg.feature_dim
    assert cfg.classical_dim == 512
    assert cfg.validate() == []

    raw = np.random.default_rng(0).normal(size=(11, 512)).astype(np.float32)
    meta = {
        "method": "seeded_random_projection",
        "input_dim": 512,
        "target_dim": 8,
        "seed": 7,
        "matrix": (np.random.default_rng(7).standard_normal((512, 8)) / np.sqrt(8)).astype(
            np.float32
        ),
    }
    projected = fq.project_features(raw, meta)
    again = fq.project_features(raw, meta)
    assert projected.shape == (11, 8)
    assert np.array_equal(projected, again)
    np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 1.0, rtol=1e-5)
    with pytest.raises(ValueError, match="expected 512 raw"):
        fq.project_features(raw[:, :16], meta)

    # Over-width vs under-width paths are deterministic without a matrix.
    pad_meta = {"input_dim": 3, "target_dim": 8, "seed": 1, "matrix": None}
    padded = fq.project_features(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), pad_meta)
    assert padded.shape == (1, 8)
    assert padded[0, 0] == pytest.approx(1.0)
    assert padded[0, 1] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Cap / seed determinism                                                      #
# --------------------------------------------------------------------------- #
def test_cap_and_seed_determinism() -> None:
    X = np.random.default_rng(2).normal(size=(40, 3))
    y = np.asarray([0] * 20 + [1] * 20)

    a_x, a_y, a_meta = fq.cap_train(X, y, 6, seed=3)
    b_x, b_y, b_meta = fq.cap_train(X, y, 6, seed=3)
    c_x, c_y, _ = fq.cap_train(X, y, 6, seed=4)

    assert a_x.shape == (6, 3) and a_y.shape == (6,)
    assert set(np.unique(a_y)) == {0, 1}
    assert np.array_equal(a_x, b_x) and np.array_equal(a_y, b_y)
    assert a_meta["train_size_before"] == 40
    assert a_meta["train_size_after"] == 6
    assert a_meta["cap_applied"] is True
    assert not np.array_equal(a_x, c_x)  # different seed -> different subset

    untouched_x, untouched_y, no_meta = fq.cap_train(X, y, 0, seed=3)
    assert untouched_x is X and untouched_y is y
    assert no_meta["cap_applied"] is False


# --------------------------------------------------------------------------- #
# Model builds with fake features                                             #
# --------------------------------------------------------------------------- #
def test_model_builds_trains_and_scores_fake_features() -> None:
    pytest.importorskip("pennylane", reason="PennyLane required for trainable QNN")
    pytest.importorskip("torch", reason="PyTorch required for trainable QNN")
    pytest.importorskip("scipy", reason="metrics require scipy")
    pytest.importorskip("sklearn", reason="metrics require sklearn")

    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(8, 8)).astype(np.float32)
    y_train = np.asarray([0, 1] * 4)
    X_val = rng.normal(size=(4, 8)).astype(np.float32)
    y_val = np.asarray([0, 1, 0, 1])
    X_test = rng.normal(size=(4, 8)).astype(np.float32)
    y_test = np.asarray([0, 1, 0, 1])

    cfg = fq.qnn_config(
        seed=0, qubits=2, layers=1, epochs=1, feature_dim=8, batch_size=4,
        class_weight="balanced",
    )
    model, history = fq.train_qnn_from_features(
        X_train, y_train, X_val, y_val, cfg, seed=0
    )
    assert history["seed"] == 0
    assert history["total_steps"] > 0
    assert history["grad_norm_first_batch"] is not None
    assert history["trainable_param_count"] is not None
    assert history["trainable_param_count"] > 0
    assert "entangling_gate_count" in history
    assert history["projection"]["target_dim"] == 4

    result = fq.evaluate_once(model, X_test, y_test)
    assert result["n"] == 4
    assert result["model"] == "qnn"
    assert len(result["confusion_matrix"]) == 2
    assert all(len(row) == 2 for row in result["confusion_matrix"])
    for key in ("accuracy", "auc", "f1", "precision", "recall"):
        assert key in result


# --------------------------------------------------------------------------- #
# TEST evaluated exactly once                                                 #
# --------------------------------------------------------------------------- #
def test_test_scored_exactly_once() -> None:
    torch = pytest.importorskip("torch", reason="PyTorch required")
    nn = torch.nn
    pytest.importorskip("scipy", reason="metrics require scipy")
    pytest.importorskip("sklearn", reason="metrics require sklearn")

    class _FakeQNN(nn.Module):
        def __init__(self, n: int) -> None:
            super().__init__()
            self.calls = 0
            self.dummy = nn.Parameter(torch.zeros(1))
            rng = np.random.default_rng(5)
            self.fixed = torch.tensor(rng.normal(size=(n, 2)), dtype=torch.float32)

        def forward(self, x):
            self.calls += 1
            return self.fixed

    X = np.zeros((6, 4), dtype=np.float32)
    y = np.asarray([0, 1, 0, 1, 0, 1])
    model = _FakeQNN(len(y))
    first = fq.evaluate_once(model, X, y)
    assert model.calls == 1
    assert first["test_evaluations"] == 1
    assert first["n"] == 6
    with pytest.raises(RuntimeError, match="exactly once"):
        fq.evaluate_once(model, X, y)
    assert model.calls == 1  # no second forward pass happened


# --------------------------------------------------------------------------- #
# Paired statistics wiring with fake classical values                         #
# --------------------------------------------------------------------------- #
def test_paired_comparison_wiring_fake_classical() -> None:
    pytest.importorskip("scipy", reason="paired_comparison requires scipy")
    qnn = {seed: 0.60 + 0.01 * seed for seed in range(6)}
    classical = {seed: 0.50 + 0.01 * seed for seed in range(6)}
    comparisons = fq.paired_against_classical(qnn, classical)

    assert len(comparisons) == 1
    item = comparisons[0]
    assert item["pair"] == ["qnn", "poly2"]
    assert item["metric"] == "accuracy"
    assert item["n"] == 6
    assert item["seeds"] == list(range(6))
    assert item["p_holm"] is not None  # holm_adjust was applied
    assert item["mean_difference"] > 0  # direction is recorded, not claimed
    assert item["decision"] == "inconclusive"  # n=6 can never be significant
    assert "n<10" in item["decision_reason"]
    assert item["qnn_win_supported"] is False

    # No common seeds -> no comparison, no significance machinery invoked.
    assert fq.paired_against_classical({0: 0.6}, {1: 0.5}) == []


# --------------------------------------------------------------------------- #
# Committed control is read from JSON, not hardcoded                          #
# --------------------------------------------------------------------------- #
def test_load_classical_poly2_from_fake_artifact(tmp_path: Path) -> None:
    fake_file = tmp_path / "results" / "fresh_deepweeds" / "metrics.json"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": [
            {"backbone": "plantnet", "model": "poly2", "seed": 0,
             "status": "ok", "metrics": {"accuracy": 0.63}},
            {"backbone": "plantnet", "model": "poly2", "seed": 1,
             "status": "ok", "metrics": {"accuracy": 0.64}},
            {"backbone": "imagenet", "model": "poly2", "seed": 0,
             "status": "ok", "metrics": {"accuracy": 0.61}},
            {"backbone": "plantnet", "model": "rbf", "seed": 0,
             "status": "ok", "metrics": {"accuracy": 0.62}},
        ]
    }
    fake_file.write_text(json.dumps(payload), encoding="utf-8")
    values = fq.load_classical_poly2("deepweeds", "plantnet", results_root=tmp_path)
    assert values == {0: pytest.approx(0.63), 1: pytest.approx(0.64)}
    assert fq.load_classical_poly2("maizeset", "plantnet", results_root=tmp_path) == {}


# --------------------------------------------------------------------------- #
# CLI dry-run is write-free                                                   #
# --------------------------------------------------------------------------- #
def test_cli_dry_run_is_write_free(tmp_path: Path, capsys) -> None:
    script = ROOT / "scripts" / "run_fresh_qnn.py"
    spec = importlib.util.spec_from_file_location("run_fresh_qnn_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "must_not_exist"
    rc = module.main(
        [
            "--dataset", "deepweeds",
            "--backbone", "plantnet",
            "--seeds", "2",
            "--output-dir", str(output),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not output.exists()
    printed = capsys.readouterr().out
    assert '"dry_run": true' in printed
    assert "CPU feasibility" in printed


# --------------------------------------------------------------------------- #
# Missing cache -> graceful error, no writes                                  #
# --------------------------------------------------------------------------- #
def test_missing_cache_graceful_error(tmp_path: Path) -> None:
    output = tmp_path / "no_run"
    with pytest.raises(FileNotFoundError, match="frozen feature cache not found"):
        fq.run_qnn_protocol(
            "deepweeds",
            "plantnet",
            [0],
            output,
            cache_dir=tmp_path / "empty_cache",
        )
    assert not output.exists()


# --------------------------------------------------------------------------- #
# Claim boundary                                                              #
# --------------------------------------------------------------------------- #
def test_claim_boundary_forbids_advantage_and_requires_power() -> None:
    low = fq.CLAIM_BOUNDARY.lower()
    assert "no quantum advantage is asserted" in low
    assert "power >= 0.8" in low
    assert "4 qubits" in low
    assert "cpu-feasibility" in low
