"""Synthetic tests for the genuine QCNN arm on the fresh protocol (QCNN-D01).

No real datasets, no network, no committed result numbers.  Fake feature
matrices and fake baseline accuracies are generated in-test.  Anything needing
torch / PennyLane / scipy / sklearn skips cleanly when those are absent; the
structural QCNN evidence and the protocol wiring are therefore verifiable in a
lightweight environment.
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

from qmlpipe import fresh_qcnn as fq  # noqa: E402  (import-safe without torch/PennyLane)


# --------------------------------------------------------------------------- #
# 1. Provable convolution+pooling structure and SHARED parameters             #
# --------------------------------------------------------------------------- #
def test_shared_convolution_parameters_structural() -> None:
    spec = fq.circuit_spec(8, "single_ancilla")
    assert spec["architecture"] == fq.ARCHITECTURE["type"]
    assert spec["is_convolution_pooling_qcnn"] is True
    # One shared vector of six parameters, regardless of the number of blocks.
    assert spec["shared_conv_parameter_vectors"] == 1
    assert spec["conv_parameter_vector_size"] == fq.CONV_PARAM_VECTOR_SIZE == 6
    assert spec["total_conv_trainable_parameters"] == fq.CONV_PARAM_VECTOR_SIZE
    assert spec["conv_block_count"] > 1
    # One entangling CNOT per local two-qubit convolution block.
    assert spec["entangling_gate_count"] == spec["conv_block_count"]
    assert spec["conv_reps"] == 2
    assert spec["pooling_depth"] > 1
    assert spec["amplitude_dim"] == 2 ** 8
    assert spec["pca_components"] == 2 ** 8 - 1  # one amplitude is the ancilla


def test_pooling_halves_the_active_register() -> None:
    for mode in fq.POOLING_MODES:
        plan = fq.pooling_plan(8, mode)
        assert plan["active_qubit_schedule"] == [8, 4, 2, 1]
        for stage in plan["stages"]:
            assert not stage["skipped"]  # power-of-two widths never skip
            active, surviving, pooled = (
                stage["active_wires"],
                stage["surviving_wires"],
                stage["pooled_wires"],
            )
            assert len(surviving) == len(active) // 2
            assert set(surviving).isdisjoint(pooled)
            assert set(surviving) | set(pooled) == set(active)
            # Two brickwork sweeps: even then odd offset.
            assert len(stage["conv_pairs"]) == plan["conv_reps"]
        assert plan["readout_wires"] == [0]


def test_single_ancilla_and_skip_modes_build_differently() -> None:
    padded = fq.pooling_plan(3, "single_ancilla")
    skipped = fq.pooling_plan(3, "skip")
    assert padded["virtual_wire_count"] == 1
    assert any(stage["ancilla_added"] for stage in padded["stages"])
    assert not any(stage["skipped"] for stage in padded["stages"])
    assert padded["active_qubit_schedule"][-1] == 1  # reaches a single readout wire

    assert skipped["virtual_wire_count"] == 0
    assert any(stage["skipped"] for stage in skipped["stages"])
    assert not any(stage["ancilla_added"] for stage in skipped["stages"])
    assert len(skipped["readout_wires"]) == 3  # odd register measured as-is

    padded_spec = fq.circuit_spec(3, "single_ancilla")
    skipped_spec = fq.circuit_spec(3, "skip")
    assert padded_spec["single_ancilla_padding"] is True
    assert padded_spec["skip_pooling_used"] is False
    assert skipped_spec["single_ancilla_padding"] is False
    assert skipped_spec["skip_pooling_used"] is True

    with pytest.raises(ValueError, match="unknown pooling mode"):
        fq.pooling_plan(8, "not_a_mode")


def test_conv_pairs_are_brickwork() -> None:
    even = fq.conv_pairs([0, 1, 2, 3, 4, 5], offset=0)
    odd = fq.conv_pairs([0, 1, 2, 3, 4, 5], offset=1)
    assert even == [(0, 1), (2, 3), (4, 5)]
    assert odd == [(1, 2), (3, 4)]
    # Position-based pairing also works on a non-contiguous pooled register.
    assert fq.conv_pairs([0, 2, 4, 6], offset=0) == [(0, 2), (4, 6)]


# --------------------------------------------------------------------------- #
# 2. TRAIN-only projection                                                    #
# --------------------------------------------------------------------------- #
def test_projection_is_fitted_on_train_only() -> None:
    rng = np.random.default_rng(31)
    X_train = rng.normal(size=(12, 5)).astype(np.float32)
    X_test = rng.normal(size=(4, 5)).astype(np.float32) + 7.0  # different distribution

    meta = fq.fit_qcnn_projection(X_train, qubits=3, pooling="single_ancilla", seed=5)
    assert meta["input_dim"] == 5
    assert meta["pca_components"] == 7  # 2**3 - one ancilla
    assert meta["amplitude_dim"] == 8
    assert meta["single_ancilla_padding"] is True
    if meta["mean"] is not None:  # PCA path (sklearn available)
        np.testing.assert_allclose(meta["mean"], X_train.mean(axis=0), rtol=1e-6)
        assert not np.allclose(meta["mean"], X_test.mean(axis=0))
    saved_meta = json.loads(json.dumps(fq._json_safe(meta)))

    projected = fq.project_qcnn_features(X_test, meta)
    assert projected.shape == (len(X_test), 8)
    np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 1.0, rtol=1e-5)
    # The single ancilla amplitude is exactly zero.
    np.testing.assert_allclose(projected[:, -1], 0.0, atol=1e-7)
    # Scoring cannot mutate the fitted transform.
    assert fq._json_safe(meta) == saved_meta

    skip_meta = fq.fit_qcnn_projection(X_train, qubits=3, pooling="skip", seed=5)
    assert skip_meta["pca_components"] == 8
    assert skip_meta["single_ancilla_padding"] is False
    assert fq.project_qcnn_features(X_test, skip_meta).shape == (len(X_test), 8)

    with pytest.raises(ValueError, match="expected 5 raw cached"):
        fq.project_qcnn_features(X_test[:, :3], meta)


# --------------------------------------------------------------------------- #
# 3. Deterministic TRAIN cap                                                  #
# --------------------------------------------------------------------------- #
def test_cap_is_deterministic_per_seed() -> None:
    X = np.random.default_rng(2).normal(size=(40, 3))
    y = np.asarray([0] * 20 + [1] * 20)
    a_x, a_y, a_meta = fq.cap_train(X, y, 6, seed=3)
    b_x, b_y, _ = fq.cap_train(X, y, 6, seed=3)
    c_x, _, _ = fq.cap_train(X, y, 6, seed=4)
    assert a_x.shape == (6, 3) and a_y.shape == (6,)
    assert set(np.unique(a_y)) == {0, 1}
    assert np.array_equal(a_x, b_x) and np.array_equal(a_y, b_y)
    assert not np.array_equal(a_x, c_x)
    assert a_meta["train_size_before"] == 40 and a_meta["train_size_after"] == 6
    untouched_x, untouched_y, no_meta = fq.cap_train(X, y, 0, seed=3)
    assert untouched_x is X and untouched_y is y
    assert no_meta["cap_applied"] is False


# --------------------------------------------------------------------------- #
# 4. Forward shape, trainable-parameter audit, sealed TEST                    #
# --------------------------------------------------------------------------- #
def test_model_forward_shape_and_parameter_audit() -> None:
    torch = pytest.importorskip("torch", reason="PyTorch required for QCNN forward")
    pytest.importorskip("pennylane", reason="PennyLane required for QCNN circuit")

    model = fq.build_qcnn_model(qubits=2, pooling="single_ancilla", seed=0)
    X = np.random.default_rng(7).normal(size=(5, 4)).astype(np.float32)
    logits = model(torch.from_numpy(X))
    assert tuple(logits.shape) == (5, 2)
    # Exactly six shared convolution parameters + Linear(1, 2) head (4 values).
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == fq.CONV_PARAM_VECTOR_SIZE + 4
    assert model.spec["shared_conv_parameter_vectors"] == 1


def test_train_and_score_synthetic_features() -> None:
    torch = pytest.importorskip("torch", reason="PyTorch required for training")
    pytest.importorskip("pennylane", reason="PennyLane required for QCNN circuit")
    pytest.importorskip("scipy", reason="metrics require scipy")
    pytest.importorskip("sklearn", reason="metrics require sklearn")

    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(8, 4)).astype(np.float32)
    y_train = np.asarray([0, 1] * 4)
    X_val = rng.normal(size=(4, 4)).astype(np.float32)
    y_val = np.asarray([0, 1, 0, 1])
    X_test = rng.normal(size=(4, 4)).astype(np.float32)
    y_test = np.asarray([0, 1, 0, 1])

    model, history = fq.train_qcnn_from_features(
        X_train, y_train, X_val, y_val, qubits=2, epochs=1, seed=0, batch_size=4
    )
    assert history["total_steps"] > 0
    assert history["grad_norm_first_batch"] is not None
    assert history["circuit_spec"]["shared_conv_parameter_vectors"] == 1
    assert history["circuit_spec"]["active_qubit_schedule"] == [2, 1]
    assert history["projection"]["pca_components"] == 3  # 2**2 minus one ancilla
    assert np.isfinite(history["param_delta"])

    result = fq.evaluate_once(model, X_test, y_test)
    assert result["model"] == "qcnn"
    assert result["n"] == len(y_test)
    assert result["test_evaluations"] == 1
    for key in ("accuracy", "auc", "f1", "precision", "recall", "confusion_matrix"):
        assert key in result
    with pytest.raises(RuntimeError, match="exactly once"):
        fq.evaluate_once(model, X_test, y_test)


def test_test_scored_exactly_once_with_fake_model() -> None:
    torch = pytest.importorskip("torch", reason="PyTorch required")
    nn = torch.nn
    pytest.importorskip("scipy", reason="metrics require scipy")
    pytest.importorskip("sklearn", reason="metrics require sklearn")

    class _FakeQCNN(nn.Module):
        def __init__(self, n: int) -> None:
            super().__init__()
            self.calls = 0
            self.dummy = nn.Parameter(torch.zeros(1))
            self.fixed = torch.tensor(
                np.random.default_rng(5).normal(size=(n, 2)), dtype=torch.float32
            )

        def forward(self, x):
            self.calls += 1
            return self.fixed

    X = np.zeros((6, 4), dtype=np.float32)
    y = np.asarray([0, 1, 0, 1, 0, 1])
    model = _FakeQCNN(len(y))
    first = fq.evaluate_once(model, X, y)
    assert first["test_evaluations"] == 1 and first["n"] == 6
    assert model.calls == 1
    with pytest.raises(RuntimeError, match="exactly once"):
        fq.evaluate_once(model, X, y)
    assert model.calls == 1  # no second forward pass happened


# --------------------------------------------------------------------------- #
# 5. Paired statistics wiring with fake baselines                             #
# --------------------------------------------------------------------------- #
def test_paired_wiring_with_fake_baselines() -> None:
    pytest.importorskip("scipy", reason="paired_comparison requires scipy")
    qcnn = {seed: 0.40 + 0.01 * seed for seed in range(6)}
    baselines = {
        "poly2": {seed: 0.30 + 0.01 * seed for seed in range(6)},
        "qnn_amplitude": {seed: 0.35 + 0.01 * seed for seed in range(6)},
    }
    comparisons = fq.paired_against_baselines(qcnn, baselines)
    assert len(comparisons) == 2
    assert {item["baseline"] for item in comparisons} == set(baselines)
    for item in comparisons:
        assert item["pair"] == ["qcnn", item["baseline"]]
        assert item["n"] == 6
        assert item["seeds"] == list(range(6))
        assert item["p_holm"] is not None  # holm_adjust was applied
        assert item["decision"] == "inconclusive"  # n < 10 can never be significant
        assert "n<10" in item["decision_reason"]
        assert item["qnn_win_supported"] is False
    assert fq.paired_against_baselines({0: 0.6}, {"poly2": {1: 0.5}}) == []


def test_baselines_are_read_from_artifacts_not_hardcoded(tmp_path: Path) -> None:
    fake_classical = tmp_path / "classical.json"
    fake_classical.write_text(
        json.dumps(
            {
                "runs": [
                    {"backbone": "plantnet", "model": "poly2", "seed": 0,
                     "status": "ok", "metrics": {"accuracy": 0.17}},
                    {"backbone": "plantnet", "model": "poly2", "seed": 1,
                     "status": "ok", "metrics": {"accuracy": 0.42}},
                    {"backbone": "imagenet", "model": "poly2", "seed": 0,
                     "status": "ok", "metrics": {"accuracy": 0.55}},
                    {"backbone": "plantnet", "model": "rbf", "seed": 0,
                     "status": "ok", "metrics": {"accuracy": 0.61}},
                ]
            }
        ),
        encoding="utf-8",
    )
    # Mirror the committed relative path so load_baselines() can find it too.
    mirror = tmp_path / "results" / "fresh_deepweeds"
    mirror.mkdir(parents=True, exist_ok=True)
    mirror.joinpath("metrics.json").write_text(fake_classical.read_text(encoding="utf-8"), encoding="utf-8")

    fake_qnn = tmp_path / "old_qnn.json"
    fake_qnn.write_text(
        json.dumps(
            {"runs": [
                {"backbone": "plantnet", "model": "qnn", "seed": 0,
                 "status": "ok", "metrics": {"accuracy": 0.23}}
            ]}
        ),
        encoding="utf-8",
    )
    values = fq.load_baseline(
        fake_classical, "plantnet", model_names=("poly2",), results_root=tmp_path
    )
    assert values == {0: pytest.approx(0.17), 1: pytest.approx(0.42)}
    assert fq.load_baseline(tmp_path / "absent.json", "plantnet", model_names=("poly2",)) == {}

    loaded = fq.load_baselines(
        "deepweeds", "plantnet", results_root=tmp_path, qnn_results=fake_qnn
    )
    assert loaded["poly2"] == values
    assert loaded["qnn_amplitude"] == {0: pytest.approx(0.23)}


# --------------------------------------------------------------------------- #
# 6. CLI dry-run is write-free                                                #
# --------------------------------------------------------------------------- #
def test_cli_dry_run_is_write_free(tmp_path: Path, capsys) -> None:
    script = ROOT / "scripts" / "run_fresh_qcnn.py"
    spec = importlib.util.spec_from_file_location("run_fresh_qcnn_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "must_not_exist"
    rc = module.main(
        [
            "--dataset", "deepweeds",
            "--backbone", "plantnet",
            "--seeds", "2",
            "--qubits", "4",
            "--pooling", "skip",
            "--output-dir", str(output),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not output.exists()
    printed = capsys.readouterr().out
    assert '"dry_run": true' in printed
    assert '"pooling": "skip"' in printed
    assert "CPU feasibility" in printed
    assert "no quantum advantage asserted" in printed


def test_cli_rejects_non_power_of_two_qubits() -> None:
    script = ROOT / "scripts" / "run_fresh_qcnn.py"
    spec = importlib.util.spec_from_file_location("run_fresh_qcnn_cli_invalid", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit):
        module.main(["--dataset", "acre", "--qubits", "3", "--dry-run"])


def test_missing_cache_graceful_error(tmp_path: Path) -> None:
    output = tmp_path / "no_run"
    with pytest.raises(FileNotFoundError, match="frozen feature cache not found"):
        fq.run_qcnn_protocol(
            "deepweeds", "plantnet", [0], output, cache_dir=tmp_path / "empty_cache"
        )
    assert not output.exists()


# --------------------------------------------------------------------------- #
# 7. Claim boundary: no advantage, no hardcoded result numbers                #
# --------------------------------------------------------------------------- #
def test_claim_boundary_forbids_advantage_and_requires_power() -> None:
    low = fq.CLAIM_BOUNDARY.lower()
    assert "no quantum advantage is asserted" in low
    assert "power >= 0.8" in low
    assert "n >= 10" in low
    assert "cpu analytic simulation" in low
    assert "train only" in low
    assert "exactly once" in low
    assert "no quantum device" in low


def test_no_hardcoded_result_numbers_in_module_constants() -> None:
    for name, path in fq.DEFAULT_CLASSICAL_PATHS.items():
        assert name in {"deepweeds", "acre"}
        assert isinstance(path, str) and path.endswith("metrics.json")
    for name, path in fq.DEFAULT_QNN_PATHS.items():
        assert isinstance(name, str) and isinstance(path, str) and path.endswith("metrics.json")
    # Architecture/provenance constants describe structure only, never a score.
    assert "accuracy" not in fq.ARCHITECTURE
    assert "mean" not in fq.ARCHITECTURE
    assert all(isinstance(value, (str, dict, bool)) for value in fq.ARCHITECTURE.values())
