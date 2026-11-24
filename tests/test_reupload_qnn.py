"""Synthetic tests for the data-re-uploading QNN arm (REUPLOAD-D01).

No real datasets, no network and no committed result numbers are used.  Fake
feature matrices and fake baseline accuracies are generated in-test.  Anything
requiring torch / PennyLane / scipy / sklearn skips cleanly when absent.  These
tests check the circuit is *provably* a re-uploading circuit, that both gradient
paths move, that preprocessing/TEST discipline holds, and that the paired
statistical wiring never produces an advantage claim on its own.
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

from qmlpipe import reupload_qnn as ru  # noqa: E402  (import-safe without quantum stack)


# --------------------------------------------------------------------------- #
# 1. The circuit is provably a re-uploading circuit (no quantum stack needed)  #
# --------------------------------------------------------------------------- #
def test_circuit_spec_is_genuinely_reuploading() -> None:
    n, L = 6, 4
    spec = ru.reupload_circuit_spec(n, L)
    assert spec["feature_dim"] == n * L == 24
    assert spec["data_insertions"] == L
    assert spec["n_variational_layers"] == L
    assert spec["distinct_variational_param_sets"] == L
    assert spec["param_shapes"]["omega"] == (L, n, 2)
    assert spec["param_shapes"]["data_scales"] == (n * L,)
    assert spec["encoding_gate_count"] == n * L
    assert spec["entangling_gate_count"] == L * n
    assert spec["n_readouts"] == n
    assert spec["shots"] is None

    enc = [op for op in spec["operations"] if op["role"] == "encoding"]
    ent = [op for op in spec["operations"] if op["role"] == "entangle"]
    var = [op for op in spec["operations"] if op["role"] == "layer_rotation"]
    # Data appears once per wire per upload, with distinct feature indices.
    assert len(enc) == L * n
    assert sorted(op["feature_index"] for op in enc) == list(range(n * L))
    assert sorted({op["upload"] for op in enc}) == list(range(L))
    # Each upload has its OWN variational parameter set (indexed by upload).
    assert sorted({op["upload"] for op in var}) == list(range(L))
    assert all(op["param"] == "omega" for op in var)
    # Independent entangling ring per upload, including the wrap-around wire.
    assert len(ent) == L * n
    assert all(op["wires"][1] == (op["wires"][0] + 1) % n for op in ent)
    assert {op["name"] for op in ent} <= {"CNOT", "CZ"}
    # Block order is U(x) then V(omega_i), repeated L times.
    first_upload_order = [op["role"] for op in spec["operations"][: 2 * n + 2 * n]]
    assert first_upload_order == ["encoding"] * n + ["entangle"] * n + ["layer_rotation"] * (2 * n)

    cz = ru.reupload_circuit_spec(3, 2, entangler="cz")
    assert {op["name"] for op in cz["operations"] if op["role"] == "entangle"} == {"CZ"}
    with pytest.raises(ValueError):
        ru.reupload_circuit_spec(1, 2)  # no ring possible on one wire


# --------------------------------------------------------------------------- #
# 2. Both gradient paths are nonzero + forward shape                          #
# --------------------------------------------------------------------------- #
def _synthetic_qnn_inputs(n: int, uploads: int, rows: int = 6):
    rng = np.random.default_rng(12)
    X = np.tanh(rng.normal(size=(rows, n * uploads))).astype(np.float32)
    y = np.asarray([0, 1] * (rows // 2), dtype=np.int64)
    return X, y


def test_both_gradient_paths_nonzero() -> None:
    pytest.importorskip("pennylane", reason="PennyLane required for trainable QNN")
    pytest.importorskip("torch", reason="PyTorch required for trainable QNN")
    cfg = ru.ReuploadConfig(seed=0, qubits=3, uploads=2, epochs=1)
    model = ru.build_reupload_qnn(cfg)
    X, y = _synthetic_qnn_inputs(cfg.qubits, cfg.uploads)
    grads = ru.gradient_probe(model, X, y, cfg=cfg)
    assert grads["data_scales"] > 0.0, "data-encoding scale gradient must move"
    assert grads["omega"] > 0.0, "independent variational-layer gradient must move"
    assert grads["loss"] > 0.0


def test_forward_shape_and_introspection() -> None:
    torch = pytest.importorskip("torch", reason="PyTorch required")
    pytest.importorskip("pennylane", reason="PennyLane required for trainable QNN")
    cfg = ru.ReuploadConfig(seed=1, qubits=4, uploads=3, epochs=1)
    model = ru.build_reupload_qnn(cfg)
    X, _ = _synthetic_qnn_inputs(cfg.qubits, cfg.uploads, rows=5)
    out = model(torch.from_numpy(X))
    assert out.shape == (5, 2)
    info = ru.reupload_circuit_introspection(model)
    assert info["uploads"] == cfg.uploads
    assert info["distinct_variational_param_sets"] == cfg.uploads
    assert info["data_insertions"] == cfg.uploads
    assert info["entangling_gate_count"] == cfg.uploads * cfg.qubits
    assert info["n_readouts"] == cfg.qubits
    assert info["trainable_param_count"] > 0
    if info["tape_verified"]:
        assert info["tape_counts_match_spec"] is True


# --------------------------------------------------------------------------- #
# 3. TRAIN-only projection                                                     #
# --------------------------------------------------------------------------- #
def test_projection_fitted_on_train_only(monkeypatch) -> None:
    rng = np.random.default_rng(4)
    X_train = rng.normal(size=(14, 16)).astype(np.float32)
    X_val = rng.normal(size=(5, 16)).astype(np.float32) * 100.0 + 50.0  # extreme holdout
    X_test = rng.normal(size=(5, 16)).astype(np.float32) * -100.0 - 50.0

    captured: dict[str, np.ndarray] = {}
    real_fit = ru._fit_pca_matrix

    def _spy(features, target, seed):
        captured["data"] = np.array(features, copy=True)
        return real_fit(features, target, seed)

    monkeypatch.setattr(ru, "_fit_pca_matrix", _spy)

    cfg = ru.ReuploadConfig(seed=3, qubits=3, uploads=2)  # target dim = 6
    meta, Z_train = ru.fit_reupload_projection(X_train, cfg)

    # The only array handed to the PCA fit is TRAIN, already centred by the TRAIN mean.
    assert "data" in captured
    np.testing.assert_allclose(captured["data"], X_train - X_train.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(meta["mean"], X_train.mean(axis=0), atol=1e-5)
    assert meta["classical_preprocessing"]["fitted_on"] == "train_only"
    assert meta["classical_preprocessing"]["val_test_used_for_fit"] is False
    assert meta["target_dim"] == cfg.feature_dim == 6

    Z_val = ru.project_reupload_features(X_val, meta)
    Z_test = ru.project_reupload_features(X_test, meta)
    assert Z_train.shape == (14, 6)
    assert Z_val.shape == Z_test.shape == (5, 6)
    assert np.all(np.abs(Z_val) <= 1.0) and np.all(np.abs(Z_test) <= 1.0)
    with pytest.raises(ValueError, match="expected 16 cached features"):
        ru.project_reupload_features(X_val[:, :8], meta)


def test_train_reupload_does_not_fit_on_val_or_test(monkeypatch) -> None:
    """Spy at the call site: only the capped TRAIN matrix reaches the fit."""
    torch = pytest.importorskip("torch", reason="PyTorch required")
    pytest.importorskip("pennylane", reason="PennyLane required for trainable QNN")
    pytest.importorskip("scipy", reason="metrics require scipy")
    pytest.importorskip("sklearn", reason="metrics require sklearn")
    rng = np.random.default_rng(9)
    X_train = rng.normal(size=(10, 8)).astype(np.float32)
    y_train = np.asarray([0, 1] * 5)
    X_val = rng.normal(size=(4, 8)).astype(np.float32) * 50.0 + 25.0
    y_val = np.asarray([0, 1, 0, 1])
    X_test = rng.normal(size=(4, 8)).astype(np.float32) * -50.0 - 25.0
    y_test = np.asarray([0, 1, 0, 1])

    seen: list[np.ndarray] = []
    real_fit = ru.fit_reupload_projection

    def _spy(X, cfg):
        seen.append(np.array(X, copy=True))
        return real_fit(X, cfg)

    monkeypatch.setattr(ru, "fit_reupload_projection", _spy)
    cfg = ru.ReuploadConfig(seed=0, qubits=2, uploads=2, epochs=1, class_weight="balanced")
    model, history = ru.train_reupload_from_features(
        X_train, y_train, X_val, y_val, cfg, seed=0
    )
    assert len(seen) == 1
    np.testing.assert_allclose(seen[0], X_train, atol=1e-6)  # TRAIN, not VAL/TEST
    assert history["classical_preprocessing"]["val_test_used_for_fit"] is False
    result = ru.evaluate_once(model, X_test, y_test)
    assert result["test_evaluations"] == 1


# --------------------------------------------------------------------------- #
# 4. TEST scored exactly once                                                  #
# --------------------------------------------------------------------------- #
def test_test_scored_exactly_once() -> None:
    torch = pytest.importorskip("torch", reason="PyTorch required")
    pytest.importorskip("scipy", reason="metrics require scipy")
    pytest.importorskip("sklearn", reason="metrics require sklearn")

    class _FakeQNN(torch.nn.Module):
        def __init__(self, n: int) -> None:
            super().__init__()
            self.calls = 0
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.fixed = torch.tensor(
                np.random.default_rng(5).normal(size=(n, 2)), dtype=torch.float32
            )

        def forward(self, x):
            self.calls += 1
            return self.fixed

    X = np.zeros((6, 4), dtype=np.float32)
    y = np.asarray([0, 1, 0, 1, 0, 1])
    model = _FakeQNN(len(y))
    first = ru.evaluate_once(model, X, y)
    assert model.calls == 1
    assert first["test_evaluations"] == 1
    assert first["model"] == "reupload_qnn"
    assert first["n"] == 6
    with pytest.raises(RuntimeError, match="exactly once"):
        ru.evaluate_once(model, X, y)
    assert model.calls == 1  # no second forward pass happened


# --------------------------------------------------------------------------- #
# 5. Deterministic TRAIN cap                                                   #
# --------------------------------------------------------------------------- #
def test_cap_determinism() -> None:
    X = np.random.default_rng(2).normal(size=(40, 3))
    y = np.asarray([0] * 20 + [1] * 20)
    ax, ay, ameta = ru.cap_train(X, y, 6, seed=3)
    bx, by, _ = ru.cap_train(X, y, 6, seed=3)
    cx, cy, _ = ru.cap_train(X, y, 6, seed=4)
    assert ax.shape == (6, 3) and ay.shape == (6,)
    assert set(np.unique(ay)) == {0, 1}
    assert np.array_equal(ax, bx) and np.array_equal(ay, by)
    assert ameta["train_size_after"] == 6 and ameta["cap_applied"] is True
    assert not np.array_equal(ax, cx)  # different seed -> different subset
    untouched_x, untouched_y, no_meta = ru.cap_train(X, y, 0, seed=3)
    assert untouched_x is X and untouched_y is y
    assert no_meta["cap_applied"] is False


# --------------------------------------------------------------------------- #
# 6. Paired wiring with fake baselines (no advantage claim)                    #
# --------------------------------------------------------------------------- #
def test_paired_wiring_fake_baselines() -> None:
    pytest.importorskip("scipy", reason="paired_comparison requires scipy")
    reupload = {s: 0.60 + 0.01 * s for s in range(6)}
    baselines = {
        "classical_poly2": {s: 0.50 + 0.01 * s for s in range(6)},
        "fresh_qnn_amplitude": {s: 0.55 + 0.01 * s for s in range(6)},
    }
    comparisons = ru.paired_comparisons(reupload, baselines)
    assert len(comparisons) == 2
    assert {c["pair"][1] for c in comparisons} == set(baselines)
    for item in comparisons:
        assert item["pair"][0] == "reupload_qnn"
        assert item["metric"] == "accuracy"
        assert item["n"] == 6
        assert item["p_holm"] is not None  # holm_adjust applied to the whole family
        assert item["mean_difference"] > 0  # direction recorded, not claimed
        assert item["decision"] == "inconclusive"  # n=6 can never be significant
        assert "n<10" in item["decision_reason"]
        assert item["quantum_advantage_supported"] is False
        assert "no quantum advantage is asserted" in item["claim_note"].lower()
    assert ru.paired_comparisons(reupload, {"empty": {9: 0.5}}) == []


def test_load_baseline_values_from_fake_artifacts(tmp_path: Path) -> None:
    classical = tmp_path / "results" / "fresh_deepweeds" / "metrics.json"
    classical.parent.mkdir(parents=True, exist_ok=True)
    classical.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    qnn = tmp_path / "old_qnn.json"
    qnn.write_text(
        json.dumps(
            {
                "runs": [
                    {"backbone": "plantnet", "model": "qnn", "seed": 0,
                     "status": "ok", "metrics": {"accuracy": 0.51}},
                ]
            }
        ),
        encoding="utf-8",
    )
    poly = ru.load_baseline_values(
        "results/fresh_deepweeds/metrics.json", ("poly2", "poly"), "plantnet",
        results_root=tmp_path,
    )
    assert poly == {0: pytest.approx(0.63), 1: pytest.approx(0.64)}
    assert ru.load_baseline_values(qnn, ("qnn",), "plantnet", results_root=tmp_path) == {
        0: pytest.approx(0.51)
    }
    assert ru.load_baseline_values("missing.json", ("qnn",), results_root=tmp_path) == {}


def test_load_baseline_values_from_directory(tmp_path: Path) -> None:
    """A DIRECTORY passed as --qnn-results resolves to <dir>/metrics.json."""
    qnn_dir = tmp_path / "qnn_matched_deepweeds"
    qnn_dir.mkdir()
    (qnn_dir / "metrics.json").write_text(
        json.dumps(
            {
                "runs": [
                    {"backbone": "plantnet", "model": "qnn", "seed": 0,
                     "status": "ok", "metrics": {"accuracy": 0.51}},
                    {"backbone": "plantnet", "model": "qnn", "seed": 1,
                     "status": "ok", "metrics": {"accuracy": 0.52}},
                    {"backbone": "imagenet", "model": "qnn", "seed": 0,
                     "status": "ok", "metrics": {"accuracy": 0.49}},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert ru._resolve_artifact(qnn_dir) == qnn_dir / "metrics.json"
    assert ru._resolve_artifact(qnn_dir / "metrics.json") == qnn_dir / "metrics.json"

    values = ru.load_baseline_values(qnn_dir, ("qnn",), "plantnet", results_root=tmp_path)
    assert values == {0: pytest.approx(0.51), 1: pytest.approx(0.52)}  # per-seed, backbone-filtered
    report = ru.load_baseline_report(qnn_dir, ("qnn",), "plantnet", results_root=tmp_path)
    assert report["status"] == "ok"
    assert report["reason"] is None
    assert report["n_values"] == 2
    assert str(report["path"]) == str(qnn_dir / "metrics.json")

    # Directory without metrics.json resolves to a non-existent file -> "missing".
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    missing = ru.load_baseline_report(empty_dir, ("qnn",), results_root=tmp_path)
    assert missing["status"] == "missing" and missing["reason"] == "missing"
    assert str(missing["path"]) == str(empty_dir / "metrics.json")


def test_baseline_report_reasons(tmp_path: Path) -> None:
    """Unreadable JSON and no-matching-runs are distinct, explicit statuses."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    unreadable = ru.load_baseline_report(bad, ("qnn",), results_root=tmp_path)
    assert unreadable["status"] == "unreadable" and unreadable["reason"] == "unreadable"

    no_runs = tmp_path / "no_runs.json"
    no_runs.write_text(json.dumps({"runs": []}), encoding="utf-8")
    empty = ru.load_baseline_report(no_runs, ("qnn",), results_root=tmp_path)
    assert empty["status"] == "no_matching_runs" and empty["reason"] == "no matching runs"
    assert empty["values"] == {}


def _stage_fake_protocol(monkeypatch, tmp_path: Path) -> None:
    """Stub cache + train/eval so the protocol can run without torch/PennyLane."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = ru._cache_npz_path("deepweeds", "plantnet", cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"")

    rng = np.random.default_rng(1)

    def _cached(*args, **kwargs):
        X = rng.normal(size=(8, 6)).astype(np.float32)
        y = np.asarray([0, 1] * 4)
        return {
            "X": {"train": X, "val": X, "test": X},
            "y": {"train": y, "val": y, "test": y},
            "cache_path": str(cache_path),
            "cache_hit": True,
            "provenance": {"group_disjoint": True},
        }

    class _FakeModel:
        def __init__(self, seed: int) -> None:
            self.seed = seed

    def _train(Xtr, ytr, Xv, yv, cfg, seed):
        history = {
            "classical_preprocessing": {"val_test_used_for_fit": False},
            "grad_norms_first_batch": {"data_scales": 1.0, "omega": 1.0, "head": 0.0},
            "param_delta": 0.1,
            "circuit": {},
        }
        return _FakeModel(int(seed)), history

    def _evaluate(model, X, y):
        return {
            "model": "reupload_qnn",
            "accuracy": 0.50 + 0.01 * model.seed,
            "auc": 0.55, "f1": 0.5, "precision": 0.5, "recall": 0.5,
            "confusion_matrix": [[2, 1], [1, 2]], "n": 6,
            "iou_class0": 0.5, "iou_class1": 0.5, "macro_iou": 0.5,
            "test_evaluations": 1,
        }

    monkeypatch.setattr(ru._fp, "cache_features", _cached)
    monkeypatch.setattr(ru, "train_reupload_from_features", _train)
    monkeypatch.setattr(ru, "evaluate_once", _evaluate)


def _write_fake_classical(tmp_path: Path) -> None:
    artifact = tmp_path / "results" / "fresh_deepweeds" / "metrics.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "runs": [
                    {"backbone": "plantnet", "model": "poly2", "seed": 0,
                     "status": "ok", "metrics": {"accuracy": 0.61}},
                    {"backbone": "plantnet", "model": "poly2", "seed": 1,
                     "status": "ok", "metrics": {"accuracy": 0.62}},
                ]
            }
        ),
        encoding="utf-8",
    )


def test_missing_qnn_baseline_warns_and_has_no_comparison(tmp_path: Path, monkeypatch) -> None:
    """A configured-but-missing old-QNN baseline is warned, never silently dropped."""
    pytest.importorskip("scipy", reason="paired_comparison requires scipy")
    _stage_fake_protocol(monkeypatch, tmp_path)
    _write_fake_classical(tmp_path)
    missing_dir = tmp_path / "qnn_matched_deepweeds"  # exists, but no metrics.json
    missing_dir.mkdir()

    result = ru.run_reupload_protocol(
        "deepweeds", "plantnet", [0, 1], tmp_path / "out",
        cache_dir=tmp_path / "cache", train_cap=0, epochs=1, qubits=2, uploads=2,
        qnn_results=missing_dir, results_root=tmp_path,
    )
    baseline_warnings = [w for w in result["warnings"] if w.get("stage") == "baseline_load"]
    assert len(baseline_warnings) == 1
    warning = baseline_warnings[0]
    assert warning["baseline"] == "fresh_qnn_amplitude"
    assert warning["reason"] == "missing"
    assert str(warning["path"]) == str(missing_dir / "metrics.json")
    assert str(missing_dir) in warning["requested"]
    assert "no comparison fabricated" in warning["error"]

    comparisons = result["metrics"]["comparisons"]
    assert [c["pair"] for c in comparisons] == [["reupload_qnn", "classical_poly2"]]
    assert all("fresh_qnn_amplitude" not in c["pair"] for c in comparisons)
    assert result["metrics"]["baselines_loaded"] == ["classical_poly2"]
    notes = result["provenance"]["comparison_notes"]
    assert notes["baselines_loaded"] == ["classical_poly2"]
    assert [s["name"] for s in notes["baseline_sources"] if not s["loaded"]] == [
        "fresh_qnn_amplitude"
    ]
    assert result["metrics"]["accounting_invariant"] is True
    assert result["metrics"]["n_runs_skipped"] == 0  # baseline warning is not a run skip


def test_unreadable_baseline_warns_and_is_excluded(tmp_path: Path, monkeypatch) -> None:
    """An unreadable, configured baseline produces a warning and no fabricated data."""
    pytest.importorskip("scipy", reason="paired_comparison requires scipy")
    _stage_fake_protocol(monkeypatch, tmp_path)
    _write_fake_classical(tmp_path)
    broken = tmp_path / "broken_qnn"
    broken.mkdir()
    (broken / "metrics.json").write_text("{not-json", encoding="utf-8")

    result = ru.run_reupload_protocol(
        "deepweeds", "plantnet", [0, 1], tmp_path / "out2",
        cache_dir=tmp_path / "cache", train_cap=0, epochs=1, qubits=2, uploads=2,
        qnn_results=broken, results_root=tmp_path,
    )
    warning = next(w for w in result["warnings"] if w.get("stage") == "baseline_load")
    assert warning["reason"] == "unreadable"
    assert warning["status"] == "unreadable"
    assert result["metrics"]["baselines_loaded"] == ["classical_poly2"]
    assert all("fresh_qnn_amplitude" not in c["pair"] for c in result["metrics"]["comparisons"])


def test_baselines_loaded_reflects_actual_loads(tmp_path: Path, monkeypatch) -> None:
    """When both baselines load, baselines_loaded names exactly those two."""
    pytest.importorskip("scipy", reason="paired_comparison requires scipy")
    _stage_fake_protocol(monkeypatch, tmp_path)
    _write_fake_classical(tmp_path)
    qnn_dir = tmp_path / "qnn_matched_deepweeds"
    qnn_dir.mkdir()
    (qnn_dir / "metrics.json").write_text(
        json.dumps(
            {
                "runs": [
                    {"backbone": "plantnet", "model": "qnn", "seed": 0,
                     "status": "ok", "metrics": {"accuracy": 0.50}},
                    {"backbone": "plantnet", "model": "qnn", "seed": 1,
                     "status": "ok", "metrics": {"accuracy": 0.51}},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = ru.run_reupload_protocol(
        "deepweeds", "plantnet", [0, 1], tmp_path / "out3",
        cache_dir=tmp_path / "cache", train_cap=0, epochs=1, qubits=2, uploads=2,
        qnn_results=qnn_dir, results_root=tmp_path,
    )
    expected = ["classical_poly2", "fresh_qnn_amplitude"]
    assert result["metrics"]["baselines_loaded"] == expected
    assert result["provenance"]["comparison_notes"]["baselines_loaded"] == expected
    assert sorted(c["pair"][1] for c in result["metrics"]["comparisons"]) == expected
    assert not [w for w in result["warnings"] if w.get("stage") == "baseline_load"]
    assert all(s["loaded"] for s in result["provenance"]["comparison_notes"]["baseline_sources"])


# --------------------------------------------------------------------------- #
# 7. CLI dry-run is write-free + defaults                                      #
# --------------------------------------------------------------------------- #
def test_cli_dry_run_is_write_free(tmp_path: Path, capsys) -> None:
    script = ROOT / "scripts" / "run_reupload_qnn.py"
    spec = importlib.util.spec_from_file_location("run_reupload_qnn_cli", script)
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
    assert '"uploads": 4' in printed
    assert '"shots": null' in printed
    assert "--qnn-results" in printed and "DIRECTORY" in printed  # accepted path forms documented


def test_missing_cache_graceful_error(tmp_path: Path) -> None:
    output = tmp_path / "no_run"
    with pytest.raises(FileNotFoundError, match="frozen feature cache not found"):
        ru.run_reupload_protocol(
            "deepweeds", "plantnet", [0], output,
            cache_dir=tmp_path / "empty_cache",
        )
    assert not output.exists()


# --------------------------------------------------------------------------- #
# 8. Claim boundary + no hardcoded project result numbers                      #
# --------------------------------------------------------------------------- #
def test_claim_boundary_forbids_advantage() -> None:
    low = ru.CLAIM_BOUNDARY.lower()
    assert "no quantum advantage is asserted" in low
    assert "power >= 0.8" in low
    assert "holm" in low
    assert "shots none" in low or "shots=none" in low
    assert "train-only" in low
    assert "test is scored exactly once" in low


def test_no_hardcoded_project_result_numbers() -> None:
    source = (ROOT / "qmlpipe" / "reupload_qnn.py").read_text(encoding="utf-8")
    for committed in ("0.918", "0.922", "0.5045", "0.7665"):
        assert committed not in source, f"hardcoded result number {committed} in module"
    assert "results/fresh_deepweeds/metrics.json" in source  # path only, not a value
    assert "results/fresh_acre_capped/metrics.json" in source
