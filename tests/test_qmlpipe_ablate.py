"""Synthetic-only tests for the QML ablation runner and model card.

No dataset, checkpoint, GPU, PennyLane install or network access is required.
Quantum-dependent integration coverage is guarded by ``pytest.importorskip`` so
the module skips cleanly under Mother's Python 3.14 verifier without PennyLane.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pytest

from experiments.metrics import bootstrap_ci, mean_std, summarize_runs
from qmlpipe.ablate import (
    ANSATZ_CHOICES,
    EMBEDDING_CHOICES,
    LAYER_CHOICES,
    MAX_GRID_SIZE,
    NOISE_CHOICES,
    QUBIT_CHOICES,
    ablation_grid,
    run_ablation,
    summarize_ablation,
)
from qmlpipe.model_card import build_model_card


# --------------------------------------------------------------------------
# Synthetic config with the same contract surface as QMLConfig (D01).
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SyntheticQMLConfig:
    seed: int = 42
    seed_offset: int = 7
    n_qubits: int = 10
    qnn_layers: int = 2
    embedding: str = "amplitude"
    ansatz: str = "strongly_entangling"
    noise_depolarizing: float | None = None
    feature_dim: int = 1024
    device: str = "default.qubit"
    shots: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> list[str]:
        if self.feature_dim != 2**self.n_qubits:
            return ["feature_dim must equal 2**n_qubits"]
        return []


# --------------------------------------------------------------------------
# Deterministic synthetic train/eval functions (no RNG, no quantum library).
# --------------------------------------------------------------------------
def _train(cfg, seed):
    return {"seed": seed, "config": cfg.as_dict()}


def _evaluate(model, cfg):
    return {
        "accuracy": (model["seed"] % 97) / 100.0,
        "n_qubits": float(cfg.n_qubits),
    }


# --------------------------------------------------------------------------
# Grid construction
# --------------------------------------------------------------------------
def test_grid_is_bounded_cartesian_and_valid():
    grid = ablation_grid(SyntheticQMLConfig())

    assert len(grid) == MAX_GRID_SIZE == 72
    assert {c.n_qubits for c in grid} == set(QUBIT_CHOICES)
    assert {c.qnn_layers for c in grid} == set(LAYER_CHOICES)
    assert {c.embedding for c in grid} == set(EMBEDDING_CHOICES)
    assert {c.ansatz for c in grid} == set(ANSATZ_CHOICES)
    assert {c.noise_depolarizing for c in grid} == set(NOISE_CHOICES)

    assert all(isinstance(c, SyntheticQMLConfig) for c in grid)
    assert all(c.validate() == [] for c in grid)
    assert all(c.feature_dim == 2**c.n_qubits for c in grid)

    first, last = grid[0], grid[-1]
    assert (
        first.n_qubits,
        first.qnn_layers,
        first.embedding,
        first.ansatz,
        first.noise_depolarizing,
    ) == (4, 1, "amplitude", "strongly_entangling", None)
    assert (
        last.n_qubits,
        last.qnn_layers,
        last.embedding,
        last.ansatz,
        last.noise_depolarizing,
    ) == (10, 3, "angle", "basic_entangler", 0.01)


def test_grid_clones_do_not_mutate_base_or_share_objects():
    base = SyntheticQMLConfig()
    before = base.as_dict()
    grid = ablation_grid(base)

    assert base.as_dict() == before
    assert all(c is not base for c in grid)
    assert len({id(c) for c in grid}) == len(grid)


def test_grid_rejects_config_with_missing_axis_field():
    @dataclass(frozen=True)
    class IncompleteConfig:
        seed: int = 42

    with pytest.raises(ValueError, match="ablation axis"):
        ablation_grid(IncompleteConfig())


# --------------------------------------------------------------------------
# Deterministic execution / honest failure recording
# --------------------------------------------------------------------------
def test_run_ablation_is_deterministic_and_respects_max_runs():
    base = SyntheticQMLConfig()
    grid = ablation_grid(base)[:3]

    first = run_ablation(base, _train, _evaluate, grid=grid, max_runs=2)
    second = run_ablation(base, _train, _evaluate, grid=grid, max_runs=2)

    assert first == second
    assert len(first) == 2
    assert [r["seed"] for r in first] == [42, 49]  # base + index * seed_offset
    assert all(r["status"] == "ok" for r in first)
    assert all(r["error"] is None for r in first)
    assert all(r["config"] == grid[i].as_dict() for i, r in enumerate(first))
    assert first[0]["config"] is not first[1]["config"]
    assert first[0]["metrics"]["accuracy"] == pytest.approx(0.42)
    assert first[0]["metrics"]["n_qubits"] == pytest.approx(4.0)


def test_run_ablation_records_failure_without_fabricating_metrics():
    def failing_train(cfg, seed):
        if cfg.n_qubits == 8:
            raise RuntimeError("synthetic backend failure")
        return _train(cfg, seed)

    base = SyntheticQMLConfig()
    grid = ablation_grid(base)
    # grid[0] is a 4-qubit arm, grid[24] the first 8-qubit arm, grid[-1] a 10-qubit arm.
    selected = [grid[0], grid[24], grid[-1]]
    assert [c.n_qubits for c in selected] == [4, 8, 10]
    results = run_ablation(base, failing_train, _evaluate, grid=selected)

    failed = [r for r in results if r["status"] == "error"]
    assert len(failed) == 1
    assert failed[0]["metrics"] == {}
    assert failed[0]["error"]["type"] == "RuntimeError"
    assert "synthetic backend failure" in failed[0]["error"]["message"]

    summary = summarize_ablation(results)
    assert summary["n_configs"] == 3
    assert summary["n_success"] == 2
    assert summary["n_failed"] == 1
    assert summary["failures"][0]["error"]["type"] == "RuntimeError"
    assert set(summary["metrics"]) == {"accuracy", "n_qubits"}


def test_run_ablation_rejects_non_mapping_eval_output():
    results = run_ablation(
        SyntheticQMLConfig(),
        _train,
        lambda model, cfg: 0.5,
        grid=ablation_grid(SyntheticQMLConfig())[:1],
    )
    assert results[0]["status"] == "error"
    assert results[0]["metrics"] == {}
    assert results[0]["error"]["type"] == "TypeError"


def test_max_runs_zero_returns_empty_and_negative_is_rejected():
    base = SyntheticQMLConfig()
    assert run_ablation(base, _train, _evaluate, max_runs=0) == []
    with pytest.raises(ValueError, match="max_runs"):
        run_ablation(base, _train, _evaluate, max_runs=-1)


# --------------------------------------------------------------------------
# Summary math
# --------------------------------------------------------------------------
def _result(index, seed, accuracy):
    return {
        "run_index": index,
        "seed": seed,
        "config": SyntheticQMLConfig().as_dict(),
        "status": "ok",
        "metrics": {"accuracy": accuracy},
        "error": None,
    }


def test_summarize_ablation_matches_metrics_helpers():
    values = [0.8, 0.9, 1.0]
    results = [_result(i, 42 + i * 7, v) for i, v in enumerate(values)]
    summary = summarize_ablation(results)

    block = summary["metrics"]["accuracy"]
    expected_mean, expected_std = mean_std(values)
    expected_lo, expected_hi = bootstrap_ci(values, seed=42)
    reference = summarize_runs("accuracy", values, seed=42)

    assert block["n"] == 3
    assert reference["n"] == 3
    assert block["mean"] == pytest.approx(expected_mean)
    assert block["mean"] == pytest.approx(reference["mean"])
    assert block["std"] == pytest.approx(expected_std)
    assert block["std"] == pytest.approx(reference["std"])
    assert block["ci95_low"] == pytest.approx(expected_lo)
    assert block["ci95_high"] == pytest.approx(expected_hi)
    assert block["values"] == values
    assert expected_lo <= expected_mean <= expected_hi
    assert summary["n_success"] == 3
    assert summary["n_failed"] == 0
    assert summary["failures"] == []
    assert summary["seed"] == 42


def test_summarize_ablation_handles_empty_and_missing_metric_values():
    empty = summarize_ablation([])
    assert empty["n_configs"] == 0
    assert empty["metrics"] == {}

    results = [_result(0, 42, 0.7)]
    results[0]["metrics"]["accuracy"] = None
    results[0]["metrics"]["auc"] = float("nan")
    summary = summarize_ablation(results)
    assert summary["metrics"] == {}  # no finite metric values => no invented stats

    skipped = summarize_ablation(results + [_result(1, 49, 0.9)])
    assert skipped["metrics"]["accuracy"]["n"] == 1


def test_summarize_ablation_does_not_use_bool_as_metric():
    results = [_result(0, 42, True), _result(1, 49, False)]
    assert summarize_ablation(results)["metrics"] == {}


# --------------------------------------------------------------------------
# Model card honesty
# --------------------------------------------------------------------------
def test_model_card_renders_provided_numbers_and_sections():
    values = [0.75, 0.80, 0.85]
    summary = summarize_ablation([_result(i, 42 + i * 7, v) for i, v in enumerate(values)])
    card = build_model_card(summary, SyntheticQMLConfig())

    for section in (
        "## Intended use",
        "## Training data provenance",
        "## Architecture",
        "## Metrics",
        "## Limitations",
        "## Ethical and deployment notes",
    ):
        assert section in card

    assert "| Qubits | 10 |" in card
    assert "| Layers | 2 |" in card
    assert "| Embedding | amplitude |" in card
    assert "| Ansatz | strongly_entangling |" in card
    assert "| Device | default.qubit |" in card
    assert "`accuracy`" in card
    assert "0.8" in card  # mean of 0.75/0.80/0.85, rendered from results only
    assert "n = 3" not in card  # n lives in the table cell, not free prose

    # Legacy paper values must never leak into a generated card.
    for forbidden in ("0.932", "0.9239", "0.0013", "0.0009"):
        assert forbidden not in card
    assert "no quantum-advantage claim" in card.lower()
    assert "not measured" in card  # provenance fields absent from synthetic cfg
    assert "QML_IMPROVEMENT_PLAN" in card
    assert "## Improvement plan" in card


def test_model_card_renders_not_measured_for_missing_metrics():
    card = build_model_card(
        {
            "n_success": 0,
            "n_failed": 1,
            "metrics": {
                "accuracy": {
                    "n": None,
                    "mean": None,
                    "std": None,
                    "ci95_low": None,
                    "ci95_high": None,
                }
            },
        },
        SyntheticQMLConfig(),
    )
    assert "not measured" in card
    assert not re.search(r"\b(?:nan|inf|infinity)\b", card, flags=re.IGNORECASE)


def test_model_card_accepts_flat_metrics_without_inventing_spread():
    card = build_model_card({"accuracy": 0.75}, {"n_qubits": 4, "layers": 1})
    assert "0.75" in card
    assert "| `accuracy` | not measured | 0.75 | not measured | not measured |" in card
    assert "| Qubits | 4 |" in card


def test_model_card_rejects_non_mapping_results():
    with pytest.raises(TypeError, match="mapping"):
        build_model_card([("accuracy", 0.5)], SyntheticQMLConfig())


# --------------------------------------------------------------------------
# Real-config integration (stdlib-only config: runs without PennyLane).
# --------------------------------------------------------------------------
def test_real_qmlconfig_grid_validates_without_pennylane():
    try:
        from qmlpipe.config import QMLConfig  # type: ignore
    except Exception as exc:  # pragma: no cover - sibling worker contract
        pytest.skip(f"QMLConfig unavailable in this environment: {exc}")

    base = QMLConfig()
    grid = ablation_grid(base)

    assert len(grid) == MAX_GRID_SIZE
    assert all(isinstance(c, QMLConfig) for c in grid)
    assert all(c.validate() == [] for c in grid)
    assert all(c.projection_dim == 2**c.qubits for c in grid)
    assert {c.qubits for c in grid} == set(QUBIT_CHOICES)
    assert {c.layers for c in grid} == set(LAYER_CHOICES)
    assert {c.embedding for c in grid} == set(EMBEDDING_CHOICES)
    assert {c.ansatz for c in grid} == set(ANSATZ_CHOICES)
    assert {c.noise_depolarizing for c in grid} == set(NOISE_CHOICES)
    assert base.projection_dim == 1024  # base config was not mutated


# --------------------------------------------------------------------------
# Quantum-dependent integration: skipped cleanly without PennyLane.
# --------------------------------------------------------------------------
def test_real_qmlconfig_integration_skips_without_pennylane():
    pytest.importorskip("pennylane")
    try:
        from qmlpipe.config import QMLConfig  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on sibling worker
        pytest.skip(f"QMLConfig unavailable in this environment: {exc}")

    try:
        cfg = QMLConfig()
        grid = ablation_grid(cfg)
    except (TypeError, ValueError) as exc:  # pragma: no cover - config contract drift
        pytest.skip(f"QMLConfig does not expose the ablation contract: {exc}")

    assert len(grid) == MAX_GRID_SIZE
    assert all(isinstance(c, QMLConfig) for c in grid)
