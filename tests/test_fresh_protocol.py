"""Synthetic tests for the fresh classical protocol (FRESH-D02).

No network, no real datasets and no legacy result numbers are used.  Heavy
dependencies (torch/torchvision/scipy/sklearn/statsmodels) are either faked
through ``sys.modules`` or skipped cleanly when absent.
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import fresh_protocol as fp  # noqa: E402


# --------------------------------------------------------------------------- #
# aggregate                                                                    #
# --------------------------------------------------------------------------- #
def test_aggregate_matches_hand_computed_mean_sd_and_is_seed_deterministic() -> None:
    values = [0.5, 0.7, 0.9, 1.1, 1.3]
    expected_mean = 0.9
    expected_sd = float(np.sqrt(0.1))  # sum((x-0.9)^2)=0.4; /(5-1)=0.1

    first = fp.aggregate(values, seed=7, n_boot=100)
    second = fp.aggregate(values, seed=7, n_boot=100)

    assert first["n"] == 5
    assert first["values"] == values
    assert first["mean"] == pytest.approx(expected_mean)
    assert first["std"] == pytest.approx(expected_sd)
    assert first["std"] != pytest.approx(float(np.std(values, ddof=0)))
    assert first["boot_B"] == 100 and first["boot_seed"] == 7
    assert first["ci95_low"] <= first["mean"] <= first["ci95_high"]
    assert first == second
    assert fp.aggregate(values, seed=8, n_boot=100)["ci95_low"] != first["ci95_low"]


# --------------------------------------------------------------------------- #
# paired statistics                                                            #
# --------------------------------------------------------------------------- #
def test_paired_comparison_is_inconclusive_for_small_n() -> None:
    pytest.importorskip("scipy")
    rng = np.random.default_rng(3)
    baseline = rng.uniform(0.5, 0.6, size=6)
    improved = baseline + 0.2 + rng.normal(0, 0.01, size=6)
    result = fp.paired_comparison(improved, baseline)
    assert result["n"] == 6
    assert result["df"] == 5
    assert result["decision"] == "inconclusive"
    assert "n<10" in result["decision_reason"]
    assert result["p_holm"] is None  # assigned by holm_adjust
    assert result["t_statistic"] is not None
    assert result["cohens_dz"] > 0
    assert result["secondary_tests"]["wilcoxon_signed_rank"]["p"] is not None


def test_paired_comparison_is_inconclusive_for_low_power() -> None:
    pytest.importorskip("scipy")
    rng = np.random.default_rng(11)
    baseline = rng.normal(0.5, 0.01, size=12)
    nearly_identical = baseline + 1e-6 * rng.normal(size=12)
    result = fp.paired_comparison(nearly_identical, baseline, alpha=0.05)
    assert result["n"] == 12
    power = result["power_at_observed_dz"]
    assert power is None or power < 0.8
    assert result["decision"] == "inconclusive"
    assert "power" in result["decision_reason"]


def test_paired_comparison_makes_no_claim_for_n_below_two() -> None:
    result = fp.paired_comparison([0.9], [0.4])
    assert result["decision"] == "inconclusive"
    assert "n<2" in result["decision_reason"]
    assert result["t_statistic"] is None


def test_holm_ordering_and_decision_rule() -> None:
    def item(p, n=12, power=0.95, low=0.01, high=0.08):
        return {
            "p_raw": p,
            "n": n,
            "power_at_observed_dz": power,
            "ci95_low": low,
            "ci95_high": high,
        }

    family = [item(0.01), item(0.04), item(0.03), item(0.20)]
    adjusted = fp.holm_adjust(family, alpha=0.05)
    assert adjusted is family
    # Holm step-down: 4*0.01=0.04; 3*0.03=0.09; 2*0.04 -> capped at running 0.09;
    # 1*0.20=0.20.
    assert [entry["p_holm"] for entry in adjusted] == pytest.approx(
        [0.04, 0.09, 0.09, 0.20]
    )
    assert all(entry["p_holm"] >= entry["p_raw"] for entry in adjusted)
    assert [entry["decision"] for entry in adjusted] == [
        "significant",
        "inconclusive",
        "inconclusive",
        "inconclusive",
    ]
    assert adjusted[0]["multiplicity"] == {"method": "holm", "m": 4, "alpha": 0.05}

    # Small-n and low-power comparisons stay inconclusive even with tiny raw p.
    guarded = fp.holm_adjust(
        [item(0.001, n=5), item(0.001, power=0.4), item(None)], alpha=0.05
    )
    assert [entry["decision"] for entry in guarded] == [
        "inconclusive",
        "inconclusive",
        "inconclusive",
    ]
    assert "n<10" in guarded[0]["decision_reason"]
    assert "power" in guarded[1]["decision_reason"]
    assert guarded[2]["p_holm"] is None


# --------------------------------------------------------------------------- #
# Classical arm                                                                #
# --------------------------------------------------------------------------- #
def test_run_svm_arm_fits_train_scores_test_once_and_is_deterministic(monkeypatch) -> None:
    sklearn_svm = pytest.importorskip("sklearn.svm")
    pytest.importorskip("scipy")

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(40, 4))
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.normal(size=(12, 4))
    y_test = (X_test[:, 0] > 0).astype(int)

    fit_calls: list[tuple[np.ndarray, np.ndarray]] = []
    predict_calls: list[np.ndarray] = []
    original_fit = sklearn_svm.SVC.fit
    original_predict = sklearn_svm.SVC.predict

    def recording_fit(self, X, y, *args, **kwargs):
        fit_calls.append((np.asarray(X).copy(), np.asarray(y).copy()))
        return original_fit(self, X, y, *args, **kwargs)

    def recording_predict(self, X, *args, **kwargs):
        predict_calls.append(np.asarray(X).copy())
        return original_predict(self, X, *args, **kwargs)

    monkeypatch.setattr(sklearn_svm.SVC, "fit", recording_fit)
    monkeypatch.setattr(sklearn_svm.SVC, "predict", recording_predict)

    result = fp.run_svm_arm(X_train, y_train, X_test, y_test, "poly2", seed=5)
    expected_keys = {
        "accuracy",
        "auc",
        "f1",
        "precision",
        "recall",
        "confusion_matrix",
        "n",
    }
    assert expected_keys <= set(result)
    assert result["model"] == "poly"
    assert result["n"] == len(y_test)
    assert result["n_train"] == len(y_train)
    assert result["confusion_matrix"] == [[0, 0], [0, 0]] or sum(
        map(sum, result["confusion_matrix"])
    ) == len(y_test)
    assert len(fit_calls) == 1
    assert fit_calls[0][0].shape == X_train.shape
    np.testing.assert_array_equal(fit_calls[0][1], y_train)
    assert len(predict_calls) == 1
    assert predict_calls[0].shape == X_test.shape

    repeated = fp.run_svm_arm(X_train, y_train, X_test, y_test, "poly2", seed=5)
    assert result == repeated
    with pytest.raises(ValueError):
        fp.run_svm_arm(X_train, y_train, X_test, y_test, "quantum", seed=5)


# --------------------------------------------------------------------------- #
# Feature cache (fully faked heavy dependencies)                               #
# --------------------------------------------------------------------------- #
def _install_fake_cache_modules(monkeypatch, split_fn):
    @dataclass(frozen=True)
    class FreshSample:
        sample_id: str
        source_image_id: str
        label: int
        path: str
        bbox: tuple[int, int, int, int] | None = None

    samples = []
    for group_index in range(10):
        for crop_index in range(2):
            samples.append(
                FreshSample(
                    sample_id=f"img{group_index:02d}_crop{crop_index}",
                    source_image_id=f"img{group_index:02d}",
                    label=(group_index + crop_index) % 2,
                    path=f"fake://img{group_index:02d}_{crop_index}.png",
                )
            )

    def prepare_dataset(name, root=None):
        assert name in {"maizeset", "deepweeds", "acre"}
        return list(samples)

    def default_grouped_split(data, seed=42, ratios=(0.7, 0.2, 0.1)):
        groups = sorted({sample.source_image_id for sample in data})
        assignment = {}
        for index, group in enumerate(groups):
            if index < 7:
                assignment[group] = "train"
            elif index < 9:
                assignment[group] = "val"
            else:
                assignment[group] = "test"
        return {
            name: [sample for sample in data if assignment[sample.source_image_id] == name]
            for name in ("train", "val", "test")
        }

    def split_provenance(name, data, splits, seed, ratios):
        return {
            "dataset": name,
            "seed": seed,
            "ratios": ratios,
            "splits": {
                split_name: {
                    "groups": sorted({s.source_image_id for s in split_samples})
                }
                for split_name, split_samples in splits.items()
            },
        }

    fresh = types.ModuleType("data.fresh_splits")
    fresh.FreshSample = FreshSample
    fresh.prepare_dataset = prepare_dataset
    fresh.grouped_split = split_fn or default_grouped_split
    fresh.split_provenance = split_provenance

    def make_feature_loader(extractor, preprocess, device):
        def loader(data, cfg, split):
            vectors = []
            for sample in data:
                digest = hashlib.sha256(sample.sample_id.encode("utf-8")).digest()[:8]
                vectors.append(np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0)
            return np.stack(vectors)

        return loader

    benchmark = types.ModuleType("scripts.run_real_benchmark")
    benchmark.make_feature_loader = make_feature_loader

    backbone_module = types.ModuleType("qmlpipe.plant_backbone")
    backbone_module.build_backbone_extractor = lambda name, cfg: (object(), object())

    config_module = types.ModuleType("qmlpipe.config")
    config_module.QMLConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)

    for name, module in {
        "data": types.ModuleType("data"),
        "scripts": types.ModuleType("scripts"),
        "scripts.run_real_benchmark": benchmark,
        "qmlpipe": types.ModuleType("qmlpipe"),
        "qmlpipe.plant_backbone": backbone_module,
        "qmlpipe.config": config_module,
    }.items():
        module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "data.fresh_splits", fresh)
    # ``from data import fresh_splits`` reads the parent attribute first, so
    # bind each fake submodule on its fake parent as CPython's import would.
    sys.modules["data"].fresh_splits = fresh
    sys.modules["scripts"].run_real_benchmark = benchmark
    sys.modules["qmlpipe"].plant_backbone = backbone_module
    sys.modules["qmlpipe"].config = config_module
    return fresh


def test_cache_features_is_reused_and_rejects_group_leakage(monkeypatch, tmp_path) -> None:
    _install_fake_cache_modules(monkeypatch, None)
    cache_dir = tmp_path / "cache"

    first = fp.cache_features(
        "maizeset", "plantnet", root=str(tmp_path), cache_dir=str(cache_dir), seed=42
    )
    assert first["cache_hit"] is False
    assert Path(first["cache_path"]).exists()
    assert set(first["X"]) == {"train", "val", "test"}
    assert first["provenance"]["group_disjoint"] is True
    assert all(len(first["y"][name]) == len(first["X"][name]) for name in first["X"])
    train_groups, test_groups = set(first["groups"]["train"]), set(first["groups"]["test"])
    assert train_groups.isdisjoint(test_groups)

    second = fp.cache_features(
        "maizeset", "plantnet", root=str(tmp_path), cache_dir=str(cache_dir), seed=7
    )
    assert second["cache_hit"] is True
    key = first["sample_ids"]["test"][0]
    first_row = first["X"]["test"][first["sample_ids"]["test"].index(key)]
    second_row = second["X"]["test"][second["sample_ids"]["test"].index(key)]
    np.testing.assert_array_equal(first_row, second_row)

    def leaking_split(data, seed=42, ratios=(0.7, 0.2, 0.1)):
        train = [s for s in data if s.source_image_id in {"img00", "img01"}]
        test = [s for s in data if s.source_image_id == "img00"]
        return {"train": train, "val": [], "test": test}

    monkeypatch.setattr(sys.modules["data.fresh_splits"], "grouped_split", leaking_split)
    with pytest.raises(ValueError, match="group"):
        fp.cache_features(
            "maizeset",
            "plantnet",
            root=str(tmp_path),
            cache_dir=str(tmp_path / "other_cache"),
            seed=42,
        )


def test_cache_features_batches_loader_calls_and_preserves_order(monkeypatch, tmp_path) -> None:
    calls: list[int] = []

    def vector_for(sample):
        digest = hashlib.sha256(sample.sample_id.encode("utf-8")).digest()[:3]
        return np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0

    def recording_loader_factory(extractor, preprocess, device):
        def loader(data, cfg, split):
            calls.append(len(data))
            return np.stack([vector_for(sample) for sample in data])

        return loader

    fresh = _install_fake_cache_modules(monkeypatch, None)
    monkeypatch.setattr(
        sys.modules["scripts.run_real_benchmark"], "make_feature_loader", recording_loader_factory
    )
    root, cache_dir = str(tmp_path), str(tmp_path / "cache")
    result = fp.cache_features(
        "deepweeds", "imagenet", root=root, cache_dir=cache_dir, seed=42, feature_batch_size=3
    )

    assert calls == [3, 3, 3, 3, 3, 3, 2]  # 20 synthetic samples, batch size 3
    assert all(size <= 3 for size in calls)
    assert sum(calls) == 20
    assert result["X"]["train"].shape == (14, 3)
    assert result["X"]["val"].shape == (4, 3)
    assert result["X"]["test"].shape == (2, 3)
    assert result["provenance"]["cache"]["feature_batch_size"] == 3

    # Original per-sample order and values are unchanged by chunking.
    samples = fresh.prepare_dataset("deepweeds", root=root)
    splits = fresh.grouped_split(samples, seed=42, ratios=fp.DEFAULT_RATIOS)
    for name in ("train", "val", "test"):
        expected = np.stack([vector_for(sample) for sample in splits[name]])
        np.testing.assert_array_equal(result["X"][name], expected)

    calls.clear()
    reuse = fp.cache_features(
        "deepweeds", "imagenet", root=root, cache_dir=cache_dir, seed=7, feature_batch_size=3
    )
    assert reuse["cache_hit"] is True
    assert calls == []  # cache reuse never invokes the loader

    with pytest.raises(ValueError, match="feature_batch_size"):
        fp.cache_features(
            "deepweeds", "imagenet", root=root, cache_dir=str(tmp_path / "bad_cache"),
            seed=42, feature_batch_size=0,
        )

    # Group disjointness is still checked before any loader call.
    calls.clear()

    def leaking_split(data, seed=42, ratios=(0.7, 0.2, 0.1)):
        return {
            "train": [s for s in data if s.source_image_id == "img00"],
            "val": [],
            "test": [s for s in data if s.source_image_id == "img00"],
        }

    monkeypatch.setattr(sys.modules["data.fresh_splits"], "grouped_split", leaking_split)
    with pytest.raises(ValueError, match="group"):
        fp.cache_features(
            "deepweeds", "imagenet", root=root, cache_dir=str(tmp_path / "leak_cache"),
            seed=42, feature_batch_size=3,
        )
    assert calls == []


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #
def test_run_protocol_writes_additive_artifacts_and_records_failures(monkeypatch, tmp_path) -> None:
    pytest.importorskip("sklearn.svm")
    pytest.importorskip("scipy")

    rng = np.random.default_rng(2)
    X_train = rng.normal(size=(30, 4))
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.normal(size=(10, 4))
    y_test = (X_test[:, 0] > 0).astype(int)

    def fake_cache(dataset, backbone, root=None, cache_dir=None, *, seed=42, ratios=fp.DEFAULT_RATIOS, force=False):
        return {
            "X": {"train": X_train, "val": X_train[:5], "test": X_test},
            "y": {"train": y_train, "val": y_train[:5], "test": y_test},
            "groups": {"train": ["g0"], "val": ["g1"], "test": ["g2"]},
            "cache_path": "fake-cache.npz",
            "cache_hit": True,
            "provenance": {"split": {"seed": seed, "group_disjoint": True}},
        }

    real_arm = fp.run_svm_arm

    def arm_or_failure(Xtr, ytr, Xte, yte, kernel, seed=42):
        if kernel == "rbf":
            raise RuntimeError("synthetic RBF failure")
        return real_arm(Xtr, ytr, Xte, yte, kernel, seed=seed)

    monkeypatch.setattr(fp, "cache_features", fake_cache)
    monkeypatch.setattr(fp, "run_svm_arm", arm_or_failure)

    output_dir = tmp_path / "out"
    result = fp.run_protocol(
        "deepweeds",
        backbones=["plantnet"],
        seeds=3,
        models=["poly2", "rbf", "logreg"],
        output_dir=str(output_dir),
        cache_dir=str(tmp_path / "cache"),
    )
    metrics_path = Path(result["metrics_path"])
    provenance_path = Path(result["provenance_path"])
    assert metrics_path.exists() and provenance_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert metrics["n_runs_expected"] == 9
    assert metrics["n_runs_succeeded"] == 6
    assert len(metrics["warnings"]) == 3
    assert all(warning["error_type"] == "RuntimeError" for warning in metrics["warnings"])
    assert "plantnet/rbf" not in metrics["aggregates"]
    assert metrics["aggregates"]["plantnet/poly2"]["accuracy"]["n"] == 3
    assert metrics["claim_boundary"] and provenance["claim_boundary"]
    assert "no quantum advantage is asserted" in metrics["claim_boundary"].lower()
    assert provenance["dataset_split_provenance"]["plantnet"][0]["seed"] == 0
    assert provenance["environment"]["packages"]

    original = metrics_path.read_text(encoding="utf-8")
    second_result = fp.run_protocol(
        "deepweeds",
        backbones=["plantnet"],
        seeds=2,
        models=["poly2"],
        output_dir=str(output_dir),
    )
    assert Path(second_result["metrics_path"]) != metrics_path
    assert metrics_path.read_text(encoding="utf-8") == original


def test_run_protocol_cache_failure_skips_every_remaining_run(monkeypatch, tmp_path) -> None:
    pytest.importorskip("sklearn.svm")
    pytest.importorskip("scipy")

    rng = np.random.default_rng(4)
    X_train = rng.normal(size=(24, 3))
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.normal(size=(8, 3))
    y_test = (X_test[:, 0] > 0).astype(int)

    def fake_cache(dataset, backbone, root=None, cache_dir=None, *, seed=42,
                   ratios=fp.DEFAULT_RATIOS, force=False):
        if backbone == "plantnet":
            raise RuntimeError("synthetic cache failure")
        return {
            "X": {"train": X_train, "test": X_test},
            "y": {"train": y_train, "test": y_test},
            "groups": {"train": ["g0"], "test": ["g1"]},
            "cache_path": "fake-cache.npz",
            "cache_hit": True,
            "provenance": {"split": {"seed": seed}},
        }

    monkeypatch.setattr(fp, "cache_features", fake_cache)
    result = fp.run_protocol(
        "acre",
        backbones=["plantnet", "imagenet"],
        seeds=3,
        models=["poly2", "logreg"],
        output_dir=str(tmp_path / "out"),
    )
    metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    provenance = json.loads(Path(result["provenance_path"]).read_text(encoding="utf-8"))

    assert metrics["n_runs_expected"] == 12
    assert metrics["n_runs_succeeded"] == 6
    assert metrics["n_runs_skipped"] == 6
    assert metrics["n_skipped_cache_failure"] == 6
    assert metrics["accounting_invariant"] is True
    assert metrics["n_runs_expected"] == metrics["n_runs_succeeded"] + metrics["n_runs_skipped"]

    skips = [warning for warning in metrics["warnings"] if warning["backbone"] == "plantnet"]
    assert len(skips) == 6
    assert {(warning["seed"], warning["model"]) for warning in skips} == {
        (seed, model) for seed in range(3) for model in ("poly2", "logreg")
    }
    assert all(warning["status"] == "skipped_cache_failure" for warning in skips)
    assert all(warning["stage"] == "feature_cache" for warning in skips)
    assert all(
        warning["error_type"] == "RuntimeError"
        and warning["error"] == "synthetic cache failure"
        for warning in skips
    )
    assert all("metrics" not in warning for warning in metrics["warnings"])
    assert all("accuracy" not in warning for warning in metrics["warnings"])

    assert {run["backbone"] for run in metrics["runs"]} == {"imagenet"}
    assert all(run["metrics"]["accuracy"] is not None for run in metrics["runs"])

    manifest = provenance["run_manifest"]
    assert len(manifest) == 12
    assert {(entry["backbone"], entry["seed"], entry["model"]) for entry in manifest} == {
        (backbone, seed, model)
        for backbone in ("plantnet", "imagenet")
        for seed in range(3)
        for model in ("poly2", "logreg")
    }
    assert {entry["status"] for entry in manifest} <= {
        "succeeded",
        "skipped_failed_arm",
        "skipped_cache_failure",
    }
    assert sum(entry["status"] == "succeeded" for entry in manifest) == 6
    assert sum(entry["status"] == "skipped_cache_failure" for entry in manifest) == 6
    assert "exact_mcnemar" in provenance["comparison_notes"]


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def test_cli_dry_run_is_write_free_and_exposes_all_flags(monkeypatch, tmp_path, capsys) -> None:
    from scripts import run_fresh_protocol as cli

    def forbidden(*args, **kwargs):
        raise AssertionError("run_protocol must not be called during a dry run")

    monkeypatch.setattr(fp, "run_protocol", forbidden)
    output_dir = tmp_path / "no-writes"
    code = cli.main(
        [
            "--dataset",
            "acre",
            "--backbone",
            "plantnet",
            "--backbone",
            "imagenet",
            "--seeds",
            "4",
            "--models",
            "poly2",
            "--models",
            "logreg",
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--root",
            str(tmp_path / "root"),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not output_dir.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    plan = payload["plan"]
    assert plan["dataset"] == "acre"
    assert plan["backbones"] == ["plantnet", "imagenet"]
    assert plan["seeds"] == [0, 1, 2, 3]
    assert plan["models"] == ["poly2", "logreg"]
    assert plan["runs_planned"] == 16
    assert "claim_boundary" in plan

    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for flag in (
        "--dataset",
        "--backbone",
        "--seeds",
        "--models",
        "--output-dir",
        "--cache-dir",
        "--root",
        "--dry-run",
    ):
        assert flag in help_text
