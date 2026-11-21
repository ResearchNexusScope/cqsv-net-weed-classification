"""Layer-0 environment manifest contract (QC-D01 §3/§4).

Covers required keys, a checksum that is independent of ``generated_at``,
deterministic JSON writing, and a JSON round-trip whose checksum is
self-consistent. Tests stay dependency-light: only stdlib + pytest + our own
modules, and no CUDA device or optional package is assumed present.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

# Allow both ``python -m pytest`` and a bare ``pytest`` invocation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.env as env
from experiments.config import ExperimentConfig

_CHECKSUM_EXCLUDED = {"generated_at", "checksum_sha256"}


def _cfg() -> ExperimentConfig:
    return ExperimentConfig(ninja_ann_dir="ann", ninja_img_dir="img", seed=123)


def _expected_checksum(manifest: dict) -> str:
    """Independently re-derive the documented checksum definition."""
    payload = {k: v for k, v in manifest.items() if k not in _CHECKSUM_EXCLUDED}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_manifest_contains_all_required_keys() -> None:
    cfg = _cfg()
    manifest = env.collect_env_manifest(cfg)

    assert env.REQUIRED_KEYS <= set(manifest)
    assert manifest["seed"] == cfg.seed == 123
    assert manifest["config"] == cfg.as_dict()  # full config snapshot
    assert manifest["config"]["feature_dim"] == 2 ** manifest["config"]["n_qubits"]
    assert manifest["feature_dim"] == 2 ** manifest["n_qubits"]

    assert env.REQUIRED_PACKAGES <= set(manifest["packages"])
    assert isinstance(manifest["python_version"], str)
    assert manifest["python_version"]
    assert isinstance(manifest["cuda_available"], bool)
    assert isinstance(manifest["cuda_device_name"], str)
    assert isinstance(manifest["git_commit_sha"], str)
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["checksum_sha256"])
    # PennyLane is optional but must always be reported explicitly.
    assert isinstance(manifest["packages"]["pennylane"], str)


def test_default_feature_dim_matches_qubit_count() -> None:
    cfg = ExperimentConfig()
    manifest = env.collect_env_manifest(cfg)
    assert manifest["config"]["feature_dim"] == 2 ** cfg.n_qubits


def test_checksum_is_stable_when_generated_at_differs(monkeypatch) -> None:
    timestamps = iter(
        ["2026-01-01T00:00:00+00:00", "2030-12-31T23:59:59+00:00"]
    )
    monkeypatch.setattr(env, "_utc_now_iso", lambda: next(timestamps))
    cfg = _cfg()

    first = env.collect_env_manifest(cfg)
    second = env.collect_env_manifest(cfg)

    assert first["generated_at"] != second["generated_at"]
    assert first["checksum_sha256"] == second["checksum_sha256"]
    assert first["checksum_sha256"] == _expected_checksum(first)
    assert second["checksum_sha256"] == _expected_checksum(second)


def test_checksum_excludes_generated_at_explicitly(monkeypatch) -> None:
    monkeypatch.setattr(env, "_utc_now_iso", lambda: "2000-01-01T00:00:00+00:00")
    manifest = env.collect_env_manifest(_cfg())
    expected = manifest["checksum_sha256"]

    # Tamper only with the wall-clock field; the checksum must not change.
    tampered = dict(manifest)
    tampered["generated_at"] = "2099-09-09T09:09:09+00:00"
    assert _expected_checksum(tampered) == expected


def test_write_env_manifest_roundtrip(tmp_path: Path) -> None:
    cfg = _cfg()
    target = tmp_path / "nested" / "env_manifest.json"

    returned = env.write_env_manifest(cfg, target)

    assert returned == target
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    loaded = json.loads(text)

    assert env.REQUIRED_KEYS <= set(loaded)
    assert loaded["seed"] == cfg.seed
    # JSON turns tuples in the config snapshot into lists.
    assert loaded["config"] == json.loads(json.dumps(cfg.as_dict()))
    # Deterministic serialisation: sort_keys=True, indent=2.
    assert text == json.dumps(
        loaded, sort_keys=True, indent=2, ensure_ascii=False, default=str
    ) + "\n"
    # The checksum survives the serialisation round-trip unchanged.
    assert loaded["checksum_sha256"] == _expected_checksum(loaded)


def test_write_env_manifest_creates_default_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    returned = env.write_env_manifest(_cfg())

    assert returned == Path("results/env_manifest.json")
    assert (tmp_path / "results" / "env_manifest.json").is_file()
