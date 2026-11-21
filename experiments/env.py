"""Environment provenance manifest for a reproducible run.

This module records *where* a result was produced: the resolved experiment
configuration, interpreter version, exact library versions, CUDA device, local
git commit, and a UTC timestamp. A SHA-256 checksum over the timestamp-
independent fields makes the manifest auditable and comparable across runs.

Design constraints (Layer 0 of QC-D01):

* No network access, builds, or test runs are performed here.
* Every probe is defensive. A missing optional dependency is reported as
  ``"not installed"`` rather than raising, and a missing git binary or absent
  repository yields ``"unavailable"``. A manifest can therefore always be
  emitted, even on a minimal machine.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.config import ExperimentConfig

__all__ = [
    "collect_env_manifest",
    "write_env_manifest",
    "REQUIRED_KEYS",
    "REQUIRED_PACKAGES",
]

# (manifest key, importable module name). PennyLane is optional but must still
# be reported explicitly (as "not installed") when absent.
PACKAGE_MODULES: tuple[tuple[str, str], ...] = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("sklearn", "sklearn"),
    ("pandas", "pandas"),
    ("pennylane", "pennylane"),
)

NOT_INSTALLED = "not installed"
UNAVAILABLE = "unavailable"

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_KEYS = frozenset(
    {
        "seed",
        "feature_dim",
        "n_qubits",
        "config",
        "python_version",
        "packages",
        "cuda_available",
        "cuda_device_name",
        "git_commit_sha",
        "generated_at",
        "checksum_sha256",
    }
)
REQUIRED_PACKAGES = frozenset(key for key, _ in PACKAGE_MODULES)

# Wall-clock time is inherently non-reproducible, and the checksum obviously
# cannot include itself. Both are excluded from the hashed payload.
_CHECKSUM_EXCLUDED = frozenset({"generated_at", "checksum_sha256"})


def _utc_now_iso() -> str:
    """UTC timestamp in ISO-8601 form with an explicit ``+00:00`` offset."""
    return datetime.now(timezone.utc).isoformat()


def _package_version(module_name: str) -> str:
    """Return the installed version of a module, or ``"not installed"``.

    Import is intentionally guarded: PennyLane (and on minimal CI images even
    torch) may be absent. Any import failure is reported uniformly so that a
    manifest is always produced.
    """
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        return str(version) if version is not None else NOT_INSTALLED
    except Exception:  # depends entirely on the host environment
        return NOT_INSTALLED


def _cuda_info() -> tuple[bool, str]:
    """Return ``(is_available, device_name)``; both degrade gracefully."""
    try:
        torch = importlib.import_module("torch")
        available = bool(torch.cuda.is_available())
        if not available:
            return False, "N/A (CUDA unavailable)"
        try:
            name = str(torch.cuda.get_device_name(0))
        except Exception:
            name = "N/A (device query failed)"
        return True, name
    except Exception:
        return False, "N/A (torch not installed)"


def _git_commit_sha() -> str:
    """Full HEAD commit SHA from the local repo, or ``"unavailable"``.

    Uses only a local ``git rev-parse`` (no network). A missing git binary, a
    non-repository directory, or a timeout all map to ``"unavailable"``.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNAVAILABLE
    if proc.returncode != 0:
        return UNAVAILABLE
    sha = proc.stdout.strip()
    return sha if sha else UNAVAILABLE


def _canonical_json(payload: dict[str, Any]) -> str:
    """Compact, key-sorted JSON encoding used for checksum computation."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _checksum(payload: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON representation of ``payload``."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def collect_env_manifest(cfg: ExperimentConfig) -> dict:
    """Build the environment manifest for ``cfg``.

    The returned mapping always contains the keys in :data:`REQUIRED_KEYS`.
    ``config`` is the authoritative full snapshot (``cfg.as_dict()``); the
    top-level ``seed``/``feature_dim``/``n_qubits`` fields are convenience
    mirrors. ``checksum_sha256`` is a SHA-256 over the canonical JSON of every
    other field *except* ``generated_at`` (wall-clock), so it is stable across
    calls made against an unchanged configuration and environment.
    """
    cuda_available, cuda_device_name = _cuda_info()

    manifest: dict[str, Any] = {
        "seed": cfg.seed,
        "feature_dim": cfg.feature_dim,
        "n_qubits": cfg.n_qubits,
        "config": cfg.as_dict(),
        "python_version": platform.python_version(),
        "packages": {key: _package_version(module) for key, module in PACKAGE_MODULES},
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "git_commit_sha": _git_commit_sha(),
        "generated_at": _utc_now_iso(),
    }

    checksum_payload = {
        key: value
        for key, value in manifest.items()
        if key not in _CHECKSUM_EXCLUDED
    }
    manifest["checksum_sha256"] = _checksum(checksum_payload)
    return manifest


def write_env_manifest(
    cfg: ExperimentConfig,
    path: str | Path = "results/env_manifest.json",
) -> Path:
    """Create parent directories and write a deterministic manifest as JSON.

    Serialisation uses ``sort_keys=True`` and ``indent=2`` so that, for a fixed
    manifest, the file bytes are reproducible. Returns the written ``Path``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = collect_env_manifest(cfg)
    text = json.dumps(
        manifest, sort_keys=True, indent=2, ensure_ascii=False, default=str
    ) + "\n"
    target.write_text(text, encoding="utf-8")
    return target
