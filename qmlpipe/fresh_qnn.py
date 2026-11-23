"""Pure variational-QNN arm on the fresh grouped-split protocol (QNN-D01).

This module is the QNN counterpart of :mod:`experiments.fresh_protocol`.  It
reuses the committed frozen-feature cache and the pre-registered statistics
helpers; it never re-extracts images, never touches TEST labels outside the
single scoring call, and never edits the classical protocol module.

Design rules
------------
* The trainable classifier is the existing
  :func:`qmlpipe.models.build_qnn` (embedding + entangled ansatz + PauliZ
  readout).  ``build_transfer_qnn`` is intentionally NOT used: it expects raw
  images and would double-extract features, whereas here features are already
  cached vectors.
* The cached feature width (e.g. 512) is projected/padded to ``2 ** qubits``
  (the AmplitudeEmbedding contract) using a TRAIN-only PCA fit (deterministic
  seeded random projection as fallback).  The identical transform is stored on
  the trained model and re-applied to VAL/TEST by :func:`evaluate_once`, so no
  test information can influence it.  The small register is recorded as an
  unequal input-information budget relative to the classical control.
* Training is CPU-bounded: at most ``_MAX_BATCHES_PER_EPOCH`` mini-batches per
  epoch and ``_MAX_TOTAL_STEPS`` optimisation steps in total.  Validation is
  used only for checkpoint selection; TEST is scored exactly once per seed.
* Aggregates use :func:`experiments.fresh_protocol.aggregate` (n, ddof=1 SD,
  seeded bootstrap CI).  Comparisons are paired per seed against the committed
  classical poly-2 arm via :func:`experiments.fresh_protocol.paired_comparison`
  followed by :func:`experiments.fresh_protocol.holm_adjust`.
* No quantum advantage is asserted.  The default 4 qubits / 3 epochs are CPU
  feasibility choices, not a hardware or scaling claim (see
  :data:`CLAIM_BOUNDARY`).

Import safety: numpy + the config/statistics modules are imported at module
scope; torch, PennyLane, scipy and sklearn are imported lazily so a minimal
verifier can import this module and run the statistics/plan helpers.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import fresh_protocol as _fp  # noqa: E402  (numpy-only at import)
from qmlpipe.config import QMLConfig  # noqa: E402  (stdlib-only at import)

__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_CLASSICAL_PATHS",
    "qnn_config",
    "project_features",
    "train_qnn_from_features",
    "evaluate_once",
    "cap_train",
    "load_classical_poly2",
    "paired_against_classical",
    "build_plan",
    "run_qnn_protocol",
]

CLAIM_BOUNDARY = (
    "Pure variational-QNN arm on the fresh grouped-split protocol. No quantum "
    "advantage is asserted. A QNN win is admissible only if the per-seed paired "
    "comparison against the committed classical poly-2 arm is 'significant' after "
    "Holm correction with achieved power >= 0.8, n >= 10, a bootstrap CI of the "
    "paired difference excluding zero, and a matched data budget. The default "
    "4 qubits and 3 epochs are CPU-feasibility choices, not a hardware or scaling "
    "claim. The QNN consumes a TRAIN-only projection of the frozen cached "
    "features into a small amplitude register; any unequal input-information "
    "budget versus the classical control "
    "is recorded and no performance difference is attributed to quantum effects. "
    "The sealed TEST is scored exactly once per seed; every reported value comes "
    "from an executed run and failures are recorded as warnings, never fabricated."
)

# Committed classical control artifacts (paths only; no result numbers live here).
DEFAULT_CLASSICAL_PATHS: dict[str, str] = {
    "deepweeds": "results/fresh_deepweeds/metrics.json",
    "acre": "results/fresh_acre_capped/metrics.json",
}

_METRICS = ("accuracy", "auc", "f1", "precision", "recall")
_MAX_BATCHES_PER_EPOCH = 32
_MAX_TOTAL_STEPS = 512
_EVAL_MARK = "_fresh_qnn_test_evaluations"
_PROJ_ATTR = "_fresh_qnn_projection"
_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Small local helpers (no private edits to fresh_protocol)                    #
# --------------------------------------------------------------------------- #
def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on the host environment
        raise RuntimeError(
            "qmlpipe.fresh_qnn requires PyTorch for QNN training/evaluation; "
            f"import failed with {type(exc).__name__}: {exc}"
        ) from exc
    return torch, F


def _normalize_seeds(seeds: int | Sequence[int]) -> list[int]:
    if isinstance(seeds, int):
        return list(range(int(seeds)))
    out = [int(value) for value in seeds]
    return out or [int(_fp.DEFAULT_SEED)]


def _default_cache_dir() -> Path:
    """Prefer the committed ``~/.cache/qc_datasets/features`` cache root.

    Falls back to the ``fresh_protocol.cache_features`` temp default when the
    home cache is absent, so direct library calls remain compatible with both
    layouts.  The CLI always passes the home-cache path explicitly.
    """
    preferred = Path.home() / ".cache" / "qc_datasets" / "features"
    if preferred.exists():
        return preferred
    return Path(tempfile.gettempdir()) / "qc_fresh_features"


def _cache_npz_path(dataset: str, backbone: str, cache_dir) -> Path:
    """Reproduce the private cache-path convention of ``fresh_protocol`` for checks."""
    base = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    root_key = hashlib.sha256(b"default").hexdigest()[:12]
    return base / f"{dataset}_{backbone}_{root_key}.npz"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _resolve_output_paths(output_dir) -> tuple[Path, Path]:
    """Additive artifact paths: never overwrite an existing metrics/provenance pair."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path, provenance_path = out / "metrics.json", out / "provenance.json"
    if metrics_path.exists() or provenance_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = out / f"fresh_qnn_{stamp}"
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = out / f"fresh_qnn_{stamp}_{suffix}"
        candidate.mkdir(parents=True, exist_ok=True)
        metrics_path, provenance_path = candidate / "metrics.json", candidate / "provenance.json"
    return metrics_path, provenance_path


def _environment_pins() -> dict:
    from importlib.metadata import version

    def pinned(name: str):
        try:
            return version(name)
        except Exception:
            return None

    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": os.environ.get("OS", "unknown"),
        "commit_hash": os.environ.get("GIT_COMMIT_HASH"),
        "packages": {
            name: pinned(name)
            for name in ("numpy", "scipy", "scikit-learn", "statsmodels", "torch", "PennyLane")
        },
        "note": "versions importable in the executing interpreter; no installs performed",
    }


# --------------------------------------------------------------------------- #
# Config + projection (testable without PennyLane)                            #
# --------------------------------------------------------------------------- #
def qnn_config(
    *,
    seed: int,
    qubits: int,
    layers: int,
    epochs: int,
    feature_dim: int | None = None,
    batch_size: int = 32,
    class_weight: str = "balanced",
) -> QMLConfig:
    """Build and validate a :class:`~qmlpipe.config.QMLConfig` for the QNN arm.

    ``projection_dim`` is always ``2 ** qubits`` (the AmplitudeEmbedding
    contract).  ``classical_dim`` records the raw cached feature width when
    supplied; no image backbone is constructed by this arm.
    """
    cfg = QMLConfig(
        seed=int(seed),
        qubits=int(qubits),
        layers=int(layers),
        epochs=int(epochs),
        batch_size=int(batch_size),
        projection_dim=2 ** int(qubits),
        classical_dim=int(feature_dim) if feature_dim is not None else 1280,
        embedding="amplitude",
        ansatz="strongly_entangling",
        device_name="default.qubit",
        shots=None,
        loss="ce",
        class_weight=str(class_weight),
        early_metric="val_loss",
    )
    cfg.validate()  # hard contract: projection_dim == 2 ** qubits
    return cfg


def _make_projection(input_dim: int, target_dim: int, seed: int):
    """Return a deterministic projection matrix, or None for pad/identity."""
    if input_dim == target_dim or input_dim < target_dim:
        return None
    rng = np.random.default_rng(int(seed))
    return (rng.standard_normal((int(input_dim), int(target_dim))) / float(np.sqrt(target_dim))).astype(
        np.float32
    )


def project_features(X, projection: Mapping[str, Any]) -> np.ndarray:
    """Apply a stored projection meta-dict to raw cached features.

    Padding is zeros; over-width inputs use the recorded TRAIN-fitted PCA (or
    seeded random fallback) matrix after optional mean subtraction.  Every row
    is L2-normalised before AmplitudeEmbedding.  The same meta dict must be used
    for TRAIN, VAL and TEST.
    """
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"features must be a 2-D matrix, got shape {arr.shape}")
    in_dim, target = int(projection["input_dim"]), int(projection["target_dim"])
    if arr.shape[1] != in_dim:
        raise ValueError(
            f"expected {in_dim} raw cached features, got {arr.shape[1]}; "
            "the projection is bound to the training feature width"
        )
    matrix = projection.get("matrix")
    mean = projection.get("mean")
    centered = arr - np.asarray(mean, dtype=np.float32) if mean is not None else arr
    if matrix is not None:
        out = centered @ np.asarray(matrix, dtype=np.float32)
    elif arr.shape[1] < target:
        out = np.zeros((arr.shape[0], target), dtype=np.float32)
        out[:, : arr.shape[1]] = centered
    else:
        out = centered
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norms, _EPS)).astype(np.float32, copy=False)


def _projection_meta(X, cfg: QMLConfig, seed: int) -> tuple[dict, np.ndarray]:
    """Fit the TRAIN->2**qubits projection on TRAIN only (never on VAL/TEST).

    PCA is preferred because it preserves the dominant frozen-feature variance
    in the small amplitude-embedding register.  When sklearn is unavailable (or
    the matrix cannot be computed) a fixed seeded random projection is used;
    either way the identical transform is stored on the model for TEST.
    """
    arr = np.asarray(X, dtype=np.float32)
    input_dim = int(arr.shape[1])
    target = int(cfg.feature_dim)  # == 2 ** qubits
    mean = None
    matrix = None
    method = "pad_or_identity"
    if input_dim > target:
        try:
            from sklearn.decomposition import PCA

            mean = arr.mean(axis=0).astype(np.float32)
            pca = PCA(n_components=target, svd_solver="full", random_state=int(seed))
            pca.fit(arr - mean)
            matrix = np.asarray(pca.components_, dtype=np.float32).T
            method = "train_pca"
        except Exception:  # pragma: no cover - lightweight verifier fallback
            matrix = _make_projection(input_dim, target, seed)
            method = "seeded_random_projection"
            mean = None
    meta = {
        "method": method,
        "input_dim": input_dim,
        "target_dim": target,
        "seed": int(seed),
        "mean": mean,
        "matrix": matrix,
        "note": (
            "Linear projection fitted on TRAIN only (PCA) or derived from the seed "
            "(random fallback); the identical transform is applied to VAL/TEST and "
            "no TEST information enters the projection."
        ),
    }
    return meta, project_features(X, meta)


def _cap_train_local(X, y, cap: int, seed: int):
    """Dependency-free fallback for the stratified TRAIN cap."""
    cap_int, n = int(cap), int(len(np.asarray(y)))
    meta: dict[str, Any] = {
        "train_cap": cap_int,
        "train_size_before": n,
        "train_size_after": n,
        "cap_applied": False,
        "note": "no cap applied (cap<=0 or train size <= cap)",
    }
    if cap_int <= 0 or n <= cap_int:
        return X, y, meta
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    k = len(classes)
    alloc = np.ones(k, dtype=int)
    if cap_int >= k:
        raw = (cap_int - k) * (counts / counts.sum())
        floors = np.floor(raw).astype(int)
        alloc += floors
        order = sorted(range(k), key=lambda i: (-(float(raw[i]) - int(floors[i])), i))
        leftover = int((cap_int - k) - floors.sum())
        for index in order[:leftover]:
            alloc[index] += 1
        for index in range(k):
            alloc[index] = min(alloc[index], int(counts[index]))
    else:
        alloc = np.zeros(k, dtype=int)
        for index in sorted(range(k), key=lambda i: (-int(counts[i]), i))[:cap_int]:
            alloc[index] = 1
    rng = np.random.default_rng(int(seed))
    chosen = [
        rng.choice(np.flatnonzero(y_arr == cls), size=int(take), replace=False)
        for cls, take in zip(classes, alloc)
        if take > 0
    ]
    selected = np.concatenate(chosen) if chosen else np.empty(0, dtype=int)
    selected.sort()
    X_sub, y_sub = np.asarray(X)[selected], y_arr[selected]
    meta.update(
        {
            "train_size_after": int(len(y_sub)),
            "cap_applied": True,
            "note": f"stratified TRAIN subsample to at most {cap_int} rows (seed {int(seed)}) [local fallback]",
        }
    )
    return X_sub, y_sub, meta


def cap_train(X, y, cap: int, seed: int):
    """Use ``experiments.fresh_protocol.cap_train`` with a local fallback."""
    cap_train_fn = getattr(_fp, "cap_train", None)
    if callable(cap_train_fn):
        return cap_train_fn(X, y, cap, seed)
    return _cap_train_local(X, y, cap, seed)


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train_qnn_from_features(X_train, y_train, X_val, y_val, cfg, seed):
    """Train one variational QNN on cached features (TRAIN/VAL only).

    Returns ``(model, history)``.  The TRAIN-fitted (seed-deterministic) feature
    projection is attached to ``model`` so that :func:`evaluate_once` applies
    the identical transform to TEST.  No TEST label or feature is read here.
    """
    torch, _F = _require_torch()
    from qmlpipe.losses import class_weights_from_labels, weighted_cross_entropy
    from qmlpipe.models import build_qnn
    from qmlpipe.train import seed_everything

    projection, X_fit = _projection_meta(X_train, cfg, seed)
    y_fit = np.asarray(y_train, dtype=np.int64).reshape(-1)
    X_val_proj = project_features(X_val, projection)
    y_val_arr = np.asarray(y_val, dtype=np.int64).reshape(-1)
    if len(y_fit) != len(X_fit) or len(y_val_arr) != len(X_val_proj):
        raise ValueError("label/feature counts disagree after projection")

    seed_everything(int(seed))
    model = build_qnn(cfg)  # TorchLayer weight init is seeded by cfg.seed
    setattr(model, _PROJ_ATTR, {key: value for key, value in projection.items()})

    weights = class_weights_from_labels(y_fit, n_classes=2, mode=str(getattr(cfg, "class_weight", "none")))
    optimizer_name = str(getattr(cfg, "optimizer", "adamw")).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay)
        )

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_fit), torch.from_numpy(y_fit)
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(cfg.batch_size), shuffle=True, generator=generator
    )
    X_val_t = torch.from_numpy(X_val_proj)

    initial = [p.detach().clone() for p in model.parameters()]
    first_grad_norm: float | None = None
    total_steps = 0
    best_val = float("inf")
    best_state = None
    epoch_log: list[dict] = []

    def _val_metrics():
        model.eval()
        with torch.no_grad():
            loss_sum, correct = 0.0, 0
            for start in range(0, len(X_val_t), int(cfg.batch_size)):
                xb = X_val_t[start : start + int(cfg.batch_size)]
                yb = torch.from_numpy(y_val_arr[start : start + len(xb)])
                logits = model(xb)
                batch_loss = weighted_cross_entropy(logits, yb, weight=weights, reduction="sum")
                loss_sum += float(batch_loss.item())
                correct += int((logits.argmax(dim=1) == yb).sum().item())
        return loss_sum / max(1, len(y_val_arr)), correct / max(1, len(y_val_arr))

    for epoch in range(int(cfg.epochs)):
        model.train()
        loss_sum, correct, seen, n_batches = 0.0, 0, 0, 0
        for xb, yb in loader:
            if n_batches >= _MAX_BATCHES_PER_EPOCH or total_steps >= _MAX_TOTAL_STEPS:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = weighted_cross_entropy(logits, yb, weight=weights)
            loss.backward()
            if first_grad_norm is None:
                grads = [
                    p.grad.detach().reshape(-1)
                    for p in model.parameters()
                    if p.grad is not None
                ]
                first_grad_norm = (
                    float(torch.linalg.vector_norm(torch.cat(grads)).item()) if grads else 0.0
                )
            optimizer.step()
            total_steps += 1
            n_batches += 1
            loss_sum += float(loss.item()) * len(yb)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            seen += len(yb)
            if not np.isfinite(loss_sum):
                raise RuntimeError("non-finite training loss; refusing to report a QNN result")
        val_loss, val_acc = _val_metrics()
        epoch_log.append(
            {
                "epoch": int(epoch),
                "train_loss": float(loss_sum / max(1, seen)),
                "train_acc": float(correct / max(1, seen)),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "n_batches": int(n_batches),
                "n_train_seen": int(seen),
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    param_delta = float(
        sum(
            float((p.detach() - start).norm().item())
            for p, start in zip(model.parameters(), initial)
        )
    )

    try:
        from qmlpipe.models import circuit_introspection

        introspection = circuit_introspection(model)
    except Exception as exc:  # pragma: no cover - depends on PennyLane version
        introspection = {
            "entangling_gate_count": None,
            "trainable_param_count": None,
            "introspection_error": f"{type(exc).__name__}",
        }

    history: dict[str, Any] = {
        "seed": int(seed),
        "epochs_configured": int(cfg.epochs),
        "max_batches_per_epoch": _MAX_BATCHES_PER_EPOCH,
        "max_total_steps": _MAX_TOTAL_STEPS,
        "total_steps": int(total_steps),
        "best_val_loss": float(best_val) if np.isfinite(best_val) else None,
        "projection": {key: value for key, value in projection.items() if key != "matrix"},
        "grad_norm_first_batch": first_grad_norm,
        "param_delta": param_delta,
        "class_weight_mode": str(getattr(cfg, "class_weight", "none")),
        "optimizer": optimizer_name,
        "entangling_gate_count": introspection.get("entangling_gate_count"),
        "trainable_param_count": introspection.get("trainable_param_count"),
        "epoch_log": epoch_log,
    }
    return model, history


# --------------------------------------------------------------------------- #
# Sealed TEST evaluation                                                       #
# --------------------------------------------------------------------------- #
def evaluate_once(model, X_test, y_test) -> dict:
    """Score the sealed TEST exactly once and return metric scalars + confusion.

    A one-shot mark on ``model`` makes a second call fail loudly, so no test
    threshold or operating point can be tuned on the test split.  Metrics come
    from :func:`experiments.metrics.binary_metrics`.
    """
    torch, _F = _require_torch()
    from experiments.metrics import binary_metrics

    seen = 0
    try:
        seen = int(getattr(model, _EVAL_MARK, 0))
    except Exception:  # pragma: no cover - exotic model objects
        seen = 0
    if seen >= 1:
        raise RuntimeError(
            "evaluate_once may score the sealed TEST exactly once per model "
            "(a second call would invalidate the held-out estimate)"
        )

    projection = getattr(model, _PROJ_ATTR, None)
    if projection is None:
        raw = np.asarray(X_test)
        if raw.ndim != 2:
            raise ValueError("TEST features must be a 2-D matrix")
        X_score = raw.astype(np.float32, copy=False)
    else:
        X_score = project_features(X_test, projection)

    y_true = np.asarray(y_test, dtype=np.int64).reshape(-1)
    if len(y_true) != len(X_score):
        raise ValueError("TEST label/feature counts disagree")

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_score))
    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    metrics = binary_metrics(y_true, preds, y_prob=probs)

    result = {
        "model": "qnn",
        "score_kind": "softmax(logits)[:,1]",
        "accuracy": float(metrics.accuracy),
        "auc": None if metrics.auc is None else float(metrics.auc),
        "f1": float(metrics.f1),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "confusion_matrix": metrics.confusion.tolist(),
        "n": int(metrics.support),
        "iou_class0": float(metrics.iou_class0),
        "iou_class1": float(metrics.iou_class1),
        "macro_iou": float(metrics.macro_iou),
        "test_evaluations": 1,
    }
    try:
        setattr(model, _EVAL_MARK, seen + 1)
    except Exception:  # pragma: no cover - exotic model objects
        pass
    return result


# --------------------------------------------------------------------------- #
# Committed classical control + paired statistics                              #
# --------------------------------------------------------------------------- #
def load_classical_poly2(dataset: str, backbone: str, results_root=None) -> dict[int, float]:
    """Read committed per-seed poly-2 accuracies (no numbers are hardcoded here)."""
    relative = DEFAULT_CLASSICAL_PATHS.get(str(dataset).lower())
    if relative is None:
        return {}
    path = (Path(results_root) if results_root is not None else _REPO_ROOT) / relative
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, float] = {}
    for run in payload.get("runs", []):
        if str(run.get("model", "")).lower() not in {"poly2", "poly"}:
            continue
        if str(run.get("backbone", "")).lower() != str(backbone).lower():
            continue
        if str(run.get("status", "ok")) != "ok":
            continue
        block = run.get("metrics", {})
        accuracy = block.get("accuracy", run.get("accuracy"))
        if accuracy is None:
            continue
        try:
            out[int(run["seed"])] = float(accuracy)
        except (KeyError, TypeError, ValueError):  # pragma: no cover - malformed artifact
            continue
    return out


def paired_against_classical(
    qnn_by_seed: Mapping[int, float],
    classical_by_seed: Mapping[int, float],
    *,
    alpha: float = 0.05,
    seed: int = _fp.BOOT_SEED,
    n_boot: int = _fp.BOOT_B,
) -> list[dict]:
    """Pair QNN vs classical accuracy per seed and Holm-adjust the family.

    A ``qnn_win_supported`` flag is True only when Holm's rule yields a
    ``significant`` decision in the QNN's favour; that decision already requires
    n >= 10, power >= 0.8 and a CI excluding zero.  The claim boundary remains
    unconditional: this flag is evidence, never a claim by itself.
    """
    common = sorted(set(int(k) for k in qnn_by_seed) & set(int(k) for k in classical_by_seed))
    if not common:
        return []
    a = [float(qnn_by_seed[s]) for s in common]
    b = [float(classical_by_seed[s]) for s in common]
    comparison = _fp.paired_comparison(
        a,
        b,
        alpha=float(alpha),
        seed=int(seed),
        n_boot=int(n_boot),
        pair=("qnn", "poly2"),
    )
    comparison["seeds"] = [int(s) for s in common]
    comparison["metric"] = "accuracy"
    comparison["mean_qnn"] = float(np.mean(a))
    comparison["mean_classical_poly2"] = float(np.mean(b))
    comparison["mean_difference"] = float(np.mean(np.asarray(a) - np.asarray(b)))
    for item in _fp.holm_adjust([comparison], alpha=float(alpha)):
        item["qnn_win_supported"] = bool(
            item.get("decision") == "significant"
            and item.get("mean_difference") is not None
            and float(item["mean_difference"]) > 0.0
        )
    return [comparison]


# --------------------------------------------------------------------------- #
# Plan + orchestration                                                         #
# --------------------------------------------------------------------------- #
def build_plan(
    dataset: str,
    backbone: str,
    seeds,
    output_dir,
    cache_dir=None,
    train_cap: int = 2000,
    epochs: int = 3,
    qubits: int = 4,
    layers: int = 2,
) -> dict:
    """Return a write-free plan (used by the CLI's ``--dry-run``)."""
    seed_list = _normalize_seeds(seeds)
    return {
        "dry_run": True,
        "dataset": str(dataset),
        "backbone": str(backbone),
        "seeds": seed_list,
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir) if cache_dir is not None else str(_default_cache_dir()),
        "train_cap": int(train_cap),
        "epochs": int(epochs),
        "qubits": int(qubits),
        "layers": int(layers),
        "projection_target_dim": 2 ** int(qubits),
        "runs_planned": len(seed_list),
        "max_batches_per_epoch": _MAX_BATCHES_PER_EPOCH,
        "max_total_steps": _MAX_TOTAL_STEPS,
        "classical_control": DEFAULT_CLASSICAL_PATHS.get(str(dataset).lower()),
        "claim_boundary": CLAIM_BOUNDARY,
        "qubit_note": (
            "4 qubits is chosen for CPU feasibility only; it is not a hardware, "
            "scaling or quantum-advantage claim."
        ),
    }


def run_qnn_protocol(
    dataset: str,
    backbone: str,
    seeds,
    output_dir,
    cache_dir=None,
    train_cap: int = 2000,
    epochs: int = 3,
    qubits: int = 4,
    layers: int = 2,
) -> dict:
    """Run one bounded QNN per seed on cached features and write additive JSON.

    Reuses ``experiments.fresh_protocol.cache_features`` so every seed sees the
    same group-disjoint TRAIN/VAL/TEST split as the committed classical arm.
    Returns the same shape as ``run_protocol``: metrics, provenance, paths,
    warnings and the claim boundary.
    """
    seed_list = _normalize_seeds(seeds)
    cache_dir = cache_dir if cache_dir is not None else _default_cache_dir()
    expected_cache = _cache_npz_path(dataset, backbone, cache_dir)
    if not expected_cache.exists():
        raise FileNotFoundError(
            f"frozen feature cache not found: {expected_cache}. "
            "Run the classical fresh-protocol featurisation first; this arm never "
            "extracts images or fabricates features."
        )

    plan = build_plan(dataset, backbone, seed_list, output_dir, cache_dir, train_cap, epochs, qubits, layers)
    warnings: list[dict] = []
    runs: list[dict] = []
    feature_dim: int | None = None
    cache_hit: bool | None = None

    for seed in seed_list:
        try:
            cached = _fp.cache_features(dataset, backbone, cache_dir=cache_dir, seed=int(seed))
            X_fit, y_fit, cap_meta = cap_train(
                cached["X"]["train"], cached["y"]["train"], train_cap, int(seed)
            )
            feature_dim = int(np.asarray(cached["X"]["train"]).shape[1])
            cache_hit = bool(cached.get("cache_hit")) if cache_hit is None else cache_hit
            cfg = qnn_config(
                seed=int(seed),
                qubits=int(qubits),
                layers=int(layers),
                epochs=int(epochs),
                feature_dim=feature_dim,
            )
            model, history = train_qnn_from_features(
                X_fit, y_fit, cached["X"]["val"], cached["y"]["val"], cfg, int(seed)
            )
            result = evaluate_once(model, cached["X"]["test"], cached["y"]["test"])
        except Exception as exc:  # record, never fabricate
            warnings.append(
                {
                    "stage": "qnn_arm",
                    "dataset": str(dataset),
                    "backbone": str(backbone),
                    "seed": int(seed),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        runs.append(
            {
                "dataset": str(dataset),
                "backbone": str(backbone),
                "seed": int(seed),
                "model": "qnn",
                "status": "ok",
                "train_cap": cap_meta,
                "metrics": result,
                "history": history,
                "cache_path": cached.get("cache_path"),
                "group_disjoint": bool(cached.get("provenance", {}).get("group_disjoint")),
            }
        )

    aggregates: dict[str, Any] = {"qnn": {"n_runs": len(runs), "seeds": [r["seed"] for r in runs]}}
    for metric in _METRICS:
        values = [
            float(run["metrics"][metric])
            for run in runs
            if run["metrics"].get(metric) is not None and np.isfinite(float(run["metrics"][metric]))
        ]
        if values:
            aggregates["qnn"][metric] = _fp.aggregate(values, name="qnn")

    qnn_by_seed = {
        int(r["seed"]): float(r["metrics"]["accuracy"])
        for r in runs
        if r["metrics"].get("accuracy") is not None
    }
    classical_by_seed = load_classical_poly2(dataset, backbone)
    comparisons = paired_against_classical(qnn_by_seed, classical_by_seed)

    n_succeeded, n_skipped = len(runs), len(warnings)
    metrics_payload = {
        "schema": "fresh_qnn.metrics.v1",
        "dataset": str(dataset),
        "backbone": str(backbone),
        "claim_boundary": CLAIM_BOUNDARY,
        "config": {key: value for key, value in plan.items() if key != "dry_run"},
        "runs": runs,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "warnings": warnings,
        "n_runs_expected": int(len(seed_list)),
        "n_runs_succeeded": int(n_succeeded),
        "n_runs_skipped": int(n_skipped),
        "accounting_invariant": bool(len(seed_list) == n_succeeded + n_skipped),
    }
    provenance_payload = {
        "schema": "fresh_qnn.provenance.v1",
        "dataset": str(dataset),
        "backbone": str(backbone),
        "claim_boundary": CLAIM_BOUNDARY,
        "config": {key: value for key, value in plan.items() if key != "dry_run"},
        "environment": _environment_pins(),
        "feature_cache": {
            "cache_path": str(expected_cache),
            "cache_hit": cache_hit,
            "feature_dim": feature_dim,
            "frozen": True,
        },
        "run_manifest": [
            {
                "dataset": str(dataset),
                "backbone": str(backbone),
                "seed": int(r["seed"]),
                "model": "qnn",
                "status": "ok",
                "train_cap": r["train_cap"],
                "projection": r["history"]["projection"],
                "grad_norm_first_batch": r["history"].get("grad_norm_first_batch"),
                "param_delta": r["history"].get("param_delta"),
                "entangling_gate_count": r["history"].get("entangling_gate_count"),
            }
            for r in runs
        ]
        + warnings,
        "comparison_notes": {
            "classical_arm": plan["classical_control"],
            "classical_seeds_matched": [int(s) for s in classical_by_seed],
            "n_budget_matched": False,
            "note": (
                "The committed classical arm used its own TRAIN budget while this arm "
                "caps TRAIN at the configured value; a paired win is therefore only "
                "admissible after budget matching in addition to the statistical gates."
            ),
            "test_scoring": "exactly one forward pass per seed via evaluate_once",
            "qubit_note": plan["qubit_note"],
        },
        "warnings": warnings,
        "additive": True,
    }

    metrics_path, provenance_path = _resolve_output_paths(output_dir)
    _write_json(metrics_path, metrics_payload)
    _write_json(provenance_path, provenance_payload)
    return {
        "metrics": metrics_payload,
        "provenance": provenance_payload,
        "metrics_path": str(metrics_path),
        "provenance_path": str(provenance_path),
        "warnings": warnings,
        "claim_boundary": CLAIM_BOUNDARY,
    }
