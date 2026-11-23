"""Hybrid classical + variational-quantum arm for the fresh 70:20:10 protocol (HYBRID-D01).

This module is the HYBRID arm only.  The pure-QNN arm is owned by the peer worker
in ``qmlpipe/fresh_qnn.py`` and is deliberately NOT imported, reimplemented or
shadowed here.  The classical control arms live in ``experiments/fresh_protocol``
and are likewise never duplicated.

Architecture choice -- parallel concatenation hybrid (NOT a pure QNN)
---------------------------------------------------------------------
For a cached feature row ``x`` (width ``D``) the model computes two branches in
parallel and fuses them:

1. classical branch: ``nn.Linear(D, hidden) -> nn.ReLU()``  (trainable);
2. quantum branch: ``nn.Linear(D, 2**qubits)`` (trainable adapter) feeding
   ``qmlpipe.models.build_qnn`` with *reduced* qubits (default 4), an amplitude
   embedding and a trainable, entangling ``StronglyEntanglingLayers`` ansatz.
   The QNN emits two logits, which are used here as the quantum representation;
3. fusion: ``Linear(concat([classical_hidden, quantum_logits]), 2)`` produces
   the final logits.

Both branches carry trainable parameters and receive non-zero gradients during
training (recorded in ``history['classical_grad_norm']`` /
``history['quantum_grad_norm']``), and the fused classifier can only use the
quantum signal through parameters that are optimised end-to-end.  A pure QNN
has no classical compression branch and would feed its circuit output straight
to a classifier; this model is therefore provably a hybrid rather than a
renamed QNN.  The trainable ansatz is entangled, so this is not a
parameter-free amplitude-embedding kernel either.

Scientific scope and caveats
----------------------------
* Features are frozen and loaded once via
  ``experiments.fresh_protocol.cache_features``; no backbone is retrained.
* Splits are group-disjoint, TRAIN is deterministically capped per seed with
  ``experiments.fresh_protocol.cap_train`` and TEST is sealed and scored exactly
  once by :func:`evaluate_once`.
* The quantum circuit runs on PennyLane's ``default.qubit`` torch-backprop
  simulator.  This is CPU simulation, **not** a quantum device and **not** a
  speed measurement.  ``2**qubits`` amplitudes mean each additional qubit
  doubles the state vector; the default 4 qubits are a feasibility choice.
* Every reported value comes from an executed run; failures become warnings and
  are never imputed.  Paired comparisons are reported with Holm correction, but
  **no quantum advantage is asserted** by this module or its artifacts.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from collections.abc import Mapping as ABCMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # pragma: no cover - depends on the host environment
    import torch
    from torch import nn
    from torch.nn import functional as F

    _TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc

# fresh_protocol imports only numpy at module scope, so this is cheap and safe.
from experiments import fresh_protocol as fp

__all__ = [
    "ARCHITECTURE",
    "CLAIM_BOUNDARY",
    "CLASSICAL_RESULT_PATHS",
    "HybridModel",
    "build_hybrid_model",
    "train_hybrid_from_features",
    "evaluate_once",
    "run_hybrid_protocol",
]

# --------------------------------------------------------------------------- #
# Static documentation / provenance constants                                  #
# --------------------------------------------------------------------------- #
ARCHITECTURE: dict[str, Any] = {
    "type": "parallel-concatenation hybrid (classical compression + variational quantum head)",
    "description": (
        "classical branch Linear(feature_dim -> hidden) -> ReLU; quantum branch "
        "Linear(feature_dim -> 2**qubits) adapter -> qmlpipe.models.build_qnn with "
        "reduced qubits, amplitude embedding and trainable StronglyEntanglingLayers "
        "emitting 2 logits; final Linear(concat([classical_hidden, quantum_logits]) -> 2)."
    ),
    "classical_branch": "nn.Linear(D, hidden) -> nn.ReLU()",
    "quantum_branch": (
        "nn.Linear(D, 2**qubits) -> build_qnn(qubits, layers, amplitude embedding, "
        "strongly_entangling ansatz) -> 2 logits"
    ),
    "fusion": "nn.Linear(hidden + 2, 2) on the concatenation of both branch outputs",
    "hybrid_evidence": (
        "classical, quantum adapter, quantum circuit weights and the fusion layer all "
        "contain trainable parameters and receive gradients; the classical branch is "
        "absent from a pure QNN, so both paths are required at inference."
    ),
    "quantum_backend": "PennyLane default.qubit, interface='torch', diff_method='backprop', analytic (shots=None); CPU simulation only",
}

CLAIM_BOUNDARY = (
    "HYBRID-D01 diagnostic arm on the frozen fresh 70:20:10 grouped splits. "
    + ARCHITECTURE["description"]
    + " Both branches are trainable and both contribute to the fused classifier; "
    "this is a hybrid model, not a pure QNN and not a parameter-free quantum kernel. "
    "All quantum execution is CPU simulation (default.qubit, analytic); no quantum "
    "advantage is asserted and no speed or hardware claim is made. Paired comparisons "
    "are reported with Holm correction and decisions are computed from the data; "
    "they are evidence for or against a difference, not a licence to claim an "
    "advantage without the full matched-control and Critic gate. TEST is sealed and "
    "scored once; failures are recorded as warnings and never fabricated."
)

# Committed classical poly-2 arms from the already-run classical protocol.
# These are paths, not result numbers: values are always read from the artifact.
CLASSICAL_RESULT_PATHS: dict[str, str] = {
    "deepweeds": "results/fresh_deepweeds/metrics.json",
    "acre": "results/fresh_acre_capped/metrics.json",
}

MAX_BATCHES_PER_EPOCH = 20  # CPU-feasibility ceiling; override via cfg attribute.

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - depends on the host environment
        raise RuntimeError(
            "qmlpipe.fresh_hybrid requires PyTorch; "
            f"import failed with {type(_TORCH_IMPORT_ERROR).__name__}: {_TORCH_IMPORT_ERROR}"
        )


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping, an object or a QMLConfig-like object."""
    if isinstance(cfg, ABCMapping) and name in cfg:
        return cfg[name]
    value = getattr(cfg, name, None)
    if value is not None:
        return value
    as_dict = getattr(cfg, "as_dict", None)
    if callable(as_dict):
        try:
            return as_dict().get(name, default)
        except Exception:  # pragma: no cover - defensive
            pass
    return default


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
if nn is None:  # pragma: no cover - torch-free verifier placeholder

    class HybridModel:  # type: ignore[no-redef]
        """Placeholder that raises on use when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

else:

    class HybridModel(nn.Module):  # type: ignore[no-redef]
        """Parallel classical-compression + variational-quantum hybrid."""

        def __init__(
            self, cfg: Any, feature_dim: int, n_classes: int = 2, hidden: int | None = None
        ) -> None:
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.n_classes = int(n_classes)
            self.qubits = int(_cfg_get(cfg, "qubits", 4))
            self.hidden_dim = int(hidden if hidden is not None else _cfg_get(cfg, "hybrid_hidden", 64))

            if self.feature_dim < 1 or self.hidden_dim < 1 or self.n_classes < 2:
                raise ValueError("feature_dim, hidden and n_classes must be positive (n_classes >= 2)")

            # --- classical branch (the component a pure QNN does not have) ---
            self.classical = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(),
            )

            # --- quantum branch: trainable adapter + small variational QNN ---
            from qmlpipe.models import build_qnn  # lazy: PennyLane may be absent

            self.qnn = build_qnn(_make_quantum_cfg(cfg))
            qnn_in = int(getattr(self.qnn, "input_dim", 2 ** self.qubits))
            self.quantum_adapter = nn.Linear(self.feature_dim, qnn_in)

            # --- fusion ---
            self.classifier = nn.Linear(self.hidden_dim + self.n_classes, self.n_classes)

            self.architecture = ARCHITECTURE["description"]
            # Sealed-TEST guard: incremented only by evaluate_once().  Validation
            # logits are collected through the private helper below instead.
            self._test_eval_count = 0

        def branch_outputs(self, x: "torch.Tensor") -> dict[str, "torch.Tensor"]:
            """Return intermediate branch tensors plus the final ``logits``."""
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32)
            if x.ndim != 2 or x.shape[-1] != self.feature_dim:
                raise ValueError(
                    f"hybrid expects (B, {self.feature_dim}) cached features, got {tuple(x.shape)}"
                )
            classical = self.classical(x)
            quantum = self.qnn(self.quantum_adapter(x))
            if isinstance(quantum, (tuple, list)):  # defensive: QNN returns (B, 2)
                quantum = quantum[0]
            if quantum.ndim == 1:
                quantum = quantum.reshape(-1, self.n_classes)
            if quantum.shape[-1] != self.n_classes:
                raise ValueError(
                    f"quantum branch must emit {self.n_classes} logits, got {tuple(quantum.shape)}"
                )
            fused = torch.cat([classical, quantum], dim=1)
            return {
                "classical": classical,
                "quantum": quantum,
                "fused": fused,
                "logits": self.classifier(fused),
            }

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.branch_outputs(x)["logits"]

        def trainable_parameter_counts(self) -> dict[str, int]:
            """Per-branch trainable parameter counts (proof of a genuine hybrid)."""
            classical = sum(p.numel() for p in self.classical.parameters() if p.requires_grad)
            quantum = sum(p.numel() for p in self.quantum_adapter.parameters() if p.requires_grad)
            quantum += sum(p.numel() for p in self.qnn.parameters() if p.requires_grad)
            fusion = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
            return {
                "classical": int(classical),
                "quantum": int(quantum),
                "fusion": int(fusion),
                "total": int(classical + quantum + fusion),
            }


def _make_quantum_cfg(cfg: Any) -> Any:
    """Build a QMLConfig for ``build_qnn`` with reduced qubits."""
    from qmlpipe.config import QMLConfig

    qubits = int(_cfg_get(cfg, "qubits", 4))
    return QMLConfig(
        seed=int(_cfg_get(cfg, "seed", 42)),
        qubits=qubits,
        layers=int(_cfg_get(cfg, "layers", 2)),
        embedding=str(_cfg_get(cfg, "embedding", "amplitude")),
        ansatz=str(_cfg_get(cfg, "ansatz", "strongly_entangling")),
        device_name="default.qubit",
        shots=None,
        projection_dim=2 ** qubits,
        noise_depolarizing=None,
    )


def build_hybrid_model(cfg: Any, feature_dim: int, n_classes: int = 2) -> Any:
    """Construct the hybrid model; ``cfg`` supplies qubits/layers/seed/hidden."""
    _require_torch()
    return HybridModel(cfg, feature_dim, n_classes=n_classes)


# --------------------------------------------------------------------------- #
# Training / evaluation                                                        #
# --------------------------------------------------------------------------- #
def _grad_norm(parameters: Any) -> float:
    squares = [
        float(p.grad.detach().float().norm(2).item()) ** 2
        for p in parameters
        if getattr(p, "grad", None) is not None
    ]
    return float(np.sqrt(sum(squares))) if squares else 0.0


def _predict_logits(model: Any, X: Any, batch_size: int = 256) -> np.ndarray:
    """Internal batched forward used for TRAIN/VAL only (never touches TEST guard)."""
    _require_torch()
    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim != 2:
        raise ValueError("feature matrix must be 2-D")
    was_training = bool(getattr(model, "training", False))
    model.eval()
    chunks: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, len(X_arr), max(1, int(batch_size))):
                batch = torch.as_tensor(X_arr[start : start + max(1, int(batch_size))])
                chunks.append(model(batch).detach().cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, model.n_classes), dtype=np.float64)


def train_hybrid_from_features(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    cfg: Any,
    seed: int,
) -> tuple[Any, dict]:
    """Train the hybrid end-to-end on cached TRAIN features; VAL is reporting-only.

    Returns ``(model, history)``.  Batches are bounded by
    ``cfg.max_batches_per_epoch`` (default 20) and epochs by ``cfg.epochs`` so
    the CPU-qubit circuit stays feasible.  TEST labels are never supplied to
    this function.  Class-balanced cross entropy is available through
    ``cfg.class_weight in {'none', 'balanced'}`` (from TRAIN labels only).
    """
    _require_torch()
    torch.manual_seed(int(seed))

    X_fit = np.asarray(X_train, dtype=np.float32)
    y_fit = np.asarray(y_train, dtype=np.int64).reshape(-1)
    X_valid = np.asarray(X_val, dtype=np.float32) if X_val is not None else np.empty((0, 0), dtype=np.float32)
    y_valid = np.asarray(y_val, dtype=np.int64).reshape(-1) if y_val is not None else np.empty(0, dtype=np.int64)
    if X_fit.ndim != 2 or X_fit.shape[0] != y_fit.shape[0]:
        raise ValueError("X_train/y_train shape mismatch")
    if X_valid.size and (X_valid.ndim != 2 or X_valid.shape[1] != X_fit.shape[1]):
        raise ValueError("X_val width must match X_train")
    if X_valid.shape[0] != y_valid.shape[0]:
        raise ValueError("X_val/y_val shape mismatch")
    if np.unique(y_fit).size < 2:
        raise ValueError("TRAIN must contain both classes (single-class collapse is inadmissible)")

    model = build_hybrid_model(cfg, X_fit.shape[1], n_classes=2)
    epochs = max(1, int(_cfg_get(cfg, "epochs", 3)))
    batch_size = max(1, int(_cfg_get(cfg, "batch_size", 32)))
    max_batches = max(1, int(_cfg_get(cfg, "max_batches_per_epoch", MAX_BATCHES_PER_EPOCH)))
    lr = float(_cfg_get(cfg, "learning_rate", 1e-3))
    weight_decay = float(_cfg_get(cfg, "weight_decay", 1e-4))
    class_weight_mode = str(_cfg_get(cfg, "class_weight", "balanced")).strip().lower()
    if class_weight_mode not in {"none", "balanced"}:
        raise ValueError("class_weight must be 'none' or 'balanced'")

    from qmlpipe.losses import class_weights_from_labels

    weights = class_weights_from_labels(y_fit, n_classes=2, mode=class_weight_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    generator = torch.Generator().manual_seed(int(seed))
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    history: dict[str, Any] = {
        "seed": int(seed),
        "epochs": [],
        "n_train": int(len(y_fit)),
        "n_val": int(len(y_valid)),
        "n_features": int(X_fit.shape[1]),
        "batch_size": batch_size,
        "max_batches_per_epoch": max_batches,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "class_weight": class_weight_mode,
        "train_loss": None,
        "final_val_loss": None,
        "final_val_accuracy": None,
        "classical_grad_norm": 0.0,
        "quantum_grad_norm": 0.0,
        "parameter_counts": model.trainable_parameter_counts(),
        "test_labels_used": False,
    }

    for epoch in range(epochs):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for features, labels in loader:
            if n_batches >= max_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            if epoch == 0 and n_batches == 0:
                history["classical_grad_norm"] = _grad_norm(model.classical.parameters())
                history["quantum_grad_norm"] = _grad_norm(model.quantum_adapter.parameters()) + _grad_norm(
                    model.qnn.parameters()
                )
            optimizer.step()
            loss_sum += float(loss.detach())
            n_batches += 1
        train_loss = loss_sum / max(1, n_batches)

        val_loss, val_accuracy = None, None
        if len(y_valid):
            val_logits = _predict_logits(model, X_valid)
            yv = torch.as_tensor(y_valid)
            val_loss = float(F.cross_entropy(torch.as_tensor(val_logits), yv).item())
            val_accuracy = float((val_logits.argmax(axis=1) == y_valid).mean())
        entry = {
            "epoch": int(epoch),
            "n_batches": int(n_batches),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history["epochs"].append(entry)
        history["train_loss"] = train_loss
        history["final_val_loss"] = val_loss
        history["final_val_accuracy"] = val_accuracy

    return model, history


def evaluate_once(model: Any, X_test: Any, y_test: Any) -> dict:
    """Score the sealed TEST split exactly once and return binary metrics.

    A per-model guard raises if this function is called a second time on the
    same model instance, so TEST cannot be reused for tuning.  Metrics come from
    ``experiments.metrics.binary_metrics``; AUC uses the positive-class softmax
    probability and no threshold is tuned on TEST.
    """
    _require_torch()
    if int(getattr(model, "_test_eval_count", 0)) >= 1:
        raise RuntimeError(
            "TEST split has already been scored once on this model; evaluate_once "
            "is single-shot by contract (no test-set tuning)."
        )
    # Mark before doing work: even a failed scoring attempt consumes the seal.
    model._test_eval_count = 1

    from experiments.metrics import binary_metrics

    X_te = np.asarray(X_test, dtype=np.float32)
    y_te = np.asarray(y_test, dtype=np.int64).reshape(-1)
    if X_te.ndim != 2 or X_te.shape[0] != y_te.shape[0]:
        raise ValueError("X_test/y_test shape mismatch")
    if X_te.shape[1] != int(getattr(model, "feature_dim", X_te.shape[1])):
        raise ValueError("X_test width does not match the model's feature_dim")

    logits = _predict_logits(model, X_te)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probability = exponentials / exponentials.sum(axis=1, keepdims=True)
    predicted = logits.argmax(axis=1)
    metrics = binary_metrics(y_te, predicted, y_prob=probability[:, 1])

    return {
        "model": "hybrid",
        "seed": int(getattr(model, "_seed", -1)),
        "accuracy": float(metrics.accuracy),
        "auc": None if metrics.auc is None else float(metrics.auc),
        "f1": float(metrics.f1),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "confusion_matrix": metrics.confusion.tolist(),
        "n": int(metrics.support),
        "n_test_rows": int(metrics.support),
        "score_kind": "softmax_proba",
        "iou_class0": float(metrics.iou_class0),
        "iou_class1": float(metrics.iou_class1),
        "macro_iou": float(metrics.macro_iou),
        "test_evaluations": int(model._test_eval_count),
        "architecture": ARCHITECTURE["description"],
    }


# --------------------------------------------------------------------------- #
# Artifact helpers                                                              #
# --------------------------------------------------------------------------- #
def _normalize_seeds(seeds: int | Sequence[int]) -> list[int]:
    if isinstance(seeds, int):
        result = list(range(int(seeds)))
    else:
        result = [int(value) for value in seeds]
    return result or [42]


def _resolve_metrics_path(path: Any) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "metrics.json"
    return candidate


def _load_per_seed_accuracy(
    path: Any,
    backbone: str,
    seeds: Sequence[int],
    model_names: Sequence[str],
) -> dict[int, float]:
    """Read per-seed TEST accuracy from a fresh-protocol-style metrics artifact.

    Accepts both the ``runs`` list and the ``aggregates`` view.  ``path`` may be
    a ``metrics.json`` file or a directory containing one.  Missing/corrupt
    files raise a clear error so the caller can record a warning rather than
    invent a baseline.
    """
    resolved = _resolve_metrics_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"per-seed accuracy artifact not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"metrics artifact is not a JSON object: {resolved}")

    wanted = {str(name).strip().lower() for name in model_names}
    per_seed: dict[int, float] = {}
    runs = payload.get("runs")
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            if str(run.get("backbone")) != str(backbone):
                continue
            if str(run.get("model", "")).strip().lower() not in wanted:
                continue
            row = run.get("metrics")
            if not isinstance(row, ABCMapping) or "accuracy" not in row:
                continue
            try:
                per_seed[int(run["seed"])] = float(row["accuracy"])
            except (KeyError, TypeError, ValueError):
                continue

    if not per_seed:
        aggregates = payload.get("aggregates")
        if isinstance(aggregates, ABCMapping):
            for name in model_names:
                entry = aggregates.get(f"{backbone}/{name}")
                if not isinstance(entry, ABCMapping):
                    continue
                accuracy = entry.get("accuracy")
                if not isinstance(accuracy, ABCMapping):
                    continue
                values = accuracy.get("values") or []
                recorded_seeds = entry.get("seeds") or list(range(len(values)))
                for seed, value in zip(recorded_seeds, values):
                    per_seed[int(seed)] = float(value)

    if not per_seed:
        raise ValueError(
            f"no per-seed accuracy for backbone {backbone!r} and models "
            f"{list(model_names)} in {resolved}"
        )
    return per_seed


def _environment_pins() -> dict:
    """Record interpreter/platform and installed package versions (no network)."""
    from importlib.metadata import version

    def pinned(name: str):
        try:
            return version(name)
        except Exception:
            return None

    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "commit_hash": os.environ.get("GIT_COMMIT_HASH"),
        "packages": {
            name: pinned(name)
            for name in ("numpy", "scipy", "scikit-learn", "statsmodels", "torch", "pennylane")
        },
        "note": "versions are those importable in the executing interpreter; no installs performed",
    }


def _resolve_output_paths(output_dir: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Return fresh (metrics.json, provenance.json) paths without overwriting."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path, provenance_path = out / "metrics.json", out / "provenance.json"
    if metrics_path.exists() or provenance_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = out / f"fresh_hybrid_{stamp}"
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = out / f"fresh_hybrid_{stamp}_{suffix}"
        candidate.mkdir(parents=True, exist_ok=True)
        metrics_path, provenance_path = candidate / "metrics.json", candidate / "provenance.json"
    return metrics_path, provenance_path


def _write_json(path: Path, payload: Any) -> None:
    """Atomically write a JSON artifact (additive: only the given path is touched)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(fp._json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _protocol_cfg(seed: int, backbone: str, *, epochs: int, qubits: int, layers: int) -> Any:
    from qmlpipe.config import QMLConfig

    return QMLConfig(
        seed=int(seed),
        qubits=int(qubits),
        layers=int(layers),
        embedding="amplitude",
        ansatz="strongly_entangling",
        epochs=int(epochs),
        backbone=str(backbone),
        projection_dim=2 ** int(qubits),
        class_weight="balanced",
        device_name="default.qubit",
        shots=None,
        noise_depolarizing=None,
    )


def _comparison(
    hybrid_by_seed: dict[int, float],
    baseline: dict[int, float],
    seed_list: Sequence[int],
    label: str,
    source: Any,
    *,
    alpha: float,
    boot_B: int,
    boot_seed: int,
) -> dict:
    """One paired hybrid-vs-baseline comparison (positive difference = hybrid higher)."""
    common = [int(s) for s in seed_list if s in hybrid_by_seed and s in baseline]
    if len(common) < 2:
        raise ValueError(f"need >=2 paired seeds for comparison with {label!r}, got {len(common)}")
    a = [hybrid_by_seed[s] for s in common]
    b = [baseline[s] for s in common]
    item = fp.paired_comparison(
        a, b, alpha=alpha, seed=boot_seed, n_boot=boot_B,
        pair=(f"hybrid", label),
    )
    item.update(
        {
            "metric": "accuracy",
            "comparison_source": str(source),
            "seeds": common,
            "hybrid_values": a,
            "baseline_values": b,
            "direction": "hybrid - baseline (positive means hybrid higher)",
        }
    )
    return item


# --------------------------------------------------------------------------- #
# Orchestration                                                                 #
# --------------------------------------------------------------------------- #
def run_hybrid_protocol(
    dataset: str,
    backbone: str,
    seeds: int | Sequence[int],
    output_dir: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] | None = None,
    train_cap: int = 2000,
    epochs: int = 3,
    qubits: int = 4,
    layers: int = 2,
    *,
    qnn_results: str | os.PathLike[str] | None = None,
    classical_results: str | os.PathLike[str] | None = None,
    alpha: float = 0.05,
    boot_B: int = 2000,
    boot_seed: int = 42,
    root: str | os.PathLike[str] | None = None,
) -> dict:
    """Run the hybrid arm across seeds and write additive metrics/provenance.

    Reuses frozen cached features, caps TRAIN deterministically, leaves TEST
    sealed, aggregates with ``experiments.fresh_protocol.aggregate`` and builds
    paired Holm-corrected comparisons against the committed classical poly-2
    arm and (optionally) the peer pure-QNN artifact at ``qnn_results``.  All
    baseline numbers are read from artifacts; none are embedded here.
    """
    seed_list = _normalize_seeds(seeds)
    if int(qubits) < 1 or int(qubits) > 10:
        raise ValueError("qubits must be in [1, 10] for a CPU-feasible state vector")
    if int(layers) < 1 or int(epochs) < 1:
        raise ValueError("layers and epochs must be >= 1")

    warnings: list[dict] = []
    runs: list[dict] = []
    for seed in seed_list:
        try:
            cached = fp.cache_features(
                dataset, backbone, root=root, cache_dir=cache_dir, seed=seed
            )
        except Exception as exc:
            warnings.append(
                {
                    "stage": "feature_cache",
                    "status": "skipped_cache_failure",
                    "seed": int(seed),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        X_fit, y_fit, cap_meta = fp.cap_train(
            cached["X"]["train"], cached["y"]["train"], train_cap, seed
        )
        X_valid = cached["X"].get("val", X_fit[:0])
        y_valid = cached["y"].get("val", y_fit[:0])
        cfg = _protocol_cfg(seed, backbone, epochs=epochs, qubits=qubits, layers=layers)
        try:
            model, history = train_hybrid_from_features(
                X_fit, y_fit, X_valid, y_valid, cfg, seed
            )
            model._seed = int(seed)
            test_metrics = evaluate_once(model, cached["X"]["test"], cached["y"]["test"])
        except Exception as exc:
            warnings.append(
                {
                    "stage": "hybrid_arm",
                    "status": "skipped_failed_arm",
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
                "status": "ok",
                "train_cap": cap_meta,
                "metrics": test_metrics,
                "history": history,
            }
        )

    aggregates: dict[str, dict] = {}
    if runs:
        entry: dict[str, Any] = {
            "n_runs": len(runs),
            "seeds": [run["seed"] for run in runs],
        }
        for metric in ("accuracy", "auc", "f1", "precision", "recall"):
            values = [
                float(run["metrics"][metric])
                for run in runs
                if run["metrics"].get(metric) is not None
            ]
            if values:
                entry[metric] = fp.aggregate(
                    values, seed=boot_seed, n_boot=boot_B, name=f"{backbone}/hybrid"
                )
        aggregates[f"{backbone}/hybrid"] = entry

    hybrid_by_seed = {
        run["seed"]: float(run["metrics"]["accuracy"])
        for run in runs
        if run["metrics"].get("accuracy") is not None
    }
    family: list[dict] = []

    def add_baseline(label: str, source: Any, names: Sequence[str]) -> None:
        if source is None:
            return
        try:
            baseline = _load_per_seed_accuracy(source, backbone, seed_list, names)
            item = _comparison(
                hybrid_by_seed, baseline, seed_list, label, source,
                alpha=alpha, boot_B=boot_B, boot_seed=boot_seed,
            )
        except Exception as exc:
            warnings.append(
                {
                    "stage": "paired_comparison",
                    "status": "skipped_comparison",
                    "seed": None,
                    "comparison": label,
                    "source": str(source),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return
        family.append(item)

    classical_path = classical_results if classical_results is not None else CLASSICAL_RESULT_PATHS.get(str(dataset))
    if classical_path is not None:
        classical_path = _resolve_repo_path(classical_path)
    add_baseline(f"{backbone}/poly2_committed", classical_path, ("poly2", "poly"))
    if qnn_results is not None:
        add_baseline(
            f"{backbone}/pure_qnn",
            _resolve_repo_path(qnn_results),
            ("qnn", "pure_qnn", "fresh_qnn", "qnn_head"),
        )

    comparisons = fp.holm_adjust(family, alpha=alpha)
    for item in comparisons:
        item["backbone"] = backbone

    config = {
        "dataset": str(dataset),
        "backbone": str(backbone),
        "seeds": seed_list,
        "train_cap": int(train_cap),
        "epochs": int(epochs),
        "qubits": int(qubits),
        "layers": int(layers),
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "root": str(root) if root is not None else None,
        "alpha": float(alpha),
        "bootstrap": {"B": int(boot_B), "seed": int(boot_seed)},
        "backend": ARCHITECTURE["quantum_backend"],
        "architecture": ARCHITECTURE,
        "classical_results": str(classical_path) if classical_path is not None else None,
        "qnn_results": str(_resolve_repo_path(qnn_results)) if qnn_results is not None else None,
        "protocol": "fresh_hybrid.v1",
    }
    n_expected = len(seed_list)
    n_succeeded = len(runs)
    # Warnings include comparison-stage skips, so keep the run-accounting
    # invariant over the per-seed stages only.
    n_skipped_runs = sum(1 for w in warnings if w["stage"] != "paired_comparison")
    metrics_payload = {
        "schema": "fresh_hybrid.metrics.v1",
        "dataset": str(dataset),
        "claim_boundary": CLAIM_BOUNDARY,
        "architecture": ARCHITECTURE,
        "config": config,
        "runs": runs,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "warnings": warnings,
        "n_runs_expected": int(n_expected),
        "n_runs_succeeded": int(n_succeeded),
        "n_runs_skipped": int(n_skipped_runs),
        "accounting_invariant": bool(n_expected == n_succeeded + n_skipped_runs),
    }
    provenance_payload = {
        "schema": "fresh_hybrid.provenance.v1",
        "dataset": str(dataset),
        "claim_boundary": CLAIM_BOUNDARY,
        "architecture": ARCHITECTURE,
        "config": config,
        "environment": _environment_pins(),
        "run_manifest": [
            {
                "dataset": run["dataset"],
                "backbone": run["backbone"],
                "seed": run["seed"],
                "status": run["status"],
                "train_cap": run["train_cap"],
                "trainable_parameter_counts": run["history"].get("parameter_counts", {}),
                "grad_norms": {
                    "classical": run["history"].get("classical_grad_norm", 0.0),
                    "quantum": run["history"].get("quantum_grad_norm", 0.0),
                },
            }
            for run in runs
        ],
        "comparison_notes": {
            "primary_test": "paired_t_test (scipy.stats.ttest_rel, two-sided)",
            "secondary_test": "Wilcoxon signed-rank",
            "multiplicity": "Holm step-down across all comparisons in this artifact",
            "direction": "hybrid - baseline (positive means hybrid higher)",
            "no_advantage_claim": (
                "Decisions are statistical descriptions of the paired seed vectors. "
                "No quantum advantage is asserted from a CPU simulation or from this "
                "single hybrid arm."
            ),
            "exact_mcnemar": (
                "not computed: only one scalar accuracy per seed is retained"
            ),
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
        "architecture": ARCHITECTURE,
    }


def _resolve_repo_path(path: Any) -> Path:
    """Resolve relative artifact paths against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _REPO_ROOT / candidate
