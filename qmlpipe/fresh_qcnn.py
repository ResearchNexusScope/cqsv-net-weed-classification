"""Genuine quantum convolutional neural network arm on the fresh splits (QCNN-D01).

This module implements the QCNN of Cong et al., *Nature Physics* 15:1273 (2019),
adapted to arbitrary feature widths following Frontiers in Physics 13:1529188
(2025): alternating quantum convolution layers (local two-qubit unitaries whose
parameters are **shared** across every pair, applied in a brickwork pattern) and
quantum pooling layers (partial trace / trace-out that halves the active qubit
register).  Two arbitrary-width strategies are exposed and selectable:
``single_ancilla`` (one zero ancilla amplitude pads the feature register) and
``skip`` (an odd-width pooling step is skipped rather than padded).

Why this model is justified: ``docs/research/QC_QNN_ARCHITECTURE_RESEARCH.md``
§2.3 and §2.5 identify QCNNs as a literature-supported, barren-plateau-resistant
family for arbitrary-width image features; §2.5 / §3 require its projection
budget to be reported and paired against the committed classical controls.

Architecture (provable from :func:`circuit_spec` / :func:`pooling_plan`,
without importing PennyLane)::

    frozen cached features
      -> PCA fitted on TRAIN only to ``2**qubits - (1 if single_ancilla else 0)``
         components (amplitude budget = ``2**qubits``, one component is the
         zero ancilla in ``single_ancilla`` mode)
      -> AmplitudeEmbedding into ``n`` wires (n a power of two)
      -> [conv(even pairs) -> conv(odd pairs) -> pooling(trace out every other
          wire)] repeated until one wire remains
      -> expval(Z) on the surviving wire(s) -> small Linear head -> 2 logits

All convolution blocks use the SAME six-parameter vector (``conv_weights``),
so ``circuit_spec()['shared_conv_parameter_vectors'] == 1`` and the total
trainable convolution-parameter count is six, independent of the number of
blocks.  Each pooling stage replaces ``active`` with ``active[0::2]``: the
remaining wires are never touched again, so the channel on them is exactly the
partial trace over the discarded subsystem (a projective measurement whose
outcome is ignored).  A generic ansatz has neither property.

Protocol discipline (``docs/METHODOLOGY_V2.md`` §4-§7)
------------------------------------------------------
* features come from ``experiments.fresh_protocol.cache_features``; the PCA
  projection is fitted on TRAIN only and the identical transform is reused for
  VAL and TEST; no test information enters preprocessing;
* TRAIN is deterministically capped per seed with
  ``experiments.fresh_protocol.cap_train``; VAL is used only for checkpointing;
  TEST is scored exactly once per seed via ``experiments.metrics.binary_metrics``;
* aggregates use ``experiments.fresh_protocol.aggregate`` (n, ddof=1 SD,
  seeded bootstrap CI); paired comparisons against the committed classical
  poly-2 arms and (when present) the old amplitude-embedding QNN use
  ``paired_comparison`` + ``holm_adjust``;
* artifacts are additive and every one carries a claim boundary stating that
  no quantum advantage is asserted.

Import safety: only numpy and repository modules that are themselves
numpy-only at import time are loaded at module scope.  torch, PennyLane, scipy
and sklearn are imported lazily, so a minimal verifier can import this module
and exercise all structural / statistical helpers without the quantum stack.
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
    "ARCHITECTURE",
    "CLAIM_BOUNDARY",
    "CONV_PARAM_VECTOR_SIZE",
    "DEFAULT_CLASSICAL_PATHS",
    "DEFAULT_QNN_PATHS",
    "POOLING_MODES",
    "conv_pairs",
    "pooling_plan",
    "circuit_spec",
    "qcnn_config",
    "fit_qcnn_projection",
    "project_qcnn_features",
    "build_qcnn_model",
    "train_qcnn_from_features",
    "evaluate_once",
    "cap_train",
    "load_baseline",
    "load_baselines",
    "paired_against_baselines",
    "build_plan",
    "run_qcnn_protocol",
]

CLAIM_BOUNDARY = (
    "QCNN-D01 diagnostic arm on the fresh grouped 70:20:10 protocol. A genuine "
    "convolution+pooling QCNN (shared-parameter brickwork convolution, partial-trace "
    "pooling) is evaluated on frozen cached features after a PCA fitted on TRAIN "
    "only; the chosen register is 2**qubits amplitudes and the pooling mode is "
    "recorded in every artifact. All quantum execution is CPU analytic simulation "
    "(PennyLane default.qubit, interface='torch', diff_method='backprop', "
    "shots=None): no quantum device, speed or scaling claim is made. The default "
    "8 qubits and bounded epochs are CPU-feasibility choices. NO QUANTUM "
    "ADVANTAGE IS ASSERTED. A difference is reportable only as paired evidence "
    "after Holm correction, power >= 0.8, n >= 10, a bootstrap CI of the paired "
    "difference excluding zero, and a matched data budget. TEST is sealed and "
    "scored exactly once per seed; every value comes from an executed run and "
    "failures are warnings, never fabricated."
)

# Architecture evidence; kept as data so it can be copied into provenance.
ARCHITECTURE: dict[str, Any] = {
    "type": "quantum convolutional neural network (QCNN)",
    "reference": "Cong et al., Nature Physics 15:1273 (2019); arbitrary-width adaptation per Frontiers in Physics 13:1529188 (2025)",
    "convolution": "local 2-qubit unitaries (two Rot + one CNOT = 6 parameters) shared across every brickwork pair; even-offset then odd-offset sweeps per stage",
    "pooling": "partial trace: surviving wires are active[0::2]; discarded wires active[1::2] are never touched or measured again",
    "arbitrary_width_modes": {
        "single_ancilla": "one zero amplitude pads the feature register to the 2**qubits amplitude budget",
        "skip": "an odd-width pooling step is skipped and the register is measured as-is",
    },
    "readout": "expval(PauliZ) on surviving wire(s) -> Linear(n_readout, 2) logits",
    "backend": "PennyLane default.qubit, interface='torch', diff_method='backprop', shots=None (analytic CPU simulation)",
}

# Paths only -- no result number is hardcoded anywhere in this module.
DEFAULT_CLASSICAL_PATHS: dict[str, str] = {
    "deepweeds": "results/fresh_deepweeds/metrics.json",
    "acre": "results/fresh_acre_capped/metrics.json",
}
DEFAULT_QNN_PATHS: dict[str, str] = {
    "deepweeds": "results/fresh_qnn_deepweeds/metrics.json",
}

POOLING_MODES = ("single_ancilla", "skip")
CONV_PARAM_VECTOR_SIZE = 6  # Rot(3) on wire A + Rot(3) on wire B + CNOT
CONV_REPS = 2               # even-offset sweep then odd-offset sweep
_METRICS = ("accuracy", "auc", "f1", "precision", "recall")
_MAX_BATCHES_PER_EPOCH = 24  # CPU feasibility ceiling
_MAX_TOTAL_STEPS = 384
_EVAL_MARK = "_fresh_qcnn_test_evaluations"
_PROJ_ATTR = "_fresh_qcnn_projection"
_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Small local helpers                                                          #
# --------------------------------------------------------------------------- #
def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - host-dependent
        raise RuntimeError(
            "qmlpipe.fresh_qcnn requires PyTorch for QCNN training/evaluation; "
            f"import failed with {type(exc).__name__}: {exc}"
        ) from exc
    return torch, F


def _require_pennylane():
    try:
        import pennylane as qml
    except Exception as exc:  # pragma: no cover - host-dependent
        raise RuntimeError(
            "qmlpipe.fresh_qcnn requires PennyLane to build the QCNN circuit; "
            f"import failed with {type(exc).__name__}: {exc}"
        ) from exc
    return qml


def _normalize_seeds(seeds: int | Sequence[int]) -> list[int]:
    if isinstance(seeds, int):
        return list(range(int(seeds)))
    out = [int(value) for value in seeds]
    return out or [int(_fp.DEFAULT_SEED)]


def _default_cache_dir() -> Path:
    preferred = Path.home() / ".cache" / "qc_datasets" / "features"
    if preferred.exists():
        return preferred
    return Path(tempfile.gettempdir()) / "qc_fresh_features"


def _cache_npz_path(dataset: str, backbone: str, cache_dir) -> Path:
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
        candidate = out / f"fresh_qcnn_{stamp}"
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = out / f"fresh_qcnn_{stamp}_{suffix}"
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
        "backend": "CPU analytic simulation (default.qubit, shots=None); no quantum hardware",
    }


# --------------------------------------------------------------------------- #
# Architecture planning -- structural evidence, NO PennyLane import needed      #
# --------------------------------------------------------------------------- #
def conv_pairs(active_wires: Sequence[int], offset: int = 0) -> list[tuple[int, int]]:
    """Brickwork pair list for one convolution sweep over ``active_wires``.

    ``offset=0`` pairs positions (0,1),(2,3),...; ``offset=1`` pairs
    (1,2),(3,4),....  Pairing is by *position in the active register*, so it
    still works after pooling when wire labels are no longer contiguous.
    """
    wires = [int(w) for w in active_wires]
    start = int(offset) % 2
    return [(wires[i], wires[i + 1]) for i in range(start, len(wires) - 1, 2)]


def pooling_plan(
    n_qubits: int, mode: str = "single_ancilla", conv_reps: int = CONV_REPS
) -> dict:
    """Build the full convolution/pooling layer plan for ``n_qubits`` wires.

    Each stage records the active wires, the brickwork convolution pair lists
    (one per repetition, alternating offsets 0/1), the wires traced out by
    pooling, and the surviving wires.  ``single_ancilla`` appends one virtual
    wire (initialised to zero by the embedding padding) when a stage width is
    odd; ``skip`` skips an odd-width pooling step and measures the register as
    it is.  No quantum operation is executed here.
    """
    key = str(mode).lower()
    if key not in POOLING_MODES:
        raise ValueError(f"unknown pooling mode {mode!r}; expected one of {POOLING_MODES}")
    n = int(n_qubits)
    if n < 2:
        raise ValueError("QCNN requires at least two qubits")
    reps = max(1, int(conv_reps))
    stages: list[dict[str, Any]] = []
    active = list(range(n))
    next_wire = n  # label for a virtual single ancilla, if ever needed
    stage_index = 0
    while len(active) > 1:
        ancilla_added = False
        if len(active) % 2 == 1 and key == "single_ancilla":
            active = active + [next_wire]
            next_wire += 1
            ancilla_added = True
        pair_lists = [conv_pairs(active, offset=rep % 2) for rep in range(reps)]
        if len(active) % 2 == 1 and key == "skip":
            stages.append(
                {
                    "stage": stage_index,
                    "active_wires": list(active),
                    "conv_pairs": pair_lists,
                    "pooled_wires": [],
                    "surviving_wires": list(active),
                    "ancilla_added": False,
                    "skipped": True,
                    "terminates": True,
                }
            )
            break
        pooled = list(active[1::2])
        surviving = list(active[0::2])
        stages.append(
            {
                "stage": stage_index,
                "active_wires": list(active),
                "conv_pairs": pair_lists,
                "pooled_wires": pooled,
                "surviving_wires": surviving,
                "ancilla_added": ancilla_added,
                "skipped": False,
                "terminates": False,
            }
        )
        active = surviving
        stage_index += 1

    schedule = [n]
    for stage in stages:
        if not stage["skipped"]:
            schedule.append(len(stage["surviving_wires"]))
    return {
        "mode": key,
        "n_qubits": n,
        "conv_reps": reps,
        "stages": stages,
        "readout_wires": list(active),
        "active_qubit_schedule": schedule,
        "virtual_wire_count": next_wire - n,
    }


def circuit_spec(
    qubits: int, pooling: str = "single_ancilla", conv_reps: int = CONV_REPS
) -> dict:
    """Return JSON-native structural evidence that this is a QCNN.

    The counts here are derived purely from :func:`pooling_plan`, so they can be
    asserted in a torch/PennyLane-free verifier: one shared parameter vector,
    one CNOT (entangler) per convolution block, and a halving qubit schedule.
    """
    plan = pooling_plan(qubits, pooling, conv_reps)
    n = int(plan["n_qubits"])
    block_count = sum(
        len(pair_list) for stage in plan["stages"] for pair_list in stage["conv_pairs"]
    )
    amplitude_dim = 2 ** n
    pca_components = amplitude_dim - (1 if plan["mode"] == "single_ancilla" else 0)
    return {
        "architecture": ARCHITECTURE["type"],
        "reference": ARCHITECTURE["reference"],
        "qubits": n,
        "pooling_mode": plan["mode"],
        "conv_reps": int(plan["conv_reps"]),
        "shared_conv_parameter_vectors": 1,
        "conv_parameter_vector_size": CONV_PARAM_VECTOR_SIZE,
        "total_conv_trainable_parameters": CONV_PARAM_VECTOR_SIZE,
        "conv_block_count": int(block_count),
        "entangling_gate_count": int(block_count),  # one CNOT per local 2-qubit block
        "pooling_depth": sum(1 for stage in plan["stages"] if not stage["skipped"]),
        "active_qubit_schedule": [int(v) for v in plan["active_qubit_schedule"]],
        "readout_wires": [int(w) for w in plan["readout_wires"]],
        "readout_dim": len(plan["readout_wires"]),
        "amplitude_dim": int(amplitude_dim),
        "pca_components": int(pca_components),
        "single_ancilla_padding": bool(plan["mode"] == "single_ancilla"),
        "virtual_pooling_ancilla": bool(any(s["ancilla_added"] for s in plan["stages"])),
        "skip_pooling_used": bool(any(s["skipped"] for s in plan["stages"])),
        "backend": ARCHITECTURE["backend"],
        "is_convolution_pooling_qcnn": True,
    }


def qcnn_config(
    *,
    seed: int,
    qubits: int,
    epochs: int,
    feature_dim: int | None = None,
    batch_size: int = 32,
    class_weight: str = "balanced",
) -> QMLConfig:
    """Build a validated :class:`QMLConfig` carrying the QCNN amplitude budget.

    ``projection_dim`` is the amplitude width ``2**qubits``; ``ansatz`` and
    ``embedding`` are recorded for provenance but the circuit itself is built by
    :func:`build_qcnn_model` (it is NOT the generic strongly-entangling ansatz).
    """
    cfg = QMLConfig(
        seed=int(seed),
        qubits=int(qubits),
        layers=int(CONV_REPS),
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
        optimizer="adam",
    )
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------- #
# TRAIN-only projection                                                        #
# --------------------------------------------------------------------------- #
def fit_qcnn_projection(
    X_train, *, qubits: int, pooling: str = "single_ancilla", seed: int = 42
) -> dict:
    """Fit the TRAIN->amplitude projection on TRAIN rows only.

    PCA reduces the cached feature width to ``pca_components`` (the amplitude
    budget minus the single ancilla in ``single_ancilla`` mode).  A fixed seeded
    random projection is the documented fallback when sklearn is unavailable.
    VAL/TEST are never passed to this function; the returned meta dict is
    attached to the trained model and reused verbatim at scoring time.
    """
    arr = np.asarray(X_train, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("TRAIN features must be a non-empty 2-D matrix")
    spec = circuit_spec(qubits, pooling)
    input_dim = int(arr.shape[1])
    target = int(spec["pca_components"])
    amplitude_dim = int(spec["amplitude_dim"])
    mean = None
    matrix = None
    method = "pad"
    if input_dim >= target:
        try:
            from sklearn.decomposition import PCA

            n_comp = min(target, int(arr.shape[0]), input_dim)
            if n_comp < target:
                raise ValueError("insufficient TRAIN rows/components for PCA")
            mean = arr.mean(axis=0).astype(np.float32)
            pca = PCA(n_components=target, svd_solver="full", random_state=int(seed))
            pca.fit(arr - mean)
            matrix = np.asarray(pca.components_, dtype=np.float32).T
            method = "train_pca"
        except Exception:  # pragma: no cover - lightweight verifier fallback
            rng = np.random.default_rng(int(seed))
            matrix = (rng.standard_normal((input_dim, target)) / float(np.sqrt(target))).astype(
                np.float32
            )
            method = "seeded_random_projection"
            mean = None
    return {
        "method": method,
        "input_dim": input_dim,
        "pca_components": target,
        "amplitude_dim": amplitude_dim,
        "pooling_mode": spec["pooling_mode"],
        "single_ancilla_padding": bool(spec["single_ancilla_padding"]),
        "seed": int(seed),
        "mean": mean,
        "matrix": matrix,
        "note": (
            "Linear projection fitted on TRAIN only (PCA) or derived from the seed "
            "(documented fallback). The identical transform is applied to VAL/TEST; "
            "no VAL/TEST information enters preprocessing. In single_ancilla mode "
            "one zero amplitude pads the register to the 2**qubits budget."
        ),
    }


def project_qcnn_features(X, projection: Mapping[str, Any]) -> np.ndarray:
    """Apply a stored projection meta dict, returning L2-normalised amplitudes."""
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"features must be a 2-D matrix, got shape {arr.shape}")
    input_dim = int(projection["input_dim"])
    target = int(projection["pca_components"])
    amplitude_dim = int(projection["amplitude_dim"])
    if arr.shape[1] != input_dim:
        raise ValueError(
            f"expected {input_dim} raw cached features, got {arr.shape[1]}; "
            "the projection is bound to the TRAIN feature width"
        )
    mean = projection.get("mean")
    matrix = projection.get("matrix")
    centered = arr - np.asarray(mean, dtype=np.float32) if mean is not None else arr
    if matrix is not None:
        out = centered @ np.asarray(matrix, dtype=np.float32)
    else:
        out = np.zeros((arr.shape[0], target), dtype=np.float32)
        take = min(target, arr.shape[1])
        out[:, :take] = centered[:, :take]
    if out.shape[1] < amplitude_dim:
        padded = np.zeros((arr.shape[0], amplitude_dim), dtype=np.float32)
        padded[:, : out.shape[1]] = out
        out = padded  # the single ancilla amplitude is exactly zero
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norms, _EPS)).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# QCNN model                                                                   #
# --------------------------------------------------------------------------- #
def build_qcnn_model(
    *, qubits: int, pooling: str = "single_ancilla", seed: int = 42,
    conv_reps: int = CONV_REPS,
):
    """Build the trainable convolution+pooling QCNN returning 2 logits.

    The QNode has exactly ONE trainable argument, ``conv_weights`` (six values),
    handed to a PennyLane ``TorchLayer``.  Every local two-qubit convolution
    block in every stage uses that same vector, so parameter sharing is
    structural rather than a documentation claim.
    """
    torch, _F = _require_torch()
    qml = _require_pennylane()
    from qmlpipe.heads import build_classification_head

    spec = circuit_spec(qubits, pooling, conv_reps)
    n = int(spec["qubits"])
    plan = pooling_plan(n, pooling, conv_reps)
    max_wire = max([w for stage in plan["stages"] for w in stage["active_wires"]] or [0])
    if max_wire >= n:
        raise ValueError(
            "single-ancilla padding needs a virtual wire outside the configured "
            "register; choose a power-of-two qubit count so the feature register "
            "fits without an extra physical wire"
        )
    amplitude_dim = int(spec["amplitude_dim"])
    readout = [int(w) for w in spec["readout_wires"]]

    dev = qml.device("default.qubit", wires=n, shots=None)  # analytic CPU simulation

    def _conv_block(wire_a: int, wire_b: int, params) -> None:
        # Generic local 2-qubit unitary: Rot⊗Rot followed by an entangling CNOT.
        qml.Rot(params[0], params[1], params[2], wires=wire_a)
        qml.Rot(params[3], params[4], params[5], wires=wire_b)
        qml.CNOT(wires=[wire_a, wire_b])

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qcnn_qnode(inputs, conv_weights):
        qml.AmplitudeEmbedding(
            features=inputs, wires=range(n), normalize=True, pad_with=0.0
        )
        for stage in plan["stages"]:
            for pair_list in stage["conv_pairs"]:
                for wire_a, wire_b in pair_list:
                    _conv_block(wire_a, wire_b, conv_weights)
            # Pooling is the partial trace over ``stage['pooled_wires']``: those
            # wires are never operated on or measured again, and the final
            # observables live only on the surviving register, so the channel on
            # the survivors is exactly Tr_discarded[rho] (measurement with an
            # ignored outcome). No separate (shot-based) measurement is needed.
        return [qml.expval(qml.PauliZ(wire)) for wire in readout]

    class _SharedConvQCNN(torch.nn.Module):
        """Amplitude embedding -> shared-parameter QCNN -> PauliZ -> linear head."""

        def __init__(self) -> None:
            super().__init__()
            self.spec = dict(spec)
            self.amplitude_dim = amplitude_dim
            self.n_qubits = n
            # Seed TorchLayer's numpy draw locally and restore caller state.
            rng_state = np.random.get_state()
            np.random.seed(int(seed))
            try:
                self.layer = qml.qnn.TorchLayer(qcnn_qnode, {"conv_weights": (CONV_PARAM_VECTOR_SIZE,)})
            finally:
                np.random.set_state(rng_state)
            self.head = build_classification_head(len(readout), n_classes=2)

        def forward(self, x):
            if x.ndim != 2 or x.shape[-1] != amplitude_dim:
                raise ValueError(
                    f"QCNN expects (B, {amplitude_dim}) amplitudes, got {tuple(x.shape)}"
                )
            out = self.layer(x.cpu())
            out = torch.as_tensor(out).to(dtype=x.dtype, device=x.device)
            return self.head(out)

    return _SharedConvQCNN()


# --------------------------------------------------------------------------- #
# Training -- bounded, VAL-only checkpointing                                  #
# --------------------------------------------------------------------------- #
def _cap_train_local(X, y, cap: int, seed: int):
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
        for index in order[: int((cap_int - k) - floors.sum())]:
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
    meta.update(
        {
            "train_size_after": int(len(selected)),
            "cap_applied": True,
            "note": f"stratified TRAIN subsample to at most {cap_int} rows (seed {int(seed)}) [local fallback]",
        }
    )
    return np.asarray(X)[selected], y_arr[selected], meta


def cap_train(X, y, cap: int, seed: int):
    """Use the committed ``experiments.fresh_protocol.cap_train`` (local fallback)."""
    fn = getattr(_fp, "cap_train", None)
    if callable(fn):
        return fn(X, y, cap, seed)
    return _cap_train_local(X, y, cap, seed)


def train_qcnn_from_features(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    qubits: int = 8,
    pooling: str = "single_ancilla",
    epochs: int = 3,
    seed: int = 42,
    conv_reps: int = CONV_REPS,
    class_weight: str = "balanced",
    batch_size: int = 32,
    learning_rate: float = 1e-3,
):
    """Train one bounded QCNN on TRAIN/VAL cached features.

    Returns ``(model, history)``.  The TRAIN-fitted projection is attached to
    ``model`` so :func:`evaluate_once` applies the identical transform to TEST.
    No TEST label or feature is read here.
    """
    torch, _F = _require_torch()
    from qmlpipe.losses import class_weights_from_labels, weighted_cross_entropy
    from qmlpipe.train import seed_everything

    projection = fit_qcnn_projection(X_train, qubits=qubits, pooling=pooling, seed=seed)
    spec = circuit_spec(qubits, pooling, conv_reps)
    X_fit = project_qcnn_features(X_train, projection)
    X_val_proj = project_qcnn_features(X_val, projection)
    y_fit = np.asarray(y_train, dtype=np.int64).reshape(-1)
    y_val_arr = np.asarray(y_val, dtype=np.int64).reshape(-1)
    if len(y_fit) != len(X_fit) or len(y_val_arr) != len(X_val_proj):
        raise ValueError("label/feature counts disagree after projection")

    seed_everything(int(seed))
    model = build_qcnn_model(
        qubits=qubits, pooling=pooling, seed=seed, conv_reps=conv_reps
    )
    setattr(model, _PROJ_ATTR, {key: value for key, value in projection.items()})

    weights = class_weights_from_labels(y_fit, n_classes=2, mode=str(class_weight))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=0.0
    )
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_fit), torch.from_numpy(y_fit)
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(batch_size), shuffle=True, generator=generator
    )
    X_val_t = torch.from_numpy(X_val_proj)

    def _val_loss() -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(X_val_t), int(batch_size)):
                xb = X_val_t[start : start + int(batch_size)]
                yb = torch.from_numpy(y_val_arr[start : start + len(xb)])
                total += float(
                    weighted_cross_entropy(
                        model(xb), yb, weight=weights, reduction="sum"
                    ).item()
                )
        return total / max(1, len(y_val_arr))

    initial = [p.detach().clone() for p in model.parameters()]
    first_grad_norm: float | None = None
    total_steps = 0
    best_val = float("inf")
    best_state = None
    epoch_log: list[dict] = []
    for epoch in range(int(epochs)):
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
                    float(torch.linalg.vector_norm(torch.cat(grads)).item())
                    if grads
                    else 0.0
                )
            optimizer.step()
            total_steps += 1
            n_batches += 1
            loss_sum += float(loss.item()) * len(yb)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            seen += len(yb)
            if not np.isfinite(loss_sum):
                raise RuntimeError("non-finite QCNN training loss; refusing to report a result")
        val_loss = _val_loss()
        epoch_log.append(
            {
                "epoch": int(epoch),
                "train_loss": float(loss_sum / max(1, seen)),
                "train_acc": float(correct / max(1, seen)),
                "val_loss": float(val_loss),
                "n_batches": int(n_batches),
            }
        )
        if val_loss < best_val:  # VAL only: checkpoint selection
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
    history: dict[str, Any] = {
        "seed": int(seed),
        "architecture": "qcnn",
        "model": "qcnn",
        "qubits": int(spec["qubits"]),
        "pooling_mode": spec["pooling_mode"],
        "epochs_configured": int(epochs),
        "max_batches_per_epoch": _MAX_BATCHES_PER_EPOCH,
        "max_total_steps": _MAX_TOTAL_STEPS,
        "total_steps": int(total_steps),
        "best_val_loss": float(best_val) if np.isfinite(best_val) else None,
        "projection": {key: value for key, value in projection.items() if key != "matrix"},
        "circuit_spec": spec,
        "grad_norm_first_batch": first_grad_norm,
        "param_delta": param_delta,
        "class_weight_mode": str(class_weight),
        "optimizer": "adam",
        "epoch_log": epoch_log,
        "test_evaluations": 0,
    }
    return model, history


# --------------------------------------------------------------------------- #
# Sealed TEST evaluation                                                       #
# --------------------------------------------------------------------------- #
def evaluate_once(model, X_test, y_test) -> dict:
    """Score the sealed TEST exactly once via ``experiments.metrics.binary_metrics``.

    A one-shot mark on ``model`` makes a second call fail loudly, so no
    operating point can be tuned on the test split.
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
    X_score = (
        project_qcnn_features(X_test, projection)
        if projection is not None
        else np.asarray(X_test, dtype=np.float32)
    )
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
        "model": "qcnn",
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
# Committed baselines + paired statistics                                      #
# --------------------------------------------------------------------------- #
def load_baseline(
    path, backbone: str, *, model_names: Sequence[str], results_root=None
) -> dict[int, float]:
    """Read per-seed accuracies from a committed metrics artifact (no numbers here).

    ``path`` may be absolute or repository-relative.  Missing files yield an
    empty mapping so optional baselines (e.g. the old QNN on acre) degrade
    honestly without fabricating a comparison.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (Path(results_root) if results_root is not None else _REPO_ROOT) / candidate
    if not candidate.exists():
        return {}
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    wanted = {str(name).lower() for name in model_names}
    out: dict[int, float] = {}
    for run in payload.get("runs", []):
        if str(run.get("model", "")).lower() not in wanted:
            continue
        if str(run.get("backbone", "")).lower() != str(backbone).lower():
            continue
        if str(run.get("status", "ok")) != "ok":
            continue
        accuracy = run.get("metrics", {}).get("accuracy", run.get("accuracy"))
        if accuracy is None:
            continue
        try:
            out[int(run["seed"])] = float(accuracy)
        except (KeyError, TypeError, ValueError):  # pragma: no cover - malformed artifact
            continue
    return out


def load_baselines(
    dataset: str,
    backbone: str,
    *,
    results_root=None,
    qnn_results=None,
) -> dict[str, dict[int, float]]:
    """Load the classical poly-2 arm and (when available) the old amplitude QNN.

    ``qnn_results`` optionally overrides the old-QNN artifact path.  Only paths
    and seed-indexed values from the artifacts are used; no baseline number is
    embedded in this module.
    """
    out: dict[str, dict[int, float]] = {}
    classical = DEFAULT_CLASSICAL_PATHS.get(str(dataset).lower())
    if classical is not None:
        values = load_baseline(
            classical, backbone, model_names=("poly2", "poly"), results_root=results_root
        )
        if values:
            out["poly2"] = values
    qnn_path = qnn_results if qnn_results is not None else DEFAULT_QNN_PATHS.get(str(dataset).lower())
    if qnn_path is not None:
        values = load_baseline(
            qnn_path, backbone, model_names=("qnn",), results_root=results_root
        )
        if values:
            out["qnn_amplitude"] = values
    return out


def paired_against_baselines(
    qcnn_by_seed: Mapping[int, float],
    baselines: Mapping[str, Mapping[int, float]],
    *,
    alpha: float = _fp.ALPHA,
    seed: int = _fp.BOOT_SEED,
    n_boot: int = _fp.BOOT_B,
) -> list[dict]:
    """Pair QCNN accuracy against each baseline per seed and Holm-adjust.

    ``qnn_win_supported`` is a flag on the evidence, never a claim: it can only
    be true when the Holm decision is ``significant`` with a positive mean
    difference (which itself requires n >= 10, power >= 0.8 and a CI excluding
    zero).  The unconditional claim boundary still forbids an advantage claim.
    """
    family: list[dict] = []
    for name, baseline in baselines.items():
        common = sorted(
            set(int(k) for k in qcnn_by_seed) & set(int(k) for k in baseline)
        )
        if len(common) < 2:
            continue
        a = [float(qcnn_by_seed[s]) for s in common]
        b = [float(baseline[s]) for s in common]
        item = _fp.paired_comparison(
            a,
            b,
            alpha=float(alpha),
            seed=int(seed),
            n_boot=int(n_boot),
            pair=("qcnn", str(name)),
        )
        item["baseline"] = str(name)
        item["seeds"] = [int(s) for s in common]
        item["metric"] = "accuracy"
        item["mean_qcnn"] = float(np.mean(a))
        item["mean_baseline"] = float(np.mean(b))
        item["mean_difference"] = float(np.mean(np.asarray(a) - np.asarray(b)))
        family.append(item)
    for item in _fp.holm_adjust(family, alpha=float(alpha)):
        item["qnn_win_supported"] = bool(
            item.get("decision") == "significant"
            and item.get("mean_difference") is not None
            and float(item["mean_difference"]) > 0.0
        )
    return family


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
    qubits: int = 8,
    pooling: str = "single_ancilla",
    *,
    conv_reps: int = CONV_REPS,
    class_weight: str = "balanced",
    qnn_results=None,
) -> dict:
    """Return a write-free plan (used by the CLI ``--dry-run``)."""
    key = str(pooling).lower()
    if key not in POOLING_MODES:
        raise ValueError(f"unknown pooling mode {pooling!r}; expected {POOLING_MODES}")
    spec = circuit_spec(qubits, key, conv_reps)
    return {
        "dry_run": True,
        "dataset": str(dataset),
        "backbone": str(backbone),
        "seeds": _normalize_seeds(seeds),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir) if cache_dir is not None else str(_default_cache_dir()),
        "train_cap": int(train_cap),
        "epochs": int(epochs),
        "qubits": int(qubits),
        "pooling": key,
        "conv_reps": int(conv_reps),
        "class_weight": str(class_weight),
        "architecture": spec,
        "runs_planned": len(_normalize_seeds(seeds)),
        "max_batches_per_epoch": _MAX_BATCHES_PER_EPOCH,
        "max_total_steps": _MAX_TOTAL_STEPS,
        "classical_control": DEFAULT_CLASSICAL_PATHS.get(str(dataset).lower()),
        "optional_qnn_control": qnn_results or DEFAULT_QNN_PATHS.get(str(dataset).lower()),
        "claim_boundary": CLAIM_BOUNDARY,
        "cpu_note": "8 qubits by default is a CPU feasibility choice; analytic simulation only, no quantum advantage asserted.",
    }


def run_qcnn_protocol(
    dataset: str,
    backbone: str,
    seeds,
    output_dir,
    cache_dir=None,
    train_cap: int = 2000,
    epochs: int = 3,
    qubits: int = 8,
    pooling: str = "single_ancilla",
    *,
    root=None,
    conv_reps: int = CONV_REPS,
    class_weight: str = "balanced",
    alpha: float = _fp.ALPHA,
    boot_B: int = _fp.BOOT_B,
    boot_seed: int = _fp.BOOT_SEED,
    results_root=None,
    qnn_results=None,
) -> dict:
    """Run one bounded QCNN per seed on cached features and write additive JSON.

    Reuses ``experiments.fresh_protocol.cache_features`` so every seed sees the
    same group-disjoint TRAIN/VAL/TEST split as the committed classical arm.
    Returns the familiar shape: metrics, provenance, paths, warnings, boundary.
    """
    seed_list = _normalize_seeds(seeds)
    key = str(pooling).lower()
    if key not in POOLING_MODES:
        raise ValueError(f"unknown pooling mode {pooling!r}; expected {POOLING_MODES}")
    cache_dir = cache_dir if cache_dir is not None else _default_cache_dir()
    expected_cache = _cache_npz_path(dataset, backbone, cache_dir)
    if not expected_cache.exists():
        raise FileNotFoundError(
            f"frozen feature cache not found: {expected_cache}. "
            "Run the classical fresh-protocol featurisation first; this arm never "
            "extracts images or fabricates features."
        )

    plan = build_plan(
        dataset, backbone, seed_list, output_dir, cache_dir, train_cap,
        epochs, qubits, key, conv_reps=conv_reps, class_weight=class_weight,
        qnn_results=qnn_results,
    )
    spec = plan["architecture"]
    warnings: list[dict] = []
    runs: list[dict] = []
    feature_dim: int | None = None
    cache_hit: bool | None = None

    for seed in seed_list:
        try:
            cached = _fp.cache_features(
                dataset, backbone, root=root, cache_dir=cache_dir, seed=int(seed)
            )
            X_fit, y_fit, cap_meta = cap_train(
                cached["X"]["train"], cached["y"]["train"], train_cap, int(seed)
            )
            feature_dim = int(np.asarray(cached["X"]["train"]).shape[1])
            cache_hit = bool(cached.get("cache_hit")) if cache_hit is None else cache_hit
            model, history = train_qcnn_from_features(
                X_fit,
                y_fit,
                cached["X"]["val"],
                cached["y"]["val"],
                qubits=qubits,
                pooling=key,
                epochs=epochs,
                seed=int(seed),
                conv_reps=conv_reps,
                class_weight=class_weight,
            )
            result = evaluate_once(model, cached["X"]["test"], cached["y"]["test"])
        except Exception as exc:  # record, never fabricate
            warnings.append(
                {
                    "stage": "qcnn_arm",
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
                "model": "qcnn",
                "status": "ok",
                "train_cap": cap_meta,
                "metrics": result,
                "history": history,
                "cache_path": cached.get("cache_path"),
                "group_disjoint": bool(cached.get("provenance", {}).get("group_disjoint")),
            }
        )

    aggregates: dict[str, Any] = {
        "qcnn": {"n_runs": len(runs), "seeds": [r["seed"] for r in runs]}
    }
    for metric in _METRICS:
        values = [
            float(run["metrics"][metric])
            for run in runs
            if run["metrics"].get(metric) is not None
            and np.isfinite(float(run["metrics"][metric]))
        ]
        if values:
            aggregates["qcnn"][metric] = _fp.aggregate(
                values, seed=boot_seed, n_boot=boot_B, name="qcnn"
            )

    qcnn_by_seed = {
        int(r["seed"]): float(r["metrics"]["accuracy"])
        for r in runs
        if r["metrics"].get("accuracy") is not None
    }
    baselines = load_baselines(
        dataset, backbone, results_root=results_root, qnn_results=qnn_results
    )
    comparisons = paired_against_baselines(
        qcnn_by_seed, baselines, alpha=alpha, seed=boot_seed, n_boot=boot_B
    )

    n_succeeded, n_skipped = len(runs), len(warnings)
    config = {k: v for k, v in plan.items() if k != "dry_run"}
    metrics_payload = {
        "schema": "fresh_qcnn.metrics.v1",
        "dataset": str(dataset),
        "backbone": str(backbone),
        "claim_boundary": CLAIM_BOUNDARY,
        "config": config,
        "architecture": spec,
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
        "schema": "fresh_qcnn.provenance.v1",
        "dataset": str(dataset),
        "backbone": str(backbone),
        "claim_boundary": CLAIM_BOUNDARY,
        "config": config,
        "architecture": spec,
        "environment": _environment_pins(),
        "feature_cache": {
            "cache_path": str(expected_cache),
            "cache_hit": cache_hit,
            "feature_dim": feature_dim,
            "frozen": True,
        },
        "projection_budget": {
            "amplitude_dim": spec["amplitude_dim"],
            "pca_components": spec["pca_components"],
            "single_ancilla_padding": spec["single_ancilla_padding"],
            "fitted_on": "train_only",
            "applied_to": ["train", "val", "test"],
            "note": "PCA (with documented seeded fallback) fitted on TRAIN only; identical transform at VAL/TEST.",
        },
        "depth": {
            "pooling_depth": spec["pooling_depth"],
            "conv_reps": spec["conv_reps"],
            "conv_block_count": spec["conv_block_count"],
            "active_qubit_schedule": spec["active_qubit_schedule"],
        },
        "run_manifest": [
            {
                "dataset": str(dataset),
                "backbone": str(backbone),
                "seed": int(r["seed"]),
                "model": "qcnn",
                "status": "ok",
                "train_cap": r["train_cap"],
                "projection": r["history"]["projection"],
                "grad_norm_first_batch": r["history"].get("grad_norm_first_batch"),
                "param_delta": r["history"].get("param_delta"),
                "entangling_gate_count": r["history"]["circuit_spec"]["entangling_gate_count"],
                "shared_conv_parameter_vectors": r["history"]["circuit_spec"]["shared_conv_parameter_vectors"],
            }
            for r in runs
        ]
        + warnings,
        "comparison_notes": {
            "baselines": {
                name: {"path": str(plan.get("classical_control") if name == "poly2" else plan.get("optional_qnn_control")),
                       "seeds_matched": [int(s) for s in baseline]}
                for name, baseline in baselines.items()
            },
            "n_budget_matched": False,
            "note": (
                "The committed classical arm used its own TRAIN budget while this "
                "arm caps TRAIN at the configured value; a paired win is only "
                "admissible after budget matching in addition to the statistical gates."
            ),
            "test_scoring": "exactly one forward pass per seed via evaluate_once",
            "backend": ARCHITECTURE["backend"],
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
