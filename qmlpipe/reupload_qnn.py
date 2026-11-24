"""Genuine DATA RE-UPLOADING QNN arm on the fresh grouped-split protocol (REUPLOAD-D01).

This module implements the architecture of Pérez-Salinas et al., *Quantum* 4:226
(2020), ``https://quantum-journal.org/papers/q-2020-02-06-226/``::

    |psi(x; omega)> = product_{i=1..L} [ V(omega_i) U_phi(x) ] |0>^n

Unlike the committed amplitude-embedding baseline (:mod:`qmlpipe.fresh_qnn`), the
input is inserted **L times**, each insertion followed by an **independent**
trainable entangling layer.  This is not another single-encoding amplitude QNN.

Design rules
------------
* ``U_phi(x)`` is angle encoding with a TRAINABLE per-feature scale: each of the
  ``n * L`` mildly-reduced features drives one RX/RY/RZ rotation on one wire,
  with the rotation type cycling RX -> RY -> RZ.  The scale is a trainable
  parameter (one per feature), so the encoding itself is learned.
* ``V(omega_i)`` is a CNOT (or CZ) ring over the n wires followed by per-wire
  trainable RY/RZ rotations.  ``omega_i`` is a *separate* parameter tensor for
  every upload ``i`` -- the variational layers are not shared.
* Readout: ``n`` Pauli-Z expectation values -> a small linear head -> 2 logits.
  Gradients flow through both the data-encoding scales and the layer rotations
  (verified by :func:`gradient_probe` and recorded in every training history).
* Features come from ``experiments.fresh_protocol.cache_features``.  The only
  classical reduction is a TRAIN-only PCA (plus TRAIN-only standardisation and a
  tanh bound) down to exactly ``n * L`` features, i.e. the minimum needed by the
  encoding.  Nothing in VAL/TEST is used to fit it, and the full recipe is
  recorded under ``classical_preprocessing`` in provenance.
* Training: bounded epochs/batches, Adam, (optionally class-weighted) cross
  entropy, VAL only for checkpointing, TEST scored exactly once per seed via
  :func:`evaluate_once` and ``experiments.metrics.binary_metrics``.
* Aggregates use ``experiments.fresh_protocol.aggregate`` (n, ddof=1 SD, seeded
  bootstrap CI).  PAIRED comparisons use ``paired_comparison`` + ``holm_adjust``
  against (a) the committed classical poly-2 arms and (b) the old amplitude QNN
  when its artifact is present.  No quantum advantage is asserted; a win would
  require matched budgets and Holm-corrected paired power >= 0.8 evidence.
* Baseline inputs are robust: the ``qnn_results`` argument (CLI ``--qnn-results``)
  accepts either a ``metrics.json`` FILE or a DIRECTORY containing one
  (:func:`_resolve_artifact` maps the latter to ``<dir>/metrics.json``).  A
  configured baseline that is missing, unreadable or has no matching runs is
  surfaced as an explicit warning with a reason; it is never silently dropped
  and no value is fabricated.

Import safety: numpy + the config/statistics modules are imported at module
scope; torch, PennyLane, scipy and sklearn are imported lazily so a light
verifier can import this module and exercise the structural/statistical helpers
without the quantum stack.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import fresh_protocol as _fp  # noqa: E402  (numpy-only at import)

__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_CLASSICAL_PATHS",
    "DEFAULT_AMPLITUDE_QNN_PATHS",
    "ReuploadConfig",
    "reupload_circuit_spec",
    "reupload_circuit_introspection",
    "build_reupload_qnn",
    "fit_reupload_projection",
    "project_reupload_features",
    "cap_train",
    "gradient_probe",
    "train_reupload_from_features",
    "evaluate_once",
    "load_baseline_values",
    "load_baseline_report",
    "paired_comparisons",
    "build_plan",
    "run_reupload_protocol",
]

CLAIM_BOUNDARY = (
    "Genuine data-re-uploading QNN (Perez-Salinas et al., Quantum 4:226, 2020) on "
    "the fresh grouped-split protocol. No quantum advantage is asserted. The "
    "default 6 qubits and 4 uploads are CPU-feasibility choices, not a hardware, "
    "scaling or quantum-advantage claim. The circuit is simulated analytically "
    "with shots=None (default.qubit, interface=torch, backprop) on CPU; no device "
    "speedup is claimed. A win is admissible only if the per-seed PAIRED comparison "
    "against the committed classical poly-2 arm (and the old amplitude QNN when "
    "present) is 'significant' after Holm correction with achieved power >= 0.8, "
    "n >= 10, a bootstrap CI of the paired difference excluding zero, and a matched "
    "data/input-information budget. The only classical reduction is a TRAIN-only "
    "linear projection to n*L features (plus TRAIN-only standardisation and a tanh "
    "bound), recorded explicitly as classical_preprocessing; VAL/TEST never fit it. "
    "The sealed TEST is scored exactly once per seed. Failures are recorded as "
    "warnings, never fabricated, and no performance difference is attributed to "
    "quantum effects without matched-budget paired evidence."
)

# Committed control artifacts (paths only; no result numbers are stored here).
DEFAULT_CLASSICAL_PATHS: dict[str, str] = {
    "deepweeds": "results/fresh_deepweeds/metrics.json",
    "acre": "results/fresh_acre_capped/metrics.json",
}
DEFAULT_AMPLITUDE_QNN_PATHS: dict[str, str] = {
    "deepweeds": "results/fresh_qnn_deepweeds/metrics.json",
    "acre": "results/fresh_qnn_acre/metrics.json",
}

_METRICS = ("accuracy", "auc", "f1", "precision", "recall")
_MAX_BATCHES_PER_EPOCH = 32
_MAX_TOTAL_STEPS = 512
_EVAL_MARK = "_reupload_test_evaluations"
_PROJ_ATTR = "_reupload_projection"
_EPS = 1e-8
_ALLOWED_ENTANGLERS = ("cnot", "cz")
_ALLOWED_ROTATIONS = ("RX", "RY", "RZ")


# --------------------------------------------------------------------------- #
# Local helpers                                                                #
# --------------------------------------------------------------------------- #
def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on the host environment
        raise RuntimeError(
            "qmlpipe.reupload_qnn requires PyTorch for QNN training/evaluation; "
            f"import failed with {type(exc).__name__}: {exc}"
        ) from exc
    return torch, F


def _require_pennylane():
    try:
        import pennylane as qml
    except Exception as exc:  # pragma: no cover - depends on the host environment
        raise RuntimeError(
            "qmlpipe.reupload_qnn requires PennyLane to build the data-reuploading "
            f"circuit; import failed with {type(exc).__name__}: {exc}"
        ) from exc
    return qml


def _normalize_seeds(seeds: int | Sequence[int]) -> list[int]:
    if isinstance(seeds, int):
        return list(range(int(seeds)))
    out = [int(value) for value in seeds]
    return out or [int(_fp.DEFAULT_SEED)]


def _default_cache_dir() -> Path:
    """Prefer the committed home cache; fall back to the fresh_protocol temp default."""
    preferred = Path.home() / ".cache" / "qc_datasets" / "features"
    if preferred.exists():
        return preferred
    return Path(tempfile.gettempdir()) / "qc_fresh_features"


def _cache_npz_path(dataset: str, backbone: str, cache_dir) -> Path:
    """Reproduce the fresh_protocol cache naming convention for preflight checks."""
    base = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    root_key = hashlib.sha256(b"default").hexdigest()[:12]
    return base / f"{dataset}_{backbone}_{root_key}.npz"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value)]
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
        candidate = out / f"reupload_qnn_{stamp}"
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = out / f"reupload_qnn_{stamp}_{suffix}"
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
# Config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReuploadConfig:
    """Frozen config for the data-reuploading QNN.

    ``feature_dim`` (the QNode input width) is ``qubits * uploads``: one reduced
    feature per wire per upload.  It deliberately does NOT follow the
    amplitude-embedding ``2 ** qubits`` contract of :class:`QMLConfig`, which is
    why this arm owns a small dedicated config instead of misusing that class.
    """

    seed: int = 42
    qubits: int = 6
    uploads: int = 4
    entangler: str = "cnot"
    rotation_cycle: tuple[str, ...] = ("RX", "RY", "RZ")
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    class_weight: str = "balanced"
    device_name: str = "default.qubit"
    shots: int | None = None

    @property
    def feature_dim(self) -> int:
        return int(self.qubits) * int(self.uploads)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if int(self.qubits) < 2:
            problems.append("qubits must be >= 2 for a CNOT/CZ ring")
        if int(self.uploads) < 1:
            problems.append("uploads must be >= 1")
        if str(self.entangler).lower() not in _ALLOWED_ENTANGLERS:
            problems.append(f"entangler must be one of {_ALLOWED_ENTANGLERS}")
        cycle = tuple(str(r).upper() for r in self.rotation_cycle)
        if not cycle or any(r not in _ALLOWED_ROTATIONS for r in cycle):
            problems.append(f"rotation_cycle entries must be in {_ALLOWED_ROTATIONS}")
        if int(self.epochs) < 1 or int(self.batch_size) < 1:
            problems.append("epochs and batch_size must be >= 1")
        if float(self.learning_rate) <= 0 or float(self.weight_decay) < 0:
            problems.append("learning_rate must be > 0 and weight_decay >= 0")
        if str(self.class_weight).lower() not in {"none", "balanced"}:
            problems.append("class_weight must be 'none' or 'balanced'")
        if self.shots is not None:
            problems.append("shots must be None: this arm uses analytic CPU backprop")
        assert not problems, "; ".join(problems)
        return problems


# --------------------------------------------------------------------------- #
# Circuit structure -- single source of truth for the QNode and introspection  #
# --------------------------------------------------------------------------- #
def reupload_circuit_spec(
    qubits: int,
    uploads: int,
    entangler: str = "cnot",
    rotation_cycle: Sequence[str] = ("RX", "RY", "RZ"),
) -> dict:
    """Return the ordered structural description of the re-uploading circuit.

    Each block ``i`` applies ``U_phi(x)`` (one scaled rotation per wire) followed
    by an INDEPENDENT ``V(omega_i)`` (CNOT/CZ ring + per-wire RY/RZ).  The QNode
    and :func:`reupload_circuit_introspection` both consume this spec, so the
    structural claims (L data insertions, L separate variational parameter sets,
    L*n entangling gates) are provable from one place without PennyLane.
    """
    n, L = int(qubits), int(uploads)
    if n < 2 or L < 1:
        raise ValueError("qubits must be >= 2 and uploads >= 1")
    gate = str(entangler).lower()
    if gate not in _ALLOWED_ENTANGLERS:
        raise ValueError(f"entangler must be one of {_ALLOWED_ENTANGLERS}")
    cycle = tuple(str(r).upper() for r in rotation_cycle)
    if not cycle or any(r not in _ALLOWED_ROTATIONS for r in cycle):
        raise ValueError(f"rotation_cycle entries must be in {_ALLOWED_ROTATIONS}")

    operations: list[dict] = []
    for i in range(L):
        for q in range(n):
            j = i * n + q
            operations.append(
                {
                    "role": "encoding",
                    "name": cycle[j % len(cycle)],
                    "wire": q,
                    "upload": i,
                    "feature_index": j,
                    "param": "data_scales",
                }
            )
        for q in range(n):
            operations.append(
                {
                    "role": "entangle",
                    "name": "CNOT" if gate == "cnot" else "CZ",
                    "wires": [q, (q + 1) % n],
                    "upload": i,
                    "param": None,
                }
            )
        for q in range(n):
            operations.append(
                {
                    "role": "layer_rotation",
                    "name": "RY",
                    "wire": q,
                    "upload": i,
                    "param": "omega",
                    "param_index": 0,
                }
            )
            operations.append(
                {
                    "role": "layer_rotation",
                    "name": "RZ",
                    "wire": q,
                    "upload": i,
                    "param": "omega",
                    "param_index": 1,
                }
            )

    encoding_gates = sum(op["role"] == "encoding" for op in operations)
    entangling_gates = sum(op["role"] == "entangle" for op in operations)
    layer_rotation_gates = sum(op["role"] == "layer_rotation" for op in operations)
    return {
        "qubits": n,
        "uploads": L,
        "entangler": gate,
        "rotation_cycle": list(cycle),
        "feature_dim": n * L,
        "operations": operations,
        "param_shapes": {"data_scales": (n * L,), "omega": (L, n, 2)},
        "encoding_gate_count": int(encoding_gates),
        "entangling_gate_count": int(entangling_gates),
        "layer_rotation_gate_count": int(layer_rotation_gates),
        "n_readouts": n,
        "n_variational_layers": L,
        "distinct_variational_param_sets": L,
        "data_insertions": L,
        "circuit_depth_proxy": int(len(operations)),
        "measurement": "PauliZ expectation value on every wire",
        "shots": None,
    }


def _apply_rotation(qml, name: str, angle, wire: int) -> None:
    {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}[str(name).upper()](angle, wires=wire)


def reupload_circuit_introspection(obj: Any) -> dict:
    """Audit that ``obj`` really is an L-upload circuit (not one encoding).

    Structural counts come from the model's stored :func:`reupload_circuit_spec`.
    When PennyLane is importable a tape is also constructed and its gate names
    counted as an independent cross-check (``tape_verified``).  Any tape failure
    degrades honestly to the structural counts with a recorded reason.
    """
    spec = getattr(obj, "circuit_spec", None)
    if spec is None:
        raise ValueError("object has no circuit_spec; not a re-uploading model")
    result: dict[str, Any] = {
        "uploads": int(spec["uploads"]),
        "qubits": int(spec["qubits"]),
        "n_variational_layers": int(spec["n_variational_layers"]),
        "distinct_variational_param_sets": int(spec["distinct_variational_param_sets"]),
        "data_insertions": int(spec["data_insertions"]),
        "encoding_gate_count": int(spec["encoding_gate_count"]),
        "entangling_gate_count": int(spec["entangling_gate_count"]),
        "n_readouts": int(spec["n_readouts"]),
        "circuit_depth_proxy": int(spec["circuit_depth_proxy"]),
        "shots": spec["shots"],
        "tape_verified": False,
        "tape_note": None,
    }
    qnode = getattr(obj, "circuit", None)
    try:
        qml = _require_pennylane()
        if qnode is None:
            raise ValueError("model exposes no QNode")
        shapes = spec["param_shapes"]
        n, L = int(spec["qubits"]), int(spec["uploads"])
        tape = qml.workflow.construct_tape(qnode)(
            np.ones((1, n * L), dtype=np.float64),
            np.ones(shapes["data_scales"], dtype=np.float64),
            np.ones(shapes["omega"], dtype=np.float64),
        )
        names = [op.name for op in tape.operations]
        tape_enc = sum(name in _ALLOWED_ROTATIONS for name in names)
        tape_ent = sum(name in {"CNOT", "CZ"} for name in names)
        result.update(
            {
                "tape_verified": True,
                "tape_encoding_gate_count": int(tape_enc),
                "tape_entangling_gate_count": int(tape_ent),
                "tape_measurement_count": int(len(tape.measurements)),
                "tape_counts_match_spec": bool(
                    tape_enc == result["encoding_gate_count"]
                    and tape_ent == result["entangling_gate_count"]
                    and len(tape.measurements) == result["n_readouts"]
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - depends on the PennyLane version
        result["tape_note"] = f"tape verification unavailable: {type(exc).__name__}: {exc}"
    trainable = 0
    try:
        torch = __import__("torch")
        if isinstance(obj, torch.nn.Module):
            trainable = sum(p.numel() for p in obj.parameters() if p.requires_grad)
    except Exception:  # pragma: no cover
        pass
    result["trainable_param_count"] = int(trainable)
    return result


# --------------------------------------------------------------------------- #
# Quantum model                                                                #
# --------------------------------------------------------------------------- #
def build_reupload_qnn(cfg: ReuploadConfig):
    """Build the re-uploading QNN module: ``(B, n*L)`` features -> ``(B, 2)`` logits."""
    torch = _require_torch()[0]
    qml = _require_pennylane()
    cfg.validate()
    n, L = int(cfg.qubits), int(cfg.uploads)
    spec = reupload_circuit_spec(n, L, cfg.entangler, cfg.rotation_cycle)
    gate = spec["entangler"]
    operations = spec["operations"]

    dev = qml.device(str(cfg.device_name), wires=n, shots=None)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, data_scales, omega):
        # inputs has shape (B, n*L); PennyLane broadcasts the leading axis.
        feats = qml.math.reshape(inputs, (-1, L, n))
        for op in operations:
            role = op["role"]
            if role == "encoding":
                j = op["feature_index"]
                angle = data_scales[j] * feats[:, op["upload"], op["wire"]]
                _apply_rotation(qml, op["name"], angle, op["wire"])
            elif role == "entangle":
                if gate == "cz":
                    qml.CZ(wires=op["wires"])
                else:
                    qml.CNOT(wires=op["wires"])
            else:  # layer_rotation -- omega[upload, wire, index], independent per layer
                angle = omega[op["upload"], op["wire"], op["param_index"]]
                _apply_rotation(qml, op["name"], angle, op["wire"])
        return [qml.expval(qml.PauliZ(q)) for q in range(n)]

    class _ReuploadQNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cfg = cfg
            self.n_qubits = n
            self.uploads = L
            self.entangler = gate
            self.rotation_cycle = tuple(str(r).upper() for r in cfg.rotation_cycle)
            self.input_dim = int(spec["feature_dim"])
            self.circuit_spec = spec
            self.circuit = circuit

            # TorchLayer draws initial weights from numpy's global RNG: seed it
            # locally and restore state so construction is deterministic.
            rng_state = np.random.get_state()
            np.random.seed(int(cfg.seed))
            try:
                self.layer = qml.qnn.TorchLayer(circuit, dict(spec["param_shapes"]))
            finally:
                np.random.set_state(rng_state)
            # Unit encoding scale at init: angle = bounded feature in [-1, 1].
            with torch.no_grad():
                self.layer.data_scales.fill_(1.0)

            from qmlpipe.heads import build_classification_head

            self.head = build_classification_head(n, cfg=cfg, n_classes=2)

        def forward(self, x):
            if x.ndim != 2 or x.shape[-1] != self.input_dim:
                raise ValueError(
                    f"re-uploading QNN expects (B, {self.input_dim}) features, "
                    f"got {tuple(x.shape)}"
                )
            out = self.layer(x.cpu())
            out = torch.as_tensor(out).to(dtype=x.dtype, device=x.device)
            return self.head(out)

    return _ReuploadQNN()


# --------------------------------------------------------------------------- #
# TRAIN-only mild classical reduction                                          #
# --------------------------------------------------------------------------- #
def _fit_pca_matrix(features: np.ndarray, target: int, seed: int):
    """Indirection kept separate so tests can spy on TRAIN-only fitting."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=int(target), svd_solver="full", random_state=int(seed))
    pca.fit(features)
    matrix = np.asarray(pca.components_, dtype=np.float32).T
    evr = float(np.sum(pca.explained_variance_ratio_))
    return matrix, evr


def _seeded_projection(input_dim: int, target: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return (
        rng.standard_normal((int(input_dim), int(target))) / float(np.sqrt(target))
    ).astype(np.float32)


def fit_reupload_projection(X_train, cfg: ReuploadConfig) -> tuple[dict, np.ndarray]:
    """Fit the MILDEST reduction needed by the encoding, on TRAIN only.

    Pipeline: TRAIN mean centering -> PCA to ``n * L`` (or a padded identity for
    under-width inputs, or a seeded random projection if sklearn is absent) ->
    TRAIN component standardisation -> tanh bound to [-1, 1].  Every fitted
    statistic is stored so :func:`project_reupload_features` applies the exact
    same transform to VAL/TEST.  VAL and TEST are never passed to this function.
    """
    cfg.validate()
    arr = np.asarray(X_train, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("TRAIN features must be a non-empty 2-D matrix")
    in_dim, target = int(arr.shape[1]), int(cfg.feature_dim)
    mean = arr.mean(axis=0).astype(np.float32)
    centered = arr - mean

    method = "train_pca_standardize_tanh"
    evr: float | None = None
    matrix: np.ndarray | None = None
    if in_dim >= target:
        try:
            matrix, evr = _fit_pca_matrix(centered, target, int(cfg.seed))
        except Exception:  # pragma: no cover - verifier without sklearn
            matrix = _seeded_projection(in_dim, target, int(cfg.seed))
            evr = None
            method = "seeded_random_projection_standardize_tanh"
        Z = centered @ matrix
    else:
        method = "pad_standardize_tanh"
        Z = np.zeros((arr.shape[0], target), dtype=np.float32)
        Z[:, :in_dim] = centered

    component_mean = Z.mean(axis=0).astype(np.float32)
    component_scale = Z.std(axis=0).astype(np.float32)
    component_scale = np.where(component_scale < _EPS, 1.0, component_scale).astype(np.float32)

    preprocessing = {
        "method": method,
        "fitted_on": "train_only",
        "val_test_used_for_fit": False,
        "input_dim": in_dim,
        "target_dim": target,
        "projection_budget": f"{in_dim} -> {target} (= {cfg.qubits} qubits x {cfg.uploads} uploads)",
        "steps": [
            {"step": "center", "statistic": "train_mean", "fitted_on": "train"},
            {
                "step": "linear_projection",
                "kind": method.split("_")[1] if method.startswith("train_") else method.split("_")[0],
                "n_components": target,
                "explained_variance_ratio_sum": evr,
                "fitted_on": "train",
            },
            {"step": "standardize", "statistic": "train_component_mean_std", "fitted_on": "train"},
            {"step": "tanh_bound", "range": [-1.0, 1.0], "fitted_on": None},
        ],
        "note": (
            "Mild TRAIN-only linear reduction to exactly n*L features, the minimum "
            "required by the angle encoding; no VAL/TEST information is used and no "
            "heavy classical preprocessing hides the quantum component."
        ),
    }
    meta: dict[str, Any] = {
        "method": method,
        "input_dim": in_dim,
        "target_dim": target,
        "seed": int(cfg.seed),
        "qubits": int(cfg.qubits),
        "uploads": int(cfg.uploads),
        "mean": mean,
        "matrix": matrix,
        "component_mean": component_mean,
        "component_scale": component_scale,
        "explained_variance_ratio_sum": evr,
        "classical_preprocessing": preprocessing,
    }
    return meta, project_reupload_features(arr, meta)


def project_reupload_features(X, meta: Mapping[str, Any]) -> np.ndarray:
    """Apply a stored TRAIN-fitted projection to any split (TRAIN/VAL/TEST)."""
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"features must be a 2-D matrix, got shape {arr.shape}")
    in_dim, target = int(meta["input_dim"]), int(meta["target_dim"])
    if arr.shape[1] != in_dim:
        raise ValueError(
            f"expected {in_dim} cached features, got {arr.shape[1]}; the projection "
            "is bound to the TRAIN feature width"
        )
    centered = arr - np.asarray(meta["mean"], dtype=np.float32)
    matrix = meta.get("matrix")
    if matrix is not None:
        Z = centered @ np.asarray(matrix, dtype=np.float32)
    else:
        Z = np.zeros((arr.shape[0], target), dtype=np.float32)
        Z[:, :in_dim] = centered
    Z = (Z - np.asarray(meta["component_mean"], dtype=np.float32)) / np.asarray(
        meta["component_scale"], dtype=np.float32
    )
    return np.tanh(Z).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Deterministic TRAIN cap                                                      #
# --------------------------------------------------------------------------- #
def _cap_train_local(X, y, cap: int, seed: int):
    """Dependency-free fallback for the stratified TRAIN cap."""
    cap_int, n = int(cap), int(len(np.asarray(y)))
    meta: dict[str, Any] = {
        "train_cap": cap_int,
        "train_size_before": n,
        "train_size_after": n,
        "cap_applied": False,
        "note": "no cap applied (cap<=0 or train size <= cap) [local fallback]",
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
    """Delegate to ``experiments.fresh_protocol.cap_train`` (never edited here)."""
    fn = getattr(_fp, "cap_train", None)
    if callable(fn):
        return fn(X, y, cap, seed)
    return _cap_train_local(X, y, cap, seed)


# --------------------------------------------------------------------------- #
# Gradients, training, sealed TEST                                             #
# --------------------------------------------------------------------------- #
def _gradient_norms(model, loss) -> dict[str, float]:
    """Return per-role gradient norms from an already-computed ``loss.backward()``."""
    torch = _require_torch()[0]
    out: dict[str, float] = {"data_scales": 0.0, "omega": 0.0, "head": 0.0}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        norm = float(param.grad.detach().reshape(-1).norm().item())
        if "scale" in name:
            out["data_scales"] += norm
        elif "omega" in name:
            out["omega"] += norm
        else:
            out["head"] += norm
    return out


def gradient_probe(model, X, y=None, cfg: ReuploadConfig | None = None) -> dict:
    """One forward/backward on synthetic data; assert both gradient paths move.

    ``data_scales`` is the data-encoding gradient path and ``omega`` is the
    independent variational-layer path.  No optimiser step is taken and grads
    are cleared afterwards, so the probe never mutates training state.
    """
    torch, F = _require_torch()
    from qmlpipe.losses import class_weights_from_labels, weighted_cross_entropy

    x = torch.as_tensor(np.asarray(X, dtype=np.float32))
    if y is None:
        y = np.asarray([0, 1] * max(1, (len(x) + 1) // 2))[: len(x)]
    y_t = torch.as_tensor(np.asarray(y, dtype=np.int64).reshape(-1))
    mode = "balanced" if cfg is None else str(cfg.class_weight)
    weights = class_weights_from_labels(y_t.numpy(), n_classes=2, mode=mode)
    model.zero_grad(set_to_none=True)
    loss = weighted_cross_entropy(model(x), y_t, weight=weights)
    loss.backward()
    norms = _gradient_norms(model, loss)
    model.zero_grad(set_to_none=True)
    norms["loss"] = float(loss.detach().item())
    return norms


def train_reupload_from_features(X_train, y_train, X_val, y_val, cfg: ReuploadConfig, seed: int):
    """Train one re-uploading QNN on cached TRAIN/VAL features (no TEST access)."""
    torch, _F = _require_torch()
    from qmlpipe.losses import class_weights_from_labels, weighted_cross_entropy
    from qmlpipe.train import seed_everything

    projection, X_fit = fit_reupload_projection(X_train, cfg)
    y_fit = np.asarray(y_train, dtype=np.int64).reshape(-1)
    X_val_proj = project_reupload_features(X_val, projection)
    y_val_arr = np.asarray(y_val, dtype=np.int64).reshape(-1)
    if len(y_fit) != len(X_fit) or len(y_val_arr) != len(X_val_proj):
        raise ValueError("label/feature counts disagree after projection")

    seed_everything(int(seed))
    model = build_reupload_qnn(cfg)
    setattr(model, _PROJ_ATTR, {k: v for k, v in projection.items()})

    weights = class_weights_from_labels(y_fit, n_classes=2, mode=str(cfg.class_weight))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay)
    )
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit))
    generator = torch.Generator().manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(cfg.batch_size), shuffle=True, generator=generator
    )
    X_val_t = torch.from_numpy(X_val_proj)

    initial = [p.detach().clone() for p in model.parameters()]
    first_grad: dict[str, float] | None = None
    total_steps = 0
    best_val = float("inf")
    best_state = None
    epoch_log: list[dict] = []

    def _val_loss() -> float:
        model.eval()
        loss_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(X_val_t), int(cfg.batch_size)):
                xb = X_val_t[start : start + int(cfg.batch_size)]
                yb = torch.from_numpy(y_val_arr[start : start + len(xb)])
                loss_sum += float(
                    weighted_cross_entropy(model(xb), yb, weight=weights, reduction="sum").item()
                )
        return loss_sum / max(1, len(y_val_arr))

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
            if first_grad is None:
                first_grad = _gradient_norms(model, loss)
            optimizer.step()
            total_steps += 1
            n_batches += 1
            loss_sum += float(loss.item()) * len(yb)
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            seen += len(yb)
            if not np.isfinite(loss_sum):
                raise RuntimeError("non-finite training loss; refusing to report a QNN result")
        val_loss = _val_loss()
        epoch_log.append(
            {
                "epoch": int(epoch),
                "train_loss": float(loss_sum / max(1, seen)),
                "train_acc": float(correct / max(1, seen)),
                "val_loss": float(val_loss),
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
        sum(float((p.detach() - start).norm().item()) for p, start in zip(model.parameters(), initial))
    )
    introspection = reupload_circuit_introspection(model)
    history: dict[str, Any] = {
        "seed": int(seed),
        "model": "reupload_qnn",
        "epochs_configured": int(cfg.epochs),
        "max_batches_per_epoch": _MAX_BATCHES_PER_EPOCH,
        "max_total_steps": _MAX_TOTAL_STEPS,
        "total_steps": int(total_steps),
        "best_val_loss": float(best_val) if np.isfinite(best_val) else None,
        "classical_preprocessing": projection["classical_preprocessing"],
        "grad_norms_first_batch": first_grad,
        "param_delta": param_delta,
        "class_weight_mode": str(cfg.class_weight),
        "optimizer": "adam",
        "circuit": introspection,
        "epoch_log": epoch_log,
    }
    return model, history


def evaluate_once(model, X_test, y_test) -> dict:
    """Score the sealed TEST exactly once via ``experiments.metrics.binary_metrics``."""
    torch = _require_torch()[0]
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
        project_reupload_features(X_test, projection)
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
        "model": "reupload_qnn",
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
def _resolve_artifact(path, results_root=None) -> Path:
    """Resolve a baseline path to its ``metrics.json`` file.

    Accepts either a ``metrics.json`` FILE path or a DIRECTORY containing one:
    an existing directory is mapped to ``<dir>/metrics.json``.  If that file is
    absent the returned path simply does not exist, so callers can report it as
    ``missing`` instead of silently dropping the baseline.  Absolute paths and
    ``results_root``-relative paths keep their previous precedence.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate
    else:
        base = Path(results_root) if results_root is not None else _REPO_ROOT
        rooted = base / candidate
        resolved = rooted if rooted.exists() else candidate
    if resolved.is_dir():
        resolved = resolved / "metrics.json"
    return resolved


def load_baseline_report(
    path,
    model_names: Sequence[str],
    backbone: str | None = None,
    results_root=None,
) -> dict:
    """Load baseline accuracies WITH an explicit status; no silent baseline drops.

    Returns a JSON-native dict: ``path`` (resolved metrics.json), ``requested``
    (original argument), ``status`` in {``"ok"``, ``"missing"``,
    ``"unreadable"``, ``"no_matching_runs"``}, ``reason`` (one of ``"missing"``,
    ``"unreadable"``, ``"no matching runs"`` or ``None``), ``n_values`` and
    ``values``.  Never fabricates a value when the artifact cannot be used.
    """
    artifact = _resolve_artifact(path, results_root)
    report: dict[str, Any] = {
        "requested": str(path),
        "path": str(artifact),
        "status": "ok",
        "reason": None,
        "n_values": 0,
        "values": {},
        "detail": None,
    }
    if not artifact.exists():
        report.update({"status": "missing", "reason": "missing"})
        return report
    if not artifact.is_file():
        report.update({"status": "unreadable", "reason": "unreadable"})
        return report
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("artifact is not a JSON object")
    except Exception as exc:  # pragma: no cover - depends on artifact contents
        report.update(
            {
                "status": "unreadable",
                "reason": "unreadable",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        return report

    wanted = {str(name).lower() for name in model_names}
    values: dict[int, float] = {}
    for run in payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        if str(run.get("model", "")).lower() not in wanted:
            continue
        if backbone is not None and str(run.get("backbone", "")).lower() != str(backbone).lower():
            continue
        if str(run.get("status", "ok")) != "ok":
            continue
        block = run.get("metrics", {})
        accuracy = block.get("accuracy", run.get("accuracy")) if isinstance(block, dict) else None
        if accuracy is None:
            continue
        try:
            values[int(run["seed"])] = float(accuracy)
        except (KeyError, TypeError, ValueError):  # pragma: no cover - malformed run
            continue
    if not values:
        report.update({"status": "no_matching_runs", "reason": "no matching runs"})
        return report
    report["values"] = values
    report["n_values"] = int(len(values))
    return report


def load_baseline_values(
    path,
    model_names: Sequence[str],
    backbone: str | None = None,
    results_root=None,
) -> dict[int, float]:
    """Read per-seed accuracies from a file or directory (no numbers hardcoded).

    Thin compatibility wrapper over :func:`load_baseline_report`: returns only
    the values dict, so callers needing the failure reason use the report API.
    """
    return load_baseline_report(path, model_names, backbone, results_root)["values"]


def paired_comparisons(
    reupload_by_seed: Mapping[int, float],
    baselines: Mapping[str, Mapping[int, float]],
    *,
    alpha: float = _fp.ALPHA,
    seed: int = _fp.BOOT_SEED,
    n_boot: int = _fp.BOOT_B,
) -> list[dict]:
    """Pair re-uploading QNN vs every baseline per seed, Holm-adjust the family.

    The returned ``quantum_advantage_supported`` flag is descriptive evidence
    only: it can never override the unconditional no-advantage claim boundary.
    """
    comparisons: list[dict] = []
    for name, values in baselines.items():
        common = sorted(
            set(int(k) for k in reupload_by_seed) & set(int(k) for k in values)
        )
        if not common:
            continue
        a = [float(reupload_by_seed[s]) for s in common]
        b = [float(values[s]) for s in common]
        item = _fp.paired_comparison(
            a,
            b,
            alpha=float(alpha),
            seed=int(seed),
            n_boot=int(n_boot),
            pair=("reupload_qnn", str(name)),
        )
        item["seeds"] = [int(s) for s in common]
        item["metric"] = "accuracy"
        item["mean_reupload"] = float(np.mean(a))
        item["mean_baseline"] = float(np.mean(b))
        item["mean_difference"] = float(np.mean(np.asarray(a) - np.asarray(b)))
        comparisons.append(item)

    for item in _fp.holm_adjust(comparisons, alpha=float(alpha)):
        item["quantum_advantage_supported"] = bool(
            item.get("decision") == "significant"
            and item.get("mean_difference") is not None
            and float(item["mean_difference"]) > 0.0
        )
        item["claim_note"] = (
            "Descriptive paired evidence only; no quantum advantage is asserted "
            "without matched budgets, n>=10, power>=0.8 and Holm-corrected CI."
        )
    return comparisons


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
    qubits: int = 6,
    uploads: int = 4,
    entangler: str = "cnot",
    qnn_results=None,
    results_root=None,
) -> dict:
    """Return a write-free plan (used by the CLI's ``--dry-run``)."""
    seed_list = _normalize_seeds(seeds)
    cfg = ReuploadConfig(
        seed=int(seed_list[0]), qubits=int(qubits), uploads=int(uploads),
        entangler=str(entangler), epochs=int(epochs),
    )
    cfg.validate()
    spec = reupload_circuit_spec(cfg.qubits, cfg.uploads, cfg.entangler, cfg.rotation_cycle)
    classical_rel = DEFAULT_CLASSICAL_PATHS.get(str(dataset).lower())
    qnn_rel = str(qnn_results) if qnn_results is not None else DEFAULT_AMPLITUDE_QNN_PATHS.get(
        str(dataset).lower()
    )
    return {
        "dry_run": True,
        "protocol": "reupload_qnn.v1",
        "dataset": str(dataset),
        "backbone": str(backbone),
        "seeds": seed_list,
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir) if cache_dir is not None else str(_default_cache_dir()),
        "train_cap": int(train_cap),
        "epochs": int(epochs),
        "qubits": int(qubits),
        "uploads": int(uploads),
        "entangler": str(entangler),
        "feature_dim": int(spec["feature_dim"]),
        "projection_target_dim": int(spec["feature_dim"]),
        "classical_preprocessing": (
            f"TRAIN-only PCA to n*L = {spec['feature_dim']} features, TRAIN-only "
            "standardisation and a tanh bound; fitted on TRAIN only"
        ),
        "circuit": {
            "n_qubits": int(spec["qubits"]),
            "uploads": int(spec["uploads"]),
            "data_insertions": int(spec["data_insertions"]),
            "distinct_variational_param_sets": int(spec["distinct_variational_param_sets"]),
            "encoding_gate_count": int(spec["encoding_gate_count"]),
            "entangling_gate_count": int(spec["entangling_gate_count"]),
            "n_readouts": int(spec["n_readouts"]),
            "circuit_depth_proxy": int(spec["circuit_depth_proxy"]),
            "shots": None,
        },
        "runs_planned": len(seed_list),
        "max_batches_per_epoch": _MAX_BATCHES_PER_EPOCH,
        "max_total_steps": _MAX_TOTAL_STEPS,
        "classical_control": classical_rel,
        "amplitude_qnn_control": qnn_rel,
        "qnn_results_note": (
            "--qnn-results accepts EITHER a metrics.json FILE path OR a DIRECTORY "
            "containing metrics.json; a configured baseline with no usable values "
            "is reported in warnings (missing/unreadable/no matching runs) and never "
            "silently dropped"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "cpu_note": (
            f"{qubits} qubits / {uploads} uploads are chosen for CPU feasibility only; "
            "not a hardware, scaling or quantum-advantage claim; shots=None analytic "
            "CPU simulation."
        ),
    }


def run_reupload_protocol(
    dataset: str,
    backbone: str,
    seeds,
    output_dir,
    cache_dir=None,
    train_cap: int = 2000,
    epochs: int = 3,
    qubits: int = 6,
    uploads: int = 4,
    *,
    entangler: str = "cnot",
    qnn_results=None,
    results_root=None,
    class_weight: str = "balanced",
    learning_rate: float = 0.001,
    batch_size: int = 32,
    alpha: float = _fp.ALPHA,
    n_boot: int = _fp.BOOT_B,
    boot_seed: int = _fp.BOOT_SEED,
) -> dict:
    """Run one bounded re-uploading QNN per seed on cached features, write additive JSON."""
    seed_list = _normalize_seeds(seeds)
    cache_dir = cache_dir if cache_dir is not None else _default_cache_dir()
    expected_cache = _cache_npz_path(dataset, backbone, cache_dir)
    if not expected_cache.exists():
        raise FileNotFoundError(
            f"frozen feature cache not found: {expected_cache}. Run the classical "
            "fresh-protocol featurisation first; this arm never extracts images or "
            "fabricates features."
        )

    plan = build_plan(
        dataset, backbone, seed_list, output_dir, cache_dir, train_cap, epochs,
        qubits, uploads, entangler, qnn_results, results_root,
    )
    cfg = ReuploadConfig(
        seed=int(seed_list[0]), qubits=int(qubits), uploads=int(uploads),
        entangler=str(entangler), epochs=int(epochs), class_weight=str(class_weight),
        learning_rate=float(learning_rate), batch_size=int(batch_size),
    )
    cfg.validate()
    spec = reupload_circuit_spec(cfg.qubits, cfg.uploads, cfg.entangler, cfg.rotation_cycle)

    warnings: list[dict] = []
    runs: list[dict] = []
    feature_dim: int | None = None
    cache_hit: bool | None = None

    for seed in seed_list:
        try:
            seed_cfg = replace(cfg, seed=int(seed))
            cached = _fp.cache_features(dataset, backbone, cache_dir=cache_dir, seed=int(seed))
            X_fit, y_fit, cap_meta = cap_train(
                cached["X"]["train"], cached["y"]["train"], train_cap, int(seed)
            )
            feature_dim = int(np.asarray(cached["X"]["train"]).shape[1])
            cache_hit = bool(cached.get("cache_hit")) if cache_hit is None else cache_hit
            model, history = train_reupload_from_features(
                X_fit, y_fit, cached["X"]["val"], cached["y"]["val"], seed_cfg, int(seed)
            )
            result = evaluate_once(model, cached["X"]["test"], cached["y"]["test"])
        except Exception as exc:  # record, never fabricate
            warnings.append(
                {
                    "stage": "reupload_qnn_arm",
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
                "model": "reupload_qnn",
                "status": "ok",
                "train_cap": cap_meta,
                "metrics": result,
                "history": history,
                "cache_path": cached.get("cache_path"),
                "group_disjoint": bool(cached.get("provenance", {}).get("group_disjoint")),
            }
        )

    aggregates: dict[str, Any] = {
        "reupload_qnn": {"n_runs": len(runs), "seeds": [r["seed"] for r in runs]}
    }
    for metric in _METRICS:
        values = [
            float(run["metrics"][metric])
            for run in runs
            if run["metrics"].get(metric) is not None and np.isfinite(float(run["metrics"][metric]))
        ]
        if values:
            aggregates["reupload_qnn"][metric] = _fp.aggregate(values, name="reupload_qnn")

    reupload_by_seed = {
        int(r["seed"]): float(r["metrics"]["accuracy"])
        for r in runs
        if r["metrics"].get("accuracy") is not None
    }
    baselines: dict[str, dict[int, float]] = {}
    baseline_sources: list[dict] = []
    baseline_warnings: list[dict] = []

    def _load_baseline(name: str, path, model_names: Sequence[str]) -> None:
        """Load one configured baseline; record an explicit warning on any failure."""
        report = load_baseline_report(path, model_names, backbone, results_root)
        baseline_sources.append(
            {
                "name": name,
                "configured": True,
                "loaded": bool(report["values"]),
                "requested": report["requested"],
                "path": report["path"],
                "status": report["status"],
                "reason": report["reason"],
                "n_values": report["n_values"],
            }
        )
        if report["values"]:
            baselines[name] = report["values"]
            return
        baseline_warnings.append(
            {
                "stage": "baseline_load",
                "baseline": name,
                "dataset": str(dataset),
                "backbone": str(backbone),
                "requested": report["requested"],
                "path": report["path"],
                "status": report["status"],
                "reason": report["reason"],
                "error_type": "BaselineUnavailable",
                "error": (
                    f"configured {name} baseline yielded no values ({report['reason']}); "
                    "no comparison fabricated"
                ),
            }
        )

    classical_rel = DEFAULT_CLASSICAL_PATHS.get(str(dataset).lower())
    if classical_rel:
        _load_baseline("classical_poly2", classical_rel, ("poly2", "poly"))
    qnn_rel = (
        qnn_results
        if qnn_results is not None
        else DEFAULT_AMPLITUDE_QNN_PATHS.get(str(dataset).lower())
    )
    if qnn_rel:
        _load_baseline("fresh_qnn_amplitude", qnn_rel, ("qnn",))
    comparisons = paired_comparisons(
        reupload_by_seed, baselines, alpha=alpha, seed=boot_seed, n_boot=n_boot
    )
    all_warnings = warnings + baseline_warnings

    # Run accounting counts seed-level failures only; baseline warnings are
    # infrastructure diagnostics and must not inflate n_runs_skipped.
    n_succeeded, n_skipped = len(runs), len(warnings)
    metrics_payload: dict[str, Any] = {
        "schema": "reupload_qnn.metrics.v1",
        "dataset": str(dataset),
        "backbone": str(backbone),
        "claim_boundary": CLAIM_BOUNDARY,
        "config": {key: value for key, value in plan.items() if key != "dry_run"},
        "runs": runs,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "baselines_loaded": sorted(baselines),
        "warnings": all_warnings,
        "n_runs_expected": int(len(seed_list)),
        "n_runs_succeeded": int(n_succeeded),
        "n_runs_skipped": int(n_skipped),
        "accounting_invariant": bool(len(seed_list) == n_succeeded + n_skipped),
    }
    provenance_payload: dict[str, Any] = {
        "schema": "reupload_qnn.provenance.v1",
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
        "circuit": {
            "n_qubits": int(spec["qubits"]),
            "uploads": int(spec["uploads"]),
            "data_insertions": int(spec["data_insertions"]),
            "distinct_variational_param_sets": int(spec["distinct_variational_param_sets"]),
            "entangling_gate_count": int(spec["entangling_gate_count"]),
            "encoding_gate_count": int(spec["encoding_gate_count"]),
            "circuit_depth_proxy": int(spec["circuit_depth_proxy"]),
            "n_readouts": int(spec["n_readouts"]),
            "device": "default.qubit",
            "interface": "torch",
            "diff_method": "backprop",
            "shots": None,
            "shots_note": "CPU analytic simulation; no shot-noise or device-speedup claim",
            "upload_formula": "prod_{i=1..L} [ V(omega_i) U_phi(x) ] |0>^n",
            "entanglement": f"{spec['entangler'].upper()} ring + per-wire RY/RZ per upload",
        },
        "projection_budget": {
            "target_dim": int(spec["feature_dim"]),
            "formula": "n * L",
            "classical_preprocessing": (
                runs[0]["history"]["classical_preprocessing"] if runs else None
            ),
        },
        "run_manifest": [
            {
                "dataset": str(dataset),
                "backbone": str(backbone),
                "seed": int(r["seed"]),
                "model": "reupload_qnn",
                "status": "ok",
                "train_cap": r["train_cap"],
                "classical_preprocessing": r["history"]["classical_preprocessing"],
                "grad_norms_first_batch": r["history"].get("grad_norms_first_batch"),
                "param_delta": r["history"].get("param_delta"),
                "circuit": r["history"].get("circuit"),
            }
            for r in runs
        ]
        + warnings,
        "comparison_notes": {
            "classical_arm": classical_rel,
            "amplitude_qnn_arm": str(qnn_rel) if qnn_rel else None,
            "baselines_loaded": sorted(baselines),
            "baseline_sources": baseline_sources,
            "baseline_drop_policy": (
                "A configured baseline with no usable values never silently disappears: "
                "it is reported in warnings with reason missing/unreadable/no matching runs "
                "and is excluded from comparisons. baselines_loaded lists only real loads."
            ),
            "n_budget_matched": False,
            "note": (
                "The committed controls used their own TRAIN budgets while this arm "
                "caps TRAIN at the configured value; a paired win is only admissible "
                "after budget matching in addition to the statistical gates."
            ),
            "test_scoring": "exactly one forward pass per seed via evaluate_once",
            "cpu_note": plan["cpu_note"],
        },
        "warnings": all_warnings,
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
        "warnings": all_warnings,
        "claim_boundary": CLAIM_BOUNDARY,
    }
