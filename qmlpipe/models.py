"""Trainable quantum-classical models for the QML pipeline (QML-D01).

Public API
----------
``build_feature_extractor(cfg)`` -- frozen EfficientNet-B0 + projected L2 features
``build_qnn(cfg)``               -- embedding + ansatz + PauliZ readout -> 2 logits
``build_transfer_qnn(cfg)``      -- feature extractor followed by ``build_qnn``
``circuit_introspection(obj)``   -- entangling gate / trainable parameter counts

All widths come from :class:`qmlpipe.config.QMLConfig`; the 4/8/10-qubit
ablations are pure parameter changes (QC-D03 §4).  The trainable QNode always
uses ``default.qubit`` with ``interface='torch'`` and ``diff_method='backprop'``
-- ``lightning.qubit`` does not support torch backprop and is reserved for the
analytic kernel in ``qmlpipe.kernel``.  QNode inputs run on CPU; outputs are
cast back to the caller's dtype and device before the classical head.

Import safety
-------------
PennyLane is imported defensively.  Importing this module never fails when the
quantum stack is absent; the quantum builders raise a clear ``RuntimeError``
only when they are actually called.
"""
from __future__ import annotations

import inspect
import warnings
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qmlpipe.config import QMLConfig

try:  # pragma: no cover - depends on the environment
    import pennylane as qml
except Exception:  # pragma: no cover
    qml = None  # type: ignore[assignment]

__all__ = [
    "build_feature_extractor",
    "build_qnn",
    "build_transfer_qnn",
    "circuit_introspection",
]

_MISSING_PENNYLANE = (
    "PennyLane is required to construct a quantum model. Install the pinned "
    "stack (PennyLane==0.40.0 and PennyLane-Lightning==0.40.0). The classical "
    "feature extractor remains available without it."
)

_EPS = 1e-8

# Native/derived multi-qubit (entangling) operations in PennyLane.
_ENTANGLING_GATE_NAMES = frozenset(
    {
        "CNOT", "CZ", "CY", "CH", "SWAP", "iSWAP", "ISWAP", "SISWAP",
        "CRX", "CRY", "CRZ", "CRot", "ControlledPhaseShift",
        "IsingXX", "IsingYY", "IsingZZ", "IsingXY",
        "MultiControlledX", "Toffoli", "CSWAP",
    }
)

# Only entangling *ansatz templates* are expanded for the gate count.  State
# preparation (AmplitudeEmbedding / AngleEmbedding) stays opaque: it is the
# parameter-free data-loading primitive, not a trainable entangler, so a bare
# embedding kernel circuit correctly reports zero entangling gates.
_EXPANDABLE_TEMPLATES = frozenset(
    {"StronglyEntanglingLayers", "BasicEntanglerLayers", "RealAmplitudes",
     "EfficientSU2", "TwoLocal"}
)


def _require_pennylane() -> None:
    if qml is None:
        raise RuntimeError(_MISSING_PENNYLANE)


# --------------------------------------------------------------------------- feature extractor

class _FrozenEfficientNet(nn.Module):
    """Local fallback feature source mirroring ``models.classical``.

    EfficientNet-B0 is constructed with ``weights=None`` (no download) and is
    permanently frozen; only the ``1280 -> cfg.feature_dim`` projection is
    trainable.  Rows are explicitly L2-normalised.
    """

    def __init__(self, cfg: QMLConfig) -> None:
        super().__init__()
        from torchvision import models as tv_models

        backbone = tv_models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.projection = nn.Linear(int(cfg.classical_dim), int(cfg.feature_dim))
        self.feature_dim = int(cfg.feature_dim)

        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f = self.features(x)
            f = self.avgpool(f)
        f = torch.flatten(f, 1)  # (B, cfg.classical_dim)
        return F.normalize(self.projection(f), p=2, dim=1, eps=_EPS)


def build_feature_extractor(cfg: QMLConfig) -> nn.Module:
    """Return a frozen EfficientNet-B0 feature extractor with L2 projection.

    The shared implementation in ``models.classical.build_efficientnet_features``
    is reused when available; a self-contained fallback (identical contract) is
    used otherwise.  Output: ``(B, 3, H, W) -> (B, cfg.feature_dim)`` with unit
    L2 rows, suitable for AmplitudeEmbedding.
    """
    try:
        from models.classical import build_efficientnet_features

        return build_efficientnet_features(cfg)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - fallback path
        return _FrozenEfficientNet(cfg)


# --------------------------------------------------------------------------- QNN

class _TorchQNN(nn.Module):
    """Embedding + entangling ansatz + PauliZ readout + 2-logit linear head."""

    def __init__(self, cfg: QMLConfig) -> None:
        super().__init__()
        _require_pennylane()

        self.cfg = cfg
        self.n_qubits = int(cfg.qubits)
        self.embedding = str(cfg.embedding).lower()
        self.ansatz = str(cfg.ansatz).lower()
        self.input_dim = int(cfg.qnn_input_dim)

        # Noisy channels require a density-matrix device and do not support the
        # analytic torch-backprop path.  Degrade honestly: report, never fake.
        if cfg.noise_depolarizing is not None:
            warnings.warn(
                "QMLConfig.noise_depolarizing is set, but depolarizing channels "
                "are not supported on the analytic default.qubit + torch/backprop "
                "QNN path. The noise setting is recorded but NOT applied "
                "(graceful degradation); the model is the noiseless circuit.",
                RuntimeWarning,
                stacklevel=2,
            )

        device_name = str(cfg.device_name)
        if device_name != "default.qubit":
            warnings.warn(
                f"Trainable QNNs require default.qubit with torch backprop; "
                f"device_name={device_name!r} is not used for the QNode "
                "(lightning.qubit does not support backprop). Falling back to "
                "default.qubit.",
                RuntimeWarning,
                stacklevel=2,
            )
            device_name = "default.qubit"

        # Finite shots are reserved for kernel diagnostics (QC-D03 §3.4) and
        # are incompatible with analytic torch backprop. Degrade honestly:
        # warn and force analytic execution rather than fabricating shot noise.
        qnn_shots = cfg.shots
        if qnn_shots is not None:
            warnings.warn(
                "QMLConfig.shots is set, but finite shots are reserved for "
                "kernel diagnostics and are incompatible with "
                "diff_method='backprop'. Forcing shots=None (analytic) for "
                "the QNN device; no shot-noise result is produced.",
                RuntimeWarning,
                stacklevel=2,
            )
            qnn_shots = None

        self.qnode = self._make_qnode(cfg, device_name, qnn_shots)

        weight_shape = (
            (int(cfg.layers), self.n_qubits)
            if self.ansatz == "basic_entangler"
            else (int(cfg.layers), self.n_qubits, 3)
        )
        self.weight_shape = weight_shape

        # TorchLayer draws its initial weights from numpy's global RNG. Seed it
        # locally and restore state so construction is deterministic without
        # perturbing the caller's RNG.
        rng_state = np.random.get_state()
        np.random.seed(int(cfg.seed))
        try:
            self.layer = qml.qnn.TorchLayer(
                self.qnode, {"weights": weight_shape}
            )
        finally:
            np.random.set_state(rng_state)

        self.head = nn.Linear(self.n_qubits, 2)

    def _make_qnode(self, cfg: QMLConfig, device_name: str, shots):
        """Build the trainable QNode.

        The data parameter is named ``inputs`` because PennyLane 0.40's
        ``qml.qnn.TorchLayer`` defaults to ``input_arg="inputs"`` and rejects a
        QNode without an argument of that name.  The device is
        ``default.qubit`` with ``interface="torch"`` and
        ``diff_method="backprop"``.  ``shots`` is always analytic (``None``)
        here: finite shots are reserved for kernel diagnostics (QC-D03 §3.4)
        and are incompatible with backprop, so a finite value is rejected with
        a warning and replaced by ``None`` in ``__init__`` before this call.
        """
        n = self.n_qubits
        dev = qml.device(device_name, wires=n, shots=shots)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            if self.embedding == "angle":
                qml.AngleEmbedding(inputs, wires=range(n), rotation="Y")
            else:
                qml.AmplitudeEmbedding(
                    features=inputs, wires=range(n),
                    normalize=True, pad_with=0.0,
                )
            if self.ansatz == "basic_entangler":
                qml.BasicEntanglerLayers(weights, wires=range(n))
            else:
                qml.StronglyEntanglingLayers(weights, wires=range(n))
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"QNN expects (B, {self.input_dim}) features "
                f"(embedding={self.embedding!r}), got {tuple(x.shape)}."
            )
        out = self.layer(x.cpu())  # QNode executes on CPU; autograd graph intact
        out = torch.as_tensor(out).to(dtype=x.dtype, device=x.device)
        return self.head(out)  # raw logits (B, 2)


class _TransferQNN(nn.Module):
    """Frozen feature extractor -> optional angle adapter -> QNN."""

    def __init__(self, cfg: QMLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.extractor = build_feature_extractor(cfg)
        self.qnn = build_qnn(cfg)
        if self.qnn.input_dim != int(cfg.projection_dim):
            # Angle ablation: the extractor emits projection_dim features but
            # AngleEmbedding consumes one angle per qubit.
            self.adapter: nn.Module = nn.Linear(
                int(cfg.projection_dim), self.qnn.input_dim
            )
        else:
            self.adapter = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnn(self.adapter(self.extractor(x)))


def build_qnn(cfg: QMLConfig) -> nn.Module:
    """Build the trainable QNN module (logits of shape ``(B, 2)``)."""
    return _TorchQNN(cfg)


def build_transfer_qnn(cfg: QMLConfig) -> nn.Module:
    """Build the full frozen-extractor + QNN transfer model (``(B, 2)``)."""
    return _TransferQNN(cfg)


# --------------------------------------------------------------------------- introspection

def _iter_qnodes(obj: Any) -> Iterator[Any]:
    """Yield distinct PennyLane QNodes referenced by ``obj``."""
    if qml is None:
        return
    if isinstance(obj, qml.QNode):
        yield obj
        return
    seen: set[int] = set()
    modules = getattr(obj, "modules", None)
    if callable(modules):
        for module in modules():
            for attr in ("circuit", "qnode", "qnode_func"):
                circuit = getattr(module, attr, None)
                if isinstance(circuit, qml.QNode) and id(circuit) not in seen:
                    seen.add(id(circuit))
                    yield circuit
    for attr in ("qnode", "circuit", "qnode_func"):
        circuit = getattr(obj, attr, None)
        if isinstance(circuit, qml.QNode) and id(circuit) not in seen:
            seen.add(id(circuit))
            yield circuit


def _resolve_weight_shape(obj: Any):
    """Resolve the QNode ansatz-weight shape for tape construction.

    PennyLane 0.40's ``TorchLayer`` does not guarantee a ``weight_shapes``
    attribute, so fall back to the shape recorded on the model here, then to a
    derivation from the config/ansatz attributes.  ``BasicEntanglerLayers``
    needs ``(layers, qubits)`` whereas ``StronglyEntanglingLayers`` needs
    ``(layers, qubits, 3)``; a wrong shape breaks tape construction even though
    the real model executes fine.
    """
    if isinstance(obj, nn.Module):
        for module in obj.modules():
            shapes = getattr(module, "weight_shapes", None)
            if isinstance(shapes, dict) and "weights" in shapes:
                return tuple(shapes["weights"])
        for module in obj.modules():
            shape = getattr(module, "weight_shape", None)
            if isinstance(shape, (tuple, list)):
                return tuple(shape)

    cfg = getattr(obj, "cfg", None)
    ansatz = getattr(obj, "ansatz", None) or getattr(cfg, "ansatz", None)
    layers = getattr(obj, "layers", None) or getattr(cfg, "layers", None)
    n = getattr(obj, "n_qubits", None) or getattr(cfg, "qubits", None)
    if ansatz is not None and layers is not None and n is not None:
        if str(ansatz).lower() == "basic_entangler":
            return (int(layers), int(n))
        return (int(layers), int(n), 3)
    return None


def _construct_tape(qnode: Any, weight_shape=None, feature_dim=None):
    """Construct (do not execute) a tape using shape-valid dummy arguments."""
    n = len(qnode.device.wires)
    fdim = int(feature_dim) if feature_dim is not None else 2 ** n
    func = getattr(qnode, "func", None) or getattr(qnode, "qfunc", None)
    if func is None:
        raise RuntimeError("Cannot introspect a QNode without its qfunc.")
    params = []
    for name in inspect.signature(func).parameters:
        if "weight" in name.lower():
            shape = tuple(weight_shape) if weight_shape is not None else (1, n, 3)
            params.append(np.zeros(shape, dtype=np.float64))
        else:
            params.append(np.ones(fdim, dtype=np.float64))
    try:
        return qml.workflow.construct_tape(qnode)(*params)
    except Exception:  # pragma: no cover - older PennyLane fallback
        with qml.tape.QuantumTape() as tape:
            func(*params)
        return tape


def _expand_operations(operations, max_depth: int = 4, max_ops: int = 20000):
    """Expand entangling ansatz templates so their CNOTs become countable."""
    ops = list(operations)
    for _ in range(max_depth):
        expanded: list[Any] = []
        changed = False
        for op in ops:
            if op.name in _EXPANDABLE_TEMPLATES:
                decomposition = None
                try:
                    decomposition = op.decomposition()
                except Exception:  # pragma: no cover
                    decomposition = None
                if decomposition and len(expanded) + len(decomposition) <= max_ops:
                    changed = True
                    expanded.extend(decomposition)
                    continue
            expanded.append(op)
        ops = expanded
        if not changed:
            break
    return ops


def circuit_introspection(obj: Any) -> dict:
    """Return ``{'entangling_gate_count': int, 'trainable_param_count': int}``.

    Works on a ``nn.Module`` (e.g. a QNN built here), on a bare PennyLane
    ``QNode`` (e.g. the parameter-free kernel circuit), or on any object that
    exposes a QNode via ``.qnode``/``.circuit``.  Embedding operations are not
    counted as entanglers; ansatz templates are decomposed first.
    """
    _require_pennylane()

    weight_shape = _resolve_weight_shape(obj)
    feature_dim = getattr(obj, "input_dim", None)
    if feature_dim is None:
        feature_dim = getattr(getattr(obj, "qnn", None), "input_dim", None)

    entangling = 0
    for qnode in _iter_qnodes(obj):
        tape = _construct_tape(
            qnode, weight_shape=weight_shape, feature_dim=feature_dim
        )
        for op in _expand_operations(tape.operations):
            if op.name in _ENTANGLING_GATE_NAMES:
                entangling += 1

    if isinstance(obj, nn.Module):
        trainable = sum(p.numel() for p in obj.parameters() if p.requires_grad)
    elif isinstance(obj, qml.QNode):
        if getattr(obj, "interface", None) is None:
            trainable = 0
        else:
            try:
                trainable = len(_construct_tape(obj).trainable_params)
            except Exception:  # pragma: no cover
                trainable = 0
    else:
        trainable = 0

    return {
        "entangling_gate_count": int(entangling),
        "trainable_param_count": int(trainable),
    }
