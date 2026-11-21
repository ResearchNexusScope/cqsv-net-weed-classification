"""Trainable hybrid QNN models for the weed-classification rebuild.

All quantum models share one canonical circuit (QC-D03 §2.1):

* ``AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True,
  pad_with=0.0)`` mapping ``2**n_qubits`` features to ``n_qubits`` wires;
* ``StronglyEntanglingLayers(weights, wires=range(n_qubits))`` with weight shape
  ``(cfg.qnn_layers, n_qubits, 3)`` (the canonical 10-qubit model has shape
  ``(2, 10, 3)`` = 60 trainable circuit parameters);
* one ``qml.expval(qml.PauliZ(w))`` per wire, fed to ``nn.Linear(n_qubits, 2)``.

The QNode runs on ``default.qubit`` (CPU) with ``interface="torch"`` and
``diff_method="backprop"``.  Inputs are moved to CPU for the QNode and the
result is cast back to the caller's dtype/device before the classical head.

Import safety
-------------
PennyLane is imported defensively.  Importing this module (or any other model
module) never fails when PennyLane is absent; a clear ``RuntimeError`` is raised
only when a quantum model is actually constructed.

Introspection helpers :func:`entangling_gate_count` and
:func:`trainable_param_count` are provided for the negative-result checks in
QC-D01 (the QSVC kernel has zero of both; the QNNs have both).
"""
from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.config import ExperimentConfig

try:  # pragma: no cover - depends on the environment
    import pennylane as qml
except Exception:  # pragma: no cover
    qml = None  # type: ignore[assignment]

__all__ = [
    "AmplitudeQNNLayer",
    "LegacyQNN",
    "FromScratchHybrid",
    "TransferHybrid",
    "build_legacy_qnn",
    "build_hybrid_from_scratch",
    "build_hybrid_transfer",
    "build_4qubit_ablation",
    "entangling_gate_count",
    "trainable_param_count",
]

_MISSING_PENNYLANE = (
    "PennyLane is required to construct a quantum model. Install the pinned "
    "stack (PennyLane==0.40.0 and PennyLane-Lightning==0.40.0). The classical "
    "models in models/classical.py remain available without it."
)

_EPS = 1e-8

# Names of native/derived multi-qubit (entangling) operations in PennyLane.
ENTANGLING_GATE_NAMES = frozenset(
    {
        "CNOT", "CZ", "CY", "CH", "SWAP", "iSWAP", "ISWAP", "SISWAP",
        "CRX", "CRY", "CRZ", "CRot", "ControlledPhaseShift",
        "IsingXX", "IsingYY", "IsingZZ", "IsingXY",
        "MultiControlledX", "Toffoli", "CSWAP",
    }
)


def _require_pennylane() -> None:
    if qml is None:
        raise RuntimeError(_MISSING_PENNYLANE)


def _canonical_qnn_circuit(n_qubits: int, shots: int | None, device_name: str = "default.qubit"):
    """Build and return the canonical amplitude-encoded QNode.

    Kept as a function (not a stored closure on a module) so the QNode object is
    easy to introspect.  ``interface='torch'`` + ``diff_method='backprop'`` is
    the only analytic path for ``default.qubit`` with TorchLayer.
    """
    _require_pennylane()
    n = int(n_qubits)
    dev = qml.device(device_name, wires=n, shots=shots)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qnn_circuit(features, weights):
        qml.AmplitudeEmbedding(
            features=features, wires=range(n), normalize=True, pad_with=0.0
        )
        qml.StronglyEntanglingLayers(weights, wires=range(n))
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return qnn_circuit


class AmplitudeQNNLayer(nn.Module):
    """Differentiable 10-qubit (configurable) amplitude-encoding QNN layer.

    ``forward`` maps ``(B, 2**n_qubits)`` CPU-or-GPU tensors to ``(B, n_qubits)``
    Pauli-Z expectation values in ``[-1, 1]``, cast back to the input dtype and
    device.  The QNode itself is always executed on CPU.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        *,
        device_name: str = "default.qubit",
        shots: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        _require_pennylane()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.device_name = str(device_name)
        self.shots = shots

        self.qnode = _canonical_qnn_circuit(self.n_qubits, shots, self.device_name)

        # TorchLayer draws its initial weights from numpy's global RNG.  Seed it
        # locally and restore its state so construction is deterministic without
        # perturbing the caller's RNG.
        rng_state = np.random.get_state()
        np.random.seed(int(seed))
        try:
            self.layer = qml.qnn.TorchLayer(
                self.qnode, {"weights": (self.n_layers, self.n_qubits, 3)}
            )
        finally:
            np.random.set_state(rng_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != 2 ** self.n_qubits:
            raise ValueError(
                f"AmplitudeQNNLayer expects (B, {2 ** self.n_qubits}) features, "
                f"got {tuple(x.shape)}."
            )
        # Keep the autograd graph: .cpu() copies but does not detach.
        out = self.layer(x.cpu())
        return out.to(dtype=x.dtype, device=x.device)


class LegacyQNN(nn.Module):
    """Labelled reproduction of ``Custom_Legacy_QNN_Upgraded``.

    A wide linear projection of the flattened image -> L2 -> QNN -> 2-logit
    head.  With 128x128 inputs the first layer is ``49152 -> 1024`` (~50M
    parameters); it is retained as a compatibility baseline only.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        device_name: str = "default.qubit",
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        run_seed = int(cfg.seed if seed is None else seed)
        self.n_qubits = int(cfg.n_qubits)
        self.feature_dim = int(cfg.feature_dim)
        self.n_layers = int(cfg.qnn_layers)
        self.ablation = False
        self.label = "legacy_qnn"

        self.projection = nn.Linear(
            3 * int(cfg.image_size) * int(cfg.image_size), self.feature_dim
        )
        self.qnn = AmplitudeQNNLayer(
            self.n_qubits, self.n_layers,
            device_name=device_name, shots=shots, seed=run_seed,
        )
        self.head = nn.Linear(self.n_qubits, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = torch.flatten(x, 1)
        f = F.normalize(self.projection(f), p=2, dim=1, eps=_EPS)
        return self.head(self.qnn(f))


class FromScratchHybrid(nn.Module):
    """Small CNN -> L2 projection -> QNN -> 2-logit head (from-scratch hybrid).

    Faithful, weights-free reproduction of ``HybridCNNQNN_Upgraded``: three
    conv stages with batch norm, adaptive average pooling to 128 channels, then
    a ``128 -> cfg.feature_dim`` projection.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        device_name: str = "default.qubit",
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        run_seed = int(cfg.seed if seed is None else seed)
        self.n_qubits = int(cfg.n_qubits)
        self.feature_dim = int(cfg.feature_dim)
        self.n_layers = int(cfg.qnn_layers)
        self.ablation = False
        self.label = "hybrid_from_scratch"

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(128, self.feature_dim)
        self.qnn = AmplitudeQNNLayer(
            self.n_qubits, self.n_layers,
            device_name=device_name, shots=shots, seed=run_seed,
        )
        self.head = nn.Linear(self.n_qubits, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        f = torch.flatten(f, 1)
        f = F.normalize(self.projection(f), p=2, dim=1, eps=_EPS)
        return self.head(self.qnn(f))


class TransferHybrid(nn.Module):
    """Canonical hybrid: frozen EfficientNet-B0 -> L2 -> QNN -> 2-logit head."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        device_name: str = "default.qubit",
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        run_seed = int(cfg.seed if seed is None else seed)
        self.n_qubits = int(cfg.n_qubits)
        self.feature_dim = int(cfg.feature_dim)
        self.n_layers = int(cfg.qnn_layers)
        self.ablation = False
        self.label = "hybrid_transfer"

        # Local import avoids a torchvision->classical circular dependency at
        # module import time (models/classical.py never imports this module).
        from models.classical import build_efficientnet_features

        self.extractor = build_efficientnet_features(cfg)
        self.qnn = AmplitudeQNNLayer(
            self.n_qubits, self.n_layers,
            device_name=device_name, shots=shots, seed=run_seed,
        )
        self.head = nn.Linear(self.n_qubits, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.extractor(x)  # already L2-normalised, (B, cfg.feature_dim)
        return self.head(self.qnn(f))


# --------------------------------------------------------------------------- builders

def build_legacy_qnn(
    cfg: ExperimentConfig,
    *,
    device_name: str = "default.qubit",
    shots: int | None = None,
    seed: int | None = None,
) -> LegacyQNN:
    """Build the legacy (flat-projection) QNN baseline."""
    return LegacyQNN(cfg, device_name=device_name, shots=shots, seed=seed)


def build_hybrid_from_scratch(
    cfg: ExperimentConfig,
    *,
    device_name: str = "default.qubit",
    shots: int | None = None,
    seed: int | None = None,
) -> FromScratchHybrid:
    """Build the from-scratch CNN + QNN hybrid."""
    return FromScratchHybrid(cfg, device_name=device_name, shots=shots, seed=seed)


def build_hybrid_transfer(
    cfg: ExperimentConfig,
    *,
    device_name: str = "default.qubit",
    shots: int | None = None,
    seed: int | None = None,
) -> TransferHybrid:
    """Build the canonical EfficientNet-transfer + QNN hybrid."""
    return TransferHybrid(cfg, device_name=device_name, shots=shots, seed=seed)


def build_4qubit_ablation(
    cfg: ExperimentConfig,
    *,
    device_name: str = "default.qubit",
    shots: int | None = None,
    seed: int | None = None,
) -> FromScratchHybrid:
    """Build an **explicitly labelled** 4-qubit / 16-feature ablation.

    This is not a headline model.  It shares the exact circuit body and the
    from-scratch hybrid architecture; only the config is overridden to
    ``n_qubits=4, feature_dim=2**4=16``.  The returned model carries
    ``ablation=True`` and ``label='q4_embamplitude_L2_ablation'`` so it can be
    filtered out of headline tables (QC-D03 §4.3).
    """
    q4_cfg = replace(cfg, n_qubits=4, feature_dim=16)
    model = FromScratchHybrid(
        q4_cfg, device_name=device_name, shots=shots, seed=seed
    )
    model.ablation = True
    model.label = "q4_embamplitude_L2_ablation"
    model.n_qubits = 4
    model.feature_dim = 16
    return model


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
            circuit = getattr(module, "circuit", None)
            circuit = circuit if circuit is not None else getattr(module, "qnode", None)
            if isinstance(circuit, qml.QNode) and id(circuit) not in seen:
                seen.add(id(circuit))
                yield circuit
    circuit = getattr(obj, "circuit", None)
    circuit = circuit if circuit is not None else getattr(obj, "qnode", None)
    if isinstance(circuit, qml.QNode) and id(circuit) not in seen:
        seen.add(id(circuit))
        yield circuit


def _construct_tape(qnode: Any):
    """Construct (do not execute) a tape for a QNode using shape-valid dummies."""
    n = len(qnode.device.wires)
    n_features = 2 ** n
    func = getattr(qnode, "func", None) or getattr(qnode, "qfunc", None)
    if func is None:
        raise RuntimeError("Cannot introspect a QNode without its qfunc.")
    params = []
    for name in inspect.signature(func).parameters:
        if "weight" in name.lower():
            params.append(np.zeros((1, n, 3), dtype=np.float64))
        else:
            params.append(np.ones(n_features, dtype=np.float64))
    try:
        return qml.workflow.construct_tape(qnode)(*params)
    except Exception:
        # Fallback for older PennyLane releases.
        with qml.tape.QuantumTape() as tape:
            func(*params)
        return tape


# Only *entangling ansatz templates* are expanded for the gate count.  State
# preparation (AmplitudeEmbedding / StatePrep) is deliberately left opaque: it
# is the parameter-free data-loading primitive, not a trainable entangler, and
# QC-D01 C5 defines the kernel circuit as having zero entangling gates.
_EXPANDABLE_TEMPLATES = frozenset(
    {
        "StronglyEntanglingLayers",
        "BasicEntanglerLayers",
        "RealAmplitudes",
        "EfficientSU2",
        "TwoLocal",
    }
)


def _expand_operations(operations, max_depth: int = 4, max_ops: int = 20000):
    """Expand entangling ansatz templates so their CNOTs become countable.

    Non-entangling/template ops (in particular ``AmplitudeEmbedding`` and its
    adjoint) are kept verbatim and never decomposed into state-preparation
    networks, which would otherwise create false entangling-gate counts.
    """
    ops = list(operations)
    for _ in range(max_depth):
        expanded: list[Any] = []
        changed = False
        for op in ops:
            if op.name in _EXPANDABLE_TEMPLATES:
                decomposition = None
                try:
                    decomposition = op.decomposition()
                except Exception:
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


def entangling_gate_count(circuit: Any) -> int:
    """Count multi-qubit (entangling) gates in a QNode or quantum nn.Module.

    Templates such as ``StronglyEntanglingLayers`` are decomposed first, so the
    returned number is the count of entangling *gates* (e.g. CNOT), not the
    number of template layers.  A bare amplitude-embedding kernel circuit has
    zero entangling gates; any QNN has at least one.
    """
    _require_pennylane()
    count = 0
    for qnode in _iter_qnodes(circuit):
        tape = _construct_tape(qnode)
        for op in _expand_operations(tape.operations):
            if op.name in ENTANGLING_GATE_NAMES:
                count += 1
    return count


def trainable_param_count(model: Any) -> int:
    """Count trainable scalar parameters in a model.

    For ``nn.Module`` objects this is ``sum(p.numel() for p in parameters() if
    p.requires_grad)``.  For a bare QNode (e.g. the kernel circuit) it reports
    the number of trainable tape parameters, which is zero for a parameter-free
    feature map.
    """
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    if qml is not None and isinstance(model, qml.QNode):
        # A circuit declared with interface=None (the QSVC kernel) has no
        # trainable parameters by construction.
        if getattr(model, "interface", None) is None:
            return 0
        try:
            return len(_construct_tape(model).trainable_params)
        except Exception:
            return 0
    return 0
