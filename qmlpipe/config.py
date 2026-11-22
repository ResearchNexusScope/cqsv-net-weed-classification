"""Configuration for the QML classification pipeline (QML-D01).

``QMLConfig`` is the single source of truth for the rebuilt quantum-classical
pipeline: qubit count, ansatz/embedding choice, devices, optimisation
hyper-parameters and output paths.  It deliberately contains **no module-level
qubit constants** -- every consumer reads its widths from a config instance, so
the 4/8/10-qubit ablations are pure parameter changes (QC-D03 §4).

Design rules
------------
* ``feature_dim`` is always ``2 ** qubits`` (AmplitudeEmbedding contract).
* ``validate()`` hard-asserts ``projection_dim == 2 ** qubits``; an invalid
  config fails loudly at construction time of an experiment, never silently.
* ``embedding="angle"`` is an *ablation*: its QNN consumes ``qubits`` inputs.
  ``qnn_input_dim`` exposes that distinction while ``feature_dim`` remains
  ``2 ** qubits`` per contract.  A transfer model inserts an adapter in that
  case (see ``qmlpipe.models.build_transfer_qnn``).
* The module imports only the standard library, so it is safe under any Python
  version (including the mother's Python 3.14 verifier without PennyLane).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

__all__ = ["QMLConfig"]


_ALLOWED_EMBEDDINGS = frozenset({"amplitude", "angle"})
_ALLOWED_ANSATZE = frozenset({"strongly_entangling", "basic_entangler"})
_ALLOWED_DEVICES = frozenset({"default.qubit", "lightning.qubit"})
_ALLOWED_OPTIMIZERS = frozenset({"adamw", "adam", "sgd", "spsa"})


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else float(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else raw


def _env_opt_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "" or raw.strip().lower() == "none":
        return None
    return int(raw)


def _env_opt_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "" or raw.strip().lower() == "none":
        return None
    return float(raw)


@dataclass(frozen=True)
class QMLConfig:
    """Frozen, serialisable configuration for the QML pipeline.

    Attributes mirror the rebuild's canonical settings (seed 42, 10 qubits,
    2 entangling layers, 1024 projected features, AdamW, early stopping).
    """

    seed: int = 42
    qubits: int = 10
    layers: int = 2
    embedding: str = "amplitude"
    ansatz: str = "strongly_entangling"
    device_name: str = "default.qubit"
    shots: int | None = None
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 5
    optimizer: str = "adamw"
    classical_dim: int = 1280
    projection_dim: int = 1024
    noise_depolarizing: float | None = None
    output_dir: str = "results/qml"

    # ------------------------------------------------------------------ properties
    @property
    def feature_dim(self) -> int:
        """Amplitude-embedding input width, ``2 ** qubits`` (contract)."""
        return 2 ** int(self.qubits)

    @property
    def qnn_input_dim(self) -> int:
        """Width of the tensor fed to the QNode's embedding.

        ``qubits`` for the angle-embedding ablation, otherwise ``feature_dim``.
        """
        return int(self.qubits) if self.embedding == "angle" else self.feature_dim

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_env(cls) -> "QMLConfig":
        """Build a config from ``QML_*`` environment variables.

        Unset variables fall back exactly to the dataclass defaults, so
        ``QMLConfig.from_env()`` is valid with a clean environment.
        """
        return cls(
            seed=_env_int("QML_SEED", 42),
            qubits=_env_int("QML_QUBITS", 10),
            layers=_env_int("QML_LAYERS", 2),
            embedding=_env_str("QML_EMBEDDING", "amplitude").lower(),
            ansatz=_env_str("QML_ANSATZ", "strongly_entangling").lower(),
            device_name=_env_str("QML_DEVICE", "default.qubit"),
            shots=_env_opt_int("QML_SHOTS"),
            batch_size=_env_int("QML_BATCH_SIZE", 32),
            epochs=_env_int("QML_EPOCHS", 20),
            learning_rate=_env_float("QML_LR", 0.001),
            weight_decay=_env_float("QML_WEIGHT_DECAY", 1e-4),
            patience=_env_int("QML_PATIENCE", 5),
            optimizer=_env_str("QML_OPTIMIZER", "adamw").lower(),
            classical_dim=_env_int("QML_CLASSICAL_DIM", 1280),
            projection_dim=_env_int("QML_PROJECTION_DIM", 1024),
            noise_depolarizing=_env_opt_float("QML_NOISE_DEPOLARIZING"),
            output_dir=_env_str("QML_OUTPUT_DIR", "results/qml"),
        )

    # ------------------------------------------------------------------ serialisation
    def as_dict(self) -> dict:
        """Return all dataclass fields as a plain ``dict`` (no properties)."""
        return asdict(self)

    def to_dict(self) -> dict:
        """Alias of :meth:`as_dict` for callers preferring the ``to_`` prefix."""
        return self.as_dict()

    # ------------------------------------------------------------------ validation
    def validate(self) -> list[str]:
        """Validate the config and return any (human-readable) problems.

        The amplitude-embedding invariant ``projection_dim == 2 ** qubits`` is a
        hard contract: when it is violated an ``AssertionError`` is raised
        (with the offending values in the message) rather than returning a
        soft problem list.  For a valid config the return value is ``[]``.
        """
        problems: list[str] = []

        if self.projection_dim != self.feature_dim:
            problems.append(
                f"projection_dim={self.projection_dim} must equal 2**qubits="
                f"{self.feature_dim} for AmplitudeEmbedding."
            )
        if int(self.qubits) <= 0:
            problems.append("qubits must be a positive integer.")
        if int(self.layers) <= 0:
            problems.append("layers must be a positive integer.")
        if int(self.batch_size) <= 0 or int(self.epochs) <= 0:
            problems.append("batch_size and epochs must be positive.")
        if int(self.patience) <= 0:
            problems.append("patience must be positive.")
        if self.learning_rate <= 0:
            problems.append("learning_rate must be positive.")
        if self.weight_decay < 0:
            problems.append("weight_decay must be non-negative.")
        if int(self.classical_dim) <= 0:
            problems.append("classical_dim must be positive.")
        if self.embedding not in _ALLOWED_EMBEDDINGS:
            problems.append(
                f"embedding={self.embedding!r} must be one of "
                f"{sorted(_ALLOWED_EMBEDDINGS)}."
            )
        if self.ansatz not in _ALLOWED_ANSATZE:
            problems.append(
                f"ansatz={self.ansatz!r} must be one of {sorted(_ALLOWED_ANSATZE)}."
            )
        if self.device_name not in _ALLOWED_DEVICES:
            problems.append(
                f"device_name={self.device_name!r} must be one of "
                f"{sorted(_ALLOWED_DEVICES)}."
            )
        if self.optimizer not in _ALLOWED_OPTIMIZERS:
            problems.append(
                f"optimizer={self.optimizer!r} must be one of "
                f"{sorted(_ALLOWED_OPTIMIZERS)}."
            )
        if self.shots is not None and int(self.shots) <= 0:
            problems.append("shots must be None (analytic) or a positive integer.")
        if self.noise_depolarizing is not None and not (
            0.0 <= float(self.noise_depolarizing) <= 1.0
        ):
            problems.append("noise_depolarizing must be None or in [0, 1].")

        # Hard contract: never construct a pipeline with a mismatched embedding
        # width.  The AssertionError message names both values for diagnostics.
        assert not problems, "; ".join(problems)
        return problems
