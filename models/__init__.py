"""Model layer for the weed-classification rebuild.

Public surface is assembled by mother after QC-I04 integration. Importing this
package must remain safe when PennyLane is not installed; quantum constructors
raise a clear error only when actually used.
"""
from .classical import (
    CLASSICAL_MODEL_NAMES,
    CustomLegacyCNN,
    EfficientNetFeatures,
    build_classical_model,
    build_efficientnet_features,
)
from .quantum import (
    AmplitudeQNNLayer,
    LegacyQNN,
    FromScratchHybrid,
    TransferHybrid,
    build_legacy_qnn,
    build_hybrid_from_scratch,
    build_hybrid_transfer,
    build_4qubit_ablation,
    entangling_gate_count,
    trainable_param_count,
)
from .qsvc import (
    QSVCWrapper,
    build_kernel_circuit,
    classical_kernel,
    kernel_equivalence_error,
    quantum_kernel,
)

__all__ = [
    "CLASSICAL_MODEL_NAMES",
    "CustomLegacyCNN",
    "EfficientNetFeatures",
    "build_classical_model",
    "build_efficientnet_features",
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
    "QSVCWrapper",
    "build_kernel_circuit",
    "classical_kernel",
    "kernel_equivalence_error",
    "quantum_kernel",
]
