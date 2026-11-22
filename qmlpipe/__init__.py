"""Quantum ML training pipeline for the weed-classification rebuild.

Mother-owned integration module. Imports are safe when PennyLane is absent;
quantum constructors raise a clear error only when actually used.
"""
from .config import QMLConfig
from .models import (
    build_feature_extractor,
    build_qnn,
    build_transfer_qnn,
    circuit_introspection,
)
from .kernel import (
    build_qsvc,
    classical_kernel_matrix,
    kernel_diagnostics,
    quantum_kernel_matrix,
)
from .train import evaluate, seed_everything, train
from .evaluate import compare_models, evaluate_qsvc
from .ablate import ablation_grid, run_ablation, summarize_ablation
from .model_card import build_model_card

__all__ = [
    "QMLConfig",
    "build_feature_extractor",
    "build_qnn",
    "build_transfer_qnn",
    "circuit_introspection",
    "quantum_kernel_matrix",
    "classical_kernel_matrix",
    "kernel_diagnostics",
    "build_qsvc",
    "seed_everything",
    "train",
    "evaluate",
    "evaluate_qsvc",
    "compare_models",
    "ablation_grid",
    "run_ablation",
    "summarize_ablation",
    "build_model_card",
]
