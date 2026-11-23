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
from .losses import (
    as_two_logits,
    class_priors_from_labels,
    class_weights_from_labels,
    focal_loss,
    weighted_cross_entropy,
)
from .heads import (
    ClassificationHead,
    build_classification_head,
    collect_logits,
    find_classification_head,
    fit_calibration,
    fit_logit_adjustment,
    fit_temperature,
    write_calibration_to_checkpoint,
)
from .plant_backbone import (
    PLANTNET_CLASSES,
    PLANTNET_FEATURE_DIM,
    PLANTNET_REPO_ID,
    PLANTNET_WEIGHTS_FILE,
    build_backbone_extractor,
    build_plantnet_extractor,
    plantnet_preprocess,
)

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
    "PLANTNET_REPO_ID",
    "PLANTNET_WEIGHTS_FILE",
    "PLANTNET_FEATURE_DIM",
    "PLANTNET_CLASSES",
    "build_plantnet_extractor",
    "plantnet_preprocess",
    "build_backbone_extractor",
    "class_weights_from_labels",
    "class_priors_from_labels",
    "weighted_cross_entropy",
    "focal_loss",
    "as_two_logits",
    "ClassificationHead",
    "build_classification_head",
    "find_classification_head",
    "collect_logits",
    "fit_temperature",
    "fit_logit_adjustment",
    "fit_calibration",
    "write_calibration_to_checkpoint",
]
