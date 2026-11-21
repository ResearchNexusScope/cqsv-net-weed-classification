"""Single source of truth for seeds, paths, and hyperparameters.

Paths are taken from environment variables so the code no longer depends on a
specific user's Downloads or kagglehub cache directory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 0.001
    weight_decay: float = 0.0

    # ImageNet normalization used by every model in the benchmark.
    norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Augmentation as stated in the paper and implemented in the SOTA script.
    rotation_degrees: int = 15
    hflip_prob: float = 0.5
    vflip_prob: float = 0.2
    jitter: float = 0.1
    scale_range: tuple[float, float] = (0.9, 1.1)

    # Quantum model settings.
    n_qubits: int = 10
    feature_dim: int = 1024
    qnn_layers: int = 2

    # DatasetNinja adaptation split (per class).
    adaptation_per_class: int = 200

    # QSVC legacy parity: 50 Kaggle source + 50 labelled Ninja adaptation
    # crops = 100 mixed training samples; 50 held-out Ninja crops for test.
    qsvc_source_per_run: int = 50
    qsvc_target_adaptation_per_run: int = 50
    qsvc_test_per_run: int = 50
    qsvc_runs: int = 10

    seed_offset: int = 42

    # Data sources. Override these in the environment on a new machine.
    kaggle_dataset: str = "aminelaatam/weed-classification"
    ninja_ann_dir: str = ""
    ninja_img_dir: str = ""
    output_dir: str = "results"

    @classmethod
    def from_env(cls) -> "ExperimentConfig":
        return cls(
            seed=int(os.getenv("QC_SEED", "42")),
            ninja_ann_dir=os.getenv("NINJA_ANN_DIR", ""),
            ninja_img_dir=os.getenv("NINJA_IMG_DIR", ""),
            output_dir=os.getenv("QC_OUTPUT_DIR", "results"),
        )

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def qsvc_mixed_train_size(self) -> int:
        """Total mixed training samples under the legacy-parity QSVC protocol."""
        return self.qsvc_source_per_run + self.qsvc_target_adaptation_per_run

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems."""
        problems = []
        if not self.ninja_ann_dir or not self.ninja_img_dir:
            problems.append(
                "NINJA_ANN_DIR and NINJA_IMG_DIR are not set; set them to the "
                "annotations and images directories of the DatasetNinja download."
            )
        if self.feature_dim != 2 ** self.n_qubits:
            problems.append(
                f"feature_dim={self.feature_dim} must equal 2**n_qubits="
                f"{2 ** self.n_qubits} for AmplitudeEmbedding."
            )
        if self.epochs <= 0 or self.batch_size <= 0:
            problems.append("epochs and batch_size must be positive.")
        return problems


DEFAULT_CONFIG = ExperimentConfig()
