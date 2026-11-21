"""Layer-0 config contract: environment overrides and validation messages.

Verifies the acceptance criterion from QC-D01 §4 (Layer 0): with both NINJA_*
paths set, ``from_env().validate() == []``; without them a single combined
error naming both variables is returned; and a feature-dimension mismatch is
reported independently.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow both ``python -m pytest`` and a bare ``pytest`` invocation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.config import ExperimentConfig

ANN = r"C:\data\ninja\annotations"
IMG = r"C:\data\ninja\images"


def _clear_ninja_env(monkeypatch) -> None:
    monkeypatch.delenv("NINJA_ANN_DIR", raising=False)
    monkeypatch.delenv("NINJA_IMG_DIR", raising=False)


def test_from_env_with_paths_validates_clean(monkeypatch) -> None:
    _clear_ninja_env(monkeypatch)
    monkeypatch.setenv("NINJA_ANN_DIR", ANN)
    monkeypatch.setenv("NINJA_IMG_DIR", IMG)
    monkeypatch.setenv("QC_SEED", "7")
    monkeypatch.setenv("QC_OUTPUT_DIR", "custom_results")

    cfg = ExperimentConfig.from_env()

    assert cfg.seed == 7
    assert cfg.output_dir == "custom_results"
    assert cfg.ninja_ann_dir == ANN
    assert cfg.ninja_img_dir == IMG
    assert cfg.validate() == []


def test_missing_ninja_paths_give_one_combined_error(monkeypatch) -> None:
    _clear_ninja_env(monkeypatch)
    problems = ExperimentConfig.from_env().validate()
    assert len(problems) == 1
    assert "NINJA_ANN_DIR" in problems[0]
    assert "NINJA_IMG_DIR" in problems[0]


def test_single_missing_path_still_names_both_variables() -> None:
    cfg = ExperimentConfig(ninja_ann_dir=ANN, ninja_img_dir="")
    problems = cfg.validate()
    assert len(problems) == 1
    assert "NINJA_ANN_DIR" in problems[0]
    assert "NINJA_IMG_DIR" in problems[0]


def test_feature_dim_mismatch_produces_dimension_error() -> None:
    cfg = ExperimentConfig(ninja_ann_dir=ANN, ninja_img_dir=IMG, feature_dim=512)
    problems = cfg.validate()
    assert len(problems) == 1
    assert "feature_dim" in problems[0]
    assert "2**n_qubits" in problems[0]


def test_default_feature_dim_matches_qubit_count() -> None:
    cfg = ExperimentConfig()
    assert cfg.feature_dim == 2 ** cfg.n_qubits
    assert cfg.feature_dim == 1024


def test_missing_paths_and_dim_mismatch_are_independent_errors() -> None:
    cfg = ExperimentConfig(feature_dim=512)
    problems = cfg.validate()
    assert len(problems) == 2
