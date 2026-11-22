"""Synthetic-only tests for the plant-backbone research note (PLANT-D03).

These tests read one Markdown research note from disk. They require no dataset,
checkpoint, model, GPU, network access or quantum install; no real run is
executed. They guard the note's decision record (selected repo id, license,
feature dim, limitations) and keep fabricated legacy run metrics out of it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "PLANT_BACKBONE_RESEARCH.md"

SELECTED_REPO_ID = "cpoisson/plantnet300k-resnet18"
EXPECTED_LICENSE = "Apache-2.0"
EXPECTED_FEATURE_DIM = "512"
SELECTED_FEATURE_DIM_LABEL = "512-d"

# Legacy / unsupported numbers from archived project runs. The research note is
# a design record, not a results artifact, so none of these may appear in it.
FORBIDDEN_RUN_METRICS = ("0.932", "0.4275", "0.804", "0.375")


def _doc_text() -> str:
    assert DOC_PATH.is_file(), f"missing research note: {DOC_PATH}"
    return DOC_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Required structural / content guards
# --------------------------------------------------------------------------
def test_research_doc_exists() -> None:
    assert DOC_PATH.is_file(), f"missing research note: {DOC_PATH}"


def test_doc_contains_selected_repo_id() -> None:
    assert SELECTED_REPO_ID in _doc_text()


def test_doc_contains_license() -> None:
    assert EXPECTED_LICENSE in _doc_text()


def test_doc_contains_feature_dim() -> None:
    text = _doc_text()
    assert EXPECTED_FEATURE_DIM in text
    assert SELECTED_FEATURE_DIM_LABEL in text


def test_doc_has_limitations_section() -> None:
    assert "## 5. Limitations" in _doc_text()


def test_doc_does_not_contain_fabricated_run_metrics() -> None:
    text = _doc_text()
    for forbidden in FORBIDDEN_RUN_METRICS:
        assert forbidden not in text, f"fabricated legacy metric in note: {forbidden}"


# --------------------------------------------------------------------------
# Additional synthetic guards on the decision record
# --------------------------------------------------------------------------
def test_doc_has_acceptance_criteria_section() -> None:
    assert "## 6. Acceptance criteria" in _doc_text()


def test_doc_disclaims_measured_results() -> None:
    assert "no numbers from project runs" in _doc_text().lower()


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/cpoisson/plantnet300k-resnet18",
        "https://github.com/plantnet/PlantNet-300K",
        "https://github.com/Imageomics/BioCLIP",
        "https://huggingface.co/imageomics/bioclip-2",
        "https://github.com/GFZ/weedsgalore",
        "https://arxiv.org/abs/2207.03881",
    ],
)
def test_doc_cites_required_sources(url: str) -> None:
    assert url in _doc_text()
