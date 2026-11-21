"""Leak-free data layer for the Quantum_Classification rebuild.

Public API is re-exported here so callers may use either
``from data import build_splits`` or the specific module
(``data.splits`` / ``data.manifest`` / ``data.transforms``).
"""
from __future__ import annotations

from data.splits import (
    ConfigError,
    DataLayerError,
    DedupReport,
    ImageDataset,
    InsufficientAdaptationCrops,
    LabelConflictError,
    ManifestError,
    QSVCProtocol,
    QSVCRun,
    Sample,
    Source,
    Split,
    SplitRecord,
    build_splits,
    content_hash,
    dedup_kaggle,
    discover_kaggle,
    load_manifest,
    make_dataloaders,
    materialise_adaptation_crops,
    parse_ninja_annotations,
    sample_qsvc_pairs,
    split_kaggle,
    split_ninja,
)
from data.manifest import SplitManifest, canonical_json, compute_checksum
from data.transforms import make_transforms

__all__ = [
    "Source",
    "Split",
    "Sample",
    "SplitRecord",
    "DedupReport",
    "QSVCProtocol",
    "QSVCRun",
    "SplitManifest",
    "DataLayerError",
    "ConfigError",
    "LabelConflictError",
    "InsufficientAdaptationCrops",
    "ManifestError",
    "content_hash",
    "discover_kaggle",
    "dedup_kaggle",
    "parse_ninja_annotations",
    "split_kaggle",
    "split_ninja",
    "build_splits",
    "load_manifest",
    "sample_qsvc_pairs",
    "materialise_adaptation_crops",
    "ImageDataset",
    "make_dataloaders",
    "make_transforms",
    "compute_checksum",
    "canonical_json",
]
