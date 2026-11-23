"""Registry of external plant/weed datasets brought into the rebuild.

This module is the single source of truth for which external datasets are known,
where they come from, their licence status and how they are cached locally.
It performs no network access and imports only the standard library, so it is
safe to use from tests and from the central verifier's Python 3.14 environment.

Acquisition is handled by ``scripts/fetch_external_datasets.py``; downloaded bytes
live outside the repository under ``~/.cache/qc_datasets`` and are never committed.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

__all__ = [
    "ExternalDataset",
    "REGISTRY",
    "get_dataset",
    "list_datasets",
    "cache_root",
    "local_path",
    "write_manifest",
    "read_manifest",
]

DEFAULT_CACHE_ENV = "QC_EXTERNAL_DATA_DIR"


@dataclass(frozen=True)
class ExternalDataset:
    """One externally hosted dataset with provenance and acquisition metadata."""

    id: str
    name: str
    task: str
    n_images: int | None
    classes: str
    license: str
    license_status: str  # verified | unverified | not_stated | per_dataset
    sources: tuple[str, ...]
    artifact: str  # file/dir name in the local cache
    format: str
    size_bytes: int | None
    status: str  # fetchable | deferred_large | manual_access | index
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sources"] = list(self.sources)
        return d


REGISTRY: tuple[ExternalDataset, ...] = (
    ExternalDataset(
        id="acre",
        name="ACRE Crop-Weed Dataset",
        task="instance segmentation (crop vs weed)",
        n_images=1000,
        classes="maize, bean vs ryegrass, mustard, matricaria, lamb's quarter",
        license="CC BY 4.0",
        license_status="verified",
        sources=(
            "https://zenodo.org/records/8102217",
            "https://doi.org/10.5281/zenodo.8102217",
        ),
        artifact="acre/The_ACRE_Crop-Weed_Dataset.zip",
        format="zip (RGB + XML instance masks)",
        size_bytes=1_220_608_171,
        status="fetchable",
        notes="1.2 GB. 2046x1080 robot-acquired maize/bean imagery.",
    ),
    ExternalDataset(
        id="deepweeds",
        name="DeepWeeds",
        task="image classification",
        n_images=17509,
        classes="8 Australian weed species + negative class",
        license="CC BY 4.0 (data), Apache-2.0 (code)",
        license_status="verified",
        sources=(
            "https://github.com/AlexOlsen/DeepWeeds",
            "https://www.tensorflow.org/datasets/catalog/deep_weeds",
            "https://huggingface.co/datasets/Project-AgML/DeepWeeds_classification",
        ),
        artifact="deepweeds/data/train-00000-of-00001.parquet",
        format="parquet (256x256 RGB, 9-class labels)",
        size_bytes=492_631_617,
        status="fetchable",
        notes="TFDS v3.0.0 fixes v1 labels. Secondary catalogs conflict on CC BY vs CC BY-SA; original repo governs.",
    ),
    ExternalDataset(
        id="maizeset",
        name="MaizeSet",
        task="image classification (maize vs weed)",
        n_images=3330,
        classes="maize plants vs weeds",
        license="not stated in repository",
        license_status="not_stated",
        sources=("https://github.com/mshahabalam/MaizeSet",),
        artifact="maizeset/",
        format="14 image zips + 14 label zips",
        size_bytes=271_000_000,
        status="fetchable",
        notes="License gap: research use only until the author confirms terms. Closest same-task match.",
    ),
    ExternalDataset(
        id="cornweed_full",
        name="CornWeed Dataset (full, annotated)",
        task="object detection (maize vs weed)",
        n_images=3574,
        classes="maize, weed",
        license="CC BY 4.0 (paper); check record",
        license_status="unverified",
        sources=(
            "https://zenodo.org/records/7961764",
            "https://weed-ai.sydney.edu.au/datasets/aec20fe7-f607-4a4c-aca5-d0f19c7c755f",
        ),
        artifact="cornweed_full/maize_dataset.zip",
        format="zip (YOLO + COCO boxes)",
        size_bytes=8_693_397_534,
        status="deferred_large",
        notes="8.7 GB. Likely superset of the Kaggle CornWeed subset currently used.",
    ),
    ExternalDataset(
        id="cottonweed12",
        name="CottonWeedDet12",
        task="object detection (12 weed species)",
        n_images=5648,
        classes="12 cotton-field weed species",
        license="check Zenodo record",
        license_status="unverified",
        sources=(
            "https://zenodo.org/records/7535814",
            "https://weed-ai.sydney.edu.au/datasets/2c14915b-0827-4b65-9908-d2a6df0d48f3",
        ),
        artifact="cottonweed12/CottonWeedDet12.7z",
        format="7z archive",
        size_bytes=28_586_468_345,
        status="deferred_large",
        notes="28.6 GB; needs py7zr or 7-Zip to extract. Measured download ~0.5 MB/s.",
    ),
    ExternalDataset(
        id="weed25",
        name="Weed25",
        task="image classification (25 weed species)",
        n_images=14035,
        classes="25 weed species, monocot + dicot",
        license="check Frontiers paper",
        license_status="unverified",
        sources=(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC9748680/",
            "https://github.com/yinandkang/Weed25-a-dataset-for-deep-learning",
            "https://pan.baidu.com/s/1rnUoDm7IxxmX1n1LmtXNXw (password rn5h)",
            "https://universe.roboflow.com/weed25-dataset/weed25-labeling/dataset/2",
        ),
        artifact="weed25/",
        format="species folders / Roboflow mirror",
        size_bytes=None,
        status="manual_access",
        notes="GitHub file is an empty LFS pointer; canonical copy is Baidu Netdisk. Roboflow mirror has 9,553 images.",
    ),
    ExternalDataset(
        id="foodcropsweeds",
        name="Dataset of Annotated Food Crops and Weed Images",
        task="object detection (crop vs weed)",
        n_images=1176,
        classes="8 weed + 6 crop species; annotations collapsed to weed/crop",
        license="check Data in Brief paper (DOI 10.1016/j.dib.2020.105833)",
        license_status="unverified",
        sources=(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC7305380",
            "https://datasetninja.com/dataset-of-annotated-food-crops-and-weed-images",
            "https://platform.ultralytics.com/ultralytics/datasets/dataset-of-annotated-food-crops-and-weed-images",
        ),
        artifact="foodcropsweeds/",
        format="JPEG + XML annotations",
        size_bytes=None,
        status="manual_access",
        notes="Greenhouse and open-field imagery; paper reports 1,176 images / 7,853 boxes.",
    ),
    ExternalDataset(
        id="weedai",
        name="Weed-AI",
        task="dataset aggregator (MS-COCO derived)",
        n_images=None,
        classes="many; per-dataset",
        license="per dataset",
        license_status="per_dataset",
        sources=(
            "https://weed-ai.sydney.edu.au/datasets/",
            "https://weed-ai.sydney.edu.au/datasets/aec20fe7-f607-4a4c-aca5-d0f19c7c755f",
            "https://weed-ai.sydney.edu.au/datasets/2c14915b-0827-4b65-9908-d2a6df0d48f3",
        ),
        artifact="weedai/index.json",
        format="index of hosted datasets",
        size_bytes=None,
        status="index",
        notes="Aggregator, not a single download. Individual records registered above where applicable.",
    ),
)

_BY_ID = {d.id: d for d in REGISTRY}


def get_dataset(dataset_id: str) -> ExternalDataset:
    try:
        return _BY_ID[dataset_id]
    except KeyError as exc:
        raise KeyError(f"unknown external dataset {dataset_id!r}; known: {sorted(_BY_ID)}") from exc


def list_datasets(status: str | None = None) -> tuple[ExternalDataset, ...]:
    if status is None:
        return REGISTRY
    return tuple(d for d in REGISTRY if d.status == status)


def cache_root(root: str | os.PathLike[str] | None = None) -> Path:
    if root is not None:
        return Path(root).expanduser()
    env = os.getenv(DEFAULT_CACHE_ENV)
    return Path(env).expanduser() if env else Path.home() / ".cache" / "qc_datasets"


def local_path(dataset: ExternalDataset | str, root: str | os.PathLike[str] | None = None) -> Path:
    d = get_dataset(dataset) if isinstance(dataset, str) else dataset
    return cache_root(root) / d.artifact


def write_manifest(
    dataset: ExternalDataset,
    path: Path,
    *,
    sha256: str | None = None,
    size_bytes: int | None = None,
    source_url: str | None = None,
) -> Path:
    """Write a JSON provenance sidecar next to a fetched artifact."""
    payload = {
        "schema_version": "1.0",
        "dataset": dataset.to_dict(),
        "local_path": str(path),
        "size_bytes": size_bytes,
        "sha256": sha256,
        "source_url": source_url,
    }
    out = path.with_suffix(path.suffix + ".manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def read_manifest(path: Path) -> dict:
    """Read a manifest sidecar; raise if absent or malformed."""
    return json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text(encoding="utf-8"))


def iter_ids(datasets: Iterable[ExternalDataset] | None = None) -> Iterable[str]:
    for d in datasets if datasets is not None else REGISTRY:
        yield d.id
