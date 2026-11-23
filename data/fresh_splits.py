"""Fresh, group-disjoint preparation and splitting of three external datasets.

Scope
-----
This module prepares MaizeSet, DeepWeeds and ACRE into a single, uniform
``FreshSample`` shape and produces train/val/test splits that obey the
group-disjoint protocol of ``docs/METHODOLOGY_V2.md`` §2: every crop cut from
one source photograph lives in exactly one fold.

It deliberately does **not** call any feature extractor, backbone or trainer.
Prepared JPEGs are written only under ``<cache>/prepared/`` and the original
cached artefacts are never modified.

Design notes
------------
* ``FreshSample`` duck-types the legacy ``Sample`` consumed by
  ``scripts/run_real_benchmark.py::make_feature_loader`` (it reads
  ``sample_id``, ``bbox`` and ``path``), so no downstream code needs changing.
* ``grouped_split`` groups on ``source_image_id``, shuffles *whole groups* with
  ``numpy.random.default_rng(seed)`` and assigns each group to the split with
  the largest remaining sample deficit. Greedy largest-deficit assignment
  approximates the requested ratios even when groups have unequal sizes.
* A group containing both classes, a split lacking a class and an empty split
  are all correctness failures (leakage / degenerate evaluation) and raise a
  ``ValueError`` rather than being silently accepted.

Only the standard library, numpy and (lazily) Pillow/pyarrow are used.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from data import external

__all__ = [
    "FreshSample",
    "label_from_filename",
    "prepare_dataset",
    "grouped_split",
    "split_provenance",
]

DATASETS = ("maizeset", "deepweeds", "acre")
_SPLIT_NAMES = ("train", "val", "test")

_DEEPWEEDS_WEED_SPECIES = frozenset(
    {
        "Parkinsonia",
        "Parthenium",
        "Chinee apple",
        "Rubber vine",
        "Lantana",
        "Siam weed",
        "Snake weed",
        "Prickly acacia",
    }
)


@dataclass(frozen=True)
class FreshSample:
    """One image-level sample (or one annotated crop) from a fresh dataset.

    ``bbox`` is kept for compatibility with ``make_feature_loader``. Preparing
    ACRE already cuts the crop into its own JPEG, so all prepared samples use
    ``bbox=None`` (the loader must not crop a second time).
    """

    sample_id: str
    source_image_id: str
    label: int  # 0 = crop / non-weed, 1 = weed
    path: str  # absolute path to a JPEG on disk
    bbox: tuple | None = None


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def label_from_filename(stem: str) -> int:
    """Return the binary MaizeSet label from an image filename stem.

    ``maize_*`` maps to 0 (crop), ``weed_*`` maps to 1 (weed). Anything else is
    rejected rather than guessed. The YOLO label files may contain several
    boxes; the image-level label is defined by this filename prefix.
    """
    prefix = str(stem).split("_", 1)[0].lower()
    if prefix == "maize":
        return 0
    if prefix == "weed":
        return 1
    raise ValueError(
        f"unknown MaizeSet filename prefix in {stem!r}; expected 'maize_*' or 'weed_*'"
    )


def _prepared_dir(name: str, root) -> Path:
    return external.cache_root(root) / "prepared" / name


def _write_jpeg(raw: bytes, out: Path) -> None:
    """Write one JPEG, re-encoding to RGB; a non-empty existing file is reused."""
    if out.exists() and out.stat().st_size > 0:
        return
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - defensive, Pillow is required here
        raise RuntimeError("preparing images requires Pillow") from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(raw)) as image:
        image.convert("RGB").save(out, "JPEG", quality=95)


def _deepweeds_label(species: str) -> int:
    """Negative -> 0 (background); each of the 8 weed species -> 1."""
    if species == "Negative":
        return 0
    if species in _DEEPWEEDS_WEED_SPECIES:
        return 1
    raise ValueError(f"unknown DeepWeeds species {species!r}; refusing to guess a label")


# --------------------------------------------------------------------------- #
# Per-dataset preparation (idempotent)
# --------------------------------------------------------------------------- #
def _prepare_maizeset(root) -> list[FreshSample]:
    source_dir = external.cache_root(root) / "maizeset"
    out_dir = _prepared_dir("maizeset", root)
    archives = sorted(source_dir.glob("images_*.zip"))
    if not archives:
        raise FileNotFoundError(f"no MaizeSet 'images_*.zip' archives under {source_dir}")

    samples: list[FreshSample] = []
    for archive in archives:
        namespace = archive.stem  # e.g. images_00 - keeps sample_ids unique across zips
        with zipfile.ZipFile(archive) as zf:
            for info in sorted(zf.infolist(), key=lambda member: member.filename):
                if info.is_dir() or Path(info.filename).suffix.lower() not in (".jpg", ".jpeg"):
                    continue
                stem = Path(info.filename).stem
                label = label_from_filename(stem)
                out = out_dir / f"{namespace}__{stem}.jpg"
                if not (out.exists() and out.stat().st_size > 0):
                    _write_jpeg(zf.read(info), out)
                samples.append(
                    FreshSample(
                        sample_id=out.stem,
                        source_image_id=stem,
                        label=label,
                        path=str(out.resolve()),
                        bbox=None,
                    )
                )
    return samples


def _prepare_deepweeds(root) -> list[FreshSample]:
    parquet_path = external.cache_root(root) / "deepweeds" / "data" / "train-00000-of-00001.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"DeepWeeds parquet not found at {parquet_path}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("reading the DeepWeeds parquet requires pyarrow") from exc

    out_dir = _prepared_dir("deepweeds", root)
    reader = pq.ParquetFile(parquet_path)
    samples: list[FreshSample] = []
    index = 0
    # Stream in small batches; never materialise the whole 17k-row table at once.
    for batch in reader.iter_batches(batch_size=256, columns=["image", "label", "species"]):
        for row in batch.to_pylist():
            label = _deepweeds_label(str(row["species"]))
            image = row["image"]
            raw = image["bytes"] if isinstance(image, dict) else image
            sample_id = f"dw_{index:06d}"
            index += 1
            out = out_dir / f"{sample_id}.jpg"
            if not (out.exists() and out.stat().st_size > 0):
                _write_jpeg(raw, out)
            samples.append(
                FreshSample(
                    sample_id=sample_id,
                    source_image_id=sample_id,
                    label=label,
                    path=str(out.resolve()),
                    bbox=None,
                )
            )
    return samples


def _prepare_acre(root) -> list[FreshSample]:
    archive = external.cache_root(root) / "acre" / "The_ACRE_Crop-Weed_Dataset.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"ACRE archive not found at {archive}")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - defensive, Pillow is required here
        raise RuntimeError("preparing ACRE crops requires Pillow") from exc

    out_dir = _prepared_dir("acre", root)
    samples: list[FreshSample] = []
    with zipfile.ZipFile(archive) as zf:
        coco = json.loads(zf.read("data/ACRE_COCO_annotations.json"))
        file_by_id = {int(image["id"]): str(image["file_name"]) for image in coco["images"]}
        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        for image_id in sorted(anns_by_image):
            file_name = file_by_id.get(image_id)
            if file_name is None:
                raise ValueError(f"ACRE annotation references unknown image_id {image_id}")
            source_stem = Path(file_name).stem
            with Image.open(io.BytesIO(zf.read(file_name))) as image:
                image = image.convert("RGB")
                width, height = image.size
                for ann in sorted(anns_by_image[image_id], key=lambda item: int(item["id"])):
                    ann_id = int(ann["id"])
                    label = 1 if int(ann["category_id"]) == 2 else 0
                    out = out_dir / f"{source_stem}_{ann_id}.jpg"
                    sample = FreshSample(
                        sample_id=out.stem,
                        source_image_id=source_stem,
                        label=label,
                        path=str(out.resolve()),
                        bbox=None,  # the crop is already cut into the saved JPEG
                    )
                    if out.exists() and out.stat().st_size > 0:
                        samples.append(sample)
                        continue
                    x, y, box_w, box_h = ann["bbox"]
                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(width, int(x + box_w))
                    y2 = min(height, int(y + box_h))
                    out.parent.mkdir(parents=True, exist_ok=True)
                    image.crop((x1, y1, x2, y2)).save(out, "JPEG", quality=95)
                    samples.append(sample)
    return samples


def prepare_dataset(name: str, root=None) -> list[FreshSample]:
    """Prepare one of ``maizeset`` / ``deepweeds`` / ``acre``.

    ``root`` overrides the cache root (defaults to ``external.cache_root()``).
    Existing prepared JPEGs are reused, so repeated calls are idempotent.
    """
    key = str(name).strip().lower()
    if key not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    if key == "maizeset":
        return _prepare_maizeset(root)
    if key == "deepweeds":
        return _prepare_deepweeds(root)
    return _prepare_acre(root)


# --------------------------------------------------------------------------- #
# Group-disjoint splitting
# --------------------------------------------------------------------------- #
def grouped_split(
    samples: Sequence[FreshSample],
    seed: int = 42,
    ratios: Sequence[float] = (0.7, 0.2, 0.1),
) -> dict[str, list[FreshSample]]:
    """Split samples into ``train`` / ``val`` / ``test`` without group leakage.

    Groups are keyed by ``source_image_id``. A seeded permutation fixes the
    order in which groups are assigned; each group goes to the split with the
    largest remaining target deficit, so whole groups stay together while the
    resulting *sample* counts approximate ``ratios``.

    Raises ``ValueError`` for: empty input, invalid ratios, fewer than three
    groups, a group that mixes classes, an empty split, or a split that does
    not contain both classes.
    """
    items = list(samples)
    if not items:
        raise ValueError("grouped_split received no samples")

    ratios_tuple = tuple(float(r) for r in ratios)
    if len(ratios_tuple) != 3 or any(r <= 0 for r in ratios_tuple):
        raise ValueError("ratios must be three positive numbers summing to 1")
    if abs(sum(ratios_tuple) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios_tuple}")

    groups: dict[str, list[FreshSample]] = {}
    for sample in items:
        label = int(sample.label)
        if label not in (0, 1):
            raise ValueError(f"sample {sample.sample_id!r} has invalid label {sample.label!r}")
        groups.setdefault(str(sample.source_image_id), []).append(sample)

    for group_id, members in groups.items():
        labels = {int(member.label) for member in members}
        if len(labels) > 1:
            raise ValueError(
                f"source group {group_id!r} mixes classes {sorted(labels)}; "
                "a group must have one class to keep the split leak-free"
            )

    group_ids = sorted(groups)
    if len(group_ids) < len(_SPLIT_NAMES):
        raise ValueError(
            f"need at least {len(_SPLIT_NAMES)} distinct source groups to fill "
            f"train/val/test, got {len(group_ids)}"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(group_ids))
    targets = [len(items) * r for r in ratios_tuple]
    filled = [0, 0, 0]
    assignment: dict[str, int] = {}
    for position in order:
        group_id = group_ids[int(position)]
        split_index = int(np.argmax([targets[j] - filled[j] for j in range(3)]))
        assignment[group_id] = split_index
        filled[split_index] += len(groups[group_id])

    splits: dict[str, list[FreshSample]] = {name: [] for name in _SPLIT_NAMES}
    for sample in items:
        splits[_SPLIT_NAMES[assignment[str(sample.source_image_id)]]].append(sample)

    # Pairwise-disjoint group sets: the core leak-free invariant.
    group_sets = [
        {str(sample.source_image_id) for sample in splits[name]} for name in _SPLIT_NAMES
    ]
    assert not (group_sets[0] & group_sets[1]), "train/val share source groups"
    assert not (group_sets[0] & group_sets[2]), "train/test share source groups"
    assert not (group_sets[1] & group_sets[2]), "val/test share source groups"

    # Full coverage, no duplication.
    assert sum(len(part) for part in splits.values()) == len(items), "split lost samples"

    for name in _SPLIT_NAMES:
        if not splits[name]:
            raise ValueError(f"split {name!r} is empty for these groups/ratios")
        observed = {int(sample.label) for sample in splits[name]}
        if observed != {0, 1}:
            raise ValueError(
                f"split {name!r} does not contain both classes (found {sorted(observed)}); "
                "cannot evaluate a binary classifier on this split"
            )
    return splits


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def split_provenance(
    name: str,
    samples: Sequence[FreshSample],
    splits: dict[str, list[FreshSample]],
    seed: int,
    ratios: Sequence[float],
) -> dict:
    """Summarise a split and hash its canonical (order-invariant) manifest."""
    records = [
        {
            "sample_id": str(sample.sample_id),
            "source_image_id": str(sample.source_image_id),
            "label": int(sample.label),
            "path": str(sample.path),
            "bbox": list(sample.bbox) if sample.bbox is not None else None,
        }
        for sample in samples
    ]
    records.sort(key=lambda record: record["sample_id"])
    manifest_bytes = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    split_of: dict[str, str] = {}
    for split_name in _SPLIT_NAMES:
        for sample in splits.get(split_name, []):
            split_of[str(sample.sample_id)] = split_name
    assignment = sorted(
        [str(sample.sample_id), split_of.get(str(sample.sample_id), "unassigned")]
        for sample in samples
    )
    assignment_bytes = json.dumps(assignment, separators=(",", ":")).encode("utf-8")

    counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for split_name in _SPLIT_NAMES:
        part = splits.get(split_name, [])
        counts[split_name] = len(part)
        group_counts[split_name] = len({str(sample.source_image_id) for sample in part})
        class_count = {"0": 0, "1": 0}
        for sample in part:
            key = str(int(sample.label))
            class_count[key] = class_count.get(key, 0) + 1
        class_counts[split_name] = class_count

    return {
        "schema_version": "1.0",
        "dataset": str(name),
        "seed": int(seed),
        "ratios": [float(r) for r in ratios],
        "counts": counts,
        "group_counts": group_counts,
        "class_counts": class_counts,
        "manifest_sha256": manifest_sha256,
        "split_assignment_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
    }
