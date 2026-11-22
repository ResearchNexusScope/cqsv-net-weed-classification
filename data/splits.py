"""Leak-free data layer: discovery, grouping, splitting and QSVC pairing.

Implementation of the accepted design spec ``docs/research/QC-D02_data_pipeline.md``.

The two group keys are deliberately derived from the *source photograph*:

* Kaggle -- ``md5(raw file bytes)`` (paper parity, see ``content_hash``).
* DatasetNinja -- ``Path(img_path).stem``; every box cut from one photo shares
  this group, so adaptation and OOD test can never share a photograph.

No image is ever copied into two logical splits; the only materialised files
are the adaptation crops (see ``materialise_adaptation_crops``), written by an
explicit caller after ``build_splits`` has returned its immutable manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import struct
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from numpy.random import SeedSequence
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from experiments.config import ExperimentConfig

__all__ = [
    "Source",
    "Split",
    "Sample",
    "SplitRecord",
    "DedupReport",
    "QSVCProtocol",
    "QSVCRun",
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
]

LABEL_NAMES = {0: "Corn", 1: "Weed"}
CLASS_TITLES = {"maize": 0, "weed": 1}
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png"})
_CHUNK = 1024 * 1024


# ---------------------------------------------------------------------------
# Value types (normative API from spec section 1.1)
# ---------------------------------------------------------------------------
class Source(str, Enum):
    KAGGLE = "kaggle"
    NINJA = "ninja"


class Split(str, Enum):
    TRAIN = "train"
    ID_VAL = "id_val"
    ID_TEST = "id_test"
    ADAPTATION = "adaptation"
    OOD_TEST = "ood_test"


class QSVCProtocol(str, Enum):
    MIXED = "mixed"
    NO_MIXING = "no_mixing_control"


@dataclass(frozen=True)
class Sample:
    """One trainable/evaluable unit. For Ninja, one bounding-box crop."""

    sample_id: str
    source: Source
    group_id: str
    content_hash: str
    path: str
    bbox: tuple[int, int, int, int] | None
    label: int
    label_name: str


@dataclass(frozen=True)
class SplitRecord:
    name: Split
    samples: tuple[Sample, ...]

    @property
    def sample_ids(self) -> frozenset[str]:
        return frozenset(s.sample_id for s in self.samples)

    @property
    def groups(self) -> frozenset[str]:
        return frozenset(s.group_id for s in self.samples)

    @property
    def hashes(self) -> frozenset[str]:
        return frozenset(s.content_hash for s in self.samples)

    @property
    def paths(self) -> frozenset[str]:
        return frozenset(s.path for s in self.samples)

    @property
    def count(self) -> int:
        return len(self.samples)

    def class_counts(self) -> dict[str, int]:
        counts = {"Corn": 0, "Weed": 0}
        for s in self.samples:
            counts[s.label_name] = counts.get(s.label_name, 0) + 1
        return counts


@dataclass(frozen=True)
class DedupReport:
    files_seen: int
    unique_hashes: int
    duplicates_removed: int
    label_conflicts: tuple[str, ...]


@dataclass(frozen=True)
class QSVCRun:
    """One arm of one of the ten paired QSVC runs (spec section 3.3)."""

    run_idx: int
    protocol: QSVCProtocol
    seed: int
    source_train: tuple[Sample, ...]
    target_train: tuple[Sample, ...]
    test: tuple[Sample, ...]

    @property
    def train(self) -> tuple[Sample, ...]:
        """Always exactly ``cfg.qsvc_mixed_train_size`` (100) samples."""
        return self.source_train + self.target_train


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class DataLayerError(RuntimeError):
    """Base class for fail-loud data-layer violations."""


class ConfigError(DataLayerError):
    """Configuration or source-root problem."""


class LabelConflictError(DataLayerError):
    """One content hash was observed with two different labels."""

    def __init__(self, message: str, conflicts: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.conflicts = tuple(conflicts)


class InsufficientAdaptationCrops(DataLayerError):
    """The 200/200 whole-source adaptation quota cannot be satisfied."""

    def __init__(self, message: str, observed: Mapping[int, int] | None = None) -> None:
        super().__init__(message)
        self.observed = dict(observed or {})


class ManifestError(DataLayerError):
    """A freshly built manifest failed its own ``verify()`` gate."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _record(name: Split, samples: Iterable[Sample]) -> SplitRecord:
    """Construct a SplitRecord with a canonical (sample_id-sorted) sample order.

    The canonical order is what the manifest serialises as ``sample_ids`` and
    what ``from_json`` uses to reconstruct the record; keeping it here makes
    the roundtrip exact.
    """
    return SplitRecord(name=name, samples=tuple(sorted(samples, key=lambda s: s.sample_id)))


def _assert_disjoint(train: Sequence[Sample], test: Sequence[Sample]) -> None:
    for attr, label in (
        ("group_id", "group"),
        ("content_hash", "hash"),
        ("path", "path"),
    ):
        a = {getattr(s, attr) for s in train}
        b = {getattr(s, attr) for s in test}
        if a & b:
            raise DataLayerError(f"{label} overlap between train and test: {sorted(a & b)[:3]}")


def _image_size(path: str | Path) -> tuple[int, int] | None:
    """Return (width, height) without requiring Pillow.

    Pillow is used when available; otherwise PNG IHDR and JPEG SOF headers are
    parsed directly.  ``None`` means bounds are unknown (no clamping).
    """
    try:
        from PIL import Image  # lazy: Pillow is optional in a minimal env
    except ImportError:
        Image = None
    if Image is not None:
        try:
            with Image.open(path) as im:
                return int(im.width), int(im.height)
        except Exception:
            return None
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR" and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    if data[:2] == b"\xff\xd8":
        i = 2
        sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
            if marker in sof:
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                return int(w), int(h)
            i += 2 + seg_len
            if seg_len == 0:
                break
    return None


def _json_native(value):
    """Recursively convert tuples to lists so a manifest equals its JSON load."""
    if isinstance(value, dict):
        return {str(k): _json_native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_native(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _library_versions() -> dict[str, str]:
    from importlib import metadata

    def version(dist: str) -> str:
        try:
            return metadata.version(dist)
        except Exception:
            return "unknown"

    return {
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", "unknown"),
        "torchvision": version("torchvision"),
        "numpy": np.__version__,
        "sklearn": version("scikit-learn"),
        "pillow": version("Pillow"),
        "pennylane": version("PennyLane"),
    }


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _pair_overlaps(splits: Mapping[Split, SplitRecord]) -> dict[str, dict[str, list[str]]]:
    names = sorted(splits, key=lambda s: s.value)
    result = {"group_overlap": {}, "hash_overlap": {}, "path_overlap": {}}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            key = f"{a.value}__{b.value}"
            result["group_overlap"][key] = sorted(splits[a].groups & splits[b].groups)
            result["hash_overlap"][key] = sorted(splits[a].hashes & splits[b].hashes)
            result["path_overlap"][key] = sorted(splits[a].paths & splits[b].paths)
    return result


# ---------------------------------------------------------------------------
# Discovery, hashing, dedup and annotation parsing
# ---------------------------------------------------------------------------
def content_hash(path: str | Path, algorithm: str = "md5") -> str:
    """Hash RAW FILE BYTES in 1 MiB chunks (paper parity: md5).

    Never hash the decoded image: two differently-encoded copies of the same
    photo must match, while different encodings of it must not.
    """
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_kaggle(root: str | Path) -> list[Sample]:
    """Scan the UNION of Kaggle source dirs for ``<Corn|Weed>/*.{jpg,jpeg,png}``.

    Both the classic ``root/<Corn|Weed>`` layout and the downloaded
    ``root/<train|test>/<Corn|Weed>`` layout are supported.  The original
    train/test directory names are ignored after discovery -- they are two
    views of one pool, and treating them as separate splits is the legacy bug.
    """
    root = Path(root)
    candidates = [root / "Corn", root / "Weed"]
    try:
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        children = []
    for child in children:
        candidates.extend((child / "Corn", child / "Weed"))

    class_dirs: list[tuple[Path, str]] = []
    seen_dirs: set[Path] = set()
    for directory in candidates:
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        if directory.is_dir():
            class_dirs.append((directory, directory.name))

    samples: list[Sample] = []
    seen_files: set[Path] = set()
    for directory, class_name in class_dirs:
        label = 0 if class_name.lower() in ("corn", "maize") else 1
        label_name = LABEL_NAMES[label]
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
                continue
            resolved = path.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            digest = content_hash(path)
            samples.append(
                Sample(
                    sample_id=f"kaggle:{digest}",
                    source=Source.KAGGLE,
                    group_id=digest,
                    content_hash=digest,
                    path=str(resolved),
                    bbox=None,
                    label=label,
                    label_name=label_name,
                )
            )
    return sorted(samples, key=lambda s: (s.group_id, s.path))


def dedup_kaggle(samples: Sequence[Sample]) -> tuple[list[Sample], DedupReport]:
    """Keep the first occurrence per content hash in sorted order.

    A hash observed under two labels is a hard error: ``LabelConflictError``.
    """
    ordered = sorted(samples, key=lambda s: (s.content_hash, s.sample_id))
    first: dict[str, Sample] = {}
    conflicts: set[str] = set()
    kept: list[Sample] = []
    for sample in ordered:
        digest = sample.content_hash
        if digest in first:
            if first[digest].label != sample.label:
                conflicts.add(digest)
            continue
        first[digest] = sample
        kept.append(sample)
    report = DedupReport(
        files_seen=len(samples),
        unique_hashes=len(kept),
        duplicates_removed=len(samples) - len(kept),
        label_conflicts=tuple(sorted(conflicts)),
    )
    if conflicts:
        raise LabelConflictError(
            f"{len(conflicts)} content hash(es) carry conflicting labels: {sorted(conflicts)[:3]}",
            tuple(sorted(conflicts)),
        )
    return kept, report


def _parse_ninja(
    ann_dir: str | Path, img_dir: str | Path
) -> tuple[list[Sample], dict[str, int]]:
    ann_root, img_root = Path(ann_dir), Path(img_dir)
    by_stem: dict[str, Path] = {}
    if img_root.is_dir():
        for path in sorted(img_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
                # The Kaggle Ninja mirror names annotations ``X.jpg.json``
                # while the image is ``X.jpg``.  Index both the stem and the
                # full filename so either annotation convention resolves.
                by_stem.setdefault(Path(path).stem, path)
                by_stem.setdefault(path.name, path)

    samples: list[Sample] = []
    stats = {
        "ann_files_seen": 0,
        "boxes_valid": 0,
        "boxes_rejected": 0,
        "boxes_ignored": 0,
        "source_images": 0,
    }
    matched_stems: set[str] = set()
    ann_files = sorted(ann_root.glob("*.json")) if ann_root.is_dir() else []

    for ann_file in ann_files:
        stats["ann_files_seen"] += 1
        # Advisory: the SAME stem (Path(...).stem) is used for grouping,
        # annotation lookup and sample ids -- never a basename in one place.
        stem = Path(ann_file).stem
        image_path = by_stem.get(stem) or by_stem.get(Path(stem).stem)
        if image_path is None:
            continue
        matched_stems.add(stem)
        bounds = _image_size(image_path)
        data = json.loads(ann_file.read_text(encoding="utf-8"))
        boxes: list[tuple[int, int, int, int, int]] = []
        for obj in data.get("objects", []):
            title = obj.get("classTitle")
            if title not in CLASS_TITLES:
                stats["boxes_ignored"] += 1
                continue
            exterior = (obj.get("points") or {}).get("exterior") or []
            if len(exterior) < 2:
                stats["boxes_rejected"] += 1
                continue
            try:
                xs = [float(point[0]) for point in exterior]
                ys = [float(point[1]) for point in exterior]
            except (TypeError, ValueError, IndexError):
                stats["boxes_rejected"] += 1
                continue
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            if bounds is not None:
                width, height = bounds
                x1 = max(0, min(width, x1))
                x2 = max(0, min(width, x2))
                y1 = max(0, min(height, y1))
                y2 = max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                stats["boxes_rejected"] += 1
                continue
            boxes.append((x1, y1, x2, y2, CLASS_TITLES[title]))

        boxes.sort(key=lambda box: (box[0], box[1], box[2], box[3], box[4]))
        for index, (x1, y1, x2, y2, label) in enumerate(boxes):
            # Advisory: the group key is literally Path(img_path).stem; the
            # annotation stem used for lookup is the same stem by construction.
            group_id = Path(image_path).stem
            digest = hashlib.sha256(
                f"{group_id}:{x1},{y1},{x2},{y2}".encode("utf-8")
            ).hexdigest()
            samples.append(
                Sample(
                    sample_id=f"ninja:{group_id}:b{index}",
                    source=Source.NINJA,
                    group_id=group_id,
                    content_hash=digest,
                    path=str(image_path.resolve()),
                    bbox=(x1, y1, x2, y2),
                    label=label,
                    label_name=LABEL_NAMES[label],
                )
            )

    samples.sort(key=lambda s: (s.group_id, s.bbox, s.label))
    stats["boxes_valid"] = len(samples)
    stats["source_images"] = len(matched_stems)
    return samples, stats


def parse_ninja_annotations(ann_dir: str | Path, img_dir: str | Path) -> list[Sample]:
    """Parse DatasetNinja JSON annotations into one ``Sample`` per valid box."""
    return _parse_ninja(ann_dir, img_dir)[0]


# ---------------------------------------------------------------------------
# Group-aware splitting
# ---------------------------------------------------------------------------
def split_kaggle(samples: Sequence[Sample], cfg: ExperimentConfig) -> dict[Split, SplitRecord]:
    """Class-stratified, group-aware 80/10/10 Kaggle split.

    ``StratifiedGroupKFold(10, shuffle=True, random_state=cfg.seed)``:
    fold 0 -> ID_TEST, fold 1 -> ID_VAL, folds 2..9 -> TRAIN.  Groups are
    disjoint by construction; post-dedup each group has size one.
    """
    ordered = sorted(samples, key=lambda s: (s.group_id, s.path, s.sample_id))
    if not ordered:
        return {
            Split.TRAIN: _record(Split.TRAIN, ()),
            Split.ID_VAL: _record(Split.ID_VAL, ()),
            Split.ID_TEST: _record(Split.ID_TEST, ()),
        }
    features = np.zeros((len(ordered), 1), dtype=np.float32)
    labels = np.asarray([s.label for s in ordered], dtype=int)
    groups = np.asarray([s.group_id for s in ordered])
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=cfg.seed)
    folds = [list(test_idx) for _, test_idx in splitter.split(features, labels, groups)]

    test_samples = [ordered[i] for i in folds[0]]
    val_samples = [ordered[i] for i in folds[1]]
    train_samples = [ordered[i] for fold in folds[2:] for i in fold]
    return {
        Split.TRAIN: _record(Split.TRAIN, train_samples),
        Split.ID_VAL: _record(Split.ID_VAL, val_samples),
        Split.ID_TEST: _record(Split.ID_TEST, test_samples),
    }


def split_ninja(samples: Sequence[Sample], cfg: ExperimentConfig) -> dict[Split, SplitRecord]:
    """Whole-source-image allocation with an exact 200/200 adaptation quota.

    See spec section 3.2: groups are permuted with an independent RNG stream;
    untaken boxes of a chosen adaptation image are deliberately discarded and
    never routed to OOD test.
    """
    by_group: dict[str, list[Sample]] = {}
    for sample in samples:
        by_group.setdefault(sample.group_id, []).append(sample)
    observed = {
        label: sum(1 for s in samples if s.label == label) for label in (0, 1)
    }
    if not by_group:
        raise InsufficientAdaptationCrops("no DatasetNinja groups found", observed)

    groups = np.asarray(sorted(by_group))
    seed_for_ninja = SeedSequence(cfg.seed).spawn(2)[1]
    order = np.random.default_rng(seed_for_ninja).permutation(len(groups))
    quota = {0: int(cfg.adaptation_per_class), 1: int(cfg.adaptation_per_class)}

    adaptation: list[Sample] = []
    ood: list[Sample] = []
    for position in order:
        group = str(groups[int(position)])
        if quota[0] == 0 and quota[1] == 0:
            ood.extend(by_group[group])
            continue
        for label in (0, 1):
            boxes = sorted(
                (s for s in by_group[group] if s.label == label),
                key=lambda s: (s.bbox, s.sample_id),
            )
            take = min(quota[label], len(boxes))
            adaptation.extend(boxes[:take])
            quota[label] -= take

    if quota[0] or quota[1]:
        raise InsufficientAdaptationCrops(
            f"cannot fill adaptation quota {quota} from class counts {observed}", observed
        )
    return {
        Split.ADAPTATION: _record(Split.ADAPTATION, adaptation),
        Split.OOD_TEST: _record(Split.OOD_TEST, ood),
    }


def materialise_adaptation_crops(
    samples: Sequence[Sample], derived_dir: str | Path
) -> list[Path]:
    """Write the labelled adaptation crops as PNGs under ``derived_dir``.

    OOD crops stay lazy ``(path, bbox)`` references; no OOD files are written.
    Sample ids contain ``:`` (illegal in Windows filenames), so the on-disk
    name is a sanitised form of the id -- the manifest remains authoritative.
    """
    try:
        from PIL import Image  # lazy: only needed when materialising
    except ImportError as exc:  # pragma: no cover - dependency-dependent
        raise DataLayerError("Pillow is required to materialise adaptation crops") from exc

    out_dir = Path(derived_dir) / "crops" / "ninja_adaptation"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for sample in samples:
        if not sample.source == Source.NINJA or sample.bbox is None:
            continue
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample.sample_id)
        destination = out_dir / f"{safe}.png"
        with Image.open(sample.path) as image:
            image.convert("RGB").crop(sample.bbox).save(destination, format="PNG")
        paths.append(destination)
    return paths


# ---------------------------------------------------------------------------
# QSVC pairing (spec section 3.3)
# ---------------------------------------------------------------------------
def _draw_group_stratified(
    samples: Sequence[Sample], per_class: Mapping[int, int], rng: np.random.Generator
) -> tuple[Sample, ...]:
    """Draw whole groups until each class quota is met, then trim excess.

    Trimming only *drops* crops of an already-chosen group; it never moves a
    group's crop to the other side, so no photograph straddles train/test.
    """
    by_key: dict[tuple[int, str], list[Sample]] = {}
    for sample in samples:
        by_key.setdefault((sample.label, sample.group_id), []).append(sample)
    picked: list[Sample] = []
    for label in (0, 1):
        need = int(per_class.get(label, 0))
        if need <= 0:
            continue
        keys = sorted(key for key in by_key if key[0] == label)
        if not keys:
            continue
        chosen: list[Sample] = []
        have = 0
        for position in rng.permutation(len(keys)):
            if have >= need:
                break
            key = keys[int(position)]
            chosen.extend(by_key[key])
            have += len(by_key[key])
        chosen.sort(key=lambda s: (s.group_id, s.sample_id))
        picked.extend(chosen[:need])
    return tuple(picked)


def sample_qsvc_pairs(
    manifest,
    cfg: ExperimentConfig,
    run_idx: int,
    protocol: QSVCProtocol = QSVCProtocol.MIXED,
) -> QSVCRun:
    """Return one arm of one paired QSVC run (50 Kaggle + 50 Ninja vs 50 OOD).

    The per-run ``SeedSequence`` is spawned, passed *itself* to ``default_rng``
    and recorded via ``generate_state(1)`` -- no second sequence is built from
    the recorded integer.
    """
    if not 0 <= run_idx < cfg.qsvc_runs:
        raise ValueError(f"run_idx {run_idx} outside range({cfg.qsvc_runs})")
    protocol = QSVCProtocol(protocol)
    run_ss = SeedSequence([cfg.seed, cfg.seed_offset]).spawn(cfg.qsvc_runs)[run_idx]
    seed = int(run_ss.generate_state(1)[0])
    rng = np.random.default_rng(run_ss)

    ood = manifest.splits[Split.OOD_TEST].samples
    kaggle_train = manifest.splits[Split.TRAIN].samples
    adaptation = manifest.splits[Split.ADAPTATION].samples

    # Test is drawn FIRST so both arms at the same run_idx see identical crops.
    half_test = int(cfg.qsvc_test_per_run) // 2
    test = _draw_group_stratified(ood, {0: half_test, 1: half_test}, rng)

    if protocol is QSVCProtocol.MIXED:
        half_source = int(cfg.qsvc_source_per_run) // 2
        source = _draw_group_stratified(kaggle_train, {0: half_source, 1: half_source}, rng)
        half_target = int(cfg.qsvc_target_adaptation_per_run) // 2
        target = _draw_group_stratified(adaptation, {0: half_target, 1: half_target}, rng)
    else:
        half_control = int(cfg.qsvc_mixed_train_size) // 2
        source = _draw_group_stratified(
            kaggle_train, {0: half_control, 1: half_control}, rng
        )
        target = ()

    run = QSVCRun(
        run_idx=int(run_idx),
        protocol=protocol,
        seed=seed,
        source_train=tuple(source),
        target_train=tuple(target),
        test=tuple(test),
    )
    if len(run.train) != int(cfg.qsvc_mixed_train_size):
        raise DataLayerError(
            f"{protocol.value} run {run_idx} has {len(run.train)} train samples, "
            f"expected {cfg.qsvc_mixed_train_size}"
        )
    if len(run.test) != int(cfg.qsvc_test_per_run):
        raise DataLayerError(
            f"{protocol.value} run {run_idx} has {len(run.test)} test samples, "
            f"expected {cfg.qsvc_test_per_run}"
        )
    _assert_disjoint(run.train, run.test)
    return run


# ---------------------------------------------------------------------------
# Dataset / dataloaders
# ---------------------------------------------------------------------------
class ImageDataset(Dataset):
    """Lazy ``(image_tensor, label, sample_id)`` dataset over ``Sample`` tuples."""

    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Callable | None = None,
    ) -> None:
        self.samples = tuple(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        try:
            from PIL import Image  # lazy: only needed when reading pixels
        except ImportError as exc:  # pragma: no cover - dependency-dependent
            raise DataLayerError("Pillow is required to read image samples") from exc
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if sample.bbox is not None:
                image = image.crop(sample.bbox)
            if self.transform is not None:
                image = self.transform(image)
            else:  # pragma: no cover - callers always pass a transform
                image = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float()
        return image, sample.label, sample.sample_id


def make_dataloaders(
    manifest,
    cfg: ExperimentConfig,
    *,
    splits: Sequence[Split] = (
        Split.TRAIN,
        Split.ID_VAL,
        Split.ID_TEST,
        Split.OOD_TEST,
    ),
    num_workers: int = 4,
) -> dict[Split, DataLoader]:
    """Train-only shuffling with an explicit seeded generator; eval is fixed."""
    from data.transforms import make_transforms

    transforms_by_kind = make_transforms(cfg)
    loaders: dict[Split, DataLoader] = {}
    for split in splits:
        dataset = ImageDataset(
            manifest.splits[split].samples,
            transform=transforms_by_kind["train" if split is Split.TRAIN else "eval"],
        )
        shuffle = split is Split.TRAIN
        generator = torch.Generator().manual_seed(cfg.seed) if shuffle else None
        kwargs = {}
        if torch.cuda.is_available():
            kwargs["pin_memory_device"] = "cuda"
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            generator=generator,
            **kwargs,
        )
    return loaders


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build_splits(
    cfg: ExperimentConfig,
    *,
    kaggle_root: str | Path | None = None,
    num_workers: int | None = None,
):
    """Validate -> discover -> dedup -> split -> manifest -> ``verify()`` gate.

    Pure function of ``(cfg, on-disk bytes)``: no random global state and no
    writes (adaptation crops are materialised by an explicit caller).
    """
    from data.manifest import SplitManifest, compute_checksum

    problems = cfg.validate()
    if problems:
        raise ConfigError("; ".join(problems))

    root = kaggle_root if kaggle_root is not None else os.getenv("KAGGLE_DATA_DIR", "")
    if not root or not Path(root).is_dir():
        raise ConfigError(
            "Kaggle source root not found; pass kaggle_root=... or set KAGGLE_DATA_DIR "
            "(no developer-path fallback by design)."
        )
    workers = int(num_workers if num_workers is not None else os.getenv("QC_NUM_WORKERS", "4"))

    kaggle_samples, kaggle_report = dedup_kaggle(discover_kaggle(root))
    ninja_samples, ninja_stats = _parse_ninja(cfg.ninja_ann_dir, cfg.ninja_img_dir)

    kaggle_splits = split_kaggle(kaggle_samples, cfg)
    ninja_splits = split_ninja(ninja_samples, cfg)
    splits: dict[Split, SplitRecord] = {**kaggle_splits, **ninja_splits}

    provenance = tuple(
        sorted(
            (sample for record in splits.values() for sample in record.samples),
            key=lambda s: s.sample_id,
        )
    )
    discarded = max(
        0,
        len(ninja_samples)
        - splits[Split.ADAPTATION].count
        - splits[Split.OOD_TEST].count,
    )

    config = _json_native(cfg.as_dict())
    config["qsvc_mixed_train_size"] = int(cfg.qsvc_mixed_train_size)
    snapshot = {
        "config": config,
        "library_versions": _library_versions(),
        "git_commit": _git_commit(),
        "num_workers": workers,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    sources = {
        "kaggle": {
            "dataset": cfg.kaggle_dataset,
            "files_seen": kaggle_report.files_seen,
            "unique_hashes": kaggle_report.unique_hashes,
            "duplicates_removed": kaggle_report.duplicates_removed,
            "label_conflicts": list(kaggle_report.label_conflicts),
            "hash_algorithm": "md5",
            "hash_input": "raw_file_bytes",
        },
        "ninja": {
            "ann_dir_env": "NINJA_ANN_DIR",
            "img_dir_env": "NINJA_IMG_DIR",
            "ann_files_seen": ninja_stats["ann_files_seen"],
            "boxes_valid": ninja_stats["boxes_valid"],
            "boxes_rejected": ninja_stats["boxes_rejected"],
            "boxes_ignored": ninja_stats["boxes_ignored"],
            "source_images": ninja_stats["source_images"],
            "discarded_boxes": discarded,
        },
    }
    overlaps = _pair_overlaps(splits)
    invariants = {
        "verified": True,
        "group_overlap": overlaps["group_overlap"],
        "hash_overlap": overlaps["hash_overlap"],
        "path_overlap": overlaps["path_overlap"],
        "class_counts": {split.value: record.class_counts() for split, record in splits.items()},
        "counts": {split.value: record.count for split, record in splits.items()},
        "discarded_boxes": discarded,
    }
    payload = {
        "schema_version": "1.0",
        "generated_by": "experiments.data.build_splits",
        "seed": int(cfg.seed),
        "config_snapshot": snapshot,
        "sources": sources,
        "splits": {
            split.value: {
                "count": record.count,
                "group_count": len(record.groups),
                "groups": sorted(record.groups),
                "class_counts": record.class_counts(),
                "sample_ids": sorted(record.sample_ids),
                "hash_digest": hashlib.sha256(
                    "".join(sorted(record.hashes)).encode("utf-8")
                ).hexdigest(),
            }
            for split, record in splits.items()
        },
        "provenance": [],  # filled by the dataclass's serialiser; see to_json
        "invariants": invariants,
        "checksum_sha256": "",
    }
    # Provenance is part of the canonical payload for the checksum.
    from data.manifest import _sample_to_dict

    payload["provenance"] = [_sample_to_dict(sample) for sample in provenance]
    checksum = compute_checksum(payload)
    payload["checksum_sha256"] = checksum

    manifest = SplitManifest(
        schema_version="1.0",
        generated_by="experiments.data.build_splits",
        seed=int(cfg.seed),
        config_snapshot=snapshot,
        sources=sources,
        splits=splits,
        provenance=provenance,
        invariants=invariants,
        checksum_sha256=checksum,
    )
    violations = manifest.verify()
    if violations:
        raise ManifestError("built manifest failed verify(): " + "; ".join(violations))
    return manifest


def load_manifest(path: str | Path):
    """Load a ``SplitManifest`` written by ``SplitManifest.to_json``."""
    from data.manifest import SplitManifest

    return SplitManifest.from_json(path)
