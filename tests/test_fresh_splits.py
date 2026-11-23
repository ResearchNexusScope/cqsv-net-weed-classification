"""Synthetic tests for fresh dataset preparation and group-disjoint splitting.

No network, no real dataset bytes, no feature extraction and no training.
Only in-memory ``FreshSample`` objects and tiny generated JPEGs in ``tmp_path``
are used.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("numpy")  # clean skip if the core numeric dep is absent

from data import fresh_splits as fs  # noqa: E402  (import after dependency guard)

RATIOS = (0.7, 0.2, 0.1)


def _sample(sample_id: str, group: str, label: int, bbox=None) -> fs.FreshSample:
    return fs.FreshSample(
        sample_id=sample_id,
        source_image_id=group,
        label=label,
        path=f"/synthetic/{sample_id}.jpg",
        bbox=bbox,
    )


def _balanced_groups(n_groups: int = 200, per_group: int = 5) -> list[fs.FreshSample]:
    """Many single-class groups, half of each class, so every split can hold both."""
    samples: list[fs.FreshSample] = []
    for group_index in range(n_groups):
        group = f"g{group_index:04d}"
        label = group_index % 2
        for member in range(per_group):
            samples.append(_sample(f"s{group_index:04d}_{member}", group, label))
    return samples


# --------------------------------------------------------------------------- #
# label_from_filename
# --------------------------------------------------------------------------- #
def test_label_from_filename_maps_known_prefixes() -> None:
    assert fs.label_from_filename("maize_1") == 0
    assert fs.label_from_filename("weed_7") == 1
    assert fs.label_from_filename("Maize_42") == 0  # prefix is case-insensitive


def test_label_from_filename_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown MaizeSet filename prefix"):
        fs.label_from_filename("tomato_1")
    with pytest.raises(ValueError):
        fs.label_from_filename("")


# --------------------------------------------------------------------------- #
# grouped_split invariants
# --------------------------------------------------------------------------- #
def test_grouped_split_is_group_disjoint_and_covers_all_samples() -> None:
    samples = _balanced_groups(60, 4)
    splits = fs.grouped_split(samples, seed=42, ratios=RATIOS)

    assert set(splits) == {"train", "val", "test"}
    group_sets = {
        name: {sample.source_image_id for sample in part} for name, part in splits.items()
    }
    assert not (group_sets["train"] & group_sets["val"])
    assert not (group_sets["train"] & group_sets["test"])
    assert not (group_sets["val"] & group_sets["test"])
    assert set().union(*group_sets.values()) == {s.source_image_id for s in samples}
    assert sum(len(part) for part in splits.values()) == len(samples)
    assert len({s.sample_id for part in splits.values() for s in part}) == len(samples)


def test_grouped_split_deterministic_for_fixed_seed() -> None:
    samples = _balanced_groups(120, 3)
    first = fs.grouped_split(samples, seed=11, ratios=RATIOS)
    second = fs.grouped_split(samples, seed=11, ratios=RATIOS)
    for name in ("train", "val", "test"):
        assert [s.sample_id for s in first[name]] == [s.sample_id for s in second[name]]


def test_grouped_split_different_seed_changes_assignment() -> None:
    samples = _balanced_groups(120, 3)
    first = fs.grouped_split(samples, seed=1, ratios=RATIOS)
    second = fs.grouped_split(samples, seed=2, ratios=RATIOS)
    assert [s.sample_id for s in first["train"]] != [s.sample_id for s in second["train"]]


def test_grouped_split_approximates_ratios() -> None:
    samples = _balanced_groups(200, 5)
    splits = fs.grouped_split(samples, seed=42, ratios=RATIOS)
    total = len(samples)
    for name, ratio in zip(("train", "val", "test"), RATIOS):
        assert abs(len(splits[name]) / total - ratio) < 0.05


def test_grouped_split_keeps_both_classes_in_every_split() -> None:
    samples = _balanced_groups(200, 5)
    splits = fs.grouped_split(samples, seed=42, ratios=RATIOS)
    for part in splits.values():
        assert {sample.label for sample in part} == {0, 1}


def test_grouped_split_handles_unequal_group_sizes() -> None:
    samples: list[fs.FreshSample] = []
    for group_index in range(70):
        group = f"u{group_index:03d}"
        label = group_index % 2
        for member in range(1 + group_index % 7):  # sizes 1..7
            samples.append(_sample(f"u{group_index:03d}_{member}", group, label))

    splits = fs.grouped_split(samples, seed=5, ratios=RATIOS)
    group_sets = {
        name: {sample.source_image_id for sample in part} for name, part in splits.items()
    }
    assert not any(
        group_sets[a] & group_sets[b]
        for i, a in enumerate(("train", "val", "test"))
        for b in ("train", "val", "test")[i + 1 :]
    )
    assert set().union(*group_sets.values()) == {s.source_image_id for s in samples}
    assert sum(len(part) for part in splits.values()) == len(samples)
    for part in splits.values():
        assert {sample.label for sample in part} == {0, 1}


def test_grouped_split_keeps_mixed_group_whole() -> None:
    """A group with both labels is allowed and must not be split across folds."""
    # Six mixed groups give each split enough samples to contain both classes.
    samples: list[fs.FreshSample] = []
    for group_index in range(6):
        group = "mixed" if group_index == 0 else f"mixed_{group_index}"
        samples.append(_sample(f"{group}_0", group, 0))
        samples.append(_sample(f"{group}_1", group, 1))

    splits = fs.grouped_split(samples, seed=42, ratios=RATIOS)

    group_sets = {
        name: {sample.source_image_id for sample in part} for name, part in splits.items()
    }
    assert not (group_sets["train"] & group_sets["val"])
    assert not (group_sets["train"] & group_sets["test"])
    assert not (group_sets["val"] & group_sets["test"])
    assert set().union(*group_sets.values()) == {s.source_image_id for s in samples}

    mixed_splits = [
        name for name, part in splits.items() if any(s.source_image_id == "mixed" for s in part)
    ]
    assert len(mixed_splits) == 1  # the mixed group is kept wholly in one split
    owner = mixed_splits[0]
    assert {s.label for s in splits[owner] if s.source_image_id == "mixed"} == {0, 1}
    assert sum(len(part) for part in splits.values()) == len(samples)
    for part in splits.values():
        assert {sample.label for sample in part} == {0, 1}


def test_grouped_split_acre_like_mixed_and_single_class_groups() -> None:
    """Real ACRE photos mix crop/weed boxes; whole-photo disjointness must hold."""
    samples: list[fs.FreshSample] = []
    # Single-class crop groups, single-class weed groups and mixed groups.
    for group_index in range(40):
        group = f"photo_{group_index:03d}"
        kind = group_index % 3
        if kind == 0:
            labels = [0, 0, 0]
        elif kind == 1:
            labels = [1, 1, 1]
        else:
            labels = [0, 1, 0, 1]
        for member, label in enumerate(labels):
            samples.append(_sample(f"{group}_{member}", group, label))

    first = fs.grouped_split(samples, seed=42, ratios=RATIOS)
    second = fs.grouped_split(samples, seed=42, ratios=RATIOS)
    for name in ("train", "val", "test"):
        assert [s.sample_id for s in first[name]] == [s.sample_id for s in second[name]]

    group_sets = {
        name: {sample.source_image_id for sample in part} for name, part in first.items()
    }
    assert not (group_sets["train"] & group_sets["val"])
    assert not (group_sets["train"] & group_sets["test"])
    assert not (group_sets["val"] & group_sets["test"])
    assert set().union(*group_sets.values()) == {s.source_image_id for s in samples}
    assert sum(len(part) for part in first.values()) == len(samples)
    for part in first.values():
        assert {sample.label for sample in part} == {0, 1}

    # class_counts must count both labels inside mixed groups per split.
    provenance = fs.split_provenance("acre-like", samples, first, 42, RATIOS)
    for name in ("train", "val", "test"):
        expected = {
            "0": sum(s.label == 0 for s in first[name]),
            "1": sum(s.label == 1 for s in first[name]),
        }
        assert provenance["class_counts"][name] == expected


def test_grouped_split_rejects_split_that_would_be_empty() -> None:
    samples = [_sample("a", "g0", 0), _sample("b", "g1", 1)]
    with pytest.raises(ValueError, match="at least 3 distinct source groups"):
        fs.grouped_split(samples, seed=42, ratios=RATIOS)


def test_grouped_split_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="no samples"):
        fs.grouped_split([], seed=42)
    with pytest.raises(ValueError, match="sum to 1"):
        fs.grouped_split(_balanced_groups(20, 2), seed=42, ratios=(0.5, 0.3, 0.3))


# --------------------------------------------------------------------------- #
# split_provenance
# --------------------------------------------------------------------------- #
def test_split_provenance_schema_and_sha256_stability() -> None:
    samples = _balanced_groups(40, 3)
    splits = fs.grouped_split(samples, seed=7, ratios=RATIOS)
    provenance = fs.split_provenance("toy", samples, splits, 7, RATIOS)

    for key in (
        "schema_version",
        "dataset",
        "seed",
        "ratios",
        "counts",
        "group_counts",
        "class_counts",
        "manifest_sha256",
        "split_assignment_sha256",
    ):
        assert key in provenance

    assert provenance["dataset"] == "toy"
    assert provenance["seed"] == 7
    assert provenance["ratios"] == list(RATIOS)
    assert set(provenance["counts"]) == {"train", "val", "test"}
    assert set(provenance["group_counts"]) == {"train", "val", "test"}
    assert set(provenance["class_counts"]) == {"train", "val", "test"}
    assert sum(provenance["counts"].values()) == len(samples)
    assert all(set(cc) == {"0", "1"} for cc in provenance["class_counts"].values())
    assert len(provenance["manifest_sha256"]) == 64

    reordered_samples = list(reversed(samples))
    reordered_splits = {name: list(reversed(part)) for name, part in splits.items()}
    again = fs.split_provenance("toy", reordered_samples, reordered_splits, 7, RATIOS)
    assert again == provenance
    assert again["manifest_sha256"] == provenance["manifest_sha256"]


# --------------------------------------------------------------------------- #
# Duck typing / prepare dispatch
# --------------------------------------------------------------------------- #
def test_fresh_sample_duck_types_with_make_feature_loader() -> None:
    """The loader keys on ``sample_id`` + ``bbox`` and opens ``sample.path``."""
    sample = fs.FreshSample("crop_1", "photo_1", 1, "/synthetic/crop_1.jpg", (1, 2, 3, 4))
    key = (sample.sample_id, tuple(sample.bbox or ()))
    assert key == ("crop_1", (1, 2, 3, 4))
    assert sample.path == "/synthetic/crop_1.jpg"
    assert sample.label == 1

    whole_image = fs.FreshSample("img_1", "photo_2", 0, "/synthetic/img_1.jpg")
    assert whole_image.bbox is None
    assert tuple(whole_image.bbox or ()) == ()


def test_prepare_dataset_unknown_name_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        fs.prepare_dataset("not-a-dataset", root=tmp_path)


@pytest.mark.parametrize("name", ["maizeset", "deepweeds", "acre"])
def test_prepare_dataset_missing_source_raises(name: str, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fs.prepare_dataset(name, root=tmp_path)


def test_prepare_maizeset_synthetic_is_idempotent(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    import io
    import zipfile

    source = tmp_path / "maizeset"
    source.mkdir(parents=True)
    archive = source / "images_00.zip"
    buffers = []
    for colour in ((10, 20, 30), (30, 20, 10)):
        buffer = io.BytesIO()
        image_module.new("RGB", (8, 8), colour).save(buffer, "JPEG")
        buffers.append(buffer.getvalue())
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("maize_1.jpg", buffers[0])
        zf.writestr("weed_1.jpg", buffers[1])

    first = fs.prepare_dataset("maizeset", root=tmp_path)
    assert {sample.source_image_id: sample.label for sample in first} == {
        "maize_1": 0,
        "weed_1": 1,
    }
    assert all(os.path.isabs(sample.path) and Path(sample.path).is_file() for sample in first)
    assert all(sample.bbox is None for sample in first)
    mtimes = {sample.path: os.path.getmtime(sample.path) for sample in first}

    second = fs.prepare_dataset("maizeset", root=tmp_path)
    assert [s.sample_id for s in second] == [s.sample_id for s in first]
    assert {s.path: os.path.getmtime(s.path) for s in second} == mtimes


def test_prepare_acre_resolves_data_prefix_and_is_collision_safe(tmp_path: Path) -> None:
    """COCO file_names omit the ``data/`` prefix that real zip entries carry."""
    image_module = pytest.importorskip("PIL.Image")
    import io
    import json
    import zipfile

    def jpeg(colour):
        buffer = io.BytesIO()
        image_module.new("RGB", (16, 16), colour).save(buffer, "JPEG")
        return buffer.getvalue()

    coco = {
        "images": [
            {"id": 1, "file_name": "Maize/seq1/rgb-x.jpg"},
            {"id": 2, "file_name": "Bean/seq1/rgb-x.jpg"},  # same stem, other directory
            {"id": 3, "file_name": "Maize/seq2/rgb-y.jpg"},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4]},
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [4, 4, 4, 4]},
            {"id": 20, "image_id": 2, "category_id": 2, "bbox": [0, 0, 4, 4]},
            {"id": 30, "image_id": 3, "category_id": 1, "bbox": [0, 0, 4, 4]},
        ],
    }
    archive = tmp_path / "acre" / "The_ACRE_Crop-Weed_Dataset.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("data/ACRE_COCO_annotations.json", json.dumps(coco))
        zf.writestr("data/Maize/seq1/rgb-x.jpg", jpeg((10, 20, 30)))
        zf.writestr("data/Bean/seq1/rgb-x.jpg", jpeg((30, 20, 10)))
        zf.writestr("data/Maize/seq2/rgb-y.jpg", jpeg((20, 30, 10)))

    samples = fs._prepare_acre(tmp_path)
    assert len(samples) == 4
    assert {sample.source_image_id for sample in samples} == {"rgb-x", "rgb-y"}
    labels = {s.sample_id.rsplit("_", 1)[-1]: s.label for s in samples}
    assert labels == {"10": 0, "11": 1, "20": 1, "30": 0}  # category 1->0, 2->1
    assert len({s.sample_id for s in samples}) == 4  # hash keeps names unique
    assert len({s.path for s in samples}) == 4
    assert all(os.path.isabs(s.path) and Path(s.path).is_file() for s in samples)
    assert all(s.bbox is None for s in samples)

    mtimes = {s.path: os.path.getmtime(s.path) for s in samples}
    again = fs._prepare_acre(tmp_path)
    assert [s.sample_id for s in again] == [s.sample_id for s in samples]
    assert {s.path: os.path.getmtime(s.path) for s in again} == mtimes


def test_acre_entry_resolver_reports_candidates() -> None:
    entries = {"data/Maize/seq/rgb-x.jpg", "data/Bean/seq/rgb-x.jpg"}
    by_basename: dict[str, list[str]] = {}
    for entry in entries:
        by_basename.setdefault(Path(entry).name, []).append(entry)

    with pytest.raises(FileNotFoundError, match="tried"):
        fs._resolve_acre_entry("Maize/seq/missing.jpg", entries, by_basename)
    with pytest.raises(FileNotFoundError, match="basename matches"):
        fs._resolve_acre_entry("Other/seq/rgb-x.jpg", entries, by_basename)
