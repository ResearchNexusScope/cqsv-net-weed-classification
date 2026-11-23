"""Synthetic, offline tests for the leak-free data layer (spec T1-T12).

No private paper content, no real dataset paths and no network are used: every
fixture is a throwaway ``tmp_path`` tree of tiny generated files.
"""
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import replace
from pathlib import Path

import pytest
from numpy.random import SeedSequence

from experiments.config import ExperimentConfig
from experiments.metrics import paired_t_test
from data.manifest import SplitManifest
from data.splits import (
    InsufficientAdaptationCrops,
    LabelConflictError,
    QSVCProtocol,
    Sample,
    Source,
    Split,
    build_splits,
    content_hash,
    dedup_kaggle,
    discover_kaggle,
    materialise_adaptation_crops,
    parse_ninja_annotations,
    sample_qsvc_pairs,
    split_kaggle,
    split_ninja,
)
from data.transforms import make_transforms

NINE_KEYS = [
    "schema_version",
    "generated_by",
    "seed",
    "config_snapshot",
    "sources",
    "splits",
    "provenance",
    "invariants",
    "checksum_sha256",
]


def _png_bytes(width: int = 8, height: int = 8, seed: int = 0) -> bytes:
    """Minimal valid truecolour PNG, generated without Pillow."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            row.extend(((seed + x) % 256, (seed + y) % 256, (seed + x + y) % 256))
        rows.append(b"\x00" + bytes(row))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )


def _sample(index: int, label: int, source: Source = Source.KAGGLE) -> Sample:
    group = f"g{index:05d}"
    return Sample(
        sample_id=f"{source.value}:{group}",
        source=source,
        group_id=group,
        content_hash=hashlib.md5(group.encode("utf-8")).hexdigest(),
        path=f"/synthetic/{source.value}/{group}.jpg",
        bbox=None,
        label=label,
        label_name="Corn" if label == 0 else "Weed",
    )


@pytest.fixture
def synthetic(tmp_path: Path):
    kaggle = tmp_path / "kaggle"
    ann = tmp_path / "ann"
    images = tmp_path / "img"
    derived = tmp_path / "derived"
    for view in ("train", "test"):
        for cls in ("Corn", "Weed"):
            (kaggle / view / cls).mkdir(parents=True, exist_ok=True)
    for i in range(150):
        cls = "Corn" if i < 75 else "Weed"
        view = "train" if i % 2 == 0 else "test"
        (kaggle / view / cls / f"k{i:04d}.png").write_bytes(_png_bytes(seed=i + 1))

    images.mkdir(parents=True, exist_ok=True)
    ann.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        stem = f"ninja_{i:03d}"
        (images / f"{stem}.png").write_bytes(_png_bytes(64, 64, seed=200 + i))
        objects = []
        for j in range(30):
            x1, y1 = (j % 5) * 12, (j // 5) * 12
            x2, y2 = x1 + 9, y1 + 9
            objects.append(
                {
                    "classTitle": "maize",
                    "points": {"exterior": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]},
                }
            )
            objects.append(
                {"classTitle": "weed", "points": {"exterior": [[x1, y1], [x2, y2]]}}
            )
        (ann / f"{stem}.json").write_text(
            json.dumps({"objects": objects}), encoding="utf-8"
        )
    cfg = ExperimentConfig(
        ninja_ann_dir=str(ann),
        ninja_img_dir=str(images),
        output_dir=str(derived),
    )
    return cfg, tmp_path


@pytest.fixture
def built(synthetic):
    cfg, root = synthetic
    manifest = build_splits(cfg, kaggle_root=root / "kaggle", num_workers=0)
    return manifest, cfg, root


# --- T1 --------------------------------------------------------------------
def test_content_hash_is_byte_based_and_chunk_stable(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 5000  # crosses the 1 MiB chunk boundary
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    first.write_bytes(payload)
    second.write_bytes(payload)
    assert content_hash(first) == hashlib.md5(payload).hexdigest()
    assert content_hash(first) == content_hash(second)

    changed = tmp_path / "c.jpg"
    changed.write_bytes(payload + b"x")
    assert content_hash(changed) != content_hash(first)

    try:
        from PIL import Image
    except ImportError:
        return  # Pillow optional locally; the byte-hash contract is already tested
    image = Image.new("RGB", (32, 32))
    image.putdata([((x * 7) % 256, (y * 11) % 256, (x + y) % 256)
                   for y in range(32) for x in range(32)])
    low = tmp_path / "q20.jpg"
    high = tmp_path / "q95.jpg"
    image.save(low, format="JPEG", quality=20)
    image.save(high, format="JPEG", quality=95)
    assert content_hash(low) != content_hash(high)  # byte hash, not decoded pixels


# --- T2 --------------------------------------------------------------------
def test_dedup_counts_and_label_conflict(tmp_path: Path) -> None:
    kaggle = tmp_path / "kaggle"
    (kaggle / "Corn").mkdir(parents=True, exist_ok=True)
    (kaggle / "Weed").mkdir(parents=True, exist_ok=True)
    payloads = [f"unique-payload-{i}\n".encode() * 100 for i in range(8)]
    for index in range(6):
        (kaggle / "Corn" / f"u{index}.jpg").write_bytes(payloads[index])
    (kaggle / "Corn" / "dup_a.jpg").write_bytes(payloads[6])
    (kaggle / "Corn" / "dup_b.jpg").write_bytes(payloads[6])
    (kaggle / "Weed" / "dup_c.jpg").write_bytes(payloads[7])
    (kaggle / "Weed" / "dup_d.jpg").write_bytes(payloads[7])

    samples = discover_kaggle(kaggle)
    kept, report = dedup_kaggle(samples)
    assert report.files_seen == 10
    assert report.unique_hashes == 8
    assert report.duplicates_removed == 2
    assert len(kept) == 8
    assert len({s.content_hash for s in kept}) == 8

    (kaggle / "Weed" / "conflict.jpg").write_bytes(payloads[0])
    with pytest.raises(LabelConflictError):
        dedup_kaggle(discover_kaggle(kaggle))


# --- T3 --------------------------------------------------------------------
def test_kaggle_splits_are_disjoint_and_stratified() -> None:
    samples = [_sample(i, 0 if i < 50 else 1) for i in range(100)]
    parts = split_kaggle(samples, ExperimentConfig())
    for left, right in (
        (Split.TRAIN, Split.ID_VAL),
        (Split.TRAIN, Split.ID_TEST),
        (Split.ID_VAL, Split.ID_TEST),
    ):
        a, b = parts[left], parts[right]
        assert a.groups & b.groups == frozenset()
        assert a.hashes & b.hashes == frozenset()
        assert a.paths & b.paths == frozenset()
    assert sum(part.count for part in parts.values()) == 100
    for part in parts.values():
        counts = part.class_counts()
        assert counts["Corn"] > 0 and counts["Weed"] > 0


# --- T4 --------------------------------------------------------------------
def test_kaggle_seed_determinism_and_sensitivity(synthetic) -> None:
    cfg, root = synthetic
    first = build_splits(cfg, kaggle_root=root / "kaggle", num_workers=0)
    second = build_splits(cfg, kaggle_root=root / "kaggle", num_workers=0)
    assert first.splits[Split.TRAIN].sample_ids == second.splits[Split.TRAIN].sample_ids
    # generated_at differs between builds but the canonical checksum does not.
    assert first.checksum_sha256 == second.checksum_sha256

    third = build_splits(replace(cfg, seed=43), kaggle_root=root / "kaggle", num_workers=0)
    assert third.splits[Split.TRAIN].sample_ids != first.splits[Split.TRAIN].sample_ids


def test_checksum_is_invariant_to_git_commit(synthetic, monkeypatch) -> None:
    """git_commit is build provenance, not data: two identical builds across a
    commit boundary must share a checksum, while the manifest still records it."""
    cfg, root = synthetic
    monkeypatch.setattr("data.splits._git_commit", lambda: "aaaaaaa")
    first = build_splits(cfg, kaggle_root=root / "kaggle", num_workers=0)
    monkeypatch.setattr("data.splits._git_commit", lambda: "bbbbbbb")
    second = build_splits(cfg, kaggle_root=root / "kaggle", num_workers=0)
    assert first.checksum_sha256 == second.checksum_sha256
    # Provenance is preserved in the full manifest even though it is not hashed.
    assert second.config_snapshot["git_commit"] == "bbbbbbb"


# --- T5 --------------------------------------------------------------------
def test_ninja_parse_polygon_and_filter(tmp_path: Path) -> None:
    images = tmp_path / "img"
    ann = tmp_path / "ann"
    images.mkdir(parents=True, exist_ok=True)
    ann.mkdir(parents=True, exist_ok=True)
    (images / "photo.png").write_bytes(_png_bytes(64, 64))
    objects = [
        {"classTitle": "maize", "points": {"exterior": [[0, 0], [20, 0], [20, 20], [0, 20]]}},
        {"classTitle": "maize", "points": {"exterior": [[30, 30], [40, 40]]}},
        {"classTitle": "weed", "points": {"exterior": [[5, 5], [15, 15]]}},
        {"classTitle": "tomato", "points": {"exterior": [[0, 0], [10, 10]]}},
        {"classTitle": "maize", "points": {"exterior": [[2, 2], [2, 12]]}},  # degenerate
    ]
    (ann / "photo.json").write_text(json.dumps({"objects": objects}), encoding="utf-8")
    samples = parse_ninja_annotations(ann, images)
    assert len(samples) == 3
    assert any(s.bbox == (0, 0, 20, 20) for s in samples)
    assert all(s.group_id == "photo" for s in samples)
    assert samples == sorted(samples, key=lambda s: (s.group_id, s.bbox, s.label))


def test_ninja_annotation_stem_includes_image_extension(tmp_path: Path) -> None:
    """Regression: real mirror uses ``X.jpg.json`` for image ``X.jpg``.

    The annotation stem is ``X.jpg`` while the image stem is ``X``; lookup must
    resolve via the full filename while group_id stays ``Path(img_path).stem``.
    """
    images = tmp_path / "img"
    ann = tmp_path / "ann"
    images.mkdir(parents=True, exist_ok=True)
    ann.mkdir(parents=True, exist_ok=True)
    (images / "X.jpg").write_bytes(_png_bytes(64, 64, seed=7))
    (ann / "X.jpg.json").write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "classTitle": "maize",
                        "points": {"exterior": [[1, 2], [11, 12]]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    samples = parse_ninja_annotations(ann, images)
    assert len(samples) == 1
    assert samples[0].group_id == "X"
    assert samples[0].bbox == (1, 2, 11, 12)
    assert samples[0].source is Source.NINJA


# --- T6 --------------------------------------------------------------------
def test_ninja_group_quota_no_cross_contamination(built) -> None:
    manifest, cfg, root = built
    adaptation = manifest.splits[Split.ADAPTATION]
    ood = manifest.splits[Split.OOD_TEST]
    assert adaptation.count == 400
    assert adaptation.class_counts() == {"Corn": 200, "Weed": 200}
    assert adaptation.groups & ood.groups == frozenset()
    for group in adaptation.groups:
        assert all(sample.group_id != group for sample in ood.samples)
    total = 10 * 60
    assert adaptation.count + ood.count <= total
    assert total - (adaptation.count + ood.count) == manifest.invariants["discarded_boxes"]

    low_ann = root / "ann_low"
    low_ann.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        stem = f"ninja_{i:03d}"
        objects = []
        for j in range(30):
            x1, y1 = (j % 5) * 12, (j // 5) * 12
            objects.append(
                {"classTitle": "maize", "points": {"exterior": [[x1, y1], [x1 + 9, y1 + 9]]}}
            )
        for j in range(15):
            x1, y1 = (j % 5) * 12, (j // 5) * 12
            objects.append(
                {"classTitle": "weed", "points": {"exterior": [[x1, y1], [x1 + 9, y1 + 9]]}}
            )
        (low_ann / f"{stem}.json").write_text(json.dumps({"objects": objects}), encoding="utf-8")
    low_samples = parse_ninja_annotations(low_ann, root / "img")
    with pytest.raises(InsufficientAdaptationCrops):
        split_ninja(low_samples, replace(cfg, ninja_ann_dir=str(low_ann)))


# --- T7 --------------------------------------------------------------------
def _minimal_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": NINE_KEYS,
        "properties": {
            "schema_version": {"const": "1.0"},
            "seed": {"type": "integer"},
            "sources": {"type": "object", "required": ["kaggle", "ninja"]},
            "provenance": {"type": "array"},
            "invariants": {
                "type": "object",
                "required": [
                    "verified",
                    "group_overlap",
                    "hash_overlap",
                    "path_overlap",
                    "class_counts",
                    "counts",
                ],
            },
            "checksum_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    }


def test_manifest_roundtrip_and_verify(built, tmp_path: Path) -> None:
    manifest, _, _ = built
    path = tmp_path / "manifest.json"
    manifest.to_json(path)
    loaded = SplitManifest.from_json(path)
    assert loaded == manifest
    assert loaded.sources == manifest.sources
    assert {"kaggle", "ninja"}.issubset(loaded.sources)
    assert loaded.provenance == manifest.provenance
    assert isinstance(loaded.provenance[0], Sample)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert list(payload) == NINE_KEYS
    assert set(payload["invariants"]) >= {
        "verified",
        "group_overlap",
        "hash_overlap",
        "path_overlap",
        "class_counts",
        "counts",
    }
    try:
        import jsonschema
    except ImportError:
        pass
    else:
        jsonschema.Draft202012Validator(_minimal_schema()).validate(payload)

    assert manifest.verify() == []
    snapshot_config = manifest.config_snapshot["config"]
    assert snapshot_config["seed"] == 42
    assert snapshot_config["image_size"] == 128
    assert list(snapshot_config["scale_range"]) == [0.9, 1.1]
    assert snapshot_config["qsvc_source_per_run"] == 50
    assert snapshot_config["qsvc_target_adaptation_per_run"] == 50
    assert snapshot_config["qsvc_test_per_run"] == 50
    assert snapshot_config["qsvc_runs"] == 10
    assert snapshot_config["qsvc_mixed_train_size"] == 100

    # Seed flexibility (seed=43) is exercised by T4 through build_splits.

    # Negative checks (relationship violations, not schema constants).
    tampered_mixed = replace(
        manifest,
        config_snapshot={
            **manifest.config_snapshot,
            "config": {**snapshot_config, "qsvc_mixed_train_size": 99},
        },
    )
    assert "qsvc_mixed_train_size" in tampered_mixed.verify()
    assert "seed_config_mismatch" in replace(manifest, seed=7).verify()

    adaptation_group = sorted(manifest.splits[Split.ADAPTATION].groups)[0]
    ood_id = min(manifest.splits[Split.OOD_TEST].sample_ids)
    provenance = list(manifest.provenance)
    for index, sample in enumerate(provenance):
        if sample.sample_id == ood_id:
            provenance[index] = replace(sample, group_id=adaptation_group)
            break
    tampered_group = replace(manifest, provenance=tuple(provenance))
    assert any("group_overlap" in violation for violation in tampered_group.verify())
    assert "checksum_mismatch" in replace(manifest, checksum_sha256="0" * 64).verify()


# --- T8 --------------------------------------------------------------------
def test_transforms_exact_parameters() -> None:
    cfg = ExperimentConfig()
    transforms = make_transforms(cfg)
    train = transforms["train"].transforms
    by_type = {type(transform).__name__: transform for transform in train}
    affine = by_type["RandomAffine"]
    assert tuple(affine.degrees) == (-15, 15)
    assert tuple(affine.scale) == (0.9, 1.1)
    assert by_type["RandomHorizontalFlip"].p == 0.5
    assert by_type["RandomVerticalFlip"].p == 0.2
    jitter = by_type["ColorJitter"]
    # torchvision normalises a scalar jitter j to (max(0, 1-j), 1+j); the
    # dependency-free fallback stores the scalar amount itself. Accept both.
    expected = (max(0.0, 1.0 - cfg.jitter), 1.0 + cfg.jitter)
    for attribute in ("brightness", "contrast", "saturation"):
        value = getattr(jitter, attribute)
        if isinstance(value, (tuple, list)):
            assert tuple(value) == pytest.approx(expected)
        else:
            assert value == pytest.approx(cfg.jitter)
    hue = jitter.hue
    if hue is None:
        pass  # this torchvision version stores None for zero hue jitter
    elif isinstance(hue, (tuple, list)):
        assert tuple(hue) == pytest.approx((0.0, 0.0))
    else:
        assert hue == pytest.approx(0.0)
    assert tuple(by_type["Normalize"].mean) == tuple(cfg.norm_mean)

    eval_types = {type(transform).__name__ for transform in transforms["eval"].transforms}
    assert not (
        {"RandomAffine", "RandomHorizontalFlip", "RandomVerticalFlip", "ColorJitter"}
        & eval_types
    )


# --- T9 --------------------------------------------------------------------
def test_transform_shapes_and_eval_determinism() -> None:
    Image = pytest.importorskip("PIL.Image")
    import torch

    cfg = ExperimentConfig()
    transforms = make_transforms(cfg)
    image = Image.new("RGB", (64, 96), (30, 60, 90))
    for kind in ("train", "eval"):
        tensor = transforms[kind](image)
        assert tuple(tensor.shape) == (3, 128, 128)
        assert tensor.dtype == torch.float32
        assert bool(torch.isfinite(tensor).all())
    assert torch.equal(transforms["eval"](image), transforms["eval"](image))

    torch.manual_seed(1)
    first = transforms["train"](image)
    torch.manual_seed(2)
    second = transforms["train"](image)
    assert not torch.equal(first, second)


# --- T10 -------------------------------------------------------------------
def test_qsvc_runs_leak_free_reproducible_exhaustive(built) -> None:
    manifest, cfg, _ = built
    ood_ids = manifest.splits[Split.OOD_TEST].sample_ids
    adaptation_ids = manifest.splits[Split.ADAPTATION].sample_ids
    train_ids = manifest.splits[Split.TRAIN].sample_ids
    for run_idx in range(cfg.qsvc_runs):
        for protocol in (QSVCProtocol.MIXED, QSVCProtocol.NO_MIXING):
            run = sample_qsvc_pairs(manifest, cfg, run_idx, protocol)
            assert len(run.train) == cfg.qsvc_mixed_train_size == 100
            assert len(run.test) == cfg.qsvc_test_per_run == 50
            assert len(run.source_train) == (50 if protocol is QSVCProtocol.MIXED else 100)
            assert len(run.target_train) == (50 if protocol is QSVCProtocol.MIXED else 0)
            assert {s.group_id for s in run.train} & {s.group_id for s in run.test} == set()
            assert {s.content_hash for s in run.train} & {s.content_hash for s in run.test} == set()
            assert {s.path for s in run.train} & {s.path for s in run.test} == set()
            expected_seed = int(
                SeedSequence([cfg.seed, cfg.seed_offset])
                .spawn(cfg.qsvc_runs)[run_idx]
                .generate_state(1)[0]
            )
            assert run.seed == expected_seed
            assert {s.sample_id for s in run.test} <= ood_ids
            assert {s.sample_id for s in run.source_train} <= train_ids
            if protocol is QSVCProtocol.MIXED:
                assert {s.sample_id for s in run.target_train} <= adaptation_ids
    assert sample_qsvc_pairs(manifest, cfg, 3, QSVCProtocol.MIXED) == sample_qsvc_pairs(
        manifest, cfg, 3, QSVCProtocol.MIXED
    )
    assert sample_qsvc_pairs(manifest, cfg, 3, QSVCProtocol.MIXED) != sample_qsvc_pairs(
        manifest, cfg, 4, QSVCProtocol.MIXED
    )


# --- T11 -------------------------------------------------------------------
def test_no_physical_train_test_aliasing(built) -> None:
    pytest.importorskip("PIL.Image")
    manifest, _, root = built
    paths = materialise_adaptation_crops(
        manifest.splits[Split.ADAPTATION].samples, root / "derived"
    )
    assert len(paths) == 400
    crop_dir = root / "derived" / "crops" / "ninja_adaptation"
    assert len(list(crop_dir.glob("*.png"))) == 400
    ood_dir = root / "derived" / "ood_test"
    assert not ood_dir.exists() or not list(ood_dir.glob("*.png"))
    assert manifest.verify() == []


# --- T12 -------------------------------------------------------------------
def test_qsvc_no_mixing_control_is_paired_and_leak_free(built) -> None:
    manifest, cfg, _ = built
    ood_ids = manifest.splits[Split.OOD_TEST].sample_ids
    adaptation_ids = manifest.splits[Split.ADAPTATION].sample_ids
    train_ids = manifest.splits[Split.TRAIN].sample_ids
    for run_idx in range(cfg.qsvc_runs):
        mixed = sample_qsvc_pairs(manifest, cfg, run_idx, QSVCProtocol.MIXED)
        control = sample_qsvc_pairs(manifest, cfg, run_idx, QSVCProtocol.NO_MIXING)
        assert [s.sample_id for s in mixed.test] == [s.sample_id for s in control.test]
        assert len(mixed.train) == len(control.train) == cfg.qsvc_mixed_train_size == 100
        assert control.target_train == ()
        assert {s.sample_id for s in mixed.target_train} <= adaptation_ids
        assert {s.sample_id for s in control.source_train} <= train_ids
        assert {s.sample_id for s in control.test} <= ood_ids
        assert {s.group_id for s in control.train} & {s.group_id for s in control.test} == set()
        assert {s.content_hash for s in control.train} & {s.content_hash for s in control.test} == set()
    result = paired_t_test([0.70 + 0.01 * i for i in range(10)], [0.60 + 0.02 * i for i in range(10)])
    assert result["n"] == 10
    assert 0.0 <= result["p_value"] <= 1.0
