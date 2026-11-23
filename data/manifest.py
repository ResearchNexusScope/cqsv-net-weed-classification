"""Immutable split manifest: normative schema, roundtrip and verification gate.

The nine authoritative top-level keys (spec section 5) are::

    schema_version, generated_by, seed, config_snapshot, sources, splits,
    provenance, invariants, checksum_sha256

``sources`` and ``provenance`` both round-trip verbatim, and ``invariants`` is
emitted in full (verified, group/hash/path overlap, class counts and counts).
The checksum is computed from the canonical JSON (sorted keys, compact
separators) with ``checksum_sha256`` **and** the volatile
``config_snapshot.generated_at`` removed, so two builds with the same data and
config have the same checksum.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from data.splits import (  # re-exported so callers may import value types here
    DedupReport,
    QSVCProtocol,
    QSVCRun,
    Sample,
    Source,
    Split,
    SplitRecord,
)

__all__ = [
    "Source",
    "Split",
    "Sample",
    "SplitRecord",
    "DedupReport",
    "QSVCProtocol",
    "QSVCRun",
    "compute_checksum",
    "canonical_json",
    "SplitManifest",
]

NORMATIVE_KEYS = (
    "schema_version",
    "generated_by",
    "seed",
    "config_snapshot",
    "sources",
    "splits",
    "provenance",
    "invariants",
    "checksum_sha256",
)
PROTOCOL_SOURCE_PER_RUN = 50
PROTOCOL_TARGET_PER_RUN = 50
PROTOCOL_MIXED_TRAIN_SIZE = 100
PROTOCOL_TEST_PER_RUN = 50
PROTOCOL_RUNS = 10


def canonical_json(payload: Mapping) -> str:
    """Canonical JSON for hashing: sorted keys, compact separators, no volatile fields."""
    clean = copy.deepcopy(dict(payload))
    clean.pop("checksum_sha256", None)
    clean.pop("generated_at", None)  # advisory: exclude volatile timestamp anywhere
    snapshot = clean.get("config_snapshot")
    if isinstance(snapshot, dict):
        snapshot.pop("generated_at", None)
        # git_commit is volatile build provenance, not data/config content.  Including
        # it would make the checksum change on every commit and break the
        # determinism contract for two identical builds made across a commit boundary.
        snapshot.pop("git_commit", None)
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_checksum(payload: Mapping) -> str:
    """SHA-256 of the canonical JSON, excluding checksum and generated_at."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _sample_to_dict(sample: Sample) -> dict:
    record = {
        "sample_id": sample.sample_id,
        "source": sample.source.value if isinstance(sample.source, Source) else str(sample.source),
        "group_id": sample.group_id,
        "content_hash": sample.content_hash,
        "path": sample.path,
        "bbox": list(sample.bbox) if sample.bbox is not None else None,
        "label": int(sample.label),
        "label_name": sample.label_name,
    }
    return record


def _sample_from_dict(record: Mapping) -> Sample:
    values = dict(record)
    values["source"] = Source(values["source"])
    if values.get("bbox") is not None:
        values["bbox"] = tuple(int(v) for v in values["bbox"])
    return Sample(**values)


def _record_to_dict(record: SplitRecord) -> dict:
    """Serialise a SplitRecord using the compact normative shape (section 5)."""
    return {
        "count": int(record.count),
        "group_count": len(record.groups),
        "groups": sorted(record.groups),
        "class_counts": record.class_counts(),
        "sample_ids": sorted(record.sample_ids),
        "hash_digest": hashlib.sha256(
            "".join(sorted(record.hashes)).encode("utf-8")
        ).hexdigest(),
    }


@dataclass(frozen=True)
class SplitManifest:
    schema_version: str = "1.0"
    generated_by: str = "experiments.data.build_splits"
    seed: int = 42
    config_snapshot: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    splits: Mapping[Split, SplitRecord] = field(default_factory=dict)
    provenance: tuple[Sample, ...] = ()
    invariants: dict = field(default_factory=dict)
    checksum_sha256: str = ""

    # -- serialisation ------------------------------------------------------
    def _to_payload(self) -> dict:
        """Build the nine-key normative JSON payload in canonical key order."""
        split_items = sorted(self.splits.items(), key=lambda item: str(item[0]))
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "seed": int(self.seed),
            "config_snapshot": copy.deepcopy(dict(self.config_snapshot)),
            "sources": copy.deepcopy(dict(self.sources)),
            "splits": {
                (key.value if isinstance(key, Split) else str(key)): _record_to_dict(record)
                for key, record in split_items
            },
            "provenance": [
                _sample_to_dict(sample)
                for sample in sorted(self.provenance, key=lambda s: s.sample_id)
            ],
            "invariants": copy.deepcopy(dict(self.invariants)),
            "checksum_sha256": self.checksum_sha256,
        }

    def to_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self._to_payload()
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SplitManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        samples = tuple(_sample_from_dict(record) for record in payload["provenance"])
        by_id = {sample.sample_id: sample for sample in samples}
        splits: dict[Split, SplitRecord] = {}
        for key, record in payload["splits"].items():
            split = Split(key)
            splits[split] = SplitRecord(
                name=split,
                samples=tuple(by_id[sample_id] for sample_id in record["sample_ids"]),
            )
        return cls(
            schema_version=payload["schema_version"],
            generated_by=payload["generated_by"],
            seed=int(payload["seed"]),
            config_snapshot=payload["config_snapshot"],
            sources=payload["sources"],
            splits=splits,
            provenance=samples,
            invariants=payload["invariants"],
            checksum_sha256=payload["checksum_sha256"],
        )

    # -- verification gate --------------------------------------------------
    def verify(self) -> list[str]:
        """Return a list of violations; ``[]`` means the manifest is sound.

        Protocol *relationships* are checked here (not as schema constants),
        so any integer seed -- including ``QC_SEED=43`` -- remains legal.
        """
        violations: list[str] = []
        config = dict(self.config_snapshot.get("config") or {})

        if config.get("seed") != int(self.seed):
            violations.append("seed_config_mismatch")

        source_n = config.get("qsvc_source_per_run")
        target_n = config.get("qsvc_target_adaptation_per_run")
        test_n = config.get("qsvc_test_per_run")
        runs_n = config.get("qsvc_runs")
        mixed_n = config.get("qsvc_mixed_train_size")

        if source_n != PROTOCOL_SOURCE_PER_RUN or target_n != PROTOCOL_TARGET_PER_RUN:
            violations.append("qsvc_protocol_mismatch")
        if mixed_n != PROTOCOL_MIXED_TRAIN_SIZE or (
            isinstance(source_n, int)
            and isinstance(target_n, int)
            and source_n + target_n != mixed_n
        ):
            violations.append("qsvc_mixed_train_size")
        if test_n != PROTOCOL_TEST_PER_RUN:
            violations.append("qsvc_test_per_run")
        if runs_n != PROTOCOL_RUNS:
            violations.append("qsvc_runs")

        if not {"kaggle", "ninja"}.issubset(set(self.sources)):
            violations.append("sources_missing")
        if self.invariants.get("verified") is not True:
            violations.append("invariants_unverified")

        by_id: dict[str, Sample] = {}
        for sample in self.provenance:
            if sample.sample_id in by_id:
                violations.append("duplicate_sample_ids")
                break
            by_id[sample.sample_id] = sample
        if len(self.provenance) != sum(record.count for record in self.splits.values()):
            violations.append("provenance_count_mismatch")

        expected_names = {split.value for split in Split}
        if {_split_name(key) for key in self.splits} != expected_names:
            violations.append("splits_missing")

        names = sorted(self.splits, key=lambda s: (s.value if isinstance(s, Split) else str(s)))
        for index, left in enumerate(names):
            left_record = self.splits[left]
            for right in names[index + 1 :]:
                right_record = self.splits[right]
                for attribute, label in (
                    ("group_id", "group_overlap"),
                    ("content_hash", "hash_overlap"),
                    ("path", "path_overlap"),
                ):
                    a = {getattr(s, attribute) for s in left_record.samples}
                    b = {getattr(s, attribute) for s in right_record.samples}
                    # Also inspect provenance: a caller may tamper the
                    # authoritative provenance without editing split records.
                    a |= {
                        getattr(by_id[sid], attribute)
                        for sid in left_record.sample_ids
                        if sid in by_id
                    }
                    b |= {
                        getattr(by_id[sid], attribute)
                        for sid in right_record.sample_ids
                        if sid in by_id
                    }
                    if a & b:
                        violations.append(
                            f"{label}:{_split_name(left)}__{_split_name(right)}"
                        )

        for split, record in self.splits.items():
            if record.count != len(record.samples):
                violations.append(f"count_mismatch:{_split_name(split)}")
            classes = record.class_counts()
            if set(classes) != {"Corn", "Weed"} or sum(classes.values()) != record.count:
                violations.append(f"class_counts_invalid:{_split_name(split)}")
            recorded = dict(self.invariants.get("counts") or {})
            if recorded.get(_split_name(split)) != record.count:
                violations.append(f"invariant_count_mismatch:{_split_name(split)}")
            recorded_classes = dict(self.invariants.get("class_counts") or {})
            if recorded_classes.get(_split_name(split)) != record.class_counts():
                violations.append(f"class_count_mismatch:{_split_name(split)}")

        if compute_checksum(self._to_payload()) != self.checksum_sha256:
            violations.append("checksum_mismatch")
        return violations


def _split_name(split) -> str:
    return split.value if isinstance(split, Split) else str(split)
