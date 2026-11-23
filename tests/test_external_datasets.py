"""Synthetic tests for the external dataset registry and fetch CLI.

No network, no downloads, no real dataset bytes: only registry metadata and a
manifest round-trip in tmp_path are exercised.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data import external


def test_registry_covers_every_listed_dataset() -> None:
    ids = {d.id for d in external.REGISTRY}
    assert ids == {
        "acre",
        "deepweeds",
        "maizeset",
        "cornweed_full",
        "cottonweed12",
        "weed25",
        "foodcropsweeds",
        "weedai",
    }
    assert len(ids) == len(external.REGISTRY)


def test_get_dataset_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown external dataset"):
        external.get_dataset("not-a-dataset")


def test_status_filters_are_disjoint() -> None:
    seen: set[str] = set()
    for status in ("fetchable", "deferred_large", "manual_access", "index"):
        rows = external.list_datasets(status)
        assert rows
        for row in rows:
            assert row.status == status
            assert row.id not in seen
            seen.add(row.id)
    assert seen == {d.id for d in external.REGISTRY}


def test_artifact_paths_stay_inside_cache_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(external.DEFAULT_CACHE_ENV, str(tmp_path / "cache"))
    for ds in external.REGISTRY:
        target = external.local_path(ds)
        assert target.is_relative_to(external.cache_root())
        assert ".." not in Path(ds.artifact).parts


def test_cache_root_precedence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(external.DEFAULT_CACHE_ENV, str(tmp_path))
    assert external.cache_root() == tmp_path
    assert external.cache_root(tmp_path / "other") == tmp_path / "other"


def test_manifest_roundtrip_records_provenance(tmp_path: Path) -> None:
    ds = external.get_dataset("deepweeds")
    payload_path = tmp_path / ds.artifact
    payload_path.parent.mkdir(parents=True)
    payload_path.write_bytes(b"x")
    out = external.write_manifest(
        ds,
        payload_path,
        sha256="abc123",
        size_bytes=1,
        source_url="https://example.test/data",
    )
    assert out == payload_path.with_suffix(payload_path.suffix + ".manifest.json")
    data = external.read_manifest(payload_path)
    assert data["dataset"]["id"] == "deepweeds"
    assert data["dataset"]["license_status"] == "verified"
    assert data["sha256"] == "abc123"
    assert data["size_bytes"] == 1
    assert data["source_url"] == "https://example.test/data"


def test_manifest_is_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "acre" / "archive.zip"
    external.write_manifest(external.get_dataset("acre"), path)
    text = path.with_suffix(path.suffix + ".manifest.json").read_text(encoding="utf-8")
    assert json.loads(text)["dataset"]["id"] == "acre"


def test_fetch_cli_list_is_offline(tmp_path: Path, monkeypatch, capsys) -> None:
    from scripts import fetch_external_datasets as fetch

    monkeypatch.setenv(external.DEFAULT_CACHE_ENV, str(tmp_path))
    assert fetch.main(["--list"]) == 0
    out = capsys.readouterr().out
    for ds in external.REGISTRY:
        assert ds.id in out


def test_fetch_cli_manual_datasets_never_download(tmp_path: Path, monkeypatch, capsys) -> None:
    from scripts import fetch_external_datasets as fetch

    monkeypatch.setenv(external.DEFAULT_CACHE_ENV, str(tmp_path))
    assert fetch.main(["--dataset", "weed25"]) == 0
    out = capsys.readouterr().out
    assert "MANUAL" in out
    assert not list(tmp_path.rglob("*.zip"))
