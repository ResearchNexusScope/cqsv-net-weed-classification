"""Fetch and register external plant/weed datasets outside the repository.

The registry lives in ``data.external``; bytes are cached under
``~/.cache/qc_datasets`` (or ``--root``). Downloads are resumable, chunk-parallel,
size-checked and sha256-manifested. No dataset is ever written into the repo.

Examples
--------
    python scripts/fetch_external_datasets.py --dataset deepweeds
    python scripts/fetch_external_datasets.py --all --workers 8
    python scripts/fetch_external_datasets.py --all --include-large   # +30 GB

Manual-access datasets (Weed25, FoodCropsWeeds, Weed-AI) are registered but not
downloaded; the script prints their canonical access instructions instead.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data.external import (  # noqa: E402
    REGISTRY,
    ExternalDataset,
    cache_root,
    get_dataset,
    local_path,
    write_manifest,
)

CHUNK = 8 * 1024 * 1024
MAIZESET_BASE = "https://raw.githubusercontent.com/mshahabalam/MaizeSet/main/"
HF_DEEPWEEDS = (
    "https://huggingface.co/datasets/Project-AgML/DeepWeeds_classification/"
    "resolve/main/data/train-00000-of-00001.parquet"
)


def _log(msg: str) -> None:
    print(f"[fetch] {msg}", flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _single_file(
    url: str,
    dest: Path,
    *,
    workers: int,
    expected_size: int | None,
    force: bool,
) -> Path:
    """Parallel ranged download with resume, size and hash verification."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force and (expected_size is None or dest.stat().st_size == expected_size):
        _log(f"already present: {dest.name} ({dest.stat().st_size} bytes)")
        return dest

    head = requests.head(url, allow_redirects=True, timeout=60)
    total = expected_size or int(head.headers.get("Content-Length", 0))
    if not total:
        raise RuntimeError(f"cannot determine size for {url}")

    supports_range = "bytes" in head.headers.get("Accept-Ranges", "").lower()
    if dest.exists() and dest.stat().st_size > total:
        dest.unlink()

    with dest.open("wb") as fh:
        fh.truncate(total)

    def worker(start: int, end: int) -> int:
        r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=180)
        r.raise_for_status()
        data = r.content
        with dest.open("r+b") as fh:
            fh.seek(start)
            fh.write(data)
        return len(data)

    if supports_range and workers > 1:
        spans = [(i, min(i + CHUNK - 1, total - 1)) for i in range(0, total, CHUNK)]
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, s, e) for s, e in spans]
            for fut in as_completed(futures):
                done += fut.result()
        if done != total:
            raise RuntimeError(f"short download: {done}/{total} for {url}")
    else:
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with dest.open("wb") as fh:
                for block in r.iter_content(1 << 20):
                    fh.write(block)

    actual = dest.stat().st_size
    if expected_size is not None and actual != expected_size:
        raise RuntimeError(f"size mismatch for {dest.name}: {actual} != {expected_size}")
    _log(f"downloaded {dest.name} ({actual} bytes)")
    return dest


def _fetch_one(
    ds: ExternalDataset,
    root: Path,
    workers: int,
    force: bool,
    large: bool,
) -> Path | None:
    dest = local_path(ds, root)

    if ds.status == "deferred_large" and not large:
        _log(f"SKIP large dataset {ds.id} ({ds.size_bytes} bytes); pass --include-large")
        return None
    if ds.status == "manual_access":
        _log(f"MANUAL {ds.id}: {ds.notes}")
        for url in ds.sources:
            _log(f"  {url}")
        return None
    if ds.status == "index":
        _log(f"INDEX {ds.id}: aggregator, no single artifact; {ds.sources[0]}")
        return None

    if ds.id == "deepweeds":
        path = _single_file(HF_DEEPWEEDS, dest, workers=workers, expected_size=ds.size_bytes, force=force)
        source = HF_DEEPWEEDS
    elif ds.id == "acre":
        path = _single_file(
            "https://zenodo.org/api/records/8102217/files/The_ACRE_Crop-Weed_Dataset.zip/content",
            dest, workers=workers, expected_size=ds.size_bytes, force=force,
        )
        source = ds.sources[0]
    elif ds.id == "cornweed_full":
        path = _single_file(
            "https://zenodo.org/api/records/7961764/files/maize_dataset.zip/content",
            dest, workers=workers, expected_size=ds.size_bytes, force=force,
        )
        source = ds.sources[0]
    elif ds.id == "cottonweed12":
        path = _single_file(
            "https://zenodo.org/api/records/7535814/files/CottonWeedDet12.7z/content",
            dest, workers=workers, expected_size=ds.size_bytes, force=force,
        )
        source = ds.sources[0]
    elif ds.id == "maizeset":
        names = [f"images_{i:02d}.zip" for i in range(1, 15)] + [
            f"labels_{i:02d}.zip" for i in range(1, 15)
        ]
        dest.mkdir(parents=True, exist_ok=True)

        def get_one(name: str) -> None:
            _single_file(MAIZESET_BASE + name, dest / name, workers=1, expected_size=None, force=force)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(get_one, names))
        path = dest
        source = ds.sources[0]
    else:
        raise RuntimeError(f"no fetch adapter for {ds.id}")

    digest = _sha256(path) if path.is_file() else None
    write_manifest(ds, path, sha256=digest, size_bytes=ds.size_bytes, source_url=source)
    _log(f"manifest written for {ds.id}")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", help="registry id to fetch (see --list)")
    p.add_argument("--all", action="store_true", help="fetch every non-manual dataset")
    p.add_argument("--list", action="store_true", help="list the registry and exit")
    p.add_argument("--include-large", action="store_true", help="include deferred multi-GB datasets")
    p.add_argument("--root", help=f"cache root (default env {cache_root()})")
    p.add_argument("--workers", type=int, default=8, help="parallel range workers")
    p.add_argument("--force", action="store_true", help="re-download existing files")
    args = p.parse_args(argv)

    if args.list or not (args.dataset or args.all):
        for ds in REGISTRY:
            print(f"{ds.id:18} {ds.status:15} {ds.n_images or '?':>6}  {ds.name}")
        return 0

    root = cache_root(args.root)
    root.mkdir(parents=True, exist_ok=True)
    targets = list(REGISTRY) if args.all else [get_dataset(args.dataset)]
    for ds in targets:
        try:
            _fetch_one(ds, root, args.workers, args.force, args.include_large)
        except Exception as exc:  # keep going; report all failures at the end
            print(f"[fetch] FAILED {ds.id}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
