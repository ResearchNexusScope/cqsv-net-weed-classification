# External dataset acquisition report (2026-11-23)

> Companion to `docs/research/QC_DATASET_RESEARCH.md`. All eight surveyed datasets
> are now registered in `data/external.py` and fetchable through
> `scripts/fetch_external_datasets.py`. Downloaded bytes live outside the repo under
> `~/.cache/qc_datasets` and are never committed. Every fetched artifact has a
> `.manifest.json` sidecar recording source URL, licence, expected size and SHA-256.

## 1. Actually fetched and verified

| id | artifact | bytes observed | bytes expected | SHA-256 (first 16) | status |
|---|---|---:|---:|---|---|
| `acre` | `acre/The_ACRE_Crop-Weed_Dataset.zip` | 1,220,608,171 | 1,220,608,171 | `9dc580237619e9cf` | verified - exact size match |
| `deepweeds` | `deepweeds/data/train-00000-of-00001.parquet` | 492,631,617 | 492,631,617 | `c86358c3299b3f17` | verified - exact size match |
| `maizeset` | `maizeset/` (28 zips: 14 images + 14 labels) | 255,094,669 | 255,094,669 | directory (no single hash) | verified - 28 files present |

Spot check: `maizeset/images_01.zip` contains 250 entries such as
`images_01/maize_1.jpg`, confirming the maize-vs-weed image layout.

DeepWeeds parquet read back after installing `pyarrow`: shape `(17509, 3)` with
columns `image`, `label`, `species`. Label counts are `{0:1125, 1:1064, 2:1031,
3:1022, 4:1062, 5:1009, 6:1074, 7:1016, 8:9106}` - eight weed species at roughly
1k each plus a 9,106-image negative class, matching the published description.

Total downloaded: **~1.93 GB**.

## 2. Registered but deliberately not downloaded

| id | reason | how to obtain |
|---|---|---|
| `cornweed_full` | 8.7 GB `maize_dataset.zip`; measured network ~0.5 MB/s (many hours) | `python scripts/fetch_external_datasets.py --dataset cornweed_full --include-large --workers 8` (resumable) |
| `cottonweed12` | 28.6 GB `.7z`; same bandwidth constraint, plus needs `py7zr`/7-Zip to extract | `--dataset cottonweed12 --include-large --workers 8` |
| `weed25` | Canonical copy is Baidu Netdisk with password; GitHub holds an empty LFS pointer. Roboflow mirror (9,553 images) needs an account | URLs recorded in the registry; manual download |
| `foodcropsweeds` | No direct public archive found; distributed via dataset portals/Ultralytics | URLs in the registry; manual download |
| `weedai` | Aggregator, not one dataset; individual records (CornWeed, CottonWeedDet12) already registered | `https://weed-ai.sydney.edu.au/datasets/` |

The fetcher prints these instructions instead of failing when run with `--all`.

## 3. Reproduce

```bash
# list the registry (offline)
python scripts/fetch_external_datasets.py --list

# fetch all feasible datasets (skips large + manual)
python scripts/fetch_external_datasets.py --all --workers 8

# fetch only one, optionally including multi-GB archives
python scripts/fetch_external_datasets.py --dataset acre
python scripts/fetch_external_datasets.py --dataset cornweed_full --include-large
```

Downloads are chunk-parallel and resumable: an existing file of the expected size
is skipped; a too-large partial file is truncated and refetched. SHA-256 is computed
after completion and stored in the sidecar manifest.

## 4. Licence and usage caveats

- **MaizeSet states no licence** in its repository. It is cached for research
  evaluation only and must not be redistributed or used in a claim until the
  author confirms terms.
- **Weed25 and FoodCropsWeeds licences are unverified**; check the publisher before
  use. DeepWeeds has conflicting secondary catalog notes (CC BY vs CC BY-SA); the
  original repository and paper govern.
- ACRE (CC BY 4.0) and DeepWeeds data (CC BY 4.0) are the only two with confirmed
  permissive terms, so they are the recommended first candidates for a new
  protocol.
- Detection datasets need group-disjoint crop extraction before they can feed the
  image-level QNN feature loader. No such extraction has been performed yet.
- Adding a new dataset changes the split manifest and every checksum. Any future
  run must start a new result tree with a new D-ID and pass the same verifier +
  Critic gates; do not mix these datasets into existing result tables.

## 5. Verification performed

- `tests/test_external_datasets.py`: 9 synthetic tests (registry completeness,
  disjoint statuses, cache-root precedence, path containment, manifest round-trip,
  offline CLI). All pass.
- Full central verifier after registry + QHEAD-D02 fix: **200 passed, 13 skipped,
  exit 0**.
- Byte sizes of all three fetched artifacts checked against the registry; MaizeSet
  archive contents spot-checked; manifest sidecars read back successfully.
