# QC-D02 — Leak-free data layer design

**D-ID:** QC-D02 (research worker, Quantum_Classification rebuild swarm)
**Owner file (ONLY file written):** `docs/research/QC-D02_data_pipeline.md`
**Status:** DESIGN SPECIFICATION for orchestrator implementation in `experiments/data.py`
**Date:** 2026-11-22
**Mode:** read-only investigation; no code, builds, tests, or commits performed by this worker.

---

## Skills loaded and how they informed this design

| Skill | Decision informed |
|---|---|
| `master-scientist` | Bottom-up order (data → metrics → experiments → manuscript); leakage and train/test identity treated as correctness bugs; claim-to-evidence classification; evidence cited by file and line. |
| `ml-for-science` | Group-aware splitting (`GroupKFold`/`StratifiedGroupKFold` discipline), split inside pipeline, no test-set tuning, fixed `random_state`, log split indices. |
| `reproducible-environments` | Config snapshot + library versions + seed recorded in the manifest; env-var paths with no silent developer-path fallback; no `latest`-style ambiguity in data provenance. |
| `deep-learning` | `Dataset`/`DataLoader` contract, train/eval transform separation, deterministic worker seeding, `shuffle=False` for all evaluation loaders, save config with every artifact. |

## Evidence base (files and lines inspected, all local)

| Source | What it establishes |
|---|---|
| `.paper/plant-1.md:67` | MD5 dedup of Kaggle; 400 Ninja adaptation crops (200 maize + 200 weed) written into train; ~2,393 mixed training images. |
| `.paper/plant-1.md:94-100` | Augmentation claim: rotation ±15°, hflip 0.5, vflip 0.2, brightness/contrast/saturation 0.1, scale 90–110%. |
| `.paper/plant-1.md:366-369` | Table 1 states same ~1,993 images used for train, in-val and in-distribution test; caveat explicitly admits train==test and bbox-level Ninja leak. |
| `.paper/plant-1.md:413` | "validation split is a copy of the training split … this is a training accuracy rather than an independent validation estimate." |
| `archives/legacy-2026-11-22/Benchmark_Clean_Eval.py:239-256` | Dedup computes MD5, then copies **each unique image into both `train` and `test`** — the train==test bug. |
| `archives/legacy-2026-11-22/Benchmark_Multiple_SOTA_extracted.py:101-105` | Comment: "Copy Kaggle images (100% of unique ones into both train and test)" — same bug, explicit. |
| `Benchmark_Clean_Eval.py:92-107`, `Benchmark_Multiple_SOTA_extracted.py:140-175` | Ninja adaptation/test split is performed by `rng.shuffle(all_samples)` over **bounding boxes**, so multiple boxes from one source photo can land in both adaptation and OOD test. |
| `Benchmark_Clean_Eval.py:267-273` | Legacy train transform implements only Resize + RandomHorizontalFlip + RandomRotation(15); vflip 0.2, colour jitter 0.1 and scale 0.9–1.1 are **absent from code**. |
| `archives/legacy-2026-11-22/Run_QSVC_Mixed_Variance.py:125-186` | Ten QSVC runs use ad-hoc seeds `run_idx*42`, `run_idx*123`, `run_idx*456`; the 100-crop Ninja pool is drawn bbox-wise from the full Ninja set, including the 400 adaptation crops. |
| `experiments/config.py:14-70` | seed 42, 128px, batch 32, 15 epochs, lr 0.001, ImageNet norm, augmentation parameters, 10 qubits/1024 features, 200/class adaptation; QSVC fields `qsvc_source_per_run=50`, `qsvc_target_adaptation_per_run=50`, `qsvc_test_per_run=50`, `qsvc_runs=10`, `seed_offset=42`, and derived `qsvc_mixed_train_size=100` (property). |
| `experiments/config.py:51-78` | `from_env()` takes `NINJA_ANN_DIR` / `NINJA_IMG_DIR`; `validate()` fails when they are unset. |
| `experiments/metrics.py:35-68` | IoU definition and `binary_metrics` already provide the metric layer the data layer must feed. |

### Claim-to-evidence table

| Claim | Classification | Note |
|---|---|---|
| Kaggle deduplicated by MD5 | **partially supported** | MD5 is computed (`Benchmark_Clean_Eval.py:250`) but the deduplicated set is then copied into both train and test; no held-out ID split exists. |
| 400 Ninja adaptation crops injected into train, remainder used as OOD test | **partially supported** | Implemented, but at bbox level (`Benchmark_Clean_Eval.py:92-107`), so source photos leak across the boundary. Paper caveat at line 369 acknowledges this. |
| Augmentation: ±15°, hflip 0.5, vflip 0.2, jitter 0.1, scale 0.9–1.1 | **partially supported** | Stated in paper (`plant-1.md:94-100`) and parameterised in `config.py:26-30`; only rotation + hflip exist in legacy code. The new data layer must implement the full claim. |
| Ten QSVC runs with 50 labelled target crops in train and 50 OOD test crops | **partially supported** | Ten-run loop exists but uses three ad-hoc seed formulas and bbox-level sampling from the full Ninja pool (`Run_QSVC_Mixed_Variance.py:125-186`). |
| 128×128 input, ImageNet normalisation, 15 epochs, batch 32, lr 0.001 | **verified** | `config.py:14-30` and legacy transform code (`Benchmark_Clean_Eval.py:262-266`). |

---

## 1. Module API for `experiments/data.py`

The orchestrator can implement this file directly. All names and signatures below are normative. Python ≥3.10, `from __future__ import annotations`.

### 1.1 Value types

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from experiments.config import ExperimentConfig


class Source(str, Enum):
    KAGGLE = "kaggle"
    NINJA = "ninja"


class Split(str, Enum):
    TRAIN = "train"            # Kaggle source-domain training pool
    ID_VAL = "id_val"          # Kaggle in-distribution validation (model selection)
    ID_TEST = "id_test"        # Kaggle in-distribution test (never used for selection)
    ADAPTATION = "adaptation"  # Ninja 200+200 labelled target-domain crops (train side)
    OOD_TEST = "ood_test"      # Ninja remainder: the headline OOD evaluation set


@dataclass(frozen=True)
class Sample:
    """One trainable/evaluable unit. For Ninja, one bounding-box crop."""
    sample_id: str                       # stable, globally unique, e.g. "ninja:img_0042:b2"
    source: Source
    group_id: str                        # KAggle: content hash. Ninja: source image stem/hash
    content_hash: str                    # md5(file bytes) | sha256(group_id + bbox)
    path: str                            # absolute path (Kaggle image OR Ninja source image)
    bbox: tuple[int, int, int, int] | None  # (x1, y1, x2, y2) pixel coords; None for Kaggle
    label: int                           # 0 = Corn/maize, 1 = Weed
    label_name: str                      # "Corn" | "Weed" (paper-facing name)


@dataclass(frozen=True)
class SplitRecord:
    name: Split
    samples: tuple[Sample, ...]
    @property
    def sample_ids(self) -> frozenset[str]: ...
    @property
    def groups(self) -> frozenset[str]: ...
    @property
    def hashes(self) -> frozenset[str]: ...
    @property
    def paths(self) -> frozenset[str]: ...
    @property
    def count(self) -> int: ...
    def class_counts(self) -> dict[str, int]: ...   # {"Corn": n, "Weed": n}


@dataclass(frozen=True)
class DedupReport:
    files_seen: int
    unique_hashes: int
    duplicates_removed: int
    label_conflicts: tuple[str, ...]     # hashes seen with both labels -> hard error upstream


class QSVCProtocol(str, Enum):
    MIXED = "mixed"                          # primary: 50 Kaggle + 50 Ninja adaptation
    NO_MIXING = "no_mixing_control"          # mandatory control: 100 Kaggle, 0 Ninja


@dataclass(frozen=True)
class QSVCRun:
    """One arm of one of the ten paired QSVC runs (see §3.3)."""
    run_idx: int                         # 0..cfg.qsvc_runs-1
    protocol: QSVCProtocol
    seed: int                            # int(SeedSequence([seed, seed_offset]).spawn(runs)[run_idx].generate_state(1)[0])
    source_train: tuple[Sample, ...]     # Kaggle TRAIN: 50 (MIXED) or 100 (NO_MIXING)
    target_train: tuple[Sample, ...]     # Ninja ADAPTATION: 50 (MIXED) or () (NO_MIXING)
    test: tuple[Sample, ...]             # the SAME 50 Ninja OOD_TEST crops in both arms
    @property
    def train(self) -> tuple[Sample, ...]:
        """Always exactly cfg.qsvc_mixed_train_size (100) samples."""
        return self.source_train + self.target_train
```

### 1.2 Manifest type

```python
@dataclass(frozen=True)
class SplitManifest:
    schema_version: str                          # "1.0"
    generated_by: str                            # "experiments.data.build_splits"
    seed: int                                    # equals config_snapshot["config"]["seed"]
    config_snapshot: dict                        # ExperimentConfig.as_dict() + versions/commit
    sources: dict[str, dict]                     # {"kaggle": {...counters...}, "ninja": {...counters...}}
    splits: Mapping[Split, SplitRecord]
    provenance: tuple[Sample, ...]               # sorted by sample_id; every Sample round-trips
    invariants: dict                             # see §5
    checksum_sha256: str                         # sha256 of canonical JSON minus this field

    def verify(self) -> list[str]: ...           # [] iff all invariants hold; see §6
    def to_json(self, path: str | Path) -> None: ...
    @classmethod
    def from_json(cls, path: str | Path) -> "SplitManifest": ...
```

`SplitManifest.verify()` is the single gate every downstream experiment must call before training.

**Serialisation contract (Blocker 1 fix — both `sources` and `provenance` are authoritative and must round-trip):**

- `to_json()` emits exactly the nine top-level keys in the §5 schema: `schema_version`, `generated_by`, `seed`, `config_snapshot`, `sources`, `splits`, `provenance`, `invariants`, `checksum_sha256`. `Split` enum keys are serialised as their `.value` strings; `Sample` dataclasses in `provenance` are serialised via `dataclasses.asdict` (with `source` → `.value`). `sources` is copied verbatim (plain JSON counters).
- `from_json()` reverses this: reconstructs `Split(...)` keys, `Sample(**record)` objects (converting `bbox` list back to a 4-tuple), and returns the identical frozen dataclass. T7 asserts `loaded == original`, `loaded.sources == original.sources` and `loaded.provenance == original.provenance` with `isinstance(loaded.provenance[0], Sample)`.
- `verify()` (Blocker 2 fix — relationships, not schema consts) returns a list of violation strings and must be empty for the default config **and** for `cfg.seed = 43`:
  - `seed != config_snapshot["config"]["seed"]` → violation `"seed_config_mismatch"`.
  - `cfg["qsvc_source_per_run"] != 50` or `cfg["qsvc_target_adaptation_per_run"] != 50` → `"qsvc_protocol_mismatch"`.
  - `cfg["qsvc_source_per_run"] + cfg["qsvc_target_adaptation_per_run"] != cfg["qsvc_mixed_train_size"]` or `cfg["qsvc_mixed_train_size"] != 100` → `"qsvc_mixed_train_size"`.
  - `cfg["qsvc_test_per_run"] != 50` → `"qsvc_test_per_run"`.
  - `cfg["qsvc_runs"] != 10` → `"qsvc_runs"`.
  - plus the group/hash/path overlap, class-count and checksum checks described in §5/§6.
  The JSON Schema validates **types only** for `seed` and all `qsvc_*` fields; any protocol-value enforcement lives here so `QC_SEED` and alternate seeds remain legal.

### 1.3 Functions

```python
def content_hash(path: str | Path, algorithm: str = "md5") -> str:
    """Hash RAW FILE BYTES in 1 MiB chunks (paper parity: md5). Never hash the
    decoded image: two differently-encoded copies of the same photo must match."""

def discover_kaggle(root: str | Path) -> list[Sample]:
    """Scan <root>/<Corn|Weed>/*.{jpg,jpeg,png}; group_id = content_hash(path);
    deterministic order sorted by (group_id, path). Does NOT split or copy."""

def dedup_kaggle(samples: Sequence[Sample]) -> tuple[list[Sample], DedupReport]:
    """Keep first occurrence per content hash in sorted order. A hash observed
    with two different labels is a label_conflict: raise LabelConflictError,
    never silently pick a class."""

def parse_ninja_annotations(ann_dir: str | Path, img_dir: str | Path) -> list[Sample]:
    """JSON per source image: objects[].classTitle in {maize, weed};
    objects[].points.exterior supplies >=2 points -> axis-aligned bbox
    (min/max over ALL points, not just first two). Clamp to image bounds,
    drop non-positive-area boxes, ignore unknown classes. group_id =
    sha256(source_image_basename); content_hash = sha256(group_id + ':' +
    f'{x1},{y1},{x2},{y2}'). Deterministic order sorted by
    (group_id, x1, y1, x2, y2, label)."""

def split_kaggle(samples: Sequence[Sample], cfg: ExperimentConfig) -> dict[Split, SplitRecord]:
    """Group-aware, class-stratified 80/10/10 split. Each group has size 1
    (post-dedup), so StratifiedGroupKFold(n_splits=10, shuffle=True,
    random_state=cfg.seed) is used: fold 0 -> ID_TEST, fold 1 -> ID_VAL,
    folds 2..9 -> TRAIN. Folds are mutually disjoint by construction."""

def split_ninja(samples: Sequence[Sample], cfg: ExperimentConfig) -> dict[Split, SplitRecord]:
    """Whole-source-image (group) allocation with exact 200/200 crop quota.
    Algorithm in §3. Raises InsufficientAdaptationCrops if quota cannot be met."""

def build_splits(cfg: ExperimentConfig) -> SplitManifest:
    """Orchestrator: validate cfg -> discover -> dedup -> split_kaggle ->
    split_ninja -> assemble provenance -> compute invariants -> verify() ->
    return manifest. Pure function of (cfg, on-disk bytes); no random global
    state, no writes except the caller's explicit manifest write."""

def load_manifest(path: str | Path) -> SplitManifest: ...

def make_transforms(cfg: ExperimentConfig) -> dict[str, Callable[[Image.Image], torch.Tensor]]:
    """Returns {'train': ..., 'eval': ...}. Specification in §4."""

def sample_qsvc_pairs(
    manifest: SplitManifest,
    cfg: ExperimentConfig,
    run_idx: int,
    protocol: QSVCProtocol = QSVCProtocol.MIXED,
) -> QSVCRun:
    """Return one arm of one paired run. Spawn the per-run SeedSequence
    run_ss = SeedSequence([cfg.seed, cfg.seed_offset]).spawn(cfg.qsvc_runs)[run_idx];
    pass run_ss itself to np.random.default_rng(run_ss) and record
    run.seed = int(run_ss.generate_state(1)[0]).

    Draw order is FIXED so the two arms are paired:
      1. test = cfg.qsvc_test_per_run crops from OOD_TEST (drawn first; identical
         for MIXED and NO_MIXING at the same run_idx);
      2. MIXED: source_train = cfg.qsvc_source_per_run (50) crops from TRAIN and
         target_train = cfg.qsvc_target_adaptation_per_run (50) crops from
         ADAPTATION -> len(train) == cfg.qsvc_mixed_train_size (100).
         NO_MIXING: source_train = cfg.qsvc_mixed_train_size (100) crops from
         TRAIN and target_train = () -> len(train) == 100.
    Never samples across the adaptation/OOD group boundary. See §3.3 and the
    RESOLVED Q7 ruling in §7.
    """

class ImageDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform: Callable[[Image.Image], torch.Tensor] | None = None):
        ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """Returns (image_tensor, label, sample_id). For Ninja samples, open
        sample.path and crop sample.bbox BEFORE applying transform."""

def make_dataloaders(
    manifest: SplitManifest,
    cfg: ExperimentConfig,
    *,
    splits: Sequence[Split] = (Split.TRAIN, Split.ID_VAL, Split.ID_TEST, Split.OOD_TEST),
    num_workers: int = 4,
) -> dict[Split, DataLoader]:
    """shuffle=True ONLY for Split.TRAIN, with an explicit
    torch.Generator().manual_seed(cfg.seed); shuffle=False everywhere else.
    pin_memory=True. pin_memory_device consistent with the selected device."""
```

**Return-type contract for the orchestrator:** `build_splits()` returns a `SplitManifest`; callers consume `manifest.splits[Split.TRAIN].samples` etc. and pass those tuples to `ImageDataset`. Nothing returns a bare list of file paths — every unit carries `sample_id`, `group_id` and `content_hash` so leakage checks remain possible all the way to evaluation.

---

## 2. Group-aware splitting and elimination of the legacy train==test bug

### 2.1 Atomic unit = group, never row

Two distinct group keys are used, both derived from the *source photograph*, not from the crop:

- **Kaggle:** `group_id = md5(raw file bytes)`. Byte-identical files collapse to one group. The legacy code also computed this hash but then violated it downstream.
- **DatasetNinja:** `group_id = sha256(source image basename)`. Every bounding box cut from one photo shares this group. Crops within a photo are never separated.

All splits are computed over the set of group IDs. A split is valid only if `groups(A) ∩ groups(B) = ∅` for every pair of splits in the three-way Kaggle partition and in the two-way Ninja partition. Sample-level path/hash disjointness follows.

### 2.2 Three-way Kaggle partition

```
raw Kaggle train/ + test/ dirs (read-only)
        |  discover all Corn/ + Weed/ images from BOTH dirs
        v
  content_hash(md5) -> dedup across the UNION (not within each dir)
        v
  unique groups (~1,993 in the paper)
        |
        +-- StratifiedGroupKFold(10, shuffle=True, random_state=42), fold 0 -> ID_TEST
        +-- fold 1 -> ID_VAL
        +-- folds 2..9 -> TRAIN
```

The original Kaggle `train/` and `test/` directory names are **ignored** after discovery: they are two views of one pool, and treating them as separate splits is precisely the legacy error. There is no physical copying of an image into two destinations. The derived data root contains only a manifest and (for Ninja) materialised adaptation crops; OOD crops stay lazy `(path, bbox)` references.

### 2.3 How the old bug is eliminated

| Legacy behaviour | Location | New behaviour |
|---|---|---|
| Each unique image copied into `SPLIT_DIR/train/<cls>` **and** `SPLIT_DIR/test/<cls>` | `Benchmark_Clean_Eval.py:254-255`; `Benchmark_Multiple_SOTA_extracted.py:104-105` | One group is assigned to exactly one of `TRAIN`, `ID_VAL`, `ID_TEST`. `SplitManifest.verify()` asserts zero group/hash/path overlap. |
| `train_dataset` and `test_dataset` read the mirrored directories | `Benchmark_Clean_Eval.py:279-280` | `TRAIN` and `ID_TEST`/`ID_VAL` sample tuples are disjoint by construction; no directory aliasing possible. |
| "Validation" accuracy was training accuracy | `plant-1.md:413` | `ID_VAL` is an independent 10% holdout used only for model selection; `ID_TEST` is reported once, never used for early stopping or tuning. |
| Model selection on the leaked split | `plant-1.md:413` | Selection uses `ID_VAL`; OOD test touched only for final evaluation, consistent with `ml-for-science` ("test-set tuning invalidates the estimate"). |

Consequence to flag in the manuscript: with a genuine ID holdout, Table 1's "~1993 train/~1993 val/~1993 test" and Table 3's in-distribution accuracies are **not reproducible**. They must be regenerated or explicitly labelled as legacy-inflated, never quietly carried forward.

---

## 3. DatasetNinja handling

### 3.1 Parsing

```
ann_dir/<stem>.json   <->   img_dir/<stem>.<ext>
{
  "objects": [
    {"classTitle": "maize", "points": {"exterior": [[x1,y1],[x2,y2], ...]}},
    {"classTitle": "weed",  "points": {"exterior": [[x1,y1],[x2,y2], ...]}}
  ]
}
```

- Map `maize -> 0 ("Corn")`, `weed -> 1 ("Weed")`; ignore other `classTitle` values and log counts.
- Compute the box from **all** exterior points (`min`/`max` over x and y). The legacy code assumes exactly two points (`Benchmark_Clean_Eval.py:87-90`); real polygon annotations with >2 vertices would be silently dropped — fix this.
- Validate `x2 > x1 and y2 > y1` and clamp to image size; reject degenerate boxes.
- Each valid box becomes one `Sample` with `group_id = sha256(stem)`, `bbox`, lazy path, deterministic sorted order.

### 3.2 Whole-image allocation with exact 200+200 quota

The legacy leak (`rng.shuffle(all_samples)` over boxes at `Benchmark_Clean_Eval.py:92-107`) is replaced by:

1. `groups = sorted(unique group_id)`; `order = np.random.default_rng(seed_for_ninja).permutation(len(groups))`, where `seed_for_ninja = np.random.SeedSequence(cfg.seed).spawn(2)[1]` (independent from the Kaggle split stream).
2. `quota = {0: cfg.adaptation_per_class, 1: cfg.adaptation_per_class}` (200/200).
3. For each group in `order`:
   - Partition its boxes by label; sort each label's boxes by `(x1,y1,x2,y2)`.
   - For each label, take `k = min(quota[label], len(boxes[label]))` boxes into `ADAPTATION`; decrement `quota[label]`.
   - **Mark the entire source image as an ADAPTATION group.** Any untaken boxes of that image are *discarded*, never routed to OOD test.
4. Once both quotas reach zero, every remaining group contributes **all** of its boxes to `OOD_TEST`.
5. Assert `quota == {0:0, 1:0}`, else raise `InsufficientAdaptationCrops` with the observed per-class box counts (fail loudly; do not underfill silently).
6. Materialise the 400 adaptation crops as PNGs under `derived/crops/ninja_adaptation/<sample_id>.png`; OOD crops remain lazy `(source_path, bbox)`. Both carry the same `group_id` semantics.

**Invariants enforced:** `groups(ADAPTATION) ∩ groups(OOD_TEST) = ∅`; `sample_id`s unique; `count(ADAPTATION) = 400` (200/200); `count(ADAPTATION) + count(OOD_TEST) = valid_boxes_total` (minus only the deliberately discarded boxes of adaptation images, which are recorded in `invariants.discarded_boxes`).

### 3.3 QSVC ten-run protocol (leak-free, mother ruling 2026-11-22)

**Primary protocol (legacy parity):** `qsvc_source_per_run=50` Kaggle source crops + `qsvc_target_adaptation_per_run=50` labelled Ninja adaptation crops = `qsvc_mixed_train_size=100` mixed training samples; `qsvc_test_per_run=50` held-out Ninja crops as test.

**Mandatory control:** the no-mixing arm uses 100 Kaggle source crops (no Ninja adaptation crops) against the **same** 50 Ninja test crops. This is the paired control for the significance comparison of domain mixing.

Per run `r` in `range(cfg.qsvc_runs)`, spawn the per-run `SeedSequence` itself: `run_ss_r = np.random.SeedSequence([cfg.seed, cfg.seed_offset]).spawn(cfg.qsvc_runs)[r]`; set `run.seed = int(run_ss_r.generate_state(1)[0])` and use `np.random.default_rng(run_ss_r)` as `rng_r` (do **not** build a second SeedSequence from the recorded integer — that would make the recorded seed inconsistent with the sampling stream). This replaces the three ad-hoc formulas `run_idx*42`, `run_idx*123`, `run_idx*456` in `Run_QSVC_Mixed_Variance.py:133-156`.

- `test` (drawn **first**, so both arms at the same `run_idx` see the identical test set): `cfg.qsvc_test_per_run = 50` crops drawn **only** from `OOD_TEST`, stratified 25 maize / 25 weed where group counts permit. These source images are disjoint from every adaptation group, so no QSVC test crop shares a photograph with any QSVC train crop and none was seen during feature-extractor training.
- `MIXED` arm — `target_train`: `cfg.qsvc_target_adaptation_per_run = 50` crops drawn **only** from `ADAPTATION` (stratified 25/25). These are the "50 labelled target-domain crops" of the paper — intentionally seen by the feature extractor; this is the declared supervised domain-mixing treatment, not leakage.
- `MIXED` arm — `source_train`: `cfg.qsvc_source_per_run = 50` crops from Kaggle `TRAIN`, stratified 25/25. `len(source_train) + len(target_train) == cfg.qsvc_mixed_train_size == 100`.
- `NO_MIXING` arm — `source_train`: `cfg.qsvc_mixed_train_size = 100` crops from Kaggle `TRAIN`, stratified 50/50; `target_train = ()`. Same `test` as the MIXED arm at that `run_idx`.
- Sample **groups**, then all crops of a chosen group enter the same side; never split a group across train and test.
- Assert per arm and per run: `len(run.train) == cfg.qsvc_mixed_train_size == 100`; `len(run.test) == cfg.qsvc_test_per_run == 50`; `groups(train) ∩ groups(test) = ∅`, `hashes(train) ∩ hashes(test) = ∅`, `paths(train) ∩ paths(test) = ∅`.
- **Paired statistics:** compute per-run accuracy vectors for both arms on the identical 50 test crops and compare with `experiments.metrics.paired_t_test` (`metrics.py:83-97`, returns mean difference, t, p and Cohen's dz); report `summarize_runs` mean ± std for each arm. Do not use an unpaired test.
- Also report the metric over the *full* `OOD_TEST` set for the final selected model. The 50-crop subsample is for the variance/significance study, not a replacement for full-test evaluation.

**Manifest note:** `ExperimentConfig.as_dict()` uses `dataclasses.asdict`, which omits `@property`. The builder must therefore inject `qsvc_mixed_train_size` (from `cfg.qsvc_mixed_train_size`) into `config_snapshot["config"]` alongside the four `qsvc_*` fields already present there, so the 100/50 protocol is auditable from the manifest alone. `verify()` should assert `config["qsvc_mixed_train_size"] == config["qsvc_source_per_run"] + config["qsvc_target_adaptation_per_run"]`.

---

## 4. Augmentation — exactly as claimed, with train/eval separation

All parameters come from `cfg` (`config.py:26-30`), not literals.

### 4.1 Train transform (PIL → tensor)

Ordered composition returned by `make_transforms(cfg)["train"]`:

| # | Op | Parameters | Paper basis |
|---|---|---|---|
| 1 | `transforms.RandomAffine` | `degrees=(-15, 15)`, `scale=(0.9, 1.1)`, `interpolation=BILINEAR`, `fill=tuple(int(255*m) for m in cfg.norm_mean)` | rotation ±15°, scaling 90–110% (`plant-1.md:94,100`) |
| 2 | `transforms.RandomHorizontalFlip` | `p=0.5` | `plant-1.md:96` |
| 3 | `transforms.RandomVerticalFlip` | `p=0.2` | `plant-1.md:96` |
| 4 | `transforms.ColorJitter` | `brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0` | `plant-1.md:98` (hue not claimed → 0) |
| 5 | `transforms.Resize` | `(128, 128)` | `plant-1.md:74` |
| 6 | `transforms.ToTensor` | — | scale pixels to [0,1] (`plant-1.md:76-78`) |
| 7 | `transforms.Normalize` | `mean=cfg.norm_mean`, `std=cfg.norm_std` | ImageNet stats (`plant-1.md:86-88`) |

Notes:
- Rotation and scale are combined in **one** `RandomAffine` to avoid two interpolation passes. `fill` is the ImageNet mean in 0–255 pixel space so padded borders normalise to ≈0 rather than a dark artefact.
- `RandomAffine(scale=(s_lo,s_hi))` applies zoom factors `1/s` internally; the factor set `{0.9, 1.1}` therefore covers ±10%. See risk Q2 for the alternative "crop area" reading.
- Geometry/photometric transforms run on PIL before `ToTensor`; normalisation is always last.
- Applied **only** to `TRAIN` (Kaggle train + Ninja adaptation crops when used for CNN/QNN training). Never to `ID_VAL`, `ID_TEST` or `OOD_TEST`.

### 4.2 Eval transform

`make_transforms(cfg)["eval"]` = `Resize((128,128)) -> ToTensor -> Normalize(cfg.norm_mean, cfg.norm_std)`. Deterministic: the same image twice yields bit-identical tensors. Used for `ID_VAL`, `ID_TEST`, `OOD_TEST`, and for feature extraction feeding QSVC (`shuffle=False`, as in `deep-learning`: no shuffling at test time).

### 4.3 Why the legacy transform was insufficient

`Benchmark_Clean_Eval.py:267-273` implements only Resize + `RandomHorizontalFlip()` + `RandomRotation(15)`. `vflip=0.2`, `ColorJitter(0.1,0.1,0.1)` and scale 0.9–1.1 exist in `config.py` and in the paper but had **no code path**. The design above closes that gap; the numbers in the paper can no longer be attributed to a transform that was never run.

### 4.4 Reproducibility of stochastic transforms

`torchvision` random ops draw from the global torch RNG. `make_dataloaders` seeds the train loader with `torch.Generator().manual_seed(cfg.seed)`; with `num_workers>0`, PyTorch derives each worker seed from the base seed, so a fixed `cfg.seed` reproduces the augmentation stream given a fixed worker count. Record `num_workers` in the manifest config snapshot.

---

## 5. Machine-readable split manifest schema

File: `derived/manifests/split_manifest_v1.json` (path derived from `cfg.output_dir`). JSON Schema draft 2020-12 (normative):

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "QCSplitManifest",
  "type": "object",
  "required": ["schema_version","generated_by","seed","config_snapshot","sources","splits","provenance","invariants","checksum_sha256"],
  "additionalProperties": false,
  "properties": {
    "schema_version": {"const": "1.0"},
    "generated_by": {"const": "experiments.data.build_splits"},
    "seed": {"type": "integer", "description": "any integer seed; value checked against config_snapshot by verify(), not pinned here"},
    "config_snapshot": {
      "type": "object",
      "required": ["config","library_versions","git_commit","num_workers","generated_at"],
      "properties": {
        "config": {
          "type": "object",
          "description": "full ExperimentConfig.as_dict() plus injected qsvc_mixed_train_size",
          "required": ["seed","image_size","qsvc_source_per_run","qsvc_target_adaptation_per_run","qsvc_test_per_run","qsvc_runs","qsvc_mixed_train_size"],
          "properties": {
            "seed": {"type": "integer"},
            "qsvc_source_per_run": {"type": "integer"},
            "qsvc_target_adaptation_per_run": {"type": "integer"},
            "qsvc_test_per_run": {"type": "integer"},
            "qsvc_runs": {"type": "integer"},
            "qsvc_mixed_train_size": {"type": "integer",
              "description": "relationship to source+target counts enforced in SplitManifest.verify(), not here"}
          }
        },
        "library_versions": {"type": "object", "description": "python, torch, torchvision, numpy, sklearn, pillow, pennylane"},
        "git_commit": {"type": "string"},
        "num_workers": {"type": "integer", "minimum": 0},
        "generated_at": {"type": "string", "format": "date-time"}
      }
    },
    "sources": {
      "type": "object",
      "required": ["kaggle","ninja"],
      "properties": {
        "kaggle": {
          "type": "object",
          "required": ["dataset","files_seen","unique_hashes","duplicates_removed","label_conflicts","hash_algorithm","hash_input"],
          "properties": {
            "dataset": {"const": "aminelaatam/weed-classification"},
            "files_seen": {"type": "integer"},
            "unique_hashes": {"type": "integer"},
            "duplicates_removed": {"type": "integer"},
            "label_conflicts": {"type": "array", "items": {"type": "string"}},
            "hash_algorithm": {"const": "md5"},
            "hash_input": {"const": "raw_file_bytes"}
          }
        },
        "ninja": {
          "type": "object",
          "required": ["ann_dir_env","img_dir_env","ann_files_seen","boxes_valid","boxes_rejected","source_images","discarded_boxes"],
          "properties": {
            "ann_dir_env": {"const": "NINJA_ANN_DIR"},
            "img_dir_env": {"const": "NINJA_IMG_DIR"},
            "ann_files_seen": {"type": "integer"},
            "boxes_valid": {"type": "integer"},
            "boxes_rejected": {"type": "integer"},
            "source_images": {"type": "integer"},
            "discarded_boxes": {"type": "integer"}
          }
        }
      }
    },
    "provenance": {
      "type": "array",
      "description": "one record per Sample, sorted by sample_id; reconstructed into Sample objects by from_json()",
      "items": {"$ref": "#/$defs/sample"}
    },
    "splits": {
      "type": "object",
      "required": ["train","id_val","id_test","adaptation","ood_test"],
      "additionalProperties": false,
      "properties": {
        "train":      {"$ref": "#/$defs/splitRecord"},
        "id_val":     {"$ref": "#/$defs/splitRecord"},
        "id_test":    {"$ref": "#/$defs/splitRecord"},
        "adaptation": {"$ref": "#/$defs/splitRecord"},
        "ood_test":   {"$ref": "#/$defs/splitRecord"}
      }
    },
    "invariants": {
      "type": "object",
      "required": ["verified","group_overlap","hash_overlap","path_overlap","class_counts","counts"],
      "properties": {
        "verified": {"const": true},
        "group_overlap": {"type": "object", "description": "pair -> [] for every split pair"},
        "hash_overlap": {"type": "object"},
        "path_overlap": {"type": "object"},
        "class_counts": {"type": "object"},
        "counts": {"type": "object", "required": ["train","id_val","id_test","adaptation","ood_test"]}
      }
    },
    "checksum_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}
  },
  "$defs": {
    "sample": {
      "type": "object",
      "required": ["sample_id","source","group_id","content_hash","path","bbox","label","label_name"],
      "additionalProperties": false,
      "properties": {
        "sample_id": {"type": "string"},
        "source": {"enum": ["kaggle","ninja"]},
        "group_id": {"type": "string"},
        "content_hash": {"type": "string", "pattern": "^[0-9a-f]{32,64}$"},
        "path": {"type": "string"},
        "bbox": {
          "description": "null for Kaggle; 4 integer pixel coords for Ninja",
          "oneOf": [
            {"type": "null"},
            {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
          ]
        },
        "label": {"type": "integer", "enum": [0, 1]},
        "label_name": {"enum": ["Corn","Weed"]}
      }
    },
    "splitRecord": {
      "type": "object",
      "required": ["count","group_count","groups","class_counts","sample_ids"],
      "properties": {
        "count": {"type": "integer", "minimum": 0},
        "group_count": {"type": "integer", "minimum": 0},
        "groups": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "class_counts": {"type": "object", "required": ["Corn","Weed"]},
        "sample_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": true},
        "hash_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$",
                        "description": "sha256 of sorted hashes; avoids embedding 100k hashes"}
      }
    }
  }
}
```

Design rule: embed **group IDs and a sorted `hash_digest`** in each split record, the source-level counters in the required `sources` object, and the full per-sample provenance (including each `content_hash`) in the required top-level `provenance` array sorted by `sample_id`. `sources` and `provenance` are both authoritative and both round-trip through `to_json()`/`from_json()` (see §1.2). This keeps the manifest machine-checkable without duplicating hashes in five places. `checksum_sha256` is computed over the canonical JSON (sorted keys, compact separators) with `checksum_sha256` removed.

Minimal illustrative instance (values synthetic, only to show shape):

```json
{
  "schema_version": "1.0",
  "generated_by": "experiments.data.build_splits",
  "seed": 42,
  "config_snapshot": {
    "config": {
      "seed": 42, "image_size": 128,
      "qsvc_source_per_run": 50, "qsvc_target_adaptation_per_run": 50,
      "qsvc_test_per_run": 50, "qsvc_runs": 10, "qsvc_mixed_train_size": 100
    },
    "library_versions": {"python": "3.11.9", "torch": "2.4.0"},
    "git_commit": "abc1234",
    "num_workers": 4,
    "generated_at": "2026-11-22T00:00:00Z"
  },
  "sources": {
    "kaggle": {
      "dataset": "aminelaatam/weed-classification",
      "files_seen": 2000, "unique_hashes": 1993, "duplicates_removed": 7,
      "label_conflicts": [], "hash_algorithm": "md5", "hash_input": "raw_file_bytes"
    },
    "ninja": {
      "ann_dir_env": "NINJA_ANN_DIR", "img_dir_env": "NINJA_IMG_DIR",
      "ann_files_seen": 150, "boxes_valid": 6212, "boxes_rejected": 12,
      "source_images": 150, "discarded_boxes": 0
    }
  },
  "provenance": [
    {
      "sample_id": "ninja:9f2a:b1", "source": "ninja", "group_id": "ninja:9f2a",
      "content_hash": "0f6d1d8c2b7a4e5f9a1c3d2e4b5a6978",
      "path": "/data/ninja/img/9f2a.jpg", "bbox": [10, 20, 90, 110],
      "label": 0, "label_name": "Corn"
    }
  ],
  "splits": {
    "train": {"count": 1594, "group_count": 1594, "groups": [], "class_counts": {"Corn": 800, "Weed": 794},
              "sample_ids": [], "hash_digest": "1111111111111111111111111111111111111111111111111111111111111111"},
    "id_val": {"count": 199, "group_count": 199, "groups": [], "class_counts": {"Corn": 100, "Weed": 99},
               "sample_ids": [], "hash_digest": "2222222222222222222222222222222222222222222222222222222222222222"},
    "id_test": {"count": 200, "group_count": 200, "groups": [], "class_counts": {"Corn": 100, "Weed": 100},
                "sample_ids": [], "hash_digest": "3333333333333333333333333333333333333333333333333333333333333333"},
    "adaptation": {
      "count": 400, "group_count": 9,
      "groups": ["ninja:9f2a", "ninja:c031"],
      "class_counts": {"Corn": 200, "Weed": 200},
      "sample_ids": ["ninja:9f2a:b1", "ninja:c031:b0"],
      "hash_digest": "3c1f0d6b8a4e2f719c3d5a7b9e1f4c6d8a2b0e5f7c9d1a3b5e7f9c2d4a6b8e0f"
    },
    "ood_test": {"count": 5812, "group_count": 141, "class_counts": {"Corn": 3102, "Weed": 2710},
                 "groups": [], "sample_ids": [],
                 "hash_digest": "aa90c1d3e5f7082b4a6c8d0e2f4a6b8c0d2e4f60718293a4b5c6d7e8f90a1b2c3"}
  },
  "invariants": {"verified": true, "group_overlap": {"adaptation__ood_test": []},
                 "counts": {"adaptation": 400, "ood_test": 5812}},
  "checksum_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
}
```

---

## 6. Unit-test plan with exact assertions

Location: `experiments/test_data.py` (orchestrator's call). Use `tmp_path`, synthetic data only — **never** the private `.paper/` content or real dataset paths in tests.

| # | Test | Exact assertions |
|---|---|---|
| T1 | `test_content_hash_is_byte_based_and_chunk_stable` | `content_hash(p) == hashlib.md5(p.read_bytes()).hexdigest()`; write the same bytes via two filenames → equal hashes; re-encode the same image to JPEG with different quality → **different** hashes (documents why byte-hash is used). |
| T2 | `test_dedup_counts_and_label_conflict` | Create 10 files: 6 unique + 2 byte-identical pairs. `r.files_seen == 10`; `r.unique_hashes == 8`; `r.duplicates_removed == 2`; `len(dedup) == 8`; `len({s.content_hash for s in dedup}) == 8`. Then place the same bytes under `Corn/` and `Weed/` and assert `pytest.raises(LabelConflictError)`. |
| T3 | `test_kaggle_splits_are_disjoint_and_stratified` | 100 synthetic unique samples (50/class), `cfg.seed=42`. For every pair in `{(TRAIN,ID_VAL),(TRAIN,ID_TEST),(ID_VAL,ID_TEST)}`: `a.groups & b.groups == frozenset()`, `a.hashes & b.hashes == frozenset()`, `a.paths & b.paths == frozenset()`. `sum(len(s.samples) for s in three) == 100`; each `class_counts()["Corn"] > 0 and ["Weed"] > 0`. |
| T4 | `test_kaggle_seed_determinism_and_sensitivity` | `build_splits(cfg)` twice → `m1.splits[TRAIN].sample_ids == m2.splits[TRAIN].sample_ids` and `m1.checksum_sha256` recomputed from config-without-timestamp is equal. With `cfg.seed=43` assert `m3.splits[TRAIN].sample_ids != m1.splits[TRAIN].sample_ids` (guards against seed being ignored). |
| T5 | `test_ninja_parse_polygon_and_filter` | JSON with **5** objects: 2 `maize`, 1 `weed`, 1 `tomato`, and 1 extra `maize` whose box is degenerate (`x2<=x1`); one of the valid maize boxes has 4 exterior points. `len(samples) == 3` (the unknown class and the degenerate box are both filtered); the 4-point box equals `(min xs, min ys, max xs, max ys)`; sorted order equals `sorted(samples, key=lambda s:(s.group_id,s.bbox,s.label))`. |
| T6 | `test_ninja_group_quota_no_cross_contamination` | 10 source images × 30 maize + 30 weed = 600 boxes; `cfg.adaptation_per_class=200`. `adapt.count == 400`; `adapt.class_counts() == {"Corn":200,"Weed":200}`; `adapt.groups & ood.groups == frozenset()`; for every `g in adapt.groups`, no OOD sample has `group_id == g`; `adapt.count + ood.count <= 600` and any shortfall equals `manifest.invariants["discarded_boxes"]`. Assert `pytest.raises(InsufficientAdaptationCrops)` when only 150 weed boxes exist. |
| T7 | `test_manifest_roundtrip_and_verify` | `m.to_json(p); loaded = SplitManifest.from_json(p)`; assert `loaded == m`, `loaded.sources == m.sources` (both `"kaggle"` and `"ninja"` counter dicts present), `loaded.provenance == m.provenance` and `isinstance(loaded.provenance[0], Sample)` — the schema-required `sources` AND `provenance` keys must both survive the roundtrip. Validate the emitted JSON against the §5 draft-2020-12 schema (`jsonschema.Draft202012Validator(...).validate(payload)`) → no exception. `m.verify() == []`; `m.config_snapshot["config"]["seed"] == 42`, `["image_size"] == 128`, `["scale_range"] == [0.9,1.1]`; `config_snapshot["config"]["qsvc_source_per_run"] == 50`, `["qsvc_target_adaptation_per_run"] == 50`, `["qsvc_test_per_run"] == 50`, `["qsvc_runs"] == 10`, and `["qsvc_mixed_train_size"] == 100` (injected into the same `config` object because `asdict` omits properties). **Seed flexibility:** build with `dataclasses.replace(cfg, seed=43)` → `m43.verify() == []` and `m43.seed == 43` (schema has no `const` on seed or qsvc fields). **Negative checks:** tamper `config["qsvc_mixed_train_size"]` to 99 → `verify()` contains `"qsvc_mixed_train_size"`; tamper top-level `seed` to 7 while `config.seed==42` → `verify()` contains `"seed_config_mismatch"`; tamper one OOD sample's `group_id` to an adaptation group → `verify()` contains `"group_overlap"`; tamper `checksum_sha256` → `verify()` reports checksum mismatch. |
| T8 | `test_transforms_exact_parameters` | Inspects `make_transforms(cfg)["train"].transforms`: one `RandomAffine` with `.degrees == (-15,15)` and `.scale == (0.9,1.1)`; `RandomHorizontalFlip.p == 0.5`; `RandomVerticalFlip.p == 0.2`; `ColorJitter.brightness == 0.1 and .contrast == 0.1 and .saturation == 0.1 and .hue == 0.0`; `Normalize.mean == cfg.norm_mean`. Assert `eval` transform contains **no** `RandomAffine/Flip/ColorJitter`. |
| T9 | `test_transform_shapes_and_eval_determinism` | For `Image.new("RGB",(64,96))`: `train(img).shape == (3,128,128)`, `eval(img).shape == (3,128,128)`, both `.dtype == torch.float32` and `torch.isfinite(t).all()`. `torch.equal(eval(img), eval(img)) is True`. With `torch.manual_seed(1)` vs `torch.manual_seed(2)`, `not torch.equal(train(img), train(img))`. |
| T10 | `test_qsvc_runs_leak_free_reproducible_exhaustive` | For `cfg.qsvc_runs == 10`, `r in range(10)` and `protocol in (MIXED, NO_MIXING)`: `len(run.train) == cfg.qsvc_mixed_train_size == 100`; `len(run.test) == cfg.qsvc_test_per_run == 50`; `len(run.source_train) == (50 if MIXED else 100)`; `len(run.target_train) == (50 if MIXED else 0)`; `groups(train)&groups(test) == ∅`, `hashes(train)&hashes(test) == ∅`, `paths(train)&paths(test) == ∅`; `run.seed == int(SeedSequence([42,42]).spawn(10)[r].generate_state(1)[0])`; `sample_qsvc_pairs(m,cfg,3,MIXED) == sample_qsvc_pairs(m,cfg,3,MIXED)` and `!= sample_qsvc_pairs(m,cfg,4,MIXED)`; `{s.sample_id for s in run.test} ⊆ m.splits[Split.OOD_TEST].sample_ids`; for MIXED, `{s.sample_id for s in run.target_train} ⊆ m.splits[Split.ADAPTATION].sample_ids`; for all arms, `{s.sample_id for s in run.source_train} ⊆ m.splits[Split.TRAIN].sample_ids`. |
| T11 | `test_no_physical_train_test_aliasing` | After `build_splits` + adaptation materialisation: `len(list(derived/"crops"/"ninja_adaptation".glob("*.png"))) == 400`; no file under `derived/train` is byte-identical to any file under `derived/ood_test` (OOD has no materialised files at all); `m.verify() == []`. |
| T12 | `test_qsvc_no_mixing_control_is_paired_and_leak_free` | For every `r in range(cfg.qsvc_runs)`: `mixed.test == control.test` (identical sample_ids in identical order — required for the paired t-test); `len(mixed.train) == len(control.train) == cfg.qsvc_mixed_train_size == 100`; `control.target_train == ()`; `{s.sample_id for s in mixed.target_train} ⊆ m.splits[Split.ADAPTATION].sample_ids`; `{s.sample_id for s in control.source_train} ⊆ m.splits[Split.TRAIN].sample_ids`; `{s.sample_id for s in control.test} ⊆ m.splits[Split.OOD_TEST].sample_ids`; `groups(control.train)&groups(control.test) == ∅` and same for hashes/paths. With two distinct 10-element accuracy vectors, `paired_t_test(a,b)["n"] == 10` and `0.0 <= result["p_value"] <= 1.0`. |

Acceptance: all of T1–T12 pass; the whole suite runs on synthetic data in <10 s without dataset env vars being set (use a `cfg` fixture with temp dirs).

---

## 7. Risks and unresolved questions

**Q1 — MD5 for dedup (paper parity) is not collision-safe.** Keep `md5` as the paper-faithful dedup key but also store a `sha256` per sample in provenance for integrity. If two *different* images ever collide under MD5, the label-conflict check catches the common case; a deliberate collision attack is out of scope but should be documented.

**Q2 — Ambiguity in "scaling 90–110%".** Implemented as affine zoom `scale=(0.9,1.1)` because `RandomResizedCrop.scale` is restricted to ≤1.0 (crop area) and cannot express >100%. If the authors meant crop-area 0.9–1.1, that is geometrically impossible for the >1 case without padding; confirm the intended semantics before the manuscript claims exact fidelity. This affects reproducibility of the augmentation claim, not leakage.

**Q3 — Reported metrics will change.** Replacing leaked train==test with a real ID holdout and group-disjoint OOD test will lower some numbers (especially Table 1/3 in-distribution accuracies). This is a **correctness fix**, not a regression; the manuscript must state it explicitly rather than reuse legacy figures.

**Q4 — Annotation schema not yet verified on real data.** The legacy parser assumes exactly two exterior points (`Benchmark_Clean_Eval.py:87-90`) and class titles exactly `maize`/`weed`. Need to inspect a sample of real `ann/*.json` for >2-point polygons, alternate class names, missing images, and image extensions. The new parser handles >2 points but must be validated against the actual download.

**Q5 — Adaptation quota feasibility unknown until data is mounted.** If fewer than 200 maize or 200 weed boxes exist across sufficiently many source images, `split_ninja` fails fast (by design). Need actual box-per-image statistics; the group-level allocation may need a fallback that documents a reduced quota rather than silently leaking. Confirm OOD test still has enough samples after reserving whole source images.

**Q6 — Near-duplicate photographs across different source images.** Content hashing and source-image grouping cannot catch re-photographs of the same plant/field. The paper's geo-referenced annotations and "same location" variation make this plausible. Optional mitigation: perceptual-hash clustering (pHash) as an *additional* group key. Not implemented in this design; ask whether the review standard requires it.

**Q7 — QSVC train-size interpretation — RESOLVED (mother ruling, 2026-11-22).** Ruling: primary protocol is **legacy parity** — 50 Kaggle source crops + 50 labelled Ninja adaptation crops = 100 mixed training samples (`cfg.qsvc_mixed_train_size`), with 50 held-out Ninja crops as test; a **mandatory no-mixing control** (100 Kaggle source crops, same 50 Ninja test crops) is required for the paired significance comparison. Mother updated `experiments/config.py:40-70`: `qsvc_source_per_run=50`, `qsvc_target_adaptation_per_run=50`, `qsvc_test_per_run=50`, `qsvc_runs=10`, derived property `qsvc_mixed_train_size=100`; the ambiguous `qsvc_train_per_run` field is removed. The API in §1 and tests T10/T12 now use these explicit fields. The `n_source_anchor` override is removed. All leak-free group guarantees are unchanged; the control adds a second arm, not a new sampling boundary. No action remains.

**Q8 — "400 random images" vs "400 crops".** `plant-1.md:356` says "400 of the images at random were reserved for training" while §3.1 says 400 *crops* (200/class). This design follows the crop reading (consistent with legacy code and config). Confirm terminology in the manuscript.

**Q9 — Full OOD test vs 50-crop QSVC subsample.** QSVC per-run accuracy is measured on 50 OOD crops; that is a small, high-variance estimate. The design mandates reporting mean ± std over 10 runs *and* a full-OOD-test number for the final model. Do not tune on the QSVC test pool.

**Q10 — Environment paths.** `cfg.validate()` correctly fails when `NINJA_ANN_DIR`/`NINJA_IMG_DIR` are unset (`config.py:66-70`). Never reintroduce hard-coded `C:\Users\inaug\Downloads\...` paths from the legacy scripts (`Benchmark_Clean_Eval.py:259-260`). Manifest records only the env **variable names**, not machine-specific absolute paths, to keep it portable.

**Q11 — Augmentation determinism with workers.** Fixed seed reproduces augmentation only for a fixed `num_workers` and torchvision version; both are recorded in `config_snapshot`. Cross-version bit-exactness of torchvision random ops is not guaranteed.

---

## Deliverable summary (master-scientist format)

- **(a) Skills loaded:** `master-scientist`, `ml-for-science`, `reproducible-environments`, `deep-learning`.
- **(b) Claim-to-evidence table:** see "Claim-to-evidence table" above (3 partially supported, 1 verified for this data-layer scope).
- **(c) Exact files changed by this worker:** `docs/research/QC-D02_data_pipeline.md` only. No code touched, no builds/tests run, no commits made, private paper read locally and not transmitted.
- **(d) Next bottom-up step & acceptance criterion:** orchestrator implements `experiments/data.py` to the API in §1, then adds `experiments/test_data.py` from §6. **Acceptance:** T1–T12 pass on synthetic data AND `SplitManifest.verify()` returns `[]` on the real mounted dataset before any model training begins.

**Ruling applied:** Q7 resolved by mother — primary QSVC = 50 Kaggle + 50 Ninja adaptation = 100 mixed train vs. 50 Ninja OOD test; mandatory paired no-mixing control (100 Kaggle, same 50 test). §1, §3.3, §5, T7, T10, T12 and §7/Q7 updated accordingly. **Blocker status: none.**
