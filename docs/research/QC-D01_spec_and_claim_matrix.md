# QC-D01 — Rebuild Specification and Claim-to-Evidence Matrix

**D-ID:** QC-D01 (research worker, Quantum_Classification rebuild swarm)
**Scope:** Read-only audit; this file is the only artifact written. No code was modified, no build or test was run, no network access was used, and the private manuscript was not transmitted.
**Evidence base:** `.paper/plant-1.md` (private, gitignored); legacy code under `archives/legacy-2026-11-22/` and `archives/Old Runs/`; new modules `experiments/config.py`, `experiments/metrics.py`, `requirements.txt`, `environment.yml`; archived logs (`outputs.md`) and CSVs.
**Skills loaded:** `master-scientist` → `scientific-writing`, `ml-for-science`, `quantum-computing`, `quantum-skills`, `reproducible-environments` (roster reviewed). Decisions on leakage/CV follow `ml-for-science`; on entanglement/kernel equivalence follow `quantum-computing` + `quantum-skills`; on pinning follow `reproducible-environments`; on claim discipline follow `scientific-writing`.

**Evidence labels.** `VERIFIED` = traceable to a named file/line or reproducible arithmetic. `PARTIALLY SUPPORTED` = directionally right but missing provenance, variance, or protocol detail, or internally inconsistent. `UNSUPPORTED` = no implementation/log, contradicted by code, or mislabeled. `UNVERIFIED` = cannot be established locally. "Legacy absence" below means no match was found by keyword search over all `.py` files (`gaussian|noise|corrupt`, `ttest|p_value|significance`, `iou|fidelity|log_loss|KFold|cross_val`); the only legacy `iou` is bounding-box NMS in `archives/Old Runs/CL_weed_maize_trail1.py:380-398`, which is not classification IoU.

---

## 1. Paper aims restated as testable objectives

State each objective as a falsifiable statement. Emit a result artifact for every objective; an objective with no artifact remains unmet.

- **O1 — Honest OOD benchmark.** Train and evaluate the eight logit-emitting neural models named in paper Table 2, plus the QSVC kernel variants, on one declared, leakage-free train/validation/test split. QSVC variants are precomputed-kernel models: they emit decision scores/probabilities, not logits, and are assessed separately. The released benchmark instead has eight neural models (including a from-scratch hybrid not listed in Table 2) plus two QSVC variants = ten models, which is the count used in the abstract. *Verify by:* a split manifest with disjoint image hashes and disjoint source-photo groups across folds, plus one metrics row per model. *Falsify if:* any hash or group appears in two folds.
- **O2 — Classical dominance.** Establish whether classical SOTA (EfficientNet-B0, YOLOv8n-cls) exceeds hybrid quantum models in OOD accuracy. *Verify by:* mean ± std and 95% CI over ≥5 seeds per model. *Falsify if:* CIs overlap or the ranked order reverses under a different seed.
- **O3 — Transfer-learning hybrid.** Test whether the EfficientNet-B0 → AmplitudeEmbedding → StronglyEntanglingLayers hybrid reaches the reported 92.39% OOD accuracy. *Verify by:* regenerated metric with n, seed, and split ID. *Falsify if:* the value cannot be reproduced on the declared split.
- **O4 — Supervised domain mixing.** Test whether adding 50 labelled target-domain crops to a 100-sample QSVC training set improves OOD accuracy. *Verify by:* ten-run mean ± std and a paired comparison against a no-mixing control on the same target test set. *Falsify if:* no control exists or target-source photographs span train and test.
- **O5 — Architectural comparison.** Compare CNN, QNN, and hybrid QC-CNN on a single declared split with the declared metric suite. *Verify by:* one confusion matrix and one metric vector per model from stored predictions. *Falsify if:* numbers are imported from a different split or different epoch count.
- **O6 — Noise robustness.** Quantify accuracy degradation under Gaussian noise of declared variance. *Verify by:* paired clean/noisy scores per seed and a deterministic noise transform keyed by seed. *Falsify if:* no noise transform is implemented.
- **O7 — Statistical significance.** Test whether model differences are significant. *Verify by:* paired t-test on run-level score vectors with effect size and n. *Falsify if:* only a single run exists per model.
- **O8 — Full metric suite.** Report accuracy, precision, recall, F1, AUC-ROC, log loss, IoU, FPS, and parameter counts from stored predictions/probabilities; report quantum fidelity only if quantum states are actually emitted. *Verify by:* one metrics JSON with all fields and provenance. *Falsify if:* a field is copied from prose.
- **O9 — Quantum-kernel negative result.** State whether the QSVC kernel supplies entanglement or quantum advantage. *Verify by:* circuit introspection (0 entangling gates, 0 trainable parameters) and a numerical equivalence test against the classical squared dot product. *Falsify if:* an entangled or parameterized feature map is introduced.

---

## 2. Claim-to-evidence matrix

| ID | Claim (paper location) | Classification | Evidence in legacy code / absence | Required correction |
| --- | --- | --- | --- | --- |
| C1 | OOD benchmark with distinct train/test sets (Abstract; §4.1–4.2; Table 1) | **UNSUPPORTED — correctness bug** | `Benchmark_Clean_Eval.py:241-256` copies every deduplicated Kaggle image into **both** `train/` and `test/`; `outputs.md:167-169` shows 2,393 train / 1,993 test with identical sources. `DatasetNinjaClassification` (`:92-116`) splits at bounding-box level, so crops from one photograph can land in adaptation and test sets. Paper already concedes this at line 369. | Rebuild splits before regenerating any result; group by MD5 hash and source photograph. |
| C2 | EfficientNet-B0 98.96%, YOLOv8n-cls 98.58% OOD accuracy (Abstract; §4.4; Conclusion) | **PARTIALLY SUPPORTED** | Values recorded once in `outputs.md:70-71, 289-290`; hardcoded into `Generate_Final_Graphs.py:15-16`; paper note at line 515 says plots reconstruct terminal output. Single run on leaked split. | Regenerate with honest split; publish n, mean, std, CI. |
| C3 | Hybrid CNN-QNN (transfer learning) 92.39% OOD accuracy (Abstract; §4.4; Conclusion) | **PARTIALLY SUPPORTED** | `outputs.md:158-159` (Acc 0.9239); hardcoded in `Generate_Final_Graphs.py:20`; model defined at `Benchmark_Clean_Eval.py:199-217`. One run, no spread. | Regenerate across seeds; report mean ± std. |
| C4 | QSVC mixing: 93.20% ± 3.01% over ten runs with 50 labelled target crops (Abstract; Conclusion) | **VERIFIED (arithmetic) / PARTIALLY SUPPORTED (protocol)** | `Run_QSVC_Mixed_Variance.py:125-210` implements 10 runs of 50 CornWeed + 50 Ninja train / 50 Ninja test on `lightning.qubit`. `variance_mixed_benchmark_metrics.csv:2-11`: mean = 0.9320, sample SD (ddof=1) = 0.0301 — exactly the stated figure. 50-sample test set; crop-level leakage persists (C1); the 100-sample QSVC train set differs from the 2,393-sample train set claimed in §3.1 (line 67). No 94.00% figure appears in the current manuscript. | Keep the number but group test crops by source photograph, add a no-mixing control, and report a 95% CI. |
| C5 | Quantum kernel feature map implies quantum advantage (§2.4; §3.2 Fig. 6; Table 2 #9; archived `README.md:19-23`) | **UNSUPPORTED** | Paper's own Fig. 6 note (line 224) states no entanglement and no trainable parameters. `Run_QSVC_Only.py:105-109` computes `(A @ B.T)**2` directly. `Run_QSVC_Mixed_Variance.py:86-90` runs the equivalent AmplitudeEmbedding + adjoint circuit on `lightning.qubit`; for L2-normalized inputs it equals `\|xᵢ·xⱼ\|²`. | Describe as a classical polynomial kernel baseline; delete "quantum advantage" and "global maximum margin guarantees" language. |
| C6 | QNN / hybrid QNN circuits use entanglement (§2.2; Table 2 #6–8) | **VERIFIED (10-qubit) / PARTIALLY SUPPORTED (4-qubit)** | `Benchmark_Clean_Eval.py:152-162` — AmplitudeEmbedding + `StronglyEntanglingLayers(2, 10 wires)`; `hybrid_QC_CNN_v2.py:78-93` — 4-qubit, 16-feature, 2-layer entangling variant. However `hybrid_QC_CNN_v2.py:74,135` uses batch 16 and **5** epochs ("Quick train loop"), contradicting Table 2 #8's 15-epoch/32-batch claim. | Keep entanglement claim for the 10-qubit models; label the 4-qubit variant as a 5-epoch prototype. |
| C7 | Table 3: CNN 95.18%, QNN 75.65%, hybrid 98.89% with training times (Table 3, §4.3) | **UNSUPPORTED as benchmark result; PARTIALLY SUPPORTED as archived artifact** | Provenance note at line 401. Traced to `archives/Old Runs/hybrid_QC_CNN__version__2_0.ipynb:443` (98.89%), `hybrid_version_2.ipynb:231` (1,993 train / 809 test), and `:11405,11507` (test loss 0.1457 / 0.0787), QNN archive `QNN_pennylane1.ipynb`; 10-epoch in-distribution split. Training times 859.44 / 153.48 / 750.24 s appear in no code or log. | Regenerate on the OOD split or delete Table 3; never present archived 809-sample numbers as OOD results. |
| C8 | 15 epochs, batch 32, Adam lr=0.001, cross-entropy for all released models (§3.2 lines 151/190; §4.2 line 407) | **VERIFIED for the released benchmark** | `Benchmark_Clean_Eval.py:291-296, 389-390` — `epochs=15`, Adam(lr=0.001), `CrossEntropyLoss`; `outputs.md` confirms per-model 15-epoch logs. | State the scope: released `Benchmark_Clean_Eval.py` only. |
| C9 | Learning-rate scheduling, early stopping, 0.03 LR, 10-epoch schedule (§3.2 line 119; earlier drafts) | **CONTRADICTED / already corrected in paper** | No scheduler or early stopping in `Benchmark_Clean_Eval.py`; `:297-324` only checkpoints best "validation" accuracy, where validation is the copied training set (`test_loader`). Paper line 407 explicitly says these are not implemented. | Remove future-tense "will be used"; describe best-checkpoint selection on a genuinely held-out split. |
| C10 | PyTorch + PennyLane only; no TensorFlow (§4.2 line 407) | **VERIFIED (released benchmark only)** | The released benchmark `Benchmark_Clean_Eval.py:12-25` imports PyTorch/PennyLane and no TensorFlow, so the paper's framework claim holds for the released benchmark. TensorFlow appears only in archived prototype scripts (`archives/Old Runs/CL_weed_maize_trail1.py:5-6`, `archives/Old Runs/GT_weed_maize.py:6-7`). | State the scope explicitly — released benchmark only, not all archived scripts; record exact versions in the run manifest. |
| C11 | Gaussian-noise robustness results (§4.2 line 417, §4.3 line 447, Table 6) | **UNSUPPORTED** | Legacy absence: no noise/corruption transform in any archived `.py` or notebook log; paper note at line 479. | Implement a seeded noise transform and rerun, or delete Table 6. |
| C12 | p-values 0.0013 / 0.0009 (§3.3 lines 282-290; §4.3 line 443; Table 4) | **UNSUPPORTED** | Legacy absence: no statistical test in archived `.py`; paper note at line 456. `experiments/metrics.py:83-97` now provides `paired_t_test`, but no run vectors exist for CNN/QNN/hybrid (single run each: §4.3 line 413). | Record ≥5 run vectors, then compute paired test + effect size; otherwise delete Table 4. |
| C13 | IoU comparison (§3.3 line 300-302; Table 5) | **UNSUPPORTED (mislabeled)** | Paper's own metric correction at line 469: listed values are precision for class 0 and recall for class 1, e.g. 594.15/(594.15+31.85+18.85)=0.921, not 0.949. Legacy `iou` exists only as bounding-box NMS (`archives/Old Runs/CL_weed_maize_trail1.py:380-398`). `experiments/metrics.py:35-38` now computes TP/(TP+FP+FN) correctly. | Recompute Table 5 from stored confusion matrices with `binary_metrics()`. |
| C14 | Log loss, k-fold CV, quantum fidelity (§3.3 lines 256-262, 294-298) | **UNSUPPORTED** | Legacy absence for `log_loss`, `KFold`, `fidelity`; no implementation in the new modules either. Related MSE/RMSE/R² exist only in `hybrid_QC_CNN_v2.py:201-203`. | Implement log loss and grouped k-fold in the metrics layer; report fidelity only if a state-producing circuit is added, else label "not measured". |
| C15 | Figures generated directly from experiment outputs (Fig. 8, 14, 15; §4.4 line 515) | **UNSUPPORTED — defect verified** | `Advanced_Evaluation_Plots.py:23-32` hardcodes 7 per-epoch accuracy arrays; `Generate_Final_Graphs.py:11-21` hardcodes all SOTA metrics feeding Figs. 14–15 and `benchmark_sota/MASTER_BENCHMARK_RESULTS.csv`; `Run_QSVC_Only.py:171-179` redefines the circuit solely for a plot. Paper note at line 515 acknowledges this. | Regenerate every figure and table from `results/metrics.json` + stored raw predictions; add a no-literal-arrays check. |
| C16 | Hardware: Intel 14th-gen i9, RTX 4090 8 GB, 32 GB RAM (§4 line 352); H100 (line 537) | **PARTIALLY SUPPORTED / CONTRADICTED** | `outputs.md:165` records RTX **4070 Laptop GPU, 8188 MiB**, Torch 2.6.0+cu124, Ultralytics 8.4.64, Python 3.13. No i9 or 4090 evidence in logs; H100 mention conflicts. | Replace with an auto-generated run manifest (device, versions, seed, git SHA); delete conflicting hardware claims. |
| C17 | Confusion-matrix and AUC narrative (§4.3 lines 423-433, 443, 447; §4.4 line 507) | **PARTIALLY SUPPORTED / internally inconsistent** | 615/195 counts and 809 support come from the archived 809-sample notebooks, while Table 3 uses different accuracies (95.18% vs 94.31% at line 443) and AUC values 0.96 / 0.8291 / 1.00 are mutually exclusive; §4.4 states `Custom_Legacy_CNN` AUC 0.500 whereas `outputs.md:93` and `Generate_Final_Graphs.py:17` record 0.9459. | Choose one split; regenerate one confusion matrix and one AUC per model from stored probabilities. |
| C18 | Augmentation: rotation ±15°, hflip 0.5, vflip 0.2, jitter 0.1, scale 0.9–1.1 (§3.1 lines 92-100) | **PARTIALLY SUPPORTED** | `Benchmark_Clean_Eval.py:267-273` implements only `RandomHorizontalFlip()` and `RandomRotation(15)`; no vertical flip, jitter, or scaling. `experiments/config.py:26-30` declares the full protocol but no training code consumes it yet. | Implement declared transforms in the data layer or narrow the paper claim to what ran. |
| C19 | 128×128 resize, ImageNet normalization, 49,152→1,280→1,024 reduction (§3.1 lines 72-90; Table 2 #7) | **VERIFIED** | `to_wsl_path` at `Benchmark_Clean_Eval.py:37-48`; resize + ImageNet normalization at `:262-273`; 1,280→1,024 projection at `:199-217`; 49,152-d flatten at `:188`; matches `experiments/config.py:15,22-23,33-34`. | Retain; assert in a data-layer unit test. |
| C20 | Code publicly available (Data Availability, line 545) | **UNVERIFIED locally** | Archived code and new modules are present; `.gitignore:6` excludes `.paper/`, so the manuscript is private as required. Remote reachability was not checked (no network use). | Commit code plus lockfile, then verify the remote URL in a separate authorized step; keep `.paper/` ignored. |

---

## 3. Target architecture — layered modules

Build strictly bottom-up. Each layer exposes a narrow interface to the layer above and never reaches around it. Use repo-relative paths.

### Layer 0 — Environment
- `experiments/config.py` (exists): single source of truth for seed, paths, hyperparameters; keep `from_env()` and `validate()`.
- `experiments/env.py` (new): collect Python and package versions, CUDA device, git SHA, and resolved config into `results/env_manifest.json`.
- `requirements.txt`, `environment.yml` (exist): tighten to exact pins and add a hash-pinned lockfile; do not let `requirements.txt` and `environment.yml` drift.

### Layer 1 — Data
- `data/splits.py`: group-aware split by MD5 of file bytes and by DatasetNinja source-photo ID; declare adaptation vs OOD test at photo level, not crop level.
- `data/transforms.py`: one train transform (full declared augmentation) and one eval transform (resize + ToTensor + ImageNet norm).
- `data/manifest.py`: emit `results/data/split_manifest.json` with per-image `{sha256, group_id, split, label}` and fold counts.

### Layer 2 — Models
- `models/classical.py`: CNN and torchvision backbones behind `build_model(name, config)`.
- `models/quantum.py`: QNN and hybrid QNN circuit builders; expose `entangling_gate_count` and `trainable_param_count` for introspection.
- `models/qsvc.py`: one kernel function backed by the PennyLane circuit, plus the algebraic reference `(A @ B.T)**2`; keep the classical path as the tested default.

### Layer 3 — Experiments
- `experiments/train.py`: seeded training with a genuinely held-out validation split; persist best-checkpoint metadata.
- `experiments/evaluate.py`: call `experiments/metrics.py`; persist per-run `y_true/y_pred/y_prob` to `results/raw/<model>/run_XX.npz`.
- `experiments/qsvc_run.py`: ten-run mixing protocol with a no-mixing control, photo-grouped test set, and seeded subsets.
- `experiments/noise.py`: seeded Gaussian corruption producing paired clean/noisy predictions.
- `experiments/stats.py`: run-vector summarization (n, mean, SD, bootstrap CI, paired test, effect size).

### Layer 4 — Reporting / figures
- `reporting/tables.py`: render Tables 1–6 from `results/metrics.json` only.
- `reporting/figures.py`: render every figure from stored raw predictions and metrics; forbid hardcoded numeric arrays.
- `reporting/manuscript_sync.py`: replace numeric literals in `.paper/plant-1.md` with values read from the same JSON; never retype a number.

---

## 4. Acceptance criteria per layer

| Layer | Criterion | Verification kind |
| --- | --- | --- |
| 0 Environment | `ExperimentConfig.from_env()` with `NINJA_ANN_DIR`/`NINJA_IMG_DIR` set returns `validate() == []`; without them it returns one combined path error naming both variables (`experiments/config.py:75-79`). `results/env_manifest.json` contains nonzero `seed`, pinned versions, and `feature_dim == 2**n_qubits`. | Unit test + JSON artifact |
| 0 Environment | `requirements.txt` and `environment.yml` declare identical version bounds for every shared package; lockfile pins hashes. | Diff/text test + lockfile artifact |
| 1 Data | For every image hash **and** every source-photo group, sets are pairwise disjoint across train/val/test; `n_train + n_val + n_test == n_manifest`; adaptation crops and OOD test crops share no `group_id`. | Unit test + `split_manifest.json` |
| 1 Data | Train transform emits all declared augmentations (rotation ±15°, hflip 0.5, vflip 0.2, jitter 0.1, scale 0.9–1.1); eval transform is deterministic across two calls on the same image. | Transform unit test |
| 2 Models | `build_model(name, config)` returns logits of shape `(batch, 2)` for each of the eight logit-emitting neural models named in paper Table 2; QSVC is excluded because precomputed-kernel variants emit decision scores/probabilities, not logits. Frozen EfficientNet parameters have `requires_grad is False`; `feature_dim` equals `2**n_qubits`. | Unit test |
| 2 Models | 10-qubit QNN reports 2 entangling layers and 10 Pauli-Z outputs; 4-qubit prototype reports 4 wires and 16-d projection. | Introspection unit test |
| 2 Models | QSVC kernel shape: `kernel(A, B)` returns shape `(len(A), len(B))`, and is symmetric when `A is B`. `max|K_quantum - (A@B.T)**2| < 1e-6` on ≥16 random L2-normalized vectors; `entangling_gate_count(K_circuit) == 0` and `trainable_param_count(K_circuit) == 0`. | Numerical equivalence test |
| 3 Experiments | Each model emits ≥5 run artifacts with distinct seeds; `results/metrics.json` rows carry `n`, `mean`, `std`, `ci95_low`, `ci95_high` — no bare accuracy. | Artifact schema test |
| 3 Experiments | `iou_from_counts(6,1,2) == 6/9`; `binary_metrics` confusion matrix on the synthetic 10-sample fixture equals `[[1,1],[2,6]]`; `paired_t_test` raises `ValueError` for `n < 2` and returns `p_value` and `cohens_dz` for `n >= 2`. | Unit tests (fixtures exist in `experiments/metrics.py:107-116`) |
| 3 Experiments | Noise experiment emits paired clean/noisy accuracy per seed; re-running with the same seed reproduces the identical corrupted tensor. | Unit test + paired artifact |
| 4 Reporting | No numeric literal arrays appear in `reporting/`; each table row equals the corresponding `results/metrics.json` field to 6 decimals. | Grep-based test + CSV artifact |
| 4 Reporting | Each generated PNG has a sidecar `.json` naming its source metric keys and raw-artifact paths; regenerating twice from the same JSON yields byte-identical CSV/PNG checksums. | Artifact + checksum test |

---

## 5. Prioritized implementation order (bottom-up)

Do not edit the manuscript numeric claims until P1–P7 artifacts exist. Treat P0 and P1 as hard blockers for every downstream result.

1. **P0 — Environment.** Tighten pins, add the lockfile, finish `env_manifest` emission. *Exit:* Layer 0 acceptance tests pass.
2. **P1 — Data.** Implement group-aware splitting, transforms, and the split manifest; eliminate train/test identity and crop leakage. *Exit:* disjoint-hash/group test passes. Nothing else may run before this.
3. **P2 — Metrics.** Extend `experiments/metrics.py` with log loss and grouped k-fold; retain IoU and paired statistics. *Exit:* all Layer 3 metric unit tests pass on synthetic fixtures.
4. **P3 — Models.** Wrap classical and quantum models behind `build_model`; add entangling-gate counting and the kernel-equivalence test. *Exit:* C5 is mechanically enforced; C6 introspection passes.
5. **P4 — Training and evaluation.** Run seeded, repeated train/evaluate with raw prediction persistence. *Exit:* ≥5 run artifacts per model; C2/C3/C7 regenerated or removed.
6. **P5 — QSVC protocol.** Ten-run mixing with photo-grouped test set and no-mixing control; recompute C4. *Exit:* mean/SD reproduced and control comparison emitted.
7. **P6 — Noise and statistics.** Implement seeded noise (C11) and compute paired tests/effect sizes (C12) from run vectors. *Exit:* paired artifacts for both experiments.
8. **P7 — Reporting.** Regenerate all tables and figures from JSON (C13, C15, C17); add the no-literal-array check. *Exit:* reporting acceptance tests pass.
9. **P8 — Manuscript sync.** Replace remaining numeric claims in `.paper/plant-1.md` from `results/metrics.json`; delete or explicitly label unsupported Tables 3, 4, 6; remove quantum-advantage and conflicting-hardware language (C5, C16). *Exit:* every numerical sentence resolves to a result artifact.
10. **P9 — Availability.** In a separately authorized step, commit code and lockfile, verify the public remote, and confirm `.paper/` remains ignored (C20).

---

## Handoff

**Files changed:** exactly one — `docs/research/QC-D01_spec_and_claim_matrix.md` (this file).
**Next bottom-up step:** P0 — finalize environment pins/lockfile and emit `results/env_manifest.json`.
**Next-step acceptance criterion:** `ExperimentConfig.from_env().validate() == []` when both `NINJA_*` paths are set, and an `env_manifest.json` containing seed, exact package versions, and `feature_dim == 2**n_qubits`.
**Blocker:** none for this document. Hard blocker for all later work: the archived pipeline copies identical images into train and test (`Benchmark_Clean_Eval.py:241-256`) and splits DatasetNinja crops at bounding-box level, so every legacy accuracy figure must be regenerated after P1 or deleted.
