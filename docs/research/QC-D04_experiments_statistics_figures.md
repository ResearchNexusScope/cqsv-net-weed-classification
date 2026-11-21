# QC-D04 — Experiments, Statistics, and Figures (design specification)

D-ID: QC-D04 | Owner: research worker (read-only except this file) | Date: 2026-11-22
Status: specification only. No code, builds, tests, or commits were run by this worker.

## 0. Scope, skills, and evidence base

**Skills loaded (master-scientist workflow):** `master-scientist` (routing + claim/evidence
discipline), core quartet `scientific-writing`, `ml-for-science`, `quantum-computing`,
`quantum-skills`; roster matches `biostatistics` (significance, effect sizes, bootstrap,
multiplicity), `reproducible-environments` (provenance/pinning). No installed statistics
skill beyond `biostatistics` was needed.

**Files inspected (local, read-only):** `experiments/config.py`, `experiments/metrics.py`,
`archives/legacy-2026-11-22/{Run_QSVC_Mixed_Variance.py,Run_QSVC_Only.py,
Generate_Final_Graphs.py,Advanced_Evaluation_Plots.py,variance_mixed_benchmark_metrics.csv,
final_benchmark_metrics.csv}`, `requirements.txt`, `environment.yml`,
`tools/verify_central.py`, `docs/SWARM_BOARD.md`, `.gitignore`, and the private paper
`.paper/plant-1.md` (read locally; not transmitted).

**Claim-to-evidence map (drives the experiment list in §2):**

| Paper claim / artefact | Audit status | Regeneration path |
| --- | --- | --- |
| QSVC mixed 93.20% ± 3.01%, 10 runs × 50 test images | partially supported (legacy CSV has the 10 run rows; no per-sample data, no seeds/CI) | E2 + E4 |
| p-values 0.0013 / 0.0009 (Table 4) | unsupported (no test implemented or logged) | E4 |
| IoU table (Table 5) | unsupported as labelled — values are precision/recall, fractional TP/FP/FN | E1/E2 with `binary_metrics` |
| Gaussian-noise degradation (Table 6) | unsupported (no experiment exists) | E3 |
| QSVC train/test identity, leakage caveat | correctness bug; must be fixed upstream | E1/E2 guard in §2.0 |
| Training curves / SOTA bars / radar (Figs 8, 14, 15) | hardcoded arrays in `Advanced_Evaluation_Plots.py`, `Generate_Final_Graphs.py` | §4 forbids these inputs |
| "quantum advantage" implication from kernel circuit | contradicted: adjoint amplitude embedding, no entanglement/trainable params → classical polynomial kernel | E2 classical-baseline arm |

**Next bottom-up step (after this spec):** implement `experiments/runner.py` to this contract,
then `experiments/figures.py`, then `tests/`; acceptance = `tools/verify_central.py` passes
with the new tests and a synthetic-fixture end-to-end run reproduces its own metrics JSON.

---

## 1. Runner API for `experiments/runner.py`

### 1.1 Stage contract (config → data → models → raw artifacts → metrics → figures)

The runner is a thin orchestrator over six **pure stages**. Each stage reads only its
declared inputs and writes only its own output subdirectory; it returns a manifest entry.
Stages must be independently re-runnable from disk so that `metrics` and `figures` never
need the dataset or model checkpoints again.

```python
# experiments/runner.py (contract; implementation is another worker's job)
from experiments.config import ExperimentConfig

@dataclass(frozen=True)
class ResolvedConfig:
    config: ExperimentConfig          # from config.py, fully resolved
    experiment_id: str                # e.g. "qsvc_mixed_variance"
    run_id: str                       # e.g. "20261122T143012Z-a1b2c3d"
    seed_map: dict[str, int]          # every derived seed, named
    config_hash: str                  # sha256 of canonical JSON of config.as_dict()
    output_dir: Path

def resolve_config(argv: list[str]) -> ResolvedConfig: ...
def load_data(cfg: ResolvedConfig, spec: ExperimentSpec) -> DataBundle: ...
def build_models(cfg: ResolvedConfig, spec: ExperimentSpec) -> dict[str, ModelAdapter]: ...
def predict_all(cfg, data: DataBundle, models, spec) -> list[PredictionArtifact]: ...
def compute_metrics(cfg: ResolvedConfig, output_dir: Path) -> MetricsBundle: ...  # reads raw/ only
def render_figures(cfg: ResolvedConfig, output_dir: Path) -> list[FigureArtifact]: ...  # reads metrics/ only

def run_experiment(cfg: ResolvedConfig, spec: ExperimentSpec) -> RunManifest:
    data = load_data(cfg, spec)
    models = build_models(cfg, spec)
    raw = predict_all(cfg, data, models, spec)
    metrics = compute_metrics(cfg, cfg.output_dir)   # from disk, not from `raw`
    figures = render_figures(cfg, cfg.output_dir)    # from disk, not from `metrics` object
    return write_manifest(cfg, data, models, raw, metrics, figures)
```

Rules:

- `ExperimentConfig.from_env()` + JSON/CLI overrides is the **only** source of seeds,
  paths, and hyperparameters; `config.validate()` must return `[]` before any work starts.
  Every stochastic component receives a named seed from `seed_map` (no implicit RNG).
- `predict_all` persists one row per (experiment, run, model, corruption, draw, image)
  **before** any aggregation. `compute_metrics` calls `experiments.metrics.binary_metrics`,
  `mean_std`, `bootstrap_ci`, `paired_t_test`, `summarize_runs` and no other functions from
  `experiments.metrics`; it may additionally call `scipy.stats.mcnemar`,
  `scipy.stats.wilcoxon`, `statsmodels.stats.multipletests.multipletests`, and
  `statsmodels.stats.power.TTestPower().power` as required by E4. `scipy.stats.ttest_rel`
  is invoked only inside `experiments.metrics.paired_t_test` (metrics.py:89), never directly
  by `compute_metrics`.
- `render_figures` imports only `metrics.json` + raw artifacts. Numeric literals in plotting
  code are limited to axis styling; all plotted numbers come from loaded files.
- Rerun safety: writing to an existing `run_id` is refused unless `--force`; manifests
  record file hashes so a partial run is detectable.

### 1.2 CLI shape

```
python -m experiments.runner <subcommand> [options]

Subcommands:
  run        config -> data -> models -> raw predictions -> metrics -> figures
  metrics    recompute metrics/ from existing raw/ (no data or model access)
  figures    re-render figures/ + captions from existing metrics/ (no data/model access)
  verify     validate schemas, provenance, and figure-to-number traceability; no writes

Options:
  --experiment {main_ood,qsvc_mixed_variance,noise_robustness,significance,calibration,all}
  --config FILE              JSON overrides of ExperimentConfig fields
  --output-dir DIR           default: <config.output_dir>/<run_id>
  --seed INT                 override config.seed (master seed)
  --models ID[,ID...]        restrict model set (default: all in spec)
  --sigma-grid 0,0.05,0.1,0.15,0.2,0.3
  --runs INT                 override qsvc_runs / training-seed count where applicable
  --noise-draws INT          default 3
  --dry-run                  resolve + validate + print plan, write nothing
  --force                    allow overwriting an existing run_id
  --strict                   non-zero exit on any provenance/schema warning
```

Example:

```powershell
python -m experiments.runner run --experiment all --seed 42 `
  --output-dir results/20261122T143012Z-main --strict
python -m experiments.runner metrics --output-dir results/20261122T143012Z-main
python -m experiments.runner figures --output-dir results/20261122T143012Z-main
python -m experiments.runner verify --output-dir results/20261122T143012Z-main
```

`tools/verify_central.py` stays the mother-owned gate; `run --dry-run` and `verify` are
worker-callable self-checks (no builds/tests run by this worker).

### 1.3 Output directory layout

```
results/<run_id>/
  config.json                     # resolved config.as_dict(), canonical key order
  provenance.json                 # commit SHA, config hash, seeds, library versions, data hash (§4 rule 3)
  manifest.json                   # stage list, inputs/outputs, file hashes, status
  data/
    manifest.json                 # split indices, sample_id -> source path/hash, group ids
  models/
    index.json                    # model_id -> checkpoint path/hash, train seeds, param count
  raw/
    predictions.parquet           # per-sample rows (§3.1); CSV mirror predictions.csv
    run_summaries.csv             # per-run rows (§3.2)
  metrics/
    metrics.json                  # aggregates, comparisons, captions (§3.3)
    metrics.csv                   # flattened aggregate table for spreadsheets
  figures/
    fig01_main_ood_accuracy.png / .svg / .caption.json / .provenance.json
    fig02_qsvc_run_variance.*
    fig03_noise_robustness.*
    fig04_significance_forest.*
    fig05_reliability.*
    figures_index.json
  logs/
    runner.log                    # stage timings, warnings, full CLI invocation
```

`data/`, `models/`, and `raw/` are immutable within a `run_id`; `metrics/` and `figures/`
are derived and may be regenerated. The private `.paper/` directory is never read by the
runner and never copied into `results/`.

---

## 2. Experiments to implement (legitimate paper aims only)

### 2.0 Preconditions / leakage guard (applies to every experiment)

1. Deduplicate by MD5/content hash **before** splitting; assign each source photograph a
   `group_id`. Splits are by `group_id`, never by crop, so crops from one photograph cannot
   appear in both adaptation/train and OOD test (the paper's own caveat, §Table 1 note).
2. "Validation accuracy" from a copy of the training split is **not** a validation metric;
   label it `train_accuracy` in artifacts. Model selection uses a genuine held-out
   validation group, or is declared fixed-hyperparameter (no selection).
3. Each run records the exact `sample_id` list used for train/adaptation/test. The 10 QSVC
   runs use different subsets; paired comparisons reuse the *same* run's test list.

### 2.1 E1 — Main OOD benchmark (`main_ood`)

- Train: deduplicated Kaggle CornWeed (+ 400 DatasetNinja adaptation crops, 200/class,
  group-disjoint from test). Test: held-out DatasetNinja OOD crops.
- Models: the paper's 9 pipelines — Legacy CNN, ResNet18, MobileNetV3, EfficientNet-B0,
  YOLOv8n-cls, Custom Legacy QNN, Hybrid CNN-QNN (from scratch), Hybrid CNN-QNN
  (transfer, 10-qubit), Hybrid CNN-QNN (transfer, 4-qubit), plus Hybrid CNN-QSVC and its
  classical-kernel control (E2). Seeds: 5 training seeds per stochastic model via
  `SeedSequence(config.seed).spawn(n_models*n_seeds)`; weights from `config` + `seed_map`.
- Outputs: per-sample predictions, per-run summaries, aggregates with n/mean/std/CI,
  confusion matrices, correct IoU (§2.6), FPS measured with `time.perf_counter` over the
  full OOD set after one warm-up batch, and parameter count from
  `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
- Claim allowed: "model A scored X ± Y (n=5 seeds, 95% CI …) on this OOD split"; no
  advantage claim from the kernel circuit (see E2).

### 2.2 E2 — 10-run QSVC mixed variance (`qsvc_mixed_variance`)

Q7 is **RESOLVED by mother**: the primary protocol is legacy parity. Per run:
`qsvc_source_per_run=50` Kaggle CornWeed source train images + `qsvc_target_adaptation_per_run=50`
labelled Ninja adaptation crops = `qsvc_mixed_train_size=100` mixed train images, with
`qsvc_test_per_run=50` held-out Ninja test images and `qsvc_runs=10`. A **no-mixing control**
(source-trained QSVC trained on 100 Kaggle source crops, with no Ninja adaptation crops,
and evaluated on the same 50 held-out Ninja test images) is required. `seed_offset=42` is
retained for legacy parity (legacy used
`random_state = run_idx*42`, `*123`, `*456`). The new seed derivation replaces these with
named `SeedSequence.spawn` keys `qsvc_run_<i>_cw`, `_ninja_pool`, `_ninja_split`; the
legacy mapping is recorded in `provenance.json` for audit.

Arms (all evaluated on the **same** 50 test images per run):

1. `qsvc_quantum_kernel` — adjoint amplitude-embedding kernel on `lightning.qubit`,
   diagonal fixed at 1.0 and lower triangle mirrored, `SVC(kernel="precomputed",
   probability=True)`.
2. `svc_rbf` and `svc_linear` on the same L2-normalised 1024-d features (classical
   controls).
3. `svc_poly2_classical` computing `(A @ B.T)**2` — the exact classical equivalent of
   the no-entanglement, no-trainable-parameter circuit; this is the fair baseline. The
   paper must not claim quantum advantage from arm 1 over arm 3.
4. `qsvc_no_mixing_control` — required no-mixing control: the same source-trained QSVC
   trained on 100 Kaggle source crops (no Ninja adaptation crops), evaluated on the same
   50 held-out Ninja test images.

Report the per-run vector (legacy values average to 93.20%, sample std 3.01% — verify,
do not retype), mean/std (ddof=1), bootstrap 95% CI over runs, **and** a per-image
binomial/cluster-robust CI. With p≈0.932 and n=50, the per-run binomial SE is
`sqrt(0.932*0.068/50) ≈ 0.036`, i.e. the observed 3% run-to-run spread is largely test-set
sampling noise, not evidence of model stability. State this explicitly in the caption.

### 2.3 E3 — Gaussian-noise robustness (`noise_robustness`)

- Corruption: additive i.i.d. Gaussian noise **after** ImageNet normalisation, applied to
  OOD test tensors only: `x_noisy = clip(x + σ·ε, clip_min, clip_max)`, `ε ~ N(0,1)`.
  Record the exact clipping policy and that the same clean images, model checkpoints, and
  train set are reused for every σ. Do **not** add noise as training augmentation here.
- Sigma grid: `σ ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.30}` (σ=0.00 is the clean anchor and
  reuses E1/E2 predictions instead of re-running). σ=0.15 is the paper-equivalent single
  corruption point; the grid supplies the dose–response curve the paper lacks.
- Seeds: for each `(run_id, σ_index, noise_draw ∈ {0,1,2}, sample_id)` derive
  `noise_seed` from `SeedSequence([config.seed, run_id_int, σ_index*1000+noise_draw,
  sample_index])`. Persist `noise_seed` in every raw row; 3 draws per σ give an
  image-level uncertainty estimate.
- Models: at minimum CNN, QNN, Hybrid QC-CNN (Table 6) and QSVC mixed (E2). Primary
  endpoint: degradation `Δ(σ) = acc(0) − acc(σ)` with a paired (same-image, same-draw)
  95% CI, plus absolute `acc(σ)` with n/mean/std/CI. Caption must say "accuracy decreased
  from A to B at σ=0.15 (paired Δ = …, n = … images)", not the old "relative degradation"
  phrasing without an uncertainty.

### 2.4 E4 — Repeated-run significance testing (`significance`)

Pre-register the pair family **before** looking at results:
(Hybrid QC-CNN vs CNN), (Hybrid QC-CNN vs QNN), (QSVC mixed vs best classical OOD model).
m=3. For each pair:

- Primary: two-sided paired t-test on run-level scores, `scipy.stats.ttest_rel(a, b)`,
  paired by `run_id` and identical test `sample_id` list. Report
  `t(df = n_runs − 1)`, exact p, mean difference, bootstrap 95% CI of the difference, and
  Cohen's `dz = mean(diff)/std(diff, ddof=1)`. `experiments.metrics.paired_t_test` returns
  `n`, `mean_difference`, `t_statistic`, `p_value` and `cohens_dz` — from `n`, derive
  `df = n − 1`; the helper does **not** return df or a CI. The difference CI comes from
  `bootstrap_ci(a - b)` and the power from `TTestPower().power(dz, nobs=n, alpha=0.05)`
  at the observed dz. With n=10, df=9.
- Secondary (more sensitive to per-image changes): exact McNemar test on the 50×10=500
  paired discordant predictions (one table per pair, aggregated over runs), plus Wilcoxon
  signed-rank as a distribution-free sensitivity check.
- Multiplicity: Holm–Bonferroni across the 3 pre-registered pairs; report all three raw
  and adjusted p-values (no selective reporting).
- Decision rule: call a difference "significant" **only if** `p_holm < 0.05` **and** the
  bootstrap CI of the difference excludes 0 **and** power at observed dz ≥ 0.80; otherwise
  print "inconclusive / descriptive (n=…, CI …, power …)". The legacy p-values 0.0013 and
  0.0009 are **not** reused or cited as results.

### 2.5 E5 — Calibration (`calibration`, if feasible)

Only for models that emit a genuine score column `score_pos` (SVC `probability=True` —
flag as internal Platt scaling; neural softmax logits — flag as uncalibrated). Compute
Brier score and expected calibration error (10 equal-width confidence bins) over all OOD
predictions, with bootstrap CIs and a reliability diagram. If a model has no scores, write
`"calibration": {"status": "N/A", "reason": "no probability output"}` — never synthesize
probabilities from hard labels. With only 50 images/run, bin counts are small: aggregate
over runs (n=500) but disclose the within-run dependence.

### 2.6 Correct IoU (all experiments)

Use `experiments.metrics.binary_metrics`, which computes IoU from integer confusion counts:
`IoU_c = TP_c / (TP_c + FP_c + FN_c)` per class (class 0 treats TN as its TP). Report
`iou_class0`, `iou_class1`, `macro_iou`, and the raw `tn/fp/fn/tp` cells. Never derive IoU
from the paper's rounded precision/recall entries or fractional counts (e.g. 594.15):
Table 5's "IoU" values are precision for class 0 and recall for class 1 and must be
regenerated, not transcribed.

---

## 3. Raw-output schemas (everything in a figure is regenerable)

### 3.1 `raw/predictions.parquet` — per-sample predictions (one row per prediction)

| column | type | meaning |
| --- | --- | --- |
| `experiment_id` | string | E1–E5 identifier |
| `run_id` | string | run identifier; join key to summaries |
| `model_id` | string | model arm, e.g. `qsvc_quantum_kernel` |
| `sigma` | float | corruption level (0.0 = clean) |
| `noise_draw` | int | 0..noise_draws-1 |
| `sample_id` | string | sha256(source image bytes + crop coords); stable across runs |
| `group_id` | string | source-photograph id (leakage guard) |
| `source_dataset` | string | `kaggle_cornweed` / `datasetninja` |
| `split` | string | `train` / `adaptation` / `ood_test` |
| `y_true` | int | 0 = crop, 1 = weed |
| `y_pred` | int | hard prediction |
| `score_pos` | float \| null | P(weed) if available, else null |
| `correct` | int | 1 if y_pred == y_true |
| `inference_ms` | float | per-batch latency attributed per sample |
| `noise_seed` | int | exact seed used for this corruption draw |
| `model_seed` | int | seed from `seed_map` for this model/run |
| `checkpoint_hash` | string | hash of model weights used |
| `created_utc` | string | ISO-8601 timestamp |

### 3.2 `raw/run_summaries.csv` — one row per (experiment, run, model, sigma, draw)

`experiment_id, run_id, model_id, sigma, noise_draw, n_samples, accuracy, precision,
recall, f1, auc (nullable), iou_class0, iou_class1, macro_iou, tn, fp, fn, tp,
brier (nullable), ece (nullable), mean_inference_ms, fps, seed, config_hash`

Every aggregate in `metrics.json` must be traceable to a set of these rows plus their
underlying `predictions.parquet` rows; the manifest records the join keys.

### 3.3 `metrics/metrics.json`

```jsonc
{
  // ILLUSTRATIVE PLACEHOLDERS BELOW: schema shape only, NOT results.
  "schema_version": "1.0",
  "experiment_id": "qsvc_mixed_variance",
  "provenance": { /* §4 rule 3, including source_artifact hashes */ },
  "runs": [ /* rows of §3.2 verbatim */ ],
  "aggregates": {
    "qsvc_quantum_kernel": {
      "accuracy": {
        "n": 10, "values": [0.94, 0.96, "..."],  // per-run vector only; per-sample rows are in raw/predictions.parquet
        "mean": 0.932, "std": 0.0301, "ci95_low": 0.915, "ci95_high": 0.949,
        "boot_B": 2000, "boot_seed": 42, "unit": "fraction"
      }
    }
  },
  "comparisons": [{
    "pair": ["hybrid_cnn_qnn", "cnn"],
    "primary_test": "paired_t_test (scipy.stats.ttest_rel, two-sided)",
    "t_statistic": 0.0, "df": 9, "p_raw": 0.0, "p_holm": 0.0,
    "mean_difference": 0.0, "ci95_low": 0.0, "ci95_high": 0.0,
    "cohens_dz": 0.0, "power_at_observed_dz": 0.0,
    "secondary_tests": {"mcnemar_exact": {"p": 0.0, "n_discordant": 0},
                        "wilcoxon_signed_rank": {"p": 0.0}},
    "decision": "inconclusive | significant"
  }],
  "figures": [{
    "figure_id": "fig02_qsvc_run_variance",
    "source_artifacts": ["raw/predictions.parquet", "metrics/metrics.json"],
    "caption_numbers": {"mean": 0.932, "std": 0.0301, "n": 10, "ci95": [0.915, 0.949]},  // ILLUSTRATIVE placeholder numbers
    "caption": "..."
  }],
  "warnings": []
}
```

All numeric values in the example above — `values`, `mean`, `std`, CIs, test statistics and
`caption_numbers` — are **ILLUSTRATIVE PLACEHOLDERS** that demonstrate the schema shape only;
they are not results and must never be transcribed into the paper. `metrics.json` carries
per-run vectors (`values`, §3.2 run rows) and aggregates only: the per-sample prediction rows
live in `raw/predictions.parquet` and are only *referenced* by `metrics.json` (via
`source_artifacts` and the run/sample join keys), never embedded in it.

Floats are rounded to a fixed number of digits at serialisation time so that the same
seed+fixture yields byte-identical `metrics.json` (golden regression test).

---

## 4. Figure generation rules

1. **Single source of truth.** Figure builders accept a path to `metrics/metrics.json`
   (and optionally `raw/predictions.parquet` for curves/curves-derived plots) and read
   numbers from those files only. Importing arrays from
   `archives/legacy-2026-11-22/Advanced_Evaluation_Plots.py` or
   `Generate_Final_Graphs.py`, or embedding lists such as `sota_data`/`training_data`, is
   forbidden. A lint/test asserts no numeric result literals appear in figure modules.
2. **Captions generated from computed numbers.** Captions are rendered from
   `caption_numbers` via templates, e.g.
   `"QSVC mixed OOD accuracy {mean:.2%} ± {std:.2%} (n={n} runs, 50 images/run;
    bootstrap 95% CI {lo:.2%}–{hi:.2%}; seed {seed})."`. Caption text and the numbers it
   contains are stored in `<fig>.caption.json` and in `metrics.json` so prose and figure
   cannot drift apart. Plots show error bars/CI only when the CI exists in the JSON.
3. **Provenance attached to every figure.** Each figure gets `<fig>.provenance.json` and
   embedded PNG/SVG metadata with: git commit SHA, resolved `config_hash`, master seed and
   `seed_map`, data-manifest hash, generating command, and library versions. Figures whose
   provenance sidecar is missing fail `runner verify`.
4. **Determinism.** Fixed model ordering (sorted `model_id`), fixed colour map by model
   family, fixed matplotlib rcParams, no RNG in plotting; output both PNG (300 dpi) and
   SVG. CI bands are computed upstream with recorded bootstrap seeds, never resampled in
   the plotting code.
5. **Traceability check.** `runner verify` re-derives each caption number from the raw
   rows and compares it to `caption_numbers` and to the value embedded in `metrics.json`;
   mismatch is a hard error. Pixel-identical PNGs are not required across matplotlib
   versions, but caption JSON and metrics JSON must match exactly.
6. **Honest visualisation.** Error bars are n/mean/std/CI, never a single point sold as
   "the result"; per-run scatter is overlaid on bar summaries where n ≤ 10; axis labels
   state units (fraction or %) and n. Reliability/noise plots use the exact σ grid and
   seeds recorded in raw rows.

---

## 5. Statistical reporting standard

Every reported number carries: **n**, **mean**, **std (ddof=1)**, **95% CI** (percentile
bootstrap, `experiments.metrics.bootstrap_ci`, B=2000, recorded seed), and the
**full per-run vector** in `metrics.json` (per-sample rows stay in `raw/predictions.parquet`,
which `metrics.json` only references). Comparisons additionally carry the
exact test name, statistic, df, p-value, effect size, and multiplicity adjustment:

| Quantity | Standard |
| --- | --- |
| Group/run summary | `n, mean, std(ddof=1), CI95_low, CI95_high, boot_B, boot_seed` |
| Paired difference | `paired_t_test` (`scipy.stats.ttest_rel`, two-sided), `t(n−1)`, mean Δ, bootstrap CI of Δ, Cohen's `dz` |
| Paired binary | exact McNemar (`scipy.stats.mcnemar`, `exact=True`) on discordant counts |
| Distribution-free sensitivity | Wilcoxon signed-rank (`scipy.stats.wilcoxon`) |
| Multiplicity | Holm–Bonferroni over the pre-registered family (`statsmodels.stats.multitest.multipletests(method="holm")`) |
| Small-sample power | `statsmodels.stats.power.TTestPower().power(dz, nobs=n, alpha=0.05)` |

Avoiding false significance:

- n=10 runs → df=9; the percentile bootstrap over 10 values is unstable. Always publish
  the 10 values alongside the CI and mark conclusions with n and power. Do not interpret
  a p-value without the effect size and CI.
- Each QSVC run uses only 50 test images; accuracy is quantised in 2% steps and has ≈3.6%
  sampling SE. A run-level std of ~3% is consistent with test-set noise; do not describe
  it as proved model stability or as evidence of a distributional difference.
- Significance is claimed only under the three-part rule in §2.4; otherwise the wording
  is "descriptive/inconclusive" with the CI quoted. The legacy p-values (0.0013, 0.0009)
  and the paper's "p < 0.05" statement are unsupported and must be replaced by E4 output.
- No post-hoc test selection: report all tests run, pre-register the primary endpoint and
  pair family in `config.json`/`manifest.json` before inspecting results.
- Use correct test assumptions: normality of run differences is unverifiable at n=10, so
  the paired t-test is reported with the Wilcoxon/McNemar sensitivity checks; if they
  disagree, say so rather than choosing the smaller p-value.

---

## 6. Unit/integration test plan and risks

### 6.1 Tests (under `tests/`, picked up by `tools/verify_central.py`)

Unit:

- `test_metrics.py`: extend the existing `experiments.metrics._self_test` — IoU identity
  `TP/(TP+FP+FN)` vs precision/recall mismatch; confusion matrix orientation; `mean_std`
  uses ddof=1; `bootstrap_ci` deterministic for fixed seed and brackets the point estimate;
  `paired_t_test` sign, `cohens_dz`, and `ValueError` for n<2; `summarize_runs` field set.
- `test_config.py`: `from_env` override precedence; `validate()` rejects missing NINJA dirs
  and `feature_dim != 2**n_qubits`; canonical `config_hash` stable under key reordering.
- `test_seeds.py`: derived seeds unique/collision-free across (run, σ, draw, sample) and
  byte-identical across two calls with the same master seed.
- `test_schemas.py`: validators accept golden rows from §3.1–3.3 and reject missing columns,
  null `y_true`, sigma outside grid, or absent provenance fields.
- `test_captions.py`: renderer injects numbers from a fixture `metrics.json`; assert a
  changed mean changes the caption and that no hardcoded numeric arrays exist in
  `experiments/figures.py` (AST/literal scan).

Integration:

- `--dry-run` resolves and validates config but creates no files.
- Synthetic-feature fixture + mock models: `run --experiment qsvc_mixed_variance` produces
  10 run rows, per-sample rows, metrics JSON, and figures; then `metrics` and `figures`
  subcommands on the same `run_id` reproduce byte-identical `metrics.json` and identical
  caption JSON **without** access to data or model files.
- `runner verify` passes on that run and fails when a provenance sidecar or a raw row is
  deliberately corrupted.
- Golden regression: same seed + fixture → byte-identical metrics JSON after fixed
  rounding; train/test group ids are disjoint (leakage assertion).
- `tools/verify_central.py` remains green: `compileall experiments`, metrics self-test,
  and `pytest -q tests`.

### 6.2 Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Train/test identity and bounding-box crop leakage (acknowledged in paper) | group-disjoint splits, persisted sample/group ids, leakage assertion in tests; otherwise all numbers are invalid |
| n=10 runs × 50 images; quantized accuracies, low power | report full vectors, CIs, power, and "inconclusive" wording; never reuse legacy p-values |
| Hardcoded arrays in legacy figure scripts | forbid imports/literals; all figures rebuilt from raw + metrics |
| Quantum kernel has no entanglement/trainable parameters (classical polynomial kernel) | include `svc_poly2_classical` control; forbid quantum-advantage claims in captions |
| O(N²) kernel circuits (100×100 per run) expensive/nondeterministic | cap matrix sizes, record device/wires/shots/transpiler seed, fix `n_jobs`, log cost |
| Gaussian-noise experiment never existed | E3 defines exact corruption, seed grid, clipping, and paired endpoint; σ=0 anchor reuses clean predictions |
| Environment drift (requirements.txt has ranges, not exact versions) | full resolved versions in `provenance.json`; prefer lockfile from `reproducible-environments` before publication |
| NINJA data paths absent in repo (`config.validate()` fails) | integration tests use synthetic fixtures; real runs require `NINJA_ANN_DIR`/`NINJA_IMG_DIR` set externally |
| Calibration on tiny bins | only if scores exist; aggregate across runs with disclosed dependence; otherwise mark N/A |

**Blocker for a full end-to-end run (not for this spec):** the external DatasetNinja image
and annotation directories and pretrained checkpoints are not in the repository. Worker
can implement and test against synthetic fixtures; real numbers require those paths plus
the mother's central verifier run.
