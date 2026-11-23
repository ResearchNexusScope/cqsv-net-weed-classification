# Methodology V2 — evidence-first protocol for the Quantum_Classification rebuild

> This document is written **backwards from the committed result artifacts** under `results/**`.
> Every measurement below is cited inline as `artifact → key = value`, taken from the artifact
> itself; no figure is copied from prose, a legacy manuscript, or memory. The machine-readable
> companion `scripts/derive_methodology.py` recomputes the **quantitative** evidence and the
> acceptance verdicts directly from those artifacts; qualitative and hand-cited rows (for example
> the model-card row) are not machine-verified. If an artifact changes, regenerate both the
> numbers and the verdicts — do not edit this document by hand to make a claim survive.

## 0. Regenerating the evidence

```bash
python scripts/derive_methodology.py --root . --output methodology_report.json
```

Exact API (see the module docstring for the output schema):

- `ARTIFACT_PATHS` — logical artifact name → repository-relative path.
- `POLICY` — methodological constants (alpha, required power, minimum n, margin multipliers). **Not observations.**
- `ArtifactError` — raised for missing, unreadable or key-less artifacts; the CLI reports it and exits non-zero.
- `load_artifacts(root, paths=ARTIFACT_PATHS) -> dict` — read and validate every declared artifact.
- `observed(artifacts, artifact_id, *keys) -> dict` — one cited observation (path, exact key path, value).
- `evidence_table(artifacts) -> list[dict]` — recompute approach status from values alone.
- `acceptance_checks(artifacts) -> list[dict]` — evaluate each criterion below against values alone.
- `build_report(root) -> dict` and `main(argv)` — full report and CLI.

No result value is written into the script: all thresholds are relations between observed controls
(for example "two control SDs"), and chance is derived from `dataset_table[0].categories`.

## 1. Evidence table (classification of each approach)

| Approach | Status | Evidence (artifact → key = value) | Interpretation |
| --- | --- | --- | --- |
| Classical poly-2 kernel + supervised domain mixing | **supported (primary)** | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → aggregates["qsvc_mixed_variance/svc_poly2_classical"].accuracy = {mean: 0.918, std: 0.0466190233, n: 10}`; its `values` equal `aggregates["qsvc_mixed_variance/qsvc_quantum_kernel"].accuracy.values` element-for-element | Reproduces the entire mixing gain on identical run vectors with no quantum backend; this is the falsifiable, device-independent baseline. |
| Supervised source-domain mixing | **partially supported** | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → aggregates["qsvc_mixed_variance/qsvc_quantum_kernel"].accuracy = {mean: 0.918, std: 0.0466190233, n: 10}` vs `aggregates["qsvc_mixed_variance/qsvc_no_mixing_control"].accuracy = {mean: 0.496, std: 0.0323865541, n: 10, ci95: [0.48, 0.5161]}`; gain `0.42200000000000004` exceeds `2 × 0.0323865541 = 0.0647731082` | Large, consistent and above the control-noise margin, but the artifact records no pre-registered paired contrast for *this* comparison, so the effect is real but not yet significance-certified. |
| QSVC "quantum advantage" | **failed / rejected** | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → comparisons[0] = {pair: ["qsvc_quantum_kernel","svc_poly2_classical"], mean_difference: 0.0, cohens_dz: 0.0, ci95_low: 0.0, ci95_high: 0.0, power_at_observed_dz: 0.05, p_holm: null, decision: "inconclusive"}`, `primary_test: "paired_t_test (scipy.stats.ttest_rel, two-sided)"`; `aggregates["qsvc_mixed_variance/svc_rbf"].accuracy.mean = 0.922` exceeds the quantum mean `0.918`; `warnings` contains the classical-oracle fallback marker | The quantum kernel is numerically identical to the classical poly-2 oracle and is beaten by a plain RBF; the paired test is zero-effect, underpowered and inconclusive. |
| QNN head for ImageNet-style OOD | **failed** | `results/qml_full/metrics.json → ood = {accuracy: 0.4275, auc: 0.37545000000000006, f1: 0.4989059080962801, confusion_matrix: [[57,143],[86,114]]}` vs `training.best_val_accuracy = 0.8040201005025126` | Validation accuracy is far above OOD performance and OOD AUC is below chance; there is no generalisation signal. |
| QNN head for PlantNet-style OOD | **failed** | `results/qml_plantnet/metrics.json → ood = {accuracy: 0.5, auc: 0.44125000000000003, f1: 0.0, precision: 0.0, recall: 0.0, confusion_matrix: [[200,0],[200,0]]}` | Single-class collapse: F1 and recall are exactly zero, so no usable classifier was trained. |
| Post-hoc OOD threshold sweep | **failed** | `results/qml_plantnet/threshold_analysis.json → best_by_balanced_accuracy = {balanced_accuracy: 0.5025, recall: 0.005, threshold: 0.4444444444444445}`, `best_by_f1 = {f1: 0.6666666666666666, precision: 0.5, recall: 1.0, balanced_accuracy: 0.5, threshold: 0.0}`, `scores_summary = {min: 0.4029652178287506, max: 0.4462487995624542, std: 0.007185933260782454}`, `provenance.note = "Thresholds sweep the saved model's own OOD probabilities; no labels were used to fit weights, only to select a reported operating point. This is post-hoc analysis, not a claim of improved generalization."` | Best balanced accuracy is chance-level with near-zero recall; the F1-optimal point is the degenerate all-positive case, and the score spread is tiny. |
| Quantum-specific noise robustness | **failed / unsupported** | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → aggregates["noise_robustness/qsvc_quantum_kernel/sigma=0.05"].accuracy.mean = 0.784` equals `aggregates["noise_robustness/svc_poly2_classical/sigma=0.05"].accuracy.mean = 0.784`; the same equality holds at `sigma=0.1` (`0.602`) and `sigma=0.2` (`0.512`), while `qsvc_no_mixing_control` stays at `0.5` | Noise degradation is a property of the classical mixing/kernel pipeline, not of a quantum channel; no quantum-specific robustness was measured. |
| Qubit/layer/encoding ablation | **partially supported, exploratory only** | `results/qml_ablation/ablation_summary.json → metrics.accuracy = {mean: 0.5241666666666666, std: 0.03394235309069385, n: 3, ci95_low: 0.48500000000000004, ci95_high: 0.545}`; `metrics.auc.ci95 = [0.401225, 0.578425]`; `n_configs: 3` | Three configurations ran, but the accuracy CI spans chance and n is below the recommended minimum; no architecture is distinguishable. |
| Model-card self-assessment | **consistent with the above** | `results/qml_ablation/model_card.md` records "No quantum-advantage claim", "CPU simulation only" and "Small subset / limited power", and states that the parameter-free amplitude-embedding feature map is exactly the classical squared dot product | The committed card already withholds the claims that the results cannot support; Methodology V2 makes that restriction operational. |

## 2. (a) Data protocol — leak-free, group-aware, adaptation/test separated

The artifact counts define the protocol's shape: `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → dataset_table[0] = {dataset: "kaggle", categories: 2, training: 1594, validation: 199, test: 200, patch_size: 128, dim_reduction: 1024}` and `dataset_table[1] = {dataset: "ninja", categories: 2, training: 400, test: 400, validation: ""}`; `results/qml_full/provenance.json → data = {adaptation_per_class: 200, ood_per_class: 200, ninja_subset_files: 1200, counts: {ood: 400, train: 1594, val: 199}}`.

1. **Group by source image, not by patch.** Derive `source_image_id` from the original image file. All patches/crops from one source image must land in the same fold. Use `GroupShuffleSplit`/`GroupKFold` (`sklearn.model_selection`) for tuning and a single grouped hold-out for the final test. Assert in the data manifest that the source-ID sets of train, validation, adaptation and OOD test are pairwise disjoint. Shuffled `KFold` on patches is a correctness bug and invalidates any result.
2. **Adaptation/test separation.** Supervised source-domain mixing is fitted *only* on the adaptation pool drawn from training source images. The OOD test set is never mixed, never used for early stopping, never used to choose a threshold, and is scored once after the protocol is frozen. `results/qml_plantnet/threshold_analysis.json → provenance.note` explicitly labels the current sweep as post-hoc; V2 forbids reporting a post-hoc threshold as a generalisation result.
3. **Validation discipline.** Validation (`train: 1594, val: 199` per the artifacts above) is used for early stopping, hyperparameter choice and threshold selection only. `results/qml_plantnet/metrics.json → training = {best_epoch: 4, best_val_accuracy: 0.6030150753768844}` must not be compared against a test metric as if it were one.
4. **Provenance pinning.** Record `commit_hash`, `config_hash`, `data_manifest_hash`, master seed and per-sample hashes for every run (the fields already present under `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → provenance` and `results/qml_full/provenance.json`). A result without a manifest hash is not reportable. Environment: commit an `environment.yml` + lockfile and tag the container with the short commit SHA so the run can be recreated exactly.
5. **Feature-cache hygiene.** The QNN runs warn that "transfer QNN's image extractor replaced by its own projection so the model consumes cached feature-loader vectors (no double extraction)" (`results/qml_plantnet/metrics.json → warnings`), and that the PlantNet backbone is frozen. The cache key must include `source_image_id` and split; verify that cached vectors cannot cross the adaptation/test boundary.

## 3. (b) Primary model — what the evidence actually supports

**Primary: a classical kernel SVM (poly-2, with an RBF sensitivity check) over frozen deep features, trained with supervised source-domain mixing.** This is the only configuration whose gain is reproduced across ten independent runs without depending on quantum hardware: `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → aggregates["qsvc_mixed_variance/svc_poly2_classical"].accuracy = {mean: 0.918, std: 0.0466190233, n: 10}` and `aggregates["qsvc_mixed_variance/svc_rbf"].accuracy.mean = 0.922`, against `aggregates["qsvc_mixed_variance/qsvc_no_mixing_control"].accuracy.mean = 0.496`.

Why it is primary: (i) the QSVC and poly-2 `values` vectors are identical element-for-element in the artifact, so the observed gain comes from the data protocol (mixing + features), not from the quantum kernel; (ii) the RBF control is at least as good; (iii) it is reproducible on any CPU with a pinned environment. The quantum kernel is retained as an **exploratory comparator only** and is never the headline model until the acceptance criteria in §7 pass.

## 4. (c) Required matched classical controls

Every reported arm must be accompanied by all of the following, on the *same* grouped folds, the *same* feature cache, the *same* seeds and the *same* corruption draws:

| Control | Evidence it must reproduce | Observed value (artifact → key) |
| --- | --- | --- |
| No-mixing control | mixing gain is not an artifact of split/features | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → aggregates["qsvc_mixed_variance/qsvc_no_mixing_control"].accuracy.mean = 0.496` |
| Classical poly-2 kernel | quantum kernel adds nothing over a matched classical kernel | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → aggregates["qsvc_mixed_variance/svc_poly2_classical"].accuracy.mean = 0.918` (identical `values` to QSVC) |
| Classical RBF kernel | the best classical kernel must be the baseline to beat | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → aggregates["qsvc_mixed_variance/svc_rbf"].accuracy.mean = 0.922` |
| Linear/logistic on the same features | deep-feature baseline without kernel or quantum step | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → aggregates["main_ood/logreg_effnet"].accuracy = {mean: 0.4825, auc: 0.51935, n: 10}` |
| Chance / majority class | absolute floor | derived: `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → dataset_table[0].categories = 2` ⇒ chance `0.5`; majority-class recall must be reported because the no-mixing control reaches `recall: 0.96` with `accuracy: 0.48` |
| Paired noise controls | noise robustness must be quantum-specific, not generic | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → noise_robustness/qsvc_quantum_kernel/sigma=0.2.accuracy.mean = 0.512` vs `noise_robustness/svc_poly2_classical/sigma=0.2.accuracy.mean = 0.512` |

**Matched adaptation data is part of the control, not a free variable.** On the committed
real artifacts the classical `qsvc_mixed_variance` arm trains with 50 labelled Ninja
(target-domain) samples mixed into the Kaggle training set, whereas the QNN arms train
only on Kaggle `Split.TRAIN`/`Split.ID_VAL` and receive **zero** labelled target samples
(`scripts/train_qml.py:548-550`). A QNN arm may therefore not be compared against the mixing accuracy cited at
`results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → aggregates["qsvc_mixed_variance/svc_poly2_classical"].accuracy.mean = 0.918`;
its matched comparator is
`results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → aggregates["qsvc_mixed_variance/qsvc_no_mixing_control"].accuracy.mean = 0.496`
unless it receives the same labelled adaptation budget under the same leakage discipline.
Every arm must record its adaptation sample count and label source in provenance. This
requirement was added after the QHEAD-D01 diagnostic runs exposed the confound; it is a
protocol rule, not a result claim.

A quantum arm reported without its matched control row is an unsupported claim, not a result.

## 5. (d) QNN head and training requirements implied by the failures

The two OOD QNN arms failed in different but diagnostic ways. ImageNet-style: `results/qml_full/metrics.json → ood.auc = 0.37545000000000006` with `training.best_val_accuracy = 0.8040201005025126`. PlantNet-style: `results/qml_plantnet/metrics.json → ood.confusion_matrix = [[200,0],[200,0]]`, `ood.f1 = 0.0`, `ood.recall = 0.0`. The threshold sweep's scores barely vary (`results/qml_plantnet/threshold_analysis.json → scores_summary = {min: 0.4029652178287506, max: 0.4462487995624542, std: 0.007185933260782454}`). Required changes before a QNN arm is admissible:

1. **Executed quantum backend, fully recorded.** `results/qml_full/metrics.json → config = {device_name: "default.qubit", shots: null, noise_depolarizing: null, qubits: 10, layers: 2}` and the `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → warnings` fallback marker show that no actual quantum execution was certified. Record device, shots, transpilation seed, noise-model parameters and library versions; `shots: null` with a claimed quantum speedup is inadmissible. Absence of PennyLane must fail the run, not silently substitute the classical `(A@B.T)^2` oracle.
2. **Trainable, entangled head with a parameter-change audit.** Assert a non-zero gradient norm and a non-zero parameter delta after training, and assert entanglement exists in the ansatz. A parameter-free amplitude-embedding feature map is a classical kernel (see the committed model card) and must be labelled as such.
3. **No single-class collapse.** Use class-balanced loss/sampling; require both classes present in the predicted labels and F1 strictly above zero on the sealed OOD test. The observed `[[200,0],[200,0]]` confusion and `f1: 0.0` (`results/qml_plantnet/metrics.json → ood`) fail this.
4. **Ranking before thresholds.** Require OOD AUC above chance with a CI excluding chance before any threshold is reported. Observed `auc: 0.44125000000000003` (`results/qml_plantnet/metrics.json → ood`) and `auc: 0.37545000000000006` (`results/qml_full/metrics.json → ood`) both fail. Calibration fields (`brier_score`, `ece` in the same `ood.calibration` objects) are reported, not claimed, unless derived from validation.
5. **Sealed test and repeated seeds.** Early stopping and threshold selection on validation only; ≥10 independent seeds; paired comparison against the poly-2 control per seed. One run with `n: 1` cannot support a claim (the legacy single-run artifact `results/real_run/qc-ac9a06bc595c/metrics/metrics.json → aggregates["qsvc_mixed_variance/qsvc_quantum_kernel"].accuracy.n = 1` is retained only as provenance, never as evidence).

## 6. (e) Statistical reporting standard

- Report **n** (independent seeds/folds), the **full value vector**, **mean**, **SD with ddof=1**, and a **percentile bootstrap 95% CI** with B and seed recorded — the artifact format already does this, e.g. `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → aggregates["qsvc_mixed_variance/qsvc_no_mixing_control"].accuracy = {n: 10, boot_B: 200, boot_seed: 1482307210, values: [...], std: 0.0323865541, ci95_low: 0.48, ci95_high: 0.5161}`.
- **Paired tests** on paired runs: `paired_t_test (scipy.stats.ttest_rel, two-sided)` as primary, Wilcoxon signed-rank and exact McNemar as secondary; apply **Holm** multiplicity correction; report **Cohen's dz** and **power**. See `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → comparisons[0]` for the required field set (`n: 10`, `df: 9`, `mean_difference: 0.0`, `cohens_dz: 0.0`, `power_at_observed_dz: 0.05`, `p_holm: null`, `decision: "inconclusive"`, `decision_reason`).
- **No significance for n < 2**, and n < 10 is exploratory only: report intervals, never a "significant" label. The legacy artifact's reason — `results/real_run/qc-ac9a06bc595c/metrics/metrics.json → table_status.comparisons.reason = "no pre-registered pair had two or more paired runs"` — is the rule applied correctly.
- A "significant" decision requires all of: Holm-adjusted p below the declared alpha, a bootstrap CI excluding zero, and power at or above the declared minimum. Otherwise report `decision: "inconclusive"` with a `decision_reason`.
- Report per-class metrics (precision, recall, F1, macro IoU, confusion matrix) alongside accuracy whenever classes are imbalanced or the control's recall is high while accuracy is near chance.

**Post-D01 diagnostic (n=1 per arm; not admissible as evidence).** The corrected
balanced/calibrated recipe was applied to both backbones. PlantNet:
`results/qml_plantnet_qhead/metrics.json → ood = {accuracy: 0.3825, auc: 0.357225, f1: 0.40481927710843374, confusion_matrix: [[69,131],[116,84]]}` with `training.best_val_accuracy = 0.8592964824120602`.
ImageNet: `results/qml_imagenet_qhead/metrics.json → ood = {accuracy: 0.4875, auc: 0.3904, f1: 0.6496, confusion_matrix: [[5,195],[10,190]]}` with `training.best_val_accuracy = 0.8543`.
The exact single-class collapse is gone on PlantNet, but both OOD AUCs remain below
chance and the large validation gains did not transfer; ImageNet instead drifts to a
soft class-1 collapse. Under §7 this clears the no-collapse criterion for PlantNet but
still fails the AUC-above-chance and ≥10-seeds criteria, so neither QNN arm is admissible.
These runs are recorded in the private provenance journal and do not revise §1.

## 7. (f) Acceptance criteria with thresholds derived from observed controls

Chance is `1 / dataset_table[0].categories` = `0.5` (`results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → dataset_table[0].categories`). The control-noise margin is `2 × no_mixing_control.accuracy.std` = `2 × 0.0323865541 = 0.0647731082` (`results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → aggregates["qsvc_mixed_variance/qsvc_no_mixing_control"].accuracy.std`). Alpha, power floor and minimum n are pre-declared policy in `POLICY`, not measured values.

| # | Criterion (pass condition) | Threshold | Current observed (artifact → key) | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Mixing gain exceeds control noise | gain > control-noise margin, n ≥ 10 per arm | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → 0.918 − 0.496 = 0.42200000000000004 > 0.0647731082` | **PASS** |
| 2 | Quantum advantage over the best matched classical control | difference > 0, CI low > 0, Holm p < alpha, power ≥ floor | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → comparisons[0].mean_difference = 0.0`, `power_at_observed_dz = 0.05`, `p_holm = null`; best classical `svc_rbf.accuracy.mean = 0.922` vs quantum `0.918` | **FAIL** |
| 3 | Classical control reproduces the gain | poly2 accuracy > control + margin at n ≥ 10 | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → 0.918 > 0.496 + 0.0647731082`, `n: 10` | **PASS** |
| 4 | Every QNN OOD arm beats chance | accuracy and AUC > 0.5 | `results/qml_full/metrics.json → ood.accuracy = 0.4275`, `ood.auc = 0.37545000000000006`; `results/qml_plantnet/metrics.json → ood.auc = 0.44125000000000003` | **FAIL** |
| 5 | QNN predictions are non-degenerate | ≥ 2 classes predicted and F1 > 0 | `results/qml_plantnet/metrics.json → ood.f1 = 0.0`, `ood.confusion_matrix = [[200,0],[200,0]]` | **FAIL** |
| 6 | Threshold is a usable operating point | balanced accuracy > 0.5; recall and precision ≥ half of chance | `results/qml_plantnet/threshold_analysis.json → best_by_balanced_accuracy.recall = 0.005`; `best_by_f1.precision = 0.5`, `balanced_accuracy = 0.5` | **FAIL** |
| 7 | Noise robustness is quantum-specific | quantum/classical paired gap > 0 at a recorded sigma and retention ≥ floor | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → max gap = 0.0`; retention at `sigma=0.05` = `0.784 / 0.918 = 0.8540305010893247` | **FAIL** |
| 8 | Ablations are conclusive | n ≥ 10 and accuracy CI excludes chance | `results/qml_ablation/ablation_summary.json → metrics.accuracy.n = 3`, `ci95 = [0.48500000000000004, 0.545]` contains `0.5` | **FAIL** |
| 9 | No classical-oracle fallback | zero fallback warnings | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → warnings` contains the fallback marker `80` times | **FAIL** |
| 10 | Reporting fields complete | n, ddof=1 SD, values and CI on every compared aggregate; n ≥ 2 | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → aggregates[…].accuracy` carries all fields with `n: 10` | **PASS** |
| 11 | No significance below minimum n | zero violations | `results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → comparisons[0].n = 10`, `decision = "inconclusive"` | **PASS** |
| 12 | Ablation manifest complete | successes + failures = configured runs | `results/qml_ablation/ablation_summary.json → n_configs: 3`, `n_success: 3`, `n_failed: 0` | **PASS** |

Current state: the data/protocol and classical-control checks pass; every claim-specific quantum or QNN check fails. That is the expected outcome of the rebuild and is the reason these claims are rejected below.

## 8. (g) Explicitly rejected claims

1. **Quantum advantage — rejected.** The quantum kernel's per-run values are identical to the classical poly-2 oracle (`results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → aggregates["qsvc_mixed_variance/qsvc_quantum_kernel"].accuracy.values == aggregates["qsvc_mixed_variance/svc_poly2_classical"].accuracy.values`), the paired difference is `0.0` with power `0.05` and an `inconclusive` decision, and the classical RBF mean `0.922` exceeds the quantum mean `0.918` (same artifact). No advantage can be asserted from a parameter-free classical kernel or a simulated, classically simulable circuit.
2. **Unsupported noise robustness — rejected.** At every recorded sigma the quantum and poly-2 accuracies are identical (`results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → noise_robustness/…`), and the original claim had no matched classical control. "Robust to noise" may only be claimed after check 7 passes with a real backend and matched corruption draws.
3. **Hardcoded figures — rejected.** No manuscript number may be retyped from prose. Values must be regenerated from artifacts by `scripts/derive_methodology.py` (or an equivalent pipeline) and cited with artifact path + key, as done throughout this document. Legacy single-run figures (`results/real_run/qc-ac9a06bc595c/metrics/metrics.json → aggregates["qsvc_mixed_variance/qsvc_quantum_kernel"].accuracy.n = 1`) and post-hoc thresholds (`results/qml_plantnet/threshold_analysis.json → provenance.note`) are not reportable results.
4. **QNN OOD generalisation — rejected as evidence.** Both QNN OOD arms are at/below chance with a single-class collapse in one arm (`results/qml_full/metrics.json → ood.auc = 0.37545000000000006`; `results/qml_plantnet/metrics.json → ood.f1 = 0.0`). Until checks 4–6 pass, QNN results are architecture diagnostics only.
5. **Significance without paired power — rejected.** Any comparison lacking ≥2 paired runs, Holm-adjusted p, an interval excluding zero and power at the declared floor must be labelled `inconclusive`, never "significant" (`results/real_run10/qc-52fcca25a5b8/metrics/metrics.json → comparisons[0].decision = "inconclusive"`).

## 9. Evidence ownership and update rule

Owned artifacts of this methodology are this document, `scripts/derive_methodology.py` and `tests/test_methodology.py`. Nothing here modifies training code, result artifacts or the manuscript. When new runs land, re-run the derivation script, require **all** acceptance checks that support a claim to pass, and only then revise the evidence table in §1 with fresh citations. A check that fails is a finding, not a number to be explained away in prose.
