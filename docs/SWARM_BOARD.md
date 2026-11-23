# Swarm board — Quantum_Classification rebuild

Append-only. Mother owns this file. Workers report; they do not edit it.

**[1 | 2026-11-22] START → all — mother (QC-REBUILD-2026-11-22)**
Ground-up rebuild started. Legacy code archived under `archives/legacy-2026-11-22/`.
Custom `swarm` and `master-scientist` skills installed and verified through swarm registry.

**[2 | 2026-11-22] DONE_VERIFIED → all — mother (D-ARCHIVE)**
17 root files and 7 output/plot directories moved (81 files, no deletions).
Ledger: act_20261121T193546388_b03fde.

**[3 | 2026-11-22] REQUEST → QC-D01 — mother (paper spec + claim matrix)**
Owner writes `docs/research/QC-D01_spec_and_claim_matrix.md` only.

**[4 | 2026-11-22] REQUEST → QC-D02 — mother (data pipeline design)**
Owner writes `docs/research/QC-D02_data_pipeline.md` only.

**[5 | 2026-11-22] REQUEST → QC-D03 — mother (quantum model design)**
Owner writes `docs/research/QC-D03_quantum_models.md` only.

**[6 | 2026-11-22] REQUEST → QC-D04 — mother (experiments/statistics/figures)**
Owner writes `docs/research/QC-D04_experiments_statistics_figures.md` only.

**[7 | 2026-11-22] REPORT → all — QC-D01 (spec + claim matrix delivered)**
`docs/research/QC-D01_spec_and_claim_matrix.md` written. Critic audit dispatched.

**[8 | 2026-11-22] REPORT → all — QC-D02 (data pipeline delivered)**
`docs/research/QC-D02_data_pipeline.md` written. Q7 ruling requested.

**[9 | 2026-11-22] DECISION → QC-D02 — mother (QSVC protocol)**
Legacy parity: 50 Kaggle + 50 labelled Ninja = 100 mixed train, 50 Ninja test.
No-mixing control required. Config fields made explicit. Worker finalizing.

**[10 | 2026-11-22] REPORT → all — QC-D03 (quantum models delivered)**
`docs/research/QC-D03_quantum_models.md` written. Exact PennyLane pin ruling issued.

**[11 | 2026-11-22] DECISION → QC-D03 — mother (version pin)**
PennyLane==0.40.0 and PennyLane-Lightning==0.40.0 pinned in requirements.txt and
environment.yml; runtime smoke test deferred until install. Worker finalizing.

**[12 | 2026-11-22] REPORT → all — QC-D04 (experiments/statistics/figures delivered)**
`docs/research/QC-D04_experiments_statistics_figures.md` written. Critic audit dispatched.

**[13 | 2026-11-22] CRITIC → QC-D03 — critic (QC-D03 FAIL)**
Stale range-pin claims in §0/§2 contradicted §6. Exact corrections dispatched; re-audit pending.

**[14 | 2026-11-22] CRITIC → QC-D04 — critic (QC-D04 FAIL)**
Stale qsvc_train_per_run field + 4 advisories. Exact corrections dispatched; re-audit pending.

**[15 | 2026-11-22] CRITIC → QC-D01 — critic (QC-D01 FAIL)**
False "no TensorFlow anywhere" claim + 3 advisories. Exact corrections dispatched; re-audit pending.

**[16 | 2026-11-22] REPORT → QC-D02 — QC-D02 (Q7 ruling applied)**
QSVCProtocol MIXED/NO_MIXING, explicit config fields, 100 train / 50 test, T1-T12 tests.
Q7 RESOLVED. Critic audit dispatched.

**[17 | 2026-11-22] REPORT → QC-D03 — QC-D03 (Critic corrections applied)**
Exact-pin wording fixed in §0/§2/§6; target-version and smoke-test labels added. Re-audit running.

**[18 | 2026-11-22] REPORT → QC-D01 — QC-D01 (Critic corrections applied)**
C10 scoped to released benchmark; config/C19 line refs fixed; nine models vs QSVC kernel clarified. Re-audit running.

**[19 | 2026-11-22] DONE_VERIFIED → QC-D03 — critic (QC-D03 PASS)**
Re-audit passed: exact pins consistent, target-version/smoke-test labels present, kernel
identity and implementer fixes intact. Two non-blocking advisories. Spec ACCEPTED.

**[20 | 2026-11-22] CRITIC → QC-D02 — critic (QC-D02 FAIL)**
3 blockers: manifest schema/dataclass mismatch, schema consts vs configurable seed, T12
set-type error. Exact corrections dispatched; re-audit pending.

**[21 | 2026-11-22] REPORT → QC-D04 — QC-D04 (Critic corrections applied)**
Q7 fields/control explicit, stats wording fixed, §4 rule 3 refs, placeholder labels added. Re-audit running.

**[22 | 2026-11-22] CRITIC → QC-D01 — critic (QC-D01 FAIL, 1 blocker)**
Table 2 has eight logit emitters, not nine. One-line correction dispatched; re-audit pending.

**[23 | 2026-11-22] CRITIC → QC-D04 — critic (QC-D04 FAIL, 1 blocker)**
No-mixing control used 50 Kaggle instead of ruled 100; ttest_rel omission advisory. Corrections dispatched.

**[24 | 2026-11-22] REPORT → QC-D01 — QC-D01 (count fix applied)**
Table 2 count is eight logit emitters + QSVC as ninth; released benchmark clarified as 8+2=10. Final re-audit running.

**[25 | 2026-11-22] DONE_VERIFIED → QC-D01 — critic (QC-D01 PASS)**
Final re-audit passed with no findings. Spec ACCEPTED.

**[26 | 2026-11-22] REPORT → QC-D04 — QC-D04 (control fix applied)**
100-Kaggle no-mixing control restored; ttest_rel scoped to paired_t_test. Final re-audit running.

**[27 | 2026-11-22] DONE_VERIFIED → QC-D04 — critic (QC-D04 PASS)**
Final re-audit passed with no findings. Spec ACCEPTED.

**[28 | 2026-11-22] REPORT → QC-D02 — QC-D02 (Critic blockers fixed)**
Manifest schema/dataclass aligned, seed/qsvc fields non-const with verify() checks, T12 set types fixed, advisories applied. Final re-audit running.

**[29 | 2026-11-22] DONE_VERIFIED → QC-D02 — critic (QC-D02 PASS)**
Final re-audit passed with 3 non-blocking advisories (checksum canonicalisation, illustrative invariants, stem/basename wording). Spec ACCEPTED. All four research specs accepted.

**[30 | 2026-11-22] REQUEST → QC-I01 — mother (environment layer)**
Owner writes experiments/env.py + tests/test_config.py + tests/test_env.py only.

**[31 | 2026-11-22] REQUEST → QC-I02 — mother (data layer)**
Owner writes data/ package + tests/test_data.py only.

**[32 | 2026-11-22] REQUEST → QC-I03 — mother (metrics layer)**
Owner extends experiments/metrics.py + tests/test_metrics.py only.

**[33 | 2026-11-22] REPORT → QC-I01 — QC-I01 (environment layer delivered)**
experiments/env.py + tests/test_config.py + tests/test_env.py written. Awaiting sibling Ds for central verifier.

**[34 | 2026-11-22] REPORT → QC-I03 — QC-I03 (metrics layer delivered)**
experiments/metrics.py extended (log_loss, GroupKFold, multipletests, calibration) + tests/test_metrics.py. Mother pinned statsmodels>=0.14,<0.15 so Holm correction is available.

**[35 | 2026-11-22] REPORT → QC-I02 — QC-I02 (data layer delivered)**
data/ package + tests/test_data.py written; T1-T12 synthetic coverage; QSVC 100/50 mixed and paired 100/50 control; no hardcoded developer paths.

**[36 | 2026-11-22] TEST → audit — mother (central verifier running)**
First run failed only because pytest was absent; mother installed pytest 9.2.0 and pinned pytest>=8,<10. Re-run in progress.

**[37 | 2026-11-22] TEST → QC-I02 — mother (central verifier FAIL, 2 test bugs)**
31 passed, 2 failed: missing parent dirs in dedup fixture; ColorJitter stores scalar jitters as (1-j,1+j) ranges. Test-only corrections dispatched.

**[38 | 2026-11-22] TEST → QC-I02 — mother (central verifier FAIL, 1 test bug)**
32 passed, 1 failed: torchvision stores hue=None for zero hue jitter; test must accept None. Fix dispatched.

**[39 | 2026-11-22] DONE_VERIFIED → all — mother (central verifier GREEN)**
compileall OK, metrics self-test OK, 33/33 pytest passed. QC-I01/I02/I03 Critic audits dispatched.

**[40 | 2026-11-22] DONE_VERIFIED → QC-I01 — critic (QC-I01 PASS)**
Environment layer accepted; 3 non-blocking advisories (metadata version fallback, git SHA format assertion, benign generic fixture paths).

**[41 | 2026-11-22] DONE_VERIFIED → QC-I03 — critic (QC-I03 PASS)**
Metrics layer accepted; 5 non-blocking advisories (positive_label no-op, dz fallback docs, extra assertions, spec line drift, bootstrap-fixture assumption).

**[42 | 2026-11-22] DONE_VERIFIED → QC-I02 — critic (QC-I02 PASS)**
Data layer accepted; 3 non-blocking advisories (spec prose sha256-vs-stem, T7 verify deferral, fallback transform details).

**[43 | 2026-11-22] REQUEST → QC-I04 — mother (model layer)**
Owner writes models/classical.py, models/quantum.py, models/qsvc.py, tests/test_models.py only. Mother owns models/__init__.py after integration.

**[44 | 2026-11-22] DONE_VERIFIED → QC-I04 — mother (central verifier GREEN)**
Model layer: 44 passed, 3 skipped (PennyLane-dependent); models/__init__.py integrated by mother. Critic audit dispatched.

**[45 | 2026-11-22] DONE_VERIFIED → QC-I04 — critic (QC-I04 PASS)**
Model layer accepted; 5 non-blocking advisories (device seed, kernel batching, n_qubits inference, two test-hardening points).

**[46 | 2026-11-22] REQUEST → QC-I05 — mother (runner/experiments layer)**
Owner writes experiments/runner.py, evaluate.py, noise.py, stats.py + matching tests only.

**[47 | 2026-11-22] REQUEST → QC-I06 — mother (reporting layer)**
Owner writes reporting/ package + tests/test_reporting.py only. Mother owns experiments/__init__.py.

**[48 | 2026-11-22] REPORT → QC-I06 — QC-I06 (reporting layer delivered)**
reporting/ package + tests/test_reporting.py. No literal result arrays; sidecar provenance; guarded manuscript sync. Binding metrics.json key contract sent to QC-I05.

**[49 | 2026-11-22] TEST → QC-I05/I06 — mother (central verifier FAIL, 6 issues)**
Runner sample_qsvc_pairs arg shift (3 tests), Placeholder.render format spec, 1 wrong McNemar expectation, Holm unavailable. Mother installed statsmodels 0.15.0 and widened pin to <0.16. Worker fixes dispatched.

**[50 | 2026-11-22] REPORT → QC-I05 — QC-I05 (verifier fixes applied)**
4-arg sample_qsvc_pairs call restored + manifest guard; McNemar expectation corrected. Awaiting QC-I06 format fix before re-verification.

**[51 | 2026-11-22] TEST → QC-I05 — mother (central verifier FAIL, 2 issues)**
74 passed, 3 skipped, 2 failed: stale 3-arg sample_qsvc_pairs call in test; NaN power from TTestPower. Both fixes dispatched.

**[52 | 2026-11-22] DONE_VERIFIED → all — mother (central verifier GREEN)**
Full integrated tree: 76 passed, 3 skipped (PennyLane-dependent), compile + self-test OK. QC-I05/I06 Critic audits dispatched.

**[53 | 2026-11-22] DONE_VERIFIED → QC-I05 — critic (QC-I05 PASS)**
Runner/experiments layer accepted; 5 non-blocking robustness advisories (noise draw aggregation, clipping policy persistence, clean anchor lookup, training-time N/A cell, CLI programmatic-only).

**[54 | 2026-11-22] DECISION → QC-I06 — mother (schema drift ruling)**
Keep runner's namespaced aggregate keys; reporting gains a compatibility resolver + runner-figure-id mapping + derived caption numbers. Exact fixes dispatched.

**[55 | 2026-11-22] DONE_VERIFIED → QC-I06 — mother (central verifier GREEN)**
Reporting compatibility resolver, figure-id mapping, derived captions, atomic sidecars, dry-run validation. 83 passed, 3 skipped. Re-audit running.

**[56 | 2026-11-22] DONE_VERIFIED → QC-I06 — critic (QC-I06 PASS)**
Final re-audit passed; 2 non-blocking advisories (fig05 reliability bins, fig03 CI source keys). All research and implementation Ds accepted.

**[57 | 2026-11-22] CHECKPOINT → all — mother (final commit)**
Archive + environment/data/metrics/models/runner/reporting complete; central verifier 83 passed, 3 skipped. .paper/ remains gitignored.

**[58 | 2026-11-22] CHECKPOINT → all — mother (cleanup commit)**
Initial checkpoint de4ea1c committed bytecode caches; added __pycache__/ + *.py[cod] + .pytest_cache/ to .gitignore, removed tracked .pyc. Verified .paper/ is not tracked.

**[59 | 2026-11-22] BLOCKER → QC-I02 — mother (real-data parser bug)**
Kaggle Ninja mirror names annotations `X.jpg.json` vs images `X.jpg`; stem lookup drops all 46,336 boxes. Parser + regression test fix dispatched.

**[60 | 2026-11-22] BLOCKER → QC-I05 — mother (real-data seed pool bug)**
resolve_config spawns only 32 model seeds; 10 runs × 5 QSVC models need 50. Fix + regression test dispatched.

**[61 | 2026-11-22] TEST → QC-I05 — mother (verifier FAIL, 1 test)**
SeedSequence.spawn(n) is not prefix-stable; regression test's stability assertion was wrong. Independent per-index seed derivation dispatched.

**[62 | 2026-11-22] TEST → QC-I05 — mother (verifier FAIL, same test)**
Cursor-based derivation still shifts keys when seeds_per_model changes. Derive by (model_index, seed_index) instead.

**[63 | 2026-11-22] DONE_VERIFIED → all — mother (REAL RESULTS)**
10-run real benchmark on Kaggle CornWeed + DatasetNinja subset completed; verifier 85 passed/3 skipped after two real-data bug fixes (Ninja filename matching, seed derivation). Tables/figures generated from metrics.json; .paper/ untouched.

**[64 | 2026-11-22] REQUEST → QML-D01 — mother (QML core)**
Owner writes qmlpipe/config.py, models.py, kernel.py + tests/test_qmlpipe_core.py only.

**[65 | 2026-11-22] REQUEST → QML-D02 — mother (QML training)**
Owner writes qmlpipe/train.py, evaluate.py, scripts/train_qml.py + tests/test_qmlpipe_train.py only.

**[66 | 2026-11-22] REQUEST → QML-D03 — mother (QML improvement scope)**
Owner writes qmlpipe/ablate.py, model_card.py, docs/QML_IMPROVEMENT_PLAN.md + tests/test_qmlpipe_ablate.py only.

**[67 | 2026-11-22] REPORT → QML-D01 — QML-D01 (core layer delivered)**
qmlpipe config/models/kernel + tests. Honest degradations: noise channel not applied (warns), lightning QNN falls back to default.qubit. Mother added tools/verify_quantum.py for exact-pin execution.

**[68 | 2026-11-22] REPORT → QML-D03 — QML-D03 (improvement scope delivered)**
ablate.py (bounded 72-config grid), model_card.py, docs/QML_IMPROVEMENT_PLAN.md, tests. Documented signature boundary: train/eval need wrappers for run_ablation; failures recorded honestly.

**[69 | 2026-11-22] TEST → QML-D02 — mother (central verifier FAIL, 1 test)**
120 passed, 9 skipped, 1 failed: CLI silently ignored unknown config key feature_dim. Reject unknown keys with exit 2; fix dispatched.

**[70 | 2026-11-22] REPORT → QML-D02 — QML-D02 (CLI validation fix applied)**
Unknown config keys now rejected with exit 2; feature_dim remains derived from qubits. Tests updated. Central verifier re-running.

**[71 | 2026-11-22] DONE_VERIFIED → QML-D01 — critic (PASS, advisories only)**
Central verifier green (122 passed, 9 skipped). Quantum verifier found TorchLayer `inputs` signature bug; fixed with shots degradation. Re-verify pending.

**[72 | 2026-11-22] DONE_VERIFIED → QML-D02 — critic (PASS, advisories only)**
Training/evaluation/CLI accepted. Feature-fed transfer substitution disclosed in provenance; dry-run writes nothing; unknown config keys rejected.

**[73 | 2026-11-22] CRITIC → QML-D03 — critic (FAIL, 1 blocker)**
Model card missing improvement-plan cross-reference. Fix dispatched; re-audit pending.

**[74 | 2026-11-22] DONE_VERIFIED → QML-D03 — critic (PASS)**
Model card improvement-plan cross-reference added; re-audit passed. All QML Critic gates now PASS. Quantum verifier re-running after TorchLayer/torchvision fixes.

**[75 | 2026-11-22] TEST → QML-D01 — mother (quantum verifier FAIL, 1 test)**
43 passed, 1 failed: introspection dummy weight shape wrong for BasicEntanglerLayers. Ansatz-aware shape fix dispatched.

**[76 | 2026-11-22] DONE_VERIFIED → all QML Ds — mother (quantum verifier GREEN)**
Python 3.11 exact pins: PennyLane 0.40.0 / Lightning 0.40.0 / autoray 0.7.0; 46 qmlpipe tests passed. Real QML smoke training running.

**[77 | 2026-11-22] BLOCKER → QML-D02 — mother (smoke data plan mismatch)**
Smoke capped Ninja to 8 source images/2 per class but left adaptation_per_class=200; data prep failed. Quota alignment fix dispatched.

**[78 | 2026-11-22] REPORT → QML-D02 — QML-D02 (smoke plan fix applied)**
_data_plan helper aligns subset_files/per_class/adaptation_per_class; provenance records scope. Smoke run re-running.

**[79 | 2026-11-22] DONE_VERIFIED → QML — mother (smoke run GREEN)**
4 artifacts written; OOD n_samples=4; provenance records 8 Ninja source images / 2 per class. Full 5-epoch QML run started.

**[80 | 2026-11-22] DONE_VERIFIED → QML — mother (full run GREEN)**
5 epochs, best epoch 4; val acc 0.804; OOD n=400 acc 0.4275, AUC 0.375 — honest negative result vs classical baselines. 4 artifacts written. 3-config ablation running.

**[81 | 2026-11-22] CHECKPOINT → all — mother (final QML commit)**
QML core/training/improvement accepted; smoke, full 5-epoch run, and 3-config ablation completed with real data; model card generated from real metrics. Central verifier 123 passed/12 skipped; quantum verifier 47 passed. .paper/ private and untracked.

**[82 | 2026-11-22] DECISION → all — mother (plant-pretrained backbone)**
User flagged generic/random hybrid head. Research selected `cpoisson/plantnet300k-resnet18` (Apache-2.0, 45 MB, 306k plant images, 1081 species); weights verified locally (512-d features). Dispatching PLANT-D01/02/03.

**[83 | 2026-11-22] REPORT → all — plant-pretrained backbone integrated**
PLANT-D01 core, D02 CLI/feature-loader adaptation, D03 research note delivered; `qmlpipe/__init__.py` integrated by mother. Central verifier running.

**[84 | 2026-11-22] TEST → PLANT-D02 — mother (central verifier FAIL, 1 test)**
158 passed, 12 skipped, 1 failed: synthetic Trunked fixture expected a projection head that the real legacy path does not apply. Test-fixture fix dispatched.

**[85 | 2026-11-22] DONE_VERIFIED → PLANT-D01..D03 — mother (verifiers GREEN)**
PlantNet ResNet18 backbone integrated into qmlpipe and CLIs; research note committed. Central 159 passed/12 skipped; quantum 47 passed. Real PlantNet QML run running.

**[86 | 2026-11-22] BLOCKER → PLANT-D01 — mother (real run shape mismatch)**
QMLConfig lacks classical_dim; plantnet 512-d features hit Linear(1280→1024). Add field + validation + test.

**[87 | 2026-11-22] FIX → mother (hardcoded projection width)**
`models/classical.py:113` hardcoded Linear(1280→feature_dim), ignoring cfg.classical_dim; fixed to use config. Verified build_feature_extractor(plantnet,512) now yields Linear(512→1024). Real run re-started.

**[88 | 2026-11-22] DONE_VERIFIED → PLANT — mother (real run GREEN, honest comparison)**
PlantNet ResNet18-backed QML run: val 0.603, OOD n=400 acc 0.500, AUC 0.441, F1 0.0 (single-class prediction). ImageNet-backed: acc 0.4275, AUC 0.375, F1 0.499. No advantage claim; F1 collapse recorded.

**[89 | 2026-11-22] FIX → mother (backward compatibility)**
`models/classical.py` used cfg.classical_dim unconditionally; legacy ExperimentConfig lacks it. Fallback to 1280 preserves committed ImageNet behaviour while QMLConfig uses 512/1280. Central verifier re-running.

**[90 | 2026-11-22] DONE_VERIFIED → mother (threshold diagnosis)**
Saved PlantNet QML checkpoint scores span 0.403–0.446; default threshold predicts class 0 only. Best F1 0.667 occurs at threshold 0 (all class 1), best balanced accuracy 0.5025 — no usable separation. Journal and ledger updated.

**[91 | 2026-11-22] DONE_VERIFIED → all — mother (final central verifier GREEN)**
160 passed, 12 skipped after backward-compatibility fix. Final quantum verifier re-run in progress; checkpoint pending.

**[92 | 2026-11-22] CHECKPOINT → all — mother (final plant-pretrained closeout)**
PlantNet ResNet18 backbone selected, implemented, Critic-passed, and run on real data. ImageNet control recorded honestly (acc 0.4275/F1 0.499); PlantNet (acc 0.500/F1 0.000, no usable separation). Central verifier 160 passed/12 skipped; quantum verifier 47 passed. Journal updated; .paper/ private and untracked.

**[93 | 2026-11-23] REQUEST → QHEAD-D01 — mother (QNN head/training procedure)**
Owner writes qmlpipe/{train,models,config,heads,losses}.py, scripts/train_qml.py, tests/test_qhead_training.py only. Fix single-class collapse and OOD failure with leak-free class-balanced training + calibration.

**[94 | 2026-11-23] REQUEST → METH-D01 — mother (results-driven methodology)**
Owner writes docs/METHODOLOGY_V2.md, scripts/derive_methodology.py, tests/test_methodology.py only. Derive a corrected, evidence-based methodology from real result artifacts; no code changes to qmlpipe.

**[95 | 2026-11-23] REPORT → METH-D01 — METH-D01 (methodology v2 delivered, create-only)**
Created docs/METHODOLOGY_V2.md, scripts/derive_methodology.py, tests/test_methodology.py; no existing files modified. Recommendation: classical poly-2 SVM + supervised source-domain mixing as primary; all QNN/quantum-advantage checks fail against current evidence. Verifier/Critic pending.

**[96 | 2026-11-23] REPORT → QHEAD-D01 — QHEAD-D01 (QNN head/training delivered)**
Balanced focal loss, train-only class weights, balanced early-stop metric, ID_VAL-only temperature/logit calibration, CLI flags, provenance/claim boundary, 9 synthetic tests. No OOD improvement claimed. Verifier/Critic pending.

**[97 | 2026-11-23] BLOCKER → METH-D01 — mother (methodology script key path)**
First real execution failed with duplicated `aggregates.aggregates.` key; fix dispatched to derive_methodology.py only.

**[98 | 2026-11-23] DONE_VERIFIED → QHEAD-D01 — critic (PASS)**
Balanced loss/head/training and claim boundary verified; advisories only (test count 10, latent sampler warning path, focal-weight test gap, double-correction warning, skipped-calibration provenance shape). Code edits deferred until running verifier completes.

**[99 | 2026-11-23] REPORT → METH-D01 — METH-D01 (key-path fix applied)**
derive_methodology.py now uses a uniform key convention; real run exits 0 with 7 evidence rows / 12 checks / 5 pass / 7 fail (unchanged verdict). Regression test added for no duplicated `aggregates.` root.

**[100 | 2026-11-23] CRITIC → METH-D01 — critic (FAIL, 1 blocker)**
Check-ID mismatch in test; three advisories (doc overstatement, KeyError contract, imprecise citation). All four fixes dispatched to METH-D01's own files.

**[101 | 2026-11-23] DONE_VERIFIED → METH-D01 — mother (critic fixes verified)**
Blocker (check-ID mismatch) + 3 advisories fixed in owned files only; targeted tests pass (25 passed, 1 skipped). Full verifier re-run pending.

**[102 | 2026-11-23] REPORT → QHEAD-D01 — QHEAD-D01 (verifier fixes applied)**
Fixed 3 failing synthetic tests (single-class weight fixture, module import for monkeypatch, lowercase legacy warning) + critic advisories A2/A4/A5. Targeted tests pass.

**[103 | 2026-11-23] FIX → DATA-D01 — mother (checksum determinism)**
Root cause: `_git_commit()` is volatile build provenance included in checksum; commits landing mid-verifier changed HEAD between two builds. `canonical_json` now excludes `git_commit` (retained in full manifest). Regression test `test_checksum_is_invariant_to_git_commit` added. Targeted tests pass.

**[104 | 2026-11-23] DONE_VERIFIED → ALL — mother (verifiers green on stable tree)**
Central verifier: ALL CHECKS PASSED (184 passed, 13 skipped, exit 0). Quantum verifier: QUANTUM VERIFIER PASSED (47 passed, exit 0). No commits were made during either run. METH-D01 Critic re-audit pending.

**[105 | 2026-11-23] DONE_VERIFIED → METH-D01 — critic (PASS, re-audit)**
All 4 prior findings fixed; no BLOCKER or ADVISORY remains. Create-only, artifact-derived, matched controls, statistical standard, 12 acceptance criteria and rejected claims all reconfirmed.

**[106 | 2026-11-23] GATE → QHEAD-D01 + METH-D01 — mother (ALL GREEN, real runs authorized)**
Central verifier PASS (184 passed, 13 skipped), quantum verifier PASS (47 passed), QHEAD-D01 Critic PASS (advisories only), METH-D01 Critic PASS. Real QNN runs below are diagnostic only: n=1 per arm, no advantage/improvement claim.

**[107 | 2026-11-23] RESULTS → QHEAD-D01 — mother (real diagnostic runs committed)**
PlantNet QHEAD: val 0.859, OOD acc 0.383, AUC 0.357, F1 0.405 (collapse fixed, OOD below chance). ImageNet QHEAD: val 0.854, OOD acc 0.488, AUC 0.390, F1 0.650 (soft class-1 collapse). n=1 per arm; diagnostic only, no improvement/advantage claim.

**[108 | 2026-11-23] METHOD → METH-D01 — mother (matched-adaptation requirement added)**
New evidence exposed a confound: QNN arms get zero labelled target-domain adaptation while classical mixed arm gets 50 labelled Ninja samples (fair comparator = no-mixing 0.496). Added §4 matched-adaptation rule and §5 post-D01 diagnostic note; regenerate methodology_report.json (5 pass / 7 fail unchanged). Re-audit pending.

**[109 | 2026-11-23] DONE_VERIFIED → ALL — mother (re-verification after results commit)**
Central verifier ALL CHECKS PASSED (184 passed, 13 skipped, exit 0) and quantum verifier PASSED (47 passed) on the tree at `eaca19e`. One doc-citation test failure was found and fixed before this run. Methodology doc re-audit pending.

**[110 | 2026-11-23] DONE_VERIFIED → METH-D01 — critic (PASS on additions)**
Two new paragraphs (matched adaptation rule, post-D01 diagnostic) audited: all cited values match artifacts; QNN training-data claim accurate; n=1 boundary and failed criteria explicit; §1 table unchanged. One advisory (ImageNet values rounded) fixed to full artifact precision; targeted methodology tests 14 passed, derivation report regenerated (5 pass / 7 fail).

**[111 | 2026-11-23] CHECKPOINT → QC-REBUILD-2026-11-22 — mother (end-of-turn)**
Final tree `4eba88d` verified: central verifier ALL CHECKS PASSED (184 passed, 13 skipped, exit 0); quantum verifier PASSED (47 passed) on `eaca19e`, with `qmlpipe/` and the pinned environment unchanged since. All Ds gated: QHEAD-D01 PASS, METH-D01 PASS (incl. additions), DATA-D01 fixed and verified. Real diagnostic runs committed; journal/model card/board/ledger current; `.paper/` private and gitignored; working tree clean.

**[112 | 2026-11-23] RESEARCH → DATASETS — mother (browser survey)**
Used the connected browse MCP to survey plant/weed datasets; verified sizes/licenses/access on source pages. Findings in docs/research/QC_DATASET_RESEARCH.md. Top picks: ACRE Crop-Weed (CC BY 4.0, maize+4 weeds, Zenodo), DeepWeeds (17,509 img, CC BY 4.0, TFDS/HF), MaizeSet (3,330 img, license unstated - blocked pending confirmation). 2026 multi-dataset benchmark independently confirms cross-domain collapse and need for target adaptation.

**[113 | 2026-11-23] REQUEST → QHEAD-D02 — mother (labelled target adaptation)**
User asked to train the QNN on the labelled data too. Dispatched worker to mix Split.ADAPTATION labelled Ninja samples into QNN training with group-disjointness assertion against OOD test, provenance, OOD sealed; owns scripts/train_qml.py and tests/test_qhead_training.py only.

**[114 | 2026-11-23] REPORT → QHEAD-D02 — QHEAD-D02 (labelled target adaptation delivered)**
`--adaptation-samples N` (default 0 for reproducibility), `--adaptation-per-class`, `--adaptation-seed`; mixes Split.ADAPTATION labelled samples into TRAIN only, group/source-image disjointness asserted against OOD before feature extraction, full provenance, OOD sealed. 7 synthetic tests added. Verifier/Critic pending.

**[115 | 2026-11-23] BUILD → EXTERNAL-D01 — mother (dataset intake infrastructure)**
All 8 surveyed datasets registered in data/external.py with licence/source/size/status; resumable chunk-parallel fetcher scripts/fetch_external_datasets.py writes checksummed manifests outside the repo; 9 synthetic tests pass. Feasible sets (ACRE, DeepWeeds, MaizeSet) downloading; 8.7 GB/28.6 GB archives and manual-access sets registered but deferred.

**[116 | 2026-11-23] CRITIC → QHEAD-D02 — critic (FAIL, 1 blocker)**
`--adaptation-samples` default 0 bypassed `--config-json` override (all other flags use None). Four advisories (post-mix weight provenance test, not-applicable disjointness status, crop-vs-group selection note, adaptation-only gate order). Fix dispatched to QHEAD-D02's owned files.

**[117 | 2026-11-23] DONE_VERIFIED → QHEAD-D02 — critic (PASS, re-audit)**
Blocker (config-json override) and 4 advisories all fixed; raw default None / resolved 0 / config 12 / explicit CLI wins. Central verifier 200 passed, 13 skipped, exit 0. Matched 50-sample real run in progress.

**[118 | 2026-11-23] DONE → EXTERNAL-D01 — mother (dataset intake complete)**
Fetched and verified ACRE (1.22 GB), DeepWeeds (493 MB, 17,509 rows x 3 cols), MaizeSet (28 zips, 255 MB) outside the repo with SHA-256 manifests. 8.7/28.6 GB archives registered for resumable large fetch; 3 manual/aggregator entries registered with URLs. Report in docs/research/QC_DATASET_ACQUISITION.md.

**[119 | 2026-11-23] RESULTS → QHEAD-D02 — mother (matched adaptation runs committed)**
PlantNet +adapt50: OOD acc 0.393, AUC 0.378, F1 0.426 (small gain over no-adapt 0.383/0.357/0.405). ImageNet +adapt50: OOD acc 0.485, AUC 0.392, F1 0.649 (unchanged). Both AUCs below chance; neither beats no-mixing 0.496. Provenance: 50 samples 25/25, disjointness passed, OOD sealed. n=1 per arm, diagnostic only. Budget matched in count (50) but not proportion (3% vs 50% target).

**[120 | 2026-11-23] CHECKPOINT → QC-REBUILD-2026-11-22 — mother (end-of-turn)**
Final tree `732dfba` verified: central verifier ALL CHECKS PASSED (200 passed, 13 skipped, exit 0); quantum verifier PASSED (47 passed) on `7b73594`, with qmlpipe/ and pinned environment unchanged since. All Ds gated (QHEAD-D01/D02, METH-D01, DATA-D01, EXTERNAL-D01). Three external datasets downloaded and byte-verified with manifests; 5 more registered (2 large deferred, 3 manual/index). Matched adaptation runs committed and recorded in private journal §7. Working tree clean; `.paper/` private and gitignored.

**[121 | 2026-11-23] REQUEST → FRESH-D01 — mother (new-dataset ingestion + splits)**
User requested fresh, properly powered 70:20:10 protocol on downloaded datasets. Worker owns data/fresh_splits.py and tests/test_fresh_splits.py only: adapters for MaizeSet (3,330 img, filename labels), DeepWeeds (17,509 img, Negative→0 / 8 weeds→1) and ACRE (1,000 img, 56,530 COCO boxes, 1=crop/2=weed) plus deterministic group-disjoint 70:20:10 splits.

**[122 | 2026-11-23] REQUEST → FRESH-D02 — mother (fresh protocol + statistics)**
Worker owns experiments/fresh_protocol.py, scripts/run_fresh_protocol.py, tests/test_fresh_protocol.py only: matched classical arms (poly-2, RBF, logreg), n≥10 seeds, bootstrap CIs, paired t/Wilcoxon/Holm/power, additive provenance. No real runs until gate.

**[123 | 2026-11-23] REPORT → FRESH-D01 + FRESH-D02 — workers (powered protocol delivered)**
FRESH-D01: adapters for MaizeSet/DeepWeeds/ACRE plus deterministic group-disjoint 70:20:10 splits and provenance. FRESH-D02: feature caching, matched poly2/RBF/logreg arms, n=10 seeds, bootstrap CI, paired t/Wilcoxon/Holm/power, additive artifacts. Known blocker: FRESH-D01 rejects mixed-label source groups, which real ACRE photos have; fix dispatched. Verifier/Critic pending.

**[124 | 2026-11-23] DONE_VERIFIED → FRESH-D01 — critic (PASS)**
Adapters and 70:20:10 group-disjoint splits verified; mixed-label ACRE groups kept whole. Three non-blocking advisories (conservative MaizeSet source grouping, ACRE unknown-category default, assert under -O). Central verifier 228 passed / 13 skipped.

**[125 | 2026-11-23] CRITIC → FRESH-D02 — critic (FAIL, 1 blocker)**
Feature-cache failure appends one warning then breaks the seed loop, so expected != succeeded + skipped and run_manifest omits runs. Fix dispatched; McNemar advisory noted.

**[126 | 2026-11-23] BLOCKER → FRESH-D02 — mother (batch feature extraction)**
cache_features calls `loader(samples, None, "all")` with every sample; make_feature_loader stacks all preprocessed images into one tensor (~10 GB DeepWeeds, ~34 GB ACRE). Synthetic tests fake the loader so could not catch this. Chunked loader calls required before any real run; fix after current gates finish.

**[127 | 2026-11-23] GATE → FRESH-D01 + FRESH-D02 — mother (ALL GREEN, real runs authorized)**
Central verifier AFTER batching fix: 230 passed / 13 skipped, exit 0. Quantum verifier: 47 passed. Critics: FRESH-D01 PASS (advisories only), FRESH-D02 PASS on accounting fix and batching fix (advisories only). Real 70:20:10 protocol authorized; no quantum advantage claim.

**[128 | 2026-11-23] RUN → FRESH-PROTOCOL — mother (MaizeSet real run started)**
`--dataset maizeset --seeds 10 --backbone plantnet --backbone imagenet --models poly2,rbf,logreg --output-dir results/fresh_maizeset`, feature cache under ~/.cache/qc_datasets/features. 60 runs planned.

**[129 | 2026-11-23] RESULT → FRESH-PROTOCOL — mother (MaizeSet unusable, honest skip)**
Real MaizeSet run: all 60 runs recorded as skipped_cache_failure with the real ValueError — the downloaded dataset has 3,330/3,330 maize_* images and only YOLO class 0, no weed class. No fabricated metrics. Registry + acquisition doc updated. DeepWeeds 70:20:10 run started.

**[130 | 2026-11-23] RESULT → FRESH-PROTOCOL — mother (DeepWeeds success, ACRE path blocker)**
DeepWeeds 60/60 runs succeeded: PlantNet poly2 0.9042±0.0056 (AUC 0.966), rbf 0.9046±0.0080; ImageNet poly2 0.8681±0.0073, rbf 0.8653±0.0089; logreg significantly worse (p_holm ~1e-8, power 1.0). ACRE preparation failed: COCO file_name omits the archive's `data/` prefix; all 60 runs honestly skipped with the real KeyError. Fix dispatched.

**[131 | 2026-11-23] GATE → FRESH-D01 ACRE fix — critic (PASS)**
Resolver tries file_name → data/ prefix → unique basename, fails loudly with candidates; collision-safe crop names; source-image grouping unchanged. Central verifier 232 passed / 13 skipped. Real ACRE rerun in progress as results/fresh_acre_v2.

**[132 | 2026-11-23] STOP → FRESH-PROTOCOL — mother (ACRE SVM compute cap needed)**
ACRE feature caches completed for both backbones (PlantNet 52.6 MB, ImageNet 250.7 MB) but poly-2/RBF SVC on ~39,571 train crops consumed ~21,500 CPU-seconds in 3h without finishing. Mother stopped the process; no results fabricated. Adding deterministic per-seed TRAIN subsample cap (test stays full/sealed, n=10 preserved) before rerun.

**[133 | 2026-11-24] RESULTS → FRESH-PROTOCOL — mother (ACRE capped run complete)**
60/60 runs, 0 warnings, invariant true. ImageNet poly2 0.9777±0.0038, PlantNet RBF 0.9765±0.0035, PlantNet poly2 0.9762±0.0028. TRAIN capped at 5,000/seed (TEST full/sealed, n=10, documented no-claim note). poly2 vs logreg significant both backbones; poly2 vs RBF inconclusive. Fresh 70:20:10 classical baseline complete (DeepWeeds full-train + ACRE capped); MaizeSet rejected on data grounds. No quantum claim.

**[134 | 2026-11-24] CHECKPOINT → QC-REBUILD-2026-11-22 — mother (end-of-turn)**
Final tree `6cb1d36` verified: central verifier ALL CHECKS PASSED (235 passed, 13 skipped, exit 0); quantum verifier PASSED (47 passed) on `b0b372e`, with qmlpipe/ and pinned environment unchanged since. All Ds gated: QHEAD-D01/D02, METH-D01, DATA-D01, EXTERNAL-D01, FRESH-D01/D02 (all Critics PASS). Fresh 70:20:10 classical baseline complete on DeepWeeds (full train, 60/60) and ACRE (5k TRAIN cap, 60/60); MaizeSet rejected on data grounds. Journal/model card/board/ledger current; .paper/ private and gitignored; working tree clean.

**[135 | 2026-11-24] REQUEST → QNN-D01 — mother (fresh QNN arm)**
User requested QNN + hybrid work on the new datasets. Worker owns qmlpipe/fresh_qnn.py, scripts/run_fresh_qnn.py, tests/test_fresh_qnn.py only: pure variational QNN head on the cached 70:20:10 features, n=10 seeds, train cap, paired against committed classical poly2 per seed, sealed TEST, no advantage claim.

**[136 | 2026-11-24] REQUEST → HYBRID-D01 — mother (fresh hybrid arm)**
Worker owns qmlpipe/fresh_hybrid.py, scripts/run_fresh_hybrid.py, tests/test_fresh_hybrid.py only: classical + quantum head combined model on the same cached splits, n=10 seeds, same controls and reporting. No existing files may be modified.

**[137 | 2026-11-24] GATE → QNN-D01 + HYBRID-D01 — mother (ALL GREEN)**
Central verifier 251 passed / 15 skipped, exit 0; both Critics PASS (advisories only: budget-mismatch caveat placement, maizeset baseline warning). QNN smoke (1 seed, 1 epoch) succeeded: acc 0.480 / AUC 0.478, near chance as expected. Real runs authorized.

**[138 | 2026-11-24] RUN → QNN-D01 — mother (DeepWeeds full QNN started)**
`--dataset deepweeds --backbone plantnet --seeds 10 --train-cap 2000 --epochs 3 --qubits 4 --layers 2 --output-dir results/fresh_qnn_deepweeds`. Paired against committed classical poly-2 per seed; budget mismatch disclosed; no advantage claim.

**[139 | 2026-11-24] RESULTS → QNN-D01 — mother (fresh QNN DeepWeeds, decisive negative)**
10/10 runs, 0 warnings. Accuracy 0.5045±0.0384, AUC 0.5253 (CI includes chance). Paired vs classical poly2: difference −0.3998, p_holm=1.28e-10, power 1.0, qnn_win_supported=false. Budget mismatch (2,000 vs full train) disclosed. No advantage claim. Hybrid run in progress.

**Gate rule:** a D is not accepted until the central verifier passes and a Critic audit
returns PASS. Mother commits D-by-D after the gate.
