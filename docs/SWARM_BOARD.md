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

**Gate rule:** a D is not accepted until the central verifier passes and a Critic audit
returns PASS. Mother commits D-by-D after the gate.
