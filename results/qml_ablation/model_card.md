# Model card — Quantum Classification QML pipeline

> Auto-generated from the ablation result dictionary. Absent values are shown as `not measured`; no result is inferred from prose, legacy artifacts or prior runs.

## Intended use

Research-only classification benchmark for comparing quantum-circuit architecture choices (qubit count, depth, data encoding, ansatz and depolarizing noise) under a declared, leakage-free protocol (QC-D01 §1–2; QC-D04 §2.0). This card reports measured run summaries only; it is not a deployment approval, a clinical or safety claim, or evidence of quantum advantage.

## Training data provenance

Provenance fields recorded in the run configuration are reproduced below. Fields that were not recorded read `not measured`. The underlying split manifest and per-sample hashes remain the authority for leakage checks; this card does not restate them from memory.

| Field | Value |
| --- | --- |
| Dataset identifier | not measured |
| Split manifest | not measured |
| Data hash | not measured |
| Config hash | not measured |

Successful runs: 3; failed runs: 0. Failed runs contribute no metrics.

## Architecture

| Field | Value |
| --- | --- |
| Qubits | 10 |
| Layers | 2 |
| Embedding | angle |
| Ansatz | strongly_entangling |
| Feature dimension | 1024 |
| Device | default.qubit |
| Shots | not measured |
| Depolarizing noise | not measured |
| Optimizer | adamw |
| Master seed | 42 |

Quoted architecture values describe the configuration actually supplied; no default has been substituted for a missing field.

## Metrics

Sample standard deviation uses ddof=1 and the 95% interval is the percentile bootstrap over run-level metric vectors (QC-D04 §5).

| Metric | n | Mean | Std (ddof=1) | 95% CI |
| --- | --- | --- | --- | --- |
| `accuracy` | 3 | 0.524167 | 0.0339424 | [0.485, 0.545] |
| `auc` | 3 | 0.508092 | 0.0940796 | [0.401225, 0.578425] |
| `f1` | 3 | 0.327506 | 0.295291 | [0, 0.573427] |
| `log_loss` | 3 | 0.691809 | 0.00274473 | [0.690104, 0.694976] |
| `macro_iou` | 3 | 0.323899 | 0.0707534 | [0.2425, 0.370655] |
| `n_samples` | 3 | 400 | 0 | [400, 400] |
| `precision` | 3 | 0.373484 | 0.324271 | [0, 0.583333] |
| `recall` | 3 | 0.31 | 0.30753 | [0, 0.615] |
| `val_accuracy` | 3 | 0.738693 | 0.157392 | [0.633166, 0.919598] |

## Limitations

- **No quantum-advantage claim.** The parameter-free amplitude-embedding feature map is exactly the classical squared dot product `((x_i·x_j)/(||x_i|| ||x_j||))^2` (QC-D01 C5; QC-D03 §3.5). A circuit with no entanglement and no trainable parameters is a classical kernel; advantage is not asserted from it. The entangled QNN arms are simulated at 4–10 qubits and remain classically simulable, so no speedup is claimed without a matched classical control.
- **CPU simulation only.** All results come from CPU simulators; no hardware fidelity, queue time, shot-calibration or wall-clock speedup was measured. Timing and device details are reported only if present in the supplied result dictionary.
- **Small subset / limited power.** Run counts and evaluation subsets are small; per-run sampling noise can dominate run-to-run spread (QC-D04 §5). Confidence intervals are reported rather than a single point estimate, and 'significant' is not claimed from these summaries without the pre-registered paired tests, effect sizes and power analysis mandated by QC-D04 §2.4.
- **Ablation arms are not headliners.** Angle-embedding and noisy arms use different input widths or noise conventions and are never averaged with the canonical amplitude/entangling configuration (QC-D03 §4.3).

## Improvement plan

See `docs/QML_IMPROVEMENT_PLAN.md` (P0 statistics/baseline honesty, P1 matched classical controls, P3 encoding/ansatz ablations). Recommendations only; no values in this card are claims of measured advantage.

## Ethical and deployment notes

- **Not for deployment.** This model is a research comparator on a bounded image-classification subset. It has not been validated for safety, robustness, fairness or calibrated decision-making.
- **Bias and subgroup audit.** Aggregate metrics can hide subgroup failures; per-subgroup performance and a decision-threshold audit are required before any operational use (ml-for-science best practice).
- **Calibration.** Scores from precomputed-kernel SVC are internal Platt probabilities and neural softmax outputs are uncalibrated; no calibration claim is made unless a calibration result is explicitly supplied in `results`.
- **Provenance and dual use.** Record the exact environment, seeds, data hashes and commit before release. Do not present simulated quantum results as hardware evidence or use them to justify high-stakes interventions.

## Addendum: QHEAD-D01 diagnostic runs (hand-written; not part of the auto-generated ablation table)

The tables above are generated from the 3-config ablation dictionary and are unchanged.
This addendum records the later corrected-training diagnostic runs; it is not generated
by `scripts/build_qml_card.py` and will be lost if that card is regenerated.

| Arm (artifact) | Val acc | OOD acc | OOD AUC | OOD F1 | Confusion |
| --- | ---: | ---: | ---: | ---: | --- |
| PlantNet QHEAD (`results/qml_plantnet_qhead/metrics.json`) | 0.8593 | 0.3825 | 0.3572 | 0.4048 | `[[69,131],[116,84]]` |
| ImageNet QHEAD (`results/qml_imagenet_qhead/metrics.json`) | 0.8543 | 0.4875 | 0.3904 | 0.6496 | `[[5,195],[10,190]]` |

The balanced focal + ID_VAL-calibrated recipe removed PlantNet's exact single-class
collapse (F1 0.0000→0.4048) but left both OOD AUCs below chance, and the large
validation gains did not transfer. ImageNet QHEAD drifts to a soft class-1 collapse.
These 5-epoch figures are n=1 per arm and are diagnostic only: no
improvement, generalisation, noise-robustness or quantum-advantage claim is made.
Calibration was fitted on ID_VAL only (`fitted_on_split=id_val`, `ood_labels_used=false`).
Also note the matched-protocol confound in `docs/METHODOLOGY_V2.md` §4: these QNN arms
receive no labelled target-domain adaptation data, unlike the classical mixed arm.
