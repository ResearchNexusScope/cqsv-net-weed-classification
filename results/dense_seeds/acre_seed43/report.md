# Dense-angle hybrid — acre — seed 43

- Branch: `feature/deepweeds-dense-angle` @ `a034187` (reviewed, not merged)
- Command-equivalent: `validate --device cpu --qubits 5 --epochs 10 --batch-size 128 --train-cap 0 --seed 43 --split-seed 43`
- Split: group-disjoint 70/20/10 train/val/test, split-seed 43
- TRAIN rows used: 39605 (cap applied: False)
- Validation rows: 11270

## Measured

- Selected epoch: 5 (val loss 0.09629660844802856)
- Validation accuracy: **0.969565**
- Paired PlantNet-RBF accuracy: 0.9773735581188997
- Difference: -0.7808340727595398 points (fixed 95, broken 183)
- Adapter gradient norm: 0.6658092940039303
- Circuit gradient norm: 0.012557659298181534
- Parameter counts: {'classical': 32832, 'quantum': 5160, 'fusion': 140, 'total': 38132}
- Split digest: `b558e0977fbf41cfc2ec0b759fc1d085bef7ed552abb38dc942f47eb030fc824`

## Artifacts

- Weights: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\acre_seed43\best.pt`
- Metrics: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\acre_seed43\metrics.json`
- Paired predictions: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\acre_seed43\paired_predictions.npz`

## Claim boundary

Diagnostic validation run. No quantum advantage is asserted. Comparisons are paired against a classical RBF baseline on identical rows; significance requires n>=10 grouped splits, Holm correction and power>=0.8. Single-seed results are reported as measured values only.
