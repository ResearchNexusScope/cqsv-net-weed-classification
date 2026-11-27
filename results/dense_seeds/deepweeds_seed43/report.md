# Dense-angle hybrid — deepweeds — seed 43

- Branch: `feature/deepweeds-dense-angle` @ `a034187` (reviewed, not merged)
- Command-equivalent: `validate --device cpu --qubits 5 --epochs 10 --batch-size 128 --train-cap 0 --seed 43 --split-seed 43`
- Split: group-disjoint 70/20/10 train/val/test, split-seed 43
- TRAIN rows used: 12256 (cap applied: False)
- Validation rows: 3502

## Measured

- Selected epoch: 3 (val loss 0.2596019208431244)
- Validation accuracy: **0.897202**
- Paired PlantNet-RBF accuracy: 0.9009137635636779
- Difference: -0.3712164477441493 points (fixed 92, broken 105)
- Adapter gradient norm: 0.2670011625966151
- Circuit gradient norm: 0.00516911456361413
- Parameter counts: {'classical': 32832, 'quantum': 5160, 'fusion': 140, 'total': 38132}
- Split digest: `a2cddbea8eb540945127069d84705ac98dc84f38f693659235d20f2422149abe`

## Artifacts

- Weights: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\deepweeds_seed43\best.pt`
- Metrics: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\deepweeds_seed43\metrics.json`
- Paired predictions: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\deepweeds_seed43\paired_predictions.npz`

## Claim boundary

Diagnostic validation run. No quantum advantage is asserted. Comparisons are paired against a classical RBF baseline on identical rows; significance requires n>=10 grouped splits, Holm correction and power>=0.8. Single-seed results are reported as measured values only.
