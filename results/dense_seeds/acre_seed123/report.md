# Dense-angle hybrid — acre — seed 123

- Branch: `feature/deepweeds-dense-angle` @ `a034187` (reviewed, not merged)
- Command-equivalent: `validate --device cpu --qubits 5 --epochs 10 --batch-size 128 --train-cap 0 --seed 123 --split-seed 123`
- Split: group-disjoint 70/20/10 train/val/test, split-seed 123
- TRAIN rows used: 39571 (cap applied: False)
- Validation rows: 11326

## Measured

- Selected epoch: 7 (val loss 0.09389957785606384)
- Validation accuracy: **0.972188**
- Paired PlantNet-RBF accuracy: 0.980752251456825
- Difference: -0.8564365177467725 points (fixed 77, broken 174)
- Adapter gradient norm: 0.6331127480229609
- Circuit gradient norm: 0.00789251271635294
- Parameter counts: {'classical': 32832, 'quantum': 5160, 'fusion': 140, 'total': 38132}
- Split digest: `54766446cfaa9a027943e9b93e510247bc1e3df61ed88bdf91d34c7fce0c63a6`

## Artifacts

- Weights: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\acre_seed123\best.pt`
- Metrics: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\acre_seed123\metrics.json`
- Paired predictions: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\acre_seed123\paired_predictions.npz`

## Claim boundary

Diagnostic validation run. No quantum advantage is asserted. Comparisons are paired against a classical RBF baseline on identical rows; significance requires n>=10 grouped splits, Holm correction and power>=0.8. Single-seed results are reported as measured values only.
