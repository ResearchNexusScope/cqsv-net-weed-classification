# Dense-angle hybrid — acre — seed 21

- Branch: `feature/deepweeds-dense-angle` @ `a034187` (reviewed, not merged)
- Command-equivalent: `validate --device cpu --qubits 5 --epochs 10 --batch-size 128 --train-cap 0 --seed 21 --split-seed 21`
- Split: group-disjoint 70/20/10 train/val/test, split-seed 21
- TRAIN rows used: 39618 (cap applied: False)
- Validation rows: 11268

## Measured

- Selected epoch: 3 (val loss 0.08787085860967636)
- Validation accuracy: **0.971690**
- Paired PlantNet-RBF accuracy: 0.9768370607028753
- Difference: -0.5147319843805453 points (fixed 98, broken 156)
- Adapter gradient norm: 0.48488971920089013
- Circuit gradient norm: 0.005862967111170292
- Parameter counts: {'classical': 32832, 'quantum': 5160, 'fusion': 140, 'total': 38132}
- Split digest: `7d18241a447153b74f2c94c1e31534e68d38167b20030ae8066b0b5c1d956ea5`

## Artifacts

- Weights: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\acre_seed21\best.pt`
- Metrics: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\acre_seed21\metrics.json`
- Paired predictions: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\acre_seed21\paired_predictions.npz`

## Claim boundary

Diagnostic validation run. No quantum advantage is asserted. Comparisons are paired against a classical RBF baseline on identical rows; significance requires n>=10 grouped splits, Holm correction and power>=0.8. Single-seed results are reported as measured values only.
