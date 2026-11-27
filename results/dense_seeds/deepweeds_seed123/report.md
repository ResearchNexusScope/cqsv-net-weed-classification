# Dense-angle hybrid — deepweeds — seed 123

- Branch: `feature/deepweeds-dense-angle` @ `a034187` (reviewed, not merged)
- Command-equivalent: `validate --device cpu --qubits 5 --epochs 10 --batch-size 128 --train-cap 0 --seed 123 --split-seed 123`
- Split: group-disjoint 70/20/10 train/val/test, split-seed 123
- TRAIN rows used: 12256 (cap applied: False)
- Validation rows: 3502

## Measured

- Selected epoch: 5 (val loss 0.23670871555805206)
- Validation accuracy: **0.906054**
- Paired PlantNet-RBF accuracy: 0.9011993146773273
- Difference: 0.4854368932038833 points (fixed 97, broken 80)
- Adapter gradient norm: 0.3508406984566918
- Circuit gradient norm: 0.007903579622507095
- Parameter counts: {'classical': 32832, 'quantum': 5160, 'fusion': 140, 'total': 38132}
- Split digest: `985d9aa7519939b897360b84ea0cbd8104fc83fb1e1226eb9c044cb77f5e2d8c`

## Artifacts

- Weights: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\deepweeds_seed123\best.pt`
- Metrics: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\deepweeds_seed123\metrics.json`
- Paired predictions: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\deepweeds_seed123\paired_predictions.npz`

## Claim boundary

Diagnostic validation run. No quantum advantage is asserted. Comparisons are paired against a classical RBF baseline on identical rows; significance requires n>=10 grouped splits, Holm correction and power>=0.8. Single-seed results are reported as measured values only.
