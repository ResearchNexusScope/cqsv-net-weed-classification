# Dense-angle hybrid — deepweeds — seed 100

- Branch: `feature/deepweeds-dense-angle` @ `a034187` (reviewed, not merged)
- Command-equivalent: `validate --device cpu --qubits 5 --epochs 10 --batch-size 128 --train-cap 0 --seed 100 --split-seed 100`
- Split: group-disjoint 70/20/10 train/val/test, split-seed 100
- TRAIN rows used: 12256 (cap applied: False)
- Validation rows: 3502

## Measured

- Selected epoch: 6 (val loss 0.2390979677438736)
- Validation accuracy: **0.905768**
- Paired PlantNet-RBF accuracy: 0.9089091947458595
- Difference: -0.31410622501427676 points (fixed 97, broken 108)
- Adapter gradient norm: 0.4355291719397444
- Circuit gradient norm: 0.004999107681214809
- Parameter counts: {'classical': 32832, 'quantum': 5160, 'fusion': 140, 'total': 38132}
- Split digest: `5d6752d093220e85e2d65226622208aaef01a2c85ff0d7666530b7de2338744a`

## Artifacts

- Weights: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\deepweeds_seed100\best.pt`
- Metrics: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\deepweeds_seed100\metrics.json`
- Paired predictions: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\deepweeds_seed100\paired_predictions.npz`

## Claim boundary

Diagnostic validation run. No quantum advantage is asserted. Comparisons are paired against a classical RBF baseline on identical rows; significance requires n>=10 grouped splits, Holm correction and power>=0.8. Single-seed results are reported as measured values only.
