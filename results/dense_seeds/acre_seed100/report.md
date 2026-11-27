# Dense-angle hybrid — acre — seed 100

- Branch: `feature/deepweeds-dense-angle` @ `a034187` (reviewed, not merged)
- Command-equivalent: `validate --device cpu --qubits 5 --epochs 10 --batch-size 128 --train-cap 0 --seed 100 --split-seed 100`
- Split: group-disjoint 70/20/10 train/val/test, split-seed 100
- TRAIN rows used: 39619 (cap applied: False)
- Validation rows: 11276

## Measured

- Selected epoch: 6 (val loss 0.0894097238779068)
- Validation accuracy: **0.973395**
- Paired PlantNet-RBF accuracy: 0.9787158566867683
- Difference: -0.5321035828307896 points (fixed 108, broken 168)
- Adapter gradient norm: 1.2256573485248028
- Circuit gradient norm: 0.01089748740196228
- Parameter counts: {'classical': 32832, 'quantum': 5160, 'fusion': 140, 'total': 38132}
- Split digest: `d028af53ab0b1f14c8bf6fa5ca33bc7d5166ec0b0573dfd242224968a4a81b96`

## Artifacts

- Weights: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\acre_seed100\best.pt`
- Metrics: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\acre_seed100\metrics.json`
- Paired predictions: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\acre_seed100\paired_predictions.npz`

## Claim boundary

Diagnostic validation run. No quantum advantage is asserted. Comparisons are paired against a classical RBF baseline on identical rows; significance requires n>=10 grouped splits, Holm correction and power>=0.8. Single-seed results are reported as measured values only.
