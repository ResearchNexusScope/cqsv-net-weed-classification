# Dense-angle hybrid — deepweeds — seed 21

- Branch: `feature/deepweeds-dense-angle` @ `a034187` (reviewed, not merged)
- Command-equivalent: `validate --device cpu --qubits 5 --epochs 10 --batch-size 128 --train-cap 0 --seed 21 --split-seed 21`
- Split: group-disjoint 70/20/10 train/val/test, split-seed 21
- TRAIN rows used: 12256 (cap applied: False)
- Validation rows: 3502

## Measured

- Selected epoch: 6 (val loss 0.24770572781562805)
- Validation accuracy: **0.902913**
- Paired PlantNet-RBF accuracy: 0.8997715591090806
- Difference: 0.31410622501427676 points (fixed 104, broken 93)
- Adapter gradient norm: 0.32178051269726643
- Circuit gradient norm: 0.00646963668987155
- Parameter counts: {'classical': 32832, 'quantum': 5160, 'fusion': 140, 'total': 38132}
- Split digest: `007ade285d7cb6a2a8e3c74262f092c636dd03b4ebb300db31e0c963223776da`

## Artifacts

- Weights: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\deepweeds_seed21\best.pt`
- Metrics: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\deepweeds_seed21\metrics.json`
- Paired predictions: `C:\Users\sugee\Github\SugeethJSA\Quantum_Classification\results\dense_seeds\deepweeds_seed21\paired_predictions.npz`

## Claim boundary

Diagnostic validation run. No quantum advantage is asserted. Comparisons are paired against a classical RBF baseline on identical rows; significance requires n>=10 grouped splits, Holm correction and power>=0.8. Single-seed results are reported as measured values only.
