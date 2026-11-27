# Dense-angle hybrid — 4-seed reproduction summary (2026-11-28)

Branch `feature/deepweeds-dense-angle` @ `a034187` (reviewed, **not merged**).
Runs driven by a private orchestration script; no repository code was edited.
Exact per-run settings: CPU, 5 qubits, 2 entangling layers, 10 epochs, batch 128,
`--train-cap 0` (entire TRAIN split), seed = split-seed ∈ {21, 43, 100, 123},
existing group-disjoint 70/20/10 train/val/test. Each run directory contains
`report.md`, `metrics.json`, `best.pt` (trained weights) and `paired_predictions.npz`.

## DeepWeeds (PlantNet frozen 512-d features)

| Seed | Hybrid val acc | Paired RBF | Difference | fixed / broken | Selected epoch |
|---:|---:|---:|---:|---:|---:|
| 21 | 0.902913 | 0.899772 | +0.314 pts | 104 / 93 | 6 |
| 43 | 0.897202 | 0.900914 | −0.371 pts | 92 / 105 | 3 |
| 100 | 0.905768 | 0.908909 | −0.314 pts | 97 / 108 | 6 |
| 123 | 0.906054 | 0.901199 | +0.485 pts | 97 / 80 | 5 |

- Hybrid: **0.902984 ± 0.004108**, bootstrap 95% CI [0.899343, 0.905911].
- Paired RBF: **0.902698 ± 0.004186**, CI [0.900128, 0.906910].
- Paired difference +0.000286, dz 0.066, power 0.051, p_holm 0.904, CI includes 0 → **inconclusive**. Essentially tied.

## ACRE (PlantNet frozen features, annotated crop instances)

| Seed | Hybrid val acc | Paired RBF | Difference | fixed / broken | Selected epoch |
|---:|---:|---:|---:|---:|---:|
| 21 | 0.971690 | 0.976837 | −0.515 pts | 98 / 156 | 3 |
| 43 | 0.969565 | 0.977374 | −0.781 pts | 95 / 183 | 5 |
| 100 | 0.973395 | 0.978716 | −0.532 pts | 108 / 168 | 6 |
| 123 | 0.972188 | 0.980752 | −0.856 pts | 77 / 174 | 7 |

- Hybrid: **0.971709 ± 0.001599**, CI [0.970221, 0.972969].
- Paired RBF: **0.978420 ± 0.001744**, CI [0.977105, 0.979908].
- Paired difference −0.006710 (−0.671 pts), dz −3.87, power 0.997, p_holm 0.0045, CI excludes 0 — but **decision `inconclusive`** because n=4 < the pre-declared minimum n=10. Descriptively the hybrid trails RBF on every seed; no formal significance claim is made at this n.

## Claim boundary

These are validation results on four seeds, not the sealed TEST and not ten grouped
splits. No quantum advantage is asserted. A significant advantage would require
n ≥ 10 paired runs, Holm correction, power ≥ 0.8 and a CI excluding zero, with
matched classical controls — the gate in `experiments/fresh_protocol.py`. The ACRE
result is a descriptive negative at n=4; it does not become a "significant loss"
claim for the same reason it cannot be a gain claim.

Artifacts: eight run directories plus this summary. Weights are the `best.pt`
checkpoints selected by lowest validation loss. Per-run split digests in
`metrics.json` bind each checkpoint to its exact sample IDs and frozen features.
