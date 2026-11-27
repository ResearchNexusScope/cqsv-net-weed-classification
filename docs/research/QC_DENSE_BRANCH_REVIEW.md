# Review: `feature/deepweeds-dense-angle` (colleague's branch, 2026-11-27)

> Read-only review performed in an isolated git worktree at detached commit
> `a034187c0fdaf97af444e09478e62521d1d3bb4a`. The branch was **not merged** into
> `main` and no branch file was modified. All numbers below come from executed
> tests/demos/pilots on this machine, not from the branch's prose.

## 1. What the branch contains

Single commit by Codex (`a034187`, 2026-11-27 01:41 IST), 8 files, +496/−25:

| File | Role |
|---|---|
| `qmlpipe/dense_statevector.py` (new) | Differentiable dense-angle statevector, pure PyTorch, CPU/CUDA, 2–20 qubits |
| `scripts/run_dense_hybrid.py` (new) | `demo`, `validate`, `final` CLI modes with paired RBF scoring |
| `tests/test_dense_statevector.py` (new) | Independent Kronecker-reference and finite-difference gradient checks |
| `qmlpipe/fresh_hybrid.py` (modified) | Adds `dense_angle` encoding, three arms (`hybrid`/`classical`/`replacement`), train-only feature standardisation, patience checkpointing, GPU device handling |
| `qmlpipe/{ablate,evaluate,train}.py` | Lazy `experiments.metrics` imports (import-cycle avoidance) |
| `docs/DENSE_HYBRID_RUN.md` (new) | Run instructions and claim discipline |

**Architecture.** Two angles per qubit (`π·tanh(feature)`), `RZ(phi) RY(theta)|0>`,
per-layer per-wire `RZ-RY-RZ` plus a CNOT ring with cyclic offset, readout of all
qubit Z expectations. This replaces the crude 16-amplitude PCA bottleneck with a
`2q`-input / `q`-output auxiliary branch fused into the same classical classifier.
The `replacement` arm substitutes a trainable classical nonlinear ring with the
**same parameter count** as the quantum branch, so `hybrid − replacement`
isolates the circuit's contribution.

## 2. Verification performed

| Check | Result |
|---|---|
| Branch unit tests (`unittest discover`) | **3/3 OK** — gates/derivatives vs independent reference, scaled matched-control training, final-checkpoint row-identity guard |
| Existing suites on branch (`pytest`) | **26 passed** (`test_dense_statevector`, `test_fresh_hybrid`, `test_fresh_protocol`) — no regressions |
| Demo (5 qubits, synthetic) | Trains; `adapter_grad_norm=1.71`, `circuit_grad_norm=0.025` — both paths receive gradients |
| Real DeepWeeds pilot (200 rows, 2 epochs) | val acc **0.7570** (n=3502); paired RBF 0.7724, −1.54 points |
| Real DeepWeeds pilot (1,000 rows, 5 epochs) | val acc **0.8172**; paired RBF 0.8210, −0.37 points |

Same split for all real runs: DeepWeeds / PlantNet, split-seed 42, seed 0,
5 qubits, 2 layers, CPU. Checkpoints selected by lowest validation loss.

## 3. Matched-control comparison (1,000 training rows, 5 epochs, 5 qubits)

| Arm | Val accuracy | vs paired RBF | fixed / broken | Selected val loss |
|---|---:|---:|---:|---:|
| dense-angle hybrid | 0.81725 | −0.371 pts | 133 / 146 | 0.40327 |
| classical replacement (matched params) | 0.81725 | −0.371 pts | 133 / 146 | 0.40735 |
| classical only (no auxiliary branch) | 0.81896 | −0.200 pts | 127 / 134 | 0.40444 |
| PlantNet + RBF SVC (paired baseline) | 0.82096 | — | — | — |

Prediction-level check: hybrid and replacement differ on **90/3,502** samples
(logit mean |Δ| 0.121) — the equal accuracy is coincidence, not duplicate output.
Hybrid vs classical-only differ on 154/3,502.

## 4. Assessment

**Strengths**
- Genuine differentiable quantum circuit, not a renamed classical layer; the
  independent Kronecker reference verifies gates, phases and both gradient paths.
- GPU-ready up to 20 qubits — a real response to the hardware ceiling documented
  in `QC_HARDWARE_OPTIONS.md`.
- Scientifically careful controls: matched-parameter classical replacement,
  classical-only arm, paired RBF on identical rows, checkpoint bound to a SHA-256
  digest of sample IDs and frozen features, locked `final` mode that scores test
  once with no refit.
- Explicit claim discipline in the docs: crossing 90% and beating paired RBF are
  separate criteria; a single winning seed is not evidence.

**Concerns / open items**
1. At the tested budget the quantum branch gives **no accuracy gain**: hybrid ties
   its matched classical replacement and trails the no-aux classical arm slightly.
   Its gradient norm is ~2% of the adapter's, so the auxiliary signal is weak.
2. **One seed, one capped budget**. Ten grouped splits and matched seeds are
   required before any conclusion; the branch's own doc says so.
3. Early stopping selects by validation *loss* (epoch 1) while validation
   *accuracy* kept rising to 0.830 by epoch 5 — worth checking whether the
   selection metric should be accuracy or a balanced metric.
4. The legacy amplitude comparison uses a different runner and only 2 readout
   logits; use `validate --encoding amplitude --qubits 4 --device cpu` with the
   same rows/seeds for a fair full-variant comparison.
5. `final` mode scores test via `_predict_logits` rather than `evaluate_once`, so
   the single-shot guard is not enforced by that path (it is called once by
   construction); the row-identity digest does not include test labels.

## 5. Recommendation

Keep the branch; it is a real improvement over the 4-qubit amplitude head. Before
merging, run the preregistered multi-seed protocol (n=10 grouped splits, matched
train caps and seeds) for all three arms at 5 qubits, then a capped GPU pilot at
higher qubit counts (e.g. 10–20) to see whether capacity changes the ordering.
Use `final` once per locked checkpoint, report paired fixed/broken counts and
variability across seeds, and make **no** quantum-advantage claim unless hybrid
beats both controls with paired power ≥ 0.8 after Holm correction.

## 6. User-requested full reproduction (2026-11-27)

Exact command supplied by the user, run in the isolated worktree at `a034187`:
CPU, 5 qubits, 10 epochs, batch 128, `--train-cap 0` (all 12,256 TRAIN rows),
seed 0, split-seed 0. Artifacts in `full-q5-reproduction/`
(`metrics.json`, `best.pt`, `paired_predictions.npz`).

Validation trajectory: 87.35% → 88.81% → 89.66% → 90.09% → 90.29% →
**90.46%** → 90.46% → 90.43% → 90.78% → 91.01%. Checkpoint selection by lowest
validation loss chose **epoch 5** (val loss 0.2330), giving val accuracy
**0.90463** on n=3,502.

Paired PlantNet-RBF on the same rows: **0.90605**; difference **−0.143 points**,
fixed 120, broken 125. So with the full training budget the dense-angle hybrid
essentially matches the classical RBF baseline — a slight net loss, no advantage.
Circuit gradient norm was 0.0037 versus adapter 0.547 (~0.7%), so the quantum
branch still receives very little learning signal.

Honest note: the selected accuracy 0.90463 coincides numerically with the
historical 90.46% figure cited in the branch doc as a ten-split RBF mean. That is
a coincidence between a single validation split here and a different multi-split
benchmark — it is **not** a reproduction of that benchmark and no equivalence is
claimed. The split here uses split-seed 0 and has its own `split_digest`.

## 7. Review evidence

Worktree `C:\Users\sugee\AppData\Local\Temp\opencode\qc-dense-branch` at `a034187`;
pilot outputs `pilot-q5/`, `pilot-q5-1000/`, `pilot-q5-1000-replacement/`,
`pilot-q5-1000-classical/`. Branch reviewed read-only; `main` untouched at
`699ecff` (this review doc added afterwards on `main`).
