# Quantum_Classification — ground-up rebuild

This repository is being rebuilt so that the released code actually satisfies the
manuscript's methodology. The legacy implementation is preserved, unmodified, in
`archives/legacy-2026-11-22/`.

## Privacy

The manuscript and its Markdown conversion live in `.paper/`, which is gitignored.
Never commit or upload that directory.

## Layout

- `experiments/` — new, leak-free implementation (config, data, models, metrics, runner)
- `docs/research/` — D-ID research specifications produced by the swarm workers
- `docs/SWARM_BOARD.md` — mother-owned D-ID status board
- `tools/verify_central.py` — single central verifier (mother only)
- `archives/legacy-2026-11-22/` — all previous scripts, outputs, CSVs and plots

## Rebuild principles

1. No train/test identity, no group leakage, no test-set tuning.
2. Every published number is generated from raw prediction artifacts by code.
3. Quantum-kernel claims are stated honestly: the current feature map is classically
   equivalent and demonstrates no quantum advantage.
4. Unsupported results are removed from the manuscript, not fabricated.
5. Work proceeds bottom-up: environment -> data -> models -> experiments -> figures -> paper.

## Environment

Python 3.11 is recommended. Install with either:

```bash
pip install -r requirements.txt
# or
mamba env create -f environment.yml
```

The DatasetNinja paths must be supplied through `NINJA_ANN_DIR` and `NINJA_IMG_DIR`.

## Quantum ML pipeline (`qmlpipe/`, 2026-11-22)

A dedicated QML training layer is included with scope for training and improvement:

- `qmlpipe/config.py` — `QMLConfig` (qubits, layers, embedding, ansatz, device, shots,
  optimizer, patience, noise) with validation and env overrides
- `qmlpipe/models.py` — amplitude/angle embedding, StronglyEntangling/BasicEntangler
  ansätze, Pauli-Z readout, frozen EfficientNet feature extractor
- `qmlpipe/kernel.py` — real `lightning.qubit` quantum kernel, classical `(A@B.T)²`
  oracle, symmetry/PSD diagnostics; explicit no-quantum-advantage statement
- `qmlpipe/train.py` / `qmlpipe/evaluate.py` — seeded training, scheduler, early
  stopping, best-checkpointing, honest metrics (n / std / CI), QSVC evaluation
- `qmlpipe/ablate.py` / `qmlpipe/model_card.py` — bounded 72-config ablation grid
  and a model card that renders `not measured` rather than inventing numbers
- `scripts/train_qml.py` — real-data CLI; reuse the leak-free data helpers
- `docs/QML_IMPROVEMENT_PLAN.md` — prioritized P0–P7 improvement roadmap

Exact-pin environment: Python 3.11, PennyLane 0.40.0, PennyLane-Lightning 0.40.0,
`autoray==0.7.0`, numpy 2.0.2. Run `python tools/verify_quantum.py` for the real
quantum tests; `tools/verify_central.py` runs under Python 3.14 and checks clean skips.

## Implementation status (2026-11-22)

All layers accepted after Critic gates: environment, data, metrics, models,
runner/experiments, and reporting. Central verifier: **83 passed, 3 skipped**
(the skips are PennyLane-dependent tests; the classical kernel oracle is tested
in their place). See `docs/SWARM_BOARD.md` for the full D-ID ledger.

No real benchmark numbers have been generated yet — the external DatasetNinja
directories and trained checkpoints are still required. All tests use synthetic
fixtures; no legacy result number was carried forward.

## Verification

Only the orchestrator runs the central verifier:

```bash
python tools/verify_central.py
```

No worker may run builds, commits, or competing test processes.
