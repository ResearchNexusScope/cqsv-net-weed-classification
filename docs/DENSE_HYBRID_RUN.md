# Run the dense-angle CQSV-Net experiment

## Show the model on a laptop

Open a terminal in the repository folder containing `qmlpipe` and `scripts`. On Windows, the following commands use Python 3.11. If your current Python already has PyTorch and NumPy, skip installation and run the last two commands with that interpreter.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install torch numpy
.\.venv\Scripts\python scripts/run_dense_hybrid.py demo --qubits 5
.\.venv\Scripts\python -m unittest discover -s tests -p test_dense_statevector.py -v
```

The demo trains the actual fused classifier on 64 synthetic feature rows and validates on 32 more. It needs no network, image data or GPU once the dependencies are installed. The output includes the first training batch's adapter and circuit gradient norms. Positive values show that the loss sends a learning signal to both components. The validation accuracy here measures performance on the synthetic fixture only. Add `--output runs/demo-5` to save weights and metrics to a new directory.

## What the branch changes

The existing model starts with a frozen PlantNet ResNet18 and a 512-number feature vector per photograph. The original four-qubit path learns 16 amplitudes and passes two quantum-branch logits to the fusion layer. The new path standardizes each feature using the training rows, learns two rotation angles per qubit, applies entangling layers and passes all qubit Z expectation values to the same fusion design. At five qubits this gives ten inputs and five readouts. At twenty it gives forty inputs and twenty readouts. The simulator still tracks 2^q amplitudes per example. Doubling the input angles per qubit changes the encoder, while simulator memory and time still rise with qubit count.

All revised arms keep the frozen visual features, train on all rows by default, and restore the checkpoint with the lowest validation loss. The `classical` arm disables the auxiliary branch. The `replacement` arm substitutes a trainable classical nonlinear layer with the same total parameter count, input projection, output width and cyclic neighbor connections as the dense-angle arm. The original amplitude experiment remains available via `scripts/run_fresh_hybrid.py`. For a fuller-budget amplitude comparison with the new runner, use `validate --encoding amplitude --qubits 4 --device cpu` and the same dataset and seed arguments shown below. This compares the complete variants, including their readouts and feature scaling.

## Run on real DeepWeeds photographs

Use a Python 3.11 environment on the H100 server with a working CUDA-enabled PyTorch and matching torchvision. From the repository root, run the commands below in that environment. They also work on a laptop with CPU PyTorch if you substitute `--device cpu`. Choose a small training cap for a laptop pilot. The first run still reads and extracts features for the dataset. Have network access or the project's pre-downloaded weights and data available for this step.

```bash
python -m pip install -r requirements.txt pyarrow huggingface_hub
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python scripts/fetch_external_datasets.py --dataset deepweeds --root ../datasets
python scripts/run_dense_hybrid.py validate --device cuda --qubits 5 --seed 0 --split-seed 0 --root ../datasets --cache-dir ../feature-cache --output runs/hybrid-q5-s0
python scripts/run_dense_hybrid.py validate --device cuda --arm classical --seed 0 --split-seed 0 --root ../datasets --cache-dir ../feature-cache --output runs/classical-s0
python scripts/run_dense_hybrid.py validate --device cuda --arm replacement --qubits 5 --seed 0 --split-seed 0 --root ../datasets --cache-dir ../feature-cache --output runs/replacement-q5-s0
```

If you have the team's parquet already, place it at `../datasets/deepweeds/data/train-00000-of-00001.parquet` and skip the fetch. Keep the same dataset root spelling and cache directory across commands. If cache contents change, the final evaluation checks the identity and order of the frozen features. Restart with a new output directory for each run.

Start with 5 qubits. To profile the upper end, run a short validation pilot using `--qubits 20 --batch-size 1 --epochs 1 --train-cap 64` and a new output directory. Validation still covers the whole validation split. A pilot checkpoint that actually capped the training set cannot be used for final evaluation. For a fair comparison, match the training examples, batch size, tuning budget, initialization seeds and split across arms. Hold `--split-seed` fixed if varying the optimizer's `--seed` on a single held-out split. Use the validation results and runtime to choose which qubit counts merit full training.

## Score the locked models

DeepWeeds here is the repository's binary task. Each photograph is labelled as one of the eight target weeds (1) or the negative category (0). The historical 90.46% figure is the mean for a frozen PlantNet extractor followed by a classical radial-basis-function support vector classifier (RBF SVC) over ten grouped splits. It is a benchmark for this particular task and split procedure. A nine-species score or an external recognition service requires its own comparable evaluation.

Choose the configuration on validation, then run final evaluation once per predeclared model and seed. For example, after the full five-qubit run above:

```bash
python scripts/run_dense_hybrid.py final --device cuda --checkpoint runs/hybrid-q5-s0/best.pt --root ../datasets --cache-dir ../feature-cache --output runs/final-hybrid-q5-s0
```

Repeat with the two locked control checkpoints and distinct output folders. Final mode reloads the exact saved network, fits the repository's PlantNet-RBF classifier to the same training rows, and scores both on identical held-out photographs. In `metrics.json`, `accuracy` is a fraction, `difference_points` is the model's accuracy minus RBF accuracy in percentage points, `fixed` counts RBF errors the model corrected, and `broken` counts RBF successes the model lost. The per-image IDs, labels, predictions and model logits are in `paired_predictions.npz`. The epoch index recorded in the JSON starts at zero.

Crossing 90% and beating the paired RBF score are separate criteria. To attribute a gain to the circuit, compare the locked hybrid with both classical controls using the same images and budgets. Report paired counts and variability across the prespecified runs, including losses. A single winning seed cannot establish a reliable advantage.
