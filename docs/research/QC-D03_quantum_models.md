# QC-D03 — Quantum model layer specification

**D-ID:** QC-D03 (Quantum_Classification rebuild swarm)
**Owner output:** this file only. No code, build, test, commit, or other file was modified.
**Status:** design specification; all PennyLane API statements are from static inspection of
the version-targeted documentation/skill references and the legacy source, **not** from a
runtime execution. Mother's ruling sets exact pins `PennyLane==0.40.0` /
`PennyLane-Lightning==0.40.0` (§6.1); residual status is **smoke test pending exact install**.
Nothing in this document reports an observed experimental result.

---

## 0. Evidence base and privacy

The private manuscript in `.paper/` was not read, quoted, or transmitted. Design decisions
are grounded in the new config, the pinned environment, and the archived legacy code.

| Claim | Evidence (file:line) | Status |
|---|---|---|
| Canonical settings are 10 qubits / 1024 features / 2 entangling layers / seed 42 | `experiments/config.py:12-43` (fields), `:71-75` (validation) | verified, static |
| Exact pins `PennyLane==0.40.0` and `PennyLane-Lightning==0.40.0` are recorded in `requirements.txt` and `environment.yml` | `requirements.txt:17-19`, `environment.yml:20-21` (numpy cap: `requirements.txt:3`, `environment.yml:9`) | resolved, exact pins set (§6) |
| Legacy QNN: 10 qubits, `default.qubit`, `interface="torch"`, `diff_method="backprop"`, AmplitudeEmbedding + `StronglyEntanglingLayers(2,10,3)` + 10 PauliZ expvals | `archives/legacy-2026-11-22/Benchmark_Clean_Eval.py:152-162,199-217` | verified, static |
| Legacy QNN drops to CPU for the QNode and moves output back to the model device | `Benchmark_Clean_Eval.py:182,196,216` | verified, static |
| Legacy QSVC circuit: AmplitudeEmbedding(x_i) + adjoint(AmplitudeEmbedding(x_j)), measure `qml.probs(all wires)`, take element 0; `lightning.qubit` | `Run_QSVC_Legit.py:84-107`; `Run_QSVC_Mixed_Variance.py:83-107` | verified, static |
| Legacy authors already state the exact classical identity `(A @ B.T)**2` and use it in `Run_QSVC_Only.py` | `Run_QSVC_Only.py:105-109` (comment + implementation), `:172-177` (plot-only circuit) | verified, static |
| Conflicting variants exist: 4 qubits/16 features; 8 qubits; AngleEmbedding vs AmplitudeEmbedding in one script | `hybrid_QC_CNN_v2.py:78-93`; `archives/Old Runs/test_run_clean.py:387`; `Benchmark_Multiple_SOTA_extracted.py:270-278` vs `:537-541` | verified, static |
| `SVC(kernel="precomputed", probability=True)` is constructed without `random_state` in every legacy QSVC script | `Run_QSVC_Legit.py:156`, `Run_QSVC_Mixed_Variance.py:172`, `Run_QSVC_Only.py:115` | verified, static; reproducibility defect |
| API/version statements in §2 | PennyLane skill references (`simulator-pennylane.md`) + `quantum-computing` workflow 6 + static legacy usage | documentation-based; **unverified at runtime — smoke test pending exact install** |

**Skills used:** `master-scientist` (claim/evidence discipline, no-advantage guardrail),
`quantum-skills` (version-sensitive API discipline), `quantum-computing` (PennyLane hybrid
workflow), `deep-learning` (frozen-backbone transfer learning, seeding, dtype/device).

---

## 1. Module API — `experiments/models.py` (proposed)

All classes read their hyperparameters from `ExperimentConfig` (`experiments/config.py`) and
contain **no module-level qubit constants**. Signatures below are the contract the implementing
worker must satisfy.

```python
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from experiments.config import ExperimentConfig

# ---------------------------------------------------------------- classical blocks

class FrozenEfficientNetB0(nn.Module):
    """ImageNet EfficientNet-B0 trunk, frozen by default.

    Constructor args:
        image_size: int = 128          # recorded only; caller supplies tensors
        weights: str = "DEFAULT"       # torchvision weights enum name or None
        trainable: bool = False        # if False: requires_grad_(False) + eval() forever
    """
    def __init__(self, *, image_size: int = 128, weights: str = "DEFAULT",
                 trainable: bool = False) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) float32 -> (B, 1280) pooled features. Under no_grad when frozen."""


class ProjectL2(nn.Module):
    """1280 -> 1024 learned projection followed by explicit L2 normalisation.

    Constructor args:
        in_features: int = 1280, out_features: int = 1024,
        eps: float = 1e-8, bias: bool = True
    Invariant: out_features == 2 ** n_qubits of the consuming quantum layer.
    """
    def __init__(self, in_features: int = 1280, out_features: int = 1024, *,
                 eps: float = 1e-8, bias: bool = True) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_features) -> (B, out_features), rows have unit L2 norm."""


# ---------------------------------------------------------------- quantum blocks

class AmplitudeQNNLayer(nn.Module):
    """10-qubit amplitude-encoded, trainable, entangling QNN layer.

    Constructor args:
        n_qubits: int = 10, n_layers: int = 2,
        device_name: str = "default.qubit",
        interface: str = "torch", diff_method: str = "backprop",
        shots: int | None = None, seed: int = 42
    Weight shape passed to TorchLayer: {"weights": (n_layers, n_qubits, 3)}.
    """
    def __init__(self, *, n_qubits: int = 10, n_layers: int = 2,
                 device_name: str = "default.qubit", interface: str = "torch",
                 diff_method: str = "backprop", shots: int | None = None,
                 seed: int = 42) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2**n_qubits) -> (B, n_qubits) Pauli-Z expectation values in [-1, 1].
        Input must be moved to CPU by the caller; output is cast back to x.dtype/x.device.
        """


# ---------------------------------------------------------------- full torch models

class HybridAmplitudeQNN(nn.Module):
    """Canonical hybrid: frozen EfficientNet-B0 -> ProjectL2 -> QNN -> linear head.

    Constructor args:
        config: ExperimentConfig, *, device_name: str = "default.qubit",
        shots: int | None = None, seed: int | None = None  # None => config.seed
    Mask: extractor frozen; projection, QNN weights and head trainable.
    """
    def __init__(self, config: ExperimentConfig, *, device_name: str = "default.qubit",
                 shots: int | None = None, seed: int | None = None) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, image_size, image_size) -> (B, 2) raw logits (CrossEntropyLoss input)."""


class LegacyQNNBaseline(nn.Module):
    """Faithful, clearly labelled reproduction of Custom_Legacy_QNN_Upgraded.

    Constructor args: same as HybridAmplitudeQNN.
    Difference: no EfficientNet; Linear(3*128*128, 1024) -> L2 -> same QNN -> head.
    Kept only as an ablation/compatibility baseline; its first layer has ~50M parameters.
    """
    def __init__(self, config: ExperimentConfig, *, device_name: str = "default.qubit",
                 shots: int | None = None, seed: int | None = None) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 128, 128) -> (B, 2) raw logits."""


MODEL_REGISTRY = {
    "hybrid_amplitude_qnn": HybridAmplitudeQNN,
    "legacy_qnn_baseline": LegacyQNNBaseline,
}
def build_model(name: str, config: ExperimentConfig, **kwargs) -> nn.Module: ...
```

### 1.1 Kernel and QSVC API (numpy / sklearn side, CPU)

```python
class ClassicalKernel:
    """Exact classical oracle for the parameter-free amplitude feature map.

    k(A, B) = clip((A @ B.T) ** 2, 0, 1) for L2-normalised rows.
    """
    def __call__(self, A: np.ndarray, B: np.ndarray, *,
                 symmetric: bool = False) -> np.ndarray: ...


class AmplitudeQuantumKernel:
    """Reusable, analytic, PSD quantum kernel k(x,x') = P(0...0 | U_x'† U_x |0...0).

    Constructor args:
        n_qubits: int = 10, device_name: str = "lightning.qubit",
        shots: int | None = None, seed: int = 42,
        interface: str | None = None, batch_size: int = 32, n_jobs: int = 1
    """
    def __init__(self, *, n_qubits: int = 10, device_name: str = "lightning.qubit",
                 shots: int | None = None, seed: int = 42,
                 interface: str | None = None, batch_size: int = 32,
                 n_jobs: int = 1) -> None: ...
    def __call__(self, A: np.ndarray, B: np.ndarray, *,
                 symmetric: bool = False) -> np.ndarray:
        """(m, D) and (n, D) L2-normalised rows -> (m, n) kernel matrix, float64."""
    def diagonal(self, A: np.ndarray) -> np.ndarray:
        """Returns ones(m) analytically; never executes a circuit for the diagonal."""
    def nearest_psd(self, K: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """Optional Frobenius-nearest PSD repair for finite-shot matrices only."""
```

```python
class QSVCWrapper:
    """sklearn SVC(kernel='precomputed') over an explicitly computed kernel matrix.

    Constructor args:
        kernel: AmplitudeQuantumKernel | ClassicalKernel,
        C: float = 1.0, probability: bool = True, seed: int = 42
    Internal estimator: sklearn.svm.SVC(kernel="precomputed", C=C,
                          probability=probability, random_state=seed)  # MANDATORY
    """
    def __init__(self, kernel, *, C: float = 1.0, probability: bool = True,
                 seed: int = 42) -> None: ...
    def fit(self, K_train: np.ndarray, y: np.ndarray) -> "QSVCWrapper": ...
    def predict(self, K_test: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, K_test: np.ndarray) -> np.ndarray: ...

def extract_features(loader, extractor: nn.Module, projection: nn.Module, *,
                     device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Frozen, no_grad feature cache -> (N, 2**n_qubits) float64 CPU array + labels.
    The projection must come from a fixed, recorded checkpoint (no test-set tuning).
    """
```

**Design note.** `QSVCWrapper` consumes precomputed matrices rather than raw features so that
kernel computation, symmetrisation, PSD repair and caching are explicit and testable. This
matches the legacy `SVC(kernel="precomputed", ...)` contract.

---

## 2. Exact PennyLane API choices per model

All statements target the exact pinned `PennyLane==0.40.0` / `PennyLane-Lightning==0.40.0`
(numpy `<2.1`). They are documentation-based — documentation for the **target** pinned
version, not for a locally installed build — and are unverified at runtime: **smoke test
pending exact install**. They must be smoke-checked against the installed versions before
the model worker claims verification.

### 2.1 `AmplitudeQNNLayer` / `HybridAmplitudeQNN` / `LegacyQNNBaseline`

```python
import pennylane as qml

dev = qml.device("default.qubit", wires=n_qubits, shots=None, seed=seed)  # smoke test pending exact install: device(seed=...)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnn_circuit(features, weights):
    qml.AmplitudeEmbedding(features=features, wires=range(n_qubits),
                           normalize=True, pad_with=0.0)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

layer = qml.qnn.TorchLayer(qnn_circuit, {"weights": (n_layers, n_qubits, 3)})
```

| Choice | Value | Rationale |
|---|---|---|
| Device | `default.qubit` (CPU), `shots=None` | Native torch backprop, exact expectation values |
| Interface | `torch` | Required for end-to-end autograd through `TorchLayer` |
| `diff_method` | `backprop` | Only supported analytic path for `default.qubit` + torch; matches legacy |
| Weights | `(n_layers, n_qubits, 3)` = `(2, 10, 3)` = 60 params | Legacy shape (`Benchmark_Clean_Eval.py:162`) |
| Readout | 10 × `qml.expval(qml.PauliZ(i))` | Gives `(B, 10)` fed to `nn.Linear(10, 2)` |
| Batch handling | `TorchLayer` vectorises the QNode over the leading batch dimension | Do not hand-roll a batch loop |

### 2.2 `AmplitudeQuantumKernel`

```python
dev = qml.device("lightning.qubit", wires=n_qubits, shots=None, seed=seed)  # smoke test pending exact install: device(seed=...)

@qml.qnode(dev, interface=None)            # plain numpy in/out; no gradients needed
def kernel_circuit(x_i, x_j):
    qml.AmplitudeEmbedding(features=x_i, wires=range(n_qubits),
                           normalize=True, pad_with=0.0)
    qml.adjoint(qml.AmplitudeEmbedding)(features=x_j, wires=range(n_qubits),
                                        normalize=True, pad_with=0.0)
    return qml.probs(wires=range(n_qubits))

k_ij = float(np.clip(kernel_circuit(A[i], B[j])[0], 0.0, 1.0))  # P(|0...0>)
```

| Choice | Value | Rationale |
|---|---|---|
| Device | `lightning.qubit`, `shots=None` | Fast C++ analytic statevector; legacy kernel device |
| Interface | `None` (numpy) | Kernel has no trainable parameters; avoids torch/autograd overhead |
| `diff_method` | default (unused) | Never request `backprop` from Lightning; if gradients are ever added use `adjoint` |
| `shots` | `None` by default | Analytic result → exactly PSD, exactly symmetric, exactly unit diagonal |
| Readout | `qml.probs(wires=range(n))[0]` | Element 0 is the all-zeros basis state under either endian convention |
| Adjoint form | `qml.adjoint(qml.AmplitudeEmbedding)(...)` | Matches legacy; construct the op inside the QNode |

### 2.3 Version-sensitive notes for 0.40–0.45

1. `default.qubit` in this range is the modern device with native torch backprop; the old
   `default.qubit.legacy` alias is deprecated/removed across this range — **do not use it**.
2. `qml.AmplitudeEmbedding(features=..., wires=..., normalize=..., pad_with=...)` and
   `qml.StronglyEntanglingLayers(weights, wires)` signatures are stable across 0.40–0.45.
3. `qml.qnn.TorchLayer(circuit, weight_shapes)` is stable. Its default weight initialisation
   draws from numpy RNG: seed `np.random` **before** construction or pass an explicit
   `init_method` with a private `np.random.default_rng(seed)`.
4. PennyLane statevector math is float64. A `TorchLayer` output may arrive as float64 even
   when the classical head is float32: always cast
   `x_q = layer(x.cpu()).to(dtype=x.dtype, device=x.device)` before the head. The legacy code
   moved devices (`.to(x.device)`) but did **not** cast dtype (`Benchmark_Clean_Eval.py:182`) —
   do not copy that omission.
5. `default.qubit` and `lightning.qubit` execute on CPU. QNode inputs must be detached CPU
   tensors/arrays; GPU tensors passed directly will fail.
6. `device(seed=...)` (both `default.qubit` and `lightning.qubit`) is documented for the
   target 0.40.0 line; **smoke test pending exact install** for this kwarg. With `shots=None`
   the result is deterministic regardless, but set it anyway for finite-shot diagnostics.
7. With finite `shots`, `qml.probs` returns a **sample-estimated** distribution, not exact
   probabilities. Shot noise can make the kernel indefinite (§3.4).
8. The `numpy<2.1` cap must remain compatible with the selected `torch`/`torchvision` wheels;
   do not relax it unilaterally.
9. Exact pins `PennyLane==0.40.0` and `PennyLane-Lightning==0.40.0` are now recorded in both
   `requirements.txt` and `environment.yml` (§6.1). For full bit-for-bit provenance the run
   manifest must still record those exact versions plus the torch/torchvision and CUDA builds;
   smoke test pending exact install.

---

## 3. Mathematically correct, reusable quantum kernel

### 3.1 Definition and derivation

Let `D = 2**n_qubits = 1024` and let `x ∈ R^D` be an L2-normalised projected feature vector
(`‖x‖₂ = 1`). `AmplitudeEmbedding` realises a unitary `U_x` with
`U_x |0…0⟩ = |φ(x)⟩`, where `|φ(x)⟩ = x` (real amplitudes, `pad_with=0.0`). Then

```
k(x, x′) = ⟨0…0| U_{x′}† U_x |0…0⟩            (transition amplitude)
         = ⟨φ(x′) | φ(x)⟩                       (state overlap)
P(0…0 | U_{x′}† U_x |0…0⟩) = |k(x, x′)|²       (what qml.probs(...)[0] returns)
         = (x · x′)²  for unit-norm real x, x′
         = ((a · b) / (‖a‖ ‖b‖))²               for arbitrary unnormalised a, b
```

The endianness/wire-order convention is irrelevant: index 0 of `probs(wires=range(n))` is the
all-zeros basis state in every convention.

### 3.2 PSD proof sketch (general, not just the classical case)

For **any fixed** parameter-free encoding unitary `U_x` (amplitude only, or amplitude plus
fixed entanglers with frozen trainable weights), define `|Φ(x)⟩ = U_x|0…0⟩` and the lifted
feature vector `ψ(x) = vec(|Φ(x)⟩⟨Φ(x)|)`. Then

```
K_ij = P(0 | U_{x_j}† U_{x_i} | 0)
     = |⟨Φ(x_j)|Φ(x_i)⟩|²
     = ⟨ψ(x_i), ψ(x_j)⟩ .
```

Hence `K = Ψ Ψᵀ` with rows `Ψ_i = ψ(x_i)`, so for any `c ∈ R^m`,

```
cᵀ K c = Σ_ij c_i c_j ⟨ψ_i, ψ_j⟩ = ‖ Σ_i c_i ψ_i ‖² ≥ 0 .
```

Symmetry follows from the real inner product; positive semidefiniteness from the norm.
This is the standard squeezed/Veronese feature map argument. **PSD is a property of the
fixed feature map, not of the simulator.** Contract: the QSVC feature map is fixed (frozen
extractor + frozen projection checkpoint + no trainable kernel parameters).

### 3.3 Diagonal, symmetry, batching

- **Diagonal:** `k(x,x) = 1` analytically for any `‖x‖ > 0`. Set `K[i,i] = 1.0` explicitly
  without executing a circuit. This is exact, saves `m` circuit evaluations, and prevents
  finite-shot estimates below 1 from breaking the PSD argument.
- **Symmetry:** if `A is B` and `symmetric=True`, evaluate only `j ≥ i` and mirror:
  `K[j,i] = K[i,j]`. Assert `np.allclose(K, K.T, atol=1e-10)` afterwards. For `A ≠ B`
  compute all `m × n` pairs; no mirroring is valid.
- **Batching:** process rows in chunks of `batch_size` to bound memory; keep `n_jobs=1` by
  default for determinism. Parallelism (`n_jobs>1`) is optional and must record the joblib
  configuration; note that nested BLAS threading can oversubscribe cores.
- **Numerical guard:** clip each probability to `[0, 1]`; reject all-zero input vectors —
  amplitude embedding with `normalize=True` on a zero vector is undefined (see §6).
- **Caching:** key the kernel cache by (feature-checksum, `n_qubits`, `device_name`,
  `shots`, `seed`, `symmetric`) so train/test matrices are never recomputed with different
  conventions.

### 3.4 Seed / shot conventions

| Regime | Default | Consequence |
|---|---|---|
| Reporting / model fitting | `shots=None` (analytic) | Exact overlap, PSD, symmetric, unit diagonal; deterministic without RNG |
| Cost/noise diagnostics only | `shots=1024`, explicit `seed` | Sampled estimate; K can be slightly indefinite; symmetrise + `nearest_psd`, and label as shot-noise mitigation, not a fix |
| Diagonal under finite shots | forced to 1.0 | Prevents artificial `k(x,x) < 1` eigenvalues |
| Lightning seed | `seed=config.seed` | Required only for finite-shot runs; record with results |

Never mix analytic and finite-shot matrices inside one experiment or figure.

### 3.5 Explicit honesty statement (must appear in code docstrings and reports)

> The canonical QSVC feature map contains **no entangling gates and no trainable
> parameters**. Its circuit value is exactly
> `P(|0…0⟩) = ((x_i·x_j)/(‖x_i‖‖x_j‖))²`, which is computed classically in
> `Run_QSVC_Only.py:105-109` as `(A @ B.T)**2`. Running the PennyLane circuit therefore
> reproduces a classical kernel; it provides **no quantum advantage**. The quantum path is
> retained only as (a) a numerical cross-check that the simulator matches the closed form
> and (b) a source of circuit diagrams. Reported QSVC results should be produced with the
> classical oracle for speed and determinism, accompanied by a passing equivalence test.
> The QNN models *do* add entanglement and trainable weights, so they are not covered by
> this identity, but at 10 simulated qubits they remain classically simulable and likewise
> warrant **no advantage claim**.

---

## 4. Eliminating the conflicting 4/8-qubit variants

### 4.1 Single source of truth

1. `ExperimentConfig` is the only place qubit count lives: `n_qubits=10`,
   `feature_dim=1024`, `qnn_layers=2`, `seed=42`. `config.validate()` already enforces
   `feature_dim == 2**n_qubits` (`experiments/config.py:71-75`); keep and extend it with
   `qnn_layers > 0` and `n_qubits in {4, 8, 10}`.
2. Every constructor takes `config` (or receives `n_qubits` from it). Add a test that scans
   `experiments/` and fails on any literal `n_qubits = <digits>` or `wires=<digits>`
   assignment — the exact anti-pattern in `hybrid_QC_CNN_v2.py:78` and the Old Runs scripts.
3. Archives stay read-only evidence; no legacy constants, scripts, or checkpoints are copied
   into the new package.

### 4.2 Legacy variant inventory and disposition

| Legacy artifact | Qubits / features | Embedding / device | Disposition |
|---|---|---|---|
| `Benchmark_Clean_Eval.py:152-162,199-217` | 10 / 1024 | Amplitude + SEL, `default.qubit`, backprop | Canonical QNN reference |
| `Run_QSVC_Legit.py:83-107` | 10 / 1024 | Amplitude kernel, `lightning.qubit`, analytic | Canonical kernel reference |
| `Run_QSVC_Only.py:105-109,172-177` | 10 / 1024 | Classical `(A@B.T)**2`; plot-only circuit | Equivalence reference |
| `Run_QSVC_Mixed_Variance.py`, `Run_QSVC_Legit - Copy.py` | 10 / 1024 | Same kernel + mixed train/test set | **Reject**: OOD data enters training → leakage; keep archived only |
| `hybrid_QC_CNN_v2.py:78-93` | 4 / 16 | Amplitude + SEL, `default.qubit` | Ablation `q4` via parameter only |
| `archives/Old Runs/test_run_clean.py:387` (+ notebooks) | 8 / 256 | Amplitude + SEL | Ablation `q8` via parameter only |
| `Benchmark_Multiple_SOTA_extracted.py:270-278` | 10 / **10** | **AngleEmbedding** + SEL | Deprecated; different model (takes `n`, not `2**n` features) |
| `Benchmark_Multiple_SOTA_extracted.py:537-541` | 10 / 1024 | AmplitudeEmbedding + SEL | Canonical embedding |

The AngleEmbedding and AmplitudeEmbedding circuits coexist in one legacy script; this is the
root of the "conflicting embedding" audit finding. Resolution: **AmplitudeEmbedding is the
only canonical embedding** because `feature_dim = 2**n_qubits` is validated by config.

### 4.3 If ablations are required

- Parameterise, never fork: `n_qubits ∈ {4, 8, 10}` with `out_features = 2**n_qubits`
  (`16 / 256 / 1024`) on `ProjectL2`; same `AmplitudeQNNLayer` code, same circuit body.
- Encode the ablation in run metadata and filenames:
  `q{n}_emb{amplitude|angle}_L{qnn_layers}` (e.g. `q4_embamplitude_L2_ablation`).
  The `embedding=angle` ablation changes the input dimension to `n_qubits` features and must
  use its own projection width; record `ablation=True` and exclude it from headline tables.
- Headline results use **only** `q10_embamplitude_L2`. Ablations live in a separate results
  table with explicit labels; they are never averaged with canonical runs.

---

## 5. Determinism, dtype/device placement, and test plan

### 5.1 Seeding (call before constructing any model or quantum device)

```python
def seed_everything(seed: int) -> None:
    import random, numpy as np, torch, pennylane as qml
    random.seed(seed)
    np.random.seed(seed)                 # also seeds TorchLayer's default init
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    qml.numpy.random.seed(seed)  # smoke test pending exact install (target pin 0.40.0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Run-level seeds derive deterministically from config:
`run_seed = config.seed + run_index * config.seed_offset` (legacy used ad-hoc
`run_idx*42`, `run_idx*123`, `run_idx*456` in `Run_QSVC_Legit.py:134,146` — replace with the
single offset rule). Also pass `random_state=seed` to every `train_test_split` and to
`SVC` (see §0 / §6).

### 5.2 Placement contract

| Stage | Device | dtype | Notes |
|---|---|---|---|
| Input images | GPU if available | float32 NCHW | ImageNet normalisation from config |
| `FrozenEfficientNetB0` | GPU | float32 | `eval()`, `requires_grad_(False)`, `no_grad` |
| `ProjectL2` | GPU | float32 | Trainable; output rows unit-norm |
| QNode input | **CPU** | float32 (cast explicitly) | `x.detach().cpu()`; requires exactly `2**n_qubits` features |
| QNode / `TorchLayer` | **CPU only** | internal float64 | No GPU execution; no grad through frozen stages |
| QNN output | back to model device | **cast to x.dtype** | `layer(x.cpu()).to(dtype=x.dtype, device=x.device)` |
| Linear head + loss | GPU | float32 | CrossEntropyLoss on raw logits |
| Kernel inputs | CPU | float64 numpy | From a frozen feature cache |
| Kernel / SVC | CPU | float64 / sklearn | `SVC(kernel="precomputed", random_state=seed)` |

The GPU→CPU→GPU hop is unavoidable with CPU quantum simulators; profile it rather than
hiding it. Cache kernel matrices once and reuse them.

### 5.3 Test plan (to be implemented under `tests/` by the owning worker, run only by mother)

| Test | Assertion | Tolerance / note |
|---|---|---|
| `test_extractor_shape` | `(B,3,128,128) -> (B,1280)` | B = 4; backbone frozen |
| `test_projection_shape_and_norm` | `(B,1280) -> (B,1024)`, `‖row‖₂ = 1` | `atol=1e-6` |
| `test_qnn_layer_shape` | `(B,1024) -> (B,10)`, values in `[-1,1]` | B = 4 |
| `test_full_model_logits` | `(B,3,128,128) -> (B,2)`, finite, no NaN grads | CPU |
| `test_legacy_baseline_shape` | `(B,3,128,128) -> (B,2)` | documents 49k→1024 first layer |
| `test_kernel_shapes` | `(m,D),(n,D) -> (m,n)`; symmetric flag | m=5, n=7 and m=n=5 |
| `test_kernel_diagonal` | `np.diag(K) == 1.0` exactly | analytic shots |
| `test_kernel_symmetry` | `np.allclose(K, K.T)` | `atol=1e-10`, analytic |
| `test_kernel_psd` | `np.linalg.eigvalsh(K).min() >= -1e-8` | analytic; m=10 |
| `test_classical_equivalence` | `K_quantum ≈ clip((A@B.T)**2,0,1)` on unit-norm random and cached features | `atol=1e-6`, `shots=None` |
| `test_seed_determinism` | two `build_model(..., seed=42)` have identical weights; repeated kernel calls identical | exact |
| `test_config_validation` | `feature_dim != 2**n_qubits` and non-canonical `n_qubits` flagged | uses `config.validate()` |
| `test_no_hardcoded_qubits` | no `n_qubits = <int>` / `wires=<int>` literals in `experiments/` | source scan |
| `test_ablation_dims` | `n_qubits ∈ {4,8,10}` → widths 16/256/1024, correct shapes | parameterised |
| `test_device_dtype` | CPU and CUDA logits within `1e-4`; output dtype == head dtype | skip if no CUDA |
| `test_svc_random_state` | `QSVCWrapper.random_state == config.seed`; repeated fit/predict identical | guards legacy defect |

---

## 6. Risks and unresolved questions

1. **RESOLVED — version pinning (Mother ruling, QC-REBUILD-2026-11-22).** The previously
   reported range-pin blocker is closed. Mother has pinned `PennyLane==0.40.0` and
   `PennyLane-Lightning==0.40.0` exactly in both `requirements.txt` and `environment.yml`,
   matching the upstream notebook's documented PennyLane 0.40 line. This worker does not
   edit those files; the residual exposure is recorded in item 2.
2. **Residual risk — smoke test pending exact install.** No runtime verification was
   performed by this worker (read-only mandate) and the pinned environment is not yet
   installed. Until `PennyLane==0.40.0` / `PennyLane-Lightning==0.40.0` are installed and the
   §5 smoke tests pass, **all API statements in §2 remain documentation-based and unverified
   at runtime**, and no result may be called verified. Status label: `smoke test pending
   exact install`.
3. **Finite-shot kernel can be indefinite.** Default to `shots=None`; any finite-shot run must
   be labelled stochastic and use symmetrisation + optional nearest-PSD repair.
4. **Zero-vector features.** `AmplitudeEmbedding(normalize=True)` on an all-zero vector is
   undefined/NaN. `ProjectL2` uses `eps=1e-8`, but add an explicit `‖x‖ > 0` assertion and a
   test; never feed an untrained/zero projection into the kernel.
5. **`SVC(probability=True)` without `random_state`** in all legacy QSVC scripts
   (`Run_QSVC_Legit.py:156`, `Run_QSVC_Mixed_Variance.py:172`, `Run_QSVC_Only.py:115`) makes
   `predict_proba` fold-shuffle-dependent. **Mandatory implementer fix (retained by Mother's
   ruling):** `SVC(..., random_state=config.seed)` in `QSVCWrapper` (§1.1); covered by
   `test_svc_random_state` (§5.3).
6. **Leakage in mixed-variance legacy runs (mandatory exclusion, retained by Mother's
   ruling).** `Run_QSVC_Mixed_Variance.py:141-180` puts DatasetNinja OOD samples into the SVC
   training set. These runs are not admissible evidence; they must stay archived and be
   excluded as canonical results and figure inputs.
7. **Legacy baseline size.** `Linear(3*128*128, 1024)` ≈ 50M trainable parameters
   (`Benchmark_Clean_Eval.py:188`) — keep it as a labelled baseline only; do not let it dominate
   parameter-count figures without a note.
8. **GPU↔CPU transfer cost** for every QNN batch at 10 qubits; measure FPS separately from
   accuracy and report honestly (the legacy QSVC rows have `FPS = NaN`,
   `Run_QSVC_Only.py:143-144`).
9. **Open question:** whether to report the analytic QSVC at all given exact classical
   equivalence. Recommendation: report it with the explicit caveat from §3.5, and keep the
   classical oracle as the primary computation.
10. **Open question:** whether `embedding=angle` ablations are wanted. If yes they require a
    separate `n_qubits`-wide projection and must never be averaged with canonical runs.

---

## Deliverable summary (master-scientist format)

- **Skills loaded:** `master-scientist`, `quantum-skills`, `quantum-computing`, `deep-learning`.
- **Claim-to-evidence table:** §0. Key claims are static-inspection verified; PennyLane API
  details are documentation-based and explicitly unverified at runtime — smoke test pending
  exact install. Version-pin blocker RESOLVED by Mother's ruling (§6.1); the two mandatory
  implementer fixes are retained unchanged in §6.5–6.6.
- **Exact files changed:** `docs/research/QC-D03_quantum_models.md` (this file only; no code,
  builds, commits, or environment files).
- **Next bottom-up step:** Mother installs `PennyLane==0.40.0` /
  `PennyLane-Lightning==0.40.0` exactly; model worker then implements `experiments/models.py`
  to the §1 API and runs the §5 smoke tests. Acceptance criterion: all §5 shape/symmetry/PSD/
  equivalence/determinism tests pass under the exact install, and `experiments/config.py`
  remains the sole source of qubit/feature/layer/seed settings.
