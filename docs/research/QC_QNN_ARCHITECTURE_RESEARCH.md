# QNN architecture research - why our first QNN was the wrong model (2026-11-24)

> Method: web search + direct reading of the source pages during this session.
> This document is public research, not private-paper content. No dataset or
> training run was performed for it.

## 1. Verdict on our current QNN/hybrid

Our QNN-D01 model is: frozen features → **PCA to 16 dimensions** → amplitude
embedding into 4 qubits → 2 `StronglyEntanglingLayers` → Z expectations → linear
head. The HYBRID-D01 model adds a trainable classical branch but keeps the same
quantum branch.

This matches almost exactly the **weak baseline** that the current literature
warns against:

- Wang et al., *Quantum Machine Intelligence* 7:35 (2025),
  `https://link.springer.com/article/10.1007/s42484-025-00247-7`: "the baseline
  model, which adopts amplitude embedding as the data encoding method, cannot
  efficiently capture the features in such a high-dimensional image dataset."
- Same paper, design principle: "Using a classical model, such as a neural
  network backbone or PCA to reduce the dimension of the data ... obscures the
  real effectiveness of the quantum machine learning model, so we should
  minimise classical preprocessing as much as possible."
- Our measured result is consistent: QNN accuracy `0.5045 ± 0.0384`, AUC CI
  `[0.480, 0.571]` (chance); hybrid `0.7665 ± 0.0086` — the classical branch
  carries nearly all the performance. So the earlier negative result is a
  **model-choice result**, not evidence that QNNs in general cannot work.

## 2. Architectures that the literature actually supports

### 2.1 Data re-uploading (highest-priority fix)
Pérez-Salinas et al., *Quantum* 4:226 (2020),
`https://quantum-journal.org/papers/q-2020-02-06-226/`.

Instead of one encoding `U(x)` followed by one variational layer `V(θ)`, the
data is re-inserted L times with **different** trainable layers:

```
|Ψ(x;ω)> = Π_{i=1..L} [ V(ω_i) U_φ(x) ] |0>^n
```

This mirrors classical deep learning (reuse of inputs across layers) and has a
proven function-approximation advantage (Yu et al. 2023, cited in Wang et al.).
Empirical impact:

- HQCNN–REGA (MDPI *Electronics* 14:185, 2025),
  `https://www.mdpi.com/2079-3197/13/8/185`: adding data re-uploading raised
  MNIST 97.9→99.0% and CIFAR-100 85.0→96.9%, with fewer epochs to converge.
- Wang et al. (2025): Hamiltonian embedding + re-uploading beat the baseline
  QCNN by **over 40%** on the MNIST test set.

### 2.2 Hamiltonian (matrix) embedding
Wang et al. (2025): Hermitianise the input matrix `H_M = (M+Mᵀ)/2` and apply
`W(t;M)=exp(-i H_M t/2)` with a trainable `t`. The Taylor expansion applies a
polynomial nonlinearity to the whole input, giving "cheaper" nonlinearity than
single-qubit rotation gates and preserving two-dimensional structure. Their
model beat QCNN on MNIST, FashionMNIST, digits and a CT dataset.

### 2.3 QCNN with arbitrary input dimensions
Frontiers in Physics 13:1529188 (2025),
`https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2025.1529188/full`:
single-ancilla padding and skip-pooling let QCNNs accept arbitrary feature
widths. Binary MNIST results: 91.75 ± 2.57% (labels 0/1), 94.59 ± 0.62%
(labels 5/6). QCNNs are highlighted as **not suffering barren plateaus**, a
likely cause of shallow-QNN failures.

Efficient QCNN (arXiv:2505.05957,
`https://arxiv.org/abs/2505.05957`): 49-qubit primitive QCNN processes 28×28
MNIST directly and reached 96.08% on IBM Heron r2. Not reproducible on our CPU,
but confirms the architecture class.

### 2.4 Encoding comparison on utility-scale hardware
arXiv:2504.10595 (`https://arxiv.org/abs/2504.10595`), four-class scene data:
block amplitude embedding (BAE) was best among quantum models (0.94 with
224 qubits), pairwise encoding (PAE) 0.88, amplitude (AAE) 0.81–0.73. Their
circuits apply a small linear layer to **several qubit expectation values**,
then softmax — closer to our head, but with far richer encoding.

### 2.5 Six design principles (Wang et al. 2025)
1. Do not make speedup the first goal; optimise predictive metrics.
2. Respect the intrinsic structure of the data.
3. **Minimise classical preprocessing** (PCA/backbone hides the QML effect).
4. Avoid direct "quantisation" of classical operations.
5. Quantum measurements are not classical activations — account for that.
6. Hardware-aware design (qubit count, depth, noise).

## 3. Concrete corrected plan

Replace QNN-D01's single-shot amplitude embedding with **two proper model
families**, both evaluated on the exact same cached 70:20:10 splits, n=10 seeds,
sealed TEST, and paired against the committed classical poly-2 arm:

- **A — Data re-uploading QNN**: angle (rotation) encoding into n qubits,
  repeated L times interleaved with independent trainable entangling layers;
  several Z expectation values → small linear head. Use a mild feature
  reduction (e.g. PCA to ~n·L features) only where needed for CPU cost, and
  record exactly how much classical preprocessing remains.
- **B — QCNN**: convolution + pooling ansatz with single-ancilla padding /
  skip-pooling for arbitrary feature width, which the literature reports as
  trainable without barren plateaus.

Both must report their projection/encoding budget and compare against BOTH the
classical poly-2 baseline and the old amplitude-embedding QNN. A quantum win is
still not claimable without matched budgets, power ≥ 0.8 and Holm-corrected
paired evidence.

## 4. Caveats

- All cited results are from other datasets (MNIST/FashionMNIST/CIFAR/scenes),
  not weed imagery; they justify architecture choice, not performance numbers.
- Several models need 16–224 qubits and are not simulable here on CPU; our
  implementations will be small-qubit analogues and must be labelled as such.
- No quantum advantage is asserted by this research note.
