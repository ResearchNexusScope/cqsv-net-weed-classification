# Hardware options for multi-qubit QNN runs (2026-11-24)

> Research from public sources during this session. No purchase or cloud account
> was created. All sizes/noises are cited to the source; local machine facts
> were measured with `Get-CimInstance` on 2026-11-24.

## 1. What this machine can actually do (measured)

| Property | Measured value |
|---|---|
| RAM | **7.8 GB** |
| GPU | Intel Iris Xe, 2 GB shared; **no NVIDIA CUDA**, no `nvidia-smi` |
| PennyLane | 0.40.0, CPU `lightning.qubit` only (`default.qubit`, `lightning.qubit`) |

Statevector memory is `2^n × 16 bytes` (complex128):

| Qubits | State | Practical note |
|---:|---:|---|
| 20 | 16 MB | fine here |
| 24 | 256 MB | fine for forward-only; gradient copies add ~2–4× |
| 26 | 1 GB | tight with 7.8 GB RAM and autograd copies |
| 28 | 4 GB | likely OOM / unusably slow here |
| 30 | 16 GB | impossible on this machine |

The 2026 scaling study (arXiv:2601.09951,
`https://arxiv.org/html/2601.09951`) reports CPU runtimes of 479 s at 24 qubits
and 1425 s at 26 qubits for a single VQE loop, versus 5.87 s and 17.71 s on one
H100 — a 3.5×–81.5× CPU/GPU ratio. That is why our proper architectures were
tested at 3–4 qubits: it is a hardware ceiling, not a modelling preference.

## 2. Tier 1 — free / near-free cloud (recommended first step)

### AWS Braket simulators
Sources: `https://aws.amazon.com/braket/getting-started/`,
`https://docs.aws.amazon.com/braket/latest/developerguide/choose-a-simulator.html`,
`https://aws.amazon.com/braket/pricing/`.

| Simulator | Max qubits | Price | Use |
|---|---:|---|---|
| Local `braket_sv` | ~25 (depends on host RAM) | free | prototyping |
| **SV1** (state vector) | **34** | **$0.075/min**, 1 free hour/month for first 12 months; 3 s minimum | general large circuits, parallel gradients |
| DM1 (density matrix) | 16–17 | $0.075/min | noise studies |
| **TN1** (tensor network) | **50** (low local entanglement) | $0.275/min | wide/shallow circuits |

AWS guidance: local below 18 qubits; 18–24 either; **>24 use on-demand**. A
34-qubit dense depth-34 SV1 task takes roughly 1–2 h.

Reuse our existing PennyLane models with one device change
(`https://www.pennylane.ai/devices/braket-aws-qubit`):

```bash
pip install amazon-braket-pennylane-plugin
```
```python
import pennylane as qml
dev = qml.device("braket.aws.qubit",
                 device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                 wires=16, shots=0, parallel=True)
# for adjoint gradient on SV1, declare diff_method="device" on the QNode
```

`parallel=True` runs gradient circuits concurrently — important for 100–500
training iterations.

### IBM Quantum Open Plan (real hardware, free)
Sources: `https://quantum.cloud.ibm.com/docs/en/ibm-quantum-platform/plans-overview`,
`https://www.ibm.com/quantum/products`.

- Free Open Plan: **~10 minutes of QPU runtime per 28-day window** (RU-based
  since Nov 2026); active users could opt into an extra 180 minutes over 12
  months (promotion announced 2026-03-16).
- Backends include Eagle r3 (127 qubits) and Heron (133 qubits).
- Queue times on popular free backends can be hours to days.
- Paid: Pay-As-You-Go ~$96/min (IBM product page; another 2026 comparison quotes
  ~$1.60/s — **verify current pricing before budgeting**), Flex from $72/min,
  Premium from $48/min.

PennyLane-Qiskit plugin (`https://docs.pennylane.ai/projects/qiskit/en/stable/`
and `https://github.com/PennyLaneAI/pennylane-qiskit`) exposes `qiskit.aer`,
`qiskit.basicsim` and `qiskit.remote`; `qiskit.remote` accepts any IBM backend.
Caveat: the plugin repo states **Python ≥3.12**, while our QML environment is
Python 3.11 — a separate env would be needed.

Real hardware is best used for **final noisy evaluation**, not for hundreds of
training iterations: parameter-shift gradients multiply circuit executions by
the parameter count, and shots/queue costs dominate.

## 3. Tier 2 — local NVIDIA GPU workstation

`lightning.gpu` uses NVIDIA cuQuantum; requires **SM 7.0+ NVIDIA GPU, Linux
x86-64/ARM64, CUDA 12** (`https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html`,
`https://pennylane.ai/devices/lightning-gpu`).

- Recommended for **20+ qubits**; parallel adjoint differentiation.
- Measured on one H100 80 GB: up to **29 qubits** with gradients; 30 qubits OOM
  (arXiv:2601.09951). Estimate is ~4× the statevector size for adjoint copies.
- Forward-only (no gradients, `diff_method=None`) can reach ~33 qubits with
  ~128 GB GPU memory (PennyLane discussion).
- Memory tuning: `batch_obs=True` / `batch_obs=1` divides observables across
  GPUs; `mpi=True` distributes the statevector.

Indicative hardware tiers:

| GPU VRAM | Approx. qubits with adjoint gradients |
|---:|---:|
| 24 GB (RTX 3090/4090) | ~26–27 |
| 48 GB (A6000 / RTX 6000 Ada) | ~28 |
| 80 GB (A100/H100) | **29 measured** (30 OOM) |

A used RTX 3090/4090 + Linux is the cheapest route to ~26 qubits locally, but
compare against SV1 first: $0.075/min means hundreds of hours of simulation for
the price of a mid-range GPU.

## 4. Tier 3 — HPC / multi-GPU (for 30–41 qubits)

`lightning.gpu` supports CUDA-aware MPI with power-of-two GPU counts
(PennyLane blog, `https://pennylane.ai/blog/2024/03/hpc-4-u-and-me/`):

- 16 × A100 40 GB → ~32 qubits.
- 500+ GPUs on Perlmutter (DOE) → **41 qubits** demonstrated.
- Scaling is near-linear beyond ~28 qubits; overheads dominate at small sizes.

Access route: university/cluster allocation. The 2026 study used an AMD EPYC
9654 + 4 × H100 node with 1.5 TB RAM — that class of machine is what the
published 16–49-qubit QML results rely on.

## 5. What this means for our experiment

- The published re-uploading/QCNN gains (5–49 qubits, 100–500 iterations) are
  **not reachable on this machine**. We correctly labelled our n=10 results as
  resource-limited negatives.
- Cheapest credible next test: **AWS Braket SV1 at 16–24 qubits**, reusing
  `qmlpipe/reupload_qnn.py` and `qmlpipe/fresh_qcnn.py` with a device swap,
  `parallel=True`, `diff_method="device"`, and the 1 free hour/month. Keep the
  same 70:20:10 splits, n=10 seeds, matched classical controls and paired gates.
- For a noise/robustness study, use DM1 (≤17 qubits) or IBM Open Plan real QPU
  for **final evaluation only**, with the classical paired control retained.
- More qubits alone do not prove QML value: barren plateaus, encoding choice and
  classical preprocessing still matter, and every claim needs the matched
  classical baseline and statistical gates already implemented in
  `experiments/fresh_protocol.py`.

## 6. Cost/claim caveats

- Prices and plan rules change; verify on the provider pages before committing.
- IBM pricing sources in this session disagree ($96/min vs ~$1.60/s) — confirm.
- No quantum advantage, speedup, or hardware claim is made by this document.
