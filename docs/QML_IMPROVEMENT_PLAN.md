# QML pipeline — prioritized improvement plan

**Audience:** Mother + the next implementation swarm (QML-D04+).
**Basis:** accepted specs QC-D01 (claim/evidence matrix), QC-D03 (quantum models),
QC-D04 (experiments/statistics/figures) and the implemented contract from
QML-D01/D02 (`QMLConfig`, `train`/`evaluate`/`seed_everything`,
`evaluate_qsvc`/`compare_models`) plus `qmlpipe/ablate.py` and
`qmlpipe/model_card.py`.

## 0. Non-negotiable guardrails (from `master-scientist`)

These are correctness rules, not stylistic preferences. A step is not accepted
if it violates one of them.

1. **No quantum-advantage claim without a matched classical control.** Every
   quantum arm must be paired with a classical baseline of the same features,
   split, seed and metric suite. The amplitude feature map has zero entangling
   gates and zero trainable parameters, so its circuit value is exactly the
   classical squared dot product `(A @ B.T)**2` (QC-D01 C5; QC-D03 §3.5).
   Calling it "quantum" is a description of the implementation, not evidence of
   advantage. Entangled QNNs at 4–10 *simulated* qubits are likewise classically
   simulable — no speedup claim.
2. **Evidence-based claims only.** Every reported number must resolve to a run
   artifact with `n`, spread, 95% CI and a recorded seed. Theory, static
   inspection and expectation are labelled as such and never written into a
   results table. Missing values render `not measured` (see `build_model_card`);
   they are never imputed or copied from the legacy paper.
3. **Leakage is a correctness bug.** Group by source photograph and content
   hash before splitting (QC-D04 §2.0). Any result produced on a leaked split
   is invalid and must be regenerated or deleted — not caveated.
4. **Bottom-up order.** Environment → data → metrics → models → training →
   experiments → reporting → manuscript. Do not patch prose to match broken
   code, or vice versa.
5. **No private-material transmission.** All paper content stays under the
   gitignored `.paper/` directory. No manuscript text or data is sent to
   external services.
6. **Reproducibility.** Exact pins (`PennyLane==0.40.0`,
   `PennyLane-Lightning==0.40.0`, numpy cap), a hash-pinned lockfile, recorded
   seeds and a resolved environment manifest accompany every result
   (`reproducible-environments` skill).

## 1. Prioritized roadmap

Work top to bottom. A step may start only when the preceding step's acceptance
criterion is met and Mother's central verifier is green.

### P0 — Statistical power and honest baseline reporting
**Why first.** The current evidence base is mostly single-run or n=10 on a
50-image test set; QC-D04 §5 shows the ~3% run-to-run spread is largely test-set
sampling noise. No conclusion is defensible until the measurement process is
fixed.

- Re-run every headline arm with the pre-registered seed map (≥5 seeds for
  neural models; the ruled 10 QSVC runs with the 100/50 mixed protocol and the
  no-mixing control).
- Emit `n`, mean, sample `std` (ddof=1), percentile-bootstrap 95% CI and the
  full per-run vector through `experiments.metrics.summarize_runs`, exactly as
  `qmlpipe.ablate.summarize_ablation` does.
- Apply QC-D04 §2.4 three-part decision rule (Holm-adjusted p < 0.05 **and**
  CI excludes 0 **and** power ≥ 0.80); otherwise label "descriptive /
  inconclusive". Never reuse the unsupported legacy p-values 0.0013 / 0.0009.

**Acceptance criterion:** every reported metric in `metrics.json` carries
`n`, `mean`, `std`, `ci95_low/high`, `boot_seed` and a per-run vector; each
pre-registered pair has raw p, adjusted p, effect size, difference CI and power;
`not measured` appears wherever an artifact is absent.

### P1 — Classical baselines (hard gate for any advantage language)
- Add `svc_poly2_classical` computing `(A @ B.T)**2` on the same L2-normalized
  features as the quantum-kernel arm, plus RBF and linear SVC controls and the
  no-mixing control (QC-D04 E2 arms 2–4).
- For the QNN arms, add a parameter-matched classical MLP on the same feature
  cache with the same optimizer and seed budget.
- Produce a single side-by-side table with identical splits, seeds and metrics.

**Acceptance criterion:** a quantum arm is never mentioned in the same sentence
as "better", "advantage" or "outperforms" unless its matched classical row and
the QC-D04 §2.4 test result are in the same artifact. `build_model_card` renders
the no-advantage limitation paragraph verbatim.

### P2 — Data scaling and leak-free expansion
- Keep MD5/content deduplication and photo-level `group_id` splits (QC-D01 P1,
  QC-D04 §2.0). Expand the adaptation and OOD subsets only by whole source
  photographs, never by additional crops from a photographed scene already in
  train.
- Persist per-run `sample_id`, `group_id`, `split` and hash in the manifest;
  assert train/adaptation/test hash *and* group sets are pairwise disjoint.
- Record the exact subset size in every run summary; do not merge a larger
  subset into an older, smaller result table.

**Acceptance criterion:** an automated leakage assertion passes on the expanded
manifest, and `n_train + n_val + n_test == n_manifest` with disjoint hash/group
sets. Any expansion that fails this check is reverted.

### P3 — Encoding and ansatz ablations
- Use `qmlpipe.ablate.ablation_grid` as the single source of truth: bounded
  72-config full factorial over qubits {4,8,10}, layers {1,2,3}, embedding
  {amplitude, angle}, ansatz {strongly_entangling, basic_entangler} and
  depolarizing noise {None, 0.01}.
- Run a **staged** subset first via `max_runs` (one-factor-at-a-time around the
  canonical `q10 / amplitude / 2 layers / strongly_entangling / no-noise`
  configuration) before paying for the full factorial.
- Keep `feature_dim = 2**n_qubits`; angle-embedding arms need a separate
  `n_qubits`-wide projection and are excluded from headline tables
  (QC-D03 §4.2–4.3).

**Acceptance criterion:** grid count equals 72 and never exceeds
`MAX_GRID_SIZE`; every generated config passes `QMLConfig.validate()`;
`summarize_ablation` reports truthful `n_success`/`n_failed` and no failed run
contributes a metric; ablation arms are labelled and never averaged with the
canonical arm.

### P4 — Optimizer and gradient choice
- Compare `backprop` on `default.qubit` (exact, CPU analytic) against
  `parameter-shift` and `SPSA` on the same data/seed budget. Record gradient
  calls, wall time and final metric vector per method.
- Keep `interface="torch"` + `TorchLayer` for backprop; use the PennyLane
  gradient API only after a smoke test under the exact 0.40.0 pin
  (QC-D03 §2.1–2.3 — all API statements there are documentation-based until
  that install exists).
- SPSA is stochastic: fix its seed, record it, and report spread over seeds —
  never a single lucky trajectory.

**Acceptance criterion:** a gradient-method comparison table with identical
split/seeds, a stated gradient-call budget, and n/mean/std/CI per method; any
target-version API detail not smoke-tested stays labelled "smoke test pending
exact install".

### P5 — Noise and shot studies
- Implement the seeded Gaussian corruption of QC-D04 E3 on test tensors only,
  with the exact σ grid `{0, 0.05, 0.1, 0.15, 0.2, 0.3}`, recorded clipping
  policy and 3 noise draws per `(run, σ, sample)`; σ=0 reuses clean predictions.
- Separately study depolarizing noise via the `noise_depolarizing` ablation axis
  (None vs 0.01) and finite shots (`shots=None` analytic vs finite-shot
  sampled). Finite-shot kernel matrices must be symmetrised, diagonal forced to
  1.0 and PSD repair labelled as mitigation, not a fix (QC-D03 §3.3–3.4).
- Never mix analytic and finite-shot matrices in one figure or summary.

**Acceptance criterion:** paired clean/noisy accuracy per seed with a paired CI;
the same seed reproduces the identical corrupted tensor; shot/noise regime is
recorded in provenance for every run; no indefinite kernel is reported as a
validated result.

### P6 — Calibration and decision thresholds
- Only for arms with genuine scores: Brier score, 10-bin ECE and a reliability
  diagram from `experiments.metrics.calibration_summary`, with CIs and
  disclosed within-run dependence at small n.
- Mark SVC `probability=True` scores as internal Platt scaling and neural
  softmax as uncalibrated. No probabilities are synthesized from hard labels;
  absent scores render `status: N/A` (QC-D04 E5).

**Acceptance criterion:** every scored arm has Brier/ECE with provenance; every
unscored arm has an explicit `N/A` reason; no threshold tuned on the test set.

### P7 — Hardware readiness (deferred, gated)
- Only after P0–P6 are green: transpile to a declared coupling map and basis
  gate set, record backend, shots, queue/time and calibration date, and compare
  hardware against the *same* classical control and noise model.
- Resource misrepresentation is prohibited: statevector memory at 10 qubits is
  small, but hardware results carry drift and shot noise — report them as
  observed hardware measurements, never as a speedup demonstration.

**Acceptance criterion:** a hardware proposal with an explicit backend, shots,
basis gates, cost estimate and matched classical control is approved before any
job is submitted; no submission occurs without Mother's authorization.

## 2. Definition of done for the current deliverables

| File | Done when |
| --- | --- |
| `qmlpipe/ablate.py` | grid is a bounded 72-config factorial; runs are deterministic; failures record type/message with empty metrics; summaries use `mean_std`/`bootstrap_ci`/`summarize_runs` with n/std/CI. |
| `qmlpipe/model_card.py` | card has intended use, provenance, architecture, metrics (n/spread/CI), limitations (no advantage, CPU simulation, small subset) and ethics; missing keys render `not measured`; no invented numbers. |
| `docs/QML_IMPROVEMENT_PLAN.md` | this document; every step has an acceptance criterion and the guardrails are explicit. |
| `tests/test_qmlpipe_ablate.py` | synthetic-only; grid bounds, deterministic cloning, failure recording, summary math and card honesty are covered; quantum-dependent tests skip cleanly without PennyLane. |

## 3. Evidence discipline for reporting (recap)

- Figures and tables read from `results/metrics.json` and raw artifacts only;
  no hardcoded numeric arrays (QC-D04 §4).
- Captions are generated from computed numbers; a changed mean changes the
  caption, and a missing number reads `not measured`.
- `master-scientist` classification: mark each claim `verified`,
  `partially supported` or `unsupported` against a named file/line before it
  enters any manuscript. Delete or label unsupported results; never fill a
  table with a plausible-looking value.
