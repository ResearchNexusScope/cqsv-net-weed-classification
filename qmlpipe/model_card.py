"""Generate an evidence-only Markdown model card for a QML ablation result.

The card renders *only* values present in ``results`` and ``cfg``. Any absent,
``None`` or non-finite value is rendered literally as ``not measured``. No score,
sample size, dataset name or hardware detail is inferred, back-filled or copied
from the legacy paper. This is the reporting-layer counterpart of the
``master-scientist`` guardrail: unsupported claims are deleted or labelled,
never fabricated.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping

__all__ = ["build_model_card"]

NOT_MEASURED = "not measured"

# Keys produced by summarize_ablation that are provenance, not metrics.
_META_KEYS = frozenset(
    {
        "n_configs",
        "n_success",
        "n_failed",
        "failures",
        "metrics",
        "seed",
        "schema_version",
    }
)

_ARCHITECTURE_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Qubits", ("n_qubits", "qubits", "num_qubits")),
    ("Layers", ("qnn_layers", "n_layers", "layers", "num_layers")),
    ("Embedding", ("embedding", "embedding_name", "feature_map")),
    ("Ansatz", ("ansatz", "ansatz_name")),
    (  # real QMLConfig exposes projection_dim; feature_dim may be a property
        "Feature dimension",
        ("projection_dim", "feature_dim", "n_features", "input_dim"),
    ),
    ("Device", ("device", "device_name", "backend")),
    ("Shots", ("shots", "n_shots")),
    ("Depolarizing noise", ("noise_depolarizing", "noise", "depol")),
    ("Optimizer", ("optimizer",)),
    ("Master seed", ("seed",)),
)

_PROVENANCE_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Dataset identifier", ("dataset", "dataset_name", "data_source")),
    ("Split manifest", ("split_manifest", "split_manifest_path", "manifest")),
    ("Data hash", ("data_hash", "dataset_hash", "data_sha256")),
    ("Config hash", ("config_hash",)),
)


def _cfg_to_dict(cfg: Any) -> dict:
    if isinstance(cfg, Mapping):
        return dict(cfg)
    as_dict = getattr(cfg, "as_dict", None)
    if callable(as_dict):
        return dict(as_dict())
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return dict(vars(cfg))


def _find(source: Mapping, candidates: tuple[str, ...]) -> tuple[bool, Any]:
    for name in candidates:
        if name in source:
            return True, source[name]
    return False, None


def _fmt(value: Any) -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return NOT_MEASURED
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    return text if text else NOT_MEASURED


def _normalise_metrics(results: Mapping) -> dict[str, Any]:
    """Return {metric_name: block} from either a summary or a flat result."""
    inner = results.get("metrics")
    if isinstance(inner, Mapping):
        if inner and all(isinstance(v, Mapping) for v in inner.values()):
            return dict(inner)
        if inner:  # a flat metrics mapping nested under "metrics"
            return dict(inner)
    return {k: v for k, v in results.items() if k not in _META_KEYS}


def _metric_cells(name: str, block: Any) -> str:
    """Render one Markdown table row; never invent a cell value."""
    if isinstance(block, Mapping):
        n = _fmt(block.get("n"))
        mean = _fmt(block.get("mean"))
        std = _fmt(block.get("std"))
        lo = _fmt(block.get("ci95_low"))
        hi = _fmt(block.get("ci95_high"))
    else:
        # A bare number is a single reported observation with no measured
        # spread or interval. Do not manufacture either.
        n = NOT_MEASURED
        mean = _fmt(block)
        std = lo = hi = NOT_MEASURED
    ci = NOT_MEASURED
    if lo != NOT_MEASURED and hi != NOT_MEASURED:
        ci = f"[{lo}, {hi}]"
    return f"| `{name}` | {n} | {mean} | {std} | {ci} |"


def _field_table(source: Mapping, fields: tuple[tuple[str, tuple[str, ...]], ...]) -> list[str]:
    lines = ["| Field | Value |", "| --- | --- |"]
    for label, candidates in fields:
        found, value = _find(source, candidates)
        lines.append(f"| {label} | {_fmt(value) if found else NOT_MEASURED} |")
    return lines


def build_model_card(results: dict, cfg: Any) -> str:
    """Build a Markdown model card from an ablation summary and a config.

    Parameters
    ----------
    results:
        Output of ``qmlpipe.ablate.summarize_ablation`` (a dict containing a
        ``metrics`` mapping of n/mean/std/CI blocks), or a flat metric dict.
        Missing or non-numeric entries render as ``not measured``.
    cfg:
        A ``QMLConfig``-like object or mapping. Only declared fields are read.

    Returns
    -------
    str
        A complete Markdown document. No value is invented: every number in the
        card is traceable to ``results`` or ``cfg``.
    """
    if not isinstance(results, Mapping):
        raise TypeError("results must be a mapping of metrics/provenance")

    cfg_dict = _cfg_to_dict(cfg)
    metric_blocks = _normalise_metrics(results)
    n_success = results.get("n_success")
    n_failed = results.get("n_failed")

    lines: list[str] = []
    lines.append("# Model card — Quantum Classification QML pipeline")
    lines.append("")
    lines.append(
        "> Auto-generated from the ablation result dictionary. "
        f"Absent values are shown as `{NOT_MEASURED}`; no result is inferred "
        "from prose, legacy artifacts or prior runs."
    )
    lines.append("")

    lines.append("## Intended use")
    lines.append("")
    lines.append(
        "Research-only classification benchmark for comparing quantum-circuit "
        "architecture choices (qubit count, depth, data encoding, ansatz and "
        "depolarizing noise) under a declared, leakage-free protocol "
        "(QC-D01 §1–2; QC-D04 §2.0). This card reports measured run summaries "
        "only; it is not a deployment approval, a clinical or safety claim, or "
        "evidence of quantum advantage."
    )
    lines.append("")

    lines.append("## Training data provenance")
    lines.append("")
    lines.append(
        "Provenance fields recorded in the run configuration are reproduced "
        f"below. Fields that were not recorded read `{NOT_MEASURED}`. The "
        "underlying split manifest and per-sample hashes remain the authority "
        "for leakage checks; this card does not restate them from memory."
    )
    lines.append("")
    lines.extend(_field_table(cfg_dict, _PROVENANCE_FIELDS))
    lines.append("")
    lines.append(
        f"Successful runs: {_fmt(n_success)}; failed runs: {_fmt(n_failed)}. "
        "Failed runs contribute no metrics."
    )
    lines.append("")

    lines.append("## Architecture")
    lines.append("")
    lines.extend(_field_table(cfg_dict, _ARCHITECTURE_FIELDS))
    lines.append("")
    lines.append(
        "Quoted architecture values describe the configuration actually "
        "supplied; no default has been substituted for a missing field."
    )
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "Sample standard deviation uses ddof=1 and the 95% interval is the "
        "percentile bootstrap over run-level metric vectors (QC-D04 §5)."
    )
    lines.append("")
    if metric_blocks:
        lines.append("| Metric | n | Mean | Std (ddof=1) | 95% CI |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name in sorted(metric_blocks):
            lines.append(_metric_cells(name, metric_blocks[name]))
    else:
        lines.append(f"No metrics were supplied. All metric fields are `{NOT_MEASURED}`.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- **No quantum-advantage claim.** The parameter-free amplitude-embedding "
        "feature map is exactly the classical squared dot product "
        "`((x_i·x_j)/(||x_i|| ||x_j||))^2` (QC-D01 C5; QC-D03 §3.5). A circuit "
        "with no entanglement and no trainable parameters is a classical kernel; "
        "advantage is not asserted from it. The entangled QNN arms are simulated "
        "at 4–10 qubits and remain classically simulable, so no speedup is "
        "claimed without a matched classical control."
    )
    lines.append(
        "- **CPU simulation only.** All results come from CPU simulators; no "
        "hardware fidelity, queue time, shot-calibration or wall-clock speedup "
        "was measured. Timing and device details are reported only if present "
        "in the supplied result dictionary."
    )
    lines.append(
        "- **Small subset / limited power.** Run counts and evaluation subsets "
        "are small; per-run sampling noise can dominate run-to-run spread "
        "(QC-D04 §5). Confidence intervals are reported rather than a single "
        "point estimate, and 'significant' is not claimed from these summaries "
        "without the pre-registered paired tests, effect sizes and power "
        "analysis mandated by QC-D04 §2.4."
    )
    lines.append(
        "- **Ablation arms are not headliners.** Angle-embedding and noisy arms "
        "use different input widths or noise conventions and are never averaged "
        "with the canonical amplitude/entangling configuration (QC-D03 §4.3)."
    )
    lines.append("")

    lines.append("## Improvement plan")
    lines.append("")
    lines.append(
        "See `docs/QML_IMPROVEMENT_PLAN.md` (P0 statistics/baseline honesty, "
        "P1 matched classical controls, P3 encoding/ansatz ablations). "
        "Recommendations only; no values in this card are claims of measured advantage."
    )
    lines.append("")

    lines.append("## Ethical and deployment notes")
    lines.append("")
    lines.append(
        "- **Not for deployment.** This model is a research comparator on a "
        "bounded image-classification subset. It has not been validated for "
        "safety, robustness, fairness or calibrated decision-making."
    )
    lines.append(
        "- **Bias and subgroup audit.** Aggregate metrics can hide subgroup "
        "failures; per-subgroup performance and a decision-threshold audit are "
        "required before any operational use (ml-for-science best practice)."
    )
    lines.append(
        "- **Calibration.** Scores from precomputed-kernel SVC are internal "
        "Platt probabilities and neural softmax outputs are uncalibrated; no "
        "calibration claim is made unless a calibration result is explicitly "
        "supplied in `results`."
    )
    lines.append(
        "- **Provenance and dual use.** Record the exact environment, seeds, "
        "data hashes and commit before release. Do not present simulated "
        "quantum results as hardware evidence or use them to justify "
        "high-stakes interventions."
    )
    lines.append("")

    return "\n".join(lines)
