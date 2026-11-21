"""Generate every paper figure from metrics.json plus saved raw artefacts.

QC-D04 §4 rules implemented here:

1. All plotted numbers are read from the metrics JSON (or, for curves, from
   the raw prediction artefacts). Importing legacy plotting scripts or
   embedding ``sota_data`` / ``training_data`` arrays is forbidden.
2. Captions are rendered from computed ``caption_numbers``; if metrics.json
   already carries a caption for a figure it is the single source of truth.
3. Each PNG is accompanied by a ``<figure_id>.provenance.json`` sidecar naming
   source metric keys, raw artefact paths, config hash, seed and git SHA.
4. Model order, colour assignment, figure order and JSON key order are fixed.
5. matplotlib is optional: when it cannot be imported every figure is returned
   with status ``skipped`` and no filesystem writes occur.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .tables import assert_no_result_literals, resolve_metric_path

# Runner experiment ids used for aggregate filtering (names, never results).
MAIN_EXPERIMENT = "main_ood"
QSVC_EXPERIMENT = "qsvc_mixed_variance"
NOISE_EXPERIMENT = "noise_robustness"

# The runner names figure descriptors ``fig_<experiment>``; these are the
# paper-facing spec ids they feed. Kept explicit so no figure silently drifts.
RUNNER_FIGURE_TO_SPEC: dict[str, str] = {
    f"fig_{MAIN_EXPERIMENT}": "fig01_main_ood_accuracy",
    f"fig_{QSVC_EXPERIMENT}": "fig02_qsvc_run_variance",
    f"fig_{NOISE_EXPERIMENT}": "fig03_noise_robustness",
    "fig_significance": "fig04_significance_forest",
    "fig_calibration": "fig05_reliability",
}


def spec_id_for_runner_figure(figure_id: str) -> str | None:
    """Map a runner figure id (``fig_<experiment>``) to a registered spec id."""
    return RUNNER_FIGURE_TO_SPEC.get(figure_id)

# matplotlib is optional. Import it defensively and select a headless backend
# before pyplot so figure rendering works on CI machines without a display.
try:  # pragma: no cover - depends on the installed environment
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _MPL_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - depends on the installed environment
    plt = None  # type: ignore[assignment]
    _MPL_ERROR = str(exc)


def matplotlib_available() -> bool:
    """Return whether a usable matplotlib/pyplot is importable."""
    return plt is not None


@dataclass(frozen=True)
class FigureSpec:
    """Declarative description of a figure.

    ``data_paths`` lists alternative dotted paths to the input data; the first
    one present in the metrics dict wins. ``caption_fields`` are fallback
    sources used only when metrics.json does not already carry a caption.
    """

    figure_id: str
    title: str
    kind: str
    data_paths: tuple[str, ...]
    source_metric_keys: tuple[str, ...]
    caption_template: str
    caption_fields: tuple[tuple[str, str], ...] = ()
    optional: bool = False


# Ordered by figure id for deterministic rendering. The last three figures are
# optional: the paper generated them from hardcoded legacy arrays, so they are
# only produced when genuine computed data is present in the metrics JSON.
FIGURE_SPECS: tuple[FigureSpec, ...] = (
    FigureSpec(
        figure_id="fig01_main_ood_accuracy",
        title="Main OOD accuracy by model",
        kind="bar_with_ci",
        data_paths=("aggregates",),
        source_metric_keys=(
            "aggregates.*.accuracy.n",
            "aggregates.*.accuracy.mean",
            "aggregates.*.accuracy.std",
            "aggregates.*.accuracy.ci95_low",
            "aggregates.*.accuracy.ci95_high",
            "aggregates.*.accuracy.values",
        ),
        caption_template="Main OOD accuracy {mean:.3f} +/- {std:.3f} (n={n} runs).",
        caption_fields=(
            ("mean", "headline.main_ood_accuracy.mean"),
            ("std", "headline.main_ood_accuracy.std"),
            ("n", "headline.main_ood_accuracy.n"),
        ),
    ),
    FigureSpec(
        figure_id="fig02_qsvc_run_variance",
        title="QSVC mixed-variance per-run accuracy",
        kind="run_scatter",
        data_paths=(
            "aggregates.qsvc_quantum_kernel.accuracy",
            "qsvc_run_summary.accuracy",
        ),
        source_metric_keys=(
            "aggregates.qsvc_quantum_kernel.accuracy.values",
            "aggregates.qsvc_quantum_kernel.accuracy.mean",
            "aggregates.qsvc_quantum_kernel.accuracy.std",
            "aggregates.qsvc_quantum_kernel.accuracy.ci95_low",
            "aggregates.qsvc_quantum_kernel.accuracy.ci95_high",
        ),
        caption_template="QSVC mixed OOD accuracy {mean:.3f} +/- {std:.3f} "
        "(n={n} runs; bootstrap 95% CI {lo:.3f}-{hi:.3f}).",
        caption_fields=(
            ("mean", "aggregates.qsvc_quantum_kernel.accuracy.mean"),
            ("std", "aggregates.qsvc_quantum_kernel.accuracy.std"),
            ("n", "aggregates.qsvc_quantum_kernel.accuracy.n"),
            ("lo", "aggregates.qsvc_quantum_kernel.accuracy.ci95_low"),
            ("hi", "aggregates.qsvc_quantum_kernel.accuracy.ci95_high"),
        ),
    ),
    FigureSpec(
        figure_id="fig03_noise_robustness",
        title="Accuracy degradation under Gaussian noise",
        kind="noise_curve",
        data_paths=("noise_table", "tables.noise_table"),
        source_metric_keys=(
            "noise_table.*.sigma_noisy",
            "noise_table.*.accuracy_noisy.mean",
            "noise_table.*.accuracy_noisy.ci95_low",
            "noise_table.*.accuracy_noisy.ci95_high",
            "noise_table.*.delta.mean",
            "noise_table.*.delta.ci95_low",
            "noise_table.*.delta.ci95_high",
        ),
        caption_template="At sigma={sigma} accuracy is {mean:.3f} (n={n}; "
        "paired delta {delta:.3f}, 95% CI {lo:.3f}-{hi:.3f}).",
        caption_fields=(
            ("sigma", "noise_headline.sigma_noisy"),
            ("mean", "noise_headline.accuracy_noisy.mean"),
            ("n", "noise_headline.delta.n"),
            ("delta", "noise_headline.delta.mean"),
            ("lo", "noise_headline.delta.ci95_low"),
            ("hi", "noise_headline.delta.ci95_high"),
        ),
    ),
    FigureSpec(
        figure_id="fig04_significance_forest",
        title="Pre-registered paired comparisons",
        kind="forest",
        data_paths=("comparisons",),
        source_metric_keys=(
            "comparisons.*.pair",
            "comparisons.*.mean_difference",
            "comparisons.*.ci95_low",
            "comparisons.*.ci95_high",
            "comparisons.*.p_raw",
            "comparisons.*.p_holm",
        ),
        caption_template="Pre-registered paired differences (n={n} runs).",
        caption_fields=(("n", "significance_headline.n_runs"),),
    ),
    FigureSpec(
        figure_id="fig05_reliability",
        title="Reliability diagram",
        kind="reliability",
        data_paths=("calibration.reliability",),
        source_metric_keys=(
            "calibration.reliability.*.bin_lower",
            "calibration.reliability.*.bin_upper",
            "calibration.reliability.*.mean_predicted",
            "calibration.reliability.*.mean_observed",
            "calibration.reliability.*.count",
            "calibration.ece",
        ),
        caption_template="Reliability diagram (ECE {ece:.3f}, n={n}).",
        caption_fields=(
            ("ece", "calibration.ece"),
            ("n", "calibration.n_samples"),
        ),
    ),
    FigureSpec(
        figure_id="fig08_training_curves",
        title="Training curves",
        kind="training_curves",
        data_paths=("training_curves", "raw.training_curves"),
        source_metric_keys=("training_curves.*.epoch", "training_curves.*.accuracy"),
        caption_template="Training curves from recorded per-epoch metrics.",
        optional=True,
    ),
    FigureSpec(
        figure_id="fig14_sota_bars",
        title="State-of-the-art accuracy bars",
        kind="sota_bars",
        data_paths=("sota_metrics",),
        source_metric_keys=("sota_metrics.*.accuracy",),
        caption_template="State-of-the-art accuracy from regenerated benchmark metrics.",
        optional=True,
    ),
    FigureSpec(
        figure_id="fig15_radar",
        title="Overall performance radar",
        kind="radar",
        data_paths=("radar_metrics",),
        source_metric_keys=("radar_metrics.*.*",),
        caption_template="Overall performance from regenerated benchmark metrics.",
        optional=True,
    ),
)

FIGURE_IDS: tuple[str, ...] = tuple(spec.figure_id for spec in FIGURE_SPECS)


@dataclass
class FigureArtifact:
    """Result of one figure render attempt.

    ``status`` is ``"rendered"`` or ``"skipped"``. Skipped figures carry a
    human-readable ``reason`` and have no output paths.
    """

    figure_id: str
    status: str
    png_path: Path | None = None
    svg_path: Path | None = None
    caption_path: Path | None = None
    provenance_path: Path | None = None
    caption: str = ""
    caption_numbers: dict[str, Any] = field(default_factory=dict)
    source_metric_keys: tuple[str, ...] = ()
    raw_artifacts: tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary used by figures_index.json."""
        return {
            "figure_id": self.figure_id,
            "status": self.status,
            "png_path": str(self.png_path) if self.png_path else None,
            "svg_path": str(self.svg_path) if self.svg_path else None,
            "caption_path": str(self.caption_path) if self.caption_path else None,
            "provenance_path": str(self.provenance_path) if self.provenance_path else None,
            "caption": self.caption,
            "caption_numbers": self.caption_numbers,
            "source_metric_keys": list(self.source_metric_keys),
            "raw_artifacts": list(self.raw_artifacts),
            "reason": self.reason,
        }


def get_figure_spec(figure_id: str) -> FigureSpec:
    """Return the registered :class:`FigureSpec` or raise ``KeyError``."""
    for spec in FIGURE_SPECS:
        if spec.figure_id == figure_id:
            return spec
    raise KeyError(f"unknown figure id {figure_id!r}; expected one of {list(FIGURE_IDS)}")


def _lookup(mapping: Mapping[str, Any], dotted: str) -> Any:
    """Resolve a dotted path through nested mappings and integer indices."""
    current: Any = mapping
    for segment in dotted.split("."):
        if isinstance(current, Mapping):
            if segment not in current:
                raise KeyError(dotted)
            current = current[segment]
        elif isinstance(current, (list, tuple)) and segment.isdigit():
            index = int(segment)
            if index >= len(current):
                raise KeyError(dotted)
            current = current[index]
        else:
            raise KeyError(dotted)
    return current


def _resolve(mapping: Mapping[str, Any], paths: Sequence[str]) -> Any:
    """Return the first dotted path present, accepting namespaced runner keys."""
    for path in paths:
        try:
            value = resolve_metric_path(mapping, path)
        except KeyError:
            continue
        if value is not None:
            return value
    return None


def _git_sha(repo_root: str | Path | None = None) -> str | None:
    """Return the local HEAD commit SHA, or ``None`` when git is unavailable.

    This is a purely local read; no network access is performed.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            text=True,
            capture_output=True,
            check=False,
        )
    except (OSError, ValueError):  # pragma: no cover - depends on the machine
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _figure_entry(spec: FigureSpec, metrics: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Find the metrics ``figures`` descriptor feeding ``spec``.

    Both the paper spec id and the runner's ``fig_<experiment>`` id are
    accepted via :data:`RUNNER_FIGURE_TO_SPEC`.
    """
    entries = metrics.get("figures", []) if isinstance(metrics, Mapping) else []
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return None
    accepted = {spec.figure_id}
    accepted.update(
        runner_id for runner_id, spec_id in RUNNER_FIGURE_TO_SPEC.items()
        if spec_id == spec.figure_id
    )
    for entry in entries:
        if isinstance(entry, Mapping) and entry.get("figure_id") in accepted:
            return entry
    return None


def _clean_aggregates(metrics: Mapping[str, Any], experiment: str | None = None) -> dict[str, Mapping[str, Any]]:
    """Return clean (sigma=0) aggregate entries, optionally for one experiment."""
    aggregates = metrics.get("aggregates", {}) if isinstance(metrics, Mapping) else {}
    if not isinstance(aggregates, Mapping):
        return {}
    selected: dict[str, Mapping[str, Any]] = {}
    for name in sorted(aggregates):
        entry = aggregates[name]
        if not isinstance(entry, Mapping):
            continue
        if experiment is not None and entry.get("experiment_id") != experiment:
            continue
        try:
            sigma = float(entry.get("sigma", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if sigma == 0.0:
            selected[name] = entry
    return selected


def derive_caption_context(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the caption-shortcut keys from the real metrics payload.

    The runner deliberately does not emit ``headline.*``,
    ``noise_headline.*``, ``significance_headline.*`` or ``calibration.ece``.
    Every value below is computed from aggregates/comparisons/noise_table/runs
    read from JSON; nothing is stored in source code or guessed.
    """
    context: dict[str, Any] = {}

    clean_main = _clean_aggregates(metrics, MAIN_EXPERIMENT)
    if clean_main:
        def _rank(name: str) -> tuple[float, str]:
            mean = _field(clean_main[name], "accuracy.mean")
            return (-float(mean) if mean is not None else 0.0, name)
        best = sorted(clean_main, key=_rank)[0]
        accuracy = clean_main[best].get("accuracy", {})
        if isinstance(accuracy, Mapping):
            context["headline"] = {
                "main_ood_accuracy": {
                    key: accuracy.get(key)
                    for key in ("mean", "std", "n", "ci95_low", "ci95_high")
                }
            }

    aggregates = metrics.get("aggregates", {}) if isinstance(metrics, Mapping) else {}
    noise_rows = _resolve(metrics, ("noise_table", "tables.noise_table"))
    rows: list[Mapping[str, Any]] = []
    if isinstance(noise_rows, Sequence) and not isinstance(noise_rows, (str, bytes)):
        rows = [row for row in noise_rows if isinstance(row, Mapping)]
    if rows:
        rows = sorted(rows, key=lambda row: float(_field(row, "sigma_noisy") or 0.0))
        last = rows[-1]
        context["noise_headline"] = {
            "sigma_noisy": _field(last, "sigma_noisy"),
            "accuracy_noisy": {"mean": _field(last, "accuracy_noisy.mean")},
            "delta": {
                key: _field(last, f"delta.{key}")
                for key in ("mean", "ci95_low", "ci95_high", "n")
            },
        }
    elif isinstance(aggregates, Mapping):
        noisy = [
            entry for entry in aggregates.values()
            if isinstance(entry, Mapping)
            and entry.get("experiment_id") == NOISE_EXPERIMENT
            and float(entry.get("sigma", 0.0) or 0.0) != 0.0
            and isinstance(entry.get("delta_accuracy_from_clean"), Mapping)
        ]
        if noisy:
            last = sorted(noisy, key=lambda entry: float(entry.get("sigma", 0.0) or 0.0))[-1]
            delta = last["delta_accuracy_from_clean"]
            context["noise_headline"] = {
                "sigma_noisy": last.get("sigma"),
                "accuracy_noisy": {"mean": _field(last, "accuracy.mean")},
                "delta": {
                    key: delta.get(key)
                    for key in ("mean", "ci95_low", "ci95_high", "n")
                },
            }

    comparisons = metrics.get("comparisons", []) if isinstance(metrics, Mapping) else []
    if isinstance(comparisons, Sequence) and comparisons and isinstance(comparisons[0], Mapping):
        context["significance_headline"] = {"n_runs": comparisons[0].get("n")}

    runs = metrics.get("runs", []) if isinstance(metrics, Mapping) else []
    if isinstance(runs, Sequence) and not isinstance(runs, (str, bytes)):
        eces = [
            float(row["ece"]) for row in runs
            if isinstance(row, Mapping) and row.get("ece") is not None
        ]
        counts = [
            int(float(row["n_samples"])) for row in runs
            if isinstance(row, Mapping) and row.get("n_samples") is not None
        ]
        if eces or counts:
            calibration: dict[str, Any] = {}
            if eces:
                calibration["ece"] = sum(eces) / len(eces)
            if counts:
                calibration["n_samples"] = max(counts)
            context["calibration"] = calibration
    return context


def _caption_numbers(spec: FigureSpec, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Collect caption numbers from metrics.json, never from source code."""
    numbers: dict[str, Any] = {}
    entry = _figure_entry(spec, metrics)
    if entry is not None:
        raw = entry.get("caption_numbers", {})
        if isinstance(raw, Mapping):
            numbers.update(dict(raw))
    derived = derive_caption_context(metrics)
    for field_name, dotted in spec.caption_fields:
        if field_name in numbers:
            continue
        for source in (metrics, derived):
            try:
                numbers[field_name] = resolve_metric_path(source, dotted)
                break
            except KeyError:
                continue
    return numbers


def _caption_text(spec: FigureSpec, numbers: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    """Return the figure caption, preferring the one stored in metrics.json."""
    entry = _figure_entry(spec, metrics)
    if entry is not None:
        caption = entry.get("caption")
        if isinstance(caption, str) and caption:
            return caption
    try:
        return spec.caption_template.format(**numbers)
    except (KeyError, ValueError, IndexError):
        # No computed numbers: fall back to a neutral title rather than
        # inventing a single numeric claim.
        return spec.title


def _load_raw_frame(raw_artifacts: Sequence[str | Path]) -> Any:
    """Load the first readable raw CSV/parquet artefact, or return ``None``.

    Used only for figures that need per-epoch data; hard failures are skipped
    cleanly because raw artefacts are optional for aggregate figures.
    """
    if not raw_artifacts:
        return None
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a declared dependency
        return None
    for raw in raw_artifacts:
        path = Path(raw)
        if not path.exists():
            continue
        try:
            if path.suffix == ".csv":
                return pd.read_csv(path)
            return pd.read_parquet(path)
        except Exception:  # pragma: no cover - depends on available engines
            continue
    return None


def _main_ood_clean_only(payload: Any) -> Any:
    """Restrict the main-OOD bar chart to clean ``main_ood`` aggregates.

    Runner aggregate keys mix experiments and sigma levels; plotting all of
    them would put QSVC and noise arms into the OOD benchmark bars. When the
    payload has no experiment metadata (an older flat fixture) it is left
    untouched so historical inputs still render.
    """
    if not isinstance(payload, Mapping):
        return payload
    entries = [value for value in payload.values() if isinstance(value, Mapping)]
    if not entries or not any("experiment_id" in value for value in entries):
        return payload
    filtered: dict[str, Any] = {}
    for name in sorted(payload):
        entry = payload[name]
        if not isinstance(entry, Mapping):
            continue
        if entry.get("experiment_id") != MAIN_EXPERIMENT:
            continue
        if "/" in name and not name.startswith(f"{MAIN_EXPERIMENT}/"):
            continue
        try:
            sigma = float(entry.get("sigma", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if sigma != 0.0:
            continue
        filtered[name] = entry
    return filtered


def _prepare_data(spec: FigureSpec, metrics: Mapping[str, Any], raw_frame: Any) -> tuple[Any, str | None]:
    """Resolve and validate figure inputs.

    Returns ``(payload, None)`` on success or ``(None, reason)`` when required
    inputs are absent (optional figures skip silently in that case).
    """
    payload = _resolve(metrics, spec.data_paths)
    if payload is None and spec.kind == "training_curves" and raw_frame is not None:
        payload = raw_frame
    if payload is None:
        return None, f"no metric data at {', '.join(spec.data_paths)}"
    if spec.figure_id == "fig01_main_ood_accuracy":
        payload = _main_ood_clean_only(payload)
        if not isinstance(payload, Mapping) or not payload:
            return None, "no clean main_ood aggregates for the OOD bar chart"
    if spec.kind in {"bar_with_ci", "sota_bars", "radar"} and not isinstance(payload, Mapping):
        return None, f"expected a mapping of model ids at {spec.data_paths[0]}"
    if spec.kind in {"run_scatter"} and not isinstance(payload, Mapping):
        return None, f"expected an aggregate mapping at {spec.data_paths[0]}"
    if spec.kind in {"noise_curve", "forest", "reliability", "training_curves"}:
        if not isinstance(payload, Sequence) and not hasattr(payload, "columns"):
            return None, f"expected rows at {spec.data_paths[0]}"
    return payload, None


def figure_input(
    figure_id: str,
    metrics: Mapping[str, Any],
    raw_artifacts: Sequence[str | Path] = (),
) -> tuple[Any, str | None]:
    """Expose the resolved, filtered payload a figure will plot.

    Useful for traceability checks (for example asserting that the main-OOD
    chart contains only clean ``main_ood`` aggregates) without rendering.
    """
    spec = get_figure_spec(figure_id)
    return _prepare_data(spec, metrics, _load_raw_frame(raw_artifacts))


def _field(item: Any, dotted: str) -> Any:
    """Best-effort dotted field read returning ``None`` instead of raising."""
    try:
        return _lookup(item, dotted)
    except (KeyError, TypeError):
        return None


def _pick_color(index: int) -> str:
    """Deterministic colour by model family index (Okabe-Ito style hex codes).

    These are styling constants only; no measured value is encoded here.
    """
    palette = (
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#D55E00",
        "#CC79A7",
        "#56B4E9",
        "#F0E442",
    )
    return palette[index % len(palette)]


def _draw(spec: FigureSpec, payload: Any, ax: Any) -> None:
    """Dispatch plotting to the figure's declared kind.

    Every value plotted below comes from ``payload`` (loaded from disk); the
    only literals are axis-styling constants and the reliability diagonal.
    """
    kind = spec.kind
    if kind == "bar_with_ci":
        model_ids = sorted(payload.keys())
        means = [float(_field(payload[mid], "accuracy.mean")) for mid in model_ids]
        errs = []
        for index, mid in enumerate(model_ids):
            lo = _field(payload[mid], "accuracy.ci95_low")
            hi = _field(payload[mid], "accuracy.ci95_high")
            errs.append(
                [float(means[index] - float(lo)), float(float(hi) - means[index])]
                if lo is not None and hi is not None else 0.0
            )
        xs = list(range(len(model_ids)))
        ax.bar(xs, means, yerr=errs, capsize=4,
               color=[_pick_color(i) for i in range(len(model_ids))])
        for i, mid in enumerate(model_ids):
            values = _field(payload[mid], "accuracy.values")
            if isinstance(values, (list, tuple)):
                ax.scatter([i] * len(values), [float(v) for v in values],
                           color=_pick_color(i), zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(model_ids, rotation=30, ha="right")
        ax.set_ylabel("accuracy (fraction)")
    elif kind == "run_scatter":
        values = payload.get("values")
        if not isinstance(values, (list, tuple)):
            values = []
        x_positions = list(range(len(values)))
        ax.scatter(x_positions, [float(v) for v in values], color=_pick_color(0), zorder=3)
        mean = payload.get("mean")
        lo = payload.get("ci95_low")
        hi = payload.get("ci95_high")
        if mean is not None:
            ax.axhline(float(mean), color=_pick_color(1), linestyle="--")
        if lo is not None and hi is not None:
            ax.axhspan(float(lo), float(hi), color=_pick_color(0), alpha=0.15)
        ax.set_xlabel("run index")
        ax.set_ylabel("accuracy (fraction)")
    elif kind == "noise_curve":
        rows = [row for row in payload if isinstance(row, Mapping)]
        rows.sort(key=lambda row: str(_field(row, "sigma_noisy")))
        sigmas = [_field(row, "sigma_noisy") for row in rows]
        means = [_field(row, "accuracy_noisy.mean") for row in rows]
        lows = [_field(row, "accuracy_noisy.ci95_low") for row in rows]
        highs = [_field(row, "accuracy_noisy.ci95_high") for row in rows]
        ax.errorbar(sigmas, means,
                    yerr=[[float(m) - float(lo) if lo is not None else 0.0
                           for m, lo in zip(means, lows)],
                          [float(hi) - float(m) if hi is not None else 0.0
                           for m, hi in zip(means, highs)]],
                    marker="o", capsize=4, color=_pick_color(0))
        ax.set_xlabel("sigma")
        ax.set_ylabel("accuracy (fraction)")
    elif kind == "forest":
        rows = [row for row in payload if isinstance(row, Mapping)]
        labels = [" vs ".join(str(v) for v in _field(row, "pair") or [])
                  for row in rows]
        diffs = [_field(row, "mean_difference") for row in rows]
        lows = [_field(row, "ci95_low") for row in rows]
        highs = [_field(row, "ci95_high") for row in rows]
        y_positions = list(range(len(rows)))
        ax.errorbar(diffs, y_positions,
                    xerr=[[float(d) - float(lo) if lo is not None else 0.0
                           for d, lo in zip(diffs, lows)],
                          [float(hi) - float(d) if hi is not None else 0.0
                           for d, hi in zip(diffs, highs)]],
                    fmt="o", capsize=4, color=_pick_color(0))
        ax.axvline(0.0, color=_pick_color(3), linestyle="--")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel("mean paired difference")
    elif kind == "reliability":
        rows = [row for row in payload if isinstance(row, Mapping)]
        predicted = [float(_field(row, "mean_predicted")) for row in rows
                     if _field(row, "mean_predicted") is not None]
        observed = [float(_field(row, "mean_observed")) for row in rows
                     if _field(row, "mean_observed") is not None]
        # Identity reference line built at runtime, not pasted as a data array.
        import numpy as np

        diagonal = np.linspace(0.0, 1.0, num=len(predicted) or 2)
        ax.plot(diagonal, diagonal, linestyle="--", color=_pick_color(3))
        ax.plot(predicted, observed, marker="o", color=_pick_color(0))
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed fraction")
    elif kind == "training_curves":
        if hasattr(payload, "columns"):
            frame = payload
            for i, model_id in enumerate(sorted(str(v) for v in frame["model"].unique())):
                sub = frame[frame["model"] == model_id].sort_values("epoch")
                ax.plot(list(sub["epoch"]), list(sub["accuracy"]),
                        label=str(model_id), color=_pick_color(i))
            ax.legend(fontsize="small")
        else:
            for i, model_id in enumerate(sorted(payload.keys())):
                curve = payload[model_id]
                ax.plot(list(curve["epoch"]), list(curve["accuracy"]),
                        label=str(model_id), color=_pick_color(i))
            ax.legend(fontsize="small")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
    elif kind == "sota_bars":
        model_ids = sorted(payload.keys())
        means = [float(_field(payload[mid], "accuracy")) for mid in model_ids]
        xs = list(range(len(model_ids)))
        ax.bar(xs, means, color=[_pick_color(i) for i in range(len(model_ids))])
        ax.set_xticks(xs)
        ax.set_xticklabels(model_ids, rotation=30, ha="right")
        ax.set_ylabel("accuracy")
    elif kind == "radar":
        model_ids = sorted(payload.keys())
        axes = sorted({str(key) for mid in model_ids for key in payload[mid]})
        for i, mid in enumerate(model_ids):
            values = [float(_field(payload[mid], axis)) for axis in axes]
            ax.plot(list(range(len(axes))), values, marker="o",
                    label=str(mid), color=_pick_color(i))
        ax.set_xticks(list(range(len(axes))))
        ax.set_xticklabels(axes)
        ax.legend(fontsize="small")
    else:  # pragma: no cover - guarded by the registry
        raise ValueError(f"unsupported figure kind {kind!r}")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON via a temporary file so readers never see a partial sidecar."""
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def render_figure(
    figure_id: str,
    metrics: Mapping[str, Any],
    raw_artifacts: Sequence[str | Path] = (),
    output_dir: str | Path | None = None,
    provenance: Mapping[str, Any] | None = None,
    git_sha: str | None = None,
) -> FigureArtifact:
    """Render one figure, or return a skipped artifact with the reason.

    The ``output_dir`` receives the PNG, SVG, caption JSON and provenance JSON.
    Calling with identical inputs overwrites the same files deterministically.
    """
    spec = get_figure_spec(figure_id)
    key_list = tuple(spec.source_metric_keys)
    raw_list = tuple(str(path) for path in raw_artifacts)
    numbers = _caption_numbers(spec, metrics)
    caption = _caption_text(spec, numbers, metrics)
    prov = dict(provenance or (metrics.get("provenance", {}) if isinstance(metrics, Mapping) else {}))

    if not matplotlib_available():
        return FigureArtifact(
            figure_id=figure_id,
            status="skipped",
            caption=caption,
            caption_numbers=numbers,
            source_metric_keys=key_list,
            raw_artifacts=raw_list,
            reason=f"matplotlib unavailable: {_MPL_ERROR}",
        )

    raw_frame = _load_raw_frame(raw_artifacts)
    payload, reason = _prepare_data(spec, metrics, raw_frame)
    if payload is None:
        return FigureArtifact(
            figure_id=figure_id,
            status="skipped",
            caption=caption,
            caption_numbers=numbers,
            source_metric_keys=key_list,
            raw_artifacts=raw_list,
            reason=reason or "missing inputs",
        )

    directory = Path(output_dir) if output_dir else Path.cwd()
    directory.mkdir(parents=True, exist_ok=True)
    png_path = directory / f"{figure_id}.png"
    svg_path = directory / f"{figure_id}.svg"
    caption_path = directory / f"{figure_id}.caption.json"
    provenance_path = directory / f"{figure_id}.provenance.json"

    # Write metadata BEFORE the image so no PNG can ever exist without its
    # sidecar, and write it atomically so an interrupted run leaves no partial
    # JSON. A failed plot removes every artefact below.
    caption_payload = {
        "figure_id": figure_id,
        "caption": caption,
        "caption_numbers": numbers,
        "source_metric_keys": list(key_list),
    }
    sidecar = {
        "figure_id": figure_id,
        "generator": "reporting.figures.render_figure",
        "source_metric_keys": list(key_list),
        "source_metric_values": numbers,
        "raw_artifacts": list(raw_list),
        "config_hash": prov.get("config_hash"),
        "seed": prov.get("master_seed", prov.get("seed")),
        "seed_map": prov.get("seed_map"),
        "git_sha": git_sha if git_sha is not None else _git_sha(),
        "data_manifest_hash": prov.get("data_manifest_hash"),
        "library_versions": prov.get("library_versions", {}),
        "status": "rendered",
    }
    _atomic_write_json(caption_path, caption_payload)
    _atomic_write_json(provenance_path, sidecar)

    plot_error: Exception | None = None
    fig, ax = plt.subplots()
    try:
        ax.set_title(spec.title)
        _draw(spec, payload, ax)
        fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
    except Exception as exc:  # malformed data must never crash a reporting run
        plot_error = exc
    finally:
        plt.close(fig)
    if plot_error is not None:
        for path in (png_path, svg_path, caption_path, provenance_path):
            try:
                path.unlink()
            except OSError:
                pass
        return FigureArtifact(
            figure_id=figure_id,
            status="skipped",
            caption=caption,
            caption_numbers=numbers,
            source_metric_keys=key_list,
            raw_artifacts=raw_list,
            reason=f"plot failed: {plot_error}",
        )

    return FigureArtifact(
        figure_id=figure_id,
        status="rendered",
        png_path=png_path,
        svg_path=svg_path,
        caption_path=caption_path,
        provenance_path=provenance_path,
        caption=caption,
        caption_numbers=numbers,
        source_metric_keys=key_list,
        raw_artifacts=raw_list,
    )


def write_figures_index(artifacts: Sequence[FigureArtifact], output_dir: str | Path) -> Path:
    """Write ``figures_index.json`` from the render results and return its path."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    index = {
        "figures": [artifact.as_dict() for artifact in sorted(artifacts, key=lambda a: a.figure_id)],
        "generator": "reporting.figures.write_figures_index",
    }
    path = directory / "figures_index.json"
    path.write_text(json.dumps(index, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def render_all_figures(
    metrics_path: str | Path,
    raw_artifacts: Sequence[str | Path] = (),
    output_dir: str | Path | None = None,
    provenance: Mapping[str, Any] | None = None,
    git_sha: str | None = None,
) -> list[FigureArtifact]:
    """Render all registered figures from a metrics JSON file on disk.

    ``output_dir`` defaults to ``<run_dir>/figures`` where ``metrics_path`` is
    ``<run_dir>/metrics/metrics.json``. No dataset or model files are read.
    """
    path = Path(metrics_path)
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if output_dir is None:
        parent = path.parent
        run_root = parent.parent if parent.name == "metrics" else parent
        output_dir = run_root / "figures"
    prov = dict(provenance or metrics.get("provenance", {}) or {})
    artifacts = [
        render_figure(
            figure_id=spec.figure_id,
            metrics=metrics,
            raw_artifacts=raw_artifacts,
            output_dir=output_dir,
            provenance=prov,
            git_sha=git_sha,
        )
        for spec in FIGURE_SPECS
    ]
    write_figures_index(artifacts, output_dir)
    return artifacts


def assert_no_hardcoded_data() -> None:
    """Raise if this module contains a hardcoded numeric result array."""
    assert_no_result_literals(sys.modules[__name__])
