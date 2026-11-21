"""Render Tables 1-6 of the paper from a results/metrics.json-like dict only.

Design contract (QC-D04 §3.3, §4, §5):

* Every cell is read from the supplied ``results`` mapping. There are no result
  numbers, no per-model arrays, and no dataset-specific values in this module.
* Each declared column names a dotted *source metric key*. The sidecar returned
  with every table lists those keys verbatim, so a reviewer can trace a cell
  back to the metrics JSON path it came from.
* Missing source keys raise :class:`MissingMetricKeys` with the full list of
  missing dotted paths rather than silently emitting blanks or inventing data.
* A small AST scanner (:func:`scan_for_literal_arrays`) enforces the
  "no hardcoded numeric arrays" rule for this module and the figure module.

The table schemas below describe *where* values live, never *what* they are.
A caller supplies a metrics dict with the documented top-level row collections
(``dataset_table``, ``pipeline_table``, ``performance_table``, ``comparisons``,
``iou_table``, ``noise_table``). Row entries may nest dictionaries; dotted
source keys such as ``accuracy.mean`` traverse them.
"""
from __future__ import annotations

import ast
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


class MissingMetricKeys(ValueError):
    """Raised when a table's required source metric keys are absent.

    The exception message lists every missing dotted path, including the row
    index, so the caller can repair the metrics JSON without guessing.
    """


@dataclass(frozen=True)
class TableSpec:
    """Declarative description of one paper table.

    ``columns`` is an ordered sequence of ``(header, dotted_source_key)``
    pairs. ``rows_key`` is the dotted path to the list of row mappings in the
    results dict. Nothing here stores an observed value.
    """

    table_id: str
    title: str
    rows_key: str
    columns: tuple[tuple[str, str], ...]
    description: str = ""


# The ordered registry of the six paper tables. Source keys are paths into the
# results dict; they intentionally contain no numbers.
TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        table_id="table1_experimental_data",
        title="Table 1. Experimental Data For Performance Evaluation",
        rows_key="dataset_table",
        columns=(
            ("Dataset", "dataset"),
            ("Categories", "categories"),
            ("Patch Size", "patch_size"),
            ("Dim Reduction", "dim_reduction"),
            ("Training", "training"),
            ("Validation", "validation"),
            ("Test", "test"),
        ),
        description="Dataset preparation summary; counts come from the data manifest.",
    ),
    TableSpec(
        table_id="table2_pipeline_description",
        title="Table 2. Pipeline Description about various models along with parameters",
        rows_key="pipeline_table",
        columns=(
            ("#", "index"),
            ("Model", "model"),
            ("Pipeline Description", "pipeline_description"),
        ),
        description="Model pipelines and trainable parameter counts from models/index.json.",
    ),
    TableSpec(
        table_id="table3_performance_comparison",
        title="Table 3. Performance Comparison of CNN, QNN, and Hybrid QC-CNN models",
        rows_key="performance_table",
        columns=(
            ("Model", "model"),
            ("Accuracy", "accuracy.mean"),
            ("Precision", "precision.mean"),
            ("Recall", "recall.mean"),
            ("F1 Score", "f1.mean"),
            ("AUC-ROC", "auc.mean"),
            ("Training Time (sec)", "training_time_seconds.mean"),
        ),
        description="Means are the group summaries in metrics.json; spread/CI are in the sidecar.",
    ),
    TableSpec(
        table_id="table4_significance",
        title="Table 4. Significance Analysis (pre-registered paired tests)",
        rows_key="comparisons",
        columns=(
            ("Model Pair", "pair"),
            ("t", "t_statistic"),
            ("df", "df"),
            ("Mean Difference", "mean_difference"),
            ("CI low", "ci95_low"),
            ("CI high", "ci95_high"),
            ("Cohen dz", "cohens_dz"),
            ("Power", "power_at_observed_dz"),
            ("p (raw)", "p_raw"),
            ("p (Holm)", "p_holm"),
            ("Decision", "decision"),
        ),
        description="All pre-registered pairs with raw and Holm-adjusted p-values.\n"
        "Legacy p-values are never reused.",
    ),
    TableSpec(
        table_id="table5_iou",
        title="Table 5. Intersection over Union (IoU) computed from integer confusion counts",
        rows_key="iou_table",
        columns=(
            ("Model", "model"),
            ("Class", "class"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("TP", "tp"),
            ("FP", "fp"),
            ("FN", "fn"),
            ("IoU", "iou"),
        ),
        description="IoU = TP/(TP+FP+FN) computed upstream from integer counts;\n"
        "it is not derived from rounded precision/recall.",
    ),
    TableSpec(
        table_id="table6_noise",
        title="Table 6. Classification Performance With Gaussian Noise",
        rows_key="noise_table",
        columns=(
            ("Model", "model"),
            ("Sigma clean", "sigma_clean"),
            ("Sigma noisy", "sigma_noisy"),
            ("Accuracy clean", "accuracy_clean.mean"),
            ("Accuracy noisy", "accuracy_noisy.mean"),
            ("Delta", "delta.mean"),
            ("Delta CI low", "delta.ci95_low"),
            ("Delta CI high", "delta.ci95_high"),
            ("n", "delta.n"),
        ),
        description="Paired same-image degradation Delta = acc(clean) - acc(noisy);\n"
        "sigma values and CI are read from the experiment artifacts.",
    ),
)

TABLE_IDS: tuple[str, ...] = tuple(spec.table_id for spec in TABLE_SPECS)


@dataclass
class RenderedTable:
    """A rendered table plus its traceability metadata.

    ``dataframe`` and ``csv`` are two views of the same cell values; ``metadata``
    is the sidecar payload naming every dotted source metric key that was read.
    """

    table_id: str
    title: str
    dataframe: pd.DataFrame
    csv: str
    source_keys: tuple[str, ...]
    rows_key: str
    n_rows: int
    columns: tuple[str, ...]
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a JSON-serialisable sidecar describing this table's inputs."""
        return {
            "table_id": self.table_id,
            "title": self.title,
            "rows_key": self.rows_key,
            "n_rows": self.n_rows,
            "columns": list(self.columns),
            "source_metric_keys": list(self.source_keys),
            "provenance": self.provenance,
            "notes": self.notes,
            "generator": "reporting.tables.render_table",
        }


def get_table_spec(table_id: str) -> TableSpec:
    """Return the :class:`TableSpec` registered under ``table_id``.

    Raises ``KeyError`` with the list of valid ids for an unknown id.
    """
    for spec in TABLE_SPECS:
        if spec.table_id == table_id:
            return spec
    raise KeyError(f"unknown table id {table_id!r}; expected one of {list(TABLE_IDS)}")


def _lookup(mapping: Mapping[str, Any], dotted: str) -> Any:
    """Resolve a dotted path through nested mappings (and integer list indices).

    Raises ``KeyError`` when any segment is absent so callers can report the
    full path rather than a generic missing-value error.
    """
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


def _namespaced_mappings(node: Any, prefix: str = "", out: list[tuple[str, Mapping[str, Any]]] | None = None) -> list[tuple[str, Mapping[str, Any]]]:
    """Collect paths of mappings that contain runner-style ``exp/model`` keys.

    The runner writes aggregate keys such as
    ``qsvc_mixed_variance/qsvc_quantum_kernel`` (and
    ``noise_robustness/cnn/sigma=0.15``). This helper records where such
    mappings live so :func:`resolve_metric_path` can expand a flat dotted key
    without knowing which experiment produced it.
    """
    if out is None:
        out = []
    if isinstance(node, Mapping):
        if any(isinstance(key, str) and "/" in key for key in node):
            out.append((prefix, node))
        for key, value in node.items():
            if not isinstance(key, str) or "." in key:
                continue
            child = f"{prefix}.{key}" if prefix else key
            _namespaced_mappings(value, child, out)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            child = f"{prefix}.{index}" if prefix else str(index)
            _namespaced_mappings(value, child, out)
    return out


def _namespaced_priority(key: str) -> tuple[int, str]:
    """Order namespaced candidate keys: clean ``exp/model`` first, then sigma=0."""
    segments = key.split("/")
    if len(segments) == 2:
        return (0, key)
    tail = segments[2:]
    if any(part.startswith("sigma=") and part.split("=", 1)[1] in ("0", "0.0")
           for part in tail):
        return (1, key)
    return (2, key)


def resolve_metric_path(results: Mapping[str, Any], dotted: str) -> Any:
    """Resolve a dotted metric key, accepting the runner's namespaced keys.

    The flat path is tried first. If it is absent, every mapping in ``results``
    that has slash-separated keys is searched for a key whose final (or any)
    slash segment equals the requested model id, and the path is retried with
    that namespaced key. Candidates are ordered deterministically so clean
    ``experiment/model`` aggregates win over noisy ``.../sigma=...`` ones.

    Raises ``KeyError(dotted)`` when no variant resolves, so callers can report
    the original source key rather than an invented value.
    """
    try:
        return _lookup(results, dotted)
    except KeyError:
        pass
    parts = dotted.split(".")
    candidates: list[tuple[int, str, str]] = []
    for prefix, mapping in _namespaced_mappings(results):
        index = len(prefix.split(".")) if prefix else 0
        if index >= len(parts):
            continue
        requested = parts[index]
        for key in mapping:
            if not isinstance(key, str) or "/" not in key:
                continue
            if requested not in key.split("/"):
                continue
            variant_parts = list(parts)
            variant_parts[index] = key
            candidates.append((*_namespaced_priority(key), ".".join(variant_parts)))
    for _, _, variant in sorted(candidates):
        try:
            return _lookup(results, variant)
        except KeyError:
            continue
    raise KeyError(dotted)


def _display(value: Any) -> Any:
    """Render a cell value without changing any numeric result.

    Lists (used for significance ``pair``) are joined for display; scalars,
    including every measurement, are returned unchanged so DataFrame equality
    with the JSON value is exact.
    """
    if isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
        return " vs ".join(parts) if len(parts) == 2 else ", ".join(parts)
    return value


def render_table(table_id: str, results: Mapping[str, Any]) -> RenderedTable:
    """Render one registered table from ``results``.

    Parameters
    ----------
    table_id:
        One of :data:`TABLE_IDS`.
    results:
        A metrics.json-like mapping holding the row collection named by the
        spec's ``rows_key``.

    Raises
    ------
    MissingMetricKeys
        If the row collection or any declared column source key is missing.
        The message enumerates every missing dotted path.
    """
    spec = get_table_spec(table_id)
    try:
        rows = _lookup(results, spec.rows_key)
    except KeyError as exc:
        raise MissingMetricKeys(
            f"{table_id}: required row collection {spec.rows_key!r} is missing "
            f"from results (key: {exc.args[0]!r})"
        ) from exc
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise MissingMetricKeys(
            f"{table_id}: {spec.rows_key!r} must be a sequence of row mappings, "
            f"got {type(rows).__name__}"
        )

    missing: list[str] = []
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            missing.append(f"{spec.rows_key}[{row_index}] (not a mapping)")
            continue
        record: dict[str, Any] = {}
        for header, source_key in spec.columns:
            try:
                value = _lookup(row, source_key)
            except KeyError:
                missing.append(f"{spec.rows_key}[{row_index}].{source_key}")
                continue
            record[header] = _display(value)
        records.append(record)
    if missing:
        raise MissingMetricKeys(
            f"{table_id}: missing source metric keys: {', '.join(missing)}"
        )

    headers = tuple(header for header, _ in spec.columns)
    frame = pd.DataFrame(records, columns=list(headers))
    source_keys = tuple(source_key for _, source_key in spec.columns)
    return RenderedTable(
        table_id=spec.table_id,
        title=spec.title,
        dataframe=frame,
        csv=frame.to_csv(index=False),
        source_keys=source_keys,
        rows_key=spec.rows_key,
        n_rows=len(records),
        columns=headers,
        provenance=dict(results.get("provenance", {}) or {}),
        notes=spec.description,
    )


def render_all_tables(results: Mapping[str, Any]) -> dict[str, RenderedTable]:
    """Render all six tables in deterministic registry order."""
    return {spec.table_id: render_table(spec.table_id, results) for spec in TABLE_SPECS}


def write_tables(results: Mapping[str, Any], output_dir: str | Path) -> dict[str, dict[str, Path]]:
    """Write each table's CSV and sidecar JSON into ``output_dir``.

    Returns a mapping ``table_id -> {"csv": path, "metadata": path}``. The
    sidecar is byte-identical across repeated calls for the same results dict.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, Path]] = {}
    for table_id, rendered in render_all_tables(results).items():
        csv_path = directory / f"{table_id}.csv"
        meta_path = directory / f"{table_id}.metadata.json"
        csv_path.write_text(rendered.csv, encoding="utf-8")
        meta_path.write_text(
            json.dumps(rendered.metadata, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        written[table_id] = {"csv": csv_path, "metadata": meta_path}
    return written


def scan_for_literal_arrays(source: str, min_length: int = 5) -> list[str]:
    """Flag hardcoded numeric arrays in module source.

    A "numeric array" is a list or tuple literal of at least ``min_length``
    elements where every element is an integer/float constant. Result series
    are always much longer than the small tuples used for axis styling, so the
    default threshold avoids false positives while still catching pasted data.
    Returns a list of human-readable findings (empty means clean).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover - source is always valid here
        return [f"unparsable source: {exc}"]
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        elements = node.elts
        if len(elements) < min_length:
            continue
        if all(isinstance(element, ast.Constant) and isinstance(element.value, (int, float))
               and not isinstance(element.value, bool) for element in elements):
            findings.append(
                f"line {getattr(node, 'lineno', '?')}: numeric literal array "
                f"with {len(elements)} elements"
            )
    return findings


def assert_no_result_literals(*modules: Any, min_length: int = 5) -> None:
    """Raise ``AssertionError`` if any supplied module contains a numeric array.

    Intended for tests; accepts modules (not paths) so it works even when the
    package is installed rather than checked out.
    """
    problems: list[str] = []
    for module in modules:
        findings = scan_for_literal_arrays(inspect.getsource(module), min_length=min_length)
        problems.extend(f"{getattr(module, '__name__', module)}: {item}" for item in findings)
    if problems:
        raise AssertionError("hardcoded numeric arrays detected: " + "; ".join(problems))
