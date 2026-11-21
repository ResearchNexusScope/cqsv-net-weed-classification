"""Reporting layer for the Quantum_Classification rebuild (QC-I06).

Everything in this package renders numbers *from* a results/metrics dictionary.
No result value, figure series, or manuscript number is stored in source code:
tables, captions, and manuscript placeholders are all resolved from the
metrics JSON at call time. Every rendered artefact carries sidecar metadata
that names the exact source metric keys it consumed.

Public API:

* :mod:`reporting.tables` — Tables 1-6 as DataFrames, CSV and sidecars.
* :mod:`reporting.figures` — paper figures from metrics + raw artefacts.
* :mod:`reporting.manuscript_sync` — auditable, opt-in manuscript updates.

Optional dependencies: matplotlib is imported lazily by :mod:`reporting.figures`;
if it is unavailable, figure rendering skips cleanly instead of raising.
"""
from __future__ import annotations

from .tables import (
    MissingMetricKeys,
    RenderedTable,
    TABLE_IDS,
    TABLE_SPECS,
    TableSpec,
    assert_no_result_literals,
    get_table_spec,
    render_all_tables,
    render_table,
    scan_for_literal_arrays,
    write_tables,
)
from .figures import (
    FIGURE_IDS,
    FIGURE_SPECS,
    FigureArtifact,
    FigureSpec,
    assert_no_hardcoded_data,
    matplotlib_available,
    render_all_figures,
    render_figure,
    write_figures_index,
)
from .manuscript_sync import (
    DEFAULT_PLACEHOLDERS,
    MissingResultKey,
    Placeholder,
    SyncDiff,
    UnsafeSyncError,
    apply_sync,
    plan_sync,
    sync_file,
)

__all__ = [
    # tables
    "MissingMetricKeys",
    "RenderedTable",
    "TABLE_IDS",
    "TABLE_SPECS",
    "TableSpec",
    "assert_no_result_literals",
    "get_table_spec",
    "render_all_tables",
    "render_table",
    "scan_for_literal_arrays",
    "write_tables",
    # figures
    "FIGURE_IDS",
    "FIGURE_SPECS",
    "FigureArtifact",
    "FigureSpec",
    "assert_no_hardcoded_data",
    "matplotlib_available",
    "render_all_figures",
    "render_figure",
    "write_figures_index",
    # manuscript sync
    "DEFAULT_PLACEHOLDERS",
    "MissingResultKey",
    "Placeholder",
    "SyncDiff",
    "UnsafeSyncError",
    "apply_sync",
    "plan_sync",
    "sync_file",
]
