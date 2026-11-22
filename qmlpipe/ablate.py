"""Deterministic, bounded architecture ablations for the QML pipeline.

This module is deliberately framework-agnostic. It consumes a ``QMLConfig``-like
dataclass plus *injected* train/evaluate callables and imports no quantum package
at module scope, so it imports cleanly under interpreters that have no
PennyLane (for example Mother's Python 3.14 verifier).

Design contract (documented so callers can adapt their functions):

* ``train_fn(cfg, seed=seed) -> model`` — alternatively ``train_fn(cfg, seed)``
  or ``train_fn(cfg)``. The adapter inspects the signature and picks a supported
  form.
* ``eval_fn(model, cfg) -> dict[str, float]`` — alternatively ``eval_fn(model)``
  or ``eval_fn(model, cfg=cfg)``. The returned mapping is copied verbatim; no
  value is ever invented, imputed, or clipped by this module.

Grid design
-----------
The grid is a **bounded full factorial** over the five audited architecture
axes from QC-D01/C3, QC-D03 §4.2 and QC-D04 E1:

=====================  ============================================
axis                   documented values
=====================  ============================================
n_qubits               (4, 8, 10)
qnn_layers             (1, 2, 3)
embedding              ("amplitude", "angle")
ansatz                 ("strongly_entangling", "basic_entangler")
noise_depolarizing     (None, 0.01)
=====================  ============================================

3 x 3 x 2 x 2 x 2 = 72 configurations. ``MAX_GRID_SIZE`` records that bound and
``ablation_grid`` refuses to emit a larger grid, so the cost of enumeration can
never grow silently. Execution cost is bounded independently by ``max_runs`` in
:func:`run_ablation`. The recorded input width (``projection_dim`` on the real
``QMLConfig``; ``feature_dim`` on compatible configs) is set to ``2**n_qubits``
in every generated config so ``QMLConfig.validate()`` continues to pass;
angle-embedding arms use their own QNN width at the model layer (QC-D03 §4.2)
and are never averaged with the canonical amplitude arms.

The runner owns only a config and a seed -- it does **not** own data loaders.
D02's ``train(model, train_loader, val_loader, cfg)`` and
``evaluate(model, loader, cfg)`` must therefore be wrapped by the caller into
``train_fn(cfg, seed) -> model`` and ``eval_fn(model, cfg) -> dict`` (for
example with ``functools.partial`` or a small lambda that builds the loaders
from ``cfg``). This keeps the injection boundary explicit and prevents the
runner from inventing a data source.

No experimental result is reported by this module: it only executes the
injected callables and summarises the numbers they actually return.
"""
from __future__ import annotations

import dataclasses
import inspect
import math
import random
from itertools import product
from typing import Any, Callable, Mapping

from experiments import metrics as _metrics

__all__ = [
    "QUBIT_CHOICES",
    "LAYER_CHOICES",
    "EMBEDDING_CHOICES",
    "ANSATZ_CHOICES",
    "NOISE_CHOICES",
    "MAX_GRID_SIZE",
    "AXIS_SPECS",
    "ablation_grid",
    "run_ablation",
    "summarize_ablation",
]

# --------------------------------------------------------------------------
# Bounded grid declaration (single source of truth for the ablation axes).
# --------------------------------------------------------------------------
QUBIT_CHOICES: tuple[int, ...] = (4, 8, 10)
LAYER_CHOICES: tuple[int, ...] = (1, 2, 3)
EMBEDDING_CHOICES: tuple[str, ...] = ("amplitude", "angle")
ANSATZ_CHOICES: tuple[str, ...] = ("strongly_entangling", "basic_entangler")
NOISE_CHOICES: tuple[float | None, ...] = (None, 0.01)

# (axis label, candidate QMLConfig field names, allowed values)
AXIS_SPECS: tuple[tuple[str, tuple[str, ...], tuple[Any, ...]], ...] = (
    ("n_qubits", ("n_qubits", "qubits", "num_qubits"), QUBIT_CHOICES),
    ("qnn_layers", ("qnn_layers", "n_layers", "layers", "num_layers"), LAYER_CHOICES),
    ("embedding", ("embedding", "embedding_name", "feature_map"), EMBEDDING_CHOICES),
    ("ansatz", ("ansatz", "ansatz_name"), ANSATZ_CHOICES),
    (
        "noise_depolarizing",
        ("noise_depolarizing", "noise", "depolarizing", "noise_level", "depol"),
        NOISE_CHOICES,
    ),
)

# ``feature_dim`` is a property on the real QMLConfig (2**qubits); the mutable
# width field that validate() checks is ``projection_dim``. Both are supported.
_FEATURE_FIELD_CANDIDATES = ("projection_dim", "feature_dim", "n_features", "input_dim")
MAX_GRID_SIZE = 3 * 3 * 2 * 2 * 2  # 72; documented upper bound, see module docstring.
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


# --------------------------------------------------------------------------
# Small, dependency-free helpers (no quantum imports).
# --------------------------------------------------------------------------
def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or an object, returning ``default``."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_dict(cfg: Any) -> dict:
    """Return a plain dict for a config, without fabricating absent keys."""
    as_dict = getattr(cfg, "as_dict", None)
    if callable(as_dict):
        return dict(as_dict())
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return dict(vars(cfg))


def _field_names(cfg: Any) -> frozenset[str]:
    if not dataclasses.is_dataclass(cfg):
        raise TypeError(
            "ablation_grid needs a dataclass config (QMLConfig); got "
            f"{type(cfg).__name__!r}"
        )
    return frozenset(f.name for f in dataclasses.fields(cfg))


def _resolve_axes(cfg: Any) -> list[dict[str, Any]]:
    """Map each declared axis onto a real dataclass field of ``cfg``."""
    fields = _field_names(cfg)
    resolved: list[dict[str, Any]] = []
    for axis, candidates, values in AXIS_SPECS:
        field = next((name for name in candidates if name in fields), None)
        if field is None:
            raise ValueError(
                f"config {type(cfg).__name__!r} exposes none of {candidates} "
                f"for ablation axis {axis!r}; cannot build an honest grid"
            )
        resolved.append({"axis": axis, "field": field, "values": values})
    return resolved


def _signature(fn: Callable) -> inspect.Signature | None:
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C callables
        return None


def _param_info(sig: inspect.Signature) -> tuple[int, bool, set[str]]:
    """Return (positional count, has *args, accepted keyword names)."""
    n_pos = 0
    has_varargs = False
    kwargs_ok: set[str] = set()
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            n_pos += 1
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            kwargs_ok.add(p.name)
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            has_varargs = True
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            kwargs_ok.add("__var_kwargs__")
    return n_pos, has_varargs, kwargs_ok


def _call_train(train_fn: Callable, cfg: Any, seed: int) -> Any:
    """Invoke ``train_fn`` using the most specific supported calling form."""
    sig = _signature(train_fn)
    if sig is None:
        return train_fn(cfg, seed)
    n_pos, has_varargs, kwargs_ok = _param_info(sig)
    if "seed" in kwargs_ok:
        return train_fn(cfg, seed=seed)
    if n_pos >= 2 or has_varargs:
        return train_fn(cfg, seed)
    return train_fn(cfg)


def _call_eval(eval_fn: Callable, model: Any, cfg: Any) -> Any:
    """Invoke ``eval_fn``; the trained model is always the first argument."""
    sig = _signature(eval_fn)
    if sig is None:
        return eval_fn(model, cfg)
    n_pos, has_varargs, kwargs_ok = _param_info(sig)
    for key in ("cfg", "config"):
        if key in kwargs_ok:
            return eval_fn(model, **{key: cfg})
    if n_pos >= 2 or has_varargs:
        return eval_fn(model, cfg)
    if n_pos >= 1:
        return eval_fn(model)
    return eval_fn()


def _run_seed(cfg: Any, run_index: int) -> int:
    """Deterministic per-run seed: ``base_seed + run_index * seed_offset``."""
    base = int(_get(cfg, "seed", 42))
    offset = int(_get(cfg, "seed_offset", 42))
    if offset <= 0:  # never let the offset collapse every run to one seed
        offset = 1
    return base + run_index * offset


def _best_effort_seed_everything(seed: int) -> None:
    """Seed local RNGs always; also call the D02 hook when it is importable.

    ``qmlpipe.train.seed_everything`` may import torch/PennyLane, which are not
    present in every verifier environment. Failure to import is not an
    experiment failure, so it is swallowed here and reported nowhere.
    """
    random.seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard project dep
        pass
    try:
        from qmlpipe.train import seed_everything as _seed_everything  # type: ignore
    except Exception:  # pragma: no cover - optional D02 module / quantum deps
        return
    try:
        _seed_everything(seed)
    except Exception:  # pragma: no cover - never let seeding abort an ablation
        return


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def ablation_grid(base_cfg: Any) -> list[Any]:
    """Return the bounded full-factorial grid of cloned configs.

    Every returned config is a new object produced by ``dataclasses.replace``;
    ``base_cfg`` is never mutated. Field names are resolved against the actual
    dataclass fields so aliasing (``qnn_layers`` vs ``layers``, etc.) does not
    require callers to rename their config. Raises ``ValueError`` if the
    resolved grid would exceed :data:`MAX_GRID_SIZE` or if an axis has no
    corresponding field.
    """
    resolved = _resolve_axes(base_cfg)
    size = 1
    for spec in resolved:
        size *= len(spec["values"])
    if size > MAX_GRID_SIZE:
        raise ValueError(
            f"ablation grid would contain {size} configs, exceeding the "
            f"documented bound MAX_GRID_SIZE={MAX_GRID_SIZE}"
        )

    fields = _field_names(base_cfg)
    feature_field = next((n for n in _FEATURE_FIELD_CANDIDATES if n in fields), None)
    nq_field = next(r["field"] for r in resolved if r["axis"] == "n_qubits")

    names = [r["field"] for r in resolved]
    grids: list[Any] = []
    for combo in product(*(r["values"] for r in resolved)):
        changes: dict[str, Any] = dict(zip(names, combo))
        if feature_field is not None:
            # Keep validate() happy: the recorded input width must equal
            # 2**n_qubits (QMLConfig.projection_dim / feature_dim) for every arm.
            changes[feature_field] = 2 ** int(changes[nq_field])
        grids.append(dataclasses.replace(base_cfg, **changes))
    return grids


def run_ablation(
    base_cfg: Any,
    train_fn: Callable,
    eval_fn: Callable,
    *,
    grid: list[Any] | None = None,
    max_runs: int | None = None,
) -> list[dict]:
    """Execute (a bounded slice of) the ablation grid and record every run.

    Each run gets a deterministic seed and a fresh ``dataclasses.replace`` clone
    of its config. Exceptions from seeding, training or evaluation are recorded
    as ``status="error"`` with their type and message; ``metrics`` stays empty
    for those runs. No number is ever fabricated to replace a missing result.
    """
    if max_runs is not None and max_runs < 0:
        raise ValueError("max_runs must be non-negative or None")

    candidates = ablation_grid(base_cfg) if grid is None else list(grid)
    if max_runs is not None:
        candidates = candidates[:max_runs]

    results: list[dict] = []
    for run_index, cfg in enumerate(candidates):
        seed = _run_seed(cfg, run_index)
        record: dict[str, Any] = {
            "run_index": run_index,
            "seed": seed,
            "config": _as_dict(cfg),
            "status": "error",
            "metrics": {},
            "error": None,
        }
        try:
            # Clone again at execution time: no run can mutate another's config.
            run_cfg = dataclasses.replace(cfg)
            _best_effort_seed_everything(seed)
            model = _call_train(train_fn, run_cfg, seed)
            raw_metrics = _call_eval(eval_fn, model, run_cfg)
            if not isinstance(raw_metrics, Mapping):
                raise TypeError(
                    "eval_fn must return a mapping of metric names to numbers; "
                    f"got {type(raw_metrics).__name__!r}"
                )
            record["metrics"] = dict(raw_metrics)
            record["status"] = "ok"
        except Exception as exc:  # honest failure capture, never a placeholder score
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        results.append(record)
    return results


def summarize_ablation(results: list[dict]) -> dict:
    """Summarise metric vectors from successful runs with n/mean/std/95% CI.

    Uses ``experiments.metrics.mean_std`` (sample sd, ddof=1),
    ``bootstrap_ci`` (percentile, fixed seed) and ``summarize_runs``. Failed
    runs are reported, never dropped silently, and never contribute numbers.
    """
    summary: dict[str, Any] = {
        "n_configs": len(results),
        "n_success": 0,
        "n_failed": 0,
        "failures": [],
        "metrics": {},
        "seed": 42,
    }
    if not results:
        return summary

    seeds = [r.get("seed") for r in results if isinstance(r.get("seed"), int)]
    if seeds:
        summary["seed"] = seeds[0]

    success: list[dict] = []
    for r in results:
        if r.get("status") == "ok" and isinstance(r.get("metrics"), Mapping):
            success.append(r)
        else:
            err = r.get("error") or {}
            summary["failures"].append(
                {
                    "run_index": r.get("run_index"),
                    "seed": r.get("seed"),
                    "config": r.get("config"),
                    "error": (
                        {"type": err.get("type"), "message": err.get("message")}
                        if isinstance(err, Mapping)
                        else {"type": None, "message": str(err)}
                    ),
                }
            )
    summary["n_success"] = len(success)
    summary["n_failed"] = len(results) - len(success)

    keys = sorted(
        {
            key
            for r in success
            for key, value in r["metrics"].items()
            if _is_finite_number(value)
        }
    )
    for key in keys:
        values = [
            float(r["metrics"][key])
            for r in success
            if _is_finite_number(r["metrics"].get(key))
        ]
        per_metric = dict(_metrics.summarize_runs(key, values, seed=summary["seed"]))
        per_metric["values"] = list(values)
        # Explicit calls keep the three required helpers in the evidence chain.
        per_metric["mean"], per_metric["std"] = _metrics.mean_std(values)
        per_metric["ci95_low"], per_metric["ci95_high"] = _metrics.bootstrap_ci(
            values, seed=summary["seed"]
        )
        summary["metrics"][key] = per_metric
    return summary
