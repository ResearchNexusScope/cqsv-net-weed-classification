"""Auditable, opt-in numeric synchronisation for the private manuscript.

This module never writes a number of its own. It replaces only tokens that a
caller has explicitly registered in a ``mapping`` (placeholder -> dotted
result key), and only with values read from the supplied results dict. There
is no regex that "finds numbers in prose", no inference, and no defaults that
silently edit text.

Safety rules enforced here:

* :func:`plan_sync` is read-only and returns a list of :class:`SyncDiff`.
* :func:`apply_sync` is a no-op unless ``apply=True`` **and** an explicit
  ``mapping`` is supplied.
* :func:`sync_file` additionally refuses to write inside a ``.paper`` directory
  unless ``allow_private=True``.
* A missing/null result key raises :class:`MissingResultKey`; values are never
  fabricated to fill a gap.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .tables import resolve_metric_path


class MissingResultKey(KeyError):
    """Raised when a registered placeholder's source key is absent or null."""


class UnsafeSyncError(ValueError):
    """Raised when a caller tries to edit without the required opt-in."""


@dataclass(frozen=True)
class Placeholder:
    """Registration for one manuscript token.

    ``token`` is the exact string in the Markdown (e.g. ``{{QSVC_ACC_MEAN}}``).
    ``source_key`` is a dotted path into the results dict. ``format_spec`` is
    an optional Python format spec applied to the value. ``template`` optionally
    wraps the formatted value in a caller-supplied claim sentence using
    ``{value}``; the number still comes only from the results dict.
    """

    token: str
    source_key: str
    format_spec: str | None = None
    template: str | None = None
    claim_id: str = ""

    def render(self, value: Any) -> str:
        """Format a value read from results into its replacement text.

        Both ``":.2f"`` and ``".2f"`` are accepted for ``format_spec``; a
        leading colon is stripped before passing the spec to ``format()``.
        """
        spec = self.format_spec.lstrip(":") if self.format_spec else ""
        formatted = format(value, spec) if spec else str(value)
        if self.template is not None:
            return self.template.format(value=formatted)
        return formatted


@dataclass(frozen=True)
class SyncDiff:
    """One proposed replacement, with offsets into the original text."""

    token: str
    source_key: str
    claim_id: str
    old: str
    new: str
    start: int
    end: int

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of this proposed edit."""
        return {
            "token": self.token,
            "source_key": self.source_key,
            "claim_id": self.claim_id,
            "old": self.old,
            "new": self.new,
            "start": self.start,
            "end": self.end,
        }


# Structural registrations only: tokens and key paths, never result values.
# Callers may pass their own mapping instead of (or in addition to) these.
DEFAULT_PLACEHOLDERS: tuple[Placeholder, ...] = (
    Placeholder("{{QSVC_ACC_MEAN}}", "aggregates.qsvc_quantum_kernel.accuracy.mean", ":.3f",
                claim_id="qsvc_mean"),
    Placeholder("{{QSVC_ACC_STD}}", "aggregates.qsvc_quantum_kernel.accuracy.std", ":.3f",
                claim_id="qsvc_std"),
    Placeholder("{{QSVC_ACC_N}}", "aggregates.qsvc_quantum_kernel.accuracy.n", None,
                claim_id="qsvc_n"),
    Placeholder("{{QSVC_CI_LOW}}", "aggregates.qsvc_quantum_kernel.accuracy.ci95_low", ":.3f",
                claim_id="qsvc_ci"),
    Placeholder("{{QSVC_CI_HIGH}}", "aggregates.qsvc_quantum_kernel.accuracy.ci95_high", ":.3f",
                claim_id="qsvc_ci"),
)


def _normalise(mapping: Mapping[Any, Any] | Sequence[Any] | None) -> tuple[Placeholder, ...]:
    """Turn a caller mapping into an ordered tuple of :class:`Placeholder`.

    Accepts a ``{token: dotted_key}`` dict, a ``{token: Placeholder}`` dict,
    a sequence of :class:`Placeholder`, or ``None`` (the default registry).
    Ordering is by token so output is deterministic regardless of dict order.
    """
    if mapping is None:
        registrations = DEFAULT_PLACEHOLDERS
    elif isinstance(mapping, Mapping):
        registrations = tuple(
            spec if isinstance(spec, Placeholder)
            else Placeholder(str(token), str(spec))
            for token, spec in mapping.items()
        )
    elif isinstance(mapping, Sequence) and not isinstance(mapping, (str, bytes)):
        registrations = tuple(mapping)
    else:
        raise UnsafeSyncError("mapping must be a dict, a sequence of Placeholder, or None")
    for spec in registrations:
        if not isinstance(spec, Placeholder):
            raise UnsafeSyncError("every registration must be a Placeholder instance")
        if not spec.token or not spec.source_key:
            raise UnsafeSyncError("placeholder token and source_key must both be non-empty")
    return tuple(sorted(registrations, key=lambda spec: spec.token))


def _occurrences(text: str, token: str) -> list[int]:
    """Return every start index of ``token`` in ``text`` (non-overlapping)."""
    found: list[int] = []
    start = 0
    while True:
        index = text.find(token, start)
        if index < 0:
            return found
        found.append(index)
        start = index + len(token)


def plan_sync(
    md_text: str,
    results: Mapping[str, Any],
    mapping: Mapping[Any, Any] | Sequence[Any] | None = None,
) -> list[SyncDiff]:
    """Propose replacements without editing anything.

    Every token in ``mapping`` that occurs in ``md_text`` is resolved against
    ``results``. If a source key is missing or null, :class:`MissingResultKey`
    is raised; no value is ever guessed. Tokens not present in the text are
    simply not proposed, and text not covered by the mapping is never touched.

    The shared key resolver accepts the runner's namespaced aggregate keys
    (``experiment/model``), so registered flat keys keep working on real
    runner-shaped metrics.
    """
    diffs: list[SyncDiff] = []
    for spec in _normalise(mapping):
        for start in _occurrences(md_text, spec.token):
            try:
                value = resolve_metric_path(results, spec.source_key)
            except KeyError as exc:
                raise MissingResultKey(
                    f"placeholder {spec.token!r} needs results key "
                    f"{spec.source_key!r}; it is missing (path: {exc.args[0]!r})"
                ) from exc
            if value is None:
                raise MissingResultKey(
                    f"placeholder {spec.token!r} needs results key "
                    f"{spec.source_key!r}; it is null and will not be invented"
                )
            diffs.append(
                SyncDiff(
                    token=spec.token,
                    source_key=spec.source_key,
                    claim_id=spec.claim_id,
                    old=spec.token,
                    new=spec.render(value),
                    start=start,
                    end=start + len(spec.token),
                )
            )
    return diffs


def apply_sync(
    md_text: str,
    results: Mapping[str, Any],
    mapping: Mapping[Any, Any] | Sequence[Any] | None = None,
    *,
    apply: bool = False,
) -> str:
    """Apply registered replacements to Markdown text, strictly opt-in.

    With the default ``apply=False`` the input text is returned byte-identical
    (dry-run), but :func:`plan_sync` is still called first so missing or null
    source keys are reported instead of being silently skipped. Setting
    ``apply=True`` requires an explicit ``mapping``; using the implicit default
    registry in an edit is rejected to prevent accidental manuscript changes.
    Only exact registered tokens are replaced.
    """
    if not apply:
        plan_sync(md_text, results, mapping)
        return md_text
    if mapping is None:
        raise UnsafeSyncError(
            "apply_sync(apply=True) requires an explicit mapping; "
            "do not rely on the default registry for manuscript edits"
        )
    result = md_text
    for diff in plan_sync(md_text, results, mapping):
        result = result.replace(diff.old, diff.new)
    for spec in _normalise(mapping):
        if spec.token in result:
            raise UnsafeSyncError(f"registered token {spec.token!r} was not fully replaced")
    return result


def sync_file(
    md_path: str | Path,
    results: Mapping[str, Any],
    mapping: Mapping[Any, Any] | Sequence[Any] | None = None,
    *,
    apply: bool = False,
    allow_private: bool = False,
) -> tuple[str, list[SyncDiff]]:
    """Read a Markdown file, plan edits, and write only with strict opt-in.

    Returns ``(text, diffs)`` where ``text`` is the would-be (dry-run) or
    written content. A path inside a ``.paper`` directory is refused unless
    ``allow_private=True``; no network access or file transmission occurs.
    """
    path = Path(md_path)
    text = path.read_text(encoding="utf-8")
    diffs = plan_sync(text, results, mapping)
    if not apply:
        return text, diffs
    if mapping is None:
        raise UnsafeSyncError("sync_file(apply=True) requires an explicit mapping")
    if ".paper" in path.parts and not allow_private:
        raise UnsafeSyncError(
            "refusing to write inside a .paper directory without allow_private=True"
        )
    updated = apply_sync(text, results, mapping, apply=True)
    path.write_text(updated, encoding="utf-8")
    return updated, diffs
