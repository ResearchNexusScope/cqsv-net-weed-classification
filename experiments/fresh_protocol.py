"""Fresh, properly powered classical protocol on grouped dataset splits (FRESH-D02).

This module is deliberately classical-only: it establishes matched control arms
(SVM poly-2, SVM RBF, logistic regression) over frozen backbone features on a
fresh 70/20/10 grouped split protocol, with leak-free feature caching and a
pre-registered paired-statistics family.  A QNN arm is a SEPARATE follow-up
stage and is neither implemented nor claimed here.

Design rules (docs/METHODOLOGY_V2.md §4-§7):

* source-image groups are disjoint across train/val/test for every seed;
* frozen features are computed once per (dataset, backbone) and cached on disk,
  then re-used to assemble per-seed splits -- no feature crosses a boundary;
* each arm fits on TRAIN only and scores TEST once;
* every aggregate carries n, the full value vector, ddof=1 SD and a seeded
  percentile bootstrap 95% CI; comparisons are paired, Holm-corrected and
  called ``inconclusive`` whenever n < 10 or achieved power < 0.8;
* failed runs are recorded as warnings, never silently dropped or imputed.

Only the standard library and numpy are imported at module scope.  scipy,
scikit-learn, statsmodels, torch and torchvision are imported lazily so that the
statistics helpers remain usable in a lightweight verifier environment.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_RATIOS",
    "aggregate",
    "paired_comparison",
    "holm_adjust",
    "cache_features",
    "run_svm_arm",
    "run_protocol",
]

CLAIM_BOUNDARY = (
    "This stage measures only classical control arms on freshly grouped splits. "
    "No quantum kernel or QNN is implemented here and no quantum advantage is "
    "asserted; QNN arms belong to a separate follow-up stage. "
    "Every reported value comes from an executed run; failures are recorded as "
    "warnings and never fabricated."
)

DEFAULT_RATIOS: tuple[float, float, float] = (0.7, 0.2, 0.1)
DEFAULT_SEED = 42
BOOT_B = 2000
BOOT_SEED = 42
MIN_N = 10
MIN_POWER = 0.8
ALPHA = 0.05
_METRICS = ("accuracy", "auc", "f1", "precision", "recall")
_MODEL_ALIASES = {"poly2": "poly", "poly": "poly", "rbf": "rbf", "logreg": "logreg"}


# --------------------------------------------------------------------------- #
# JSON helpers                                                                 #
# --------------------------------------------------------------------------- #
def _json_safe(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and non-finite floats for JSON."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value)]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    """Atomically write a JSON artifact (additive: only the given path is touched)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _resolve_output_paths(output_dir: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Return ``(metrics.json, provenance.json)`` without overwriting existing ones."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path, provenance_path = out / "metrics.json", out / "provenance.json"
    if metrics_path.exists() or provenance_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = out / f"fresh_protocol_{stamp}"
        suffix = 1
        while candidate.exists():
            suffix += 1
            candidate = out / f"fresh_protocol_{stamp}_{suffix}"
        candidate.mkdir(parents=True, exist_ok=True)
        metrics_path, provenance_path = candidate / "metrics.json", candidate / "provenance.json"
    return metrics_path, provenance_path


# --------------------------------------------------------------------------- #
# Statistics                                                                   #
# --------------------------------------------------------------------------- #
def aggregate(
    values: Sequence[float],
    seed: int = BOOT_SEED,
    n_boot: int = BOOT_B,
    name: str | None = None,
) -> dict:
    """Summarise a run vector with n, mean, ddof=1 SD and a seeded 95% CI.

    The percentile bootstrap resamples ``values`` with replacement ``n_boot``
    times using ``numpy.random.default_rng(seed)`` and reports the 2.5th and
    the 97.5th percentile of the resampled means.  ``boot_B`` and ``boot_seed``
    are always returned so the interval is reproducible.  Values are returned
    verbatim; nothing is rounded or fabricated.
    """
    vector = [float(value) for value in values]
    if not vector:
        raise ValueError("aggregate requires a non-empty run vector")
    array = np.asarray(vector, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(vector) > 1 else 0.0
    rng = np.random.default_rng(int(seed))
    resampled_means = np.asarray(
        [rng.choice(array, size=len(vector), replace=True).mean() for _ in range(int(n_boot))],
        dtype=float,
    )
    low, high = np.percentile(resampled_means, [2.5, 97.5])
    result: dict[str, Any] = {
        "n": len(vector),
        "values": vector,
        "mean": mean,
        "std": std,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "boot_B": int(n_boot),
        "boot_seed": int(seed),
    }
    if name is not None:
        result["model"] = str(name)
    return result


def _achieved_power(dz: float | None, n: int, alpha: float) -> float | None:
    """Achieved power of a two-sided paired t-test at the observed dz.

    statsmodels is optional; when absent (or when it returns a non-finite
    number) ``None`` is returned so significance can never be asserted.
    """
    if dz is None or int(n) < 2:
        return None
    try:
        from statsmodels.stats.power import TTestPower
    except ImportError:
        return None
    try:
        power = float(
            TTestPower().power(
                effect_size=abs(float(dz)),
                nobs=int(n),
                alpha=float(alpha),
                alternative="two-sided",
            )
        )
    except Exception:  # pragma: no cover - version/degenerate-effect dependent
        return None
    return power if np.isfinite(power) else None


def paired_comparison(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float = 0.05,
    *,
    seed: int = BOOT_SEED,
    n_boot: int = BOOT_B,
    pair: Sequence[str] | None = None,
) -> dict:
    """One paired comparison of two seed-matched run vectors.

    Primary test is ``scipy.stats.ttest_rel`` (two-sided); the Wilcoxon
    signed-rank test is reported as a secondary test.  Cohen's ``dz``, the
    achieved power and a bootstrap CI of the paired differences are recorded.
    ``holm_adjust`` must be called on the whole family to assign ``p_holm`` and
    finalise the decision.  The decision is ``inconclusive`` whenever n < 10,
    power < 0.8, n < 2, the CI includes zero, or Holm p >= alpha.  No
    significance claim is made from n < 2.
    """
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    if first.shape != second.shape:
        raise ValueError("paired_comparison requires equal-length vectors")
    n = int(first.size)
    base: dict[str, Any] = {
        "pair": [str(name) for name in pair] if pair is not None else None,
        "primary_test": "paired_t_test (scipy.stats.ttest_rel, two-sided)",
        "n": n,
        "df": max(0, n - 1),
        "p_raw": None,
        "p_holm": None,
        "t_statistic": None,
        "wilcoxon_p": None,
        "wilcoxon_note": None,
        "mean_difference": None,
        "cohens_dz": None,
        "power_at_observed_dz": None,
        "ci95_low": None,
        "ci95_high": None,
        "boot_B": int(n_boot),
        "boot_seed": int(seed),
        "alpha": float(alpha),
    }
    if n < 2:
        base.update(
            {
                "secondary_tests": {"wilcoxon_signed_rank": {"p": None}},
                "decision": "inconclusive",
                "decision_reason": "n<2: no significance claim",
            }
        )
        return base

    from scipy import stats as scipy_stats  # lazy: scipy is not needed for aggregate()

    diff = first - second
    t_statistic, p_raw = scipy_stats.ttest_rel(first, second, alternative="two-sided")
    wilcoxon_p: float | None = None
    wilcoxon_note: str | None = None
    try:
        wilcoxon_p = float(scipy_stats.wilcoxon(diff, alternative="two-sided").pvalue)
    except Exception as exc:  # e.g. all-zero differences
        wilcoxon_note = f"unavailable: {type(exc).__name__}"

    spread = float(diff.std(ddof=1))
    dz = 0.0 if spread == 0.0 else float(diff.mean() / spread)
    ci = aggregate(diff, seed=seed, n_boot=n_boot)
    power = _achieved_power(dz, n, alpha)

    reasons: list[str] = []
    if n < MIN_N:
        reasons.append("n<10")
    if power is None or float(power) < MIN_POWER:
        reasons.append("power<0.8 or unavailable")
    base.update(
        {
            "p_raw": float(p_raw),
            "t_statistic": float(t_statistic),
            "wilcoxon_p": wilcoxon_p,
            "wilcoxon_note": wilcoxon_note,
            "mean_difference": float(diff.mean()),
            "cohens_dz": dz,
            "power_at_observed_dz": power,
            "ci95_low": ci["ci95_low"],
            "ci95_high": ci["ci95_high"],
            "secondary_tests": {
                "wilcoxon_signed_rank": {"p": wilcoxon_p, "note": wilcoxon_note}
            },
            "decision": "inconclusive",
            "decision_reason": (
                "descriptive: " + ", ".join(reasons) if reasons else "pending Holm correction"
            ),
        }
    )
    return base


def holm_adjust(comparisons: Sequence[dict], alpha: float = 0.05) -> list[dict]:
    """Apply Holm-Bonferroni to a family and finalise each decision in place.

    The step-down adjusted p-value for rank ``j`` (0-based, ascending raw p)
    among ``m`` finite tests is ``max_{i<=j} (m-i) * p_(i)``, capped at 1.
    Tests with a missing/non-finite raw p receive ``p_holm = None`` and remain
    inconclusive.  A ``significant`` decision additionally requires n >= 10,
    achieved power >= 0.8 and a bootstrap CI of the difference excluding zero.
    """
    items = comparisons if isinstance(comparisons, list) else list(comparisons)
    valid = [
        index
        for index, item in enumerate(items)
        if item.get("p_raw") is not None and np.isfinite(float(item["p_raw"]))
    ]
    m = len(valid)
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, index in enumerate(sorted(valid, key=lambda i: float(items[i]["p_raw"]))):
        running = max(running, (m - rank) * float(items[index]["p_raw"]))
        adjusted[index] = min(1.0, running)

    for index, item in enumerate(items):
        item["p_holm"] = adjusted.get(index)
        item["multiplicity"] = {"method": "holm", "m": m, "alpha": float(alpha)}
        failing: list[str] = []
        n = int(item.get("n", 0) or 0)
        if n < 2:
            failing.append("n<2")
        elif n < MIN_N:
            failing.append("n<10")
        power = item.get("power_at_observed_dz")
        if power is None or float(power) < MIN_POWER:
            failing.append("power<0.8 or unavailable")
        p_holm = item.get("p_holm")
        if p_holm is None or not (float(p_holm) < float(alpha)):
            failing.append("p_holm>=alpha")
        low, high = item.get("ci95_low"), item.get("ci95_high")
        ci_excludes_zero = (
            low is not None and high is not None and (float(low) > 0.0 or float(high) < 0.0)
        )
        if not ci_excludes_zero:
            failing.append("bootstrap CI includes 0")
        if failing:
            item["decision"] = "inconclusive"
            item["decision_reason"] = "descriptive: " + ", ".join(failing)
        else:
            item["decision"] = "significant"
            item["decision_reason"] = (
                "p_holm<alpha, bootstrap CI excludes 0, power>=0.8, n>=10"
            )
    return items


# --------------------------------------------------------------------------- #
# Leak-free grouped splits and frozen feature cache                            #
# --------------------------------------------------------------------------- #
def _fresh_module():
    """Import the peer's split module by name; never edit or shadow it."""
    try:
        from data import fresh_splits
    except ImportError as exc:  # pragma: no cover - depends on peer delivery
        raise RuntimeError(
            "data.fresh_splits is required for the fresh protocol; it is owned by "
            "worker FRESH-D01 and must expose prepare_dataset/grouped_split/split_provenance"
        ) from exc
    return fresh_splits


def _group_id(sample: Any) -> str:
    """Source-image identity from a duck-typed FreshSample (with fallbacks)."""
    for attribute in ("source_image_id", "group_id"):
        value = getattr(sample, attribute, None)
        if value is not None:
            return str(value)
    return str(getattr(sample, "sample_id"))


def _sample_groups(splits: Mapping[str, Iterable[Any]]) -> dict[str, set[str]]:
    """Derive source-image group sets from duck-typed FreshSample objects."""
    return {
        str(name): {_group_id(sample) for sample in samples}
        for name, samples in splits.items()
    }


def _recorded_group_sets(
    prov: Any, split: str, sample_to_group: Mapping[str, str]
) -> list[set[str]]:
    """All plausible group sets recorded for ``split`` in split_provenance.

    Provenance may store group ids directly or sample ids that can be mapped
    back to groups; every interpretation is returned so the caller can confirm
    at least one agrees with the samples.
    """
    if not isinstance(prov, Mapping):
        return []
    candidates: list[Any] = []
    splits_map = prov.get("splits")
    if isinstance(splits_map, Mapping) and split in splits_map:
        node = splits_map[split]
        if isinstance(node, Mapping):
            for key in ("groups", "source_image_ids", "source_ids", "group_ids", "sample_ids", "samples"):
                if key in node:
                    candidates.append(node[key])
        elif isinstance(node, (list, tuple, set)):
            candidates.append(node)
    for key in (
        f"{split}_groups",
        f"{split}_source_image_ids",
        f"{split}_group_ids",
        f"{split}_sample_ids",
        split,
    ):
        if key in prov:
            candidates.append(prov[key])
    group_sets = prov.get("group_sets")
    if isinstance(group_sets, Mapping) and split in group_sets:
        candidates.append(group_sets[split])

    interpretations: list[set[str]] = []
    for candidate in candidates:
        if isinstance(candidate, (Mapping, str, bytes)):
            continue
        try:
            values = list(candidate)
        except TypeError:
            continue
        direct = {str(value) for value in values}
        if direct:
            interpretations.append(direct)
        mapped = {
            sample_to_group[str(getattr(value, "sample_id", value))]
            for value in values
            if str(getattr(value, "sample_id", value)) in sample_to_group
        }
        if mapped:
            interpretations.append(mapped)
    return interpretations


def _assert_split_disjoint(splits: Mapping[str, Iterable[Any]], prov: Any) -> dict[str, set[str]]:
    """Raise on any source-image group shared by two splits; cross-check provenance."""
    groups = _sample_groups(splits)
    sample_to_group = {
        str(sample.sample_id): _group_id(sample)
        for samples in splits.values()
        for sample in samples
    }
    for left, right in itertools.combinations(list(groups), 2):
        overlap = groups[left] & groups[right]
        if overlap:
            example = sorted(overlap)[0]
            raise ValueError(
                f"source-image group {example!r} appears in both {left!r} and {right!r}; "
                "split is not group-disjoint"
            )
    if isinstance(prov, Mapping) and prov.get("group_disjoint") is False:
        raise ValueError("split_provenance reports group_disjoint=False")
    for name, derived in groups.items():
        recorded = _recorded_group_sets(prov, name, sample_to_group)
        if recorded and not any(candidate == derived for candidate in recorded):
            raise ValueError(
                f"split_provenance groups for {name!r} disagree with the samples; "
                "refusing to featurise"
            )
    return groups


def _encode_bbox(sample: Any) -> str:
    bbox = getattr(sample, "bbox", None)
    if not bbox:
        return ""
    return ",".join(str(int(value)) for value in bbox)


def _cache_paths(dataset: str, backbone: str, root: str | os.PathLike[str] | None, cache_dir):
    base = (
        Path(cache_dir)
        if cache_dir is not None
        else Path(tempfile.gettempdir()) / "qc_fresh_features"
    )
    root_key = hashlib.sha256(
        str(root if root is not None else "default").encode("utf-8")
    ).hexdigest()[:12]
    stem = f"{dataset}_{backbone}_{root_key}"
    return base / f"{stem}.npz", base / f"{stem}.provenance.json"


def _load_cache(npz_path: Path, samples: Sequence[Any]):
    """Return a (sample_id, bbox) -> feature lookup, or (None, missing)."""
    with np.load(npz_path, allow_pickle=False) as data:
        ids = [str(value) for value in data["sample_ids"].tolist()]
        boxes = [str(value) for value in data["bboxes"].tolist()]
        features = np.asarray(data["features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(ids):
        raise ValueError("feature cache shape is inconsistent")
    lookup = {(key, box): features[index] for index, (key, box) in enumerate(zip(ids, boxes))}
    missing = [sample for sample in samples if (str(sample.sample_id), _encode_bbox(sample)) not in lookup]
    return lookup, missing


def cache_features(
    dataset: str,
    backbone: str,
    root: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    force: bool = False,
) -> dict:
    """Compute/reuse frozen features for a fresh grouped 70/20/10 split.

    Features are computed for *all* samples once per (dataset, backbone, root)
    and stored at a deterministic cache path; subsequent calls with different
    split seeds re-assemble train/val/test arrays from that cache without
    re-extracting.  Group disjointness is asserted with ``split_provenance``
    BEFORE any image is featurised, so no feature can cross a split boundary.

    Returns a dict with ``X``/``y`` (per-split arrays), ``X_train``/``y_train``
    /``X_test``/``y_test`` aliases, ``sample_ids``, ``groups``, ``cache_path``,
    ``cache_hit`` and ``provenance`` (split + cache + disjointness evidence).
    """
    fresh = _fresh_module()
    samples = list(fresh.prepare_dataset(dataset, root=root))
    splits = {
        str(name): list(items)
        for name, items in dict(
            fresh.grouped_split(samples, seed=int(seed), ratios=tuple(ratios))
        ).items()
    }
    split_prov = fresh.split_provenance(dataset, samples, splits, int(seed), tuple(ratios))
    groups = _assert_split_disjoint(splits, split_prov)  # before featurising

    npz_path, prov_path = _cache_paths(dataset, backbone, root, cache_dir)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    lookup = None
    cache_hit = False
    if npz_path.exists() and not force:
        try:
            lookup, missing = _load_cache(npz_path, samples)
            if not missing:
                cache_hit = True
            else:
                lookup = None
        except Exception:  # corrupt/foreign cache -> recompute
            lookup = None

    if lookup is None:
        from scripts.run_real_benchmark import make_feature_loader
        from qmlpipe.plant_backbone import build_backbone_extractor
        from qmlpipe.config import QMLConfig

        try:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:  # pragma: no cover - lightweight verifier
            device = "cpu"

        extractor, preprocess = build_backbone_extractor(
            backbone, QMLConfig(backbone=backbone)
        )
        loader = make_feature_loader(extractor, preprocess, device)
        features = np.asarray(loader(samples, None, "all"), dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(len(samples), -1)
        if features.ndim != 2 or features.shape[0] != len(samples):
            raise ValueError(
                f"feature loader returned shape {features.shape} for {len(samples)} samples"
            )
        ids = np.asarray([str(sample.sample_id) for sample in samples], dtype=str)
        boxes = np.asarray([_encode_bbox(sample) for sample in samples], dtype=str)
        np.savez_compressed(
            npz_path,
            features=features,
            sample_ids=ids,
            bboxes=boxes,
            dataset=np.asarray(str(dataset)),
            backbone=np.asarray(str(backbone)),
            root=np.asarray(str(root)),
        )
        cache_prov = {
            "cache_path": str(npz_path),
            "dataset": str(dataset),
            "backbone": str(backbone),
            "root": str(root),
            "n_samples": int(len(samples)),
            "feature_dim": int(features.shape[1]),
            "feature_hash": hashlib.sha256(features.tobytes()).hexdigest(),
            "frozen": True,
            "note": (
                "Frozen L2-normalised backbone features; extracted once for all samples "
                "and assigned to splits only after the group-disjointness assertion."
            ),
        }
        _write_json(prov_path, cache_prov)
        lookup = {
            (key, box): features[index]
            for index, (key, box) in enumerate(zip(ids.tolist(), boxes.tolist()))
        }
    else:
        if prov_path.exists():
            cache_prov = json.loads(prov_path.read_text(encoding="utf-8"))
        else:  # pragma: no cover - cache created by an older run
            cache_prov = {"cache_path": str(npz_path), "note": "reused cache without metadata"}
        cache_prov = dict(cache_prov)
        cache_prov["reused"] = True

    X: dict[str, np.ndarray] = {}
    y: dict[str, np.ndarray] = {}
    sample_ids: dict[str, list[str]] = {}
    for name, items in splits.items():
        keys = [(str(sample.sample_id), _encode_bbox(sample)) for sample in items]
        absent = [key for key in keys if key not in lookup]
        if absent:
            raise ValueError(
                f"cached features are missing for split {name!r}: {absent[:3]} "
                "(refusing to fabricate features)"
            )
        X[name] = np.stack([lookup[key] for key in keys]).astype(np.float32, copy=False)
        y[name] = np.asarray([int(sample.label) for sample in items], dtype=int)
        sample_ids[name] = [key[0] for key in keys]

    result: dict[str, Any] = {
        "dataset": str(dataset),
        "backbone": str(backbone),
        "seed": int(seed),
        "ratios": [float(value) for value in ratios],
        "X": X,
        "y": y,
        "sample_ids": sample_ids,
        "groups": {name: sorted(value) for name, value in groups.items()},
        "cache_path": str(npz_path),
        "cache_hit": bool(cache_hit),
        "provenance": {
            "split": _json_safe(split_prov),
            "cache": _json_safe(cache_prov),
            "group_disjoint": True,
            "groups": {name: sorted(value) for name, value in groups.items()},
        },
        "split_provenance": _json_safe(split_prov),
    }
    for name in ("train", "val", "test"):
        if name in X:
            result[f"X_{name}"] = X[name]
            result[f"y_{name}"] = y[name]
    return result


# --------------------------------------------------------------------------- #
# Classical arms                                                               #
# --------------------------------------------------------------------------- #
def _make_estimator(kernel: str, seed: int):
    """Build a fresh sklearn Pipeline; preprocessing is fitted on TRAIN only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if kernel == "poly":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="poly", degree=2, C=1.0, gamma="scale", random_state=int(seed)),
        )
    if kernel == "rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", random_state=int(seed)),
        )
    if kernel == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=int(seed)),
        )
    raise ValueError(f"unknown classical arm {kernel!r}; expected poly2/poly/rbf/logreg")


def run_svm_arm(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    kernel: str,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Fit one classical arm on TRAIN only and score TEST once.

    ``kernel`` accepts ``poly2``/``poly`` (degree-2 polynomial SVC), ``rbf``
    (RBF SVC) and ``logreg`` (logistic-regression baseline).  Reported metrics
    are accuracy, AUC, F1, precision, recall, a 2x2 confusion matrix and n.
    AUC uses ``decision_function`` for SVMs and ``predict_proba`` for logreg;
    no threshold is tuned on the test split.
    """
    name = _MODEL_ALIASES.get(str(kernel).lower())
    if name is None:
        raise ValueError(f"unknown classical arm {kernel!r}; expected poly2/poly/rbf/logreg")
    from experiments.metrics import binary_metrics  # lazy: imports scipy + sklearn

    X_fit = np.asarray(X_train)
    X_score = np.asarray(X_test)
    y_fit = np.asarray(y_train, dtype=int)
    y_true = np.asarray(y_test, dtype=int)
    if X_fit.ndim != 2 or X_score.ndim != 2 or X_fit.shape[1] != X_score.shape[1]:
        raise ValueError("train/test feature matrices must be 2-D with equal width")
    if len(y_fit) != len(X_fit) or len(y_true) != len(X_score):
        raise ValueError("label and feature counts disagree")

    estimator = _make_estimator(name, int(seed))
    estimator.fit(X_fit, y_fit)  # TRAIN only
    y_pred = estimator.predict(X_score)  # TEST scored exactly once
    if name == "logreg":
        score = estimator.predict_proba(X_score)[:, 1]
        score_kind = "predict_proba"
    else:
        score = estimator.decision_function(X_score)
        score_kind = "decision_function"

    metrics = binary_metrics(y_true, y_pred, y_prob=score)
    return {
        "model": name,
        "requested_model": str(kernel),
        "kernel": name,
        "seed": int(seed),
        "accuracy": float(metrics.accuracy),
        "auc": None if metrics.auc is None else float(metrics.auc),
        "f1": float(metrics.f1),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "confusion_matrix": metrics.confusion.tolist(),
        "n": int(metrics.support),
        "n_train": int(len(y_fit)),
        "score_kind": score_kind,
        "iou_class0": float(metrics.iou_class0),
        "iou_class1": float(metrics.iou_class1),
        "macro_iou": float(metrics.macro_iou),
    }


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def _normalize_seeds(seeds: int | Sequence[int]) -> list[int]:
    if isinstance(seeds, int):
        result = list(range(int(seeds)))
    else:
        result = [int(value) for value in seeds]
    return result or [DEFAULT_SEED]


def _environment_pins() -> dict:
    """Record interpreter/platform and installed package versions (no network)."""
    from importlib.metadata import version

    def pinned(name: str):
        try:
            return version(name)
        except Exception:
            return None

    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "commit_hash": os.environ.get("GIT_COMMIT_HASH"),
        "packages": {
            name: pinned(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-learn",
                "statsmodels",
                "torch",
                "torchvision",
                "Pillow",
            )
        },
        "note": "versions are those importable in the executing interpreter; no installs performed",
    }


def run_protocol(
    dataset: str,
    backbones: Sequence[str] = ("plantnet", "imagenet"),
    seeds: int | Sequence[int] = 10,
    models: Sequence[str] = ("poly2", "rbf", "logreg"),
    output_dir: str | os.PathLike[str] = "results/fresh_protocol",
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
    alpha: float = ALPHA,
    boot_B: int = BOOT_B,
    boot_seed: int = BOOT_SEED,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> dict:
    """Loop seeds x backbones x classical models and write additive artifacts.

    Every expected ``(backbone, seed, model)`` tuple gets one manifest row with
    a final status of ``succeeded``, ``skipped_failed_arm`` or
    ``skipped_cache_failure``.  Skipped rows carry the original exception type
    and message and no metric values; all counts derive from that manifest, so
    a failed cache build can never silently drop the remaining seeds of a
    backbone.  ``metrics.json`` and ``provenance.json`` are written to
    ``output_dir`` (or a fresh child directory if they already exist).  Both
    carry ``claim_boundary`` stating that no quantum advantage is asserted.
    """
    seed_list = _normalize_seeds(seeds)
    model_list = list(dict.fromkeys(str(model) for model in models))
    backbone_list = list(dict.fromkeys(str(backbone) for backbone in backbones))

    # Single source of truth for accounting: one row per expected run.  Every
    # list and count below is derived from this manifest.
    manifest: list[dict] = [
        {
            "dataset": str(dataset),
            "backbone": backbone,
            "seed": int(seed),
            "model": model,
            "status": "not_run",
            "stage": None,
            "error_type": None,
            "error": None,
        }
        for backbone in backbone_list
        for seed in seed_list
        for model in model_list
    ]
    manifest_index = {
        (entry["backbone"], entry["seed"], entry["model"]): index
        for index, entry in enumerate(manifest)
    }
    run_metrics: dict[tuple[str, int, str], dict] = {}
    cache_info: dict[str, dict] = {}
    split_provenance: dict[str, list[dict]] = {}

    def _mark(
        entry: dict, status: str, stage: str | None = None, exc: Exception | None = None
    ) -> None:
        entry["status"] = status
        entry["stage"] = stage
        if exc is not None:
            entry["error_type"] = type(exc).__name__
            entry["error"] = str(exc)

    for backbone in backbone_list:
        first_cache = None
        cache_failed = False
        for seed in seed_list:
            if cache_failed:
                break
            try:
                if first_cache is not None and seed == seed_list[0]:
                    cached = first_cache
                else:
                    cached = cache_features(
                        dataset,
                        backbone,
                        root=root,
                        cache_dir=cache_dir,
                        seed=seed,
                        ratios=ratios,
                    )
                    if seed == seed_list[0]:
                        first_cache = cached
                cache_info.setdefault(
                    backbone,
                    {
                        "cache_path": cached.get("cache_path"),
                        "cache_hit": bool(cached.get("cache_hit")),
                        "feature_dim": int(np.asarray(cached["X"]["train"]).shape[1]),
                    },
                )
                split_provenance.setdefault(backbone, []).append(
                    {
                        "seed": int(seed),
                        "split_provenance": cached.get("provenance", {}).get("split"),
                        "groups": cached.get("groups"),
                    }
                )
            except Exception as exc:
                # This backbone cannot be featurised.  Finalise EVERY remaining
                # (seed, model) tuple for it with the original error, so the
                # accounting invariant holds and nothing is silently omitted.
                for entry in manifest:
                    if entry["backbone"] == backbone and entry["status"] == "not_run":
                        _mark(entry, "skipped_cache_failure", "feature_cache", exc)
                cache_failed = True
                break
            for model in model_list:
                entry = manifest[manifest_index[(backbone, seed, model)]]
                try:
                    arm = run_svm_arm(
                        cached["X"]["train"],
                        cached["y"]["train"],
                        cached["X"]["test"],
                        cached["y"]["test"],
                        model,
                        seed=seed,
                    )
                    _mark(entry, "succeeded")
                    run_metrics[(backbone, seed, model)] = arm
                except Exception as exc:
                    _mark(entry, "skipped_failed_arm", "arm", exc)

    # Derive all downstream views from the single manifest.
    runs: list[dict] = []
    for entry in manifest:
        if entry["status"] != "succeeded":
            continue
        key = (entry["backbone"], entry["seed"], entry["model"])
        runs.append(
            {
                "dataset": str(dataset),
                "backbone": entry["backbone"],
                "seed": entry["seed"],
                "model": entry["model"],
                "status": "ok",
                "metrics": run_metrics[key],
            }
        )
    warnings: list[dict] = [
        {
            "stage": entry["stage"],
            "status": entry["status"],
            "dataset": entry["dataset"],
            "backbone": entry["backbone"],
            "seed": entry["seed"],
            "model": entry["model"],
            "error_type": entry["error_type"],
            "error": entry["error"],
        }
        for entry in manifest
        if entry["status"] != "succeeded"
    ]

    aggregates: dict[str, dict] = {}
    for backbone in backbone_list:
        for model in model_list:
            subset = [
                run
                for run in runs
                if run["backbone"] == backbone
                and run["model"] == model
                and run["status"] == "ok"
            ]
            if not subset:
                continue
            entry: dict[str, Any] = {
                "n_runs": len(subset),
                "seeds": [run["seed"] for run in subset],
            }
            for metric in _METRICS:
                values = [
                    run["metrics"].get(metric)
                    for run in subset
                    if run["metrics"].get(metric) is not None
                ]
                clean = [float(value) for value in values if np.isfinite(float(value))]
                if clean:
                    entry[metric] = aggregate(
                        clean, seed=boot_seed, n_boot=boot_B, name=f"{backbone}/{model}"
                    )
            aggregates[f"{backbone}/{model}"] = entry

    comparisons: list[dict] = []
    for backbone in backbone_list:
        family: list[dict] = []
        for left, right in itertools.combinations(model_list, 2):
            a_values: list[float] = []
            b_values: list[float] = []
            for seed in seed_list:
                left_run = next(
                    (
                        run
                        for run in runs
                        if run["backbone"] == backbone
                        and run["model"] == left
                        and run["seed"] == seed
                        and run["status"] == "ok"
                    ),
                    None,
                )
                right_run = next(
                    (
                        run
                        for run in runs
                        if run["backbone"] == backbone
                        and run["model"] == right
                        and run["seed"] == seed
                        and run["status"] == "ok"
                    ),
                    None,
                )
                if left_run is not None and right_run is not None:
                    a_values.append(float(left_run["metrics"]["accuracy"]))
                    b_values.append(float(right_run["metrics"]["accuracy"]))
            if len(a_values) < 2:
                continue
            family.append(
                paired_comparison(
                    a_values,
                    b_values,
                    alpha=alpha,
                    seed=boot_seed,
                    n_boot=boot_B,
                    pair=(left, right),
                )
            )
        for item in holm_adjust(family, alpha=alpha):
            item["backbone"] = backbone
            item["metric"] = "accuracy"
            comparisons.append(item)

    config = {
        "dataset": str(dataset),
        "backbones": backbone_list,
        "seeds": seed_list,
        "models": model_list,
        "root": str(root),
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "ratios": [float(value) for value in ratios],
        "alpha": float(alpha),
        "bootstrap": {"B": int(boot_B), "seed": int(boot_seed)},
        "protocol": "fresh_protocol.v1",
    }
    counts = {
        status: sum(1 for entry in manifest if entry["status"] == status)
        for status in ("succeeded", "skipped_failed_arm", "skipped_cache_failure")
    }
    n_expected = len(manifest)
    n_succeeded = counts["succeeded"]
    n_skipped = counts["skipped_failed_arm"] + counts["skipped_cache_failure"]
    metrics_payload = {
        "schema": "fresh_protocol.metrics.v1",
        "dataset": str(dataset),
        "claim_boundary": CLAIM_BOUNDARY,
        "config": config,
        "runs": runs,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "warnings": warnings,
        "n_runs_expected": int(n_expected),
        "n_runs_succeeded": int(n_succeeded),
        "n_runs_skipped": int(n_skipped),
        "n_skipped_failed_arm": int(counts["skipped_failed_arm"]),
        "n_skipped_cache_failure": int(counts["skipped_cache_failure"]),
        "accounting_invariant": bool(n_expected == n_succeeded + n_skipped),
    }
    provenance_payload = {
        "schema": "fresh_protocol.provenance.v1",
        "dataset": str(dataset),
        "claim_boundary": CLAIM_BOUNDARY,
        "config": config,
        "environment": _environment_pins(),
        "dataset_split_provenance": split_provenance,
        "feature_cache": cache_info,
        "run_manifest": [
            {
                "dataset": entry["dataset"],
                "backbone": entry["backbone"],
                "seed": entry["seed"],
                "model": entry["model"],
                "status": entry["status"],
                "stage": entry["stage"],
                "error_type": entry["error_type"],
                "error": entry["error"],
            }
            for entry in manifest
        ],
        "comparison_notes": {
            "secondary_test": "Wilcoxon signed-rank",
            "exact_mcnemar": (
                "not computed: this stage stores only one scalar accuracy per seed; "
                "paired per-image predictions are not retained, so Wilcoxon is the "
                "secondary paired test"
            ),
        },
        "warnings": warnings,
        "additive": True,
    }

    metrics_path, provenance_path = _resolve_output_paths(output_dir)
    _write_json(metrics_path, metrics_payload)
    _write_json(provenance_path, provenance_payload)
    return {
        "metrics": metrics_payload,
        "provenance": provenance_payload,
        "metrics_path": str(metrics_path),
        "provenance_path": str(provenance_path),
        "warnings": warnings,
        "claim_boundary": CLAIM_BOUNDARY,
    }
