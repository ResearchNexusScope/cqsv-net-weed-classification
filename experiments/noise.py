"""Seeded Gaussian corruption of post-normalisation OOD test tensors (QC-D04 §2.3).

The corruption is deliberately narrow: additive i.i.d. Gaussian noise is applied
to *test* tensors only, after the canonical ImageNet normalisation, with a
per-sample seed derived from the master seed.  It is never a training
augmentation, and the clean ``sigma=0`` arm reuses the unmodified tensor so that
the degradation endpoint is genuinely paired (same image, same draw).

No dataset path, sample count or result value is embedded here.  The sigma grid
is a protocol constant from the accepted specification, not a measured number.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.random import SeedSequence

__all__ = [
    "SIGMA_GRID",
    "NoiseSpec",
    "derive_noise_seed",
    "sigma_index",
    "run_index",
    "make_noise_seeds",
    "clipping_bounds_from_config",
    "add_gaussian_noise",
]

# QC-D04 §2.3 -- the σ=0 entry is the clean anchor (no random draw is consumed).
SIGMA_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)

_SIGMA_TOL = 1e-12


@dataclass(frozen=True)
class NoiseSpec:
    """Resolved description of one noise experiment.

    ``clip_bounds`` is an optional ``(low, high)`` pair of arrays broadcastable
    to the feature matrix.  ``None`` means no clipping is applied and the raw
    post-normalisation tensor space is used; the chosen policy is always
    recorded in the raw artifact metadata by the caller.
    """

    sigma_grid: tuple[float, ...] = SIGMA_GRID
    noise_draws: int = 3
    clip_bounds: tuple[Any, Any] | None = None

    def validate(self) -> list[str]:
        """Return human-readable protocol violations; ``[]`` means sound."""
        problems: list[str] = []
        grid = [float(v) for v in self.sigma_grid]
        if not grid:
            problems.append("sigma_grid must contain at least one level")
        if any(v < 0.0 for v in grid):
            problems.append("sigma_grid levels must be non-negative")
        if any(b <= a for a, b in zip(grid, grid[1:])):
            problems.append("sigma_grid must be strictly increasing")
        if grid and grid[0] != 0.0:
            problems.append("sigma_grid must start at the clean anchor 0.0")
        if int(self.noise_draws) < 1:
            problems.append("noise_draws must be >= 1")
        return problems

    @property
    def clipping_policy(self) -> str:
        """Short, auditable description of the clipping policy in use."""
        if self.clip_bounds is None:
            return "none (raw post-normalisation tensor space)"
        low, high = self.clip_bounds
        return f"clip(x + sigma*eps, {np.asarray(low).shape}, {np.asarray(high).shape})"


def derive_noise_seed(
    master_seed: int,
    run_id_int: int,
    sigma_index: int,
    noise_draw: int,
    sample_index: int,
) -> int:
    """Named ``SeedSequence`` draw for one (run, sigma, draw, sample) tuple.

    The four-component entropy list follows QC-D04 §2.3 exactly;
    ``sigma_index * 1000 + noise_draw`` keeps draws of the same σ adjacent
    without colliding with neighbouring σ levels.
    """
    entropy = [
        int(master_seed),
        int(run_id_int),
        int(sigma_index) * 1000 + int(noise_draw),
        int(sample_index),
    ]
    return int(SeedSequence(entropy).generate_state(1)[0])


def sigma_index(sigma: float, grid: Sequence[float] = SIGMA_GRID) -> int:
    """Return the index of ``sigma`` on ``grid``; raise if it is off-grid."""
    for index, value in enumerate(grid):
        if abs(float(sigma) - float(value)) <= _SIGMA_TOL:
            return int(index)
    raise ValueError(f"sigma {sigma!r} is not on the grid {tuple(grid)!r}")


def run_index(run_id: str | int) -> int:
    """Stable integer identity for a run label (parsed suffix, else hash)."""
    if isinstance(run_id, (int, np.integer)):
        return int(run_id)
    text = str(run_id)
    digits = "".join(character for character in text if character.isdigit())
    if digits:
        return int(digits[:9])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def make_noise_seeds(
    master_seed: int,
    run_id: str | int,
    sigma: float,
    noise_draw: int,
    n_samples: int,
    grid: Sequence[float] = SIGMA_GRID,
) -> tuple[int, ...]:
    """One derived seed per sample, in the caller's sample order."""
    level = sigma_index(sigma, grid)
    run = run_index(run_id)
    return tuple(
        derive_noise_seed(master_seed, run, level, noise_draw, sample_index)
        for sample_index in range(int(n_samples))
    )


def clipping_bounds_from_config(
    cfg: Any, n_features: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature ImageNet clip bounds for a normalised image tensor.

    With ``n_features=None`` the bounds are per channel ``(3,)``.  Otherwise the
    channel bounds are repeated across the flattened spatial positions, so the
    result broadcasts against ``(n, n_features)``.  A non-multiple of the
    channel count is rejected rather than guessed.
    """
    mean = np.asarray(cfg.norm_mean, dtype=np.float64)
    std = np.asarray(cfg.norm_std, dtype=np.float64)
    if np.any(std <= 0.0):
        raise ValueError("norm_std entries must be positive")
    low = (0.0 - mean) / std
    high = (1.0 - mean) / std
    if n_features is None:
        return low, high
    channels = int(low.shape[0])
    if int(n_features) % channels:
        raise ValueError(
            f"n_features={n_features} is not a multiple of the {channels} channels"
        )
    spatial = int(n_features) // channels
    return np.repeat(low, spatial), np.repeat(high, spatial)


def add_gaussian_noise(
    X: np.ndarray,
    sigma: float,
    seeds: Sequence[int],
    clip_bounds: tuple[Any, Any] | None = None,
) -> np.ndarray:
    """Return ``X + sigma * eps`` with one seeded draw per row.

    ``sigma=0`` returns an exact copy of ``X`` without consuming any RNG and
    without clipping: it is the clean anchor, not a corrupted sample.  For
    ``sigma>0`` each row uses ``numpy.random.default_rng(seed).standard_normal``
    with the seed persisted by the caller.  Optional ``clip_bounds`` are applied
    after addition, matching ``clip(x + sigma*eps, clip_min, clip_max)``.
    """
    array = np.asarray(X, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {array.shape}")
    seeds = tuple(int(s) for s in seeds)
    if len(seeds) != array.shape[0]:
        raise ValueError(
            f"One noise seed per sample required: {len(seeds)} seeds for "
            f"{array.shape[0]} samples"
        )
    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")

    noisy = array.copy()
    if sigma == 0.0:
        return noisy
    for row_index, seed in enumerate(seeds):
        eps = np.random.default_rng(seed).standard_normal(size=array.shape[1])
        noisy[row_index] = array[row_index] + sigma * eps
    if clip_bounds is not None:
        low, high = clip_bounds
        noisy = np.clip(noisy, np.asarray(low, dtype=float), np.asarray(high, dtype=float))
    return noisy
