"""Synthetic, offline tests for seeded Gaussian corruption (QC-D04 §2.3).

No dataset, paper value or external file is used: every array is generated in
memory from a fixed seed.  The assertions target the seed derivation, the exact
sigma grid, the clean-anchor identity, clipping and reproducibility.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import noise  # noqa: E402


def test_sigma_grid_is_the_accepted_protocol_constant() -> None:
    assert noise.SIGMA_GRID == (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
    assert noise.sigma_index(0.15) == noise.SIGMA_GRID.index(0.15)
    with pytest.raises(ValueError):
        noise.sigma_index(0.07)
    assert noise.NoiseSpec().validate() == []


def test_noise_spec_rejects_bad_grids_and_draw_counts() -> None:
    assert noise.NoiseSpec(sigma_grid=(0.05, 0.1)).validate()
    assert noise.NoiseSpec(sigma_grid=(0.0, 0.1, 0.05)).validate()
    assert noise.NoiseSpec(sigma_grid=(0.0, -0.1)).validate()
    assert noise.NoiseSpec(noise_draws=0).validate()


def test_seeds_are_unique_reproducible_and_master_dependent() -> None:
    master = 1234
    run_id = "run_03"
    n_samples = 7
    first = noise.make_noise_seeds(master, run_id, 0.15, 2, n_samples)
    second = noise.make_noise_seeds(master, run_id, 0.15, 2, n_samples)
    assert first == second
    assert len(set(first)) == n_samples
    assert all(isinstance(seed, int) for seed in first)

    assert noise.make_noise_seeds(master + 1, run_id, 0.15, 2, n_samples) != first
    assert noise.make_noise_seeds(master, run_id, 0.20, 2, n_samples) != first
    assert noise.make_noise_seeds(master, run_id, 0.15, 0, n_samples) != first
    assert noise.make_noise_seeds(master, "run_04", 0.15, 2, n_samples) != first

    expected = noise.derive_noise_seed(master, noise.run_index(run_id), 3, 2, 0)
    assert first[0] == expected


def test_sigma_zero_is_an_exact_clean_anchor() -> None:
    rng = np.random.default_rng(9)
    clean = rng.normal(size=(5, 4))
    seeds = noise.make_noise_seeds(42, "run_00", 0.0, 0, len(clean))
    result = noise.add_gaussian_noise(clean, 0.0, seeds, clip_bounds=(-1.0, 1.0))
    np.testing.assert_array_equal(result, clean)


def test_noise_is_seeded_clipped_and_nontrivial() -> None:
    rng = np.random.default_rng(11)
    clean = rng.normal(size=(6, 3))
    seeds = noise.make_noise_seeds(77, "run_01", 0.30, 1, len(clean))
    first = noise.add_gaussian_noise(clean, 0.30, seeds)
    second = noise.add_gaussian_noise(clean, 0.30, seeds)
    np.testing.assert_array_equal(first, second)
    assert not np.allclose(first, clean)

    other = noise.add_gaussian_noise(
        clean, 0.30, noise.make_noise_seeds(78, "run_01", 0.30, 1, len(clean))
    )
    assert not np.allclose(first, other)

    low = np.full(3, -0.5)
    high = np.full(3, 0.5)
    clipped = noise.add_gaussian_noise(clean, 0.30, seeds, clip_bounds=(low, high))
    assert np.all(clipped >= low) and np.all(clipped <= high)


def test_clipping_bounds_come_only_from_config() -> None:
    cfg = SimpleNamespace(norm_mean=(0.5, 0.25, 0.125), norm_std=(0.5, 0.5, 0.25))
    low, high = noise.clipping_bounds_from_config(cfg)
    np.testing.assert_allclose(low, [-1.0, -0.5, -0.5])
    np.testing.assert_allclose(high, [1.0, 1.5, 3.5])
    low_flat, high_flat = noise.clipping_bounds_from_config(cfg, n_features=6)
    assert low_flat.shape == (6,) and high_flat.shape == (6,)
    np.testing.assert_allclose(low_flat[:2], low[0])
    with pytest.raises(ValueError):
        noise.clipping_bounds_from_config(cfg, n_features=5)


def test_bad_seed_count_is_rejected() -> None:
    clean = np.zeros((3, 2))
    with pytest.raises(ValueError):
        noise.add_gaussian_noise(clean, 0.1, [1, 2])
    with pytest.raises(ValueError):
        noise.add_gaussian_noise(clean, -0.1, [1, 2, 3])
