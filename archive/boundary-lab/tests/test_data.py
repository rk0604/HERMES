"""Tests for the synthetic bulk generator."""

from __future__ import annotations

import numpy as np
import pytest

from hermes.config import DEFAULT_PATTERNS, DataConfig
from hermes.data import (
    PATTERN_GENERATORS,
    generate_dataset,
    generate_sample,
    index_to_pattern,
    make_train_val,
    pattern_to_index,
)


@pytest.mark.parametrize("pattern", DEFAULT_PATTERNS)
def test_grid_is_binary_and_correctly_shaped(pattern: str) -> None:
    sample = generate_sample(pattern=pattern, seed=0, grid_size=10)
    assert sample.grid.shape == (10, 10)
    assert sample.grid.dtype == np.uint8
    assert set(np.unique(sample.grid)).issubset({0, 1})
    assert sample.pattern_type == pattern


@pytest.mark.parametrize("pattern", DEFAULT_PATTERNS)
def test_every_generator_is_reproducible_under_a_seed(pattern: str) -> None:
    first = generate_sample(pattern=pattern, seed=1234, grid_size=10)
    second = generate_sample(pattern=pattern, seed=1234, grid_size=10)
    assert np.array_equal(first.grid, second.grid)
    assert first.metadata == second.metadata


@pytest.mark.parametrize("pattern", DEFAULT_PATTERNS)
def test_different_seeds_generally_give_different_grids(pattern: str) -> None:
    grids = {generate_sample(pattern=pattern, seed=s, grid_size=10).grid.tobytes() for s in range(24)}
    assert len(grids) > 1, f"{pattern} ignored its seed"


@pytest.mark.parametrize("pattern", DEFAULT_PATTERNS)
def test_generators_produce_non_empty_grids(pattern: str) -> None:
    for seed in range(8):
        assert generate_sample(pattern=pattern, seed=seed, grid_size=10).grid.sum() > 0


def test_every_declared_pattern_has_a_generator() -> None:
    assert set(DEFAULT_PATTERNS) == set(PATTERN_GENERATORS)


def test_pattern_index_roundtrip() -> None:
    for pattern in DEFAULT_PATTERNS:
        assert index_to_pattern(pattern_to_index(pattern)) == pattern


def test_unknown_pattern_raises() -> None:
    with pytest.raises(ValueError):
        generate_sample(pattern="not_a_pattern", seed=0)


def test_dataset_shape_labels_and_reproducibility() -> None:
    config = DataConfig(seed=7)
    first = generate_dataset(32, config)
    second = generate_dataset(32, config)

    assert first.grids.shape == (32, 10, 10)
    assert len(first) == 32
    assert len(first.pattern_types) == len(first.metadata) == len(first.seeds) == 32
    assert set(np.unique(first.grids)).issubset({0, 1})
    assert np.array_equal(first.grids, second.grids)
    assert first.pattern_types == second.pattern_types


def test_per_sample_seed_regenerates_the_same_grid() -> None:
    dataset = generate_dataset(16, DataConfig(seed=3))
    for index in (0, 5, 15):
        sample = dataset[index]
        again = generate_sample(pattern=sample.pattern_type, seed=sample.seed, grid_size=10)
        assert np.array_equal(again.grid, sample.grid)


def test_restricted_pattern_set_is_respected() -> None:
    config = DataConfig(pattern_types=("circle", "cluster"), seed=0)
    dataset = generate_dataset(40, config)
    assert set(dataset.pattern_types).issubset({"circle", "cluster"})


def test_train_val_splits_are_differently_seeded() -> None:
    config = DataConfig(train_size=64, val_size=64, seed=0)
    train, val = make_train_val(config)
    assert len(train) == 64 and len(val) == 64
    assert not np.array_equal(train.grids, val.grids)


def test_noise_flips_cells_but_keeps_grid_binary() -> None:
    clean = generate_sample(pattern="rectangle_filled", seed=5, noise_prob=0.0)
    noisy = generate_sample(pattern="rectangle_filled", seed=5, noise_prob=0.5)
    assert set(np.unique(noisy.grid)).issubset({0, 1})
    assert not np.array_equal(clean.grid, noisy.grid)


def test_hollow_rectangle_has_an_empty_interior() -> None:
    for seed in range(20):
        sample = generate_sample(pattern="rectangle_hollow", seed=seed)
        row, col = sample.metadata["row"], sample.metadata["col"]
        height, width = sample.metadata["height"], sample.metadata["width"]
        if height > 2 and width > 2:
            interior = sample.grid[row + 1 : row + height - 1, col + 1 : col + width - 1]
            assert interior.sum() == 0


def test_symmetric_pattern_is_actually_symmetric() -> None:
    for seed in range(30):
        sample = generate_sample(pattern="symmetric", seed=seed)
        axis = sample.metadata["axis"]
        if axis in ("vertical", "both"):
            assert np.array_equal(sample.grid, sample.grid[:, ::-1])
        if axis in ("horizontal", "both"):
            assert np.array_equal(sample.grid, sample.grid[::-1, :])


def test_filter_by_pattern_keeps_alignment() -> None:
    dataset = generate_dataset(60, DataConfig(seed=2))
    circles = dataset.filter_by_pattern("circle")
    assert all(p == "circle" for p in circles.pattern_types)
    assert circles.grids.shape[0] == len(circles.pattern_types)


def test_grid_size_is_configurable() -> None:
    dataset = generate_dataset(8, DataConfig(grid_size=16, seed=0))
    assert dataset.grids.shape == (8, 16, 16)


def test_cluster_growth_always_terminates() -> None:
    """Regression: the blob loop used to spin forever when the frontier collapsed
    to a single cell whose neighbours were all taken, so a rare seed would hang
    dataset generation indefinitely."""
    from hermes.data import cluster

    for seed in range(4000):
        grid, metadata = cluster(np.random.default_rng(seed), 10)
        assert 1 <= metadata["size"] <= 8
        assert grid.sum() == metadata["size"]


def test_cluster_cells_are_connected() -> None:
    for seed in range(300):
        sample = generate_sample(pattern="cluster", seed=seed)
        cells = {tuple(cell) for cell in sample.metadata["cells"]}
        reached = {next(iter(cells))}
        stack = list(reached)
        while stack:
            row, col = stack.pop()
            for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbour in cells and neighbour not in reached:
                    reached.add(neighbour)
                    stack.append(neighbour)
        assert reached == cells, f"cluster seed {seed} is disconnected"


def test_large_dataset_generation_completes() -> None:
    """Regression guard for the same hang, at the size that first exposed it."""
    dataset = generate_dataset(4096, DataConfig(seed=0))
    assert dataset.grids.shape == (4096, 10, 10)
    assert dataset.grids.min() >= 0 and dataset.grids.max() <= 1
    assert all(dataset.grids[i].any() for i in range(0, 4096, 97))
