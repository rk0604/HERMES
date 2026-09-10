"""Synthetic bulk-grid generator.

The "bulk" is a binary ``grid_size x grid_size`` array (10x10 by default). Each
sample is drawn from one of several *structured* pattern families -- lines,
rectangles, clusters, circles, symmetric motifs, and combinations.

Why structure matters
---------------------
The boundary is deliberately smaller than the bulk. A dataset of independent
random pixels would be 100 incompressible bits, so no substantially smaller
boundary could reconstruct it and the experiment would only measure the
bottleneck's raw capacity. Structured patterns have low description length, so
a compressive boundary *can* in principle carry them -- which makes the
interesting question "how is that information laid out around the ring?" rather
than "does it fit?".

Every generator takes a :class:`numpy.random.Generator` and returns both the
grid and a metadata dict describing exactly what was drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .config import DEFAULT_PATTERNS, DataConfig

__all__ = [
    "BulkSample",
    "BulkDataset",
    "PATTERN_GENERATORS",
    "generate_sample",
    "generate_dataset",
    "pattern_to_index",
    "index_to_pattern",
]

Grid = np.ndarray  # (grid_size, grid_size) uint8, values in {0, 1}
Generator = Callable[[np.random.Generator, int], tuple[Grid, dict[str, Any]]]

_MAX_RESAMPLE = 32
#: Stream id that separates pattern selection from shape drawing (see
#: :func:`generate_sample`), so a recorded ``(pattern_type, seed)`` regenerates a grid.
_SELECTOR_STREAM = 0xC0FFEE


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BulkSample:
    """One generated bulk grid plus a description of how it was made.

    Attributes:
        grid: ``(grid_size, grid_size)`` ``uint8`` array with values in ``{0, 1}``.
        pattern_type: Which family produced it, e.g. ``"circle"``.
        metadata: Generator parameters (centre, radius, endpoints, ...).
        seed: The seed that reproduces this exact sample.
    """

    grid: Grid
    pattern_type: str
    metadata: dict[str, Any]
    seed: int

    @property
    def grid_size(self) -> int:
        return int(self.grid.shape[0])

    @property
    def density(self) -> float:
        """Fraction of cells that are on."""
        return float(self.grid.mean())

    def describe(self) -> str:
        """Short human-readable summary, used by the Streamlit app."""
        parts = [f"{k}={v}" for k, v in self.metadata.items() if k != "pattern_type"]
        return f"{self.pattern_type}({', '.join(parts)})"


@dataclass
class BulkDataset:
    """A batch of generated grids with aligned labels and metadata."""

    grids: np.ndarray  # (N, G, G) uint8
    pattern_types: list[str]
    metadata: list[dict[str, Any]]
    seeds: list[int]
    config: DataConfig = field(default_factory=DataConfig)

    def __len__(self) -> int:
        return int(self.grids.shape[0])

    def __getitem__(self, index: int) -> BulkSample:
        return BulkSample(
            grid=self.grids[index],
            pattern_type=self.pattern_types[index],
            metadata=self.metadata[index],
            seed=self.seeds[index],
        )

    def __iter__(self) -> Iterable[BulkSample]:
        for i in range(len(self)):
            yield self[i]

    @property
    def grid_size(self) -> int:
        return int(self.grids.shape[1])

    def pattern_ids(self) -> np.ndarray:
        """Integer pattern label per sample, indexed into :data:`DEFAULT_PATTERNS`."""
        return np.array([pattern_to_index(p) for p in self.pattern_types], dtype=np.int64)

    def filter_by_pattern(self, pattern: str) -> "BulkDataset":
        keep = [i for i, p in enumerate(self.pattern_types) if p == pattern]
        return BulkDataset(
            grids=self.grids[keep],
            pattern_types=[self.pattern_types[i] for i in keep],
            metadata=[self.metadata[i] for i in keep],
            seeds=[self.seeds[i] for i in keep],
            config=self.config,
        )

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for p in self.pattern_types:
            counts[p] = counts.get(p, 0) + 1
        return {
            "num_samples": len(self),
            "grid_size": self.grid_size,
            "mean_density": float(self.grids.mean()),
            "pattern_counts": counts,
        }


def pattern_to_index(pattern: str) -> int:
    return DEFAULT_PATTERNS.index(pattern)


def index_to_pattern(index: int) -> str:
    return DEFAULT_PATTERNS[index]


# ---------------------------------------------------------------------------
# Pattern generators
# ---------------------------------------------------------------------------


def _blank(grid_size: int) -> Grid:
    return np.zeros((grid_size, grid_size), dtype=np.uint8)


def _span(rng: np.random.Generator, size: int, min_len: int = 3) -> tuple[int, int]:
    """A random inclusive ``[start, end]`` interval of length >= ``min_len``."""
    length = int(rng.integers(min_len, size + 1))
    start = int(rng.integers(0, size - length + 1))
    return start, start + length - 1


def horizontal_line(rng: np.random.Generator, grid_size: int) -> tuple[Grid, dict[str, Any]]:
    """A single horizontal segment, optionally 2 cells thick."""
    grid = _blank(grid_size)
    row = int(rng.integers(0, grid_size))
    c0, c1 = _span(rng, grid_size)
    thickness = int(rng.integers(1, 3))
    rows = [r for r in range(row, min(row + thickness, grid_size))]
    grid[np.ix_(rows, range(c0, c1 + 1))] = 1
    return grid, {"row": row, "col_start": c0, "col_end": c1, "thickness": len(rows)}


def vertical_line(rng: np.random.Generator, grid_size: int) -> tuple[Grid, dict[str, Any]]:
    """A single vertical segment, optionally 2 cells thick."""
    grid = _blank(grid_size)
    col = int(rng.integers(0, grid_size))
    r0, r1 = _span(rng, grid_size)
    thickness = int(rng.integers(1, 3))
    cols = [c for c in range(col, min(col + thickness, grid_size))]
    grid[np.ix_(range(r0, r1 + 1), cols)] = 1
    return grid, {"col": col, "row_start": r0, "row_end": r1, "thickness": len(cols)}


def diagonal_line(rng: np.random.Generator, grid_size: int) -> tuple[Grid, dict[str, Any]]:
    """A diagonal segment along either the main or the anti-diagonal direction."""
    grid = _blank(grid_size)
    direction = "main" if rng.random() < 0.5 else "anti"
    offset = int(rng.integers(-(grid_size - 3), grid_size - 2))
    cells: list[tuple[int, int]] = []
    for i in range(grid_size):
        r = i
        c = i + offset if direction == "main" else (grid_size - 1 - i) + offset
        if 0 <= c < grid_size:
            cells.append((r, c))
    if not cells:
        return diagonal_line(rng, grid_size)
    # Optionally use only part of the diagonal.
    if len(cells) > 4 and rng.random() < 0.5:
        keep = int(rng.integers(3, len(cells) + 1))
        start = int(rng.integers(0, len(cells) - keep + 1))
        cells = cells[start : start + keep]
    rows, cols = zip(*cells)
    grid[list(rows), list(cols)] = 1
    return grid, {
        "direction": direction,
        "offset": offset,
        "length": len(cells),
        "start": list(cells[0]),
        "end": list(cells[-1]),
    }


def _rectangle_bounds(rng: np.random.Generator, grid_size: int, min_side: int) -> tuple[int, int, int, int]:
    height = int(rng.integers(min_side, min(grid_size, 7) + 1))
    width = int(rng.integers(min_side, min(grid_size, 7) + 1))
    r0 = int(rng.integers(0, grid_size - height + 1))
    c0 = int(rng.integers(0, grid_size - width + 1))
    return r0, c0, height, width


def rectangle_filled(rng: np.random.Generator, grid_size: int) -> tuple[Grid, dict[str, Any]]:
    """A solid axis-aligned rectangle."""
    grid = _blank(grid_size)
    r0, c0, h, w = _rectangle_bounds(rng, grid_size, min_side=2)
    grid[r0 : r0 + h, c0 : c0 + w] = 1
    return grid, {"row": r0, "col": c0, "height": h, "width": w, "filled": True}


def rectangle_hollow(rng: np.random.Generator, grid_size: int) -> tuple[Grid, dict[str, Any]]:
    """A rectangle outline (border cells only)."""
    grid = _blank(grid_size)
    r0, c0, h, w = _rectangle_bounds(rng, grid_size, min_side=3)
    grid[r0 : r0 + h, c0 : c0 + w] = 1
    if h > 2 and w > 2:
        grid[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = 0
    return grid, {"row": r0, "col": c0, "height": h, "width": w, "filled": False}


def cluster(rng: np.random.Generator, grid_size: int) -> tuple[Grid, dict[str, Any]]:
    """A small connected blob grown by random 4-neighbour accretion.

    Each step either adds a cell or retires an exhausted frontier cell, so the
    loop always terminates: growth is bounded by ``size`` and retirement by the
    number of cells ever added.
    """
    grid = _blank(grid_size)
    size = int(rng.integers(3, 9))
    r, c = int(rng.integers(0, grid_size)), int(rng.integers(0, grid_size))
    cells = {(r, c)}
    frontier = [(r, c)]
    while len(cells) < size and frontier:
        index = int(rng.integers(0, len(frontier)))
        br, bc = frontier[index]
        options = [
            (br + dr, bc + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= br + dr < grid_size
            and 0 <= bc + dc < grid_size
            and (br + dr, bc + dc) not in cells
        ]
        if not options:
            # This cell is boxed in; retire it rather than retrying forever.
            frontier.pop(index)
            continue
        nr, nc = options[int(rng.integers(0, len(options)))]
        cells.add((nr, nc))
        frontier.append((nr, nc))
    rows, cols = zip(*sorted(cells))
    grid[list(rows), list(cols)] = 1
    return grid, {"origin": [r, c], "size": len(cells), "cells": [list(x) for x in sorted(cells)]}


def circle(rng: np.random.Generator, grid_size: int) -> tuple[Grid, dict[str, Any]]:
    """A discretised circle: a ring by default, occasionally a filled disc."""
    grid = _blank(grid_size)
    radius = float(rng.integers(2, max(3, grid_size // 2)))
    lo, hi = int(np.ceil(radius)), int(grid_size - 1 - np.ceil(radius))
    if hi < lo:
        lo, hi = 0, grid_size - 1
    cr = float(rng.integers(lo, hi + 1))
    cc = float(rng.integers(lo, hi + 1))
    filled = bool(rng.random() < 0.3)

    rows, cols = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    dist = np.sqrt((rows - cr) ** 2 + (cols - cc) ** 2)
    mask = dist <= radius + 0.35 if filled else np.abs(dist - radius) <= 0.6
    grid[mask] = 1
    return grid, {
        "center": [int(cr), int(cc)],
        "radius": radius,
        "filled": filled,
        "num_on": int(grid.sum()),
    }


def symmetric(rng: np.random.Generator, grid_size: int) -> tuple[Grid, dict[str, Any]]:
    """A random motif mirrored across one or both axes."""
    axis = ["vertical", "horizontal", "both"][int(rng.integers(0, 3))]
    half = grid_size // 2
    grid = _blank(grid_size)

    n_seeds = int(rng.integers(3, 9))
    rows_hi = half if axis in ("horizontal", "both") else grid_size
    cols_hi = half if axis in ("vertical", "both") else grid_size
    seeds: list[tuple[int, int]] = []
    for _ in range(n_seeds):
        r = int(rng.integers(0, max(1, rows_hi)))
        c = int(rng.integers(0, max(1, cols_hi)))
        seeds.append((r, c))
        grid[r, c] = 1

    if axis in ("vertical", "both"):
        grid |= grid[:, ::-1]
    if axis in ("horizontal", "both"):
        grid |= grid[::-1, :]
    return grid, {"axis": axis, "num_seeds": n_seeds, "seeds": [list(s) for s in seeds]}


def combined(rng: np.random.Generator, grid_size: int) -> tuple[Grid, dict[str, Any]]:
    """The union of two simpler shapes drawn from the non-composite families."""
    simple = [
        "horizontal_line",
        "vertical_line",
        "diagonal_line",
        "rectangle_filled",
        "rectangle_hollow",
        "cluster",
        "circle",
    ]
    picks = [simple[int(rng.integers(0, len(simple)))] for _ in range(2)]
    grid = _blank(grid_size)
    parts: list[dict[str, Any]] = []
    for name in picks:
        sub, meta = PATTERN_GENERATORS[name](rng, grid_size)
        grid |= sub
        parts.append({"pattern_type": name, **meta})
    return grid, {"parts": parts, "num_parts": len(parts)}


#: Name -> generator function. Extend this dict to add a new pattern family
#: (and add the name to :data:`hermes.config.DEFAULT_PATTERNS`).
PATTERN_GENERATORS: dict[str, Generator] = {
    "horizontal_line": horizontal_line,
    "vertical_line": vertical_line,
    "diagonal_line": diagonal_line,
    "rectangle_filled": rectangle_filled,
    "rectangle_hollow": rectangle_hollow,
    "cluster": cluster,
    "circle": circle,
    "symmetric": symmetric,
    "combined": combined,
}


# ---------------------------------------------------------------------------
# Sampling entry points
# ---------------------------------------------------------------------------


def generate_sample(
    pattern: str | None = None,
    seed: int | None = None,
    grid_size: int = 10,
    noise_prob: float = 0.0,
    allow_empty: bool = False,
    pattern_types: Sequence[str] = DEFAULT_PATTERNS,
    rng: np.random.Generator | None = None,
) -> BulkSample:
    """Generate one bulk grid.

    Args:
        pattern: Pattern family name, or ``None`` to pick uniformly from
            ``pattern_types``.
        seed: Seed for a fresh generator. Ignored when ``rng`` is supplied.
            Pattern selection draws from a separate stream from shape drawing,
            so ``generate_sample(None, seed=s)`` and
            ``generate_sample(<the pattern it chose>, seed=s)`` return the *same*
            grid. That is what makes a recorded ``(pattern_type, seed)`` pair
            enough to regenerate a sample exactly.
        grid_size: Side length of the square grid.
        noise_prob: Per-cell flip probability applied after drawing the pattern.
        allow_empty: If ``False``, resample when the grid comes out all zeros.
        pattern_types: Candidate families when ``pattern`` is ``None``.
        rng: Explicit generator; use this when drawing many samples in a stream.

    Returns:
        A :class:`BulkSample` whose ``grid`` is ``uint8`` with values in ``{0, 1}``.
    """
    if rng is None:
        seed = int(np.random.SeedSequence().entropy % (2**31)) if seed is None else int(seed)
        rng = np.random.default_rng(seed)
        # Independent stream, so choosing the pattern does not shift the shape draw.
        selector = np.random.default_rng([int(seed), _SELECTOR_STREAM])
    else:
        seed = -1 if seed is None else int(seed)
        selector = rng

    for _ in range(_MAX_RESAMPLE):
        name = pattern or str(pattern_types[int(selector.integers(0, len(pattern_types)))])
        if name not in PATTERN_GENERATORS:
            raise ValueError(f"Unknown pattern {name!r}; expected one of {sorted(PATTERN_GENERATORS)}")
        grid, meta = PATTERN_GENERATORS[name](rng, grid_size)
        if noise_prob > 0.0:
            flips = rng.random(grid.shape) < noise_prob
            grid = (grid ^ flips.astype(np.uint8)).astype(np.uint8)
            meta = {**meta, "noise_prob": noise_prob, "num_flipped": int(flips.sum())}
        if allow_empty or grid.any():
            grid = np.ascontiguousarray(grid, dtype=np.uint8)
            return BulkSample(grid=grid, pattern_type=name, metadata=meta, seed=seed)

    # Extremely unlikely; fall back to a guaranteed non-empty grid.
    grid = _blank(grid_size)
    grid[grid_size // 2, grid_size // 2] = 1
    return BulkSample(grid=grid, pattern_type="cluster", metadata={"fallback": True}, seed=seed)


def generate_dataset(
    num_samples: int,
    config: DataConfig | None = None,
    seed: int | None = None,
) -> BulkDataset:
    """Generate ``num_samples`` grids with per-sample reproducible seeds.

    Each sample gets its own seed derived from the master seed, so a single grid
    can be regenerated later without replaying the whole stream.
    """
    config = config or DataConfig()
    master = config.seed if seed is None else int(seed)
    seed_rng = np.random.default_rng(master)
    sample_seeds = seed_rng.integers(0, 2**31 - 1, size=num_samples, dtype=np.int64)

    grids = np.zeros((num_samples, config.grid_size, config.grid_size), dtype=np.uint8)
    pattern_types: list[str] = []
    metadata: list[dict[str, Any]] = []
    for i, s in enumerate(sample_seeds):
        sample = generate_sample(
            pattern=None,
            seed=int(s),
            grid_size=config.grid_size,
            noise_prob=config.noise_prob,
            allow_empty=config.allow_empty,
            pattern_types=config.pattern_types,
        )
        grids[i] = sample.grid
        pattern_types.append(sample.pattern_type)
        metadata.append(sample.metadata)

    return BulkDataset(
        grids=grids,
        pattern_types=pattern_types,
        metadata=metadata,
        seeds=[int(s) for s in sample_seeds],
        config=config,
    )


def make_train_val(config: DataConfig | None = None) -> tuple[BulkDataset, BulkDataset]:
    """Generate disjointly-seeded train and validation splits."""
    config = config or DataConfig()
    train = generate_dataset(config.train_size, config, seed=config.seed)
    val = generate_dataset(config.val_size, config, seed=config.seed + 10_000_019)
    return train, val
