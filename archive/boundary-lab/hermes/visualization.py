"""Matplotlib figures for grids, the boundary ring and experiment curves.

Every function returns a :class:`matplotlib.figure.Figure` built without
``pyplot``, so figures carry no global state and are safe to create repeatedly
inside Streamlit reruns.

The boundary ring is the intuitive view of the topology, but reading exact
values off a circle is hard -- so every ring plot has a companion bar chart
(:func:`plot_boundary_bar`) that shows the same numbers precisely.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .topology import ring_coordinates

__all__ = [
    "plot_grid",
    "plot_grid_panels",
    "plot_reconstruction_panels",
    "plot_boundary_ring",
    "plot_boundary_bar",
    "plot_subset_curves",
    "plot_training_history",
    "plot_influence_heatmap",
]

_BINARY_CMAP = "gray_r"
_PROB_CMAP = "magma"
_DIFF_CMAP = "coolwarm"


def _style_grid_axis(ax, grid_size: int, show_ticks: bool = True) -> None:
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="#b0b0b0", linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    if show_ticks:
        step = max(1, grid_size // 10)
        ax.set_xticks(np.arange(0, grid_size, step))
        ax.set_yticks(np.arange(0, grid_size, step))
        ax.tick_params(labelsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def plot_grid(
    grid: np.ndarray,
    title: str = "",
    cmap: str = _BINARY_CMAP,
    vmin: float = 0.0,
    vmax: float = 1.0,
    highlight: tuple[int, int] | None = None,
    colorbar: bool = False,
    figsize: tuple[float, float] = (3.2, 3.2),
) -> Figure:
    """Render one 2-D grid.

    Args:
        grid: ``(G, G)`` array.
        title: Axis title.
        cmap: Matplotlib colormap name.
        vmin / vmax: Colour limits.
        highlight: Optional ``(row, col)`` to outline, e.g. the selected query.
        colorbar: Attach a colour bar (useful for probability grids).
    """
    grid = np.asarray(grid)
    fig = Figure(figsize=figsize, dpi=140)
    ax = fig.add_subplot(111)
    image = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    _style_grid_axis(ax, grid.shape[0])
    if highlight is not None:
        row, col = highlight
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#e63946", linewidth=2.0
            )
        )
    if title:
        ax.set_title(title, fontsize=9)
    if colorbar:
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_grid_panels(
    grids: Sequence[np.ndarray],
    titles: Sequence[str],
    cmaps: Sequence[str] | None = None,
    limits: Sequence[tuple[float, float]] | None = None,
    highlight: tuple[int, int] | None = None,
    colorbars: Sequence[bool] | None = None,
    panel_size: float = 2.7,
) -> Figure:
    """Render several grids side by side with independent colormaps and limits."""
    count = len(grids)
    cmaps = list(cmaps) if cmaps else [_BINARY_CMAP] * count
    limits = list(limits) if limits else [(0.0, 1.0)] * count
    colorbars = list(colorbars) if colorbars else [False] * count

    fig = Figure(figsize=(panel_size * count, panel_size + 0.4), dpi=140)
    axes = fig.subplots(1, count, squeeze=False)[0]
    for ax, grid, title, cmap, (vmin, vmax), bar in zip(
        axes, grids, titles, cmaps, limits, colorbars
    ):
        grid = np.asarray(grid)
        image = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        _style_grid_axis(ax, grid.shape[0], show_ticks=False)
        ax.set_title(title, fontsize=9)
        if highlight is not None:
            row, col = highlight
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#e63946", linewidth=1.6
                )
            )
        if bar:
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_reconstruction_panels(
    truth: np.ndarray,
    probabilities: np.ndarray,
    prediction: np.ndarray,
    error_map: np.ndarray,
    highlight: tuple[int, int] | None = None,
) -> Figure:
    """The standard four-panel reconstruction view."""
    return plot_grid_panels(
        [truth, probabilities, prediction, error_map],
        ["Original bulk", "Reconstructed p(cell=1)", "Thresholded", "Errors"],
        cmaps=[_BINARY_CMAP, _PROB_CMAP, _BINARY_CMAP, "Reds"],
        limits=[(0, 1), (0, 1), (0, 1), (0, 1)],
        colorbars=[False, True, False, False],
        highlight=highlight,
    )


def plot_boundary_ring(
    values: np.ndarray,
    ablated: Iterable[int] | None = None,
    highlight: Iterable[int] | None = None,
    title: str = "Boundary ring",
    cmap: str = "viridis",
    size_scale: bool = True,
    figsize: tuple[float, float] = (4.2, 4.2),
) -> Figure:
    """Draw the boundary slots as a ring, with marker colour/size showing ``values``.

    Args:
        values: ``(num_slots,)`` metric per slot (activation magnitude, attention,
            probability change, ...).
        ablated: Slots to cross out as removed.
        highlight: Slots to ring in a contrasting colour, e.g. a selected subset.
        title: Figure title.
        cmap: Colormap for the value encoding.
        size_scale: Also scale marker area by value (on by default; helps when
            the colormap alone is hard to read).

    Returns:
        A figure with slot indices labelled outside the ring.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    num_slots = values.shape[0]
    ablated = set(int(a) for a in (ablated or []))
    highlight = set(int(h) for h in (highlight or []))

    coords = ring_coordinates(num_slots)
    span = float(values.max() - values.min())
    normed = (values - values.min()) / span if span > 1e-12 else np.full(num_slots, 0.5)
    sizes = 120.0 + 520.0 * normed if size_scale else np.full(num_slots, 260.0)

    fig = Figure(figsize=figsize, dpi=140)
    ax = fig.add_subplot(111)

    circle_theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(circle_theta), np.sin(circle_theta), color="#cccccc", linewidth=1.0, zorder=0)

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        s=sizes,
        cmap=cmap,
        norm=Normalize(vmin=float(values.min()), vmax=float(values.max()) or 1.0),
        edgecolors="#333333",
        linewidths=0.6,
        zorder=2,
    )

    for slot in range(num_slots):
        x, y = coords[slot]
        ax.annotate(
            str(slot),
            (x * 1.19, y * 1.19),
            ha="center",
            va="center",
            fontsize=7,
            color="#444444",
        )
        if slot in highlight:
            ax.scatter([x], [y], s=sizes[slot] + 260, facecolors="none",
                       edgecolors="#1d3557", linewidths=1.8, zorder=3)
        if slot in ablated:
            ax.scatter([x], [y], marker="x", s=90, c="#e63946", linewidths=2.2, zorder=4)

    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.42, 1.42)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_boundary_bar(
    values: np.ndarray,
    ablated: Iterable[int] | None = None,
    highlight: Iterable[int] | None = None,
    title: str = "Per-slot values",
    ylabel: str = "value",
    figsize: tuple[float, float] = (6.0, 2.6),
) -> Figure:
    """Bar chart of the same per-slot values shown on the ring.

    Precise reading is much easier here than on a circle; the two views are
    meant to be used together.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    ablated = set(int(a) for a in (ablated or []))
    highlight = set(int(h) for h in (highlight or []))

    colors = []
    for slot in range(values.shape[0]):
        if slot in ablated:
            colors.append("#e63946")
        elif slot in highlight:
            colors.append("#1d3557")
        else:
            colors.append("#457b9d")

    fig = Figure(figsize=figsize, dpi=140)
    ax = fig.add_subplot(111)
    ax.bar(np.arange(values.shape[0]), values, color=colors)
    ax.set_xticks(np.arange(values.shape[0]))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("boundary slot", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_subset_curves(
    results: Sequence[Any],
    baseline: float | None = None,
    majority_baseline: float | None = None,
    title: str = "Reconstruction vs. available boundary slots",
    figsize: tuple[float, float] = (6.2, 3.6),
) -> Figure:
    """Plot one or more :class:`~hermes.analysis.SubsetCurveResult` objects.

    Args:
        results: Curves to overlay (typically contiguous vs. random).
        baseline: Optional horizontal line for full-boundary accuracy.
        majority_baseline: Optional line for the all-zeros baseline, which makes
            it obvious when "high accuracy" is really just sparsity.
    """
    fig = Figure(figsize=figsize, dpi=140)
    ax = fig.add_subplot(111)
    palette = ["#1d3557", "#e76f51", "#2a9d8f", "#8338ec"]

    for index, result in enumerate(results):
        color = palette[index % len(palette)]
        sizes = np.asarray(result.sizes)
        mean = np.asarray(result.accuracy_mean)
        std = np.asarray(result.accuracy_std)
        ax.plot(sizes, mean, marker="o", color=color, label=f"{result.mode}")
        ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.15)

    if baseline is not None:
        ax.axhline(baseline, color="#606060", linestyle="--", linewidth=1.0,
                   label=f"full boundary ({baseline:.3f})")
    if majority_baseline is not None:
        ax.axhline(majority_baseline, color="#b0b0b0", linestyle=":", linewidth=1.0,
                   label=f"all-zeros baseline ({majority_baseline:.3f})")

    ax.set_xlabel("number of readable boundary slots", fontsize=9)
    ax.set_ylabel("cell accuracy", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def plot_training_history(
    history: Sequence[dict[str, float]],
    figsize: tuple[float, float] = (6.2, 3.0),
) -> Figure:
    """Two panels: loss curves and accuracy curves over epochs."""
    epochs = [h["epoch"] for h in history]
    fig = Figure(figsize=figsize, dpi=140)
    ax_loss, ax_acc = fig.subplots(1, 2)

    ax_loss.plot(epochs, [h["train_loss"] for h in history], color="#1d3557", label="train")
    val_epochs = [h["epoch"] for h in history if "val_loss" in h]
    if val_epochs:
        ax_loss.plot(
            val_epochs, [h["val_loss"] for h in history if "val_loss" in h],
            color="#e76f51", label="val (all cells)",
        )
    ax_loss.set_xlabel("epoch", fontsize=8)
    ax_loss.set_ylabel("BCE", fontsize=8)
    ax_loss.set_title("Loss", fontsize=9)
    ax_loss.legend(fontsize=7, frameon=False)

    ax_acc.plot(epochs, [h["train_accuracy"] for h in history], color="#1d3557", label="train")
    if val_epochs:
        ax_acc.plot(
            val_epochs, [h["val_accuracy"] for h in history if "val_accuracy" in h],
            color="#e76f51", label="val (all cells)",
        )
    ax_acc.set_xlabel("epoch", fontsize=8)
    ax_acc.set_ylabel("cell accuracy", fontsize=8)
    ax_acc.set_title("Accuracy", fontsize=9)
    ax_acc.legend(fontsize=7, frameon=False)

    for ax in (ax_loss, ax_acc):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def plot_influence_heatmap(
    matrix: np.ndarray,
    title: str = "Bulk cell -> boundary slot attention",
    figsize: tuple[float, float] = (6.0, 3.4),
) -> Figure:
    """Heatmap of a ``(num_cells, num_slots)`` influence matrix.

    Reminder: attention is an association measure, not a causal attribution.
    """
    matrix = np.asarray(matrix)
    fig = Figure(figsize=figsize, dpi=140)
    ax = fig.add_subplot(111)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("boundary slot", fontsize=9)
    ax.set_ylabel("bulk cell (row-major)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
