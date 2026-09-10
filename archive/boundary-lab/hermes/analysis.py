"""Reconstruction, evaluation and mechanistic-analysis helpers.

Three families of tool live here:

* **Reconstruction** -- query every cell and rebuild the grid from the boundary
  alone (:func:`reconstruct_grid`, :func:`reconstruct_batch`).
* **Evaluation** -- accuracy / BCE overall and broken down by pattern type
  (:func:`evaluate_dataset`).
* **Association measures** -- encoder cross-attention as a *descriptive* map of
  which boundary slots read which bulk cells (:func:`bulk_to_boundary_influence`).

.. warning::
   Attention weights are an **exploratory association measure, not a causal
   explanation**. A slot can attend strongly to a cell whose value it never
   propagates, and a slot can carry information about a cell it barely attends
   to (for instance via the boundary self-attention layers, which let content
   spread around the ring after the read). Use
   :mod:`hermes.interventions` -- ablation and subset decoding -- when you want
   evidence that a slot is actually *used*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import topology
from .data import BulkDataset
from .model import HERMESModel, all_cell_queries, as_bulk_tensor

__all__ = [
    "Reconstruction",
    "BatchReconstruction",
    "SubsetCurveResult",
    "reconstruct_grid",
    "reconstruct_batch",
    "evaluate_dataset",
    "bulk_to_boundary_influence",
    "influence_matrix",
    "boundary_slot_magnitudes",
    "decoder_query_attention",
    "boundary_subset_curve",
    "default_subset_sizes",
]


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


@dataclass
class Reconstruction:
    """Full-grid reconstruction of one bulk, produced by querying every cell.

    The bulk grid is used **only** to score the result -- the decoder saw the
    boundary and the query coordinates, nothing else.
    """

    probabilities: np.ndarray  # (G, G) float32 in [0, 1]
    prediction: np.ndarray  # (G, G) uint8
    truth: np.ndarray  # (G, G) uint8
    error_map: np.ndarray  # (G, G) uint8, 1 where prediction != truth
    logits: np.ndarray = field(repr=False)
    accuracy: float = 0.0
    bce: float = 0.0
    exact_match: bool = False
    boundary: torch.Tensor | None = field(default=None, repr=False)

    def to_dict(self, include_grids: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "accuracy": self.accuracy,
            "bce": self.bce,
            "exact_match": self.exact_match,
            "num_errors": int(self.error_map.sum()),
        }
        if include_grids:
            payload["probabilities"] = self.probabilities.tolist()
            payload["prediction"] = self.prediction.tolist()
            payload["truth"] = self.truth.tolist()
        return payload


@dataclass
class BatchReconstruction:
    """Reconstructions for a batch of grids."""

    probabilities: np.ndarray  # (B, G, G)
    prediction: np.ndarray  # (B, G, G)
    truth: np.ndarray  # (B, G, G)
    accuracy: float
    bce: float
    exact_match_rate: float
    per_sample_accuracy: np.ndarray  # (B,)


def _as_boundary(model: HERMESModel, boundary: torch.Tensor) -> torch.Tensor:
    if boundary.dim() == 2:
        boundary = boundary.unsqueeze(0)
    return boundary.to(model.device)


@torch.no_grad()
def reconstruct_grid(
    model: HERMESModel,
    bulk,
    boundary: torch.Tensor | None = None,
    slot_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> Reconstruction:
    """Rebuild a single grid by querying all ``grid_size ** 2`` cells.

    Args:
        model: Trained (or untrained) :class:`~hermes.model.HERMESModel`.
        bulk: The ``(G, G)`` ground-truth grid. Used to encode the boundary when
            ``boundary`` is ``None``, and always used for scoring.
        boundary: Optional precomputed / perturbed boundary. Supply this to
            decode from an ablated state without re-encoding.
        slot_mask: Optional boolean mask of readable slots, ``(S,)`` or ``(1, S)``.
        threshold: Probability cut for the binarised prediction.

    Returns:
        A :class:`Reconstruction` with the probability grid, the thresholded
        grid, the error map, cell accuracy and mean BCE.
    """
    was_training = model.training
    model.eval()
    try:
        bulk_tensor = as_bulk_tensor(bulk, device=model.device)
        if bulk_tensor.shape[0] != 1:
            raise ValueError("reconstruct_grid expects a single grid; use reconstruct_batch")

        boundary = model.encode_bulk(bulk_tensor) if boundary is None else _as_boundary(model, boundary)

        grid_size = model.grid_size
        queries = all_cell_queries(grid_size, device=model.device).unsqueeze(0)
        if slot_mask is not None and not isinstance(slot_mask, torch.Tensor):
            slot_mask = torch.as_tensor(np.asarray(slot_mask))
        if slot_mask is not None:
            slot_mask = slot_mask.to(model.device)

        logits = model.decode_query(boundary, queries, slot_mask)[0]  # (N,)
        target = bulk_tensor.reshape(-1).to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target).item()

        probs = torch.sigmoid(logits).reshape(grid_size, grid_size).cpu().numpy()
        truth = bulk_tensor.reshape(grid_size, grid_size).cpu().numpy().astype(np.uint8)
        prediction = (probs >= threshold).astype(np.uint8)
        error_map = (prediction != truth).astype(np.uint8)

        return Reconstruction(
            probabilities=probs.astype(np.float32),
            prediction=prediction,
            truth=truth,
            error_map=error_map,
            logits=logits.reshape(grid_size, grid_size).cpu().numpy(),
            accuracy=float((prediction == truth).mean()),
            bce=float(bce),
            exact_match=bool(error_map.sum() == 0),
            boundary=boundary,
        )
    finally:
        model.train(was_training)


@torch.no_grad()
def reconstruct_batch(
    model: HERMESModel,
    bulks,
    slot_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
    batch_size: int = 256,
) -> BatchReconstruction:
    """Full-grid reconstruction for many grids at once."""
    was_training = model.training
    model.eval()
    try:
        all_bulk = as_bulk_tensor(bulks, device=model.device)
        grid_size = model.grid_size
        queries = all_cell_queries(grid_size, device=model.device)
        if slot_mask is not None:
            if not isinstance(slot_mask, torch.Tensor):
                slot_mask = torch.as_tensor(np.asarray(slot_mask))
            slot_mask = slot_mask.to(model.device)

        probs_chunks, loss_sum, count = [], 0.0, 0
        for start in range(0, all_bulk.shape[0], batch_size):
            chunk = all_bulk[start : start + batch_size]
            boundary = model.encode_bulk(chunk)
            batch_queries = queries.unsqueeze(0).expand(chunk.shape[0], -1, -1)
            logits = model.decode_query(boundary, batch_queries, slot_mask)
            target = chunk.reshape(chunk.shape[0], -1).to(logits.dtype)
            loss_sum += F.binary_cross_entropy_with_logits(
                logits, target, reduction="sum"
            ).item()
            count += target.numel()
            probs_chunks.append(torch.sigmoid(logits).cpu())

        probs = torch.cat(probs_chunks).reshape(-1, grid_size, grid_size).numpy()
        truth = all_bulk.reshape(-1, grid_size, grid_size).cpu().numpy().astype(np.uint8)
        prediction = (probs >= threshold).astype(np.uint8)
        correct = prediction == truth
        per_sample = correct.reshape(correct.shape[0], -1).mean(axis=1)

        return BatchReconstruction(
            probabilities=probs.astype(np.float32),
            prediction=prediction,
            truth=truth,
            accuracy=float(correct.mean()),
            bce=float(loss_sum / max(count, 1)),
            exact_match_rate=float((per_sample == 1.0).mean()),
            per_sample_accuracy=per_sample.astype(np.float32),
        )
    finally:
        model.train(was_training)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_dataset(
    model: HERMESModel,
    dataset: BulkDataset,
    threshold: float = 0.5,
    batch_size: int = 256,
    slot_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Score a model on a dataset by querying all cells of every grid.

    Returns:
        A JSON-safe metrics dict containing overall cell accuracy and BCE, the
        exact-grid-match rate, per-pattern breakdowns, and an all-zeros majority
        baseline for context (these grids are sparse, so raw accuracy alone is
        easy to misread).
    """
    recon = reconstruct_batch(
        model, dataset.grids, slot_mask=slot_mask, threshold=threshold, batch_size=batch_size
    )

    accuracy_by_pattern: dict[str, float] = {}
    exact_by_pattern: dict[str, float] = {}
    count_by_pattern: dict[str, int] = {}
    patterns = np.array(dataset.pattern_types)
    for pattern in sorted(set(dataset.pattern_types)):
        mask = patterns == pattern
        subset = recon.per_sample_accuracy[mask]
        accuracy_by_pattern[pattern] = float(subset.mean())
        exact_by_pattern[pattern] = float((subset == 1.0).mean())
        count_by_pattern[pattern] = int(mask.sum())

    density = float(dataset.grids.mean())
    return {
        "num_samples": len(dataset),
        "cell_accuracy": recon.accuracy,
        "full_grid_cell_accuracy": recon.accuracy,
        "bce": recon.bce,
        "exact_grid_match_rate": recon.exact_match_rate,
        "accuracy_by_pattern": accuracy_by_pattern,
        "exact_match_by_pattern": exact_by_pattern,
        "count_by_pattern": count_by_pattern,
        "mean_density": density,
        "majority_baseline_accuracy": float(max(density, 1.0 - density)),
    }


# ---------------------------------------------------------------------------
# Association measures (exploratory, not causal)
# ---------------------------------------------------------------------------


@torch.no_grad()
def bulk_to_boundary_influence(
    model: HERMESModel,
    bulk,
    cell: tuple[int, int],
    layer: int = -1,
    head: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """How much each boundary slot attends to one bulk cell.

    .. warning::
       This is encoder cross-attention weight, i.e. an **association measure**.
       High weight means "this slot's read focused here", not "this slot causally
       carries this cell". For causal evidence, ablate the slot and measure the
       change in the reconstruction of that cell.

    Args:
        model: The model to inspect.
        bulk: A single ``(G, G)`` grid.
        cell: ``(row, col)`` of the bulk cell of interest.
        layer: Which encoder cross-attention layer to read (``-1`` = last).
        head: A specific attention head, or ``None`` to average over heads.
        normalize: Rescale the returned vector to sum to 1.

    Returns:
        ``(num_boundary_slots,)`` float array. Entry ``s`` is slot ``s``'s
        attention to the requested cell.
    """
    was_training = model.training
    model.eval()
    try:
        bulk_tensor = as_bulk_tensor(bulk, device=model.device)
        _, attention_maps = model.encode_bulk_with_attention(bulk_tensor)
        attn = attention_maps[layer]  # (B, H, S, N)
        attn = attn.mean(dim=1) if head is None else attn[:, head]  # (B, S, N)

        row, col = int(cell[0]), int(cell[1])
        if not (0 <= row < model.grid_size and 0 <= col < model.grid_size):
            raise ValueError(f"cell {cell} outside the {model.grid_size}x{model.grid_size} grid")
        index = row * model.grid_size + col

        influence = attn[..., index].mean(dim=0).cpu().numpy().astype(np.float32)
        if normalize:
            total = float(influence.sum())
            if total > 0:
                influence = influence / total
        return influence
    finally:
        model.train(was_training)


@torch.no_grad()
def influence_matrix(
    model: HERMESModel, bulk, layer: int = -1, head: int | None = None, normalize: bool = True
) -> np.ndarray:
    """Attention from every bulk cell to every boundary slot.

    Returns:
        ``(num_cells, num_boundary_slots)`` in row-major cell order, so
        ``matrix[r * G + c]`` is the influence vector for cell ``(r, c)``.
        Same caveat as :func:`bulk_to_boundary_influence`: association, not cause.
    """
    was_training = model.training
    model.eval()
    try:
        bulk_tensor = as_bulk_tensor(bulk, device=model.device)
        _, attention_maps = model.encode_bulk_with_attention(bulk_tensor)
        attn = attention_maps[layer]
        attn = attn.mean(dim=1) if head is None else attn[:, head]  # (B, S, N)
        matrix = attn.mean(dim=0).transpose(0, 1).cpu().numpy().astype(np.float32)  # (N, S)
        if normalize:
            totals = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
        return matrix
    finally:
        model.train(was_training)


def boundary_slot_magnitudes(boundary: torch.Tensor, ord: float = 2.0) -> np.ndarray:
    """L2 (by default) norm of each boundary slot vector: ``(num_boundary_slots,)``.

    A crude "how active is this slot" readout, averaged over the batch if the
    boundary is batched.
    """
    tensor = boundary.detach()
    if tensor.dim() == 3:
        tensor = tensor.mean(dim=0)
    return torch.linalg.vector_norm(tensor, ord=ord, dim=-1).cpu().numpy().astype(np.float32)


@torch.no_grad()
def decoder_query_attention(
    model: HERMESModel,
    boundary: torch.Tensor,
    cell: tuple[int, int],
    slot_mask: torch.Tensor | None = None,
    layer: int = -1,
    head: int | None = None,
) -> np.ndarray:
    """Which boundary slots the decoder reads when answering a given query.

    Returns:
        ``(num_boundary_slots,)`` attention weights. Exploratory, like all
        attention measures.
    """
    was_training = model.training
    model.eval()
    try:
        queries = torch.tensor([[[int(cell[0]), int(cell[1])]]], device=model.device)
        _, maps = model.decode_query_with_attention(boundary, queries, slot_mask)
        attn = maps[layer]  # (B, H, Q, S)
        attn = attn.mean(dim=1) if head is None else attn[:, head]
        return attn[0, 0].cpu().numpy().astype(np.float32)
    finally:
        model.train(was_training)


# ---------------------------------------------------------------------------
# Boundary-subset decoding
# ---------------------------------------------------------------------------


def default_subset_sizes(num_slots: int) -> list[int]:
    """The standard sweep ``1, 2, 4, 8, 12, 16, 20`` clipped to the ring size."""
    canonical = [1, 2, 4, 8, 12, 16, 20]
    sizes = sorted({min(s, num_slots) for s in canonical if s <= num_slots} | {num_slots})
    return sizes


@dataclass
class SubsetCurveResult:
    """Reconstruction quality as a function of how many boundary slots are readable."""

    mode: str  # "contiguous" or "random"
    num_slots: int
    sizes: list[int]
    accuracy_mean: list[float]
    accuracy_std: list[float]
    bce_mean: list[float]
    exact_match_mean: list[float]
    trials: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@torch.no_grad()
def boundary_subset_curve(
    model: HERMESModel,
    bulks,
    sizes: Sequence[int] | None = None,
    mode: str = "contiguous",
    trials: int = 5,
    seed: int = 0,
    threshold: float = 0.5,
    batch_size: int = 256,
) -> SubsetCurveResult:
    """Measure reconstruction as the readable subset of the boundary grows.

    Slots outside the subset are hidden from the decoder's cross-attention
    (a masking intervention), so the decoder truly reads only the subset.

    Args:
        model: Model to evaluate.
        bulks: ``(B, G, G)`` grids, or a :class:`~hermes.data.BulkDataset`.
        sizes: Subset sizes to sweep; defaults to :func:`default_subset_sizes`.
        mode: ``"contiguous"`` (an arc of the ring) or ``"random"``
            (slots scattered around it).
        trials: Random restarts per size; results are averaged. The arc start
            (or the random subset) is redrawn each trial.
        seed: Seed controlling subset choice, for reproducibility.
        threshold: Binarisation threshold.

    Returns:
        A :class:`SubsetCurveResult` ready to plot or serialise.
    """
    if mode not in ("contiguous", "random"):
        raise ValueError("mode must be 'contiguous' or 'random'")
    if isinstance(bulks, BulkDataset):
        bulks = bulks.grids

    num_slots = model.num_boundary_slots
    sizes = list(sizes) if sizes is not None else default_subset_sizes(num_slots)
    rng = np.random.default_rng(seed)

    accuracy_mean, accuracy_std, bce_mean, exact_mean = [], [], [], []
    for size in sizes:
        size = int(np.clip(size, 0, num_slots))
        accuracies, bces, exacts = [], [], []
        effective_trials = 1 if size == num_slots else trials
        for _ in range(effective_trials):
            if mode == "contiguous":
                start = int(rng.integers(0, num_slots))
                slots = topology.contiguous_slots(start, size, num_slots)
            else:
                slots = [int(i) for i in rng.choice(num_slots, size=size, replace=False)]
            mask = torch.zeros(num_slots, dtype=torch.bool)
            mask[torch.as_tensor(slots, dtype=torch.long)] = True
            recon = reconstruct_batch(
                model, bulks, slot_mask=mask, threshold=threshold, batch_size=batch_size
            )
            accuracies.append(recon.accuracy)
            bces.append(recon.bce)
            exacts.append(recon.exact_match_rate)
        accuracy_mean.append(float(np.mean(accuracies)))
        accuracy_std.append(float(np.std(accuracies)))
        bce_mean.append(float(np.mean(bces)))
        exact_mean.append(float(np.mean(exacts)))

    return SubsetCurveResult(
        mode=mode,
        num_slots=num_slots,
        sizes=[int(s) for s in sizes],
        accuracy_mean=accuracy_mean,
        accuracy_std=accuracy_std,
        bce_mean=bce_mean,
        exact_match_mean=exact_mean,
        trials=trials,
        seed=seed,
    )
