"""Inference-time interventions on the boundary state.

Two complementary ways to remove a slot's contribution:

``ablate_slots``
    Overwrite the slot's *content* (zero it, replace with the mean of surviving
    slots, randomise it, ...). The slot is still attended to, but carries no
    information about this particular bulk.
``slot_mask``
    Make the slot unreadable: the decoder's cross-attention simply cannot see
    it. This is the cleaner "the channel is gone" intervention and is what the
    boundary-subset experiments use.

Ablation is a causal intervention on the forward pass -- unlike attention
weights, a measured accuracy drop is evidence that the ablated slots were
actually *used*. It still does not tell you what they encoded, only that
removing them mattered.

All functions are non-mutating: the input boundary tensor is never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import torch

from . import topology

__all__ = [
    "ABLATION_MODES",
    "SELECTION_STRATEGIES",
    "AblationSpec",
    "AblationResult",
    "ablate_slots",
    "slot_mask_from_indices",
    "select_single",
    "select_contiguous",
    "select_random",
    "select_every_other",
    "select_fraction",
    "compare_ablation",
]

AblationMode = Literal["zero", "mean", "noise", "shuffle"]

#: Supported content-replacement modes for :func:`ablate_slots`.
ABLATION_MODES: tuple[str, ...] = ("zero", "mean", "noise", "shuffle")

#: Supported slot-selection strategies for :class:`AblationSpec`.
SELECTION_STRATEGIES: tuple[str, ...] = (
    "manual",
    "single",
    "contiguous",
    "random",
    "every_other",
    "fraction",
    "nearest_k",
    "none",
)


# ---------------------------------------------------------------------------
# Slot selection
# ---------------------------------------------------------------------------


def select_single(slot: int, num_slots: int) -> list[int]:
    """Exactly one slot."""
    return [topology.wrap_index(slot, num_slots)]


def select_contiguous(start: int, length: int, num_slots: int) -> list[int]:
    """A contiguous arc of the ring, wrapping past the last slot."""
    return topology.contiguous_slots(start, length, num_slots)


def select_random(
    num_slots: int,
    count: int | None = None,
    fraction: float | None = None,
    seed: int | None = None,
) -> list[int]:
    """A uniformly random subset, given either an explicit ``count`` or a ``fraction``."""
    if count is None:
        if fraction is None:
            raise ValueError("Provide either count or fraction")
        count = int(round(float(fraction) * num_slots))
    count = int(np.clip(count, 0, num_slots))
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(num_slots, size=count, replace=False))


def select_every_other(num_slots: int, offset: int = 0, step: int = 2) -> list[int]:
    """Every ``step``-th slot around the ring."""
    return topology.every_other_slots(num_slots, offset=offset, step=step)


def select_fraction(
    num_slots: int,
    fraction: float,
    strategy: str = "random",
    seed: int | None = None,
    offset: int = 0,
) -> list[int]:
    """A given fraction of slots, laid out randomly, contiguously or interleaved."""
    count = int(round(float(np.clip(fraction, 0.0, 1.0)) * num_slots))
    if strategy == "random":
        return select_random(num_slots, count=count, seed=seed)
    if strategy == "contiguous":
        return select_contiguous(offset, count, num_slots)
    if strategy == "every_other":
        return select_every_other(num_slots, offset=offset)[:count]
    raise ValueError(f"Unknown strategy {strategy!r}")


@dataclass
class AblationSpec:
    """A declarative, JSON-serialisable description of which slots to ablate.

    Storing the *spec* rather than the resolved index list keeps experiment
    records readable and lets the same intervention be replayed on a ring of a
    different size.
    """

    strategy: str = "none"
    mode: str = "zero"
    slots: tuple[int, ...] = ()
    start: int = 0
    length: int = 1
    count: int | None = None
    fraction: float | None = None
    offset: int = 0
    step: int = 2
    center: int = 0
    k: int = 1
    seed: int | None = None
    noise_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.strategy not in SELECTION_STRATEGIES:
            raise ValueError(f"Unknown strategy {self.strategy!r}")
        if self.mode not in ABLATION_MODES:
            raise ValueError(f"Unknown mode {self.mode!r}")
        self.slots = tuple(int(s) for s in self.slots)

    def resolve(self, num_slots: int) -> list[int]:
        """Turn the spec into concrete slot indices for a ring of ``num_slots``."""
        if self.strategy == "none":
            return []
        if self.strategy == "manual":
            return sorted({topology.wrap_index(s, num_slots) for s in self.slots})
        if self.strategy == "single":
            return select_single(self.start, num_slots)
        if self.strategy == "contiguous":
            return select_contiguous(self.start, self.length, num_slots)
        if self.strategy == "random":
            return select_random(num_slots, count=self.count, fraction=self.fraction, seed=self.seed)
        if self.strategy == "every_other":
            return select_every_other(num_slots, offset=self.offset, step=self.step)
        if self.strategy == "fraction":
            return select_fraction(
                num_slots, float(self.fraction or 0.0), "random", self.seed, self.offset
            )
        if self.strategy == "nearest_k":
            return topology.nearest_k_slots(self.center, self.k, num_slots)
        raise ValueError(f"Unhandled strategy {self.strategy!r}")

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        payload = asdict(self)
        payload["slots"] = list(self.slots)
        return payload


# ---------------------------------------------------------------------------
# The intervention itself
# ---------------------------------------------------------------------------


def ablate_slots(
    boundary: torch.Tensor,
    slot_indices: Sequence[int],
    mode: str = "zero",
    noise_scale: float = 1.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Return a copy of ``boundary`` with the given slots' contents destroyed.

    Args:
        boundary: ``(B, S, D)`` or ``(S, D)`` boundary state.
        slot_indices: Slots to ablate; indices are wrapped onto the ring, and
            duplicates are ignored.
        mode: One of :data:`ABLATION_MODES`.

            - ``"zero"``: overwrite with zeros.
            - ``"mean"``: overwrite with the mean of the *surviving* slots, so
              the ablated slot carries no slot-specific information but keeps a
              plausible activation scale.
            - ``"noise"``: overwrite with Gaussian noise scaled to the surviving
              slots' standard deviation.
            - ``"shuffle"``: replace with the same slot taken from a different
              batch element (a rolled batch), destroying the bulk-specific
              content while preserving the marginal distribution.
        noise_scale: Multiplier for the noise standard deviation (``"noise"`` only).
        seed: Seed for the ``"noise"`` mode, for reproducibility.

    Returns:
        A new tensor with the same shape and device as ``boundary``. The input
        is never modified.

    Raises:
        ValueError: On an unknown ``mode``.
    """
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode {mode!r}; expected one of {ABLATION_MODES}")

    squeezed = boundary.dim() == 2
    working = boundary.unsqueeze(0) if squeezed else boundary
    result = working.clone()

    num_slots = working.shape[1]
    indices = sorted({topology.wrap_index(int(s), num_slots) for s in slot_indices})
    if not indices:
        return boundary.clone()

    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=working.device)
    keep = torch.ones(num_slots, dtype=torch.bool, device=working.device)
    keep[index_tensor] = False

    if mode == "zero":
        result[:, index_tensor, :] = 0.0
    elif mode == "mean":
        if keep.any():
            replacement = working[:, keep, :].mean(dim=1, keepdim=True)
        else:
            replacement = torch.zeros_like(working[:, :1, :])
        result[:, index_tensor, :] = replacement.expand(-1, len(indices), -1)
    elif mode == "noise":
        source = working[:, keep, :] if keep.any() else working
        std = source.std().clamp_min(1e-6) * float(noise_scale)
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(
            (working.shape[0], len(indices), working.shape[2]), generator=generator
        ).to(device=working.device, dtype=working.dtype)
        result[:, index_tensor, :] = noise * std
    elif mode == "shuffle":
        rolled = torch.roll(working, shifts=1, dims=0)
        result[:, index_tensor, :] = rolled[:, index_tensor, :]

    return result.squeeze(0) if squeezed else result


def slot_mask_from_indices(
    ablated: Sequence[int],
    num_slots: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Boolean mask of *readable* slots: ``False`` at every index in ``ablated``.

    Pass the result as ``slot_mask`` to :meth:`hermes.model.HERMESModel.decode_query`
    to make those slots invisible to the decoder's cross-attention.
    """
    mask = torch.ones(num_slots, dtype=torch.bool)
    for slot in ablated:
        mask[topology.wrap_index(int(slot), num_slots)] = False
    return mask.to(device) if device is not None else mask


# ---------------------------------------------------------------------------
# Before / after comparison
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    """Baseline vs. ablated full-grid reconstruction for a single bulk grid."""

    ablated_slots: list[int]
    mode: str
    baseline_accuracy: float
    ablated_accuracy: float
    accuracy_drop: float
    baseline_bce: float
    ablated_bce: float
    baseline_probs: np.ndarray = field(repr=False)
    ablated_probs: np.ndarray = field(repr=False)
    prob_delta: np.ndarray = field(repr=False)
    baseline_prediction: np.ndarray = field(repr=False)
    ablated_prediction: np.ndarray = field(repr=False)
    prediction_diff: np.ndarray = field(repr=False)
    mean_abs_prob_change: float = 0.0
    num_cells_flipped: int = 0

    def to_dict(self, include_grids: bool = False) -> dict[str, Any]:
        """JSON-safe summary; grids are optional because they are bulky."""
        payload: dict[str, Any] = {
            "ablated_slots": list(self.ablated_slots),
            "mode": self.mode,
            "baseline_accuracy": self.baseline_accuracy,
            "ablated_accuracy": self.ablated_accuracy,
            "accuracy_drop": self.accuracy_drop,
            "baseline_bce": self.baseline_bce,
            "ablated_bce": self.ablated_bce,
            "mean_abs_prob_change": self.mean_abs_prob_change,
            "num_cells_flipped": self.num_cells_flipped,
        }
        if include_grids:
            payload["baseline_probs"] = self.baseline_probs.tolist()
            payload["ablated_probs"] = self.ablated_probs.tolist()
            payload["prob_delta"] = self.prob_delta.tolist()
        return payload


def compare_ablation(
    model,
    bulk,
    slot_indices: Sequence[int] | None = None,
    mode: str = "zero",
    spec: AblationSpec | None = None,
    use_mask: bool = False,
    threshold: float = 0.5,
    seed: int | None = None,
) -> AblationResult:
    """Reconstruct one grid before and after an intervention and diff the results.

    Args:
        model: A :class:`hermes.model.HERMESModel`.
        bulk: A single ``(G, G)`` grid (numpy array or tensor).
        slot_indices: Slots to ablate. Ignored when ``spec`` is given.
        mode: Content-replacement mode, see :func:`ablate_slots`. Ignored when
            ``use_mask`` is ``True``.
        spec: Optional :class:`AblationSpec`; takes precedence over
            ``slot_indices`` / ``mode``.
        use_mask: If ``True``, hide the slots from the decoder instead of
            overwriting their contents.
        threshold: Probability threshold for binarising the reconstruction.
        seed: Seed forwarded to noise-mode ablation.

    Returns:
        An :class:`AblationResult` with baseline/ablated accuracy, the accuracy
        drop, the per-cell probability change and the reconstruction difference map.
    """
    from .analysis import reconstruct_grid  # local import avoids a cycle

    num_slots = model.num_boundary_slots
    if spec is not None:
        slots = spec.resolve(num_slots)
        mode = spec.mode
        seed = spec.seed if seed is None else seed
    else:
        slots = sorted({topology.wrap_index(int(s), num_slots) for s in (slot_indices or [])})

    baseline = reconstruct_grid(model, bulk, threshold=threshold)

    if use_mask:
        mask = slot_mask_from_indices(slots, num_slots, device=baseline.boundary.device)
        ablated = reconstruct_grid(
            model, bulk, boundary=baseline.boundary, slot_mask=mask, threshold=threshold
        )
    else:
        perturbed = ablate_slots(
            baseline.boundary, slots, mode=mode, seed=seed,
            noise_scale=spec.noise_scale if spec else 1.0,
        )
        ablated = reconstruct_grid(
            model, bulk, boundary=perturbed, threshold=threshold
        )

    prob_delta = ablated.probabilities - baseline.probabilities
    prediction_diff = (ablated.prediction != baseline.prediction).astype(np.int8)

    return AblationResult(
        ablated_slots=slots,
        mode="mask" if use_mask else mode,
        baseline_accuracy=baseline.accuracy,
        ablated_accuracy=ablated.accuracy,
        accuracy_drop=baseline.accuracy - ablated.accuracy,
        baseline_bce=baseline.bce,
        ablated_bce=ablated.bce,
        baseline_probs=baseline.probabilities,
        ablated_probs=ablated.probabilities,
        prob_delta=prob_delta,
        baseline_prediction=baseline.prediction,
        ablated_prediction=ablated.prediction,
        prediction_diff=prediction_diff,
        mean_abs_prob_change=float(np.abs(prob_delta).mean()),
        num_cells_flipped=int(prediction_diff.sum()),
    )
