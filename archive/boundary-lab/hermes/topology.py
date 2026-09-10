"""Ring topology utilities for the boundary.

The boundary is a cycle of ``num_slots`` positions: slot ``num_slots - 1`` is
adjacent to slot ``0``. Version 1 does **not** restrict communication to local
neighbourhoods -- attention over the boundary is global. The topology is kept
explicit and separate so that locality constraints (banded attention, local-only
message passing, distance-decayed priors) can be layered on later without
touching the model code.

All functions are pure NumPy / Python and safe to use outside torch.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "wrap_index",
    "circular_distance",
    "circular_distance_matrix",
    "contiguous_slots",
    "nearest_k_slots",
    "every_other_slots",
    "ring_coordinates",
    "sinusoidal_ring_encoding",
    "is_contiguous",
    "slots_to_mask",
    "mask_to_slots",
]


def _check_num_slots(num_slots: int) -> int:
    if num_slots < 1:
        raise ValueError(f"num_slots must be >= 1, got {num_slots}")
    return int(num_slots)


def wrap_index(index: int | np.ndarray, num_slots: int) -> int | np.ndarray:
    """Wrap an index (possibly negative or out of range) onto the ring.

    ``wrap_index(-1, 20) == 19`` and ``wrap_index(20, 20) == 0``.
    """
    num_slots = _check_num_slots(num_slots)
    if isinstance(index, np.ndarray):
        return np.mod(index, num_slots)
    return int(index) % num_slots


def circular_distance(
    i: int | np.ndarray, j: int | np.ndarray, num_slots: int
) -> int | np.ndarray:
    """Shortest number of steps between two slots, going either way round.

    Args:
        i: Slot index or array of indices.
        j: Slot index or array of indices (broadcastable against ``i``).
        num_slots: Ring size.

    Returns:
        ``min(|i - j|, num_slots - |i - j|)``, so the maximum possible distance
        is ``num_slots // 2``.

    Example:
        >>> circular_distance(0, 19, 20)
        1
    """
    num_slots = _check_num_slots(num_slots)
    raw = np.abs(np.mod(np.asarray(i), num_slots) - np.mod(np.asarray(j), num_slots))
    dist = np.minimum(raw, num_slots - raw)
    if np.isscalar(i) and np.isscalar(j):
        return int(dist)
    return dist


def circular_distance_matrix(num_slots: int) -> np.ndarray:
    """``(num_slots, num_slots)`` matrix of pairwise circular distances."""
    num_slots = _check_num_slots(num_slots)
    idx = np.arange(num_slots)
    return circular_distance(idx[:, None], idx[None, :], num_slots)


def contiguous_slots(start: int, length: int, num_slots: int) -> list[int]:
    """A contiguous arc of ``length`` slots starting at ``start``, wrapping round.

    Args:
        start: First slot of the arc; wrapped onto the ring.
        length: Arc length; must be in ``[0, num_slots]``.
        num_slots: Ring size.

    Returns:
        Slot indices in walk order, e.g. ``contiguous_slots(18, 4, 20) ==
        [18, 19, 0, 1]``.

    Raises:
        ValueError: If ``length`` is negative or exceeds ``num_slots``.
    """
    num_slots = _check_num_slots(num_slots)
    if length < 0:
        raise ValueError("length must be >= 0")
    if length > num_slots:
        raise ValueError(f"length {length} exceeds ring size {num_slots}")
    start = wrap_index(start, num_slots)
    return [(start + offset) % num_slots for offset in range(length)]


def nearest_k_slots(center: int, k: int, num_slots: int, include_center: bool = True) -> list[int]:
    """The ``k`` slots closest to ``center`` on the ring.

    Slots are added in order of increasing circular distance; ties (a slot at
    distance ``d`` on each side) are broken deterministically by taking the
    clockwise (increasing-index) side first.

    Args:
        center: Slot to centre the neighbourhood on.
        k: How many slots to return.
        num_slots: Ring size.
        include_center: Whether ``center`` itself counts toward ``k``.

    Returns:
        ``k`` distinct slot indices, ordered by distance from ``center``.
    """
    num_slots = _check_num_slots(num_slots)
    if k < 0:
        raise ValueError("k must be >= 0")
    center = wrap_index(center, num_slots)

    order: list[int] = [center] if include_center else []
    offset = 1
    while len(order) < min(k, num_slots) and offset <= num_slots:
        for candidate in ((center + offset) % num_slots, (center - offset) % num_slots):
            if candidate not in order and len(order) < min(k, num_slots):
                order.append(candidate)
        offset += 1
    return order[:k]


def every_other_slots(num_slots: int, offset: int = 0, step: int = 2) -> list[int]:
    """Every ``step``-th slot, starting from ``offset``.

    With the default ``step=2`` this is the classic "every other slot" pattern
    used for interleaved ablations.
    """
    num_slots = _check_num_slots(num_slots)
    if step < 1:
        raise ValueError("step must be >= 1")
    return [s % num_slots for s in range(wrap_index(offset, num_slots), num_slots, step)]


def ring_coordinates(
    num_slots: int,
    radius: float = 1.0,
    start_angle: float = np.pi / 2,
    clockwise: bool = True,
) -> np.ndarray:
    """2-D positions of each slot on a circle, for plotting.

    Slot 0 sits at ``start_angle`` (top of the circle by default) and subsequent
    slots proceed clockwise.

    Returns:
        ``(num_slots, 2)`` array of ``(x, y)`` coordinates.
    """
    num_slots = _check_num_slots(num_slots)
    step = -1.0 if clockwise else 1.0
    angles = start_angle + step * 2.0 * np.pi * np.arange(num_slots) / num_slots
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def sinusoidal_ring_encoding(num_slots: int, dim: int) -> np.ndarray:
    """Fixed circular positional encoding, periodic in the ring index.

    Uses harmonics of the ring angle ``theta_s = 2*pi*s / num_slots``, so the
    encoding wraps exactly: slot ``num_slots`` would coincide with slot ``0``.

    Args:
        num_slots: Ring size.
        dim: Encoding width; must be even.

    Returns:
        ``(num_slots, dim)`` ``float32`` array.
    """
    num_slots = _check_num_slots(num_slots)
    if dim % 2 != 0:
        raise ValueError("dim must be even for sin/cos pairs")
    theta = 2.0 * np.pi * np.arange(num_slots) / num_slots
    harmonics = np.arange(1, dim // 2 + 1)
    angles = theta[:, None] * harmonics[None, :]
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1).astype(np.float32)


def is_contiguous(slots: Sequence[int], num_slots: int) -> bool:
    """Whether ``slots`` form one unbroken arc (wrap-around counts as unbroken)."""
    num_slots = _check_num_slots(num_slots)
    unique = sorted({wrap_index(s, num_slots) for s in slots})
    if not unique or len(unique) == num_slots:
        return True
    for start in unique:
        if set(contiguous_slots(start, len(unique), num_slots)) == set(unique):
            return True
    return False


def slots_to_mask(slots: Iterable[int], num_slots: int, value: bool = True) -> np.ndarray:
    """Boolean mask of length ``num_slots`` that is ``value`` at the given slots."""
    num_slots = _check_num_slots(num_slots)
    mask = np.zeros(num_slots, dtype=bool) if value else np.ones(num_slots, dtype=bool)
    for s in slots:
        mask[wrap_index(s, num_slots)] = value
    return mask


def mask_to_slots(mask: np.ndarray) -> list[int]:
    """Indices where a boolean mask is ``True``."""
    return [int(i) for i in np.flatnonzero(np.asarray(mask).astype(bool))]
