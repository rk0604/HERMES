"""Ring topology: wraparound, contiguity, neighbourhoods and encodings."""

from __future__ import annotations

import numpy as np
import pytest

from hermes.topology import (
    circular_distance,
    circular_distance_matrix,
    contiguous_slots,
    every_other_slots,
    is_contiguous,
    mask_to_slots,
    nearest_k_slots,
    ring_coordinates,
    sinusoidal_ring_encoding,
    slots_to_mask,
    wrap_index,
)

N = 20


def test_wrap_index_handles_negatives_and_overflow() -> None:
    assert wrap_index(-1, N) == 19
    assert wrap_index(N, N) == 0
    assert wrap_index(N + 3, N) == 3
    assert wrap_index(-N - 1, N) == 19


def test_circular_distance_wraps_around_the_seam() -> None:
    assert circular_distance(0, 19, N) == 1
    assert circular_distance(19, 0, N) == 1
    assert circular_distance(18, 1, N) == 3
    assert circular_distance(0, 0, N) == 0


def test_circular_distance_is_bounded_by_half_the_ring() -> None:
    assert circular_distance(0, 10, N) == 10
    for i in range(N):
        for j in range(N):
            assert 0 <= circular_distance(i, j, N) <= N // 2


def test_circular_distance_is_symmetric_and_vectorised() -> None:
    matrix = circular_distance_matrix(N)
    assert matrix.shape == (N, N)
    assert np.array_equal(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0)
    assert matrix[0, 19] == 1

    left = np.array([0, 5, 19])
    right = np.array([19, 5, 1])
    assert np.array_equal(circular_distance(left, right, N), np.array([1, 0, 2]))


def test_odd_ring_distance() -> None:
    assert circular_distance(0, 4, 7) == 3
    assert circular_distance(0, 3, 7) == 3


def test_contiguous_selection_wraps_past_the_last_slot() -> None:
    assert contiguous_slots(18, 4, N) == [18, 19, 0, 1]
    assert contiguous_slots(0, 3, N) == [0, 1, 2]
    assert contiguous_slots(19, 1, N) == [19]


def test_contiguous_selection_edge_cases() -> None:
    assert contiguous_slots(5, 0, N) == []
    assert sorted(contiguous_slots(7, N, N)) == list(range(N))
    assert contiguous_slots(-2, 3, N) == [18, 19, 0]
    with pytest.raises(ValueError):
        contiguous_slots(0, N + 1, N)
    with pytest.raises(ValueError):
        contiguous_slots(0, -1, N)


def test_contiguous_slots_are_all_distinct() -> None:
    for start in range(N):
        for length in range(N + 1):
            slots = contiguous_slots(start, length, N)
            assert len(slots) == len(set(slots)) == length


def test_nearest_k_is_centred_and_wraps() -> None:
    assert nearest_k_slots(0, 1, N) == [0]
    assert set(nearest_k_slots(0, 3, N)) == {0, 1, 19}
    assert set(nearest_k_slots(0, 5, N)) == {0, 1, 19, 2, 18}
    assert set(nearest_k_slots(19, 3, N)) == {19, 0, 18}


def test_nearest_k_orders_by_distance() -> None:
    slots = nearest_k_slots(10, 7, N)
    distances = [circular_distance(10, s, N) for s in slots]
    assert distances == sorted(distances)
    assert len(set(slots)) == 7


def test_nearest_k_can_exclude_the_centre_and_saturates() -> None:
    assert 5 not in nearest_k_slots(5, 4, N, include_center=False)
    assert sorted(nearest_k_slots(0, 999, N)) == list(range(N))


def test_every_other_slots() -> None:
    assert every_other_slots(N) == list(range(0, N, 2))
    assert every_other_slots(N, offset=1) == list(range(1, N, 2))
    assert every_other_slots(N, step=5) == [0, 5, 10, 15]
    assert len(every_other_slots(N)) == N // 2


def test_is_contiguous_detects_arcs_including_wrapped_ones() -> None:
    assert is_contiguous([18, 19, 0, 1], N)
    assert is_contiguous([3, 4, 5], N)
    assert is_contiguous([7], N)
    assert is_contiguous(list(range(N)), N)
    assert not is_contiguous([0, 5, 10], N)
    assert not is_contiguous([0, 1, 3], N)


def test_mask_helpers_roundtrip() -> None:
    slots = [0, 4, 19]
    mask = slots_to_mask(slots, N)
    assert mask.shape == (N,) and mask.dtype == bool
    assert mask_to_slots(mask) == sorted(slots)
    assert slots_to_mask([-1], N)[19]


def test_ring_coordinates_lie_on_the_unit_circle_and_close_the_loop() -> None:
    coords = ring_coordinates(N)
    assert coords.shape == (N, 2)
    assert np.allclose(np.linalg.norm(coords, axis=1), 1.0)
    # Consecutive slots are equally spaced, including from the last back to the first.
    gaps = np.linalg.norm(coords - np.roll(coords, -1, axis=0), axis=1)
    assert np.allclose(gaps, gaps[0])


def test_sinusoidal_encoding_is_periodic_on_the_ring() -> None:
    encoding = sinusoidal_ring_encoding(N, 8)
    assert encoding.shape == (N, 8)
    assert encoding.dtype == np.float32
    # Slot N would coincide exactly with slot 0.
    theta = 2 * np.pi * N / N
    harmonics = np.arange(1, 5)
    wrapped = np.concatenate([np.sin(theta * harmonics), np.cos(theta * harmonics)])
    assert np.allclose(wrapped, encoding[0], atol=1e-6)


def test_sinusoidal_encoding_rejects_odd_dims() -> None:
    with pytest.raises(ValueError):
        sinusoidal_ring_encoding(N, 7)


def test_invalid_ring_sizes_raise() -> None:
    with pytest.raises(ValueError):
        circular_distance(0, 1, 0)
    with pytest.raises(ValueError):
        contiguous_slots(0, 1, -3)
