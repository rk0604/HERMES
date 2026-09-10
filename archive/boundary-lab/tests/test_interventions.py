"""Boundary ablation: selection, non-mutation, and before/after comparison."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hermes.analysis import reconstruct_grid
from hermes.config import ModelConfig
from hermes.data import generate_sample
from hermes.interventions import (
    ABLATION_MODES,
    AblationSpec,
    ablate_slots,
    compare_ablation,
    select_contiguous,
    select_every_other,
    select_fraction,
    select_random,
    select_single,
    slot_mask_from_indices,
)
from hermes.model import HERMESModel

N = 20


@pytest.fixture(scope="module")
def model() -> HERMESModel:
    torch.manual_seed(0)
    return HERMESModel(ModelConfig()).eval()


@pytest.fixture(scope="module")
def grid() -> np.ndarray:
    return generate_sample(pattern="rectangle_filled", seed=11).grid


@pytest.fixture()
def boundary(model: HERMESModel, grid: np.ndarray) -> torch.Tensor:
    with torch.no_grad():
        return model.encode_bulk(torch.from_numpy(grid.astype(np.float32)))


# -- non-mutation ----------------------------------------------------------


@pytest.mark.parametrize("mode", ABLATION_MODES)
def test_ablation_never_mutates_the_original_boundary(
    boundary: torch.Tensor, mode: str
) -> None:
    original = boundary.clone()
    ablated = ablate_slots(boundary, [0, 3, 7], mode=mode, seed=0)
    assert torch.equal(boundary, original), f"mode {mode} mutated its input"
    assert ablated is not boundary
    assert ablated.data_ptr() != boundary.data_ptr()


def test_ablation_preserves_shape_dtype_and_device(boundary: torch.Tensor) -> None:
    ablated = ablate_slots(boundary, [1, 2])
    assert ablated.shape == boundary.shape
    assert ablated.dtype == boundary.dtype
    assert ablated.device == boundary.device


def test_unbatched_boundary_keeps_its_rank(boundary: torch.Tensor) -> None:
    single = boundary[0]
    assert ablate_slots(single, [0]).shape == single.shape


# -- modes -----------------------------------------------------------------


def test_zero_mode_zeroes_only_the_selected_slots(boundary: torch.Tensor) -> None:
    ablated = ablate_slots(boundary, [2, 5], mode="zero")
    assert torch.count_nonzero(ablated[:, [2, 5], :]) == 0
    keep = [s for s in range(N) if s not in (2, 5)]
    assert torch.equal(ablated[:, keep, :], boundary[:, keep, :])


def test_mean_mode_uses_the_surviving_slots(boundary: torch.Tensor) -> None:
    ablated = ablate_slots(boundary, [4], mode="mean")
    keep = [s for s in range(N) if s != 4]
    expected = boundary[:, keep, :].mean(dim=1)
    assert torch.allclose(ablated[:, 4, :], expected, atol=1e-6)


def test_noise_mode_is_reproducible_under_a_seed(boundary: torch.Tensor) -> None:
    first = ablate_slots(boundary, [1, 9], mode="noise", seed=42)
    second = ablate_slots(boundary, [1, 9], mode="noise", seed=42)
    third = ablate_slots(boundary, [1, 9], mode="noise", seed=43)
    assert torch.allclose(first, second)
    assert not torch.allclose(first, third)


def test_shuffle_mode_borrows_from_another_batch_element(model: HERMESModel) -> None:
    batch = torch.randn(3, N, model.config.boundary_dim)
    ablated = ablate_slots(batch, [6], mode="shuffle")
    rolled = torch.roll(batch, shifts=1, dims=0)
    assert torch.allclose(ablated[:, 6, :], rolled[:, 6, :])


def test_empty_selection_is_a_no_op_copy(boundary: torch.Tensor) -> None:
    ablated = ablate_slots(boundary, [], mode="zero")
    assert torch.equal(ablated, boundary)
    assert ablated is not boundary


def test_indices_are_wrapped_and_deduplicated(boundary: torch.Tensor) -> None:
    ablated = ablate_slots(boundary, [-1, 19, N + 0, 0], mode="zero")
    assert torch.count_nonzero(ablated[:, [0, 19], :]) == 0
    assert torch.equal(ablated[:, 1:19, :], boundary[:, 1:19, :])


def test_unknown_mode_raises(boundary: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        ablate_slots(boundary, [0], mode="teleport")


def test_ablating_every_slot_zeroes_the_whole_boundary(boundary: torch.Tensor) -> None:
    ablated = ablate_slots(boundary, list(range(N)), mode="zero")
    assert torch.count_nonzero(ablated) == 0


# -- selection helpers -----------------------------------------------------


def test_selection_helpers() -> None:
    assert select_single(3, N) == [3]
    assert select_single(-1, N) == [19]
    assert select_contiguous(18, 4, N) == [18, 19, 0, 1]
    assert select_every_other(N) == list(range(0, N, 2))

    random_slots = select_random(N, count=6, seed=0)
    assert len(random_slots) == len(set(random_slots)) == 6
    assert select_random(N, count=6, seed=0) == random_slots
    assert select_random(N, fraction=0.5, seed=1) != []
    assert len(select_random(N, fraction=0.25, seed=1)) == 5

    assert len(select_fraction(N, 0.5, "contiguous")) == 10
    assert len(select_fraction(N, 0.5, "every_other")) == 10
    with pytest.raises(ValueError):
        select_fraction(N, 0.5, "spiral")
    with pytest.raises(ValueError):
        select_random(N)


def test_slot_mask_marks_readable_slots() -> None:
    mask = slot_mask_from_indices([0, 1, 2], N)
    assert mask.shape == (N,) and mask.dtype == torch.bool
    assert not mask[:3].any()
    assert mask[3:].all()


# -- AblationSpec ----------------------------------------------------------


def test_spec_resolves_each_strategy() -> None:
    assert AblationSpec(strategy="none").resolve(N) == []
    assert AblationSpec(strategy="manual", slots=(5, 1, 1)).resolve(N) == [1, 5]
    assert AblationSpec(strategy="single", start=4).resolve(N) == [4]
    assert AblationSpec(strategy="contiguous", start=19, length=3).resolve(N) == [19, 0, 1]
    assert len(AblationSpec(strategy="random", count=7, seed=0).resolve(N)) == 7
    assert AblationSpec(strategy="every_other").resolve(N) == list(range(0, N, 2))
    assert len(AblationSpec(strategy="fraction", fraction=0.3, seed=0).resolve(N)) == 6
    assert set(AblationSpec(strategy="nearest_k", center=0, k=3).resolve(N)) == {0, 1, 19}


def test_spec_is_json_safe_and_validated() -> None:
    import json

    spec = AblationSpec(strategy="contiguous", start=2, length=5, mode="mean")
    assert json.loads(json.dumps(spec.to_dict()))["strategy"] == "contiguous"
    with pytest.raises(ValueError):
        AblationSpec(strategy="wormhole")
    with pytest.raises(ValueError):
        AblationSpec(strategy="single", mode="evaporate")


def test_spec_resolution_is_seed_reproducible() -> None:
    spec = AblationSpec(strategy="random", count=5, seed=3)
    assert spec.resolve(N) == spec.resolve(N)


# -- comparison ------------------------------------------------------------


def test_compare_ablation_reports_a_full_grid_diff(
    model: HERMESModel, grid: np.ndarray
) -> None:
    result = compare_ablation(model, grid, slot_indices=[0, 1, 2, 3], mode="zero")
    assert result.baseline_probs.shape == (10, 10)
    assert result.ablated_probs.shape == (10, 10)
    assert result.prob_delta.shape == (10, 10)
    assert result.prediction_diff.shape == (10, 10)
    assert result.ablated_slots == [0, 1, 2, 3]
    assert result.accuracy_drop == pytest.approx(
        result.baseline_accuracy - result.ablated_accuracy
    )
    assert 0.0 <= result.baseline_accuracy <= 1.0
    assert result.num_cells_flipped == int(result.prediction_diff.sum())


def test_compare_ablation_with_no_slots_changes_nothing(
    model: HERMESModel, grid: np.ndarray
) -> None:
    result = compare_ablation(model, grid, slot_indices=[])
    assert result.accuracy_drop == pytest.approx(0.0)
    assert np.allclose(result.prob_delta, 0.0, atol=1e-6)


def test_compare_ablation_accepts_a_spec_and_mask_mode(
    model: HERMESModel, grid: np.ndarray
) -> None:
    spec = AblationSpec(strategy="contiguous", start=0, length=10, mode="zero")
    by_value = compare_ablation(model, grid, spec=spec)
    by_mask = compare_ablation(model, grid, slot_indices=spec.resolve(N), use_mask=True)
    assert by_value.ablated_slots == by_mask.ablated_slots
    assert by_mask.mode == "mask"
    assert not np.allclose(by_value.ablated_probs, by_mask.ablated_probs)


def test_compare_ablation_result_serialises(model: HERMESModel, grid: np.ndarray) -> None:
    import json

    result = compare_ablation(model, grid, slot_indices=[1, 2])
    payload = json.loads(json.dumps(result.to_dict(include_grids=True)))
    assert payload["mode"] == "zero"
    assert np.array(payload["prob_delta"]).shape == (10, 10)


def test_querying_all_cells_returns_a_10x10_reconstruction(
    model: HERMESModel, grid: np.ndarray
) -> None:
    recon = reconstruct_grid(model, grid)
    assert recon.probabilities.shape == (10, 10)
    assert recon.prediction.shape == (10, 10)
    assert recon.error_map.shape == (10, 10)
    assert recon.truth.shape == (10, 10)
    assert set(np.unique(recon.prediction)).issubset({0, 1})
    assert ((recon.probabilities >= 0) & (recon.probabilities <= 1)).all()
    assert recon.accuracy == pytest.approx(1.0 - recon.error_map.mean())


def test_reconstruction_from_an_ablated_boundary_uses_the_given_state(
    model: HERMESModel, grid: np.ndarray, boundary: torch.Tensor
) -> None:
    baseline = reconstruct_grid(model, grid)
    perturbed = ablate_slots(boundary, list(range(N)), mode="zero")
    ablated = reconstruct_grid(model, grid, boundary=perturbed)
    assert not np.allclose(baseline.probabilities, ablated.probabilities)
    # The ground truth is only ever used for scoring, so it is unchanged.
    assert np.array_equal(baseline.truth, ablated.truth)
