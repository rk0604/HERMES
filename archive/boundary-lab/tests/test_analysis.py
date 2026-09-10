"""Reconstruction, evaluation, influence measures and subset-decoding curves."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from hermes.analysis import (
    boundary_slot_magnitudes,
    boundary_subset_curve,
    bulk_to_boundary_influence,
    decoder_query_attention,
    default_subset_sizes,
    evaluate_dataset,
    influence_matrix,
    reconstruct_batch,
    reconstruct_grid,
)
from hermes.config import DataConfig, ModelConfig
from hermes.data import generate_dataset, generate_sample
from hermes.model import HERMESModel

N = 20


@pytest.fixture(scope="module")
def model() -> HERMESModel:
    torch.manual_seed(0)
    return HERMESModel(ModelConfig()).eval()


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(24, DataConfig(seed=5))


def test_reconstruct_batch_shapes_and_ranges(model: HERMESModel, dataset) -> None:
    recon = reconstruct_batch(model, dataset.grids)
    assert recon.probabilities.shape == (24, 10, 10)
    assert recon.prediction.shape == (24, 10, 10)
    assert recon.truth.shape == (24, 10, 10)
    assert recon.per_sample_accuracy.shape == (24,)
    assert 0.0 <= recon.accuracy <= 1.0
    assert 0.0 <= recon.exact_match_rate <= 1.0
    assert np.array_equal(recon.truth, dataset.grids)


def test_batch_and_single_reconstruction_agree(model: HERMESModel, dataset) -> None:
    single = reconstruct_grid(model, dataset.grids[3])
    batched = reconstruct_batch(model, dataset.grids)
    assert np.allclose(single.probabilities, batched.probabilities[3], atol=1e-5)


def test_reconstruct_grid_rejects_a_batch(model: HERMESModel, dataset) -> None:
    with pytest.raises(ValueError):
        reconstruct_grid(model, dataset.grids[:2])


def test_reconstruction_leaves_the_model_in_eval_mode(model: HERMESModel, dataset) -> None:
    model.train()
    try:
        reconstruct_grid(model, dataset.grids[0])
        assert model.training, "training flag should be restored"
    finally:
        model.eval()


def test_evaluate_dataset_reports_per_pattern_metrics(model: HERMESModel, dataset) -> None:
    metrics = evaluate_dataset(model, dataset)
    assert metrics["num_samples"] == 24
    assert 0.0 <= metrics["cell_accuracy"] <= 1.0
    assert metrics["cell_accuracy"] == metrics["full_grid_cell_accuracy"]
    assert metrics["bce"] > 0
    assert set(metrics["accuracy_by_pattern"]) == set(dataset.pattern_types)
    assert sum(metrics["count_by_pattern"].values()) == 24
    assert 0.5 <= metrics["majority_baseline_accuracy"] <= 1.0
    json.dumps(metrics)  # must be JSON-safe for run records


def test_influence_vector_shape_and_normalisation(model: HERMESModel) -> None:
    grid = generate_sample(pattern="circle", seed=2).grid
    influence = bulk_to_boundary_influence(model, grid, (4, 4))
    assert influence.shape == (N,)
    assert influence.sum() == pytest.approx(1.0, abs=1e-5)
    assert (influence >= 0).all()

    raw = bulk_to_boundary_influence(model, grid, (4, 4), normalize=False)
    assert raw.shape == (N,)
    assert (raw >= 0).all()


def test_influence_accepts_a_single_head(model: HERMESModel) -> None:
    grid = generate_sample(pattern="circle", seed=2).grid
    per_head = bulk_to_boundary_influence(model, grid, (0, 0), head=0)
    averaged = bulk_to_boundary_influence(model, grid, (0, 0))
    assert per_head.shape == averaged.shape == (N,)


def test_influence_rejects_out_of_range_cells(model: HERMESModel) -> None:
    grid = generate_sample(pattern="circle", seed=2).grid
    with pytest.raises(ValueError):
        bulk_to_boundary_influence(model, grid, (10, 0))
    with pytest.raises(ValueError):
        bulk_to_boundary_influence(model, grid, (-1, 0))


def test_influence_matrix_rows_match_the_per_cell_vectors(model: HERMESModel) -> None:
    grid = generate_sample(pattern="cluster", seed=8).grid
    matrix = influence_matrix(model, grid)
    assert matrix.shape == (100, N)
    for row, col in [(0, 0), (3, 7), (9, 9)]:
        expected = bulk_to_boundary_influence(model, grid, (row, col))
        assert np.allclose(matrix[row * 10 + col], expected, atol=1e-5)


def test_boundary_slot_magnitudes(model: HERMESModel, dataset) -> None:
    boundary = model.encode_bulk(torch.from_numpy(dataset.grids[:4].astype(np.float32)))
    magnitudes = boundary_slot_magnitudes(boundary)
    assert magnitudes.shape == (N,)
    assert (magnitudes >= 0).all()
    assert boundary_slot_magnitudes(boundary[0]).shape == (N,)


def test_decoder_query_attention_is_a_distribution(model: HERMESModel, dataset) -> None:
    boundary = model.encode_bulk(torch.from_numpy(dataset.grids[:1].astype(np.float32)))
    attention = decoder_query_attention(model, boundary, (5, 5))
    assert attention.shape == (N,)
    assert attention.sum() == pytest.approx(1.0, abs=1e-5)


def test_decoder_query_attention_respects_a_slot_mask(model: HERMESModel, dataset) -> None:
    boundary = model.encode_bulk(torch.from_numpy(dataset.grids[:1].astype(np.float32)))
    mask = torch.ones(N, dtype=torch.bool)
    mask[:5] = False
    attention = decoder_query_attention(model, boundary, (5, 5), slot_mask=mask)
    assert np.allclose(attention[:5], 0.0)
    assert attention.sum() == pytest.approx(1.0, abs=1e-5)


def test_default_subset_sizes() -> None:
    assert default_subset_sizes(20) == [1, 2, 4, 8, 12, 16, 20]
    assert default_subset_sizes(8) == [1, 2, 4, 8]
    assert default_subset_sizes(10) == [1, 2, 4, 8, 10]
    assert default_subset_sizes(1) == [1]


@pytest.mark.parametrize("mode", ["contiguous", "random"])
def test_subset_curve_shapes_and_determinism(model: HERMESModel, dataset, mode: str) -> None:
    sizes = [1, 4, 20]
    first = boundary_subset_curve(model, dataset.grids, sizes=sizes, mode=mode, trials=2, seed=0)
    second = boundary_subset_curve(model, dataset.grids, sizes=sizes, mode=mode, trials=2, seed=0)

    assert first.sizes == sizes
    assert first.mode == mode
    assert first.num_slots == N
    for field in ("accuracy_mean", "accuracy_std", "bce_mean", "exact_match_mean"):
        assert len(getattr(first, field)) == len(sizes)
    assert first.accuracy_mean == second.accuracy_mean
    json.dumps(first.to_dict())


def test_full_subset_matches_the_unmasked_reconstruction(model: HERMESModel, dataset) -> None:
    curve = boundary_subset_curve(model, dataset.grids, sizes=[20], mode="contiguous", trials=1)
    unmasked = reconstruct_batch(model, dataset.grids)
    assert curve.accuracy_mean[0] == pytest.approx(unmasked.accuracy, abs=1e-6)
    assert curve.accuracy_std[0] == pytest.approx(0.0)


def test_subset_curve_accepts_a_dataset_object(model: HERMESModel, dataset) -> None:
    curve = boundary_subset_curve(model, dataset, sizes=[2], mode="random", trials=1)
    assert len(curve.accuracy_mean) == 1


def test_subset_curve_rejects_an_unknown_mode(model: HERMESModel, dataset) -> None:
    with pytest.raises(ValueError):
        boundary_subset_curve(model, dataset.grids, mode="spiral")


def test_subset_decoding_only_reads_the_chosen_slots(model: HERMESModel, dataset) -> None:
    """A subset decode must be unaffected by the contents of excluded slots."""
    from hermes.interventions import ablate_slots

    grid = dataset.grids[0]
    boundary = model.encode_bulk(torch.from_numpy(grid.astype(np.float32)))
    mask = torch.zeros(N, dtype=torch.bool)
    mask[[1, 2, 3]] = True

    kept = reconstruct_grid(model, grid, boundary=boundary, slot_mask=mask)
    scrambled = ablate_slots(boundary, [s for s in range(N) if s not in (1, 2, 3)], mode="noise", seed=1)
    kept_after = reconstruct_grid(model, grid, boundary=scrambled, slot_mask=mask)
    assert np.allclose(kept.probabilities, kept_after.probabilities, atol=1e-5)
