"""Tensor-shape contracts and the bulk/decoder separation."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from hermes.config import ModelConfig
from hermes.data import generate_dataset
from hermes.model import HERMESModel, all_cell_queries, as_bulk_tensor, as_query_tensor

BATCH = 3


@pytest.fixture(scope="module")
def model() -> HERMESModel:
    torch.manual_seed(0)
    return HERMESModel(ModelConfig()).eval()


@pytest.fixture(scope="module")
def bulk() -> torch.Tensor:
    grids = generate_dataset(BATCH).grids.astype(np.float32)
    return torch.from_numpy(grids)


def test_bulk_tokens_have_expected_shape(model: HERMESModel, bulk: torch.Tensor) -> None:
    tokens = model.bulk_encoder(bulk)
    assert tokens.shape == (BATCH, model.config.num_cells, model.config.hidden_dim)


def test_encode_bulk_returns_boundary_shape(model: HERMESModel, bulk: torch.Tensor) -> None:
    boundary = model.encode_bulk(bulk)
    assert boundary.shape == (BATCH, model.config.num_boundary_slots, model.config.boundary_dim)


def test_boundary_is_smaller_than_a_dense_bulk_encoding(model: HERMESModel) -> None:
    # The boundary is the only channel to the decoder; keep the ring narrower
    # than an unconstrained per-cell encoding of the bulk.
    assert model.config.num_boundary_slots < model.config.num_cells


def test_decode_query_shapes(model: HERMESModel, bulk: torch.Tensor) -> None:
    boundary = model.encode_bulk(bulk)
    queries = torch.tensor([[[0, 0], [4, 5], [9, 9]]]).expand(BATCH, -1, -1)
    logits = model.decode_query(boundary, queries)
    assert logits.shape == (BATCH, 3)
    assert torch.isfinite(logits).all()


def test_forward_matches_encode_then_decode(model: HERMESModel, bulk: torch.Tensor) -> None:
    queries = all_cell_queries(model.grid_size).unsqueeze(0).expand(BATCH, -1, -1)
    with torch.no_grad():
        direct = model(bulk, queries)
        staged = model.decode_query(model.encode_bulk(bulk), queries)
    assert direct.shape == (BATCH, model.config.num_cells)
    assert torch.allclose(direct, staged, atol=1e-6)


def test_encoder_attention_shape(model: HERMESModel, bulk: torch.Tensor) -> None:
    boundary, maps = model.encode_bulk_with_attention(bulk)
    assert boundary.shape[1:] == (model.config.num_boundary_slots, model.config.boundary_dim)
    assert len(maps) == model.config.num_encoder_cross_layers
    for attn in maps:
        assert attn.shape == (
            BATCH,
            model.config.num_heads,
            model.config.num_boundary_slots,
            model.config.num_cells,
        )
        assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5)


def test_decoder_attention_shape(model: HERMESModel, bulk: torch.Tensor) -> None:
    boundary = model.encode_bulk(bulk)
    queries = torch.tensor([[[2, 3], [7, 1]]]).expand(BATCH, -1, -1)
    _, maps = model.decode_query_with_attention(boundary, queries)
    assert len(maps) == model.config.num_decoder_layers
    assert maps[-1].shape == (BATCH, model.config.num_heads, 2, model.config.num_boundary_slots)


def test_decoder_signature_cannot_accept_a_bulk(model: HERMESModel) -> None:
    """The bulk must be architecturally unreachable from the decoder."""
    params = set(inspect.signature(model.decode_query).parameters)
    assert params == {"boundary", "queries", "slot_mask"}
    assert not any("bulk" in name or "grid" in name for name in params)

    decoder_params = set(inspect.signature(model.decoder.forward).parameters)
    assert not any("bulk" in name or "grid" in name for name in decoder_params)


def test_decoder_runs_on_a_boundary_with_no_bulk_in_scope(model: HERMESModel) -> None:
    """A synthetic boundary decodes fine: the decoder needs nothing else."""
    boundary = torch.randn(2, model.config.num_boundary_slots, model.config.boundary_dim)
    logits = model.decode_query(boundary, torch.tensor([[[1, 1]], [[2, 2]]]))
    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


def test_no_gradient_reaches_the_bulk_when_the_boundary_is_detached(model: HERMESModel) -> None:
    grid = torch.rand(1, model.grid_size, model.grid_size, requires_grad=True)
    boundary = model.encode_bulk(grid).detach()
    logits = model.decode_query(boundary, torch.tensor([[[3, 3]]]))
    logits.sum().backward()
    assert grid.grad is None or torch.count_nonzero(grid.grad) == 0


def test_gradient_does_flow_through_the_boundary(model: HERMESModel) -> None:
    """Sanity check on the previous test: the path exists when not detached."""
    grid = torch.rand(1, model.grid_size, model.grid_size, requires_grad=True)
    logits = model.decode_query(model.encode_bulk(grid), torch.tensor([[[3, 3]]]))
    logits.sum().backward()
    assert grid.grad is not None and torch.count_nonzero(grid.grad) > 0


def test_querying_all_cells_reconstructs_a_grid_shaped_output(
    model: HERMESModel, bulk: torch.Tensor
) -> None:
    queries = all_cell_queries(model.grid_size)
    assert queries.shape == (100, 2)
    boundary = model.encode_bulk(bulk[:1])
    logits = model.decode_query(boundary, queries.unsqueeze(0))
    assert logits.reshape(model.grid_size, model.grid_size).shape == (10, 10)


def test_slot_mask_changes_the_prediction(model: HERMESModel, bulk: torch.Tensor) -> None:
    boundary = model.encode_bulk(bulk)
    queries = all_cell_queries(model.grid_size).unsqueeze(0).expand(BATCH, -1, -1)
    mask = torch.ones(model.config.num_boundary_slots, dtype=torch.bool)
    mask[:10] = False
    with torch.no_grad():
        full = model.decode_query(boundary, queries)
        partial = model.decode_query(boundary, queries, mask)
    assert not torch.allclose(full, partial)


def test_fully_masked_boundary_is_finite_not_nan(model: HERMESModel, bulk: torch.Tensor) -> None:
    """Masking every slot must degrade gracefully, not produce NaNs."""
    boundary = model.encode_bulk(bulk)
    queries = all_cell_queries(model.grid_size).unsqueeze(0).expand(BATCH, -1, -1)
    mask = torch.zeros(model.config.num_boundary_slots, dtype=torch.bool)
    _, maps = model.decode_query_with_attention(boundary, queries, mask)
    logits = model.decode_query(boundary, queries, mask)
    assert torch.isfinite(logits).all()
    assert torch.count_nonzero(maps[-1]) == 0


def test_masked_slots_receive_zero_attention(model: HERMESModel, bulk: torch.Tensor) -> None:
    boundary = model.encode_bulk(bulk)
    mask = torch.ones(model.config.num_boundary_slots, dtype=torch.bool)
    mask[3:8] = False
    _, maps = model.decode_query_with_attention(boundary, torch.tensor([[[0, 0]]] * BATCH), mask)
    assert torch.count_nonzero(maps[-1][..., 3:8]) == 0
    assert torch.allclose(maps[-1].sum(-1), torch.ones_like(maps[-1].sum(-1)), atol=1e-5)


@pytest.mark.parametrize("slots,dim,heads", [(8, 16, 2), (32, 64, 8), (20, 32, 4)])
def test_alternate_boundary_geometries(slots: int, dim: int, heads: int) -> None:
    config = ModelConfig(num_boundary_slots=slots, boundary_dim=dim, num_heads=heads)
    small = HERMESModel(config).eval()
    grids = torch.from_numpy(generate_dataset(2).grids.astype(np.float32))
    boundary = small.encode_bulk(grids)
    assert boundary.shape == (2, slots, dim)
    assert small.decode_query(boundary, torch.tensor([[[0, 0]]] * 2)).shape == (2, 1)


def test_input_coercion_helpers() -> None:
    assert as_bulk_tensor(np.zeros((10, 10))).shape == (1, 10, 10)
    assert as_bulk_tensor(torch.zeros(4, 10, 10)).shape == (4, 10, 10)
    assert as_query_tensor([3, 4]).shape == (1, 1, 2)
    assert as_query_tensor(np.zeros((7, 2)), batch_size=5).shape == (5, 7, 2)
    with pytest.raises(ValueError):
        as_bulk_tensor(np.zeros((2, 3, 4, 5)))


def test_checkpoint_roundtrip_preserves_predictions(tmp_path, bulk: torch.Tensor) -> None:
    torch.manual_seed(1)
    original = HERMESModel(ModelConfig(num_boundary_slots=12, boundary_dim=16)).eval()
    path = tmp_path / "model.pt"
    original.save(path)
    restored = HERMESModel.load(path)

    queries = all_cell_queries(10).unsqueeze(0)
    with torch.no_grad():
        assert torch.allclose(
            original.decode_query(original.encode_bulk(bulk[:1]), queries),
            restored.decode_query(restored.encode_bulk(bulk[:1]), queries),
            atol=1e-6,
        )
    assert restored.config.num_boundary_slots == 12
