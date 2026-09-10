"""The HERMES bulk -> boundary -> query architecture.

Pipeline
--------
1. **Bulk tokens.** Each of the ``grid_size ** 2`` cells becomes a token built
   from its binary value plus a learned row embedding and a learned column
   embedding: ``(B, num_cells, hidden_dim)``.
2. **Boundary.** A learnable ring of ``num_boundary_slots`` slots (slot
   embedding + ring positional embedding) reads the bulk tokens by
   cross-attention, producing ``(B, num_boundary_slots, boundary_dim)``.
3. **Query-conditioned decode.** A query ``(row, col)`` is embedded from learned
   row/column embeddings and cross-attends *only* to the boundary, emitting one
   binary logit per query.

The hard architectural constraint is that :meth:`HERMESModel.decode_query` takes
the boundary and the queries and nothing else -- it has no parameter through
which the original bulk grid or bulk tokens could reach it.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .topology import sinusoidal_ring_encoding

__all__ = [
    "HERMESModel",
    "MultiHeadAttention",
    "CrossAttentionBlock",
    "BulkEncoder",
    "BoundaryEncoder",
    "QueryDecoder",
    "as_bulk_tensor",
    "as_query_tensor",
    "all_cell_queries",
]


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def as_bulk_tensor(
    bulk: "torch.Tensor | np.ndarray | Sequence",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Coerce a grid (or batch of grids) to a ``(B, G, G)`` float tensor."""
    tensor = bulk if isinstance(bulk, torch.Tensor) else torch.as_tensor(np.asarray(bulk))
    tensor = tensor.to(dtype=dtype)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise ValueError(f"Expected bulk of shape (G, G) or (B, G, G), got {tuple(tensor.shape)}")
    return tensor.to(device) if device is not None else tensor


def as_query_tensor(
    queries: "torch.Tensor | np.ndarray | Sequence",
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Coerce queries to a ``(B, Q, 2)`` long tensor of ``(row, col)`` pairs.

    Accepts a single ``(row, col)`` pair, a ``(Q, 2)`` array (broadcast across the
    batch) or an explicit ``(B, Q, 2)`` array.
    """
    tensor = queries if isinstance(queries, torch.Tensor) else torch.as_tensor(np.asarray(queries))
    tensor = tensor.to(dtype=torch.long)
    if tensor.dim() == 1:
        tensor = tensor.view(1, 1, -1)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3 or tensor.shape[-1] != 2:
        raise ValueError(f"Expected queries of shape (..., 2), got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1 and batch_size > 1:
        tensor = tensor.expand(batch_size, -1, -1)
    return tensor.to(device) if device is not None else tensor


def all_cell_queries(grid_size: int = 10, device: torch.device | str | None = None) -> torch.Tensor:
    """Every ``(row, col)`` pair in row-major order: shape ``(grid_size ** 2, 2)``."""
    rows, cols = torch.meshgrid(
        torch.arange(grid_size), torch.arange(grid_size), indexing="ij"
    )
    queries = torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=-1).long()
    return queries.to(device) if device is not None else queries


# ---------------------------------------------------------------------------
# Attention primitives
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Multi-head attention that always returns its per-head weights.

    Written out rather than using :class:`torch.nn.MultiheadAttention` so that
    (a) attention maps are available for mechanistic analysis without relying on
    hooks, and (b) key masking has explicit, well-defined behaviour when *every*
    key is masked (the output is exactly zero rather than NaN or a uniform
    average over masked keys).
    """

    def __init__(self, query_dim: int, kv_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(kv_dim, query_dim)
        self.v_proj = nn.Linear(kv_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend from ``query`` to ``key_value``.

        Args:
            query: ``(B, Lq, query_dim)``.
            key_value: ``(B, Lk, kv_dim)``.
            key_mask: Optional ``(B, Lk)`` or ``(Lk,)`` boolean mask where
                ``True`` means the key is *available*. Fully-masked rows produce
                a zero attention row and a zero attention output.

        Returns:
            ``(output, attn)`` with shapes ``(B, Lq, query_dim)`` and
            ``(B, num_heads, Lq, Lk)``. Attention rows sum to 1, except
            fully-masked rows which sum to 0.
        """
        batch, len_q, _ = query.shape
        len_k = key_value.shape[1]

        def split(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = split(self.q_proj(query))
        k = split(self.k_proj(key_value))
        v = split(self.v_proj(key_value))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, Lq, Lk)

        if key_mask is not None:
            mask = key_mask.to(torch.bool)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(batch, -1)
            mask = mask.view(batch, 1, 1, len_k)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            # Renormalise so fully-masked rows are exactly zero instead of uniform.
            attn = attn * mask.to(attn.dtype)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            attn = torch.where(mask.any(dim=-1, keepdim=True), attn, torch.zeros_like(attn))
        else:
            attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(self.attn_dropout(attn), v)  # (B, H, Lq, head_dim)
        out = out.transpose(1, 2).reshape(batch, len_q, self.num_heads * self.head_dim)
        return self.out_dropout(self.out_proj(out)), attn


class FeedForward(nn.Module):
    """Position-wise MLP with GELU."""

    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """Pre-norm block: ``x = x + attn(x, kv)`` then ``x = x + ffn(x)``."""

    def __init__(
        self, query_dim: int, kv_dim: int, num_heads: int, ffn_mult: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.attn = MultiHeadAttention(query_dim, kv_dim, num_heads, dropout)
        self.norm_ff = nn.LayerNorm(query_dim)
        self.ffn = FeedForward(query_dim, ffn_mult, dropout)

    def forward(
        self, x: torch.Tensor, key_value: torch.Tensor, key_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, attn = self.attn(self.norm_q(x), self.norm_kv(key_value), key_mask)
        x = x + attended
        x = x + self.ffn(self.norm_ff(x))
        return x, attn


class SelfAttentionBlock(nn.Module):
    """Cross-attention block wired to attend to itself."""

    def __init__(self, dim: int, num_heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.block = CrossAttentionBlock(dim, dim, num_heads, ffn_mult, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.block(x, x)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


class BulkEncoder(nn.Module):
    """Turns a binary grid into ``(B, num_cells, hidden_dim)`` bulk tokens.

    A token is ``value_proj(cell) + row_embedding[r] + col_embedding[c]``. The
    value path is a ``Linear(1, hidden_dim)``, which for binary inputs is exactly
    as expressive as a two-entry embedding table while also accepting soft or
    perturbed grids (useful for later gradient-based analyses).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.value_proj = nn.Linear(1, dim)
        self.row_embedding = nn.Embedding(config.grid_size, dim)
        self.col_embedding = nn.Embedding(config.grid_size, dim)
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            SelfAttentionBlock(dim, config.num_heads, config.ffn_mult, config.dropout)
            for _ in range(config.num_bulk_layers)
        )

        rows, cols = torch.meshgrid(
            torch.arange(config.grid_size), torch.arange(config.grid_size), indexing="ij"
        )
        self.register_buffer("cell_rows", rows.reshape(-1), persistent=False)
        self.register_buffer("cell_cols", cols.reshape(-1), persistent=False)

    def forward(self, bulk: torch.Tensor) -> torch.Tensor:
        """Args: ``bulk`` of shape ``(B, G, G)``. Returns ``(B, G*G, hidden_dim)``."""
        batch = bulk.shape[0]
        values = bulk.reshape(batch, -1, 1)  # (B, N, 1)
        tokens = (
            self.value_proj(values)
            + self.row_embedding(self.cell_rows).unsqueeze(0)
            + self.col_embedding(self.cell_cols).unsqueeze(0)
        )
        tokens = self.norm(tokens)
        for layer in self.layers:
            tokens, _ = layer(tokens)
        return tokens


class BoundaryEncoder(nn.Module):
    """Learnable ring of slots that reads the bulk tokens by cross-attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        slots, dim = config.num_boundary_slots, config.boundary_dim

        self.slot_embedding = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.ring_position = nn.Parameter(torch.randn(slots, dim) * 0.02)
        if config.use_sinusoidal_ring:
            ring = torch.from_numpy(sinusoidal_ring_encoding(slots, dim - dim % 2))
            if ring.shape[1] < dim:  # odd boundary_dim: pad the final column
                ring = F.pad(ring, (0, dim - ring.shape[1]))
            self.register_buffer("ring_sinusoid", ring, persistent=False)
        else:
            self.register_buffer("ring_sinusoid", torch.zeros(slots, dim), persistent=False)

        self.cross_layers = nn.ModuleList(
            CrossAttentionBlock(
                dim, config.hidden_dim, config.num_heads, config.ffn_mult, config.dropout
            )
            for _ in range(config.num_encoder_cross_layers)
        )
        self.self_layers = nn.ModuleList(
            SelfAttentionBlock(dim, config.num_heads, config.ffn_mult, config.dropout)
            for _ in range(config.num_boundary_self_layers)
        )
        self.norm = nn.LayerNorm(dim)

    def initial_slots(self, batch: int) -> torch.Tensor:
        base = self.slot_embedding + self.ring_position + self.ring_sinusoid
        return base.unsqueeze(0).expand(batch, -1, -1)

    def forward(self, bulk_tokens: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Args: ``bulk_tokens`` ``(B, N, hidden_dim)``.

        Returns:
            ``(boundary, cross_attention_maps)`` where ``boundary`` is
            ``(B, num_boundary_slots, boundary_dim)`` and each attention map is
            ``(B, num_heads, num_boundary_slots, N)``.
        """
        slots = self.initial_slots(bulk_tokens.shape[0])
        attention_maps: list[torch.Tensor] = []
        for layer in self.cross_layers:
            slots, attn = layer(slots, bulk_tokens)
            attention_maps.append(attn)
        for layer in self.self_layers:
            slots, _ = layer(slots)
        return self.norm(slots), attention_maps


class QueryDecoder(nn.Module):
    """Predicts one cell's value by attending from a ``(row, col)`` query to the boundary.

    Receives the boundary only. There is no code path from here back to the bulk.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.boundary_dim
        self.row_embedding = nn.Embedding(config.grid_size, dim)
        self.col_embedding = nn.Embedding(config.grid_size, dim)
        self.query_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.norm_in = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            CrossAttentionBlock(dim, dim, config.num_heads, config.ffn_mult, config.dropout)
            for _ in range(config.num_decoder_layers)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def embed_queries(self, queries: torch.Tensor) -> torch.Tensor:
        """``(B, Q, 2)`` -> ``(B, Q, boundary_dim)``."""
        rows, cols = queries[..., 0], queries[..., 1]
        embedded = self.row_embedding(rows) + self.col_embedding(cols) + self.query_token
        return self.norm_in(embedded)

    def forward(
        self,
        boundary: torch.Tensor,
        queries: torch.Tensor,
        slot_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Args:
            boundary: ``(B, S, boundary_dim)``.
            queries: ``(B, Q, 2)`` of ``(row, col)`` pairs.
            slot_mask: Optional ``(B, S)`` or ``(S,)`` boolean mask; ``True``
                marks a slot as readable.

        Returns:
            ``(logits, attention_maps)`` with logits of shape ``(B, Q)`` and each
            attention map of shape ``(B, num_heads, Q, S)``.
        """
        x = self.embed_queries(queries)
        attention_maps: list[torch.Tensor] = []
        for layer in self.layers:
            x, attn = layer(x, boundary, slot_mask)
            attention_maps.append(attn)
        return self.head(x).squeeze(-1), attention_maps


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class HERMESModel(nn.Module):
    """Query-conditioned bulk-to-boundary model.

    Example:
        >>> model = HERMESModel(ModelConfig())
        >>> boundary = model.encode_bulk(bulk)            # (B, 20, 32)
        >>> logits = model.decode_query(boundary, queries)  # (B, Q)
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.bulk_encoder = BulkEncoder(self.config)
        self.boundary_encoder = BoundaryEncoder(self.config)
        self.decoder = QueryDecoder(self.config)

    # -- convenience properties ---------------------------------------

    @property
    def grid_size(self) -> int:
        return self.config.grid_size

    @property
    def num_boundary_slots(self) -> int:
        return self.config.num_boundary_slots

    @property
    def boundary_dim(self) -> int:
        return self.config.boundary_dim

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        return sum(p.numel() for p in params if p.requires_grad or not trainable_only)

    # -- core interface ------------------------------------------------

    def encode_bulk(self, bulk: torch.Tensor) -> torch.Tensor:
        """Encode a bulk grid into the boundary state.

        Args:
            bulk: ``(B, G, G)`` or ``(G, G)`` binary grid.

        Returns:
            ``(B, num_boundary_slots, boundary_dim)``.
        """
        boundary, _ = self.encode_bulk_with_attention(bulk)
        return boundary

    def encode_bulk_with_attention(
        self, bulk: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Like :meth:`encode_bulk` but also returns the encoder cross-attention maps.

        Each map has shape ``(B, num_heads, num_boundary_slots, num_cells)``.
        """
        bulk = as_bulk_tensor(bulk, device=self.device)
        tokens = self.bulk_encoder(bulk)
        return self.boundary_encoder(tokens)

    def decode_query(
        self,
        boundary: torch.Tensor,
        queries: torch.Tensor,
        slot_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict queried cells from the boundary alone.

        This method never sees the bulk: its only inputs are the boundary state,
        the ``(row, col)`` queries and an optional slot availability mask.

        Args:
            boundary: ``(B, S, boundary_dim)`` from :meth:`encode_bulk`.
            queries: ``(B, Q, 2)``, ``(Q, 2)`` or a single ``(row, col)`` pair.
            slot_mask: Optional ``(B, S)`` / ``(S,)`` boolean mask of readable slots.

        Returns:
            ``(B, Q)`` binary logits.
        """
        logits, _ = self.decode_query_with_attention(boundary, queries, slot_mask)
        return logits

    def decode_query_with_attention(
        self,
        boundary: torch.Tensor,
        queries: torch.Tensor,
        slot_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Like :meth:`decode_query` but also returns query->boundary attention."""
        if boundary.dim() == 2:
            boundary = boundary.unsqueeze(0)
        queries = as_query_tensor(queries, batch_size=boundary.shape[0], device=boundary.device)
        if queries.shape[0] != boundary.shape[0]:
            raise ValueError(
                f"Batch mismatch: boundary {boundary.shape[0]} vs queries {queries.shape[0]}"
            )
        if slot_mask is not None and not isinstance(slot_mask, torch.Tensor):
            slot_mask = torch.as_tensor(np.asarray(slot_mask), device=boundary.device)
        return self.decoder(boundary, queries, slot_mask)

    def forward(
        self,
        bulk: torch.Tensor,
        queries: torch.Tensor,
        slot_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode then decode in one call. Returns ``(B, Q)`` logits."""
        bulk = as_bulk_tensor(bulk, device=self.device)
        boundary = self.encode_bulk(bulk)
        queries = as_query_tensor(queries, batch_size=bulk.shape[0], device=bulk.device)
        return self.decode_query(boundary, queries, slot_mask)

    # -- checkpointing -------------------------------------------------

    def save(self, path) -> None:
        """Save weights together with the :class:`ModelConfig` that built them."""
        from dataclasses import asdict
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location: str | torch.device = "cpu") -> "HERMESModel":
        """Rebuild a model from a checkpoint written by :meth:`save`."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ModelConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
