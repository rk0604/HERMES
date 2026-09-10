"""Configuration dataclasses for HERMES Boundary Lab.

Every experiment is fully described by an :class:`ExperimentConfig`, which
serialises to / from JSON. Combined with the recorded seeds this is what makes
runs reproducible: a config file plus the code revision is enough to regenerate
a dataset, a model and a training run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "ExperimentConfig",
    "DEFAULT_PATTERNS",
]

#: Canonical list of pattern families produced by :mod:`hermes.data`.
DEFAULT_PATTERNS: tuple[str, ...] = (
    "horizontal_line",
    "vertical_line",
    "diagonal_line",
    "rectangle_filled",
    "rectangle_hollow",
    "cluster",
    "circle",
    "symmetric",
    "combined",
)


def _filter_known(cls: type, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Drop keys that are not fields of ``cls`` so old configs still load."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in payload.items() if k in known}


@dataclass
class DataConfig:
    """Controls the synthetic bulk-grid generator.

    Attributes:
        grid_size: Side length of the square bulk grid. The bulk therefore has
            ``grid_size ** 2`` cells.
        pattern_types: Pattern families to sample from. Must be a subset of
            :data:`DEFAULT_PATTERNS`.
        seed: Master seed for dataset generation.
        train_size: Number of training grids.
        val_size: Number of validation grids.
        noise_prob: Probability of flipping each cell after the pattern is
            drawn. Kept at ``0.0`` by default -- independent random pixels are a
            deliberately bad fit for a compressive boundary (100 arbitrary bits
            cannot be packed losslessly into a much smaller state), so noise is
            an ablation knob rather than the main dataset.
        allow_empty: If ``False``, regenerate a sample whose grid came out
            entirely blank.
    """

    grid_size: int = 10
    pattern_types: tuple[str, ...] = DEFAULT_PATTERNS
    seed: int = 0
    train_size: int = 4096
    val_size: int = 512
    noise_prob: float = 0.0
    allow_empty: bool = False

    def __post_init__(self) -> None:
        self.pattern_types = tuple(self.pattern_types)
        unknown = set(self.pattern_types) - set(DEFAULT_PATTERNS)
        if unknown:
            raise ValueError(f"Unknown pattern types: {sorted(unknown)}")
        if not self.pattern_types:
            raise ValueError("pattern_types must not be empty")
        if self.grid_size < 3:
            raise ValueError("grid_size must be >= 3")
        if not 0.0 <= self.noise_prob <= 1.0:
            raise ValueError("noise_prob must be in [0, 1]")

    @property
    def num_cells(self) -> int:
        return self.grid_size * self.grid_size


@dataclass
class ModelConfig:
    """Shapes and depths of the bulk -> boundary -> query architecture.

    Attributes:
        grid_size: Must match :attr:`DataConfig.grid_size`.
        hidden_dim: Width of a bulk token.
        boundary_dim: Width of a boundary slot. This, times
            :attr:`num_boundary_slots`, is the entire channel the decoder sees.
        num_boundary_slots: Number of slots on the boundary ring.
        num_heads: Attention heads used by every attention block.
        num_bulk_layers: Self-attention layers applied to bulk tokens *before*
            the boundary reads them.
        num_encoder_cross_layers: Cross-attention layers where boundary slots
            read bulk tokens. Keeping this at 1 makes the recorded attention map
            a direct ``bulk cell -> boundary slot`` read, which is much easier to
            interpret.
        num_boundary_self_layers: Self-attention layers among boundary slots
            after the cross-attention. Note these let information spread around
            the ring, so a slot's content is not only what it directly read.
        num_decoder_layers: Cross-attention layers where the query reads the
            boundary.
        ffn_mult: Feed-forward expansion factor.
        dropout: Dropout probability used inside attention and feed-forwards.
        use_sinusoidal_ring: Add a fixed circular (sin/cos) encoding to the
            learned ring positional embedding, making the ring geometry explicit
            rather than purely learned.
    """

    grid_size: int = 10
    hidden_dim: int = 64
    boundary_dim: int = 32
    num_boundary_slots: int = 20
    num_heads: int = 4
    num_bulk_layers: int = 1
    num_encoder_cross_layers: int = 1
    num_boundary_self_layers: int = 1
    num_decoder_layers: int = 1
    ffn_mult: int = 2
    dropout: float = 0.0
    use_sinusoidal_ring: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads or self.boundary_dim % self.num_heads:
            raise ValueError("hidden_dim and boundary_dim must be divisible by num_heads")
        if self.num_boundary_slots < 1:
            raise ValueError("num_boundary_slots must be >= 1")
        if self.num_encoder_cross_layers < 1 or self.num_decoder_layers < 1:
            raise ValueError("encoder cross layers and decoder layers must be >= 1")

    @property
    def num_cells(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def boundary_floats(self) -> int:
        """Total float count of the boundary state, per grid."""
        return self.num_boundary_slots * self.boundary_dim


@dataclass
class TrainConfig:
    """Optimisation settings.

    Attributes:
        epochs: Passes over the training set.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Grids per optimiser step.
        queries_per_grid: How many cells are queried per grid per step. Cells are
            sampled without replacement; set to ``grid_size ** 2`` to query the
            whole grid every step.
        seed: Seed for parameter init, query sampling and batch shuffling.
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or an explicit torch device.
        grad_clip: Global grad-norm clip; ``0`` disables.
        eval_every: Run validation every N epochs (always runs on the last).
        num_workers: DataLoader workers. ``0`` is right for CPU demos.
        scheduler: ``"cosine"`` (linear warmup then cosine decay) or ``"none"``.
            Training spends its first phase with near-uniform encoder attention,
            during which the model sits at the all-zeros baseline; the warmup
            plus a healthy peak learning rate shortens that stall noticeably.
        warmup_frac: Fraction of total steps spent warming up.
        min_lr_frac: Floor of the cosine decay, as a fraction of ``lr``.
    """

    epochs: int = 40
    lr: float = 4e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    queries_per_grid: int = 100
    seed: int = 0
    device: str = "auto"
    grad_clip: float = 1.0
    eval_every: int = 1
    num_workers: int = 0
    scheduler: str = "cosine"
    warmup_frac: float = 0.05
    min_lr_frac: float = 0.05

    def __post_init__(self) -> None:
        if self.queries_per_grid < 1:
            raise ValueError("queries_per_grid must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.scheduler not in ("cosine", "none"):
            raise ValueError("scheduler must be 'cosine' or 'none'")
        if not 0.0 <= self.warmup_frac < 1.0:
            raise ValueError("warmup_frac must be in [0, 1)")


@dataclass
class ExperimentConfig:
    """Top-level container: everything needed to reproduce a run."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    name: str = "hermes-mvp"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.data.grid_size != self.model.grid_size:
            raise ValueError(
                "data.grid_size and model.grid_size must match "
                f"({self.data.grid_size} != {self.model.grid_size})"
            )

    # -- serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data"]["pattern_types"] = list(self.data.pattern_types)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentConfig":
        return cls(
            data=DataConfig(**_filter_known(DataConfig, payload.get("data", {}))),
            model=ModelConfig(**_filter_known(ModelConfig, payload.get("model", {}))),
            train=TrainConfig(**_filter_known(TrainConfig, payload.get("train", {}))),
            name=payload.get("name", "hermes-mvp"),
            notes=payload.get("notes", ""),
        )

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def replace(self, **overrides: Any) -> "ExperimentConfig":
        """Return a copy with dotted overrides applied, e.g. ``model.hidden_dim=128``."""
        payload = self.to_dict()
        for key, value in overrides.items():
            if "." in key:
                section, attr = key.split(".", 1)
                payload[section][attr] = value
            else:
                payload[key] = value
        return ExperimentConfig.from_dict(payload)


def small_demo_config(
    pattern_types: Sequence[str] | None = None, **overrides: Any
) -> ExperimentConfig:
    """A config sized for a CPU demo (seconds to a couple of minutes)."""
    cfg = ExperimentConfig(
        data=DataConfig(
            train_size=1024,
            val_size=256,
            pattern_types=tuple(pattern_types) if pattern_types else DEFAULT_PATTERNS,
        ),
        model=ModelConfig(),
        train=TrainConfig(epochs=25, batch_size=64, queries_per_grid=100),
        name="hermes-cpu-demo",
    )
    return cfg.replace(**overrides) if overrides else cfg
