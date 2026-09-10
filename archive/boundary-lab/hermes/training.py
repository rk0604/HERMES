"""Training loop, evaluation and checkpointing.

During training each grid is queried at ``queries_per_grid`` randomly chosen
cells (sampled without replacement), and the loss is binary cross-entropy with
logits on those cells. Validation is stricter: every cell of every validation
grid is queried, which is the full-grid reconstruction setting.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .analysis import evaluate_dataset
from .config import ExperimentConfig
from .data import BulkDataset, make_train_val
from .model import HERMESModel

__all__ = [
    "GridDataset",
    "TrainResult",
    "train_model",
    "set_seed",
    "resolve_device",
    "make_lr_lambda",
    "save_checkpoint",
    "load_checkpoint",
    "save_run_record",
    "json_default",
]

EpochCallback = Callable[[int, dict[str, float]], None]


# ---------------------------------------------------------------------------
# Reproducibility / device
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_lr_lambda(config, total_steps: int) -> Callable[[int], float]:
    """Linear warmup followed by cosine decay, as a multiplier on the base LR."""
    if config.scheduler == "none":
        return lambda step: 1.0
    warmup_steps = max(1, int(config.warmup_frac * total_steps))
    floor = float(config.min_lr_frac)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
        return floor + (1.0 - floor) * cosine

    return lr_lambda


def resolve_device(device: str = "auto") -> torch.device:
    """Turn ``"auto"`` into the best available device; pass anything else through."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset plumbing
# ---------------------------------------------------------------------------


class GridDataset(Dataset):
    """Wraps a :class:`~hermes.data.BulkDataset` as float grids plus pattern ids."""

    def __init__(self, dataset: BulkDataset) -> None:
        self.grids = torch.from_numpy(dataset.grids.astype(np.float32))
        self.pattern_ids = torch.from_numpy(dataset.pattern_ids())
        self.source = dataset

    def __len__(self) -> int:
        return int(self.grids.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.grids[index], self.pattern_ids[index]


def sample_queries(
    batch_size: int,
    num_cells: int,
    queries_per_grid: int,
    grid_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample distinct cell indices per grid.

    Returns:
        ``(queries, flat_indices)`` where ``queries`` is ``(B, Q, 2)`` of
        ``(row, col)`` pairs and ``flat_indices`` is ``(B, Q)`` row-major indices
        for gathering targets out of the flattened grid.
    """
    count = min(queries_per_grid, num_cells)
    noise = torch.rand(batch_size, num_cells, device=device, generator=generator)
    flat = noise.argsort(dim=1)[:, :count]  # distinct indices per row
    rows = flat // grid_size
    cols = flat % grid_size
    return torch.stack([rows, cols], dim=-1), flat


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Everything a run produced: the model, per-epoch history and final metrics."""

    model: HERMESModel
    history: list[dict[str, float]] = field(default_factory=list)
    final_metrics: dict[str, Any] = field(default_factory=dict)
    config: ExperimentConfig | None = None
    duration_seconds: float = 0.0

    def to_record(self) -> dict[str, Any]:
        """JSON-safe record of the run, including the config that produced it."""
        return {
            "config": self.config.to_dict() if self.config else None,
            "num_parameters": self.model.num_parameters(),
            "duration_seconds": self.duration_seconds,
            "history": self.history,
            "final_metrics": self.final_metrics,
        }


def train_model(
    config: ExperimentConfig | None = None,
    train_data: BulkDataset | None = None,
    val_data: BulkDataset | None = None,
    callback: EpochCallback | None = None,
    verbose: bool = False,
) -> TrainResult:
    """Train a HERMES model.

    Args:
        config: Full experiment config; defaults to :class:`ExperimentConfig`.
        train_data: Optional pre-generated training grids. Generated from
            ``config.data`` when omitted.
        val_data: Optional pre-generated validation grids.
        callback: Called as ``callback(epoch, stats)`` after each epoch, where
            ``stats`` holds train loss/accuracy and (when validation ran)
            full-grid validation metrics. Used by the Streamlit progress bar.
        verbose: Print a line per epoch.

    Returns:
        A :class:`TrainResult`. The model is left in eval mode.
    """
    config = config or ExperimentConfig()
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)

    if train_data is None or val_data is None:
        generated_train, generated_val = make_train_val(config.data)
        train_data = train_data or generated_train
        val_data = val_data or generated_val

    model = HERMESModel(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )

    loader_generator = torch.Generator().manual_seed(config.train.seed)
    loader = DataLoader(
        GridDataset(train_data),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        generator=loader_generator,
        drop_last=False,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(config.train, max(1, len(loader) * config.train.epochs))
    )

    num_cells = config.model.num_cells
    grid_size = config.model.grid_size
    query_generator = torch.Generator(device=device).manual_seed(config.train.seed + 1)

    history: list[dict[str, float]] = []
    start_time = time.perf_counter()

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0

        for grids, _ in loader:
            grids = grids.to(device)
            queries, flat = sample_queries(
                grids.shape[0],
                num_cells,
                config.train.queries_per_grid,
                grid_size,
                device,
                query_generator,
            )
            targets = grids.reshape(grids.shape[0], -1).gather(1, flat)

            logits = model(grids, queries)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
            scheduler.step()

            loss_sum += loss.item() * targets.numel()
            correct += int(((logits > 0).float() == targets).sum().item())
            total += targets.numel()

        stats: dict[str, float] = {
            "epoch": epoch,
            "train_loss": loss_sum / max(total, 1),
            "train_accuracy": correct / max(total, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        is_last = epoch == config.train.epochs
        if is_last or epoch % max(config.train.eval_every, 1) == 0:
            metrics = evaluate_dataset(model, val_data)
            stats.update(
                {
                    "val_loss": metrics["bce"],
                    "val_accuracy": metrics["cell_accuracy"],
                    "val_exact_match": metrics["exact_grid_match_rate"],
                }
            )

        history.append(stats)
        if callback is not None:
            callback(epoch, stats)
        if verbose:
            parts = [f"{k}={v:.4f}" for k, v in stats.items() if k != "epoch"]
            print(f"epoch {epoch:>3}/{config.train.epochs}  " + "  ".join(parts))

    model.eval()
    final_metrics = evaluate_dataset(model, val_data)
    final_metrics["train_metrics"] = evaluate_dataset(model, train_data)

    return TrainResult(
        model=model,
        history=history,
        final_metrics=final_metrics,
        config=config,
        duration_seconds=time.perf_counter() - start_time,
    )


# ---------------------------------------------------------------------------
# Checkpoints and run records
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    model: HERMESModel,
    config: ExperimentConfig | None = None,
    history: list[dict[str, float]] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Write weights plus the config, history and metrics needed to interpret them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "experiment_config": config.to_dict() if config else None,
        "history": history or [],
        "metrics": metrics or {},
        "torch_version": torch.__version__,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[HERMESModel, dict[str, Any]]:
    """Rebuild a model from a checkpoint and return it with the stored payload."""
    from .config import ModelConfig

    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = HERMESModel(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def save_run_record(
    path: str | Path,
    config: ExperimentConfig,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a JSON record of a run: config, seeds, metrics and anything extra.

    This is the research-hygiene artefact -- config plus seeds plus code revision
    is enough to reproduce the numbers stored alongside them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "name": config.name,
        "notes": config.notes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config.to_dict(),
        "seeds": {
            "data_seed": config.data.seed,
            "train_seed": config.train.seed,
        },
        "metrics": metrics,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    if extra:
        record.update(extra)
    path.write_text(json.dumps(record, indent=2, default=json_default), encoding="utf-8")
    return path


def json_default(obj: Any) -> Any:
    """``json.dump(default=...)`` hook for NumPy scalars, arrays and torch tensors."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
