"""Training loop, reproducibility and checkpointing."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from hermes.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from hermes.data import generate_dataset
from hermes.training import (
    load_checkpoint,
    make_lr_lambda,
    resolve_device,
    sample_queries,
    save_checkpoint,
    save_run_record,
    set_seed,
    train_model,
)


def tiny_config(**overrides) -> ExperimentConfig:
    """A deliberately small setup so tests stay fast."""
    config = ExperimentConfig(
        data=DataConfig(train_size=16, val_size=8, pattern_types=("rectangle_filled",), seed=0),
        model=ModelConfig(hidden_dim=32, boundary_dim=16, num_boundary_slots=8, num_heads=2),
        train=TrainConfig(
            epochs=60, batch_size=16, queries_per_grid=100, lr=5e-3, eval_every=100, device="cpu"
        ),
        name="tiny-test",
    )
    return config.replace(**overrides) if overrides else config


def test_training_loss_decreases_on_a_tiny_overfitting_dataset() -> None:
    data = generate_dataset(16, DataConfig(pattern_types=("rectangle_filled",), seed=0))
    result = train_model(tiny_config(), train_data=data, val_data=data)

    losses = [epoch["train_loss"] for epoch in result.history]
    assert len(losses) == 60
    assert all(np.isfinite(losses))

    early = float(np.mean(losses[:5]))
    late = float(np.mean(losses[-5:]))
    assert late < early, f"loss did not decrease: {early:.4f} -> {late:.4f}"
    assert late < 0.9 * early, f"loss barely moved: {early:.4f} -> {late:.4f}"


def test_training_improves_over_the_all_zeros_baseline() -> None:
    data = generate_dataset(24, DataConfig(pattern_types=("rectangle_filled",), seed=3))
    result = train_model(tiny_config(**{"train.epochs": 120}), train_data=data, val_data=data)
    metrics = result.final_metrics
    assert metrics["cell_accuracy"] > metrics["majority_baseline_accuracy"]


def test_history_and_metrics_are_well_formed() -> None:
    result = train_model(tiny_config(**{"train.epochs": 4, "train.eval_every": 1}))
    for epoch_stats in result.history:
        assert {"epoch", "train_loss", "train_accuracy", "lr"} <= set(epoch_stats)
        assert 0.0 <= epoch_stats["train_accuracy"] <= 1.0
    assert result.history[-1]["epoch"] == 4
    assert "val_accuracy" in result.history[-1]

    metrics = result.final_metrics
    for key in (
        "cell_accuracy",
        "bce",
        "exact_grid_match_rate",
        "accuracy_by_pattern",
        "majority_baseline_accuracy",
    ):
        assert key in metrics
    assert result.duration_seconds > 0
    assert not result.model.training  # returned in eval mode


def test_runs_are_reproducible_under_a_seed() -> None:
    config = tiny_config(**{"train.epochs": 3})
    first = train_model(config)
    second = train_model(config)
    assert [h["train_loss"] for h in first.history] == pytest.approx(
        [h["train_loss"] for h in second.history]
    )
    assert first.final_metrics["cell_accuracy"] == pytest.approx(
        second.final_metrics["cell_accuracy"]
    )


def test_different_seeds_diverge() -> None:
    first = train_model(tiny_config(**{"train.epochs": 3, "train.seed": 0}))
    second = train_model(tiny_config(**{"train.epochs": 3, "train.seed": 1}))
    assert first.history[-1]["train_loss"] != second.history[-1]["train_loss"]


def test_epoch_callback_fires_once_per_epoch() -> None:
    seen: list[int] = []
    train_model(tiny_config(**{"train.epochs": 3}), callback=lambda e, s: seen.append(e))
    assert seen == [1, 2, 3]


def test_sample_queries_are_distinct_and_in_range() -> None:
    queries, flat = sample_queries(4, 100, 30, 10, torch.device("cpu"))
    assert queries.shape == (4, 30, 2)
    assert flat.shape == (4, 30)
    for row in range(4):
        assert len(set(flat[row].tolist())) == 30
    assert int(queries.min()) >= 0 and int(queries.max()) <= 9
    assert torch.equal(queries[..., 0] * 10 + queries[..., 1], flat)


def test_sample_queries_cannot_exceed_the_number_of_cells() -> None:
    queries, _ = sample_queries(2, 100, 500, 10, torch.device("cpu"))
    assert queries.shape == (2, 100, 2)


def test_lr_schedule_warms_up_then_decays() -> None:
    config = TrainConfig(warmup_frac=0.1, min_lr_frac=0.05)
    schedule = make_lr_lambda(config, total_steps=100)
    assert schedule(0) < schedule(9)
    assert schedule(9) == pytest.approx(1.0)
    assert schedule(99) < schedule(50) < schedule(10)
    assert schedule(99) >= 0.05

    flat = make_lr_lambda(TrainConfig(scheduler="none"), total_steps=100)
    assert flat(0) == flat(99) == 1.0


def test_checkpoint_roundtrip_restores_weights_and_payload(tmp_path) -> None:
    config = tiny_config(**{"train.epochs": 2})
    result = train_model(config)
    path = save_checkpoint(
        tmp_path / "ckpt.pt", result.model, config, result.history, result.final_metrics
    )
    assert path.exists()

    restored, payload = load_checkpoint(path)
    assert restored.config.num_boundary_slots == config.model.num_boundary_slots
    assert payload["experiment_config"]["name"] == "tiny-test"
    assert len(payload["history"]) == 2

    grid = torch.from_numpy(generate_dataset(1).grids.astype(np.float32))
    queries = torch.tensor([[[1, 1], [5, 5]]])
    with torch.no_grad():
        original = result.model.decode_query(result.model.encode_bulk(grid), queries)
        loaded = restored.decode_query(restored.encode_bulk(grid), queries)
    assert torch.allclose(original, loaded, atol=1e-6)


def test_run_record_is_valid_json_with_seeds_and_metrics(tmp_path) -> None:
    config = tiny_config(**{"train.epochs": 2})
    result = train_model(config)
    path = save_run_record(
        tmp_path / "run.json",
        config,
        result.final_metrics,
        extra={"ablation": {"strategy": "contiguous", "length": 4}},
    )
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["seeds"] == {"data_seed": 0, "train_seed": 0}
    assert record["config"]["model"]["num_boundary_slots"] == 8
    assert record["config"]["data"]["pattern_types"] == ["rectangle_filled"]
    assert record["metrics"]["cell_accuracy"] >= 0.0
    assert record["ablation"]["length"] == 4
    assert "torch_version" in record and "timestamp" in record


def test_config_json_roundtrip(tmp_path) -> None:
    config = tiny_config()
    path = config.save_json(tmp_path / "config.json")
    restored = ExperimentConfig.load_json(path)
    assert restored.to_dict() == config.to_dict()


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        ModelConfig(hidden_dim=10, num_heads=4)
    with pytest.raises(ValueError):
        DataConfig(pattern_types=("not_a_pattern",))
    with pytest.raises(ValueError):
        ExperimentConfig(data=DataConfig(grid_size=8), model=ModelConfig(grid_size=10))
    with pytest.raises(ValueError):
        TrainConfig(scheduler="exponential")


def test_set_seed_and_resolve_device() -> None:
    set_seed(123)
    first = torch.randn(4)
    set_seed(123)
    assert torch.equal(first, torch.randn(4))
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}
