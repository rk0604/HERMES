"""Train a HERMES model from the command line.

Examples::

    python scripts/train.py --epochs 40 --name baseline
    python scripts/train.py --config runs/baseline/config.json
    python scripts/train.py --boundary-slots 8 --boundary-dim 16 --name narrow-ring

Every run writes three files into ``runs/<name>/``: the resolved config, a JSON
record with seeds and metrics, and the model checkpoint. Config plus seeds plus
the code revision is enough to reproduce the numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig  # noqa: E402
from hermes.training import save_checkpoint, save_run_record, train_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a HERMES bulk -> boundary -> query model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="Load an ExperimentConfig JSON; flags override it.")
    parser.add_argument("--name", default="baseline", help="Run name; output goes to runs/<name>/.")
    parser.add_argument("--notes", default="", help="Free-text note stored in the run record.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints")

    data = parser.add_argument_group("data")
    data.add_argument("--grid-size", type=int, default=10)
    data.add_argument("--train-size", type=int, default=4096)
    data.add_argument("--val-size", type=int, default=512)
    data.add_argument("--data-seed", type=int, default=0)
    data.add_argument("--noise-prob", type=float, default=0.0)
    data.add_argument("--patterns", nargs="+", default=None, help="Restrict the pattern families.")

    model = parser.add_argument_group("model")
    model.add_argument("--boundary-slots", type=int, default=20)
    model.add_argument("--boundary-dim", type=int, default=32)
    model.add_argument("--hidden-dim", type=int, default=64)
    model.add_argument("--num-heads", type=int, default=4)
    model.add_argument("--bulk-layers", type=int, default=1)
    model.add_argument("--encoder-cross-layers", type=int, default=1)
    model.add_argument("--boundary-self-layers", type=int, default=1)
    model.add_argument("--decoder-layers", type=int, default=1)
    model.add_argument("--dropout", type=float, default=0.0)

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--lr", type=float, default=4e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--queries-per-grid", type=int, default=100)
    train.add_argument("--train-seed", type=int, default=0)
    train.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    train.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    train.add_argument("--eval-every", type=int, default=1)
    train.add_argument("--quiet", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build the config, letting explicit flags override a loaded config file."""
    if args.config:
        base = ExperimentConfig.load_json(args.config)
        given = {arg.lstrip("-").replace("-", "_") for arg in sys.argv[1:] if arg.startswith("--")}
        overrides: dict[str, object] = {}
        mapping = {
            "epochs": "train.epochs",
            "lr": "train.lr",
            "batch_size": "train.batch_size",
            "queries_per_grid": "train.queries_per_grid",
            "train_seed": "train.seed",
            "device": "train.device",
            "scheduler": "train.scheduler",
            "boundary_slots": "model.num_boundary_slots",
            "boundary_dim": "model.boundary_dim",
            "hidden_dim": "model.hidden_dim",
            "num_heads": "model.num_heads",
            "train_size": "data.train_size",
            "val_size": "data.val_size",
            "data_seed": "data.seed",
            "noise_prob": "data.noise_prob",
            "name": "name",
            "notes": "notes",
        }
        for flag, path in mapping.items():
            if flag in given:
                overrides[path] = getattr(args, flag)
        return base.replace(**overrides) if overrides else base

    return ExperimentConfig(
        data=DataConfig(
            grid_size=args.grid_size,
            train_size=args.train_size,
            val_size=args.val_size,
            seed=args.data_seed,
            noise_prob=args.noise_prob,
            pattern_types=tuple(args.patterns) if args.patterns else DataConfig().pattern_types,
        ),
        model=ModelConfig(
            grid_size=args.grid_size,
            hidden_dim=args.hidden_dim,
            boundary_dim=args.boundary_dim,
            num_boundary_slots=args.boundary_slots,
            num_heads=args.num_heads,
            num_bulk_layers=args.bulk_layers,
            num_encoder_cross_layers=args.encoder_cross_layers,
            num_boundary_self_layers=args.boundary_self_layers,
            num_decoder_layers=args.decoder_layers,
            dropout=args.dropout,
        ),
        train=TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            queries_per_grid=args.queries_per_grid,
            seed=args.train_seed,
            device=args.device,
            scheduler=args.scheduler,
            eval_every=args.eval_every,
        ),
        name=args.name,
        notes=args.notes,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    run_dir = Path(args.output_dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save_json(run_dir / "config.json")

    print(f"Run '{config.name}'")
    print(f"  boundary : {config.model.num_boundary_slots} slots x {config.model.boundary_dim} dim "
          f"({config.model.boundary_floats} floats) for {config.model.num_cells} bulk cells")
    print(f"  data     : {config.data.train_size} train / {config.data.val_size} val grids, "
          f"seed {config.data.seed}")
    print(f"  training : {config.train.epochs} epochs, lr {config.train.lr}, "
          f"seed {config.train.seed}, device {config.train.device}")
    print()

    result = train_model(config, verbose=not args.quiet)

    checkpoint = save_checkpoint(
        Path(args.checkpoint_dir) / f"{config.name}.pt",
        result.model,
        config,
        result.history,
        result.final_metrics,
    )
    record = save_run_record(
        run_dir / "run.json",
        config,
        result.final_metrics,
        extra={
            "num_parameters": result.model.num_parameters(),
            "duration_seconds": result.duration_seconds,
            "history": result.history,
            "checkpoint": str(checkpoint),
        },
    )

    metrics = result.final_metrics
    print()
    print(f"Finished in {result.duration_seconds:.1f}s ({result.model.num_parameters():,} params)")
    print(f"  cell accuracy        : {metrics['cell_accuracy']:.4f}")
    print(f"  exact grid match     : {metrics['exact_grid_match_rate']:.4f}")
    print(f"  BCE                  : {metrics['bce']:.4f}")
    print(f"  all-zeros baseline   : {metrics['majority_baseline_accuracy']:.4f}  <- compare against this")
    print("  accuracy by pattern  :")
    for pattern, value in sorted(metrics["accuracy_by_pattern"].items()):
        print(f"      {pattern:<18} {value:.4f}")
    print()
    print(f"Checkpoint : {checkpoint}")
    print(f"Run record : {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
