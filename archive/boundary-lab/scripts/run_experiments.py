"""Run the standard HERMES analysis suite and save JSON + figures.

Experiments
-----------
1. **Baseline** -- full-boundary reconstruction, overall and per pattern type.
2. **Single-slot ablation** -- ablate each slot alone; how much does each matter?
3. **Contiguous-arc ablation** -- remove arcs of growing length.
4. **Structured ablation** -- random fractions vs. every-other-slot, at matched budgets.
5. **Boundary-subset decoding** -- accuracy vs. subset size, contiguous vs. random.

Examples::

    python scripts/run_experiments.py --checkpoint checkpoints/baseline.pt
    python scripts/run_experiments.py --train --epochs 30 --name quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes import DISCLAIMER  # noqa: E402
from hermes.analysis import (  # noqa: E402
    boundary_subset_curve,
    default_subset_sizes,
    evaluate_dataset,
    reconstruct_batch,
)
from hermes.config import DataConfig, ExperimentConfig  # noqa: E402
from hermes.data import generate_dataset  # noqa: E402
from hermes.interventions import AblationSpec, slot_mask_from_indices  # noqa: E402
from hermes.training import (  # noqa: E402
    json_default,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_model,
)
from hermes.visualization import (  # noqa: E402
    plot_boundary_bar,
    plot_boundary_ring,
    plot_subset_curves,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ablation and boundary-subset experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint to analyse.")
    parser.add_argument("--train", action="store_true", help="Train a fresh model first.")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs when --train is used.")
    parser.add_argument("--name", default="experiments", help="Output goes to runs/<name>/.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--eval-size", type=int, default=512, help="Grids in the evaluation set.")
    parser.add_argument("--eval-seed", type=int, default=4242)
    parser.add_argument("--trials", type=int, default=8, help="Random restarts per subset size.")
    parser.add_argument("--seed", type=int, default=0, help="Master seed for the experiments.")
    parser.add_argument("--no-figures", action="store_true", help="Skip writing PNGs.")
    return parser


def masked_accuracy(model, grids, ablated: list[int]) -> dict[str, float]:
    """Accuracy when the listed slots are hidden from the decoder."""
    mask = slot_mask_from_indices(ablated, model.num_boundary_slots)
    recon = reconstruct_batch(model, grids, slot_mask=mask)
    return {"accuracy": recon.accuracy, "bce": recon.bce, "exact_match": recon.exact_match_rate}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)

    run_dir = Path(args.output_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # -- model ------------------------------------------------------------
    if args.train or not args.checkpoint:
        config = ExperimentConfig(name=args.name).replace(**{"train.epochs": args.epochs})
        print(f"Training a model for {args.epochs} epochs...")
        result = train_model(config, verbose=True)
        model = result.model
        save_checkpoint(
            REPO_ROOT / "checkpoints" / f"{args.name}.pt",
            model,
            config,
            result.history,
            result.final_metrics,
        )
    else:
        print(f"Loading {args.checkpoint}")
        model, payload = load_checkpoint(args.checkpoint)
        stored = payload.get("experiment_config")
        config = ExperimentConfig.from_dict(stored) if stored else ExperimentConfig()

    num_slots = model.num_boundary_slots
    dataset = generate_dataset(args.eval_size, DataConfig(seed=args.eval_seed))
    print(f"\nEvaluating on {len(dataset)} grids, ring size {num_slots}\n")

    results: dict[str, object] = {
        "disclaimer": DISCLAIMER,
        "name": args.name,
        "seeds": {
            "master": args.seed,
            "eval_data": args.eval_seed,
            "data": config.data.seed,
            "train": config.train.seed,
        },
        "config": config.to_dict(),
        "model": {
            "num_parameters": model.num_parameters(),
            "num_boundary_slots": num_slots,
            "boundary_dim": model.boundary_dim,
            "boundary_floats": num_slots * model.boundary_dim,
            "num_bulk_cells": model.grid_size**2,
        },
        "eval_dataset": dataset.summary(),
    }

    # -- 1. baseline ------------------------------------------------------
    baseline = evaluate_dataset(model, dataset)
    results["baseline"] = baseline
    print(f"[1] baseline cell accuracy {baseline['cell_accuracy']:.4f} "
          f"(all-zeros baseline {baseline['majority_baseline_accuracy']:.4f}), "
          f"exact match {baseline['exact_grid_match_rate']:.4f}")

    # -- 2. single-slot ablation ------------------------------------------
    single = []
    for slot in range(num_slots):
        scores = masked_accuracy(model, dataset.grids, [slot])
        single.append({
            "slot": slot,
            **scores,
            "accuracy_drop": baseline["cell_accuracy"] - scores["accuracy"],
        })
    drops = np.array([entry["accuracy_drop"] for entry in single])
    results["single_slot_ablation"] = {
        "per_slot": single,
        "mean_drop": float(drops.mean()),
        "max_drop": float(drops.max()),
        "min_drop": float(drops.min()),
        "most_important_slot": int(drops.argmax()),
        "least_important_slot": int(drops.argmin()),
    }
    print(f"[2] single-slot ablation: mean drop {drops.mean():.4f}, "
          f"worst slot {int(drops.argmax())} (drop {drops.max():.4f})")

    # -- 3. contiguous arcs ------------------------------------------------
    arcs = []
    rng = np.random.default_rng(args.seed)
    for length in sorted({1, 2, 4, num_slots // 4, num_slots // 2, 3 * num_slots // 4}):
        if not 0 < length <= num_slots:
            continue
        per_start = []
        for _ in range(args.trials):
            spec = AblationSpec(
                strategy="contiguous", start=int(rng.integers(0, num_slots)), length=int(length)
            )
            per_start.append(masked_accuracy(model, dataset.grids, spec.resolve(num_slots)))
        accuracies = [entry["accuracy"] for entry in per_start]
        arcs.append({
            "length": int(length),
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "accuracy_drop": float(baseline["cell_accuracy"] - np.mean(accuracies)),
            "trials": args.trials,
        })
    results["contiguous_arc_ablation"] = arcs
    print("[3] contiguous arcs: " + ", ".join(
        f"len {a['length']} -> {a['accuracy_mean']:.4f}" for a in arcs))

    # -- 4. structured vs random at matched budgets ------------------------
    budget = num_slots // 2
    every_other = AblationSpec(strategy="every_other").resolve(num_slots)
    random_matched = [
        masked_accuracy(
            model,
            dataset.grids,
            AblationSpec(strategy="random", count=budget, seed=int(rng.integers(0, 10_000)))
            .resolve(num_slots),
        )["accuracy"]
        for _ in range(args.trials)
    ]
    contiguous_matched = [
        masked_accuracy(
            model,
            dataset.grids,
            AblationSpec(strategy="contiguous", start=int(rng.integers(0, num_slots)), length=budget)
            .resolve(num_slots),
        )["accuracy"]
        for _ in range(args.trials)
    ]
    results["matched_budget_ablation"] = {
        "num_ablated": budget,
        "every_other": masked_accuracy(model, dataset.grids, every_other),
        "random_mean": float(np.mean(random_matched)),
        "random_std": float(np.std(random_matched)),
        "contiguous_mean": float(np.mean(contiguous_matched)),
        "contiguous_std": float(np.std(contiguous_matched)),
    }
    print(f"[4] removing {budget} slots: every-other "
          f"{results['matched_budget_ablation']['every_other']['accuracy']:.4f}, "
          f"random {np.mean(random_matched):.4f}, contiguous {np.mean(contiguous_matched):.4f}")

    # -- 5. subset decoding ------------------------------------------------
    sizes = default_subset_sizes(num_slots)
    contiguous_curve = boundary_subset_curve(
        model, dataset.grids, sizes=sizes, mode="contiguous", trials=args.trials, seed=args.seed
    )
    random_curve = boundary_subset_curve(
        model, dataset.grids, sizes=sizes, mode="random", trials=args.trials, seed=args.seed
    )
    results["subset_decoding"] = {
        "contiguous": contiguous_curve.to_dict(),
        "random": random_curve.to_dict(),
    }
    print("[5] subset decoding (contiguous / random):")
    for i, size in enumerate(sizes):
        print(f"      {size:>3} slots  {contiguous_curve.accuracy_mean[i]:.4f} / "
              f"{random_curve.accuracy_mean[i]:.4f}")

    # -- outputs -----------------------------------------------------------
    results_path = run_dir / "experiments.json"
    results_path.write_text(
        json.dumps(results, indent=2, default=json_default), encoding="utf-8"
    )
    print(f"\nResults -> {results_path}")

    if not args.no_figures:
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        plot_subset_curves(
            [contiguous_curve, random_curve],
            baseline=baseline["cell_accuracy"],
            majority_baseline=baseline["majority_baseline_accuracy"],
        ).savefig(figures_dir / "subset_curve.png", dpi=150, bbox_inches="tight")
        plot_boundary_ring(
            drops, title="Accuracy drop from ablating each slot alone"
        ).savefig(figures_dir / "single_slot_ring.png", dpi=150, bbox_inches="tight")
        plot_boundary_bar(
            drops, title="Accuracy drop per ablated slot", ylabel="accuracy drop"
        ).savefig(figures_dir / "single_slot_bar.png", dpi=150, bbox_inches="tight")
        print(f"Figures -> {figures_dir}")

    print(f"\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
