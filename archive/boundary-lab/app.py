"""HERMES Boundary Lab -- interactive Streamlit interface.

Run with::

    streamlit run app.py

Six sections, matching the experiment workflow: generate a bulk grid, train or
load a model, inspect the encoding, reconstruct the bulk from the boundary,
ablate boundary slots, and sweep boundary-subset size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch

from hermes import DISCLAIMER, __version__
from hermes.analysis import (
    boundary_slot_magnitudes,
    boundary_subset_curve,
    bulk_to_boundary_influence,
    decoder_query_attention,
    default_subset_sizes,
    evaluate_dataset,
    reconstruct_grid,
)
from hermes.config import (
    DEFAULT_PATTERNS,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from hermes.data import generate_dataset, generate_sample
from hermes.interventions import (
    ABLATION_MODES,
    AblationSpec,
    ablate_slots,
    slot_mask_from_indices,
)
from hermes.model import HERMESModel
from hermes.training import load_checkpoint, resolve_device, save_checkpoint, train_model
from hermes.visualization import (
    plot_boundary_bar,
    plot_boundary_ring,
    plot_grid,
    plot_grid_panels,
    plot_reconstruction_panels,
    plot_subset_curves,
    plot_training_history,
)

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

st.set_page_config(page_title="HERMES Boundary Lab", page_icon="◯", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_state() -> None:
    """Seed every key the app reads, so no section depends on tab visit order."""
    state = st.session_state
    state.setdefault("seed", 0)
    state.setdefault("pattern", "random")
    state.setdefault("sample", generate_sample(pattern=None, seed=0, grid_size=10))
    state.setdefault("model", None)
    state.setdefault("experiment_config", None)
    state.setdefault("history", [])
    state.setdefault("final_metrics", {})
    state.setdefault("selected_cell", (5, 5))
    state.setdefault("ablated_slots", [])
    state.setdefault("ablation_mode", "zero")
    state.setdefault("ablation_use_mask", False)
    state.setdefault("subset_results", None)


def get_model() -> HERMESModel:
    """The current model, creating an untrained one on first use."""
    if st.session_state.model is None:
        torch.manual_seed(0)
        st.session_state.model = HERMESModel(ModelConfig()).eval()
        st.session_state.experiment_config = ExperimentConfig()
    return st.session_state.model


def model_is_trained() -> bool:
    return bool(st.session_state.history)


def untrained_warning() -> None:
    if not model_is_trained():
        st.warning(
            "This model has not been trained yet, so the numbers below are noise. "
            "Train or load a model in **2. Train or load model** first.",
            icon="⚠️",
        )


init_state()


# ---------------------------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------------------------

st.title("HERMES Boundary Lab")
st.caption(f"**{DISCLAIMER}**")
st.markdown(
    "A structured **bulk** grid is encoded into a smaller ring-shaped **boundary**. "
    "The decoder then never sees the bulk: given a query `(row, col)` it must "
    "reconstruct that cell from the boundary alone."
)

with st.sidebar:
    st.header("Status")
    model = get_model()
    config = st.session_state.experiment_config or ExperimentConfig()
    st.metric("Boundary slots", model.num_boundary_slots)
    st.metric("Boundary dim", model.boundary_dim)
    st.metric("Parameters", f"{model.num_parameters():,}")
    st.write(f"**Device:** `{resolve_device(config.train.device)}`")
    st.write(f"**Trained:** {'yes' if model_is_trained() else 'no (random weights)'}")
    if st.session_state.final_metrics:
        metrics = st.session_state.final_metrics
        st.write(f"**Val cell accuracy:** {metrics.get('cell_accuracy', 0):.4f}")
        st.write(f"**Exact grid match:** {metrics.get('exact_grid_match_rate', 0):.4f}")
        st.write(f"**All-zeros baseline:** {metrics.get('majority_baseline_accuracy', 0):.4f}")
    st.divider()
    st.caption(
        f"HERMES v{__version__}. Bulk = {model.grid_size}x{model.grid_size} binary grid "
        f"({model.grid_size ** 2} cells). Boundary = {model.num_boundary_slots} slots on a ring."
    )
    st.caption(
        "Attention maps are exploratory association measures. Ablation and "
        "subset decoding are the causal tests."
    )


tab_generate, tab_train, tab_inspect, tab_reconstruct, tab_ablate, tab_subset = st.tabs(
    [
        "1. Generate bulk",
        "2. Train or load model",
        "3. Inspect encoding",
        "4. Reconstruct bulk",
        "5. Ablate boundary",
        "6. Boundary-subset experiment",
    ]
)


# ---------------------------------------------------------------------------
# 1. Generate bulk
# ---------------------------------------------------------------------------

with tab_generate:
    st.subheader("Generate a bulk grid")
    controls, display = st.columns([1, 1.4])

    with controls:
        pattern = st.selectbox(
            "Pattern type",
            ["random"] + list(DEFAULT_PATTERNS),
            index=(["random"] + list(DEFAULT_PATTERNS)).index(st.session_state.pattern),
            help="'random' picks a family uniformly at each regeneration.",
        )
        seed = st.number_input(
            "Random seed", min_value=0, max_value=2**31 - 2, value=int(st.session_state.seed), step=1
        )
        noise = st.slider(
            "Per-cell noise probability",
            0.0,
            0.3,
            0.0,
            0.01,
            help=(
                "Flips cells after the pattern is drawn. Independent random pixels are "
                "incompressible, so a small boundary cannot encode them -- this is an "
                "ablation knob, not the main dataset."
            ),
        )

        left, right = st.columns(2)
        if left.button("Generate", width="stretch", type="primary"):
            st.session_state.seed = int(seed)
            st.session_state.pattern = pattern
            st.session_state.sample = generate_sample(
                pattern=None if pattern == "random" else pattern,
                seed=int(seed),
                grid_size=10,
                noise_prob=noise,
            )
        if right.button("New random seed", width="stretch"):
            st.session_state.seed = int(np.random.randint(0, 2**31 - 1))
            st.session_state.pattern = pattern
            st.session_state.sample = generate_sample(
                pattern=None if pattern == "random" else pattern,
                seed=st.session_state.seed,
                grid_size=10,
                noise_prob=noise,
            )

        sample = st.session_state.sample
        st.markdown("**Pattern metadata**")
        st.json(
            {
                "pattern_type": sample.pattern_type,
                "seed": sample.seed,
                "density": round(sample.density, 3),
                "cells_on": int(sample.grid.sum()),
                **{k: v for k, v in sample.metadata.items() if k != "cells"},
            },
            expanded=False,
        )

    with display:
        sample = st.session_state.sample
        st.pyplot(
            plot_grid(sample.grid, title=f"Bulk: {sample.pattern_type} (seed {sample.seed})"),
            width="content",
        )

    with st.expander("Why structured patterns, not random pixels?"):
        st.markdown(
            "A grid of independent random pixels is 100 incompressible bits. No "
            "boundary substantially smaller than the bulk could reconstruct it, so the "
            "experiment would only measure raw bottleneck capacity. Structured patterns "
            "have a short description, so the boundary *can* carry them -- which makes "
            "the interesting question **how the information is laid out around the ring**, "
            "not whether it fits."
        )


# ---------------------------------------------------------------------------
# 2. Train or load model
# ---------------------------------------------------------------------------

with tab_train:
    st.subheader("Train a model")
    st.caption("Sized for a CPU demo. Use `scripts/train.py` or Colab for larger runs.")

    col_model, col_optim, col_data = st.columns(3)
    with col_model:
        st.markdown("**Architecture**")
        num_slots = st.slider("Boundary slots", 4, 64, 20, 1)
        boundary_dim = st.select_slider("Boundary dim", [16, 32, 64, 128], value=32)
        hidden_dim = st.select_slider("Hidden dim", [32, 64, 128], value=64)
        num_heads = st.select_slider("Attention heads", [1, 2, 4, 8], value=4)
    with col_optim:
        st.markdown("**Optimisation**")
        epochs = st.slider("Epochs", 1, 120, 25, 1)
        lr = st.select_slider(
            "Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 4e-3, 6e-3, 1e-2], value=4e-3
        )
        batch_size = st.select_slider("Batch size", [16, 32, 64, 128, 256], value=64)
        queries_per_grid = st.slider("Queries per grid", 1, 100, 100, 1)
    with col_data:
        st.markdown("**Data**")
        train_size = st.select_slider("Training grids", [256, 512, 1024, 2048, 4096], value=1024)
        val_size = st.select_slider("Validation grids", [128, 256, 512], value=256)
        data_seed = st.number_input("Data seed", 0, 2**31 - 2, 0, 1)
        train_seed = st.number_input("Training seed", 0, 2**31 - 2, 0, 1)

    if boundary_dim % num_heads or hidden_dim % num_heads:
        st.error("Hidden dim and boundary dim must both be divisible by the number of heads.")
    elif st.button("Train model", type="primary"):
        experiment = ExperimentConfig(
            data=DataConfig(train_size=train_size, val_size=val_size, seed=int(data_seed)),
            model=ModelConfig(
                hidden_dim=hidden_dim,
                boundary_dim=boundary_dim,
                num_boundary_slots=num_slots,
                num_heads=num_heads,
            ),
            train=TrainConfig(
                epochs=epochs,
                lr=float(lr),
                batch_size=batch_size,
                queries_per_grid=queries_per_grid,
                seed=int(train_seed),
                device="cpu",
            ),
            name="streamlit-run",
        )
        progress = st.progress(0.0, text="Starting...")
        live = st.empty()

        def on_epoch(epoch: int, stats: dict) -> None:
            progress.progress(
                epoch / epochs,
                text=(
                    f"Epoch {epoch}/{epochs} - train loss {stats['train_loss']:.4f}, "
                    f"train acc {stats['train_accuracy']:.4f}"
                    + (
                        f" | val acc {stats['val_accuracy']:.4f}"
                        f", exact {stats['val_exact_match']:.3f}"
                        if "val_accuracy" in stats
                        else ""
                    )
                ),
            )
            live.dataframe(
                [
                    {k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()}
                    for s in reversed(st.session_state.get("_live_history", []) + [stats])
                ][:8],
                width="stretch",
                hide_index=True,
            )
            st.session_state["_live_history"] = st.session_state.get("_live_history", []) + [stats]

        st.session_state["_live_history"] = []
        with st.spinner("Training on CPU..."):
            result = train_model(experiment, callback=on_epoch)
        progress.empty()
        live.empty()

        st.session_state.model = result.model
        st.session_state.experiment_config = experiment
        st.session_state.history = result.history
        st.session_state.final_metrics = result.final_metrics
        st.session_state.ablated_slots = []
        st.session_state.subset_results = None
        st.success(f"Trained in {result.duration_seconds:.1f}s.")
        st.rerun()

    if st.session_state.history:
        st.divider()
        st.markdown("**Training history**")
        st.pyplot(plot_training_history(st.session_state.history), width="content")

        metrics = st.session_state.final_metrics
        a, b, c, d = st.columns(4)
        a.metric("Cell accuracy", f"{metrics['cell_accuracy']:.4f}")
        b.metric("Exact grid match", f"{metrics['exact_grid_match_rate']:.4f}")
        c.metric("BCE", f"{metrics['bce']:.4f}")
        d.metric(
            "All-zeros baseline",
            f"{metrics['majority_baseline_accuracy']:.4f}",
            help="These grids are sparse, so always predicting 0 already scores well. "
            "Cell accuracy is only meaningful relative to this.",
        )

        st.markdown("**Accuracy by pattern type**")
        st.dataframe(
            [
                {
                    "pattern": pattern,
                    "cell accuracy": round(value, 4),
                    "exact match": round(metrics["exact_match_by_pattern"][pattern], 4),
                    "n": metrics["count_by_pattern"][pattern],
                }
                for pattern, value in sorted(metrics["accuracy_by_pattern"].items())
            ],
            width="stretch",
            hide_index=True,
        )

    st.divider()
    st.subheader("Checkpoints")
    save_col, load_col = st.columns(2)

    with save_col:
        st.markdown("**Save**")
        name = st.text_input("Checkpoint name", value="hermes_model")
        if st.button("Save checkpoint", width="stretch"):
            path = save_checkpoint(
                CHECKPOINT_DIR / f"{name}.pt",
                get_model(),
                st.session_state.experiment_config,
                st.session_state.history,
                st.session_state.final_metrics,
            )
            st.success(f"Saved to `{path.relative_to(Path(__file__).parent)}`")

    with load_col:
        st.markdown("**Load**")
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        available = sorted(p.name for p in CHECKPOINT_DIR.glob("*.pt"))
        if not available:
            st.info("No checkpoints in `checkpoints/` yet.")
        else:
            chosen = st.selectbox("Available checkpoints", available)
            if st.button("Load checkpoint", width="stretch"):
                loaded, payload = load_checkpoint(CHECKPOINT_DIR / chosen)
                st.session_state.model = loaded
                st.session_state.history = payload.get("history", [])
                st.session_state.final_metrics = payload.get("metrics", {})
                if payload.get("experiment_config"):
                    st.session_state.experiment_config = ExperimentConfig.from_dict(
                        payload["experiment_config"]
                    )
                st.session_state.ablated_slots = []
                st.session_state.subset_results = None
                st.success(f"Loaded `{chosen}`.")
                st.rerun()


# ---------------------------------------------------------------------------
# 3. Inspect encoding
# ---------------------------------------------------------------------------

with tab_inspect:
    st.subheader("Inspect the encoding of a single cell")
    untrained_warning()

    model = get_model()
    sample = st.session_state.sample
    grid = sample.grid

    picker, readout = st.columns([1, 1.3])
    with picker:
        st.markdown("**Select a bulk cell**")
        row = st.number_input("Row", 0, model.grid_size - 1, st.session_state.selected_cell[0])
        col = st.number_input("Column", 0, model.grid_size - 1, st.session_state.selected_cell[1])
        st.session_state.selected_cell = (int(row), int(col))

        st.caption("...or click a cell directly (filled cells are marked).")
        for grid_row in range(model.grid_size):
            columns = st.columns(model.grid_size, gap="small")
            for grid_col in range(model.grid_size):
                label = "■" if grid[grid_row, grid_col] else "·"
                if columns[grid_col].button(
                    label, key=f"cell_{grid_row}_{grid_col}", width="stretch"
                ):
                    st.session_state.selected_cell = (grid_row, grid_col)
                    st.rerun()

    cell = st.session_state.selected_cell
    with torch.no_grad():
        boundary = model.encode_bulk(torch.from_numpy(grid.astype(np.float32)))
        reconstruction = reconstruct_grid(model, grid, boundary=boundary)
    probability = float(reconstruction.probabilities[cell])
    true_value = int(grid[cell])

    with readout:
        st.pyplot(
            plot_grid(grid, title=f"Bulk with query ({cell[0]}, {cell[1]})", highlight=cell),
            width="content",
        )
        a, b, c = st.columns(3)
        a.metric("Selected cell", f"({cell[0]}, {cell[1]})")
        b.metric("True value", true_value)
        c.metric(
            "Predicted p(cell=1)",
            f"{probability:.4f}",
            delta=f"{probability - true_value:+.3f} vs truth",
            delta_color="off",
        )

    st.divider()
    metric_choice = st.radio(
        "Ring metric",
        [
            "Boundary slot activation magnitude",
            "Bulk cell -> boundary attention (encoder)",
            "Query -> boundary attention (decoder)",
        ],
        horizontal=True,
    )

    if metric_choice.startswith("Boundary slot"):
        values = boundary_slot_magnitudes(boundary)
        label = "L2 norm"
        caption = (
            "How large each slot's vector is for this bulk. A crude activity readout: "
            "it says nothing about *which* cells a slot encodes."
        )
    elif metric_choice.startswith("Bulk cell"):
        values = bulk_to_boundary_influence(model, grid, cell)
        label = "attention weight"
        caption = (
            f"Encoder cross-attention from bulk cell ({cell[0]}, {cell[1]}) to each slot, "
            "normalised to sum to 1. **Association, not causation** -- a slot can attend "
            "to a cell without carrying it, and boundary self-attention lets content "
            "spread around the ring after this read. Use tab 5 to test causally."
        )
    else:
        values = decoder_query_attention(model, boundary, cell)
        label = "attention weight"
        caption = (
            f"Where the decoder looks on the ring when answering query ({cell[0]}, {cell[1]}). "
            "Also an association measure."
        )

    st.caption(caption)
    ring_col, bar_col = st.columns([1, 1.3])
    with ring_col:
        st.pyplot(
            plot_boundary_ring(values, title=metric_choice, highlight=[]),
            width="content",
        )
    with bar_col:
        st.pyplot(
            plot_boundary_bar(values, title=metric_choice, ylabel=label),
            width="content",
        )
        st.caption("The bar chart shows the same numbers as the ring, but readable precisely.")


# ---------------------------------------------------------------------------
# 4. Reconstruct bulk
# ---------------------------------------------------------------------------

with tab_reconstruct:
    st.subheader("Reconstruct the whole bulk from the boundary")
    st.caption(
        "Every one of the 100 cells is queried independently. The decoder sees only the "
        "boundary and the query coordinates."
    )
    untrained_warning()

    model = get_model()
    grid = st.session_state.sample.grid
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)
    reconstruction = reconstruct_grid(model, grid, threshold=threshold)

    st.pyplot(
        plot_reconstruction_panels(
            reconstruction.truth,
            reconstruction.probabilities,
            reconstruction.prediction,
            reconstruction.error_map,
            highlight=st.session_state.selected_cell,
        ),
        width="stretch",
    )

    a, b, c, d = st.columns(4)
    a.metric("Cell accuracy", f"{reconstruction.accuracy:.4f}")
    b.metric("Cells wrong", int(reconstruction.error_map.sum()))
    c.metric("BCE", f"{reconstruction.bce:.4f}")
    d.metric("Exact match", "yes" if reconstruction.exact_match else "no")

    with st.expander("Score this model on a fresh validation set"):
        eval_size = st.select_slider("Validation grids", [64, 128, 256, 512], value=128)
        if st.button("Evaluate"):
            dataset = generate_dataset(eval_size, DataConfig(seed=999))
            metrics = evaluate_dataset(model, dataset, threshold=threshold)
            left, right = st.columns(2)
            left.metric("Cell accuracy", f"{metrics['cell_accuracy']:.4f}")
            left.metric("Exact grid match", f"{metrics['exact_grid_match_rate']:.4f}")
            right.metric("BCE", f"{metrics['bce']:.4f}")
            right.metric("All-zeros baseline", f"{metrics['majority_baseline_accuracy']:.4f}")
            st.dataframe(
                [
                    {"pattern": p, "cell accuracy": round(v, 4)}
                    for p, v in sorted(metrics["accuracy_by_pattern"].items())
                ],
                width="stretch",
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# 5. Ablate boundary
# ---------------------------------------------------------------------------

with tab_ablate:
    st.subheader("Ablate boundary slots")
    st.caption(
        "Inference-time intervention -- no retraining. Unlike attention weights, an "
        "accuracy drop here is causal evidence that the removed slots were being used."
    )
    untrained_warning()

    model = get_model()
    grid = st.session_state.sample.grid
    num_slots = model.num_boundary_slots

    select_col, mode_col = st.columns([1.4, 1])
    with select_col:
        strategy = st.radio(
            "Selection",
            ["Manual", "Contiguous interval", "Random percentage", "Every other slot", "None"],
            horizontal=True,
        )
        if strategy == "Manual":
            chosen = st.multiselect(
                "Slots to ablate", list(range(num_slots)), default=st.session_state.ablated_slots
            )
            spec = AblationSpec(strategy="manual", slots=tuple(chosen))
        elif strategy == "Contiguous interval":
            start = st.slider("Arc start slot", 0, num_slots - 1, 0)
            length = st.slider("Arc length", 0, num_slots, max(1, num_slots // 4))
            spec = AblationSpec(strategy="contiguous", start=start, length=length)
            st.caption("The arc wraps past the last slot back to slot 0.")
        elif strategy == "Random percentage":
            fraction = st.slider("Percentage of slots", 0, 100, 25, 5) / 100.0
            random_seed = st.number_input("Selection seed", 0, 10_000, 0, 1)
            spec = AblationSpec(strategy="fraction", fraction=fraction, seed=int(random_seed))
        elif strategy == "Every other slot":
            offset = st.slider("Offset", 0, 1, 0)
            spec = AblationSpec(strategy="every_other", offset=offset)
        else:
            spec = AblationSpec(strategy="none")

    with mode_col:
        use_mask = st.checkbox(
            "Hide slots from the decoder (mask)",
            value=st.session_state.ablation_use_mask,
            help=(
                "Masking removes the slot from the decoder's attention entirely. "
                "Unchecked, the slot stays visible but its contents are overwritten."
            ),
        )
        ablation_mode = st.selectbox(
            "Content replacement", list(ABLATION_MODES), disabled=use_mask
        )
        st.session_state.ablation_use_mask = use_mask
        if st.button("Reset ablation", width="stretch"):
            st.session_state.ablated_slots = []
            st.rerun()

    slots = spec.resolve(num_slots)
    st.session_state.ablated_slots = slots

    baseline = reconstruct_grid(model, grid)
    if slots:
        if use_mask:
            mask = slot_mask_from_indices(slots, num_slots, device=baseline.boundary.device)
            ablated = reconstruct_grid(model, grid, boundary=baseline.boundary, slot_mask=mask)
        else:
            perturbed = ablate_slots(baseline.boundary, slots, mode=ablation_mode, seed=0)
            ablated = reconstruct_grid(model, grid, boundary=perturbed)
    else:
        ablated = baseline

    prob_delta = ablated.probabilities - baseline.probabilities
    st.write(
        f"**Ablating {len(slots)}/{num_slots} slots** "
        f"({'masked' if use_mask else ablation_mode}): `{slots}`"
    )

    ring_col, metric_col = st.columns([1, 1.2])
    with ring_col:
        st.pyplot(
            plot_boundary_ring(
                boundary_slot_magnitudes(baseline.boundary),
                ablated=slots,
                title="Boundary ring (x = ablated)",
            ),
            width="content",
        )
    with metric_col:
        a, b, c = st.columns(3)
        a.metric("Baseline accuracy", f"{baseline.accuracy:.4f}")
        b.metric("Ablated accuracy", f"{ablated.accuracy:.4f}")
        c.metric(
            "Accuracy drop",
            f"{baseline.accuracy - ablated.accuracy:+.4f}",
            delta=f"{ablated.accuracy - baseline.accuracy:+.4f}",
            delta_color="inverse",
        )
        d, e, f = st.columns(3)
        d.metric("Cells flipped", int((ablated.prediction != baseline.prediction).sum()))
        e.metric("Mean |Δp|", f"{np.abs(prob_delta).mean():.4f}")
        f.metric("BCE", f"{ablated.bce:.4f}", delta=f"{ablated.bce - baseline.bce:+.4f}",
                 delta_color="inverse")
        st.pyplot(
            plot_boundary_bar(
                boundary_slot_magnitudes(baseline.boundary),
                ablated=slots,
                title="Slot activation magnitude",
                ylabel="L2 norm",
            ),
            width="content",
        )

    st.pyplot(
        plot_grid_panels(
            [
                baseline.truth,
                baseline.probabilities,
                ablated.probabilities,
                prob_delta,
                ablated.error_map,
            ],
            [
                "Original bulk",
                "Baseline reconstruction",
                "Post-ablation reconstruction",
                "Difference (Δ probability)",
                "Post-ablation errors",
            ],
            cmaps=["gray_r", "magma", "magma", "coolwarm", "Reds"],
            limits=[(0, 1), (0, 1), (0, 1), (-1, 1), (0, 1)],
            colorbars=[False, True, True, True, False],
        ),
        width="stretch",
    )
    st.caption(
        "The difference map is signed: blue means the ablation pushed the cell toward 0, "
        "red toward 1."
    )


# ---------------------------------------------------------------------------
# 6. Boundary-subset experiment
# ---------------------------------------------------------------------------

with tab_subset:
    st.subheader("How much boundary do you need?")
    st.caption(
        "Decode using only a subset of slots, growing the subset. Contiguous subsets are "
        "one arc of the ring; random subsets are scattered around it. A large gap between "
        "the two curves would say the encoding is spatially localised on the ring."
    )
    untrained_warning()

    model = get_model()
    num_slots = model.num_boundary_slots

    a, b, c = st.columns(3)
    with a:
        sizes = st.multiselect(
            "Subset sizes",
            list(range(1, num_slots + 1)),
            default=default_subset_sizes(num_slots),
        )
    with b:
        trials = st.slider("Trials per size", 1, 20, 5, help="Subset placement is redrawn each trial.")
        eval_size = st.select_slider("Evaluation grids", [64, 128, 256, 512], value=128)
    with c:
        curve_seed = st.number_input("Experiment seed", 0, 10_000, 0, 1)
        run = st.button("Run experiment", type="primary", width="stretch")

    if run and sizes:
        dataset = generate_dataset(eval_size, DataConfig(seed=4242))
        with st.spinner("Sweeping subset sizes..."):
            contiguous = boundary_subset_curve(
                model, dataset.grids, sizes=sorted(sizes), mode="contiguous",
                trials=trials, seed=int(curve_seed),
            )
            random_curve = boundary_subset_curve(
                model, dataset.grids, sizes=sorted(sizes), mode="random",
                trials=trials, seed=int(curve_seed),
            )
            full = evaluate_dataset(model, dataset)
        st.session_state.subset_results = {
            "contiguous": contiguous,
            "random": random_curve,
            "full_accuracy": full["cell_accuracy"],
            "majority": full["majority_baseline_accuracy"],
        }

    results = st.session_state.subset_results
    if results:
        st.pyplot(
            plot_subset_curves(
                [results["contiguous"], results["random"]],
                baseline=results["full_accuracy"],
                majority_baseline=results["majority"],
            ),
            width="content",
        )
        st.dataframe(
            [
                {
                    "slots": size,
                    "contiguous accuracy": round(results["contiguous"].accuracy_mean[i], 4),
                    "contiguous std": round(results["contiguous"].accuracy_std[i], 4),
                    "random accuracy": round(results["random"].accuracy_mean[i], 4),
                    "random std": round(results["random"].accuracy_std[i], 4),
                    "contiguous exact match": round(results["contiguous"].exact_match_mean[i], 4),
                    "random exact match": round(results["random"].exact_match_mean[i], 4),
                }
                for i, size in enumerate(results["contiguous"].sizes)
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Read against the all-zeros baseline, not against zero: these grids are sparse, "
            "so a decoder with no useful information still scores around 0.86."
        )
    else:
        st.info("Configure the sweep and press **Run experiment**.")
