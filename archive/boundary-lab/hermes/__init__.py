"""HERMES Boundary Lab.

A toy holography-inspired neural-memory experiment. A structured binary "bulk"
grid is encoded into a smaller ring-shaped "boundary" state; the explicit bulk is
then withheld, and a decoder must reconstruct any requested cell from the
boundary alone.

This is an information-processing prototype. It does **not** simulate black-hole
physics, resolve the black-hole information problem, or implement AdS/CFT -- the
holographic vocabulary is a naming convention for a bulk/boundary bottleneck,
nothing more.

Typical use::

    from hermes import ExperimentConfig, train_model, reconstruct_grid

    result = train_model(ExperimentConfig())
    recon = reconstruct_grid(result.model, grid)
"""

from __future__ import annotations

__version__ = "0.1.0"

#: Shown in the Streamlit header and in saved run records.
DISCLAIMER = (
    "Toy holography-inspired neural-memory experiment; "
    "not a physical black-hole simulation."
)

from .analysis import (  # noqa: E402
    Reconstruction,
    SubsetCurveResult,
    boundary_slot_magnitudes,
    boundary_subset_curve,
    bulk_to_boundary_influence,
    decoder_query_attention,
    evaluate_dataset,
    influence_matrix,
    reconstruct_batch,
    reconstruct_grid,
)
from .config import (  # noqa: E402
    DEFAULT_PATTERNS,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    small_demo_config,
)
from .data import (  # noqa: E402
    BulkDataset,
    BulkSample,
    generate_dataset,
    generate_sample,
    make_train_val,
)
from .interventions import (  # noqa: E402
    AblationResult,
    AblationSpec,
    ablate_slots,
    compare_ablation,
    slot_mask_from_indices,
)
from .model import HERMESModel, all_cell_queries  # noqa: E402
from .training import (  # noqa: E402
    TrainResult,
    load_checkpoint,
    save_checkpoint,
    save_run_record,
    set_seed,
    train_model,
)

__all__ = [
    "__version__",
    "DISCLAIMER",
    # config
    "ExperimentConfig",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "DEFAULT_PATTERNS",
    "small_demo_config",
    # data
    "BulkSample",
    "BulkDataset",
    "generate_sample",
    "generate_dataset",
    "make_train_val",
    # model
    "HERMESModel",
    "all_cell_queries",
    # training
    "train_model",
    "TrainResult",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
    "save_run_record",
    # analysis
    "Reconstruction",
    "SubsetCurveResult",
    "reconstruct_grid",
    "reconstruct_batch",
    "evaluate_dataset",
    "bulk_to_boundary_influence",
    "influence_matrix",
    "boundary_slot_magnitudes",
    "decoder_query_attention",
    "boundary_subset_curve",
    # interventions
    "ablate_slots",
    "compare_ablation",
    "slot_mask_from_indices",
    "AblationSpec",
    "AblationResult",
]
