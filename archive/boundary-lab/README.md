# HERMES Boundary Lab

**Holographic Encoding for Robust, Memory-Efficient Sequence models** — a small research
prototype for a holography-*inspired* neural-memory idea.

> **Toy holography-inspired neural-memory experiment; not a physical black-hole simulation.**
>
> This project does not simulate black holes, does not address the black-hole information
> problem, and does not implement AdS/CFT. "Bulk" and "boundary" are borrowed vocabulary for
> an information-layout question: *if an interior state can only be accessed through a smaller
> ring-shaped exterior state, how does the information end up arranged on that ring?*

---

## The idea in one picture

```text
   bulk (10x10 binary grid)        boundary (ring of 20 slots)      query-conditioned decode
   +-------------------+                    o o o
   | . . # # # . . . . |            o                 o              query (row=3, col=7)
   | . # . . . # . . . |  cross-  o                     o                     |
   | # . . . . . # . . |  attn   o                       o   cross-attn       v
   | # . . . . . # . . |  ----->  o                     o    ---------->  p(cell = 1)
   | . # . . . # . . . |            o                 o
   +-------------------+                    o o o
        100 cells                    20 slots x 32 dims         the bulk is NOT available here
```

1. Every bulk cell becomes a token: `value + row embedding + column embedding`.
2. Learnable boundary slots cross-attend over all 100 bulk tokens.
3. **The bulk is then discarded.** A query `(row, col)` cross-attends to the boundary and
   predicts one binary logit.

---

## Concepts

### 1. The bulk

The bulk is a binary `10 x 10` grid — 100 cells, the "interior" state to be remembered. Grids
come from a configurable generator with nine structured pattern families: horizontal and
vertical lines, diagonals, filled and hollow rectangles, small clusters, circles, symmetric
motifs, and combinations of two shapes. Each sample carries metadata describing exactly what
was drawn, and a seed that regenerates it.

**Why not random pixels?** A grid of independent random pixels is 100 incompressible bits. No
boundary substantially smaller than the bulk could reconstruct it, so the experiment would only
measure the bottleneck's raw capacity. Structured patterns have a short description, so the
boundary *can* carry them — which makes the interesting question **how the information is laid
out around the ring**, not whether it fits. Independent noise is available as an ablation knob
(`DataConfig.noise_prob`), not as the main dataset.

### 2. The boundary

The boundary is a ring of `num_boundary_slots` (default 20) learnable slots, each with a slot
embedding and a ring positional embedding (learned, optionally plus a fixed circular sin/cos
encoding). Slots read the bulk by cross-attention, producing a
`(batch, num_boundary_slots, boundary_dim)` state.

The ring is a cycle: slot 19 is adjacent to slot 0. Version 1 keeps attention **global** —
locality is *not* imposed yet — but the topology is explicit and separate (`hermes/topology.py`)
so local-only communication can be layered on later without touching the model. Utilities cover
circular distance, contiguous arcs (with wraparound), nearest-`k` neighbourhoods, and
every-other-slot patterns.

### 3. Query-conditioned reconstruction

The decoder receives the boundary and a query `(row, col)` — and nothing else. This is an
architectural guarantee, not a convention:

```python
boundary = model.encode_bulk(bulk)                # (B, 20, 32)
logits   = model.decode_query(boundary, queries)  # (B, Q)
logits   = model.forward(bulk, queries)           # encode + decode in one call
```

`decode_query` has exactly three parameters — `boundary`, `queries`, `slot_mask` — with no path
back to the bulk. The test suite asserts this by signature inspection, by decoding a purely
synthetic boundary, and by checking that no gradient reaches a bulk tensor once the boundary is
detached. The bulk grid is used *only* to score predictions.

Reconstructing the whole bulk means querying all 100 positions and reassembling a `10 x 10` grid.

### 4. Distributed encoding

Nothing tells slot 7 to store the top-left corner. The model is free to spread each cell across
many slots, or to partition the grid into arcs. Three tools probe what it actually did:

- **Encoder cross-attention** — for a chosen bulk cell, a normalised influence vector over the
  20 slots.
- **Boundary-slot magnitudes** — how active each slot is for a given bulk.
- **Subset decoding** — reconstruct using only `1, 2, 4, 8, 12, 16, 20` slots, comparing
  contiguous arcs against randomly scattered slots. If contiguous subsets do much worse than
  random ones at the same budget, the encoding is spread out; if they do similarly well, each
  region of the ring carries broadly redundant information.

> **Attention weights are an exploratory association measure, not a causal explanation.** A
> slot can attend to a cell it never propagates, and the boundary self-attention layers let
> content spread around the ring *after* the read. Ablation is the causal test.

### 5. Boundary ablation

Inference-time interventions, no retraining:

```python
ablated = ablate_slots(boundary, [3, 4, 5], mode="zero")
```

Modes are `zero`, `mean` (replace with the mean of surviving slots), `noise`, and `shuffle`
(borrow the slot from another batch element). Slots can also be **masked** — hidden from the
decoder's attention entirely, which is the cleaner "this channel is gone" intervention. Selection
supports one slot, a contiguous interval (wrapping), random slots, every other slot, and a
percentage. Ablation never mutates the input boundary.

`compare_ablation` reports baseline accuracy, ablated accuracy, the accuracy drop, per-cell
probability change, and a reconstruction difference map.

### 6. Why this is initially just a neural bottleneck

Be honest about what the MVP is: **a query-conditioned autoencoder with an attention
bottleneck.** With the defaults, the boundary holds `20 x 32 = 640` floats for 100 bits, so it is
not information-theoretically tight at all; the constraint is *structural* (a fixed ring of slots,
read through attention) rather than a hard capacity limit. Nothing so far requires the ring
geometry to matter, and none of the observed behaviour yet needs a holographic story — an
ordinary MLP autoencoder with a query head would likely score similarly.

Reported accuracy needs the same honesty. These grids are sparse — with the default pattern mix
about 15% of cells are on — so a decoder that always predicts 0 already scores ~0.85 cell
accuracy on its own. **Always read cell accuracy against
the `majority_baseline_accuracy` that every evaluation reports**, and prefer
`exact_grid_match_rate` as the headline number.

### 7. What would distinguish HERMES from an ordinary autoencoder

The MVP is deliberately the null hypothesis. These are the follow-ups that could break the tie:

| Experiment | What it would show |
|---|---|
| **Force a real bottleneck** — quantise slots to a few bits, or add channel noise, and shrink `boundary_dim` until capacity binds | Whether the ring layout matters when information is genuinely scarce, rather than merely routed |
| **Locality constraints** — restrict boundary self-attention to a band of circular distance `k` | If reconstruction survives local-only communication, the ring geometry is doing real work; a flat autoencoder has no analogue |
| **Contiguous vs. random subsets at matched budget** — already implemented, currently the sharpest test | A large, systematic gap means information is spatially organised on the ring, not just distributed |
| **Redundancy scaling** — how many slots must be removed before accuracy collapses, as a function of ring size | Graceful degradation under arbitrary slot removal is the interesting "holographic-like" behaviour; a hard cliff is ordinary compression |
| **Region queries** — ask for a `k x k` patch rather than a single cell | Tests whether nearby bulk cells are read from nearby ring positions |
| **Bulk depth** — give the bulk an interior/exterior structure and check whether interior cells need more of the boundary | The bulk/boundary framing only earns its name if depth costs something |
| **Causal patching** — swap slot contents between two bulks and see which cells follow | Turns association measures into causal attributions |

Until several of those come back positive, the right description of this repo is "a
query-conditioned attention bottleneck with good instrumentation", and the README should keep
saying so.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Windows (PowerShell)** — same steps, but activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow it for the current session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Windows (cmd.exe)**:

```bat
.venv\Scripts\activate.bat
```

Everything in the demo runs on CPU. Python 3.11+ required.

---

## Running things

### Streamlit app

```bash
streamlit run app.py
```

Six sections:

1. **Generate bulk** — pattern type, seed, regenerate; shows the grid and its metadata.
2. **Train or load model** — epochs, learning rate, boundary-slot count, hidden dimension;
   live epoch loss and validation accuracy; save/load checkpoints.
3. **Inspect encoding** — pick a cell (number inputs or by clicking the grid); see its true
   value, predicted probability, boundary-slot magnitudes, and bulk-to-boundary attention,
   drawn both as a ring and as a bar chart.
4. **Reconstruct bulk** — original, probability grid, thresholded grid, and error map.
5. **Ablate boundary** — manual, contiguous, random-percentage, or every-other selection;
   ring view, before/after reconstruction, difference map, and the accuracy drop. Updates
   immediately, no retraining.
6. **Boundary-subset experiment** — accuracy vs. subset size, contiguous vs. random.

### Command line

```bash
python scripts/train.py --name baseline --epochs 40
```

```bash
python scripts/run_experiments.py --checkpoint checkpoints/baseline.pt --name baseline-experiments
```

`train.py` writes `runs/<name>/config.json`, `runs/<name>/run.json` and `checkpoints/<name>.pt`.
`run_experiments.py` writes `runs/<name>/experiments.json` plus figures, covering baseline
metrics, single-slot ablation, contiguous-arc ablation, matched-budget structured ablation, and
the subset-decoding sweep.

### Tests

```bash
python -m pytest
```

Covers: binary/shape validity of generated grids, per-pattern seed reproducibility, model tensor
shapes, the decoder's inability to reach the bulk, ring-distance wraparound, contiguous selection
wraparound, ablation non-mutation, `10 x 10` reconstruction from 100 queries, and loss decrease on
a tiny overfitting dataset.

### Google Colab

Heavy runs (bigger rings, longer training, sweeps) belong on a GPU. Open
[`notebooks/HERMES_Colab.ipynb`](notebooks/HERMES_Colab.ipynb) in Colab: it clones this repo,
installs dependencies, trains with `device="cuda"`, runs the analysis suite and offers to save
checkpoints to Drive. The model code is device-agnostic — `TrainConfig(device="auto")` picks CUDA
when available.

---

## Project layout

```text
hermes-boundary-lab/
├── app.py                    # Streamlit interface (6 sections)
├── README.md
├── requirements.txt
├── pyproject.toml
├── hermes/
│   ├── config.py             # DataConfig / ModelConfig / TrainConfig / ExperimentConfig (JSON)
│   ├── data.py               # 9 structured pattern generators + seeded datasets
│   ├── model.py              # BulkEncoder, BoundaryEncoder, QueryDecoder, HERMESModel
│   ├── topology.py           # ring distance, arcs, neighbourhoods, circular encodings
│   ├── interventions.py      # ablation modes, slot selection, before/after comparison
│   ├── analysis.py           # reconstruction, evaluation, influence, subset curves
│   ├── training.py           # training loop, checkpoints, run records
│   └── visualization.py      # grid panels, boundary ring, bar charts, curves
├── scripts/
│   ├── train.py
│   └── run_experiments.py
├── tests/
│   ├── test_data.py
│   ├── test_model_shapes.py
│   ├── test_topology.py
│   ├── test_interventions.py
│   ├── test_analysis.py
│   └── test_training.py
├── notebooks/
│   └── HERMES_Colab.ipynb
├── checkpoints/
└── runs/
```

---

## Research hygiene

Every run is described by an `ExperimentConfig` that round-trips through JSON, recording the data
seed, training seed, model dimensions, boundary-slot count, dataset configuration, training
hyperparameters, evaluation metrics and ablation settings. `save_run_record` stores all of it
alongside library versions and a timestamp. Config plus seeds plus the code revision reproduces a
run: `train_model` seeds Python, NumPy and torch, and the tests assert that two runs with the same
seed produce identical loss curves.

Reporting conventions used throughout:

- Every evaluation reports `majority_baseline_accuracy`; cell accuracy is meaningless without it.
- `exact_grid_match_rate` is the headline metric.
- Per-pattern accuracy is always broken out — aggregate numbers hide which families fail.
- Attention-based measures are labelled associational wherever they appear: in code, in
  docstrings, and in the UI.

---

## Extending

- **New pattern family** — add a generator to `PATTERN_GENERATORS` in `hermes/data.py` and its
  name to `DEFAULT_PATTERNS` in `hermes/config.py`.
- **Different ring size or width** — `ModelConfig(num_boundary_slots=..., boundary_dim=...)`;
  everything downstream reads the ring size from the model.
- **Locality** — build a mask from `circular_distance_matrix` and pass it to the boundary
  self-attention blocks in `hermes/model.py`.
- **New intervention** — add a mode to `ABLATION_MODES` and a branch in `ablate_slots`.

## License

MIT.
