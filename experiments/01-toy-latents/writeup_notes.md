# Toy demo: reconstruction vs. task-only latents

**Scope.** This is a deliberately minimal, self-contained proof-of-concept for
*one* mechanism the full Hermes project will test at scale — whether a
world-model latent trained purely through a task-relevant prediction head (no
reconstruction loss) can match a reconstruction-trained latent on the task while
handling distractors differently. It is **not** a result about Hermes itself.
Everything below comes from a single real training + analysis run
(`train.py` → `analyze.py`); no number here is simulated or hand-picked.

---

## Methods

**Data (`data.py`).** Each sample is a 16×16 RGB image containing one white
Gaussian blob on a randomly coloured, noisy background.

- *Task-relevant factor* `t`: the blob's horizontal position, `~U(0,1)`, mapped
  to a pixel column (vertical position fixed at centre). This is the only thing
  a downstream controller would need.
- *Distractors, drawn independently of `t`*: the background RGB colour and
  per-pixel Gaussian noise (σ=0.05).

By construction the background dominates raw pixel variance (the moving dot
accounts for only **15.4%** of total pixel variance), so a reconstruction loss
is *pressured* to encode the background while a task loss has no reason to. The
data is rigged to make the encoders reveal that asymmetry — not to guarantee an
outcome.

**Encoders (`model.py`, hand-written numpy, no autograd framework).** Identical
architecture and latent size (8) for both:

- Shared encoder: `768 → Dense(64) → ReLU → Dense(8)`.
- **Encoder A (reconstruction):** + decoder `8 → 64 → 768 → sigmoid`,
  MSE against the full image.
- **Encoder B (task-only):** + head `8 → 16 → 1`, MSE against `t`.
  **No reconstruction term at all.**

Both trained 60 epochs, batch 128, Adam (lr 2e-3), from the *same* weight
initialisation. Latents were snapshotted on a fixed 1000-sample eval set every 5
epochs. Total runtime ≈ 45 s on CPU.

**Probing.** "Variance explained" = held-out R² of a linear read-out from the
latent to a factor (task `t`, or the RGB background). Probes are fit on one half
of the eval snapshot and scored on the other.

---

## Results

### 0. Headline

Read the object's position out of each latent with the *same* linear probe and
the *same* train/test split (Encoder A has no task head of its own, so a probe is
the only fair comparison):

| Encoder | Position readout R² | Mean abs. error | …in pixels (16 px image) |
|---|---|---|---|
| A — reconstruction | 0.915 | 0.065 | 0.97 px |
| **B — task-only** | **0.986** | **0.026** | **0.39 px** |

The task-only encoder localises the object **2.5× more precisely**, despite
never receiving a reconstruction signal.

### 1. Task fidelity and distractor content in the latent

| Encoder | Task R² (position) | Distractor R² (RGB) |
|---|---|---|
| Random init (shared) | 0.473 | 0.837 |
| **A — reconstruction** | 0.915 | **0.998** |
| **B — task-only** | **0.986** | 0.963 |

Two honest observations:

1. **B matches/beats A on the task** (0.986 vs 0.915) while never seeing a
   reconstruction signal — the first half of the hypothesis holds cleanly.
2. **Task-only training does NOT scrub distractors from the latent.** Even a
   *random* encoder already linearly exposes 84% of the background (a random
   projection of background-dominated pixels). Training B *raised* that to 0.963
   rather than lowering it. So B's latent still carries the background; it simply
   isn't *required* to. This is the opposite of the naïve "task-only ⇒
   distractor-free" expectation, and it is the main caveat of the demo.

### 2. Latent ablation — causal importance (item 5)

Zeroing each latent dim and re-running each encoder's **own** downstream module:

| | Metric | Importance by dim (sorted) |
|---|---|---|
| **B — task head** | drop in task R² | dim6 **28.9**, dim4 4.9, then ≤1.5 … |
| **A — decoder** | rise in recon MSE | 0.090, 0.076, 0.066, 0.059, … 0.008 (spread) |

On a **common** linear task-probe (comparable units, share of total task-MSE
increase when a dim is ablated):

| Encoder | Baseline task R² | Top-dim share of task importance |
|---|---|---|
| A — reconstruction | 0.915 | 42% (spread over ≥4 dims) |
| **B — task-only** | 0.986 | **82% (one dim, #6)** |

**This is the sharpest, cleanest contrast in the demo.** B's task prediction
rests almost entirely on a *single* latent dimension; A's reconstruction relies
on *all* dimensions roughly evenly (it must, to reproduce background everywhere).
So even though both latents *contain* the background, the task-relevant
information in B is *causally concentrated and isolated*, whereas in A it is
distributed and entangled with the background-encoding machinery.

### 3. PCA of the final latent space (item 6)

| Encoder | Var(PC1) | Var(PC2) | Task R² from 2 PCs | Distractor R² from 2 PCs | PC1↔brightness corr |
|---|---|---|---|---|---|
| A — reconstruction | 0.48 | 0.19 | 0.21 | 0.75 | −0.85 |
| B — task-only | 0.80 | 0.12 | 0.46 | 0.75 | −0.86 |

**This did not match the hoped-for pattern, and we report it as-is.** The
*top* principal component of **both** encoders is aligned with the background
brightness distractor (corr ≈ −0.85), because the background is the
highest-variance thing in either latent. PCA — a pure variance method — surfaces
it first in both. Task structure is present (in B it appears on PC2, corr −0.67,
and is fully linearly decodable at R²=0.986) but it is *not* the dominant
variance direction. Under the task colouring, B's projection is cleaner than A's
(0.46 vs 0.21 task-R²-from-2-PCs), but neither latent is "distractor-free" in a
variance sense.

The lesson: **correlational/variance tools (PCA, linear probes) and causal tools
(ablation) tell different stories here.** By variance, background dominates both
latents. By causal importance, the task circuit in B is cleanly separated. For
mechanistic interpretability of task-only world models, the causal view is the
more informative one.

**Consequence for visualisation.** Because PCA axes are variance axes, a PCA
scatter of these latents is dominated by background and shows no legible task
structure for *either* encoder. The interactive demo therefore plots a
*task-aligned* projection instead: axis 1 is the latent direction a probe uses to
read out position, axis 2 is the background direction orthogonalised against it.
On those axes the difference is directly visible — the correlation between axis 1
and true position is **0.994 for B** vs **0.959 for A**. The raw PCA view is kept
in an appendix for completeness.

### 4. Distractor-shift robustness (supplementary)

Freeze both encoders, fit a task probe in-distribution, then ramp background
noise out-of-distribution and measure task R²:

| bg noise σ | A — recon | B — task-only |
|---|---|---|
| 0.05 (train) | 0.919 | **0.987** |
| 0.10 | 0.910 | **0.970** |
| 0.20 | 0.876 | **0.907** |
| 0.30 | 0.825 | 0.824 |
| 0.40 | 0.761 | 0.734 |

B's task read-out is more accurate across the in- and near-distribution regime
(σ ≤ 0.2), the advantage vanishes at σ≈0.3, and A is marginally better under
*severe* corruption. So the task-only representation is better *where it counts*
but is **not** categorically more robust to arbitrarily strong nuisance — an
honest, non-cheerleading result.

---

## Takeaways (honest, and for the LaTeX section later)

- **Supported:** a purely task-trained latent can *match or exceed* a
  reconstruction latent on the task (0.986 vs 0.915), and its task information is
  **causally concentrated in a single latent dimension** (82% of ablation
  importance) versus distributed across all dimensions for the autoencoder.
- **Not supported (reported honestly):** task-only training did **not** remove
  the distractor from the latent — background remained linearly decodable
  (R²=0.96) and dominated the top PCA component of *both* encoders. "Task-relevant
  objective" bought *causal isolation of task info*, not *distractor invariance*.
- **Implication for Hermes:** to turn "task-relevant" into "distractor-robust"
  will likely need an explicit pressure the task head alone does not provide —
  e.g. a capacity/information bottleneck, or an invariance objective. This toy is
  a clean testbed for adding exactly that and re-running the same four analyses.

*Reproduce:* `python train.py && python analyze.py && python export_demo.py`;
explore live with `python interactive_viz.py`. All figures derive from the CSV/
npz files in `results/`.


---

## Extension: does the task-only latent survive multi-step imagination?

*(Separate experiment. The static results above are unchanged and were not
re-run. Code: `data_seq.py`, `model_seq.py`, `train_rollout.py`,
`analyze_rollout.py`. This arm uses PyTorch; the static demo remains
hand-written numpy.)*

### Motivation

Everything above is a single-frame result, and a single-frame win is a weak
claim for a world model. The concern worth testing is temporal: a latent trained
only to answer "where is the object *now*" might carry just enough information
for one step and then collapse once the model has to imagine forward without new
observations.

### Setup

Trajectories of 20 frames: the dot moves at roughly constant velocity and
**reflects off the left and right walls**; the background colour is drawn once
per trajectory and held fixed; pixel noise is redrawn each frame. Walls matter —
without them, 10-step prediction is pure linear extrapolation and nothing can
distinguish two models that both recover velocity.

A GRU (hidden 32) sits on each encoder. Both models are trained end to end to
predict position 1–10 steps ahead, and differ **only** in the loss:

| | encoder | decoder | loss |
|---|---|---|---|
| **A** | 768→64→8 | yes | task + reconstruction |
| **B** | 768→64→8 | **none** | task only |

Both receive the task gradient, so this is a fair comparison rather than a
handicap match. Initial weights of the shared encoder/GRU/head are asserted
byte-identical (53,801 params each). Training uses teacher forcing (real frames
at every step). At evaluation the model consumes 5 real frames and then steps
the GRU forward on **zero input** for 10 steps — training and evaluation use the
identical procedure, so the curve measures representation quality rather than
train/test mismatch. Results are averaged over **5 seeds**; each seed re-runs
data generation, initialisation and training.

### Results

Blind-rollout position error, px on a 16-px image, mean over 5 seeds. "Seeds
agreeing" counts how many seeds put B behind A:

| Horizon | A (recon+task) | B (task-only) | Gap (B−A) | Seeds agreeing | Hold-last | Linear extrap. |
|---|---|---|---|---|---|---|
| 1 | 0.361 | 0.473 | +0.112 | 4/5 | 0.582 | 0.127 |
| 3 | 0.423 | 0.535 | +0.113 | 5/5 | 1.644 | 0.515 |
| 5 | 0.559 | 0.643 | +0.084 | 3/5 | 2.533 | 1.196 |
| 10 | 0.932 | 1.074 | +0.142 | 3/5 | 4.138 | 4.128 |

Degradation rate — err(h=10)/err(h=1), the metric that literally answers
"degrades faster":

| Model | Growth factor | h=1 → h=10 |
|---|---|---|
| A · recon + task | 2.71 ± 0.64 | 0.361 → 0.932 px |
| B · task-only | 2.30 ± 0.31 | 0.473 → 1.074 px |
| baseline: hold last | 7.11 | 0.582 → 4.138 px |
| baseline: linear | 32.63 | 0.127 → 4.128 px |

Split by whether a wall bounce falls inside the scored window:

| Subset | A @ h1 | B @ h1 | A @ h10 | B @ h10 |
|---|---|---|---|---|
| no bounce | 0.364 | 0.517 | 0.859 | 0.939 |
| bounce | 0.358 | 0.440 | 0.990 | 1.181 |

### What this actually shows

**The collapse hypothesis is not supported.** B is slightly worse than A at
every horizon, but by a roughly *constant* offset (~0.1 px) rather than a
widening one: the gap is +0.112 px at one step and
+0.142 px at ten. On the literal rate question B is
the *more* stable of the two (2.30× growth
versus 2.71×), i.e. the opposite direction
from the worry that motivated the experiment.

**The A/B difference is small and seed-unstable.** All 5 seeds agreed on the
sign of the gap at only 2 of 10 horizons; at horizon 10 it was 3/5, with two
seeds putting B ahead. Per-seed h=10 gaps ranged from −0.080 to +0.362 px. The
honest summary is that A is *probably* slightly better in absolute terms and
that the difference is comparable to run-to-run noise.

**A pilot finding that did not replicate.** A single-seed pilot suggested A's
entire advantage came from wall bounces — a tidy mechanistic story about
reconstruction encoding scene geometry. It did not survive the 5-seed sweep:
the bounce and no-bounce gaps cross over with horizon and neither dominates. It
is recorded here because it is exactly the kind of pattern that one seed will
manufacture and a sweep will destroy.

**Both models did learn the dynamics.** Both beat hold-last by 3–4× at long
horizons, and both beat linear extrapolation from horizon ~6 onward — which is
only possible by having represented the wall, since linear extrapolation walks
straight through it.

### Limitations

- One toy, one architecture, one dynamics; effects are fractions of a pixel.
- We cannot separate "reconstruction teaches the model about the scene" from "a
  second loss regularises the encoder". A control arm with a non-pixel auxiliary
  objective (e.g. predicting the background colour) would distinguish these, and
  is the obvious next run.
- A 10-step horizon over near-linear dynamics is a gentle test. Longer horizons,
  faster motion, or dynamics requiring more scene structure could still separate
  the two where this does not.
- `LAMBDA_RECON = 1.0` was not tuned; A's balance between its two losses is
  arbitrary and unswept.

*Reproduce:* `python train_rollout.py && python analyze_rollout.py`
(~40 min on CPU for 5 seeds; no GPU required).
