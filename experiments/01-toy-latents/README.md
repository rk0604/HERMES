# Experiment 1: task only latents at small scale

Two phases sharing one codebase. First a static, single frame comparison of a
reconstruction latent against a task only latent. Then a temporal extension where the
object moves and the model must imagine forward with the pixels cut off.

Everything here is hand written NumPy with explicit forward and backward passes, apart
from the rollout extension which uses PyTorch because backpropagation through time by
hand would be a large pile of error prone code for no scientific gain. Runs on CPU.

## The setup

Each sample is a 16 by 16 RGB image containing one white Gaussian blob on a randomly
coloured, noisy background.

* **Task relevant factor.** The blob horizontal position, drawn uniformly, mapped to a
  pixel column. This is the only thing a downstream controller would need.
* **Distractors, drawn independently of the task.** The background RGB colour and per
  pixel Gaussian noise at sigma 0.05.

The data is rigged so the asymmetry can show itself, not so a particular outcome is
guaranteed. By construction the background dominates raw pixel variance: the moving blob
accounts for only **15.4 percent** of total pixel variance. A reconstruction loss is
therefore under pressure to spend latent capacity on background, while a task loss has no
reason to.

Both encoders share an identical architecture and latent size of 8, and start from the
same weights:

```
shared encoder:  768 -> Dense(64) -> ReLU -> Dense(8)
Encoder A:       + decoder 8 -> 64 -> 768 -> sigmoid,  MSE against the full image
Encoder B:       + head    8 -> 16 -> 1,               MSE against position
                   no reconstruction term anywhere
```

Sixty epochs, batch 128, Adam at 2e-3. Latents are snapshotted on a fixed evaluation set
every 5 epochs.

## Phase 1 results: the static comparison

Position is read out of both latents with the same linear probe and the same train and
test split. Encoder A has no task head of its own, so a probe is the only fair
instrument.

| encoder | position readout R squared | mean absolute error | in pixels |
| :--- | :--- | :--- | :--- |
| A, reconstruction | 0.915 | 0.065 | 0.97 px |
| **B, task only** | **0.986** | **0.026** | **0.39 px** |

The task only encoder localises the object roughly **2.5 times more precisely** despite
never receiving a reconstruction signal.

### What is in each latent

| encoder | task R squared | distractor R squared (background RGB) |
| :--- | :--- | :--- |
| random initialisation, shared | 0.473 | 0.837 |
| A, reconstruction | 0.915 | **0.998** |
| B, task only | **0.986** | 0.963 |

Two honest observations. B beats A on the task cleanly. But **task only training did not
scrub the distractors out of the latent.** A random encoder already exposes 84 percent of
the background linearly, because a random projection of background dominated pixels
retains it. Training B *raised* that number rather than lowering it. B's latent still
carries the background, it simply is not required to use it.

This was published as the main caveat of the experiment. Experiment 2 reverses it at
scale, which is worth reading as a pair.

### Causal ablation

Zeroing each latent dimension in turn and re running the downstream module, measured on a
common linear task probe so the units are comparable:

| encoder | baseline task R squared | share of task importance in the top dimension |
| :--- | :--- | :--- |
| A, reconstruction | 0.915 | 42 percent, spread over four or more dimensions |
| **B, task only** | **0.986** | **82 percent, concentrated in one dimension** |

This is the sharpest contrast in the phase. B's prediction rests almost entirely on a
single latent dimension. A's reconstruction relies on all dimensions roughly evenly,
because it must reproduce background everywhere. So although both latents *contain* the
background, the task information in B is causally isolated while in A it is distributed
and entangled.

### A note on visualisation

PCA finds directions of greatest variance, and in this dataset that is always the
background. A raw PCA scatter is therefore dominated by the distractor and shows no
legible task structure for either encoder. The interactive demo plots a **task aligned**
projection instead, where axis one is the direction a probe uses to read out position.
On those axes the correlation between axis one and the true position is 0.994 for B
against 0.959 for A. The raw PCA view is kept in an appendix for completeness.

## Phase 2 results: blind rollout

The dataset is extended into trajectories of 20 frames. The blob moves at roughly
constant velocity and reflects off the walls. Reflecting walls are deliberate: without
them, ten step prediction is pure linear extrapolation and any model that recovers
velocity solves it perfectly, leaving nothing for a horizon curve to reveal.

A GRU sits on each encoder. Both models are trained end to end to predict position 1 to
10 steps ahead, differing only in the loss:

| arm | loss |
| :--- | :--- |
| A | task plus reconstruction |
| B | task only, no decoder |

Both receive the task gradient, so this is a fair comparison rather than a handicap
match. Training uses teacher forcing with real frames at every step. At evaluation the
model consumes 5 real frames and then steps the GRU forward on **zero input** for 10
steps. Training and evaluation use the identical procedure, so the curve measures
representation quality rather than a train and test mismatch. Averaged over **5 seeds**.

| horizon | A | B | gap | seeds agreeing | hold last | linear |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.361 | 0.473 | +0.112 | 4 of 5 | 0.582 | 0.127 |
| 3 | 0.423 | 0.535 | +0.113 | 5 of 5 | 1.644 | 0.515 |
| 5 | 0.559 | 0.643 | +0.084 | 3 of 5 | 2.533 | 1.196 |
| 10 | 0.932 | 1.074 | +0.142 | 3 of 5 | 4.138 | 4.128 |

Degradation rate, which is the metric that literally answers "does it degrade faster":

| model | growth factor, 1 step to 10 steps |
| :--- | :--- |
| A, reconstruction plus task | 2.71 plus or minus 0.64 |
| **B, task only** | **2.30 plus or minus 0.31** |
| baseline, hold last | 7.11 |
| baseline, linear | 32.63 |

### What this shows

**The collapse hypothesis is not supported.** B is slightly worse than A at every
horizon, but by a roughly *constant* offset of about 0.1 px rather than a widening one.
On the rate question B is the *more* stable of the two, which is the opposite direction
from the worry that motivated the experiment.

**The difference is small and seed unstable.** All 5 seeds agreed on the sign at only 2
of 10 horizons. At the longest horizon it was 3 of 5, with two seeds putting B ahead. Per
seed gaps at 10 steps ranged from minus 0.080 to plus 0.362 px. The honest summary is
that A is probably slightly better in absolute terms and that the difference is
comparable to run to run noise.

**A pilot finding that did not replicate.** A single seed pilot produced a tidy
mechanistic story: A's entire advantage came from wall bounces, implying reconstruction
encodes scene geometry. It was convincing enough to be drafted into the writeup. It did
not survive the five seed sweep, where the bounce and no bounce gaps cross over and
neither dominates. It is recorded here because it is exactly the kind of pattern that one
seed will manufacture and a sweep will destroy.

**Both models learned the dynamics.** Both beat hold last by three to four times at long
horizons, and both beat linear extrapolation from about horizon three onward, which is
only possible by having represented the wall.

## Files

| file | purpose |
| :--- | :--- |
| `data.py` | static dataset, blob rendering, pixel variance self check |
| `model.py` | NumPy layers, Adam, the two encoders, weight serialisation |
| `train.py` | trains both encoders, logs latent snapshots and probe metrics |
| `analyze.py` | latent ablation, PCA, task aligned projection, robustness sweep |
| `interactive_viz.py` | live Dash dashboard, narrative ordered |
| `export_demo.py` | animated GIF plus a self contained interactive HTML page |
| `data_seq.py` | trajectory dataset for the rollout phase |
| `model_seq.py` | encoder plus GRU world model, PyTorch |
| `train_rollout.py` | multi seed rollout sweep with paired statistics |
| `analyze_rollout.py` | aggregates the sweep into summary tables |
| `viz_rollout.py` | rollout figures for the dashboard and the HTML export |
| `writeup_notes.md` | full methods and results notes |

## Running it

```bash
pip install -r requirements.txt

python train.py            # about 45 seconds on CPU
python analyze.py          # ablation, PCA, robustness
python export_demo.py      # GIF plus results/demo.html
python interactive_viz.py  # dashboard at 127.0.0.1:8050

python train_rollout.py    # rollout sweep, about 40 minutes for 5 seeds
python analyze_rollout.py  # aggregates the sweep
```

Two artifacts are excluded from git because they are large and fully regenerated by the
commands above: `results/demo.html` at 6 MB, produced by `export_demo.py`, and
`results/rollout_predictions.npz` at 25 MB, produced by `train_rollout.py`.

## Limitations

* One toy, one architecture. Effects are fractions of a pixel.
* `LAMBDA_RECON` is fixed at 1.0 and was never swept, so arm A's balance between its two
  losses is arbitrary.
* A ten step horizon over near linear dynamics is a gentle test.
* Task only training did not produce distractor invariance here, only a cleaner and more
  concentrated task representation. Getting invariance would need an additional pressure
  this experiment does not have, such as an information bottleneck.
