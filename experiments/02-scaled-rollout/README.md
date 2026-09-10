# Experiment 2: scaled rollout with control arms

Experiment 1 compared two arms in a 16 by 16 world with one object and a ten step
horizon, and found a null: the task only latent neither collapsed nor clearly won. That
left two things unresolved. Does the result survive at a scale where the encoder actually
has to work, and if any arm does benefit, is that because reconstruction teaches
something about the scene, or merely because a second loss regularises the encoder?

This experiment answers both. Self contained Colab notebook, GPU runtime, roughly 400
minutes on a T4 for the full configuration.

## What changed from Experiment 1

| | Experiment 1 | Experiment 2 |
| :--- | :--- | :--- |
| frames | 16 by 16, one object | **48 by 48, target plus two moving distractor objects** |
| motion | one dimension | **two dimensions, reflecting off all four walls** |
| encoder | two layer MLP | **four block convolutional network** |
| latent | 8 | **32** |
| horizon | 10 blind steps | **25 blind steps** |
| arms | 2 | **4** |
| seeds | 5 | 4 |

## The four arms

All arms share an identical convolutional encoder, GRU and task head, initialised byte
identically and asserted so in code at runtime. They differ **only** in the extra loss
attached:

| arm | extra loss | literature analogue |
| :--- | :--- | :--- |
| `A_recon` | pixel reconstruction through a transposed convolution decoder | DreamerV3 |
| `B_task` | none, no decoder exists in the module | TD-MPC2 |
| `C_scene` | predict background colour and distractor positions, never touching a pixel | the negative control |
| `D_latent` | predict its own next latent, no decoder | JEPA style |

`C_scene` is the arm that settles the open question. It is scene aware but pixel free, so
if it recovers the benefit of reconstruction then the effect was never about pixels.

## Environment

Each trajectory is 45 frames containing a white target blob moving in two dimensions at
roughly constant velocity and reflecting off all four walls, two independently moving
coloured distractor blobs, a background colour drawn once per trajectory, and fresh per
pixel noise each frame.

Reflecting walls are deliberate. With straight line motion, 25 step prediction is pure
linear extrapolation and any model that recovers velocity solves it, so a horizon curve
would reveal nothing. Bounces make the dynamics piecewise linear and force the model to
represent the wall rather than merely the velocity.

Frames are rendered **on the GPU inside the training loop** from small stored factor
arrays. Materialising the full dataset would be roughly 7 GB.

## Evaluation

The model consumes 10 real frames, then steps the GRU forward on **zero input** for 25
steps. No pixels reach it during the rollout. Training uses the identical procedure with
teacher forcing on the context, so the curve measures representation quality rather than
a train and test mismatch.

Two baselines bracket every curve. **Hold last position** repeats the final observed
position and is the do nothing floor. **Linear extrapolation** fits velocity from the
true context and extrapolates, which is near optimal for straight line motion but walks
straight through walls, so beating it means having learned the bounce.

## Results

Mean over 4 seeds, Euclidean pixel error on a 48 pixel frame.

| arm | 1 step | 25 steps | growth |
| :--- | :--- | :--- | :--- |
| A, task plus reconstruction | 0.795 | 3.095 | 3.94x |
| **B, task only** | 0.804 | **3.067** | 3.90x |
| C, task plus scene auxiliary | 0.932 | 3.473 | 3.81x |
| D, task plus next latent | 16.750 | 18.185 | 1.09x (failed, see below) |
| baseline, hold last | 1.396 | 22.573 | 16.2x |
| baseline, linear extrapolation | 0.613 | 40.126 | 65.4x |

Paired comparisons against arm B, negative meaning B is better. Arms see identical
trajectories, so differences are compared pairwise rather than by asking whether separate
error bars overlap.

| comparison | 1 step | 10 steps | 25 steps | consistency |
| :--- | :--- | :--- | :--- | :--- |
| B against A | +0.009 | minus 0.051 | minus 0.027 | sign unanimous at only 15 of 25 horizons, seeds split 2 against 2 at the longest horizon |
| B against C | minus 0.128 | minus 0.284 | minus 0.405 | **unanimous at 25 of 25 horizons, B better in 4 of 4 seeds** |

### What is in each latent

This is the headline of the experiment. Linear probe R squared from each final latent,
averaged over 4 seeds.

| arm | task | background | distractor positions |
| :--- | :--- | :--- | :--- |
| A, task plus reconstruction | 0.973 | **0.988** | 0.012 |
| **B, task only** | 0.957 | **0.081** | 0.007 |
| C, task plus scene auxiliary | 0.974 | 0.997 | **0.491** |
| D, failed | 0.645 | 0.973 | 0.010 |

Per seed background readability for arm B: 0.035, 0.170, 0.084, 0.035.

## Reading the results

**One. A and B are indistinguishable on the task.** The separation is 0.027 px on a 48
pixel frame, which is 0.06 percent of the image, and the four seeds split two against two
on which arm is ahead. This is a null and should be reported as one.

**Two. Their latents are not remotely the same.** Arm A encodes essentially all of the
background at 0.988 while arm B encodes almost none of it at 0.081, roughly twelve times
less, for no measurable cost in accuracy. This **reverses the caveat published with
Experiment 1**, where task only training left the background almost fully readable at
0.963. At this scale, with a convolutional encoder and genuine moving clutter, task only
training does discard it.

**Three. The control arm answers the open question, and sharply.** A non pixel scene
auxiliary is not neutral, it is **harmful**: worse than task only at every horizon, in
every seed, unanimously. Arm C's latent proves it learned exactly what it was told, with
distractor readability jumping from 0.007 to 0.491, and it paid for that in task
accuracy. So the answer is neither "reconstruction helps" nor "any second loss helps".
Auxiliary objectives that force the encoder to represent task irrelevant content are at
best free, as with reconstruction, and at worst a tax, as with the explicit scene
auxiliary.

**Four. No arm degrades faster.** Growth factors of 3.94, 3.90 and 3.81 are
indistinguishable. The worry that a task only latent works one step ahead and then falls
apart is not supported out to 25 steps.

**Five. The models genuinely learned.** At the longest horizon they sit at roughly 3.1 px
against 22.6 px for hold last, seven times better, and 40.1 px for linear extrapolation,
thirteen times better. Beating linear extrapolation is only possible by having
represented the walls.

## Arm D failed

`D_latent` never trained. Its numbers are **not** evidence about self supervised
objectives and should not be presented as such.

The evidence that it is a constant predictor sitting at chance:

1. Always guessing the centre of the frame scores **18.22 px** for this configuration,
   computed from the trajectory distribution. Arm D scored between 16.8 and 18.9 px.
2. Its error **decreases** after horizon 10. A real dynamics model cannot improve with a
   longer horizon. A constant predictor can, because the target distribution shifts.
3. Its latent ablation magnitudes are **six orders of magnitude** larger than the other
   arms, a median of 3539 against roughly 0.001. The latent norms exploded.
4. Its latent carries far less task information, 0.645 against roughly 0.97.

**Root cause.** The objective was L2 normalised to stop the loss exploding, but nothing
bounded the latent norm, so the explosion simply moved. Latents grew until they saturated
the recurrent state and killed task learning.

**Fix before reusing this arm.** Apply a LayerNorm to the latent or an explicit norm
penalty, keep the stop gradient on the target, and consider a predictor and EMA target
asymmetry as in BYOL to prevent representational collapse. Verify by watching the latent
norm during training, not only the loss.

## Guardrails, and one that was not enough

Several protections in this notebook exist because a specific run failed without them.

**Rollout retention check, before training.** A freshly initialised GRU driven by zero
input contracts to a fixed point. Measured on this architecture it retained only about 1
percent of its initial hidden state after 10 steps and 0 percent after 25. The rollout
then emits a constant no matter what the context encoded, the model predicts the centre
of the frame, and the signal dies over the same steps so no gradient survives to teach it
otherwise. An earlier version of this experiment trained 4000 steps at exactly chance for
this reason. The fix biases the GRU update gate toward keeping state, which lifts
retention to 66 percent at 10 steps and 36 percent at 25, and is applied identically to
every arm. The notebook now measures retention at the configured horizon before training
and warns if it is too low.

**Learning gate, after training.** Every arm is checked against the trivial baselines,
and if one loses the notebook says loudly that nothing downstream is interpretable.

**The gate was not sufficient, and this run proved it.** Arm D **passed** the gate, at
18.2 px against hold last at 22.6 px, while sitting at chance. At a 25 step horizon hold
last has degraded so far that beating it proves nothing. The gate needs a chance floor
check as well, and future experiments should include one. Guardrails need their own
guardrails.

Other protections carried through: identical initialisation asserted at runtime, per seed
points plotted on every comparison figure, deterministic evaluation with seeded pixel
noise, and checkpoint plus resume so a dropped Colab session costs only the run in
flight.

## Files

| path | purpose |
| :--- | :--- |
| `hermes_rollout_colab.ipynb` | the experiment, self contained, drop into Colab |
| `nb_src.py` | the notebook source in plain Python, easier to diff and edit |
| `results/` | the full 4 seed run: CSVs, figures, animations, VERDICT.txt |
| `results-v1-failed/` | the earlier run where every arm sat at chance, kept as a record of the failure that motivated the retention check |

`results/VERDICT.txt` reports level, rate and robustness as three separate questions,
because they can and do disagree.

## Running it

Open the notebook in Colab, set the runtime to GPU, and run every cell. Leave
`QUICK_TEST = True` for the first pass, which takes about two minutes and executes every
cell so you learn the notebook works before committing GPU time. That preset is
**expected to fail the learning gate**, since fifteen optimiser steps trains nothing.

Then set `QUICK_TEST = False` and choose a scale:

| scale | seeds | epochs | steps per run | time |
| :--- | :--- | :--- | :--- | :--- |
| `mini` | 1 | 30 | about 690 | about 1 minute on a T4 |
| `medium` | 3 | 42 | about 1700 | 35 to 55 minutes |
| `full` | 4 | 60 | about 5000 | 2 to 3.5 hours |

Step counts are sized from a measured learning curve: this setup crosses the hold last
baseline at roughly 300 optimiser steps and is still improving at 800.

## Limitations

* One toy, one architecture, one family of dynamics.
* Auxiliary loss weights were never tuned and are all fixed at 1.0.
* The task is two dimensional position, so a 32 dimensional latent has ample room and
  capacity pressure is barely tested. Shrinking the latent is the cheapest way to add it.
* Robustness under distribution shift was **not** tested. These results measure what a
  latent contains, not whether a cleaner latent survives shifted distractors better.
* Model weights were not saved in the checkpoints, only logs and predictions, so an
  evaluation only follow up would currently require a full retrain. Future runs should
  save the state dict.
