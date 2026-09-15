# Experiment 3: action-conditioned selective relevance

Experiment 2 found that a task only latent matches a reconstruction latent while carrying
roughly twelve times less background. A skeptic can dismiss that in one line: colour never
mattered, so of course it was dropped. A model that ignored all colour would have scored
the same.

This experiment closes that gap in three ways:

1. **Actions.** The agent is steered, and the world model must imagine the consequences of
   acting, which is what a world model is for.
2. **Relevance is defined by reward**, not by a coordinate chosen by hand. The task loss is
   reward prediction, so whatever predicts return is task relevant by definition.
3. **Selective relevance.** One piece of colour is essential and another is useless, so a
   model that discards colour wholesale now fails visibly.

**Status: not yet run.** No results are reported here until a real run exists. The notebook
has been verified end to end on a CPU with the `quick` preset, including a simulated
disconnect in the middle of training and a full re-run against cached results. The GPU
specific paths (mixed precision, GPU random generators) are exercised by a preflight at
the start of every Colab run, before any long training begins.

## The environment

| | |
| :--- | :--- |
| frame | 48 by 48 RGB (32 by 32 for the `medium` preset) |
| agent | white Gaussian blob, steered by a continuous action in [-1, 1]² |
| dynamics | `v' = 0.95 v + 0.0025 a`, `p' = clip(p + v')`, velocity zeroed at walls. Top speed 2.35 px per step, full braking about 13.5 steps |
| goals | a black hollow ring and a black plus, static, at least 0.35 apart |
| selector | background hue 0.05 to 0.45 (warm) makes the ring pay; 0.55 to 0.95 (cool) makes the plus pay. Dead zones around both hue boundaries are never drawn |
| distractors | two full saturation coloured blobs bouncing at constant velocity; hues 0.60 to 0.72 are held out of training |
| noise | fresh per pixel noise, standard deviation 0.05 |
| reward | `r = -d + 0.5 exp(-d² / (2 · 0.05²))`, where `d` is the distance to the active goal |

Goals are black shapes rather than coloured ones, because a fixed goal colour vanishes on a
background of the same hue. This also leaves colour meaning exactly two things: the
background, which is relevant, and the distractors, which are not.

### Why damping 0.95: the design check

With direct position control, one step planning is optimal and a world model is pointless.
`design_check.py` and `design_check_momentum.py` simulate the **true** dynamics with a
perfect knowledge CEM planner, before any network exists, to confirm that momentum and the
selector both matter. These are numbers from that numpy simulation, 100 episodes, 50 steps,
not experimental results.

| true dynamics planner | damping 0.95 | damping 0.97 |
| :--- | :--- | :--- |
| no-op | -0.449 | -0.463 |
| oracle, 1 step planning | +0.037 | -0.117 |
| oracle, 4 steps | +0.150 | -0.027 |
| oracle, 8 steps | +0.194 | +0.064 |
| oracle, 15 steps | +0.187 | +0.088 |
| colour blind oracle, 8 steps | -0.132 | -0.173 |

At damping 0.90 (a separate 40 step run) a one step planner reached 92 percent of the best
return, so momentum did not matter. At 0.95 it reaches 76 percent and planning saturates by
about 8 steps. The colour blind planner, with perfect state, reaches about half and picks
the correct goal less than half the time, so ignoring the background genuinely costs return.

## The offline data

The world model learns from a recording, collected once per seed, identical for every arm.
The behaviour policy never looks at the background: 30 percent of episodes use smooth
random actions, 70 percent chase waypoints that are the ring, the plus or a random point by
coin flip.

Two tempting alternatives would break the experiment. A driver that heads for the active
goal lets the model read the answer off the actions without looking at the background.
Online collection gives every arm different data and destroys the paired comparison. A
leakage check verifies that the active goal cannot be predicted from the movements alone.

## The four arms

All arms share an identical encoder, action conditioned GRU and reward head, start from
byte identical weights (asserted at runtime), and train on the same batches in the same
order with the same pixel noise. They differ only in the extra loss, all weighted 1.0.

| arm | extra loss | analogue |
| :--- | :--- | :--- |
| `A_recon` | pixel reconstruction from the latent | DreamerV3, in its loss |
| `B_task` | none, no decoder exists | MuZero style value equivalence |
| `C_scene` | predict only irrelevant factors: distractor positions and hues, background saturation and brightness | negative control |
| `D_latent` | predict the future latent of an EMA target encoder | TD-MPC2 consistency, JEPA |

Arm C differs from Experiment 2's version. Predicting background **colour** would now
directly supervise the selector and hand C an unearned advantage, so it is excluded.

Arm D failed in Experiment 2 because nothing bounded the latent norm. The latent now passes
through a LayerNorm with no learnable gain in **every** arm, since a learnable gain would
reopen the same escape route and adding it to one arm alone would make the trunks differ.
Latent norm, per dimension spread and effective rank are logged every epoch.

Every arm uses a deterministic GRU rather than an RSSM. An RSSM's KL term is itself a
representation loss, and it would stop arm B from being reward only.

## Evaluation

**Level 1, reward prediction.** On 500 held out behaviour episodes, R² for each horizon
from 0 to 15 imagined steps, given the true actions. References: predict the mean, hold the
current reward, and a colour blind predictor using the true future positions. Re-run along
the planner's own trajectories to see whether prediction holds up where the model is used.

**Level 2, planning return, the headline.** 200 fixed episodes. CEM plans 10 steps ahead
inside each latent, 500 samples, 50 elites, 6 iterations, replanning every step. The real
return is recorded. The oracle uses the same planner with the true state, so the gap splits
into world model quality and planner quality. Baselines: no-op, random, oracle, one step
oracle, colour blind oracle, wrong goal oracle. Goal choice accuracy asks whether the agent
ends nearer the active goal than the inactive one.

**Level 3, latent probes.** Linear and small MLP probes on the encoder latent and on the
GRU state, trained and scored on different **episodes**. Experiment 2 split by frame, which
inflates scores for per episode constants. Targets are grouped as relevant (agent position
and velocity, active goal, selector bit), goal like but irrelevant (inactive goal), identity
(ring and plus positions) and irrelevant (distractors, background saturation and
brightness). Every probe also runs on the untrained starting weights, because a random conv
encoder already carries background colour. The headline statistic is selectivity, R² of the
active goal minus R² of the inactive goal.

**Distribution shift.** Evaluation only, from saved weights: four distractors, twice the
distractor speed, held out distractor hues, and three times the pixel noise. Goals,
backgrounds and start states are identical across conditions, so the comparison is paired.

## Guardrails

Carried from earlier experiments: GRU update gate bias toward keeping state, with a
retention check before training; byte identical initialisation asserted in code; per seed
points on every figure; paired comparisons on identical episodes; deterministic evaluation
with seeded noise; separate reporting of level and rate.

New in this experiment:

* **The learning gate checks the chance floor.** Experiment 2's gate compared only against
  hold last, and a broken arm passed it. Every seed and arm must now have reward R² above 0.5
  at one step, above zero at the full horizon, below hold current error, a prediction spread
  of at least 0.3 of the true spread (a constant predictor has none), planning return above
  the floor with a confidence interval excluding zero, and an uncollapsed latent. Error that
  falls with horizon is flagged.
* **Leakage check** on the behaviour policy.
* **Checkpoints include weights**, both every few epochs and at the end of each run, so
  evaluation only follow ups need no retraining.
* **Preflight** runs every GPU code path on a toy batch before the sweep.
* **Config lock.** A run folder refuses to resume under a configuration that would
  invalidate its checkpoints.

## Running it

1. Open `hermes_exp3_colab.ipynb` in Colab. The notebook requests a GPU runtime; confirm
   it under Runtime, Change runtime type.
2. Runtime, Run all.
3. Approve the Google Drive prompt. Everything is written to
   `MyDrive/hermes_exp3/<scale>/`.
4. If Colab disconnects, reconnect and Run all again. Finished runs, half finished runs and
   finished evaluations are loaded from Drive.

| scale | frames | seeds | training data | epochs | purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `quick` | 24 px | 1 | 96 episodes | 2 | proves every cell runs; expected to fail the gate |
| `medium` | 32 px | 3 | 6000 episodes | 40 | **default**: does every arm pass the learning gate |
| `full` | 48 px | 4 | 8000 episodes | 60 | the headline configuration |

Start with `medium`. If every arm passes the gate, set `SCALE = "full"` in the configuration
cell and Run all again; each scale has its own Drive folder. No run time is promised. The
notebook prints a measured estimate after the first epoch of each arm.

### What lands on Drive

```
hermes_exp3/<scale>/
  config.json        the configuration this folder is locked to
  progress.json      which stages have finished, with timestamps
  ckpt/              seed{s}_{arm}_final.pt (weights, log, reward stats); _partial.pt while training
  eval/              cached baselines, per arm evaluations, probe tables
  results/           CSVs, figures, GIFs, VERDICT.txt
  hermes_exp3_<scale>_results.zip
```

`results/` holds `VERDICT.txt`; CSVs for training, level 1, planning, the learning gate,
probes, selectivity, optimism and shift; figures `fig01` to `fig11`; and three GIFs,
including `gif02_planning.gif`, which shows each arm steering with its chosen plan and a
probe readout of its imagined positions overlaid on the real frames.

## Limitations

* Auxiliary loss weights are not tuned.
* Arm A is Dreamer like in its loss, not its architecture.
* The selector is the most salient pixel feature, which favours reconstruction on selection
  for a reason unrelated to the hypothesis. A low salience cue would favour arm B instead.
  Neither variant is run here.
* A 32 dimensional latent against roughly 7 relevant quantities applies no capacity
  pressure.
* The planner has no value function and sees 10 steps ahead. The oracle shares the limit.
* Probes can mislead in both directions, which is why they are read alongside behaviour.

## Files

| path | purpose |
| :--- | :--- |
| `hermes_exp3_colab.ipynb` | the experiment, self contained, drop into Colab |
| `nb_src.py` | the notebook source in plain Python, easier to diff and edit |
| `build_notebook.py` | regenerates the notebook from `nb_src.py` |
| `design_check.py` | numpy design check: proposed frames, true dynamics planners |
| `design_check_momentum.py` | the damping sweep behind the choice of 0.95 |
