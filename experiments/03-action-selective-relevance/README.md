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

**Current state: v1 and v2 have both been run at `medium` scale. Neither supports a claim
about reconstruction versus reward only. One narrow observation replicates across both
runs: reward only training strips task-irrelevant background detail that reconstruction
keeps. v3 is being built to fix what v2 exposed.** All notebooks are kept; each writes to
its own Drive folder.

| notebook | source | status |
| :--- | :--- | :--- |
| `hermes_exp3_colab.ipynb` | `nb_src.py` | v1, run once at `medium`; results in `hermes_exp3_medium_results/` |
| `hermes_exp3_v2_colab.ipynb` | `nb_src_v2.py` | v2, run once at `medium`; results in `hermes_exp3_v2_medium_results/` |

## What the v1 medium run showed

Three seeds, 32 px, 40 epochs, 200 planning episodes per arm. What held up:

* The design itself is sound. The active goal cannot be predicted from movement alone
  (0.488 and 0.51 against a chance band of 0.50 ± 0.022); imagination retains 50 percent of
  its starting state at 15 steps; arm D trained without the latent explosion that killed it
  in Experiment 2.
* **Reward only training learned the colour signal.** B, C and D head for the active goal 99
  to 100 percent of the time, against 52 percent for a colour blind planner with perfect
  state, and predict reward 15 steps out at R squared 0.89 to 0.90 against 0.40 for a true
  state predictor that cannot see the selector.
* **B removes irrelevant background detail.** Background saturation and brightness read 0.63
  from the untrained starting weights and 0.00 from B in every seed. Arm A keeps 0.55.
* B, C and D are tied on planning return: the gaps are smaller than the seed spread and the
  seeds disagree on the sign.

What blocks any claim about reconstruction versus reward only:

1. **Arm A was still training.** In two of three seeds its loss fell 53 percent in the last
   five epochs and its 15 step reward R squared jumped from roughly 0.27 to 0.61 to 0.68
   between epochs 35 and 40. The third seed escaped the plateau at epoch 25 and finished
   level with the others. "A is worse" is indistinguishable from "A is slower".
2. **The two sharpest probes had no dynamic range.** Distractor positions and the inactive
   goal read 0.00 from every latent, including the untrained weights. There was no way to
   tell "B dropped it" from "nothing could read it".
3. **The negative control never learned its task.** C's auxiliary loss barely moved and it
   read distractor positions back at 0.03 to 0.35, so irrelevant content was never actually
   forced into its latent.
4. **Planning measured exploitation, not selection.** Every arm's imagined reward was
   systematically optimistic (bias close to the entire error), reward R squared went negative
   along the planner's own trajectories within a handful of steps, and the arms scored about
   level with the colour blind oracle in return despite choosing the right goal almost always.
5. **The shift test was uninformative**: changing the distractors moved return by at most
   0.006 for any arm, because no latent encodes distractors at all.

## What v2 changes

| v1 problem | v2 fix |
| :--- | :--- |
| fixed 40 epochs, arm A unfinished | every arm trains to a **held-out plateau** under the same ceiling; **epochs needed** is reported as its own metric, and the gate flags any run that hit the ceiling while still improving |
| probes with no dynamic range | a **trained CNN ceiling**: a conv encoder of the same architecture trained directly on frames to read each factor. A target the ceiling cannot read is reported as untestable rather than as dropped |
| arm C never learned its target | C now predicts an **identity-free distractor occupancy heatmap**. v1 asked for each distractor's position *by slot*, but slots have no stable identity, so the target was ambiguous |
| planner exploited off-distribution model error | part of the offline data is now driven by **the evaluation planner itself, aimed at a coin-flip goal**, which covers the approach-and-stop states the planner visits while still leaking nothing |
| one failed seed discarded a whole arm | claims are computed on the seeds where **both** arms passed the gate |

Measurements taken while building v2, each from a real run:

* **The parking driver leaks nothing.** Its coin-flip goal agrees with the active goal
  0.4996 of the time over 40 seeds by 4000 episodes. The v1-style warning seen in a 96
  episode smoke test is small sample noise (sd 0.094 at 29 parked episodes).
* **A raw pixel probe is too weak to be the positive control.** A linear probe on an 8 by 8
  downsample reads nothing but agent position (0.44), so it would mark every interesting
  target untestable. The trained CNN ceiling is used instead.
* **Probe targets must be weighted per group.** With one pooled objective, the 64
  dimensional heatmap drowned the 2 dimensional targets: ring position went from 0.00 to
  0.52 once each group was weighted equally. The same fix is applied to the MLP probe.
* **Frame size decides what the probes can test.** CNN ceiling, 3000 scenes:

  | target | 32 px | 48 px |
  | :--- | :--- | :--- |
  | agent position | 0.90 | 0.89 |
  | background saturation and value | 0.90 | 0.89 |
  | distractor heatmap | 0.68 | 0.76 |
  | ring position | 0.52 | 0.74 |
  | inactive goal | 0.27 | 0.63 |
  | distractor positions, by slot | 0.08 | 0.17 |

  **The inactive goal, which is the sharpest selectivity test, is only testable at 48 px.**
  Run `medium` to check the gate, but make selectivity claims from the `full` run.
  Per-slot distractor positions stay unreadable at both sizes, which is the direct evidence
  that v1's by-slot target was ill posed.

## What the v2 medium run showed

Three seeds, 32 px, a 120 epoch ceiling, and the same 200 planning episodes as v1 (every
baseline matches v1 exactly). Results in `hermes_exp3_v2_medium_results/`.

What holds up, and now replicates across both runs:

* **B strips irrelevant background detail.** On the seeds that passed the gate, background
  saturation and brightness read 0.00 and 0.00 from B, and the shade within the warm or cool
  band 0.00 and 0.00. Arm A keeps 0.61 and 0.69 (band shade 0.72 and 0.68), the untrained
  weights read 0.59 to 0.66 (0.43 to 0.83), and the CNN ceiling 0.91. v1 gave B 0.00 in all
  three seeds. B's one failed seed, which never learned the task, still reads the band shade
  at 0.90, so the stripping comes from learning.
* **The negative control now learns its target**: C reads the distractor heatmap at 0.70 to
  0.83, where v1's C never encoded distractors at all.

What blocks a claim:

1. **The early-stopping rule stopped four of twelve runs on the initial plateau.** A seed 0
   (epoch 40), B seed 2 (35) and C seeds 0 and 2 (30 and 35) were stopped with held-out reward
   R squared still at or below zero. Runs that learned broke out between epochs 20 and 35: D
   seed 2 sat at -0.04 at epoch 30 and reached 0.72 at epoch 35. The convergence flag called
   the stopped runs converged, because a flat plateau looks converged. Every gate failure in
   this run is an artefact of that rule, and A versus B rests on a single paired seed.
2. **The verdict's selection criteria were judged on that single seed**, which "in every
   usable seed" passes trivially. Its "SELECTION" line is not evidence.
3. **Selectivity does not separate A from B.** The inactive goal is readable in principle
   (ceiling 0.84) and every trained arm drops it: A 0.06 and 0.00, B 0.00 and 0.01, D 0.00,
   0.12 and 0.00. The active goal is level: A 0.85 and 0.88, B 0.85 and 0.90. A's decoder
   barely draws the goals (contrast kept 0.05 to 0.32 of the real frame's).
4. **The CNN ceiling is not a ceiling for every target.** It reads the distractor heatmap at
   0.24 while C reads it at 0.70 to 0.83: one network trained on thirteen target groups at once
   under-serves some of them.
5. **Planning regressed against v1 on identical episodes.** On passing seeds B's goal choice
   fell from 1.00 and 0.995 to 0.775 and 0.87, and it stopped 7 to 9 px from the goal instead
   of about 3. D fell from 0.995 to 1.00 down to 0.885 to 0.95, stopping 5 to 7 px away instead
   of 2 to 4. Offline reward R squared on the held-out set is comparable in both runs (about
   0.89 to 0.91 for B), but imagined reward became more optimistic (bias 0.30 to 0.45 for B,
   against 0.16 to 0.24 in v1). v2 changed the data, the stopping rule and C's target together,
   so the cause is not identified; the parking data is the leading suspect.
6. **The shift test is still flat**: the mean change in return under the distractor shifts is
   at most 0.008 for any arm.

## Next: v3

1. **Stopping that cannot end a run on the plateau**: patience only starts counting once an arm
   has learned (held-out R squared at least 0.5), a higher minimum epoch count, and runs that
   never learn within the ceiling reported as such rather than as converged.
2. **A ceiling that is a ceiling**: one network per target group, flagged invalid for any
   target an arm reads better than it does.
3. **No criteria on fewer than three usable paired seeds**; the verdict says "insufficient
   seeds" instead.
4. **A parking ablation** to identify the planning regression: B and D trained with and without
   the planner-parked episodes, compared on identical planning episodes.
5. **Five seeds.**

## The environment

| | |
| :--- | :--- |
| frame | 48 by 48 RGB (32 by 32 for the `medium` preset) |
| agent | white Gaussian blob, steered by a continuous action in [-1, 1]² |
| dynamics | `v' = 0.95 v + 0.0025 a`, `p' = clip(p + v')`, velocity zeroed at walls. Top speed 2.35 px per step, full braking about 13.5 steps |
| goals | a black hollow ring and a black plus, static, at least 0.35 apart |
| selector | background hue 0.05 to 0.45 (warm) makes the ring pay; 0.55 to 0.95 (cool) makes the plus pay. Dead zones around both hue boundaries are never drawn |
| distractors | two full saturation coloured blobs bouncing at constant velocity; hues 0.60 to 0.72 are held out of training |
| reward | `r = -d + 0.5 exp(-d² / (2 · 0.05²))`, where `d` is the distance to the active goal |

Goals are black shapes rather than coloured ones, because a fixed goal colour vanishes on a
background of the same hue. This also leaves colour meaning exactly two things: the
background, which is relevant, and the distractors, which are not.

### Why damping 0.95: the design check

With direct position control, one step planning is optimal and a world model is pointless.
`design_check.py` and `design_check_momentum.py` simulate the true dynamics with a perfect
knowledge CEM planner, before any network exists. Numbers from that numpy simulation, 100
episodes, 50 steps:

| true dynamics planner | damping 0.95 | damping 0.97 |
| :--- | :--- | :--- |
| no-op | -0.449 | -0.463 |
| oracle, 1 step planning | +0.037 | -0.117 |
| oracle, 8 steps | +0.194 | +0.064 |
| oracle, 15 steps | +0.187 | +0.088 |
| colour blind oracle, 8 steps | -0.132 | -0.173 |

At damping 0.90 a one step planner reached 92 percent of the best return, so momentum did
not matter. At 0.95 it reaches 76 percent. The v1 run confirmed both ends on the real task:
the one step oracle scored -0.001 against +0.160 for the full oracle, and the colour blind
oracle -0.145.

## The offline data

Collected once per seed, identical for every arm, cached on Drive. The driver never looks at
the background: smooth random actions, waypoint chasing where each waypoint is the ring, the
plus or a random point by coin flip, and (v2) planner-driven parking aimed at a coin-flip
goal. A leakage check verifies the active goal cannot be predicted from the movements alone.

## The four arms

All arms share an identical encoder, action conditioned GRU and reward head, start from byte
identical weights (asserted at runtime), and train on the same batches in the same order.
They differ only in the extra loss, all weighted 1.0.

| arm | extra loss | analogue |
| :--- | :--- | :--- |
| `A_recon` | pixel reconstruction from the latent | DreamerV3, in its loss |
| `B_task` | none, no decoder exists | MuZero style value equivalence |
| `C_scene` | predict only irrelevant factors: a distractor occupancy heatmap, background saturation and brightness | negative control |
| `D_latent` | predict the future latent of an EMA target encoder | TD-MPC2 consistency, JEPA |

The latent passes through a LayerNorm with no learnable gain in every arm, which is the fix
for Experiment 2's arm D explosion. Every arm uses a deterministic GRU rather than an RSSM,
because an RSSM's KL term is itself a representation loss and would stop B being reward only.

## Evaluation

**Level 1, reward prediction.** Held out behaviour episodes, R² for horizons 0 to 15 given
the true actions, against three references: predict the mean, hold the current reward, and a
colour blind predictor with true future positions. Also measured along the planner's own
trajectories, in raw reward units as well as R².

**Level 2, planning return.** 200 fixed episodes, CEM with 500 samples and 10 step horizon
inside each latent, replanning every step. Baselines: no-op, random, oracle, one step oracle,
colour blind oracle, wrong goal oracle. Goal choice accuracy is the selection headline;
return and stopping distance are the model quality headline.

**Level 3, latent probes.** Linear and MLP probes on the encoder latent and the GRU state,
split by episode, with the untrained weights and the CNN ceiling as references. Headline
statistic: selectivity, R² of the active goal minus R² of the inactive goal.

**Distribution shift.** Evaluation only: four distractors, double speed, held out distractor
hues, triple pixel noise, with goals, backgrounds and starts identical across conditions.

## Guardrails

Carried over: GRU keep-state bias with a retention check, byte identical initialisation
asserted in code, per seed points on every figure, paired comparisons, deterministic
evaluation, level and rate reported separately, and a learning gate that checks the chance
floor, a constant-predictor detector and the planning floor.

Added in v2: convergence checking, the CNN probe ceiling, a validity check on arm C's own
auxiliary target, a reconstruction fidelity check on arm A, and dataset caching.

## Running it

1. Open `hermes_exp3_v2_colab.ipynb` in Colab (GPU runtime).
2. Runtime, Run all, and approve the Google Drive prompt. Everything lands in
   `MyDrive/hermes_exp3_v2/<scale>/`.
3. If Colab disconnects, reconnect and Run all again: finished runs, half finished runs,
   cached datasets and finished evaluations are all picked up.

| scale | frames | seeds | training data | max epochs | purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `quick` | 24 px | 1 | 96 episodes | 4 | proves every cell runs; expected to fail the gate |
| `medium` | 32 px | 3 | 6000 episodes | 120 | gate check |
| `full` | 48 px | 4 | 8000 episodes | 150 | the headline configuration, and the only one where the inactive goal probe is testable |

Epochs are a ceiling: arms stop at a held-out plateau, so run time is not fixed. For
reference, v1 measured 2.8 s/epoch for the task only arm and 5.0 s/epoch for the
reconstruction arm at 32 px.

## Limitations

* Auxiliary loss weights are not tuned.
* Arm A is Dreamer like in its loss, not its architecture.
* The selector is the most salient pixel feature, which favours reconstruction for a reason
  unrelated to the hypothesis. A low salience cue would favour arm B instead; that variant
  is deliberately deferred.
* A 32 dimensional latent against roughly seven relevant quantities applies no capacity
  pressure.
* The planner has no value function and sees 10 steps ahead; the oracle shares the limit.
* Parking data makes the offline set partly planner shaped, so on-policy accuracy is now
  partly a property of the data. The behaviour data column remains the out of distribution
  reading.

## Files

| path | purpose |
| :--- | :--- |
| `hermes_exp3_v2_colab.ipynb` | v2, run once at `medium` |
| `nb_src_v2.py` | v2 source in plain Python |
| `hermes_exp3_v2_medium_results/` | the v2 medium run: CSVs, figures, GIFs, VERDICT.txt |
| `hermes_exp3_colab.ipynb`, `nb_src.py` | v1, kept unchanged for reference |
| `build_notebook.py` | regenerates a notebook: `python build_notebook.py nb_src_v2.py` |
| `design_check.py`, `design_check_momentum.py` | numpy design checks behind the dynamics constants |
| `hermes_exp3_medium_results/` | the v1 medium run: CSVs, figures, VERDICT.txt |
