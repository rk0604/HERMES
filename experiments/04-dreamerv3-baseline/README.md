# Experiment 4: DreamerV3 baseline and learning-signal ablations

**Status: not yet run on a GPU.** No training result is reported here until a real run
exists. Everything in this folder has been verified on a CPU with DreamerV3's tiny `debug`
configuration (see [Verification status](#verification-status) for exactly what that
covers and what it does not).

This experiment leaves the toy world of Experiments 1 to 3 and moves to the real thing:
Hafner et al.'s own DreamerV3 code, unmodified except for two small, verified patches. The
purpose of this phase is narrow and deliberate: **reproduce the paper's learning-signal
ablation** (Nature 2025, Fig. 6b; arXiv v3, Fig. 6b and Fig. 18) on two tasks, with a
proper unablated baseline, and in doing so build a working, instrumented understanding of
the codebase before any HERMES-specific change is made to it.

## What the paper did

From the paper and its supplement (`references/`):

* Four arms: **Dreamer**, **No value gradients**, **No reward or value gradients**,
  **No reconstruction gradients**. The ablations *stop gradients* from a loss into the
  world-model representation; every loss is still computed and its head still trains.
* 14 tasks, the **200M** model, one A100 per run, **5 seeds**, mean and one standard
  deviation reported. Crafter was run to 5M steps in the ablation figure; the Atari
  ablations used Atlantis, Breakout and Montezuma at the full 200M-frame budget, not the
  Atari100k benchmark.
* Finding: Dreamer "rests predominantly on the unsupervised reconstruction loss of its
  world model"; removing reward and value gradients costs little, removing reconstruction
  gradients costs a lot on average.

## What this experiment runs

| | |
| :--- | :--- |
| code | upstream `danijar/dreamerv3` at `e3f0224` (2026-05-25) plus two patches below |
| tasks | `crafter_reward` (config block `crafter`, 1.1M env steps) and `atari100k_pong` (block `atari100k`, 110K agent steps = 440K frames) |
| arms | baseline, `abl_novalue`, `abl_norewval`, `abl_norecon` |
| seeds | 1 to start; the notebook's timing section projects the cost of more |
| model size | chosen from measured timing; the 200M default is what the paper used and is expected to be far beyond a Colab budget for Crafter |
| compute | one Colab GPU session at a time, resumable, everything on Google Drive |

The arms map onto the code as follows. The first two flags already exist upstream; the
third is added by the patch.

| paper arm | config block | flags it sets | where the stop-gradient acts |
| :--- | :--- | :--- | :--- |
| Dreamer | (none) | | |
| No value gradients | `abl_novalue` | `agent.repval_grad: False` | `agent.py:220`, critic loss on replayed states |
| No reward or value gradients | `abl_norewval` | `agent.reward_grad: False`, `agent.repval_grad: False` | `agent.py:172` and `agent.py:220` |
| No reconstruction gradients | `abl_norecon` | `agent.rec_grad: False` (new) | `agent.py:170-171`, decoder input |

Two things to keep in mind when reading results: the **continue** predictor always shapes
the latent (no flag exists for it, `agent.py:177`), and the actor and critic losses on
imagined trajectories never do (`ac_grads: False`), so "no reward or value gradients" still
leaves continuation, the two KL terms and reconstruction.

## The patches

`rec_grad.patch` (12 lines) adds `rec_grad: True` to the defaults, wraps the decoder input
in the codebase's own `sg(x, skip=flag)` idiom, and adds the three named blocks. It is
deliberately a stop-gradient rather than `loss_scales.rec: 0`: with the scale at zero the
decoder would stop learning, whereas with the gradient stopped the decoder keeps training
on a detached latent and becomes a probe of how much pixel detail that latent retains, the
same role the background-readability probe played in Experiment 2.

`atari_ale_compat.patch` (1 line) makes the Atari wrapper's `setInt` call compatible with
`ale_py` 0.10 and later (`str` key, Python `int`). It is harmless under the author's
pinned 0.9.0, which the notebook tries first.

Both patches are printed, written to Drive next to the runs, applied with `git apply`, and
verified by assertions on the patched files. Nothing else in the upstream code is changed.

## Files

| file | purpose |
| :--- | :--- |
| `ARCHITECTURE_MAP.md` | paper concepts traced to file and line in the code; read this first |
| `nb_src.py` | source of the notebook, percent format; **edit this, not the `.ipynb`** |
| `build_notebook.py` | builds `hermes_exp04_colab.ipynb` from `nb_src.py`, embedding the three files below |
| `hermes_exp04_colab.ipynb` | the Colab notebook |
| `inspect_agent.py` | builds the agent from config flags and prints the parameter tree, the RSSM tensor shapes, and a gradient-flow table with assertions |
| `rec_grad.patch`, `atari_ale_compat.patch` | the two patches |

## How to run

Open `hermes_exp04_colab.ipynb` in Colab with an **A100** (L4 works; T4 has no bfloat16
hardware and is not recommended). Set `PRESET = 'smoke'` and **Run all** once: every stage,
including the run matrix and the plots, executes with the `debug` network in minutes and
proves the pipeline on that machine. Then set `PRESET = 'full'`, read the timing section's
cost projection, choose `SIZE` and `SEEDS`, and run again. If Colab disconnects, reconnect
and Run all: finished runs are skipped, the run in flight resumes from its last checkpoint.

The notebook writes to `MyDrive/HERMES/exp04-dreamerv3-baseline/`:

```
runs/<task>/<arm>/seed<N>/   DreamerV3 logdir: metrics.jsonl, scores.jsonl, config.yaml,
                             ckpt/, replay/, scope/, plus stdout.log (timestamped),
                             command.txt, sessions.jsonl, DONE.json
timing/                      the short runs used to measure speed
smoke/                       throw-away smoke-test logdirs
plots/                       scores.png, world_model_losses.png, openloop_seed<N>.png
records/                     env_info.json, versions.txt, pip_freeze.txt, the patches,
                             inspect_<arm>.{log,json}, timing.json, summary.csv
```

## Deciding the budget

The notebook refuses to guess. For each size in `TIMING_SIZES` it runs the real
configuration for a few hundred gradient updates into a logdir on Drive, reads the
`fps/policy` DreamerV3 logs (environment steps per second of wall time, training
included), and projects hours per run and per matrix; with `UNITS_PER_HOUR` from Colab's
resources panel it also prints compute units. The arithmetic that makes Crafter the
expensive task: gradient updates per environment step are `train_ratio / (batch_size ×
batch_length)`, i.e. 0.5 for Crafter and 0.25 for Atari100k, so a Crafter run is about
550K updates and an Atari100k run about 27.5K.

## Verification status

| what | verified how | result |
| :--- | :--- | :--- |
| pinned dependency set installs and imports | local CPU venv, Python 3.13.14 | jax 0.5.0, numpy 2.5.3, elements 3.22.1, ninjax 3.6.3, portal 3.8.1, scope 0.7.1, optax 0.2.5, chex 0.1.90, einops 0.8.2, crafter 1.8.3, ale_py 0.12.1, av 18.1.0 |
| `crafter debug`, `atari100k debug`, `dummy debug` train end to end | local CPU | exit 0; metrics, scores, checkpoint, replay chunks, scope outputs written |
| resume by re-running with the same `--logdir` | local CPU | second run loads `ckpt/latest`, continues; re-logged steps handled by `lineage()` in the notebook |
| `rec_grad.patch` applies to `e3f0224` and does what it claims | `git apply --check`; `inspect_agent.py` gradient-flow table on all four arms | `image → enc/RSSM` gradient exactly 0 only under `abl_norecon`; `rew` and `repval` exactly 0 under `abl_norewval`; `repval` 0 under `abl_novalue`; every other row unchanged |
| `atari_ale_compat.patch` | Atari100k debug run with ale_py 0.12.1 | exit 0 (fails without it: `setInt(): incompatible function arguments`) |
| RSSM shapes and one-hot latents | `inspect_agent.py`, debug config | `z` and `ẑ` one-hot; imagination starts from all `B×T` states for 15 steps |
| whole notebook, `PRESET='smoke'` | executed locally with nbconvert on the CPU venv kernel (2026-09-15) | all 19 code cells ran, 0 errors: clone + patches, 4 inspection arms, 2 CPU smoke runs, 2 timing runs, the 8-run matrix, 3 plots, summary table; ~8 minutes |
| the install cell on Colab (JAX CUDA wheel, Python version, `ale_py==0.9.0` wheel) | **not verified** | most likely place to need a fix on first Colab run; the cell prints the real error |
| bfloat16 on the assigned GPU, the GPU smoke test, the profiler window | **not verified** | the notebook's smoke section covers them on first Colab run |
| timing, cost, model size choice | **not verified** | measured by the notebook, never estimated here |

## Known limitations and things that bit us

* **Environments are not seeded** by the stock config (`main.py:241-242` seeds only
  suites with `use_seed`; Crafter and Atari get none). `--seed` fixes every JAX draw, but
  episode content differs run to run. Seeded evaluation, a HERMES guardrail, would need
  `train_eval` or a small addition later; the paper's own numbers are training-episode
  returns, which is what we plot.
* **Checkpoints are on a wall-clock timer** (`run.save_every`, 600 s here) and none is
  written when the loop ends. The final checkpoint can be up to `SAVE_EVERY` older than
  the final metrics. A run that resumes from a stale checkpoint re-logs the steps in
  between; the notebook's `lineage()` keeps only the final agent's history.
* **The replay buffer is written to Drive** as compressed chunks at every checkpoint and
  nothing deletes them; the folder grows for the whole run.
* **The JAX profiler cannot be disabled from the config** and fires at updates 100-120;
  the GPU smoke test runs past it on purpose.
* **`ale_py`:** 0.9.0 is the author's pin and has ROMs bundled; it may lack a wheel for the
  Python version Colab ships, in which case the notebook falls back to 0.12.1 and says so.
* **Windows only:** `elements.LocalPath.glob` returns backslash paths, `Path.name` then
  fails to recognise `latest`, and the checkpoint cleanup deletes the checkpoint it just
  wrote. Irrelevant on Colab; the local tests patched the test venv's copy of `elements`.
* **Zero-initialised heads.** The reward and value heads use `outscale: 0.0`, so at step 0
  they send no gradient into the latent regardless of any flag; the gradient-flow test
  gives those kernels a small random value to test the path rather than the initial
  value. The KL terms sit at the 1-nat free-bits floor at initialisation.
