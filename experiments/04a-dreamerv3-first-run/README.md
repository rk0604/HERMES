# DreamerV3 first run: one stock agent on Atari100k Pong, from Colab

**What this is.** A single Colab notebook that trains **one unmodified DreamerV3 agent**
(Hafner et al., upstream code at a pinned commit) on **Atari100k Pong**, writes everything
to Google Drive, survives disconnects, and shows what the agent is doing. Its purpose is
familiarity with the real codebase: run it once, end to end, and see the pieces. It is not
an experiment. The ablation study built on the same machinery lives in
`../04-dreamerv3-baseline/`, together with `ARCHITECTURE_MAP.md`, the paper-concept →
code-line map that this notebook's explanations point into.

**Status: not yet run on a GPU.** Every stage was executed on a CPU with DreamerV3's tiny
`debug` network (see [What was verified, and where](#what-was-verified-and-where)). No
number in this folder is a GPU result.

## Files

| file | purpose |
| :--- | :--- |
| `hermes_dreamer_first_run_colab.ipynb` | the notebook; upload it to Colab as is |
| `first_run_src.py` | source of the notebook in percent format; **edit this, not the `.ipynb`** |
| `build_notebook.py` | builds the notebook from the source, embedding the two files below |
| `inspect_agent.py` | builds the agent from config flags and prints its parameter tree and the tensor shapes flowing through the RSSM |
| `atari_ale_compat.patch` | one line for the Atari *wrapper*, applied only if the author's pinned `ale_py==0.9.0` cannot be installed |

## What the notebook does, cell by cell

1. **Settings.** `CONFIG_BLOCK='atari100k'`, `TASK='atari100k_pong'`, `SIZE='size12m'`,
   `SEED=0`, the step budget (`None` = the block's own 1.1e5 agent steps = 440K frames),
   the checkpoint and log timers in seconds, and `UNITS_PER_HOUR` for cost projection.
2. **Machine and Drive.** Mounts Drive, creates
   `MyDrive/HERMES/dreamerv3-first-run/{runs,smoke,timing,plots,records}`, records GPU name
   and memory (`nvidia-smi`), Python version, and warns if the GPU is a T4 (no bfloat16
   hardware; DreamerV3 computes in bfloat16).
3. **Install.** Pins `jax[cuda12]==0.5.0` (the version in the repo's `Dockerfile`),
   `elements`, `ninjax`, `portal`, `scope`, `optax`, `chex`, `einops`, `av`, and
   `ale_py==0.9.0` with a fallback to `0.12.1`. Then runs JAX **in a subprocess** and asserts
   it sees a `CudaDevice` and can do a bfloat16 matmul. The notebook process never imports
   JAX: no runtime restart, no GPU memory held by the notebook, no kernel death on a crash.
4. **Clone DreamerV3** at `e3f0224` and print the commit. If `ale_py` had to fall back, apply
   the one-line wrapper patch and say so; otherwise assert the tree is byte-identical to the
   commit.
5. **See the architecture.** Writes and runs `inspect_agent.py` with the run's own config
   flags: the full parameter dictionary with shapes and per-module totals (the real size of
   `size12m`), and the shapes of image → tokens → `h` → `z` → prior logits → imagined `h`,
   `ẑ` for one batch, with checks that `z` and `ẑ` are one-hot and that imagination starts
   from every posterior state. Saved to `records/inspect.{log,json}`.
6. **The runner.** `run_dreamer(logdir, configs, flags)`: launches `dreamerv3/main.py` as a
   subprocess, streams the important lines, writes every line with a timestamp to
   `stdout.log`, writes `DONE.json` on a clean exit, appends the session's wall time to
   `sessions.jsonl`. Resuming is DreamerV3's own mechanism: same command, same `--logdir`.
   Also reads the config block's budget, action repeat and train ratio from `configs.yaml`
   so that nothing is typed in twice.
7. **Smoke tests.** Pong with the `debug` block on CPU (600 steps), then on the GPU (1500
   steps, past the built-in profiler window at updates 100–120). Each is checked for
   metrics, scores, a complete checkpoint and saved replay chunks.
8. **Timing.** The real configuration for 3000 steps (about 500 updates after the 1024-step
   replay warm-up) into a Drive logdir, with a forced checkpoint so Drive I/O is measured.
   Reads `fps/policy` from DreamerV3's own log and prints hours for the full run, and
   compute units if the rate was entered.
9. **The run.** `runs/<task>/<size>/seed<N>/`. Prints fresh / resuming from step N / done,
   then runs to the budget. Interrupting is safe; running the cell again resumes.
10. **Look at it.** Score against the paper's five 200M Pong seeds and PPO (shipped in the
    repo under `scores/`); the world-model losses and speeds; the open-loop video at the
    earliest and latest report (observed half vs imagined half); six frames of the agent's
    latest recorded episode; the timer breakdown and `nvidia-smi` capture; a summary JSON
    with the mean return of the last 10% of the budget next to the paper's Table 4 value.

## What "stock" means here

No change to the algorithm, the networks, the losses or the training loop. The command
line is the repo's own plus `--configs atari100k size12m`, `--task`, `--seed`, and
`--run.save_every` / `--run.log_every` (600 s and 60 s instead of the defaults 900 s and
120 s, so that a disconnect costs less). The only possible source change is the one-line
Atari wrapper fix, and only if the pinned `ale_py` has no wheel for Colab's Python.

## What is deliberately left out

* Ablations, config patches, a run matrix, seeds sweeps: `../04-dreamerv3-baseline/`.
* Crafter: about twenty times the gradient updates of Atari100k (1.1M steps at train ratio
  512 versus 110K at 256). Switching is a two-string change once Pong has run.
* Background execution with a live dashboard: Colab runs one cell at a time; a background
  process would add a second way to corrupt a resumable logdir. Progress streams into the
  run cell instead, and the results cells work on whatever exists.
* Seeded evaluation episodes: the stock `train` script has none, and adding one means
  code. What is plotted is what the paper plots: training-episode returns.
* The `scope` web viewer: it needs a server and a port; the notebook reads its files
  directly, which was verified.

## Honest limitations

* **Colab sessions end**, and on Colab Pro the tab must stay open (background execution
  is Pro+). Every reconnect repeats the install (minutes), recompiles, and loses up to
  `SAVE_EVERY` seconds of training. The run itself is safe on Drive.
* **Runtime and cost are unknown until measured.** The timing cell is the only source of
  truth; nothing here estimates them.
* **The GPU is a lottery.** The paper used an A100. L4 should work. T4 lacks bfloat16
  hardware; the notebook warns and names the flag to switch to float32.
* **The install cell is the fragile one.** JAX 0.5.0 against whatever image Colab runs
  today is the biggest unknown; the check prints the real error rather than letting it
  surface later as a CUDA error.
* **Not GPU-verified by the author of this folder.** The CUDA path is verified on your
  first run by the GPU smoke test, before real compute is spent.
* **`size12m` is not the paper's model.** Its Table 4 Pong score (−4) and the shipped
  curves are for the 200M model; the paper's scaling results say smaller models learn more
  slowly. Expect "it learns", not a match.
* **Environments are not seeded** by the stock config; the same seed plays different
  episodes on a rerun. True of the paper's runs too.
* **No checkpoint at the end of training** (wall-clock timer only); `scores.jsonl` and
  `metrics.jsonl` are complete regardless. The replay buffer on Drive grows for the whole
  run (Pong: order of a few GB).

## What was verified, and where

| what | how | result |
| :--- | :--- | :--- |
| pinned dependencies install and import together | local CPU venv, Python 3.13.14 | jax 0.5.0, numpy 2.5.3, elements 3.22.1, ninjax 3.6.3, portal 3.8.1, scope 0.7.1, optax 0.2.5, chex 0.1.90, einops 0.8.2, av 18.1.0, ale_py 0.12.1 (0.9.0 has no wheel for 3.13; the fallback path is the one exercised locally) |
| Pong with `atari100k debug` trains end to end | local CPU | exit 0; metrics, scores, checkpoint, replay chunks, scope outputs |
| resume by re-running with the same `--logdir` | local CPU | second run loads `ckpt/latest` and continues; overlapping re-logged steps handled by `lineage()` |
| `inspect_agent.py` | local CPU, `debug` and via the notebook | shapes match the config arithmetic; `z`, `ẑ` one-hot; `B×T` rollouts |
| whole notebook, `PRESET='smoke'` | executed locally with nbconvert on the CPU venv kernel (2026-09-16) | all 16 code cells ran, 0 errors: install check, clone + conditional patch, inspection, CPU smoke run, timing run, the run itself, all result cells (with "no finished episode yet" where the toy run is too short); about 3 minutes |
| the install cell on Colab; bfloat16 on the assigned GPU; the GPU smoke test; timing | **not verified** | first Colab run |

## How to run

1. Upload `hermes_dreamer_first_run_colab.ipynb` to Colab. Runtime → Change runtime type →
   **A100 GPU**.
2. Optional but recommended: set `PRESET = 'smoke'` in the settings cell and Run all once.
   Every stage finishes in minutes with the toy network; if the install cell is going to
   fail on today's Colab image, this is where you find out.
3. Set `PRESET = 'full'`, Run all, approve the Drive pop-up. Read the timing projection
   before the run cell commits you to it (interrupt if you want to change `SIZE`).
4. On a disconnect: reconnect, Run all. The run resumes.
5. To try another game: change `TASK` (any `atari100k_<game>`). To try the paper's model:
   `SIZE = ''`. Both make a new logdir; the old run is untouched.
