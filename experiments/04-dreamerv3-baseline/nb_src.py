# %% [markdown]
# # HERMES · Experiment 4 — Reproducing DreamerV3's learning-signal ablations
#
# This notebook runs **stock DreamerV3** (Hafner et al., upstream code at a pinned commit) on
# **Crafter** and **Atari100k Pong**, as a baseline and as the three *learning-signal
# ablations* from the paper (Nature 2025, Fig. 6b): no value gradients, no reward or value
# gradients, and no reconstruction gradients. It is a runner and a teaching document at the
# same time: every code cell is preceded by an explanation of what it does and why, and the
# architecture is made visible with real tensor shapes and a gradient-flow table before any
# long training starts.
#
# **Reference docs in the repo:** `ARCHITECTURE_MAP.md` (paper concepts → file and line in
# the code) and `README.md` (status, run plan, what has and has not been verified).
#
# ## Contents
#
# 1. [Settings](#settings) — every knob in one place
# 2. [Where are we running](#env) — Colab or local, GPU, Drive
# 3. [Install](#install) — pinned JAX + DreamerV3 dependencies, with a loud GPU check
# 4. [Clone and patch DreamerV3](#clone) — pinned commit + two small verified patches
# 5. [See the architecture](#inspect) — parameter tree, RSSM shapes, gradient-flow table
# 6. [The runner](#runner) — one function that launches, logs and resumes a training run
# 7. [Smoke tests](#smoke) — the `debug` config on CPU and GPU, to prove the pipeline
# 8. [Timing and cost](#timing) — measure, then project the whole run matrix
# 9. [The run matrix](#matrix) — baseline + 3 ablations × tasks × seeds, resumable
# 10. [Results](#results) — score curves, world-model losses, open-loop videos, summary table
# 11. [Aggregate and write-up figures](#aggregate) — all runs in three CSV files, four figures
#
# ## How to run, and how to resume
#
# 1. **Runtime → Change runtime type → A100 GPU** (L4 works; T4 lacks bfloat16 hardware, see
#    the GPU cell). Then **Runtime → Run all** and approve the Google Drive pop-up.
# 2. Everything durable is written to Drive under `MyDrive/HERMES/exp04-dreamerv3-baseline/`:
#    per-run logs, metrics, checkpoints, replay buffers, plots and provenance records.
# 3. **If Colab disconnects:** reconnect and **Run all** again. Finished runs are skipped,
#    the run in flight resumes from its last checkpoint (DreamerV3's own convention: same
#    command, same `--logdir`), and you lose at most `SAVE_EVERY` seconds of work.
# 4. Before the first real run, set `PRESET = 'smoke'` once and run everything: it exercises
#    every stage, including the run matrix and the plots, with a tiny network in minutes.
#    Then set `PRESET = 'full'`.

# %% [markdown]
# <a name="settings"></a>
# ## 1. Settings
#
# Everything you might want to change is here. Nothing below this cell needs editing.
#
# * `PRESET`: `'smoke'` replaces the network with DreamerV3's tiny `debug` configuration and
#   shrinks every step budget, so the *whole* notebook runs in minutes on a CPU. `'full'` is
#   the real experiment.
# * `SIZE`: DreamerV3's model-size preset (`size12m`, `size25m`, `size50m`, `size100m`; `''`
#   is the stock default, which is the 200M model the paper used on a single A100). The
#   timing section measures whatever you list in `TIMING_SIZES` and projects the cost of the
#   full matrix for each, so you can choose with numbers rather than guesses.
# * `ARMS`: the four arms of the paper's Fig. 6b. Each maps to a config block; the three
#   ablation blocks are added by the patch in section 4 and verified in section 5.
# * `UNITS_PER_HOUR`: Colab shows this in *Runtime → View resources* while a GPU runtime is
#   connected. The notebook never guesses it; leave `None` to get wall-clock projections only.

# %%
PRESET = 'full'              # 'smoke' or 'full'

DREAMER_REPO = 'https://github.com/danijar/dreamerv3.git'
DREAMER_COMMIT = 'e3f02248693a79dc8b0ebd62c93683888ddaccfe'   # 2026-05-25, "Fix Atari frame maxpooling on reset"

TASKS = {                    # name -> DreamerV3 config block and task id
    'crafter': dict(configs=['crafter'], task='crafter_reward'),
    'atari100k': dict(configs=['atari100k'], task='atari100k_pong'),
}
ARMS = {                     # name -> extra config blocks (see the patch in section 4)
    'baseline': [],
    'novalue': ['abl_novalue'],        # paper: "No value gradients"
    'norewval': ['abl_norewval'],      # paper: "No reward or value gradients"
    'norecon': ['abl_norecon'],        # paper: "No reconstruction gradients"
}
SEEDS = [0]                  # the paper used 5 seeds per run; add more once the timing is known
SIZE = 'size12m'             # model-size block for the real runs ('' = stock 200M default)
TIMING_SIZES = ['size12m', 'size50m']   # sizes to time before choosing SIZE
STEPS = {}                   # per-task override of run.steps; empty = the config block's budget
                             # (crafter 1.1e6 env steps, atari100k 1.1e5 agent steps = 4.4e5 frames)
SAVE_EVERY = 600             # seconds between checkpoints to Drive (stock: 900)
LOG_EVERY = 60               # seconds between metric writes (stock: 120)
UNITS_PER_HOUR = None        # Colab compute units per hour for this GPU, from Runtime -> View resources

# %% [markdown]
# <a name="env"></a>
# ## 2. Where are we running
#
# The notebook works both on Colab and on a local machine (the local path exists so the
# notebook logic could be dry-run on a CPU before it ever touched a GPU; the results in the
# repo README say what was verified where). On Colab, Google Drive is mounted and becomes the
# root for everything durable. The directory layout:
#
# ```
# <ROOT>/
#   runs/<task>/<arm>/seed<N>/     one DreamerV3 logdir per run: metrics.jsonl, scores.jsonl,
#                                  ckpt/, replay/, scope/, stdout.log, DONE.json
#   smoke/                          throw-away logdirs of the smoke tests
#   timing/                         short runs used only to measure speed
#   plots/                          every figure this notebook produces
#   records/                        provenance: versions, GPU, patches, inspection outputs
# ```
#
# Two facts about the machine matter for the rest: which **GPU** we got (Colab hands out
# different ones, and JAX's default compute type here is bfloat16, which only Ampere-class or
# newer GPUs run natively: A100 and L4 yes, T4 no), and which **Python** version, because the
# pinned packages need wheels for it.

# %%
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import time

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = pathlib.Path('/content/drive/MyDrive/HERMES/exp04-dreamerv3-baseline')
    REPO_DIR = pathlib.Path('/content/dreamerv3')          # code lives on local disk; nothing durable here
    CACHE_DIR = pathlib.Path('/content/jax_cache')          # compiled-program cache, local disk on purpose
else:
    ROOT = pathlib.Path('hermes_exp04_local').resolve()
    REPO_DIR = ROOT / 'dreamerv3'
    CACHE_DIR = ROOT / 'jax_cache'
PYTHON = sys.executable
for sub in ('runs', 'smoke', 'timing', 'plots', 'records'):
    (ROOT / sub).mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def sh(cmd, check=True, **kw):
    """Run a command, return its combined output as text; raise with that output on failure."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    if check and proc.returncode:
        raise RuntimeError(f'command failed ({proc.returncode}): {cmd}\n{proc.stdout}')
    return proc.stdout

GPU = None
if shutil.which('nvidia-smi'):
    line = sh(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], check=False).strip()
    if line and 'not found' not in line.lower():
        name, mem, drv = [x.strip() for x in line.split(',')]
        GPU = dict(name=name, memory=mem, driver=drv)
PLATFORM = 'cuda' if GPU else 'cpu'

ENV_INFO = dict(
    in_colab=IN_COLAB, python=platform.python_version(), interpreter=PYTHON, platform=platform.platform(),
    gpu=GPU, jax_platform=PLATFORM, preset=PRESET, root=str(ROOT), time=time.strftime('%Y-%m-%d %H:%M:%S'))
(ROOT / 'records' / 'env_info.json').write_text(json.dumps(ENV_INFO, indent=1))
print(json.dumps(ENV_INFO, indent=1))
if GPU and 'T4' in GPU['name']:
    print('\nWARNING: T4 has no bfloat16 hardware. DreamerV3 computes in bfloat16 by default. If the GPU '
          'smoke test below is very slow or fails, restart with an A100/L4, or add '
          "'--jax.compute_dtype', 'float32' to COMMON_FLAGS in the runner section.")
if not GPU:
    print('\nNo GPU found: every run below will use the CPU. Fine for PRESET="smoke", useless for "full".')

# %% [markdown]
# <a name="install"></a>
# ## 3. Install
#
# DreamerV3 needs JAX plus three small libraries by the same author (`elements` for config,
# flags, logging and checkpoints; `ninjax`, the module system the networks are written in;
# `scope`, the metric writer) and the two environments. Versions are **pinned to what was
# tested**; the repo's own `requirements.txt` only gives lower bounds for most of them.
#
# Three traps, all handled here:
#
# * `jax[cuda12]==0.5.0` is the version in the repo's `Dockerfile` (the author-tested one).
#   The pip wheel bundles its own CUDA libraries, so nothing else needs installing, but the
#   JAX that Colab pre-installs gets replaced. **This is the cell most likely to need attention
#   on a new Colab image**; the check at the end tells you exactly what went wrong instead of
#   letting a later "CUDA error" hide it (the README warns that those are usually downstream).
# * `numpy<2` in `requirements.txt` exists for DMLab/MineRL only; we do not need it and it has
#   no wheels for newer Pythons, so it is dropped. Everything here was verified with numpy 2.
# * `ale_py==0.9.0` is the Atari version the author pinned (ROMs are bundled, no licence
#   prompt). If there is no wheel for this Python we fall back to `0.12.1`, which needs a
#   one-line compatibility patch to the Atari wrapper (section 4); the patch is harmless
#   under 0.9.0 too. The version actually installed is written to `records/`.
#
# Nothing in this notebook imports JAX into the notebook process itself. Every JAX program
# runs as a **subprocess**: a crash cannot take the notebook down, and the notebook never
# grabs GPU memory that a training run then cannot get.

# %%
PINNED = [
    'jax[cuda12]==0.5.0',
    'elements==3.22.1', 'ninjax==3.6.3', 'portal==3.8.1', 'scope==0.7.1',
    'optax==0.2.5', 'chex==0.1.90', 'einops==0.8.2', 'cloudpickle==3.1.2',
    'ruamel.yaml==0.19.1', 'av==18.1.0', 'crafter==1.8.3', 'pillow', 'psutil',
]
if IN_COLAB:
    log = ROOT / 'records' / 'pip_install.log'
    out = sh([PYTHON, '-m', 'pip', 'install', '-q', *PINNED], check=False)
    log.write_text(out)
    print(out[-3000:] or 'pip: no output (ok)')
    ale = sh([PYTHON, '-m', 'pip', 'install', '-q', 'ale_py==0.9.0'], check=False)
    if 'ERROR' in ale or 'No matching distribution' in ale:
        print('ale_py==0.9.0 has no wheel for this Python; falling back to ale_py==0.12.1\n', ale[-1500:])
        sh([PYTHON, '-m', 'pip', 'install', '-q', 'ale_py==0.12.1'])
    # Colab's image can ship a newer JAX CUDA plugin (jax-cuda13-*) next to the pinned CUDA 12
    # one. Both register the "cuda" device and JAX 0.5.0 aborts on the newer one, so remove it.
    listing = sh([PYTHON, '-m', 'pip', 'list', '--format=freeze'])
    stray = [l.split('==')[0] for l in listing.splitlines() if re.match(r'jax[-_]cuda13', l, re.I)]
    if stray:
        print('removing JAX CUDA 13 packages that conflict with the pinned CUDA 12 build:', stray)
        sh([PYTHON, '-m', 'pip', 'uninstall', '-y', *stray])
    (ROOT / 'records' / 'pip_freeze.txt').write_text(sh([PYTHON, '-m', 'pip', 'freeze']))
else:
    print('Not on Colab: assuming the current interpreter already has the pinned packages.')

check = sh([PYTHON, '-c', (
    'import importlib.metadata as m, jax, jax.numpy as jnp, sys\n'
    'print("python", sys.version.split()[0])\n'
    'for p in ["jax","jaxlib","numpy","elements","ninjax","portal","scope","optax","chex","einops","ale_py","crafter","av"]:\n'
    '    print(f"{p}=={m.version(p)}")\n'
    'print("devices", jax.devices())\n'
    'x = jnp.ones((1024, 1024), jnp.bfloat16)\n'
    'print("bfloat16 matmul ok, sum =", float((x @ x).sum()), "on", (x @ x).devices())\n'
)], check=False)
print(check)
if PLATFORM == 'cuda':
    assert 'CudaDevice' in check and 'bfloat16 matmul ok' in check, (
        'JAX does not see the GPU. Read the output above: the real cause is usually a wheel/CUDA mismatch.')
(ROOT / 'records' / 'versions.txt').write_text(check)

# %% [markdown]
# <a name="clone"></a>
# ## 4. Clone and patch DreamerV3
#
# We clone the **upstream** repository at a pinned commit and apply two patches. Both are
# shown in full (they are small), written to `records/` for provenance, and verified after
# applying. `git apply` refuses to apply a patch twice, so the cell is safe to re-run.
#
# **Patch 1, `rec_grad.patch`: the missing switch for the paper's third ablation.**
# `configs.yaml` already has `reward_grad` and `repval_grad`, which stop the reward and the
# replay-value losses from sending gradient into the world-model latent (`agent.py:172` and
# `agent.py:220` use them through `sg(x, skip=flag)`, where `sg` is JAX's *stop-gradient*: the
# value passes through unchanged, but no gradient flows back). There is **no** such flag for
# the decoder: `agent.py:170-171` always feeds the latent straight into it. The paper's
# ablation "stops the task-agnostic reconstruction gradients from shaping the
# representations" (Nature version, p. 5), so the decoder keeps training on a detached
# latent. The patch adds `rec_grad: True` to the defaults, wraps the decoder input in the same
# `sg(..., skip=...)` idiom, and adds three named config blocks, one per ablation arm.
#
# This is deliberately **not** `loss_scales.rec: 0`. Zeroing the scale would also stop the
# decoder itself from learning; with the stop-gradient the decoder becomes a probe that
# tells us how much pixel detail a latent trained without reconstruction still carries.
#
# **Patch 2, `atari_ale_compat.patch`:** newer `ale_py` versions require a `str` key and a
# Python `int` in `setInt`; the wrapper passes `bytes` and a `numpy.int64`. One line.

# %% [file] rec_grad.patch

# %% [file] atari_ale_compat.patch

# %%
if not (REPO_DIR / '.git').exists():
    print(sh(['git', 'clone', DREAMER_REPO, str(REPO_DIR)]))
head = sh(['git', 'rev-parse', 'HEAD'], cwd=REPO_DIR).strip()
if head != DREAMER_COMMIT:
    sh(['git', 'checkout', '-q', DREAMER_COMMIT], cwd=REPO_DIR)
print(sh(['git', 'log', '-1', '--format=%H %ad %s', '--date=short'], cwd=REPO_DIR))

for name in ('rec_grad.patch', 'atari_ale_compat.patch'):
    patch = os.path.abspath(name)
    shutil.copy(patch, ROOT / 'records' / name)
    if sh(['git', 'apply', '--check', '--reverse', patch], check=False, cwd=REPO_DIR).strip() == '':
        print(f'{name}: already applied')
    else:
        sh(['git', 'apply', '--verbose', patch], cwd=REPO_DIR)
        print(f'{name}: applied')

agent_py = (REPO_DIR / 'dreamerv3' / 'agent.py').read_text()
configs_yaml = (REPO_DIR / 'dreamerv3' / 'configs.yaml').read_text()
atari_py = (REPO_DIR / 'embodied' / 'envs' / 'atari.py').read_text()
assert 'sg(repfeat, skip=self.config.rec_grad)' in agent_py, 'rec_grad patch not in agent.py'
assert '\n    rec_grad: True\n' in configs_yaml, 'rec_grad default missing from configs.yaml'
assert all(f'\n{b}:\n' in configs_yaml for b in ('abl_novalue', 'abl_norewval', 'abl_norecon')), 'ablation blocks missing'
assert "setInt('random_seed', int(" in atari_py, 'atari compat patch not applied'
print(sh(['git', 'diff', '--stat'], cwd=REPO_DIR))
print('patches verified')

# %% [markdown]
# <a name="inspect"></a>
# ## 5. See the architecture
#
# Before training anything, we build the agent exactly the way `main.py` does and look at it.
# The script below is written to disk and then run as a subprocess (once per arm). It prints:
#
# 1. **The parameter tree.** In JAX, a model's weights are not hidden inside objects; they
#    live in a plain dictionary of arrays (a *pytree*: any nested structure of arrays) keyed
#    by path, such as `dyn/dyngru/kernel`. The listing is the model, with nothing hidden. The
#    per-module totals give the true parameter count of the `SIZE` preset. Keys under `opt/`
#    are optimizer moments, `slowval/` is the EMA copy of the critic, and `retnorm/` etc. are
#    running statistics; they are saved in the checkpoint too but are not network weights.
# 2. **The shapes flowing through the RSSM** for one batch of `B` sequences of `T` steps:
#    image → encoder `tokens` → recurrent state `h` (`deter`) → posterior sample `z` (`stoch`,
#    one-hot over classes) → prior logits `p(z|h)` → the imagined `h` and `ẑ` of the rollouts,
#    which start from *every* posterior state and unroll `imag_length` steps on the prior
#    without ever seeing an observation. `scan` (used inside `observe` and `imagine`) is JAX's
#    compiled for-loop: the same step function applied `T` times, threading the state through.
#    Two checks run on the arrays: `z` and `ẑ` are genuinely one-hot, and the rollout count is
#    `B × T`.
# 3. **The gradient-flow table.** For every loss term, the L2 norm of the gradient it sends into
#    the encoder and into the RSSM. This is *the* verification for the ablations: a flag that
#    claims to stop a gradient must produce an exact `0` in this table, and nothing else may
#    change. Two subtleties the script handles: the reward and value heads are zero-initialised
#    (`outscale: 0.0`), so at step 0 they send no gradient regardless of any flag, and the
#    script gives those kernels a small random value first to test the *path*; and the KL
#    terms are clipped below 1 free nat, so at initialisation they typically send nothing.
#
# The run for the **baseline** is the one to read in full. The three ablation runs are only
# checked (exit code 0 means the expected zeros were found); their tables are kept in
# `records/inspect_<arm>.json`.

# %% [file] inspect_agent.py

# %%
EXPECT_ZERO = {'baseline': '', 'novalue': 'repval', 'norewval': 'rew,repval', 'norecon': 'image'}

def size_blocks():
    return ['debug'] if PRESET == 'smoke' else ([SIZE] if SIZE else [])

def inspect_arm(arm, task='crafter', show=False):
    out = ROOT / 'records' / f'inspect_{arm}.json'
    cmd = [PYTHON, 'inspect_agent.py', '--repo', str(REPO_DIR), '--expect_zero', EXPECT_ZERO[arm], '--out', str(out),
           '--configs', *TASKS[task]['configs'], *size_blocks(), *ARMS[arm], '--jax.platform', PLATFORM]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    text = proc.stdout
    (ROOT / 'records' / f'inspect_{arm}.log').write_text(text)
    if show or proc.returncode:
        print(text[text.find('PARAMETER TREE') - 80:] if 'PARAMETER TREE' in text else text)
    else:
        print(text[text.find('GRADIENT NORM'):].rstrip())
    assert proc.returncode == 0, f'inspect_agent.py failed for arm {arm}'
    return json.loads(out.read_text())

INSPECT = {'baseline': inspect_arm('baseline', show=True)}

# %% [markdown]
# The same table for the three ablation arms. Each must show an exact zero for the loss its
# flag disables (`repval`; `rew` and `repval`; `image`) and leave every other row alone.
#
# The second half of the cell checks that "leave alone" across arms. It compares within a
# tolerance rather than demanding identical numbers, because each arm is a separate process
# whose computation graph differs — a stop-gradient changes what XLA compiles, so operations
# fuse and accumulate in a different order — and DreamerV3 computes in bfloat16, which keeps
# only about three decimal digits. On a CPU these came out bit-identical; on an A100 the arms
# agree to roughly four significant figures. A gradient that genuinely leaked would go to
# zero or move by orders of magnitude, not by hundredths of a percent, so a few percent of
# tolerance separates "numerical noise" from "semantic change" with room to spare.

# %%
for arm in ('novalue', 'norewval', 'norecon'):
    if arm in INSPECT:         # already inspected in this session: re-running the check is instant
        print(f'{arm}: using the inspection already in memory')
        continue
    print(f'\n===== {arm}: --configs {" ".join(ARMS[arm])} =====')
    INSPECT[arm] = inspect_arm(arm)

TOL = 0.05                    # 5%; observed noise between arms is ~0.02% on an A100
base = INSPECT['baseline']['grad_norms']
worst_rel, worst_where = 0.0, 'nothing'
for arm in ('novalue', 'norewval', 'norecon'):
    for loss, norms in INSPECT[arm]['grad_norms'].items():
        if loss in EXPECT_ZERO[arm].split(','):
            continue           # this one is supposed to change: it is checked above, per arm
        for module, value in norms.items():
            ref = base[loss][module]
            rel = abs(value - ref) / ref if ref else abs(value - ref)
            if rel > worst_rel:
                worst_rel, worst_where = rel, f'{arm}: {loss} -> {module}'
            assert rel <= TOL, (
                f'{arm} changed the gradient of {loss} into {module} by {rel:.1%} '
                f'({value:.4e} against baseline {ref:.4e}), which it should not touch')
print(f'\nEvery ablation zeroes exactly the gradients it claims to and leaves the rest alone '
      f'(largest disagreement with the baseline: {worst_rel:.3%}, {worst_where}).')

# %% [markdown]
# <a name="runner"></a>
# ## 6. The runner
#
# One function launches `dreamerv3/main.py` as a subprocess with a given `--logdir` and set of
# config blocks, streams the important lines to the notebook, and writes **everything** to
# `<logdir>/stdout.log` with a wall-clock timestamp on every line. When the process exits
# cleanly it writes `<logdir>/DONE.json`; if it is interrupted or crashes it does not, and
# the next call with the same logdir resumes from the last checkpoint because that is what
# DreamerV3 does when it finds `ckpt/latest` (`embodied/run/train.py:83-90`). Each session's
# wall time is appended to `<logdir>/sessions.jsonl`, so the total cost of a run that
# spanned several Colab sessions is still known.
#
# Two facts worth knowing about DreamerV3's own logging, both visible in `stdout.log`:
# checkpoints and metric writes happen on **wall-clock** timers (`run.save_every`,
# `run.log_every`, in seconds), not on step counts; and a checkpoint is **not** written when
# the loop finishes, so the last checkpoint of a completed run can be up to `SAVE_EVERY`
# seconds older than its final metrics. The score curves (what the paper reports) come from
# `metrics.jsonl` and `scores.jsonl`, which are complete.

# %%
# Timers for the real runs. The smoke preset keeps the seconds-scale timers of the debug block.
TIMER_FLAGS = [] if PRESET == 'smoke' else ['--run.save_every', str(SAVE_EVERY), '--run.log_every', str(LOG_EVERY)]
INTERESTING = ('Agent Step', 'fps/policy', 'Error', 'error', 'Traceback', 'checkpoint', 'Compiling', 'Done compiling',
               'Start training', 'devices', 'Optimizer opt has', 'ALERT', 'Logdir')

def dreamer_cmd(logdir, configs, flags=()):
    # Flags come after the config blocks, so a flag always wins over a block (main.py:27-29).
    return [PYTHON, str(REPO_DIR / 'dreamerv3' / 'main.py'), '--logdir', str(logdir), '--configs', *configs, *flags]

def run_dreamer(logdir, configs, flags=(), label=None, quiet=False):
    """Run (or resume) one DreamerV3 training job. Returns a dict with exit code and wall time."""
    logdir = pathlib.Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    cmd = dreamer_cmd(logdir, configs, flags)
    (logdir / 'command.txt').write_text(' '.join(cmd) + '\n')
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONUTF8': '1', 'JAX_COMPILATION_CACHE_DIR': str(CACHE_DIR)}
    print(f'[{time.strftime("%H:%M:%S")}] {label or logdir.name}: {" ".join(cmd[1:])}')
    t0 = time.time()
    with (logdir / 'stdout.log').open('a', encoding='utf-8') as log:
        log.write(f'\n===== session start {time.strftime("%Y-%m-%d %H:%M:%S")} =====\n')
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding='utf-8', errors='replace', env=env, cwd=REPO_DIR)
        try:
            for line in proc.stdout:
                stamp = time.strftime('%H:%M:%S')
                log.write(f'[{stamp}] {line}')
                if not quiet and any(k in line for k in INTERESTING):
                    print(f'[{stamp}] {line.rstrip()[:300]}')
            code = proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            proc.wait()
            log.write('===== interrupted =====\n')
            raise
    wall = time.time() - t0
    result = dict(exit=code, wall_seconds=wall, finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    with (logdir / 'sessions.jsonl').open('a') as f:
        f.write(json.dumps(result) + '\n')
    if code == 0:
        (logdir / 'DONE.json').write_text(json.dumps(result, indent=1))
    print(f'[{time.strftime("%H:%M:%S")}] exit {code} after {wall / 60:.1f} min')
    return result

def read_jsonl(path):
    path = pathlib.Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def lineage(rows):
    """Keep only the rows that belong to the final agent's history.

    When a run resumes from a checkpoint older than its last metric write, it re-logs steps
    that are already in the file. A restart is visible as a step number smaller than the one
    before it; everything logged earlier at or beyond that step belonged to a discarded
    branch and is dropped."""
    out = []
    for row in rows:
        while out and out[-1]['step'] > row['step']:
            out.pop()
        out.append(row)
    return out

def last_step(logdir):
    rows = lineage(read_jsonl(pathlib.Path(logdir) / 'metrics.jsonl'))
    return rows[-1]['step'] if rows else 0

# %% [markdown]
# <a name="smoke"></a>
# ## 7. Smoke tests
#
# DreamerV3's `debug` config block shrinks every network to a handful of units, the batch to
# 8×10, and every timer to seconds. It **does not learn anything** and it runs on the **CPU**
# (`configs.yaml`, the `debug` block sets `jax.platform: cpu`), so it proves the pipeline
# (environment, replay, training loop, checkpoints, metrics) without proving the GPU path.
# Three runs, each throw-away and written to local disk:
#
# 1. Crafter with `debug` on CPU.
# 2. Atari100k Pong with `debug` on CPU (this is where the ALE compatibility patch matters).
# 3. Crafter with `debug` but forced back onto the GPU: same tiny network, real CUDA path,
#    bfloat16 arithmetic, and enough updates to pass **120**, because DreamerV3 starts its JAX
#    profiler at update 100 and stops it at 120 (`embodied/jax/agent.py:302-307`), a path that
#    cannot be turned off from the config and must be known to work.
#
# After each run the cell checks what should exist: `metrics.jsonl`, `scores.jsonl`, a
# checkpoint with a `done` marker, replay chunks, the `scope/` folder. Runs that already
# finished are not repeated, so re-running this cell is quick.
#
# **How the profiler is checked, and why not with a metric.** `train/opt/updates` in
# `metrics.jsonl` looks like the update count but is not: DreamerV3 averages every training
# metric over each log window before writing it, so the logged value is the window's *mean*
# counter and trails the true count, and nothing is written after the last window. (Measured
# locally: largest logged value 131, true count 135.) The direct evidence is what the
# profiler itself leaves behind: the lines `Start JAX profiler` and `Stop JAX profiler` in the
# run's `stdout.log`, and a `.xplane.pb` trace file under `plugins/profile/`.

# %%
def profiler_ran(logdir):
    """True if the run started and stopped DreamerV3's JAX profiler and wrote its trace."""
    logdir = pathlib.Path(logdir)
    log = (logdir / 'stdout.log').read_text(errors='replace') if (logdir / 'stdout.log').exists() else ''
    return 'Start JAX profiler' in log and 'Stop JAX profiler' in log and bool(list(logdir.rglob('*.xplane.pb')))

def checkpoint_counters(logdir):
    """Update/action counters as of the last checkpoint (which can trail the end of a run)."""
    import pickle
    ckpt = pathlib.Path(logdir) / 'ckpt'
    return pickle.loads((ckpt / (ckpt / 'latest').read_text().strip() / 'agent.pkl').read_bytes())['counters']

def check_logdir(logdir, min_step):
    logdir = pathlib.Path(logdir)
    for f in ('metrics.jsonl', 'scores.jsonl', 'config.yaml', 'ckpt/latest', 'DONE.json'):
        assert (logdir / f).exists(), f'missing {f} in {logdir}'
    latest = (logdir / 'ckpt' / (logdir / 'ckpt' / 'latest').read_text().strip())
    assert (latest / 'done').exists(), f'checkpoint {latest} has no done marker'
    assert (latest / 'agent.pkl').exists() and (latest / 'replay.pkl').exists(), 'checkpoint incomplete'
    assert list((logdir / 'replay').glob('*.npz')), 'no replay chunks were saved'
    step = last_step(logdir)
    assert step >= min_step, f'only reached step {step} < {min_step}'
    keys = sorted({k for r in read_jsonl(logdir / 'metrics.jsonl') for k in r})
    print(f'ok: {logdir.name}: reached step {step}, {len(keys)} metric keys, '
          f'{len(list((logdir / "replay").glob("*.npz")))} replay chunks, checkpoint {latest.name}')
    return keys

smoke = ROOT / 'smoke'
# Checkpoint and log every 5 s so that a one-minute run exercises both paths at least once.
SMOKE_FLAGS = ['--run.save_every', '5', '--run.log_every', '5']

for name, block in (('crafter_cpu', 'crafter'), ('atari_cpu', 'atari100k')):
    if (smoke / name / 'DONE.json').exists():
        print(f'{name}: already finished, not repeated')
    else:
        r = run_dreamer(smoke / name, [block, 'debug'], ['--run.steps', '600', *SMOKE_FLAGS], quiet=True)
        assert r['exit'] == 0, f'{block} debug run failed; read smoke/{name}/stdout.log'
    keys = check_logdir(smoke / name, 500)
print('world-model loss keys:', [k for k in keys if k.startswith('train/loss/')])

if PLATFORM == 'cuda':
    gpu = smoke / 'crafter_gpu'
    if profiler_ran(gpu):
        print('crafter_gpu: already ran on the GPU through the profiler window, not repeated')
    else:
        if gpu.exists():
            # An earlier attempt stopped short. It is not resumed: a resume that restarts between
            # updates 100 and 120 makes DreamerV3 stop a profiler this process never started,
            # which raises "No profile started". It is renamed, not deleted, and the test restarts.
            aside = gpu.with_name(f'crafter_gpu_incomplete_{time.strftime("%Y%m%d_%H%M%S")}')
            gpu.rename(aside)
            print(f'crafter_gpu: earlier attempt did not reach the profiler window; moved to {aside.name}')
        # debug: train_ratio 8 over a batch of 8x10 = 80 steps is one update per 10 env steps
        # after an 80-step warm-up, so 2000 steps is about 190 updates, well past 120.
        r = run_dreamer(gpu, ['crafter', 'debug'], ['--jax.platform', 'cuda', '--run.steps', '2000', *SMOKE_FLAGS])
        assert r['exit'] == 0, 'GPU debug run failed; scroll up: the first error is the real one'
    check_logdir(gpu, 1400)
    assert profiler_ran(gpu), (
        'the GPU run never started and stopped the JAX profiler (no "Stop JAX profiler" line in '
        'smoke/crafter_gpu/stdout.log, or no .xplane.pb trace), so updates 100-120 were not reached')
    counters = checkpoint_counters(gpu)
    print(f'GPU path ok: the profiler started and stopped, trace written; '
          f'{counters["updates"]} gradient updates as of the last checkpoint')
else:
    print('no GPU: the CUDA smoke test is skipped')

# %% [markdown]
# <a name="timing"></a>
# ## 8. Timing and cost
#
# Never commit to a long run on an estimate. For each task and each candidate size, this
# runs the **real configuration** for a short budget into a logdir on Drive (so Drive I/O is
# part of what is measured), long enough to get past the replay warm-up (training starts once
# the buffer holds `batch_size × batch_length = 1024` steps) and past compilation, and then
# reads the speed DreamerV3 logs itself:
#
# * `fps/policy`: environment steps per second of wall time in the last log window, with
#   training interleaved. This is the number that turns a step budget into hours.
# * `fps/train`: replayed steps trained per second; divided by `batch_size × batch_length` it
#   is gradient updates per second.
#
# Cost of a run = `steps / fps_policy`. The projection below multiplies that by the number of
# runs in the matrix and, if `UNITS_PER_HOUR` is set, by the compute-unit rate. Gradient
# updates per environment step are fixed by the config (`train_ratio / (batch_size ×
# batch_length)`: 0.5 for Crafter, 0.25 for Atari100k), so Crafter's 1.1M steps cost about
# 550K updates and Atari100k's 110K steps about 27.5K; Crafter dominates.
#
# In `smoke` mode the timing runs use the `debug` network and the numbers mean nothing; the
# cell still runs to prove the code path.

# %%
TIMING_STEPS = {'crafter': 2200, 'atari100k': 3000}       # ~600 and ~500 updates after warm-up
if PRESET == 'smoke':
    TIMING_STEPS = {t: 300 for t in TASKS}

def run_steps(task):
    if PRESET == 'smoke':
        return 400
    return int(STEPS.get(task, {'crafter': 1.1e6, 'atari100k': 1.1e5}[task]))

TIMING = {}
for size in (['debug'] if PRESET == 'smoke' else TIMING_SIZES):
    for task, spec in TASKS.items():
        logdir = ROOT / 'timing' / f'{task}_{size or "default"}'   # 'debug' in smoke mode, so sizes never mix
        if not (logdir / 'DONE.json').exists():
            flags = ['--jax.platform', PLATFORM, '--run.steps', str(TIMING_STEPS[task])]
            if PRESET != 'smoke':      # log often, and checkpoint to Drive at least once, to measure both
                flags += ['--run.log_every', '30', '--run.save_every', '120']
            r = run_dreamer(logdir, [*spec['configs'], *([size] if size else [])], flags,
                            label=f'timing {task} {size or "default"}')
            assert r['exit'] == 0, f'timing run failed: {logdir}'
        rows = [r for r in lineage(read_jsonl(logdir / 'metrics.jsonl')) if 'fps/policy' in r and r.get('fps/train', 0) > 0]
        assert rows, f'no log window with training in {logdir}; increase TIMING_STEPS'
        last = rows[-1]
        wall = sum(s['wall_seconds'] for s in read_jsonl(logdir / 'sessions.jsonl'))
        params = next((r['train/opt/param_count'] for r in rows if 'train/opt/param_count' in r), float('nan'))
        TIMING[(task, size)] = dict(fps_policy=last['fps/policy'], fps_train=last['fps/train'],
                                    updates_per_s=last['fps/train'] / 1024, wall_seconds=wall, params=params)
        print(f'{task:<10} {size or "default":<8} params {params:>14,.0f}  env steps/s {last["fps/policy"]:8.2f}  '
              f'updates/s {last["fps/train"] / 1024:6.2f}   (whole timing run incl. compile: {wall / 60:.1f} min)')
(ROOT / 'records' / 'timing.json').write_text(json.dumps({f'{t}|{s}': v for (t, s), v in TIMING.items()}, indent=1))

n_runs = len(ARMS) * len(SEEDS)
print(f'\nProjection for the full matrix: {len(ARMS)} arms x {len(SEEDS)} seed(s) = {n_runs} runs per task')
print(f'{"task":<10} {"size":<8} {"steps/run":>10} {"hours/run":>10} {"hours/task":>11} {"units/task":>11}')
totals = {}
for (task, size), t in TIMING.items():
    hours = run_steps(task) / t['fps_policy'] / 3600
    units = hours * n_runs * UNITS_PER_HOUR if UNITS_PER_HOUR else float('nan')
    totals[size] = totals.get(size, 0) + hours * n_runs
    print(f'{task:<10} {size or "default":<8} {run_steps(task):>10,} {hours:>10.1f} {hours * n_runs:>11.1f} {units:>11.0f}')
for size, hours in totals.items():
    units = f'{hours * UNITS_PER_HOUR:.0f} compute units' if UNITS_PER_HOUR else 'set UNITS_PER_HOUR for units'
    print(f'TOTAL {size or "default":<8} {hours:6.1f} hours of this GPU  ->  {units}')
print('\nThese are extrapolations from a few hundred updates; compile time and Drive checkpoints add a little on top.')

# %% [markdown]
# <a name="matrix"></a>
# ## 9. The run matrix
#
# Every combination of task × arm × seed gets its own logdir on Drive. The cell first prints
# the status of every run (done, in progress with its last step, or not started), then works
# through the incomplete ones **in order, one at a time** (one GPU, one process). Re-running
# the cell after a disconnect is the resume mechanism: finished runs are skipped because of
# `DONE.json`, the interrupted one continues from `ckpt/latest`.
#
# The command for every run is the stock one plus: the size block, the arm's ablation block,
# `--seed`, and the checkpoint/log timers from the settings. Nothing else. The full command
# line is saved next to each run in `command.txt`, and DreamerV3 itself saves the resolved
# configuration as `config.yaml`, so every run is exactly reproducible from its folder.
#
# **Seeds.** DreamerV3 derives every random draw (network init, sampling, replay) from
# `--seed`, but the environments are not seeded by the stock config (`main.py:241-242` only
# seeds suites with `use_seed`), so episode content still varies between runs. The paper used
# 5 seeds and reported the mean and one standard deviation; the results section plots every
# seed as its own line and never a mean alone.

# %%
def run_logdir(task, arm, seed):
    # The size block is part of the path ('debug' in smoke mode): a checkpoint can only
    # resume a network of the same size, so different sizes must never share a logdir.
    return ROOT / 'runs' / task / ('_'.join(size_blocks()) or 'default') / arm / f'seed{seed}'

def run_flags(task, seed):
    return ['--jax.platform', PLATFORM, '--seed', str(seed), '--run.steps', str(run_steps(task)), *TIMER_FLAGS]

MATRIX = [(task, arm, seed) for task in TASKS for arm in ARMS for seed in SEEDS]
print(f'{"task":<10} {"arm":<9} {"seed":>4} {"status":<12} {"last step":>10} {"target":>10}')
for task, arm, seed in MATRIX:
    d = run_logdir(task, arm, seed)
    status = 'done' if (d / 'DONE.json').exists() else ('in progress' if (d / 'metrics.jsonl').exists() else 'not started')
    print(f'{task:<10} {arm:<9} {seed:>4} {status:<12} {last_step(d):>10,} {run_steps(task):>10,}')

# %%
for task, arm, seed in MATRIX:
    d = run_logdir(task, arm, seed)
    if (d / 'DONE.json').exists():
        continue
    print(f'\n===== {task} / {arm} / seed {seed}  (resuming from step {last_step(d):,})' if last_step(d) else
          f'\n===== {task} / {arm} / seed {seed}')
    r = run_dreamer(d, [*TASKS[task]['configs'], *size_blocks(), *ARMS[arm]], run_flags(task, seed), label=f'{task}/{arm}/seed{seed}')
    if r['exit'] != 0:
        raise RuntimeError(f'{task}/{arm}/seed{seed} exited with {r["exit"]}; see {d / "stdout.log"}. '
                           'Re-run this cell to resume from the last checkpoint once the cause is fixed.')
print('\nAll runs in the matrix are complete.')

# %% [markdown]
# <a name="results"></a>
# ## 10. Results
#
# Everything below reads the files DreamerV3 wrote; no number is typed in by hand. Three
# sources per run:
#
# * `scores.jsonl`: one line per finished episode, `{step, episode/score}`. This is what the
#   paper plots (training-episode return against environment steps). For Atari the logged step
#   is already in **frames** (the logger multiplies by the action repeat of 4), which matches
#   the 400K-frame Atari100k budget and the reference curves.
# * `metrics.jsonl`: every scalar, written every `LOG_EVERY` seconds: the world-model losses
#   (`train/loss/image`, `rew`, `con`, `dyn`, `rep`), the actor/critic losses, speed, replay
#   and memory statistics.
# * `scope/`: the non-scalar outputs, read with `scope.Reader`: the open-loop video
#   (`report/openloop/image`), the parameter summary, the timer breakdown and the raw
#   `nvidia-smi` output. The `scope` web viewer needs a server and a port; parsing the files
#   directly is simpler and works anywhere, so that is what we do.
#
# **Reference data.** The repo ships the paper's own Atari100k curves in
# `scores/atari100k-dreamerv3.json.gz` (5 seeds per game, x in frames) and PPO's; they are
# overlaid on the Pong plot. For Crafter no per-run curve is shipped, so Crafter compares arms
# against each other only. The paper's Atari100k scores (Supplementary Table 4: DreamerV3 on
# Pong −4, random −21, human 15) come from its default **200M** model over **5 seeds**. Its
# learning-signal ablation (Supplementary Figure 9) covers 14 tasks, among them Crafter to 5M
# steps and three Atari games (Atlantis, Breakout, Montezuma's Revenge) to 20M steps, but not
# Pong and not Atari100k; the ablation section does not state its model size or seed count.
# So the expectation for a smaller model is the *ordering* of the arms, not the numbers.

# %%
import gzip

import matplotlib.pyplot as plt
import numpy as np

COLORS = {'baseline': '#0b0b0b', 'novalue': '#2a78d6', 'norewval': '#1baf7a', 'norecon': '#eb6834'}   # colour-blind checked
LABELS = {'baseline': 'Dreamer (baseline)', 'novalue': 'No value gradients',
          'norewval': 'No reward or value gradients', 'norecon': 'No reconstruction gradients'}
PLOTS = ROOT / 'plots'

def load_runs():
    runs = {}
    for task, arm, seed in MATRIX:
        d = run_logdir(task, arm, seed)
        scores = lineage(read_jsonl(d / 'scores.jsonl'))
        metrics = lineage(read_jsonl(d / 'metrics.jsonl'))
        if scores or metrics:
            runs[(task, arm, seed)] = dict(scores=scores, metrics=metrics, done=(d / 'DONE.json').exists(), logdir=d)
    return runs

def smooth(y, k):
    """Trailing mean over the last k points, drawn only once the window is full; the raw
    points are always drawn as well. A centred mean padded with zeros would drag both ends
    of every curve toward zero, which on Pong made a flat curve at -21 appear to rise."""
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    out = np.full(len(y), np.nan)
    c = np.cumsum(np.insert(y, 0, 0.0))
    out[k - 1:] = (c[k:] - c[:-k]) / k
    return out

def reference_curves(game='pong'):
    out = {}
    for method in ('dreamerv3', 'ppo_fixhp'):
        path = REPO_DIR / 'scores' / f'atari100k-{method}.json.gz'
        if path.exists():
            runs = [r for r in json.load(gzip.open(path)) if r['task'] == f'atari_{game}']
            out[method] = [(np.asarray(r['xs']), np.asarray(r['ys'])) for r in runs]
    return out

RUNS = load_runs()
print(f'{len(RUNS)} runs with data; complete: {sum(r["done"] for r in RUNS.values())}')

# %% [markdown]
# **Score curves.** One panel per task; one thin line per seed, the running mean over episodes
# drawn on top; the four arms in the paper's ordering. For Pong the paper's five DreamerV3
# seeds and five PPO seeds are the grey bands. Incomplete runs are drawn as far as they got and
# marked in the legend.

# %%
def plot_scores(runs, window=20):
    fig, axes = plt.subplots(1, len(TASKS), figsize=(7 * len(TASKS), 4.5), squeeze=False)
    for ax, task in zip(axes[0], TASKS):
        if task == 'atari100k':
            for method, curves in reference_curves(TASKS[task]['task'].split('_', 1)[1]).items():
                for i, (xs, ys) in enumerate(curves):
                    ax.plot(xs, ys, color='0.75' if method == 'dreamerv3' else '0.88', lw=1,
                            label=f'paper {"DreamerV3" if method == "dreamerv3" else "PPO"}, 5 seeds' if i == 0 else None)
        for arm in ARMS:
            for seed in SEEDS:
                run = runs.get((task, arm, seed))
                if not run or not run['scores']:
                    continue
                x = np.array([r['step'] for r in run['scores']])
                y = np.array([r['episode/score'] for r in run['scores']])
                tag = '' if run['done'] else f' (incomplete, step {x[-1]:,})'
                ax.plot(x, y, color=COLORS[arm], alpha=0.25, lw=0.8)
                ax.plot(x, smooth(y, window), color=COLORS[arm], lw=1.8, label=f'{LABELS[arm]}, seed {seed}{tag}')
        ax.set_title(f'{task} ({TASKS[task]["task"]}), size: {" ".join(size_blocks()) or "default"}')
        ax.set_xlabel('environment steps' + (' (frames)' if task == 'atari100k' else ''))
        ax.set_ylabel('episode return')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / 'scores.png', dpi=150)
    return fig

plot_scores(RUNS);

# %% [markdown]
# **World-model losses.** The direct view of what each ablation does to the world model.
# Under `norecon`, the image loss is the loss of a decoder that trains on a latent it is not
# allowed to shape: a probe of how much pixel detail the latent keeps anyway. Under
# `norewval`, the reward loss is likewise a probe of how predictable reward is from a latent
# shaped only by pixels and continuation. `dyn` and `rep` sit at the 1-nat free-bits floor
# whenever the KL is below it. `train/opt/loss` is the sum of everything with its scales, and
# is only shown to make the point that it is not a world-model metric.

# %%
LOSS_KEYS = ['train/loss/image', 'train/loss/rew', 'train/loss/con', 'train/loss/dyn', 'train/loss/rep', 'train/opt/loss']

def plot_losses(runs):
    fig, axes = plt.subplots(len(TASKS), len(LOSS_KEYS), figsize=(3.3 * len(LOSS_KEYS), 3.4 * len(TASKS)), squeeze=False)
    for row, task in zip(axes, TASKS):
        for ax, key in zip(row, LOSS_KEYS):
            for arm in ARMS:
                for seed in SEEDS:
                    run = runs.get((task, arm, seed))
                    if not run:
                        continue
                    pts = [(r['step'], r[key]) for r in run['metrics'] if key in r]
                    if pts:
                        x, y = map(np.array, zip(*pts))
                        ax.plot(x, y, color=COLORS[arm], lw=1.2, alpha=0.9 if seed == SEEDS[0] else 0.5,
                                label=LABELS[arm] if seed == SEEDS[0] else None)
            ax.set_title(f'{task}: {key.split("/", 1)[1]}', fontsize=10)
            ax.set_xlabel('environment steps')
            ax.grid(alpha=0.3)
            if key == 'train/loss/image':
                ax.set_yscale('log')
        row[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / 'world_model_losses.png', dpi=150)
    return fig

plot_losses(RUNS);

# %% [markdown]
# **Open-loop predictions.** Every `report_every` seconds DreamerV3 takes six replayed
# sequences, feeds the first half to the model (green border) and lets it **imagine** the
# second half from the prior alone (red border), then decodes both halves. Each column of the
# video is one sequence; the three rows are the true frames, the decoded frames, and their
# difference. The frames below are taken from the latest report of each run: the last observed
# step, the first imagined step, and the last imagined step. For `norecon` the decoder never
# shaped the latent, so whatever it can still draw is what the latent kept without being asked.

# %%
def openloop_frames(logdir):
    import av
    import scope
    reader = scope.Reader(str(logdir))
    if 'report/openloop/image' not in reader.keys():
        return None, None
    steps, files = reader['report/openloop/image']
    container = av.open(str(files[-1]))
    frames = np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])
    container.close()
    return int(steps[-1]), frames

def plot_openloop(runs, seed=None):
    seed = SEEDS[0] if seed is None else seed
    rows = [(task, arm) for task in TASKS for arm in ARMS if (task, arm, seed) in runs]
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 2.4 * len(rows)), squeeze=False)
    for (task, arm), row in zip(rows, axes):
        step, frames = openloop_frames(runs[(task, arm, seed)]['logdir'])
        if frames is None:
            for ax in row:
                ax.axis('off')
            row[0].set_title(f'{task} / {arm}: no report yet', fontsize=9, loc='left')
            continue
        n = len(frames)                      # the report appends min(10, T) blank frames to the video
        T = n - 10 if n >= 20 else n // 2
        picks = {'last observed': T // 2 - 1, 'first imagined': T // 2, 'last imagined': T - 1}
        for ax, (name, t) in zip(row, picks.items()):
            ax.imshow(frames[t])
            ax.set_title(f'{task} / {arm} @ step {step:,}: {name} (t={t})', fontsize=8)
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(PLOTS / f'openloop_seed{seed}.png', dpi=150)
    return fig

plot_openloop(RUNS);

# %% [markdown]
# **Summary table.** Per run: final score (mean return of the episodes in the last 10% of the
# step budget), mean return over the whole run (the area under the curve, normalised), the
# final world-model losses, wall time across all sessions, and whether the run is complete.
# The table is also written to `records/summary.csv`. With one seed, differences between arms
# that are smaller than the spread between the paper's five reference seeds mean nothing.

# %%
import pandas as pd

rows = []
for (task, arm, seed), run in sorted(RUNS.items()):
    target = run_steps(task) * (4 if task == 'atari100k' else 1)
    sc = np.array([(r['step'], r['episode/score']) for r in run['scores']]) if run['scores'] else np.zeros((0, 2))
    final = sc[sc[:, 0] >= 0.9 * target, 1] if len(sc) else np.array([])
    last = next((r for r in reversed(run['metrics']) if 'train/loss/image' in r), {})
    sessions = read_jsonl(run['logdir'] / 'sessions.jsonl')
    rows.append(dict(
        task=task, arm=arm, seed=seed, done=run['done'], episodes=len(sc),
        last_step=last_step(run['logdir']),
        final_score=final.mean() if len(final) else np.nan, n_final_episodes=len(final),
        mean_score=sc[:, 1].mean() if len(sc) else np.nan,
        loss_image=last.get('train/loss/image', np.nan), loss_rew=last.get('train/loss/rew', np.nan),
        loss_dyn=last.get('train/loss/dyn', np.nan), loss_rep=last.get('train/loss/rep', np.nan),
        wall_hours=sum(s['wall_seconds'] for s in sessions) / 3600, sessions=len(sessions)))
summary = pd.DataFrame(rows)
summary.to_csv(ROOT / 'records' / 'summary.csv', index=False)
pd.set_option('display.width', 200)
print(summary.round(3).to_string(index=False))
print(f'\nplots: {sorted(p.name for p in PLOTS.glob("*.png"))}\nroot:  {ROOT}')

# %% [markdown]
# <a name="aggregate"></a>
# ## 11. One copy of everything, and figures for a write-up
#
# The cells above read each run's files and draw quick diagnostic plots. This section writes
# every run into three files under `records/aggregate/` on Drive, so later analysis needs
# neither this notebook's state nor a GPU:
#
# * `scores_all.csv`: every finished episode of every run, one row each, labelled with
#   `task`, `size`, `arm` and `seed`. Steps that were logged again after a resume are already
#   removed (`lineage()`).
# * `metrics_all.csv`: every logged scalar of every run, labelled the same way.
# * `summary.csv`: one row per run. `final_score` is the mean return over the episodes in the
#   last 10% of the budget, with its standard deviation *across those episodes* (one seed, so
#   not across seeds) and their count. The `loss_*` columns are means over the last 10% of
#   logged steps, steadier than the single last value in the table above.
#
# `wall_hours` and `sessions` count only sessions that ended with the run cell returning or
# being interrupted. A session that Colab killed leaves no record, so both are lower bounds.

# %%
AGG = ROOT / 'records' / 'aggregate'
AGG.mkdir(parents=True, exist_ok=True)
SIZE_NAME = '_'.join(size_blocks()) or 'default'

score_frames, metric_frames, agg_rows = [], [], []
for (task, arm, seed), run in sorted(RUNS.items()):
    tag = dict(task=task, size=SIZE_NAME, arm=arm, seed=seed)
    scores, metrics = pd.DataFrame(run['scores']), pd.DataFrame(run['metrics'])
    score_frames.append(scores.assign(**tag))
    metric_frames.append(metrics.assign(**tag))
    budget = run_steps(task) * (4 if task == 'atari100k' else 1)   # scores.jsonl logs Atari in frames
    final = scores[scores.step >= 0.9 * budget]['episode/score'] if len(scores) else pd.Series(dtype=float)
    sessions = read_jsonl(run['logdir'] / 'sessions.jsonl')
    row = dict(**tag, done=run['done'], last_step=int(metrics.step.max()) if len(metrics) else 0,
               episodes=len(scores), final_score=final.mean(), final_score_std=final.std(),
               n_final_episodes=len(final), mean_score=scores['episode/score'].mean() if len(scores) else np.nan,
               wall_hours=sum(s['wall_seconds'] for s in sessions) / 3600, sessions=len(sessions))
    for key in ('image', 'rew', 'con', 'dyn', 'rep'):
        col = f'train/loss/{key}'
        if col in metrics:
            row[f'loss_{key}'] = metrics[metrics.step >= 0.9 * metrics.step.max()][col].dropna().mean()
    agg_rows.append(row)

pd.concat(score_frames, ignore_index=True).to_csv(AGG / 'scores_all.csv', index=False)
pd.concat(metric_frames, ignore_index=True).to_csv(AGG / 'metrics_all.csv', index=False)
pd.DataFrame(agg_rows).to_csv(AGG / 'summary.csv', index=False)
print(pd.DataFrame(agg_rows).round(3).to_string(index=False))
print(f'\nwritten to {AGG}: scores_all.csv, metrics_all.csv, summary.csv')

# %% [markdown]
# **Figures for a write-up.** Four figures, drawn only from the three files above and saved to
# `plots/writeup/` at 200 dpi:
#
# 1. `learning_curves.png`: return over training, as a trailing mean over episodes.
# 2. `final_scores.png`: each arm's final score with its spread and episode count.
# 3. `world_model_losses.png`: the KL between posterior and prior, with the free-nats floor
#    marked, and the decoder loss on a log scale. In the no-reconstruction arm that decoder
#    trains on a detached latent, so its loss measures how much pixel detail the latent holds.
# 4. `kl_vs_return.png`: final KL against final score, one point per arm.
#
# The line ends carry direct labels, the colours were checked for colour-blind separation,
# and the free-nats value is read from the baseline run's own `config.yaml`. This cell needs
# only `ROOT`: to redraw the figures later on a **CPU runtime**, run the Settings and "Where
# are we running" cells, then this one.

# %%
import numpy as np, pandas as pd, matplotlib.pyplot as plt, yaml
from matplotlib.ticker import FuncFormatter

AGG = ROOT / 'records' / 'aggregate'
FIG_DIR = ROOT / 'plots' / 'writeup'
FIG_DIR.mkdir(parents=True, exist_ok=True)

scores = pd.read_csv(AGG / 'scores_all.csv').rename(columns={'episode/score': 'score'})
metrics = pd.read_csv(AGG / 'metrics_all.csv', low_memory=False)
agg_summary = pd.read_csv(AGG / 'summary.csv')

FIG_TASKS = {'crafter': 'Crafter', 'atari100k': 'Atari100k Pong'}
FIG_XLABEL = {'crafter': 'environment steps', 'atari100k': 'environment frames'}
FIG_ARMS = ['baseline', 'novalue', 'norewval', 'norecon']
FIG_LABEL = {'baseline': 'Baseline (full DreamerV3)', 'novalue': 'No value gradients',
             'norewval': 'No reward or value gradients', 'norecon': 'No reconstruction gradients'}
FIG_SHORT = {'baseline': 'baseline', 'novalue': 'no value', 'norewval': 'no rew+val', 'norecon': 'no recon'}
FIG_COLOR = {'baseline': '#0b0b0b', 'novalue': '#2a78d6', 'norewval': '#1baf7a', 'norecon': '#eb6834'}
EP_WINDOW = {'crafter': 100, 'atari100k': 10}           # episodes per trailing mean (Pong episodes are long)
METRIC_WINDOW = {'crafter': 20, 'atari100k': 3}         # log rows per trailing mean (one row per minute; Pong runs ~40 min)
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.titlecolor': INK, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2,
    'font.size': 10, 'legend.frameon': False})
THOUSANDS = FuncFormatter(lambda v, _: f'{v / 1e3:,.0f}k' if v else '0')

def trailing(df, col, window):
    """Trailing mean, drawn only once the window is full. Unlike a centred mean with zero
    padding, it cannot bend the ends of a curve (the bug in the notebook's first plots)."""
    df = df.dropna(subset=[col]).sort_values('step')
    window = max(1, min(window, len(df)))          # smoke runs are shorter than one full window
    return df.step.to_numpy(), df[col].rolling(window, min_periods=window).mean().to_numpy()

def label_ends(ax, ends, fmt='{:.1f}'):
    """Name and last value just right of each line's end, nudged apart so labels never overlap.
    Call it last. Positions come from the final axis limits, so log axes are handled too."""
    for x, y, arm in ends:
        ax.plot(x, y, 'o', ms=6, color=FIG_COLOR[arm], mec=SURF, mew=2, zorder=5)
    lo, hi = ax.get_ylim()                     # reading the limits forces autoscaling to finish
    f = np.log10 if ax.get_yscale() == 'log' else (lambda v: v)
    placed = []
    for x, y, arm in sorted(ends, key=lambda e: e[1]):
        fy = (f(y) - f(lo)) / (f(hi) - f(lo))  # position as a fraction of the axis height
        fy = max(fy, placed[-1] + 0.075) if placed else fy
        placed.append(fy)
        ax.annotate(f'{FIG_SHORT[arm]}  {fmt.format(y)}', (1.02, fy), xycoords='axes fraction',
                    color=INK2, fontsize=9, va='center', annotation_clip=False)

def shared_legend(fig, y=1.07):
    handles = [plt.Line2D([], [], color=FIG_COLOR[a], lw=2) for a in FIG_ARMS]
    fig.legend(handles, [FIG_LABEL[a] for a in FIG_ARMS], loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, y), labelcolor=INK2)

def save_fig(fig, name):
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches='tight')
    plt.show()
    print('saved', FIG_DIR / name)

def find_key(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            found = v if k == key else find_key(v, key)
            if found is not None:
                return found
    return None

try:   # read the KL floor from the run's own config instead of typing it in
    cfg_path = next(ROOT.glob('runs/crafter/*/baseline/seed*/config.yaml'))
    FREE_NATS = find_key(yaml.safe_load(cfg_path.read_text()), 'free_nats')
except Exception as e:
    FREE_NATS = None
    print('could not read free_nats from config.yaml:', repr(e))
print('free_nats from config.yaml:', FREE_NATS)

# 1. Learning curves ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
for ax, (task, name) in zip(axes, FIG_TASKS.items()):
    ends = []
    for arm in FIG_ARMS:
        x, y = trailing(scores[(scores.task == task) & (scores.arm == arm)], 'score', EP_WINDOW[task])
        ax.plot(x, y, color=FIG_COLOR[arm])
        ok = ~np.isnan(y)
        if ok.any():
            ends.append((x[ok][-1], y[ok][-1], arm))
    ax.set_title(name, loc='left')
    ax.set_xlabel(FIG_XLABEL[task])
    ax.set_ylabel(f'episode return ({EP_WINDOW[task]}-episode trailing mean)')
    ax.xaxis.set_major_formatter(THOUSANDS)
    label_ends(ax, ends)
fig.subplots_adjust(wspace=0.45)
shared_legend(fig)
save_fig(fig, 'learning_curves.png')

# 2. Final performance ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
for ax, (task, name) in zip(axes, FIG_TASKS.items()):
    s = agg_summary[agg_summary.task == task].set_index('arm').loc[FIG_ARMS[::-1]]
    ypos = np.arange(len(s))
    for yi, (arm, r) in zip(ypos, s.iterrows()):
        ax.plot([r.final_score - r.final_score_std, r.final_score + r.final_score_std], [yi, yi],
                color=FIG_COLOR[arm], lw=2, solid_capstyle='round', alpha=0.45)
        ax.plot(r.final_score, yi, 'o', ms=9, color=FIG_COLOR[arm], mec=SURF, mew=2)
        ax.annotate(f'{r.final_score:.1f}  (n={int(r.n_final_episodes)} ep.)', (r.final_score + r.final_score_std, yi),
                    xytext=(8, 0), textcoords='offset points', ha='left', va='center', fontsize=9, color=INK2)
    ax.set_yticks(ypos, [FIG_LABEL[a] for a in s.index])
    ax.set_ylim(-0.6, len(s) - 0.2)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.3 * np.ptp(ax.get_xlim()))   # room for the labels
    ax.grid(axis='y', visible=False)
    ax.set_title(name, loc='left')
    ax.set_xlabel('mean episode return, last 10% of training')
fig.text(0.01, -0.08, 'Dot: mean over the episodes in the last 10% of the budget. Line: ±1 std across those '
         'episodes (one seed, so this is not seed-to-seed variance).', fontsize=8.5, color=INK2)
fig.subplots_adjust(wspace=0.9)
save_fig(fig, 'final_scores.png')

# 3. What the world model learned: KL (information taken from each frame) and decoder probe ----
fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
for row, (task, name) in enumerate(FIG_TASKS.items()):
    for col, (key, title, log) in enumerate([
            ('train/loss/dyn', 'KL between posterior and prior (nats)', False),
            ('train/loss/image', 'Decoder reconstruction loss (log scale)', True)]):
        ax, ends = axes[row, col], []
        for arm in FIG_ARMS:
            x, y = trailing(metrics[(metrics.task == task) & (metrics.arm == arm)], key, METRIC_WINDOW[task])
            ax.plot(x, y, color=FIG_COLOR[arm])
            ok = ~np.isnan(y)
            if ok.any():
                ends.append((x[ok][-1], y[ok][-1], arm))
        if log:
            ax.set_yscale('log')
        if key == 'train/loss/dyn' and FREE_NATS is not None:
            lo, hi = ax.get_ylim()
            ax.set_ylim(max(0, min(lo, FREE_NATS - 0.15 * (hi - FREE_NATS))), hi)   # room for the note under the line
            ax.axhline(FREE_NATS, color=INK2, lw=1, ls='--')
            ax.annotate(f'free-nats floor ({FREE_NATS:g}): no gradient below this', (0.99, FREE_NATS),
                        xycoords=('axes fraction', 'data'), xytext=(0, -4), textcoords='offset points',
                        ha='right', va='top', fontsize=8.5, color=INK2)
        ax.set_title(f'{name}: {title}', loc='left', fontsize=11)
        ax.set_xlabel(FIG_XLABEL[task])
        ax.xaxis.set_major_formatter(THOUSANDS)
        label_ends(ax, ends, '{:.2f}' if task == 'atari100k' else '{:.1f}')
fig.subplots_adjust(wspace=0.5, hspace=0.45)
shared_legend(fig, y=0.97)
fig.text(0.01, 0.005, 'Without reconstruction the decoder still trains, on a detached latent: it becomes a probe of '
         'how much pixel detail the latent holds.', fontsize=8.5, color=INK2)
save_fig(fig, 'world_model_losses.png')

# 4. Information in the latent against return ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (task, name) in zip(axes, FIG_TASKS.items()):
    s = agg_summary[agg_summary.task == task].set_index('arm')
    for arm in FIG_ARMS:
        ax.plot(s.loc[arm, 'loss_dyn'], s.loc[arm, 'final_score'], 'o', ms=10, color=FIG_COLOR[arm], mec=SURF, mew=2)
        ax.annotate(FIG_SHORT[arm], (s.loc[arm, 'loss_dyn'], s.loc[arm, 'final_score']), xytext=(8, -3),
                    textcoords='offset points', fontsize=9, color=INK2)
    if FREE_NATS is not None:
        ax.axvline(FREE_NATS, color=INK2, lw=1, ls='--')
    ax.set_title(name, loc='left')
    ax.set_xlabel('KL, mean over last 10% of training (nats)')
    ax.set_ylabel('episode return, last 10%')
    ax.margins(0.2)
fig.text(0.01, -0.06, 'One point per arm, one seed each. Four points show a pattern, not a fitted relationship.',
         fontsize=8.5, color=INK2)
fig.subplots_adjust(wspace=0.3)
save_fig(fig, 'kl_vs_return.png')

# The same numbers as a table, for anyone who cannot tell the colours apart
cols = ['task', 'arm', 'final_score', 'final_score_std', 'n_final_episodes', 'loss_dyn', 'loss_image']
print(agg_summary[cols].round(2).to_string(index=False))
