# %% [markdown]
# # HERMES · DreamerV3 first run — one stock agent on Atari100k Pong
#
# This notebook trains **one unmodified DreamerV3 agent** (Hafner et al., upstream code at a
# pinned commit) on **Atari100k Pong** from a Colab GPU, writes everything to Google Drive,
# survives disconnects, and shows what the agent is doing while it trains. Its purpose is
# familiarity: run the real thing once, end to end, and see the pieces. No ablations, no
# experiment matrix; those come later and live in `experiments/04-dreamerv3-baseline/`.
#
# The paper concept → code map is in `../04-dreamerv3-baseline/ARCHITECTURE_MAP.md`; the
# `README.md` next to this notebook says what every cell does and what was verified where.
#
# ## Contents
#
# 1. [Settings](#settings) — the few knobs there are
# 2. [Machine and Drive](#machine) — GPU, Python, where files go
# 3. [Install](#install) — pinned JAX + DreamerV3 dependencies, with a loud GPU check
# 4. [Clone DreamerV3](#clone) — pinned commit, unmodified
# 5. [See the architecture](#inspect) — parameter tree and RSSM tensor shapes
# 6. [The runner](#runner) — launch, log, resume
# 7. [Smoke tests](#smoke) — the `debug` config on CPU and on the GPU
# 8. [Timing](#timing) — measure speed, then project the cost of the real run
# 9. [The run](#run) — resumable
# 10. [Look at it](#results) — score against the paper's curves, losses, imagination, the agent playing
#
# ## How to run, and how to resume
#
# 1. **Runtime → Change runtime type → A100 GPU.** (L4 should work. T4 has no bfloat16
#    hardware and DreamerV3 computes in bfloat16; the machine cell warns about it.)
# 2. **Runtime → Run all**, approve the Google Drive pop-up. Everything durable goes to
#    `MyDrive/HERMES/dreamerv3-first-run/`.
# 3. **If Colab disconnects** (it will, eventually): reconnect, **Run all** again. The install
#    and the smoke tests repeat (a few minutes), and the run continues from its last
#    checkpoint. That is DreamerV3's own convention: the same command with the same
#    `--logdir` resumes. You lose at most `SAVE_EVERY` seconds of training.
# 4. **To stop early:** interrupt the run cell (Runtime → Interrupt). The results cells work on
#    whatever exists; running the run cell again resumes.
#
# Set `PRESET = 'smoke'` and Run all once first if you want to see every stage finish in
# minutes with a toy-sized network before spending real GPU time.

# %% [markdown]
# <a name="settings"></a>
# ## 1. Settings
#
# * `CONFIG_BLOCK` and `TASK`: DreamerV3 bundles per-benchmark settings in named blocks of
#   `configs.yaml`. `atari100k` sets the step budget (`1.1e5` agent steps; each agent step is 4
#   frames, so 440K frames, the standard 400K-frame Atari100k budget plus a margin), one
#   environment, a train ratio of 256, 64×64 colour frames, and the Atari100k protocol. Any
#   Atari100k game works as `TASK`.
# * `SIZE`: model-size preset. `size12m` is the smallest that is still a normal DreamerV3
#   (the paper's results use the 200M default, which `''` selects). The timing section
#   measures whatever you pick before the real run starts.
# * `STEPS = None` keeps the block's budget. `SAVE_EVERY`/`LOG_EVERY` are wall-clock seconds
#   (that is how DreamerV3's timers work).
# * `UNITS_PER_HOUR`: Colab shows it in *Runtime → View resources* while a GPU runtime is
#   attached. The notebook never guesses it.

# %%
PRESET = 'full'              # 'full' = the real run; 'smoke' = every stage with the tiny debug network, in minutes

DREAMER_REPO = 'https://github.com/danijar/dreamerv3.git'
DREAMER_COMMIT = 'e3f02248693a79dc8b0ebd62c93683888ddaccfe'   # 2026-05-25, "Fix Atari frame maxpooling on reset"

CONFIG_BLOCK = 'atari100k'   # which block of configs.yaml sets up the benchmark
TASK = 'atari100k_pong'      # which game; the config block's own default is also pong
SIZE = 'size12m'             # model-size block; '' = stock 200M default
SEED = 0
STEPS = None                 # None = the config block's budget (1.1e5 agent steps for atari100k)
SAVE_EVERY = 600             # seconds between checkpoints to Drive (stock: 900)
LOG_EVERY = 60               # seconds between metric writes (stock: 120)
UNITS_PER_HOUR = None        # Colab compute units per hour for this GPU, from Runtime -> View resources

# %% [markdown]
# <a name="machine"></a>
# ## 2. Machine and Drive
#
# Mounts Drive (on Colab), creates the folder layout, and records which GPU and Python we
# got, because both matter: Colab assigns different GPUs, and the pinned packages need
# wheels for the Python version. Layout under `ROOT`:
#
# ```
# runs/<task>/<size>/seed<N>/   the DreamerV3 logdir: metrics.jsonl, scores.jsonl, config.yaml,
#                               ckpt/, replay/, scope/, plus stdout.log, command.txt,
#                               sessions.jsonl, DONE.json written by this notebook
# smoke/                        throw-away smoke-test logdirs
# timing/                       the short run used to measure speed
# plots/                        every figure this notebook produces
# records/                      provenance: versions, GPU, inspection output
# ```
#
# The notebook also works outside Colab (it was dry-run that way on a CPU); then `ROOT` is a
# local folder and the install cell is skipped.

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
    ROOT = pathlib.Path('/content/drive/MyDrive/HERMES/dreamerv3-first-run')
    REPO_DIR = pathlib.Path('/content/dreamerv3')          # code on local disk; nothing durable lives here
    CACHE_DIR = pathlib.Path('/content/jax_cache')          # compiled-program cache, local disk on purpose
else:
    ROOT = pathlib.Path('dreamerv3_first_run_local').resolve()
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
    print('\nWARNING: T4 has no bfloat16 hardware and DreamerV3 computes in bfloat16 by default. If the GPU '
          "smoke test is very slow or fails, switch to an A100/L4, or add '--jax.compute_dtype', 'float32' "
          'to the flags in the runner section (slower, more memory, but exact).')
if not GPU:
    print('\nNo GPU found: every run below uses the CPU. Fine for PRESET="smoke", useless for "full".')

# %% [markdown]
# <a name="install"></a>
# ## 3. Install
#
# DreamerV3 needs JAX plus three small libraries by the same author (`elements`: config,
# flags, logging, checkpoints; `ninjax`: the module system the networks are written in;
# `scope`: the metric writer) and the Atari emulator. Versions are **pinned to what was
# tested**; the repo's `requirements.txt` gives only lower bounds for most of them.
#
# Three things to know:
#
# * `jax[cuda12]==0.5.0` is the version in the repo's `Dockerfile`, i.e. the author-tested
#   one. The wheel bundles its CUDA libraries, so nothing else is needed, but it replaces the
#   JAX that Colab pre-installs. **This is the cell most likely to need attention on a new
#   Colab image.** The check at the end runs JAX in a subprocess and asserts it sees a
#   `CudaDevice`, so a wheel/CUDA mismatch shows up here with its real error instead of as a
#   mysterious "CUDA error" later (the repo's README warns that those are usually downstream).
# * `numpy<2` in `requirements.txt` is for DMLab/MineRL only; it is dropped (everything here
#   was verified with numpy 2).
# * `ale_py==0.9.0` is the Atari version the author pinned; its ROMs are bundled, so there is
#   no licence prompt. If there is no wheel for this Python, the cell falls back to `0.12.1`,
#   which needs a one-line compatibility change to the Atari *wrapper* (section 4) and says so.
#
# Nothing in this notebook imports JAX into the notebook process. Every JAX program is a
# **subprocess**, so a crash cannot take the notebook down, no runtime restart is needed
# after the install, and the notebook never holds GPU memory that a training run needs.

# %%
PINNED = [
    'jax[cuda12]==0.5.0',
    'elements==3.22.1', 'ninjax==3.6.3', 'portal==3.8.1', 'scope==0.7.1',
    'optax==0.2.5', 'chex==0.1.90', 'einops==0.8.2', 'cloudpickle==3.1.2',
    'ruamel.yaml==0.19.1', 'av==18.1.0', 'pillow', 'psutil',
]
if IN_COLAB:
    out = sh([PYTHON, '-m', 'pip', 'install', '-q', *PINNED], check=False)
    (ROOT / 'records' / 'pip_install.log').write_text(out)
    print(out[-3000:] or 'pip: no output (ok)')
    ale = sh([PYTHON, '-m', 'pip', 'install', '-q', 'ale_py==0.9.0'], check=False)
    if 'ERROR' in ale or 'No matching distribution' in ale:
        print('ale_py==0.9.0 has no wheel for this Python; falling back to ale_py==0.12.1\n', ale[-1500:])
        sh([PYTHON, '-m', 'pip', 'install', '-q', 'ale_py==0.12.1'])
    (ROOT / 'records' / 'pip_freeze.txt').write_text(sh([PYTHON, '-m', 'pip', 'freeze']))
else:
    print('Not on Colab: assuming the current interpreter already has the pinned packages.')

check = sh([PYTHON, '-c', (
    'import importlib.metadata as m, jax, jax.numpy as jnp, sys\n'
    'print("python", sys.version.split()[0])\n'
    'for p in ["jax","jaxlib","numpy","elements","ninjax","portal","scope","optax","chex","einops","ale_py","av"]:\n'
    '    print(f"{p}=={m.version(p)}")\n'
    'print("devices", jax.devices())\n'
    'x = jnp.ones((1024, 1024), jnp.bfloat16)\n'
    'print("bfloat16 matmul ok, sum =", float((x @ x).sum()), "on", (x @ x).devices())\n'
)], check=False)
print(check)
if PLATFORM == 'cuda':
    assert 'CudaDevice' in check, (
        'JAX does not see the GPU. Read the output above: the real cause is usually a wheel/CUDA mismatch.')
found = re.search(r'ale_py==(\S+)', check)
assert found, 'the version check did not complete; the output above is the real error'
ALE_VERSION = found.group(1)
(ROOT / 'records' / 'versions.txt').write_text(check)

# %% [markdown]
# <a name="clone"></a>
# ## 4. Clone DreamerV3
#
# The **upstream** repository at a pinned commit, and the exact commit is printed. The
# algorithm is not touched. The only conditional change is to the Atari environment
# *wrapper*: if the install had to fall back to a newer `ale_py`, its `setInt` requires a
# `str` key and a Python `int` where the wrapper passes `bytes` and a `numpy.int64`, and the
# one-line patch below is applied and announced. With `ale_py==0.9.0` nothing is applied and
# the cell asserts that the working tree equals the commit.

# %% [file] atari_ale_compat.patch

# %%
if not (REPO_DIR / '.git').exists():
    print(sh(['git', 'clone', DREAMER_REPO, str(REPO_DIR)]))
if sh(['git', 'rev-parse', 'HEAD'], cwd=REPO_DIR).strip() != DREAMER_COMMIT:
    sh(['git', 'checkout', '-q', DREAMER_COMMIT], cwd=REPO_DIR)
print(sh(['git', 'log', '-1', '--format=%H %ad %s', '--date=short'], cwd=REPO_DIR))

NEEDS_ALE_PATCH = tuple(int(x) for x in ALE_VERSION.split('.')[:2]) >= (0, 10)
patch = os.path.abspath('atari_ale_compat.patch')
if NEEDS_ALE_PATCH:
    if sh(['git', 'apply', '--check', '--reverse', patch], check=False, cwd=REPO_DIR).strip() == '':
        print(f'ale_py {ALE_VERSION}: wrapper compatibility patch already applied')
    else:
        sh(['git', 'apply', '--verbose', patch], cwd=REPO_DIR)
        print(f'ale_py {ALE_VERSION}: applied the one-line wrapper compatibility patch (embodied/envs/atari.py)')
    print(sh(['git', 'diff', '--stat'], cwd=REPO_DIR))
else:
    assert sh(['git', 'status', '--porcelain'], cwd=REPO_DIR).strip() == '', 'working tree differs from the commit'
    print(f'ale_py {ALE_VERSION}: no patches applied, the code is exactly the upstream commit')

# %% [markdown]
# <a name="inspect"></a>
# ## 5. See the architecture
#
# Before training, build the agent the way `main.py` does and look at it. The script below
# is written to disk and run as a subprocess with the same config flags the real run will
# use. It prints:
#
# 1. **The parameter tree.** In JAX a model's weights are not hidden inside objects; they are
#    a plain dictionary of arrays (a *pytree*: any nested structure of arrays) keyed by path,
#    such as `dyn/dyngru/kernel`. The listing is the whole model. The per-module totals are
#    the true parameter count behind the `SIZE` preset. Keys under `opt/` are optimizer
#    moments, `slowval/` is the slow-moving copy of the critic, `retnorm/` etc. are running
#    statistics; they are saved in checkpoints too but are not network weights.
# 2. **The shapes flowing through the RSSM** for one batch of `B` sequences of `T` steps:
#    image → encoder `tokens` → recurrent state `h` (`deter`) → posterior sample `z` (`stoch`,
#    one-hot over classes) → prior logits `p(z|h)` → the imagined `h` and `ẑ` of the rollouts,
#    which start from *every* posterior state and unroll `imag_length` steps on the prior
#    alone, never seeing an observation. `scan`, used inside `observe` and `imagine`, is JAX's
#    compiled for-loop: one step function applied `T` times with the state threaded through.
#    Two checks run on the arrays: `z` and `ẑ` are genuinely one-hot, and there are `B × T`
#    rollouts.
#
# The result is also saved as `records/inspect.json`. Where to read the code for each line:
# encoder `dreamerv3/rssm.py:179-250`, RSSM `rssm.py:16-176` (`_core` updates `h`,
# `_observe` is the posterior, `_prior` the prior, `imagine` the rollout), decoder
# `rssm.py:253-359`, heads and losses `dreamerv3/agent.py:156-245`.

# %% [file] inspect_agent.py

# %%
def size_blocks():
    return ['debug'] if PRESET == 'smoke' else ([SIZE] if SIZE else [])

cmd = [PYTHON, 'inspect_agent.py', '--repo', str(REPO_DIR), '--out', str(ROOT / 'records' / 'inspect.json'),
       '--configs', CONFIG_BLOCK, *size_blocks(), '--task', TASK, '--jax.platform', PLATFORM]
proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                      env={**os.environ, 'PYTHONUNBUFFERED': '1'})
(ROOT / 'records' / 'inspect.log').write_text(proc.stdout)
text = proc.stdout
print(text[text.find('PARAMETER TREE') - 80:] if 'PARAMETER TREE' in text else text)
assert proc.returncode == 0, 'inspect_agent.py failed; the traceback above is the real error'

# %% [markdown]
# <a name="runner"></a>
# ## 6. The runner
#
# One function launches `dreamerv3/main.py` as a subprocess with a `--logdir` and a list of
# config blocks and flags, streams the important lines into the notebook, and writes
# **everything** to `<logdir>/stdout.log` with a wall-clock timestamp on every line. When the
# process exits cleanly it writes `<logdir>/DONE.json`; if it is interrupted or crashes it
# does not, and the next call with the same logdir resumes from the last checkpoint because
# that is what DreamerV3 does when it finds `ckpt/latest` (`embodied/run/train.py:83-90`).
# Each session's wall time goes to `<logdir>/sessions.jsonl`, so the total cost of a run that
# spanned several Colab sessions is still known.
#
# Two facts about DreamerV3's own logging, both visible in `stdout.log`: checkpoints and
# metric writes happen on **wall-clock** timers (`run.save_every`, `run.log_every`), not on
# step counts; and no checkpoint is written when the loop finishes, so the last checkpoint of
# a completed run can be up to `SAVE_EVERY` seconds older than its final metrics. The score
# curve (what the paper reports) comes from `scores.jsonl`, which is complete.

# %%
INTERESTING = ('Agent Step', 'fps/policy', 'Error', 'error', 'Traceback', 'checkpoint', 'Compiling', 'Done compiling',
               'Start training', 'devices', 'Optimizer opt has', 'ALERT', 'Logdir')

def run_dreamer(logdir, configs, flags=(), label=None, quiet=False):
    """Run (or resume) one DreamerV3 training job. Returns a dict with exit code and wall time."""
    logdir = pathlib.Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    # Flags come after the config blocks, so a flag always wins over a block (main.py:27-29).
    cmd = [PYTHON, str(REPO_DIR / 'dreamerv3' / 'main.py'), '--logdir', str(logdir), '--configs', *configs, *flags]
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
    that are already in the file. A restart shows as a step number smaller than the one
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

# The config block's own numbers, read from the file rather than typed in here.
import ruamel.yaml as yaml
CONFIGS = yaml.YAML(typ='safe').load((REPO_DIR / 'dreamerv3' / 'configs.yaml').read_text())
BLOCK_STEPS = int(CONFIGS[CONFIG_BLOCK]['run']['steps'])
FRAMES_PER_STEP = CONFIGS['defaults']['env'].get(CONFIG_BLOCK, {}).get('repeat', 1)   # the logger multiplies steps by this
BATCH_STEPS = CONFIGS['defaults']['batch_size'] * CONFIGS['defaults']['batch_length']
TRAIN_RATIO = CONFIGS[CONFIG_BLOCK].get('run', {}).get('train_ratio', CONFIGS['defaults']['run']['train_ratio'])

def run_steps():
    if PRESET == 'smoke':
        return 400
    return int(STEPS) if STEPS else BLOCK_STEPS

print(f'{CONFIG_BLOCK}: {BLOCK_STEPS:,} agent steps = {BLOCK_STEPS * FRAMES_PER_STEP:,} frames; '
      f'{TRAIN_RATIO / BATCH_STEPS:.3g} gradient updates per agent step, so {BLOCK_STEPS * TRAIN_RATIO / BATCH_STEPS:,.0f} updates per run')

# %% [markdown]
# <a name="smoke"></a>
# ## 7. Smoke tests
#
# DreamerV3's `debug` config block shrinks every network to a handful of units and the batch
# to 8×10. It **learns nothing** and runs on the **CPU** (the block sets `jax.platform: cpu`),
# so it proves the pipeline (emulator, replay, training loop, checkpoints, metrics) without
# proving the GPU path. So there are two runs, both throw-away:
#
# 1. Pong with `debug` on CPU.
# 2. Pong with `debug` forced back onto the GPU: same tiny network, the real CUDA path,
#    bfloat16 arithmetic, and enough updates to pass **120**, because DreamerV3 starts its JAX
#    profiler at update 100 and stops it at 120 (`embodied/jax/agent.py:302-307`), which
#    cannot be switched off from the config and must be known to work.
#
# Checkpoint and log timers are set to 5 s so a one-minute run exercises both. After each run
# the cell checks what must exist: `metrics.jsonl`, `scores.jsonl`, a checkpoint with its
# `done` marker, saved replay chunks.

# %%
def check_logdir(logdir, min_step):
    logdir = pathlib.Path(logdir)
    for f in ('metrics.jsonl', 'scores.jsonl', 'config.yaml', 'ckpt/latest', 'DONE.json'):
        assert (logdir / f).exists(), f'missing {f} in {logdir}'
    latest = logdir / 'ckpt' / (logdir / 'ckpt' / 'latest').read_text().strip()
    assert (latest / 'done').exists(), f'checkpoint {latest} has no done marker'
    assert (latest / 'agent.pkl').exists() and (latest / 'replay.pkl').exists(), 'checkpoint incomplete'
    chunks = list((logdir / 'replay').glob('*.npz'))
    assert chunks, 'no replay chunks were saved'
    step = last_step(logdir)
    assert step >= min_step, f'only reached step {step} < {min_step}'
    keys = sorted({k for r in read_jsonl(logdir / 'metrics.jsonl') for k in r})
    print(f'ok: {logdir.name}: reached step {step:,}, {len(keys)} metric keys, {len(chunks)} replay chunks, '
          f'checkpoint {latest.name}')
    return keys

smoke = ROOT / 'smoke'
SMOKE_FLAGS = ['--task', TASK, '--run.save_every', '5', '--run.log_every', '5']
r = run_dreamer(smoke / 'cpu', [CONFIG_BLOCK, 'debug'], ['--run.steps', '600', *SMOKE_FLAGS], quiet=True)
assert r['exit'] == 0, 'debug run failed; read smoke/cpu/stdout.log'
keys = check_logdir(smoke / 'cpu', 500)
print('loss keys:', [k for k in keys if k.startswith('train/loss/')])

if PLATFORM == 'cuda':
    # debug: train_ratio 8 with batch 8x10 = 80, i.e. one update per 10 agent steps after an
    # 80-step warm-up; 1500 steps gives ~140 updates, past the profiler window.
    r = run_dreamer(smoke / 'gpu', [CONFIG_BLOCK, 'debug'], ['--jax.platform', 'cuda', '--run.steps', '1500', *SMOKE_FLAGS])
    assert r['exit'] == 0, 'GPU debug run failed; scroll up: the first error is the real one'
    check_logdir(smoke / 'gpu', 1400)
    rows = lineage(read_jsonl(smoke / 'gpu' / 'metrics.jsonl'))
    updates = max(r.get('train/opt/updates', 0) for r in rows)
    assert updates >= 120, f'only {updates} updates; the profiler window (100-120) was not exercised'
    print(f'GPU path ok: {updates:.0f} gradient updates, profiler trace written: '
          f'{bool(list((smoke / "gpu").rglob("*.xplane.pb")))}')
else:
    print('no GPU: the CUDA smoke test is skipped')

# %% [markdown]
# <a name="timing"></a>
# ## 8. Timing
#
# Never start a long run on a guess. This runs the **real configuration** (`CONFIG_BLOCK` +
# `SIZE`) for a short budget into a logdir on Drive, long enough to get past compilation and
# the replay warm-up (training starts once the buffer holds `batch_size × batch_length =
# 1024` steps), and reads the speed DreamerV3 logs itself:
#
# * `fps/policy`: agent steps per second of wall time in the last log window, with the
#   training interleaved. This turns the step budget into hours.
# * `fps/train`: replayed steps trained per second; divided by 1024 it is gradient updates
#   per second.
#
# A checkpoint to Drive is forced once during this run (`save_every 120`) so that Drive I/O
# is part of the measurement. The projection multiplies by `UNITS_PER_HOUR` if you set it.
# In `smoke` mode the numbers are for the toy network and mean nothing.

# %%
TIMING_STEPS = 300 if PRESET == 'smoke' else 3000        # ~500 updates after the 1024-step warm-up
SIZE_NAME = '_'.join(size_blocks()) or 'default'          # 'debug' in smoke mode, so logdirs never mix network sizes
logdir = ROOT / 'timing' / f'{TASK}_{SIZE_NAME}'
if not (logdir / 'DONE.json').exists():
    flags = ['--task', TASK, '--jax.platform', PLATFORM, '--run.steps', str(TIMING_STEPS)]
    if PRESET != 'smoke':
        flags += ['--run.log_every', '30', '--run.save_every', '120']
    r = run_dreamer(logdir, [CONFIG_BLOCK, *size_blocks()], flags, label='timing')
    assert r['exit'] == 0, f'timing run failed: {logdir}'
rows = [r for r in lineage(read_jsonl(logdir / 'metrics.jsonl')) if 'fps/policy' in r and r.get('fps/train', 0) > 0]
assert rows, f'no log window with training in {logdir}; increase TIMING_STEPS'
last = rows[-1]
wall = sum(s['wall_seconds'] for s in read_jsonl(logdir / 'sessions.jsonl'))
params = next((r['train/opt/param_count'] for r in rows if 'train/opt/param_count' in r), float('nan'))
hours = run_steps() / last['fps/policy'] / 3600
TIMING = dict(fps_policy=last['fps/policy'], fps_train=last['fps/train'], updates_per_s=last['fps/train'] / BATCH_STEPS,
              timing_wall_seconds=wall, params=params, projected_hours=hours,
              projected_units=hours * UNITS_PER_HOUR if UNITS_PER_HOUR else None)
(ROOT / 'records' / 'timing.json').write_text(json.dumps(TIMING, indent=1))
print(f'{TASK} {SIZE_NAME}: {params:,.0f} params, {last["fps/policy"]:.2f} agent steps/s, '
      f'{TIMING["updates_per_s"]:.2f} updates/s  (timing run incl. compile: {wall / 60:.1f} min)')
print(f'Projection for the real run: {run_steps():,} agent steps -> {hours:.1f} hours of this GPU'
      + (f' -> {hours * UNITS_PER_HOUR:.0f} compute units' if UNITS_PER_HOUR else '  (set UNITS_PER_HOUR to see units)'))
print('This extrapolates from a few hundred updates; each resume adds the install, compilation and up to SAVE_EVERY of lost work.')

# %% [markdown]
# <a name="run"></a>
# ## 9. The run
#
# The command is the stock one plus the size block, `--task`, `--seed`, and the two timers
# from the settings; nothing else. The full command line is saved next to the run in
# `command.txt`, and DreamerV3 itself saves the resolved configuration as `config.yaml`, so
# the run is exactly reproducible from its folder. The status line before it says whether
# this is a fresh start, a resume, or already done.
#
# **Seeds.** `--seed` fixes every JAX draw (initialisation, sampling, replay), but the stock
# config does not seed the environment (`main.py:241-242` seeds only suites with `use_seed`),
# so two runs with the same seed still play different episodes. This is also true of the
# paper's runs.

# %%
RUN_DIR = ROOT / 'runs' / TASK / SIZE_NAME / f'seed{SEED}'   # a different size is a different run; never resume across sizes
RUN_FLAGS = ['--task', TASK, '--jax.platform', PLATFORM, '--seed', str(SEED), '--run.steps', str(run_steps())]
if PRESET != 'smoke':
    RUN_FLAGS += ['--run.save_every', str(SAVE_EVERY), '--run.log_every', str(LOG_EVERY)]

if (RUN_DIR / 'DONE.json').exists():
    print(f'already done: {RUN_DIR}')
else:
    print(f'{"resuming from step " + format(last_step(RUN_DIR), ",") if last_step(RUN_DIR) else "fresh start"}: {RUN_DIR}')
    r = run_dreamer(RUN_DIR, [CONFIG_BLOCK, *size_blocks()], RUN_FLAGS, label=f'{TASK} {SIZE_NAME} seed{SEED}')
    if r['exit'] != 0:
        raise RuntimeError(f'exit {r["exit"]}; see {RUN_DIR / "stdout.log"}. Fix the cause, then run this cell again to resume.')

# %% [markdown]
# <a name="results"></a>
# ## 10. Look at it
#
# Everything below reads files DreamerV3 wrote; nothing is typed in by hand. Three sources:
#
# * `scores.jsonl`: one line per finished episode, `{step, episode/score}`; this is what the
#   paper plots. For Atari the logged step is in **frames** (the logger multiplies by the
#   action repeat of 4), matching the 400K-frame budget and the reference curves.
# * `metrics.jsonl`: every scalar, every `LOG_EVERY` seconds: the world-model losses
#   (`train/loss/image`, `rew`, `con`, `dyn`, `rep`), the actor/critic losses, speed, replay
#   and memory statistics.
# * `scope/`: the non-scalar outputs, read with `scope.Reader`: the open-loop video
#   (`report/openloop/image`), a video of the agent playing (`epstats/policy_image`), the
#   timer breakdown, and the raw `nvidia-smi` output. (The `scope` web viewer needs a server
#   and a port; reading the files directly is simpler and works anywhere.)
#
# **Reference.** The repo ships the paper's own Atari100k curves in
# `scores/atari100k-dreamerv3.json.gz` (5 seeds per game, x in frames) and PPO's. They were
# made with the **200M** model; the paper's Table 4 gives DreamerV3 −4 on Pong at 400K frames
# (random −21, human 15, PPO −20). With `size12m` the honest expectation is "learns, probably
# below the 200M curves", not a match; the paper's scaling results show smaller models learn
# more slowly.

# %%
import gzip

import matplotlib.pyplot as plt
import numpy as np

PLOTS = ROOT / 'plots'
SCORES = lineage(read_jsonl(RUN_DIR / 'scores.jsonl'))
METRICS = lineage(read_jsonl(RUN_DIR / 'metrics.jsonl'))
DONE = (RUN_DIR / 'DONE.json').exists()
TARGET_FRAMES = run_steps() * FRAMES_PER_STEP
print(f'{RUN_DIR}\n{len(SCORES)} finished episodes, last logged step {last_step(RUN_DIR):,} of {TARGET_FRAMES:,} frames, '
      f'{"complete" if DONE else "incomplete"}')

def smooth(y, k):
    """Centred running mean over k points; the raw points are always drawn as well."""
    return np.convolve(y, np.ones(k) / k, mode='same') if len(y) >= k else np.asarray(y, float)

def reference_curves(game):
    out = {}
    for method in ('dreamerv3', 'ppo_fixhp'):
        path = REPO_DIR / 'scores' / f'atari100k-{method}.json.gz'
        if path.exists():
            runs = [r for r in json.load(gzip.open(path)) if r['task'] == f'atari_{game}']
            out[method] = [(np.asarray(r['xs']), np.asarray(r['ys'])) for r in runs]
    return out

fig, ax = plt.subplots(figsize=(8, 4.5))
for method, curves in reference_curves(TASK.split('_', 1)[1]).items():
    for i, (xs, ys) in enumerate(curves):
        ax.plot(xs, ys, color='0.7' if method == 'dreamerv3' else '0.88', lw=1,
                label=f'paper {"DreamerV3 200M" if method == "dreamerv3" else "PPO"}, 5 seeds' if i == 0 else None)
if SCORES:
    x = np.array([r['step'] for r in SCORES])
    y = np.array([r['episode/score'] for r in SCORES])
    ax.plot(x, y, color='#d62728', alpha=0.3, lw=0.8)
    ax.plot(x, smooth(y, 10), color='#d62728', lw=2, label=f'this run: {SIZE_NAME}, seed {SEED}'
            + ('' if DONE else f' (incomplete, {x[-1]:,} frames)'))
ax.set_title(f'{TASK}: training-episode return')
ax.set_xlabel('environment frames')
ax.set_ylabel('episode return')
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOTS / 'score.png', dpi=150);

# %% [markdown]
# **What the world model is doing.** `image` is the reconstruction loss (sum of squared pixel
# errors per frame, so its scale is large), `rew` and `con` the reward and continue
# predictors, `dyn` and `rep` the two KL terms, which sit at the 1-nat free-bits floor until
# posterior and prior drift apart. `opt/loss` is the sum of everything with its scales (one
# optimizer updates world model, actor and critic together; only stop-gradients separate
# them), so it is not a world-model metric on its own. `fps/policy` and `fps/train` are the
# speeds the timing section used.

# %%
KEYS = ['train/loss/image', 'train/loss/rew', 'train/loss/con', 'train/loss/dyn', 'train/loss/rep',
        'train/opt/loss', 'fps/policy', 'fps/train']
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for ax, key in zip(axes.flat, KEYS):
    pts = [(r['step'], r[key]) for r in METRICS if key in r]
    if pts:
        x, y = map(np.array, zip(*pts))
        ax.plot(x, y, color='#1f77b4', lw=1.2)
    ax.set_title(key, fontsize=10)
    ax.set_xlabel('environment frames')
    ax.grid(alpha=0.3)
    if key == 'train/loss/image':
        ax.set_yscale('log')
fig.tight_layout()
fig.savefig(PLOTS / 'world_model.png', dpi=150);

# %% [markdown]
# **Imagination, seen.** Every `report_every` seconds (300 by default) DreamerV3 takes six
# replayed sequences, feeds the first half to the model (green border) and lets it **imagine**
# the second half from the prior alone (red border), then decodes both halves. Each column is
# one sequence; the three rows are the true frames, the decoded frames, and their difference.
# Below: the earliest and the latest report, at the last observed step, the first imagined
# step, and the last imagined step, so you can see the decoder and the dynamics improve.

# %%
import av
import scope

def video(logdir, key, which=-1):
    reader = scope.Reader(str(logdir))
    if key not in reader.keys():
        return None, None
    steps, files = reader[key]
    container = av.open(str(files[which]))
    frames = np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])
    container.close()
    return int(steps[which]), frames

reports = scope.Reader(str(RUN_DIR)).length('report/openloop/image') if 'report/openloop/image' in scope.Reader(str(RUN_DIR)).keys() else 0
which = [0, -1] if reports > 1 else ([-1] if reports else [])
fig, axes = plt.subplots(max(len(which), 1), 3, figsize=(14, 2.6 * max(len(which), 1)), squeeze=False)
for row, w in zip(axes, which):
    step, frames = video(RUN_DIR, 'report/openloop/image', w)
    n = len(frames)                       # the report appends min(10, T) blank frames to the video
    T = n - 10 if n >= 20 else n // 2
    picks = {'last observed': T // 2 - 1, 'first imagined': T // 2, 'last imagined': T - 1}
    for ax, (name, t) in zip(row, picks.items()):
        ax.imshow(frames[t])
        ax.set_title(f'report at {step:,} frames: {name} (t={t} of {T})', fontsize=8)
        ax.axis('off')
if not which:
    for ax in axes.flat:
        ax.axis('off')
    print('no open-loop report yet (the first one is written report_every seconds into training)')
fig.tight_layout()
fig.savefig(PLOTS / 'openloop.png', dpi=150);

# %% [markdown]
# **The agent playing.** `epstats/policy_image` is the full first environment's latest
# finished episode as the agent saw it (64×64, the model input, not the emulator's native
# frame). Six frames spread across the episode.

# %%
step, frames = video(RUN_DIR, 'epstats/policy_image')
if frames is None:
    print('no finished episode recorded yet')
else:
    picks = np.linspace(0, len(frames) - 1, 6).astype(int)
    fig, axes = plt.subplots(1, 6, figsize=(15, 2.8))
    for ax, t in zip(axes, picks):
        ax.imshow(frames[t])
        ax.set_title(f'frame {t} of {len(frames)}', fontsize=8)
        ax.axis('off')
    fig.suptitle(f'episode recorded at {step:,} frames', fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / 'episode.png', dpi=150)

# %% [markdown]
# **Where the time went, and the summary.** DreamerV3's timer breakdown says how much wall
# time was spent training versus acting versus everything else; `nvidia-smi` as captured
# during the run says how much of the GPU the run actually used. The final numbers: mean
# return over the episodes in the last 10% of the budget, against the paper's Table 4 value
# for the 200M model.

# %%
reader = scope.Reader(str(RUN_DIR))
for key in ('timer', 'usage/nvsmi/output'):
    if key in reader.keys():
        steps, files = reader[key]
        print(f'--- {key} at {int(steps[-1]):,} frames ---')
        print(files[-1].read_bytes().decode(errors='replace').strip()[:1500])

sessions = read_jsonl(RUN_DIR / 'sessions.jsonl')
sc = np.array([(r['step'], r['episode/score']) for r in SCORES]) if SCORES else np.zeros((0, 2))
final = sc[sc[:, 0] >= 0.9 * TARGET_FRAMES, 1] if len(sc) else np.array([])
summary = dict(
    task=TASK, size=SIZE_NAME, seed=SEED, complete=DONE, gpu=GPU and GPU['name'],
    episodes=len(sc), last_step_frames=last_step(RUN_DIR),
    final_score_last_10pct=float(final.mean()) if len(final) else None, n_final_episodes=int(len(final)),
    mean_score_all_episodes=float(sc[:, 1].mean()) if len(sc) else None,
    wall_hours_all_sessions=sum(s['wall_seconds'] for s in sessions) / 3600, sessions=len(sessions),
    paper_table4_pong_200M=-4, paper_random=-21, paper_human=15)
(ROOT / 'records' / 'summary.json').write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
print(f'\nplots: {sorted(p.name for p in PLOTS.glob("*.png"))}\nroot:  {ROOT}')
