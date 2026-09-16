# %% [markdown]
# # Hermes · Experiment 3 **v2** — Does a reward-only world model learn to *select* what matters?
#
# **Action-conditioned, reward-defined, selective relevance.** Drop this notebook into
# Colab, make sure the runtime is a **GPU**, and press **Runtime → Run all**.
#
# ### What v2 fixes, and why
#
# v1 ran at `medium` scale and produced four problems that made its comparison
# uninterpretable. v1 is kept untouched; this notebook writes to its own Drive folder.
#
# | v1 problem | what the v1 run showed | v2 fix |
# |---|---|---|
# | **Arm A was still training** at the fixed 40 epochs | A's loss fell 53% in its last 5 epochs; 2 of 3 seeds had not escaped a long plateau, so "A is worse" was really "A is slower" | train to a **held-out plateau** with the same maximum budget for every arm, and report **learning speed** as its own metric |
# | **Probes had no dynamic range** | distractor positions and the inactive goal read 0.00 from *every* latent, including untrained weights, so "B dropped them" was untestable | add a **raw-pixel probe** as a positive control, an **identity-free distractor heatmap** target, and a **reconstruction fidelity** check on arm A |
# | **The negative control never learned its task** | C's auxiliary loss barely moved and it read distractors at 0.03–0.35, so "forcing irrelevant content" was never actually forced | give C an **identity-free** target (a distractor occupancy heatmap): v1 asked for each distractor's xy *by slot*, but slots have no stable identity |
# | **Planning measured model exploitation** | reward R² was 0.89 on recorded data but negative along the planner's own trajectories, with systematically optimistic imagined reward | add **planner-driven parking episodes** (aimed at a coin-flip goal, so still no leakage) to the offline data, and report on-policy error in **raw units** |
#
# Also fixed: the verdict now evaluates claims on the seeds where **both** arms passed the
# gate, instead of dropping an arm entirely when one seed fails.
#
# ### The question
#
# Experiment 2 showed a task-only latent matches a reconstruction latent while carrying
# ~12x less background. A skeptic dismisses that in one line: *"colour never mattered,
# so of course it was dropped."* A model that ignores **all** colour would have scored
# the same. This experiment makes one piece of colour essential and another useless, so
# blanket discarding now fails visibly.
#
# ### The game
#
# * A **white blob** (the agent) is steered by a 2-D continuous action. It has
#   **momentum**: it cannot stop instantly, so planning several steps ahead matters.
# * Two **black goals**, a hollow **ring** and a **plus**, sit still for the episode.
# * The **background hue secretly selects which goal pays**: warm hues (red, yellow,
#   green) → the ring; cool hues (blue, purple, pink) → the plus. Nothing else marks it.
# * Two **coloured distractor blobs** bounce around, plus fresh per-pixel noise. They
#   never matter.
# * Reward is dense: closer to the *active* goal is better, with a bonus for arriving.
#
# ### What is new versus Experiment 2
#
# | | Experiment 2 | Experiment 3 |
# |---|---|---|
# | control | none, objects move on their own | **agent steered by actions, with momentum** |
# | task signal | target position, hand-picked | **reward**: relevance is defined by what predicts it |
# | relevance | background never relevant | **background hue decides which goal pays** |
# | headline metric | position error | **real return from planning inside the latent** |
# | latent | unconstrained (arm D exploded) | **LayerNorm without learnable gain, in every arm** |
# | checkpoints | logs only | **full weights, mid-run and final, on Google Drive** |
#
# ### The four arms
#
# All arms share an identical encoder + GRU + reward head, start from **byte-identical
# weights** (asserted in code), train on **the same data in the same order**, and all
# receive the same reward-prediction loss. They differ **only** in the extra loss:
#
# | arm | extra loss | analogue | role |
# |---|---|---|---|
# | `A_recon` | redraw the frame from the latent | DreamerV3 (loss only) | the reconstruction camp |
# | `B_task` | none, no decoder exists | MuZero-style value equivalence | **the bet** |
# | `C_scene` | predict *irrelevant* scene factors (distractors, background saturation/brightness) | none | negative control |
# | `D_latent` | predict the future latent of an EMA encoder | TD-MPC2 consistency / JEPA | pixel-free self-supervision |
#
# ### How to run, and how to resume
#
# 1. **Runtime → Change runtime type → T4 GPU** (or better). The notebook stops with a
#    clear message if there is no GPU.
# 2. **Runtime → Run all.**
# 3. Approve the **Google Drive** pop-up. Everything is written to
#    `MyDrive/hermes_exp3/<run name>/`: checkpoints, cached evaluations, CSVs, figures,
#    GIFs and `VERDICT.txt`.
# 4. **If Colab disconnects:** reconnect, and press **Run all** again. Finished training
#    runs, half-finished training runs (saved every few epochs), and finished evaluations
#    are all picked up from Drive. Only the minutes since the last save are repeated.
#
# ### Map of the notebook
#
# 0 setup and Drive · 1 configuration · 2 environment and data · 3 models · 4 planner ·
# 5 training (resumable) · 6 evaluation (resumable) · 7 analysis and the learning gate ·
# 8 figures · 9 animations · 10 verdict
# %%
# ============================================================================
# 0. Setup: imports, GPU, Google Drive
# ============================================================================
import os, sys, time, math, json, copy, zipfile, warnings, datetime, contextlib, io
from dataclasses import dataclass, asdict, replace as dc_replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
HEADLESS = bool(os.environ.get("HERMES_HEADLESS"))      # set for smoke tests outside Colab
if HEADLESS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import hsv_to_rgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"torch  : {torch.__version__}")
print(f"device : {DEVICE}")
if DEVICE == "cuda":
    print(f"gpu    : {torch.cuda.get_device_name(0)}")
else:
    print("gpu    : NONE")

# ---- Google Drive: every checkpoint and result lives here, so a disconnect loses
# ---- nothing that was already saved. Outside Colab a local folder is used instead.
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    STORAGE_ROOT = "/content/drive/MyDrive/hermes_exp3_v2"
else:
    STORAGE_ROOT = os.environ.get("HERMES_OUT", os.path.abspath("hermes_exp3_v2_runs"))
os.makedirs(STORAGE_ROOT, exist_ok=True)
print(f"storage: {STORAGE_ROOT}")

# ---- house style, shared with Experiment 2 so the figures read as one series ----
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 120, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
})
ARM_COLOR = {"A_recon": "#d1495b", "B_task": "#2e86ab", "C_scene": "#e8a33d",
             "D_latent": "#5f8d4e",
             "oracle": "#222222", "oracle_myopic": "#777777",
             "oracle_colourblind": "#9b59b6", "oracle_wrong": "#c0a0c0",
             "random": "#bbbbbb", "noop": "#dddddd",
             "chance": "#bbbbbb", "hold_current": "#8a8a8a", "colourblind_true": "#9b59b6",
             "init": "#9a9a9a", "ceiling": "#444444"}
ARM_LABEL = {"A_recon": "A · reward + reconstruction", "B_task": "B · reward only",
             "C_scene": "C · reward + irrelevant-scene aux",
             "D_latent": "D · reward + next-latent (EMA)",
             "oracle": "oracle (true state, same planner)",
             "oracle_myopic": "oracle, 1-step planning",
             "oracle_colourblind": "oracle, colour-blind",
             "oracle_wrong": "oracle, wrong goal",
             "random": "random actions", "noop": "no-op (zero action)",
             "chance": "chance (predict mean)", "hold_current": "hold current reward",
             "colourblind_true": "colour-blind, true state",
             "init": "untrained (shared starting weights)",
             "ceiling": "CNN trained on frames (ceiling)"}
SHORT = {"A_recon": "A", "B_task": "B", "C_scene": "C", "D_latent": "D", "init": "init",
         "ceiling": "ceil"}


def show(fig):
    """Display in Colab; close silently in headless smoke tests."""
    if HEADLESS:
        plt.close(fig)
    else:
        plt.show()
# %% [markdown]
# ## 1. Configuration
#
# **The only line you normally touch is `SCALE`.**
#
# | scale | frames | seeds | training data | max epochs | purpose |
# |---|---|---|---|---|---|
# | `quick` | 24 px | 1 | 96 episodes | 4 | proves every cell runs; **numbers are meaningless and expected to fail the gate** |
# | `medium` | 32 px | 3 | 6000 x 60 steps | 120 | **run this first**: does every arm pass the learning gate? |
# | `full` | 48 px | 4 | 8000 x 60 steps | 150 | the headline configuration |
#
# **Epochs are a ceiling, not a length (v2).** Every arm trains until its held-out reward
# R2 stops improving (`min_delta` over `patience_snaps` snapshots, never before
# `min_epochs`), under the same ceiling. v1 used a fixed 40 epochs and arm A was still
# improving steeply when it stopped - its loss fell 53% in the last 5 epochs - which made
# "A is worse" indistinguishable from "A is slower". How long each arm needed is now a
# reported metric instead of a confound.
#
# The brief's rule, learned the hard way: **start small until the learning gate passes,
# then scale up.** So the default is `medium`. When it finishes and every arm passes the
# gate, set `SCALE = "full"` and Run all again. Each scale writes to its own Drive
# folder, so they never overwrite each other.
#
# No run time is promised here. Measured in the v1 medium run at 32 px: 2.8 s/epoch for
# the task-only arm, 5.0 s/epoch for the reconstruction arm. v2 adds the parking data
# (once per seed, cached) and trains longer, and prints its own **measured** estimate
# after the first epoch of each arm.
#
# **Resume safety.** The chosen configuration is written to `config.json` in the run
# folder. If you change a setting that would make old checkpoints invalid, the cell
# refuses to continue and tells you what changed, rather than silently mixing two
# experiments. Adding seeds is allowed.
#
# Constants that were chosen from a measurement rather than a guess are marked
# *design check*. They come from `design_check.py`, a numpy simulation with the **true**
# dynamics and a perfect-knowledge planner, run before any model existed.
# %%
SCALE = os.environ.get("HERMES_SCALE", "medium")     # "quick" | "medium" | "full"
RUN_NAME = os.environ.get("HERMES_RUN_NAME", SCALE)   # Drive sub-folder for this run


@dataclass
class Cfg:
    # ---- frames and episodes ----
    img_size: int = 48
    seq_len: int = 60            # steps per training episode
    plan_T: int = 45             # steps per planning episode (<= seq_len - horizon)
    n_train: int = 8000          # training episodes per seed
    n_l1: int = 500              # held-out behaviour-policy episodes (level 1 + probes)
    n_probe: int = 300           # of those, episodes used for latent probes
    probe_stride: int = 2        # probe every 2nd frame
    probe_grid: int = 8          # distractor occupancy heatmap resolution (v2)
    n_ceiling: int = 4000        # scenes used to train the CNN probe ceiling (v2)
    ceiling_steps: int = 2500
    recon_eval_n: int = 64       # episodes used for arm A's reconstruction fidelity check
    n_plan: int = 200            # fixed planning episodes (level 2)
    n_shift: int = 100           # episodes per distribution-shift condition

    # ---- agent dynamics (design check: at damping 0.90 a 1-step planner reached 92%
    # ---- of the best return, so momentum did not matter; at 0.95 it reaches 76%) ----
    damping: float = 0.95
    max_accel: float = 0.0025    # terminal speed 0.05 arena/step; full braking ~13.5 steps
    v0_max: float = 0.025

    # ---- goals and reward ----
    goal_lo: float = 0.15
    goal_hi: float = 0.85
    goal_min_sep: float = 0.35
    bonus: float = 0.5
    bonus_sigma: float = 0.05
    success_radius: float = 0.05

    # ---- background: hue selects the active goal; dead zones around both boundaries ----
    hue_band_ring: tuple = (0.05, 0.45)     # warm -> ring active
    hue_band_plus: tuple = (0.55, 0.95)     # cool -> plus active
    bg_sat: tuple = (0.5, 0.9)
    bg_val: tuple = (0.45, 0.8)
    pixel_noise: float = 0.05

    # ---- distractors (irrelevant) ----
    n_distract: int = 2
    d_speed: tuple = (0.015, 0.045)
    d_hue_holdout: tuple = (0.60, 0.72)     # never seen in training; used by a shift test

    # ---- shapes, in pixels at 48 px (scaled with img_size) ----
    blob_sigma: float = 2.2
    ring_radius: float = 3.5
    ring_width: float = 1.0
    plus_arm: float = 4.0
    plus_halfwidth: float = 0.9

    # ---- behaviour policy for the offline data (independent of the selector) ----
    p_park: float = 0.3          # share driven by the CEM planner to a COIN-FLIP goal (v2)
    park_cem_samples: int = 200  # cheaper CEM than the evaluation planner; data only
    park_cem_iters: int = 4
    park_chunk: int = 600        # episodes parked in parallel on the GPU
    park_action_noise: float = 0.05
    p_ou: float = 0.3            # share of episodes with smooth random actions
    ou_theta: float = 0.15
    ou_sigma: float = 0.35
    kp: float = 9.0              # waypoint controller gains
    kd: float = 76.0
    pd_noise: float = 0.3
    wp_switch: tuple = (10, 21)  # steps between waypoint changes

    # ---- model ----
    latent: int = 32
    hidden: int = 192
    ch: int = 32
    reward_mlp: int = 128

    # ---- training: same MAXIMUM budget for every arm, stop on a held-out plateau (v2) ----
    epochs: int = 150            # maximum, not a fixed length
    min_epochs: int = 30
    patience_snaps: int = 4      # snapshots without improvement before stopping
    min_delta: float = 0.005     # improvement in held-out reward R2 that counts
    converged_delta: float = 0.01  # R2 gain over the last quarter -> "still improving"
    batch: int = 128
    lr: float = 1e-3
    grad_clip: float = 5.0
    horizon: int = 15            # reward predicted k = 0..horizon steps ahead
    anchor_stride: int = 3
    ema_tau: float = 0.99        # arm D's target encoder
    w_recon: float = 1.0         # auxiliary weights are NOT tuned (see limitations)
    w_scene: float = 1.0
    w_latent: float = 1.0
    amp: bool = True             # mixed precision on GPU
    seeds: tuple = (0, 1, 2, 3)
    arms: tuple = ("A_recon", "B_task", "C_scene", "D_latent")
    ckpt_every: int = 5          # epochs between mid-run checkpoints
    snap_every: int = 5          # epochs between level-1 snapshots

    # ---- planner (CEM), identical for every arm and for the oracle ----
    plan_h: int = 10
    cem_samples: int = 500
    cem_elites: int = 50
    cem_iters: int = 6
    cem_init_std: float = 0.7
    cem_min_std: float = 0.05
    plan_chunk: int = 100        # episodes planned in parallel on the GPU
    run_shift: bool = True


PRESETS = {
    # Proves the notebook runs end to end, even on a CPU. NUMBERS ARE MEANINGLESS.
    "quick": dict(img_size=24, seq_len=20, plan_T=10, n_train=96, n_l1=40, n_probe=40,
                  n_plan=8, n_shift=6, latent=16, hidden=48, ch=8, reward_mlp=32,
                  epochs=4, min_epochs=2, patience_snaps=2, batch=32, horizon=5,
                  anchor_stride=2, seeds=(0,), ckpt_every=1, snap_every=1, plan_h=4,
                  cem_samples=32, cem_elites=6, cem_iters=2, plan_chunk=8, probe_grid=4, n_ceiling=300,
                  ceiling_steps=80, recon_eval_n=4, park_cem_samples=16, park_cem_iters=2, park_chunk=8),
    # Gate-first run at 32 px, 3 seeds. Arms stop when held-out reward R2 plateaus.
    "medium": dict(img_size=32, n_train=6000, epochs=120, ch=24, seeds=(0, 1, 2)),
    # The headline configuration: 48 px, 4 seeds, same stopping rule.
    "full": dict(),
}
CFG = Cfg(**PRESETS[SCALE])
assert CFG.plan_T <= CFG.seq_len - CFG.horizon, "planning episodes must fit training anchors"
assert CFG.img_size % 8 == 0, "encoder uses three stride-2 blocks"

if DEVICE != "cuda" and SCALE != "quick" and not os.environ.get("HERMES_ALLOW_CPU"):
    raise RuntimeError(
        "\n\nNo GPU detected. Runtime > Change runtime type > T4 GPU, then Run all again.\n"
        "(This run would take days on a CPU. Nothing has been written yet.)")

# ---- fixed seeds for the shared evaluation sets: identical for every arm and seed ----
L1_SEED, PLAN_SEED, SHIFT_SEED = 424242, 777, 31337

# ---- run folders on Drive ----
RUN_DIR = os.path.join(STORAGE_ROOT, RUN_NAME)
CKPT_DIR = os.path.join(RUN_DIR, "ckpt")          # model weights, mid-run and final
EVAL_DIR = os.path.join(RUN_DIR, "eval")          # cached evaluation outputs
DATA_DIR = os.path.join(RUN_DIR, "data")          # cached offline datasets (v2: parking costs GPU)
RESULTS = os.path.join(RUN_DIR, "results")        # CSVs, figures, GIFs, VERDICT.txt
for _d in (RUN_DIR, CKPT_DIR, EVAL_DIR, DATA_DIR, RESULTS):
    os.makedirs(_d, exist_ok=True)

# ---- config consistency: never silently mix two different experiments ----
_RESUME_SAFE_KEYS = {"seeds", "ckpt_every", "plan_chunk", "run_shift", "amp"}
_cfg_now = json.loads(json.dumps(asdict(CFG)))
_cfg_path = os.path.join(RUN_DIR, "config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path) as f:
        _cfg_old = json.load(f)
    _diff = {k: (_cfg_old.get(k), v) for k, v in _cfg_now.items()
             if k not in _RESUME_SAFE_KEYS and _cfg_old.get(k) != v}
    if _diff:
        raise RuntimeError(
            "\n\nThe run folder %s already holds a run with a DIFFERENT configuration:\n%s\n"
            "Either restore those settings, or set RUN_NAME to a new name to start a "
            "separate run." % (RUN_DIR, "\n".join(f"  {k}: saved={a}  now={b}"
                                                   for k, (a, b) in _diff.items())))
    print("config matches the existing run folder -> resuming where it stopped")
with open(_cfg_path, "w") as f:
    json.dump(_cfg_now, f, indent=2)

# ---- mixed precision: bfloat16 on Ampere+ GPUs, float16 + loss scaling on a T4 ----
if DEVICE == "cuda" and CFG.amp:
    AMP_DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    AMP_DTYPE = None


def autocast():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if AMP_DTYPE else contextlib.nullcontext()


def make_scaler():
    return torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))


# ---- disconnect-proof saving: write to a temp file, then atomically rename ----
def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def safe_load(path):
    """Load a checkpoint; a file truncated by a disconnect is reported and ignored."""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  !! unreadable file {os.path.basename(path)} ({type(e).__name__}): "
              "ignoring it and recomputing")
        return None


def mark_progress(stage, **info):
    """Human-readable progress log on Drive, so you can see where a run stopped."""
    path = os.path.join(RUN_DIR, "progress.json")
    prog = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                prog = json.load(f)
        except Exception:
            prog = {}
    prog[stage] = dict(info, time=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(path + ".tmp", "w") as f:
        json.dump(prog, f, indent=2)
    os.replace(path + ".tmp", path)


def savefig(fig, name):
    fig.savefig(os.path.join(RESULTS, name), bbox_inches="tight")
    print("saved " + name)
    show(fig)


PXS = CFG.img_size - 1          # arena units -> pixels

print("SCALE = %s    RUN_DIR = %s" % (SCALE, RUN_DIR))
if SCALE == "quick":
    print("!! QUICK smoke test: the numbers below are NOT results.")
print(json.dumps({k: str(v) for k, v in asdict(CFG).items()}, indent=1))
print(f"\narms x seeds = {len(CFG.arms)} x {len(CFG.seeds)} = "
      f"{len(CFG.arms) * len(CFG.seeds)} training runs")
_done = [f for f in os.listdir(CKPT_DIR) if f.endswith("_final.pt")]
_part = [f for f in os.listdir(CKPT_DIR) if f.endswith("_partial.pt")]
print(f"already on Drive: {len(_done)} finished runs, {len(_part)} half-finished runs")
mark_progress("config", scale=SCALE)

# %% [markdown]
# ## 2. Environment and data
#
# ### 2a. The world
#
# Everything is simulated as a few small arrays of **factors** (positions, colours) in
# numpy, and frames are rendered **on the GPU per batch**. Materialising 8000 × 60
# frames at 48×48×3 would be ~33 GB.
#
# * **Agent dynamics**: `v' = 0.95·v + 0.0025·a`, `p' = clip(p + v')`, with the velocity
#   component zeroed on wall contact. Top speed is 2.35 px/step and full braking takes
#   ~13.5 steps, so the planner must start braking early.
# * **Reward**: `r = −d + 0.5·exp(−d² / (2·0.05²))`, where `d` is the distance to the
#   **active** goal. Far away it is about `−d`; on the goal it is `+0.5`.
# * **Background**: hue is drawn from the warm band (ring active) or the cool band (plus
#   active). The **dead zones** 0.45–0.55 and 0.95–0.05 are never drawn: hue is circular,
#   so a threshold creates two boundaries where two almost identical colours would mean
#   opposite goals, and those episodes would be coin-flips no model could learn.
# * **Goals are black shapes**, not coloured. A fixed goal colour would vanish on a
#   same-hue background (Experiment 2's data figure shows a green distractor nearly
#   invisible on green). Black also keeps colour meaning exactly two things: the
#   background (relevant) and the distractors (irrelevant).
# * **Distractor hues** are full-saturation, so never white like the agent, and never
#   drawn from 0.60–0.72, which is reserved for the unseen-colour shift test.
#
# ### 2b. The offline data, and why it ignores the active goal
#
# The world model learns from a **recording** of a behaviour policy. Two tempting
# choices would break the experiment:
#
# * a driver that heads for the **active** goal lets the model read the answer off the
#   **actions** (which it receives as input) without ever looking at the background;
# * collecting data **online** with each arm's own planner gives every arm different
#   data, and destroys the paired comparison.
#
# So the data is recorded **once per seed, offline, identically for every arm**, by a
# driver that never looks at the background: 30% of episodes use smooth random actions,
# 70% chase waypoints that are the ring, the plus or a random point **by coin flip**. The
# leakage check below verifies that the active goal cannot be predicted from the
# movements alone.
# %%
def _hsv(h, s, v):
    return hsv_to_rgb(np.stack([h, s, v], -1)).astype(np.float32)


def sample_scene(n, T, seed, cfg, n_distract=None, speed_mult=1.0, holdout_hues=False):
    """Everything about an episode except the agent's actions.

    Independent random streams: [seed,0] for goals/background/agent start, [seed,1] for
    distractors. A shift condition that changes only the distractors therefore keeps
    every goal, background and start state identical, so the comparison stays paired.
    """
    rng = np.random.default_rng([seed, 0])
    rng_d = np.random.default_rng([seed, 1])
    nd = cfg.n_distract if n_distract is None else n_distract

    sel = rng.integers(0, 2, n)                              # 0 -> ring active, 1 -> plus
    lo = np.where(sel == 0, cfg.hue_band_ring[0], cfg.hue_band_plus[0])
    hi = np.where(sel == 0, cfg.hue_band_ring[1], cfg.hue_band_plus[1])
    bg_hsv = np.stack([rng.uniform(lo, hi), rng.uniform(*cfg.bg_sat, n),
                       rng.uniform(*cfg.bg_val, n)], -1).astype(np.float32)

    goals = rng.uniform(cfg.goal_lo, cfg.goal_hi, (n, 2, 2))
    bad = np.linalg.norm(goals[:, 0] - goals[:, 1], axis=-1) < cfg.goal_min_sep
    while bad.any():
        goals[bad] = rng.uniform(cfg.goal_lo, cfg.goal_hi, (int(bad.sum()), 2, 2))
        bad = np.linalg.norm(goals[:, 0] - goals[:, 1], axis=-1) < cfg.goal_min_sep

    p0 = rng.uniform(0.1, 0.9, (n, 2))
    ang = rng.uniform(0, 2 * np.pi, n)
    v0 = np.stack([np.cos(ang), np.sin(ang)], -1) * rng.uniform(0, cfg.v0_max, n)[:, None]

    # distractors: constant velocity, reflecting off all four walls
    dp = rng_d.uniform(0.1, 0.9, (n, nd, 2))
    dang = rng_d.uniform(0, 2 * np.pi, (n, nd))
    dv = (np.stack([np.cos(dang), np.sin(dang)], -1)
          * rng_d.uniform(*cfg.d_speed, (n, nd))[..., None] * speed_mult)
    u = rng_d.uniform(0, 1, (n, nd))
    lo_h, hi_h = cfg.d_hue_holdout
    if holdout_hues:
        dhue = lo_h + u * (hi_h - lo_h)
    else:                                   # uniform over the wheel minus the held-out band
        w = hi_h - lo_h
        dhue = u * (1 - w)
        dhue = np.where(dhue >= lo_h, dhue + w, dhue)
    dpos = np.zeros((n, T, nd, 2), np.float32)
    for t in range(T):
        dpos[:, t] = dp
        dp = dp + dv
        for _ in range(4):                  # loop so a fast object cannot tunnel out
            lo_w, hi_w = dp < 0, dp > 1
            if not (lo_w.any() or hi_w.any()):
                break
            dp = np.where(lo_w, -dp, dp)
            dp = np.where(hi_w, 2 - dp, dp)
            dv = np.where(lo_w | hi_w, -dv, dv)
    dpos = np.clip(dpos, 0, 1)

    return {"sel": sel, "bg_hsv": bg_hsv,
            "bg_rgb": _hsv(bg_hsv[:, 0], bg_hsv[:, 1], bg_hsv[:, 2]),
            "goals": goals.astype(np.float32), "p0": p0.astype(np.float32),
            "v0": v0.astype(np.float32), "dpos": dpos, "dhue": dhue.astype(np.float32),
            "drgb": _hsv(dhue, np.ones_like(dhue), np.ones_like(dhue))}


def agent_step_np(p, v, a, cfg):
    v = cfg.damping * v + cfg.max_accel * np.clip(a, -1, 1)
    p = p + v
    hit = (p < 0) | (p > 1)
    return np.clip(p, 0, 1), np.where(hit, 0.0, v)


def reward_np(p, g, cfg):
    d = np.linalg.norm(p - g, axis=-1)
    return -d + cfg.bonus * np.exp(-d ** 2 / (2 * cfg.bonus_sigma ** 2))


def active_goal(goals, sel):
    return goals[np.arange(len(sel)), sel]


def behaviour_pd(sc, T, seed, cfg):
    """Selector-independent driver: smooth random actions, or waypoint chasing.

    Convention: frame t shows p_t; action a_t moves p_t -> p_{t+1}; r_t = R(p_t).
    Returns p, v, a and a driver label (0 = smooth random, 1 = waypoints). Section 4b
    replaces a share of these episodes with planner-driven parking.
    """
    n = len(sc["sel"])
    rng = np.random.default_rng([seed, 2])
    p, v = sc["p0"].astype(np.float64), sc["v0"].astype(np.float64)
    goals = sc["goals"]
    mode_ou = rng.random(n) < cfg.p_ou
    ou = np.zeros((n, 2))

    def pick_waypoint(m):
        # ring, plus or a random point with equal probability. NEVER looks at `sel`.
        return rng.integers(0, 3, m), rng.uniform(0.1, 0.9, (m, 2))

    choice, rand_wp = pick_waypoint(n)
    t_next = rng.integers(*cfg.wp_switch, n)
    P = np.zeros((n, T, 2), np.float32)
    V = np.zeros_like(P)
    A = np.zeros_like(P)
    for t in range(T):
        P[:, t], V[:, t] = p, v
        sw = t >= t_next
        if sw.any():
            c, w = pick_waypoint(int(sw.sum()))
            choice[sw], rand_wp[sw] = c, w
            t_next[sw] = t + rng.integers(*cfg.wp_switch, int(sw.sum()))
        wp = np.where((choice == 2)[:, None], rand_wp,
                      goals[np.arange(n), np.minimum(choice, 1)])
        ou = ou - cfg.ou_theta * ou + cfg.ou_sigma * rng.standard_normal((n, 2))
        pd_a = cfg.kp * (wp - p) - cfg.kd * v + cfg.pd_noise * rng.standard_normal((n, 2))
        a = np.clip(np.where(mode_ou[:, None], ou, pd_a), -1, 1)
        A[:, t] = a
        p, v = agent_step_np(p, v, a, cfg)

    return P, V, A, np.where(mode_ou, 0, 1).astype(np.int64)


def to_torch(split):
    """CPU tensors (pinned when a GPU is present) for fast per-batch transfer."""
    out = {}
    for k, v in split.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        out[k] = t.pin_memory() if DEVICE == "cuda" else t
    return out


def render(agent, goals, dpos, drgb, bg_rgb, cfg, noise_std=None, generator=None):
    """Render frames on-device from factors. Vectorised over batch and time.

    agent (B,T,2)  goals (B,2,2)  dpos (B,T,nd,2)  drgb (B,nd,3)  bg_rgb (B,3)
    returns (B,T,3,S,S) in [0,1].

    ALPHA compositing (not additive): adding a blob to a bright background saturates
    every channel and turns coloured objects white. Draw order: background + noise,
    distractors, goals (never hidden by a distractor), agent (always on top).
    """
    dev = agent.device
    B, T = agent.shape[:2]
    S = cfg.img_size
    s = S / 48.0
    P = S - 1
    noise_std = cfg.pixel_noise if noise_std is None else noise_std
    g = torch.arange(S, device=dev, dtype=torch.float32)
    sig2 = 2.0 * (cfg.blob_sigma * s) ** 2

    img = bg_rgb.view(B, 1, 3, 1, 1).expand(B, T, 3, S, S)
    if noise_std > 0:
        img = img + noise_std * torch.randn((B, T, 3, S, S), device=dev, generator=generator)
    else:
        img = img.clone()

    nd = dpos.shape[2]
    if nd:
        cx = (dpos[..., 0] * P)[..., None, None]                    # (B,T,nd,1,1)
        cy = (dpos[..., 1] * P)[..., None, None]
        al = torch.exp(-((g.view(1, 1, 1, 1, S) - cx) ** 2
                         + (g.view(1, 1, 1, S, 1) - cy) ** 2) / sig2)
        for k in range(nd):
            a = al[:, :, k].unsqueeze(2)
            img = img * (1 - a) + drgb[:, k].view(B, 1, 3, 1, 1) * a

    gx, gy = g.view(1, 1, S), g.view(1, S, 1)
    cx, cy = (goals[:, 0, 0] * P).view(B, 1, 1), (goals[:, 0, 1] * P).view(B, 1, 1)
    rr = torch.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    a_ring = (1 - (rr - cfg.ring_radius * s).abs() / max(cfg.ring_width * s, 0.8)).clamp(0, 1)
    cx, cy = (goals[:, 1, 0] * P).view(B, 1, 1), (goals[:, 1, 1] * P).view(B, 1, 1)
    dx, dy = (gx - cx).abs(), (gy - cy).abs()
    L, hw = cfg.plus_arm * s, max(cfg.plus_halfwidth * s, 0.7)

    def bar(u, w):
        return (hw + 0.5 - u).clamp(0, 1) * (L + 0.5 - w).clamp(0, 1)

    a_plus = torch.maximum(bar(dy, dx), bar(dx, dy))
    img = img * (1 - a_ring.view(B, 1, 1, S, S)) * (1 - a_plus.view(B, 1, 1, S, S))

    cx = (agent[..., 0] * P)[..., None, None]                        # (B,T,1,1)
    cy = (agent[..., 1] * P)[..., None, None]
    a = torch.exp(-((g.view(1, 1, 1, S) - cx) ** 2
                    + (g.view(1, 1, S, 1) - cy) ** 2) / sig2).unsqueeze(2)
    img = img * (1 - a) + a
    return img.clamp(0.0, 1.0)


def object_masks(agent, goals, dpos, cfg):
    """Soft masks for each object type, on the geometry the renderer uses.

    Used to ask whether arm A's decoder actually draws the small objects, or only the
    background it cannot avoid.
    """
    dev = agent.device
    B, T = agent.shape[:2]
    S = cfg.img_size
    s = S / 48.0
    P = S - 1
    g = torch.arange(S, device=dev, dtype=torch.float32)
    sig2 = 2.0 * (cfg.blob_sigma * s) ** 2

    def blob(pos):
        cx = (pos[..., 0] * P)[..., None, None]
        cy = (pos[..., 1] * P)[..., None, None]
        return torch.exp(-((g.view(1, 1, 1, S) - cx) ** 2
                           + (g.view(1, 1, S, 1) - cy) ** 2) / sig2)

    gx, gy = g.view(1, 1, S), g.view(1, S, 1)
    cx, cy = (goals[:, 0, 0] * P).view(B, 1, 1), (goals[:, 0, 1] * P).view(B, 1, 1)
    rr = torch.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    a_ring = (1 - (rr - cfg.ring_radius * s).abs() / max(cfg.ring_width * s, 0.8)).clamp(0, 1)
    cx, cy = (goals[:, 1, 0] * P).view(B, 1, 1), (goals[:, 1, 1] * P).view(B, 1, 1)
    dx, dy = (gx - cx).abs(), (gy - cy).abs()
    L, hw = cfg.plus_arm * s, max(cfg.plus_halfwidth * s, 0.7)

    def bar(u, w):
        return (hw + 0.5 - u).clamp(0, 1) * (L + 0.5 - w).clamp(0, 1)

    goal_mask = torch.maximum(a_ring, torch.maximum(bar(dy, dx), bar(dx, dy)))
    dis = torch.zeros(B, T, S, S, device=dev)
    for k in range(dpos.shape[2]):
        dis = torch.maximum(dis, blob(dpos[:, :, k]))
    return {"agent": blob(agent), "goals": goal_mask.view(B, 1, S, S).expand(B, T, S, S),
            "distractors": dis}


def distractor_heatmap(dpos, cfg):
    """Identity-free target for arm C: a coarse Gaussian occupancy grid of the distractors.

    v1 asked arm C for each distractor's xy BY SLOT, but slots have no stable identity
    (colours are drawn at random), so the target was ambiguous and C never learned it. A
    heatmap sums over the objects, so there is no slot identity to get wrong.
    """
    G = cfg.probe_grid
    B, T = dpos.shape[:2]
    c = (torch.arange(G, device=dpos.device, dtype=torch.float32) + 0.5) / G
    s2 = 2.0 * (1.0 / G) ** 2
    gx = torch.exp(-(dpos[..., 0].unsqueeze(-1) - c.view(1, 1, 1, G)) ** 2 / s2)
    gy = torch.exp(-(dpos[..., 1].unsqueeze(-1) - c.view(1, 1, 1, G)) ** 2 / s2)
    return torch.einsum("btni,btnj->btij", gy, gx).reshape(B, T, G * G)


def render_np(split, idx, cfg, agent=None, seed=0, noise_std=None):
    """CPU rendering of a few episodes for figures and GIFs -> (n,T,S,S,3)."""
    gen = torch.Generator().manual_seed(seed)
    agent = split["p"][idx] if agent is None else agent
    T = agent.shape[1]
    with torch.no_grad():
        x = render(torch.from_numpy(np.ascontiguousarray(agent, dtype=np.float32)),
                   torch.from_numpy(split["goals"][idx]),
                   torch.from_numpy(np.ascontiguousarray(split["dpos"][idx][:, :T])),
                   torch.from_numpy(split["drgb"][idx]),
                   torch.from_numpy(split["bg_rgb"][idx]), cfg,
                   noise_std=noise_std, generator=gen)
    return x.permute(0, 1, 3, 4, 2).numpy()


# ---------------------------------------------------------------------------
# Leakage check: can the ACTIVE goal be predicted from movements alone?
# ---------------------------------------------------------------------------
def leakage_check(split, cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    p, a, g, sel = split["p"], split["a"], split["goals"], split["sel"]
    d0 = np.linalg.norm(p - g[:, None, 0], axis=-1)
    d1 = np.linalg.norm(p - g[:, None, 1], axis=-1)
    R = cfg.success_radius
    feats = np.column_stack([
        (d0 - d1).mean(1), (d0 - d1)[:, -1], d0.min(1) - d1.min(1),
        (d0 < R).mean(1) - (d1 < R).mean(1), (d0 < 3 * R).mean(1) - (d1 < 3 * R).mean(1),
        g.reshape(len(g), 4), np.abs(a).mean(1), a.mean(1), split["driver"]])
    X = StandardScaler().fit_transform(feats)
    cv = min(5, max(2, len(sel) // 20))
    acc_lin = cross_val_score(LogisticRegression(max_iter=2000), X, sel, cv=cv).mean()
    acc_gbm = cross_val_score(HistGradientBoostingClassifier(max_iter=100), X, sel, cv=cv).mean()
    d_act = np.where(sel[:, None] == 0, d0, d1)
    d_ina = np.where(sel[:, None] == 0, d1, d0)
    return {"n_episodes": len(sel), "acc_logistic": acc_lin, "acc_boosted_trees": acc_gbm,
            "chance": 0.5, "ci95_halfwidth": 1.96 * math.sqrt(0.25 / len(sel)),
            "frac_steps_near_active": float((d_act < R).mean()),
            "frac_steps_near_inactive": float((d_ina < R).mean())}

# %% [markdown]
# ## 3. Models
#
# Every arm shares this trunk:
#
# ```
# encoder : 3xSxS -> conv/2 -> conv/2 -> conv/2 -> Linear -> LayerNorm (no gain) -> z_t
# filter  : h_t     = GRU([z_t, a_{t-1}, 1], h_{t-1})        a real frame arrives
# imagine : h_{t+k} = GRU([0,   a_{t+k-1}, 0], h_{t+k-1})    no pixels; flag says "imagining"
# reward  : r_hat   = MLP(h)
# ```
#
# **Reward loss, identical for every arm.** From anchor steps across each episode, the
# model filters the real frames up to `t`, then imagines `k = 1..15` steps ahead using the
# **real** actions and predicts each reward. `k = 0` (the current reward) is included.
# Rewards are standardised with the training set's mean and standard deviation, so
# "always predict the mean" scores exactly MSE 1, R² 0. Every horizon is weighted equally.
#
# **What each arm adds** (weights fixed at 1.0, not tuned):
#
# * `A_recon`: a transposed-conv decoder redraws each frame from `z_t` (pixel MSE).
# * `B_task`: nothing. No decoder module exists.
# * `C_scene`: an MLP on `z_t` predicts **only irrelevant** factors: a coarse **distractor
#   occupancy heatmap** and background saturation and brightness. Experiment 2's version
#   predicted background **colour**, which here would directly supervise the goal selector
#   and hand C an unearned win, so it is excluded. v1 asked for each distractor's xy **by
#   slot**; slots have no stable identity, the target was ambiguous, and C's auxiliary loss
#   barely moved (it read distractors back at 0.03-0.35), so it was never a valid control.
#   A heatmap sums over the objects and has no slot identity to get wrong.
# * `D_latent`: from each imagined state, an MLP predicts what a slow-moving **EMA copy**
#   of the encoder will produce from the real future frame (stop-gradient target). This
#   is TD-MPC2's latent-consistency idea with a BYOL-style EMA target.
#
# **The arm-D fix from Experiment 2.** Its latent norm exploded by six orders of
# magnitude and saturated the GRU. The latent now passes through a **LayerNorm with no
# learnable gain**, in **every** arm (otherwise the trunks would differ). The gain has to
# go: a learnable gain would simply reopen the same escape route. Latent norm, per-dim
# spread and effective rank are logged every epoch so collapse is visible.
#
# **The GRU keep-state bias from Experiment 2 is kept.** A default GRU driven by weak
# input contracts to a fixed point and forgets its starting state within ~10 steps, which
# once left a run stuck at chance for 4000 steps. The retention check below measures how
# much of the starting state survives to the horizons actually used.
# %%
GRU_KEEP_BIAS = 3.0
TRUNK_EXCLUDE = ("decoder.", "scene_head.", "latent_pred.", "ema_encoder.")


def n_scene_targets(cfg):
    # v2: an identity-free distractor occupancy heatmap, plus background saturation/value.
    # v1 asked for each distractor's xy by slot, which is ambiguous, and C never learned it.
    return cfg.probe_grid ** 2 + 2


class Encoder(nn.Module):
    """Three stride-2 conv blocks, a linear map, and a gain-free LayerNorm."""

    def __init__(self, cfg):
        super().__init__()
        c = cfg.ch
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(c, 2 * c, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(2 * c, 4 * c, 4, 2, 1), nn.ReLU())
        self.fc = nn.Linear(4 * c * (cfg.img_size // 8) ** 2, cfg.latent)
        self.norm = nn.LayerNorm(cfg.latent, elementwise_affine=False)

    def forward(self, x):
        return self.norm(self.fc(self.net(x).flatten(1)))


class Decoder(nn.Module):
    """Mirror of the encoder, for arm A only."""

    def __init__(self, cfg):
        super().__init__()
        c = cfg.ch
        self.sp, self.c4 = cfg.img_size // 8, 4 * c
        self.fc = nn.Linear(cfg.latent, 4 * c * self.sp ** 2)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(4 * c, 2 * c, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(2 * c, c, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(c, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, z):
        return self.net(self.fc(z).view(-1, self.c4, self.sp, self.sp))


class WorldModel(nn.Module):
    """Encoder + action-conditioned GRU + reward head, plus one arm-specific extra."""

    def __init__(self, cfg, arm):
        super().__init__()
        assert arm in ("A_recon", "B_task", "C_scene", "D_latent")
        self.cfg, self.arm = cfg, arm

        # --- shared trunk: always built first, in this order, so identical seeding
        # --- gives every arm byte-identical starting weights (asserted below) ---
        self.encoder = Encoder(cfg)
        self.gru = nn.GRU(cfg.latent + 3, cfg.hidden, batch_first=True)
        with torch.no_grad():   # PyTorch packs biases [b_r | b_z | b_n]; push z toward "keep"
            self.gru.bias_ih_l0[cfg.hidden: 2 * cfg.hidden].fill_(GRU_KEEP_BIAS)
        self.reward_head = nn.Sequential(nn.Linear(cfg.hidden, cfg.reward_mlp), nn.ELU(),
                                         nn.Linear(cfg.reward_mlp, 1))

        # --- arm-specific extras: always built last ---
        self.decoder = Decoder(cfg) if arm == "A_recon" else None
        self.scene_head = (nn.Sequential(nn.Linear(cfg.latent, 128), nn.ELU(),
                                         nn.Linear(128, n_scene_targets(cfg)))
                           if arm == "C_scene" else None)
        self.latent_pred, self.ema_encoder = None, None
        if arm == "D_latent":
            self.latent_pred = nn.Sequential(nn.Linear(cfg.hidden, 128), nn.ELU(),
                                             nn.Linear(128, cfg.latent))
            self.ema_encoder = copy.deepcopy(self.encoder)
            for p in self.ema_encoder.parameters():
                p.requires_grad_(False)

    # ---- pieces ----------------------------------------------------------------
    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("ema_encoder.")]

    def encode(self, x):
        """(B,T,3,S,S) -> (B,T,latent). Frames are encoded independently."""
        B, T = x.shape[:2]
        return self.encoder(x.reshape(B * T, *x.shape[2:])).view(B, T, -1)

    def filter_seq(self, z, a_prev):
        """Real frames in: z (B,T,L), a_prev (B,T,2) -> h (B,T,hidden)."""
        inp = torch.cat([z, a_prev.to(z.dtype), torch.ones_like(z[..., :1])], -1)
        out, _ = self.gru(inp)
        return out

    def filter_step(self, z, a_prev, h):
        """One real frame: z (B,L), a_prev (B,2), h (B,hidden) -> (B,hidden)."""
        inp = torch.cat([z, a_prev.to(z.dtype), torch.ones_like(z[:, :1])], -1).unsqueeze(1)
        out, _ = self.gru(inp, h.to(inp.dtype).unsqueeze(0).contiguous())
        return out[:, 0]

    def imagine(self, h, actions):
        """No pixels: zero latent, flag 0. h (N,hidden), actions (N,K,2) -> (N,K,hidden)."""
        N, K = actions.shape[:2]
        zeros = torch.zeros(N, K, self.cfg.latent, device=actions.device, dtype=actions.dtype)
        inp = torch.cat([zeros, actions, torch.zeros_like(actions[..., :1])], -1)
        out, _ = self.gru(inp, h.to(inp.dtype).unsqueeze(0).contiguous())
        return out

    def reward(self, hs):
        return self.reward_head(hs).squeeze(-1)

    # ---- losses ----------------------------------------------------------------
    def losses(self, x, a, r_std, scene_tgt, anchors):
        """x (B,T,3,S,S), a (B,T,2), r_std (B,T), scene_tgt (B,T,n_scene), anchors (nA,).

        Returns (task_loss, aux_loss, per-horizon task MSE [k=0..H]).
        """
        cfg = self.cfg
        B, T = x.shape[:2]
        H, L = cfg.horizon, cfg.latent
        z = self.encode(x)
        a_prev = torch.cat([torch.zeros_like(a[:, :1]), a[:, :-1]], 1)
        hs = self.filter_seq(z, a_prev)
        r0 = self.reward(hs).float()                                  # k = 0

        offs = torch.arange(H, device=x.device)
        act_idx = anchors[:, None] + offs[None]                       # a_t .. a_{t+H-1}
        tgt_idx = act_idx + 1                                         # r_{t+1} .. r_{t+H}
        nA = anchors.numel()
        ho = self.imagine(hs[:, anchors].reshape(B * nA, -1),
                          a[:, act_idx].reshape(B * nA, H, 2))
        rk = self.reward(ho).float().view(B, nA, H)

        mse_k = torch.cat([((r0 - r_std) ** 2).mean().view(1),
                           ((rk - r_std[:, tgt_idx]) ** 2).mean(dim=(0, 1))])
        task = mse_k.mean()

        if self.arm == "A_recon":
            rec = self.decoder(z.reshape(B * T, L)).float()
            aux = ((rec - x.reshape(B * T, *x.shape[2:])) ** 2).mean() * cfg.w_recon
        elif self.arm == "C_scene":
            aux = ((self.scene_head(z).float() - scene_tgt) ** 2).mean() * cfg.w_scene
        elif self.arm == "D_latent":
            with torch.no_grad():
                zt = self.ema_encoder(x.reshape(B * T, *x.shape[2:])).float().view(B, T, L)
            pred = F.layer_norm(self.latent_pred(ho).float(), (L,)).view(B, nA, H, L)
            aux = ((pred - zt[:, tgt_idx]) ** 2).mean() * cfg.w_latent
        else:
            aux = torch.zeros((), device=x.device)
        return task, aux, mse_k.detach()

    @torch.no_grad()
    def update_ema(self):
        if self.ema_encoder is None:
            return
        tau = self.cfg.ema_tau
        for pe, p in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            pe.mul_(tau).add_(p.detach(), alpha=1 - tau)


def build_arms(cfg, seed):
    """One model per arm, each built right after re-seeding with the same value."""
    models = {}
    for arm in cfg.arms:
        torch.manual_seed(seed)
        models[arm] = WorldModel(cfg, arm).to(DEVICE)
    return models


def assert_same_trunk(models):
    """Fail loudly unless every arm starts from byte-identical shared weights."""
    names = list(models)
    ref = models[names[0]].state_dict()
    checked = 0
    for other in names[1:]:
        for n, t in models[other].state_dict().items():
            if n.startswith(TRUNK_EXCLUDE):
                continue
            if n not in ref or not torch.equal(ref[n], t):
                raise AssertionError(f"trunk weights differ at {n} ({other} vs {names[0]})")
            checked += 1
    return checked


# --- Can imagination carry information as far as we ask it to? ------------------
# Two copies of the GRU start from different hidden states and receive the SAME
# actions. How much of their difference survives k steps? If it dies, the rollout
# forgets what the frames said and every arm will predict the mean.
def rollout_retention(cfg, horizon, keep_bias=True, n=256):
    torch.manual_seed(0)
    m = WorldModel(cfg, "B_task").to(DEVICE)
    if not keep_bias:                          # PyTorch's default bias init, for comparison
        with torch.no_grad():
            b = 1.0 / math.sqrt(cfg.hidden)
            m.gru.bias_ih_l0[cfg.hidden: 2 * cfg.hidden].uniform_(-b, b)
    hA = torch.randn(n, cfg.hidden, device=DEVICE)
    hB = torch.randn(n, cfg.hidden, device=DEVICE)
    acts = torch.rand(n, horizon, 2, device=DEVICE) * 2 - 1
    with torch.no_grad():
        gap = (m.imagine(hA, acts) - m.imagine(hB, acts)).norm(dim=-1).mean(0)
    return (gap / (hA - hB).norm(dim=-1).mean()).cpu().numpy()


_Hmax = max(CFG.horizon, CFG.plan_h)
_ret_fix = rollout_retention(CFG, _Hmax, keep_bias=True)
_ret_def = rollout_retention(CFG, _Hmax, keep_bias=False)
pd.DataFrame({"step": np.arange(1, _Hmax + 1), "retention_keep_bias": _ret_fix,
              "retention_default_init": _ret_def}).to_csv(
    os.path.join(RESULTS, "retention.csv"), index=False)
print("imagination retention (fraction of the starting-state difference that survives):")
for k in sorted({1, 5, CFG.plan_h, CFG.horizon}):
    print(f"  step {k:2d}:  with keep-bias {_ret_fix[k-1]:.2f}   default init {_ret_def[k-1]:.2f}")
if _ret_fix[-1] < 0.10:
    print(f"  !! WARNING: under 10% survives {_Hmax} steps. Imagination will forget the frames")
    print("  !! and every arm will collapse to predicting the mean. Raise GRU_KEEP_BIAS.")
else:
    print(f"  ok: {100*_ret_fix[-1]:.0f}% survives to step {_Hmax}.")

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(range(1, _Hmax + 1), _ret_fix, "-o", ms=3, color="#2e86ab", label="keep-bias +3 (used)")
ax.plot(range(1, _Hmax + 1), _ret_def, "--", color="#999", label="default GRU init")
ax.axhline(0.10, color="#d1495b", lw=1, ls=":", label="10% warning line")
ax.set_xlabel("imagined steps"); ax.set_ylabel("retention"); ax.set_ylim(0, 1.05)
ax.set_title("Does imagination remember the starting state?"); ax.legend(fontsize=8)
savefig(fig, "fig03_retention.png")

_m = build_arms(CFG, 0)
print(f"\ntrunk verified byte-identical across {len(_m)} arms "
      f"({assert_same_trunk(_m)} tensor comparisons)")
for a, m in _m.items():
    tot = sum(p.numel() for p in m.trainable_parameters())
    trunk = sum(p.numel() for n, p in m.named_parameters() if not n.startswith(TRUNK_EXCLUDE))
    print(f"  {a:9s} trunk={trunk:>9,}  trainable={tot:>9,}  (+{tot-trunk:,} arm-specific)")
del _m

# %% [markdown]
# ## 4. The planner (CEM) and the planning environment
#
# **CEM in one breath.** Imagine a few hundred random 10-step action plans inside the
# model, score each by the rewards the model predicts, keep the best, sample new plans
# around them, repeat a few times, then execute only the **first** action, look at the
# next real frame, and plan again.
#
# **Held constant so the comparison is about the world model, not the planner:**
#
# * the same CEM settings for every arm **and** for the oracle;
# * the same 200 planning episodes (start state, goals, background, distractor motion);
# * the same pixel noise and the same CEM random numbers, seeded per episode chunk and
#   per step. Two arms given the same episode draw identical random numbers, so their
#   returns are paired even in the planner's randomness.
#
# **Baselines, and what each one proves:**
#
# | baseline | what it knows | why it is here |
# |---|---|---|
# | `noop`, `random` | nothing | the chance floor: an arm that cannot beat these learned nothing |
# | `oracle` | the true state, the true dynamics | the ceiling for **this** planner, so the gap splits into world-model quality vs planner quality |
# | `oracle_myopic` | the true state, but plans 1 step | proves momentum makes multi-step imagination necessary |
# | `oracle_colourblind` | the true state, but not which goal pays | proves the background genuinely matters; blanket colour-discarding should score like this |
# | `oracle_wrong` | heads for the inactive goal | the floor for a model that got selection backwards |
#
# Frames are rendered, encoded, filtered and planned **on the GPU**, with `plan_chunk`
# episodes and `cem_samples` plans each processed in one batch.
# %%
def agent_step_t(p, v, a, cfg):
    v = cfg.damping * v + cfg.max_accel * a.clamp(-1, 1)
    p = p + v
    hit = (p < 0) | (p > 1)
    return p.clamp(0, 1), torch.where(hit, torch.zeros_like(v), v)


def reward_t(p, g, cfg):
    d = (p - g).norm(dim=-1)
    return -d + cfg.bonus * torch.exp(-d ** 2 / (2 * cfg.bonus_sigma ** 2))


def cem(score_fn, mu, gen, cfg):
    """Cross-entropy method over action plans. mu (E,Hp,2) is the warm start."""
    E, Hp, _ = mu.shape
    N = cfg.cem_samples
    K = max(2, min(cfg.cem_elites, N // 2))
    std = torch.full_like(mu, cfg.cem_init_std)
    for _ in range(cfg.cem_iters):
        eps = torch.randn((E, N, Hp, 2), device=mu.device, generator=gen)
        A = (mu[:, None] + std[:, None] * eps).clamp(-1, 1)
        A[:, 0] = mu.clamp(-1, 1)                      # the current mean always competes
        top = score_fn(A).topk(K, dim=1).indices
        el = torch.gather(A, 1, top[:, :, None, None].expand(-1, -1, Hp, 2))
        mu, std = el.mean(1), el.std(1).clamp_min(cfg.cem_min_std)
    return mu


def model_score_fn(model, h):
    """Score plans by the sum of rewards the WORLD MODEL imagines. h (E,hidden)."""
    def score(A):
        E, N, Hp, _ = A.shape
        with autocast():
            r = model.reward(model.imagine(h.repeat_interleave(N, 0), A.reshape(E * N, Hp, 2)))
        return r.float().view(E, N, Hp).sum(-1)
    return score


def true_score_fn(p, v, targets, cfg):
    """Score plans with the TRUE dynamics and reward (oracles). targets: [(goal, weight)]."""
    def score(A):
        E, N, Hp, _ = A.shape
        pp, vv = p[:, None].expand(E, N, 2), v[:, None].expand(E, N, 2)
        R = torch.zeros(E, N, device=A.device)
        for k in range(Hp):
            pp, vv = agent_step_t(pp, vv, A[:, :, k], cfg)
            for g, w in targets:
                R = R + w * reward_t(pp, g[:, None], cfg)
        return R
    return score


CONTROLLERS = ("noop", "random", "oracle", "oracle_myopic", "oracle_colourblind", "oracle_wrong")


@torch.no_grad()
def run_planning(scene, cfg, kind, model=None, noise_std=None, n_record=0):
    """Play every episode in `scene` with one controller ("model" or a CONTROLLERS entry).

    Returns numpy arrays: rewards (E,T), traj (E,T+1,2), actions (E,T,2),
    pred_r (E,T,Hp) standardised rewards the model imagined for the plan it executed,
    and, for the first `n_record` episodes, the plans and imagined hidden states (GIFs).
    """
    E_all, Tp = len(scene["sel"]), cfg.plan_T
    Hp = 1 if kind == "oracle_myopic" else cfg.plan_h
    keep = {k: [] for k in ("rewards", "traj", "actions", "pred_r", "plan", "imag_h")}
    if model is not None:
        model.eval()
    for c0 in range(0, E_all, cfg.plan_chunk):
        sl = slice(c0, min(c0 + cfg.plan_chunk, E_all))
        E = sl.stop - sl.start
        T_ = lambda k: torch.from_numpy(np.ascontiguousarray(scene[k][sl])).to(DEVICE)
        goals, sel, dpos, drgb, bg = T_("goals"), T_("sel"), T_("dpos"), T_("drgb"), T_("bg_rgb")
        p, v = T_("p0"), T_("v0")
        ar = torch.arange(E, device=DEVICE)
        g_act, g_ina = goals[ar, sel], goals[ar, 1 - sel]
        targets = {"oracle": [(g_act, 1.0)], "oracle_myopic": [(g_act, 1.0)],
                   "oracle_wrong": [(g_ina, 1.0)],
                   "oracle_colourblind": [(goals[:, 0], 0.5), (goals[:, 1], 0.5)]}.get(kind)
        gen_plan = torch.Generator(device=DEVICE).manual_seed(7919 * PLAN_SEED + c0)
        gen_noise = torch.Generator(device=DEVICE).manual_seed(104729 * PLAN_SEED + c0)

        mu = torch.zeros(E, Hp, 2, device=DEVICE)
        a_prev = torch.zeros(E, 2, device=DEVICE)
        h = torch.zeros(E, cfg.hidden, device=DEVICE)
        R = torch.zeros(E, Tp, device=DEVICE)
        TR = torch.zeros(E, Tp + 1, 2, device=DEVICE)
        ACT = torch.zeros(E, Tp, 2, device=DEVICE)
        PR = torch.zeros(E, Tp, Hp, device=DEVICE)
        nrec = max(0, min(n_record - c0, E))
        PLAN = torch.zeros(nrec, Tp, Hp, 2, device=DEVICE)
        IH = torch.zeros(nrec, Tp, Hp, cfg.hidden if model is not None else 1, device=DEVICE)

        for t in range(Tp):
            TR[:, t] = p
            if kind == "model":
                x = render(p[:, None], goals, dpos[:, t:t + 1], drgb, bg, cfg,
                           noise_std=noise_std, generator=gen_noise)
                with autocast():
                    h = model.filter_step(model.encode(x)[:, 0], a_prev, h)
                h = h.float()
                mu = cem(model_score_fn(model, h), mu, gen_plan, cfg)
                with autocast():
                    hi = model.imagine(h, mu)
                    PR[:, t] = model.reward(hi).float()
                if nrec:
                    IH[:, t] = hi[:nrec].float()
            elif targets is not None:
                mu = cem(true_score_fn(p, v, targets, cfg), mu, gen_plan, cfg)
            if nrec:
                PLAN[:, t] = mu[:nrec]

            if kind == "random":
                a = torch.rand(E, 2, device=DEVICE, generator=gen_plan) * 2 - 1
            elif kind == "noop":
                a = torch.zeros(E, 2, device=DEVICE)
            else:
                a = mu[:, 0].clamp(-1, 1)
                mu = torch.cat([mu[:, 1:], torch.zeros_like(mu[:, :1])], 1)  # warm start
            p, v = agent_step_t(p, v, a, cfg)
            R[:, t] = reward_t(p, g_act, cfg)
            ACT[:, t] = a
            a_prev = a
        TR[:, Tp] = p

        for k, tens in (("rewards", R), ("traj", TR), ("actions", ACT), ("pred_r", PR),
                        ("plan", PLAN), ("imag_h", IH)):
            keep[k].append(tens.cpu().numpy().astype(np.float16 if k == "imag_h" else np.float32))
    return {k: np.concatenate(v) for k, v in keep.items()}


def planning_metrics(res, scene, cfg):
    traj, goals, sel = res["traj"], scene["goals"], scene["sel"]
    d_act = np.linalg.norm(traj[:, 1:] - active_goal(goals, sel)[:, None], axis=-1)
    d_ina = np.linalg.norm(traj[:, 1:] - active_goal(goals, 1 - sel)[:, None], axis=-1)
    return {"return": res["rewards"].mean(1),
            "success": (d_act[:, -1] < cfg.success_radius).astype(float),
            "time_in_radius": (d_act < cfg.success_radius).mean(1),
            "chose_active": (d_act[:, -1] < d_ina[:, -1]).astype(float),
            "final_dist_px": d_act[:, -1] * (cfg.img_size - 1)}

# %% [markdown]
# ## 4b. Building the offline dataset
#
# **What changed in v2.** The v1 driver wandered and chased waypoints with a PD
# controller. That never produced the slow, precise *approach and stop* that a CEM planner
# spends most of its time doing, so at evaluation the planner drove the model into states
# its training data barely covered. The v1 run showed it plainly: reward R² was 0.89 on the
# recorded data but went **negative** along the planner's own trajectories within a handful
# of steps, and every arm's imagined reward was systematically **optimistic** (bias almost
# equal to the whole error). The planner was exploiting model error rather than revealing
# which latent is better.
#
# So a share of episodes (`p_park`) is now driven by **the same CEM planner the evaluation
# uses**, with the true dynamics, aimed at a goal chosen by **coin flip**.
#
# * It parks precisely, and it uses the bang-bang style action sequences CEM produces, so
#   the training data covers the states and actions the evaluation planner visits.
# * It is aimed at a coin-flip goal, **not** the active one, so it still leaks nothing about
#   which goal pays. The leakage check below is run on the combined dataset and must sit at
#   chance.
#
# The rest of the episodes keep the v1 mix: smooth random actions, and waypoint chasing
# where each waypoint is the ring, the plus or a random point by coin flip.
#
# Datasets are **cached to Drive per seed**, because the parking episodes cost GPU time.
# %%
@torch.no_grad()
def collect_parking(scene, cfg, seed, sl=slice(None)):
    """Drive episodes with the evaluation planner toward a COIN-FLIP goal (never `sel`).

    Uses the true dynamics, so no model is involved and nothing is rendered: this is the
    same CEM code the evaluation uses, which is the point. Returns p, v, a and the goal
    index it aimed at (kept only for diagnostics).
    """
    goals_all = scene["goals"][sl]
    n, T = len(goals_all), cfg.seq_len
    gsel = np.random.default_rng([seed, 7]).integers(0, 2, n)      # independent of sel
    P = np.zeros((n, T, 2), np.float32)
    V = np.zeros_like(P)
    A = np.zeros_like(P)
    park_cfg = dc_replace(cfg, cem_samples=cfg.park_cem_samples, cem_iters=cfg.park_cem_iters,
                          cem_elites=max(2, cfg.park_cem_samples // 10))
    p0_all, v0_all = scene["p0"][sl], scene["v0"][sl]
    for c0 in range(0, n, cfg.park_chunk):
        c1 = min(c0 + cfg.park_chunk, n)
        E = c1 - c0
        goals = torch.from_numpy(goals_all[c0:c1]).to(DEVICE)
        tgt = goals[torch.arange(E, device=DEVICE),
                    torch.from_numpy(gsel[c0:c1]).to(DEVICE)]
        p = torch.from_numpy(p0_all[c0:c1]).to(DEVICE)
        v = torch.from_numpy(v0_all[c0:c1]).to(DEVICE)
        mu = torch.zeros(E, cfg.plan_h, 2, device=DEVICE)
        gen = torch.Generator(device=DEVICE).manual_seed(9176 * seed + c0)
        for t in range(T):
            P[c0:c1, t] = p.cpu().numpy()
            V[c0:c1, t] = v.cpu().numpy()
            mu = cem(true_score_fn(p, v, [(tgt, 1.0)], cfg), mu, gen, park_cfg)
            a = (mu[:, 0] + cfg.park_action_noise
                 * torch.randn(E, 2, device=DEVICE, generator=gen)).clamp(-1, 1)
            A[c0:c1, t] = a.cpu().numpy()
            p, v = agent_step_t(p, v, a, cfg)
            mu = torch.cat([mu[:, 1:], torch.zeros_like(mu[:, :1])], 1)
    return P, V, A, gsel


def make_split(n, T, seed, cfg, park=None):
    """Scene + behaviour. `driver`: 0 = smooth random, 1 = waypoints, 2 = planner parking."""
    sc = sample_scene(n, T, seed, cfg)
    P, V, A, driver = behaviour_pd(sc, T, seed, cfg)
    n_park = int(round((cfg.p_park if park is None else park) * n))
    if n_park > 0:
        sl = slice(0, n_park)                       # episodes are already in random order
        pp, vv, aa, _ = collect_parking(sc, cfg, seed, sl)
        P[sl], V[sl], A[sl] = pp, vv, aa
        driver[sl] = 2
    R = reward_np(P, active_goal(sc["goals"], sc["sel"])[:, None], cfg).astype(np.float32)
    sc.update({"p": P, "v": V, "a": A, "r": R, "driver": driver})
    return sc


SPLIT_KEYS = ("sel", "bg_hsv", "bg_rgb", "goals", "p0", "v0", "dpos", "dhue", "drgb",
              "p", "v", "a", "r", "driver")


def cached_split(n, T, seed, cfg, label):
    """Datasets are cached on Drive: the parking episodes are the expensive part."""
    path = os.path.join(DATA_DIR, f"{label}_seed{seed}_n{n}_T{T}.npz")
    if os.path.exists(path):
        try:
            with np.load(path) as z:
                return {k: z[k] for k in SPLIT_KEYS}
        except Exception as e:
            print(f"  !! unreadable dataset cache {os.path.basename(path)} "
                  f"({type(e).__name__}); regenerating")
    t0 = time.time()
    sp = make_split(n, T, seed, cfg)
    np.savez(path + ".tmp.npz", **{k: sp[k] for k in SPLIT_KEYS})
    os.replace(path + ".tmp.npz", path)
    print(f"  built {label} seed {seed}: {n} episodes x {T} steps "
          f"({100*cfg.p_park:.0f}% planner-parked) in {time.time()-t0:.0f}s -> cached", flush=True)
    return sp


# ---------------------------------------------------------------------------
# Look at the data, and check it leaks nothing, BEFORE training anything.
# ---------------------------------------------------------------------------
_look = cached_split(min(1000, CFG.n_train), CFG.seq_len, 99, CFG, "preview")
print(f"reward range [{_look['r'].min():+.3f}, {_look['r'].max():+.3f}]  "
      f"mean {_look['r'].mean():+.3f}  std {_look['r'].std():.3f}")
print(f"active goal = ring in {100*(_look['sel']==0).mean():.1f}% of episodes (want ~50)")
_d_act = np.linalg.norm(_look["p"] - active_goal(_look["goals"], _look["sel"])[:, None], axis=-1)
for name, code in (("smooth random", 0), ("waypoints", 1), ("planner parking", 2)):
    m = _look["driver"] == code
    if m.any():
        print(f"  {name:16s}: {100*m.mean():4.1f}% of episodes, "
              f"{100*(_d_act[m] < CFG.success_radius).mean():5.2f}% of steps within "
              f"{CFG.success_radius} of the ACTIVE goal")

_leak = leakage_check(_look, CFG)
pd.DataFrame([_leak]).to_csv(os.path.join(RESULTS, "leakage_check.csv"), index=False)
print("\nLEAKAGE CHECK  (predict the active goal from movements + goal positions, no pixels)")
print(f"  logistic regression : {_leak['acc_logistic']:.3f}")
print(f"  boosted trees       : {_leak['acc_boosted_trees']:.3f}")
print(f"  chance 0.500 +/- {_leak['ci95_halfwidth']:.3f} (95% band for this many episodes)")
print(f"  steps within {CFG.success_radius} of the active goal   : "
      f"{100*_leak['frac_steps_near_active']:.2f}%")
print(f"  steps within {CFG.success_radius} of the inactive goal : "
      f"{100*_leak['frac_steps_near_inactive']:.2f}%   (should match the line above)")
if max(_leak["acc_logistic"], _leak["acc_boosted_trees"]) > 0.5 + _leak["ci95_halfwidth"]:
    print("  !! WARNING: the behaviour policy leaks the active goal. A model could read it")
    print("  !! off the actions without looking at the background. Do not trust results.")
else:
    print("  ok: movements carry no detectable information about which goal pays.")
# %% [markdown]
# ### 4c. Look at the data before trusting anything downstream
#
# The **cyan square** marks the active goal; it is drawn on the figure only, never in the
# frames the models see. The coverage panels now also split by driver, so you can see that
# the planner-parked episodes are the ones that actually sit on a goal.
# %%
GOAL_BOX = 5.5 * CFG.img_size / 48.0     # half-width (px) of the figure-only active-goal marker


def goal_box(ax, g_px, lw=1.2):
    ax.add_patch(plt.Rectangle((g_px[0] - GOAL_BOX, g_px[1] - GOAL_BOX), 2 * GOAL_BOX,
                               2 * GOAL_BOX, fill=False, ec="#00e5ff", lw=lw))


def fig_frames_and_coverage():
    n_show, k_show = 4, 6
    idx = np.concatenate([np.flatnonzero(_look["driver"] == 2)[:2],
                          np.flatnonzero(_look["driver"] != 2)[:2]])[:n_show]
    frames = render_np(_look, idx, CFG, seed=5)
    ts = np.linspace(0, CFG.seq_len - 1, k_show).astype(int)
    ga_all = active_goal(_look["goals"], _look["sel"]) * PXS
    names = {0: "smooth random", 1: "waypoints", 2: "planner parking"}
    fig, axes = plt.subplots(n_show, k_show, figsize=(1.7 * k_show, 1.85 * n_show))
    for r, e in enumerate(idx):
        for c, t in enumerate(ts):
            ax = axes[r, c]
            ax.imshow(frames[r, t], interpolation="nearest")
            goal_box(ax, ga_all[e])
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if r == 0:
                ax.set_title(f"t={t}", fontsize=9)
        axes[r, 0].set_ylabel("%s\n%s active" % (names[_look["driver"][e]],
                                                 ("ring", "plus")[_look["sel"][e]]), fontsize=7.5)
    fig.suptitle("Offline data (cyan square = active goal, figure only)", y=1.0)
    fig.tight_layout()
    savefig(fig, "fig01_frames_grid.png")

    d_ina = np.linalg.norm(_look["p"] - active_goal(_look["goals"], 1 - _look["sel"])[:, None], axis=-1)
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    ax = axes[0]
    ax.hist2d(_look["p"][..., 0].ravel(), _look["p"][..., 1].ravel(), bins=40, cmap="viridis")
    ax.set_title("where the agent goes"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.invert_yaxis(); ax.grid(False)
    ax = axes[1]
    bins = np.linspace(0, 1.2, 50)
    ax.hist(_d_act.ravel(), bins, alpha=.6, label="to ACTIVE goal", color="#2e86ab")
    ax.hist(d_ina.ravel(), bins, alpha=.6, label="to INACTIVE goal", color="#d1495b")
    ax.set_title("distances (should overlap: no leakage)"); ax.legend(fontsize=8)
    ax = axes[2]
    ax.hist(_look["r"].ravel(), 60, color="#555")
    ax.set_title("reward distribution"); ax.set_xlabel("reward")
    ax = axes[3]
    for code, col, name in ((0, "#bbbbbb", "smooth random"), (1, "#5f8d4e", "waypoints"),
                            (2, "#2e86ab", "planner parking")):
        m = _look["driver"] == code
        if m.any():
            ax.hist(_d_act[m].ravel() * PXS, np.linspace(0, 20, 60), alpha=.55,
                    color=col, label=name, density=True)
    ax.axvline(CFG.success_radius * PXS, color="#d1495b", lw=1.2, ls=":")
    ax.set_xlabel("distance to active goal (px)"); ax.set_title("coverage near the goal, by driver")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig02_data_coverage.png")


def gif_dataset(path, n=4, fps=8):
    idx = np.concatenate([np.flatnonzero(_look["driver"] == 2)[:2],
                          np.flatnonzero(_look["driver"] != 2)[:2]])[:n]
    frames = render_np(_look, idx, CFG, seed=5)
    ga_all = active_goal(_look["goals"], _look["sel"]) * PXS
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 2.9))
    ims, txts = [], []
    for i, e in enumerate(idx):
        ax = axes[i]
        ims.append(ax.imshow(frames[i, 0], interpolation="nearest"))
        goal_box(ax, ga_all[e])
        txts.append(ax.text(0.5, -0.08, "", transform=ax.transAxes, ha="center", fontsize=9))
        ax.set_title(("random", "waypoints", "parking")[_look["driver"][e]], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    sup = fig.suptitle("")
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    def upd(t):
        for i, e in enumerate(idx):
            ims[i].set_data(frames[i, t])
            txts[i].set_text("reward %+.2f" % _look["r"][e, t])
        sup.set_text("Offline data - step %d/%d" % (t, CFG.seq_len - 1))
        return ims + txts + [sup]

    FuncAnimation(fig, upd, frames=CFG.seq_len, blit=False).save(
        path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("saved " + os.path.basename(path))


fig_frames_and_coverage()
if not os.path.exists(os.path.join(RESULTS, "gif01_dataset.gif")):
    gif_dataset(os.path.join(RESULTS, "gif01_dataset.gif"))

# %% [markdown]
# ## 5. Training — resumable
#
# **What happens.** For every seed, the offline data is generated (deterministically from
# the seed), every arm is built from byte-identical weights, and each arm trains on the
# **same batches in the same order with the same pixel noise**: batch order and noise are
# derived from `(seed, epoch)`, not from a running random state.
#
# **Checkpointing and resume.** Each `(seed, arm)` writes to `ckpt/` on Drive:
#
# * `seed{s}_{arm}_partial.pt` every `ckpt_every` epochs: weights, optimiser, loss
#   scaler, log. A disconnect costs at most `ckpt_every` epochs of that one arm.
# * `seed{s}_{arm}_final.pt` when the arm finishes: weights, training log, reward
#   statistics and config. The partial file is then deleted.
#
# Files are written to a temporary name and renamed, so a disconnect in the middle of a
# save can never leave a corrupt checkpoint behind. Because batches come from
# `(seed, epoch)`, a resumed arm sees exactly the batches it would have seen anyway.
#
# **Preflight.** Before any real training, every GPU code path (a training step for all
# four arms under mixed precision, model planning, oracle planning, reward prediction) is
# run once on a toy batch. A crash there costs seconds, not a crash hours in.
#
# **Measured time estimate.** After the first epoch of each arm, the cell prints the
# measured seconds per epoch and the remaining training time. Arms not yet measured are
# assumed to cost as much as the slowest measured one, and it says so.
#
# Printed every `snap_every` epochs: training reward MSE (standardised, averaged over all
# horizons), the auxiliary loss, held-out reward R² at 1 step and at the full horizon, and
# the latent's per-dimension spread (near 0 means the latent has collapsed).
# %%
from dataclasses import replace as dc_replace
import gc


def reward_stats(split):
    return {"r_mean": float(split["r"].mean()), "r_std": float(split["r"].std() + 1e-8)}


def data_fingerprint(split):
    """Detects data regenerated differently on resume (e.g. a numpy version change)."""
    return round(float(split["r"][:50].sum() + split["p"][:50].sum()), 3)


BATCH_KEYS = ("p", "a", "r", "goals", "dpos", "drgb", "bg_rgb", "bg_hsv", "dhue")


def get_batch(split_t, idx, cfg, stats, gen, noise_std=None):
    """Move one batch of factors to the GPU and render its frames there."""
    bt = {k: split_t[k][idx].to(DEVICE, non_blocking=True) for k in BATCH_KEYS}
    x = render(bt["p"], bt["goals"], bt["dpos"], bt["drgb"], bt["bg_rgb"], cfg,
               noise_std=noise_std, generator=gen)
    B, T = bt["p"].shape[:2]
    scene = torch.cat([distractor_heatmap(bt["dpos"], cfg),
                       bt["bg_hsv"][:, 1:].view(B, 1, 2).expand(B, T, 2)], -1)
    r_std = (bt["r"] - stats["r_mean"]) / stats["r_std"]
    return x, bt["a"], r_std, scene


def train_anchors(cfg, T=None):
    T = cfg.seq_len if T is None else T
    return torch.arange(0, T - cfg.horizon, cfg.anchor_stride, device=DEVICE)


@torch.no_grad()
def level1_predict(model, split_t, cfg, stats, n=None, batch=32, seed=0, keep_latents=0):
    """Reward prediction on held-out episodes given the TRUE actions.

    Returns standardised `pred` and `tgt` of shape (N, n_anchors, horizon+1), where
    column k is k steps ahead (k = 0 is the current reward). Optionally also keeps the
    encoder latent z and filtered GRU state h (every probe_stride-th frame) for probes.
    """
    model.eval()
    N = split_t["p"].shape[0] if n is None else min(n, split_t["p"].shape[0])
    T = split_t["p"].shape[1]
    anchors = train_anchors(cfg, T)
    nA, H = anchors.numel(), cfg.horizon
    act_idx = anchors[:, None] + torch.arange(H, device=DEVICE)[None]
    preds, tgts, Zs, Hs = [], [], [], []
    gen = torch.Generator(device=DEVICE)
    for i in range(0, N, batch):
        gen.manual_seed(seed + i)                     # identical pixels for every arm
        idx = torch.arange(i, min(i + batch, N))
        x, a, r_std, _ = get_batch(split_t, idx, cfg, stats, gen)
        B = x.shape[0]
        with autocast():
            z = model.encode(x)
            hs = model.filter_seq(z, torch.cat([torch.zeros_like(a[:, :1]), a[:, :-1]], 1))
            r0 = model.reward(hs[:, anchors]).float()
            ho = model.imagine(hs[:, anchors].reshape(B * nA, -1),
                               a[:, act_idx].reshape(B * nA, H, 2))
            rk = model.reward(ho).float().view(B, nA, H)
        preds.append(torch.cat([r0[..., None], rk], -1).cpu())
        tgts.append(torch.cat([r_std[:, anchors][..., None], r_std[:, act_idx + 1]], -1).cpu())
        if keep_latents > i:
            m = min(B, keep_latents - i)
            Zs.append(z[:m, ::cfg.probe_stride].float().cpu().half())
            Hs.append(hs[:m, ::cfg.probe_stride].float().cpu().half())
    out = {"pred": torch.cat(preds).numpy(), "tgt": torch.cat(tgts).numpy(),
           "anchors": anchors.cpu().numpy()}
    if Zs:
        out["z"], out["h"] = torch.cat(Zs).numpy(), torch.cat(Hs).numpy()
    return out


def r2_per_k(pred, tgt):
    """R^2 and MSE for each horizon column, pooled over episodes and anchors."""
    k = pred.shape[-1]
    mse = ((pred - tgt) ** 2).reshape(-1, k).mean(0)
    return 1.0 - mse / tgt.reshape(-1, k).var(0), mse


@torch.no_grad()
def latent_health(model, x_fixed):
    """Latent norm, mean per-dimension spread, and effective rank on a fixed batch."""
    with autocast():
        z = model.encode(x_fixed).float().reshape(-1, model.cfg.latent)
    s = torch.linalg.svdvals(z - z.mean(0))
    p = s / (s.sum() + 1e-12)
    eff_rank = torch.exp(-(p * torch.log(p + 1e-12)).sum())
    return float(z.norm(dim=-1).mean()), float(z.std(0).mean()), float(eff_rank)


def ckpt_paths(seed, arm):
    return (os.path.join(CKPT_DIR, f"seed{seed}_{arm}_final.pt"),
            os.path.join(CKPT_DIR, f"seed{seed}_{arm}_partial.pt"))


EPOCH_SECONDS = {}


def print_eta(seed, arm, epochs_done):
    slowest = max(EPOCH_SECONDS.values())
    remaining = 0.0
    for s in CFG.seeds:
        for a in CFG.arms:
            if os.path.exists(ckpt_paths(s, a)[0]):
                continue
            left = CFG.epochs - (epochs_done if (s, a) == (seed, arm) else 0)
            remaining += EPOCH_SECONDS.get(a, slowest) * left
    unmeasured = [a for a in CFG.arms if a not in EPOCH_SECONDS]
    print(f"    measured {EPOCH_SECONDS[arm]:.1f} s/epoch for {arm}  ->  training still to do "
          f"~{remaining/60:.0f} min (~{remaining/3600:.1f} h)"
          + (f"; {', '.join(unmeasured)} not measured yet, assumed as slow as the slowest"
             if unmeasured else ""), flush=True)


def learning_speed(arm, seed, log, snaps, cfg):
    """How long this arm needed, and whether it had actually stopped improving.

    v1 trained every arm for a fixed 40 epochs while arm A was still improving steeply, so
    its numbers described an unfinished run. Speed is now measured, not assumed.
    """
    hist = [(s["epoch"], float(s["r2_k"][-1])) for s in snaps]
    reached = [e for e, v in hist if v > 0.8]
    q = max(1, len(hist) // 4)
    gain = hist[-1][1] - hist[-1 - q][1] if len(hist) > q else float("nan")
    return {"arm": arm, "seed": seed, "epochs_trained": log[-1]["epoch"],
            "max_epochs": cfg.epochs, "stopped_early": bool(log[-1]["epoch"] < cfg.epochs),
            "epochs_to_r2_0.8": float(reached[0]) if reached else float("nan"),
            "final_r2_kH": hist[-1][1], "best_r2_kH": max(v for _, v in hist),
            "r2_gain_last_quarter": gain,
            "converged": bool(gain == gain and gain < cfg.converged_delta),
            "minutes": round(sum(r["sec"] for r in log) / 60, 2)}


def train_arm(model, arm, seed, train_t, l1_t, cfg, stats, fingerprint):
    final_path, partial_path = ckpt_paths(seed, arm)
    opt = torch.optim.Adam(model.trainable_parameters(), lr=cfg.lr)
    scaler = make_scaler()
    log, snaps, start = [], [], 1
    best_r2, since_improve = -9.9, 0

    blob = safe_load(partial_path)
    if blob is not None and blob.get("fingerprint") != fingerprint:
        print("    !! half-finished checkpoint was trained on different data; restarting this arm")
        blob = None
    if blob is not None:
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        scaler.load_state_dict(blob["scaler"])
        log, snaps, start = blob["log"], blob["snaps"], blob["epoch"] + 1
        best_r2, since_improve = blob.get("best_r2", -9.9), blob.get("since_improve", 0)
        print(f"    resuming {arm} from epoch {blob['epoch']} (half-finished checkpoint)")

    anchors = train_anchors(cfg)
    n = train_t["p"].shape[0]
    n_batches = n // cfg.batch
    gen = torch.Generator(device=DEVICE)
    x_fixed = get_batch(l1_t, torch.arange(min(16, l1_t["p"].shape[0])), cfg, stats,
                        torch.Generator(device=DEVICE).manual_seed(4242))[0]

    for ep in range(start, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        perm = np.random.default_rng([seed, ep]).permutation(n)   # same for every arm
        gen.manual_seed(seed * 100003 + ep)                       # same pixel noise too
        s_task = torch.zeros((), device=DEVICE)
        s_aux = torch.zeros((), device=DEVICE)
        s_k = torch.zeros(cfg.horizon + 1, device=DEVICE)
        bad = 0
        for b in range(n_batches):
            idx = torch.from_numpy(perm[b * cfg.batch:(b + 1) * cfg.batch])
            x, a, r_std, scene = get_batch(train_t, idx, cfg, stats, gen)
            with autocast():
                task, aux, mse_k = model.losses(x, a, r_std, scene, anchors)
                loss = task + aux
            if not torch.isfinite(loss):
                bad += 1
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            model.update_ema()
            s_task += task.detach()
            s_aux += aux.detach()
            s_k += mse_k
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        sec = time.time() - t0
        nb = max(n_batches - bad, 1)
        zn, zs, er = latent_health(model, x_fixed)
        s_k = (s_k / nb).cpu().numpy()
        row = {"arm": arm, "seed": seed, "epoch": ep, "task_mse": float(s_task) / nb,
               "aux": float(s_aux) / nb, "mse_k0": s_k[0], "mse_k1": s_k[1],
               "mse_kH": s_k[-1], "z_norm": zn, "z_dim_std": zs, "z_eff_rank": er,
               "nonfinite_batches": bad, "sec": sec}
        log.append(row)

        stop = False
        if ep == 1 or ep % cfg.snap_every == 0 or ep == cfg.epochs:
            out = level1_predict(model, l1_t, cfg, stats, n=128, seed=999)
            r2, _ = r2_per_k(out["pred"], out["tgt"])
            snaps.append({"epoch": ep, "r2_k": r2})
            score = float(r2[-1])
            if score > best_r2 + cfg.min_delta:
                best_r2, since_improve = score, 0
            else:
                since_improve += 1
            stop = since_improve >= cfg.patience_snaps and ep >= cfg.min_epochs
            print(f"    ep {ep:3d}  task {row['task_mse']:.4f}  aux {row['aux']:.4f}  "
                  f"held-out R2 k=1 {r2[1]:+.3f}  k={cfg.horizon} {r2[-1]:+.3f}  "
                  f"z-dim-spread {zs:.3f}  ({sec:.1f}s)"
                  + (f"  !! {bad} non-finite batches skipped" if bad else ""), flush=True)
        if ep == start:
            EPOCH_SECONDS[arm] = max(EPOCH_SECONDS.get(arm, 0.0), sec)
            print_eta(seed, arm, ep)
        if (ep % cfg.ckpt_every == 0 or stop) and ep < cfg.epochs:
            atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                         "scaler": scaler.state_dict(), "epoch": ep, "log": log,
                         "snaps": snaps, "fingerprint": fingerprint,
                         "best_r2": best_r2, "since_improve": since_improve}, partial_path)
        if stop:
            print(f"    plateau: held-out R2 gained less than {cfg.min_delta} for "
                  f"{cfg.patience_snaps} snapshots -> stopping at epoch {ep} "
                  f"(ceiling {cfg.epochs})", flush=True)
            break

    speed = learning_speed(arm, seed, log, snaps, cfg)
    atomic_save({"model": model.state_dict(), "log": log, "snaps": snaps, "stats": stats,
                 "fingerprint": fingerprint, "arm": arm, "seed": seed, "cfg": asdict(cfg),
                 "speed": speed}, final_path)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    return log, snaps, speed


def preflight(train_t, stats):
    """Exercise every GPU code path once on a toy batch before committing hours."""
    t0 = time.time()
    tiny = dc_replace(CFG, cem_samples=16, cem_elites=4, cem_iters=2, plan_T=3, plan_chunk=2)
    models = build_arms(CFG, 12345)
    anchors = train_anchors(CFG)
    x, a, r_std, scene = get_batch(train_t, torch.arange(2), CFG, stats,
                                   torch.Generator(device=DEVICE).manual_seed(0))
    for arm, m in models.items():
        opt = torch.optim.Adam(m.trainable_parameters(), lr=CFG.lr)
        scaler = make_scaler()
        with autocast():
            task, aux, _ = m.losses(x, a, r_std, scene, anchors)
            loss = task + aux
        assert torch.isfinite(loss), f"preflight: non-finite loss for {arm}"
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        scaler.step(opt)
        scaler.update()
        m.update_ema()
    sc = sample_scene(2, tiny.plan_T + 1, 1, tiny)
    run_planning(sc, tiny, "model", model=models["B_task"], n_record=1)
    run_planning(sc, tiny, "oracle", n_record=1)
    level1_predict(models["A_recon"], train_t, CFG, stats, n=2, keep_latents=2)
    del models
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print(f"preflight ok: a training step for all {len(CFG.arms)} arms, model and oracle "
          f"planning, reward prediction ({time.time()-t0:.1f}s)\n")


# ---------------------------------------------------------------------------
# The sweep: every seed x every arm. Re-running this cell skips finished work.
# ---------------------------------------------------------------------------
l1_split = cached_split(CFG.n_l1, CFG.seq_len, L1_SEED, CFG, "l1")   # held out, shared
l1_t = to_torch(l1_split)
all_logs, all_snaps, STATS, SPEED = [], [], {}, []
t_sweep, preflight_done = time.time(), False

for seed in CFG.seeds:
    need = [a for a in CFG.arms if not os.path.exists(ckpt_paths(seed, a)[0])]
    train_split = cached_split(CFG.n_train, CFG.seq_len, 1000 * seed + 1, CFG, "train")
    STATS[seed] = reward_stats(train_split)
    fp = data_fingerprint(train_split)
    train_t = to_torch(train_split) if need else None
    if need and not preflight_done:
        preflight(train_t, STATS[seed])
        preflight_done = True

    models = build_arms(CFG, seed + 10)
    n_chk = assert_same_trunk(models)
    print(f"seed {seed}: trunk byte-identical across arms ({n_chk} tensors); "
          f"{len(need)} of {len(CFG.arms)} arms still to train")

    for arm in CFG.arms:
        blob = safe_load(ckpt_paths(seed, arm)[0])
        if blob is not None:
            if blob.get("fingerprint") != fp:
                print(f"  !! seed {seed} {arm}: checkpoint data fingerprint differs from "
                      "regenerated data (numpy version change?). Using the checkpoint anyway.")
            all_logs += blob["log"]
            all_snaps += [dict(s, arm=arm, seed=seed) for s in blob["snaps"]]
            SPEED.append(blob.get("speed")
                         or learning_speed(arm, seed, blob["log"], blob["snaps"], CFG))
            STATS[seed] = blob["stats"]
            print(f"  [seed {seed}] {arm:9s} finished earlier -> loaded from Drive")
            continue
        print(f"  [seed {seed}] training {arm} ...", flush=True)
        log, snaps, speed = train_arm(models[arm], arm, seed, train_t, l1_t, CFG,
                                      STATS[seed], fp)
        all_logs += log
        all_snaps += [dict(s, arm=arm, seed=seed) for s in snaps]
        SPEED.append(speed)
        mark_progress(f"trained_seed{seed}_{arm}", epochs=CFG.epochs,
                      minutes=round(sum(r["sec"] for r in log) / 60, 1))
        print(f"  [seed {seed}] {arm} done, saved to Drive", flush=True)

    del models, train_t, train_split
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

log_df = pd.DataFrame(all_logs)
log_df.to_csv(os.path.join(RESULTS, "training_log.csv"), index=False)
snap_df = pd.DataFrame([{"seed": s["seed"], "arm": s["arm"], "epoch": s["epoch"],
                         "horizon": k, "r2": float(v)}
                        for s in all_snaps for k, v in enumerate(s["r2_k"])])
snap_df.to_csv(os.path.join(RESULTS, "training_level1_snapshots.csv"), index=False)
speed_df = pd.DataFrame(SPEED)
speed_df.to_csv(os.path.join(RESULTS, "learning_speed.csv"), index=False)
print("\nHOW LONG EACH ARM NEEDED  (v2 trains to a plateau, not a fixed length)")
print(speed_df[["arm", "seed", "epochs_trained", "stopped_early", "epochs_to_r2_0.8",
                "final_r2_kH", "r2_gain_last_quarter", "converged", "minutes"]]
      .sort_values(["arm", "seed"]).to_string(index=False))
if (~speed_df.converged).any():
    print("  !! these arms were STILL IMPROVING when they hit the epoch ceiling:")
    for _, r in speed_df[~speed_df.converged].iterrows():
        print(f"     seed {r.seed} {r.arm}: R2 still rising by {r.r2_gain_last_quarter:+.3f} "
              "over the last quarter -> raise CFG.epochs before comparing this arm")
mark_progress("training_done")
print(f"\nTRAINING DONE ({(time.time()-t_sweep)/60:.1f} min in this session)")

# %% [markdown]
# ## 6. Evaluation — resumable, cached on Drive
#
# Every expensive result is saved to `eval/` the moment it is computed, and re-running
# loads it instead. Nothing here trains anything; every arm is loaded from its final
# checkpoint.
#
# **Shared evaluation sets** (identical for every arm and every seed):
#
# * **Level 1 set**: 500 held-out behaviour-policy episodes. Reward prediction given the
#   true actions, plus the latents used by the probes.
# * **Planning set**: 200 fixed episodes. Each arm plans with CEM in its own latent and we
#   record the **real** reward it collects.
# * **On-policy check**: reward prediction re-run along the planner's **own** trajectories.
#   The planner visits states the behaviour policy rarely did (parking precisely on a
#   goal), so this measures how well each model holds up where it is actually used.
# * **Distribution shift** (bonus): 100 episodes per condition, with goals, background and
#   start states identical across conditions, so only the distractors or noise change:
#
# | condition | change |
# |---|---|
# | `in_distribution` | none (the reference) |
# | `4_distractors` | twice as many distractors |
# | `fast_distractors` | distractors move twice as fast |
# | `unseen_distractor_hues` | distractor hues from 0.60–0.72, never seen in training |
# | `heavy_pixel_noise` | pixel noise 0.05 → 0.15 |
#
# The background and goals are never shifted: that would change the task, not the
# distraction.
# %%
plan_scene = sample_scene(CFG.n_plan, CFG.plan_T + 1, PLAN_SEED, CFG)
SHIFTS = {"in_distribution": {}, "4_distractors": {"n_distract": 4},
          "fast_distractors": {"speed_mult": 2.0},
          "unseen_distractor_hues": {"holdout_hues": True},
          "heavy_pixel_noise": {"noise_std": 0.15}}
SHIFT_SCENES = {}
if CFG.run_shift:
    for _name, _kw in SHIFTS.items():
        _kw = {k: v for k, v in _kw.items() if k != "noise_std"}
        SHIFT_SCENES[_name] = sample_scene(CFG.n_shift, CFG.plan_T + 1, SHIFT_SEED, CFG, **_kw)


def cached(path, fn, label):
    blob = safe_load(path)
    if blob is not None:
        print(f"  {label:44s} loaded from Drive")
        return blob
    t0 = time.time()
    blob = fn()
    atomic_save(blob, path)
    print(f"  {label:44s} computed and saved ({time.time()-t0:.0f}s)", flush=True)
    return blob


print("baselines on the planning set:")
BASE = {k: cached(os.path.join(EVAL_DIR, f"baseline_{k}.pt"),
                  lambda k=k: run_planning(plan_scene, CFG, k, n_record=3), k)
        for k in CONTROLLERS}
BASE_SHIFT = {}
if CFG.run_shift:
    # True-state controllers never see pixels, and the shift scenes share goals,
    # backgrounds and starts, so one run on the reference scenes serves every condition.
    print("baselines on the shift episodes:")
    BASE_SHIFT = {k: cached(os.path.join(EVAL_DIR, f"baseline_shift_{k}.pt"),
                            lambda k=k: run_planning(SHIFT_SCENES["in_distribution"], CFG, k),
                            k)
                  for k in ("noop", "random", "oracle")}
mark_progress("baselines_done")


@torch.no_grad()
def recon_fidelity(model, split_t, cfg, stats, seed=4242):
    """Does arm A's decoder DRAW the small objects, or only the background?

    For each object type: the contrast it has against the background in the real frame,
    and the contrast the reconstruction gives it. A ratio near 1 means the object is drawn;
    near 0 means reconstruction is ignoring it, which is how a "reconstruct everything" arm
    can still fail to encode the goals or the distractors. v1 could not tell these apart.
    """
    n = min(cfg.recon_eval_n, split_t["p"].shape[0])
    idx = torch.arange(n)
    x, _, _, _ = get_batch(split_t, idx, cfg, stats,
                           torch.Generator(device=DEVICE).manual_seed(seed))
    with autocast():
        z = model.encode(x)
        rec = model.decoder(z.reshape(-1, cfg.latent)).float().view_as(x)
    bt = {k: split_t[k][idx].to(DEVICE) for k in ("p", "goals", "dpos", "bg_rgb")}
    masks = object_masks(bt["p"], bt["goals"], bt["dpos"], cfg)
    # Contrast is measured against each image's OWN background, not the true background
    # colour: otherwise a decoder that emits flat grey scores well for objects it never
    # drew, which is exactly what the first version of this check did.
    empty = (1 - torch.clamp(masks["agent"] + masks["goals"] + masks["distractors"], 0, 1))
    w_bg = (empty > 0.95).float().unsqueeze(2)

    def own_background(img):
        return ((img * w_bg).sum((-1, -2), keepdim=True)
                / (w_bg.sum((-1, -2), keepdim=True) + 1e-6))

    bg_real, bg_rec = own_background(x), own_background(rec)
    out = {"mse_all": float(((rec - x) ** 2).mean())}
    for name, m in masks.items():
        w = m.unsqueeze(2)
        tot = float(w.sum()) + 1e-6
        c_real = float((w * (x - bg_real).abs().mean(2, keepdim=True)).sum()) / tot
        c_rec = float((w * (rec - bg_rec).abs().mean(2, keepdim=True)).sum()) / tot
        out[f"contrast_real_{name}"] = c_real
        out[f"contrast_recon_{name}"] = c_rec
        out[f"contrast_ratio_{name}"] = c_rec / (c_real + 1e-6)
        out[f"mse_{name}"] = float((w * (rec - x) ** 2).sum()) / (3 * tot)
    step = max(1, x.shape[1] // 4)
    out["examples_real"] = x[:3, ::step].cpu().numpy().astype(np.float16)
    out["examples_recon"] = rec[:3, ::step].cpu().numpy().astype(np.float16)
    return out


def load_trained(seed, arm):
    blob = safe_load(ckpt_paths(seed, arm)[0])
    if blob is None:
        raise RuntimeError(f"missing final checkpoint for seed {seed} {arm}: re-run section 5")
    m = WorldModel(CFG, arm).to(DEVICE)
    m.load_state_dict(blob["model"])
    m.eval()
    return m, blob["stats"]


def trajectory_split(res, scene, cfg):
    """Turn planner trajectories into a split, for on-policy reward prediction."""
    T = cfg.plan_T
    p = res["traj"][:, :T]
    ga = active_goal(scene["goals"], scene["sel"])
    sp = {k: scene[k] for k in ("goals", "drgb", "bg_rgb", "bg_hsv", "dhue")}
    sp.update({"p": p.astype(np.float32), "a": res["actions"].astype(np.float32),
               "r": reward_np(p, ga[:, None], cfg).astype(np.float32),
               "dpos": np.ascontiguousarray(scene["dpos"][:, :T])})
    return sp


EVAL = {}
for seed in CFG.seeds:
    for arm in CFG.arms:
        path = os.path.join(EVAL_DIR, f"seed{seed}_{arm}.pt")
        ev = safe_load(path) or {}
        want = ["level1", "planning", "onpolicy"] + (["recon"] if arm == "A_recon" else [])
        todo = [k for k in want if k not in ev]
        shift_todo = [s for s in SHIFTS if s not in ev.get("shift", {})] if CFG.run_shift else []
        if not todo and not shift_todo:
            EVAL[(seed, arm)] = ev
            print(f"[seed {seed}] {arm:9s} evaluation loaded from Drive")
            continue
        print(f"[seed {seed}] {arm:9s} evaluating: {', '.join(todo + (['shift'] if shift_todo else []))}",
              flush=True)
        model, stats = load_trained(seed, arm)
        t0 = time.time()
        if "level1" not in ev:
            ev["level1"] = level1_predict(model, l1_t, CFG, stats, seed=L1_SEED,
                                          keep_latents=CFG.n_probe)
            atomic_save(ev, path)
        if "planning" not in ev:
            ev["planning"] = run_planning(plan_scene, CFG, "model", model=model, n_record=3)
            atomic_save(ev, path)
            print(f"    planning return {ev['planning']['rewards'].mean():+.3f} per step "
                  f"({time.time()-t0:.0f}s so far)", flush=True)
        if "onpolicy" not in ev:
            ev["onpolicy"] = level1_predict(
                model, to_torch(trajectory_split(ev["planning"], plan_scene, CFG)), CFG,
                stats, seed=PLAN_SEED)
            atomic_save(ev, path)
        if "recon" in todo:
            ev["recon"] = recon_fidelity(model, l1_t, CFG, stats)
            atomic_save(ev, path)
            print("    reconstruction contrast ratio  agent %.2f  goals %.2f  distractors %.2f"
                  % (ev["recon"]["contrast_ratio_agent"], ev["recon"]["contrast_ratio_goals"],
                     ev["recon"]["contrast_ratio_distractors"]), flush=True)
        for name in shift_todo:
            ev.setdefault("shift", {})[name] = run_planning(
                SHIFT_SCENES[name], CFG, "model", model=model,
                noise_std=SHIFTS[name].get("noise_std"))
            atomic_save(ev, path)
        EVAL[(seed, arm)] = ev
        mark_progress(f"evaluated_seed{seed}_{arm}")
        print(f"    done ({time.time()-t0:.0f}s), saved to Drive", flush=True)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# Reference for the probes: the SHARED untrained starting weights. Every arm began here, so
# "what did training add or remove" is read against this, not against zero. It matters: a
# random conv encoder already carries a lot of background colour, which dominates the pixels.
EVAL_INIT = {}
for seed in CFG.seeds:
    path = os.path.join(EVAL_DIR, f"seed{seed}_init.pt")
    blob = safe_load(path)
    if blob is None:
        torch.manual_seed(seed + 10)                    # exactly how build_arms seeds the trunk
        m0 = WorldModel(CFG, "B_task").to(DEVICE)
        blob = {"level1": level1_predict(m0, l1_t, CFG, STATS[seed], seed=L1_SEED,
                                         keep_latents=CFG.n_probe)}
        atomic_save(blob, path)
        del m0
    EVAL_INIT[seed] = blob
print("untrained-encoder reference latents ready")

mark_progress("evaluation_done")
print("\nplanning return (mean reward per step) on the shared planning set:")
for k in CONTROLLERS:
    print(f"  {ARM_LABEL[k]:40s} {BASE[k]['rewards'].mean():+.3f}")
for arm in CFG.arms:
    vals = [EVAL[(s, arm)]["planning"]["rewards"].mean() for s in CFG.seeds]
    print(f"  {ARM_LABEL[arm]:40s} " + "  ".join(f"{v:+.3f}" for v in vals) + "   (per seed)")

# %% [markdown]
# ## 7. Analysis and the learning gate
#
# Everything in this section is cheap: it reads the cached evaluations and writes CSVs.
# Sign convention in **every** paired table: **positive means B (reward-only) is worse**.
#
# ### 7a. Level 1: reward prediction versus horizon
#
# R² per horizon `k` (0 = current reward, 15 = fifteen imagined steps ahead) on the 500
# held-out behaviour episodes, given the true actions. References:
#
# * `chance`: always predict the training mean (R² = 0 by construction);
# * `hold_current`: repeat the true current reward for every future step (the analogue of
#   Experiment 2's hold-last);
# * `colourblind_true`: the **true** future positions, but averaging the reward of both
#   goals. A strong reference: it knows everything except which goal pays.
# %%
H = CFG.horizon
ARMS = list(CFG.arms)
SEEDS = list(CFG.seeds)


def colourblind_true_pred(split, anchors, stats, cfg):
    idx = anchors[:, None] + np.arange(cfg.horizon + 1)[None]         # (nA, H+1)
    pos = split["p"][:, idx]                                          # (N, nA, H+1, 2)
    g = split["goals"]
    r = 0.5 * (reward_np(pos, g[:, None, None, 0], cfg) + reward_np(pos, g[:, None, None, 1], cfg))
    return ((r - stats["r_mean"]) / stats["r_std"]).astype(np.float32)


l1_rows, L1_EP = [], {}
for seed in SEEDS:
    ref = EVAL[(seed, ARMS[0])]["level1"]
    tgt, anchors = ref["tgt"], ref["anchors"]
    series = {a: EVAL[(seed, a)]["level1"]["pred"] for a in ARMS}
    series["chance"] = np.zeros_like(tgt)
    series["hold_current"] = np.repeat(tgt[..., :1], H + 1, axis=-1)
    series["colourblind_true"] = colourblind_true_pred(l1_split, anchors, STATS[seed], CFG)
    for name, pred in series.items():
        r2, mse = r2_per_k(pred, tgt)
        spread = pred.reshape(-1, H + 1).std(0) / (tgt.reshape(-1, H + 1).std(0) + 1e-8)
        for k in range(H + 1):
            l1_rows.append({"seed": seed, "model": name, "horizon": k, "mse": float(mse[k]),
                            "r2": float(r2[k]), "pred_spread_ratio": float(spread[k])})
        L1_EP[(seed, name)] = ((pred - tgt) ** 2).mean(1)             # (N, H+1)

l1_df = pd.DataFrame(l1_rows)
l1_df.to_csv(os.path.join(RESULTS, "level1_reward_error.csv"), index=False)
l1_sum = l1_df.groupby(["model", "horizon"])[["r2", "mse"]].agg(["mean", "std"]).reset_index()
l1_sum.columns = ["model", "horizon", "r2_mean", "r2_std", "mse_mean", "mse_std"]
l1_sum.to_csv(os.path.join(RESULTS, "level1_summary.csv"), index=False)

l1_paired = []
for seed in SEEDS:
    base = L1_EP[(seed, "B_task")]
    for arm in [a for a in ARMS if a != "B_task"]:
        diff = base - L1_EP[(seed, arm)]                              # > 0 => B worse
        for k in range(H + 1):
            l1_paired.append({"seed": seed, "vs": arm, "horizon": k,
                              "delta_mse": float(diff[:, k].mean()),
                              "b_worse_frac": float((diff[:, k] > 0).mean())})
l1_paired_df = pd.DataFrame(l1_paired)
l1_paired_df.to_csv(os.path.join(RESULTS, "level1_paired.csv"), index=False)

deg_rows = []
for (m, s), g in l1_df.groupby(["model", "seed"]):
    g = g.set_index("horizon")
    deg_rows.append({"model": m, "seed": s, "mse_k1": g.loc[1, "mse"], "mse_kH": g.loc[H, "mse"],
                     "mse_ratio": g.loc[H, "mse"] / max(g.loc[1, "mse"], 1e-8),
                     "r2_k1": g.loc[1, "r2"], "r2_kH": g.loc[H, "r2"],
                     "r2_drop": g.loc[1, "r2"] - g.loc[H, "r2"]})
deg_df = pd.DataFrame(deg_rows)
deg_df.to_csv(os.path.join(RESULTS, "level1_degradation.csv"), index=False)

onpol_rows = []
for seed in SEEDS:
    for arm in ARMS:
        lv, op = EVAL[(seed, arm)]["level1"], EVAL[(seed, arm)]["onpolicy"]
        r2_off, mse_off = r2_per_k(lv["pred"], lv["tgt"])
        r2_on, mse_on = r2_per_k(op["pred"], op["tgt"])
        # R2 on planner trajectories is unfair: rewards vary less there, so the denominator
        # shrinks. RMSE in raw reward units is the honest comparison, so report both.
        rs = STATS[seed]["r_std"]
        for k in range(H + 1):
            onpol_rows.append({"seed": seed, "arm": arm, "horizon": k,
                               "r2_behaviour_data": float(r2_off[k]),
                               "r2_planner_trajectories": float(r2_on[k]),
                               "rmse_raw_behaviour_data": float(np.sqrt(mse_off[k]) * rs),
                               "rmse_raw_planner_trajectories": float(np.sqrt(mse_on[k]) * rs)})
onpol_df = pd.DataFrame(onpol_rows)
onpol_df.to_csv(os.path.join(RESULTS, "level1_onpolicy.csv"), index=False)

_show_k = sorted({0, 1, 5, 10, H} & set(range(H + 1)))
print("Level 1: reward-prediction R² (mean over seeds) by horizon")
print(l1_sum[l1_sum.horizon.isin(_show_k)].pivot(index="horizon", columns="model",
                                                 values="r2_mean").round(3).to_string())
# %% [markdown]
# ### 7b. Level 2: planning return — the headline metric
#
# Mean real reward per step over each planning episode. Also: the **normalised score**
# `(return − floor) / (oracle − floor)` where the floor is the better of no-op and random;
# **success** (ends within 2.35 px of the active goal); **goal-choice accuracy** (ends
# closer to the active than the inactive goal; 50% means it cannot tell them apart).
#
# **Optimism** compares the rewards the model *imagined* for the action it executed with
# the reward that actually arrived. A model the planner can exploit shows large positive
# bias.
# %%
def metrics_frame(res, scene, model, seed, condition):
    m = planning_metrics(res, scene, CFG)
    return pd.DataFrame({"seed": seed, "model": model, "condition": condition,
                         "episode": np.arange(len(m["return"])), **m})


def boot_ci(d, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


plan_ep = pd.concat([metrics_frame(BASE[k], plan_scene, k, -1, "planning") for k in CONTROLLERS]
                    + [metrics_frame(EVAL[(s, a)]["planning"], plan_scene, a, s, "planning")
                       for s in SEEDS for a in ARMS], ignore_index=True)
plan_ep.to_csv(os.path.join(RESULTS, "planning_episodes.csv"), index=False)

FLOOR = max(("noop", "random"), key=lambda k: BASE[k]["rewards"].mean())
R_FLOOR, R_ORACLE = BASE[FLOOR]["rewards"].mean(), BASE["oracle"]["rewards"].mean()
_cols = ["return", "success", "time_in_radius", "chose_active", "final_dist_px"]
plan_seed = plan_ep.groupby(["model", "seed"])[_cols].mean().reset_index()
plan_seed["normalised"] = (plan_seed["return"] - R_FLOOR) / (R_ORACLE - R_FLOOR)
plan_seed.to_csv(os.path.join(RESULTS, "planning_per_seed.csv"), index=False)
plan_sum = plan_seed.groupby("model")[_cols + ["normalised"]].agg(["mean", "std"])
plan_sum.columns = [f"{a}_{b}" for a, b in plan_sum.columns]
plan_sum = plan_sum.reset_index()
plan_sum.to_csv(os.path.join(RESULTS, "planning_summary.csv"), index=False)

plan_paired = []
for seed in SEEDS:
    rb = EVAL[(seed, "B_task")]["planning"]["rewards"].mean(1)
    for arm in [a for a in ARMS if a != "B_task"]:
        d = EVAL[(seed, arm)]["planning"]["rewards"].mean(1) - rb      # > 0 => B worse
        lo, hi = boot_ci(d)
        plan_paired.append({"seed": seed, "vs": arm, "delta_return": float(d.mean()),
                            "ci95_lo": float(lo), "ci95_hi": float(hi),
                            "b_worse_frac": float((d > 0).mean())})
plan_paired_df = pd.DataFrame(plan_paired)
plan_paired_df.to_csv(os.path.join(RESULTS, "planning_paired.csv"), index=False)

opt_rows = []
for seed in SEEDS:
    st = STATS[seed]
    for arm in ARMS:
        res = EVAL[(seed, arm)]["planning"]
        pred = res["pred_r"] * st["r_std"] + st["r_mean"]                  # (E,T,Hp) raw units
        real = res["rewards"]                                               # (E,T)
        Tp, Hp = real.shape[1], pred.shape[2]
        plan_sum_pred = pred[:, :Tp - Hp + 1].sum(-1)
        plan_sum_real = np.stack([real[:, t:t + Hp].sum(1) for t in range(Tp - Hp + 1)], 1)
        opt_rows.append({"seed": seed, "arm": arm,
                         "one_step_bias": float((pred[:, :, 0] - real).mean()),
                         "one_step_mae": float(np.abs(pred[:, :, 0] - real).mean()),
                         "plan_sum_bias": float((plan_sum_pred - plan_sum_real).mean())})
opt_df = pd.DataFrame(opt_rows)
opt_df.to_csv(os.path.join(RESULTS, "planning_optimism.csv"), index=False)

print(f"floor = {FLOOR} ({R_FLOOR:+.3f})   oracle = {R_ORACLE:+.3f}\n")
print("Planning (mean over seeds; baselines have one run):")
print(plan_sum.set_index("model")[["return_mean", "return_std", "normalised_mean", "success_mean",
                                   "chose_active_mean"]].round(3).to_string())
# %% [markdown]
# ### 7c. The learning gate — read this before any comparison
#
# Experiment 2's gate compared only against hold-last, and a broken arm **passed** it
# while emitting a constant. So every `(seed, arm)` must now pass **all** of:
#
# | check | why |
# |---|---|
# | reward R² at k=1 > 0.5 | it learned to read the current frame at all |
# | reward R² at k=15 > 0 | beats the **chance floor** (predict the mean) at the full horizon |
# | MSE at k=15 below `hold_current` | imagination adds something over repeating the present |
# | prediction spread ≥ 0.3 × true spread | a **constant predictor** has zero spread, however clean its curve looks |
# | planning return above the floor, 95% CI excluding zero | the latent actually steers the agent |
# | latent per-dim spread > 0.05 | the latent has not collapsed (arm D's risk) |
#
# Also flagged, as a warning: error that **falls** with horizon, which a real dynamics
# model cannot do and a constant predictor can (arm D's tell in Experiment 2).
# %%
gate_rows = []
_last = log_df.sort_values("epoch").groupby(["seed", "arm"]).tail(1).set_index(["seed", "arm"])
_speed_ix = speed_df.set_index(["seed", "arm"])
for seed in SEEDS:
    g = l1_df[l1_df.seed == seed].set_index(["model", "horizon"])
    floor_ep = BASE[FLOOR]["rewards"].mean(1)
    for arm in ARMS:
        lo, _ = boot_ci(EVAL[(seed, arm)]["planning"]["rewards"].mean(1) - floor_ep)
        checks = {
            "r2_k1>0.5": g.loc[(arm, 1), "r2"] > 0.5,
            f"r2_k{H}>0": g.loc[(arm, H), "r2"] > 0,
            "beats_hold_at_H": g.loc[(arm, H), "mse"] < g.loc[("hold_current", H), "mse"],
            "not_constant": g.loc[(arm, 1), "pred_spread_ratio"] >= 0.3,
            "return>floor": lo > 0,
            "latent_ok": _last.loc[(seed, arm), "z_dim_std"] > 0.05,
        }
        gate_rows.append({"seed": seed, "arm": arm, **{k: bool(v) for k, v in checks.items()},
                          "passed": bool(all(checks.values())),
                          "converged": bool(_speed_ix.loc[(seed, arm), "converged"]),
                          "epochs_trained": int(_speed_ix.loc[(seed, arm), "epochs_trained"]),
                          "warn_error_falls_with_horizon":
                              bool(g.loc[(arm, H), "mse"] < 0.9 * g.loc[(arm, 1), "mse"]),
                          "r2_k1": g.loc[(arm, 1), "r2"], f"r2_k{H}": g.loc[(arm, H), "r2"],
                          "return_ci95_lo_vs_floor": lo})
gate_df = pd.DataFrame(gate_rows)
gate_df.to_csv(os.path.join(RESULTS, "learning_gate.csv"), index=False)
FAILED_ARMS = sorted(gate_df.loc[~gate_df.passed, "arm"].unique().tolist())
# v2: claims are made on the seeds where BOTH arms passed, instead of dropping an arm
# entirely because one seed failed.
PASS_SEEDS = {a: set(gate_df.loc[(gate_df.arm == a) & gate_df.passed, "seed"]) for a in ARMS}


def paired_seeds(a, b):
    return sorted(PASS_SEEDS[a] & PASS_SEEDS[b])

print("LEARNING GATE")
print(gate_df.drop(columns=["return_ci95_lo_vs_floor"]).to_string(index=False))
for _, r in gate_df[gate_df.warn_error_falls_with_horizon].iterrows():
    print(f"  !! WARNING seed {r.seed} {r.arm}: error FALLS with horizon - "
          "the signature of a constant predictor")
for _, r in gate_df[~gate_df.converged].iterrows():
    print(f"  !! WARNING seed {r.seed} {r.arm}: hit the epoch ceiling while still improving "
          "- its numbers describe an unfinished run")
print("  seeds passing the gate per arm: "
      + ",  ".join(f"{SHORT[a]} {sorted(PASS_SEEDS[a])}" for a in ARMS))
if FAILED_ARMS:
    print("\n" + "!" * 78)
    print(f"!! GATE FAILED for: {', '.join(FAILED_ARMS)}")
    print("!! In at least one seed these arms did not learn. Their numbers are NOT")
    print("!! interpretable and are excluded from every claim in the verdict. Report them")
    print("!! as failed arms, not as findings about their objectives.")
    if SCALE == "quick":
        print("!! (Expected: SCALE='quick' trains for 2 epochs. Nothing here is a result.)")
    else:
        print("!! Likely causes: too few epochs, learning rate, or (arm D) latent collapse.")
    print("!" * 78)
else:
    print("\nAll arms passed in every seed: the comparisons below are meaningful.")
# %% [markdown]
# ### 7d. Level 3: what is in each latent? (the selective-relevance test)
#
# Probes read quantities back out of the encoder latent `z` (one frame) and the filtered
# GRU state `h` (the frame history). Probes are trained on half the **episodes** and scored
# on the other half. Experiment 2 split by frame, which puts frames of the same episode on
# both sides and inflates scores for per-episode constants such as background colour.
#
# Two probe types: **linear** (ridge) and a small **MLP** (one hidden layer). A quantity
# can be present but not *linearly* readable. For example, a model that stores both goals
# plus the selector bit holds the active goal as a product of the two, which a linear probe
# under-reads.
#
# | group | targets | selective hypothesis for B |
# |---|---|---|
# | relevant | agent position and velocity, active goal, agent→active-goal vector, **selector bit** (balanced accuracy) | kept |
# | goal-like but irrelevant | inactive goal, agent→inactive-goal vector | dropped |
# | identity | ring position, plus position | each is the active goal half the time |
# | irrelevant | distractor positions and hues, background saturation/brightness, hue within its band | dropped |
#
# **The `init` reference.** Every probe is also run on the **untrained** shared starting
# weights. A random conv encoder already carries plenty of background colour (it dominates
# the pixels), so "B reads the selector" means little on its own. What matters is what
# training **added or removed** relative to `init`.
#
# **Selectivity** = R²(active goal) − R²(inactive goal). A model that encodes goal-like
# things without caring which one pays scores about 0. A model that encodes only what
# reward needs scores high.
#
# Honest caveat: a perfectly selective model may use the selector bit inside the encoder
# and keep only the selected goal, so a *low* selector-bit score does **not** mean the model
# ignores colour. Read the probes together with goal-choice accuracy from planning.
# %%
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import balanced_accuracy_score

REG_TARGETS = ["agent_position", "agent_velocity", "active_goal", "agent_to_active_goal",
               "inactive_goal", "agent_to_inactive_goal", "ring_position", "plus_position",
               "distractor_positions", "distractor_heatmap", "distractor_hues",
               "background_sat_val", "background_hue_within_band"]
TARGET_GROUP = {"agent_position": "relevant", "agent_velocity": "relevant",
                "active_goal": "relevant", "agent_to_active_goal": "relevant",
                "selector_bit": "relevant", "inactive_goal": "goal-like irrelevant",
                "agent_to_inactive_goal": "goal-like irrelevant", "ring_position": "identity",
                "plus_position": "identity", "distractor_positions": "irrelevant",
                "distractor_heatmap": "irrelevant",
                "distractor_hues": "irrelevant", "background_sat_val": "irrelevant",
                "background_hue_within_band": "irrelevant"}


def _heatmap_np(dpos, cfg):
    """The same identity-free distractor target arm C is trained on (numpy version)."""
    G = cfg.probe_grid
    c = (np.arange(G, dtype=np.float32) + 0.5) / G
    s2 = 2.0 * (1.0 / G) ** 2
    hx = np.exp(-(dpos[..., 0][..., None] - c) ** 2 / s2)
    hy = np.exp(-(dpos[..., 1][..., None] - c) ** 2 / s2)
    return np.einsum("btni,btnj->btij", hy, hx).reshape(dpos.shape[0], dpos.shape[1], G * G)


def probe_targets(split, n, stride, cfg):
    s = slice(None, None, stride)
    p, v = split["p"][:n, s], split["v"][:n, s]
    Tf = p.shape[1]
    goals, sel = split["goals"][:n], split["sel"][:n]
    rep = lambda x: np.repeat(np.asarray(x, np.float32)[:, None], Tf, 1)
    ga, gi = active_goal(goals, sel), active_goal(goals, 1 - sel)
    ang = 2 * np.pi * split["dhue"][:n]
    band_lo = np.where(sel == 0, cfg.hue_band_ring[0], cfg.hue_band_plus[0])
    band_w = cfg.hue_band_ring[1] - cfg.hue_band_ring[0]
    raw = {"agent_position": p, "agent_velocity": v * 20.0,
           "active_goal": rep(ga), "agent_to_active_goal": p - rep(ga),
           "inactive_goal": rep(gi), "agent_to_inactive_goal": p - rep(gi),
           "ring_position": rep(goals[:, 0]), "plus_position": rep(goals[:, 1]),
           "distractor_positions": split["dpos"][:n, s].reshape(n, Tf, -1),
           "distractor_heatmap": _heatmap_np(split["dpos"][:n, s], cfg),
           "distractor_hues": rep(np.concatenate([np.cos(ang), np.sin(ang)], 1)),
           "background_sat_val": rep(split["bg_hsv"][:n, 1:]),
           "background_hue_within_band": rep(((split["bg_hsv"][:n, 0] - band_lo) / band_w)[:, None]),
           "selector_bit": rep(sel[:, None])}
    return {k: v_.reshape(n * Tf, -1).astype(np.float32) for k, v_ in raw.items()}, Tf


def _r2(y, pred):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-12))


def run_probes(Z, Y, train_mask, seed=0, steps=600):
    """Linear (ridge / logistic) and MLP probes on one latent space. Returns {(probe,target): score}."""
    out = {}
    mu, sd = Z[train_mask].mean(0), Z[train_mask].std(0) + 1e-6
    Zs = (Z - mu) / sd
    Ztr, Zte = Zs[train_mask], Zs[~train_mask]
    for k in REG_TARGETS:
        pred = Ridge(alpha=1.0).fit(Ztr, Y[k][train_mask]).predict(Zte)
        out[("linear", k)] = _r2(Y[k][~train_mask], pred)
    ysel = Y["selector_bit"][:, 0].astype(int)
    if len(np.unique(ysel[train_mask])) > 1:
        clf = LogisticRegression(max_iter=2000).fit(Ztr, ysel[train_mask])
        out[("linear", "selector_bit")] = float(balanced_accuracy_score(ysel[~train_mask], clf.predict(Zte)))
    else:
        out[("linear", "selector_bit")] = float("nan")

    # one multi-output MLP per latent space, trained on the GPU
    Yreg = np.concatenate([Y[k] for k in REG_TARGETS], 1)
    ym, ys = Yreg[train_mask].mean(0), Yreg[train_mask].std(0) + 1e-6
    # Each target group gets equal weight. Pooling one MSE lets a 64-dim heatmap drown a
    # 2-dim position: measured on the CNN ceiling, ring position went 0.00 -> 0.52 when
    # this was fixed.
    bounds, _c = [], 0
    for k in REG_TARGETS:
        bounds.append((_c, _c + Y[k].shape[1]))
        _c += Y[k].shape[1]
    dev = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(DEVICE)
    xtr, xte = dev(Ztr), dev(Zte)
    ytr, ysl = dev((Yreg[train_mask] - ym) / ys), dev(ysel[train_mask])
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Z.shape[1], 128), nn.ReLU(),
                        nn.Linear(128, Yreg.shape[1] + 1)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        o = net(xtr)
        loss = (torch.stack([F.mse_loss(o[:, a:b], ytr[:, a:b]) for a, b in bounds]).mean()
                + F.binary_cross_entropy_with_logits(o[:, -1], ysl))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        o = net(xte).cpu().numpy()
    pred = o[:, :-1] * ys + ym
    col = 0
    for k in REG_TARGETS:
        d = Y[k].shape[1]
        out[("mlp", k)] = _r2(Y[k][~train_mask], pred[:, col:col + d])
        col += d
    out[("mlp", "selector_bit")] = float(balanced_accuracy_score(ysel[~train_mask], (o[:, -1] > 0).astype(int)))
    return out


def cnn_ceiling(cfg, n=None, steps=None, seed=20250915, batch=128):
    """What a conv encoder of the SAME architecture reads from the frame when trained to.

    This is the probe's positive control. v1 had none, so targets that read 0.00 from every
    latent (distractor positions, the inactive goal) could not be told apart from targets
    nothing could read. A target the ceiling cannot read is untestable, not "dropped".
    """
    n, steps = n or cfg.n_ceiling, steps or cfg.ceiling_steps
    sc = sample_scene(n, 1, seed, cfg)
    pseudo = dict(sc, p=sc["p0"][:, None], v=np.zeros((n, 1, 2), np.float32))
    Y, _ = probe_targets(pseudo, n, 1, cfg)
    names = [k for k in REG_TARGETS if k != "agent_velocity"]     # single frame: no velocity
    Yc = np.concatenate([Y[k] for k in names] + [Y["selector_bit"]], 1).astype(np.float32)
    frames = []
    for i in range(0, n, 256):
        s = slice(i, min(i + 256, n))
        d = lambda k: torch.from_numpy(np.ascontiguousarray(sc[k][s])).to(DEVICE)
        with torch.no_grad():
            frames.append(render(d("p0")[:, None], d("goals"), d("dpos"), d("drgb"),
                                 d("bg_rgb"), cfg,
                                 generator=torch.Generator(device=DEVICE).manual_seed(seed + i))[:, 0])
    X = torch.cat(frames)
    tr = np.zeros(n, bool)
    tr[np.random.default_rng(0).permutation(n)[: n // 2]] = True
    ym, ys = Yc[tr].mean(0), Yc[tr].std(0) + 1e-6
    Xtr, Xte = X[tr], X[~tr]
    Ytr = torch.from_numpy((Yc[tr] - ym) / ys).to(DEVICE)
    bounds, c = [], 0
    for k in names + ["selector_bit"]:
        bounds.append((c, c + Y[k].shape[1]))
        c += Y[k].shape[1]
    torch.manual_seed(0)
    net = nn.Sequential(Encoder(cfg), nn.ReLU(), nn.Linear(cfg.latent, Yc.shape[1])).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(Xtr), (min(batch, len(Xtr)),), device=DEVICE)
        with autocast():
            o = net(Xtr[idx])
        loss = torch.stack([F.mse_loss(o[:, a:b].float(), Ytr[idx][:, a:b])
                            for a, b in bounds]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad(), autocast():
        pred = net(Xte).float().cpu().numpy() * ys + ym
    yte = Yc[~tr]
    out = {}
    for (a, b), k in zip(bounds, names + ["selector_bit"]):
        y, p = yte[:, a:b], pred[:, a:b]
        if k == "selector_bit":
            out[k] = float(balanced_accuracy_score(y[:, 0].astype(int), (p[:, 0] > 0.5).astype(int)))
        else:
            out[k] = float(1 - ((y - p) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-12))
    return out


CEILING = cached(os.path.join(EVAL_DIR, "cnn_ceiling.pt"), lambda: cnn_ceiling(CFG),
                 "CNN probe ceiling (what a trained conv encoder reads from the frame)")
print("   " + "  ".join(f"{k} {v:.2f}" for k, v in CEILING.items()))
UNTESTABLE = sorted(k for k, v in CEILING.items() if v < 0.3)
if UNTESTABLE:
    print("   targets the ceiling cannot read, so no arm can be judged on them: "
          + ", ".join(UNTESTABLE))

Y_PROBE, TF_PROBE = probe_targets(l1_split, CFG.n_probe, CFG.probe_stride, CFG)
_ep_perm = np.random.default_rng(0).permutation(CFG.n_probe)
_train_eps = np.zeros(CFG.n_probe, bool)
_train_eps[_ep_perm[: CFG.n_probe // 2]] = True
TRAIN_MASK = np.repeat(_train_eps, TF_PROBE)                    # split by EPISODE

# "ceiling" is a conv encoder of the same architecture trained directly on the frames:
# what is readable in principle. In v1, distractor positions and the inactive goal read
# 0.00 from every latent AND from untrained weights, and there was no way to tell
# "the arm dropped it" from "nothing could read it".
PROBE_ARMS = ["ceiling", "init"] + ARMS
probe_frames = []
for seed in SEEDS:
    for arm in PROBE_ARMS:
        path = os.path.join(EVAL_DIR, f"probes_seed{seed}_{arm}.csv")
        if arm != "ceiling" and os.path.exists(path):
            probe_frames.append(pd.read_csv(path))
            continue
        rows = []
        if arm == "ceiling":     # not a latent: one run-wide reference, added per seed
            probe_frames.append(pd.DataFrame(
                [{"seed": seed, "arm": "ceiling", "space": "frame", "probe": p,
                  "target": t, "group": TARGET_GROUP[t],
                  "metric": "balanced_acc" if t == "selector_bit" else "r2", "score": v}
                 for t, v in CEILING.items() for p in ("linear", "mlp")]))
            continue
        lv = EVAL_INIT[seed]["level1"] if arm == "init" else EVAL[(seed, arm)]["level1"]
        for space in ("z", "h"):
            Z = lv[space].astype(np.float32).reshape(-1, lv[space].shape[-1])
            for (probe, target), score in run_probes(Z, Y_PROBE, TRAIN_MASK, seed).items():
                rows.append({"seed": seed, "arm": arm, "space": space, "probe": probe,
                             "target": target, "group": TARGET_GROUP[target],
                             "metric": "balanced_acc" if target == "selector_bit" else "r2",
                             "score": score})
        df = pd.DataFrame(rows)
        df.to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        probe_frames.append(df)
        print(f"  probes seed {seed} {arm} computed")
probe_df = pd.concat(probe_frames, ignore_index=True)
probe_df["score_clipped"] = np.where(probe_df.metric == "r2", probe_df.score.clip(lower=0), probe_df.score)
probe_df.to_csv(os.path.join(RESULTS, "latent_probes.csv"), index=False)

_piv = probe_df.pivot_table(index=["seed", "arm", "space", "probe"], columns="target",
                            values="score_clipped").reset_index()
_piv["selectivity_goal"] = _piv["active_goal"] - _piv["inactive_goal"]
_piv["selectivity_vector"] = _piv["agent_to_active_goal"] - _piv["agent_to_inactive_goal"]
sel_df = _piv[["seed", "arm", "space", "probe", "selectivity_goal", "selectivity_vector",
               "selector_bit"]]
sel_df.to_csv(os.path.join(RESULTS, "latent_selectivity.csv"), index=False)

print("\nProbe scores on z (mean over seeds; R², or balanced accuracy for selector_bit):")
print(probe_df[probe_df.space == "z"].pivot_table(index="target", columns=["probe", "arm"],
                                                  values="score_clipped").round(3).to_string())
# %% [markdown]
# ### 7e. Distribution shift (bonus): does a cleaner latent buy robustness?
#
# For each condition, the **drop** in return relative to `in_distribution` on the same
# episodes. The paired column compares each arm's drop with B's drop; positive means B
# was hurt more.
# %%
shift_df, shift_paired_df = pd.DataFrame(), pd.DataFrame()
if CFG.run_shift:
    frames = [metrics_frame(EVAL[(s, a)]["shift"][c], SHIFT_SCENES[c], a, s, c)
              for s in SEEDS for a in ARMS for c in SHIFTS]
    frames += [metrics_frame(BASE_SHIFT[k], SHIFT_SCENES["in_distribution"], k, -1, "any")
               for k in BASE_SHIFT]
    shift_df = pd.concat(frames, ignore_index=True)
    shift_df.to_csv(os.path.join(RESULTS, "shift_episodes.csv"), index=False)

    rows = []
    for s in SEEDS:
        ref = {a: EVAL[(s, a)]["shift"]["in_distribution"]["rewards"].mean(1) for a in ARMS}
        for c in SHIFTS:
            drops = {a: EVAL[(s, a)]["shift"][c]["rewards"].mean(1) - ref[a] for a in ARMS}
            for a in ARMS:
                rows.append({"seed": s, "arm": a, "condition": c,
                             "return": float(EVAL[(s, a)]["shift"][c]["rewards"].mean()),
                             "return_drop": float(drops[a].mean()),
                             "drop_minus_B_drop": float((drops[a] - drops["B_task"]).mean())})
    shift_paired_df = pd.DataFrame(rows)
    shift_paired_df.to_csv(os.path.join(RESULTS, "shift_summary.csv"), index=False)
    print("Return under shift (mean over seeds):")
    print(shift_paired_df.pivot_table(index="condition", columns="arm", values="return").round(3).to_string())
    print("\nDrop vs in-distribution (mean over seeds):")
    print(shift_paired_df.pivot_table(index="condition", columns="arm", values="return_drop").round(3).to_string())

# %% [markdown]
# ## 8. Figures
#
# Every comparison figure plots **one point or thin line per seed**, never only a mean.
# Arms that failed the learning gate are labelled `[FAILED GATE]` or marked `*`.
#
# ### 8a. Training health
#
# Task loss, auxiliary loss, and the latent monitors. **Per-dimension spread** falling
# toward zero means the latent has collapsed; Experiment 2's arm D failed silently where
# this panel would have shown it.
# %%
def arm_lab(a):
    return ARM_LABEL[a] + ("  [FAILED GATE]" if a in FAILED_ARMS else "")


def tick_lab(a):
    return SHORT[a] + ("*" if a in FAILED_ARMS else "")


def per_seed_lines(ax, df, x, y, keys=None, logy=False):
    for a in (ARMS if keys is None else keys):
        g = df[df["arm"] == a]
        if g.empty:
            continue
        for _, gs in g.groupby("seed"):
            ax.plot(gs[x], gs[y], color=ARM_COLOR[a], alpha=.3, lw=1)
        m = g.groupby(x)[y].mean()
        ax.plot(m.index, m.values, color=ARM_COLOR[a], lw=2.4, label=tick_lab(a))
    if logy:
        ax.set_yscale("log")


def dots_and_mean(ax, j, v, color, s=55):
    """One dot per seed plus a thick tick at the mean."""
    ax.scatter(np.full(len(v), j) + np.linspace(-.1, .1, len(v)), v, s=s, color=color,
               edgecolor="white", zorder=3)
    ax.hlines(np.mean(v), j - .28, j + .28, color=color, lw=3, zorder=4)


def fig_training():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    per_seed_lines(axes[0, 0], log_df, "epoch", "task_mse", logy=True)
    axes[0, 0].set_title("reward loss (standardised MSE, all horizons)")
    per_seed_lines(axes[0, 1], log_df, "epoch", "aux", keys=[a for a in ARMS if a != "B_task"], logy=True)
    axes[0, 1].set_title("auxiliary loss (B has none)")
    per_seed_lines(axes[0, 2], log_df, "epoch", "mse_kH", logy=True)
    axes[0, 2].set_title(f"training reward MSE at k={H}")
    per_seed_lines(axes[1, 0], log_df, "epoch", "z_dim_std")
    axes[1, 0].axhline(0.05, color="#d1495b", ls=":", lw=1)
    axes[1, 0].set_title("latent per-dim spread (collapse if -> 0)")
    per_seed_lines(axes[1, 1], log_df, "epoch", "z_eff_rank")
    axes[1, 1].set_title("latent effective rank")
    per_seed_lines(axes[1, 2], snap_df[snap_df.horizon == H], "epoch", "r2")
    axes[1, 2].axhline(0, color="#555", lw=1)
    axes[1, 2].set_title(f"held-out reward R² at k={H} during training")
    for ax in axes.ravel():
        ax.set_xlabel("epoch")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Training health (thin = individual seeds, bold = mean)", y=1.0)
    fig.tight_layout()
    savefig(fig, "fig04_training.png")


fig_training()
# %% [markdown]
# ### 8b. Level 1: reward prediction versus horizon
#
# **Level** and **rate** are different questions and can disagree (Experiment 1's rollout
# run had the task-only arm slightly worse in level but degrading more slowly). The middle
# panel shows the rate on its own.
# %%
def fig_level1():
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.4, 1, 1.3]})
    ax = axes[0]
    for ref in ("chance", "hold_current", "colourblind_true"):
        m = l1_df[l1_df.model == ref].groupby("horizon")["r2"].mean()
        ax.plot(m.index, m.values, "--", color=ARM_COLOR[ref], lw=1.6, label=ARM_LABEL[ref])
    for a in ARMS:
        g = l1_df[l1_df.model == a]
        for _, gs in g.groupby("seed"):
            ax.plot(gs.horizon, gs.r2, color=ARM_COLOR[a], alpha=.3, lw=1)
        m = g.groupby("horizon")["r2"].mean()
        ax.plot(m.index, m.values, "-o", ms=3, color=ARM_COLOR[a], lw=2.4, label=arm_lab(a))
    ax.set_xlabel("imagined steps ahead (k)"); ax.set_ylabel("reward R² (held out)")
    ax.set_ylim(min(-0.2, ax.get_ylim()[0]), 1.02)
    ax.set_title("Level: how well is future reward predicted?")
    ax.legend(fontsize=7.5, loc="lower left")

    ax = axes[1]
    for j, a in enumerate(ARMS):
        dots_and_mean(ax, j, deg_df[deg_df.model == a]["r2_drop"].values, ARM_COLOR[a])
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([tick_lab(a) for a in ARMS])
    ax.set_ylabel(f"R²(k=1) − R²(k={H})")
    ax.set_title("Rate: how much is lost over the horizon\n(one point per seed)")

    ax = axes[2]
    for a in [a for a in ARMS if a != "B_task"]:
        g = l1_paired_df[l1_paired_df.vs == a]
        for _, gs in g.groupby("seed"):
            ax.plot(gs.horizon, gs.delta_mse, color=ARM_COLOR[a], alpha=.35, lw=1)
        m = g.groupby("horizon")["delta_mse"].mean()
        ax.plot(m.index, m.values, color=ARM_COLOR[a], lw=2.6, label=f"B vs {tick_lab(a)}")
    ax.axhline(0, color="#555", lw=1.2)
    ax.set_xlabel("k"); ax.set_ylabel("MSE(B) − MSE(arm)")
    ax.set_title("Paired gap on identical episodes\n(above 0 = B worse; thin = seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig05_level1_horizon.png")


fig_level1()
# %% [markdown]
# ### 8c. Level 2: planning return — the headline figure
#
# Dots are seeds, thick ticks are means, dashed lines are baselines. The right panel is the
# selection test in behaviour: goal-choice accuracy. The colour-blind oracle shows where a
# model that ignores the background would sit.
# %%
def fig_planning():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ps = plan_seed.set_index("model")
    base_keys = ("oracle", "oracle_myopic", "oracle_colourblind", FLOOR)
    for ax, col, title in ((axes[0], "return", "real return (mean reward per step)"),
                           (axes[1], "normalised", "normalised: 0 = floor, 1 = oracle"),
                           (axes[2], "chose_active", "goal-choice accuracy")):
        for k in base_keys:
            y = (ps.loc[k, "return"] - R_FLOOR) / (R_ORACLE - R_FLOOR) if col == "normalised" else ps.loc[k, col]
            ax.axhline(y, color=ARM_COLOR[k], ls="--", lw=1.4,
                       label=ARM_LABEL[k] + (" (floor)" if k == FLOOR else ""))
        for j, a in enumerate(ARMS):
            dots_and_mean(ax, j, plan_seed[plan_seed.model == a][col].values, ARM_COLOR[a], s=60)
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([tick_lab(a) for a in ARMS])
        ax.set_xlim(-0.6, len(ARMS) - 0.4)
        ax.set_title(title)
    axes[2].axhline(0.5, color="#555", lw=1, ls=":", label="50% = cannot tell the goals apart")
    axes[2].set_ylim(0, 1.05)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("Does a policy planned inside each latent reach the ACTIVE goal?"
                 + ("   (* = failed learning gate)" if FAILED_ARMS else ""), y=1.0)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    savefig(fig, "fig06_planning_return.png")

    others = [a for a in ARMS if a != "B_task"]
    fig, axes = plt.subplots(1, len(others), figsize=(5 * len(others), 4), squeeze=False)
    for ax, a in zip(axes[0], others):
        g = plan_paired_df[plan_paired_df.vs == a].sort_values("seed")
        ax.errorbar(np.arange(len(g)), g.delta_return,
                    yerr=[g.delta_return - g.ci95_lo, g.ci95_hi - g.delta_return],
                    fmt="o", color=ARM_COLOR[a], capsize=4, ms=7)
        ax.axhline(0, color="#555", lw=1.2)
        ax.set_xticks(np.arange(len(g))); ax.set_xticklabels([f"seed {s}" for s in g.seed])
        ax.set_ylabel("return(arm) − return(B)")
        ax.set_title(f"{arm_lab(a)}\nvs B (above 0 = B worse; bars = 95% CI over episodes)", fontsize=9.5)
    fig.tight_layout()
    savefig(fig, "fig07_planning_paired.png")


fig_planning()
# %% [markdown]
# ### 8d. Level 3: what is in each latent — the selective-relevance figure
#
# Bars are means over seeds, dots are seeds; the grey `init` bar is the untrained shared
# starting point. Targets run from task-relevant (left) to irrelevant (right). The
# selective-relevance hypothesis predicts B high on the left, low on the right, and, the
# sharpest test, low on the **inactive** goal.
# %%
PROBE_ORDER = ["agent_position", "agent_velocity", "active_goal", "agent_to_active_goal",
               "selector_bit", "inactive_goal", "agent_to_inactive_goal", "ring_position",
               "plus_position", "distractor_positions", "distractor_hues",
               "background_sat_val", "background_hue_within_band"]
GROUP_SHADE = {"relevant": "#e8f4ea", "goal-like irrelevant": "#fbeaea",
               "identity": "#f2f2f2", "irrelevant": "#fdf3e3"}


def fig_probes(space, name):
    fig, axes = plt.subplots(2, 1, figsize=(17, 8.8), sharex=True)
    xs = np.arange(len(PROBE_ORDER))
    w = 0.84 / len(PROBE_ARMS)
    for ax, probe in zip(axes, ("linear", "mlp")):
        for i, t in enumerate(PROBE_ORDER):
            ax.axvspan(i - .5, i + .5, color=GROUP_SHADE[TARGET_GROUP[t]], zorder=0)
        d = probe_df[probe_df.probe == probe]
        for j, a in enumerate(PROBE_ARMS):
            da = d[(d.arm == a) & (d.space == ("frame" if a == "ceiling" else space))]
            off = (j - (len(PROBE_ARMS) - 1) / 2) * w
            ax.bar(xs + off, [da[da.target == t].score_clipped.mean() for t in PROBE_ORDER], w,
                   color=ARM_COLOR[a], alpha=.6, label=arm_lab(a), zorder=2)
            for i, t in enumerate(PROBE_ORDER):
                v = da[da.target == t].score_clipped.values
                ax.scatter(np.full(len(v), i + off), v, s=9, color="#333", zorder=3, edgecolor="none")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"{probe} probe\nR² (selector: balanced acc.)")
        ax.grid(False)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([t.replace("_", "\n") for t in PROBE_ORDER], fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle(f"What can be read out of {'the encoder latent z' if space == 'z' else 'the GRU state h'}?"
                 "   green = relevant · red = goal-like but irrelevant · grey = identity · orange = irrelevant",
                 y=1.0, fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    savefig(fig, name)


fig_probes("z", "fig08_probes_z.png")
fig_probes("h", "fig09_probes_h.png")


def fig_learning_speed():
    """v1 stopped every arm at 40 epochs while A was still improving. Speed is now its own
    result rather than a confound hiding inside the accuracy comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, col, title in ((axes[0], "epochs_trained", "epochs until the held-out plateau"),
                           (axes[1], "epochs_to_r2_0.8", "epochs to reach reward R² 0.8"),
                           (axes[2], "minutes", "GPU minutes per run")):
        for j, a in enumerate(ARMS):
            v = speed_df[speed_df.arm == a][col].values.astype(float)
            v = v[~np.isnan(v)]
            if len(v):
                dots_and_mean(ax, j, v, ARM_COLOR[a])
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([tick_lab(a) for a in ARMS])
        ax.set_title(title)
    axes[0].axhline(CFG.epochs, color="#d1495b", ls=":", lw=1.2)
    axes[0].text(len(ARMS) - 0.6, CFG.epochs, " ceiling", fontsize=8, color="#d1495b", va="bottom")
    not_conv = speed_df[~speed_df.converged]
    fig.suptitle("How long each arm needed" + ("  —  %d run(s) hit the ceiling while still "
                 "improving" % len(not_conv) if len(not_conv) else "  —  all runs plateaued"),
                 y=1.03)
    fig.tight_layout()
    savefig(fig, "fig12_learning_speed.png")


fig_learning_speed()


def fig_recon_fidelity():
    """Does arm A's decoder actually draw the small objects? If it does not, the claim that
    reconstruction 'encodes everything' fails before any latent probe is consulted."""
    have = [(s, EVAL[(s, "A_recon")]["recon"]) for s in SEEDS
            if "recon" in EVAL.get((s, "A_recon"), {})]
    if not have:
        return
    rows = []
    for s, rec in have:
        for obj in ("agent", "goals", "distractors"):
            rows.append({"seed": s, "object": obj, "contrast_real": rec[f"contrast_real_{obj}"],
                         "contrast_recon": rec[f"contrast_recon_{obj}"],
                         "ratio": rec[f"contrast_ratio_{obj}"]})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(RESULTS, "recon_fidelity.csv"), index=False)

    ex = have[0][1]
    real, recon = ex["examples_real"].astype(np.float32), ex["examples_recon"].astype(np.float32)
    n_ep, n_t = real.shape[0], real.shape[1]
    fig = plt.figure(figsize=(3.1 * n_t, 3.0 * (2 * n_ep) / 2 + 2.6))
    gs = fig.add_gridspec(2 * n_ep + 1, n_t, height_ratios=[1] * (2 * n_ep) + [1.25])
    for e in range(n_ep):
        for t in range(n_t):
            for r, img, lab in ((0, real, "real"), (1, recon, "arm A reconstruction")):
                ax = fig.add_subplot(gs[2 * e + r, t])
                ax.imshow(np.clip(img[e, t].transpose(1, 2, 0), 0, 1), interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                if t == 0:
                    ax.set_ylabel(lab, fontsize=8)
    ax = fig.add_subplot(gs[-1, :])
    w = 0.35
    xs = np.arange(3)
    objs = ["agent", "goals", "distractors"]
    ax.bar(xs - w / 2, [rdf[rdf.object == o].contrast_real.mean() for o in objs], w,
           label="real frame", color="#888")
    ax.bar(xs + w / 2, [rdf[rdf.object == o].contrast_recon.mean() for o in objs], w,
           label="reconstruction", color=ARM_COLOR["A_recon"])
    for i, o in enumerate(objs):
        ax.text(i, 0, "  ratio %.2f" % rdf[rdf.object == o].ratio.mean(), fontsize=9,
                ha="center", va="bottom")
    ax.set_xticks(xs); ax.set_xticklabels(objs)
    ax.set_ylabel("contrast against\nthe background")
    ax.set_title("Which objects does reconstruction actually draw? (mean over seeds)")
    ax.legend(fontsize=8)
    fig.suptitle("Arm A: reconstruction fidelity per object type", y=1.0)
    fig.tight_layout()
    savefig(fig, "fig13_recon_fidelity.png")


fig_recon_fidelity()


def fig_selectivity():
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
    for ax, (space, probe) in zip(axes, [("z", "linear"), ("z", "mlp"), ("h", "linear"), ("h", "mlp")]):
        d = sel_df[(sel_df.space == space) & (sel_df.probe == probe)]
        for j, a in enumerate(PROBE_ARMS):
            v = d[d.arm == a].selectivity_goal.values
            if len(v):                      # the ceiling row lives in its own space
                dots_and_mean(ax, j, v, ARM_COLOR[a], s=50)
        ax.axhline(0, color="#555", lw=1)
        ax.set_xticks(range(len(PROBE_ARMS))); ax.set_xticklabels([tick_lab(a) for a in PROBE_ARMS])
        ax.set_title(f"{space} · {probe} probe")
    axes[0].set_ylabel("R²(active goal) − R²(inactive goal)")
    fig.suptitle("Selectivity: does the latent keep the goal that pays and drop the one that does not?", y=1.03)
    fig.tight_layout()
    savefig(fig, "fig10_selectivity.png")


fig_selectivity()
# %% [markdown]
# ### 8e. Where the model is actually used, and under distribution shift
#
# Left: reward R² on the behaviour data (solid) versus along the planner's own trajectories
# (dashed). Right: real return under each shift condition (dots = seeds).
# %%
def fig_onpolicy_and_shift():
    ncol = 2 if CFG.run_shift else 1
    fig, axes = plt.subplots(1, ncol, figsize=(7.5 * ncol, 4.8), squeeze=False)
    ax = axes[0, 0]
    for a in ARMS:
        g = onpol_df[onpol_df.arm == a].groupby("horizon")[
            ["rmse_raw_behaviour_data", "rmse_raw_planner_trajectories"]].mean()
        ax.plot(g.index, g.rmse_raw_behaviour_data, "-", color=ARM_COLOR[a], lw=2,
                label=f"{tick_lab(a)} behaviour data")
        ax.plot(g.index, g.rmse_raw_planner_trajectories, "--", color=ARM_COLOR[a], lw=2,
                label=f"{tick_lab(a)} planner's trajectories")
    ax.set_xlabel("k"); ax.set_ylabel("reward RMSE (raw units)")
    ax.set_title("Does prediction hold up where the planner goes?\n(raw units: R² is unfair "
                 "here, rewards vary less)")
    ax.legend(fontsize=7, ncol=2)
    if CFG.run_shift:
        ax = axes[0, 1]
        conds = list(SHIFTS)
        w = 0.8 / len(ARMS)
        for k in ("oracle", FLOOR):
            ax.axhline(BASE_SHIFT[k]["rewards"].mean(), color=ARM_COLOR[k], ls="--", lw=1.2,
                       label=ARM_LABEL[k])
        for j, a in enumerate(ARMS):
            d = shift_paired_df[shift_paired_df.arm == a]
            off = (j - (len(ARMS) - 1) / 2) * w
            for i, c in enumerate(conds):
                v = d[d.condition == c]["return"].values
                ax.scatter(np.full(len(v), i + off), v, s=18, color=ARM_COLOR[a], zorder=3,
                           label=arm_lab(a) if i == 0 else None)
                ax.hlines(v.mean(), i + off - w / 2, i + off + w / 2, color=ARM_COLOR[a], lw=2.5)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=8)
        ax.set_ylabel("real return")
        ax.set_title("Planning return under distribution shift")
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    savefig(fig, "fig11_onpolicy_and_shift.png")


fig_onpolicy_and_shift()

# %% [markdown]
# ## 9. Animations
#
# ### 9a. The planning GIF — watch each arm steer
#
# Columns are the first three episodes of the fixed planning set (not cherry-picked). Rows
# are the oracle and each arm, first seed. Every panel shows the **real frame** that
# controller saw, plus:
#
# * **white line**: where the agent has been;
# * **dashed line**: the plan CEM just chose, traced through the **true** physics, i.e.
#   where that plan would really take the agent;
# * **dots**: where the model **imagines** the agent will be over the plan. The model has
#   no position output, so these are read out of the imagined GRU states by a linear probe
#   fit on filtered states. It is an illustration of the imagination, not a model output;
# * **cyan square**: the active goal (drawn on the figure only).
#
# No arm is ever shown as a reconstructed image: B has no decoder, and drawing one would
# misstate what these models do.
# %%
def replay_states(scene, res, cfg):
    """Re-simulate positions AND velocities from the recorded actions."""
    E, T = res["actions"].shape[:2]
    p, v = scene["p0"][:E].astype(np.float64), scene["v0"][:E].astype(np.float64)
    P, V = np.zeros((E, T + 1, 2)), np.zeros((E, T + 1, 2))
    for t in range(T):
        P[:, t], V[:, t] = p, v
        p, v = agent_step_np(p, v, res["actions"][:, t], cfg)
    P[:, T], V[:, T] = p, v
    return P, V


def gif_planning(path, n_ep=3, fps=6):
    seed = SEEDS[0]
    rows = ["oracle"] + ARMS
    idx = np.arange(n_ep)
    ga = active_goal(plan_scene["goals"], plan_scene["sel"])[idx] * PXS
    panels = {}
    for r in rows:
        res = BASE[r] if r == "oracle" else EVAL[(seed, r)]["planning"]
        P, V = replay_states(plan_scene, {"actions": res["actions"][:n_ep]}, CFG)
        frames = render_np(plan_scene, idx, CFG, agent=P.astype(np.float32), seed=7)
        imag = None
        if r != "oracle":
            lv = EVAL[(seed, r)]["level1"]
            hz = lv["h"].astype(np.float32).reshape(-1, lv["h"].shape[-1])
            probe = Ridge(alpha=1.0).fit(hz, Y_PROBE["agent_position"])
            ih = res["imag_h"][:n_ep].astype(np.float32)                   # (n,T,Hp,hidden)
            imag = probe.predict(ih.reshape(-1, ih.shape[-1])).reshape(*ih.shape[:3], 2)
        panels[r] = dict(P=P, V=V, frames=frames, plan=res["plan"][:n_ep], imag=imag,
                         pred=res["pred_r"][:n_ep], rew=res["rewards"][:n_ep])

    T = CFG.plan_T
    fig, axes = plt.subplots(len(rows), n_ep, figsize=(2.3 * n_ep + 1.4, 2.35 * len(rows)))
    art = {}
    for i, r in enumerate(rows):
        for j in range(n_ep):
            ax = axes[i, j]
            im = ax.imshow(panels[r]["frames"][j, 0], interpolation="nearest")
            goal_box(ax, ga[j], lw=1.1)
            trail, = ax.plot([], [], "-", color="white", lw=1.2, alpha=.8)
            plan, = ax.plot([], [], "--", color=ARM_COLOR[r], lw=1.8)
            dots, = ax.plot([], [], "o", color=ARM_COLOR[r], ms=3.2, mec="white", mew=.4)
            txt = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=6.5, color="white",
                          bbox=dict(fc="black", alpha=.45, lw=0, pad=1))
            ax.set_xlim(-0.5, PXS + 0.5); ax.set_ylim(PXS + 0.5, -0.5)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            art[(i, j)] = (im, trail, plan, dots, txt)
            if j == 0:
                ax.set_ylabel(SHORT.get(r, "oracle") + ("*" if r in FAILED_ARMS else ""),
                              fontsize=11, color=ARM_COLOR[r])
            if i == 0:
                ax.set_title("episode %d (%s active)" % (j, ("ring", "plus")[plan_scene["sel"][j]]),
                             fontsize=9)
    sup = fig.suptitle("", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    def upd(t):
        out = []
        for i, r in enumerate(rows):
            pn = panels[r]
            for j in range(n_ep):
                im, trail, plan, dots, txt = art[(i, j)]
                im.set_data(pn["frames"][j, t])
                trail.set_data(pn["P"][j, :t + 1, 0] * PXS, pn["P"][j, :t + 1, 1] * PXS)
                if t < T:
                    pp, vv = pn["P"][j, t].copy(), pn["V"][j, t].copy()
                    path_ = [pp]
                    for k in range(pn["plan"].shape[2]):
                        pp, vv = agent_step_np(pp, vv, pn["plan"][j, t, k], CFG)
                        path_.append(pp)
                    path_ = np.array(path_) * PXS
                    plan.set_data(path_[:, 0], path_[:, 1])
                    if pn["imag"] is not None:
                        dots.set_data(pn["imag"][j, t, :, 0] * PXS, pn["imag"][j, t, :, 1] * PXS)
                    txt.set_text("reward %+.2f" % pn["rew"][j, t])
                out += [im, trail, plan, dots, txt]
        sup.set_text("Planning in each latent (seed %d), step %d/%d\n"
                     "dashed = chosen plan in true physics · dots = imagined positions "
                     "(probe readout)\ncyan = active goal · * = failed learning gate" % (seed, t, T))
        return out + [sup]

    FuncAnimation(fig, upd, frames=T + 1, blit=False).save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print("saved " + os.path.basename(path))


gif_planning(os.path.join(RESULTS, "gif02_planning.gif"))
# %% [markdown]
# ### 9b. Reward prediction improving over training
#
# The held-out R²-versus-horizon curve for every arm, rebuilt at each logged epoch (first
# seed). The chance floor is R² = 0.
# %%
def gif_level1_training(path, fps=3):
    seed = SEEDS[0]
    d = snap_df[snap_df.seed == seed]
    epochs = sorted(d.epoch.unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    lines = {a: ax.plot([], [], "-o", ms=3, lw=2.4, color=ARM_COLOR[a], label=arm_lab(a))[0] for a in ARMS}
    ax.axhline(0, color="#555", lw=1, ls="--", label="chance (predict the mean)")
    ax.set_xlim(-0.5, H + 0.5)
    ax.set_ylim(min(-0.3, float(d.r2.min()) - 0.05), 1.02)
    ax.set_xlabel("imagined steps ahead (k)"); ax.set_ylabel("held-out reward R²")
    ax.legend(fontsize=8, loc="lower left")
    ttl = ax.set_title("")

    def upd(i):
        for a in ARMS:
            g = d[(d.arm == a) & (d.epoch == epochs[i])].sort_values("horizon")
            lines[a].set_data(g.horizon, g.r2)
        ttl.set_text("Reward prediction vs horizon - epoch %d (seed %d)" % (epochs[i], seed))
        return list(lines.values()) + [ttl]

    FuncAnimation(fig, upd, frames=len(epochs), blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("saved " + os.path.basename(path))


gif_level1_training(os.path.join(RESULTS, "gif03_level1_training.gif"))

# %% [markdown]
# ## 10. Verdict
#
# Written by the data, not by what we hoped. Questions that are easy to conflate are
# answered separately, and **v2 adds the checks v1 was missing**:
#
# 0. **GATE** — did each arm learn, and had it stopped improving when it stopped?
# 1. **SPEED** — how long did each arm need? (v1 read an unfinished arm A as a worse one.)
# 2. **LEVEL** — which latent plans best, and how well does each predict reward?
# 3. **RATE** — does any arm lose more over the imagination horizon?
# 4. **SELECTIVE RELEVANCE** — does B keep what reward needs and drop what it does not,
#    judged only on targets the raw-pixel probe shows are readable at all?
# 5. **CONTROLS** — did C actually learn its irrelevant target; did D train; does
#    reconstruction even draw the objects?
# 6. **MODEL QUALITY WHERE IT IS USED** — on-policy error and planner optimism.
# 7. **DISTRIBUTION SHIFT.**
#
# Claims are made on the seeds where **both** arms passed the gate. Any gap smaller than
# the seed-to-seed spread is reported as a tie.
# %%
def verdict():
    L = []
    n = len(SEEDS)
    ps = plan_seed.set_index(["model", "seed"])
    pz = probe_df[(probe_df.space == "z") & (probe_df.probe == "mlp")]
    ceil = probe_df[probe_df.arm == "ceiling"]

    def per_seed(model, col, seeds=None):
        return np.array([ps.loc[(model, s), col] for s in (seeds or SEEDS)])

    def probe_mean(arm, target, seeds=None, frame=None):
        f = pz if frame is None else frame
        d = f[(f.arm == arm) & (f.target == target)]
        if seeds is not None:
            d = d[d.seed.isin(seeds)]
        return float(d.score_clipped.mean())

    def compare(d, name, unit="", better="B worse"):
        d = np.asarray(d, dtype=float)
        if len(d) == 0:
            return [f"{name}: no seed passed the gate in both arms"]
        worse = int((d > 0).sum())
        sd = d.std(ddof=1) if len(d) > 1 else float("nan")
        line = (f"{name}: mean {d.mean():+.4f} {unit}; {better} in {worse}/{len(d)} seeds; "
                f"seed-to-seed sd {sd:.4f}")
        if len(d) < 3:
            why = f"only {len(d)} usable seed(s) - says nothing about robustness"
        elif abs(d.mean()) < sd:
            why = "SMALLER than the seed spread -> treat as a tie"
        elif worse == len(d):
            why = f"{better} in every usable seed"
        elif worse == 0:
            why = "B better in every usable seed"
        else:
            why = "seeds disagree on the sign -> not robust"
        return [line, f"      -> {why}"]

    L += ["=" * 78,
          f"HERMES EXPERIMENT 3 (v2) VERDICT   scale={SCALE}  seeds={n}  "
          f"frames={CFG.img_size}px  planning episodes={CFG.n_plan}", "=" * 78, ""]
    if SCALE == "quick":
        L += ["!! QUICK SMOKE TEST. NOTHING BELOW IS A RESULT.", ""]

    # ---- 0. gate -----------------------------------------------------------------
    L.append("0. LEARNING GATE")
    for a in ARMS:
        g = gate_df[gate_df.arm == a]
        L.append(f"   {ARM_LABEL[a]:38s} passed {int(g.passed.sum())}/{len(g)} seeds"
                 f"   converged {int(g.converged.sum())}/{len(g)}"
                 + ("   -> claims use seeds " + str(sorted(PASS_SEEDS[a])) if not g.passed.all()
                    else ""))
    if (~gate_df.converged).any():
        L.append("   !! some runs hit the epoch ceiling while still improving; raise CFG.epochs:")
        for _, r in gate_df[~gate_df.converged].iterrows():
            L.append(f"      seed {r.seed} {r.arm} (trained {r.epochs_trained} epochs)")
    L.append("")

    # ---- 1. speed ----------------------------------------------------------------
    L.append("1. SPEED - epochs needed to reach the held-out plateau (v2 measures this)")
    for a in ARMS:
        s = speed_df[speed_df.arm == a]
        to08 = s["epochs_to_r2_0.8"].astype(float)
        L.append(f"   {ARM_LABEL[a]:38s} epochs {s.epochs_trained.mean():6.1f}   "
                 f"to R2 0.8 {('%.1f' % to08.mean()) if to08.notna().any() else 'never':>6s}   "
                 f"final R2 {s.final_r2_kH.mean():.3f}   {s.minutes.mean():.1f} min")
    L.append("")

    # ---- 2. level ----------------------------------------------------------------
    L.append("2. LEVEL - planning return (real reward per step, mean +/- sd over seeds)")
    for k in CONTROLLERS:
        L.append(f"   {ARM_LABEL[k]:38s} {BASE[k]['rewards'].mean():+.3f}")
    for a in ARMS:
        r = per_seed(a, "return")
        L.append(f"   {ARM_LABEL[a]:38s} {r.mean():+.3f} +/- {r.std(ddof=1) if n > 1 else 0:.3f}"
                 f"   normalised {per_seed(a, 'normalised').mean():.2f}"
                 f"   goal-choice {per_seed(a, 'chose_active').mean():.2f}"
                 f"   success {per_seed(a, 'success').mean():.2f}"
                 f"   stops {per_seed(a, 'final_dist_px').mean():.1f} px away"
                 + ("   [gate failed in some seeds]" if a in FAILED_ARMS else ""))
    for a in [x for x in ARMS if x != "B_task"]:
        sd = paired_seeds("B_task", a)
        d = plan_paired_df[(plan_paired_df.vs == a) & (plan_paired_df.seed.isin(sd))]
        L += compare(d.sort_values("seed").delta_return.values, f"   return({SHORT[a]}) - return(B)")
    L.append("   reward prediction R2 at k=1 (mean over seeds): " + "  ".join(
        f"{SHORT[a]} {deg_df[deg_df.model == a].r2_k1.mean():.3f}" for a in ARMS))
    L.append("")

    # ---- 3. rate -----------------------------------------------------------------
    L.append(f"3. RATE - reward R2 lost between k=1 and k={H} (a separate question from level)")
    for a in ARMS:
        v = deg_df[deg_df.model == a]
        L.append(f"   {ARM_LABEL[a]:38s} R2 drop {v.r2_drop.mean():.3f}   "
                 f"MSE growth {v.mse_ratio.mean():.2f}x")
    for a in [x for x in ARMS if x != "B_task"]:
        sd = paired_seeds("B_task", a)
        if sd:
            dB = deg_df[(deg_df.model == "B_task") & (deg_df.seed.isin(sd))].sort_values("seed")
            dA = deg_df[(deg_df.model == a) & (deg_df.seed.isin(sd))].sort_values("seed")
            L += compare(dB.r2_drop.values - dA.r2_drop.values,
                         f"   drop(B) - drop({SHORT[a]})", "R2", better="B degrades more")
    L.append("")

    # ---- 4. robustness of the sign -----------------------------------------------
    L.append("4. ROBUST - do the seeds agree on the sign?")
    for a in [x for x in ARMS if x != "B_task"]:
        sd = paired_seeds("B_task", a)
        piv = l1_paired_df[(l1_paired_df.vs == a) & (l1_paired_df.seed.isin(sd))].pivot(
            index="horizon", columns="seed", values="delta_mse")
        unanimous = int(((piv > 0).all(axis=1) | (piv < 0).all(axis=1)).sum()) if len(piv.columns) else 0
        pl = plan_paired_df[(plan_paired_df.vs == a) & (plan_paired_df.seed.isin(sd))]
        L.append(f"   B vs {SHORT[a]} (on {len(sd)} usable seeds): level-1 sign unanimous at "
                 f"{unanimous}/{H + 1} horizons; planning: B worse in "
                 f"{int((pl.delta_return > 0).sum())}/{len(pl)} seeds")
    if n < 3:
        L.append(f"   !! only {n} seed(s): this says nothing about robustness.")
    L.append("")

    # ---- 5. selective relevance --------------------------------------------------
    L.append("5. SELECTIVE RELEVANCE")
    cb = planning_metrics(BASE["oracle_colourblind"], plan_scene, CFG)["chose_active"].mean()
    orc = planning_metrics(BASE["oracle"], plan_scene, CFG)["chose_active"].mean()
    L.append(f"   goal-choice accuracy: oracle {orc:.2f}, colour-blind oracle {cb:.2f} "
             "(what ignoring the background looks like)")
    for a in ARMS:
        L.append(f"   {ARM_LABEL[a]:38s} goal-choice {per_seed(a, 'chose_active').mean():.2f}")
    L.append("   probe scores (MLP on z, mean over seeds). 'ceiling' = what a conv encoder of")
    L.append("   the same architecture reads from the frame when trained to; a target it")
    L.append("   scores low on cannot be tested at all:")
    for t in ("selector_bit", "active_goal", "inactive_goal", "distractor_heatmap",
              "distractor_positions", "background_sat_val", "background_hue_within_band"):
        c = probe_mean("ceiling", t, frame=ceil[ceil.probe == "mlp"])
        flag = "   <- UNREADABLE even by the ceiling: not testable" if c < 0.3 else ""
        L.append(f"     {t:26s} ceiling {c:.2f} | " + "  ".join(
            f"{SHORT[a]} {probe_mean(a, t):.2f}" for a in PROBE_ARMS if a != "ceiling") + flag)
    L.append("   selectivity = R2(active goal) - R2(inactive goal), mean over seeds:")
    for space in ("z", "h"):
        for probe in ("linear", "mlp"):
            d = sel_df[(sel_df.space == space) & (sel_df.probe == probe)]
            L.append(f"     {space} {probe:6s}: " + "  ".join(
                f"{SHORT[a]} {d[d.arm == a].selectivity_goal.mean():+.2f}"
                for a in PROBE_ARMS if a != "ceiling"))
    sd_ab = paired_seeds("A_recon", "B_task")
    if sd_ab:
        selB = sel_df[(sel_df.arm == "B_task") & (sel_df.space == "z") & (sel_df.probe == "mlp")
                      & (sel_df.seed.isin(sd_ab))].sort_values("seed").selectivity_goal.values
        selA = sel_df[(sel_df.arm == "A_recon") & (sel_df.space == "z") & (sel_df.probe == "mlp")
                      & (sel_df.seed.isin(sd_ab))].sort_values("seed").selectivity_goal.values
        bg_ceiling = probe_mean("ceiling", "background_sat_val", frame=ceil[ceil.probe == "mlp"])
        crit = {
            "B picks the active goal far more often than a colour-blind planner (by > 0.15)":
                per_seed("B_task", "chose_active", PASS_SEEDS["B_task"]).mean() > cb + 0.15,
            "B selectivity (MLP, z) > 0 in every usable seed": bool((selB > 0).all()),
            "B selectivity > A selectivity in every usable seed": bool((selB > selA).all()),
            "B drops irrelevant background detail below the untrained weights":
                probe_mean("B_task", "background_sat_val") < probe_mean("init", "background_sat_val") - 0.1,
            "that background test is meaningful (readable from pixels)": bg_ceiling > 0.3,
        }
        L.append(f"   criteria (stated in advance), judged on seeds {sd_ab}:")
        for k, v in crit.items():
            L.append(f"     [{'x' if v else ' '}] {k}")
        if all(crit.values()):
            L.append("   -> B keeps what reward needs and drops irrelevant detail: SELECTION,")
            L.append("      not blanket discarding.")
        elif not list(crit.values())[0]:
            L.append("   -> B does not reliably use the background: consistent with blanket")
            L.append("      discarding, or with not having learned the task.")
        else:
            L.append("   -> mixed: read the probe figures before claiming anything.")
    else:
        L.append("   -> A and B never passed the gate in the same seed: not assessable.")
    L.append("")

    # ---- 6. controls -------------------------------------------------------------
    L.append("6. CONTROLS")
    c_hm, b_hm = probe_mean("C_scene", "distractor_heatmap"), probe_mean("B_task", "distractor_heatmap")
    px_hm = probe_mean("ceiling", "distractor_heatmap", frame=ceil[ceil.probe == "mlp"])
    L.append(f"   did C learn its irrelevant target? distractor heatmap R2: C {c_hm:.2f} vs "
             f"B {b_hm:.2f} (ceiling {px_hm:.2f})")
    if c_hm < b_hm + 0.05:
        L.append("   !! C did NOT encode the content it was told to. The negative control is")
        L.append("      INVALID this run: C vs B says nothing about the cost of irrelevant content.")
    else:
        sd = paired_seeds("B_task", "C_scene")
        d = plan_paired_df[(plan_paired_df.vs == "C_scene") & (plan_paired_df.seed.isin(sd))]
        L += compare(d.sort_values("seed").delta_return.values, "   return(C) - return(B)")
        L.append("      (below 0 = forcing irrelevant content costs return)")
    if "D_latent" in FAILED_ARMS:
        L.append("   D_latent failed the gate in some seeds; its numbers are not evidence about")
        L.append("   self-supervised objectives there.")
    else:
        L.append(f"   D_latent trained (final latent per-dim spread "
                 f"{log_df[log_df.arm == 'D_latent'].groupby('seed').z_dim_std.last().mean():.3f}).")
    rec = [EVAL[(s, "A_recon")]["recon"] for s in SEEDS if "recon" in EVAL.get((s, "A_recon"), {})]
    if rec:
        L.append("   does reconstruction actually DRAW each object? contrast kept vs the real frame:")
        L.append("     " + "  ".join(
            f"{o} {np.mean([r[f'contrast_ratio_{o}'] for r in rec]):.2f}"
            for o in ("agent", "goals", "distractors")))
        weak = [o for o in ("goals", "distractors")
                if np.mean([r[f"contrast_ratio_{o}"] for r in rec]) < 0.5]
        if weak:
            L.append(f"     -> reconstruction largely ignores: {', '.join(weak)}. 'Recon encodes")
            L.append("        everything' is false here, which limits what A-vs-B can show.")
    L.append("")

    # ---- 7. model quality where the planner uses it -------------------------------
    L.append("7. MODEL QUALITY WHERE THE PLANNER USES IT  (v2 added the parking data for this)")
    L.append("   reward RMSE in raw units at k=1, behaviour data -> planner trajectories:")
    for a in ARMS:
        g = onpol_df[(onpol_df.arm == a) & (onpol_df.horizon == 1)]
        o = opt_df[opt_df.arm == a]
        L.append(f"   {ARM_LABEL[a]:38s} {g.rmse_raw_behaviour_data.mean():.3f} -> "
                 f"{g.rmse_raw_planner_trajectories.mean():.3f}"
                 f"   optimism bias {o.one_step_bias.mean():+.3f} of {o.one_step_mae.mean():.3f} error")
    L.append("   (bias close to the whole error = the planner is exploiting the model)")
    L.append("")

    # ---- 8. shift ----------------------------------------------------------------
    if CFG.run_shift:
        L.append("8. DISTRIBUTION SHIFT - drop in return vs in-distribution (mean over seeds)")
        for c in [c for c in SHIFTS if c != "in_distribution"]:
            d = shift_paired_df[shift_paired_df.condition == c]
            L.append(f"   {c:24s} " + "  ".join(
                f"{SHORT[a]} {d[d.arm == a]['return_drop'].mean():+.3f}" for a in ARMS))
        if abs(shift_paired_df[shift_paired_df.condition != "heavy_pixel_noise"]["return_drop"]).max() < 0.02:
            L.append("   the distractor shifts move nothing, because no arm encodes distractors")
            L.append("   at all (see section 5): this tests indifference, not robustness.")
        L.append("")

    L += ["=" * 78,
          "Level and rate are different claims. A tie is a result. Failed or unconverged arms",
          "are failures, not findings. Every number above comes from this run's CSVs.",
          "=" * 78]
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(RESULTS, "VERDICT.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


verdict()
mark_progress("verdict_written")
# %% [markdown]
# ### Limitations — read before quoting any of this
#
# * **Auxiliary loss weights are not tuned.** All are 1.0.
# * **Arm A is Dreamer-like in its loss, not its architecture.** Every arm uses a
#   deterministic GRU rather than an RSSM, because an RSSM's KL term is itself a
#   representation loss and would stop B from being reward-only.
# * **The selector is the most salient pixel feature**, which favours reconstruction on
#   selection for a reason unrelated to the hypothesis. A small, low-salience cue would
#   favour B instead. Neither variant is run here (that was option 5, deferred).
# * **No capacity pressure.** A 32-dimensional latent against roughly 7 relevant
#   quantities means A never has to sacrifice anything to reconstruct.
# * **The planner has no value function**, so it sees 10 steps ahead. The oracle shares
#   that limit, so the ceiling is fair, but returns are not comparable to a value-based agent.
# * **Probes can mislead in both directions.** The raw-pixel row says what is readable in
#   principle; a target it scores low on is untestable rather than absent.
# * **Parking data makes the offline set partly planner-shaped.** It removes the v1
#   distribution gap by construction, so "the model holds up on-policy" is now partly a
#   property of the data, not only of the arm. The behaviour-data column is the honest
#   out-of-distribution reading.
# * **One toy, one architecture, one family of dynamics.**
# %%
# ============================================================================
# 11. Bundle the results folder (it is already on Drive; this is a single-file copy)
# ============================================================================
def bundle():
    path = os.path.join(RUN_DIR, f"hermes_exp3_v2_{RUN_NAME}_results.zip")
    with zipfile.ZipFile(path + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(RESULTS)):
            z.write(os.path.join(RESULTS, f), f)
    os.replace(path + ".tmp", path)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)\n")
    print(f"Everything for this run is in {RUN_DIR}:")
    for sub in ("results", "eval", "ckpt", "data"):
        files = sorted(os.listdir(os.path.join(RUN_DIR, sub)))
        mb = sum(os.path.getsize(os.path.join(RUN_DIR, sub, f)) for f in files) / 1e6
        print(f"  {sub}/  {len(files)} files, {mb:.1f} MB")
    for f in sorted(os.listdir(RESULTS)):
        print("    results/" + f)


bundle()
mark_progress("finished")

