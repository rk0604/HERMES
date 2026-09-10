# %% [markdown]
# # Hermes — Does a task-only latent survive multi-step imagination?
#
# **Scaled-up rollout experiment.** Drop this notebook into Colab, set the runtime to
# **GPU (T4 is plenty)**, and run all cells top to bottom.
#
# ### The question
#
# A world model has to *imagine forward*. The worry: a latent trained only to answer
# "where is the object **now**" might carry just enough for one step and then collapse
# once you cut off the pixels and make it run on its own hidden state.
#
# ### What is new versus the 16x16 toy
#
# | | toy | this notebook |
# |---|---|---|
# | frames | 16x16, 1 object | **48x48, target + 2 moving distractor objects** |
# | motion | 1-D, bounces | **2-D, bounces off all four walls** |
# | encoder | 2-layer MLP | **4-block CNN** |
# | horizon | 10 steps | **25 steps** |
# | arms | 2 | **4 (adds the control that the toy could not settle)** |
#
# ### The four arms
#
# All share an identical CNN encoder + GRU + task head, initialised identically, and
# all receive the task gradient. They differ **only** in what extra loss is attached:
#
# | arm | extra loss | what it tests |
# |---|---|---|
# | `A_recon` | pixel reconstruction | the classic world model |
# | `B_task` | none | the Hermes bet |
# | `C_scene` | predict background colour + distractor positions (**no pixels**) | *is it about pixels, or just about knowing the scene?* |
# | `D_latent` | predict its own next latent (self-supervised, **no decoder**) | a Hermes-flavoured alternative |
#
# `C_scene` is the important addition. The small-scale run could not separate
# *"reconstruction teaches the model about the scene"* from the duller
# *"any second loss regularises the encoder"*. Arm C is a scene-aware objective that
# never touches a pixel; arm D is a pixel-free self-supervised objective. Between
# them they pin down which explanation is doing the work.
#
# ### Honesty rules carried over
#
# * Every arm starts from **byte-identical** shared weights (asserted, not assumed).
# * Training and evaluation use the **identical** rollout procedure, so the curves
#   measure representation quality rather than train/test mismatch.
# * Everything is repeated over **multiple seeds**; a one-seed gap is not evidence.
#   In the small-scale version a beautiful single-seed finding evaporated under a
#   5-seed sweep, so per-seed points are plotted everywhere rather than hidden
#   inside a mean.
# * `B_task` never gets a decoder anywhere, including for visualisation. Where we
#   need to show "what B predicted", we draw its predicted **position**, never a
#   reconstructed image.
# %%
# ============================================================================
# 0. Environment
# ============================================================================
import os, sys, time, math, json, zipfile, warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
if os.environ.get("HERMES_HEADLESS"):      # set when smoke-testing outside Colab
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import hsv_to_rgb

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

print(f"torch  : {torch.__version__}")
print(f"device : {DEVICE}")
if DEVICE == "cuda":
    print(f"gpu    : {torch.cuda.get_device_name(0)}")
else:
    print("gpu    : NONE — set Runtime > Change runtime type > GPU, or expect a long run")

# Consistent house style for every figure in the notebook.
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 120, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
})

# One colour per arm, used everywhere so the figures read as a set.
ARM_COLOR = {"A_recon": "#d1495b", "B_task": "#2e86ab",
             "C_scene": "#e8a33d", "D_latent": "#5f8d4e",
             "baseline_hold": "#bbbbbb", "baseline_linear": "#8a8a8a"}
ARM_LABEL = {"A_recon": "A · task + reconstruction", "B_task": "B · task only",
             "C_scene": "C · task + scene aux (no pixels)",
             "D_latent": "D · task + next-latent (no decoder)",
             "baseline_hold": "baseline: hold last", "baseline_linear": "baseline: linear"}
# %% [markdown]
# ## 1. Configuration
#
# **Run it once with `QUICK_TEST = True` first.** That is a ~2 minute pass that
# executes every cell end to end, so you find out the notebook works *before*
# committing an hour of GPU time. Its numbers are meaningless — three epochs is
# nowhere near converged — and the config cell says so on screen.
#
# Then set `QUICK_TEST = False` and choose `SCALE`:
#
# | scale | frames | horizon | seeds | epochs | steps/run | rough T4 time |
# |---|---|---|---|---|---|---|
# | `quick` | 24px | 10 | 1 | 3 | ~15 | ~2 min (CPU fine) |
# | `mini` | 16px | 10 | 1 | 30 | ~690 | ~10 min on CPU, ~1 min on a T4 |
# | `medium` | 32px | 18 | 3 | 42 | ~1700 | ~35-55 min |
# | `full` | 48px | 25 | 4 | 60 | ~5000 | ~2-3.5 h |
#
# Those step counts are not guesses. A measured learning curve for this setup
# crosses the hold-last baseline at **~300 optimiser steps** and is still
# improving at 800, so `medium` and `full` are sized well past that.
#
# **`quick` is expected to FAIL the learning gate** further down — 15 steps
# trains nothing. That is the gate doing its job, not a bug. Use **`mini`** if
# you want the cheapest run that actually produces a valid result.
#
# `medium` is the better first real run: enough seeds to say something about
# robustness, short enough to survive a flaky Colab session. The training cell
# prints a **measured** time estimate after the first epoch, so you can abort early
# rather than discovering the cost an hour in.
# %%
QUICK_TEST = True          # <-- flip to False for a real run, then pick SCALE below

@dataclass
class Cfg:
    # ---- data ----
    img_size: int = 48
    n_frames: int = 45          # frames per trajectory
    context: int = 10           # real frames the model sees before going blind
    horizon: int = 25           # steps it must then imagine
    n_distract: int = 2         # moving distractor objects (task-irrelevant)
    n_train: int = 8000
    n_eval: int = 1500
    v_min: float = 0.015
    v_max: float = 0.045
    step_noise: float = 0.004
    pixel_noise: float = 0.05
    dot_sigma: float = 2.2      # blob radius, in pixels

    # ---- model ----
    latent: int = 32
    hidden: int = 192
    ch: int = 32                # base CNN width

    # ---- training ----
    epochs: int = 60
    batch: int = 96
    lr: float = 1.5e-3
    grad_clip: float = 5.0
    seeds: tuple = (0, 1, 2, 3)
    arms: tuple = ("A_recon", "B_task", "C_scene", "D_latent")

    # ---- loss weights (not tuned; see the limitations cell) ----
    w_recon: float = 1.0
    w_scene: float = 1.0
    w_latent: float = 1.0

    # ---- logging ----
    snap_every: int = 5         # epochs between latent/rollout snapshots

    @property
    def anchors(self):
        """Timesteps we predict from: enough context behind, enough horizon ahead."""
        return list(range(self.context - 1, self.n_frames - self.horizon))


PRESETS = {
    # ~2 min even on CPU. Proves the notebook runs; the NUMBERS ARE MEANINGLESS
    # (3 epochs is nowhere near converged). Never quote results from this.
    "quick": dict(img_size=24, n_frames=22, context=5, horizon=10, n_distract=1,
                  n_train=300, n_eval=200, latent=16, hidden=64, ch=16,
                  epochs=3, batch=64, seeds=(0,), snap_every=1),
    # ~20-35 min on a T4. Enough seeds to say something, small enough to survive
    # a flaky Colab session. Good default for a first real run.
    # A real (if small) run that actually LEARNS: ~690 steps per arm, past the
    # ~300 where this setup starts beating the hold-last baseline. Runs on CPU
    # in ~10 min, so it is the cheapest way to see the pipeline produce a valid
    # result rather than merely execute.
    "mini": dict(img_size=16, n_frames=18, context=5, horizon=10, n_distract=1,
                 n_train=1500, n_eval=500, latent=16, hidden=64, ch=16,
                 epochs=30, batch=64, seeds=(0,), snap_every=5),
    # ~35-55 min on a T4. Sized from a measured learning curve: this setup
    # crosses the hold-last baseline at ~300 optimiser steps and is still
    # improving at 800, so medium is set to give ~1700 steps per run.
    "medium": dict(img_size=32, n_frames=35, context=8, horizon=18, n_distract=2,
                   n_train=4000, n_eval=1000, latent=24, hidden=128, ch=24,
                   epochs=42, batch=96, seeds=(0, 1, 2), snap_every=6),
    # ~2-3.5 h on a T4. The headline configuration: ~5000 steps per run, well
    # past the point where the measured learning curve flattens.
    "full": dict(),                      # = the Cfg defaults
}

# "quick" | "mini" | "medium" | "full"   (env var lets you override without editing)
SCALE = os.environ.get("HERMES_SCALE", "quick" if QUICK_TEST else "full")
CFG = Cfg(**PRESETS[SCALE])
print("SCALE = " + SCALE)

if SCALE == "quick":
    print("!! QUICK smoke test: the numbers below are NOT results. "
          "Set QUICK_TEST=False (and pick SCALE) for a real run.")

print(json.dumps({k: str(v) for k, v in asdict(CFG).items()}, indent=2))
print(f"\nanchors per trajectory : {len(CFG.anchors)}  {CFG.anchors[:6]}...")
print(f"arms x seeds           : {len(CFG.arms)} x {len(CFG.seeds)} "
      f"= {len(CFG.arms) * len(CFG.seeds)} training runs")
# %% [markdown]
# ## 2. Data — 2-D bouncing target plus moving distractors
#
# Each trajectory contains:
#
# * a **white target blob** moving in 2-D at roughly constant velocity, reflecting off
#   all four walls. Its `(x, y)` position is the *only* task-relevant quantity.
# * **`n_distract` coloured blobs** with their own independent motion. They are drawn
#   from the same distribution as the target but are completely uncorrelated with it,
#   so they carry zero task information while being expensive to reconstruct.
# * a **background colour** drawn once per trajectory, plus fresh per-pixel noise
#   every frame.
#
# Why reflecting walls: with straight-line motion, 25-step prediction is just linear
# extrapolation, and any model that recovers velocity solves it perfectly — there
# would be nothing for a horizon curve to reveal. Bounces make the dynamics
# piecewise-linear, so the model has to represent the walls, not merely the velocity.
#
# **Memory note.** We store only the factors (positions and colours — a few MB) and
# render frames **on the GPU inside the training loop**. Materialising 6000 x 45
# frames at 48x48x3 would be ~7 GB; this way it is a few hundred MB of activations
# per batch and the dataset size is limited only by patience.
# %%
def simulate_factors(n, cfg, seed):
    """Simulate positions for the target and the distractors. Pure numpy, fast.

    Returns a dict of small arrays — no images. Reflection is applied per step and
    looped so a single step cannot tunnel through a wall.
    """
    rng = np.random.default_rng(seed)
    T, nd = cfg.n_frames, cfg.n_distract
    n_obj = 1 + nd                              # object 0 is the target

    # Start positions away from the walls; random direction, random speed.
    p = rng.uniform(0.15, 0.85, size=(n, n_obj, 2)).astype(np.float32)
    speed = rng.uniform(cfg.v_min, cfg.v_max, size=(n, n_obj, 1)).astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi, size=(n, n_obj, 1)).astype(np.float32)
    v = np.concatenate([np.cos(ang), np.sin(ang)], axis=-1) * speed

    pos = np.zeros((n, T, n_obj, 2), dtype=np.float32)
    pos[:, 0] = p
    bounced = np.zeros((n, T, n_obj), dtype=bool)

    for k in range(1, T):
        p = p + v + rng.normal(0, cfg.step_noise, size=p.shape).astype(np.float32)
        flipped = np.zeros(p.shape[:2], dtype=bool)
        # Reflect until inside [0,1]; two passes is ample for these speeds.
        for _ in range(2):
            lo, hi = p < 0.0, p > 1.0
            if not (lo.any() or hi.any()):
                break
            p = np.where(lo, -p, p)
            p = np.where(hi, 2.0 - p, p)
            v = np.where(lo | hi, -v, v)
            flipped |= (lo | hi).any(axis=-1)
        pos[:, k] = np.clip(p, 0.0, 1.0)
        bounced[:, k] = flipped

    bg = rng.uniform(0.05, 0.95, size=(n, 3)).astype(np.float32)
    # Distractor colours: a random HUE at full saturation and value. This matters.
    # Sampling RGB uniformly and rescaling gives a near-WHITE distractor a few
    # percent of the time, which makes the task genuinely ambiguous — the model
    # cannot know which of two white blobs is the target. Full saturation forces
    # the minimum channel to 0, so a distractor is never confusable with the
    # white target, and "track the white one" is always a well-posed instruction.
    hue = rng.uniform(0.0, 1.0, size=(n, nd, 1)).astype(np.float32)
    dcol = hsv_to_rgb(np.concatenate(
        [hue, np.ones_like(hue), np.ones_like(hue)], axis=-1)).astype(np.float32)

    # Did the TARGET bounce inside the scored window? Used to split the analysis.
    lo, hi = cfg.context, cfg.context + cfg.horizon
    bounce_in_window = bounced[:, lo:hi, 0].any(axis=1)

    return {
        "pos": pos,                       # (n, T, n_obj, 2)  all objects
        "target": pos[:, :, 0, :].copy(),  # (n, T, 2)        the task signal
        "bg": bg, "dcol": dcol,
        "bounce_in_window": bounce_in_window,
        "n_bounce": bounced[:, :, 0].sum(axis=1),
    }


def render(pos, bg, dcol, cfg, noise_std=None, generator=None, device=None):
    """Render frames on-device from factors. Fully vectorised over batch and time.

    pos  : (B, T, n_obj, 2) in [0,1]      bg : (B, 3)      dcol : (B, nd, 3)
    Returns (B, T, 3, H, W) in [0, 1].
    """
    device = device or pos.device
    noise_std = cfg.pixel_noise if noise_std is None else noise_std
    B, T, n_obj, _ = pos.shape
    H = W = cfg.img_size
    s2 = 2.0 * (cfg.dot_sigma ** 2)

    # Pixel-centre grids, shaped to broadcast against (B, T, n_obj, 1, 1).
    gx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, 1, W)
    gy = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, 1, H, 1)

    cx = (pos[..., 0] * (W - 1)).unsqueeze(-1).unsqueeze(-1)   # (B,T,n_obj,1,1)
    cy = (pos[..., 1] * (H - 1)).unsqueeze(-1).unsqueeze(-1)
    blobs = torch.exp(-(((gx - cx) ** 2) + ((gy - cy) ** 2)) / s2)   # (B,T,n_obj,H,W)

    # Background + per-frame noise.
    img = bg.view(B, 1, 3, 1, 1).expand(B, T, 3, H, W).clone()
    if noise_std > 0:
        img = img + torch.randn(img.shape, device=device, generator=generator) * noise_std

    # ALPHA compositing, not additive. Adding a blob to a bright background
    # saturates every channel toward 1, which turns coloured distractors white
    # and makes "track the white one" ambiguous. Compositing keeps each object's
    # true colour on any background: objects occlude rather than glow through.
    for k in range(1, n_obj):                      # distractors first ...
        a = blobs[:, :, k].unsqueeze(2)            # (B,T,1,H,W)
        img = img * (1 - a) + dcol[:, k - 1].view(B, 1, 3, 1, 1) * a
    a = blobs[:, :, 0].unsqueeze(2)                # ... target painted on top,
    img = img * (1 - a) + a                        # pure white, never occluded,
    return img.clamp_(0.0, 1.0)                    # so the task stays well posed


def make_split(n, cfg, seed):
    """Factors for one split, as torch tensors kept on the CPU (they are small)."""
    f = simulate_factors(n, cfg, seed)
    return {
        "pos": torch.from_numpy(f["pos"]),
        "target": torch.from_numpy(f["target"]),
        "bg": torch.from_numpy(f["bg"]),
        "dcol": torch.from_numpy(f["dcol"]),
        "bounce_in_window": f["bounce_in_window"],
        "n_bounce": f["n_bounce"],
    }


_probe = simulate_factors(400, CFG, 0)
print(f"target position range : [{_probe['target'].min():.3f}, {_probe['target'].max():.3f}]")
print(f"mean |step| per frame : {np.abs(np.diff(_probe['target'], axis=1)).mean():.4f}")
print(f"trajectories bouncing in the scored window : "
      f"{_probe['bounce_in_window'].mean()*100:.1f}%")
_lum = _probe["bg"] @ np.array([0.299, 0.587, 0.114], np.float32)
print(f"corr(bg luminance, target start x) : "
      f"{np.corrcoef(_lum, _probe['target'][:, 0, 0])[0,1]:+.4f}  (want ~0)")
# Distractors must never be confusable with the white target: at full saturation
# the darkest channel is 0, so "whiteness" stays far from 1.
print(f"distractor whiteness (min RGB channel, want ~0) : "
      f"{_probe['dcol'].min(axis=-1).mean():.3f}")

# Memory saved by rendering on the fly rather than materialising frames:
_full_gb = CFG.n_train * CFG.n_frames * 3 * CFG.img_size**2 * 4 / 1e9
print(f"\nframes if materialised : {_full_gb:.2f} GB  -> rendered per batch instead")
# %% [markdown]
# ### 2b. Look at the data before trusting anything downstream
#
# A still grid, then an animated GIF of a few trajectories. The white blob is the
# target; the coloured ones are distractors the model is free to ignore.
# %%
_prev = make_split(8, CFG, seed=999)
with torch.no_grad():
    _frames = render(_prev["pos"], _prev["bg"], _prev["dcol"], CFG, device="cpu")

n_show, k_show = 4, 6
step = max(1, CFG.n_frames // k_show)
fig, axes = plt.subplots(n_show, k_show, figsize=(1.5 * k_show, 1.5 * n_show))
for r in range(n_show):
    for c in range(k_show):
        t = min(c * step, CFG.n_frames - 1)
        axes[r, c].imshow(_frames[r, t].permute(1, 2, 0).numpy())
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([]); axes[r, c].grid(False)
        if r == 0:
            axes[r, c].set_title(f"t={t}", fontsize=9)
fig.suptitle("Sample trajectories — white = target, coloured = distractors", y=0.99)
fig.tight_layout(); plt.savefig(f"{RESULTS}/fig01_dataset_grid.png", bbox_inches="tight")
plt.show()


def gif_dataset(path, n=4, fps=10):
    """Animate a few trajectories so the motion and the distractors are visible."""
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 2.4))
    ims, marks = [], []
    for i, ax in enumerate(axes):
        ims.append(ax.imshow(_frames[i, 0].permute(1, 2, 0).numpy()))
        m, = ax.plot([], [], "o", mfc="none", mec="#00e5ff", mew=2, ms=13)
        marks.append(m)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0].set_ylabel("cyan ring = target")
    sup = fig.suptitle("")
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    def upd(t):
        for i in range(n):
            ims[i].set_data(_frames[i, t].permute(1, 2, 0).numpy())
            marks[i].set_data([_prev["target"][i, t, 0] * (CFG.img_size - 1)],
                              [_prev["target"][i, t, 1] * (CFG.img_size - 1)])
        sup.set_text(f"Dataset — frame {t}/{CFG.n_frames - 1}")
        return ims + marks + [sup]

    FuncAnimation(fig, upd, frames=CFG.n_frames, blit=False).save(
        path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {path}")


gif_dataset(f"{RESULTS}/gif01_dataset.gif")
# %% [markdown]
# ## 3. Models
#
# Every arm shares the same trunk:
#
# ```
# CNN encoder : 3x48x48 -> conv/2 -> conv/2 -> conv/2 -> flatten -> Linear -> latent
# GRU         : latent -> hidden
# task head   : hidden -> 2      (the target's x, y)
# ```
#
# and differs only in what is bolted on:
#
# * `A_recon`  — a transposed-conv decoder, MSE against the full frame.
# * `B_task`   — nothing. No decoder exists in this module at all.
# * `C_scene`  — a small MLP predicting background RGB and every distractor's
#   position from the latent. Scene-aware, but never touches a pixel.
# * `D_latent` — predicts the *next* latent from the GRU state, self-supervised
#   against the encoder's own output (stop-gradient on the target so it cannot
#   collapse to a constant).
#
# **Rollout convention** — the crux. The GRU eats `context` real frames, then steps
# forward on a **zero input** for `horizon` steps, reading a position out each step.
# No pixels reach it during that phase. Training and evaluation use the same
# procedure, so the horizon curve measures the representation, not a train/test gap.
# %%
GRU_KEEP_BIAS = 3.0   # see WorldModel.__init__ for why this exists


class Encoder(nn.Module):
    """Small CNN: three stride-2 blocks, then a linear map to the latent."""

    def __init__(self, cfg):
        super().__init__()
        c = cfg.ch
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1), nn.ReLU(),        # H/2
            nn.Conv2d(c, c * 2, 4, 2, 1), nn.ReLU(),    # H/4
            nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.ReLU(),  # H/8
        )
        self.spatial = cfg.img_size // 8
        self.fc = nn.Linear(c * 4 * self.spatial ** 2, cfg.latent)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class Decoder(nn.Module):
    """Mirror of the encoder, for arm A only."""

    def __init__(self, cfg):
        super().__init__()
        c = cfg.ch
        self.spatial = cfg.img_size // 8
        self.fc = nn.Linear(cfg.latent, c * 4 * self.spatial ** 2)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(c, 3, 4, 2, 1), nn.Sigmoid(),
        )
        self.c4 = c * 4

    def forward(self, z):
        h = self.fc(z).view(-1, self.c4, self.spatial, self.spatial)
        return self.net(h)


class WorldModel(nn.Module):
    """Encoder + GRU + task head, plus exactly one arm-specific auxiliary."""

    def __init__(self, cfg, arm):
        super().__init__()
        assert arm in ("A_recon", "B_task", "C_scene", "D_latent")
        self.cfg, self.arm = cfg, arm

        # --- shared trunk, always built first and in the same order so that
        # --- seeding identically gives every arm the same starting weights ---
        self.encoder = Encoder(cfg)
        self.gru = nn.GRU(cfg.latent, cfg.hidden, batch_first=True)
        # --- Bias the update gate toward "keep the state" -------------------
        # A freshly initialised GRU driven by ZERO input CONTRACTS to a fixed
        # point: measured on this architecture, only ~1% of the variation in the
        # initial hidden state survives 10 blind steps, and ~0% survives 25. The
        # rollout therefore emits a constant no matter what the context encoded
        # (the model predicts the centre of the frame), and because the signal
        # dies over those same steps there is no gradient with which to learn
        # otherwise. An earlier version of this notebook was stuck at exactly
        # chance for 4000 steps for precisely this reason.
        #
        # PyTorch packs GRU biases as [b_r | b_z | b_n]. In its update rule
        #     h' = (1 - z) * n + z * h,      z = sigmoid(... + b_z + ...)
        # pushing b_z up drives z toward 1, i.e. h' -> h, so the recurrence
        # starts near the IDENTITY and dynamics are learned as a perturbation on
        # top of it. At +3.0 retention goes from 1% to 66% at 10 steps and from
        # 0% to 36% at 25. This is a plain initialisation choice, applied
        # identically to every arm, so it cannot favour one of them.
        with torch.no_grad():
            self.gru.bias_ih_l0[cfg.hidden: 2 * cfg.hidden].fill_(GRU_KEEP_BIAS)
        self.head = nn.Linear(cfg.hidden, 2)

        # --- arm-specific extras, always built last ---
        self.decoder = Decoder(cfg) if arm == "A_recon" else None
        self.scene_head = None
        if arm == "C_scene":
            n_out = 3 + 2 * cfg.n_distract          # bg RGB + distractor xy
            self.scene_head = nn.Sequential(
                nn.Linear(cfg.latent, 64), nn.ReLU(), nn.Linear(64, n_out))
        self.latent_head = None
        if arm == "D_latent":
            self.latent_head = nn.Sequential(
                nn.Linear(cfg.hidden, 64), nn.ReLU(), nn.Linear(64, cfg.latent))

    # ---- pieces ----------------------------------------------------------
    def encode_seq(self, x):
        """(B,T,3,H,W) -> (B,T,latent). Frames encoded independently; all temporal
        structure lives in the GRU."""
        B, T = x.shape[:2]
        return self.encoder(x.reshape(B * T, *x.shape[2:])).view(B, T, -1)

    def rollout(self, h, horizon):
        """Step the GRU `horizon` times on ZERO input. h: (B,hidden) -> (B,horizon,2).

        A sequence of zero vectors through the fused nn.GRU is exactly equivalent to
        calling a GRUCell step by step with a zero input — verified in the small-scale
        version against a manual unroll — but runs as one fused kernel.
        """
        zeros = torch.zeros(h.shape[0], horizon, self.cfg.latent,
                            device=h.device, dtype=h.dtype)
        out, _ = self.gru(zeros, h.unsqueeze(0).contiguous())
        return self.head(out)

    def predict_from_frames(self, x, context, horizon):
        """Evaluation path: real frames in, blind multi-step positions out."""
        z = self.encode_seq(x[:, :context])
        out, _ = self.gru(z)
        return self.rollout(out[:, -1], horizon)

    # ---- losses ----------------------------------------------------------
    def losses(self, x, target, pos, bg, dcol, anchors):
        """All loss terms for one batch.

        x       : (B,T,3,H,W) rendered frames
        target  : (B,T,2)     true target position (the task signal)
        pos     : (B,T,n_obj,2), bg : (B,3), dcol : (B,nd,3)   [for arm C]
        anchors : timesteps to predict from; the GRU has consumed real frames
                  0..a and must then predict a+1 .. a+horizon blind.

        Returns (task_loss, aux_loss). aux is a zero tensor for arm B.
        """
        cfg = self.cfg
        B, T = x.shape[:2]
        z = self.encode_seq(x)                       # (B,T,latent)
        hs, _ = self.gru(z)                          # teacher-forced: real frames

        # Roll out from every anchor at once by folding the anchor axis into the
        # batch. Equivalent to looping (equal group sizes => mean of means is the
        # overall mean) but issues `horizon` fused calls instead of A*horizon.
        n_a = len(anchors)
        h_anchor = torch.stack([hs[:, a] for a in anchors], 0)        # (A,B,hidden)
        preds = self.rollout(h_anchor.reshape(n_a * B, cfg.hidden),
                             cfg.horizon).view(n_a, B, cfg.horizon, 2)
        tgt = torch.stack([target[:, a + 1: a + 1 + cfg.horizon] for a in anchors], 0)
        task_loss = ((preds - tgt) ** 2).mean()

        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        if self.arm == "A_recon":
            rec = self.decoder(z.reshape(B * T, cfg.latent))
            aux = ((rec - x.reshape(B * T, *x.shape[2:])) ** 2).mean() * cfg.w_recon
        elif self.arm == "C_scene":
            # Non-pixel scene knowledge: background colour + where the distractors are.
            pred = self.scene_head(z.reshape(B * T, cfg.latent))
            tgt_bg = bg.unsqueeze(1).expand(B, T, 3).reshape(B * T, 3)
            tgt_d = pos[:, :, 1:, :].reshape(B * T, -1)
            aux = ((pred - torch.cat([tgt_bg, tgt_d], -1)) ** 2).mean() * cfg.w_scene
        elif self.arm == "D_latent":
            # Predict the NEXT latent from the current GRU state. The target is
            # detached, so the encoder cannot trivially minimise this by
            # collapsing all latents to a constant.
            #
            # Both sides are L2-NORMALISED first (the BYOL/SimSiam trick). Raw
            # MSE on unnormalised latents is unusable here: nothing bounds the
            # latent norm, so as the encoder's activations grow during training
            # the target grows with them and the squared error grows
            # quadratically. In a smoke test this loss went 0.05 -> 11.8 -> 162
            # in three epochs and would have swamped the task loss entirely.
            # Normalising makes the objective scale-invariant and bounded in
            # [0, 4]: this arm predicts the DIRECTION of the next latent, not
            # its magnitude.
            pred_z = self.latent_head(hs[:, :-1].reshape(B * (T - 1), cfg.hidden))
            tgt_z = z[:, 1:].reshape(B * (T - 1), cfg.latent).detach()
            aux = ((F.normalize(pred_z, dim=-1) - F.normalize(tgt_z, dim=-1)) ** 2
                   ).sum(-1).mean() * cfg.w_latent
        else:
            aux = zero
        return task_loss, aux


def build_arms(cfg, seed):
    """One model per arm, all sharing byte-identical trunk weights.

    Each is constructed after re-seeding torch with the same value, and the trunk
    (encoder, GRU, head) is always built before any arm-specific module, so the
    shared parameters draw the same random numbers in every arm.
    """
    models = {}
    for arm in cfg.arms:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        models[arm] = WorldModel(cfg, arm).to(DEVICE)
    return models


def assert_same_trunk(models):
    """Fail loudly if the arms do not start from identical shared weights."""
    names = list(models)
    ref = dict(models[names[0]].named_parameters())
    checked = 0
    for other in names[1:]:
        for n, p in models[other].named_parameters():
            if n.startswith(("decoder", "scene_head", "latent_head")):
                continue                      # arm-specific by design
            if not torch.equal(ref[n].detach(), p.detach()):
                raise AssertionError(f"trunk weights differ at {n} ({other})")
            checked += 1
    return checked


# --- Can the rollout even carry information as far as we are asking? ---------
# A zero-input GRU can contract to a fixed point, in which case the rollout
# emits a constant regardless of the context, the model predicts the centre of
# the frame, and no gradient survives to teach it otherwise. That failure is
# silent: you get clean-looking curves from a model that learned nothing. So we
# measure it up front, at the horizon actually configured.
def rollout_retention(cfg, horizon=None, n=256):
    horizon = horizon or cfg.horizon
    m = WorldModel(cfg, "B_task").to(DEVICE)
    h0 = torch.randn(n, cfg.hidden, device=DEVICE)
    with torch.no_grad():
        zeros = torch.zeros(n, horizon, cfg.latent, device=DEVICE)
        out, _ = m.gru(zeros, h0.unsqueeze(0).contiguous())
    sp = out.std(dim=0).mean(dim=-1)          # spread across the batch, per step
    return (sp / sp[0]).cpu().numpy()


_ret = rollout_retention(CFG)
print(f"\nrollout information retention (fraction of initial-state spread kept):")
print("  " + "  ".join(f"h{h}={_ret[h-1]:.2f}"
                       for h in sorted({1, 2, 5, 10, CFG.horizon} & set(range(1, CFG.horizon + 1)))))
if _ret[-1] < 0.10:
    print("  !! WARNING: the rollout forgets its starting state within "
          f"{CFG.horizon} steps.")
    print("  !! The model will emit a constant and cannot learn. Raise "
          "GRU_KEEP_BIAS.")
else:
    print(f"  ok: {_ret[-1]*100:.0f}% survives to horizon {CFG.horizon}, so the "
          "rollout can carry the context.")


_m = build_arms(CFG, 0)
print(f"trunk verified identical across {len(_m)} arms "
      f"({assert_same_trunk(_m)} tensor comparisons)")
for a, m in _m.items():
    tot = sum(p.numel() for p in m.parameters())
    trunk = sum(p.numel() for n, p in m.named_parameters()
                if not n.startswith(("decoder", "scene_head", "latent_head")))
    print(f"  {a:10s} trunk={trunk:>9,}  total={tot:>9,}  (+{tot-trunk:,} aux)")
del _m
# %% [markdown]
# ## 4. Training
#
# Teacher forcing: real frames at every step, the model's own predictions are never
# fed back. Frames are rendered on-device inside the loop.
#
# After the first epoch the cell prints a **measured** time estimate for the whole
# sweep, so you can bail out early if the configuration is too ambitious rather than
# discovering it an hour in.
# %%
def evaluate(model, split, cfg, batch=256, noise_std=None, seed=12345):
    """Blind rollout on a held-out split. Returns predictions (N, horizon, 2).

    The evaluation frames are rendered with a FIXED generator, so the per-pixel
    noise is identical every time. Without this the same model scores slightly
    differently on repeat evaluations, which is exactly the kind of wobble that
    makes a small effect look real (or hides one).
    """
    model.eval()
    outs = []
    gen = torch.Generator(device=DEVICE)
    with torch.no_grad():
        for i in range(0, len(split["target"]), batch):
            gen.manual_seed(seed + i)          # deterministic per chunk
            sl = slice(i, i + batch)
            pos = split["pos"][sl].to(DEVICE)
            x = render(pos[:, :cfg.context], split["bg"][sl].to(DEVICE),
                       split["dcol"][sl].to(DEVICE), cfg, noise_std=noise_std,
                       generator=gen)
            outs.append(model.predict_from_frames(x, cfg.context, cfg.horizon).cpu())
    return torch.cat(outs).numpy()


def latent_snapshot(model, split, cfg, n=400, seed=999):
    """Per-frame latents on a fixed subset, for the probing / PCA figures.

    Fixed generator for the same reason as `evaluate`: identical pixels every
    time, so latent comparisons across arms and epochs are apples-to-apples.
    """
    model.eval()
    gen = torch.Generator(device=DEVICE); gen.manual_seed(seed)
    with torch.no_grad():
        pos = split["pos"][:n].to(DEVICE)
        x = render(pos, split["bg"][:n].to(DEVICE), split["dcol"][:n].to(DEVICE),
                   cfg, generator=gen)
        z = model.encode_seq(x).cpu().numpy()
    return z


def train_arm(model, arm, train, evalsplit, cfg, seed, time_probe=None):
    """Train one arm. Returns (loss log, snapshot list)."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = len(train["target"])
    rng = np.random.default_rng(seed)          # same batch order for every arm
    gen = torch.Generator(device=DEVICE); gen.manual_seed(seed)
    log, snaps = [], []

    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        perm = rng.permutation(n)
        tot_task = tot_aux = nb = 0.0
        for i in range(0, n, cfg.batch):
            idx = torch.from_numpy(perm[i:i + cfg.batch].copy())
            pos = train["pos"][idx].to(DEVICE, non_blocking=True)
            bg = train["bg"][idx].to(DEVICE, non_blocking=True)
            dcol = train["dcol"][idx].to(DEVICE, non_blocking=True)
            tgt = train["target"][idx].to(DEVICE, non_blocking=True)
            x = render(pos, bg, dcol, cfg, generator=gen)   # rendered on-device

            task_loss, aux = model.losses(x, tgt, pos, bg, dcol, cfg.anchors)
            loss = task_loss + aux

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            tot_task += task_loss.item(); tot_aux += float(aux); nb += 1

        log.append({"arm": arm, "seed": seed, "epoch": ep,
                    "task_mse": tot_task / nb, "aux": tot_aux / nb,
                    "sec": time.time() - t0})

        if ep % cfg.snap_every == 0 or ep == 1 or ep == cfg.epochs:
            snaps.append({"arm": arm, "seed": seed, "epoch": ep,
                          "pred": evaluate(model, evalsplit, cfg),
                          "latent": latent_snapshot(model, evalsplit, cfg)})

        if time_probe is not None and ep == 1:
            time_probe.append(time.time() - t0)
    return log, snaps


# ---------------------------------------------------------------------------
# Run the full sweep: every arm x every seed.
# ---------------------------------------------------------------------------
all_logs, all_snaps, all_preds, splits = [], [], {}, {}
probe, t_start = [], time.time()
total_runs = len(CFG.seeds) * len(CFG.arms)
run_i = 0

# --- checkpoint / resume ----------------------------------------------------
# The full sweep is 16 training runs and a couple of hours. Colab disconnects.
# Each (seed, arm) is checkpointed the moment it finishes, and re-running this
# cell SKIPS anything already on disk — so a dropped session costs you the run
# in flight, not the whole sweep. Delete results/ckpt to force a fresh start.
CKPT = os.path.join(RESULTS, "ckpt")
os.makedirs(CKPT, exist_ok=True)


def ckpt_path(seed, arm):
    return os.path.join(CKPT, f"{arm}_seed{seed}.pt")


for seed in CFG.seeds:
    train = make_split(CFG.n_train, CFG, seed=1000 * seed)
    ev = make_split(CFG.n_eval, CFG, seed=1000 * seed + 1)   # disjoint
    splits[seed] = ev
    models = build_arms(CFG, seed + 10)
    assert_same_trunk(models)

    for arm in CFG.arms:
        run_i += 1
        path = ckpt_path(seed, arm)

        if os.path.exists(path):
            blob = torch.load(path, weights_only=False)
            all_logs += blob["log"]
            all_snaps += blob["snaps"]
            all_preds[(seed, arm)] = blob["pred"]
            print(f"[{run_i}/{total_runs}] seed {seed}  arm {arm} ... "
                  f"resumed from checkpoint", flush=True)
            continue

        print(f"[{run_i}/{total_runs}] seed {seed}  arm {arm} ...",
              end=" ", flush=True)
        log, snaps = train_arm(models[arm], arm, train, ev, CFG, seed,
                               time_probe=probe)
        pred = evaluate(models[arm], ev, CFG)
        all_logs += log
        all_snaps += snaps
        all_preds[(seed, arm)] = pred

        # Snapshots carry per-epoch latents and are only needed for the
        # evolution animations, which use the first seed. Storing them for
        # every seed would make the checkpoints large for no benefit.
        keep = snaps if seed == CFG.seeds[0] else []
        torch.save({"log": log, "snaps": keep, "pred": pred}, path)

        print(f"task_mse {log[-1]['task_mse']:.5f}  "
              f"({sum(l['sec'] for l in log):.0f}s)  [saved]")

        if run_i == 1 and probe:
            est = probe[0] * CFG.epochs * total_runs / 60
            print(f"    ~{probe[0]:.1f}s/epoch  ->  estimated total "
                  f"~{est:.1f} min for {total_runs} runs")
            if est > 60:
                print(f"    NOTE: ~{est/60:.1f} h. Checkpointing is on, so a "
                      "disconnect only costs the run in flight —")
                print("    just re-run this cell to pick up where it stopped.")

    del models
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print(f"\nSWEEP DONE in {(time.time()-t_start)/60:.1f} min")
log_df = pd.DataFrame(all_logs)
log_df.to_csv(f"{RESULTS}/training_log.csv", index=False)
# %% [markdown]
# ## 5. Evaluation
#
# Error is the **Euclidean distance in pixels** between the predicted and true target
# position, at each horizon step, on held-out trajectories.
#
# Two reference baselines bracket the models. Without them a horizon curve is
# unreadable, because you cannot tell whether either model learned any dynamics:
#
# * **hold last** — repeat the last observed position. The do-nothing floor.
# * **linear** — fit velocity from the true context positions and extrapolate. Near
#   optimal for straight-line motion, but it knows nothing about walls and walks
#   straight through them, so **beating it means having learned the bounce**.
#
# Both baselines are handed the *true* positions, which slightly favours them over
# the models (which only ever see pixels). That is deliberate — it makes them
# conservative references.
# %%
PXS = CFG.img_size - 1


def baseline_hold(t_np, cfg):
    return np.repeat(t_np[:, cfg.context - 1][:, None, :], cfg.horizon, axis=1)


def baseline_linear(t_np, cfg):
    v = (t_np[:, cfg.context - 1] - t_np[:, 0]) / max(cfg.context - 1, 1)
    steps = np.arange(1, cfg.horizon + 1)[None, :, None]
    return t_np[:, cfg.context - 1][:, None, :] + v[:, None, :] * steps


def err_curve(pred, t_np, cfg, mask=None):
    """Per-horizon Euclidean pixel error. pred/(true): (N, horizon, 2)."""
    true = t_np[:, cfg.context: cfg.context + cfg.horizon, :]
    d = np.linalg.norm(pred - true, axis=-1) * PXS          # (N, horizon)
    return d[mask] if mask is not None else d


rows, paired_rows = [], []
for seed in CFG.seeds:
    ev = splits[seed]
    t_np = ev["target"].numpy()
    bounce = ev["bounce_in_window"]
    series = dict(all_preds_for_seed := {a: all_preds[(seed, a)] for a in CFG.arms})
    series["baseline_hold"] = baseline_hold(t_np, CFG)
    series["baseline_linear"] = baseline_linear(t_np, CFG)

    for name, pred in series.items():
        for subset, mask in (("all", None), ("no_bounce", ~bounce), ("bounce", bounce)):
            d = err_curve(pred, t_np, CFG, mask)
            if len(d) < 2:
                continue
            for h in range(CFG.horizon):
                rows.append({"seed": seed, "model": name, "subset": subset,
                             "horizon": h + 1, "n": len(d),
                             "err_px": float(d[:, h].mean()),
                             "median_px": float(np.median(d[:, h]))})

    # Paired against B: same trajectories, so compare the difference directly
    # rather than asking whether two independent error bars overlap.
    base = err_curve(all_preds_for_seed["B_task"], t_np, CFG)
    for arm in CFG.arms:
        if arm == "B_task":
            continue
        other = err_curve(all_preds_for_seed[arm], t_np, CFG)
        for h in range(CFG.horizon):
            diff = base[:, h] - other[:, h]          # >0 => B worse than `arm`
            paired_rows.append({"seed": seed, "vs": arm, "horizon": h + 1,
                                "delta_px": float(diff.mean()),
                                "b_worse_frac": float((diff > 0).mean())})

err_df = pd.DataFrame(rows)
paired_df = pd.DataFrame(paired_rows)
err_df.to_csv(f"{RESULTS}/rollout_error.csv", index=False)
paired_df.to_csv(f"{RESULTS}/rollout_paired.csv", index=False)

summary = (err_df[err_df.subset == "all"]
           .groupby(["model", "horizon"])["err_px"]
           .agg(["mean", "std", "count"]).reset_index())
summary.to_csv(f"{RESULTS}/rollout_summary.csv", index=False)

# Degradation ratio: err(last horizon) / err(h=1), per model per seed. This is the
# literal "degrades faster" metric, and it is NOT the same question as "who is more
# accurate" — a model can be uniformly worse while decaying at the same rate.
deg = []
for (m, s), g in err_df[err_df.subset == "all"].groupby(["model", "seed"]):
    g = g.set_index("horizon")
    deg.append({"model": m, "seed": s, "h1": g.loc[1, "err_px"],
                "hN": g.loc[CFG.horizon, "err_px"],
                "ratio": g.loc[CFG.horizon, "err_px"] / g.loc[1, "err_px"]})
deg_df = pd.DataFrame(deg)
deg_df.to_csv(f"{RESULTS}/rollout_degradation.csv", index=False)

# ---------------------------------------------------------------------------
# LEARNING GATE — check the models actually learned before reading anything else
# ---------------------------------------------------------------------------
# A model that cannot beat "repeat the last observed position" has learned no
# dynamics at all, and every comparison after it is a ranking among broken
# models. This is not hypothetical: the first smoke test of this notebook
# produced a full set of confident-looking curves in which all four arms were
# simply emitting the centre of the frame. The tell was that they lost to a
# baseline requiring no learning whatsoever.
def learning_gate():
    H = CFG.horizon
    ev = splits[CFG.seeds[0]]
    true = ev["target"].numpy()[:, CFG.context: CFG.context + H, :]
    # What "always guess the centre of the frame" would score — the chance floor.
    chance = float(np.linalg.norm(true - 0.5, axis=-1).mean()) * PXS
    hold = summary[(summary.model == "baseline_hold") & (summary.horizon == H)]["mean"].iloc[0]
    lin = summary[(summary.model == "baseline_linear") & (summary.horizon == H)]["mean"].iloc[0]

    print("")
    print("LEARNING GATE  (at horizon %d)" % H)
    print("  chance, always guess centre : %6.2f px" % chance)
    print("  baseline hold-last          : %6.2f px   <- models MUST beat this" % hold)
    print("  baseline linear             : %6.2f px" % lin)

    bad = []
    for a in CFG.arms:
        e = summary[(summary.model == a) & (summary.horizon == H)]["mean"].iloc[0]
        if e >= hold:
            bad.append(a)
        print("  %-34s %6.2f px   %s" % (ARM_LABEL[a], e, "ok" if e < hold else "FAILS GATE"))

    if bad:
        print("")
        print("!" * 74)
        print("!! %d of %d ARMS DID NOT LEARN. They lose to a baseline that needs no"
              % (len(bad), len(CFG.arms)))
        print("!! learning at all, so NOTHING below this point is interpretable.")
        print("!! Likely causes, in order:")
        print("!!   1. SCALE='quick', or too few epochs  ->  raise CFG.epochs")
        print("!!   2. too little data                   ->  raise CFG.n_train")
        print("!!   3. learning rate                     ->  try CFG.lr = 3e-3")
        print("!! Do not report these numbers.")
        print("!" * 74)
    else:
        print("")
        print("  All arms beat hold-last, so they learned real dynamics and the")
        print("  comparison below is meaningful.")
    return bad


_failed_gate = learning_gate()

show_h = sorted({1, 2, 5, 10, CFG.horizon // 2, CFG.horizon} & set(range(1, CFG.horizon + 1)))
print(f"Mean Euclidean error (px) over {len(CFG.seeds)} seed(s):\n")
print(summary[summary.horizon.isin(show_h)]
      .pivot(index="horizon", columns="model", values="mean").round(3).to_string())
print("\nDegradation ratio  err(hN)/err(h1):")
print(deg_df.groupby("model")["ratio"].agg(["mean", "std"]).round(3).to_string())
# %% [markdown]
# ## 6. Figures
#
# ### 6a. The main result — error vs horizon
# %%
def fig_horizon_curves():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    s = summary
    for m in list(CFG.arms) + ["baseline_hold", "baseline_linear"]:
        g = s[s.model == m].sort_values("horizon")
        if g.empty:
            continue
        is_arm = m in CFG.arms
        for ax, logy in zip(axes, (False, True)):
            ax.plot(g.horizon, g["mean"], "-o" if is_arm else "--",
                    color=ARM_COLOR[m], lw=2.6 if is_arm else 1.6,
                    ms=4 if is_arm else 0, label=ARM_LABEL[m])
            if is_arm and g["std"].notna().any():
                ax.fill_between(g.horizon, g["mean"] - g["std"].fillna(0),
                                g["mean"] + g["std"].fillna(0),
                                color=ARM_COLOR[m], alpha=0.13)
            if logy:
                ax.set_yscale("log")
    for ax, t in zip(axes, ("linear scale", "log scale")):
        ax.set_xlabel("prediction horizon (steps ahead, no new frames)")
        ax.set_ylabel("position error (px)")
        ax.set_title(t)
    axes[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Blind rollout: how far off is the imagined position?", y=1.02)
    fig.tight_layout()
    plt.savefig(f"{RESULTS}/fig02_horizon_curves.png", bbox_inches="tight")
    plt.show()


fig_horizon_curves()
# %% [markdown]
# ### 6b. The rate question, with the offset removed
#
# "Degrades faster" is about **rate**, not level. A model can be uniformly worse
# while decaying at exactly the same speed. Dividing each arm by its own horizon-1
# error strips out the constant offset and leaves only the growth.
# %%
def fig_degradation():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                             gridspec_kw={"width_ratios": [1.3, 1, 1]})

    ax = axes[0]
    for m in CFG.arms:
        g = summary[summary.model == m].sort_values("horizon")
        base = g[g.horizon == 1]["mean"].iloc[0]
        ax.plot(g.horizon, g["mean"] / base, "-o", ms=4, lw=2.6,
                color=ARM_COLOR[m], label=ARM_LABEL[m])
    ax.set_xlabel("horizon"); ax.set_ylabel("error ÷ own h=1 error")
    ax.set_title("Growth relative to each arm's own start")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    for j, m in enumerate(CFG.arms):
        r = deg_df[deg_df.model == m]["ratio"].values
        ax.scatter(np.full(len(r), j) + np.random.uniform(-.09, .09, len(r)), r,
                   color=ARM_COLOR[m], s=70, alpha=.8, edgecolor="white", zorder=3)
        ax.hlines(r.mean(), j - .25, j + .25, color=ARM_COLOR[m], lw=3, zorder=4)
    ax.set_xticks(range(len(CFG.arms)))
    ax.set_xticklabels([a.split("_")[0] for a in CFG.arms])
    ax.set_ylabel(f"err(h={CFG.horizon}) / err(h=1)")
    ax.set_title("Growth factor, one point per seed")

    ax = axes[2]
    for j, m in enumerate(CFG.arms):
        g = err_df[(err_df.model == m) & (err_df.subset == "all")]
        for h, mk in ((1, "o"), (CFG.horizon, "s")):
            v = g[g.horizon == h]["err_px"].values
            ax.scatter(np.full(len(v), j) + (0.16 if h != 1 else -0.16),
                       v, marker=mk, color=ARM_COLOR[m], s=55, alpha=.8,
                       edgecolor="white", zorder=3,
                       label=(f"h={h}" if j == 0 else None))
    ax.set_xticks(range(len(CFG.arms)))
    ax.set_xticklabels([a.split("_")[0] for a in CFG.arms])
    ax.set_ylabel("error (px)"); ax.set_title("Absolute error, per seed")
    ax.legend(fontsize=9)
    fig.suptitle("Does any arm degrade faster than the others?", y=1.03)
    fig.tight_layout()
    plt.savefig(f"{RESULTS}/fig03_degradation.png", bbox_inches="tight")
    plt.show()


fig_degradation()
# %% [markdown]
# ### 6c. Paired comparison against the task-only arm
#
# Every arm sees the identical held-out trajectories, so we compare the *paired*
# difference. Positive means **B (task-only) is worse** than that arm.
# The dots show individual seeds — if their sign disagrees, the effect is not real.
# %%
def fig_paired():
    others = [a for a in CFG.arms if a != "B_task"]
    fig, axes = plt.subplots(1, len(others), figsize=(5 * len(others), 4.3),
                             squeeze=False)
    for ax, arm in zip(axes[0], others):
        g = paired_df[paired_df["vs"] == arm]
        piv = g.pivot(index="horizon", columns="seed", values="delta_px")
        for sd in piv.columns:
            ax.plot(piv.index, piv[sd], color=ARM_COLOR[arm], alpha=.32, lw=1.2)
        ax.plot(piv.index, piv.mean(axis=1), color=ARM_COLOR[arm], lw=3,
                label="mean over seeds")
        ax.axhline(0, color="#555", lw=1.4)
        ax.set_xlabel("horizon"); ax.set_ylabel("B error − arm error (px)")
        ax.set_title(f"B  vs  {ARM_LABEL[arm]}", fontsize=10)
        ax.legend(fontsize=9)
        ax.text(.02, .97, "above 0 → task-only is worse", transform=ax.transAxes,
                va="top", fontsize=8.5, color="#666")
    fig.suptitle("Paired gap (thin lines = individual seeds)", y=1.03)
    fig.tight_layout()
    plt.savefig(f"{RESULTS}/fig04_paired.png", bbox_inches="tight")
    plt.show()


fig_paired()
# %% [markdown]
# ### 6d. Training curves and the bounce split
# %%
def fig_training_and_bounce():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))

    ax = axes[0]
    for m in CFG.arms:
        g = log_df[log_df.arm == m].groupby("epoch")["task_mse"].mean()
        ax.plot(g.index, g.values, color=ARM_COLOR[m], lw=2.4, label=ARM_LABEL[m])
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("task MSE")
    ax.set_title("Task loss (mean over seeds)"); ax.legend(fontsize=8)

    ax = axes[1]
    for m in CFG.arms:
        g = log_df[log_df.arm == m].groupby("epoch")["aux"].mean()
        if g.max() > 0:
            ax.plot(g.index, g.values, color=ARM_COLOR[m], lw=2.4, label=ARM_LABEL[m])
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("auxiliary loss")
    ax.set_title("Auxiliary losses (B has none)"); ax.legend(fontsize=8)

    ax = axes[2]
    for m in CFG.arms:
        for sub, ls in (("no_bounce", "--"), ("bounce", "-")):
            g = (err_df[(err_df.model == m) & (err_df.subset == sub)]
                 .groupby("horizon")["err_px"].mean())
            if g.empty:
                continue
            ax.plot(g.index, g.values, ls, color=ARM_COLOR[m], lw=2,
                    label=f"{m.split('_')[0]} {sub}")
    ax.set_xlabel("horizon"); ax.set_ylabel("error (px)")
    ax.set_title("Solid = target bounces in window, dashed = no bounce")
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    plt.savefig(f"{RESULTS}/fig05_training_bounce.png", bbox_inches="tight")
    plt.show()


fig_training_and_bounce()
# %% [markdown]
# ### 6e. What is actually in each latent?
#
# Linear probes on the final latents: how much of the **task** (target position) and
# how much of the **distractors** (background colour, distractor positions) can be
# read out linearly. Plus a causal check — zero each latent dimension and measure the
# damage to the task read-out, which says which dimensions the arm actually *relies*
# on rather than merely correlates with.
# %%
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as _tts


def probe_r2(Z, y, seed=0):
    """Held-out R² of a linear read-out from latent Z to target y."""
    Ztr, Zte, ytr, yte = _tts(Z, y, test_size=0.5, random_state=seed)
    p = Ridge(alpha=1.0).fit(Ztr, ytr).predict(Zte)
    ss_res = ((yte - p) ** 2).sum()
    ss_tot = ((yte - yte.mean(axis=0)) ** 2).sum()
    return float(max(1 - ss_res / ss_tot, 0.0))


final_lat = {}
for s in all_snaps:
    if s["epoch"] == CFG.epochs:
        final_lat.setdefault(s["arm"], []).append(s)

probe_rows, abl_rows = [], []
for arm, snaps in final_lat.items():
    for sn in snaps:
        ev = splits[sn["seed"]]
        n = sn["latent"].shape[0]
        Z = sn["latent"].reshape(-1, CFG.latent)                    # (n*T, latent)
        y_task = ev["target"][:n].numpy().reshape(-1, 2)
        y_bg = np.repeat(ev["bg"][:n].numpy()[:, None, :], CFG.n_frames, 1).reshape(-1, 3)
        y_dis = ev["pos"][:n, :, 1:, :].numpy().reshape(n * CFG.n_frames, -1)

        probe_rows.append({"arm": arm, "seed": sn["seed"],
                           "task_r2": probe_r2(Z, y_task),
                           "bg_r2": probe_r2(Z, y_bg),
                           "distractor_r2": probe_r2(Z, y_dis)})

        # Causal: ablate one dim at a time and refit nothing — measure the drop.
        Ztr, Zte, ytr, yte = _tts(Z, y_task, test_size=0.5, random_state=0)
        reg = Ridge(alpha=1.0).fit(Ztr, ytr)
        base = ((yte - reg.predict(Zte)) ** 2).mean()
        for d in range(CFG.latent):
            Zab = Zte.copy(); Zab[:, d] = 0.0
            abl_rows.append({"arm": arm, "seed": sn["seed"], "dim": d,
                             "mse_increase": float(((yte - reg.predict(Zab)) ** 2).mean() - base)})

probe_df = pd.DataFrame(probe_rows)
abl_df = pd.DataFrame(abl_rows)
probe_df.to_csv(f"{RESULTS}/latent_probes.csv", index=False)
abl_df.to_csv(f"{RESULTS}/latent_ablation.csv", index=False)


def fig_latent_content():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))

    ax = axes[0]
    w, arms = 0.26, list(CFG.arms)
    xs = np.arange(len(arms))
    for k, (col, lab, c) in enumerate([("task_r2", "task (target xy)", "#2a9d8f"),
                                       ("bg_r2", "background colour", "#bbbbbb"),
                                       ("distractor_r2", "distractor positions", "#888888")]):
        vals = [probe_df[probe_df.arm == a][col].mean() for a in arms]
        ax.bar(xs + (k - 1) * w, vals, w, label=lab, color=c)
    ax.set_xticks(xs); ax.set_xticklabels([a.split("_")[0] for a in arms])
    ax.set_ylabel("linear probe R²"); ax.set_ylim(0, 1.05)
    ax.set_title("What is linearly readable from the latent?")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    for a in arms:
        g = abl_df[abl_df.arm == a].groupby("dim")["mse_increase"].mean()
        share = g.sort_values(ascending=False).values
        share = share / (share.sum() + 1e-12)
        ax.plot(range(1, len(share) + 1), np.cumsum(share), "-o", ms=3,
                color=ARM_COLOR[a], lw=2.2, label=ARM_LABEL[a])
    ax.set_xlabel("latent dims, ranked by causal importance")
    ax.set_ylabel("cumulative share of task importance")
    ax.set_title("Is the task concentrated or smeared?")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.02)

    ax = axes[2]
    for a in arms:
        g = abl_df[abl_df.arm == a].groupby("dim")["mse_increase"].mean()
        v = np.sort(g.values)[::-1]
        v = v / (v.sum() + 1e-12)
        ax.bar(np.arange(len(v)) + .8 * (arms.index(a) / len(arms) - .5),
               v, width=.8 / len(arms), color=ARM_COLOR[a],
               label=ARM_LABEL[a].split(" · ")[0])
    ax.set_xlabel("dim rank"); ax.set_ylabel("share of task importance")
    ax.set_title("Per-dimension causal importance"); ax.legend(fontsize=8)
    fig.tight_layout()
    plt.savefig(f"{RESULTS}/fig06_latent_content.png", bbox_inches="tight")
    plt.show()


fig_latent_content()
# %% [markdown]
# ### 6f. Latent geometry
#
# PCA picks the directions of greatest **variance**, which in this dataset is the
# background and the distractors — so a raw PCA scatter tends to hide the task. We
# therefore also plot a **task-aligned** projection: axis 1 is the latent direction a
# probe uses to read out the target's x, axis 2 the direction for its y. Colouring by
# the true position should make a smooth 2-D gradient if the task is cleanly encoded.
# %%
def fig_latent_geometry():
    arms = list(CFG.arms)
    fig, axes = plt.subplots(2, len(arms), figsize=(3.5 * len(arms), 7))
    sd = CFG.seeds[0]
    for j, arm in enumerate(arms):
        sn = [s for s in final_lat[arm] if s["seed"] == sd][0]
        ev = splits[sd]
        n = sn["latent"].shape[0]
        Z = sn["latent"].reshape(-1, CFG.latent)
        y = ev["target"][:n].numpy().reshape(-1, 2)
        Zc = Z - Z.mean(0)

        P = PCA(n_components=2).fit_transform(Zc)
        ax = axes[0, j]
        ax.scatter(P[:, 0], P[:, 1], c=y[:, 0], s=3, cmap="viridis", alpha=.6)
        ax.set_title(ARM_LABEL[arm].split(" · ")[0] + "  — raw PCA", fontsize=10)
        ax.set_xlabel("PC1"); ax.grid(False)
        if j == 0:
            ax.set_ylabel("PC2\n(colour = true x)")

        # Task-aligned axes: the directions a linear probe uses for x and y.
        W = Ridge(alpha=1.0).fit(Zc, y).coef_          # (2, latent)
        w1 = W[0] / (np.linalg.norm(W[0]) + 1e-9)
        w2 = W[1] - (W[1] @ w1) * w1                   # orthogonalise
        w2 = w2 / (np.linalg.norm(w2) + 1e-9)
        A = np.stack([Zc @ w1, Zc @ w2], 1)
        ax = axes[1, j]
        ax.scatter(A[:, 0], A[:, 1], c=y[:, 0], s=3, cmap="viridis", alpha=.6)
        ax.set_xlabel("task axis (x)"); ax.grid(False)
        if j == 0:
            ax.set_ylabel("task axis (y)\n(colour = true x)")
        ax.set_title("task-aligned", fontsize=10)
    fig.suptitle("Latent geometry — top: variance axes, bottom: task axes", y=1.0)
    fig.tight_layout()
    plt.savefig(f"{RESULTS}/fig07_latent_geometry.png", bbox_inches="tight")
    plt.show()


fig_latent_geometry()
# %% [markdown]
# ## 7. Animations
#
# ### 7a. The money GIF — watch each arm imagine forward
#
# During the **observing** phase the models see real frames. Then the pixels are cut
# off and each arm runs on its own hidden state alone. The true target is the cyan
# ring; each arm's guess is its own coloured marker.
#
# The frames keep being *drawn* after the cutoff so you can see the ground truth —
# but the models are no longer receiving them. And note that no arm's prediction is
# ever rendered as a reconstructed image: `B_task` has no decoder, so drawing one
# would misstate what these models do. Predictions are positions, drawn as markers.
# %%
def gif_imagination(path, n_traj=4, fps=6, seed=None):
    seed = CFG.seeds[0] if seed is None else seed
    ev = splits[seed]
    t_np = ev["target"].numpy()
    # Show trajectories that bounce inside the window — that is the interesting
    # regime — and the caption says so, so this is illustration, not a quiet boost.
    cand = np.flatnonzero(ev["bounce_in_window"])
    if len(cand) < n_traj:
        cand = np.arange(len(t_np))
    idx = np.random.default_rng(0).choice(cand, n_traj, replace=False)

    with torch.no_grad():
        frames = render(ev["pos"][idx], ev["bg"][idx], ev["dcol"][idx],
                        CFG, device="cpu").numpy()

    preds = {a: all_preds[(seed, a)][idx] for a in CFG.arms}
    total = CFG.context + CFG.horizon
    S = CFG.img_size - 1

    fig, axes = plt.subplots(1, n_traj, figsize=(2.6 * n_traj, 3.2))
    axes = np.atleast_1d(axes)
    ims, truth, marks, notes = [], [], {a: [] for a in CFG.arms}, []
    for i, ax in enumerate(axes):
        ims.append(ax.imshow(frames[i, 0].transpose(1, 2, 0)))
        tm, = ax.plot([], [], "o", mfc="none", mec="#00e5ff", mew=2.2, ms=15)
        truth.append(tm)
        for a in CFG.arms:
            m, = ax.plot([], [], "X", color=ARM_COLOR[a], ms=9, mec="white", mew=0.8)
            marks[a].append(m)
        notes.append(ax.text(0.5, -0.09, "", transform=ax.transAxes, ha="center",
                             fontsize=9, color="#444"))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    handles = [plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#00e5ff",
                          mew=2, label="true target")]
    handles += [plt.Line2D([], [], marker="X", ls="", color=ARM_COLOR[a],
                           label=ARM_LABEL[a].split(" - ")[0].split(" · ")[0])
                for a in CFG.arms]
    fig.legend(handles=handles, loc="lower center", ncol=len(CFG.arms) + 1,
               fontsize=9, frameon=False)
    sup = fig.suptitle("")
    fig.tight_layout(rect=[0, 0.16, 1, 0.90])

    def upd(t):
        observing = t < CFG.context
        for i in range(n_traj):
            ims[i].set_data(frames[i, t].transpose(1, 2, 0))
            truth[i].set_data([t_np[idx[i], t, 0] * S], [t_np[idx[i], t, 1] * S])
            for a in CFG.arms:
                if observing:
                    marks[a][i].set_data([], [])
                else:
                    p = preds[a][i, t - CFG.context]
                    marks[a][i].set_data([p[0] * S], [p[1] * S])
            notes[i].set_text("observing" if observing
                              else "imagining +%d" % (t - CFG.context + 1))
        sup.set_text(("OBSERVING - real frames going in" if observing else
                      "IMAGINING - pixels cut off, %d step(s) blind"
                      % (t - CFG.context + 1)) + "   (frame %d)" % t)
        return ims + truth + [m for a in CFG.arms for m in marks[a]] + [sup]

    FuncAnimation(fig, upd, frames=total, blit=False).save(
        path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote " + path)


gif_imagination(RESULTS + "/gif02_imagination.gif")
# %% [markdown]
# ### 7b. The error curve growing over training
# %%
def gif_error_growth(path, fps=4):
    """The horizon curve for every arm, rebuilt at each logged epoch."""
    epochs = sorted({s["epoch"] for s in all_snaps})
    curves = {}
    for ep in epochs:
        for a in CFG.arms:
            ds = []
            for sn in all_snaps:
                if sn["epoch"] == ep and sn["arm"] == a:
                    ds.append(err_curve(sn["pred"],
                                        splits[sn["seed"]]["target"].numpy(),
                                        CFG).mean(0))
            if ds:
                curves[(ep, a)] = np.mean(ds, 0)

    ymax = max(c.max() for c in curves.values()) * 1.12
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    lines = {a: ax.plot([], [], "-o", ms=4, lw=2.6, color=ARM_COLOR[a],
                        label=ARM_LABEL[a])[0] for a in CFG.arms}
    hold = summary[summary.model == "baseline_hold"].sort_values("horizon")["mean"]
    ax.plot(range(1, CFG.horizon + 1), hold.values, "--", color="#bbb",
            label="baseline: hold last")
    ax.set_xlim(0.5, CFG.horizon + 0.5); ax.set_ylim(0, ymax)
    ax.set_xlabel("prediction horizon"); ax.set_ylabel("position error (px)")
    ax.legend(fontsize=8.5, loc="upper left")
    ttl = ax.set_title("")

    def upd(i):
        ep = epochs[i]
        for a in CFG.arms:
            if (ep, a) in curves:
                lines[a].set_data(range(1, CFG.horizon + 1), curves[(ep, a)])
        ttl.set_text("Blind-rollout error vs horizon - epoch %d" % ep)
        return list(lines.values()) + [ttl]

    FuncAnimation(fig, upd, frames=len(epochs), blit=False).save(
        path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote " + path)


gif_error_growth(RESULTS + "/gif03_error_growth.gif")
# %% [markdown]
# ### 7c. Latent structure emerging during training
# %%
def gif_latent_evolution(path, fps=4, seed=None):
    """Task-aligned latent projection for every arm, over training epochs."""
    seed = CFG.seeds[0] if seed is None else seed
    epochs = sorted({s["epoch"] for s in all_snaps})
    ev = splits[seed]

    proj = {}
    for ep in epochs:
        for a in CFG.arms:
            sn = [s for s in all_snaps
                  if s["epoch"] == ep and s["arm"] == a and s["seed"] == seed]
            if not sn:
                continue
            Z = sn[0]["latent"].reshape(-1, CFG.latent)
            n = sn[0]["latent"].shape[0]
            y = ev["target"][:n].numpy().reshape(-1, 2)
            Zc = Z - Z.mean(0)
            W = Ridge(alpha=1.0).fit(Zc, y).coef_
            w1 = W[0] / (np.linalg.norm(W[0]) + 1e-9)
            w2 = W[1] - (W[1] @ w1) * w1
            w2 = w2 / (np.linalg.norm(w2) + 1e-9)
            A = np.stack([Zc @ w1, Zc @ w2], 1)
            A = A / (A.std(0) + 1e-9)          # comparable scales across arms
            proj[(ep, a)] = (A[::3], y[::3, 0])

    fig, axes = plt.subplots(1, len(CFG.arms), figsize=(3.3 * len(CFG.arms), 3.8))
    axes = np.atleast_1d(axes)
    scs = {}
    for ax, a in zip(axes, CFG.arms):
        scs[a] = ax.scatter([], [], s=4, c=[], cmap="viridis", vmin=0, vmax=1)
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.grid(False)
        ax.set_title(ARM_LABEL[a].split(" · ")[0], fontsize=10)
        ax.set_xlabel("task axis x")
    axes[0].set_ylabel("task axis y\n(colour = true x)")
    sup = fig.suptitle("")
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    def upd(i):
        ep = epochs[i]
        for a in CFG.arms:
            if (ep, a) in proj:
                A, c = proj[(ep, a)]
                scs[a].set_offsets(A)
                scs[a].set_array(c)
        sup.set_text("Task-aligned latent projection - epoch %d" % ep)
        return list(scs.values()) + [sup]

    FuncAnimation(fig, upd, frames=len(epochs), blit=False).save(
        path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote " + path)


gif_latent_evolution(RESULTS + "/gif04_latent_evolution.gif")
# %% [markdown]
# ## 8. Verdict
#
# The cell below writes the summary tables and states what the numbers say — chosen
# by the data, not by what we hoped. It checks three *different* questions that are
# easy to conflate:
#
# 1. **Is B less accurate?** (level)
# 2. **Does B degrade faster?** (rate — a model can be uniformly worse while decaying
#    at exactly the same speed)
# 3. **Is any of it robust?** (do the seeds even agree on the sign)
#
# It also runs the control that the small-scale experiment could not settle: if
# `C_scene` and `D_latent` match `A_recon`, the benefit is *not* about pixels — it is
# just "a second loss helps". If only `A_recon` helps, reconstruction is doing
# something specific.
# %%
def verdict():
    lines = []
    n_seeds = len(CFG.seeds)
    H = CFG.horizon

    m1 = {a: summary[(summary.model == a) & (summary.horizon == 1)]["mean"].iloc[0]
          for a in CFG.arms}
    mH = {a: summary[(summary.model == a) & (summary.horizon == H)]["mean"].iloc[0]
          for a in CFG.arms}
    ratio = deg_df.groupby("model")["ratio"].mean().to_dict()

    lines.append("=" * 74)
    lines.append("RESULT  (%d seeds, %d-step blind rollout, %dpx frames)"
                 % (n_seeds, H, CFG.img_size))
    lines.append("=" * 74)
    lines.append("")
    lines.append("%-34s %9s %9s %9s" % ("arm", "err h=1", "err h=%d" % H, "growth"))
    for a in CFG.arms:
        lines.append("%-34s %9.3f %9.3f %8.2fx"
                     % (ARM_LABEL[a], m1[a], mH[a], ratio[a]))
    for b in ("baseline_hold", "baseline_linear"):
        r = summary[summary.model == b]
        lines.append("%-34s %9.3f %9.3f %8.2fx"
                     % (ARM_LABEL[b], r[r.horizon == 1]["mean"].iloc[0],
                        r[r.horizon == H]["mean"].iloc[0], ratio.get(b, float("nan"))))
    lines.append("")

    # --- 1. level -------------------------------------------------------
    best = min(CFG.arms, key=lambda a: mH[a])
    lines.append("1. LEVEL   most accurate at h=%d : %s" % (H, ARM_LABEL[best]))
    gap = mH["B_task"] - mH[best]
    if best == "B_task":
        lines.append("           the task-only arm is the most accurate arm.")
    else:
        lines.append("           task-only is %+.3f px behind it (%.1f%% of a pixel)."
                     % (gap, 100 * gap))

    # --- 2. rate --------------------------------------------------------
    slowest = min(CFG.arms, key=lambda a: ratio[a])
    lines.append("")
    lines.append("2. RATE    slowest degradation : %s (%.2fx)"
                 % (ARM_LABEL[slowest], ratio[slowest]))
    lines.append("           task-only grows %.2fx vs %.2fx for A_recon."
                 % (ratio["B_task"], ratio["A_recon"]))
    lines.append("           -> B degrades %s than the reconstruction arm."
                 % ("FASTER" if ratio["B_task"] > ratio["A_recon"] else "SLOWER"))

    # --- 3. robustness --------------------------------------------------
    lines.append("")
    lines.append("3. ROBUST  per-horizon sign agreement across the %d seeds:" % n_seeds)
    for arm in [a for a in CFG.arms if a != "B_task"]:
        g = paired_df[paired_df["vs"] == arm]
        piv = g.pivot(index="horizon", columns="seed", values="delta_px")
        unanimous = int(((piv > 0).all(axis=1) | (piv < 0).all(axis=1)).sum())
        at_H = piv.loc[H]
        lines.append("           B vs %-9s unanimous at %2d/%d horizons; "
                     "at h=%d, %d/%d seeds put B behind"
                     % (arm, unanimous, H, H, int((at_H > 0).sum()), n_seeds))
    if n_seeds < 3:
        lines.append("           !! %d seed(s) only - this says nothing about "
                     "robustness. Raise CFG.seeds." % n_seeds)

    # --- 4. the control -------------------------------------------------
    lines.append("")
    lines.append("4. CONTROL is the benefit about PIXELS or just about a 2nd loss?")
    if "C_scene" in CFG.arms and "D_latent" in CFG.arms:
        aid = mH["B_task"] - mH["A_recon"]      # >0 => recon helped
        cid = mH["B_task"] - mH["C_scene"]
        did = mH["B_task"] - mH["D_latent"]
        lines.append("           benefit over task-only at h=%d, in px:" % H)
        lines.append("             reconstruction (A) : %+.3f" % aid)
        lines.append("             scene aux, no pixels (C) : %+.3f" % cid)
        lines.append("             next-latent, no decoder (D) : %+.3f" % did)
        if aid > 0 and max(cid, did) >= 0.6 * aid:
            lines.append("           -> a NON-PIXEL auxiliary recovers most of the")
            lines.append("              benefit: this looks like a generic")
            lines.append("              second-loss effect, NOT something special")
            lines.append("              about reconstructing pixels.")
        elif aid > max(cid, did) and aid > 0:
            lines.append("           -> reconstruction helps MORE than the non-pixel")
            lines.append("              auxiliaries: pixels appear to add something")
            lines.append("              the others do not.")
        else:
            lines.append("           -> no arm shows a clear benefit over task-only.")
    lines.append("")
    lines.append("=" * 74)
    lines.append("Read points 1 and 2 as separate claims. 'Worse' and 'degrades")
    lines.append("faster' are different questions and can disagree.")
    lines.append("=" * 74)

    txt = "\n".join(lines)
    print(txt)
    with open(RESULTS + "/VERDICT.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return txt


_ = verdict()
# %% [markdown]
# ### Limitations — read before quoting any of this
#
# * **Loss weights are not tuned.** `w_recon`, `w_scene` and `w_latent` are all 1.0.
#   Arm A's balance between its two objectives is arbitrary, and a different weight
#   could change the ordering. Sweeping these is the obvious next run.
# * **One architecture, one dynamics.** Constant velocity with reflecting walls is
#   still simple. Richer dynamics (acceleration, collisions between objects,
#   occlusion) could separate the arms where this does not.
# * **Effect sizes are small.** Report them in pixels, next to the seed spread, and
#   resist reading a trend into a gap that a re-seed would erase.
# * **The task is low-dimensional.** Two numbers. A latent with 32 dimensions has
#   plenty of room, so capacity pressure — a big part of the real Hermes argument —
#   is barely being tested here. Shrinking `CFG.latent` is a cheap way to add it.
# %%
# ============================================================================
# 9. Bundle everything for download
# ============================================================================
def bundle(path="hermes_rollout_results.zip"):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(RESULTS)):
            z.write(os.path.join(RESULTS, f), f)
    mb = os.path.getsize(path) / 1e6
    print("wrote %s (%.1f MB)" % (path, mb))
    for f in sorted(os.listdir(RESULTS)):
        print("   " + f)
    return path


_zip = bundle()

try:                      # only works inside Colab; harmless elsewhere
    from google.colab import files
    files.download(_zip)
except Exception as e:
    print("\n(not in Colab, skipping auto-download: %s)" % type(e).__name__)
