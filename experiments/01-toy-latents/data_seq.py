"""
data_seq.py — Trajectory version of the toy dataset, for the rollout experiment.

This EXTENDS data.py; it does not replace it. The static single-frame demo and
its results are untouched. We reuse `render()` verbatim, so a frame here is
pixel-for-pixel the same kind of image as in the static experiment: one white
Gaussian blob on a randomly-coloured, noisy background.

What is new is time. Each trajectory is a short sequence of frames in which the
blob moves with (approximately) constant velocity and bounces off the left and
right walls:

    t_{k+1} = t_k + v_k + eps ,    eps ~ N(0, STEP_NOISE^2)
    if the dot leaves [0, 1] it REFLECTS:  t -> 2-t (or -t), v -> -v

The distractor design carries over unchanged and is the whole point of the demo:

  * The background colour is drawn once per trajectory and held FIXED across its
    frames. It is independent of the motion, so it carries zero information
    about where the dot is or where it is going.
  * Per-pixel noise is redrawn every frame, so it is temporally uncorrelated.

Why reflecting walls rather than a plain straight line: with constant velocity
and no walls, predicting 10 steps ahead is exactly linear extrapolation, and any
model that recovers velocity solves the task perfectly — there would be nothing
for a horizon curve to reveal. Bounces make the dynamics piecewise-linear, so
long-horizon prediction requires the model to represent the wall, not just the
velocity. Because bounces dominate long-horizon error, `bounce_in_horizon` is
returned so the analysis can report errors split by whether a bounce occurs —
otherwise the curves would be an uninterpretable mix of two regimes.
"""

import numpy as np

from data import IMG_SIZE, N_CHANNELS, IMG_DIM, PIXEL_NOISE_STD, render

# ----------------------------------------------------------------------------
# Trajectory constants. Kept small so the whole experiment runs on CPU quickly.
# ----------------------------------------------------------------------------
N_FRAMES = 20        # frames per trajectory
CONTEXT = 5          # frames fed to the model before it must predict blind
HORIZON = 10         # how many steps ahead we ask it to predict
V_MIN, V_MAX = 0.02, 0.06   # |velocity| per step, in units of t (0..1)
STEP_NOISE = 0.005   # process noise on the position update


def _step(t, v):
    """Advance one step with reflection at the [0, 1] walls.

    Reflection is applied after the update, and may need to run twice in the
    (very unlikely) event a single step overshoots the far wall too, so we loop
    until the position is back inside the interval.
    """
    t = t + v
    while t < 0.0 or t > 1.0:
        if t < 0.0:
            t = -t
            v = -v
        if t > 1.0:
            t = 2.0 - t
            v = -v
    return t, v


def make_trajectories(n_traj, n_frames=N_FRAMES, seed=0, noise_std=PIXEL_NOISE_STD):
    """Generate `n_traj` trajectories of `n_frames` frames each.

    Returns a dict with:
      X       : (n, T, IMG_DIM) float32 — flattened frames, the encoder input.
      images  : (n, T, H, W, 3) float32 — same frames, for plotting.
      t       : (n, T) float32          — true blob position at every frame.
      v0      : (n,) float32            — initial velocity (diagnostic only).
      bg      : (n, 3) float32          — per-trajectory background colour.
      n_bounces : (n,) int32            — bounces over the whole trajectory.
      bounce_in_horizon : (n,) bool     — did a bounce occur in the prediction
                                          window that the rollout is scored on,
                                          i.e. between CONTEXT and CONTEXT+HORIZON.

    The model NEVER sees `t`, `v0` or `bg` — they are supervision / analysis
    only. The model only ever sees pixels.
    """
    rng = np.random.default_rng(seed)

    t = np.zeros((n_traj, n_frames), dtype=np.float32)
    v0 = np.zeros(n_traj, dtype=np.float32)
    n_bounces = np.zeros(n_traj, dtype=np.int32)
    bounce_in_horizon = np.zeros(n_traj, dtype=bool)

    # Background colour: one per trajectory, independent of the motion.
    bg = rng.uniform(0.0, 1.0, size=(n_traj, 3)).astype(np.float32)

    for i in range(n_traj):
        # Start away from the walls so the first few frames are usually clean.
        pos = float(rng.uniform(0.15, 0.85))
        # Velocity: random magnitude, random direction.
        vel = float(rng.uniform(V_MIN, V_MAX) * rng.choice([-1.0, 1.0]))
        v0[i] = vel

        t[i, 0] = pos
        for k in range(1, n_frames):
            vel_before = vel
            pos, vel = _step(pos + float(rng.normal(0.0, STEP_NOISE)) - 0.0, vel)
            t[i, k] = pos
            if vel != vel_before:          # sign flipped => we hit a wall
                n_bounces[i] += 1
                if CONTEXT <= k < CONTEXT + HORIZON:
                    bounce_in_horizon[i] = True

    # Render every frame. Background colour is constant within a trajectory;
    # pixel noise is redrawn per frame (that is what `render` does internally).
    images = np.empty((n_traj, n_frames, IMG_SIZE, IMG_SIZE, N_CHANNELS),
                      dtype=np.float32)
    for i in range(n_traj):
        for k in range(n_frames):
            images[i, k] = render(t[i, k], bg[i], rng, noise_std=noise_std)

    X = images.reshape(n_traj, n_frames, IMG_DIM).astype(np.float32)
    return {
        "X": X, "images": images, "t": t, "v0": v0, "bg": bg,
        "n_bounces": n_bounces, "bounce_in_horizon": bounce_in_horizon,
    }


if __name__ == "__main__":
    # Self-check: confirm the dynamics and the distractor design behave as
    # described, and report how often bounces land inside the scored window.
    d = make_trajectories(500, seed=0)
    t = d["t"]
    print(f"X shape                    : {d['X'].shape}  (n, T, pixels)")
    print(f"position range             : [{t.min():.3f}, {t.max():.3f}]")

    step = np.diff(t, axis=1)
    print(f"mean |step| per frame      : {np.abs(step).mean():.4f} "
          f"(configured |v| in [{V_MIN}, {V_MAX}])")
    print(f"trajectories with >=1 bounce: {(d['n_bounces'] > 0).mean() * 100:.1f}%")
    print(f"bounce inside scored window: {d['bounce_in_horizon'].mean() * 100:.1f}%"
          f"  (frames {CONTEXT}..{CONTEXT + HORIZON})")

    # The background must carry NO information about the motion. Correlate the
    # per-trajectory background luminance against position and velocity.
    lum = d["bg"] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    print(f"corr(bg luminance, start pos): {np.corrcoef(lum, t[:, 0])[0, 1]:+.4f}")
    print(f"corr(bg luminance, velocity) : {np.corrcoef(lum, d['v0'])[0, 1]:+.4f}")
    print("=> background is independent of the dynamics, as intended.")
