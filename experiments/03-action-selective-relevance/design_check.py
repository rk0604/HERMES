"""Experiment 3 design check. Numpy only, no networks, no training.

1. Renders the proposed frames at 48px and 32px so object legibility can be judged by eye.
2. Runs TRUE-dynamics planners and trivial baselines to check, before any model exists,
   that (a) momentum makes multi-step planning necessary and (b) ignoring the background
   selector actually costs return.
"""
import sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
HUE_BANDS = ((0.05, 0.45), (0.55, 0.95))   # selector 0 -> ring active, 1 -> plus active
DIST_HUE_HOLDOUT = (0.60, 0.72)            # reserved for the unseen-colour shift test


def sample_episodes(E, rng, min_sep=0.35):
    sel = rng.integers(0, 2, E)
    lo = np.where(sel == 0, HUE_BANDS[0][0], HUE_BANDS[1][0])
    hi = np.where(sel == 0, HUE_BANDS[0][1], HUE_BANDS[1][1])
    bg = np.stack([rng.uniform(lo, hi), rng.uniform(0.5, 0.9, E),
                   rng.uniform(0.45, 0.8, E)], -1)
    goals = rng.uniform(0.15, 0.85, (E, 2, 2))
    bad = np.linalg.norm(goals[:, 0] - goals[:, 1], axis=-1) < min_sep
    while bad.any():
        goals[bad] = rng.uniform(0.15, 0.85, (bad.sum(), 2, 2))
        bad = np.linalg.norm(goals[:, 0] - goals[:, 1], axis=-1) < min_sep
    p0 = rng.uniform(0.1, 0.9, (E, 2))
    ang = rng.uniform(0, 2 * np.pi, E)
    v0 = np.stack([np.cos(ang), np.sin(ang)], -1) * rng.uniform(0, 0.025, E)[:, None]
    return dict(sel=sel, bg=bg, goals=goals.astype(np.float32),
                p0=p0.astype(np.float32), v0=v0.astype(np.float32))


def sample_dhue(shape, rng):
    w = DIST_HUE_HOLDOUT[1] - DIST_HUE_HOLDOUT[0]
    u = rng.uniform(0, 1 - w, shape)
    return np.where(u >= DIST_HUE_HOLDOUT[0], u + w, u)


# ---------------------------------------------------------------- rendering
def render(img, p, goals, dpos, dhue, bg, rng, noise=0.05):
    s = img / 48.0
    yy, xx = np.mgrid[0:img, 0:img].astype(np.float32)
    im = np.broadcast_to(hsv_to_rgb(bg), (img, img, 3)) + rng.normal(0, noise, (img, img, 3))

    def over(im, a, col):
        return im * (1 - a[..., None]) + np.asarray(col, np.float32) * a[..., None]

    def blob(c, sig):
        cx, cy = c * (img - 1)
        return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))

    sig = 2.2 * s
    for c, h in zip(dpos, dhue):                                   # distractors
        im = over(im, blob(c, sig), hsv_to_rgb([h, 1, 1]))
    cx, cy = goals[0] * (img - 1)                                  # goal 0: hollow ring
    r = np.hypot(xx - cx, yy - cy)
    im = over(im, np.clip(1.0 - np.abs(r - 3.5 * s) / max(1.0 * s, 0.8), 0, 1), (0, 0, 0))
    cx, cy = goals[1] * (img - 1)                                  # goal 1: plus
    dx, dy = np.abs(xx - cx), np.abs(yy - cy)
    L, hw = 4.0 * s, max(0.9 * s, 0.7)
    bar = lambda u, v: np.clip(hw + 0.5 - u, 0, 1) * np.clip(L + 0.5 - v, 0, 1)
    im = over(im, np.maximum(bar(dy, dx), bar(dx, dy)), (0, 0, 0))
    im = over(im, blob(p, sig), (1, 1, 1))                         # agent on top
    return np.clip(im, 0, 1)


def figure(path):
    rng = np.random.default_rng(3)
    n = 8
    ep = sample_episodes(n, rng)
    dpos = rng.uniform(0.05, 0.95, (n, 2, 2))
    dhue = sample_dhue((n, 2), rng)
    fig, axes = plt.subplots(3, n, figsize=(1.9 * n, 6.4),
                             gridspec_kw={"height_ratios": [1, 1, 0.18]})
    for j in range(n):
        for i, img in enumerate((48, 32)):
            fr = render(img, ep["p0"][j], ep["goals"][j], dpos[j], dhue[j], ep["bg"][j],
                        np.random.default_rng(j))
            ax = axes[i, j]
            ax.imshow(fr, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title("active: %s\nhue %.2f" % (("ring", "plus")[ep["sel"][j]],
                                                      ep["bg"][j][0]), fontsize=9)
        axes[0, 0].set_ylabel("48 px"); axes[1, 0].set_ylabel("32 px")
    gs = axes[2, 0].get_gridspec()
    for ax in axes[2]:
        ax.remove()
    ax = fig.add_subplot(gs[2, :])
    hues = np.linspace(0, 1, 400)
    ax.imshow(hsv_to_rgb(np.stack([hues, np.full_like(hues, .7), np.full_like(hues, .65)],
                                  -1))[None], aspect="auto", extent=[0, 1, 0, 1])
    for a, b in ((0, .05), (.45, .55), (.95, 1)):
        ax.axvspan(a, b, color="k", alpha=.55)
    ax.text(.25, .5, "ring active", ha="center", va="center", fontsize=9, color="w")
    ax.text(.75, .5, "plus active", ha="center", va="center", fontsize=9, color="w")
    ax.set_yticks([]); ax.set_xlabel("background hue (dark = excluded dead zones)")
    fig.suptitle("Proposed Experiment 3 frames: white = agent, black ring / plus = goals, "
                 "coloured blobs = distractors", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print("wrote", path)


# ---------------------------------------------------------------- dynamics
def step(p, v, a, d, alpha):
    v = d * v + alpha * a
    p = p + v
    hit = (p < 0) | (p > 1)
    return np.clip(p, 0, 1), np.where(hit, 0.0, v).astype(np.float32)


def rew(p, g, bonus=0.5, sig=0.05):
    dist = np.linalg.norm(p - g, axis=-1)
    return -dist + bonus * np.exp(-dist ** 2 / (2 * sig ** 2))


def cem_plan(p, v, targets, H, dyn, rng, mu, N=200, K=20, iters=4, sd0=0.7):
    E = p.shape[0]
    sd = np.full_like(mu, sd0)
    for _ in range(iters):
        A = np.clip(mu[:, None] + sd[:, None] *
                    rng.standard_normal((E, N, H, 2)).astype(np.float32), -1, 1)
        pp = np.repeat(p[:, None], N, 1); vv = np.repeat(v[:, None], N, 1)
        R = np.zeros((E, N), np.float32)
        for k in range(H):
            pp, vv = step(pp, vv, A[:, :, k], *dyn)
            for g, w in targets:
                R += w * rew(pp, g[:, None])
        top = np.argpartition(-R, K, axis=1)[:, :K]
        el = np.take_along_axis(A, top[:, :, None, None], 1)
        mu, sd = el.mean(1), el.std(1) + 0.05
    return mu


def run(policy, ep, dyn, T=40, H=12, seed=7):
    rng = np.random.default_rng(seed)
    E = len(ep["sel"]); ix = np.arange(E)
    p, v = ep["p0"].copy(), ep["v0"].copy()
    ga, gi = ep["goals"][ix, ep["sel"]], ep["goals"][ix, 1 - ep["sel"]]
    mu = np.zeros((E, H, 2), np.float32)
    rs = []
    for t in range(T):
        if policy == "noop":
            a = np.zeros((E, 2), np.float32)
        elif policy == "random":
            a = rng.uniform(-1, 1, (E, 2)).astype(np.float32)
        else:
            tg = {"oracle": [(ga, 1.0)], "wrong_goal": [(gi, 1.0)],
                  "colour_blind": [(ep["goals"][:, 0], .5), (ep["goals"][:, 1], .5)]}[policy]
            mu = cem_plan(p, v, tg, H, dyn, rng, mu)
            a = mu[:, 0]
            mu = np.concatenate([mu[:, 1:], np.zeros((E, 1, 2), np.float32)], 1)
        p, v = step(p, v, a, *dyn)
        rs.append(rew(p, ga))
    rs = np.stack(rs, 1)
    dA, dI = np.linalg.norm(p - ga, axis=-1), np.linalg.norm(p - gi, axis=-1)
    return dict(ret=rs.mean(1), late=rs[:, T // 2:].mean(1), success=dA < 0.05,
                chose_active=dA < dI, final_px=dA * 47)


if __name__ == "__main__":
    figure(OUT + "/design_frames.png")

    for d, al in ((0.9, 0.005), (0.8, 0.01)):
        vt = al / (1 - d)
        k_stop = np.log(0.5) / np.log(d)
        print(f"\ndamping {d}, max_accel {al}: terminal speed {vt:.3f}/step "
              f"({vt*47:.2f} px), coast-to-stop distance {vt*d/(1-d):.2f} arena, "
              f"full-brake stop time {k_stop:.1f} steps")

    ep = sample_episodes(100, np.random.default_rng(11))
    print("\nepisodes: 100 (identical across rows), T=40, CEM N=200 K=20 iters=4")
    print("%-14s %-12s %3s  %8s %8s %8s %8s %9s" %
          ("policy", "dynamics", "H", "ret/step", "late", "success", "chose_act", "final_px"))
    rows = [("noop", (0.9, 0.005), 0), ("random", (0.9, 0.005), 0)]
    rows += [("oracle", (0.9, 0.005), h) for h in (1, 3, 6, 12, 20)]
    rows += [("colour_blind", (0.9, 0.005), 12), ("wrong_goal", (0.9, 0.005), 12)]
    rows += [("oracle", (0.8, 0.01), h) for h in (1, 12)]
    rows += [("oracle", (0.0, 0.05), h) for h in (1, 12)]
    t0 = time.time()
    for pol, dyn, h in rows:
        r = run(pol, ep, dyn, H=max(h, 1))
        print("%-14s d=%.1f a=%.3f %3s  %+8.3f %+8.3f %7.0f%% %8.0f%% %9.2f" %
              (pol, dyn[0], dyn[1], h or "-", r["ret"].mean(), r["late"].mean(),
               100 * r["success"].mean(), 100 * r["chose_active"].mean(),
               r["final_px"].mean()))
    print("\n(%.0fs)" % (time.time() - t0))
