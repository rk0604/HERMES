"""
train_rollout.py — Multi-step rollout experiment (ADDITION to the static demo).

Question
--------
The static demo showed that a task-only latent (B) reads out position more
accurately than a reconstruction latent (A) from a single frame. That is a
one-step result. The claim actually at stake for Hermes is temporal: does a
task-only latent still hold up when the model has to imagine forward several
steps with no new observations, or does it degrade faster than a latent that was
forced to model the whole scene?

Setup
-----
Two models, identical architecture and identical initial weights (asserted, not
assumed), differing ONLY in the loss:

    A  "recon + task"  : task loss + LAMBDA_RECON * reconstruction loss
    B  "task-only"     : task loss only; no decoder exists in the model

Both are trained end to end. Both receive the task gradient, so this is a fair
comparison and isolates the effect of the extra reconstruction pressure.

Training feeds REAL frames at every step (teacher forcing); the model's own
predictions are never fed back. From each anchor timestep the GRU then rolls
forward HORIZON steps on zero input, exactly as it will at test time.

Nothing here touches the static experiment. It writes only rollout_* files.

Run:  python train_rollout.py     (after / independently of train.py)
"""

import os
import time

import numpy as np
import pandas as pd
import torch

from data import IMG_SIZE
from data_seq import make_trajectories, CONTEXT, HORIZON, N_FRAMES
from model_seq import (RolloutModel, build_pair, assert_same_init, count_params,
                       LAMBDA_RECON)

# ----------------------------------------------------------------------------
# Configuration. Small on purpose — the whole script runs in ~1-2 min on CPU.
# ----------------------------------------------------------------------------
SEED = 0
SEEDS = [0, 1, 2, 3, 4]   # repeat the whole experiment; a one-seed gap is not
                          # evidence, and the A/B difference here is small
N_TRAIN_TRAJ = 2000
N_EVAL_TRAJ = 500
EPOCHS = 80       # task loss is still falling at 60; 80 gets both closer to
                  # convergence so the comparison is not just about who trains faster
BATCH = 128       # 128 is ~4x faster per epoch than 64 here: the encoder/decoder
                  # matmuls are big enough to keep BLAS busy only at this size.
LR = 2e-3
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Anchors: timesteps we predict from. `a` needs a+HORIZON <= N_FRAMES-1, and we
# want at least CONTEXT real frames consumed first, so a >= CONTEXT-1.
ANCHORS = list(range(CONTEXT - 1, N_FRAMES - HORIZON))

PX = IMG_SIZE - 1  # convert position units (0..1) into pixels for reporting


def to_tensor(d):
    return torch.from_numpy(d["X"]), torch.from_numpy(d["t"])


def train_one(model, X, t, name, epochs=EPOCHS, batch=BATCH, lr=LR, seed=SEED,
              verbose=True):
    """Train one model, returning a per-epoch loss log.

    Both models are trained with identical hyper-parameters and identical batch
    ORDER (the shuffling RNG is re-seeded per model), so any difference in the
    result comes from the loss, not from the data order.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    log = []

    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        tot_task = tot_recon = nb = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, tb = X[idx], t[idx]

            task_loss, recon_loss = model.losses(xb, tb, ANCHORS)
            loss = task_loss + (model.decoder is not None) * recon_loss * LAMBDA_RECON

            opt.zero_grad()
            loss.backward()
            # Recurrent nets can spike; clipping keeps both runs comparable.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tot_task += task_loss.detach().item()
            tot_recon += recon_loss.detach().item()
            nb += 1

        log.append({"model": name, "epoch": ep,
                    "task_mse": tot_task / nb, "recon_mse": tot_recon / nb})
        if verbose and (ep % 20 == 0 or ep == 1):
            print(f"  [{name}] epoch {ep:3d}  task_mse={tot_task/nb:.5f}  "
                  f"recon_mse={tot_recon/nb:.5f}")
    return log


# ---------------------------------------------------------------------------
# Evaluation: the actual blind rollout
# ---------------------------------------------------------------------------
def rollout_predictions(model, X, context=CONTEXT, horizon=HORIZON):
    """Feed `context` real frames, then predict `horizon` steps with no input."""
    model.eval()
    with torch.no_grad():
        return model.predict_from_frames(X, context=context,
                                         horizon=horizon).cpu().numpy()


def baseline_constant(t_np, context=CONTEXT, horizon=HORIZON):
    """Predict the last observed position for every horizon.

    This is the do-nothing floor. Any model that has learned no dynamics at all
    should land here; a model worse than this is actively harmful.

    Note it is given the TRUE position at the end of the context window, so it
    is slightly favoured relative to the models (which only ever see pixels).
    That is deliberate — it makes it a conservative floor.
    """
    last = t_np[:, context - 1][:, None]
    return np.repeat(last, horizon, axis=1)


def baseline_linear(t_np, context=CONTEXT, horizon=HORIZON):
    """Fit velocity from the true context positions and extrapolate linearly.

    With constant-velocity motion and no walls this would be near-optimal, so it
    is the natural ceiling — EXCEPT that it knows nothing about the walls, so it
    walks straight through them. Comparing the models against it therefore says
    something specific: beating it means having learned the bounce.
    """
    v = (t_np[:, context - 1] - t_np[:, 0]) / max(context - 1, 1)
    steps = np.arange(1, horizon + 1)[None, :]
    return t_np[:, context - 1][:, None] + v[:, None] * steps


def score(pred, t_np, mask=None, context=CONTEXT, horizon=HORIZON, n_boot=1000,
          seed=0):
    """Per-horizon error metrics, with a bootstrap CI over trajectories.

    Returns a list of dicts, one per horizon step.
    """
    rows = []
    rng = np.random.default_rng(seed)
    for h in range(1, horizon + 1):
        true = t_np[:, context - 1 + h]
        p = pred[:, h - 1]
        if mask is not None:
            true, p = true[mask], p[mask]
        if len(true) < 2:
            continue
        err = np.abs(p - true) * PX
        ss_res = np.sum((p - true) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # Bootstrap the mean pixel error so the plot can show uncertainty.
        boot = rng.choice(err, size=(n_boot, len(err)), replace=True).mean(axis=1)
        rows.append({
            "horizon": h, "n": int(len(true)),
            "mae_px": float(err.mean()),
            "mae_px_lo": float(np.percentile(boot, 2.5)),
            "mae_px_hi": float(np.percentile(boot, 97.5)),
            "rmse_px": float(np.sqrt(np.mean((p - true) ** 2)) * PX),
            "r2": float(r2),
        })
    return rows


def paired_diff(pred_a, pred_b, t_np, mask=None, context=CONTEXT,
                horizon=HORIZON, n_boot=2000, seed=0):
    """Paired comparison of B against A, per horizon.

    A and B are evaluated on the SAME held-out trajectories, so the per-image
    errors are paired and we should test the difference directly rather than
    ask whether two independent confidence intervals happen to overlap. Pairing
    removes the (large) trajectory-to-trajectory variance and is far more
    sensitive; comparing separate CIs is a well-known way to miss a real effect.

    Positive `delta_px` means B is WORSE than A.
    `frac_boot_favours_A` is the fraction of bootstrap resamples in which B is
    worse — a one-sided bootstrap significance readout.
    """
    rows = []
    rng = np.random.default_rng(seed)
    for h in range(1, horizon + 1):
        true = t_np[:, context - 1 + h]
        ea = np.abs(pred_a[:, h - 1] - true) * PX
        eb = np.abs(pred_b[:, h - 1] - true) * PX
        if mask is not None:
            ea, eb = ea[mask], eb[mask]
        d = eb - ea                                  # >0 => B worse
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boot = d[idx].mean(axis=1)
        rows.append({
            "horizon": h, "n": int(len(d)),
            "delta_px": float(d.mean()),
            "delta_lo": float(np.percentile(boot, 2.5)),
            "delta_hi": float(np.percentile(boot, 97.5)),
            "frac_boot_favours_A": float((boot > 0).mean()),
        })
    return rows


def run_one_seed(seed, verbose=True):
    """Full train + eval for a single seed. Returns (rows, extras)."""
    train_d = make_trajectories(N_TRAIN_TRAJ, seed=1000 * seed)
    eval_d = make_trajectories(N_EVAL_TRAJ, seed=1000 * seed + 1)   # disjoint
    Xtr, ttr = to_tensor(train_d)
    Xte, _ = to_tensor(eval_d)
    t_np = eval_d["t"]

    model_a, model_b = build_pair(seed + 10)
    n_checked = assert_same_init(model_a, model_b)
    if verbose:
        print(f"  A/B verified identical across {n_checked} shared tensors "
              f"({count_params(model_a, True)} params each)")

    log_a = train_one(model_a, Xtr, ttr, "A_recon_task", seed=seed,
                      verbose=verbose)
    log_b = train_one(model_b, Xtr, ttr, "B_task_only", seed=seed,
                      verbose=verbose)

    pred_a = rollout_predictions(model_a, Xte)
    pred_b = rollout_predictions(model_b, Xte)
    series = {"A_recon_task": pred_a, "B_task_only": pred_b,
              "baseline_constant": baseline_constant(t_np),
              "baseline_linear": baseline_linear(t_np)}
    bounce = eval_d["bounce_in_horizon"]
    subsets = (("all", None), ("no_bounce", ~bounce), ("bounce", bounce))

    rows = []
    for name, pred in series.items():
        for subset, mask in subsets:
            for r in score(pred, t_np, mask=mask):
                rows.append({"seed": seed, "model": name, "subset": subset, **r})

    diffs = []
    for subset, mask in subsets:
        for r in paired_diff(pred_a, pred_b, t_np, mask=mask):
            diffs.append({"seed": seed, "subset": subset, **r})

    extras = {"log": [{**r, "seed": seed} for r in log_a + log_b],
              "diffs": diffs, "eval_d": eval_d,
              "pred_a": pred_a, "pred_b": pred_b, "series": series,
              "models": (model_a, model_b)}
    return rows, extras


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 2))

    print(f"Rollout experiment: context={CONTEXT} horizon={HORIZON} "
          f"anchors={ANCHORS}")
    print(f"seeds={SEEDS}  train={N_TRAIN_TRAJ} eval={N_EVAL_TRAJ} "
          f"epochs={EPOCHS} batch={BATCH}\n")

    all_rows, all_diffs, all_logs = [], [], []
    first_extras = None
    for s in SEEDS:
        print(f"--- seed {s} ---")
        rows, extras = run_one_seed(s, verbose=(s == SEEDS[0]))
        all_rows += rows
        all_diffs += extras["diffs"]
        all_logs += extras["log"]
        if first_extras is None:
            first_extras = extras
        a10 = [r for r in rows if r["model"] == "A_recon_task"
               and r["subset"] == "all" and r["horizon"] == 10][0]
        b10 = [r for r in rows if r["model"] == "B_task_only"
               and r["subset"] == "all" and r["horizon"] == 10][0]
        print(f"  h=10 px error:  A={a10['mae_px']:.3f}  B={b10['mae_px']:.3f}"
              f"   (B-A = {b10['mae_px'] - a10['mae_px']:+.3f})")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "rollout_error.csv"), index=False)
    pd.DataFrame(all_diffs).to_csv(
        os.path.join(RESULTS_DIR, "rollout_paired_diff.csv"), index=False)
    pd.DataFrame(all_logs).to_csv(
        os.path.join(RESULTS_DIR, "rollout_training.csv"), index=False)

    # Per-seed artefacts for the visualisation come from the FIRST seed only.
    ev = first_extras["eval_d"]
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "rollout_predictions.npz"),
        t_true=ev["t"], images=ev["images"],
        bounce_in_horizon=ev["bounce_in_horizon"],
        pred_A=first_extras["pred_a"], pred_B=first_extras["pred_b"],
        pred_constant=first_extras["series"]["baseline_constant"],
        pred_linear=first_extras["series"]["baseline_linear"],
        context=CONTEXT, horizon=HORIZON, seed=SEEDS[0],
    )
    ma, mb = first_extras["models"]
    torch.save({"A": ma.state_dict(), "B": mb.state_dict()},
               os.path.join(RESULTS_DIR, "rollout_weights.pt"))

    # --- console report: averaged over seeds -------------------------------
    agg = (df[df.subset == "all"].groupby(["model", "horizon"])["mae_px"]
           .agg(["mean", "std"]).reset_index())
    piv = agg.pivot(index="horizon", columns="model", values="mean")[
        ["A_recon_task", "B_task_only", "baseline_constant", "baseline_linear"]]
    print(f"\nBlind-rollout position error, mean over {len(SEEDS)} seeds "
          f"(px on a 16px image)")
    print(piv.round(3).to_string())

    dd = pd.DataFrame(all_diffs)
    print("\nPaired B-minus-A gap (px; >0 = B worse), by subset:")
    for subset in ["all", "no_bounce", "bounce"]:
        s = dd[dd.subset == subset].groupby("horizon")["delta_px"].agg(
            ["mean", "std"])
        line = "  ".join(f"h{h}={s.loc[h, 'mean']:+.3f}" for h in [1, 3, 5, 10])
        print(f"  {subset:10s} {line}")
    print(f"\ndone in {time.time() - t0:.1f}s -> results/rollout_*.csv/.npz")


if __name__ == "__main__":
    main()
