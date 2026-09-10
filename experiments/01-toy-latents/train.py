"""
train.py — Train the two encoders and log everything analyze.py / the viz need.

What it does, end to end:
  1. Build a fixed train set and a fixed eval set from data.py.
  2. Train Encoder A (Autoencoder, reconstruction loss) and Encoder B
     (TaskNet, task-prediction loss) for the SAME number of epochs with the
     SAME optimiser settings.
  3. Every SNAPSHOT_EVERY epochs, freeze-frame the eval-set latents of BOTH
     encoders and linear-probe them for how much variance they carry about
     (a) the task feature t and (b) the background distractor.
  4. Save to results/:
        latent_snapshots.npz  — epoch grid, eval factors, and the latent
                                 activations of both encoders at every snapshot.
        probing_metrics.csv   — task vs distractor R^2 per encoder per snapshot.
        weights_A.npz / weights_B.npz — final trained weights, for analyze.py.

Run:  python train.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from data import make_dataset
from model import Autoencoder, TaskNet, Adam, state_dict

# ----------------------------------------------------------------------------
# Experiment configuration. Small on purpose: this whole script runs in seconds.
# ----------------------------------------------------------------------------
SEED = 0
N_TRAIN = 4000
N_EVAL = 1000
EPOCHS = 60
BATCH = 128
LR = 2e-3
SNAPSHOT_EVERY = 5          # log latents/probes every N epochs (plus epoch 0)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def probe_r2(Z, y, seed=0):
    """How much of `y` a LINEAR read-out of the latent Z can explain.

    We split the snapshot in half, fit an ordinary least-squares probe on one
    half and report R^2 on the other. Using a held-out half keeps the number
    honest (an 8-dim probe could otherwise slightly overfit a scalar target).

    Z : (n, latent_dim) latent activations.
    y : (n,) or (n, k) target(s). For k>1 we average R^2 across columns.
    Returns a single float R^2 (clipped at 0 from below for readability).
    """
    y = np.asarray(y)
    Ztr, Zte, ytr, yte = train_test_split(Z, y, test_size=0.5, random_state=seed)
    reg = LinearRegression().fit(Ztr, ytr)
    pred = reg.predict(Zte)
    # multioutput='uniform_average' -> mean R^2 across target columns when k>1.
    r2 = r2_score(yte, pred, multioutput="uniform_average")
    return float(max(r2, 0.0))


def snapshot_metrics(Z, factors):
    """Compute the task vs distractor probe R^2 for one latent snapshot."""
    return {
        "task_r2": probe_r2(Z, factors["t"]),           # x-position
        "distractor_r2": probe_r2(Z, factors["bg"]),    # full RGB background
        "brightness_r2": probe_r2(Z, factors["brightness"]),  # bg luminance
    }


def iterate_minibatches(n, batch, rng):
    """Yield shuffled index batches for one epoch."""
    idx = rng.permutation(n)
    for start in range(0, n, batch):
        yield idx[start:start + batch]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- data -------------------------------------------------------------
    train = make_dataset(N_TRAIN, seed=SEED)
    ev = make_dataset(N_EVAL, seed=SEED + 1)   # disjoint eval set
    Xtr, ttr = train["X"], train["t"]
    Xev = ev["X"]
    eval_factors = {"t": ev["t"], "bg": ev["bg"], "brightness": ev["brightness"]}

    # --- models (identical encoder architecture, separate weights) --------
    # Separate RNGs so both encoders get the SAME initialisation distribution
    # but their own draw — a fair, controlled comparison.
    ae = Autoencoder(np.random.default_rng(SEED + 10))
    task = TaskNet(np.random.default_rng(SEED + 10))
    opt_ae = Adam(ae.params(), lr=LR)
    opt_task = Adam(task.params(), lr=LR)

    # --- logging buffers --------------------------------------------------
    snap_epochs = []
    ZA_snaps, ZB_snaps = [], []   # eval latents per snapshot
    metric_rows = []

    def take_snapshot(epoch):
        ZA = ae.encode(Xev)
        ZB = task.encode(Xev)
        snap_epochs.append(epoch)
        ZA_snaps.append(ZA.copy())
        ZB_snaps.append(ZB.copy())
        for name, Z in (("A_recon", ZA), ("B_task", ZB)):
            m = snapshot_metrics(Z, eval_factors)
            m.update(epoch=epoch, encoder=name)
            metric_rows.append(m)

    # Snapshot at initialisation (epoch 0) so we can see structure EMERGE.
    take_snapshot(0)

    # --- training loop ----------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        ae_loss = task_loss = 0.0
        n_batches = 0
        for bidx in iterate_minibatches(N_TRAIN, BATCH, rng):
            xb = Xtr[bidx]
            tb = ttr[bidx]
            # Encoder A step: reconstruction loss only.
            ae_loss += ae.loss_and_backward(xb)
            opt_ae.step()
            # Encoder B step: task-prediction loss only (same batch).
            task_loss += task.loss_and_backward(xb, tb)
            opt_task.step()
            n_batches += 1

        if epoch % SNAPSHOT_EVERY == 0 or epoch == EPOCHS:
            take_snapshot(epoch)
            last = {r["encoder"]: r for r in metric_rows if r["epoch"] == epoch}
            print(f"epoch {epoch:3d} | "
                  f"AE mse={ae_loss / n_batches:.4f} "
                  f"task mse={task_loss / n_batches:.4f} | "
                  f"task_r2  A={last['A_recon']['task_r2']:.3f} "
                  f"B={last['B_task']['task_r2']:.3f} | "
                  f"distr_r2 A={last['A_recon']['distractor_r2']:.3f} "
                  f"B={last['B_task']['distractor_r2']:.3f}")

    # --- persist ----------------------------------------------------------
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "latent_snapshots.npz"),
        epochs=np.array(snap_epochs),
        ZA=np.stack(ZA_snaps),   # (n_snap, N_EVAL, latent)
        ZB=np.stack(ZB_snaps),
        t=eval_factors["t"],
        bg=eval_factors["bg"],
        brightness=eval_factors["brightness"],
    )
    np.savez_compressed(os.path.join(RESULTS_DIR, "weights_A.npz"), **state_dict(ae))
    np.savez_compressed(os.path.join(RESULTS_DIR, "weights_B.npz"), **state_dict(task))

    df = pd.DataFrame(metric_rows)[
        ["epoch", "encoder", "task_r2", "distractor_r2", "brightness_r2"]
    ].sort_values(["encoder", "epoch"]).reset_index(drop=True)
    df.to_csv(os.path.join(RESULTS_DIR, "probing_metrics.csv"), index=False)

    print("\nSaved to results/: latent_snapshots.npz, probing_metrics.csv, "
          "weights_A.npz, weights_B.npz")
    print("\nFinal-epoch probe R^2:")
    print(df[df.epoch == EPOCHS].to_string(index=False))


if __name__ == "__main__":
    main()
