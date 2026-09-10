"""
analyze.py — Mechanistic XAI on the FINAL trained encoders.

Two analyses, both run on the exact weights train.py saved (no re-training):

  (5) Latent ablation = CAUSAL importance of each latent dimension.
        - Native objective (spec item 5, per-encoder):
            Encoder A: zero dim d, re-run the DECODER, measure the rise in
                       reconstruction MSE.
            Encoder B: zero dim d, re-run the TASK HEAD, measure the drop in
                       task R^2.
          These use each model's OWN downstream module, so they answer
          "which dims does THIS model causally rely on for ITS objective?"
        - Common task probe (supplementary, comparable across encoders):
            Fit one linear task probe on each latent, then zero each dim and
            measure the drop in task R^2. Same metric for A and B, so the
            side-by-side bar chart is apples-to-apples.

  (6) PCA of the final latent space, projected to 2-D, with two colourings
      (task feature t vs background brightness). We report the ACTUAL
      separation numerically for each encoder and do not assume a pattern.

Outputs to results/:
    ablation_native.csv     — per-dim native-objective importance, per encoder.
    ablation_task_probe.csv — per-dim task-R^2 drop via a common linear probe.
    pca_coords.csv          — 2-D PCA coords + colourings, per encoder.
    pca_summary.csv         — explained variance + PC/task/distractor structure.

Run:  python analyze.py   (after train.py)
"""

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from data import make_dataset
from model import Autoencoder, TaskNet, LATENT_DIM, load_state_dict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SEED = 0  # must match train.py so we rebuild the SAME eval set / models


def load_everything():
    """Rebuild both trained models and reload the eval set used for snapshots."""
    ae = Autoencoder(np.random.default_rng(SEED + 10))
    task = TaskNet(np.random.default_rng(SEED + 10))
    load_state_dict(ae, np.load(os.path.join(RESULTS_DIR, "weights_A.npz")))
    load_state_dict(task, np.load(os.path.join(RESULTS_DIR, "weights_B.npz")))

    snaps = np.load(os.path.join(RESULTS_DIR, "latent_snapshots.npz"))
    factors = {"t": snaps["t"], "bg": snaps["bg"], "brightness": snaps["brightness"]}
    # Rebuild the eval images (deterministic: same seed as train.py's eval set).
    ev = make_dataset(len(factors["t"]), seed=SEED + 1)
    return ae, task, ev["X"], factors


# ---------------------------------------------------------------------------
# (5) Native-objective ablation
# ---------------------------------------------------------------------------
def ablation_native(ae, task, X, factors):
    """Zero each latent dim and re-run each model's OWN downstream module."""
    t = factors["t"]
    rows = []

    # --- Encoder B: task head, metric = task R^2 -------------------------
    ZB = task.encode(X)
    base_pred = task.head.forward(ZB).reshape(-1)
    base_r2_B = r2_score(t, base_pred)
    for d in range(LATENT_DIM):
        Zabl = ZB.copy()
        Zabl[:, d] = 0.0
        pred = task.head.forward(Zabl).reshape(-1)
        r2 = r2_score(t, pred)
        rows.append({
            "encoder": "B_task", "dim": d,
            "metric": "task_r2",
            "baseline": base_r2_B, "ablated": r2,
            "importance": base_r2_B - r2,      # drop in task R^2
        })

    # --- Encoder A: decoder, metric = reconstruction MSE -----------------
    ZA = ae.encode(X)
    base_recon = ae.decoder.forward(ZA)
    base_mse_A = float(np.mean((base_recon - X) ** 2))
    for d in range(LATENT_DIM):
        Zabl = ZA.copy()
        Zabl[:, d] = 0.0
        recon = ae.decoder.forward(Zabl)
        mse = float(np.mean((recon - X) ** 2))
        rows.append({
            "encoder": "A_recon", "dim": d,
            "metric": "recon_mse",
            "baseline": base_mse_A, "ablated": mse,
            "importance": mse - base_mse_A,    # rise in reconstruction MSE
        })
    return pd.DataFrame(rows), base_r2_B, base_mse_A


# ---------------------------------------------------------------------------
# (5b) Common task-probe ablation (comparable across encoders)
# ---------------------------------------------------------------------------
def ablation_task_probe(ae, task, X, factors, seed=0):
    """Fit ONE linear task probe per encoder, then ablate each dim and record
    how much the probe's TASK-PREDICTION MSE rises. Same metric for both, so the
    side-by-side bar chart is directly comparable.

    Why MSE-increase rather than R^2-drop: the latent dims are correlated, so a
    linear probe spreads compensating weights across them. Zeroing one dim can
    send R^2 wildly negative (an artifact of probe-weight scale, not of true
    importance). MSE-increase is bounded and interpretable, and we also report
    `importance_frac` = each dim's share of the total, which makes
    "concentrated (B) vs spread (A)" directly visible.
    """
    t = factors["t"]
    rows = []
    for name, model in (("A_recon", ae), ("B_task", task)):
        Z = model.encode(X)
        Ztr, Zte, ttr, tte = train_test_split(Z, t, test_size=0.5, random_state=seed)
        probe = LinearRegression().fit(Ztr, ttr)
        base_r2 = r2_score(tte, probe.predict(Zte))
        base_mse = float(np.mean((probe.predict(Zte) - tte) ** 2))
        imps = []
        for d in range(LATENT_DIM):
            Zabl = Zte.copy()
            Zabl[:, d] = 0.0
            mse = float(np.mean((probe.predict(Zabl) - tte) ** 2))
            imps.append(mse - base_mse)  # rise in task-prediction MSE
        total = sum(imps) if sum(imps) > 0 else 1.0
        for d in range(LATENT_DIM):
            rows.append({
                "encoder": name, "dim": d,
                "baseline_r2": base_r2, "baseline_mse": base_mse,
                "importance": imps[d],                # MSE increase
                "importance_frac": imps[d] / total,   # share of total (sums to 1)
            })
    return pd.DataFrame(rows)


def robustness_shift(ae, task, factors, seed=0,
                     noise_grid=(0.05, 0.10, 0.20, 0.30, 0.40)):
    """Distractor-shift robustness — the most direct test of the Hermes claim.

    Procedure: freeze both encoders. Fit ONE linear task probe per encoder on
    the in-distribution eval latents (noise_std = 0.05, the training value).
    Then regenerate the eval images with progressively MORE per-pixel background
    noise (an out-of-distribution nuisance neither encoder was trained on) and
    measure each frozen probe's task R^2. The encoder whose task read-out decays
    less is the more distractor-robust representation.

    We keep the object positions `t` fixed across the sweep (same seed), so the
    ONLY thing changing is the strength of the background distractor.
    """
    t = factors["t"]
    n = len(t)
    rows = []
    # Fit the task probes once, in-distribution.
    probes = {}
    for name, model in (("A_recon", ae), ("B_task", task)):
        Z0 = model.encode(make_dataset(n, seed=SEED + 1, noise_std=0.05)["X"])
        Ztr, _, ttr, _ = train_test_split(Z0, t, test_size=0.5, random_state=seed)
        probes[name] = LinearRegression().fit(Ztr, ttr)

    for noise in noise_grid:
        # Same seed => same t and same background colours; only noise differs.
        Xn = make_dataset(n, seed=SEED + 1, noise_std=noise)["X"]
        for name, model in (("A_recon", ae), ("B_task", task)):
            Zn = model.encode(Xn)
            r2 = r2_score(t, probes[name].predict(Zn))
            rows.append({"encoder": name, "noise_std": noise,
                         "task_r2": float(r2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (6) PCA projection + honest separation report
# ---------------------------------------------------------------------------
def pca_analysis(ae, task, X, factors):
    t = factors["t"]
    brightness = factors["brightness"]
    coord_rows = []
    summary_rows = []
    for name, model in (("A_recon", ae), ("B_task", task)):
        Z = model.encode(X)
        # Keep 3 components so the interactive viz can show a rotatable 3-D
        # scatter; the R^2 / correlation report below still uses the top 2.
        pca = PCA(n_components=3).fit(Z)
        P = pca.transform(Z)  # (n, 3)

        for i in range(len(t)):
            coord_rows.append({
                "encoder": name, "pc1": P[i, 0], "pc2": P[i, 1], "pc3": P[i, 2],
                "task": t[i], "brightness": brightness[i],
            })

        # How much of the projection is explained by task vs distractor?
        # R^2 of predicting each factor from the top-2 PCs (linear).
        P2 = P[:, :2]

        def r2_from_pcs(y):
            return float(max(r2_score(y, LinearRegression().fit(P2, y).predict(P2)), 0.0))

        summary_rows.append({
            "encoder": name,
            "explained_var_pc1": float(pca.explained_variance_ratio_[0]),
            "explained_var_pc2": float(pca.explained_variance_ratio_[1]),
            "explained_var_pc3": float(pca.explained_variance_ratio_[2]),
            "task_r2_from_2pcs": r2_from_pcs(t),
            "distractor_r2_from_2pcs": r2_from_pcs(brightness),
            # Correlation of each PC with each factor (sign not meaningful).
            "pc1_task_corr": float(np.corrcoef(P[:, 0], t)[0, 1]),
            "pc1_bright_corr": float(np.corrcoef(P[:, 0], brightness)[0, 1]),
            "pc2_task_corr": float(np.corrcoef(P[:, 1], t)[0, 1]),
            "pc2_bright_corr": float(np.corrcoef(P[:, 1], brightness)[0, 1]),
        })
    return pd.DataFrame(coord_rows), pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# (7) Narrative outputs — the head-to-head story, in directly readable form
# ---------------------------------------------------------------------------
def task_readout(ae, task, X, factors, seed=0):
    """Fair head-to-head: read the object's position out of each latent.

    Both encoders get the SAME treatment — a linear probe fit on half the eval
    set and scored on the other half — so the comparison is apples-to-apples
    (Encoder A has no task head of its own, so a probe is the only fair option).
    We return the held-out true/predicted pairs, which plot as the classic
    "predicted vs actual" scatter: tight on the diagonal = accurate readout.
    """
    t = factors["t"]
    rows = []
    for name, model in (("A_recon", ae), ("B_task", task)):
        Z = model.encode(X)
        Ztr, Zte, ttr, tte = train_test_split(Z, t, test_size=0.5, random_state=seed)
        probe = LinearRegression().fit(Ztr, ttr)
        pred = probe.predict(Zte)
        for a, b in zip(tte, pred):
            rows.append({"encoder": name, "t_true": float(a), "t_pred": float(b)})
    return pd.DataFrame(rows)


def task_aligned_projection(ae, task, X, factors, seed=0):
    """Project each latent onto INTERPRETABLE axes instead of variance axes.

    PCA answers "where is the most variance?" — and in this dataset that is
    always the background, so PCA hides the story. Here we instead build axes
    that mean something:

        axis 1 ("task axis")       = the latent direction a linear probe uses
                                     to read out the object's position.
        axis 2 ("distractor axis") = the direction used to read out background
                                     brightness, with the task direction removed
                                     (Gram-Schmidt), so the two axes are
                                     independent and the plot is honest.
        axis 3                     = the leading direction of what is left over.

    Colour the cloud by position and a good task representation becomes a clean
    gradient along axis 1. Colour by background and you see how much distractor
    structure each latent still carries on axis 2.
    """
    t, brightness = factors["t"], factors["brightness"]
    rows = []
    for name, model in (("A_recon", ae), ("B_task", task)):
        Z = model.encode(X)
        Zc = Z - Z.mean(axis=0)  # centre, so the axes pass through the cloud

        # Direction that linearly predicts the task feature.
        w_task = LinearRegression().fit(Zc, t).coef_.astype(np.float64)
        w_task /= np.linalg.norm(w_task) + 1e-12

        # Direction that predicts the distractor, orthogonalised against w_task
        # so axis 2 carries only distractor structure the task axis does not.
        w_dist = LinearRegression().fit(Zc, brightness).coef_.astype(np.float64)
        w_dist -= (w_dist @ w_task) * w_task
        w_dist /= np.linalg.norm(w_dist) + 1e-12

        # Axis 3: the biggest remaining direction after removing axes 1 and 2.
        resid = Zc - np.outer(Zc @ w_task, w_task) - np.outer(Zc @ w_dist, w_dist)
        w_res = PCA(n_components=1).fit(resid).components_[0]

        a1, a2, a3 = Zc @ w_task, Zc @ w_dist, Zc @ w_res
        # Scale each axis to unit std so A and B are visually comparable
        # (we care about the SHAPE of the cloud, not each encoder's raw scale).
        a1, a2, a3 = [a / (a.std() + 1e-12) for a in (a1, a2, a3)]
        for i in range(len(t)):
            rows.append({
                "encoder": name,
                "task_axis": float(a1[i]),
                "distractor_axis": float(a2[i]),
                "residual_axis": float(a3[i]),
                "task": float(t[i]), "brightness": float(brightness[i]),
            })
    return pd.DataFrame(rows)


def recon_examples(ae, task, n=6, seed=0):
    """A few eval images, Encoder A's reconstruction of them, and Encoder B's
    predicted object position for the same images.

    Makes the capacity argument concrete. A's decoder faithfully reproduces the
    background colour — the part of the image carrying no task information.
    Encoder B has NO decoder and cannot draw anything at all; the only thing it
    outputs is a position. We save that scalar so the demo can render "what B
    knows" (a dot at its predicted location) next to A's full repaint.
    """
    # Draw a pool, then keep n examples SPREAD ACROSS the position range so the
    # strip visibly shows the object (and B's predicted dot) moving left to
    # right. The selection is on position only — never on how well either model
    # happened to do — so it illustrates without cherry-picking accuracy.
    pool = make_dataset(200, seed=SEED + 77)
    order = np.argsort(pool["t"])
    pick = order[(np.linspace(0.06, 0.94, n) * (len(order) - 1)).astype(int)]

    X = pool["X"][pick]
    recon = ae.decoder.forward(ae.encode(X))
    b_pred = task.predict(X)  # Encoder B's own task head output
    return (pool["images"][pick], recon.reshape(pool["images"][pick].shape),
            pool["t"][pick], b_pred)


def readout_over_time(n_show=300, n_track=8, seed=0):
    """Per-epoch position read-outs, for the two animated panels in the demo.

    train.py already saved the eval-set latents at every logged epoch, so we can
    recover "what could be read out of this latent at epoch N" without touching
    the training loop again. At each snapshot we fit a linear probe on a fixed
    train half and predict the held-out half — the same fair protocol used for
    the static head-to-head, applied at every epoch.

    Both encoders are read with a PROBE (not their own head), because Encoder A
    has no task head at all; using the same instrument on both is what makes the
    animation an honest comparison rather than a rigged one.

    Saves:
      t_show / predA_show / predB_show : a fixed subset of held-out points,
          for the "predicted vs actual" scatter that tightens over training.
      track_* : a handful of individual held-out images spanning the position
          range, plus each encoder's predicted position for them at every epoch,
          for the object-tracker view.
    """
    snaps = np.load(os.path.join(RESULTS_DIR, "latent_snapshots.npz"))
    epochs, ZA, ZB, t = snaps["epochs"], snaps["ZA"], snaps["ZB"], snaps["t"]
    n = len(t)

    # One fixed split reused at every epoch, so frames are directly comparable.
    idx = np.arange(n)
    idx_tr, idx_te = train_test_split(idx, test_size=0.5, random_state=seed)

    rng = np.random.default_rng(seed)
    show = rng.choice(idx_te, size=min(n_show, len(idx_te)), replace=False)

    # Tracker examples: held-out images spread across the position range.
    te_sorted = idx_te[np.argsort(t[idx_te])]
    track = te_sorted[(np.linspace(0.07, 0.93, n_track) * (len(te_sorted) - 1)).astype(int)]

    predA_show, predB_show, predA_tr, predB_tr = [], [], [], []
    for i in range(len(epochs)):
        row = {}
        for key, Z in (("A", ZA), ("B", ZB)):
            probe = LinearRegression().fit(Z[i][idx_tr], t[idx_tr])
            row[key] = probe.predict(Z[i])          # predict everything once
        predA_show.append(row["A"][show]);  predB_show.append(row["B"][show])
        predA_tr.append(row["A"][track]);   predB_tr.append(row["B"][track])

    ev = make_dataset(n, seed=SEED + 1)  # same eval set train.py snapshotted
    return {
        "epochs": epochs,
        "t_show": t[show],
        "predA_show": np.stack(predA_show),      # (n_snap, n_show)
        "predB_show": np.stack(predB_show),
        "track_t": t[track],
        "track_images": ev["images"][track],
        "predA_track": np.stack(predA_tr),       # (n_snap, n_track)
        "predB_track": np.stack(predB_tr),
    }


def main():
    ae, task, X, factors = load_everything()

    # (5) native ablation
    df_native, base_r2_B, base_mse_A = ablation_native(ae, task, X, factors)
    df_native.to_csv(os.path.join(RESULTS_DIR, "ablation_native.csv"), index=False)

    # (5b) common task-probe ablation
    df_probe = ablation_task_probe(ae, task, X, factors)
    df_probe.to_csv(os.path.join(RESULTS_DIR, "ablation_task_probe.csv"), index=False)

    # (bonus) distractor-shift robustness
    df_robust = robustness_shift(ae, task, factors)
    df_robust.to_csv(os.path.join(RESULTS_DIR, "robustness_shift.csv"), index=False)

    # (6) PCA
    df_coords, df_pca = pca_analysis(ae, task, X, factors)
    df_coords.to_csv(os.path.join(RESULTS_DIR, "pca_coords.csv"), index=False)
    df_pca.to_csv(os.path.join(RESULTS_DIR, "pca_summary.csv"), index=False)

    # (7) narrative outputs: fair head-to-head readout + interpretable axes
    df_readout = task_readout(ae, task, X, factors)
    df_readout.to_csv(os.path.join(RESULTS_DIR, "task_readout.csv"), index=False)
    df_aligned = task_aligned_projection(ae, task, X, factors)
    df_aligned.to_csv(os.path.join(RESULTS_DIR, "task_aligned.csv"), index=False)
    imgs, recons, ts, b_pred = recon_examples(ae, task)
    np.savez_compressed(os.path.join(RESULTS_DIR, "recon_examples.npz"),
                        images=imgs, recons=recons, t=ts, b_pred=b_pred)

    # (8) per-epoch read-outs powering the two animated panels
    rot = readout_over_time()
    np.savez_compressed(os.path.join(RESULTS_DIR, "readout_over_time.npz"), **rot)

    # ---- console report --------------------------------------------------
    np.set_printoptions(precision=3, suppress=True)
    print(f"Encoder B baseline task R^2 (task head)   : {base_r2_B:.4f}")
    print(f"Encoder A baseline recon MSE (decoder)    : {base_mse_A:.5f}\n")

    print("NATIVE-OBJECTIVE ablation (item 5) — importance per dim:")
    for enc in ("B_task", "A_recon"):
        sub = df_native[df_native.encoder == enc].sort_values("importance", ascending=False)
        imp = sub["importance"].values
        metric = sub["metric"].iloc[0]
        print(f"  {enc:8s} ({metric}): "
              f"top dims {sub['dim'].values[:3].tolist()}  "
              f"importances {np.round(imp, 4).tolist()}")

    print("\nCOMMON task-probe ablation (comparable) — task-MSE-increase share per dim:")
    for enc in ("A_recon", "B_task"):
        sub = df_probe[df_probe.encoder == enc].sort_values("importance_frac", ascending=False)
        top = sub.iloc[0]
        # Concentration: how much of total importance the single top dim holds.
        print(f"  {enc:8s}: baseline R^2={top['baseline_r2']:.3f} | "
              f"top dim={int(top['dim'])} holds {top['importance_frac']*100:.1f}% "
              f"of task importance | frac-vector="
              f"{np.round(sub.sort_values('dim')['importance_frac'].values, 3).tolist()}")

    print("\nPCA summary (item 6) — 2-D projection structure:")
    print(df_pca.round(4).to_string(index=False))

    print("\nDISTRACTOR-SHIFT robustness (bonus) — task R^2 vs background noise:")
    piv = df_robust.pivot(index="noise_std", columns="encoder", values="task_r2")
    print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
