"""
analyze_rollout.py — Aggregate the multi-seed rollout sweep into the numbers
that actually answer the question.

train_rollout.py runs the experiment once per seed and dumps raw per-seed rows.
This script turns those into the summary tables used by the writeup and the
demo page. It computes nothing new about the models — it only aggregates.

Three things get computed, because the headline question ("does B degrade
faster?") is genuinely ambiguous and the answers can differ:

  1. ABSOLUTE error at each horizon, averaged over seeds. Answers "which model
     is more accurate at horizon h?"

  2. PAIRED gap (B minus A) at each horizon. A and B see identical held-out
     trajectories, so the errors are paired; testing the difference directly is
     much more sensitive than asking whether two separate confidence intervals
     overlap. Answers "is the difference real?"

  3. DEGRADATION RATIO, err(h=10) / err(h=1), per model. This is the one that
     literally answers "degrades faster", and it is NOT the same question as
     (1): a model can be uniformly worse at every horizon while degrading at
     exactly the same RATE. Reporting only absolute error would conflate a
     constant offset with a difference in temporal stability.

Run:  python analyze_rollout.py    (after train_rollout.py)
"""

import os

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

MODELS = ["A_recon_task", "B_task_only", "baseline_constant", "baseline_linear"]
NICE = {"A_recon_task": "A · recon + task", "B_task_only": "B · task-only",
        "baseline_constant": "baseline: hold last", "baseline_linear": "baseline: linear"}


def load():
    err = pd.read_csv(os.path.join(RESULTS_DIR, "rollout_error.csv"))
    diff = pd.read_csv(os.path.join(RESULTS_DIR, "rollout_paired_diff.csv"))
    return err, diff


def summarise_absolute(err):
    """Mean +/- std across seeds of the per-horizon pixel error."""
    g = (err.groupby(["model", "subset", "horizon"])
            .agg(mae_px_mean=("mae_px", "mean"),
                 mae_px_std=("mae_px", "std"),
                 r2_mean=("r2", "mean"),
                 r2_std=("r2", "std"),
                 n_seeds=("mae_px", "size"),
                 n_traj=("n", "first"))
            .reset_index())
    return g


def summarise_paired(diff):
    """Mean paired gap across seeds, plus how consistently it points one way.

    `seeds_favouring_A` counts how many individual seeds had B worse than A.
    With only a handful of seeds this is a blunt but honest instrument: if the
    sign flips across seeds, the effect is not robust and we say so.
    """
    g = (diff.groupby(["subset", "horizon"])
             .agg(delta_px_mean=("delta_px", "mean"),
                  delta_px_std=("delta_px", "std"),
                  n_seeds=("delta_px", "size"),
                  seeds_favouring_A=("delta_px", lambda s: int((s > 0).sum())),
                  mean_boot_frac=("frac_boot_favours_A", "mean"))
             .reset_index())
    return g


def degradation_ratio(err, h_lo=1, h_hi=10):
    """err(h_hi) / err(h_lo) per model per seed — the "degrades faster" metric.

    Computed per seed and then averaged, so the spread across seeds is visible.
    A ratio of 3 means error triples between horizon 1 and horizon 10.
    """
    rows = []
    for (model, subset, seed), g in err.groupby(["model", "subset", "seed"]):
        g = g.set_index("horizon")
        if h_lo not in g.index or h_hi not in g.index:
            continue
        lo, hi = g.loc[h_lo, "mae_px"], g.loc[h_hi, "mae_px"]
        rows.append({"model": model, "subset": subset, "seed": seed,
                     "err_h1": lo, "err_h10": hi, "ratio": hi / lo})
    d = pd.DataFrame(rows)
    return (d.groupby(["model", "subset"])
             .agg(err_h1=("err_h1", "mean"), err_h10=("err_h10", "mean"),
                  ratio_mean=("ratio", "mean"), ratio_std=("ratio", "std"),
                  n_seeds=("ratio", "size"))
             .reset_index())


def _fmt_table(df, index, columns, values, fmt="{:.3f}"):
    p = df.pivot(index=index, columns=columns, values=values)
    cols = [c for c in MODELS if c in p.columns]
    return p[cols].map(lambda v: fmt.format(v) if pd.notna(v) else "—")


def main():
    err, diff = load()
    abs_g = summarise_absolute(err)
    pair_g = summarise_paired(diff)
    ratio_g = degradation_ratio(err)

    abs_g.to_csv(os.path.join(RESULTS_DIR, "rollout_summary.csv"), index=False)
    pair_g.to_csv(os.path.join(RESULTS_DIR, "rollout_paired_summary.csv"),
                  index=False)
    ratio_g.to_csv(os.path.join(RESULTS_DIR, "rollout_degradation.csv"),
                   index=False)

    n_seeds = int(abs_g.n_seeds.max())
    print(f"Rollout summary over {n_seeds} seeds "
          f"({int(abs_g.n_traj.max())} held-out trajectories each)\n")

    print("=== 1. Blind-rollout position error, px on a 16px image (all traj) ===")
    a = abs_g[abs_g.subset == "all"]
    print(_fmt_table(a, "horizon", "model", "mae_px_mean").to_string())

    print("\n=== 2. Paired gap, B minus A (px; >0 means B is worse) ===")
    print("subset      " + "".join(f"{'h'+str(h):>12}" for h in [1, 3, 5, 10]))
    for subset in ["all", "no_bounce", "bounce"]:
        s = pair_g[pair_g.subset == subset].set_index("horizon")
        cells = []
        for h in [1, 3, 5, 10]:
            if h in s.index:
                r = s.loc[h]
                star = "*" if r.seeds_favouring_A == r.n_seeds else " "
                cells.append(f"{r.delta_px_mean:+.3f}{star}".rjust(12))
        print(f"{subset:12s}" + "".join(cells))
    print("  * = every seed agreed on the sign")

    print("\n=== 3. Degradation ratio  err(h=10) / err(h=1) ===")
    print("  (this is the literal 'degrades faster' metric)")
    r = ratio_g[ratio_g.subset == "all"].set_index("model")
    for m in MODELS:
        if m in r.index:
            row = r.loc[m]
            std = "" if pd.isna(row.ratio_std) else f" ± {row.ratio_std:.2f}"
            print(f"  {NICE[m]:24s} {row.ratio_mean:5.2f}{std}"
                  f"   ({row.err_h1:.3f} -> {row.err_h10:.3f} px)")

    print("\n=== 4. Split by whether the dot bounces inside the window ===")
    for subset in ["no_bounce", "bounce"]:
        s = abs_g[(abs_g.subset == subset) & (abs_g.horizon.isin([1, 10]))]
        piv = s.pivot(index="horizon", columns="model", values="mae_px_mean")
        cols = [c for c in ["A_recon_task", "B_task_only"] if c in piv.columns]
        n = int(abs_g[abs_g.subset == subset].n_traj.max())
        print(f"  {subset} (n={n}):")
        print("    " + piv[cols].round(3).to_string().replace("\n", "\n    "))

    print("\nwrote rollout_summary.csv, rollout_paired_summary.csv, "
          "rollout_degradation.csv")


if __name__ == "__main__":
    main()
