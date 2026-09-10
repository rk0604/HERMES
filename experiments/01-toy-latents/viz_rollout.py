"""
viz_rollout.py — Plotly figures for the multi-step rollout experiment.

Kept in its own module so that adding this experiment to the demo page is a
purely additive change: interactive_viz.py and export_demo.py each import from
here and gain a section, and nothing about the existing static figures moves.

Everything is graceful if the rollout has not been run yet — `available()`
returns False and the page simply omits the section, so the static demo still
works standalone.

Figures
-------
rollout_curve_figure   : position error vs prediction horizon, A vs B, with the
                         two reference baselines and a seed-spread band.
rollout_gap_figure     : the PAIRED B-minus-A gap vs horizon, split by whether
                         the dot bounces off a wall inside the window. This is
                         where the mechanism shows up.
rollout_examples_figure: individual held-out trajectories — true position over
                         time against each model's blind prediction. Positions
                         are drawn as points on a time axis; there is no
                         reconstructed image anywhere, because Encoder B has no
                         decoder and inventing one for a picture would misstate
                         what the model does.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS = os.path.join(os.path.dirname(__file__), "results")

COL_A, COL_B = "#d1495b", "#2e86ab"          # same palette as the static demo
COL_CONST, COL_LIN = "#bbbbbb", "#8a8a8a"
LABEL = {"A_recon_task": "A · recon + task", "B_task_only": "B · task-only",
         "baseline_constant": "baseline: hold last position",
         "baseline_linear": "baseline: linear extrapolation"}
COLOR = {"A_recon_task": COL_A, "B_task_only": COL_B,
         "baseline_constant": COL_CONST, "baseline_linear": COL_LIN}

PLOT_BG = dict(paper_bgcolor="white", plot_bgcolor="#fafafa",
               font=dict(family="system-ui, sans-serif", size=13))

_REQUIRED = ["rollout_summary.csv", "rollout_paired_summary.csv",
             "rollout_degradation.csv", "rollout_predictions.npz"]


def available():
    """True only if train_rollout.py AND analyze_rollout.py have both run."""
    return all(os.path.exists(os.path.join(RESULTS, f)) for f in _REQUIRED)


def _load():
    summ = pd.read_csv(os.path.join(RESULTS, "rollout_summary.csv"))
    paired = pd.read_csv(os.path.join(RESULTS, "rollout_paired_summary.csv"))
    degr = pd.read_csv(os.path.join(RESULTS, "rollout_degradation.csv"))
    preds = np.load(os.path.join(RESULTS, "rollout_predictions.npz"))
    return summ, paired, degr, preds


def headline():
    """Key numbers for the page's prose, read straight from the results.

    Deliberately includes the numbers that could FALSIFY the hypothesis as well
    as the ones that support it — how many seeds agreed, whether the gap widens
    with horizon, and the degradation ratio — so the prose on the page is
    generated from what happened rather than from what we expected.
    """
    summ, paired, degr, _ = _load()
    a = summ[(summ.subset == "all")].set_index(["model", "horizon"])
    d = degr[degr.subset == "all"].set_index("model")
    p = paired[paired.subset == "all"].set_index("horizon")

    out = {"n_seeds": int(summ.n_seeds.max())}
    for m in ("A_recon_task", "B_task_only"):
        out[m] = {
            "h1": float(a.loc[(m, 1), "mae_px_mean"]),
            "h10": float(a.loc[(m, 10), "mae_px_mean"]),
            "ratio": float(d.loc[m, "ratio_mean"]),
            "ratio_std": float(d.loc[m, "ratio_std"]),
        }
    out["gap_h1"] = float(p.loc[1, "delta_px_mean"])
    out["gap_h10"] = float(p.loc[10, "delta_px_mean"])

    # Robustness: at how many horizons did ALL seeds agree on the sign?
    n_seeds = int(p.n_seeds.max())
    unanimous = [int(h) for h in p.index
                 if int(p.loc[h, "seeds_favouring_A"]) == n_seeds]
    out["unanimous_horizons"] = unanimous
    out["n_horizons"] = int(len(p.index))
    out["seeds_agree_h10"] = int(p.loc[10, "seeds_favouring_A"])

    # Does the gap actually WIDEN with horizon? That is the hypothesis under
    # test, and it is a different question from "is B worse on average".
    out["gap_widens"] = bool(p.loc[10, "delta_px_mean"] >
                             2.0 * p.loc[1, "delta_px_mean"])
    # Who degrades faster, in relative terms?
    out["b_degrades_faster"] = bool(out["B_task_only"]["ratio"] >
                                    out["A_recon_task"]["ratio"])
    for sub in ("bounce", "no_bounce"):
        s = paired[paired.subset == sub].set_index("horizon")
        out[f"gap_h10_{sub}"] = float(s.loc[10, "delta_px_mean"])
        out[f"gap_h1_{sub}"] = float(s.loc[1, "delta_px_mean"])
    return out


def rollout_curve_figure():
    """Position error vs horizon for both models plus the two baselines.

    The shaded band is +/- one standard deviation ACROSS SEEDS, so it shows how
    much of the A/B separation survives re-running the whole experiment.
    """
    summ, _, _, _ = _load()
    s = summ[summ.subset == "all"]
    fig = go.Figure()
    for m in ["baseline_constant", "baseline_linear", "A_recon_task", "B_task_only"]:
        g = s[s.model == m].sort_values("horizon")
        if g.empty:
            continue
        is_model = m.startswith(("A_", "B_"))
        if is_model:
            std = g.mae_px_std.fillna(0.0).values
            fig.add_trace(go.Scatter(
                x=np.r_[g.horizon, g.horizon[::-1]],
                y=np.r_[g.mae_px_mean + std, (g.mae_px_mean - std)[::-1]],
                fill="toself", fillcolor=COLOR[m], opacity=0.15,
                line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=g.horizon, y=g.mae_px_mean, mode="lines+markers", name=LABEL[m],
            line=dict(color=COLOR[m], width=3 if is_model else 1.8,
                      dash=None if is_model else "dot"),
            marker=dict(size=7 if is_model else 5),
            hovertemplate=f"{LABEL[m]}<br>horizon %{{x}}<br>"
                          "%{y:.3f} px<extra></extra>"))
    fig.update_layout(
        height=430, xaxis_title="prediction horizon (steps ahead, no new frames)",
        yaxis_title="position error (px, 16px image)",
        xaxis=dict(dtick=1), margin=dict(l=65, r=25, t=55, b=50),
        title="Blind rollout: how far off is the predicted position?",
        legend=dict(orientation="h", y=-0.20), **PLOT_BG)
    return fig


def rollout_gap_figure():
    """Paired B-minus-A gap vs horizon, split by wall bounces.

    Positive means the task-only model is worse. Because A and B are scored on
    the identical trajectories we compare them pairwise, which is far more
    sensitive than checking whether two separate error bars overlap.
    """
    _, paired, _, _ = _load()
    fig = go.Figure()
    styles = {"all": ("#333333", "all trajectories", None),
              "bounce": ("#c1121f", "bounces off a wall in the window", "dash"),
              "no_bounce": ("#457b9d", "no bounce (straight-line motion)", "dot")}
    for sub, (colr, name, dash) in styles.items():
        g = paired[paired.subset == sub].sort_values("horizon")
        if g.empty:
            continue
        err = g.delta_px_std.fillna(0.0).values
        fig.add_trace(go.Scatter(
            x=g.horizon, y=g.delta_px_mean, mode="lines+markers", name=name,
            line=dict(color=colr, width=2.5, dash=dash),
            error_y=dict(type="data", array=err, visible=True,
                         color=colr, thickness=1.2, width=3),
            hovertemplate=f"{name}<br>horizon %{{x}}<br>"
                          "B is %{y:+.3f} px worse<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#888", width=1.5))
    fig.add_annotation(x=0.02, y=0.97, xref="paper", yref="paper",
                       xanchor="left", yanchor="top", showarrow=False,
                       text="<span style='font-size:12px;color:#666'>"
                            "above the line = task-only (B) is worse</span>")
    fig.update_layout(
        height=400, xaxis_title="prediction horizon (steps ahead)",
        yaxis_title="B minus A  (px; error bars = spread over seeds)",
        xaxis=dict(dtick=1), margin=dict(l=65, r=25, t=55, b=50),
        title="Paired gap between the two models, and where it comes from",
        legend=dict(orientation="h", y=-0.22), **PLOT_BG)
    return fig


def rollout_normalised_figure():
    """The 'degrades faster' question, plotted directly.

    Left: each model's error DIVIDED BY its own error at horizon 1. This strips
    out the constant offset between them and shows only the growth rate, which
    is what "degrades faster" actually means. Right: the per-seed degradation
    ratio err(h10)/err(h1) as points, so the seed spread is visible instead of
    hidden inside a mean.

    Plotting the absolute curves alone would let a constant offset masquerade as
    a difference in temporal stability; these two panels separate them.
    """
    summ, _, degr, _ = _load()
    s = summ[summ.subset == "all"]
    err = pd.read_csv(os.path.join(RESULTS, "rollout_error.csv"))
    err = err[err.subset == "all"]

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.13,
        subplot_titles=("Error relative to each model's own horizon-1 error",
                        "Growth factor h1→h10, per seed"))

    for m in ("A_recon_task", "B_task_only"):
        g = s[s.model == m].sort_values("horizon")
        base = float(g[g.horizon == 1].mae_px_mean.iloc[0])
        fig.add_trace(go.Scatter(
            x=g.horizon, y=g.mae_px_mean / base, mode="lines+markers",
            name=LABEL[m], line=dict(color=COLOR[m], width=3),
            marker=dict(size=7),
            hovertemplate=f"{LABEL[m]}<br>horizon %{{x}}<br>"
                          "%{y:.2f}x its h1 error<extra></extra>"), row=1, col=1)

    # Per-seed growth factors, jittered so overlapping points stay visible.
    rng = np.random.default_rng(0)
    for j, m in enumerate(("A_recon_task", "B_task_only")):
        ratios = []
        for _, gg in err[err.model == m].groupby("seed"):
            gg = gg.set_index("horizon")
            ratios.append(gg.loc[10, "mae_px"] / gg.loc[1, "mae_px"])
        fig.add_trace(go.Scatter(
            x=np.full(len(ratios), j) + rng.uniform(-0.08, 0.08, len(ratios)),
            y=ratios, mode="markers", showlegend=False,
            marker=dict(color=COLOR[m], size=11, opacity=0.75,
                        line=dict(color="white", width=1)),
            hovertemplate=f"{LABEL[m]}<br>%{{y:.2f}}x<extra></extra>"),
            row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[j - 0.22, j + 0.22], y=[np.mean(ratios)] * 2, mode="lines",
            line=dict(color=COLOR[m], width=3), showlegend=False,
            hoverinfo="skip"), row=1, col=2)

    fig.update_xaxes(title_text="prediction horizon", dtick=1, row=1, col=1)
    fig.update_yaxes(title_text="error ÷ own h1 error", row=1, col=1)
    fig.update_xaxes(tickvals=[0, 1], ticktext=["A", "B"], range=[-0.5, 1.5],
                     row=1, col=2)
    fig.update_yaxes(title_text="err(h10) / err(h1)", row=1, col=2)
    fig.update_layout(
        height=400, margin=dict(l=65, r=25, t=60, b=50),
        title="Do the two degrade at different RATES? (offset removed)",
        legend=dict(orientation="h", y=-0.20), **PLOT_BG)
    return fig


def rollout_examples_figure(n_show=4, seed=0):
    """Individual held-out rollouts: true position over time vs predictions.

    Solid grey = the true trajectory. The shaded span marks the context frames
    the models actually saw; everything to the right of it is predicted blind.
    We deliberately show examples that DO bounce, since that is where the two
    models differ — the caption on the page says so, so this is illustration,
    not a cherry-picked score.
    """
    _, _, _, preds = _load()
    t_true = preds["t_true"]
    ctx, hor = int(preds["context"]), int(preds["horizon"])
    bounce = preds["bounce_in_horizon"]

    idx = np.flatnonzero(bounce)
    rng = np.random.default_rng(seed)
    idx = rng.choice(idx, size=min(n_show, len(idx)), replace=False) \
        if len(idx) else np.arange(min(n_show, len(t_true)))

    cols = len(idx)
    fig = make_subplots(rows=1, cols=cols, horizontal_spacing=0.045,
                        subplot_titles=[f"trajectory {int(i)}" for i in idx],
                        shared_yaxes=True)
    steps = np.arange(1, hor + 1)
    for c, i in enumerate(idx, start=1):
        frames = np.arange(ctx + hor)
        fig.add_trace(go.Scatter(
            x=frames, y=t_true[i, :ctx + hor], mode="lines",
            line=dict(color="#999", width=2), name="true position",
            showlegend=(c == 1),
            hovertemplate="frame %{x}<br>true %{y:.3f}<extra></extra>"), row=1, col=c)
        for key, m in (("pred_A", "A_recon_task"), ("pred_B", "B_task_only")):
            fig.add_trace(go.Scatter(
                x=ctx - 1 + steps, y=preds[key][i], mode="markers",
                marker=dict(color=COLOR[m], size=6,
                            symbol="triangle-up" if m.startswith("A") else "circle"),
                name=LABEL[m], showlegend=(c == 1),
                hovertemplate=f"{LABEL[m]}<br>frame %{{x}}<br>"
                              "predicted %{y:.3f}<extra></extra>"), row=1, col=c)
        # Shade the observed context window.
        fig.add_vrect(x0=-0.5, x1=ctx - 0.5, fillcolor="#000", opacity=0.06,
                      line_width=0, row=1, col=c)
        fig.update_xaxes(title_text="frame", row=1, col=c)
    fig.update_yaxes(title_text="position (0 = left wall, 1 = right)",
                     range=[-0.05, 1.05], row=1, col=1)
    fig.update_layout(
        height=360, margin=dict(l=70, r=25, t=55, b=50),
        title="Blind rollouts on held-out trajectories "
              "(shaded = frames the models saw; everything after is imagined)",
        legend=dict(orientation="h", y=-0.25), **PLOT_BG)
    return fig


if __name__ == "__main__":
    if not available():
        raise SystemExit("rollout results missing — run train_rollout.py then "
                         "analyze_rollout.py first")
    h = headline()
    print(f"seeds: {h['n_seeds']}")
    for m in ("A_recon_task", "B_task_only"):
        v = h[m]
        print(f"{LABEL[m]:20s} h1={v['h1']:.3f}px  h10={v['h10']:.3f}px  "
              f"ratio={v['ratio']:.2f}")
    print(f"paired gap (B-A): h1={h['gap_h1']:+.3f}  h10={h['gap_h10']:+.3f}")
    print(f"  bounce h10={h['gap_h10_bounce']:+.3f}  "
          f"no_bounce h10={h['gap_h10_no_bounce']:+.3f}")
    for f in (rollout_curve_figure, rollout_gap_figure, rollout_examples_figure):
        f()
    print("all rollout figures build OK")
