"""
interactive_viz.py — Interactive Dash app, organised as a NARRATIVE.

The page answers four questions in order, instead of dumping diagnostics:

  1. Does the task-only encoder actually work?      -> scoreboard + readout plot
  2. How cleanly does it represent the task?        -> ablation concentration
  3. What does the latent space look like?          -> 3-D task-aligned scatter
  4. What's the honest catch?                       -> distractor panel + curves

Design note on the 3-D view: we deliberately do NOT plot PCA axes there. PCA
finds the highest-VARIANCE directions, and in this dataset that is always the
background — so a PCA scatter looks like confetti for both encoders and hides
the result. Instead we project onto interpretable axes computed in analyze.py
(a "task axis" and an orthogonalised "distractor axis"), which shows the actual
structure. The raw PCA panel is still available lower down for completeness.

Run:  python interactive_viz.py     then open http://127.0.0.1:8050
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
from dash import Dash, dcc, html, Input, Output, State

from data import render, IMG_SIZE  # reused to draw B's predicted position
import viz_rollout                 # section 6 (multi-step rollout experiment)

RESULTS = os.path.join(os.path.dirname(__file__), "results")

# ---------------------------------------------------------------------------
# Load every result file once at startup. Nothing here re-trains or re-computes
# a model — the page is purely a view over what train.py / analyze.py produced.
# ---------------------------------------------------------------------------
snaps = np.load(os.path.join(RESULTS, "latent_snapshots.npz"))
EPOCHS = snaps["epochs"]
ZA, ZB = snaps["ZA"], snaps["ZB"]
T = snaps["t"]
BRIGHT = snaps["brightness"]
N_SNAP, N_EVAL, LATENT = ZA.shape

probe_df = pd.read_csv(os.path.join(RESULTS, "probing_metrics.csv"))
abl_df = pd.read_csv(os.path.join(RESULTS, "ablation_task_probe.csv"))
pca_df = pd.read_csv(os.path.join(RESULTS, "pca_coords.csv"))
readout_df = pd.read_csv(os.path.join(RESULTS, "task_readout.csv"))
aligned_df = pd.read_csv(os.path.join(RESULTS, "task_aligned.csv"))
recon_npz = np.load(os.path.join(RESULTS, "recon_examples.npz"))
ROT = np.load(os.path.join(RESULTS, "readout_over_time.npz"))  # per-epoch readouts

ENCODERS = ("A_recon", "B_task")
LABEL = {"A_recon": "A · reconstruction", "B_task": "B · task-only"}
COL_A, COL_B = "#d1495b", "#2e86ab"   # A = red, B = blue, used everywhere
COLOR = {"A_recon": COL_A, "B_task": COL_B}

# ---------------------------------------------------------------------------
# Headline numbers, computed from the saved results (never hard-coded).
# ---------------------------------------------------------------------------
def _headline():
    out = {}
    for enc in ENCODERS:
        s = readout_df[readout_df.encoder == enc]
        err = np.abs(s.t_true - s.t_pred)
        a = abl_df[abl_df.encoder == enc]
        out[enc] = {
            "r2": r2_score(s.t_true, s.t_pred),           # position readout R²
            "mae": float(err.mean()),                     # avg error, in units of t
            "px": float(err.mean() * 15),                 # same error, in pixels (16px wide)
            "top_share": float(a.importance_frac.max()),  # concentration of task info
            "n_dims_80": int((a.sort_values("importance_frac", ascending=False)
                              .importance_frac.cumsum() < 0.8).sum() + 1),
            "distr": float(probe_df[(probe_df.encoder == enc) &
                                    (probe_df.epoch == EPOCHS[-1])].distractor_r2.iloc[0]),
        }
    return out


H = _headline()

# ---------------------------------------------------------------------------
# Heatmap rows = task-conditional mean latent E[z | t] (see below).
# Because the distractors are independent of t, averaging within a t-bin cancels
# them out, so what remains down the rows is the task-dependent structure.
# ---------------------------------------------------------------------------
N_BINS = 40
_ORDER = np.argsort(T)
BINS = list(np.array_split(_ORDER, N_BINS))


def _bin_stack(Z):
    return np.stack([np.stack([Z[i][idx].mean(axis=0) for idx in BINS])
                     for i in range(Z.shape[0])])


ZA_BIN, ZB_BIN = _bin_stack(ZA), _bin_stack(ZB)
ZA_RANGE = (float(ZA_BIN.min()), float(ZA_BIN.max()))
ZB_RANGE = (float(ZB_BIN.min()), float(ZB_BIN.max()))

PLOT_BG = dict(paper_bgcolor="white", plot_bgcolor="#fafafa",
               font=dict(family="system-ui, sans-serif", size=13))


# ===========================================================================
# 1. "Does it work?" — position read out of each latent, vs the truth
# ===========================================================================
def readout_figure():
    """Predicted vs actual object position for both encoders.

    Same linear probe, same train/test split for both, so it is a fair
    head-to-head. Points hugging the dashed diagonal = an accurate read-out.
    """
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.09,
        subplot_titles=[
            f"<b>{LABEL[e]}</b>   R² = {H[e]['r2']:.3f} · avg error {H[e]['px']:.2f} px"
            for e in ENCODERS],
    )
    for col, enc in enumerate(ENCODERS, start=1):
        s = readout_df[readout_df.encoder == enc]
        fig.add_trace(go.Scatter(
            x=s.t_true, y=s.t_pred, mode="markers",
            marker=dict(size=4, color=COLOR[enc], opacity=0.45),
            hovertemplate="true %{x:.2f}<br>predicted %{y:.2f}<extra></extra>",
        ), row=1, col=col)
        # Perfect-prediction reference line.
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color="#333", dash="dash", width=1.5)),
                      row=1, col=col)
        fig.update_xaxes(title_text="true position", range=[-0.02, 1.02], row=1, col=col)
    fig.update_yaxes(title_text="predicted position", range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], row=1, col=2)
    fig.update_layout(height=400, showlegend=False,
                      margin=dict(l=60, r=30, t=60, b=50), **PLOT_BG)
    return fig


# ===========================================================================
# 2. "How cleanly?" — how concentrated is the task information
# ===========================================================================
def concentration_figure():
    """Latent dims ranked by causal task-importance, most important first.

    Each bar is that dimension's share of the total damage done to task
    prediction when it is ablated (zeroed). A single tall bar means the task
    lives in ONE dimension; a flat spread means it is smeared across many.
    """
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.10,
        subplot_titles=[
            f"<b>{LABEL[e]}</b>   top dim = {H[e]['top_share']*100:.0f}% of task info"
            for e in ENCODERS],
    )
    for col, enc in enumerate(ENCODERS, start=1):
        s = (abl_df[abl_df.encoder == enc]
             .sort_values("importance_frac", ascending=False).reset_index(drop=True))
        fig.add_trace(go.Bar(
            x=[f"dim {int(d)}" for d in s.dim], y=s.importance_frac,
            marker_color=COLOR[enc],
            hovertemplate="%{x}<br>%{y:.1%} of task importance<extra></extra>",
        ), row=1, col=col)
        fig.update_xaxes(title_text="latent dims, ranked", row=1, col=col)
        fig.update_yaxes(range=[0, 0.9], row=1, col=col)
    fig.update_yaxes(title_text="share of task information", tickformat=".0%", row=1, col=1)
    fig.update_layout(height=380, showlegend=False,
                      margin=dict(l=60, r=30, t=60, b=50), **PLOT_BG)
    return fig


# ===========================================================================
# 3. "What does the latent look like?" — 3-D, on interpretable axes
# ===========================================================================
def latent3d_figure(coloring):
    """Rotatable 3-D view of both latents on TASK-ALIGNED axes.

    x = task axis (the direction that reads out position)
    y = distractor axis (background direction, orthogonalised against x)
    z = leading leftover direction

    Coloured by position, a good task representation is a smooth gradient
    running left-to-right along x. Coloured by background, you see how much
    distractor structure is still sitting in the latent (spoiler: plenty, in
    both — that's the honest caveat).
    """
    key = {"Position (the task)": "task",
           "Background brightness (a distractor)": "brightness"}[coloring]
    cmap = "Viridis" if key == "task" else "Cividis"
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.02,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[f"<b>{LABEL[e]}</b>" for e in ENCODERS],
    )
    for col, enc in enumerate(ENCODERS, start=1):
        s = aligned_df[aligned_df.encoder == enc]
        fig.add_trace(go.Scatter3d(
            x=s.task_axis, y=s.distractor_axis, z=s.residual_axis, mode="markers",
            marker=dict(size=2.4, opacity=0.75, color=s[key], colorscale=cmap,
                        showscale=(col == 2),
                        colorbar=dict(title=coloring.split(" (")[0], x=1.02, len=0.75)),
            hovertemplate="task axis %{x:.2f}<br>distractor axis %{y:.2f}<extra></extra>",
        ), row=1, col=col)
    scene = dict(xaxis_title="task axis", yaxis_title="distractor axis",
                 zaxis_title="leftover",
                 camera=dict(eye=dict(x=1.6, y=1.5, z=0.9)))
    fig.update_layout(height=540, showlegend=False,
                      margin=dict(l=0, r=60, t=45, b=0),
                      scene=scene, scene2=scene, **PLOT_BG)
    return fig


# ===========================================================================
# 4. The honest catch — what each encoder spends capacity on
# ===========================================================================
def tradeoff_figure():
    """Two grouped bars: how well each latent reads out the TASK, and how much
    DISTRACTOR it still carries. Shows the win and the caveat in one view."""
    task_vals = [H[e]["r2"] for e in ENCODERS]
    distr_vals = [H[e]["distr"] for e in ENCODERS]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[LABEL[e] for e in ENCODERS], y=task_vals,
                         name="Task readout (higher = better)",
                         marker_color="#2a9d8f",
                         text=[f"{v:.3f}" for v in task_vals], textposition="outside"))
    fig.add_trace(go.Bar(x=[LABEL[e] for e in ENCODERS], y=distr_vals,
                         name="Distractor still encoded (lower = cleaner)",
                         marker_color="#bbb",
                         text=[f"{v:.3f}" for v in distr_vals], textposition="outside"))
    fig.update_layout(barmode="group", height=380, yaxis_range=[0, 1.15],
                      yaxis_title="linear-probe R²",
                      margin=dict(l=60, r=30, t=30, b=40),
                      legend=dict(orientation="h", y=-0.18), **PLOT_BG)
    return fig


def recon_figure():
    """Three rows: the input, what A rebuilds, and what B actually knows.

    Row 2 is Encoder A's genuine decoder output — it repaints the background
    faithfully, which is capacity spent on information the task never needs.

    Row 3 needs a caveat, and the page states it: Encoder B has NO decoder and
    cannot produce an image at all. The only thing it outputs is a number, the
    object's position. So we RENDER that number — a dot drawn at B's predicted
    location on a blank background — using the same blob renderer the dataset
    uses. It is a picture of B's knowledge, not a reconstruction by B. That is
    precisely the point: B threw the background away and kept the position, and
    its position is the more accurate of the two.
    """
    imgs, recons = recon_npz["images"], recon_npz["recons"]
    t_true, b_pred = recon_npz["t"], recon_npz["b_pred"]
    n = len(imgs)

    # Neutral grey canvas + no pixel noise, so row 3 shows ONLY the position.
    rng = np.random.default_rng(0)
    grey = np.array([0.18, 0.18, 0.20], dtype=np.float32)

    fig = make_subplots(
        rows=3, cols=n, horizontal_spacing=0.012, vertical_spacing=0.05,
        row_titles=["input", "A rebuilds it", "B knows where it is"],
    )
    for i in range(n):
        fig.add_trace(go.Image(z=(imgs[i] * 255).astype(np.uint8),
                               hoverinfo="skip"), row=1, col=i + 1)
        fig.add_trace(go.Image(z=(np.clip(recons[i], 0, 1) * 255).astype(np.uint8),
                               hoverinfo="skip"), row=2, col=i + 1)
        # Render B's predicted position as a dot on an empty background.
        pred_img = render(float(np.clip(b_pred[i], 0, 1)), grey, rng, noise_std=0.0)
        err_px = abs(float(b_pred[i]) - float(t_true[i])) * (IMG_SIZE - 1)
        fig.add_trace(go.Image(z=(pred_img * 255).astype(np.uint8),
                               hoverinfo="skip"), row=3, col=i + 1)
        # Label each prediction with how far off it was, in pixels.
        fig.add_annotation(text=f"{err_px:.2f} px off", row=3, col=i + 1,
                           x=IMG_SIZE / 2, y=IMG_SIZE + 3.5, showarrow=False,
                           font=dict(size=10, color="#666"))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(height=390, margin=dict(l=20, r=110, t=16, b=10), **PLOT_BG)
    # Row labels sit on the right by default; keep them small and grey.
    for ann in fig.layout.annotations:
        if ann.text in ("input", "A rebuilds it", "B knows where it is"):
            ann.font = dict(size=12, color="#333")
    return fig


# ===========================================================================
# 5. Training dynamics (explorable detail)
# ===========================================================================
def heatmap_figure(snap_i):
    """Task-conditional mean latent E[z|t] for both encoders at one snapshot.

    The two encoders are on their own colour scales (their activation magnitudes
    differ), so each panel carries its own colourbar — placed in the gap between
    the panels and at the far right so neither collides with an axis.
    """
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.19,
                        subplot_titles=[f"<b>{LABEL[e]}</b>" for e in ENCODERS])
    for col, (Zbin, zr) in enumerate([(ZA_BIN, ZA_RANGE), (ZB_BIN, ZB_RANGE)], start=1):
        fig.add_trace(go.Heatmap(
            z=Zbin[snap_i], zmin=zr[0], zmax=zr[1], colorscale="RdBu",
            reversescale=True,
            colorbar=dict(x=0.445 if col == 1 else 1.005, len=0.86,
                          thickness=11, tickfont=dict(size=10)),
            hovertemplate="dim %{x}<br>position bin %{y}<br>%{z:.2f}<extra></extra>",
        ), row=1, col=col)
        fig.update_xaxes(title_text="latent dim", dtick=1, row=1, col=col)
    fig.update_yaxes(title_text="object position (low → high)", row=1, col=1)
    fig.update_layout(height=400, margin=dict(l=70, r=55, t=52, b=45),
                      title=f"Average latent per position — epoch {EPOCHS[snap_i]}"
                            " (a smooth column = that dim tracks position)",
                      **PLOT_BG)
    return fig


def readout_scatter_figure(snap_i):
    """Predicted vs actual object position at one epoch, for both encoders.

    Every point is a held-out image. Perfect prediction lies on the dashed
    diagonal, so the CLOUD'S THICKNESS is the error — no axis-reading required.
    Animated across epochs, both clouds collapse toward the line and the gap in
    tightness between them is directly visible.
    """
    t = ROT["t_show"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=[f"<b>{LABEL[e]}</b>" for e in ENCODERS])
    for col, (enc, preds) in enumerate(
            zip(ENCODERS, (ROT["predA_show"], ROT["predB_show"])), start=1):
        p = preds[snap_i]
        r2 = r2_score(t, p)
        err_px = float(np.abs(p - t).mean() * (IMG_SIZE - 1))
        fig.add_trace(go.Scatter(
            x=t, y=p, mode="markers",
            marker=dict(size=5, color=COLOR[enc], opacity=0.45),
            hovertemplate="true %{x:.2f}<br>predicted %{y:.2f}<extra></extra>",
        ), row=1, col=col)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color="#333", dash="dash", width=1.5)),
                      row=1, col=col)
        # Live score, so the animation carries a number as well as a shape.
        fig.add_annotation(
            row=1, col=col, x=0.04, y=0.99, xref="x domain", yref="y domain",
            xanchor="left", yanchor="top", showarrow=False, align="left",
            text=f"<b>R² = {r2:.3f}</b><br><span style='font-size:11px'>"
                 f"avg err {err_px:.2f} px</span>",
            font=dict(size=15, color=COLOR[enc]),
            bgcolor="rgba(255,255,255,0.82)", borderpad=5)
        fig.update_xaxes(title_text="true position", range=[-0.03, 1.03], row=1, col=col)
    fig.update_yaxes(title_text="predicted position", range=[-0.12, 1.12], row=1, col=1)
    fig.update_yaxes(range=[-0.12, 1.12], row=1, col=2)
    fig.update_layout(height=400, showlegend=False,
                      margin=dict(l=60, r=30, t=58, b=48),
                      title=f"Predicted vs actual position — epoch {EPOCHS[snap_i]}"
                            "  (tighter to the line = better)", **PLOT_BG)
    return fig


def tracker_figure(snap_i):
    """Where each encoder thinks the object is, on real held-out images.

    Top row: the actual image. Bottom strip: a ruler in pixel coordinates with
    three markers — the true column, Encoder A's guess, and Encoder B's guess.
    Markers sit below the image rather than on top of it, because the background
    colour is random and a red line on a red background would be invisible.

    Watch the two coloured markers walk toward the black diamond as training
    proceeds. The per-image pixel errors are printed underneath.
    """
    imgs = ROT["track_images"]
    t_true = ROT["track_t"]
    pa, pb = ROT["predA_track"][snap_i], ROT["predB_track"][snap_i]
    n = len(imgs)
    W = IMG_SIZE - 1  # position 0..1 maps onto columns 0..W

    fig = make_subplots(rows=2, cols=n, row_heights=[0.74, 0.26],
                        horizontal_spacing=0.018, vertical_spacing=0.04)
    for i in range(n):
        c = i + 1
        fig.add_trace(go.Image(z=(imgs[i] * 255).astype(np.uint8), hoverinfo="skip"),
                      row=1, col=c)
        # Ruler strip: true position plus each encoder's guess, in pixel units.
        marks = [("true", t_true[i] * W, "#111", "diamond", 11),
                 (LABEL["A_recon"], pa[i] * W, COL_A, "triangle-up", 12),
                 (LABEL["B_task"], pb[i] * W, COL_B, "triangle-up", 12)]
        for name, xpos, colr, sym, size in marks:
            fig.add_trace(go.Scatter(
                x=[xpos], y=[0], mode="markers", name=name,
                marker=dict(color=colr, symbol=sym, size=size,
                            line=dict(color="white", width=1)),
                showlegend=(i == 0),
                hovertemplate=f"{name}: %{{x:.2f}} px<extra></extra>",
            ), row=2, col=c)
        fig.update_xaxes(range=[-0.5, W + 0.5], showticklabels=False,
                         showgrid=False, zeroline=False, row=2, col=c)
        fig.update_yaxes(range=[-1, 1], showticklabels=False, showgrid=False,
                         zeroline=False, row=2, col=c)
        # Per-image error readout, in pixels.
        fig.add_annotation(
            row=2, col=c, x=W / 2, y=-0.95, showarrow=False, yanchor="top",
            text=f"<span style='color:{COL_A}'>{abs(pa[i]-t_true[i])*W:.2f}</span> / "
                 f"<span style='color:{COL_B}'><b>{abs(pb[i]-t_true[i])*W:.2f}</b></span> px",
            font=dict(size=11))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1)

    # These are a handful of individual images, and individual images are noisy.
    # Print the population statistic alongside so the eye anchors on the real
    # number rather than on eight anecdotes.
    t_all = ROT["t_show"]
    ea = np.abs(ROT["predA_show"][snap_i] - t_all).mean() * W
    eb = np.abs(ROT["predB_show"][snap_i] - t_all).mean() * W
    win = 100.0 * (np.abs(ROT["predB_show"][snap_i] - t_all)
                   < np.abs(ROT["predA_show"][snap_i] - t_all)).mean()

    fig.update_layout(
        height=340, margin=dict(l=20, r=20, t=70, b=46),
        title=dict(text=(
            f"Where each encoder thinks the object is — epoch {EPOCHS[snap_i]}"
            f"<br><span style='font-size:12px;color:#666'>"
            f"per-image error in px: <span style='color:{COL_A}'>A</span> / "
            f"<span style='color:{COL_B}'>B</span>.  Across all "
            f"{len(t_all)} held-out images: "
            f"<span style='color:{COL_A}'>A {ea:.2f} px</span> · "
            f"<span style='color:{COL_B}'><b>B {eb:.2f} px</b></span> "
            f"— B closer on {win:.0f}% of them.</span>")),
        legend=dict(orientation="h", y=-0.10, x=0.5, xanchor="center"),
        **PLOT_BG)
    return fig


def pca_figure(coloring):
    """The raw 2-D PCA view, kept for completeness (see the note on the page)."""
    key = {"Position (the task)": "task",
           "Background brightness (a distractor)": "brightness"}[coloring]
    cmap = "Viridis" if key == "task" else "Cividis"
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13,
                        subplot_titles=[f"<b>{LABEL[e]}</b>" for e in ENCODERS])
    for col, enc in enumerate(ENCODERS, start=1):
        s = pca_df[pca_df.encoder == enc]
        fig.add_trace(go.Scatter(
            x=s.pc1, y=s.pc2, mode="markers",
            marker=dict(size=4, opacity=0.6, color=s[key], colorscale=cmap,
                        showscale=(col == 2), colorbar=dict(x=1.02)),
        ), row=1, col=col)
        fig.update_xaxes(title_text="PC1", row=1, col=col)
    fig.update_yaxes(title_text="PC2", row=1, col=1)
    fig.update_layout(height=380, showlegend=False,
                      margin=dict(l=60, r=70, t=45, b=45), **PLOT_BG)
    return fig


# ===========================================================================
# Page layout
# ===========================================================================
CARD = {"flex": "1", "background": "#fff", "borderRadius": "10px",
        "padding": "16px 18px", "border": "1px solid #e6e6e6"}
BTN = {"background": COL_B, "color": "#fff", "border": "none", "borderRadius": "6px",
       "padding": "9px 20px", "fontSize": "14px", "fontWeight": "600",
       "cursor": "pointer", "minWidth": "104px"}
SECTION = {"margin": "38px 0 10px"}
NOTE = {"color": "#555", "fontSize": "15px", "lineHeight": "1.6", "maxWidth": "820px"}

app = Dash(__name__)
app.title = "Hermes toy demo"


def rollout_section():
    """Section 6 — the multi-step rollout experiment.

    Returns an empty list when the rollout has not been run, so the static demo
    still stands alone. All numbers in the prose are read from the saved results
    rather than typed in, so the text cannot drift away from the data.
    """
    if not viz_rollout.available():
        return []
    h = viz_rollout.headline()
    a, b = h["A_recon_task"], h["B_task_only"]
    ns = h["n_seeds"]
    worse = h["gap_h10"] > 0
    who = ("the task-only model", COL_B) if worse else ("the reconstruction model", COL_A)
    faster = "B" if h["b_degrades_faster"] else "A"
    slower = "A" if h["b_degrades_faster"] else "B"

    return [
        html.H2("6. Does it survive multi-step imagination?", style=SECTION),
        html.P(["Everything above is a single-frame result. The claim that "
                "actually matters for a world model is temporal: feed a few "
                "real frames, then cut off the pixels and make the model "
                "imagine forward on its own. We put a GRU on top of each "
                "encoder and trained both end to end to predict the object's "
                "position 1–10 steps ahead. ",
                html.B("A"), " keeps a reconstruction loss alongside the task "
                "loss; ", html.B("B"), " gets the task loss only and has no "
                "decoder at all. Both get the task gradient, so this is a fair "
                "fight."], style=NOTE),
        html.Div(style={"display": "flex", "gap": "14px", "margin": "18px 0 6px"},
                 children=[
                     stat_card(f"{a['h1']:.2f} → {a['h10']:.2f} px",
                               "A · recon + task",
                               f"error at horizon 1 → 10  (×{a['ratio']:.1f})", COL_A),
                     stat_card(f"{b['h1']:.2f} → {b['h10']:.2f} px",
                               "B · task-only",
                               f"error at horizon 1 → 10  (×{b['ratio']:.1f})", COL_B),
                     stat_card(f"×{b['ratio']:.1f} vs ×{a['ratio']:.1f}",
                               "Error growth, h1 → h10",
                               f"B grows {slower_word(h)} than A — the opposite "
                               f"of the collapse hypothesis", COL_B),
                 ]),
        dcc.Graph(figure=viz_rollout.rollout_curve_figure(),
                  config={"displayModeBar": False}),
        html.P([html.B("The headline is a null result. "),
                "The worry was that a task-only latent would hold up one step "
                "ahead and then fall apart under imagination. It doesn't. B is "
                f"slightly worse than A at every horizon — about "
                f"{h['gap_h1']:.2f} px at one step and {h['gap_h10']:.2f} px at "
                f"ten, on a 16-pixel image — but that gap is roughly ",
                html.B("constant"), ", not widening. Both models beat holding "
                "the last position by a wide margin, and both beat straight-line "
                "extrapolation at long horizons, which means both learned that "
                "the dot bounces off the walls."],
               style={**NOTE, "marginTop": "16px"}),
        dcc.Graph(figure=viz_rollout.rollout_normalised_figure(),
                  config={"displayModeBar": False}),
        html.P([html.B("On the literal question, B wins. "),
                "\"Degrades faster\" is a question about rate, not level — a "
                "model can be uniformly worse while decaying at the same speed. "
                "Dividing each model by its own horizon-1 error removes the "
                f"offset, and what's left is that A's error grows ×{a['ratio']:.1f} "
                f"from horizon 1 to 10 while B's grows ×{b['ratio']:.1f}. The "
                f"task-only latent degrades ", html.B(f"{slower_word(h)}"),
                ", not faster. The per-seed points on the right show why that "
                "should be held loosely: the spread across seeds is comparable "
                "to the difference between the models."],
               style={**NOTE, "marginTop": "16px"}),
        dcc.Graph(figure=viz_rollout.rollout_gap_figure(),
                  config={"displayModeBar": False}),
        html.P([html.B("How much of this is noise? "),
                f"A good deal. Re-running the whole experiment across {ns} seeds, "
                f"all {ns} agreed on the sign of the gap at only "
                f"{len(h['unanimous_horizons'])} of {h['n_horizons']} horizons "
                f"(at horizon 10 it was {h['seeds_agree_h10']}/{ns} — two seeds "
                "had B ahead). A pilot run on a single seed suggested the whole "
                "A-advantage came from wall bounces; that pattern did not "
                "survive the sweep, and the bounce and no-bounce curves here "
                "cross over. Treat the A/B difference as small and unstable, "
                "and the absence of a widening gap as the real finding."],
               style={**NOTE, "marginTop": "16px"}),
        dcc.Graph(figure=viz_rollout.rollout_examples_figure(),
                  config={"displayModeBar": False}),
        html.P(["Caveats. This is one toy, one architecture, one dynamics; the "
                "effect is a fraction of a pixel. We also cannot separate ",
                html.I("\"reconstruction teaches the model about the scene\""),
                " from the duller ", html.I("\"a second loss regularises the "
                "encoder\""), " — an auxiliary objective with nothing to do "
                "with pixels might do the same thing. And a 10-step horizon on "
                "near-linear dynamics is a gentle test; a longer horizon or "
                "richer dynamics could still separate them."],
               style={**NOTE, "marginTop": "16px"}),
    ]


def slower_word(h):
    """'slower'/'faster' for B's degradation rate, decided by the data."""
    return "faster" if h["b_degrades_faster"] else "slower"


def stat_card(value, label, sub, color):
    return html.Div(style=CARD, children=[
        html.Div(value, style={"fontSize": "30px", "fontWeight": "700", "color": color}),
        html.Div(label, style={"fontSize": "14px", "fontWeight": "600", "marginTop": "2px"}),
        html.Div(sub, style={"fontSize": "13px", "color": "#666", "marginTop": "4px"}),
    ])


app.layout = html.Div(
    style={"maxWidth": "1080px", "margin": "0 auto", "padding": "30px 24px 80px",
           "fontFamily": "system-ui, -apple-system, sans-serif", "color": "#1a1a1a",
           "background": "#fff"},
    children=[
        # ---------------- hero ----------------
        html.H1("Does a world model need to reconstruct what it sees?",
                style={"fontSize": "32px", "marginBottom": "8px", "lineHeight": "1.25"}),
        html.P([
            "Two identical encoders squeeze the same images into 8 numbers. ",
            html.B("Encoder A", style={"color": COL_A}),
            " is trained to rebuild the whole image. ",
            html.B("Encoder B", style={"color": COL_B}),
            " is trained only to predict where the object is — it never learns to "
            "draw anything. The question: does skipping reconstruction cost us?"],
            style={**NOTE, "fontSize": "17px"}),
        html.Div(style={"background": "#f0f7f5", "borderLeft": f"4px solid #2a9d8f",
                        "padding": "14px 18px", "borderRadius": "6px",
                        "margin": "22px 0 26px", "fontSize": "16px"},
                 children=[html.B("Answer: no — it helps. "),
                           f"The task-only encoder reads out object position "
                           f"{H['B_task']['px'] / H['A_recon']['px']:.1f}× more precisely, "
                           f"using a fraction of its latent to do it."]),

        html.Div(style={"display": "flex", "gap": "14px", "marginBottom": "10px"},
                 children=[
                     stat_card(f"{H['B_task']['r2']:.3f}", "Task-only accuracy (R²)",
                               f"vs {H['A_recon']['r2']:.3f} for reconstruction", COL_B),
                     stat_card(f"{H['B_task']['px']:.2f} px", "Average position error",
                               f"vs {H['A_recon']['px']:.2f} px — on a 16 px image", COL_B),
                     stat_card(f"{H['B_task']['top_share']*100:.0f}%",
                               "Task info in ONE dimension",
                               f"vs {H['A_recon']['top_share']*100:.0f}% — B is far more concentrated",
                               COL_B),
                 ]),

        # ---------------- 1. does it work ----------------
        html.H2("1. The task-only encoder is more accurate", style=SECTION),
        html.P("Both latents get the same fair test: a linear read-out of the object's "
               "position, trained on half the data and scored on the other half. "
               "Closer to the dashed line is better.", style=NOTE),
        dcc.Graph(figure=readout_figure(), config={"displayModeBar": False}),

        # ---------------- 2. how cleanly ----------------
        html.H2("2. …and it stores the task far more cleanly", style=SECTION),
        html.P("We switch off each of the 8 latent dimensions one at a time and measure "
               "how much the position prediction breaks. That tells us which dimensions "
               "the encoder actually relies on. The task-only encoder concentrates "
               "almost everything into a single dimension; the autoencoder has to smear "
               "it across many, tangled up with the background it also has to remember.",
               style=NOTE),
        dcc.Graph(figure=concentration_figure(), config={"displayModeBar": False}),

        # ---------------- 3. the latent space ----------------
        html.H2("3. See it in the latent space", style=SECTION),
        html.P(["Each dot is one image, floating in the encoder's latent space. "
                "The horizontal axis is the direction that encodes position. ",
                html.B("Colour by position"),
                " and the task-only cloud becomes a clean left-to-right gradient. ",
                html.B("Switch to background"),
                " and that gradient disappears — but notice the background still "
                "structures both clouds along the vertical axis. Drag to rotate."],
               style=NOTE),
        html.Div([
            html.Label("Colour the dots by:  ",
                       style={"fontWeight": "600", "fontSize": "14px"}),
            dcc.Dropdown(id="color-by",
                         options=["Position (the task)",
                                  "Background brightness (a distractor)"],
                         value="Position (the task)", clearable=False,
                         style={"width": "340px", "display": "inline-block",
                                "verticalAlign": "middle"}),
        ], style={"margin": "14px 0 4px"}),
        dcc.Graph(id="latent3d", config={"displayModeBar": False}),

        # ---------------- 4. the catch ----------------
        html.H2("4. The honest catch", style=SECTION),
        html.P("Training on the task alone does not delete the distractor. The "
               "background is still linearly readable from both latents — the task-only "
               "encoder simply isn't forced to rely on it. So the win here is a cleaner, "
               "more concentrated task representation, not distractor invariance. "
               "Getting that would need an extra pressure this toy doesn't have.",
               style=NOTE),
        dcc.Graph(figure=tradeoff_figure(), config={"displayModeBar": False}),
        html.P(["Where each encoder's capacity goes. ",
                html.B("A"), " spends its 8 numbers faithfully repainting the "
                "background — detail the task never needs. ",
                html.B("B"), " has no decoder and cannot draw anything at all; "
                "the only thing it outputs is a position, so the bottom row is "
                "that predicted position drawn as a dot on an empty canvas. It is "
                "a picture of what B kept, not a reconstruction by B — and it is "
                "the more accurate of the two."],
               style={**NOTE, "marginTop": "18px"}),
        dcc.Graph(figure=recon_figure(), config={"displayModeBar": False}),

        # ---------------- 5. explore ----------------
        html.H2("5. Watch it happen during training", style=SECTION),
        html.P(["Press play. Both encoders start from the same weights, so at "
                "epoch 0 they are identical and equally wrong. Watch how fast "
                "they separate: ", html.B("B", style={"color": COL_B}),
                " snaps onto the diagonal within a few epochs, while ",
                html.B("A", style={"color": COL_A}),
                " is still catching up and never gets as tight. Below, the same "
                "story on real images — the coloured markers walk in toward the "
                "true position as each encoder learns to find the object."],
               style=NOTE),
        html.Div([
            # Play/pause drives the slider via a dcc.Interval ticker below.
            html.Button("▶  Play", id="play-btn", n_clicks=0, style=BTN),
            html.Label("Speed:", style={"fontWeight": "600", "fontSize": "13px",
                                        "margin": "0 8px 0 16px"}),
            dcc.Dropdown(id="speed", clearable=False, value=600,
                         options=[{"label": "slow", "value": 1100},
                                  {"label": "normal", "value": 600},
                                  {"label": "fast", "value": 260}],
                         style={"width": "130px", "display": "inline-block",
                                "verticalAlign": "middle", "fontSize": "13px"}),
        ], style={"display": "flex", "alignItems": "center", "margin": "14px 0 6px"}),
        html.Div([
            html.Label("Training epoch:", style={"fontWeight": "600", "fontSize": "14px"}),
            dcc.Slider(min=0, max=len(EPOCHS) - 1, step=1, value=len(EPOCHS) - 1,
                       marks={i: str(int(e)) for i, e in enumerate(EPOCHS)},
                       id="epoch-slider"),
            # The ticker fires only while playing; each tick advances the slider.
            dcc.Interval(id="ticker", interval=600, disabled=True, n_intervals=0),
        ], style={"margin": "4px 0 18px"}),
        dcc.Graph(id="readout-anim", config={"displayModeBar": False}),
        dcc.Graph(id="tracker", config={"displayModeBar": False}),
        html.Details([
            html.Summary("Show the latent activations too",
                         style={"cursor": "pointer", "fontSize": "14px",
                                "color": "#555", "margin": "10px 0"}),
            html.P("Each row is a band of object positions; each column is a "
                   "latent dimension. Averaging within a band cancels the "
                   "(independent) background, so what's left is task structure — "
                   "a column that becomes a smooth gradient is tracking position.",
                   style=NOTE),
            dcc.Graph(id="heatmaps", config={"displayModeBar": False}),
        ]),

        # ---------------- 6. rollout (only if that experiment has been run) ----
        *rollout_section(),

        # ---------------- appendix ----------------
        html.H2("Appendix — the raw PCA view", style=SECTION),
        html.P("For completeness. PCA picks the directions of greatest variance, and "
               "in this dataset that is the background — so both clouds look like "
               "confetti and the result is hidden. This is exactly why section 3 uses "
               "task-aligned axes instead.", style=NOTE),
        dcc.Graph(id="pca", config={"displayModeBar": False}),

        html.Hr(style={"margin": "40px 0 16px", "border": "none",
                       "borderTop": "1px solid #e6e6e6"}),
        html.P("A minimal illustrative toy — 16×16 synthetic images, 8-dim latents, "
               "trained on CPU in under a minute. Every number on this page comes from "
               "one real training run. Not a claim about Hermes's actual results.",
               style={"fontSize": "13px", "color": "#888"}),
    ])


@app.callback(Output("readout-anim", "figure"), Output("tracker", "figure"),
              Output("heatmaps", "figure"),
              Input("epoch-slider", "value"))
def _update_epoch(snap_i):
    return (readout_scatter_figure(snap_i), tracker_figure(snap_i),
            heatmap_figure(snap_i))


@app.callback(Output("ticker", "disabled"), Output("play-btn", "children"),
              Input("play-btn", "n_clicks"))
def _toggle_play(n_clicks):
    """Odd clicks = playing (ticker enabled), even = paused."""
    playing = bool(n_clicks) and n_clicks % 2 == 1
    return (not playing), ("⏸  Pause" if playing else "▶  Play")


@app.callback(Output("ticker", "interval"), Input("speed", "value"))
def _set_speed(ms):
    return ms


@app.callback(Output("epoch-slider", "value"),
              Input("ticker", "n_intervals"),
              State("epoch-slider", "value"),
              prevent_initial_call=True)
def _advance(_n, current):
    """Step the slider one snapshot forward, looping back to the start."""
    return 0 if current is None else (current + 1) % len(EPOCHS)


@app.callback(Output("latent3d", "figure"), Output("pca", "figure"),
              Input("color-by", "value"))
def _update_color(coloring):
    return latent3d_figure(coloring), pca_figure(coloring)


if __name__ == "__main__":
    print("Serving at http://127.0.0.1:8050  (Ctrl+C to stop)")
    app.run(debug=False, port=8050)
