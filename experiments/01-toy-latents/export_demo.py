"""
export_demo.py — Shareable exports of the toy demo.

Produces two artefacts in results/:

  1. position_readout.gif
        ~12 s animation of the predicted-vs-actual position scatter for both
        encoders across training. Both start from the same weights and are
        identical in frame 1; the task-only cloud then collapses onto the
        diagonal while the reconstruction cloud stays visibly fatter, with a
        live R² counter in each panel. This is the clip meant for a tweet /
        LinkedIn post. Built with matplotlib + Pillow (no ffmpeg needed).

  2. demo.html
        A single, self-contained interactive Plotly page (plotly.js embedded, so
        it works offline with no server). Contains the full narrative:
          - the head-to-head position read-out and per-dimension ablation,
          - the 3-D task-aligned latent view with a colour dropdown,
          - the honest-caveat panel and reconstruction examples,
          - three play/pause animations over training epochs (read-out scatter,
            object tracker, latent activations),
          - the raw PCA view as an appendix.

We reuse the data + a couple of figure builders from interactive_viz.py so the
exports match the live app exactly (importing it does NOT start the server).

Run:  python export_demo.py   (after train.py and analyze.py)
"""

import os

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Reuse the exact loaded arrays + figure builders from the live app, so the
# static export and the live page always tell the same story. (Importing the
# module does NOT start the Dash server.)
from interactive_viz import (
    ZA_BIN, ZB_BIN, EPOCHS, ZA_RANGE, ZB_RANGE, ROT, IMG_SIZE,
    probe_df, pca_df, aligned_df, COL_A, COL_B, N_SNAP, H, LABEL, ENCODERS,
    readout_figure, concentration_figure, tradeoff_figure, recon_figure,
    latent3d_figure, pca_figure,
    readout_scatter_figure, tracker_figure, heatmap_figure,
)
import viz_rollout   # section 6 (multi-step rollout), skipped if not yet run

RESULTS = os.path.join(os.path.dirname(__file__), "results")


# ===========================================================================
# 1. Animated GIF of the position read-out over training
# ===========================================================================
def make_gif(path, hold_last=8, frames_per_snap=3, fps=4):
    """Animate the predicted-vs-actual position scatter across training.

    This is the clip meant for a tweet / LinkedIn post, so it shows the claim in
    the most universally readable form there is: points on a diagonal. Both
    encoders start from the same initialisation (identical first frame), then
    separate — the task-only cloud collapses onto the line while the
    reconstruction cloud stays visibly fatter. A live R² counter runs in each
    panel.

    Every frame is a REAL logged snapshot; frames are only repeated to control
    pacing, never interpolated or invented.
    """
    plan = []
    for i in range(N_SNAP):
        plan += [i] * frames_per_snap
    plan += [N_SNAP - 1] * hold_last

    t = ROT["t_show"]
    preds = (ROT["predA_show"], ROT["predB_show"])
    W = IMG_SIZE - 1

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
    scatters, texts = [], []
    for ax, enc, colr in zip(axes, ENCODERS, (COL_A, COL_B)):
        ax.plot([0, 1], [0, 1], "--", color="#333", lw=1.4, zorder=1)
        sc = ax.scatter(t, preds[0][0], s=11, alpha=0.45, color=colr, zorder=2)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.12, 1.12)
        ax.set_title(LABEL[enc], fontsize=11, color=colr, fontweight="bold")
        ax.set_xlabel("true position")
        txt = ax.text(0.04, 0.96, "", transform=ax.transAxes, va="top", ha="left",
                      fontsize=12, color=colr, fontweight="bold",
                      bbox=dict(fc="white", ec="none", alpha=0.82, pad=3))
        scatters.append(sc); texts.append(txt)
    axes[0].set_ylabel("predicted position")
    axes[1].set_yticklabels([])
    suptitle = fig.suptitle("", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    def update(frame):
        i = plan[frame]
        for k in range(2):
            p = preds[k][i]
            scatters[k].set_offsets(np.c_[t, p])
            r2 = 1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)
            texts[k].set_text(f"R² = {r2:.3f}\n{np.abs(p - t).mean() * W:.2f} px err")
        suptitle.set_text(f"Reading the object's position out of each latent — "
                          f"epoch {int(EPOCHS[i])}")
        return scatters + texts + [suptitle]

    anim = FuncAnimation(fig, update, frames=len(plan), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {path}  ({len(plan)} frames, ~{len(plan)/fps:.1f}s)")


# ===========================================================================
# 2. Self-contained interactive HTML
# ===========================================================================
def animate(builder, frame_ms=600, slider_pad=52):
    """Turn a per-epoch figure builder into a self-contained animated figure.

    The Dash app rebuilds a figure on every slider move; a static HTML page
    cannot call Python, so instead we pre-render one Plotly *frame* per epoch and
    let plotly.js swap between them. We reuse the SAME builder functions the live
    app uses, so the shared file and the served page cannot drift apart.

    Each frame carries both data and layout, so per-epoch titles and annotations
    (the live R2 readouts) animate along with the points.
    """
    fig = builder(N_SNAP - 1)          # opening view = fully trained
    fig.frames = [
        go.Frame(name=str(int(EPOCHS[i])), data=builder(i).data,
                 layout=builder(i).layout)
        for i in range(N_SNAP)
    ]

    play_pause = dict(
        type="buttons", direction="left", showactive=False,
        x=0.0, y=-0.20, xanchor="left", yanchor="top", pad=dict(t=0, r=10),
        buttons=[
            dict(label="▶  Play", method="animate",
                 args=[None, dict(mode="immediate", fromcurrent=True,
                                  frame=dict(duration=frame_ms, redraw=True),
                                  transition=dict(duration=0))]),
            dict(label="⏸  Pause", method="animate",
                 args=[[None], dict(mode="immediate",
                                    frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0))]),
        ],
    )
    steps = [dict(method="animate", label=str(int(e)),
                  args=[[str(int(e))],
                        dict(mode="immediate", frame=dict(duration=0, redraw=True),
                             transition=dict(duration=0))])
             for e in EPOCHS]
    fig.update_layout(
        margin=dict(l=60, r=60, t=fig.layout.margin.t or 70, b=110),
        updatemenus=[play_pause],
        sliders=[dict(active=N_SNAP - 1, currentvalue=dict(prefix="epoch "),
                      pad=dict(t=slider_pad), steps=steps)],
    )
    fig.update_layout(height=(fig.layout.height or 400) + 90)
    return fig


def dropdown_3d_figure():
    """The section-3 latent view, with the colour toggle baked in as a Plotly
    dropdown so the static HTML keeps the same interaction as the live app."""
    fig = latent3d_figure("Position (the task)")
    task_colors, bright_colors = [], []
    for enc in ENCODERS:
        sub = aligned_df[aligned_df.encoder == enc]
        task_colors.append(sub["task"].values)
        bright_colors.append(sub["brightness"].values)
    fig.update_layout(
        margin=dict(l=0, r=60, t=70, b=0),
        updatemenus=[dict(
            type="dropdown", x=0.0, y=1.10, xanchor="left",
            buttons=[
                dict(label="Colour by: position (the task)", method="restyle",
                     args=[{"marker.color": task_colors,
                            "marker.colorscale": "Viridis"}, [0, 1]]),
                dict(label="Colour by: background (a distractor)", method="restyle",
                     args=[{"marker.color": bright_colors,
                            "marker.colorscale": "Cividis"}, [0, 1]]),
            ],
        )],
    )
    return fig


CSS = """
body{max-width:1080px;margin:0 auto;padding:30px 24px 70px;
 font-family:system-ui,-apple-system,sans-serif;color:#1a1a1a;background:#fff}
h1{font-size:32px;line-height:1.25;margin-bottom:8px}
h2{margin:38px 0 8px;font-size:23px}
p.note{color:#555;font-size:15px;line-height:1.6;max-width:820px}
.lede{font-size:17px}
.answer{background:#f0f7f5;border-left:4px solid #2a9d8f;padding:14px 18px;
 border-radius:6px;margin:22px 0 26px;font-size:16px}
.cards{display:flex;gap:14px;margin-bottom:6px}
.card{flex:1;background:#fff;border:1px solid #e6e6e6;border-radius:10px;padding:16px 18px}
.card .v{font-size:30px;font-weight:700}
.card .l{font-size:14px;font-weight:600;margin-top:2px}
.card .s{font-size:13px;color:#666;margin-top:4px}
footer{margin-top:40px;border-top:1px solid #e6e6e6;padding-top:16px;
 font-size:13px;color:#888}
"""


def rollout_html():
    """Section 6 for the static export. Empty if the rollout has not been run.

    Mirrors the live page's section 6 and, like it, reads every number from the
    saved results — including the ones that contradict the original hypothesis.
    """
    if not viz_rollout.available():
        return []
    h = viz_rollout.headline()
    a, b = h["A_recon_task"], h["B_task_only"]
    ns = h["n_seeds"]
    rate = "faster" if h["b_degrades_faster"] else "slower"
    n_unan = len(h["unanimous_horizons"])

    return [
        "<h2>6. Does it survive multi-step imagination?</h2>",
        '<p class="note">Everything above is a single-frame result. The claim '
        "that actually matters for a world model is temporal: feed a few real "
        "frames, then cut off the pixels and make the model imagine forward on "
        "its own. We put a GRU on top of each encoder and trained both end to "
        "end to predict the object's position 1&ndash;10 steps ahead. <b>A</b> "
        "keeps a reconstruction loss alongside the task loss; <b>B</b> gets the "
        "task loss only and has no decoder at all. Both get the task gradient, "
        f"so this is a fair fight. Averaged over {ns} seeds.</p>",
        '<div class="cards">'
        + _card(f"{a['h1']:.2f} &rarr; {a['h10']:.2f} px", "A &middot; recon + task",
                f"error at horizon 1 &rarr; 10 (&times;{a['ratio']:.1f})", COL_A)
        + _card(f"{b['h1']:.2f} &rarr; {b['h10']:.2f} px", "B &middot; task-only",
                f"error at horizon 1 &rarr; 10 (&times;{b['ratio']:.1f})", COL_B)
        + _card(f"&times;{b['ratio']:.1f} vs &times;{a['ratio']:.1f}",
                "Error growth, h1 &rarr; h10",
                f"B grows {rate} than A &mdash; the opposite of the collapse "
                "hypothesis", COL_B)
        + "</div>",
        pio.to_html(viz_rollout.rollout_curve_figure(), include_plotlyjs=False,
                    full_html=False, config={"displayModeBar": False}),
        '<p class="note"><b>The headline is a null result.</b> The worry was '
        "that a task-only latent would hold up one step ahead and then fall "
        "apart under imagination. It doesn't. B is slightly worse than A at "
        f"every horizon &mdash; about {h['gap_h1']:.2f} px at one step and "
        f"{h['gap_h10']:.2f} px at ten, on a 16-pixel image &mdash; but that gap "
        "is roughly <b>constant</b>, not widening. Both models beat holding the "
        "last position by a wide margin, and both beat straight-line "
        "extrapolation at long horizons, which means both learned that the dot "
        "bounces off the walls.</p>",
        pio.to_html(viz_rollout.rollout_normalised_figure(), include_plotlyjs=False,
                    full_html=False, config={"displayModeBar": False}),
        '<p class="note"><b>On the literal question, B wins.</b> "Degrades '
        'faster" is a question about rate, not level &mdash; a model can be '
        "uniformly worse while decaying at the same speed. Dividing each model "
        "by its own horizon-1 error removes the offset, and what is left is "
        f"that A's error grows &times;{a['ratio']:.1f} from horizon 1 to 10 "
        f"while B's grows &times;{b['ratio']:.1f}. The task-only latent degrades "
        f"<b>{rate}</b>, not faster. The per-seed points on the right show why "
        "that should be held loosely: the spread across seeds is comparable to "
        "the difference between the models.</p>",
        pio.to_html(viz_rollout.rollout_gap_figure(), include_plotlyjs=False,
                    full_html=False, config={"displayModeBar": False}),
        '<p class="note"><b>How much of this is noise?</b> A good deal. '
        f"Re-running the whole experiment across {ns} seeds, all {ns} agreed on "
        f"the sign of the gap at only {n_unan} of {h['n_horizons']} horizons (at "
        f"horizon 10 it was {h['seeds_agree_h10']}/{ns} &mdash; two seeds had B "
        "ahead). A pilot run on a single seed suggested the whole A-advantage "
        "came from wall bounces; that pattern did not survive the sweep, and the "
        "bounce and no-bounce curves here cross over. Treat the A/B difference "
        "as small and unstable, and the absence of a widening gap as the real "
        "finding.</p>",
        pio.to_html(viz_rollout.rollout_examples_figure(), include_plotlyjs=False,
                    full_html=False, config={"displayModeBar": False}),
        '<p class="note">Caveats. This is one toy, one architecture, one '
        "dynamics; the effect is a fraction of a pixel. We also cannot separate "
        "<i>reconstruction teaches the model about the scene</i> from the duller "
        "<i>a second loss regularises the encoder</i>; an auxiliary objective "
        "with nothing to do with pixels might do the same thing. And a 10-step "
        "horizon on near-linear dynamics is a gentle test; a longer horizon or "
        "richer dynamics could still separate them.</p>",
    ]


def _card(value, label, sub, color):
    return (f'<div class="card"><div class="v" style="color:{color}">{value}</div>'
            f'<div class="l">{label}</div><div class="s">{sub}</div></div>')


def make_html(path):
    """Write the whole narrative into ONE self-contained HTML file.

    Same figures and same story as the live Dash app, but static: the epoch
    slider and the colour dropdown are Plotly-native controls, so the page needs
    no server and no network (plotly.js is embedded in the first figure).
    """
    b, a = H["B_task"], H["A_recon"]
    ratio = b["px"] / a["px"]

    parts = [
        "<html><head><meta charset='utf-8'><title>Hermes toy demo</title>"
        f"<style>{CSS}</style></head><body>",

        "<h1>Does a world model need to reconstruct what it sees?</h1>",
        '<p class="note lede">Two identical encoders squeeze the same images into 8 '
        f'numbers. <b style="color:{COL_A}">Encoder A</b> is trained to rebuild the '
        f'whole image. <b style="color:{COL_B}">Encoder B</b> is trained only to '
        'predict where the object is — it never learns to draw anything. '
        'The question: does skipping reconstruction cost us?</p>',
        f'<div class="answer"><b>Answer: no — it helps.</b> The task-only encoder '
        f'reads out object position {1/ratio:.1f}× more precisely, using a fraction '
        f'of its latent to do it.</div>',

        '<div class="cards">'
        + _card(f"{b['r2']:.3f}", "Task-only accuracy (R²)",
                f"vs {a['r2']:.3f} for reconstruction", COL_B)
        + _card(f"{b['px']:.2f} px", "Average position error",
                f"vs {a['px']:.2f} px — on a 16 px image", COL_B)
        + _card(f"{b['top_share']*100:.0f}%", "Task info in ONE dimension",
                f"vs {a['top_share']*100:.0f}% — B is far more concentrated", COL_B)
        + "</div>",

        "<h2>1. The task-only encoder is more accurate</h2>",
        '<p class="note">Both latents get the same fair test: a linear read-out of '
        "the object's position, trained on half the data and scored on the other "
        "half. Closer to the dashed line is better.</p>",
        pio.to_html(readout_figure(), include_plotlyjs=True, full_html=False,
                    config={"displayModeBar": False}),

        "<h2>2. …and it stores the task far more cleanly</h2>",
        '<p class="note">We switch off each of the 8 latent dimensions one at a time '
        'and measure how much the position prediction breaks. The task-only encoder '
        'concentrates almost everything into a single dimension; the autoencoder has '
        'to smear it across many, tangled up with the background it also has to '
        'remember.</p>',
        pio.to_html(concentration_figure(), include_plotlyjs=False, full_html=False,
                    config={"displayModeBar": False}),

        "<h2>3. See it in the latent space</h2>",
        "<p class=\"note\">Each dot is one image, floating in the encoder's latent "
        'space. The horizontal axis is the direction that encodes position. '
        '<b>Colour by position</b> and the task-only cloud becomes a clean '
        'left-to-right gradient. <b>Switch to background</b> and that gradient '
        'disappears — but the background still structures both clouds along the '
        'vertical axis. Drag to rotate; use the dropdown to switch colour.</p>',
        pio.to_html(dropdown_3d_figure(), include_plotlyjs=False, full_html=False),

        "<h2>4. The honest catch</h2>",
        '<p class="note">Training on the task alone does not delete the distractor. '
        'The background is still linearly readable from both latents — the task-only '
        "encoder simply isn't forced to rely on it. So the win is a cleaner, more "
        'concentrated task representation, not distractor invariance.</p>',
        pio.to_html(tradeoff_figure(), include_plotlyjs=False, full_html=False,
                    config={"displayModeBar": False}),
        "<p class=\"note\">Where each encoder's capacity goes. <b>A</b> spends its "
        "8 numbers faithfully repainting the background — detail the task never "
        "needs. <b>B</b> has no decoder and cannot draw anything at all; the only "
        "thing it outputs is a position, so the bottom row is that predicted "
        "position drawn as a dot on an empty canvas. It is a picture of what B "
        "kept, not a reconstruction by B — and it is the more accurate of the "
        "two.</p>",
        pio.to_html(recon_figure(), include_plotlyjs=False, full_html=False,
                    config={"displayModeBar": False}),

        "<h2>5. Watch it happen during training</h2>",
        "<p class=\"note\">Press play. Both encoders start from the same weights, "
        f"so at epoch 0 they are identical and equally wrong. Watch how fast they "
        f"separate: <b style=\"color:{COL_B}\">B</b> snaps onto the diagonal within "
        f"a few epochs, while <b style=\"color:{COL_A}\">A</b> is still catching up "
        "and never gets as tight.</p>",
        pio.to_html(animate(readout_scatter_figure), include_plotlyjs=False,
                    full_html=False),
        '<p class="note">The same story on real held-out images: the coloured '
        'markers walk in toward the true position as each encoder learns to find '
        'the object. Individual images are noisy, so the population figures are '
        'printed above them.</p>',
        pio.to_html(animate(tracker_figure), include_plotlyjs=False, full_html=False),
        '<p class="note">And the latent activations themselves — each row is a band '
        'of object positions, each column a latent dimension. Averaging within a '
        'band cancels the (independent) background, so a column that becomes a '
        'smooth gradient is one that tracks position.</p>',
        pio.to_html(animate(heatmap_figure), include_plotlyjs=False, full_html=False),

        *rollout_html(),

        "<h2>Appendix — the raw PCA view</h2>",
        '<p class="note">For completeness. PCA picks the directions of greatest '
        'variance, and here that is the background — so both clouds look like '
        'confetti and the result is hidden. That is exactly why section 3 uses '
        'task-aligned axes instead.</p>',
        pio.to_html(pca_figure("Position (the task)"), include_plotlyjs=False,
                    full_html=False, config={"displayModeBar": False}),

        "<footer>A minimal illustrative toy — 16×16 synthetic images, 8-dim latents, "
        "trained on CPU in under a minute. Every number on this page comes from one "
        "real training run. Not a claim about Hermes's actual results.</footer>",
        "</body></html>",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"wrote {path}  ({os.path.getsize(path)/1e6:.1f} MB, self-contained)")


if __name__ == "__main__":
    make_gif(os.path.join(RESULTS, "position_readout.gif"))
    make_html(os.path.join(RESULTS, "demo.html"))
