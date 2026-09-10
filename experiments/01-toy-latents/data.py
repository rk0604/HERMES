"""
data.py — Synthetic 2D-state image dataset for the Hermes toy demo.

Each sample is a small RGB image containing exactly ONE object (a white
Gaussian blob) drawn on a randomly-colored, noisy background.

The generative factors are split into two groups:

  * task-relevant  : the object's horizontal position `t` in [0, 1].
                     This is the ONLY thing a downstream "world model" needs
                     in order to act. It maps to the pixel column of the blob.

  * distractors    : the background colour (r, g, b) and per-pixel Gaussian
                     noise. These are drawn INDEPENDENTLY of `t`, so they carry
                     zero information about the task, yet they dominate the
                     pixel-level variance of the image.

The whole point of the demo lives in that last sentence: because the background
colour accounts for far more raw pixel variance than the tiny dot, a
reconstruction loss is *pressured* to spend latent capacity encoding it, while
a task-prediction loss has no reason to. We build the data so the encoders can
reveal that asymmetry — we do NOT hard-code any particular outcome.

Everything here is plain numpy so it is fully transparent and dependency-light.
"""

import numpy as np

# ----------------------------------------------------------------------------
# Image / rendering constants. Kept tiny so training runs in seconds on CPU.
# ----------------------------------------------------------------------------
IMG_SIZE = 16          # images are IMG_SIZE x IMG_SIZE
N_CHANNELS = 3         # RGB
IMG_DIM = IMG_SIZE * IMG_SIZE * N_CHANNELS  # flattened input dimension (768)

DOT_ROW = IMG_SIZE // 2   # the blob's vertical position is FIXED (row = centre)
DOT_SIGMA = 1.6           # Gaussian blob radius in pixels
DOT_BRIGHTNESS = 1.0      # peak intensity added by the blob (white dot)
PIXEL_NOISE_STD = 0.05    # std of per-pixel background noise (a distractor)


def _blob_column_profile():
    """Pre-compute the (IMG_SIZE, IMG_SIZE) Gaussian footprint for a blob whose
    centre sits at (DOT_ROW, col). We build it once per column on the fly in
    `render`, but the vertical (row) part never changes, so cache it here.

    Returns a 1-D array `row_weight[y]` = exp(-(y-DOT_ROW)^2 / (2 sigma^2)).
    """
    ys = np.arange(IMG_SIZE)
    row_weight = np.exp(-((ys - DOT_ROW) ** 2) / (2.0 * DOT_SIGMA ** 2))
    return row_weight  # shape (IMG_SIZE,)


_ROW_WEIGHT = _blob_column_profile()  # shape (IMG_SIZE,)


def render(t, bg_color, rng, noise_std=PIXEL_NOISE_STD):
    """Render ONE image from its generative factors.

    Parameters
    ----------
    t         : float in [0, 1]      — task-relevant horizontal position.
    bg_color  : array shape (3,)     — background RGB in [0, 1] (a distractor).
    rng       : np.random.Generator  — source of the per-pixel noise.
    noise_std : float                — per-pixel Gaussian noise std. Defaults to
                the training value; analyze.py raises it to build an
                out-of-distribution "distractor shift" for the robustness test.

    Returns
    -------
    img : float32 array shape (IMG_SIZE, IMG_SIZE, 3) in [0, 1].
    """
    # 1. Fill the whole frame with the (distractor) background colour.
    img = np.ones((IMG_SIZE, IMG_SIZE, N_CHANNELS), dtype=np.float32) * bg_color

    # 2. Add unstructured per-pixel Gaussian noise (also a distractor).
    img += rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)

    # 3. Paint the white blob. Its column encodes the task feature `t`.
    center_col = t * (IMG_SIZE - 1)
    xs = np.arange(IMG_SIZE)
    col_weight = np.exp(-((xs - center_col) ** 2) / (2.0 * DOT_SIGMA ** 2))
    # Outer product of the (fixed) row profile and the (t-dependent) col profile
    # gives a 2-D Gaussian footprint in [0, 1].
    blob = np.outer(_ROW_WEIGHT, col_weight)[:, :, None]  # (H, W, 1)

    # Additively blend a white dot on top, then clip to a valid image range.
    img = img + DOT_BRIGHTNESS * blob
    np.clip(img, 0.0, 1.0, out=img)
    return img


def make_dataset(n, seed=0, noise_std=PIXEL_NOISE_STD):
    """Generate `n` samples. `noise_std` overrides the per-pixel background
    noise (used by the distractor-shift robustness test in analyze.py).

    Returns a dict with:
      X          : (n, IMG_DIM) float32 flattened images, the encoder input.
      images     : (n, H, W, 3) float32, handy for plotting.
      t          : (n,) float32 task feature (x-position in [0, 1]).
      bg         : (n, 3) float32 background RGB (the primary distractors).
      brightness : (n,) float32 mean background luminance (derived distractor).

    We keep BOTH the flattened X (for the MLP) and the image tensor (for viz).
    """
    rng = np.random.default_rng(seed)

    # Task feature: horizontal position, uniform over the full width.
    t = rng.uniform(0.0, 1.0, size=n).astype(np.float32)

    # Distractor: background colour, drawn INDEPENDENTLY of t. This independence
    # is what makes the background carry zero task information.
    bg = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)

    images = np.empty((n, IMG_SIZE, IMG_SIZE, N_CHANNELS), dtype=np.float32)
    for i in range(n):
        images[i] = render(t[i], bg[i], rng, noise_std=noise_std)

    X = images.reshape(n, IMG_DIM).astype(np.float32)
    # Luminance (Rec. 601-ish) as a single scalar summarising the colour
    # distractor — convenient as a 1-D "distractor" target for probing/PCA.
    brightness = (bg @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.float32)

    return {
        "X": X,
        "images": images,
        "t": t,
        "bg": bg,
        "brightness": brightness,
    }


if __name__ == "__main__":
    # Tiny self-check: report how the pixel variance splits between the
    # background distractor and the task-carrying dot. This is the empirical
    # justification for the whole demo, so it is worth printing.
    d = make_dataset(2000, seed=0)
    X = d["X"]
    print(f"dataset X shape          : {X.shape}")
    print(f"task feature t range     : [{d['t'].min():.3f}, {d['t'].max():.3f}]")

    # Variance attributable to background: render each image with the SAME dot
    # but its own bg vs a FIXED bg, and compare total pixel variance.
    total_var = X.var(axis=0).sum()

    # Re-render with background fixed to grey -> only the dot moves.
    rng = np.random.default_rng(1)
    fixed_bg = np.array([0.5, 0.5, 0.5], np.float32)
    dot_only = np.stack([render(t, fixed_bg, rng).reshape(-1) for t in d["t"]])
    dot_var = dot_only.var(axis=0).sum()

    print(f"total pixel variance     : {total_var:.2f}")
    print(f"variance from moving dot  : {dot_var:.2f} "
          f"({100 * dot_var / total_var:.1f}% of total)")
    print("=> background distractor dominates pixel variance, as intended.")
