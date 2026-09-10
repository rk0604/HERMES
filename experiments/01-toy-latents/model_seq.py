"""
model_seq.py — Encoder + GRU world model for the rollout experiment (PyTorch).

This is an ADDITION to the numpy demo, not a replacement. The static experiment
in model.py / train.py is untouched and still runs on hand-written numpy
gradients. We switch to torch only here, because this experiment needs
backpropagation through time and hand-writing GRU gradients would be a large
pile of error-prone code for no scientific gain.

The architecture deliberately mirrors the static demo so the comparison carries
over. Both models get:

    encoder :  768 -> Dense(64) -> ReLU -> Dense(8)        [same as make_encoder]
    gru     :  GRUCell(latent=8 -> hidden=32)
    head    :  Dense(hidden -> 1)                          [reads out position]

and Encoder A additionally gets

    decoder :  8 -> Dense(64) -> ReLU -> Dense(768) -> Sigmoid   [same as the AE]

The ONLY difference between the two models is the loss:

    Model A ("recon + task") : task loss + LAMBDA_RECON * reconstruction loss.
                               Both gradients flow into the shared encoder.
    Model B ("task-only")    : task loss only. No decoder exists anywhere in
                               this model, so nothing can reconstruct pixels.

Both therefore receive the task gradient — this is a fair fight, and the
question it asks is the one Hermes actually cares about: does the extra
reconstruction pressure buy you stability under multi-step imagination?

Rollout convention (this is the crux of the experiment):
  * CONTEXT frames are encoded and fed to the GRU one at a time (teacher
    forcing — always real frames, never the model's own output).
  * Then the GRU keeps stepping with a ZERO input vector for HORIZON steps,
    reading a position out of its hidden state at each step. No new pixels
    reach the model during this phase, so it must carry the dynamics in its
    hidden state alone. Training and evaluation use the identical procedure —
    if they differed, the horizon curve would measure train/test mismatch
    rather than the quality of the representation.
"""

import numpy as np
import torch
import torch.nn as nn

from data import IMG_DIM
from data_seq import CONTEXT, HORIZON

LATENT_DIM = 8      # identical to the static demo's bottleneck
HIDDEN_DIM = 32     # GRU hidden state
LAMBDA_RECON = 1.0  # weight on A's reconstruction term (both losses are MSEs
                    # of comparable scale here; train_rollout.py logs each term
                    # separately so you can see the reconstruction really trained)


class RolloutModel(nn.Module):
    """Encoder + GRU + position head, optionally with a decoder.

    `use_recon=True` builds Model A (reconstruction + task).
    `use_recon=False` builds Model B (task only, no decoder at all).

    Submodules are constructed in a fixed order — encoder, gru, head, then the
    optional decoder — so that seeding the RNG identically before building each
    model gives A and B byte-identical encoder/gru/head initial weights. The
    decoder is built last precisely so that drawing its weights cannot perturb
    the shared ones. This mirrors how train.py hands both static encoders a
    fresh `default_rng(SEED + 10)`.
    """

    def __init__(self, use_recon: bool):
        super().__init__()
        self.use_recon = use_recon

        self.encoder = nn.Sequential(
            nn.Linear(IMG_DIM, 64), nn.ReLU(), nn.Linear(64, LATENT_DIM),
        )
        # nn.GRU (the fused sequence version) rather than nn.GRUCell in a Python
        # loop: mathematically the same single-layer GRU, but the whole sequence
        # runs in one fused call instead of T dispatches, which is ~10x faster
        # here. The zero-input rollout is expressed the same way — by feeding a
        # sequence of zero vectors — so it stays fused too.
        self.gru = nn.GRU(LATENT_DIM, HIDDEN_DIM, batch_first=True)
        self.head = nn.Linear(HIDDEN_DIM, 1)

        self.decoder = None
        if use_recon:
            self.decoder = nn.Sequential(
                nn.Linear(LATENT_DIM, 64), nn.ReLU(),
                nn.Linear(64, IMG_DIM), nn.Sigmoid(),
            )

    # -- pieces -------------------------------------------------------------
    def encode_seq(self, X):
        """(B, T, IMG_DIM) -> (B, T, LATENT_DIM). Frames are encoded
        independently; all temporal structure lives in the GRU."""
        B, T, D = X.shape
        return self.encoder(X.reshape(B * T, D)).reshape(B, T, LATENT_DIM)

    def run_context(self, z, context=CONTEXT):
        """Feed the first `context` real latents through the GRU.

        Returns the hidden state after the last context frame. This is the only
        thing that survives into the rollout — the model's entire knowledge of
        where the dot is and where it is going has to be in here.
        """
        out, _ = self.gru(z[:, :context])   # (B, context, HIDDEN_DIM)
        return out[:, -1]

    def rollout(self, h, horizon=HORIZON):
        """Step the GRU forward `horizon` times with ZERO input.

        No frames are available here, so the input is a sequence of zero vectors
        and the GRU must evolve its own hidden state. We read a position out of
        every step, giving predictions for horizons 1..horizon.

        h : (B, HIDDEN_DIM) starting hidden state.
        Returns (B, horizon).
        """
        zeros = torch.zeros(h.shape[0], horizon, LATENT_DIM,
                            device=h.device, dtype=h.dtype)
        # nn.GRU wants the initial hidden state as (num_layers, B, HIDDEN_DIM).
        out, _ = self.gru(zeros, h.unsqueeze(0).contiguous())
        return self.head(out).squeeze(-1)

    # -- convenience --------------------------------------------------------
    def predict_from_frames(self, X, context=CONTEXT, horizon=HORIZON):
        """Full pipeline used at evaluation time: real frames in, blind
        multi-step position predictions out."""
        z = self.encode_seq(X[:, :context])
        h = self.run_context(z, context=context)
        return self.rollout(h, horizon=horizon)

    def losses(self, X, t_true, anchors, horizon=HORIZON):
        """Training losses for one batch.

        X       : (B, T, IMG_DIM) real frames.
        t_true  : (B, T) true positions.
        anchors : list of timesteps to predict from. For each anchor `a`, the
                  GRU has consumed real frames 0..a and must then predict
                  positions a+1 .. a+horizon with no further input.

        Using several anchors per trajectory (rather than only the end of the
        context window) gives more supervision per sample and checks that the
        model works from any point in the sequence, not just one privileged one.

        Returns (task_loss, recon_loss). recon_loss is a zero tensor for B.
        """
        B, T, D = X.shape
        z = self.encode_seq(X)

        # One teacher-forced pass over the whole sequence (real frames at every
        # step), giving the hidden state after every frame so each anchor can
        # branch off from it.
        hs, _ = self.gru(z)                                    # (B, T, HIDDEN)

        # Roll out from every anchor AT ONCE by folding the anchor axis into the
        # batch: A anchors x B trajectories becomes one batch of A*B. This is
        # arithmetically identical to looping over anchors (every anchor group
        # has the same size, so a mean of means equals the overall mean) but
        # issues `horizon` fused GRU calls instead of A*horizon.
        n_a = len(anchors)
        h_anchor = torch.stack([hs[:, a] for a in anchors], dim=0)   # (A, B, H)
        preds = self.rollout(h_anchor.reshape(n_a * B, HIDDEN_DIM),
                             horizon=horizon).reshape(n_a, B, horizon)
        target = torch.stack([t_true[:, a + 1: a + 1 + horizon] for a in anchors],
                             dim=0)                                  # (A, B, horizon)
        task_loss = ((preds - target) ** 2).mean()

        if self.use_recon:
            recon = self.decoder(z.reshape(B * T, LATENT_DIM))
            recon_loss = ((recon - X.reshape(B * T, D)) ** 2).mean()
        else:
            recon_loss = torch.zeros((), device=X.device, dtype=X.dtype)

        return task_loss, recon_loss


def build_pair(seed):
    """Construct Model A and Model B with IDENTICAL initial shared weights.

    Each model is built after re-seeding torch with the same value, so the
    encoder, GRU and head draw the same random numbers in both. Verified by
    `assert_same_init` below, which train_rollout.py calls before training so a
    silent divergence can never masquerade as a result.
    """
    torch.manual_seed(seed)
    model_a = RolloutModel(use_recon=True)
    torch.manual_seed(seed)
    model_b = RolloutModel(use_recon=False)
    return model_a, model_b


def assert_same_init(model_a, model_b):
    """Fail loudly unless A and B start from the same encoder/GRU/head weights.

    The decoder is A-only and is deliberately excluded.
    """
    a_params = dict(model_a.named_parameters())
    b_params = dict(model_b.named_parameters())
    checked = 0
    for name, pb in b_params.items():
        assert name in a_params, f"{name} missing from model A"
        pa = a_params[name]
        if not torch.equal(pa.detach(), pb.detach()):
            raise AssertionError(f"initial weights differ at {name}")
        checked += 1
    return checked


def count_params(model, shared_only=False):
    """Parameter count. `shared_only=True` excludes A's decoder, which is the
    number to quote when claiming the two encoders are the same size."""
    return sum(p.numel() for n, p in model.named_parameters()
               if not (shared_only and n.startswith("decoder")))


if __name__ == "__main__":
    a, b = build_pair(0)
    n = assert_same_init(a, b)
    print(f"A and B share identical init across {n} tensors  [OK]")
    print(f"shared params (encoder+gru+head) : A={count_params(a, True)} "
          f"B={count_params(b, True)}")
    print(f"total params                     : A={count_params(a)} "
          f"B={count_params(b)}   (difference = A's decoder)")

    X = torch.randn(4, 20, IMG_DIM)
    t = torch.rand(4, 20)
    for name, m in (("A", a), ("B", b)):
        tl, rl = m.losses(X, t, anchors=[4, 5])
        out = m.predict_from_frames(X)
        print(f"{name}: task={tl.item():.4f} recon={rl.item():.4f} "
              f"rollout shape={tuple(out.shape)}")
