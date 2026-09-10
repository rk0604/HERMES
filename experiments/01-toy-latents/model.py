"""
model.py — Minimal numpy MLP building blocks + the two encoders.

Everything is hand-written forward/backward (no autograd framework) so that
every gradient is explicit and explainable. The pieces are deliberately small:

    Linear   : y = x @ W + b
    ReLU     : elementwise max(0, x)
    Sigmoid  : elementwise 1 / (1 + e^-x)   (used only on the AE's output)
    MLP      : an ordered list of the above, with chained forward / backward
    Adam     : the optimiser, kept tiny but standard

The two models under test share the SAME encoder architecture and latent size;
they differ only in what sits on top of the latent and therefore in their loss:

    Autoencoder (Encoder A) : encoder -> decoder, MSE against the full image.
    TaskNet     (Encoder B) : encoder -> head,    MSE against the task feature.
"""

import numpy as np

from data import IMG_DIM

LATENT_DIM = 8   # size of the bottleneck for BOTH encoders (kept identical)


# ----------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------
class Linear:
    """Fully-connected layer:  y = x @ W + b."""

    def __init__(self, n_in, n_out, rng):
        # He initialisation: good default for ReLU-ish nets, keeps activations
        # from vanishing/exploding at the start of training.
        self.W = (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(np.float32)
        self.b = np.zeros(n_out, dtype=np.float32)
        # Gradient buffers, filled in during backward().
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)
        self._x = None  # cache of the forward input, needed for the gradient

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, g_out):
        # g_out is dLoss/dy for this layer's output y (shape: batch x n_out).
        # dLoss/dW = x^T @ g_out ; dLoss/db = sum over batch ; dLoss/dx = g_out @ W^T
        self.gW[...] = self._x.T @ g_out
        self.gb[...] = g_out.sum(axis=0)
        return g_out @ self.W.T

    def params(self):
        return [(self.W, self.gW), (self.b, self.gb)]


class ReLU:
    def forward(self, x):
        self._mask = x > 0.0
        return x * self._mask

    def backward(self, g_out):
        return g_out * self._mask

    def params(self):
        return []


class Sigmoid:
    def forward(self, x):
        self._out = 1.0 / (1.0 + np.exp(-x))
        return self._out

    def backward(self, g_out):
        return g_out * self._out * (1.0 - self._out)

    def params(self):
        return []


# ----------------------------------------------------------------------------
# A small sequential container
# ----------------------------------------------------------------------------
class MLP:
    """Ordered list of layers with chained forward / backward passes."""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, g):
        # Walk the layers in reverse, threading the gradient back to the input.
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def params(self):
        out = []
        for layer in self.layers:
            out.extend(layer.params())
        return out


# ----------------------------------------------------------------------------
# Optimiser
# ----------------------------------------------------------------------------
class Adam:
    """Standard Adam. Operates on a flat list of (param, grad) references.

    The param/grad arrays are updated IN PLACE, so the layers keep pointing at
    the same buffers throughout training.
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, (p, g) in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)  # in-place update


# ----------------------------------------------------------------------------
# The two models under test
# ----------------------------------------------------------------------------
def make_encoder(rng):
    """The bottleneck encoder shared (architecturally) by both models:
        768 -> Dense(64) -> ReLU -> Dense(LATENT_DIM)   [linear latent]
    """
    return MLP([
        Linear(IMG_DIM, 64, rng),
        ReLU(),
        Linear(64, LATENT_DIM, rng),
    ])


class Autoencoder:
    """Encoder A: trained ONLY with reconstruction loss.

    latent -> Dense(64) -> ReLU -> Dense(768) -> Sigmoid   (reconstruct image)
    """

    def __init__(self, rng):
        self.encoder = make_encoder(rng)
        self.decoder = MLP([
            Linear(LATENT_DIM, 64, rng),
            ReLU(),
            Linear(64, IMG_DIM, rng),
            Sigmoid(),  # image pixels live in [0, 1]
        ])

    def encode(self, X):
        return self.encoder.forward(X)

    def forward(self, X):
        z = self.encoder.forward(X)
        recon = self.decoder.forward(z)
        return recon, z

    def loss_and_backward(self, X):
        """MSE reconstruction loss + full backward pass. Returns scalar loss."""
        recon, _ = self.forward(X)
        n = X.shape[0]
        diff = recon - X
        loss = np.mean(diff ** 2)
        # d(MSE)/d(recon) = 2/(n*D) * diff. Push it back decoder -> encoder.
        g = (2.0 / (n * X.shape[1])) * diff
        g = self.decoder.backward(g)
        self.encoder.backward(g)
        return loss

    def params(self):
        return self.encoder.params() + self.decoder.params()


class TaskNet:
    """Encoder B: trained ONLY with a task-prediction head. NO reconstruction.

    latent -> Dense(16) -> ReLU -> Dense(1)   (predict task feature t)
    """

    def __init__(self, rng):
        self.encoder = make_encoder(rng)
        self.head = MLP([
            Linear(LATENT_DIM, 16, rng),
            ReLU(),
            Linear(16, 1, rng),
        ])

    def encode(self, X):
        return self.encoder.forward(X)

    def forward(self, X):
        z = self.encoder.forward(X)
        pred = self.head.forward(z)  # shape (n, 1)
        return pred, z

    def loss_and_backward(self, X, t):
        """MSE(prediction, t) + full backward pass. Returns scalar loss."""
        pred, _ = self.forward(X)
        n = X.shape[0]
        target = t.reshape(-1, 1)
        diff = pred - target
        loss = np.mean(diff ** 2)
        g = (2.0 / n) * diff  # d(MSE)/d(pred)
        g = self.head.backward(g)
        self.encoder.backward(g)
        return loss

    def predict(self, X):
        pred, _ = self.forward(X)
        return pred.reshape(-1)

    def params(self):
        return self.encoder.params() + self.head.params()


# ----------------------------------------------------------------------------
# Weight (de)serialisation — so train.py can hand the EXACT trained encoders to
# analyze.py without any re-training. We just collect every Linear layer's W & b
# into a flat, order-stable dict of numpy arrays and save/load it via np.savez.
# ----------------------------------------------------------------------------
def _linears(model):
    """Return this model's Linear layers in a fixed order (encoder first)."""
    if isinstance(model, Autoencoder):
        mlps = [model.encoder, model.decoder]
    elif isinstance(model, TaskNet):
        mlps = [model.encoder, model.head]
    else:
        raise TypeError(type(model))
    return [l for mlp in mlps for l in mlp.layers if isinstance(l, Linear)]


def state_dict(model):
    d = {}
    for i, lin in enumerate(_linears(model)):
        d[f"W{i}"] = lin.W
        d[f"b{i}"] = lin.b
    return d


def load_state_dict(model, d):
    for i, lin in enumerate(_linears(model)):
        lin.W[...] = d[f"W{i}"]
        lin.b[...] = d[f"b{i}"]
    return model
