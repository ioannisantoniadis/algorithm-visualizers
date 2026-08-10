"""
algorithm.py — From-scratch Variational Autoencoder, trained by hand-derived
backpropagation (no autograd).

Architecture (all fixed-width, tanh, full-batch gradient descent):

    encoder:  x (2) --tanh--> h_e (H) --linear--> mu (2)
                                        --linear--> logvar (2)   [VAE only]
    latent:   z = mu                                              [plain AE]
              z = mu + exp(0.5*logvar) * eps,  eps ~ N(0, I)       [VAE — the
                                                  "reparameterisation trick"]
    decoder:  z (2) --tanh--> h_d (H) --linear--> x_hat (2)

Loss (mean over the batch):
    recon_loss = 0.5 * mean_i ||x_hat_i - x_i||^2        (Gaussian NLL, unit var)
    kl_loss    = -0.5 * mean_i sum_d (1 + logvar - mu^2 - exp(logvar))
    total_loss = recon_loss + beta * kl_loss              [kl_loss = 0 for AE]

The KL term is the whole story of *why* a VAE's latent space looks different
from a plain autoencoder's: it is the closed-form KL divergence between the
encoder's output distribution q(z|x) = N(mu, diag(exp(logvar))) and the
standard normal prior N(0, I). Minimising it pulls every point's latent
distribution toward a shared, overlapping, unit-scale blob — which is
exactly what makes VAE latent space smooth and interpolatable, and exactly
what a plain autoencoder (whose loss has no such term) has no reason to do.

Every gradient below was derived by hand with the chain rule and verified
against a numerical (finite-difference) gradient check before being used
here — see the module docstring in visualize.py for how each piece maps to
the visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Phase = Literal["init", "train"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """State of one model (AE or VAE) after a given number of training epochs."""

    # Data — constant across snapshots of the same run
    points: np.ndarray  # (N, 2)
    labels: np.ndarray  # (N,) cluster/class id, used only for colouring

    epoch: int
    phase: Phase
    variational: bool

    mu: np.ndarray               # (N, 2) latent means
    logvar: np.ndarray | None    # (N, 2) latent log-variances; None for plain AE
    z: np.ndarray                # (N, 2) latent codes actually fed to the decoder
    reconstruction: np.ndarray   # (N, 2) decoder output for each point

    recon_loss: float
    kl_loss: float
    total_loss: float

    # Decoder parameters at this instant — lets the app decode *any* (z1, z2)
    # live, e.g. for the "click a point in latent space" interactive demo.
    decoder_params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

    # Decoder output evaluated on a fixed grid of latent points, shape
    # (res, res, 2). Visualises the manifold the decoder currently represents
    # in data space — the 2-D analogue of "what does this model generate".
    decoded_grid: np.ndarray

    @property
    def title(self) -> str:
        kind = "VAE" if self.variational else "Plain AE"
        if self.phase == "init":
            return f"{kind} — initialisation (epoch 0)"
        return f"{kind} — epoch {self.epoch}"


@dataclass
class TrainingRun:
    """Full result of fit(): every epoch's snapshot plus shared latent grid."""

    snapshots: list[Snapshot]
    grid_zx: np.ndarray  # (res, res) latent-space meshgrid, x1 coordinate
    grid_zy: np.ndarray  # (res, res) latent-space meshgrid, x2 coordinate
    variational: bool
    beta: float
    hidden_size: int


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _init_weight(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Xavier/Glorot-uniform initialisation — keeps tanh activations from
    saturating at the start of training regardless of layer width."""
    limit = np.sqrt(6.0 / sum(shape))
    return rng.uniform(-limit, limit, size=shape)


def _make_latent_grid(res: int, extent: float = 3.0):
    """Fixed grid over latent space, matching the +-3 sigma range of the
    standard normal prior the VAE is regularised toward."""
    lin = np.linspace(-extent, extent, res)
    zx, zy = np.meshgrid(lin, lin)
    grid_points = np.column_stack([zx.ravel(), zy.ravel()])
    return zx, zy, grid_points


def decode(z: np.ndarray, decoder_params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Run latent points through the decoder only. z: (n, 2) -> (n, 2).

    Used both internally (to fill in decoded_grid on every snapshot) and by
    the app for the interactive "pick a point in latent space, see what it
    decodes to" demo — the same function, the same weights, just called on
    whatever (z1, z2) the user chooses.
    """
    W1d, b1d, W2d, b2d = decoder_params
    h_d = np.tanh(z @ W1d + b1d)
    return h_d @ W2d + b2d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit(
    X: np.ndarray,
    labels: np.ndarray,
    variational: bool = True,
    beta: float = 1.0,
    hidden_size: int = 16,
    lr: float = 0.1,
    epochs: int = 150,
    seed: int = 0,
    grid_res: int = 14,
) -> TrainingRun:
    """Train a 2 -> hidden_size -> 2 (latent) -> hidden_size -> 2 VAE (or
    plain autoencoder) with full-batch gradient descent, recording every
    epoch as a Snapshot.

    Parameters
    ----------
    X, labels:   data (N, 2) and colouring labels (N,); labels are never
                 used by the model itself
    variational: if True, sample z via the reparameterisation trick and add
                 the KL term to the loss (a VAE); if False, z = mu exactly
                 and there is no KL term (a plain autoencoder)
    beta:        weight on the KL term (ignored if variational=False).
                 beta=0 recovers a plain autoencoder even with variational=True,
                 which is a good way to see the KL term's effect in isolation
    hidden_size: width of the single hidden layer in the encoder and decoder
    lr:          gradient-descent learning rate
    epochs:      number of full-batch gradient steps
    seed:        random seed for weight init and (if variational) the noise
                 used in the reparameterisation trick
    grid_res:    resolution of the fixed latent grid whose decoded image is
                 stored on every snapshot (the "manifold" overlay)

    Returns
    -------
    TrainingRun with one Snapshot per epoch (plus the initial, untrained one).
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    H, L = hidden_size, 2  # latent dim fixed at 2 for visualisability

    W1e, b1e = _init_weight((D, H), rng), np.zeros(H)
    Wmu, bmu = _init_weight((H, L), rng), np.zeros(L)
    Wlv, blv = _init_weight((H, L), rng), np.zeros(L)
    W1d, b1d = _init_weight((L, H), rng), np.zeros(H)
    W2d, b2d = _init_weight((H, D), rng), np.zeros(D)

    zx, zy, grid_points = _make_latent_grid(grid_res)

    def _forward(eps: np.ndarray | None):
        h_e = np.tanh(X @ W1e + b1e)
        mu = h_e @ Wmu + bmu
        if variational:
            logvar = h_e @ Wlv + blv
            std = np.exp(0.5 * logvar)
            z = mu + std * eps
        else:
            logvar = None
            std = None
            z = mu
        h_d = np.tanh(z @ W1d + b1d)
        x_hat = h_d @ W2d + b2d

        recon_loss = 0.5 * float(np.mean(np.sum((x_hat - X) ** 2, axis=1)))
        if variational:
            kl_loss = -0.5 * float(np.mean(np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1)))
        else:
            kl_loss = 0.0
        total_loss = recon_loss + beta * kl_loss

        cache = dict(h_e=h_e, mu=mu, logvar=logvar, std=std, z=z, h_d=h_d, x_hat=x_hat, eps=eps)
        return recon_loss, kl_loss, total_loss, cache

    def _snapshot(epoch: int, phase: Phase, recon_loss, kl_loss, total_loss, cache) -> Snapshot:
        dec_params = (W1d.copy(), b1d.copy(), W2d.copy(), b2d.copy())
        grid_dec = decode(grid_points, dec_params).reshape(grid_res, grid_res, D)
        return Snapshot(
            points=X,
            labels=labels,
            epoch=epoch,
            phase=phase,
            variational=variational,
            mu=cache["mu"].copy(),
            logvar=cache["logvar"].copy() if variational else None,
            z=cache["z"].copy(),
            reconstruction=cache["x_hat"].copy(),
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            total_loss=total_loss,
            decoder_params=dec_params,
            decoded_grid=grid_dec,
        )

    # Epoch 0: random weights, no training yet
    eps0 = rng.standard_normal(size=(N, L)) if variational else None
    recon0, kl0, total0, cache0 = _forward(eps0)
    snapshots = [_snapshot(0, "init", recon0, kl0, total0, cache0)]

    for epoch in range(1, epochs + 1):
        # Forward + backward with the weights as they stand entering this
        # epoch (i.e. the result of `epoch - 1` previous updates).
        eps = rng.standard_normal(size=(N, L)) if variational else None
        _, _, _, cache = _forward(eps)
        h_e, mu, logvar, std, z, h_d, x_hat = (
            cache["h_e"], cache["mu"], cache["logvar"], cache["std"],
            cache["z"], cache["h_d"], cache["x_hat"],
        )

        # --- backward pass (hand-derived, verified against finite differences) ---
        dx_hat = (x_hat - X) / N
        grad_W2d = h_d.T @ dx_hat
        grad_b2d = dx_hat.sum(axis=0)
        dh_d = dx_hat @ W2d.T
        dz2 = dh_d * (1 - h_d**2)  # d(tanh)/dz using the activation itself
        grad_W1d = z.T @ dz2
        grad_b1d = dz2.sum(axis=0)
        dz = dz2 @ W1d.T  # gradient flowing back into the latent code

        if variational:
            # z = mu + std * eps  =>  dz/dmu = 1,  dz/dlogvar = eps * std * 0.5
            dmu = dz + beta * (mu / N)
            dlogvar = dz * eps * std * 0.5 + beta * ((np.exp(logvar) - 1) / (2 * N))
        else:
            dmu = dz
            dlogvar = None

        grad_Wmu = h_e.T @ dmu
        grad_bmu = dmu.sum(axis=0)
        dh_e = dmu @ Wmu.T

        if variational:
            grad_Wlv = h_e.T @ dlogvar
            grad_blv = dlogvar.sum(axis=0)
            dh_e = dh_e + dlogvar @ Wlv.T
        else:
            grad_Wlv = np.zeros_like(Wlv)
            grad_blv = np.zeros_like(blv)

        dz1 = dh_e * (1 - h_e**2)
        grad_W1e = X.T @ dz1
        grad_b1e = dz1.sum(axis=0)

        # --- gradient descent update ---
        W1e -= lr * grad_W1e; b1e -= lr * grad_b1e
        Wmu -= lr * grad_Wmu; bmu -= lr * grad_bmu
        if variational:
            Wlv -= lr * grad_Wlv; blv -= lr * grad_blv
        W1d -= lr * grad_W1d; b1d -= lr * grad_b1d
        W2d -= lr * grad_W2d; b2d -= lr * grad_b2d

        # Recompute forward with the just-updated weights so the recorded
        # loss, mu/logvar/z, reconstruction and decoder_params all describe
        # the *same* model state (state after `epoch` updates) — otherwise
        # decoder_params (captured post-update) would disagree with the
        # reconstruction shown (which would still be pre-update).
        eps_report = rng.standard_normal(size=(N, L)) if variational else None
        recon_loss, kl_loss, total_loss, report_cache = _forward(eps_report)
        snapshots.append(_snapshot(epoch, "train", recon_loss, kl_loss, total_loss, report_cache))

    return TrainingRun(
        snapshots=snapshots,
        grid_zx=zx,
        grid_zy=zy,
        variational=variational,
        beta=beta,
        hidden_size=hidden_size,
    )
