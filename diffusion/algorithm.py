"""
algorithm.py — From-scratch Denoising Diffusion Probabilistic Model (DDPM)

Three pieces, kept deliberately separate so each can be watched in isolation:

  1. Forward (noising) process — a fixed, hand-written Markov chain that
     dissolves the data distribution into standard Gaussian noise over T
     steps. No learning involved; this is pure probability.

  2. Score network training — a tiny MLP, trained by hand-derived
     backpropagation (no autograd), to predict the noise eps that was added
     to a sample x_t. This is "denoising score matching": learning
     eps_theta(x_t, t) approx eps is mathematically equivalent to learning
     the score (gradient of log-density) of the noised distribution at
     every step.

  3. Reverse (denoising) process — starting from pure noise x_T ~ N(0, I),
     repeatedly ask the trained network "what noise do you see?" and
     subtract a small amount of it, following the DDPM ancestral sampling
     rule. If training worked, this recovers the original data shape.

Forward process (fixed, per step t = 1..T):
    x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * eps_t,   eps_t ~ N(0, I)

Marginal (closed form, used for training — the standard DDPM shortcut that
lets us jump straight from x_0 to a noisy x_t at any t without simulating
every intermediate step):
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,  eps ~ N(0, I)
    alpha_t      = 1 - beta_t
    alpha_bar_t  = prod_{s=1..t} alpha_s

Training objective (the "simple" DDPM loss):
    L = E_{x_0, t, eps} [ || eps - eps_theta(x_t, t) ||^2 ]

Reverse sampling step (t = T .. 1):
    eps_hat = eps_theta(x_t, t)
    mean    = (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_t)
    x_{t-1} = mean + sqrt(beta_t) * z,   z ~ N(0, I)   (z = 0 when t == 1)

Every gradient below was derived by hand with the chain rule; see the
_backward() docstring for the exact derivation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Phase = Literal["forward", "reverse"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NoiseSchedule:
    """Precomputed beta/alpha/alpha_bar arrays, indexed 0..T-1 for t = 1..T."""

    betas: np.ndarray        # (T,)  beta_t
    alphas: np.ndarray       # (T,)  alpha_t = 1 - beta_t
    alpha_bars: np.ndarray   # (T,)  alpha_bar_t = cumprod(alpha)
    kind: str
    T: int

    def snr(self, t: int) -> float:
        """Signal-to-noise ratio alpha_bar_t / (1 - alpha_bar_t) at step t (1-indexed).

        t = 0 is defined as the noise-free signal (SNR = +inf).
        """
        if t <= 0:
            return float("inf")
        ab = float(self.alpha_bars[t - 1])
        return ab / (1.0 - ab) if ab < 1.0 else float("inf")


@dataclass
class Snapshot:
    """State of the point cloud at one diffusion step."""

    points: np.ndarray  # (N, 2) — current positions at this step
    step: int           # t, 0..T   (0 = clean data, T = pure noise)
    phase: Phase
    beta_t: float = 0.0
    alpha_bar_t: float = 1.0
    snr_t: float = float("inf")

    @property
    def title(self) -> str:
        if self.phase == "forward":
            if self.step == 0:
                return "Forward process — original data (t=0)"
            return f"Forward process — step t={self.step} (adding noise)"
        if self.step == 0:
            return "Reverse process — final sample (t=0)"
        return f"Reverse process — step t={self.step} (denoising)"


@dataclass
class TrainingRun:
    """Result of training the score network."""

    loss_history: list[float]           # one entry per epoch
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    layer_sizes: list[int]
    hidden_size: int
    time_embed_dim: int = 3


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def make_schedule(
    T: int,
    kind: str = "linear",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> NoiseSchedule:
    """Build a beta_t schedule.

    linear — betas increase linearly from beta_start to beta_end, the
             original DDPM (Ho et al., 2020) choice.
    cosine — betas derived from a cosine alpha_bar schedule (Nichol &
             Dhariwal, 2021); alpha_bar decays more gently at the start and
             end of the chain, spending relatively more steps in the
             "medium noise" regime that is hardest to denoise.
    """
    if kind == "linear":
        betas = np.linspace(beta_start, beta_end, T)
    elif kind == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=float)
        f = np.cos(((steps / T + s) / (1 + s)) * np.pi / 2) ** 2
        alpha_bars_full = f / f[0]
        betas = np.clip(1.0 - alpha_bars_full[1:] / alpha_bars_full[:-1], 1e-5, 0.999)
    else:
        raise ValueError(f"Unknown schedule kind {kind!r}. Choose 'linear' or 'cosine'.")

    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return NoiseSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars, kind=kind, T=T)


# ---------------------------------------------------------------------------
# 1. Forward (noising) process — fixed Markov chain, no learning
# ---------------------------------------------------------------------------

def forward_process(
    X0: np.ndarray,
    schedule: NoiseSchedule,
    seed: int = 0,
) -> list[Snapshot]:
    """Run the fixed forward Markov chain and record every step.

    Each step samples fresh Gaussian noise and blends it in — this is the
    literal, sequential definition of the process (as opposed to the
    closed-form shortcut used during training), so consecutive frames show
    a continuous, physically plausible "dissolving" trajectory per point.
    """
    rng = np.random.default_rng(seed)
    x = X0.astype(float).copy()

    snapshots = [
        Snapshot(points=x.copy(), step=0, phase="forward", beta_t=0.0, alpha_bar_t=1.0, snr_t=float("inf"))
    ]
    for t in range(1, schedule.T + 1):
        beta_t = float(schedule.betas[t - 1])
        eps = rng.standard_normal(size=x.shape)
        x = np.sqrt(1.0 - beta_t) * x + np.sqrt(beta_t) * eps
        snapshots.append(
            Snapshot(
                points=x.copy(),
                step=t,
                phase="forward",
                beta_t=beta_t,
                alpha_bar_t=float(schedule.alpha_bars[t - 1]),
                snr_t=schedule.snr(t),
            )
        )
    return snapshots


# ---------------------------------------------------------------------------
# 2. Score network — tiny MLP, hand-derived backprop
# ---------------------------------------------------------------------------

def _time_embed(t: np.ndarray, T: int) -> np.ndarray:
    """Map integer timestep(s) to a small fixed (non-learned) embedding.

    [t/T, sin(pi * t/T), cos(pi * t/T)] gives the network a monotonic
    signal (t/T) plus two smooth periodic-ish features that make it easier
    for a tiny MLP to interpolate its behaviour between far-apart t values
    than a bare scalar would.
    """
    frac = t.astype(float) / T
    return np.column_stack([frac, np.sin(np.pi * frac), np.cos(np.pi * frac)])


def _init_params(layer_sizes: list[int], rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Xavier/Glorot-uniform initialisation — keeps tanh activations from
    saturating at the start of training regardless of layer width."""
    weights, biases = [], []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        limit = np.sqrt(6.0 / (n_in + n_out))
        weights.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
        biases.append(np.zeros(n_out, dtype=float))
    return weights, biases


def _forward_mlp(
    net_in: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]
) -> list[np.ndarray]:
    """Push net_in through every layer: tanh on hidden layers, linear output
    (the predicted noise eps_hat is unbounded, so no squashing on output).

    Returns the list of activations (activations[0] = net_in, activations[-1]
    = eps_hat); intermediate values needed by the backward pass are the
    activations themselves (tanh's derivative is 1 - a^2).
    """
    activations = [net_in]
    a = net_in
    n_layers = len(weights)
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = a @ W + b
        a = z if i == n_layers - 1 else np.tanh(z)
        activations.append(a)
    return activations


def _backward_mlp(
    target: np.ndarray,
    activations: list[np.ndarray],
    weights: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Backpropagate the MSE loss L = mean((eps_hat - eps)^2) through every layer.

    The output layer is linear, so its local gradient is simply
    dL/dz_out = 2*(eps_hat - eps) / eps.size (mean over every element, both
    the batch axis and the 2 output dimensions). Every earlier layer's local
    gradient is the upstream gradient projected back through W^T, scaled by
    the tanh derivative (1 - a^2) evaluated at that layer's own activation —
    exactly the "gradient flowing backward" story of backprop.
    """
    n_layers = len(weights)
    grad_w: list[np.ndarray | None] = [None] * n_layers
    grad_b: list[np.ndarray | None] = [None] * n_layers

    pred = activations[-1]
    dz = 2.0 * (pred - target) / target.size
    for i in reversed(range(n_layers)):
        a_prev = activations[i]
        grad_w[i] = a_prev.T @ dz
        grad_b[i] = dz.sum(axis=0)
        if i > 0:
            da_prev = dz @ weights[i].T
            dz = da_prev * (1.0 - activations[i] ** 2)  # d(tanh)/dz using the activation

    return grad_w, grad_b  # type: ignore[return-value]


def predict_noise(
    x: np.ndarray, t: np.ndarray | int, weights: list[np.ndarray], biases: list[np.ndarray], T: int
) -> np.ndarray:
    """Run the trained score network on x at timestep(s) t. Shared by training
    (loss evaluation) and sampling (the reverse process) so both call the
    exact same forward pass."""
    t_arr = np.full(len(x), t, dtype=float) if np.ndim(t) == 0 else np.asarray(t, dtype=float)
    temb = _time_embed(t_arr, T)
    net_in = np.concatenate([x, temb], axis=1)
    return _forward_mlp(net_in, weights, biases)[-1]


def train_score_network(
    X0: np.ndarray,
    schedule: NoiseSchedule,
    hidden_size: int = 64,
    lr: float = 0.05,
    epochs: int = 300,
    seed: int = 0,
) -> TrainingRun:
    """Train eps_theta(x_t, t) to predict the noise added to x_0.

    Every epoch draws a fresh random timestep t and fresh Gaussian noise eps
    *per data point* (not one shared t for the whole batch), and jumps
    straight to x_t via the closed-form marginal. This is standard DDPM
    training: the network only ever needs to solve the *local* denoising
    problem at some particular (x_t, t); the chain rule across steps is
    handled entirely by the sampling procedure, not by training.

    Parameters
    ----------
    X0:          training data, shape (N, 2)
    schedule:    NoiseSchedule defining alpha_bar_t for every t
    hidden_size: width of the single hidden layer
    lr:          full-batch gradient-descent learning rate
    epochs:      number of full-batch gradient steps
    seed:        random seed for weight init and the training noise/timesteps

    Returns
    -------
    TrainingRun with the loss curve and the final trained weights/biases.
    """
    rng = np.random.default_rng(seed)
    N, D = X0.shape
    T = schedule.T
    time_embed_dim = 3
    layer_sizes = [D + time_embed_dim, hidden_size, D]
    weights, biases = _init_params(layer_sizes, rng)
    alpha_bars = schedule.alpha_bars

    loss_history: list[float] = []
    for _ in range(epochs):
        t_idx = rng.integers(1, T + 1, size=N)          # random t per point
        eps = rng.standard_normal(size=(N, D))            # random noise per point
        ab = alpha_bars[t_idx - 1][:, None]                # (N, 1)
        x_t = np.sqrt(ab) * X0 + np.sqrt(1.0 - ab) * eps

        temb = _time_embed(t_idx.astype(float), T)
        net_in = np.concatenate([x_t, temb], axis=1)
        activations = _forward_mlp(net_in, weights, biases)
        pred = activations[-1]

        loss = float(np.mean((pred - eps) ** 2))
        loss_history.append(loss)

        grad_w, grad_b = _backward_mlp(eps, activations, weights)
        weights = [w - lr * gw for w, gw in zip(weights, grad_w)]
        biases = [b - lr * gb for b, gb in zip(biases, grad_b)]

    return TrainingRun(
        loss_history=loss_history,
        weights=weights,
        biases=biases,
        layer_sizes=layer_sizes,
        hidden_size=hidden_size,
        time_embed_dim=time_embed_dim,
    )


# ---------------------------------------------------------------------------
# 3. Reverse (denoising / sampling) process
# ---------------------------------------------------------------------------

def reverse_process(
    run: TrainingRun,
    schedule: NoiseSchedule,
    n_points: int,
    seed: int = 0,
    x_T: np.ndarray | None = None,
) -> list[Snapshot]:
    """Sample from the trained model by ancestral sampling: start at pure
    noise and repeatedly denoise one step using the trained network's noise
    prediction, following the DDPM reverse update rule.

    Parameters
    ----------
    run:      TrainingRun holding the trained weights/biases
    schedule: same NoiseSchedule used for training (must match — alpha_bar_t
              at each t is what the network was trained to expect)
    n_points: number of independent samples to generate
    seed:     random seed for the initial noise draw and the per-step noise
    x_T:      optional fixed starting noise (n_points, 2); if omitted, drawn
              fresh from N(0, I) — the standard "sample from the prior" start

    Returns
    -------
    Snapshots ordered from t=T (pure noise) down to t=0 (generated sample),
    so playing them forward animates noise condensing into structure.
    """
    rng = np.random.default_rng(seed)
    T = schedule.T
    D = 2
    x = x_T.copy() if x_T is not None else rng.standard_normal(size=(n_points, D))

    snapshots = [
        Snapshot(
            points=x.copy(), step=T, phase="reverse",
            beta_t=float(schedule.betas[-1]), alpha_bar_t=float(schedule.alpha_bars[-1]),
            snr_t=schedule.snr(T),
        )
    ]

    for t in range(T, 0, -1):
        beta_t = float(schedule.betas[t - 1])
        alpha_t = float(schedule.alphas[t - 1])
        alpha_bar_t = float(schedule.alpha_bars[t - 1])

        eps_hat = predict_noise(x, t, run.weights, run.biases, T)
        mean = (x - (beta_t / np.sqrt(1.0 - alpha_bar_t)) * eps_hat) / np.sqrt(alpha_t)

        if t > 1:
            z = rng.standard_normal(size=(n_points, D))
            x = mean + np.sqrt(beta_t) * z
        else:
            x = mean

        snapshots.append(
            Snapshot(
                points=x.copy(), step=t - 1, phase="reverse",
                beta_t=beta_t, alpha_bar_t=schedule.alpha_bars[t - 2] if t > 1 else 1.0,
                snr_t=schedule.snr(t - 1),
            )
        )

    return snapshots
