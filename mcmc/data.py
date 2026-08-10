"""
data.py — Target (unnormalised) probability densities for the MCMC visualiser.

Metropolis-Hastings only ever needs the *ratio* of densities at two points,
so every distribution here is defined up to an unknown normalising constant
— exactly the situation MCMC was invented for (in real Bayesian problems
that constant is usually an intractable high-dimensional integral).

Public API
----------
log_target(name, xy)      — vectorised log unnormalised density, xy shape (..., 2)
grid_density(name, n)     — (xs, ys, Z) for contour plots, Z peak-normalised to 1
marginal_x(name, n)       — (xs, px) true marginal of the first coordinate, found by
                             numerically integrating out the second (no closed form
                             exists for the banana or ring targets)

TARGET_KEYS / TARGET_NAMES — registry, kept in sync with app.py
TARGET_DEFAULTS            — per-target plot extent, starting point, default proposal σ
"""

from __future__ import annotations

import numpy as np

TARGET_KEYS = ["gaussian", "banana", "bimodal", "ring"]

TARGET_NAMES = [
    "Correlated Gaussian",
    "Banana",
    "Bimodal (two modes)",
    "Ring",
]

# extent:        half-width of the square plotting/grid window
# default_sigma: a sensible starting proposal std for the sidebar slider
# init:          starting point deliberately placed away from the mass, so
#                the early "burn-in" phase is visible on the chain's path
TARGET_DEFAULTS: dict[str, dict] = {
    "gaussian": {"extent": 6.0, "default_sigma": 1.0, "init": (4.5, -4.5)},
    "banana":   {"extent": 8.0, "default_sigma": 0.8, "init": (5.5, 6.0)},
    "bimodal":  {"extent": 7.0, "default_sigma": 1.0, "init": (0.0, 5.5)},
    "ring":     {"extent": 6.0, "default_sigma": 0.6, "init": (0.0, 0.0)},
}


# ---------------------------------------------------------------------------
# Unnormalised log-densities — each takes xy of shape (..., 2)
# ---------------------------------------------------------------------------

def _log_gaussian(xy: np.ndarray, rho: float = 0.85, sx: float = 1.8, sy: float = 1.0) -> np.ndarray:
    """Correlated 2-D Gaussian — the 'easy' baseline target.

    A random-walk proposal with roughly matched, isotropic variance mixes
    well here; it is the control case the other three targets are meant
    to contrast with.
    """
    x, y = xy[..., 0], xy[..., 1]
    zx, zy = x / sx, y / sy
    return -(zx ** 2 - 2 * rho * zx * zy + zy ** 2) / (2 * (1 - rho ** 2))


def _log_banana(xy: np.ndarray, b: float = 0.6, s1: float = 1.6) -> np.ndarray:
    """Rosenbrock-style 'banana' — a warped Gaussian with strong curvature.

    No single fixed, isotropic proposal covariance matches this shape
    everywhere along its curve, which is exactly why it is a classic
    torture test for random-walk Metropolis-Hastings.
    """
    x1, x2 = xy[..., 0], xy[..., 1]
    return -(x1 ** 2) / (2 * s1 ** 2) - (x2 - b * (x1 ** 2 - s1 ** 2)) ** 2 / 2


def _log_bimodal(xy: np.ndarray, sep: float = 3.2, sigma: float = 0.9, w: float = 0.5) -> np.ndarray:
    """Two well-separated Gaussian modes — the textbook 'chain gets stuck' target.

    With a small proposal σ the chain can spend thousands of steps trapped
    in a single mode because crossing the low-density valley between the
    two peaks requires an improbable run of accepted uphill... then downhill
    proposals.
    """
    x, y = xy[..., 0], xy[..., 1]
    d1 = ((x + sep) ** 2 + y ** 2) / (2 * sigma ** 2)
    d2 = ((x - sep) ** 2 + y ** 2) / (2 * sigma ** 2)
    a, c = -d1 + np.log(w), -d2 + np.log(1 - w)
    m = np.maximum(a, c)
    return m + np.log(np.exp(a - m) + np.exp(c - m))


def _log_ring(xy: np.ndarray, radius: float = 3.0, sigma: float = 0.45) -> np.ndarray:
    """A thin ring of probability mass — non-convex support, no single mode.

    The chain must curve around the centre rather than cut through it,
    so an overly large isotropic step size mostly proposes points inside
    the "hole" or outside the ring and gets rejected.
    """
    x, y = xy[..., 0], xy[..., 1]
    r = np.sqrt(x ** 2 + y ** 2)
    return -((r - radius) ** 2) / (2 * sigma ** 2)


_LOG_FUNCS = {
    "gaussian": _log_gaussian,
    "banana": _log_banana,
    "bimodal": _log_bimodal,
    "ring": _log_ring,
}


def log_target(name: str, xy: np.ndarray) -> np.ndarray:
    """Vectorised log p̃(xy) for xy of shape (..., 2). Returns shape (...)."""
    if name not in _LOG_FUNCS:
        raise ValueError(f"Unknown target {name!r}. Choose from: {TARGET_KEYS}")
    return _LOG_FUNCS[name](np.asarray(xy, dtype=float))


def grid_density(name: str, n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the target on an n×n grid for contour plotting.

    Returns xs, ys (each length n) and Z of shape (n, n), laid out the way
    Plotly's go.Contour expects: Z[j, i] corresponds to (xs[i], ys[j]).
    Z is scaled so its peak is 1 — fine for an unnormalised contour, and
    keeps the colour scale identical across every target.
    """
    extent = TARGET_DEFAULTS[name]["extent"]
    xs = np.linspace(-extent, extent, n)
    ys = np.linspace(-extent, extent, n)
    Xg, Yg = np.meshgrid(xs, ys)  # shape (n, n); Xg[j, i] = xs[i], Yg[j, i] = ys[j]
    logZ = log_target(name, np.stack([Xg, Yg], axis=-1))
    Z = np.exp(logZ - logZ.max())
    return xs, ys, Z


def marginal_x(name: str, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """True marginal density of the first coordinate.

    Found by numerically integrating the unnormalised joint over the second
    coordinate on a fine grid, then renormalising the result to integrate
    to 1. This is the curve the growing sample histogram should converge to
    — for the banana and ring targets there is no closed-form marginal.
    """
    extent = TARGET_DEFAULTS[name]["extent"]
    xs = np.linspace(-extent, extent, n)
    ys = np.linspace(-extent, extent, n)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")  # Xg[i, j] = xs[i], Yg[i, j] = ys[j]
    logZ = log_target(name, np.stack([Xg, Yg], axis=-1))
    Z = np.exp(logZ - logZ.max())
    px = Z.sum(axis=1)
    # Manual trapezoidal rule (uniform spacing) — avoids the np.trapz/np.trapezoid
    # split across numpy versions.
    dx = xs[1] - xs[0]
    area = dx * (px.sum() - 0.5 * (px[0] + px[-1]))
    px = px / area
    return xs, px
