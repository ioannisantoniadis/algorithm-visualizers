"""
data.py — 2-D toy dataset generators

All generators return (X, labels) where:
  X      — float64 array of shape (n_points, 2), normalised to roughly [-2.5, 2.5]
  labels — int array of shape (n_points,), a cluster/class id used *only* for
           colouring the plots. The VAE never sees these labels — it is an
           unsupervised model trained on X alone.

Three shapes are offered because they exercise the latent space differently:
  blobs   — several well-separated clusters; a good encoder should place each
            cluster in its own region of latent space
  moons   — a single smooth non-linear manifold; a good encoder should curl
            it into a shape the decoder can invert cleanly
  circles — one manifold nested inside another; the hardest case for a
            2-D bottleneck with no spare capacity to "cheat" with extra
            dimensions

Public API
----------
make_blobs(n_points, n_blobs, spread, seed)
make_moons(n_points, noise, seed)
make_circles(n_points, noise, factor, seed)

make_dataset(shape, n_points, seed) — single entry-point dispatcher

SHAPE_NAMES — ordered list of human-readable display names
SHAPE_KEYS  — corresponding internal keys for make_dataset()
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Registry — keep display name <-> key in sync with app.py
# ---------------------------------------------------------------------------

SHAPE_KEYS = ["blobs", "moons", "circles"]

SHAPE_NAMES = [
    "Gaussian blobs (4 clusters)",
    "Two moons",
    "Concentric circles",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(X: np.ndarray, scale: float = 2.5) -> np.ndarray:
    """Rescale X so that max(abs) ~= scale, centred at the origin.

    Keeping every dataset on a comparable scale means the encoder/decoder
    architecture and learning rate defaults work reasonably well no matter
    which shape is selected.
    """
    centre = (X.max(axis=0) + X.min(axis=0)) / 2.0
    X = X - centre
    span = np.abs(X).max()
    if span > 0:
        X = X / span * scale
    return X


def _shuffle(X: np.ndarray, labels: np.ndarray, rng: np.random.Generator):
    perm = rng.permutation(len(X))
    return X[perm], labels[perm]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def make_blobs(
    n_points: int = 240,
    n_blobs: int = 4,
    spread: float = 0.35,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """n_blobs Gaussian clouds arranged evenly around a circle.

    Well-separated clusters are the easiest case: a smooth (VAE) latent
    space should place each cluster in its own compact region, while a
    plain autoencoder's latent space, having no pressure to stay compact,
    may spread the same clusters arbitrarily far apart.
    """
    rng = np.random.default_rng(seed)
    n_each = n_points // n_blobs
    remainder = n_points - n_each * n_blobs
    sizes = [n_each] * n_blobs
    sizes[0] += remainder

    angles = np.linspace(0, 2 * np.pi, n_blobs, endpoint=False)
    centres = 1.6 * np.column_stack([np.cos(angles), np.sin(angles)])

    X_parts, y_parts = [], []
    for i, (centre, size) in enumerate(zip(centres, sizes)):
        pts = centre + rng.normal(0, spread, size=(size, 2))
        X_parts.append(pts)
        y_parts.append(np.full(size, i, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    X, y = _shuffle(X, y, rng)
    return _normalise(X), y


def make_moons(
    n_points: int = 240,
    noise: float = 0.12,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved crescent shapes — a single smooth non-linear manifold."""
    rng = np.random.default_rng(seed)
    n_each = n_points // 2
    remainder = n_points - 2 * n_each

    t0 = np.linspace(0, np.pi, n_each)
    moon0 = np.column_stack([np.cos(t0), np.sin(t0)])

    t1 = np.linspace(0, np.pi, n_each + remainder)
    moon1 = np.column_stack([1 - np.cos(t1), 1 - np.sin(t1) - 0.5])

    X = np.vstack([moon0, moon1])
    y = np.concatenate([
        np.zeros(n_each, dtype=int),
        np.ones(n_each + remainder, dtype=int),
    ])

    X += rng.normal(0, noise, size=X.shape)
    X, y = _shuffle(X, y, rng)
    return _normalise(X), y


def make_circles(
    n_points: int = 240,
    noise: float = 0.06,
    factor: float = 0.45,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner circle nested inside an outer ring.

    The hardest case for a 2-D latent bottleneck: the decoder has to fold
    two topologically distinct loops into the same plane without letting
    them overlap after reconstruction.
    """
    rng = np.random.default_rng(seed)
    n_outer = n_points // 2
    n_inner = n_points - n_outer

    t_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    outer = np.column_stack([np.cos(t_outer), np.sin(t_outer)])

    t_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    inner = np.column_stack([factor * np.cos(t_inner), factor * np.sin(t_inner)])

    X = np.vstack([outer, inner])
    y = np.concatenate([np.zeros(n_outer, dtype=int), np.ones(n_inner, dtype=int)])

    X += rng.normal(0, noise, size=X.shape)
    X, y = _shuffle(X, y, rng)
    return _normalise(X), y


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def make_dataset(
    shape: str,
    n_points: int = 240,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, labels) for the requested shape.

    Parameters
    ----------
    shape:    one of SHAPE_KEYS
    n_points: total number of data points
    seed:     random seed
    """
    if shape == "blobs":
        return make_blobs(n_points=n_points, seed=seed)
    if shape == "moons":
        return make_moons(n_points=n_points, seed=seed)
    if shape == "circles":
        return make_circles(n_points=n_points, seed=seed)
    raise ValueError(f"Unknown shape {shape!r}. Choose from: {SHAPE_KEYS}")
