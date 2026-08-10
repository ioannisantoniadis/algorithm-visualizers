"""
data.py — 2-D toy dataset generators for the diffusion visualiser

All generators return (X, labels) where:
  X      — float64 array of shape (n_points, 2), normalised to roughly [-2, 2]
           so its scale is comparable to the standard normal noise the
           forward process eventually turns it into
  labels — int array of shape (n_points,), a cluster/class id used *only*
           for colouring the plots. The diffusion model never sees these
           labels — it is trained on X alone.

Three shapes are offered because they stress the reverse (denoising) process
differently:
  moons — a single smooth non-convex manifold; the classic DDPM toy example
  ring  — a closed 1-D loop; the score network must learn a *radial*
          restoring force pointing back onto the circle from every angle
  blobs — several disconnected clusters; the reverse process must learn to
          commit to one mode per sample rather than blur between them

Public API
----------
make_moons(n_points, noise, seed)
make_ring(n_points, noise, seed)
make_blobs(n_points, n_blobs, spread, seed)

make_dataset(shape, n_points, seed) — single entry-point dispatcher

SHAPE_NAMES — ordered list of human-readable display names
SHAPE_KEYS  — corresponding internal keys for make_dataset()
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Registry — keep display name <-> key in sync with app.py
# ---------------------------------------------------------------------------

SHAPE_KEYS = ["moons", "ring", "blobs"]

SHAPE_NAMES = [
    "Two moons",
    "Ring",
    "Gaussian blobs (3 clusters)",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(X: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Rescale X so that max(abs) ~= scale, centred at the origin.

    Keeping the data on a scale close to the standard normal prior N(0, I)
    means the forward process needs only a modest number of steps T to fully
    dissolve the structure into noise, and the reverse process starts from
    noise of a directly comparable magnitude.
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

def make_moons(
    n_points: int = 240,
    noise: float = 0.08,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved crescent shapes — the classic DDPM toy distribution."""
    rng = np.random.default_rng(seed)
    n_each = n_points // 2
    remainder = n_points - 2 * n_each

    t0 = np.linspace(0, np.pi, n_each)
    moon0 = np.column_stack([np.cos(t0), np.sin(t0)])

    t1 = np.linspace(0, np.pi, n_each + remainder)
    moon1 = np.column_stack([1 - np.cos(t1), 1 - np.sin(t1) - 0.5])

    X = np.vstack([moon0, moon1])
    labels = np.concatenate([
        np.zeros(n_each, dtype=int),
        np.ones(n_each + remainder, dtype=int),
    ])

    X += rng.normal(0, noise, size=X.shape)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_ring(
    n_points: int = 240,
    noise: float = 0.06,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Points scattered around a single circle.

    A single connected 1-D manifold with no natural sub-clusters — the
    reverse process has to learn a purely radial restoring force (push
    every point back onto the circle) rather than deciding *which* mode
    a noisy point belongs to.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n_points)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    X += rng.normal(0, noise, size=X.shape)
    labels = np.zeros(n_points, dtype=int)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_blobs(
    n_points: int = 240,
    n_blobs: int = 3,
    spread: float = 0.25,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """n_blobs well-separated Gaussian clusters arranged around a circle.

    Disconnected modes are the hardest case for a diffusion model: the
    reverse process must commit early to *which* blob a sample is heading
    toward, since once low-noise steps are reached there is no way to jump
    between disconnected regions.
    """
    rng = np.random.default_rng(seed)
    n_each = n_points // n_blobs
    remainder = n_points - n_each * n_blobs
    sizes = [n_each] * n_blobs
    sizes[0] += remainder

    angles = np.linspace(0, 2 * np.pi, n_blobs, endpoint=False)
    centres = 1.3 * np.column_stack([np.cos(angles), np.sin(angles)])

    X_parts, y_parts = [], []
    for i, (centre, size) in enumerate(zip(centres, sizes)):
        pts = centre + rng.normal(0, spread, size=(size, 2))
        X_parts.append(pts)
        y_parts.append(np.full(size, i, dtype=int))

    X = np.vstack(X_parts)
    labels = np.concatenate(y_parts)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


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
    if shape == "moons":
        return make_moons(n_points=n_points, seed=seed)
    if shape == "ring":
        return make_ring(n_points=n_points, seed=seed)
    if shape == "blobs":
        return make_blobs(n_points=n_points, seed=seed)
    raise ValueError(f"Unknown shape {shape!r}. Choose from: {SHAPE_KEYS}")
