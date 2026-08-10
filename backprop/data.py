"""
data.py — 2-D binary classification dataset generators

All generators return (X, y) where:
  X — float64 array of shape (n_points, 2), normalised to roughly [-4, 4]
  y — int array of shape (n_points,), values in {0, 1}

The four shapes are chosen specifically because they span the boundary
between "a single perceptron can solve this" and "you need a hidden
layer": XOR is the textbook example a linear model provably cannot
separate, moons/circles are smooth non-linear boundaries, and the
linear set is the sanity-check case where even the tiniest network
converges almost immediately.

Public API
----------
make_xor(n_points, noise, seed)      — four-quadrant XOR pattern
make_moons(n_points, noise, seed)    — two interleaved crescents
make_circles(n_points, noise, seed)  — inner circle vs. outer ring
make_linear(n_points, noise, seed)   — linearly separable half-planes

make_dataset(shape, n_points, noise, seed) — single entry-point dispatcher

SHAPE_NAMES — ordered list of human-readable display names
SHAPE_KEYS  — corresponding internal keys for make_dataset()
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Registry — keep display name ↔ key in sync with app.py
# ---------------------------------------------------------------------------

SHAPE_KEYS = ["xor", "moons", "circles", "linear"]

SHAPE_NAMES = [
    "XOR pattern",
    "Two moons",
    "Concentric circles",
    "Linearly separable",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(X: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """Rescale X so that max(abs) ≈ scale, centred at the origin."""
    centre = (X.max(axis=0) + X.min(axis=0)) / 2.0
    X = X - centre
    span = np.abs(X).max()
    if span > 0:
        X = X / span * scale
    return X


def _shuffle(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def make_xor(
    n_points: int = 200,
    noise: float = 0.35,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Four Gaussian blobs at the corners of a square, labelled like XOR.

    Blobs in the (+,+) and (-,-) quadrants are class 0; (+,-) and (-,+)
    are class 1. No straight line can separate class 0 from class 1 —
    this is the classical proof that a single-layer perceptron cannot
    represent XOR, and the reason hidden layers exist at all.
    """
    rng = np.random.default_rng(seed)
    n_each = n_points // 4
    remainder = n_points - 4 * n_each
    sizes = [n_each] * 4
    sizes[0] += remainder

    corners = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=float)
    labels_per_corner = [0, 0, 1, 1]

    X_parts, y_parts = [], []
    for corner, label, size in zip(corners, labels_per_corner, sizes):
        pts = corner + rng.normal(0, noise, size=(size, 2))
        X_parts.append(pts)
        y_parts.append(np.full(size, label, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    X, y = _shuffle(X, y, rng)
    return _normalise(X), y


def make_moons(
    n_points: int = 200,
    noise: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved crescent (moon) shapes — a smooth non-linear boundary.

    A network with a single hidden neuron cannot bend a line enough to
    separate these; a few hidden units with a non-linear activation can.
    """
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
    n_points: int = 200,
    noise: float = 0.08,
    factor: float = 0.45,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner circle (class 1) surrounded by an outer ring (class 0).

    The decision boundary is a closed curve — impossible for any linear
    model, and a good demonstration that the hidden layer learns a
    radial feature rather than a directional one.
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


def make_linear(
    n_points: int = 200,
    noise: float = 0.35,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two Gaussian clouds separated by a comfortable margin.

    A single linear boundary solves this perfectly, so even the smallest
    network converges in a handful of epochs — a useful contrast to the
    other three shapes.
    """
    rng = np.random.default_rng(seed)
    n0 = n_points // 2
    n1 = n_points - n0

    c0 = rng.normal([-1.1, -0.6], noise, size=(n0, 2))
    c1 = rng.normal([1.1, 0.6], noise, size=(n1, 2))

    X = np.vstack([c0, c1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    X, y = _shuffle(X, y, rng)
    return _normalise(X), y


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def make_dataset(
    shape: str,
    n_points: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the requested shape.

    Parameters
    ----------
    shape:      one of SHAPE_KEYS
    n_points:   total number of data points
    seed:       random seed
    """
    if shape == "xor":
        return make_xor(n_points=n_points, seed=seed)
    if shape == "moons":
        return make_moons(n_points=n_points, seed=seed)
    if shape == "circles":
        return make_circles(n_points=n_points, seed=seed)
    if shape == "linear":
        return make_linear(n_points=n_points, seed=seed)
    raise ValueError(f"Unknown shape {shape!r}. Choose from: {SHAPE_KEYS}")
