"""
data.py — 2-D classification dataset generators for the Random Forest visualiser

All generators return (X, y) where:
  X — float64 array of shape (n_points, 2), normalised to roughly [-4, 4]
  y — int array of shape (n_points,), values in [0, n_classes)

The five shapes are chosen to probe different things about an
axis-aligned, tree-based ensemble:

  blobs        — trivial for a single tree; the sanity-check baseline
  checkerboard — a grid of alternating classes literally *is* a union of
                 axis-aligned rectangles, the exact hypothesis class a
                 CART split can represent exactly — the best case for trees
  moons        — a smooth curved boundary no axis-aligned split can match
                 exactly, only approximate with a staircase of small boxes
  circles      — a closed, radially symmetric boundary; the same staircase
                 story as moons, but curving in every direction at once
  spiral       — a genuinely hard 3-class non-linear boundary; a shallow
                 forest underfits badly, a deeper one can trace the arms

Public API
----------
make_blobs(n_points, seed)          — three well-separated Gaussian blobs
make_checkerboard(n_points, seed)   — alternating grid of two classes
make_moons(n_points, noise, seed)   — two interleaved crescents
make_circles(n_points, noise, seed) — inner circle vs. outer ring
make_spiral(n_points, noise, seed)  — three interleaved spiral arms

make_dataset(shape, n_points, seed) — single entry-point dispatcher

SHAPE_NAMES — ordered list of human-readable display names
SHAPE_KEYS  — corresponding internal keys for make_dataset()
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Registry — keep display name ↔ key in sync with app.py
# ---------------------------------------------------------------------------

SHAPE_KEYS = ["blobs", "checkerboard", "moons", "circles", "spiral"]

SHAPE_NAMES = [
    "Three blobs",
    "Checkerboard",
    "Two moons",
    "Concentric circles",
    "Three-arm spiral",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(X: np.ndarray, scale: float = 4.0) -> np.ndarray:
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

def make_blobs(
    n_points: int = 240,
    std: float = 0.65,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated Gaussian blobs (3 classes).

    Any single shallow tree already solves this almost perfectly — a
    baseline showing that ensembling isn't required when the classes
    are trivially separable, only helpful.
    """
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 2.2], [-2.0, -1.4], [2.0, -1.4]])

    n_each = n_points // 3
    sizes = [n_each, n_each, n_points - 2 * n_each]

    X_parts, y_parts = [], []
    for k, (centre, size) in enumerate(zip(centres, sizes)):
        pts = rng.normal(loc=centre, scale=std, size=(size, 2))
        X_parts.append(pts)
        y_parts.append(np.full(size, k, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    X, y = _shuffle(X, y, rng)
    return _normalise(X), y


def make_checkerboard(
    n_points: int = 260,
    cells: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A cells×cells grid of alternating classes (2 classes).

    This boundary literally *is* a union of axis-aligned rectangles — the
    exact hypothesis class a CART split can represent perfectly, so it's
    the friendliest possible shape for a tree-based model and the
    clearest place to watch max_features change how fast the forest
    finds it.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_points, 2))

    cell_size = 2.0 / cells
    ix = np.clip(np.floor((X[:, 0] + 1.0) / cell_size).astype(int), 0, cells - 1)
    iy = np.clip(np.floor((X[:, 1] + 1.0) / cell_size).astype(int), 0, cells - 1)
    y = (ix + iy) % 2

    X, y = _shuffle(X, y, rng)
    return _normalise(X), y


def make_moons(
    n_points: int = 240,
    noise: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved crescent shapes (2 classes).

    No single axis-aligned split matches this boundary exactly — a tree
    approximates the curve with a staircase of small rectangles, getting
    finer (and more overfit to noise) the deeper it's allowed to grow.
    """
    rng = np.random.default_rng(seed)
    n0 = n_points // 2
    n1 = n_points - n0

    t0 = np.linspace(0, np.pi, n0)
    moon0 = np.column_stack([np.cos(t0), np.sin(t0)])

    t1 = np.linspace(0, np.pi, n1)
    moon1 = np.column_stack([1 - np.cos(t1), 1 - np.sin(t1) - 0.5])

    X = np.vstack([moon0, moon1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    X += rng.normal(0, noise, size=X.shape)
    X, y = _shuffle(X, y, rng)
    return _normalise(X), y


def make_circles(
    n_points: int = 240,
    noise: float = 0.08,
    factor: float = 0.45,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner circle (class 1) surrounded by an outer ring (class 0).

    A closed, radially symmetric boundary — the same "staircase of
    rectangles" story as moons, except the tree has to curve around the
    origin in every direction, needing splits on both features at
    multiple depths.
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


def make_spiral(
    n_points: int = 300,
    noise: float = 0.22,
    n_arms: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Three interleaved spiral arms (3 classes).

    A genuinely hard non-linear, multi-class boundary: a shallow tree (or
    a forest with too few estimators) underfits into a handful of blunt
    rectangles, while enough depth and enough trees can trace the arms
    reasonably well. Good for showing where trees start to struggle.
    """
    rng = np.random.default_rng(seed)
    n_each = n_points // n_arms
    remainder = n_points - n_arms * n_each

    X_parts, y_parts = [], []
    for k in range(n_arms):
        size = n_each + (remainder if k == n_arms - 1 else 0)
        t = np.linspace(0.2, 1.0, size)
        angle = 3.0 * np.pi * t + k * (2 * np.pi / n_arms)
        r = 3.2 * t
        pts = np.column_stack([r * np.cos(angle), r * np.sin(angle)])
        pts += rng.normal(0, noise, size=pts.shape)
        X_parts.append(pts)
        y_parts.append(np.full(size, k, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
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
    """Return (X, y) for the requested shape.

    Parameters
    ----------
    shape:      one of SHAPE_KEYS
    n_points:   total number of data points
    seed:       random seed
    """
    if shape == "blobs":
        return make_blobs(n_points=n_points, seed=seed)
    if shape == "checkerboard":
        return make_checkerboard(n_points=n_points, seed=seed)
    if shape == "moons":
        return make_moons(n_points=n_points, seed=seed)
    if shape == "circles":
        return make_circles(n_points=n_points, seed=seed)
    if shape == "spiral":
        return make_spiral(n_points=n_points, seed=seed)
    raise ValueError(f"Unknown shape {shape!r}. Choose from: {SHAPE_KEYS}")
