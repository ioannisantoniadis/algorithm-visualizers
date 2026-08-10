"""
data.py — 2-D binary classification dataset generators for the SVM visualiser

All generators return (X, y) where:
  X — float64 array of shape (n_points, 2), normalised to roughly [-4, 4]
  y — int array of shape (n_points,), values in {-1, +1}

Public API
----------
make_linear_separable(n_points, margin, seed)   — two clearly separated blobs
make_linear_overlap(n_points, overlap, seed)     — same, but classes bleed
                                                    into each other (needs a
                                                    soft margin / slack)
make_moons(n_points, noise, seed)                — interleaved crescents
make_circles(n_points, noise, factor, seed)      — concentric rings
make_xor(n_points, noise, seed)                  — four-quadrant XOR pattern

make_dataset(shape, n_points, seed, **kwargs)    — single entry-point dispatcher

SHAPE_NAMES — ordered list of human-readable display names
SHAPE_KEYS  — corresponding internal keys for make_dataset()
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Registry — keep display name ↔ key in sync with app.py
# ---------------------------------------------------------------------------

SHAPE_KEYS = [
    "linear_separable",
    "linear_overlap",
    "moons",
    "circles",
    "xor",
]

SHAPE_NAMES = [
    "Linearly separable",
    "Overlapping classes (soft margin)",
    "Two moons",
    "Concentric circles",
    "XOR pattern",
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

def make_linear_separable(
    n_points: int = 120,
    gap: float = 2.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two Gaussian blobs with a wide, clean gap between them.

    A hard(ish) margin is achievable — any reasonable C finds a boundary
    that gets every point right.
    """
    rng = np.random.default_rng(seed)
    n0, n1 = n_points // 2, n_points - n_points // 2

    pos = rng.normal(loc=[gap, gap], scale=0.9, size=(n0, 2))
    neg = rng.normal(loc=[-gap, -gap], scale=0.9, size=(n1, 2))

    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n0), -np.ones(n1)])
    X, y = _shuffle(X, y, rng)
    return _normalise(X), y.astype(int)


def make_linear_overlap(
    n_points: int = 120,
    overlap: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two blobs close enough that their tails overlap.

    No hyperplane classifies every point correctly — a soft margin with
    slack variables is required. Great for watching C trade off margin
    width against the number of violations.
    """
    rng = np.random.default_rng(seed)
    n0, n1 = n_points // 2, n_points - n_points // 2

    centre = max(0.6, 1.6 - 0.5 * overlap)
    pos = rng.normal(loc=[centre, centre], scale=1.15, size=(n0, 2))
    neg = rng.normal(loc=[-centre, -centre], scale=1.15, size=(n1, 2))

    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n0), -np.ones(n1)])
    X, y = _shuffle(X, y, rng)
    return _normalise(X), y.astype(int)


def make_moons(
    n_points: int = 120,
    noise: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved crescents — the classic "no linear boundary works" case.

    A linear kernel is stuck near 50% accuracy; RBF wraps around the
    crescents easily.
    """
    rng = np.random.default_rng(seed)
    n0 = n_points // 2
    n1 = n_points - n0

    t0 = np.linspace(0, np.pi, n0)
    moon0 = np.column_stack([np.cos(t0), np.sin(t0)])

    t1 = np.linspace(0, np.pi, n1)
    moon1 = np.column_stack([1 - np.cos(t1), 1 - np.sin(t1) - 0.5])

    X = np.vstack([moon0, moon1])
    y = np.concatenate([np.ones(n0), -np.ones(n1)])

    X += rng.normal(0, noise, size=X.shape)
    X, y = _shuffle(X, y, rng)
    return _normalise(X), y.astype(int)


def make_circles(
    n_points: int = 120,
    noise: float = 0.08,
    factor: float = 0.45,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner circle (class +1) surrounded by an outer ring (class -1).

    Radially symmetric — a textbook demonstration of the RBF kernel, whose
    decision boundary is naturally a closed curve around the origin.
    """
    rng = np.random.default_rng(seed)
    n_out = n_points // 2
    n_in = n_points - n_out

    t_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    outer = np.column_stack([np.cos(t_out), np.sin(t_out)])

    t_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
    inner = np.column_stack([factor * np.cos(t_in), factor * np.sin(t_in)])

    X = np.vstack([outer, inner])
    y = np.concatenate([-np.ones(n_out), np.ones(n_in)])

    X += rng.normal(0, noise, size=X.shape)
    X, y = _shuffle(X, y, rng)
    return _normalise(X), y.astype(int)


def make_xor(
    n_points: int = 120,
    noise: float = 0.35,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Four Gaussian blobs at the corners of a square, labelled like XOR.

    Diagonal corners share a class, so no single straight line separates
    them — but a degree-2 polynomial kernel (which can represent products
    like x1*x2) solves it exactly.
    """
    rng = np.random.default_rng(seed)
    quarter = n_points // 4
    sizes = [quarter] * 3 + [n_points - 3 * quarter]
    centres = [(1.5, 1.5), (-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5)]
    labels = [1, 1, -1, -1]

    X_parts, y_parts = [], []
    for (cx, cy), lbl, size in zip(centres, labels, sizes):
        pts = rng.normal(loc=[cx, cy], scale=noise + 0.5, size=(size, 2))
        X_parts.append(pts)
        y_parts.append(np.full(size, lbl))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    X, y = _shuffle(X, y, rng)
    return _normalise(X), y.astype(int)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def make_dataset(
    shape: str,
    n_points: int = 120,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the requested shape.

    Parameters
    ----------
    shape:      one of SHAPE_KEYS
    n_points:   total number of data points
    seed:       random seed
    """
    if shape == "linear_separable":
        return make_linear_separable(n_points=n_points, seed=seed)
    if shape == "linear_overlap":
        return make_linear_overlap(n_points=n_points, seed=seed)
    if shape == "moons":
        return make_moons(n_points=n_points, seed=seed)
    if shape == "circles":
        return make_circles(n_points=n_points, seed=seed)
    if shape == "xor":
        return make_xor(n_points=n_points, seed=seed)
    raise ValueError(f"Unknown shape {shape!r}. Choose from: {SHAPE_KEYS}")
