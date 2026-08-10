"""
data.py — 2-D dataset generators for the K-means visualiser

All generators return (X, true_labels) where:
  X           — float64 array of shape (n_points, 2), normalised to roughly [-4, 4]
  true_labels — int array of shape (n_points,), values in [0, n_groups)
                (-1 means "no natural label", e.g. uniform noise)

Public API
----------
make_blobs(n_points, n_clusters, seed)        — well-separated spherical Gaussians
make_anisotropic(n_points, n_clusters, seed)  — elongated/rotated ellipses
make_varied(n_points, n_clusters, seed)       — blobs with very different radii
make_moons(n_points, noise, seed)             — two interleaved crescents
make_circles(n_points, noise, factor, seed)   — inner and outer circle
make_uniform(n_points, seed)                  — 2-D uniform random noise

make_dataset(shape, n_points, n_clusters, seed) — single entry-point dispatcher

SHAPE_NAMES — ordered list of human-readable display names
SHAPE_KEYS  — corresponding internal keys for make_dataset()
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Registry — keep display name ↔ key in sync with app.py
# ---------------------------------------------------------------------------

SHAPE_KEYS = [
    "blobs",
    "anisotropic",
    "varied",
    "moons",
    "circles",
    "uniform",
]

SHAPE_NAMES = [
    "Gaussian blobs",
    "Anisotropic blobs",
    "Varied density",
    "Two moons",
    "Concentric rings",
    "Uniform noise",
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


def _shuffle(X: np.ndarray, labels: np.ndarray, rng: np.random.Generator):
    perm = rng.permutation(len(X))
    return X[perm], labels[perm]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def make_blobs(
    n_points: int = 200,
    n_clusters: int = 4,
    std: float = 0.55,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Well-separated spherical Gaussian blobs.

    Cluster centres placed on a jittered regular grid so blobs never overlap.
    K-means solves this trivially with any reasonable initialisation.
    """
    rng = np.random.default_rng(seed)

    cols = int(np.ceil(np.sqrt(n_clusters)))
    rows = int(np.ceil(n_clusters / cols))
    spacing = max(4.0 * std * 3, 4.0)

    grid_x, grid_y = np.meshgrid(
        np.arange(cols) * spacing,
        np.arange(rows) * spacing,
    )
    centres = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:n_clusters]
    centres += rng.uniform(-0.2 * spacing, 0.2 * spacing, size=centres.shape)

    sizes = np.full(n_clusters, n_points // n_clusters, dtype=int)
    sizes[: n_points % n_clusters] += 1

    X_parts, label_parts = [], []
    for k, (centre, size) in enumerate(zip(centres, sizes)):
        cov = np.diag(rng.uniform(0.8, 1.2, size=2) * std ** 2)
        pts = rng.multivariate_normal(centre, cov, size=size)
        X_parts.append(pts)
        label_parts.append(np.full(size, k, dtype=int))

    X = np.vstack(X_parts)
    labels = np.concatenate(label_parts)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_anisotropic(
    n_points: int = 200,
    n_clusters: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Elongated, randomly-rotated elliptical blobs.

    Each blob has a random rotation angle and aspect ratio 4:1, making
    K-means often mis-classify points along the elongated axis when the
    initialisation is poor.
    """
    rng = np.random.default_rng(seed)

    cols = int(np.ceil(np.sqrt(n_clusters)))
    rows = int(np.ceil(n_clusters / cols))
    spacing = 7.0

    grid_x, grid_y = np.meshgrid(
        np.arange(cols) * spacing,
        np.arange(rows) * spacing,
    )
    centres = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:n_clusters]
    centres += rng.uniform(-0.15 * spacing, 0.15 * spacing, size=centres.shape)

    sizes = np.full(n_clusters, n_points // n_clusters, dtype=int)
    sizes[: n_points % n_clusters] += 1

    X_parts, label_parts = [], []
    for k, (centre, size) in enumerate(zip(centres, sizes)):
        angle = rng.uniform(0, np.pi)
        # Elongated covariance in local axes, then rotate
        s1, s2 = rng.uniform(1.5, 2.5), rng.uniform(0.3, 0.6)
        D = np.diag([s1 ** 2, s2 ** 2])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        cov = R @ D @ R.T
        pts = rng.multivariate_normal(centre, cov, size=size)
        X_parts.append(pts)
        label_parts.append(np.full(size, k, dtype=int))

    X = np.vstack(X_parts)
    labels = np.concatenate(label_parts)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_varied(
    n_points: int = 200,
    n_clusters: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Blobs with very different radii and point counts.

    The largest blob is ~5× wider than the smallest; its sheer size causes
    K-means to assign neighbouring small-cluster points to it, demonstrating
    the algorithm's bias toward equal-volume Voronoi cells.
    """
    rng = np.random.default_rng(seed)

    cols = int(np.ceil(np.sqrt(n_clusters)))
    rows = int(np.ceil(n_clusters / cols))
    spacing = 10.0

    grid_x, grid_y = np.meshgrid(
        np.arange(cols) * spacing,
        np.arange(rows) * spacing,
    )
    centres = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:n_clusters]
    centres += rng.uniform(-0.1 * spacing, 0.1 * spacing, size=centres.shape)

    # Exponentially varied standard deviations so the difference is stark
    stds = np.exp(rng.uniform(np.log(0.4), np.log(2.5), size=n_clusters))

    # Also vary point counts (one blob is much larger)
    raw_sizes = rng.uniform(0.5, 3.0, size=n_clusters)
    raw_sizes /= raw_sizes.sum()
    sizes = (raw_sizes * n_points).astype(int)
    sizes[-1] += n_points - sizes.sum()  # fix rounding

    X_parts, label_parts = [], []
    for k, (centre, size, std) in enumerate(zip(centres, sizes, stds)):
        cov = np.eye(2) * std ** 2
        pts = rng.multivariate_normal(centre, cov, size=size)
        X_parts.append(pts)
        label_parts.append(np.full(size, k, dtype=int))

    X = np.vstack(X_parts)
    labels = np.concatenate(label_parts)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_moons(
    n_points: int = 200,
    noise: float = 0.08,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved crescent (moon) shapes.

    K-means cannot correctly separate these with K=2 because the clusters
    are non-convex — it always produces a vertical split instead.
    Labels are 0 (top moon) and 1 (bottom moon).
    """
    rng = np.random.default_rng(seed)
    n_each = n_points // 2
    remainder = n_points - 2 * n_each

    # Top moon: upper semicircle
    t0 = np.linspace(0, np.pi, n_each)
    moon0 = np.column_stack([np.cos(t0), np.sin(t0)])

    # Bottom moon: lower semicircle, offset so they interleave
    t1 = np.linspace(0, np.pi, n_each + remainder)
    moon1 = np.column_stack([1 - np.cos(t1), 1 - np.sin(t1) - 0.5])

    X = np.vstack([moon0, moon1])
    labels = np.concatenate([np.zeros(n_each, dtype=int),
                              np.ones(n_each + remainder, dtype=int)])

    X += rng.normal(0, noise, size=X.shape)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_circles(
    n_points: int = 200,
    noise: float = 0.05,
    factor: float = 0.45,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner circle surrounded by an outer circle.

    K-means with K=2 will bisect the plane vertically/horizontally instead
    of recovering the two rings, because K-means decision boundaries are
    always linear (hyperplanes).
    Labels are 0 (outer) and 1 (inner).
    """
    rng = np.random.default_rng(seed)
    n_outer = n_points // 2
    n_inner = n_points - n_outer

    t_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    outer = np.column_stack([np.cos(t_outer), np.sin(t_outer)])

    t_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    inner = np.column_stack([factor * np.cos(t_inner), factor * np.sin(t_inner)])

    X = np.vstack([outer, inner])
    labels = np.concatenate([np.zeros(n_outer, dtype=int),
                              np.ones(n_inner, dtype=int)])

    X += rng.normal(0, noise, size=X.shape)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_uniform(
    n_points: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Points drawn uniformly at random from the unit square.

    There are no natural clusters. K-means will still converge to K cells
    that tile the space in a roughly Voronoi pattern, showing that the
    algorithm always produces *some* result regardless of whether the data
    has cluster structure.
    All labels are -1 to signal that there is no ground truth.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_points, 2))
    labels = np.full(n_points, -1, dtype=int)
    return _normalise(X), labels


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def make_dataset(
    shape: str,
    n_points: int = 200,
    n_clusters: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, labels) for the requested shape.

    Parameters
    ----------
    shape:      one of SHAPE_KEYS
    n_points:   total number of data points
    n_clusters: number of Gaussian components (ignored for moons/circles/uniform)
    seed:       random seed
    """
    if shape == "blobs":
        return make_blobs(n_points=n_points, n_clusters=n_clusters, seed=seed)
    if shape == "anisotropic":
        return make_anisotropic(n_points=n_points, n_clusters=n_clusters, seed=seed)
    if shape == "varied":
        return make_varied(n_points=n_points, n_clusters=n_clusters, seed=seed)
    if shape == "moons":
        return make_moons(n_points=n_points, seed=seed)
    if shape == "circles":
        return make_circles(n_points=n_points, seed=seed)
    if shape == "uniform":
        return make_uniform(n_points=n_points, seed=seed)
    raise ValueError(f"Unknown shape {shape!r}. Choose from: {SHAPE_KEYS}")
