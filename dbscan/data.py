"""
data.py — 2-D dataset generators for the DBSCAN visualiser

Six shapes chosen to highlight what DBSCAN can and cannot do:
  blobs       — trivial for DBSCAN (and K-means); good baseline
  anisotropic — elongated ellipses; DBSCAN handles these well
  varied      — different-density blobs; requires careful ε tuning
  moons       — non-convex crescents; DBSCAN's showcase strength
  circles     — concentric rings; another DBSCAN win, K-means fails
  uniform     — no structure; DBSCAN labels everything as noise

All generators return (X, true_labels) where:
  X           — float64 array shape (n_points, 2), normalised to [-4, 4]
  true_labels — int array shape (n_points,); -1 = no natural label
"""

from __future__ import annotations

import numpy as np

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

# DBSCAN hint: suggested (ε, minPts) defaults per shape, tuned to look good
SHAPE_DEFAULTS: dict[str, dict[str, float]] = {
    "blobs":       {"eps": 0.5, "min_samples": 5},
    "anisotropic": {"eps": 0.6, "min_samples": 5},
    "varied":      {"eps": 0.6, "min_samples": 4},
    "moons":       {"eps": 0.65, "min_samples": 5},
    "circles":     {"eps": 0.85, "min_samples": 5},
    "uniform":     {"eps": 0.4, "min_samples": 5},
}


def _normalise(X: np.ndarray, scale: float = 4.0) -> np.ndarray:
    centre = (X.max(axis=0) + X.min(axis=0)) / 2.0
    X = X - centre
    span = np.abs(X).max()
    if span > 0:
        X = X / span * scale
    return X


def _shuffle(X: np.ndarray, labels: np.ndarray, rng: np.random.Generator):
    perm = rng.permutation(len(X))
    return X[perm], labels[perm]


def make_blobs(n_points: int = 200, n_clusters: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    std = 0.55
    cols = int(np.ceil(np.sqrt(n_clusters)))
    rows = int(np.ceil(n_clusters / cols))
    spacing = max(4.0 * std * 3, 4.0)
    grid_x, grid_y = np.meshgrid(np.arange(cols) * spacing, np.arange(rows) * spacing)
    centres = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:n_clusters]
    centres += rng.uniform(-0.2 * spacing, 0.2 * spacing, size=centres.shape)
    sizes = np.full(n_clusters, n_points // n_clusters, dtype=int)
    sizes[: n_points % n_clusters] += 1
    X_parts, label_parts = [], []
    for k, (c, s) in enumerate(zip(centres, sizes)):
        cov = np.diag(rng.uniform(0.8, 1.2, size=2) * std ** 2)
        X_parts.append(rng.multivariate_normal(c, cov, size=s))
        label_parts.append(np.full(s, k, dtype=int))
    X, labels = np.vstack(X_parts), np.concatenate(label_parts)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_anisotropic(n_points: int = 200, n_clusters: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    spacing = 7.0
    cols = int(np.ceil(np.sqrt(n_clusters)))
    rows = int(np.ceil(n_clusters / cols))
    grid_x, grid_y = np.meshgrid(np.arange(cols) * spacing, np.arange(rows) * spacing)
    centres = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:n_clusters]
    centres += rng.uniform(-0.15 * spacing, 0.15 * spacing, size=centres.shape)
    sizes = np.full(n_clusters, n_points // n_clusters, dtype=int)
    sizes[: n_points % n_clusters] += 1
    X_parts, label_parts = [], []
    for k, (c, s) in enumerate(zip(centres, sizes)):
        angle = rng.uniform(0, np.pi)
        s1, s2 = rng.uniform(1.5, 2.5), rng.uniform(0.3, 0.6)
        D = np.diag([s1 ** 2, s2 ** 2])
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        X_parts.append(rng.multivariate_normal(c, R @ D @ R.T, size=s))
        label_parts.append(np.full(s, k, dtype=int))
    X, labels = np.vstack(X_parts), np.concatenate(label_parts)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_varied(n_points: int = 200, n_clusters: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    spacing = 10.0
    cols = int(np.ceil(np.sqrt(n_clusters)))
    rows = int(np.ceil(n_clusters / cols))
    grid_x, grid_y = np.meshgrid(np.arange(cols) * spacing, np.arange(rows) * spacing)
    centres = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:n_clusters]
    centres += rng.uniform(-0.1 * spacing, 0.1 * spacing, size=centres.shape)
    stds = np.exp(rng.uniform(np.log(0.4), np.log(2.5), size=n_clusters))
    raw_sizes = rng.uniform(0.5, 3.0, size=n_clusters)
    raw_sizes /= raw_sizes.sum()
    sizes = (raw_sizes * n_points).astype(int)
    sizes[-1] += n_points - sizes.sum()
    X_parts, label_parts = [], []
    for k, (c, s, std) in enumerate(zip(centres, sizes, stds)):
        X_parts.append(rng.multivariate_normal(c, np.eye(2) * std ** 2, size=s))
        label_parts.append(np.full(s, k, dtype=int))
    X, labels = np.vstack(X_parts), np.concatenate(label_parts)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_moons(n_points: int = 200, noise: float = 0.08, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_each = n_points // 2
    remainder = n_points - 2 * n_each
    t0 = np.linspace(0, np.pi, n_each)
    moon0 = np.column_stack([np.cos(t0), np.sin(t0)])
    t1 = np.linspace(0, np.pi, n_each + remainder)
    moon1 = np.column_stack([1 - np.cos(t1), 1 - np.sin(t1) - 0.5])
    X = np.vstack([moon0, moon1])
    labels = np.concatenate([np.zeros(n_each, dtype=int), np.ones(n_each + remainder, dtype=int)])
    X += rng.normal(0, noise, size=X.shape)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_circles(n_points: int = 200, noise: float = 0.05, factor: float = 0.45, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_outer, n_inner = n_points // 2, n_points - n_points // 2
    t_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    t_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    outer = np.column_stack([np.cos(t_outer), np.sin(t_outer)])
    inner = np.column_stack([factor * np.cos(t_inner), factor * np.sin(t_inner)])
    X = np.vstack([outer, inner])
    labels = np.concatenate([np.zeros(n_outer, dtype=int), np.ones(n_inner, dtype=int)])
    X += rng.normal(0, noise, size=X.shape)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_uniform(n_points: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_points, 2))
    labels = np.full(n_points, -1, dtype=int)
    return _normalise(X), labels


def make_dataset(shape: str, n_points: int = 200, n_clusters: int = 4, seed: int = 0):
    """Return (X, true_labels) for the requested shape."""
    dispatch = {
        "blobs":       lambda: make_blobs(n_points, n_clusters, seed),
        "anisotropic": lambda: make_anisotropic(n_points, n_clusters, seed),
        "varied":      lambda: make_varied(n_points, n_clusters, seed),
        "moons":       lambda: make_moons(n_points, seed=seed),
        "circles":     lambda: make_circles(n_points, seed=seed),
        "uniform":     lambda: make_uniform(n_points, seed),
    }
    if shape not in dispatch:
        raise ValueError(f"Unknown shape {shape!r}. Choose from: {SHAPE_KEYS}")
    return dispatch[shape]()
