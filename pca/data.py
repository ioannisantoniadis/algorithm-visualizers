"""
data.py — 2-D and 3-D dataset generators for the PCA visualiser

Labels colour points only; PCA ignores them.
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


def _grid23_centres(
    n_clusters: int, spacing: float, rng: np.random.Generator, dim: int,
) -> np.ndarray:
    if dim == 2:
        cols = int(np.ceil(np.sqrt(n_clusters)))
        rows = int(np.ceil(n_clusters / cols))
        gx, gy = np.meshgrid(
            np.arange(cols) * spacing, np.arange(rows) * spacing,
        )
        c = np.column_stack([gx.ravel(), gy.ravel()])[:n_clusters]
    else:
        side = int(np.ceil(n_clusters ** (1 / 3)))
        gx, gy, gz = np.meshgrid(
            np.arange(side) * spacing,
            np.arange(side) * spacing,
            np.arange(side) * spacing,
        )
        c = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])[:n_clusters]
    c = c.astype(float)
    c += rng.uniform(-0.15 * spacing, 0.15 * spacing, size=c.shape)
    return c


# ---------------------------------------------------------------------------
# 2-D generators
# ---------------------------------------------------------------------------

def make_blobs(
    n_points: int = 200,
    n_clusters: int = 4,
    std: float = 0.55,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    spacing = max(4.0 * std * 3, 4.0)
    centres = _grid23_centres(n_clusters, spacing, rng, 2)
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
    rng = np.random.default_rng(seed)
    # One shared elongation axis for *every* blob, and a tight grid spacing
    # relative to that elongation. This is what makes the sidebar's claim
    # ("PC1 follows the longest axis of each ellipsoid") literally true: all
    # ellipsoids share the same axis, and within-cluster variance along it
    # dominates the between-cluster (grid-position) variance. Drawing an
    # independent random angle per cluster (the old behaviour) and spacing
    # blobs far apart made the global PC1 barely edge out PC2 (~54-60%
    # across seeds) — indistinguishable from plain isotropic blobs and a
    # poor "best demo".
    spacing = 2.0
    centres = _grid23_centres(n_clusters, spacing, rng, 2)
    sizes = np.full(n_clusters, n_points // n_clusters, dtype=int)
    sizes[: n_points % n_clusters] += 1
    angle = rng.uniform(0, np.pi)
    s1, s2 = rng.uniform(1.5, 2.5), rng.uniform(0.3, 0.6)
    D = np.diag([s1 ** 2, s2 ** 2])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    cov = R @ D @ R.T
    X_parts, label_parts = [], []
    for k, (centre, size) in enumerate(zip(centres, sizes)):
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
    rng = np.random.default_rng(seed)
    spacing = 10.0
    centres = _grid23_centres(n_clusters, spacing, rng, 2)
    stds = np.exp(rng.uniform(np.log(0.4), np.log(2.5), size=n_clusters))
    raw_sizes = rng.uniform(0.5, 3.0, size=n_clusters)
    raw_sizes /= raw_sizes.sum()
    sizes = (raw_sizes * n_points).astype(int)
    sizes[-1] += n_points - sizes.sum()
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
    rng = np.random.default_rng(seed)
    n_each = n_points // 2
    remainder = n_points - 2 * n_each
    t0 = np.linspace(0, np.pi, n_each)
    moon0 = np.column_stack([np.cos(t0), np.sin(t0)])
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
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_points, 2))
    labels = np.full(n_points, -1, dtype=int)
    return _normalise(X), labels


# ---------------------------------------------------------------------------
# 3-D generators
# ---------------------------------------------------------------------------

def make_blobs_3d(
    n_points: int = 200,
    n_clusters: int = 4,
    std: float = 0.55,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    spacing = max(4.0 * std * 3, 4.0)
    centres = _grid23_centres(n_clusters, spacing, rng, 3)
    sizes = np.full(n_clusters, n_points // n_clusters, dtype=int)
    sizes[: n_points % n_clusters] += 1
    X_parts, label_parts = [], []
    for k, (centre, size) in enumerate(zip(centres, sizes)):
        cov = np.diag(rng.uniform(0.8, 1.2, size=3) * std ** 2)
        pts = rng.multivariate_normal(centre, cov, size=size)
        X_parts.append(pts)
        label_parts.append(np.full(size, k, dtype=int))
    X = np.vstack(X_parts)
    labels = np.concatenate(label_parts)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def _random_spd(rng: np.random.Generator, scales: np.ndarray) -> np.ndarray:
    """Random SPD with given diagonal spectrum in its eigenbasis."""
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    return Q @ np.diag(scales ** 2) @ Q.T


def make_anisotropic_3d(
    n_points: int = 200,
    n_clusters: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Shared elongation axes across every blob + tight grid spacing — see
    # make_anisotropic() for why (same fix, same reasoning, 2-D vs 3-D).
    spacing = 2.0
    centres = _grid23_centres(n_clusters, spacing, rng, 3)
    sizes = np.full(n_clusters, n_points // n_clusters, dtype=int)
    sizes[: n_points % n_clusters] += 1
    # Strong elongation: one large, two small axes
    s = np.array([
        rng.uniform(2.0, 3.0),
        rng.uniform(0.35, 0.65),
        rng.uniform(0.25, 0.55),
    ])
    cov = _random_spd(rng, s)
    X_parts, label_parts = [], []
    for k, (centre, size) in enumerate(zip(centres, sizes)):
        pts = rng.multivariate_normal(centre, cov, size=size)
        X_parts.append(pts)
        label_parts.append(np.full(size, k, dtype=int))
    X = np.vstack(X_parts)
    labels = np.concatenate(label_parts)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_varied_3d(
    n_points: int = 200,
    n_clusters: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    spacing = 10.0
    centres = _grid23_centres(n_clusters, spacing, rng, 3)
    stds = np.exp(rng.uniform(np.log(0.4), np.log(2.5), size=n_clusters))
    raw_sizes = rng.uniform(0.5, 3.0, size=n_clusters)
    raw_sizes /= raw_sizes.sum()
    sizes = (raw_sizes * n_points).astype(int)
    sizes[-1] += n_points - sizes.sum()
    X_parts, label_parts = [], []
    for k, (centre, size, std) in enumerate(zip(centres, sizes, stds)):
        cov = np.eye(3) * std ** 2
        pts = rng.multivariate_normal(centre, cov, size=size)
        X_parts.append(pts)
        label_parts.append(np.full(size, k, dtype=int))
    X = np.vstack(X_parts)
    labels = np.concatenate(label_parts)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_moons_3d(
    n_points: int = 200,
    noise: float = 0.08,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xy, labels = make_moons(n_points=n_points, noise=noise, seed=seed)
    # Break coplanarity: lift with a slanted plane + noise so PCA must use 3-D
    z = 0.45 * xy[:, 0] + 0.35 * xy[:, 1] + rng.normal(0, 0.12, size=len(xy))
    X = np.column_stack([xy, z])
    return _normalise(X), labels


def make_circles_3d(
    n_points: int = 200,
    noise: float = 0.05,
    factor: float = 0.45,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xy, labels = make_circles(n_points=n_points, noise=noise, factor=factor, seed=seed)
    z = rng.normal(0, 0.2, size=len(xy))
    X = np.column_stack([xy, z])
    return _normalise(X), labels


def make_uniform_3d(
    n_points: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_points, 3))
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
    dim: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    if dim == 2:
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
    else:
        if shape == "blobs":
            return make_blobs_3d(n_points=n_points, n_clusters=n_clusters, seed=seed)
        if shape == "anisotropic":
            return make_anisotropic_3d(n_points=n_points, n_clusters=n_clusters, seed=seed)
        if shape == "varied":
            return make_varied_3d(n_points=n_points, n_clusters=n_clusters, seed=seed)
        if shape == "moons":
            return make_moons_3d(n_points=n_points, seed=seed)
        if shape == "circles":
            return make_circles_3d(n_points=n_points, seed=seed)
        if shape == "uniform":
            return make_uniform_3d(n_points=n_points, seed=seed)
    raise ValueError(f"Unknown shape {shape!r}. Choose from: {SHAPE_KEYS}")
