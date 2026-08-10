"""
Synthetic high-D data for t-SNE: latent shape → random rotation into ℝᴰ (+ noise).

Uses the same generators as `umap-viz` so you can compare embeddings across tools.

Returns (X, labels, latent) with latent (n,2)/(n,3) or None (uniform); mixed uses NaN rows.
"""

from __future__ import annotations

import numpy as np

DATASET_KEYS = [
    "blobs2",
    "blobs3",
    "moons2",
    "rings2",
    "mixed",
    "uniform",
]

DATASET_NAMES = [
    "Two blobs (latent 2-D)",
    "Three blobs (latent 3-D)",
    "Two moons (latent 2-D)",
    "Concentric rings (latent 2-D)",
    "Mixed blobs + noise",
    "Uniform (control)",
]


def _lift(
    X2: np.ndarray,
    dim: int,
    rng: np.random.Generator,
    noise: float = 0.04,
) -> np.ndarray:
    a = rng.standard_normal((dim, 2))
    q, _ = np.linalg.qr(a)
    x_h = X2 @ q.T
    x_h += rng.normal(0.0, noise, size=x_h.shape)
    return x_h.astype(np.float64)


def _lift3(
    X3: np.ndarray,
    dim: int,
    rng: np.random.Generator,
    noise: float = 0.04,
) -> np.ndarray:
    a = rng.standard_normal((dim, 3))
    q, _ = np.linalg.qr(a)
    x_h = X3 @ q.T
    x_h += rng.normal(0.0, noise, size=x_h.shape)
    return x_h.astype(np.float64)


def _normalise_rows(X: np.ndarray, scale: float = 4.0) -> np.ndarray:
    c = (X.max(axis=0) + X.min(axis=0)) / 2.0
    X = X - c
    span = np.abs(X).max()
    if span > 1e-9:
        X = X / span * scale
    return X


def _normalise_latent(Z: np.ndarray, scale: float = 4.0) -> np.ndarray:
    c = (Z.max(axis=0) + Z.min(axis=0)) / 2.0
    z = Z - c
    span = np.abs(z).max()
    if span > 1e-9:
        z = z / span * scale
    return z.astype(np.float64)


def make_blobs2(
    n_points: int,
    ambient_dim: int,
    seed: int,
    *,
    lift_noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n0 = n_points // 2
    n1 = n_points - n0
    c0 = rng.normal(0, 0.15, size=(n0, 2))
    c1 = np.column_stack([
        rng.normal(2.2, 0.12, size=n1),
        rng.normal(1.0, 0.12, size=n1),
    ])
    X2 = np.vstack([c0, c1])
    labels = np.array([0] * n0 + [1] * n1, dtype=int)
    perm = rng.permutation(len(X2))
    latent = _normalise_latent(X2[perm].copy())
    X = _normalise_rows(_lift(X2[perm], ambient_dim, rng, noise=lift_noise))
    labels = labels[perm]
    return X, labels, latent


def make_blobs3(
    n_points: int,
    ambient_dim: int,
    seed: int,
    *,
    lift_noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n0 = n_points // 3
    n1 = n_points // 3
    n2 = n_points - n0 - n1
    centers = np.asarray(
        [[0.0, 0.0, 0.0], [2.1, 1.0, -0.9], [-1.1, 2.0, 1.1]],
        dtype=np.float64,
    )
    c0 = rng.normal(0, 0.12, size=(n0, 3)) + centers[0]
    c1 = rng.normal(0, 0.12, size=(n1, 3)) + centers[1]
    c2 = rng.normal(0, 0.12, size=(n2, 3)) + centers[2]
    X3 = np.vstack([c0, c1, c2])
    labels = np.array([0] * n0 + [1] * n1 + [2] * n2, dtype=int)
    perm = rng.permutation(len(X3))
    latent = _normalise_latent(X3[perm].copy())
    X = _normalise_rows(_lift3(X3[perm], ambient_dim, rng, noise=lift_noise))
    labels = labels[perm]
    return X, labels, latent


def make_moons2(
    n_points: int,
    ambient_dim: int,
    seed: int,
    *,
    lift_noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_each = n_points // 2
    rem = n_points - 2 * n_each
    t0 = np.linspace(0, np.pi, n_each)
    m0 = np.column_stack([np.cos(t0), np.sin(t0)])
    t1 = np.linspace(0, np.pi, n_each + rem)
    m1 = np.column_stack([1 - np.cos(t1), 1 - np.sin(t1) - 0.5])
    X2 = np.vstack([m0, m1])
    labels = np.array([0] * n_each + [1] * (n_each + rem), dtype=int)
    X2 += rng.normal(0, 0.08, size=X2.shape)
    perm = rng.permutation(len(X2))
    latent = _normalise_latent(X2[perm].copy())
    X = _normalise_rows(_lift(X2[perm], ambient_dim, rng, noise=lift_noise))
    labels = labels[perm]
    return X, labels, latent


def make_rings2(
    n_points: int,
    ambient_dim: int,
    seed: int,
    factor: float = 0.45,
    *,
    lift_noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n0 = n_points // 2
    n1 = n_points - n0
    t0 = np.linspace(0, 2 * np.pi, n0, endpoint=False)
    outer = np.column_stack([np.cos(t0), np.sin(t0)])
    t1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    inner = np.column_stack([factor * np.cos(t1), factor * np.sin(t1)])
    X2 = np.vstack([outer, inner])
    labels = np.array([0] * n0 + [1] * n1, dtype=int)
    X2 += rng.normal(0, 0.05, size=X2.shape)
    perm = rng.permutation(len(X2))
    latent = _normalise_latent(X2[perm].copy())
    X = _normalise_rows(_lift(X2[perm], ambient_dim, rng, noise=lift_noise))
    labels = labels[perm]
    return X, labels, latent


def make_mixed(
    n_points: int,
    ambient_dim: int,
    seed: int,
    *,
    lift_noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_b = int(n_points * 0.7)
    n_u = n_points - n_b
    xb, lb, latent_b = make_blobs2(
        n_b, ambient_dim, seed ^ 12345, lift_noise=lift_noise,
    )
    xu = rng.uniform(-3, 3, size=(n_u, ambient_dim))
    xu = _normalise_rows(xu, scale=4.0)
    labels_u = np.full(n_u, -1, dtype=int)
    latent_u = np.full((n_u, 2), np.nan, dtype=np.float64)
    X = np.vstack([xb, xu])
    labels = np.concatenate([lb, labels_u])
    latent = np.vstack([latent_b, latent_u])
    perm = rng.permutation(len(X))
    return X[perm], labels[perm], latent[perm]


def make_uniform(
    n_points: int,
    ambient_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, None]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_points, ambient_dim))
    labels = np.full(n_points, -1, dtype=int)
    return _normalise_rows(X), labels, None


def make_dataset(
    key: str,
    n_points: int = 200,
    ambient_dim: int = 32,
    seed: int = 0,
    *,
    lift_noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    ln = float(lift_noise)
    if key == "blobs2":
        return make_blobs2(n_points, ambient_dim, seed, lift_noise=ln)
    if key == "blobs3":
        return make_blobs3(n_points, ambient_dim, seed, lift_noise=ln)
    if key == "moons2":
        return make_moons2(n_points, ambient_dim, seed, lift_noise=ln)
    if key == "rings2":
        return make_rings2(n_points, ambient_dim, seed, lift_noise=ln)
    if key == "mixed":
        return make_mixed(n_points, ambient_dim, seed, lift_noise=ln)
    if key == "uniform":
        return make_uniform(n_points, ambient_dim, seed)
    raise ValueError(key)


def has_full_latent(key: str) -> bool:
    return key in ("blobs2", "blobs3", "moons2", "rings2")


def latent_intrinsic_dim(key: str) -> int | None:
    if key == "blobs3":
        return 3
    if key in ("blobs2", "moons2", "rings2"):
        return 2
    if key == "mixed":
        return 2
    return None
