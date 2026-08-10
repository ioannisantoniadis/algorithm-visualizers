"""
data.py — 2-D dataset generators and geometric "augmentations"

All generators return (X, labels) where:
  X      — float64 array of shape (n_points, 2), normalised to roughly [-4, 4]
  labels — int array of shape (n_points,), values in [0, n_clusters); ground
           truth only used to colour the plots, never seen by the algorithm

Contrastive learning needs no labels at all. What it needs is a way to
produce two *different* views of the *same* underlying sample — with real
images that's a random crop/flip/colour-jitter; here, where a "sample" is
just a point in the plane, the natural analogue is a small random rigid
transform (rotation + scale) plus additive noise applied around the point's
own location. Two augmented views of the same point should still be more
similar to each other than to a view of any other point — that one property
is the entire training signal.

Public API
----------
make_blobs(n_points, n_clusters, seed)  — well-separated Gaussian blobs
make_ring(n_points, n_clusters, seed)   — small blobs spaced around a circle
make_dataset(shape, n_points, n_clusters, seed) — dispatcher

augment(X, rng, strength, kind) — produce one noisy/rotated/rescaled view

SHAPE_NAMES / SHAPE_KEYS   — dataset registry
AUG_NAMES / AUG_KEYS       — augmentation-type registry
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Registries — keep display name <-> key in sync with app.py
# ---------------------------------------------------------------------------

SHAPE_KEYS = ["blobs", "ring"]
SHAPE_NAMES = ["Gaussian blobs", "Clusters on a circle"]

AUG_KEYS = ["jitter", "rotate", "scale", "all"]
AUG_NAMES = ["Jitter only", "Rotate only", "Scale only", "Jitter + rotate + scale"]


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
# Dataset generators
# ---------------------------------------------------------------------------

def make_blobs(
    n_points: int = 120,
    n_clusters: int = 4,
    std: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Well-separated spherical Gaussian blobs on a jittered grid.

    Each blob is one "identity" that the encoder must learn to keep
    together in embedding space, purely from augmented-view agreement.
    """
    rng = np.random.default_rng(seed)

    cols = int(np.ceil(np.sqrt(n_clusters)))
    rows = int(np.ceil(n_clusters / cols))
    # Centre spacing relative to std matters a lot here: too generous (the
    # old 4*std*3, i.e. ~24 std between neighbours) and the blobs are so
    # separated that *any* random encoder — trained or not — already keeps
    # them apart on the unit circle, since a smooth map preserves distant
    # clusters' separation regardless of its weights. That made "before vs.
    # after" look identical on load. 4*std (~4 std between neighbours) still
    # reads as visually distinct blobs but leaves the untrained encoder
    # genuinely confused about some of them, so training has real work to do.
    spacing = max(4.0 * std, 1.5)

    grid_x, grid_y = np.meshgrid(np.arange(cols) * spacing, np.arange(rows) * spacing)
    centres = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:n_clusters]
    centres += rng.uniform(-0.2 * spacing, 0.2 * spacing, size=centres.shape)

    sizes = np.full(n_clusters, n_points // n_clusters, dtype=int)
    sizes[: n_points % n_clusters] += 1

    X_parts, label_parts = [], []
    for k, (centre, size) in enumerate(zip(centres, sizes)):
        cov = np.diag(rng.uniform(0.8, 1.2, size=2) * std**2)
        pts = rng.multivariate_normal(centre, cov, size=size)
        X_parts.append(pts)
        label_parts.append(np.full(size, k, dtype=int))

    X = np.vstack(X_parts)
    labels = np.concatenate(label_parts)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_ring(
    n_points: int = 120,
    n_clusters: int = 5,
    std: float = 0.9,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Small Gaussian blobs spaced evenly around a circle.

    Harder than plain blobs: neighbouring clusters are closer together,
    so the encoder has to push harder to keep them apart while still
    pulling each cluster's own augmented views together.
    """
    rng = np.random.default_rng(seed)
    radius = 4.0

    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    angles += rng.uniform(0, 2 * np.pi / n_clusters * 0.3)  # slight jitter of layout
    centres = radius * np.column_stack([np.cos(angles), np.sin(angles)])

    sizes = np.full(n_clusters, n_points // n_clusters, dtype=int)
    sizes[: n_points % n_clusters] += 1

    X_parts, label_parts = [], []
    for k, (centre, size) in enumerate(zip(centres, sizes)):
        pts = rng.normal(centre, std, size=(size, 2))
        X_parts.append(pts)
        label_parts.append(np.full(size, k, dtype=int))

    X = np.vstack(X_parts)
    labels = np.concatenate(label_parts)
    X, labels = _shuffle(X, labels, rng)
    return _normalise(X), labels


def make_dataset(
    shape: str,
    n_points: int = 120,
    n_clusters: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, labels) for the requested shape. See SHAPE_KEYS."""
    if shape == "blobs":
        return make_blobs(n_points=n_points, n_clusters=n_clusters, seed=seed)
    if shape == "ring":
        return make_ring(n_points=n_points, n_clusters=n_clusters, seed=seed)
    raise ValueError(f"Unknown shape {shape!r}. Choose from: {SHAPE_KEYS}")


# ---------------------------------------------------------------------------
# Augmentation — the "positive pair" mechanism
# ---------------------------------------------------------------------------

def augment(
    X: np.ndarray,
    rng: np.random.Generator,
    strength: float = 0.5,
    kind: str = "all",
) -> np.ndarray:
    """Return one randomly-augmented view of each row of X.

    strength in [0, 1] scales every effect; kind selects which geometric
    transforms are active. All transforms are centred so a strength of 0
    reproduces X exactly (an "augmentation" that changes nothing is a
    trivial positive pair, useful for sanity-checking the loss at strength=0).

    - rotate: small random rotation, up to ~18° at strength=1
    - scale:  random rescale, ±25% at strength=1
    - jitter: additive Gaussian noise, sigma up to ~0.35 at strength=1
      (data lives on a scale of roughly [-4, 4])
    """
    X = np.atleast_2d(X).astype(float)
    n = len(X)
    out = X.copy()

    if kind in ("rotate", "all") and strength > 0:
        theta = rng.uniform(-1.0, 1.0, size=n) * (0.32 * strength)
        cos, sin = np.cos(theta), np.sin(theta)
        x0, y0 = out[:, 0].copy(), out[:, 1].copy()
        out[:, 0] = cos * x0 - sin * y0
        out[:, 1] = sin * x0 + cos * y0

    if kind in ("scale", "all") and strength > 0:
        scale = 1.0 + rng.uniform(-1.0, 1.0, size=(n, 1)) * (0.25 * strength)
        out = out * scale

    if kind in ("jitter", "all") and strength > 0:
        out = out + rng.normal(0.0, 0.35 * strength, size=out.shape)

    return out
