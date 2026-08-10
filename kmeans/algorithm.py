"""
algorithm.py — From-scratch K-means with full iteration history

The algorithm alternates between two steps until convergence:

  E-step (assign): each point is assigned to its nearest centroid.
  M-step (update): each centroid is moved to the mean of its cluster.

Every E-step and M-step is recorded as a Snapshot so the visualiser
can play back the full convergence story frame by frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """State of the algorithm at one point in time."""

    # All data points — constant across snapshots (shape N×2)
    points: np.ndarray

    # Cluster index for each point (shape N,)
    labels: np.ndarray

    # Centroid positions (shape K×2)
    centroids: np.ndarray

    # Iteration number (1-indexed); 0 = initialisation frame
    iteration: int

    # Which sub-step produced this snapshot
    phase: Literal["init", "assign", "update"]

    # How many points changed cluster in the most recent assign step
    # (meaningful only when phase == "assign")
    n_changed: int = 0

    # True once centroids have stopped moving
    converged: bool = False

    # Whether the initial centroids were supplied manually by the user
    manual_init: bool = False

    @property
    def title(self) -> str:
        if self.phase == "init":
            kind = "manual" if self.manual_init else "random"
            return f"Initialisation — {kind} centroids"
        verb = "Assign points" if self.phase == "assign" else "Move centroids"
        tag = " ✓ converged" if self.converged else ""
        return f"Iteration {self.iteration} — {verb}{tag}"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return the index of the closest centroid for each point.

    We compute the squared Euclidean distance without the sqrt because
    we only need relative ordering; this is faster.

    Distance matrix shape: (N, K)
    """
    # X[:, np.newaxis, :] broadcasts to (N, 1, 2)
    # centroids[np.newaxis, :, :] broadcasts to (1, K, 2)
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]   # (N, K, 2)
    sq_dist = (diff ** 2).sum(axis=2)                           # (N, K)
    return sq_dist.argmin(axis=1)                               # (N,)


def _update(X: np.ndarray, labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Move each centroid to the mean of its assigned points.

    If a cluster is empty (can happen with bad initialisation), the
    centroid is re-initialised to a random data point to avoid NaNs.

    ``rng`` must be the same seeded generator used elsewhere in ``fit()``
    so that a run with a fixed seed is fully reproducible — including the
    empty-cluster fallback, which previously drew from the unseeded
    global NumPy random state.
    """
    centroids = np.empty((k, X.shape[1]), dtype=float)
    for c in range(k):
        mask = labels == c
        if mask.any():
            centroids[c] = X[mask].mean(axis=0)
        else:
            # Re-seed empty cluster at a random data point
            centroids[c] = X[rng.integers(len(X))]
    return centroids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 0,
    init_centroids: np.ndarray | None = None,
) -> list[Snapshot]:
    """Run K-means on X and return every intermediate snapshot.

    Parameters
    ----------
    X:              data matrix, shape (N, 2)
    k:              number of clusters
    max_iter:       hard cap on EM iterations
    tol:            centroid shift threshold below which we declare convergence
                    (we use the max shift across all centroids)
    seed:           random seed used only when init_centroids is None
    init_centroids: optional (K, 2) array of manually chosen starting centroids;
                    when supplied the random initialisation is skipped entirely

    Returns
    -------
    A list of Snapshot objects in chronological order:
      [init, assign_1, update_1, assign_2, update_2, ..., final_assign]
    """
    n = len(X)
    manual_init = init_centroids is not None

    # A single seeded generator drives every random choice in this run —
    # both the initial centroid draw (random mode) and the empty-cluster
    # fallback in _update() (either mode) — so a fixed seed is fully
    # reproducible end to end.
    rng = np.random.default_rng(seed)

    if manual_init:
        centroids = np.asarray(init_centroids, dtype=float).copy()
        if centroids.shape != (k, 2):
            raise ValueError(
                f"init_centroids must have shape ({k}, 2), got {centroids.shape}"
            )
    else:
        # Pick K distinct data points as starting centroids
        init_indices = rng.choice(n, size=k, replace=False)
        centroids = X[init_indices].copy()

    # Before the first assign step every point gets label -1 (unassigned)
    labels = np.full(n, -1, dtype=int)

    snapshots: list[Snapshot] = [
        Snapshot(
            points=X,
            labels=labels.copy(),
            centroids=centroids.copy(),
            iteration=0,
            phase="init",
            manual_init=manual_init,
        )
    ]

    for iteration in range(1, max_iter + 1):
        # ---- E-step: assign each point to nearest centroid ----------------
        prev_labels = labels.copy()
        labels = _assign(X, centroids)
        n_changed = int((labels != prev_labels).sum())

        snapshots.append(
            Snapshot(
                points=X,
                labels=labels.copy(),
                centroids=centroids.copy(),
                iteration=iteration,
                phase="assign",
                n_changed=n_changed,
            )
        )

        # ---- M-step: move centroids to cluster means ----------------------
        prev_centroids = centroids.copy()
        centroids = _update(X, labels, k, rng)
        max_shift = float(np.linalg.norm(centroids - prev_centroids, axis=1).max())
        converged = max_shift < tol

        snapshots.append(
            Snapshot(
                points=X,
                labels=labels.copy(),
                centroids=centroids.copy(),
                iteration=iteration,
                phase="update",
                n_changed=n_changed,
                converged=converged,
            )
        )

        if converged:
            break

    return snapshots
