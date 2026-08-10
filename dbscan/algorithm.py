"""
algorithm.py — From-scratch DBSCAN with full step history

DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
classifies each point as one of three types:

  Core point   — has at least minPts neighbours within radius ε (including itself)
  Border point — within ε of a core point but not a core point itself
  Noise point  — neither core nor border; isolated outlier

Clusters are formed by linking core points that are within ε of each other
and absorbing any border points reachable from them.

Key difference from K-means
----------------------------
  • No K to specify — the number of clusters emerges from the data
  • Non-convex shapes are handled naturally (density connectivity, not distance to centroid)
  • Outliers are explicitly labelled as noise (label = -1)
  • Result depends on ε and minPts, not on a random initialisation

Snapshot strategy
-----------------
Recording one snapshot per point would produce hundreds of frames. Instead
we record at the level of cluster formation events:

  1. init          — all points unvisited
  2. examine       — seed point found; ε-neighbourhood highlighted, ε-circle shown
  3. expand        — BFS complete; full cluster coloured in
  4. (repeat 2–3 for each cluster)
  5. done          — all points classified; noise shown as ✕

This gives 2 × n_clusters + 2 frames, typically 6–20 for well-parameterised data.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Point role constants (stored in roles array)
# ---------------------------------------------------------------------------
UNVISITED = 0
CORE      = 1
BORDER    = 2
NOISE     = 3

ROLE_NAMES = {UNVISITED: "unvisited", CORE: "core", BORDER: "border", NOISE: "noise"}

# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """Algorithm state at one recorded moment."""

    # All data points — constant across snapshots (N×2)
    points: np.ndarray

    # Cluster assignment: -2 = unvisited, -1 = noise, 0..K = cluster id
    labels: np.ndarray

    # Point role: UNVISITED / CORE / BORDER / NOISE
    roles: np.ndarray

    # Index of the point whose ε-neighbourhood is being shown (examine phase)
    # None in all other phases
    active_point: int | None

    # Indices of points in the active ε-neighbourhood (examine phase)
    neighbors: np.ndarray | None

    # ε radius used (needed by visualiser to draw the circle)
    eps: float

    phase: Literal["init", "examine", "expand", "done"]

    # 0-indexed cluster being built (−1 before any cluster starts)
    cluster_id: int

    @property
    def n_clusters(self) -> int:
        return int(self.labels.max()) + 1 if self.labels.max() >= 0 else 0

    @property
    def n_noise(self) -> int:
        return int((self.labels == -1).sum())

    @property
    def n_unvisited(self) -> int:
        return int((self.labels == -2).sum())

    @property
    def title(self) -> str:
        if self.phase == "init":
            return "Initialisation — all points unvisited"
        if self.phase == "examine":
            return (
                f"Cluster {self.cluster_id + 1} — examine seed point "
                f"({len(self.neighbors)} neighbours in ε-ball)"
            )
        if self.phase == "expand":
            return (
                f"Cluster {self.cluster_id + 1} complete "
                f"({(self.labels == self.cluster_id).sum()} points)"
            )
        # done
        return (
            f"Done — {self.n_clusters} cluster{'s' if self.n_clusters != 1 else ''}, "
            f"{self.n_noise} noise point{'s' if self.n_noise != 1 else ''}"
        )


# ---------------------------------------------------------------------------
# Core algorithm helpers
# ---------------------------------------------------------------------------

def _range_query(X: np.ndarray, idx: int, eps: float) -> np.ndarray:
    """Return indices of all points within Euclidean distance eps of X[idx]."""
    diff = X - X[idx]
    sq_dist = (diff ** 2).sum(axis=1)
    return np.where(sq_dist <= eps ** 2)[0]


def _copy(snap: Snapshot) -> Snapshot:
    """Deep-copy mutable arrays so each snapshot is independent."""
    return Snapshot(
        points=snap.points,           # immutable — shared reference is fine
        labels=snap.labels.copy(),
        roles=snap.roles.copy(),
        active_point=snap.active_point,
        neighbors=snap.neighbors.copy() if snap.neighbors is not None else None,
        eps=snap.eps,
        phase=snap.phase,
        cluster_id=snap.cluster_id,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> list[Snapshot]:
    """Run DBSCAN on X and return the full snapshot history.

    Parameters
    ----------
    X:           data matrix, shape (N, 2)
    eps:         neighbourhood radius ε
    min_samples: minimum points (including the point itself) to form a core point

    Returns
    -------
    List of Snapshots: [init, examine_0, expand_0, examine_1, expand_1, ..., done]
    """
    n = len(X)
    labels = np.full(n, -2, dtype=int)   # -2 = unvisited
    roles  = np.full(n, UNVISITED, dtype=int)

    base = Snapshot(
        points=X,
        labels=labels,
        roles=roles,
        active_point=None,
        neighbors=None,
        eps=eps,
        phase="init",
        cluster_id=-1,
    )
    snapshots: list[Snapshot] = [_copy(base)]

    cluster_id = -1

    for i in range(n):
        if labels[i] != -2:
            continue  # already processed

        neighbors = _range_query(X, i, eps)

        if len(neighbors) < min_samples:
            # Not enough neighbours — tentatively noise
            # (may be upgraded to border later if absorbed by a cluster)
            labels[i] = -1
            roles[i] = NOISE
            continue

        # ---- Core point found — start a new cluster ----------------------
        cluster_id += 1

        # Examine snapshot: show the ε-ball around this seed
        labels[i] = cluster_id
        roles[i] = CORE
        snapshots.append(Snapshot(
            points=X,
            labels=labels.copy(),
            roles=roles.copy(),
            active_point=i,
            neighbors=neighbors.copy(),
            eps=eps,
            phase="examine",
            cluster_id=cluster_id,
        ))

        # BFS expansion
        seeds: deque[int] = deque(
            j for j in neighbors if j != i
        )
        in_queue: set[int] = set(seeds)

        while seeds:
            j = seeds.popleft()
            in_queue.discard(j)

            if labels[j] == -1:
                # Was noise — promote to border
                labels[j] = cluster_id
                roles[j] = BORDER

            if labels[j] != -2:
                continue  # already in a cluster

            labels[j] = cluster_id
            neighbors_j = _range_query(X, j, eps)

            if len(neighbors_j) >= min_samples:
                roles[j] = CORE
                # Add unvisited or noise neighbours to the queue
                for k in neighbors_j:
                    if labels[k] == -2 or labels[k] == -1:
                        if k not in in_queue:
                            seeds.append(k)
                            in_queue.add(k)
            else:
                roles[j] = BORDER

        # Expand snapshot: show the completed cluster
        snapshots.append(Snapshot(
            points=X,
            labels=labels.copy(),
            roles=roles.copy(),
            active_point=None,
            neighbors=None,
            eps=eps,
            phase="expand",
            cluster_id=cluster_id,
        ))

    # Mark any remaining unvisited points as noise
    unvisited_mask = labels == -2
    labels[unvisited_mask] = -1
    roles[unvisited_mask] = NOISE

    snapshots.append(Snapshot(
        points=X,
        labels=labels.copy(),
        roles=roles.copy(),
        active_point=None,
        neighbors=None,
        eps=eps,
        phase="done",
        cluster_id=cluster_id,
    ))

    return snapshots
