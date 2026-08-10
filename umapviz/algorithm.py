"""
From-scratch UMAP-style embedding (McInnes, Healy, Melville, 2018).

Stages recorded for the visualiser:
  1. k-nearest neighbour distances
  2. Symmetrised fuzzy topological graph (fuzzy union of local simplicial sets)
  3. Stochastic gradient embedding in R² (UMAP lower-dimensional fuzzy set CE)

Implementation follows the structure of umap-learn (nearest neighbours →
smooth_knn_dist → membership → union → layout SGD) with a readable Python
SGD loop instead of numba. Intended for small n (≲ 400) suitable for animation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Phase = Literal["knn", "fuzzy", "embed"]

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INF = 1e10


@dataclass
class Snapshot:
    phase: Phase
    step: int
    X: np.ndarray           # (n, D) high-dimensional data
    labels: np.ndarray      # colouring only
    latent_xy: np.ndarray | None  # (n, 2) or (n, 3) before lift; NaN row = no latent (mixed noise)
    pca2: np.ndarray        # (n, 2) PCA preview of X
    Y: np.ndarray           # (n, 2) embedding (meaningful in embed phase)
    knn_indices: np.ndarray | None
    knn_dists: np.ndarray | None
    fuzzy_edges: np.ndarray | None  # shape (E, 3) i, j, weight (upper triangle)
    a_param: float
    b_param: float
    n_neighbors: int

    @property
    def title(self) -> str:
        if self.phase == "knn":
            return "Stage 1/3 — kNN graph in ℝᴰ (edges drawn in PCA₂)"
        if self.phase == "fuzzy":
            return "Stage 2/3 — fuzzy simplicial union (projected with PCA₂)"
        if self.step == 0:
            return "Stage 3/3 — embedding initialisation (PCA + jitter)"
        return f"Stage 3/3 — UMAP embedding (SGD epoch {self.step})"


def _pca_first_two(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    y = x @ vt[:2].T
    y = y / (np.abs(y).max() + 1e-9) * 4.0
    y += rng.normal(0, 0.02, size=y.shape)
    return y.astype(np.float64)


def _smooth_knn_dist(
    distances: np.ndarray,
    k: float,
    local_connectivity: float = 1.0,
    bandwidth: float = 1.0,
    n_iter: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of umap `smooth_knn_dist` (see umap/umap_.py).

    Note: umap-learn's own `distances` array includes each point's self
    distance (0.0) at column 0, so its inner loop starts at `j = 1` to skip
    it. Here `distances` is `knn_dists`, which already excludes self (see
    `fit()` — `knn_indices = knn_full[:, 1:]`), so every column is a real
    neighbour and must be included; starting at `j = 1` would silently drop
    the closest neighbour from the sigma/rho solve.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float64)
    result = np.zeros(distances.shape[0], dtype=np.float64)
    mean_distances = float(np.mean(distances))

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INF
        mid = 1.0
        ith_distances = distances[i]
        nonzero = ith_distances[ith_distances > 0.0]
        if nonzero.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = nonzero[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (nonzero[index] - nonzero[index - 1])
            else:
                rho[i] = interpolation * nonzero[0]
        elif nonzero.shape[0] > 0:
            rho[i] = float(np.max(nonzero))

        for _ in range(n_iter):
            psum = 0.0
            for j in range(distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0
            if abs(psum - target) < SMOOTH_K_TOLERANCE:
                break
            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INF:
                    mid *= 2.0
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid
        if rho[i] > 0.0:
            mean_ith = float(np.mean(ith_distances))
            if result[i] < MIN_K_DIST_SCALE * mean_ith:
                result[i] = MIN_K_DIST_SCALE * mean_ith
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho


def _membership_rows(
    knn_indices: np.ndarray,
    knn_dists: np.ndarray,
    sigmas: np.ndarray,
    rhos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directed fuzzy simplicial 1-skeleton (COO), excluding self loops."""
    n_samples, n_neighbors = knn_indices.shape
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n_samples):
        for j in range(n_neighbors):
            jdx = int(knn_indices[i, j])
            if jdx < 0 or jdx == i:
                continue
            dij = float(knn_dists[i, j])
            if dij - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = float(np.exp(-((dij - rhos[i]) / (sigmas[i]))))
            rows.append(i)
            cols.append(jdx)
            vals.append(val)
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(vals, dtype=np.float64),
    )


def _fuzzy_union_coo(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    n_vertices: int,
) -> np.ndarray:
    """Return symmetric adjacency in dense form (small n only)."""
    directed = np.zeros((n_vertices, n_vertices), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals, strict=True):
        directed[int(r), int(c)] = max(directed[int(r), int(c)], v)
    sym = directed + directed.T - directed * directed.T
    np.fill_diagonal(sym, 0.0)
    return sym


def _find_ab_params(spread: float, min_dist: float) -> tuple[float, float]:
    """Fit (a, b) for q(d) = 1/(1 + a d^{2b}) — grid search, no scipy."""
    xv = np.linspace(0.0, spread * 3, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread)).astype(np.float64)
    best_a, best_b = 1.0, 1.0
    best_err = 1e100
    for a in np.geomspace(0.05, 25.0, 36):
        for b in np.linspace(0.15, 2.2, 36):
            yhat = 1.0 / (1.0 + a * xv ** (2.0 * b))
            err = float(np.mean((yhat - yv) ** 2))
            if err < best_err:
                best_err = err
                best_a, best_b = float(a), float(b)
    return best_a, best_b


def _rdist_sq(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    return float(np.dot(d, d))


def _clip(v: float, lo: float = -4.0, hi: float = 4.0) -> float:
    return float(max(lo, min(hi, v)))


def _optimize_epochs_simple(
    embedding: np.ndarray,
    sym: np.ndarray,
    n_epochs: int,
    a: float,
    b: float,
    rng: np.random.Generator,
    initial_alpha: float,
    gamma: float,
    negative_sample_rate: int,
) -> None:
    """Simpler per-epoch schedule: shuffle all weighted edges each epoch."""
    n = sym.shape[0]
    rows, cols = np.nonzero(np.triu(sym, 1))
    wts = sym[rows, cols]
    mask = wts > 1e-8
    rows = rows[mask]
    cols = cols[mask]
    wts = wts[mask]
    if rows.size == 0:
        return

    for epoch in range(n_epochs):
        alpha = initial_alpha
        order = rng.permutation(len(rows))
        for idx in order:
            i, j = int(rows[idx]), int(cols[idx])
            w = float(wts[idx])
            if i == j:
                continue
            current = embedding[i]
            other = embedding[j]
            dist_sq = _rdist_sq(current, other)
            if dist_sq > 0.0:
                grad_coeff = (
                    -2.0
                    * a
                    * b
                    * (dist_sq ** (b - 1.0))
                    / (a * (dist_sq ** b) + 1.0)
                )
            else:
                grad_coeff = 0.0
            # Symmetric attractive update: both endpoints of the edge move.
            # `rows, cols` only ever list i < j (upper triangle), so without
            # this the higher-index endpoint of every edge — including
            # whichever point ends up with the largest index after the
            # dataset's row permutation — would never receive a direct
            # attractive pull and could drift untouched by attraction.
            for d in range(2):
                gd = _clip(w * grad_coeff * (current[d] - other[d]))
                embedding[i, d] += gd * alpha
                embedding[j, d] += -gd * alpha

            for _ in range(negative_sample_rate):
                k = int(rng.integers(0, n))
                if k == i or k == j:
                    continue
                current = embedding[i]
                oth = embedding[k]
                dsq = _rdist_sq(current, oth)
                if dsq <= 0.0:
                    continue
                gr = 2.0 * gamma * b / ((0.001 + dsq) * (a * (dsq ** b) + 1.0))
                for d in range(2):
                    gd = _clip(gr * (current[d] - oth[d]))
                    embedding[i, d] += gd * alpha


def fit(
    X: np.ndarray,
    labels: np.ndarray,
    latent_xy: np.ndarray | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.15,
    spread: float = 1.0,
    n_epochs: int = 220,
    snap_embed_every: int = 30,
    random_state: int = 0,
    negative_sample_rate: int = 5,
    learning_rate: float = 1.0,
    repulsion_strength: float = 1.0,
) -> list[Snapshot]:
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_features)")
    n, _dim_h = X.shape
    rng = np.random.default_rng(random_state)
    pca2 = _pca_first_two(X, rng)
    lat_copy = None if latent_xy is None else latent_xy.astype(np.float64).copy()

    # --- brute-force kNN (including point 0 at self with dist inf skip) ---
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    knn_full = np.argsort(d2, axis=1)[:, : n_neighbors + 1]
    knn_indices = knn_full[:, 1:].astype(np.int32)
    knn_dists = np.sqrt(
        np.take_along_axis(d2, knn_indices.astype(np.int64), axis=1),
    ).astype(np.float64)

    sigmas, rhos = _smooth_knn_dist(
        knn_dists.astype(np.float64),
        float(n_neighbors),
        local_connectivity=1.0,
    )
    rows, cols, vals = _membership_rows(knn_indices, knn_dists, sigmas, rhos)
    sym = _fuzzy_union_coo(rows, cols, vals, n)
    # prune very weak edges (umap-style)
    thr = float(np.max(sym) / max(n_epochs, 10))
    sym[sym < thr] = 0.0

    a, b = _find_ab_params(spread, min_dist)

    snaps: list[Snapshot] = []

    def snap(
        phase: Phase,
        step: int,
        y: np.ndarray,
        fuzzy: np.ndarray | None,
    ) -> None:
        snaps.append(
            Snapshot(
                phase=phase,
                step=step,
                X=X.copy(),
                labels=labels.copy(),
                latent_xy=None if lat_copy is None else lat_copy.copy(),
                pca2=pca2.copy(),
                Y=y.copy(),
                knn_indices=knn_indices.copy() if phase == "knn" else None,
                knn_dists=knn_dists.copy() if phase == "knn" else None,
                fuzzy_edges=fuzzy.copy() if fuzzy is not None else None,
                a_param=a,
                b_param=b,
                n_neighbors=n_neighbors,
            )
        )

    # --- kNN snapshot (draw in PCA space) ---
    y0 = np.zeros((n, 2), dtype=np.float64)
    snap("knn", 0, y0, None)

    # --- fuzzy graph edges (upper triangle, cap count) ---
    ui, uj = np.nonzero(np.triu(sym, 1))
    wij = sym[ui, uj]
    keep = wij > 0
    ui, uj, wij = ui[keep], uj[keep], wij[keep]
    order = np.argsort(-wij)
    cap = min(4000, len(order))
    edges = np.column_stack(
        [ui[order[:cap]], uj[order[:cap]], wij[order[:cap]]],
    ).astype(np.float64)
    snap("fuzzy", 0, y0, edges)

    # --- initial embedding: PCA + jitter ---
    y = _pca_first_two(X, rng).astype(np.float64)
    snap("embed", 0, y.copy(), None)

    step_no = 0
    remaining = n_epochs
    while remaining > 0:
        chunk = min(snap_embed_every, remaining)
        lr = learning_rate * (1.0 - step_no / max(n_epochs, 1))
        _optimize_epochs_simple(
            y,
            sym,
            chunk,
            a,
            b,
            rng,
            initial_alpha=lr,
            gamma=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
        )
        step_no += chunk
        remaining -= chunk
        snap("embed", step_no, y.copy(), None)

    return snaps
