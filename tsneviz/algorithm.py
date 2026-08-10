"""
t-Distributed Stochastic Neighbour Embedding (t-SNE) — educational NumPy core.

High level
----------
1. Gaussian affinities P_ij in input space (perplexity-matched two-sided kernels).
2. Map to 2-D with Student-t (Cauchy) kernel Q_ij; minimise KL(P || Q) with GD + momentum.
3. Early exaggeration: scale P during early iterations so repulsive forces do not collapse the map.

Complexity O(n²) per step — intended for n ≲ 500 in the visualiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Phase = Literal["early", "main"]


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    sum_x = np.sum(X**2, axis=1, keepdims=True)
    return sum_x + sum_x.T - 2 * (X @ X.T)


def _joint_probabilities(
    dist_sq: np.ndarray,
    perplexity: float,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> np.ndarray:
    """Symmetric joint distribution P (sums to 1, zero diagonal)."""
    n = dist_sq.shape[0]
    P = np.zeros((n, n), dtype=np.float64)
    target_entropy = np.log(perplexity)
    ind = np.arange(n)

    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        beta = 1.0
        not_i = ind != i
        di = dist_sq[i, not_i].astype(np.float64)

        for _ in range(max_iter):
            p_unnorm = np.exp(-di * beta)
            sum_p = np.sum(p_unnorm) + 1e-12
            p_row = p_unnorm / sum_p
            entropy = -np.sum(p_row * np.log(p_row + 1e-15))
            h_diff = entropy - target_entropy
            if abs(h_diff) < tol:
                break
            if h_diff > 0:
                betamin = beta
                beta = (beta + betamax) / 2 if betamax != np.inf else beta * 2.0
            else:
                betamax = beta
                beta = (beta + betamin) / 2 if betamin != -np.inf else beta / 2.0

        p_unnorm = np.exp(-di * beta)
        p_unnorm /= np.sum(p_unnorm) + 1e-12
        P[i, not_i] = p_unnorm

    P = (P + P.T) / (2 * n)
    np.fill_diagonal(P, 0.0)
    P = np.maximum(P, 1e-12)
    np.fill_diagonal(P, 0.0)
    P /= np.sum(P)
    return P.astype(np.float64)


def _pca_two_components(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    y = x @ vt[:2].T
    span = np.abs(y).max() + 1e-9
    y = y / span * 4.0
    y += rng.normal(0, 0.015, size=y.shape)
    return y.astype(np.float64)


def _kl_gradient(Y: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, float]:
    """Gradient of KL(P||Q) w.r.t. Y; Q from Student-t kernel (df=1)."""
    n = Y.shape[0]
    diff = Y[:, None, :] - Y[None, :, :]
    d_sq = np.sum(diff**2, axis=2)
    q_num = 1.0 / (1.0 + d_sq)
    np.fill_diagonal(q_num, 0.0)
    z = np.sum(q_num) + 1e-12
    q = q_num / z
    pq = P - q
    w = 4.0 * pq / (1.0 + d_sq)
    np.fill_diagonal(w, 0.0)
    grad = w.sum(axis=1, keepdims=True) * Y - w @ Y
    mask = ~np.eye(n, dtype=bool)
    kl = float(np.sum(P[mask] * np.log((P[mask] + 1e-15) / (q[mask] + 1e-15))))
    return grad.astype(np.float64), kl


@dataclass
class Snapshot:
    Y: np.ndarray                  # (n, 2) embedding
    pca2: np.ndarray              # (n, 2) PCA preview of X
    labels: np.ndarray
    latent_xy: np.ndarray | None  # reference latent (or None)
    kl_divergence: float
    iteration: int                # 0 .. n_iter-1
    phase: Phase
    perplexity: float


def fit(
    X: np.ndarray,
    labels: np.ndarray,
    latent_xy: np.ndarray | None,
    *,
    perplexity: float = 30.0,
    n_iter: int = 750,
    early_exaggeration: float = 12.0,
    early_exaggeration_iter: int = 150,
    learning_rate: float = 250.0,
    momentum: float = 0.8,
    mom_switch_iter: int = 250,
    initial_momentum: float = 0.5,
    random_state: int = 0,
    snap_every: int = 15,
) -> list[Snapshot]:
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    cap = max(5.0, (n - 1) / 3.0)
    eff_perp = float(np.clip(perplexity, 5.0, cap))
    dist_sq = _pairwise_sq_dists(X.astype(np.float64))
    P = _joint_probabilities(dist_sq, eff_perp)
    pca2 = _pca_two_components(X, rng)

    Y = rng.normal(0.0, 1e-4, size=(n, 2)).astype(np.float64)
    vel = np.zeros_like(Y)

    snaps: list[Snapshot] = []

    def record(it: int, phase: Phase, p_scaled: np.ndarray) -> None:
        p_norm = p_scaled / (p_scaled.sum() + 1e-12)
        _, kl = _kl_gradient(Y, p_norm)
        snaps.append(
            Snapshot(
                Y=Y.copy(),
                pca2=pca2.copy(),
                labels=labels.copy(),
                latent_xy=None if latent_xy is None else latent_xy.copy(),
                kl_divergence=kl,
                iteration=it,
                phase=phase,
                perplexity=eff_perp,
            ),
        )

    record(0, "early", P * early_exaggeration)

    for it in range(n_iter):
        phase: Phase = "early" if it < early_exaggeration_iter else "main"
        p_work = P * (early_exaggeration if phase == "early" else 1.0)
        mom = initial_momentum if it < mom_switch_iter else momentum

        grad, _ = _kl_gradient(Y, p_work)
        vel = mom * vel - learning_rate * grad
        Y = Y + vel
        Y -= Y.mean(axis=0, keepdims=True)

        if it == 0 or (it + 1) % snap_every == 0 or it == n_iter - 1:
            record(it + 1, phase, p_work)

    return snaps
