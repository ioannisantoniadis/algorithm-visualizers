"""
Expectation–Maximisation for Gaussian mixture models (full covariance per component).

Snapshots alternate **M-step** (new ellipses; point colours still from the previous
E-step) and **E-step** (updated soft responsibilities for the current parameters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Substep = Literal["init", "e", "m"]


def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    return s.squeeze(axis)


def _log_gaussian_pdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log N(x | mean, cov) for each row of X."""
    d = X.shape[1]
    xc = X - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        logdet = -1e6
    cov_inv = np.linalg.inv(cov)
    quad = np.sum(xc @ cov_inv * xc, axis=1)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


@dataclass
class Snapshot:
    X: np.ndarray                 # (n, 2)
    responsibilities: np.ndarray  # (n, K) soft assignments shown for this frame
    weights: np.ndarray           # (K,)
    means: np.ndarray             # (K, 2)
    covs: np.ndarray              # (K, 2, 2)
    log_likelihood: float           # under theta implied by means/covs/weights
    iteration: int                  # 0 = after init E-step
    substep: Substep


def _init_parameters(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    means = X[idx].astype(np.float64).copy()
    data_cov = np.cov(X.T)
    if not np.isfinite(data_cov).all() or data_cov.shape != (2, 2):
        data_cov = np.eye(2)
    scale = float(np.trace(data_cov) / 2.0 * 0.25)
    covs = np.stack([np.eye(2, dtype=np.float64) * scale for _ in range(k)])
    weights = np.full(k, 1.0 / k, dtype=np.float64)
    return weights, means, covs


def _e_step(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
) -> tuple[np.ndarray, float]:
    n, k = X.shape[0], len(weights)
    log_prob = np.zeros((n, k), dtype=np.float64)
    for j in range(k):
        log_prob[:, j] = np.log(weights[j] + 1e-16) + _log_gaussian_pdf(X, means[j], covs[j])
    log_sum = _logsumexp(log_prob, axis=1)
    gamma = np.exp(log_prob - log_sum[:, None])
    ll = float(log_sum.sum())
    return gamma, ll


def _m_step(
    X: np.ndarray,
    gamma: np.ndarray,
    reg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = X.shape
    k = gamma.shape[1]
    nk = gamma.sum(axis=0) + 1e-10
    weights = nk / n
    means = (gamma.T @ X) / nk[:, None]
    covs = np.zeros((k, d, d), dtype=np.float64)
    for j in range(k):
        xc = X - means[j]
        covs[j] = (gamma[:, j : j + 1] * xc).T @ xc / nk[j]
        covs[j] += reg * np.eye(d)
    return weights, means, covs


def fit(
    X: np.ndarray,
    n_components: int,
    *,
    max_iter: int = 25,
    reg_covar: float = 1e-6,
    random_state: int = 0,
) -> list[Snapshot]:
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must be shape (n, 2)")
    rng = np.random.default_rng(random_state)
    k = int(n_components)
    if k < 1:
        raise ValueError("n_components must be >= 1")

    weights, means, covs = _init_parameters(X, k, rng)
    snaps: list[Snapshot] = []

    gamma, ll = _e_step(X, weights, means, covs)
    snaps.append(
        Snapshot(
            X=X.copy(),
            responsibilities=gamma.copy(),
            weights=weights.copy(),
            means=means.copy(),
            covs=covs.copy(),
            log_likelihood=ll,
            iteration=0,
            substep="init",
        ),
    )

    for it in range(1, max_iter + 1):
        weights, means, covs = _m_step(X, gamma, reg_covar)
        _, ll_m = _e_step(X, weights, means, covs)
        snaps.append(
            Snapshot(
                X=X.copy(),
                responsibilities=gamma.copy(),
                weights=weights.copy(),
                means=means.copy(),
                covs=covs.copy(),
                log_likelihood=ll_m,
                iteration=it,
                substep="m",
            ),
        )

        gamma, ll = _e_step(X, weights, means, covs)
        snaps.append(
            Snapshot(
                X=X.copy(),
                responsibilities=gamma.copy(),
                weights=weights.copy(),
                means=means.copy(),
                covs=covs.copy(),
                log_likelihood=ll,
                iteration=it,
                substep="e",
            ),
        )

    return snaps
