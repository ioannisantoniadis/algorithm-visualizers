"""
algorithm.py — PCA from scratch with educational Snapshot sequence (d = 2 or 3)

Pipeline:
  raw → centred → covariance geometry → eigenvectors → rank‑k reconstruction

Uses numpy.linalg.eigh on the sample covariance **S** = X̃ᵀX̃ / (n−1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Phase = Literal["raw", "centered", "covariance", "eigen", "reconstruct"]


@dataclass
class Snapshot:
    """One animation frame."""

    phase: Phase
    X: np.ndarray              # original data (n×d), d ∈ {2, 3}
    mu: np.ndarray             # sample mean (d,)
    X_centered: np.ndarray     # X − μ
    cov: np.ndarray            # sample covariance S (d×d)
    evals: np.ndarray          # eigenvalues descending (d,)
    evecs: np.ndarray          # eigenvectors as columns (d×d)
    k_components: int          # 0…d during reconstruct; −1 otherwise
    X_hat: np.ndarray | None   # reconstruction for reconstruct phase
    mse: float                 # mean squared entry of (X − X̂), i.e. ||X−X̂||_F² / (n·d)
    cumulative_variance: float # ∑ⱼ λⱼ / trace(S) for kept j (fraction of variance along PCs)

    @property
    def ndim(self) -> int:
        return int(self.X.shape[1])

    @property
    def title(self) -> str:
        d = self.ndim
        if self.phase == "raw":
            return f"Raw data — ℝ^{d} cloud before centering"
        if self.phase == "centered":
            return "Centered cloud — subtract sample mean"
        if self.phase == "covariance":
            if d == 2:
                return "Covariance ellipse — ~95% Mahalanobis contour"
            return "Covariance ellipsoid — ~95% Mahalanobis surface"
        if self.phase == "eigen":
            tot = float(self.evals.sum())
            if tot <= 0:
                return "Eigenvectors"
            parts = [f"PC{j + 1}: {100.0 * float(self.evals[j]) / tot:.1f}%"
                     for j in range(d)]
            return "Eigenvectors — " + ", ".join(parts)
        # reconstruct
        k = self.k_components
        pct = 100.0 * self.cumulative_variance
        if k == 0:
            return (
                "0 PCs — all points collapse to the mean "
                f"({pct:.1f}% var., MSE = {self.mse:.4f})"
            )
        if k < d:
            return (
                f"{k} PC{'s' if k > 1 else ''} — best linear {k}-dim subspace "
                f"({pct:.1f}% var., MSE = {self.mse:.4f})"
            )
        return (
            f"{d} PCs — perfect reconstruction in ℝ^{d} "
            f"(100% var., MSE = {self.mse:.4f})"
        )


def _reconstruct(
    X: np.ndarray,
    X_centered: np.ndarray,
    mu: np.ndarray,
    evecs: np.ndarray,
    k: int,
) -> np.ndarray:
    n = X.shape[0]
    if k == 0:
        return np.tile(mu, (n, 1))
    Vk = evecs[:, :k]
    return X_centered @ Vk @ Vk.T + mu


def fit(X: np.ndarray) -> list[Snapshot]:
    """Run PCA and return ordered snapshots for the visualiser."""
    if X.ndim != 2 or X.shape[1] not in (2, 3):
        raise ValueError("fit() expects X of shape (n_points, 2) or (n_points, 3)")
    n, d = X.shape
    mu = X.mean(axis=0)
    X_centered = X - mu
    dof = max(n - 1, 1)
    cov = (X_centered.T @ X_centered) / dof
    evals_asc, evecs_asc = np.linalg.eigh(cov)
    order = np.argsort(evals_asc)[::-1]
    evals = np.maximum(evals_asc[order], 1e-15)
    evecs = evecs_asc[:, order]

    trace = float(evals.sum())
    var_ratios = evals / trace if trace > 0 else np.ones(d, dtype=float) / d

    snaps: list[Snapshot] = []

    def snap(
        phase: Phase,
        k_comp: int,
        X_hat: np.ndarray | None,
    ) -> None:
        if X_hat is None:
            mse = 0.0
            cum = 0.0
        else:
            err = X - X_hat
            mse = float(np.mean(err ** 2))
            if k_comp <= 0:
                cum = 0.0
            else:
                cum = float(var_ratios[:k_comp].sum())
        snaps.append(
            Snapshot(
                phase=phase,
                X=X.copy(),
                mu=mu.copy(),
                X_centered=X_centered.copy(),
                cov=cov.copy(),
                evals=evals.copy(),
                evecs=evecs.copy(),
                k_components=k_comp,
                X_hat=None if X_hat is None else X_hat.copy(),
                mse=mse,
                cumulative_variance=cum,
            )
        )

    snap("raw", -1, None)
    snap("centered", -1, None)
    snap("covariance", -1, None)
    snap("eigen", -1, None)

    for k in range(0, d + 1):
        X_hat = _reconstruct(X, X_centered, mu, evecs, k)
        snap("reconstruct", k, X_hat)

    return snaps
