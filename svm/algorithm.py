"""
algorithm.py — From-scratch soft-margin kernel SVM, trained with a simplified
Sequential Minimal Optimisation (SMO) solver.

Why SMO instead of vanilla gradient descent on the primal?  The primal hinge
loss

    L(w, b) = (1/2)||w||^2 + C * sum_i max(0, 1 - y_i (w·x_i + b))

is easy to minimise directly *only* for the linear kernel — w lives in the
same space as the data.  RBF and polynomial kernels map x into a (possibly
infinite-dimensional) feature space where w is never formed explicitly.  The
standard trick is to optimise the dual instead:

    max_alpha  sum_i alpha_i - 1/2 sum_i sum_j alpha_i alpha_j y_i y_j K(x_i, x_j)
    s.t.       0 <= alpha_i <= C,  sum_i alpha_i y_i = 0

which only ever touches the data through the kernel function K. Points with
alpha_i > 0 are exactly the **support vectors** — everything else could be
deleted from the training set without changing the decision boundary at all.

Simplified SMO (Platt, 1998) optimises this dual two coordinates at a time —
the smallest update that can keep the equality constraint satisfied. Each
full sweep over the data is recorded as one Snapshot, so the visualiser can
replay how the margin tightens and the support-vector set shrinks to just
the points that matter.

Because dual coordinate ascent on this objective is mathematically the same
as descending the primal hinge loss (strong duality holds for this convex
problem), the `dual_objective` recorded on every snapshot — which climbs
monotonically by construction — is a faithful mirror of the hinge loss
falling during training, even though no gradient is ever computed
explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

KernelName = Literal["linear", "rbf", "poly"]


# ---------------------------------------------------------------------------
# Kernels — the only place the feature map is ever touched
# ---------------------------------------------------------------------------

def kernel_matrix(
    A: np.ndarray,
    B: np.ndarray,
    kernel: KernelName,
    gamma: float,
    degree: int,
    coef0: float,
) -> np.ndarray:
    """Gram matrix K[i, j] = K(A[i], B[j]) for any of the three kernels.

    A: (m, 2), B: (n, 2) -> returns (m, n)
    """
    if kernel == "linear":
        return A @ B.T
    if kernel == "rbf":
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b, computed without a python loop
        sq_a = (A ** 2).sum(axis=1)[:, None]
        sq_b = (B ** 2).sum(axis=1)[None, :]
        sq_dist = sq_a + sq_b - 2 * A @ B.T
        np.maximum(sq_dist, 0, out=sq_dist)  # guard against tiny negative fp noise
        return np.exp(-gamma * sq_dist)
    if kernel == "poly":
        return (gamma * (A @ B.T) + coef0) ** degree
    raise ValueError(f"Unknown kernel {kernel!r}")


def default_gamma(X: np.ndarray) -> float:
    """scikit-learn's 'scale' heuristic: 1 / (n_features * Var[X])."""
    var = float(X.var())
    return 1.0 / (2 * var) if var > 0 else 1.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """State of the SMO solver after one full pass over the training set."""

    points: np.ndarray      # (N, 2) — constant across snapshots
    labels: np.ndarray      # (N,) in {-1, +1} — constant across snapshots
    alpha: np.ndarray       # (N,) dual coefficients at this point in training
    b: float                # bias term
    decision: np.ndarray    # (N,) f(x_i) evaluated at this snapshot
    iteration: int          # sweep number; 0 = before any update (all alpha=0)
    n_changed: int = 0      # alpha pairs updated during this sweep
    active_pair: tuple[int, int] | None = None  # last (i, j) touched this sweep
    dual_objective: float = 0.0  # Σα − ½ αᵀ(yyᵀ⊙K)α — see _dual_objective()
    converged: bool = False

    @property
    def support_mask(self) -> np.ndarray:
        """Points with non-negligible dual weight — the support vectors."""
        return self.alpha > 1e-6

    @property
    def n_support(self) -> int:
        return int(self.support_mask.sum())

    @property
    def slack(self) -> np.ndarray:
        """Hinge slack xi_i = max(0, 1 - y_i f(x_i)) — margin violation per point."""
        return np.maximum(0.0, 1.0 - self.labels * self.decision)

    @property
    def title(self) -> str:
        if self.iteration == 0:
            return "Initialisation — all α = 0 (no support vectors yet)"
        tag = " ✓ converged" if self.converged else ""
        return f"Sweep {self.iteration} — {self.n_changed} α pairs updated{tag}"


# ---------------------------------------------------------------------------
# Simplified SMO
# ---------------------------------------------------------------------------

def _decision_function(
    K: np.ndarray, alpha: np.ndarray, y: np.ndarray, b: float
) -> np.ndarray:
    """f(x_i) for every training point, using the precomputed Gram matrix."""
    return (alpha * y) @ K + b


def _dual_objective(K: np.ndarray, alpha: np.ndarray, y: np.ndarray) -> float:
    """The SMO dual objective: Σᵢ αᵢ − ½ ΣᵢΣⱼ αᵢαⱼ yᵢyⱼ K(xᵢ, xⱼ).

    Every single-pair SMO update is chosen to increase (or leave unchanged)
    this exact quantity, so across a full training run it climbs
    monotonically toward its maximum — a clean, honest convergence curve to
    plot, unlike the primal hinge loss, which does not decrease monotonically
    under coordinate ascent with a randomly chosen second index.

    By strong duality, this maximum equals the minimum of the primal
    objective ½||w||² + C·Σ hinge(xᵢ) — so watching this curve rise is
    equivalent to watching the hinge loss (and the margin regulariser) fall.
    """
    reg = 0.5 * (alpha * y) @ K @ (alpha * y)
    return float(alpha.sum() - reg)


def fit(
    X: np.ndarray,
    y: np.ndarray,
    kernel: KernelName = "linear",
    C: float = 1.0,
    gamma: float | None = None,
    degree: int = 3,
    coef0: float = 1.0,
    tol: float = 1e-3,
    max_epochs: int = 30,
    seed: int = 0,
) -> list[Snapshot]:
    """Train a soft-margin SVM with simplified SMO, recording one snapshot per sweep.

    Parameters
    ----------
    X:          (N, 2) training points
    y:          (N,) labels in {-1, +1}
    kernel:     "linear" | "rbf" | "poly"
    C:          soft-margin penalty; larger C = less tolerance for violations
    gamma:      RBF/poly kernel width; None -> data-driven default
    degree:     polynomial kernel degree
    coef0:      polynomial kernel additive constant
    tol:        KKT violation tolerance used to decide which alphas to touch
    max_epochs: hard cap on sweeps AND the "no progress" patience (3 clean
                sweeps in a row with zero updates ends training early)
    seed:       random seed for the j-index heuristic and tie-breaking

    Returns
    -------
    List of Snapshot objects, index 0 = initial state (alpha = 0).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(X)
    g = gamma if gamma is not None else default_gamma(X)

    K = kernel_matrix(X, X, kernel, g, degree, coef0)
    rng = np.random.default_rng(seed)

    alpha = np.zeros(n, dtype=float)
    b = 0.0

    snapshots: list[Snapshot] = [
        Snapshot(
            points=X, labels=y, alpha=alpha.copy(), b=b,
            decision=_decision_function(K, alpha, y, b),
            iteration=0, dual_objective=_dual_objective(K, alpha, y),
        )
    ]

    no_change_streak = 0
    for epoch in range(1, max_epochs + 1):
        n_changed = 0
        active_pair: tuple[int, int] | None = None
        order = rng.permutation(n)

        for i in order:
            f_i = float((alpha * y) @ K[:, i] + b)
            E_i = f_i - y[i]

            # KKT check: is alpha_i violating optimality within tolerance?
            violates = (y[i] * E_i < -tol and alpha[i] < C) or \
                       (y[i] * E_i > tol and alpha[i] > 0)
            if not violates:
                continue

            # Pick a second index at random (Platt's simplified heuristic)
            j = int(rng.integers(0, n - 1))
            if j >= i:
                j += 1
            f_j = float((alpha * y) @ K[:, j] + b)
            E_j = f_j - y[j]

            alpha_i_old, alpha_j_old = alpha[i], alpha[j]
            if y[i] != y[j]:
                lo = max(0.0, alpha[j] - alpha[i])
                hi = min(C, C + alpha[j] - alpha[i])
            else:
                lo = max(0.0, alpha[i] + alpha[j] - C)
                hi = min(C, alpha[i] + alpha[j])
            if lo >= hi:
                continue

            eta = 2 * K[i, j] - K[i, i] - K[j, j]
            if eta >= 0:
                continue  # non-convex direction — skip this pair

            alpha_j_new = alpha[j] - y[j] * (E_i - E_j) / eta
            alpha_j_new = min(max(alpha_j_new, lo), hi)
            if abs(alpha_j_new - alpha_j_old) < 1e-7:
                continue

            alpha_i_new = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha_j_new)

            b1 = b - E_i - y[i] * (alpha_i_new - alpha_i_old) * K[i, i] \
                   - y[j] * (alpha_j_new - alpha_j_old) * K[i, j]
            b2 = b - E_j - y[i] * (alpha_i_new - alpha_i_old) * K[i, j] \
                   - y[j] * (alpha_j_new - alpha_j_old) * K[j, j]
            if 0 < alpha_i_new < C:
                b = b1
            elif 0 < alpha_j_new < C:
                b = b2
            else:
                b = (b1 + b2) / 2

            alpha[i], alpha[j] = alpha_i_new, alpha_j_new
            n_changed += 1
            active_pair = (int(i), int(j))

        converged = n_changed == 0
        no_change_streak = no_change_streak + 1 if converged else 0

        snapshots.append(
            Snapshot(
                points=X, labels=y, alpha=alpha.copy(), b=b,
                decision=_decision_function(K, alpha, y, b),
                iteration=epoch, n_changed=n_changed, active_pair=active_pair,
                dual_objective=_dual_objective(K, alpha, y),
                converged=(no_change_streak >= 3),
            )
        )

        if no_change_streak >= 3:
            break

    return snapshots
