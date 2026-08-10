"""
algorithm.py — From-scratch Perceptron and Gradient Descent, with full history

Two related learning stories share this module because they answer the same
question — "how do we find a separating line?" — with two different update
rules:

  perceptron_fit()
      The classical mistake-driven rule: scan the data, and whenever a point
      is on the wrong side of the current boundary, nudge the weight vector
      directly towards that point's label. No loss function, no learning
      rate schedule — just "if wrong, correct". Every weight update is
      recorded as a Snapshot so the visualiser can replay the boundary
      rotating step by step.

  gradient_descent_fit()
      The smooth alternative: define a differentiable loss (logistic loss)
      over the weight vector and descend its gradient. Three optimisers run
      side by side from the same starting point — vanilla gradient descent,
      momentum, and Adam — so their trajectories on the same loss surface
      can be compared directly.

To keep the loss surface plottable as a 2-D contour, the gradient-descent
weight vector has no bias term: the data is generated symmetrically about
the origin (see data.py), so a bias-free line through the origin is exactly
the class of separators worth searching over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Part 1 — Perceptron
# ---------------------------------------------------------------------------

PerceptronPhase = Literal["init", "update", "epoch_end"]


@dataclass
class PerceptronSnapshot:
    """State of the Perceptron at one meaningful point in its history.

    Only frames where something changes are recorded: the initial state,
    every misclassification-triggered weight update, and the summary at the
    end of each epoch. This keeps the step-through short and meaningful
    instead of replaying every "already correct, no-op" check.
    """

    points: np.ndarray      # all data points, constant across snapshots (N×2)
    labels: np.ndarray      # true labels in {-1, +1} (N,)
    weights: np.ndarray     # [bias, w1, w2] after this snapshot
    prev_weights: np.ndarray  # weights immediately before this snapshot's update
    epoch: int              # 1-indexed epoch; 0 = initialisation frame
    point_idx: int           # index of the point that triggered this update; -1 if none
    phase: PerceptronPhase
    n_updates: int           # cumulative weight updates so far
    n_mistakes_epoch: int    # mistakes made in the current/just-finished epoch
    converged: bool = False  # True once a full epoch made zero mistakes

    @property
    def title(self) -> str:
        if self.phase == "init":
            return "Initialisation — w = (bias=0, w₁=0, w₂=0)"
        if self.phase == "update":
            return (
                f"Epoch {self.epoch} — point #{self.point_idx} misclassified "
                f"→ update #{self.n_updates}"
            )
        tag = "0 mistakes, converged ✓" if self.converged else f"{self.n_mistakes_epoch} mistakes"
        return f"Epoch {self.epoch} complete — {tag}"


def _augment(X: np.ndarray) -> np.ndarray:
    """Prepend a constant 1 column so the bias can be learned like any weight."""
    return np.hstack([np.ones((len(X), 1)), X])


def predict_sign(w: np.ndarray, x_aug: np.ndarray) -> float:
    """+1/-1 prediction for an augmented point. Ties (dot == 0) count as -1,
    which forces the perceptron to keep pushing until there is a real margin.
    """
    return 1.0 if np.dot(w, x_aug) > 0 else -1.0


def accuracy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of points correctly classified by weights w = [bias, w1, w2]."""
    Xa = _augment(X)
    preds = np.where(Xa @ w > 0, 1.0, -1.0)
    return float(np.mean(preds == y))


def perceptron_fit(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 1.0,
    max_epochs: int = 15,
    seed: int = 0,
) -> list[PerceptronSnapshot]:
    """Run the Perceptron learning rule and return every meaningful snapshot.

    Parameters
    ----------
    X:          data matrix, shape (N, 2)
    y:          labels in {-1, +1}, shape (N,)
    lr:         learning rate (only scales the update step; the perceptron
                converges for any lr > 0 on separable data)
    max_epochs: hard cap on full passes over the data
    seed:       controls the per-epoch shuffle order

    Returns
    -------
    [init, update, update, ..., epoch_end, update, ..., epoch_end]
    The final snapshot's `converged` flag tells you whether a separating
    line was found within max_epochs.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    Xa = _augment(X)
    w = np.zeros(3)
    n_updates = 0

    snapshots: list[PerceptronSnapshot] = [
        PerceptronSnapshot(
            points=X, labels=y, weights=w.copy(), prev_weights=w.copy(),
            epoch=0, point_idx=-1, phase="init",
            n_updates=0, n_mistakes_epoch=0, converged=False,
        )
    ]

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(n)
        mistakes = 0
        for idx in order:
            pred = predict_sign(w, Xa[idx])
            if pred != y[idx]:
                prev_w = w.copy()
                w = w + lr * y[idx] * Xa[idx]
                mistakes += 1
                n_updates += 1
                snapshots.append(
                    PerceptronSnapshot(
                        points=X, labels=y, weights=w.copy(), prev_weights=prev_w,
                        epoch=epoch, point_idx=int(idx), phase="update",
                        n_updates=n_updates, n_mistakes_epoch=mistakes, converged=False,
                    )
                )

        converged = mistakes == 0
        snapshots.append(
            PerceptronSnapshot(
                points=X, labels=y, weights=w.copy(), prev_weights=w.copy(),
                epoch=epoch, point_idx=-1, phase="epoch_end",
                n_updates=n_updates, n_mistakes_epoch=mistakes, converged=converged,
            )
        )
        if converged:
            break

    return snapshots


# ---------------------------------------------------------------------------
# Part 2 — Gradient descent on the logistic loss (SGD / Momentum / Adam)
# ---------------------------------------------------------------------------

@dataclass
class GDSnapshot:
    """Joint state of all three optimisers after `step` gradient updates.

    Each optimiser gets its own weight vector, loss, gradient norm, and full
    path so far, all starting from the same w0 — this is what makes the
    trajectories directly comparable on one loss surface.
    """

    step: int
    w_vanilla: np.ndarray
    w_momentum: np.ndarray
    w_adam: np.ndarray
    loss_vanilla: float
    loss_momentum: float
    loss_adam: float
    grad_norm_vanilla: float
    grad_norm_momentum: float
    grad_norm_adam: float
    path_vanilla: np.ndarray    # (step+1, 2) — full trajectory up to this step
    path_momentum: np.ndarray
    path_adam: np.ndarray

    @property
    def title(self) -> str:
        return (
            f"Step {self.step} — loss:  vanilla {self.loss_vanilla:.3f}  ·  "
            f"momentum {self.loss_momentum:.3f}  ·  adam {self.loss_adam:.3f}"
        )


def _logistic_loss_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean logistic loss and its gradient at w = (w1, w2), no bias.

    L(w) = mean( log(1 + exp(-y · (X w))) )
    ∇L(w) = mean( -y · x · sigmoid(-y · (X w)) )

    Written with np.logaddexp / a shifted sigmoid so it never overflows,
    even as ||w|| grows large chasing a separable dataset's vanishing loss.
    """
    z = y * (X @ w)                      # (n,) — signed margin
    loss = float(np.mean(np.logaddexp(0.0, -z)))
    s = 1.0 / (1.0 + np.exp(np.clip(z, -30, 30)))   # sigmoid(-z), stable
    grad = -(X * (y * s)[:, None]).mean(axis=0)
    return loss, grad


def loss_grid(
    X: np.ndarray,
    y: np.ndarray,
    w1_range: tuple[float, float],
    w2_range: tuple[float, float],
    resolution: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the logistic loss over a (w1, w2) grid for a contour plot."""
    w1s = np.linspace(*w1_range, resolution)
    w2s = np.linspace(*w2_range, resolution)
    W1, W2 = np.meshgrid(w1s, w2s)                       # (res, res)

    # z[i, j, k] = y_k * (w1_ij * x1_k + w2_ij * x2_k)
    z = y[None, None, :] * (
        W1[:, :, None] * X[None, None, :, 0] + W2[:, :, None] * X[None, None, :, 1]
    )
    Z = np.mean(np.logaddexp(0.0, -z), axis=2)           # (res, res)
    return w1s, w2s, Z


def accuracy_nobias(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction correctly classified by a bias-free weight vector w = (w1, w2)."""
    preds = np.where(X @ w > 0, 1.0, -1.0)
    return float(np.mean(preds == y))


def gradient_descent_fit(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.5,
    adam_lr: float = 0.3,
    momentum_beta: float = 0.9,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    n_steps: int = 50,
    init_scale: float = 2.5,
    seed: int = 7,
) -> list[GDSnapshot]:
    """Run vanilla GD, momentum, and Adam from the same random w0.

    All three take a *full-batch* gradient step each iteration — "vanilla"
    here means "no momentum, no adaptive scaling", not single-sample SGD.
    A full batch keeps the three trajectories comparable: any difference in
    their paths comes purely from the update rule, not from gradient noise.

    Parameters
    ----------
    lr:            step size for vanilla GD and momentum
    adam_lr:       Adam's step size (kept separate — Adam's adaptive scaling
                   means the same lr behaves very differently)
    momentum_beta: velocity decay for classical momentum
    n_steps:       number of gradient updates to run
    init_scale:    all three optimisers start from the same
                   w0 ~ Uniform(-init_scale, init_scale)²
    seed:          controls only the random initial weight vector
    """
    rng = np.random.default_rng(seed)
    w0 = rng.uniform(-init_scale, init_scale, size=2)

    w_v, w_m, w_a = w0.copy(), w0.copy(), w0.copy()
    v_velocity = np.zeros(2)          # momentum velocity
    m_adam = np.zeros(2)               # Adam 1st moment
    v_adam = np.zeros(2)               # Adam 2nd moment

    loss_v, grad_v = _logistic_loss_grad(X, y, w_v)
    loss_m, grad_m = _logistic_loss_grad(X, y, w_m)
    loss_a, grad_a = _logistic_loss_grad(X, y, w_a)

    path_v, path_m, path_a = [w_v.copy()], [w_m.copy()], [w_a.copy()]

    snapshots: list[GDSnapshot] = [
        GDSnapshot(
            step=0,
            w_vanilla=w_v.copy(), w_momentum=w_m.copy(), w_adam=w_a.copy(),
            loss_vanilla=loss_v, loss_momentum=loss_m, loss_adam=loss_a,
            grad_norm_vanilla=float(np.linalg.norm(grad_v)),
            grad_norm_momentum=float(np.linalg.norm(grad_m)),
            grad_norm_adam=float(np.linalg.norm(grad_a)),
            path_vanilla=np.array(path_v), path_momentum=np.array(path_m),
            path_adam=np.array(path_a),
        )
    ]

    for t in range(1, n_steps + 1):
        # ---- Vanilla gradient descent -------------------------------------
        _, grad_v = _logistic_loss_grad(X, y, w_v)
        w_v = w_v - lr * grad_v
        loss_v, _ = _logistic_loss_grad(X, y, w_v)

        # ---- Classical momentum --------------------------------------------
        _, grad_m = _logistic_loss_grad(X, y, w_m)
        v_velocity = momentum_beta * v_velocity + grad_m
        w_m = w_m - lr * v_velocity
        loss_m, _ = _logistic_loss_grad(X, y, w_m)

        # ---- Adam -----------------------------------------------------------
        _, grad_a = _logistic_loss_grad(X, y, w_a)
        m_adam = adam_beta1 * m_adam + (1 - adam_beta1) * grad_a
        v_adam = adam_beta2 * v_adam + (1 - adam_beta2) * (grad_a ** 2)
        m_hat = m_adam / (1 - adam_beta1 ** t)
        v_hat = v_adam / (1 - adam_beta2 ** t)
        w_a = w_a - adam_lr * m_hat / (np.sqrt(v_hat) + adam_eps)
        loss_a, _ = _logistic_loss_grad(X, y, w_a)

        path_v.append(w_v.copy())
        path_m.append(w_m.copy())
        path_a.append(w_a.copy())

        snapshots.append(
            GDSnapshot(
                step=t,
                w_vanilla=w_v.copy(), w_momentum=w_m.copy(), w_adam=w_a.copy(),
                loss_vanilla=loss_v, loss_momentum=loss_m, loss_adam=loss_a,
                grad_norm_vanilla=float(np.linalg.norm(grad_v)),
                grad_norm_momentum=float(np.linalg.norm(grad_m)),
                grad_norm_adam=float(np.linalg.norm(grad_a)),
                path_vanilla=np.array(path_v), path_momentum=np.array(path_m),
                path_adam=np.array(path_a),
            )
        )

    return snapshots
