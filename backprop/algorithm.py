"""
algorithm.py — From-scratch MLP trained with backpropagation

A small fully-connected network (2 -> ... -> 1) is trained by full-batch
gradient descent on a binary classification task. Every epoch is split
into the two sub-steps that make backprop what it is:

  forward  — push the data through the current weights, measure the loss
  backward — push the error back through the network to get gradients,
             then take a gradient-descent step

Every forward and backward sub-step is recorded as a Snapshot, together
with the network's prediction surface evaluated on a fixed background
grid, so the visualiser can replay the whole training run frame by
frame: decision boundary, hidden-neuron activations, and signed gradient
magnitudes on every edge.

Architecture and activations are deliberately fixed and simple:
tanh in every hidden layer (zero-centred, so the activation heat-maps
have a natural diverging colour scale), sigmoid on the single output
unit, binary cross-entropy loss. This keeps every line of the forward
and backward pass legible without hiding the calculus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Phase = Literal["init", "forward", "backward"]


@dataclass
class Snapshot:
    """State of the network at one point in time."""

    # Data — constant across snapshots
    points: np.ndarray   # (N, 2)
    targets: np.ndarray  # (N,)

    # Epoch number (1-indexed); 0 = initial random weights, pre-training
    epoch: int

    # Which sub-step produced this snapshot
    phase: Phase

    # Parameters at this instant (list of per-layer arrays)
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    # Gradients w.r.t. the loss — populated only on "backward" snapshots
    grad_weights: list[np.ndarray] | None = None
    grad_biases: list[np.ndarray] | None = None

    # Scalar metrics measured with the weights above, before any update
    loss: float = 0.0
    accuracy: float = 0.0

    # Prediction surface evaluated on a fixed background grid, shape (res, res).
    # Shared by reference between a forward snapshot and its paired backward
    # snapshot (same weights, no recomputation, no extra memory).
    grid_probs: np.ndarray | None = None

    # First-hidden-layer activations over the same grid, shape (res, res, n_units)
    hidden_activations: np.ndarray | None = None

    # True once the loss has stopped improving meaningfully
    converged: bool = False

    @property
    def title(self) -> str:
        if self.phase == "init":
            return "Initialisation — random weights"
        verb = "Forward pass" if self.phase == "forward" else "Backward pass (gradients)"
        tag = " ✓ converged" if self.converged else ""
        return f"Epoch {self.epoch} — {verb}{tag}"


@dataclass
class TrainingRun:
    """Full result of fit(): every snapshot plus the shared background grid."""

    snapshots: list[Snapshot]
    grid_x: np.ndarray  # (res, res) meshgrid X coordinates
    grid_y: np.ndarray  # (res, res) meshgrid Y coordinates
    layer_sizes: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip to avoid overflow in exp() for very confident (large |z|) predictions
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def _init_params(
    layer_sizes: list[int], rng: np.random.Generator
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Xavier/Glorot-uniform initialisation — keeps tanh activations from
    saturating at the start of training regardless of layer width."""
    weights, biases = [], []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        limit = np.sqrt(6.0 / (n_in + n_out))
        weights.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
        biases.append(np.zeros(n_out, dtype=float))
    return weights, biases


def _forward(
    X: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run X through every layer, keeping every pre- and post-activation.

    activations[0] is the input itself; activations[-1] is the sigmoid
    output. zs holds the pre-activation values needed by the backward pass.
    """
    activations = [X]
    zs: list[np.ndarray] = []
    a = X
    n_layers = len(weights)
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = a @ W + b
        zs.append(z)
        a = _sigmoid(z) if i == n_layers - 1 else np.tanh(z)
        activations.append(a)
    return zs, activations


def _backward(
    y: np.ndarray,
    activations: list[np.ndarray],
    weights: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Backpropagate the binary cross-entropy error through every layer.

    The output layer's local gradient collapses to (prediction - target) —
    the well-known simplification that falls out of pairing a sigmoid
    output with cross-entropy loss. Every earlier layer's local gradient
    is the upstream gradient projected back through W, scaled by the
    tanh derivative (1 - a^2), which is exactly the "gradient flowing
    backward" the visualisation shows on the network diagram.
    """
    n = y.shape[0]
    n_layers = len(weights)
    grad_w: list[np.ndarray | None] = [None] * n_layers
    grad_b: list[np.ndarray | None] = [None] * n_layers

    dz = activations[-1] - y.reshape(-1, 1)  # (N, 1)
    for i in reversed(range(n_layers)):
        a_prev = activations[i]
        grad_w[i] = a_prev.T @ dz / n
        grad_b[i] = dz.mean(axis=0)
        if i > 0:
            da_prev = dz @ weights[i].T
            dz = da_prev * (1.0 - activations[i] ** 2)  # d(tanh)/dz using the activation

    return grad_w, grad_b  # type: ignore[return-value]


def _bce(p: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(p.ravel(), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _accuracy(p: np.ndarray, y: np.ndarray) -> float:
    pred = (p.ravel() > 0.5).astype(int)
    return float((pred == y).mean())


def _make_grid(X: np.ndarray, res: int, pad: float = 0.7):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid_points


def _copy_params(params: list[np.ndarray]) -> list[np.ndarray]:
    return [p.copy() for p in params]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit(
    X: np.ndarray,
    y: np.ndarray,
    hidden_sizes: tuple[int, ...] = (4,),
    lr: float = 0.5,
    epochs: int = 150,
    seed: int = 0,
    tol: float = 5e-4,
    grid_res: int = 45,
) -> TrainingRun:
    """Train a 2 -> hidden_sizes -> 1 MLP on (X, y) with full-batch gradient
    descent, recording every forward/backward sub-step.

    Parameters
    ----------
    X, y:         training data, X shape (N, 2), y shape (N,) in {0, 1}
    hidden_sizes: widths of the hidden layers, e.g. (4,) or (8, 8)
    lr:           gradient-descent learning rate
    epochs:       hard cap on training epochs
    seed:         random seed for weight initialisation
    tol:          stop early once the loss improves by less than this for
                  several consecutive epochs (the network has converged).
                  Default (5e-4) is tuned so the "converged" state is
                  actually reachable within the UI's epoch-cap range for
                  most shapes/architectures -- an absolute tolerance as
                  tight as 1e-6 never fires within 300 epochs for any
                  shape here and leaves the whole converged/✓ branch dead.
    grid_res:     resolution of the background grid used for the decision
                  boundary contour and the hidden-activation heat-maps

    Returns
    -------
    TrainingRun with the full snapshot history:
      [init, forward_1, backward_1, forward_2, backward_2, ..., backward_T]
    """
    rng = np.random.default_rng(seed)
    layer_sizes = [X.shape[1], *hidden_sizes, 1]
    weights, biases = _init_params(layer_sizes, rng)
    xx, yy, grid_points = _make_grid(X, res=grid_res)

    def _grid_eval(w, b):
        _, grid_acts = _forward(grid_points, w, b)
        probs = grid_acts[-1].reshape(xx.shape)
        hidden1 = grid_acts[1].reshape(xx.shape[0], xx.shape[1], -1)
        return probs, hidden1

    _, acts0 = _forward(X, weights, biases)
    loss0 = _bce(acts0[-1], y)
    acc0 = _accuracy(acts0[-1], y)
    probs0, hidden0 = _grid_eval(weights, biases)

    snapshots: list[Snapshot] = [
        Snapshot(
            points=X,
            targets=y,
            epoch=0,
            phase="init",
            weights=_copy_params(weights),
            biases=_copy_params(biases),
            loss=loss0,
            accuracy=acc0,
            grid_probs=probs0,
            hidden_activations=hidden0,
        )
    ]

    prev_loss = loss0
    stable_epochs = 0
    for epoch in range(1, epochs + 1):
        zs, acts = _forward(X, weights, biases)
        p = acts[-1]
        loss = _bce(p, y)
        acc = _accuracy(p, y)
        probs, hidden = _grid_eval(weights, biases)

        forward_snap = Snapshot(
            points=X,
            targets=y,
            epoch=epoch,
            phase="forward",
            weights=_copy_params(weights),
            biases=_copy_params(biases),
            loss=loss,
            accuracy=acc,
            grid_probs=probs,
            hidden_activations=hidden,
        )
        snapshots.append(forward_snap)

        grad_w, grad_b = _backward(y, acts, weights)

        backward_snap = Snapshot(
            points=X,
            targets=y,
            epoch=epoch,
            phase="backward",
            weights=_copy_params(weights),
            biases=_copy_params(biases),
            grad_weights=_copy_params(grad_w),
            grad_biases=_copy_params(grad_b),
            loss=loss,
            accuracy=acc,
            # Same weights as forward_snap -> identical prediction surface.
            # Shared by reference: no recomputation, no extra memory.
            grid_probs=probs,
            hidden_activations=hidden,
        )
        snapshots.append(backward_snap)

        weights = [w - lr * gw for w, gw in zip(weights, grad_w)]
        biases = [b - lr * gb for b, gb in zip(biases, grad_b)]

        # Early stop once loss has plateaued for a few consecutive epochs
        if abs(prev_loss - loss) < tol:
            stable_epochs += 1
        else:
            stable_epochs = 0
        prev_loss = loss

        if stable_epochs >= 3 and epoch > 5:
            forward_snap.converged = True
            backward_snap.converged = True
            break

    return TrainingRun(snapshots=snapshots, grid_x=xx, grid_y=yy, layer_sizes=layer_sizes)
