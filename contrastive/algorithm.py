"""
algorithm.py — From-scratch SimCLR-style contrastive encoder

A small fully-connected network (2 -> ... -> 2) maps each raw 2-D point to
a 2-D embedding, which is then L2-normalised onto the unit circle. Two
augmented views of the *same* point form a positive pair; every other view
in the batch is a negative. The encoder is trained with the NT-Xent
(normalised temperature-scaled cross-entropy) loss — pull positives
together, push negatives apart, entirely without labels.

Every training step is split into the two sub-steps that make backprop
what it is, exactly as in the sibling backprop-viz project:

  forward  — augment a batch, embed both views, measure the NT-Xent loss
  backward — propagate the loss gradient back through normalisation and
             every dense layer, then take a gradient-descent step

Because the embedding dimension is fixed at 2, the *entire* representation
lives in the plane and, after normalisation, on the unit circle — nothing
is hidden by a t-SNE/PCA projection. What you see in the "Normalised
embedding" panel is exactly what NT-Xent optimises.

Deriving the backward pass
---------------------------
Let E be the (N, 2) matrix of L2-normalised embeddings for one batch of N
augmented views (N = 2 * batch_size). Cosine similarity is just the Gram
matrix S = E @ E.T since every row of E has unit norm. Logits are S / tau.
For anchor i, NT-Xent is a softmax cross-entropy over the N-1 other rows
(self-similarity is masked out), with the *single* positive index p(i)
as the target class:

    L = (1/N) * sum_i  -log( exp(logit[i, p(i)]) / sum_{k != i} exp(logit[i, k]) )

dL/dlogit[i, k] is the standard softmax-minus-one-hot gradient, call it
G (N, N). Because logits = S / tau, dL/dS = G / tau =: M. The subtlety is
that S is symmetric (S_ij = S_ji = e_i . e_j) yet M is *not* symmetric (row
i and row j give different gradients for the same entry). Differentiating
carefully w.r.t. each embedding row shows the two contributions add:

    dL/dE = (M + M.T) @ E

From there, backprop through the L2 normalisation (a standard
"normalize" Jacobian) and then through the plain dense/tanh layers, same
pattern as backprop.algorithm._backward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .data import augment

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Phase = Literal["init", "forward", "backward"]


@dataclass
class Snapshot:
    """State of the encoder at one point in training."""

    # Full dataset — constant across snapshots
    points: np.ndarray  # (N_total, 2)
    labels: np.ndarray  # (N_total,) ground-truth cluster id, for colouring only

    # Step number (1-indexed); 0 = initial random weights, pre-training
    step: int
    phase: Phase

    # Parameters at this instant
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    grad_weights: list[np.ndarray] | None = None
    grad_biases: list[np.ndarray] | None = None

    # Normalised embedding of every point in the dataset (no augmentation) —
    # "where does the whole dataset currently live in embedding space"
    embed_full: np.ndarray | None = None

    # This step's batch: which base points were sampled, their two augmented
    # views in raw input space, and those views' normalised embeddings
    batch_idx: np.ndarray | None = None
    view_a: np.ndarray | None = None
    view_b: np.ndarray | None = None
    embed_a: np.ndarray | None = None
    embed_b: np.ndarray | None = None

    # Full (2B, 2B) cosine-similarity matrix for this step's batch
    sim_matrix: np.ndarray | None = None

    # Scalar metrics, computed with the weights above before any update
    loss: float = 0.0
    pos_sim: float = 0.0  # mean cosine similarity of positive pairs
    neg_sim: float = 0.0  # mean cosine similarity of negative pairs

    converged: bool = False

    @property
    def title(self) -> str:
        if self.phase == "init":
            return "Initialisation — random encoder, before any training"
        verb = "Forward pass (embed + NT-Xent loss)" if self.phase == "forward" else "Backward pass (gradients)"
        tag = " ✓ converged" if self.converged else ""
        return f"Step {self.step} — {verb}{tag}"


@dataclass
class TrainingRun:
    """Full result of fit(): every snapshot plus fixed run metadata."""

    snapshots: list[Snapshot]
    layer_sizes: list[int]
    tau: float


# ---------------------------------------------------------------------------
# Math helpers — encoder forward/backward
# ---------------------------------------------------------------------------

def _init_params(layer_sizes: list[int], rng: np.random.Generator):
    """Xavier/Glorot-uniform initialisation, same rationale as backprop-viz:
    keeps tanh activations from saturating regardless of layer width."""
    weights, biases = [], []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        limit = np.sqrt(6.0 / (n_in + n_out))
        weights.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
        biases.append(np.zeros(n_out, dtype=float))
    return weights, biases


def _forward(X: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]):
    """Push X through every layer. Hidden layers use tanh; the final layer
    is linear (no squashing) — its raw output is what gets L2-normalised
    into the embedding, so it must be free to point in any direction."""
    activations = [X]
    a = X
    n_layers = len(weights)
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = a @ W + b
        a = np.tanh(z) if i < n_layers - 1 else z
        activations.append(a)
    return activations


def _normalise_rows(Z: np.ndarray, eps: float = 1e-8):
    norm = np.linalg.norm(Z, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return Z / norm, norm


def _embed(X: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]):
    """Full forward pass to a unit-norm embedding. Returns everything the
    backward pass needs: intermediate activations, raw (pre-norm) output,
    the norm used to rescale it, and the final normalised embedding."""
    activations = _forward(X, weights, biases)
    z_raw = activations[-1]
    e, norm = _normalise_rows(z_raw)
    return activations, z_raw, norm, e


def _nt_xent(E: np.ndarray, tau: float):
    """NT-Xent loss and its gradient w.r.t. the embedding matrix E.

    E: (N, d) L2-normalised embeddings, rows [0:B] and [B:2B] are the two
    augmented views of the same B base points in matching order, i.e. row
    i's positive is row (i + B) mod 2B.

    Returns
    -------
    loss:     scalar NT-Xent loss (already averaged over the N anchors)
    dE:       (N, d) dLoss/dE
    sim:      (N, N) raw cosine-similarity matrix (for the heat-map view)
    pos_sim:  mean cosine similarity across positive pairs
    neg_sim:  mean cosine similarity across negative pairs
    """
    n = E.shape[0]
    half = n // 2
    pos_idx = np.concatenate([np.arange(half, n), np.arange(0, half)])

    sim = E @ E.T
    logits = sim / tau

    diag = np.eye(n, dtype=bool)
    logits_masked = np.where(diag, -np.inf, logits)

    row_max = logits_masked.max(axis=1, keepdims=True)
    exp_l = np.exp(logits_masked - row_max)          # diag entries -> 0
    sum_exp = exp_l.sum(axis=1, keepdims=True)

    pos_logit = logits[np.arange(n), pos_idx]
    log_softmax_pos = pos_logit - row_max.ravel() - np.log(sum_exp.ravel())
    loss = float(-log_softmax_pos.mean())

    # dL/dlogits = softmax - one_hot(positive), averaged over the N anchors
    softmax = exp_l / sum_exp
    G = softmax.copy()
    G[np.arange(n), pos_idx] -= 1.0
    G /= n
    G[diag] = 0.0  # no self-similarity term

    M = G / tau  # dL/dS
    dE = (M + M.T) @ E  # see module docstring for the derivation

    pos_sim = float(sim[np.arange(n), pos_idx].mean())
    neg_mask = ~diag
    neg_mask[np.arange(n), pos_idx] = False
    neg_sim = float(sim[neg_mask].mean())

    return loss, dE, sim, pos_sim, neg_sim


def _backward_through_encoder(
    dE: np.ndarray,
    E: np.ndarray,
    norm: np.ndarray,
    activations: list[np.ndarray],
    weights: list[np.ndarray],
):
    """Chain dL/dE back through L2-normalisation, then through every dense
    layer (linear final layer, tanh hidden layers) — the same layer-by-
    layer loop as backprop.algorithm._backward, just with a different
    starting gradient."""
    # Jacobian of row-wise L2 normalisation: de/dz = (I - e e^T) / ||z||.
    # Contracting with the upstream gradient dE gives, per row:
    #   dz = (dE - (dE . e) e) / ||z||
    dot = np.sum(dE * E, axis=1, keepdims=True)
    dz = (dE - dot * E) / norm  # gradient w.r.t. the final (linear) layer's z

    n_layers = len(weights)
    grad_w: list[np.ndarray | None] = [None] * n_layers
    grad_b: list[np.ndarray | None] = [None] * n_layers

    for i in reversed(range(n_layers)):
        a_prev = activations[i]
        # dE already carried the 1/N averaging from _nt_xent, so these are
        # plain sums over the batch dimension, not means.
        grad_w[i] = a_prev.T @ dz
        grad_b[i] = dz.sum(axis=0)
        if i > 0:
            da_prev = dz @ weights[i].T
            dz = da_prev * (1.0 - activations[i] ** 2)  # d(tanh)/dz

    return grad_w, grad_b  # type: ignore[return-value]


def _copy_params(params: list[np.ndarray]) -> list[np.ndarray]:
    return [p.copy() for p in params]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit(
    X: np.ndarray,
    labels: np.ndarray,
    hidden_sizes: tuple[int, ...] = (16,),
    tau: float = 0.2,
    batch_size: int = 16,
    lr: float = 0.5,
    steps: int = 80,
    aug_strength: float = 0.5,
    aug_kind: str = "all",
    seed: int = 0,
    tol: float = 1e-5,
) -> TrainingRun:
    """Train a 2 -> hidden_sizes -> 2 encoder on X with NT-Xent contrastive
    loss, recording every forward/backward sub-step.

    Parameters
    ----------
    X, labels:    dataset, X shape (N, 2); labels are ground truth used only
                  for colouring plots — never touched by the algorithm
    hidden_sizes: widths of the hidden layers, e.g. (16,) or (8, 8)
    tau:          NT-Xent temperature; smaller -> sharper, more punishing
                  softmax over negatives
    batch_size:   number of *base* points sampled each step (each yields
                  two augmented views, so the loss sees 2 * batch_size rows)
    lr:           gradient-descent learning rate
    steps:        number of mini-batch training steps
    aug_strength: strength in [0, 1] passed to data.augment
    aug_kind:     one of data.AUG_KEYS
    seed:         random seed for weight init, batch sampling and augmentation
    tol:          early-stop once the loss improves by less than this for
                  several consecutive steps

    Returns
    -------
    TrainingRun with the full snapshot history:
      [init, forward_1, backward_1, forward_2, backward_2, ..., backward_T]
    """
    n_total = len(X)
    batch_size = int(min(batch_size, n_total))
    rng = np.random.default_rng(seed)

    layer_sizes = [2, *hidden_sizes, 2]
    weights, biases = _init_params(layer_sizes, rng)

    def _full_embedding(w, b):
        _, _, _, e = _embed(X, w, b)
        return e

    snapshots: list[Snapshot] = [
        Snapshot(
            points=X,
            labels=labels,
            step=0,
            phase="init",
            weights=_copy_params(weights),
            biases=_copy_params(biases),
            embed_full=_full_embedding(weights, biases),
        )
    ]

    prev_loss = None
    stable_steps = 0

    for step in range(1, steps + 1):
        batch_idx = rng.choice(n_total, size=batch_size, replace=False)
        base = X[batch_idx]
        view_a = augment(base, rng, strength=aug_strength, kind=aug_kind)
        view_b = augment(base, rng, strength=aug_strength, kind=aug_kind)
        batch_X = np.vstack([view_a, view_b])  # rows [0:B]=A, [B:2B]=B

        activations, z_raw, norm, E = _embed(batch_X, weights, biases)
        loss, dE, sim, pos_sim, neg_sim = _nt_xent(E, tau)
        embed_full = _full_embedding(weights, biases)

        forward_snap = Snapshot(
            points=X,
            labels=labels,
            step=step,
            phase="forward",
            weights=_copy_params(weights),
            biases=_copy_params(biases),
            embed_full=embed_full,
            batch_idx=batch_idx,
            view_a=view_a,
            view_b=view_b,
            embed_a=E[:batch_size],
            embed_b=E[batch_size:],
            sim_matrix=sim,
            loss=loss,
            pos_sim=pos_sim,
            neg_sim=neg_sim,
        )
        snapshots.append(forward_snap)

        grad_w, grad_b = _backward_through_encoder(dE, E, norm, activations, weights)

        backward_snap = Snapshot(
            points=X,
            labels=labels,
            step=step,
            phase="backward",
            weights=_copy_params(weights),
            biases=_copy_params(biases),
            grad_weights=_copy_params(grad_w),
            grad_biases=_copy_params(grad_b),
            # Same weights as forward_snap -> identical embeddings; shared
            # by reference so no recomputation and no extra memory.
            embed_full=embed_full,
            batch_idx=batch_idx,
            view_a=view_a,
            view_b=view_b,
            embed_a=E[:batch_size],
            embed_b=E[batch_size:],
            sim_matrix=sim,
            loss=loss,
            pos_sim=pos_sim,
            neg_sim=neg_sim,
        )
        snapshots.append(backward_snap)

        weights = [w - lr * gw for w, gw in zip(weights, grad_w)]
        biases = [b - lr * gb for b, gb in zip(biases, grad_b)]

        if prev_loss is not None and abs(prev_loss - loss) < tol:
            stable_steps += 1
        else:
            stable_steps = 0
        prev_loss = loss

        if stable_steps >= 5 and step > 10:
            forward_snap.converged = True
            backward_snap.converged = True
            break

    return TrainingRun(snapshots=snapshots, layer_sizes=layer_sizes, tau=tau)
