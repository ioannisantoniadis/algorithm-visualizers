"""
tree.py — minimal CART decision tree (Gini impurity, axis-aligned splits)

This is the single-tree building block used internally by the Random
Forest ensemble in algorithm.py. It is deliberately bare-bones — no
pruning, no cost-complexity tuning, no categorical-feature handling.
Production decision-tree mechanics are not the point of this repo (see
the standalone CART implementation for that); what matters here are the
two hooks the *forest* needs that a lone tree wouldn't otherwise have:

  - `max_features`: at every single split, only a random subset of the
    features is even considered as a candidate. This is the difference
    between "Bagged Trees" (bootstrap only) and a genuine Random Forest
    (bootstrap + per-split feature subsampling) — it decorrelates trees
    that would otherwise all discover the same dominant feature first.
    With only two input features in this app, restricting to one
    candidate per split also costs real per-split accuracy, so the
    trade-off doesn't always favour subsampling here — see algorithm.py.
  - every accepted split reports its impurity decrease so the forest can
    accumulate a feature-importance score across the whole ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    """One node of a binary decision tree. Leaves carry a class
    distribution; internal nodes carry a single axis-aligned test
    `x[feature] <= threshold`."""

    is_leaf: bool
    proba: np.ndarray          # (n_classes,) empirical class distribution at this node
    prediction: int            # argmax(proba) — cached so predict() never re-derives it
    feature: int = -1
    threshold: float = 0.0
    left: "Node | None" = None
    right: "Node | None" = None


def _gini(counts: np.ndarray) -> float:
    """Gini impurity 1 - sum(p_c^2) from raw class counts (not yet normalised)."""
    n = counts.sum()
    if n <= 0:
        return 0.0
    p = counts / n
    return 1.0 - float(np.sum(p ** 2))


def _best_split(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    feature_candidates: np.ndarray,
    min_samples_leaf: int,
) -> tuple[float, int, float] | None:
    """Scan every candidate feature for the threshold that most reduces
    Gini impurity. Returns (impurity_decrease, feature, threshold), or
    None if no split satisfies min_samples_leaf on both sides.

    The sweep is the standard O(n log n) trick: sort by the feature once,
    then walk the sorted order maintaining running left/right class
    counts instead of recomputing impurity from scratch at every cut.
    """
    n = len(y)
    parent_counts = np.bincount(y, minlength=n_classes).astype(float)
    parent_impurity = _gini(parent_counts)

    best: tuple[float, int, float] | None = None
    for f in feature_candidates:
        order = np.argsort(X[:, f], kind="mergesort")
        xf = X[order, f]
        yf = y[order]

        left_counts = np.zeros(n_classes)
        right_counts = parent_counts.copy()

        for i in range(n - 1):
            c = yf[i]
            left_counts[c] += 1
            right_counts[c] -= 1

            if xf[i] == xf[i + 1]:
                continue  # can't place a threshold between identical values

            n_left, n_right = i + 1, n - (i + 1)
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue

            weighted = (n_left * _gini(left_counts) + n_right * _gini(right_counts)) / n
            decrease = parent_impurity - weighted
            if best is None or decrease > best[0]:
                threshold = (xf[i] + xf[i + 1]) / 2.0
                best = (decrease, int(f), float(threshold))

    return best


def _build(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    depth: int,
    max_depth: int,
    min_samples_leaf: int,
    max_features: int,
    rng: np.random.Generator,
    importance: np.ndarray,
    n_total: int,
) -> Node:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    n = len(y)
    proba = counts / max(n, 1)
    majority = int(np.argmax(counts))

    is_pure = _gini(counts) == 0.0
    if depth >= max_depth or n < 2 * min_samples_leaf or is_pure:
        return Node(is_leaf=True, proba=proba, prediction=majority)

    n_features = X.shape[1]
    k = max(1, min(max_features, n_features))
    candidates = rng.choice(n_features, size=k, replace=False)

    split = _best_split(X, y, n_classes, candidates, min_samples_leaf)
    if split is None:
        return Node(is_leaf=True, proba=proba, prediction=majority)

    decrease, feature, threshold = split
    # Importance credit is weighted by how much of the *tree's* data
    # reaches this node — a split near the root that cleanly separates
    # everything matters more than one deep in a rare branch.
    importance[feature] += (n / n_total) * decrease

    mask = X[:, feature] <= threshold
    left = _build(X[mask], y[mask], n_classes, depth + 1, max_depth,
                  min_samples_leaf, max_features, rng, importance, n_total)
    right = _build(X[~mask], y[~mask], n_classes, depth + 1, max_depth,
                    min_samples_leaf, max_features, rng, importance, n_total)
    return Node(is_leaf=False, proba=proba, prediction=majority,
                feature=feature, threshold=threshold, left=left, right=right)


def build_tree(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    max_depth: int,
    min_samples_leaf: int,
    max_features: int,
    rng: np.random.Generator,
) -> tuple[Node, np.ndarray, int, int]:
    """Grow one CART tree on (X, y) with per-split feature subsampling.

    Returns (root, feature_importance, depth_reached, n_leaves).
    feature_importance has shape (n_features,) and is *not* normalised —
    the caller accumulates it across every tree in the forest before
    normalising once at the end.
    """
    importance = np.zeros(X.shape[1])
    root = _build(X, y, n_classes, 0, max_depth, min_samples_leaf,
                  max_features, rng, importance, len(y))
    return root, importance, _depth(root), _n_leaves(root)


def _depth(node: Node) -> int:
    if node.is_leaf:
        return 0
    return 1 + max(_depth(node.left), _depth(node.right))


def _n_leaves(node: Node) -> int:
    if node.is_leaf:
        return 1
    return _n_leaves(node.left) + _n_leaves(node.right)


def predict(node: Node, X: np.ndarray) -> np.ndarray:
    """Vectorised class prediction for every row of X.

    Rather than looping row by row, each recursive call carries the set
    of row indices still "alive" at that node and narrows it with a
    single boolean mask — so a tree of depth d costs O(d) array
    operations, not O(n) Python-level tree walks.
    """
    out = np.empty(len(X), dtype=int)
    _predict_into(node, X, np.arange(len(X)), out)
    return out


def _predict_into(node: Node, X: np.ndarray, idx: np.ndarray, out: np.ndarray) -> None:
    if len(idx) == 0:
        return
    if node.is_leaf:
        out[idx] = node.prediction
        return
    values = X[idx, node.feature]
    go_left = values <= node.threshold
    _predict_into(node.left, X, idx[go_left], out)
    _predict_into(node.right, X, idx[~go_left], out)
