"""
algorithm.py — From-scratch Random Forest: bagging + feature subsampling +
majority vote, with a full tree-by-tree training history.

A Random Forest is two independent randomisation tricks stacked on top of
plain decision trees (see tree.py for the CART building block):

  1. Bagging — each tree trains on a bootstrap resample of the training
     set (sampled with replacement, same size as the original by default).
     Every tree therefore sees a slightly different dataset, so their
     errors are only weakly correlated.
  2. Feature subsampling — each *split* (not each tree) is only allowed to
     consider a random subset of the features as candidates. With just
     two input features this means max_features=1 forces every split to
     commit to a single, randomly-chosen axis, which measurably
     decorrelates the trees: without it, every tree in the bag tends to
     discover the same strongest feature first and look nearly identical
     (this is "Bagged Trees", not yet a Random Forest).

     Decorrelation is not free, though: with only two candidate features
     to begin with, dropping to max_features=1 throws away half the
     information available at every split, which raises each tree's own
     bias. In this two-feature toy setting that bias cost usually
     outweighs the variance win, so out-of-bag error with max_features=1
     is often *not* lower than with max_features=2 — the opposite of
     the folklore "subsampling always helps". It reliably wins in real
     Random Forests because production feature sets have dozens to
     hundreds of (often redundant) columns, so losing some candidates
     per split costs little bias while still buying decorrelation. The
     UI's feature-sampling toggle is deliberately left in so you can see
     this bias/variance trade-off fail to pay off, not just see it win.

Averaging many high-variance, low-bias trees (majority vote for
classification) cancels out their independent mistakes while barely
raising the bias — the textbook bias-variance trade-off, made visible by
watching the jagged single-tree boundary smooth out as more trees join
the vote.

Two by-products of bagging are recorded every step because they are the
whole reason practitioners like Random Forests:

  - Out-of-bag (OOB) error — each tree's bootstrap resample leaves out
    ~37% of the data on average (e^-1 for sample-size resampling); voting
    only the trees for which a point was OOB gives an honest estimate of
    test error *without* holding out a separate validation set.
  - Feature importance — the total impurity decrease credited to each
    feature, summed over every split in every tree and normalised to sum
    to 1. A feature the forest never needed to split on contributes ~0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tree import Node, build_tree, predict


@dataclass
class Snapshot:
    """State of the ensemble immediately after adding one new tree."""

    points: np.ndarray                     # (N, 2) — constant across snapshots
    labels: np.ndarray                      # (N,) — constant across snapshots
    tree_index: int                         # 1-indexed: which tree this is
    bootstrap_counts: np.ndarray            # (N,) times each point was resampled (0 = OOB for this tree)

    tree_grid: np.ndarray                   # (res, res) this tree's own prediction over the background grid
    ensemble_grid: np.ndarray               # (res, res) majority vote of trees[0:tree_index]
    ensemble_confidence_grid: np.ndarray    # (res, res) fraction of trees agreeing with the winning class

    oob_error: float | None                 # running OOB error using every tree so far; None until a point has been OOB at least once
    train_accuracy: float                   # running resubstitution accuracy of the ensemble so far
    feature_importance: np.ndarray          # (2,) normalised, cumulative across trees[0:tree_index]

    tree_depth: int                         # depth reached by *this* tree
    n_leaves: int                           # leaf count of *this* tree
    n_oob_points: int                       # how many training points were OOB for this tree

    @property
    def oob_mask(self) -> np.ndarray:
        return self.bootstrap_counts == 0

    @property
    def title(self) -> str:
        return f"Tree {self.tree_index} — depth {self.tree_depth}, {self.n_leaves} leaves"


@dataclass
class ForestRun:
    """Full result of fit(): every snapshot plus the shared background grid."""

    snapshots: list[Snapshot]
    grid_x: np.ndarray   # (res, res) meshgrid X coordinates
    grid_y: np.ndarray   # (res, res) meshgrid Y coordinates
    n_classes: int


def _make_grid(X: np.ndarray, res: int, pad: float = 0.8):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    return gx, gy, grid_points


def fit(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 25,
    max_depth: int = 6,
    min_samples_leaf: int = 3,
    max_features: int = 1,
    bootstrap_frac: float = 1.0,
    seed: int = 0,
    grid_res: int = 70,
) -> ForestRun:
    """Train a Random Forest classifier on (X, y), recording one Snapshot
    per tree added so the visualiser can replay the whole ensemble
    forming, tree by tree.

    Parameters
    ----------
    X, y:              training data; X shape (N, 2), y shape (N,) with
                        integer class labels in [0, n_classes)
    n_estimators:       number of trees in the forest
    max_depth:          hard cap on each tree's depth
    min_samples_leaf:   minimum samples required on both sides of a split
    max_features:       candidate features considered at each split
                        (1 = true Random Forest; 2 = Bagging only, since
                        every split can see both available features)
    bootstrap_frac:     fraction of N resampled (with replacement) per
                        tree; 1.0 is the standard choice. Lower values
                        starve each tree of data (more bias) but leave
                        more points OOB per tree (denser OOB curve early on)
    seed:               random seed — controls bootstrap draws and every
                        tree's feature-subsampling choices
    grid_res:           resolution of the background grid used for every
                        decision-boundary heatmap

    Returns
    -------
    ForestRun with the full snapshot history, one entry per tree.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n, n_features = X.shape
    n_classes = int(y.max()) + 1
    rng = np.random.default_rng(seed)

    gx, gy, grid_points = _make_grid(X, grid_res)
    n_grid = grid_points.shape[0]
    m = max(1, int(round(bootstrap_frac * n)))

    cumulative_importance = np.zeros(n_features)
    oob_vote_counts = np.zeros((n, n_classes))
    oob_ever = np.zeros(n, dtype=bool)
    train_vote_counts = np.zeros((n, n_classes))
    grid_vote_counts = np.zeros((n_grid, n_classes))

    snapshots: list[Snapshot] = []

    for t in range(1, n_estimators + 1):
        boot_idx = rng.integers(0, n, size=m)
        bootstrap_counts = np.bincount(boot_idx, minlength=n)
        oob_mask = bootstrap_counts == 0

        root: Node
        root, importance, depth, n_leaves = build_tree(
            X[boot_idx], y[boot_idx], n_classes,
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            max_features=max_features, rng=rng,
        )
        cumulative_importance += importance

        # --- Out-of-bag bookkeeping ------------------------------------
        n_oob = int(oob_mask.sum())
        if n_oob:
            oob_idx = np.where(oob_mask)[0]
            oob_pred = predict(root, X[oob_idx])
            oob_vote_counts[oob_idx, oob_pred] += 1
            oob_ever[oob_idx] = True

        if oob_ever.any():
            oob_final = oob_vote_counts[oob_ever].argmax(axis=1)
            oob_error = float((oob_final != y[oob_ever]).mean())
        else:
            oob_error = None

        # --- Running (optimistic) resubstitution accuracy --------------
        train_pred = predict(root, X)
        train_vote_counts[np.arange(n), train_pred] += 1
        train_accuracy = float((train_vote_counts.argmax(axis=1) == y).mean())

        # --- Background-grid predictions, this tree and the ensemble ---
        tree_grid_flat = predict(root, grid_points)
        grid_vote_counts[np.arange(n_grid), tree_grid_flat] += 1
        ensemble_grid_flat = grid_vote_counts.argmax(axis=1)
        confidence_flat = grid_vote_counts.max(axis=1) / t

        importance_sum = float(cumulative_importance.sum())
        norm_importance = (
            cumulative_importance / importance_sum
            if importance_sum > 1e-12
            else np.full(n_features, 1.0 / n_features)
        )

        snapshots.append(
            Snapshot(
                points=X,
                labels=y,
                tree_index=t,
                bootstrap_counts=bootstrap_counts,
                tree_grid=tree_grid_flat.reshape(gx.shape),
                ensemble_grid=ensemble_grid_flat.reshape(gx.shape),
                ensemble_confidence_grid=confidence_flat.reshape(gx.shape),
                oob_error=oob_error,
                train_accuracy=train_accuracy,
                feature_importance=norm_importance,
                tree_depth=depth,
                n_leaves=n_leaves,
                n_oob_points=n_oob,
            )
        )

    return ForestRun(snapshots=snapshots, grid_x=gx, grid_y=gy, n_classes=n_classes)
