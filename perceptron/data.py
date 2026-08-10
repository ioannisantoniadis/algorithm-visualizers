"""
data.py — 2-D dataset generator for the Perceptron / Gradient Descent visualiser

The dataset is built from two Gaussian blobs placed symmetrically about the
origin along a "true" normal direction, so:

  - the data-generating boundary always passes through the origin, matching
    the bias-free weight space (w1, w2) used by the gradient-descent tab
  - a single `separation` control smoothly interpolates between overlapping
    (non linearly-separable) and cleanly separable data, which is exactly the
    knob that makes the Perceptron either converge quickly or never at all

make_dataset(...) → (X, y, true_normal)
  X:           float64 array, shape (n_points, 2)
  y:           labels in {-1, +1}, shape (n_points,)
  true_normal: unit vector (2,) — the boundary direction used to generate
               the data; handy as a dashed reference line ("ground truth")
"""

from __future__ import annotations

import numpy as np


def make_dataset(
    n_points: int = 60,
    separation: float = 3.5,
    angle_deg: float = 35.0,
    noise_std: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two Gaussian blobs straddling a line through the origin.

    Parameters
    ----------
    n_points:    total number of points (split evenly between classes)
    separation:  distance between the two blob centres, in units of
                 noise_std. separation ≲ 1.5 → heavy class overlap (the
                 Perceptron essentially never converges); separation ≳ 4
                 → most random draws become linearly separable, though with
                 only ~50 points a handful of seeds still put a stray point
                 on the wrong side even at separation 4-5 — that's sampling
                 noise, not a bug, and it's exactly what the "did not
                 converge" message is for. (Empirically, with the defaults
                 below, separation needs to reach ~5 before *every* seed is
                 separable — the "≳3" folklore threshold for 1-D Gaussians
                 doesn't hold once you require every one of ~50 2-D points
                 to land on the right side.)
    angle_deg:   orientation of the true boundary's normal, in degrees
    noise_std:   standard deviation of the per-class Gaussian noise
    seed:        random seed, controls both blob placement noise and shuffle
    """
    rng = np.random.default_rng(seed)

    n_pos = n_points // 2
    n_neg = n_points - n_pos

    theta = np.radians(angle_deg)
    true_normal = np.array([np.cos(theta), np.sin(theta)])

    half_gap = (separation * noise_std) / 2.0
    pos = true_normal * half_gap + rng.normal(0.0, noise_std, size=(n_pos, 2))
    neg = -true_normal * half_gap + rng.normal(0.0, noise_std, size=(n_neg, 2))

    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n_pos), -np.ones(n_neg)])

    perm = rng.permutation(n_points)
    X, y = X[perm], y[perm]

    return X, y, true_normal
