"""
algorithm.py — From-scratch discrete linear Kalman filter with full history

State vector x = [px, py, vx, vy]^T under a constant-velocity motion model.
Each time step is split into two recorded phases, mirroring the classic
predict-update loop:

  Predict: propagate the state forward through the motion model and grow
           the covariance by the process noise Q — the filter's belief
           about how much the target can accelerate/manoeuvre between
           measurements. Uncertainty always grows here.

  Update:  fold in the new noisy measurement, weighted by the Kalman
           gain K, which balances trust between the (uncertain)
           prediction and the (noisy) observation. Uncertainty always
           shrinks here.

Every predict/update pair is recorded as a Snapshot so the visualiser can
replay the growing/shrinking uncertainty ellipse frame by frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """State of the filter at one predict or update phase."""

    step: int                      # time index; 0 = prior belief before any update
    phase: Literal["init", "predict", "update"]

    mean: np.ndarray               # state estimate (4,): [px, py, vx, vy]
    cov: np.ndarray                # state covariance (4, 4)

    true_pos: np.ndarray           # (2,) ground-truth position at this step (plotting/metrics only)

    # Populated only on "init" and "update" phases (once a measurement has
    # actually been folded in)
    observation: np.ndarray | None = None
    innovation: np.ndarray | None = None     # y = z - H @ x_pred
    kalman_gain: np.ndarray | None = None    # K
    nis: float = 0.0                         # normalised innovation squared

    @property
    def pos_mean(self) -> np.ndarray:
        return self.mean[:2]

    @property
    def pos_cov(self) -> np.ndarray:
        return self.cov[:2, :2]

    @property
    def vel_mean(self) -> np.ndarray:
        return self.mean[2:]

    @property
    def pos_uncertainty(self) -> float:
        """Trace of the position covariance block — a single scalar summary
        of "how big is the uncertainty ellipse right now"."""
        return float(np.trace(self.pos_cov))

    @property
    def title(self) -> str:
        if self.phase == "init":
            return "Initial belief — before any measurement"
        if self.phase == "predict":
            return f"t={self.step} — Predict step (uncertainty grows)"
        return f"t={self.step} — Update step (uncertainty shrinks)"


# ---------------------------------------------------------------------------
# Model matrices
# ---------------------------------------------------------------------------

_H = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])


def _transition_matrix(dt: float) -> np.ndarray:
    """Constant-velocity motion model: position += velocity * dt."""
    return np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _process_noise_cov(dt: float, q: float) -> np.ndarray:
    """Discretised white-noise-acceleration process noise covariance.

    `q` is the std-dev of the unmodelled acceleration assumed to drive the
    target between steps. It is the filter's *belief*, not the true
    physical noise — larger q means the filter expects more manoeuvring
    and therefore weighs new measurements more heavily.
    """
    dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
    block = np.array([
        [dt4 / 4.0, dt3 / 2.0],
        [dt3 / 2.0, dt2],
    ])
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = block   # x position/velocity
    Q[np.ix_([1, 3], [1, 3])] = block   # y position/velocity
    return (q ** 2) * Q


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit(
    observations: np.ndarray,
    true_positions: np.ndarray,
    dt: float = 1.0,
    q: float = 0.1,
    r: float = 1.0,
    init_velocity_var: float = 10.0,
) -> list[Snapshot]:
    """Run the Kalman filter over a sequence of noisy position measurements.

    Parameters
    ----------
    observations:       (n, 2) noisy [x, y] measurements — the only input
                         the filter actually uses
    true_positions:      (n, 2) ground-truth positions — for plotting and
                         RMSE metrics only, never touched by the filter
    dt:                  time step between measurements
    q:                   process noise std — how much manoeuvring the filter expects
    r:                   measurement noise std — how much it trusts each observation
    init_velocity_var:   initial variance on the (unknown) velocity components

    Returns
    -------
    A list of Snapshot objects in chronological order:
      [init, predict_1, update_1, predict_2, update_2, ..., update_n]
    """
    n = len(observations)
    F = _transition_matrix(dt)
    Q = _process_noise_cov(dt, q)
    R = (r ** 2) * np.eye(2)
    I4 = np.eye(4)

    # Initial belief: position = first observation (only thing we know),
    # velocity completely unknown -> large variance
    x = np.array([observations[0, 0], observations[0, 1], 0.0, 0.0])
    P = np.diag([r ** 2, r ** 2, init_velocity_var, init_velocity_var])

    snapshots: list[Snapshot] = [
        Snapshot(
            step=0,
            phase="init",
            mean=x.copy(),
            cov=P.copy(),
            true_pos=true_positions[0],
            observation=observations[0].copy(),
        )
    ]

    for k in range(1, n):
        # ---- Predict: propagate belief through the motion model -----------
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        snapshots.append(
            Snapshot(
                step=k,
                phase="predict",
                mean=x_pred.copy(),
                cov=P_pred.copy(),
                true_pos=true_positions[k],
            )
        )

        # ---- Update: fold in the new noisy measurement ---------------------
        z = observations[k]
        y = z - _H @ x_pred                       # innovation (measurement residual)
        S = _H @ P_pred @ _H.T + R                # innovation covariance
        K = P_pred @ _H.T @ np.linalg.inv(S)       # Kalman gain

        x = x_pred + K @ y
        P = (I4 - K @ _H) @ P_pred
        nis = float(y @ np.linalg.solve(S, y))     # normalised innovation squared

        snapshots.append(
            Snapshot(
                step=k,
                phase="update",
                mean=x.copy(),
                cov=P.copy(),
                true_pos=true_positions[k],
                observation=z.copy(),
                innovation=y.copy(),
                kalman_gain=K.copy(),
                nis=nis,
            )
        )

    return snapshots
