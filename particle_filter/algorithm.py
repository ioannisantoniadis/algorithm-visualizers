"""
algorithm.py — From-scratch bootstrap (SIR) particle filter

Tracks a 2-D target from noisy position observations using a set of weighted
particles instead of a single Gaussian belief (contrast with kalman/, which
solves the same tracking problem with a closed-form Gaussian filter). Each
particle carries a constant-velocity state [px, py, vx, vy] — the same state
shape as the Kalman filter's mean vector — so the two visualisers are a fair
side-by-side comparison of "Gaussian belief" vs. "particle cloud" for
identical trajectories. (An earlier draft carried only [px, py] and let
particles do a pure random walk in position; that model can't keep up with
directed motion at all — see the "Why velocity is part of the particle
state" note below.)

Each time step is split into up to three recorded phases:

  Predict:   propagate every particle's position by its own velocity, then
             perturb every particle's velocity with process noise — this is
             where particles spread out to cover where the target might
             have gone or how it might have turned.

  Update:    re-weight each particle by how likely the new observation is
             given that particle's *position* (Gaussian likelihood; velocity
             isn't observed directly). Particles near the observation gain
             weight; particles far away lose it.

  Resample:  when the weights become too concentrated on a few particles
             (effective sample size drops below a threshold), draw a new
             particle set proportional to the weights and reset weights to
             uniform. Skipped on any step where ESS stays high enough.

Why velocity is part of the particle state
-------------------------------------------
A pure position random walk only diffuses by `process_noise` per step. If
the true target moves at speed 1.3/step (typical for these trajectories)
and `process_noise` is a plausible-looking 0.3, the particle cloud
systematically lags behind a translating target and the filter's RMSE ends
up *worse* than just using the raw noisy observations — confirmed by
deliberately testing this while building this module. Carrying velocity
lets each particle predict "where I'm already heading", exactly like the
Kalman filter's constant-velocity transition, and the particle filter
handles it via diffusion on velocity rather than matrix algebra.
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
    """State of the filter at one predict, update, or resample phase."""

    step: int                      # time index; 0 = initial particle cloud
    phase: Literal["init", "predict", "update", "resample"]

    particles: np.ndarray          # (N, 4) particle states: [px, py, vx, vy]
    weights: np.ndarray            # (N,) normalised importance weights

    true_pos: np.ndarray           # (2,) ground-truth position (plotting/metrics only)

    observation: np.ndarray | None = None
    ess: float = 0.0               # effective sample size = 1 / sum(w_i^2)

    @property
    def weighted_mean(self) -> np.ndarray:
        """Full (4,) weighted mean state: [px, py, vx, vy]."""
        return (self.particles * self.weights[:, None]).sum(axis=0)

    @property
    def pos_mean(self) -> np.ndarray:
        return self.weighted_mean[:2]

    @property
    def n_particles(self) -> int:
        return len(self.particles)

    @property
    def title(self) -> str:
        if self.phase == "init":
            return "Initial particle cloud — scattered around the first observation"
        if self.phase == "predict":
            return f"t={self.step} — Predict step (particles diffuse)"
        if self.phase == "update":
            return f"t={self.step} — Update step (re-weight by observation)"
        return f"t={self.step} — Resample (ESS was low, particles refocus)"


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: one random offset, N evenly-spaced draws.

    Lower variance than naive multinomial resampling (drawing N i.i.d.
    samples from the weight distribution) for the same particle count,
    which is why it's the standard choice for bootstrap particle filters.
    """
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against floating-point drift leaving the last bucket short
    return np.searchsorted(cumulative, positions)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit(
    observations: np.ndarray,
    true_positions: np.ndarray,
    n_particles: int = 300,
    process_noise: float = 0.5,
    measurement_noise: float = 1.2,
    resample_frac: float = 0.5,
    init_velocity_std: float = 1.0,
    seed: int = 0,
) -> list[Snapshot]:
    """Run a bootstrap particle filter over a sequence of noisy position
    measurements.

    Parameters
    ----------
    observations:         (n, 2) noisy [x, y] measurements — the only input
                          the filter actually uses
    true_positions:       (n, 2) ground-truth positions — for plotting and
                          RMSE metrics only, never touched by the filter
    n_particles:          number of particles in the filter
    process_noise:        std-dev of the velocity perturbation applied to
                          every particle each predict — how much manoeuvring
                          the filter expects between measurements
    measurement_noise:    std-dev of the Gaussian likelihood used to weight
                          particles' *positions* against each observation
    resample_frac:        resample whenever ESS falls below
                          resample_frac * n_particles
    init_velocity_std:    std-dev of the initial random velocity guess for
                          each particle (true velocity is unknown at t=0)

    Returns
    -------
    A list of Snapshot objects in chronological order:
      [init, predict_1, update_1, (resample_1?), predict_2, update_2, ...]
    """
    rng = np.random.default_rng(seed)
    n = len(observations)

    positions = observations[0] + rng.normal(0.0, measurement_noise, size=(n_particles, 2))
    velocities = rng.normal(0.0, init_velocity_std, size=(n_particles, 2))
    particles = np.concatenate([positions, velocities], axis=1)
    weights = np.full(n_particles, 1.0 / n_particles)

    snapshots: list[Snapshot] = [
        Snapshot(
            step=0,
            phase="init",
            particles=particles.copy(),
            weights=weights.copy(),
            true_pos=true_positions[0],
            observation=observations[0].copy(),
            ess=float(n_particles),
        )
    ]

    resample_threshold = resample_frac * n_particles

    for k in range(1, n):
        # ---- Predict: integrate position by velocity, diffuse velocity ----
        particles = particles.copy()
        particles[:, :2] += particles[:, 2:]
        particles[:, 2:] += rng.normal(0.0, process_noise, size=(n_particles, 2))
        snapshots.append(
            Snapshot(
                step=k,
                phase="predict",
                particles=particles.copy(),
                weights=weights.copy(),
                true_pos=true_positions[k],
            )
        )

        # ---- Update: re-weight by Gaussian likelihood of the observation --
        # (position only — velocity is never observed directly)
        z = observations[k]
        sq_dist = np.sum((particles[:, :2] - z) ** 2, axis=1)
        likelihood = np.exp(-0.5 * sq_dist / measurement_noise ** 2)
        weights = weights * likelihood
        weight_sum = weights.sum()
        if weight_sum <= 0.0:
            # Every particle assigned ~zero likelihood (severe mismatch) —
            # fall back to uniform rather than dividing by zero.
            weights = np.full(n_particles, 1.0 / n_particles)
        else:
            weights = weights / weight_sum
        ess = float(1.0 / np.sum(weights ** 2))

        snapshots.append(
            Snapshot(
                step=k,
                phase="update",
                particles=particles.copy(),
                weights=weights.copy(),
                true_pos=true_positions[k],
                observation=z.copy(),
                ess=ess,
            )
        )

        # ---- Resample: only when particle diversity has collapsed ---------
        if ess < resample_threshold:
            idx = _systematic_resample(weights, rng)
            particles = particles[idx].copy()
            weights = np.full(n_particles, 1.0 / n_particles)
            snapshots.append(
                Snapshot(
                    step=k,
                    phase="resample",
                    particles=particles.copy(),
                    weights=weights.copy(),
                    true_pos=true_positions[k],
                    ess=float(n_particles),
                )
            )

    return snapshots
