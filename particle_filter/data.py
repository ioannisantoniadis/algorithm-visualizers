"""
data.py — synthetic 2-D trajectory + noisy observation generator

Each generator returns (true_pos, observations) where:
  true_pos      — float64 array (n_steps, 2): the ground-truth path.
                  The filter never sees this; it exists only so the
                  visualiser and the RMSE metrics can grade the estimate.
  observations  — float64 array (n_steps, 2): true_pos + measurement
                  noise. This is the only thing the particle filter is
                  given.

Same four trajectory shapes as kalman/data.py (this is the same tracking
problem, solved with a different filter), so the two visualisers are
directly comparable on identical ground truth.

Public API
----------
make_linear(...)    — near-constant-velocity, (almost) straight path
make_circular(...)  — orbiting motion; a curved path a CV model can't follow exactly
make_weave(...)     — sinusoidal weaving path
make_maneuver(...)  — piecewise-linear path with abrupt turns

make_trajectory(kind, n_steps, dt, seed, measurement_noise) — dispatcher

TRAJ_KEYS / TRAJ_NAMES — registry, kept in sync with app.py
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Registry — keep display name <-> key in sync with app.py
# ---------------------------------------------------------------------------

TRAJ_KEYS = ["linear", "circular", "weave", "maneuver"]

TRAJ_NAMES = [
    "Constant velocity",
    "Circular orbit",
    "Sine weave",
    "Maneuvering turns",
]

# Physical randomness baked into the *true* motion, independent of the
# filter's assumed process noise. A real target is never perfectly linear.
_TRUE_PROCESS_STD = 0.05


def _add_measurement_noise(true_pos: np.ndarray, measurement_noise: float, rng) -> np.ndarray:
    return true_pos + rng.normal(0.0, measurement_noise, size=true_pos.shape)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def make_linear(
    n_steps: int = 60,
    dt: float = 1.0,
    seed: int = 0,
    measurement_noise: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Near-constant-velocity motion in a random direction — the scenario
    the constant-velocity particle motion model was designed for."""
    rng = np.random.default_rng(seed)
    speed = 1.3
    angle0 = rng.uniform(0, 2 * np.pi)
    v = speed * np.array([np.cos(angle0), np.sin(angle0)])

    pos = np.zeros((n_steps, 2))
    for t in range(1, n_steps):
        v = v + rng.normal(0.0, _TRUE_PROCESS_STD, size=2)
        pos[t] = pos[t - 1] + v * dt

    obs = _add_measurement_noise(pos, measurement_noise, rng)
    return pos, obs


def make_circular(
    n_steps: int = 60,
    dt: float = 1.0,
    seed: int = 0,
    measurement_noise: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Orbiting motion at (roughly) constant angular velocity — a curved
    path a constant-velocity motion model can only approximate locally."""
    rng = np.random.default_rng(seed)
    radius = 8.0
    n_orbits = 1.3
    theta = np.linspace(0, n_orbits * 2 * np.pi, n_steps)
    drift = np.cumsum(rng.normal(0.0, _TRUE_PROCESS_STD / radius, size=n_steps))
    pos = np.column_stack([
        radius * np.cos(theta + drift),
        radius * np.sin(theta + drift),
    ])
    obs = _add_measurement_noise(pos, measurement_noise, rng)
    return pos, obs


def make_weave(
    n_steps: int = 60,
    dt: float = 1.0,
    seed: int = 0,
    measurement_noise: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sinusoidal weaving path, like a vehicle changing lanes repeatedly."""
    rng = np.random.default_rng(seed)
    length, amplitude, n_cycles = 22.0, 4.0, 2.2
    x = np.linspace(0, length, n_steps)
    y = amplitude * np.sin(2 * np.pi * n_cycles * x / length)
    pos = np.column_stack([x, y])
    pos += np.cumsum(rng.normal(0.0, _TRUE_PROCESS_STD, size=pos.shape), axis=0)
    obs = _add_measurement_noise(pos, measurement_noise, rng)
    return pos, obs


def make_maneuver(
    n_steps: int = 60,
    dt: float = 1.0,
    seed: int = 0,
    measurement_noise: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear path with a handful of abrupt direction changes —
    the hardest case: at each turn the true velocity jumps discontinuously,
    which a constant-velocity particle motion model can only explain away
    as process noise."""
    rng = np.random.default_rng(seed)
    speed = 1.4
    n_legs = int(rng.integers(3, 6))
    cut_points = np.sort(rng.choice(np.arange(5, n_steps - 5), size=n_legs - 1, replace=False))
    bounds = np.concatenate([[0], cut_points, [n_steps]])

    pos = np.zeros((n_steps, 2))
    for i in range(len(bounds) - 1):
        start, end = int(bounds[i]), int(bounds[i + 1])
        angle = rng.uniform(0, 2 * np.pi)
        v = speed * np.array([np.cos(angle), np.sin(angle)])
        for t in range(max(start, 1), end):
            pos[t] = pos[t - 1] + v * dt + rng.normal(0.0, _TRUE_PROCESS_STD, size=2)

    obs = _add_measurement_noise(pos, measurement_noise, rng)
    return pos, obs


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def make_trajectory(
    kind: str,
    n_steps: int = 60,
    dt: float = 1.0,
    seed: int = 0,
    measurement_noise: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (true_pos, observations) for the requested trajectory kind."""
    if kind == "linear":
        return make_linear(n_steps, dt, seed, measurement_noise)
    if kind == "circular":
        return make_circular(n_steps, dt, seed, measurement_noise)
    if kind == "weave":
        return make_weave(n_steps, dt, seed, measurement_noise)
    if kind == "maneuver":
        return make_maneuver(n_steps, dt, seed, measurement_noise)
    raise ValueError(f"Unknown trajectory kind {kind!r}. Choose from: {TRAJ_KEYS}")
