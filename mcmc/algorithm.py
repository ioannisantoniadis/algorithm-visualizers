"""
algorithm.py — Random-walk Metropolis-Hastings, from scratch.

At every step we propose a new point from a symmetric Gaussian random walk
centred on the current state, then accept it with probability

    alpha = min(1, p̃(proposed) / p̃(current))

Because the proposal is symmetric (q(y|x) == q(x|y)), the Hastings
correction term cancels and only the target-density ratio survives — this
is why "vanilla" random-walk MH is one of the simplest MCMC samplers to
implement, and why it only ever needs an *unnormalised* target density.

Every proposal — accepted or not — is recorded as a Snapshot so the
visualiser can replay the chain's full trajectory, including the rejected
side-steps that never make it into the final sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import TARGET_DEFAULTS, log_target


@dataclass
class Snapshot:
    """State of the chain immediately after one Metropolis-Hastings step."""

    iteration: int                  # 0 = start, before any proposal has been tested
    current: np.ndarray             # (2,) accepted position after this step
    proposed: np.ndarray | None     # (2,) point tested this step (None at iteration 0)
    accepted: bool | None           # outcome of this step's test (None at iteration 0)
    chain: np.ndarray               # (iteration+1, 2) every accepted-or-repeated state so far
    n_accepted: int                 # cumulative count of accepted moves
    log_density: float              # log p̃(current) — how "likely" the current state is

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposals accepted so far (undefined at iteration 0)."""
        return self.n_accepted / self.iteration if self.iteration > 0 else 0.0

    @property
    def title(self) -> str:
        if self.iteration == 0:
            return "Start — chain initialised (deliberately far from the mass)"
        verb = "accepted" if self.accepted else "rejected"
        return f"Step {self.iteration} — proposal {verb}"


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation of a 1-D chain for lags 0..max_lag.

    A slowly-decaying ACF means consecutive samples are still highly
    correlated (poor mixing, low effective sample size); a fast decay
    toward 0 means the chain forgets its past quickly and samples behave
    more like independent draws from the target.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    var = np.dot(x, x) / n
    max_lag = max(0, min(max_lag, n - 1))
    acf = np.ones(max_lag + 1)
    if var > 0:
        for lag in range(1, max_lag + 1):
            acf[lag] = np.dot(x[: n - lag], x[lag:]) / (n * var)
    return acf


def run_mcmc(
    target: str,
    n_steps: int,
    proposal_sigma: float,
    init: tuple[float, float] | None = None,
    seed: int = 0,
) -> list[Snapshot]:
    """Run random-walk Metropolis-Hastings on `target` and return every snapshot.

    Parameters
    ----------
    target:         key into data.TARGET_KEYS
    n_steps:        number of proposal attempts (chain length excluding the start)
    proposal_sigma: std of the isotropic Gaussian random-walk proposal
    init:           starting (x, y); defaults to data.TARGET_DEFAULTS[target]["init"]
    seed:           random seed

    Returns
    -------
    A list of n_steps + 1 Snapshot objects, index 0 being the start state.
    """
    rng = np.random.default_rng(seed)
    start = np.asarray(init if init is not None else TARGET_DEFAULTS[target]["init"], dtype=float)

    current = start.copy()
    log_current = float(log_target(target, current))
    chain = [current.copy()]

    snapshots: list[Snapshot] = [
        Snapshot(
            iteration=0,
            current=current.copy(),
            proposed=None,
            accepted=None,
            chain=np.array(chain),
            n_accepted=0,
            log_density=log_current,
        )
    ]

    n_accepted = 0
    for t in range(1, n_steps + 1):
        proposed = current + rng.normal(0.0, proposal_sigma, size=2)
        log_proposed = float(log_target(target, proposed))

        # log of the Metropolis acceptance ratio; the proposal is symmetric
        # so the target densities alone determine it
        log_alpha = log_proposed - log_current
        accept = bool(np.log(rng.uniform()) < log_alpha)

        if accept:
            current = proposed
            log_current = log_proposed
            n_accepted += 1

        chain.append(current.copy())
        snapshots.append(
            Snapshot(
                iteration=t,
                current=current.copy(),
                proposed=proposed.copy(),
                accepted=accept,
                chain=np.array(chain),
                n_accepted=n_accepted,
                log_density=log_current,
            )
        )

    return snapshots
