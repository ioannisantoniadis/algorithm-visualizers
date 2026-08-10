"""
algorithm.py — from-scratch tabular Q-learning and SARSA

Both algorithms share one training loop; the only difference is how the
TD target for the *next* state is computed:

  Q-learning (off-policy): target = r + gamma * max_a' Q[s', a']
      Learns the value of the GREEDY policy regardless of which action
      exploration actually takes next — it bootstraps off the best
      possible next move, even if the agent won't take it.

  SARSA (on-policy): target = r + gamma * Q[s', a']
      where a' is the action epsilon-greedy exploration will *actually*
      take next. It learns the value of the policy it is really
      following, exploration and all.

That single difference is why the two disagree on the Cliff Walking
preset: Q-learning happily learns to hug the cliff edge because it always
bootstraps off the best next action, while SARSA learns a safer path one
row back, because its target accounts for the real chance that
exploration steps off the edge.

Every environment step — one action, one stochastic transition, one
Q-update — is recorded as a Snapshot, so the visualiser can replay the
Q-table's evolution and the agent's changing policy one step at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .data import Cell, GridWorld

# Action encoding — indices double as 90-degree rotation steps, so
# "one step clockwise" (right of the direction you're facing) is simply
# (action + 1) % 4 and "one step counter-clockwise" is (action - 1) % 4.
N, E, S, W = 0, 1, 2, 3
ACTIONS = (N, E, S, W)
ACTION_NAMES = {N: "north", E: "east", S: "south", W: "west"}
ACTION_ARROWS = {N: "↑", E: "→", S: "↓", W: "←"}
_DELTAS = {N: (-1, 0), E: (0, 1), S: (1, 0), W: (0, -1)}


@dataclass
class Snapshot:
    """State immediately after one environment step and one Q-update
    (or, for step 0, the untrained Q-table before any learning)."""

    grid: GridWorld  # constant across all snapshots
    algorithm: Literal["qlearning", "sarsa"]

    episode: int  # 1-indexed; 0 = init frame, before any training
    step_in_episode: int
    global_step: int  # steps taken across all episodes so far

    phase: Literal["init", "explore", "exploit"]
    state: Cell | None
    action: int | None  # intended action, chosen by epsilon-greedy
    actual_action: int | None  # action the stochastic env actually executed
    next_state: Cell | None
    reward: float | None
    done: bool

    epsilon: float
    q_table: np.ndarray  # (rows, cols, 4) copy AFTER this step's update
    td_error: float | None

    episode_return: float = 0.0  # cumulative reward so far this episode
    episode_length: int = 0

    @property
    def slipped(self) -> bool:
        return self.action is not None and self.actual_action != self.action

    @property
    def title(self) -> str:
        name = "Q-learning" if self.algorithm == "qlearning" else "SARSA"
        if self.phase == "init":
            return f"{name} — untrained (Q = 0 everywhere)"
        verb = "exploring" if self.phase == "explore" else "exploiting"
        tag = " · episode ended" if self.done else ""
        return f"{name} — ep {self.episode}, step {self.step_in_episode} ({verb}){tag}"


def _move(state: Cell, action: int, rows: int, cols: int) -> Cell:
    dr, dc = _DELTAS[action]
    r, c = state
    return (min(max(r + dr, 0), rows - 1), min(max(c + dc, 0), cols - 1))


def _slip(action: int, rng: np.random.Generator) -> int:
    """80% intended direction, 10% one step clockwise, 10% one step
    counter-clockwise — the exact transition model from grid-world-mdp."""
    u = rng.random()
    if u < 0.8:
        return action
    if u < 0.9:
        return (action + 1) % 4
    return (action - 1) % 4


def env_step(
    grid: GridWorld, state: Cell, action: int, rng: np.random.Generator
) -> tuple[Cell, float, bool, int]:
    """Execute `action` from `state` under the stochastic transition model.

    Returns (next_state, reward, done, actual_action). Landing on a
    "respawn" cell (the cliff) applies its reward but snaps the agent
    back to start without ending the episode.
    """
    actual_action = _slip(action, rng)
    next_state = _move(state, actual_action, grid.rows, grid.cols)
    reward = float(grid.reward[next_state])
    done = bool(grid.terminal[next_state])
    if grid.respawn[next_state]:
        next_state = grid.start
        done = False
    return next_state, reward, done, actual_action


def _epsilon_greedy(
    q_table: np.ndarray, state: Cell, epsilon: float, rng: np.random.Generator
) -> tuple[int, bool]:
    """Return (action, was_exploration_step)."""
    if rng.random() < epsilon:
        return int(rng.integers(4)), True
    row = q_table[state]
    best = np.flatnonzero(row == row.max())
    return int(rng.choice(best)), False


def _epsilon_at(ep: int, start: float, end: float, decay_episodes: int) -> float:
    """Linear decay from `start` to `end` over the first `decay_episodes`
    episodes, then held constant at `end` — a simple, visualisable
    exploration-to-exploitation schedule."""
    if decay_episodes <= 1:
        return end
    frac = min(1.0, (ep - 1) / (decay_episodes - 1))
    return start + (end - start) * frac


def train(
    grid: GridWorld,
    algorithm: Literal["qlearning", "sarsa"],
    n_episodes: int = 100,
    max_steps: int = 60,
    alpha: float = 0.2,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 60,
    seed: int = 0,
) -> list[Snapshot]:
    """Run tabular Q-learning or SARSA and return every step as a Snapshot.

    Returns
    -------
    [init, ep1_step1, ep1_step2, ..., ep2_step1, ...] — one init frame
    followed by every environment step across every episode, in order.
    """
    rng = np.random.default_rng(seed)
    rows, cols = grid.rows, grid.cols
    q_table = np.zeros((rows, cols, 4), dtype=float)

    snapshots: list[Snapshot] = [
        Snapshot(
            grid=grid, algorithm=algorithm, episode=0, step_in_episode=0, global_step=0,
            phase="init", state=grid.start, action=None, actual_action=None,
            next_state=None, reward=None, done=False,
            epsilon=epsilon_start, q_table=q_table.copy(), td_error=None,
        )
    ]

    global_step = 0
    for ep in range(1, n_episodes + 1):
        epsilon = _epsilon_at(ep, epsilon_start, epsilon_end, epsilon_decay_episodes)
        state = grid.start
        episode_return = 0.0

        # SARSA needs "the action we're about to take" chosen up front,
        # since its target reuses that same action next step (on-policy).
        if algorithm == "sarsa":
            action, is_explore = _epsilon_greedy(q_table, state, epsilon, rng)

        for t in range(1, max_steps + 1):
            global_step += 1
            if algorithm == "qlearning":
                action, is_explore = _epsilon_greedy(q_table, state, epsilon, rng)

            next_state, reward, done, actual_action = env_step(grid, state, action, rng)

            if algorithm == "qlearning":
                best_next = 0.0 if done else float(q_table[next_state].max())
                td_target = reward + gamma * best_next
            else:  # sarsa
                if done:
                    next_action, next_is_explore = None, False
                    td_target = reward
                else:
                    next_action, next_is_explore = _epsilon_greedy(q_table, next_state, epsilon, rng)
                    td_target = reward + gamma * float(q_table[next_state][next_action])

            td_error = td_target - float(q_table[state][action])
            q_table[state][action] += alpha * td_error

            episode_return += reward

            snapshots.append(
                Snapshot(
                    grid=grid, algorithm=algorithm, episode=ep, step_in_episode=t,
                    global_step=global_step,
                    phase="explore" if is_explore else "exploit",
                    state=state, action=action, actual_action=actual_action,
                    next_state=next_state, reward=reward, done=done,
                    epsilon=epsilon, q_table=q_table.copy(), td_error=float(td_error),
                    episode_return=episode_return, episode_length=t,
                )
            )

            if done:
                break
            state = next_state
            if algorithm == "sarsa":
                action, is_explore = next_action, next_is_explore

    return snapshots


def episode_summaries(
    snapshots: list[Snapshot],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (episode_numbers, total_reward, length, epsilon) — one entry
    per completed episode, read off each episode's final recorded step."""
    per_ep: dict[int, Snapshot] = {}
    for s in snapshots:
        if s.episode == 0:
            continue
        per_ep[s.episode] = s  # later steps overwrite earlier ones
    eps = sorted(per_ep)
    if not eps:
        return np.array([]), np.array([]), np.array([]), np.array([])
    returns = np.array([per_ep[e].episode_return for e in eps])
    lengths = np.array([per_ep[e].episode_length for e in eps])
    epsilons = np.array([per_ep[e].epsilon for e in eps])
    return np.array(eps), returns, lengths, epsilons


def greedy_action(q_table: np.ndarray, state: Cell) -> int:
    """Argmax action at `state`, breaking ties toward the lowest index
    so the displayed policy is deterministic frame to frame."""
    row = q_table[state]
    return int(np.argmax(row))


def rollout_policy_path(grid: GridWorld, q_table: np.ndarray, max_steps: int = 200) -> list[Cell]:
    """Trace the greedy policy deterministically (ignoring the env's 10/10
    slip) — this is what "the agent's current best route" means in the
    policy-arrow view, as opposed to any one noisy training trajectory."""
    state = grid.start
    path = [state]
    seen = {state}
    for _ in range(max_steps):
        if grid.terminal[state]:
            break
        action = greedy_action(q_table, state)
        next_state = _move(state, action, grid.rows, grid.cols)
        if grid.respawn[next_state] or next_state == state or next_state in seen:
            break  # cliff step, wall bump, or a policy loop — stop drawing
        path.append(next_state)
        seen.add(next_state)
        state = next_state
    return path
