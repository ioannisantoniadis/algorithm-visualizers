"""
data.py — grid-world environment definitions for the TD-learning visualiser

This extends grid-world-mdp's 3x5 resource grid (iron / diamond / lava / pit,
stochastic 80/10/10 movement) to a handful of presets an agent can *learn*
about through trial and error rather than by being handed the transition
model. Every preset returns a GridWorld: a compact description of cell
kinds, rewards, terminal/respawn behaviour and the start cell — enough for
algorithm.py to run episodes without knowing anything about how the layout
was generated.

Public API
----------
make_classic()                       — the exact grid from grid-world-mdp
make_cliff(rows, cols)                — Sutton & Barto's Cliff Walking
make_random_hazards(rows, cols, seed) — scattered resources/hazards

make_grid(preset, rows, cols, seed)  — single entry-point dispatcher

PRESET_KEYS / PRESET_NAMES — ordered keys/names kept in sync with app.py
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

Cell = tuple[int, int]

PRESET_KEYS = ["classic", "cliff", "random_hazards"]
PRESET_NAMES = ["Classic (grid-world-mdp)", "Cliff walk", "Random hazards"]

# Cell kinds and the colour/legend metadata visualize.py draws them with.
# "respawn" kinds (cliff) are non-terminal but snap the agent back to start.
KIND_REWARD = {
    "empty": 0.0,
    "start": 0.0,
    "iron": 3.0,
    "diamond": 10.0,
    "lava": -10.0,
    "pit": -3.0,
    "cliff": -100.0,
    "goal": 0.0,
}
TERMINAL_KINDS = {"iron", "diamond", "lava", "pit", "goal"}
RESPAWN_KINDS = {"cliff"}


@dataclass
class GridWorld:
    """Everything algorithm.py needs to run episodes; immutable after creation."""

    kind: np.ndarray  # (rows, cols) of str — cell category, drives reward/behaviour
    reward: np.ndarray  # (rows, cols) float — reward for ENTERING this cell
    terminal: np.ndarray  # (rows, cols) bool — episode ends on entry
    respawn: np.ndarray  # (rows, cols) bool — agent bounces back to start on entry
    start: Cell
    step_reward: float  # reward for entering any plain "empty"/"start" cell
    rows: int
    cols: int
    name: str


def _empty_kind_grid(rows: int, cols: int) -> np.ndarray:
    return np.full((rows, cols), "empty", dtype=object)


def _finalize(kind: np.ndarray, start: Cell, step_reward: float, name: str) -> GridWorld:
    rows, cols = kind.shape
    kind = kind.copy()
    kind[start] = "start"
    reward = np.zeros((rows, cols), dtype=float)
    terminal = np.zeros((rows, cols), dtype=bool)
    respawn = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            k = kind[r, c]
            if k in ("empty", "start"):
                reward[r, c] = step_reward
            else:
                reward[r, c] = KIND_REWARD[k]
            terminal[r, c] = k in TERMINAL_KINDS
            respawn[r, c] = k in RESPAWN_KINDS
    return GridWorld(
        kind=kind, reward=reward, terminal=terminal, respawn=respawn,
        start=start, step_reward=step_reward, rows=rows, cols=cols, name=name,
    )


def make_classic() -> GridWorld:
    """The exact 3x5 grid from grid-world-mdp: iron/lava/diamond up top,
    two lava cells cutting through the middle row. Reward comes only from
    the terminal cell you land on (step_reward=0), so a learner is guided
    purely by discounting — reaching the diamond sooner is worth more only
    because gamma < 1, not because moving costs anything."""
    kind = _empty_kind_grid(3, 5)
    kind[0, 2] = "iron"
    kind[0, 3] = "lava"
    kind[0, 4] = "diamond"
    kind[1, 2] = "lava"
    kind[1, 3] = "lava"
    start = (2, 0)
    return _finalize(kind, start, step_reward=0.0, name="Classic (grid-world-mdp)")


def make_cliff(rows: int = 4, cols: int = 12) -> GridWorld:
    """Sutton & Barto's Cliff Walking (Example 6.6): start bottom-left, goal
    bottom-right, a row of cliff cells in between. Every step costs -1;
    stepping off the cliff costs -100 and resets to start (episode keeps
    going). This is the textbook example where Q-learning (off-policy)
    converges to the optimal path hugging the cliff edge, while SARSA
    (on-policy) learns a safer route one row up — because SARSA's target
    uses the action epsilon-greedy exploration will *actually* take next,
    which occasionally steps off the edge during training."""
    rows = max(rows, 3)
    cols = max(cols, 4)
    kind = _empty_kind_grid(rows, cols)
    kind[rows - 1, 1:cols - 1] = "cliff"
    start = (rows - 1, 0)
    kind[rows - 1, cols - 1] = "goal"
    return _finalize(kind, start, step_reward=-1.0, name="Cliff walk")


def _bfs_reachable(terminal_ok: np.ndarray, start: Cell, blocked: np.ndarray) -> bool:
    rows, cols = blocked.shape
    seen = np.zeros((rows, cols), dtype=bool)
    q = deque([start])
    seen[start] = True
    while q:
        r, c = q.popleft()
        if terminal_ok[r, c]:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and not blocked[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return False


def make_random_hazards(rows: int = 5, cols: int = 7, seed: int = 0) -> GridWorld:
    """Scattered iron/diamond/lava/pit cells. A small step cost (-0.05)
    rewards efficiency on top of the terminal payoffs. Regenerating with a
    new seed gives a fresh layout to re-learn from scratch."""
    rng = np.random.default_rng(seed)
    kind = _empty_kind_grid(rows, cols)
    start = (rows - 1, 0)

    n_cells = rows * cols
    budget = {
        "diamond": max(1, n_cells // 18),
        "iron": max(1, n_cells // 12),
        "lava": max(1, n_cells // 8),
        "pit": max(1, n_cells // 10),
    }

    all_cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != start]
    rng.shuffle(all_cells)

    placed = 0
    for label, count in budget.items():
        for _ in range(count):
            if placed >= len(all_cells):
                break
            r, c = all_cells[placed]
            placed += 1
            kind[r, c] = label

    # Guarantee at least one positive terminal is reachable without crossing
    # a hazard — otherwise the agent has literally nothing to learn.
    blocked = np.isin(kind, ["lava", "pit"])
    is_positive = np.isin(kind, ["diamond", "iron"])
    if not _bfs_reachable(is_positive, start, blocked):
        # carve a guaranteed-safe diamond near the far corner
        goal = (0, cols - 1)
        kind[goal] = "diamond"
        r, c = start
        gr, gc = goal
        while c != gc:
            c += 1 if gc > c else -1
            if kind[r, c] in ("lava", "pit"):
                kind[r, c] = "empty"
        while r != gr:
            r += 1 if gr > r else -1
            if kind[r, c] in ("lava", "pit"):
                kind[r, c] = "empty"

    return _finalize(kind, start, step_reward=-0.05, name="Random hazards")


def make_grid(preset: str, rows: int = 5, cols: int = 7, seed: int = 0) -> GridWorld:
    """Single entry point dispatcher used by app.py. `rows`/`cols` are
    ignored by the "classic" preset, which reproduces a fixed 3x5 layout."""
    if preset == "classic":
        return make_classic()
    if preset == "cliff":
        return make_cliff(rows, cols)
    if preset == "random_hazards":
        return make_random_hazards(rows, cols, seed)
    raise ValueError(f"Unknown preset {preset!r}. Choose from: {PRESET_KEYS}")
