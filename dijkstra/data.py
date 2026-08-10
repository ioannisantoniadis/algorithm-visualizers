"""
data.py — grid / maze generators for the pathfinding visualiser

All generators return a grid_cost array: a float64 (rows, cols) matrix
where a finite value is the cost to *enter* that cell and np.inf marks
an impassable wall. make_grid() wraps them and guarantees the chosen
start and goal are free and mutually reachable.

Public API
----------
make_open(rows, cols)                              — uniform-cost open field
make_random_obstacles(rows, cols, density, seed)   — scattered wall blocks
make_maze(rows, cols, seed)                         — perfect maze (Prim's algorithm)
make_weighted_terrain(rows, cols, seed)             — no walls, costly terrain patches

make_grid(preset, rows, cols, seed, density) — single entry-point dispatcher

PRESET_KEYS / PRESET_NAMES — ordered keys/names kept in sync with app.py
"""

from __future__ import annotations

from collections import deque

import numpy as np

Cell = tuple[int, int]

PRESET_KEYS = ["open", "random_obstacles", "maze", "weighted_terrain"]
PRESET_NAMES = ["Open field", "Random obstacles", "Maze", "Weighted terrain"]


def _odd(n: int) -> int:
    """Round up to the nearest odd number — maze carving needs odd dimensions
    so every "room" cell lands on an even (row, col) coordinate."""
    return n if n % 2 == 1 else n + 1


def make_open(rows: int, cols: int) -> np.ndarray:
    """Uniform-cost field with no obstacles at all.

    Useful as a baseline: on a uniform-cost grid Dijkstra degenerates into
    plain breadth-first search (every unvisited neighbour is equally
    attractive), while A* still trims the search by biasing expansion
    toward the goal — this preset isolates that effect cleanly.
    """
    return np.ones((rows, cols), dtype=float)


def make_random_obstacles(
    rows: int,
    cols: int,
    density: float = 0.25,
    seed: int = 0,
) -> np.ndarray:
    """Uniform-cost field with randomly scattered impassable blocks."""
    rng = np.random.default_rng(seed)
    grid = np.ones((rows, cols), dtype=float)
    grid[rng.random((rows, cols)) < density] = np.inf
    return grid


def make_maze(rows: int, cols: int, seed: int = 0) -> np.ndarray:
    """Perfect maze carved with randomised Prim's algorithm.

    "Rooms" live on even (row, col) coordinates; the odd cells between
    them are walls until carved open. A perfect maze has exactly one
    path between any two rooms, so a dead end looks identical to the
    true corridor until fully explored — which is exactly what makes
    this preset a good demo of how much the A* heuristic can prune
    compared to undirected Dijkstra.
    """
    rows, cols = _odd(rows), _odd(cols)
    rng = np.random.default_rng(seed)
    grid = np.full((rows, cols), np.inf)

    def carved(r: int, c: int) -> bool:
        return np.isfinite(grid[r, c])

    start = (0, 0)
    grid[start] = 1.0
    frontier: list[tuple[int, int, int, int]] = []

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not carved(nr, nc):
                frontier.append((r, c, nr, nc))

    add_frontier(*start)
    while frontier:
        idx = int(rng.integers(len(frontier)))
        r, c, nr, nc = frontier.pop(idx)
        if carved(nr, nc):
            continue
        grid[nr, nc] = 1.0
        grid[(r + nr) // 2, (c + nc) // 2] = 1.0  # knock down the wall between
        add_frontier(nr, nc)

    return grid


def make_weighted_terrain(rows: int, cols: int, seed: int = 0) -> np.ndarray:
    """Open field (no walls) with patches of costly terrain — mud and water.

    Because every cell is reachable, this preset isolates *cost* from
    *connectivity*: the shortest path in number of steps is not always
    the cheapest path, which is exactly the distinction Dijkstra makes
    over plain BFS.
    """
    rng = np.random.default_rng(seed)
    grid = np.ones((rows, cols), dtype=float)
    for cost, n_patches, radius in ((3.0, 5, 4), (6.0, 3, 3)):
        for _ in range(n_patches):
            cr, cc = int(rng.integers(rows)), int(rng.integers(cols))
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    r, c = cr + dr, cc + dc
                    if (
                        0 <= r < rows
                        and 0 <= c < cols
                        and dr * dr + dc * dc <= radius * radius
                        and rng.random() < 0.75
                    ):
                        grid[r, c] = max(grid[r, c], cost)
    return grid


def _bfs_connected(grid: np.ndarray, start: Cell, goal: Cell) -> bool:
    rows, cols = grid.shape
    seen = np.zeros((rows, cols), dtype=bool)
    q = deque([start])
    seen[start] = True
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and not seen[nr, nc]
                and np.isfinite(grid[nr, nc])
            ):
                seen[nr, nc] = True
                q.append((nr, nc))
    return False


def _carve_fallback_path(grid: np.ndarray, start: Cell, goal: Cell) -> np.ndarray:
    """Force an L-shaped corridor open so start and goal are always reachable.

    Random obstacle placement can occasionally wall off the goal entirely;
    rather than silently failing, we guarantee a (possibly ugly) route.
    """
    grid = grid.copy()
    r, c = start
    gr, gc = goal
    grid[r, c] = 1.0
    while c != gc:
        c += 1 if gc > c else -1
        grid[r, c] = 1.0
    while r != gr:
        r += 1 if gr > r else -1
        grid[r, c] = 1.0
    return grid


def make_grid(
    preset: str,
    rows: int,
    cols: int,
    seed: int = 0,
    density: float = 0.25,
) -> tuple[np.ndarray, Cell, Cell]:
    """Return (grid_cost, start, goal) for the requested preset.

    start is always the top-left cell and goal the bottom-right cell;
    both are forced free and, if necessary, forcibly reconnected.
    """
    if preset == "maze":
        grid = make_maze(rows, cols, seed)
    elif preset == "random_obstacles":
        grid = make_random_obstacles(rows, cols, density, seed)
    elif preset == "weighted_terrain":
        grid = make_weighted_terrain(rows, cols, seed)
    elif preset == "open":
        grid = make_open(rows, cols)
    else:
        raise ValueError(f"Unknown preset {preset!r}. Choose from: {PRESET_KEYS}")

    rows, cols = grid.shape
    start, goal = (0, 0), (rows - 1, cols - 1)
    grid[start] = 1.0
    grid[goal] = 1.0
    if not _bfs_connected(grid, start, goal):
        grid = _carve_fallback_path(grid, start, goal)
    return grid, start, goal
