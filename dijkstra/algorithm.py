"""
algorithm.py — from-scratch Dijkstra / A* on a weighted grid

Both algorithms share one implementation: A* is just Dijkstra with
priority f = g + h instead of f = g, where g is the exact cost-so-far
and h is an admissible (never-overestimating) estimate of the
remaining cost. Setting h ≡ 0 recovers plain Dijkstra exactly — the
`heuristic_mode` parameter is simply ignored when algorithm="dijkstra".

Every time a node is popped off the priority queue and finalised
(its shortest distance is locked in), the full search state — distances,
the visited set, the open set, and the tentative path — is recorded as
a Snapshot, so the visualiser can replay the frontier's growth one node
at a time.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Cell = tuple[int, int]


@dataclass
class Snapshot:
    """State of the search immediately after finalising exactly one node."""

    grid_cost: np.ndarray  # (rows, cols); np.inf marks a wall — constant across snapshots
    start: Cell
    goal: Cell
    algorithm: Literal["dijkstra", "astar"]

    step: int  # nodes finalised so far (0 = init frame, before anything is expanded)
    phase: Literal["init", "visit", "found", "no_path"]
    current: Cell | None  # node just finalised this step (None for init / no_path)

    dist: np.ndarray  # (rows, cols) best known cost-so-far; np.inf where still unknown
    closed: np.ndarray  # (rows, cols) bool — finalised ("visited") nodes
    open_mask: np.ndarray  # (rows, cols) bool — nodes currently sitting in the open set

    # Best few live entries in the open set, sorted by priority — feeds the
    # "priority queue" side panel. Each item is (priority, g_cost, cell).
    frontier_preview: list[tuple[float, float, Cell]] = field(default_factory=list)
    updated: list[Cell] = field(default_factory=list)  # neighbours relaxed this step
    path_so_far: list[Cell] = field(default_factory=list)  # start → current, via came_from
    path: list[Cell] | None = None  # final shortest path — set only when phase == "found"
    total_cost: float | None = None

    @property
    def title(self) -> str:
        name = "A*" if self.algorithm == "astar" else "Dijkstra"
        if self.phase == "init":
            return f"{name} — ready to search"
        if self.phase == "found":
            return f"{name} — goal reached · cost {self.total_cost:.1f} · {self.step} nodes visited"
        if self.phase == "no_path":
            return f"{name} — no path exists · {self.step} nodes exhausted"
        return f"{name} — step {self.step}: expanding ({self.current[0]}, {self.current[1]})"


def _neighbors(node: Cell, rows: int, cols: int):
    r, c = node
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield (nr, nc)


def _heuristic(a: Cell, b: Cell, mode: Literal["manhattan", "euclidean"], min_cost: float) -> float:
    """Admissible cost-to-go estimate.

    Scaling the raw grid distance by the cheapest cell cost in the whole
    grid keeps the heuristic a true lower bound even on weighted terrain
    (where a "closer" cell might still be reached faster than a "farther"
    one) — that lower-bound property is what lets A* guarantee the
    optimal path instead of just a fast-but-wrong one.
    """
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    raw = math.hypot(dr, dc) if mode == "euclidean" else float(dr + dc)
    return raw * min_cost


def _reconstruct(came_from: dict[Cell, Cell], node: Cell, start: Cell) -> list[Cell]:
    path = [node]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def _frontier_preview(
    heap: list[tuple[float, float, Cell]],
    closed: np.ndarray,
    dist: np.ndarray,
    limit: int,
) -> list[tuple[float, float, Cell]]:
    """De-duplicated view of the best `limit` live entries in the open set.

    We push a new heap entry every time a node's distance improves rather
    than decrease-key the old one (heapq has no cheap decrease-key), so
    the heap accumulates stale entries. This filters those out so the
    priority-queue panel always reflects genuinely current information.
    """
    seen: set[Cell] = set()
    live: list[tuple[float, float, Cell]] = []
    for priority, g, node in heap:
        if node in seen or closed[node] or g > dist[node] + 1e-9:
            continue
        seen.add(node)
        live.append((priority, g, node))
    live.sort(key=lambda t: t[0])
    return live[:limit]


def search(
    grid_cost: np.ndarray,
    start: Cell,
    goal: Cell,
    algorithm: Literal["dijkstra", "astar"] = "dijkstra",
    heuristic_mode: Literal["manhattan", "euclidean"] = "manhattan",
    frontier_limit: int = 10,
) -> list[Snapshot]:
    """Run Dijkstra or A* on grid_cost and return every finalisation step.

    Parameters
    ----------
    grid_cost:      (rows, cols) float array; np.inf marks impassable walls
    start, goal:    (row, col) cells — must be free
    algorithm:      "dijkstra" (h ≡ 0) or "astar" (h = heuristic_mode estimate)
    heuristic_mode: "manhattan" (matches 4-directional moves) or "euclidean"
    frontier_limit: how many open-set entries to keep in each snapshot's
                    frontier_preview, for the priority-queue side panel

    Returns
    -------
    [init, visit_1, visit_2, ..., found]  or  [init, ..., no_path] if the
    goal is unreachable from start.
    """
    rows, cols = grid_cost.shape
    finite_costs = grid_cost[np.isfinite(grid_cost)]
    min_cost = float(finite_costs.min()) if finite_costs.size else 1.0

    def h(node: Cell) -> float:
        return _heuristic(node, goal, heuristic_mode, min_cost) if algorithm == "astar" else 0.0

    dist = np.full((rows, cols), np.inf)
    dist[start] = 0.0
    closed = np.zeros((rows, cols), dtype=bool)
    came_from: dict[Cell, Cell] = {}

    open_heap: list[tuple[float, float, Cell]] = [(h(start), 0.0, start)]
    open_mask = np.zeros((rows, cols), dtype=bool)
    open_mask[start] = True

    def snap(**kwargs) -> Snapshot:
        return Snapshot(
            grid_cost=grid_cost, start=start, goal=goal, algorithm=algorithm, **kwargs
        )

    snapshots: list[Snapshot] = [
        snap(
            step=0,
            phase="init",
            current=None,
            dist=dist.copy(),
            closed=closed.copy(),
            open_mask=open_mask.copy(),
            frontier_preview=_frontier_preview(open_heap, closed, dist, frontier_limit),
            path_so_far=[start],
        )
    ]

    step = 0
    while open_heap:
        priority, g, node = heapq.heappop(open_heap)
        if closed[node] or g > dist[node] + 1e-9:
            continue  # stale entry left behind by an earlier, worse push

        closed[node] = True
        open_mask[node] = False
        step += 1
        path_so_far = _reconstruct(came_from, node, start)

        if node == goal:
            snapshots.append(
                snap(
                    step=step,
                    phase="found",
                    current=node,
                    dist=dist.copy(),
                    closed=closed.copy(),
                    open_mask=open_mask.copy(),
                    frontier_preview=_frontier_preview(open_heap, closed, dist, frontier_limit),
                    path_so_far=path_so_far,
                    path=path_so_far,
                    total_cost=float(dist[node]),
                )
            )
            return snapshots

        updated: list[Cell] = []
        for neighbor in _neighbors(node, rows, cols):
            step_cost = grid_cost[neighbor]
            if not np.isfinite(step_cost) or closed[neighbor]:
                continue
            new_g = dist[node] + step_cost
            if new_g < dist[neighbor] - 1e-9:
                dist[neighbor] = new_g
                came_from[neighbor] = node
                open_mask[neighbor] = True
                updated.append(neighbor)
                heapq.heappush(open_heap, (new_g + h(neighbor), new_g, neighbor))

        snapshots.append(
            snap(
                step=step,
                phase="visit",
                current=node,
                dist=dist.copy(),
                closed=closed.copy(),
                open_mask=open_mask.copy(),
                frontier_preview=_frontier_preview(open_heap, closed, dist, frontier_limit),
                updated=updated,
                path_so_far=path_so_far,
            )
        )

    snapshots.append(
        snap(
            step=step,
            phase="no_path",
            current=None,
            dist=dist.copy(),
            closed=closed.copy(),
            open_mask=open_mask.copy(),
            frontier_preview=[],
            path_so_far=[],
        )
    )
    return snapshots
