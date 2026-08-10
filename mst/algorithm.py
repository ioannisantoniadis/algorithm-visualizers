"""
algorithm.py — from-scratch Kruskal's and Prim's algorithms for the
Minimum Spanning Tree (MST) problem.

Both algorithms are greedy, and both are provably optimal — but they lean
on two *different* halves of the same proof:

  Cycle property:  the heaviest edge on any cycle is never in an MST.
                   Kruskal exploits this directly — it sorts every edge
                   globally and rejects any edge that would close a cycle
                   with edges already accepted (checked in ~O(1) via a
                   Union-Find / disjoint-set structure). Many disconnected
                   fragments grow at once and merge as the sort progresses.

  Cut property:    for any cut that separates the graph into two sides,
                   the cheapest edge crossing it is always safe to add.
                   Prim exploits this directly — it grows a single tree
                   from one seed node, and at every step adds the cheapest
                   edge crossing the cut between "in the tree" and
                   "everything else".

Every edge examined — accepted or rejected — is recorded as a Snapshot so
the visualiser can replay the construction one edge at a time.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .data import Edge

Phase = Literal["init", "accept", "reject", "grow", "done"]


@dataclass
class Snapshot:
    """State of the MST construction immediately after considering (Kruskal)
    or adding (Prim) exactly one edge."""

    positions: np.ndarray  # (N, 2) node layout — constant across snapshots
    edges: list[Edge]  # full candidate edge list — constant across snapshots
    algorithm: Literal["kruskal", "prim"]

    step: int  # edges considered (Kruskal) / added (Prim) so far
    phase: Phase
    current_edge: Edge | None  # edge just accepted / rejected / grown

    mst_edges: list[Edge] = field(default_factory=list)
    total_weight: float = 0.0
    n_components: int = 1  # connected components among {accepted edges}; 1 == spanning tree done

    # --- Kruskal: Union-Find state -----------------------------------------
    component_id: np.ndarray | None = None  # (N,) current UF root per node
    rejected_edges: list[Edge] = field(default_factory=list)
    cycle_path: list[int] = field(default_factory=list)  # MST path that + current_edge closes a cycle
    sorted_index: int = 0  # position reached in the globally sorted edge order

    # --- Prim: frontier state -----------------------------------------------
    in_tree: np.ndarray | None = None  # (N,) bool — which nodes have joined the tree
    frontier_preview: list[Edge] = field(default_factory=list)  # best few crossing edges

    @property
    def title(self) -> str:
        name = "Kruskal" if self.algorithm == "kruskal" else "Prim"
        if self.phase == "init":
            return f"{name} — ready"
        if self.phase == "done":
            n_edges = len(self.mst_edges)
            forest_note = "" if self.n_components == 1 else f" · {self.n_components} fragments (disconnected)"
            return f"{name} — done · {n_edges} edges · total weight {self.total_weight:.1f}{forest_note}"
        e = self.current_edge
        if self.phase == "accept":
            return f"{name} — accept ({e.u}, {e.v}) w={e.w:.1f}"
        if self.phase == "reject":
            return f"{name} — reject ({e.u}, {e.v}) w={e.w:.1f} — closes a cycle"
        return f"{name} — grow: add ({e.u}, {e.v}) w={e.w:.1f}"


class _UnionFind:
    """Disjoint-set with union-by-size and path compression — the data
    structure that makes Kruskal's "would this close a cycle?" check run
    in ~O(1) amortised instead of an O(n) tree walk per candidate edge."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression: flatten as we go
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        """Merge the sets containing a and b. Returns False (no-op) if they
        were already the same set — that's precisely a would-be cycle."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def _path_in_forest(mst_edges: list[Edge], u: int, v: int) -> list[int]:
    """BFS path from u to v using only already-accepted MST edges.

    This path, together with the edge that just got rejected, is exactly
    the cycle the cycle property is about: every other edge on this path
    was accepted *earlier* (Kruskal processes edges in increasing weight
    order), so the rejected edge is guaranteed to be the heaviest edge on
    the cycle it would have formed.
    """
    adj: dict[int, list[int]] = {}
    for e in mst_edges:
        adj.setdefault(e.u, []).append(e.v)
        adj.setdefault(e.v, []).append(e.u)
    if u not in adj or v not in adj:
        return []

    prev: dict[int, int | None] = {u: None}
    q = deque([u])
    while q:
        node = q.popleft()
        if node == v:
            break
        for nxt in adj.get(node, []):
            if nxt not in prev:
                prev[nxt] = node
                q.append(nxt)
    if v not in prev:
        return []

    path = [v]
    while path[-1] != u:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def kruskal(positions: np.ndarray, edges: list[Edge]) -> list[Snapshot]:
    """Run Kruskal's algorithm and return every edge-consideration step.

    Returns
    -------
    [init, step_1, step_2, ..., done] where each step_i has phase "accept"
    or "reject". The list stops early once a full spanning tree (n-1
    accepted edges) is found; any edges after that point were never even
    examined, exactly as the real algorithm would skip them once done.
    """
    n = len(positions)
    order = sorted(edges, key=lambda e: (e.w, e.u, e.v))
    uf = _UnionFind(n)

    def comp_ids() -> np.ndarray:
        return np.array([uf.find(i) for i in range(n)])

    mst: list[Edge] = []
    rejected: list[Edge] = []
    total = 0.0

    snapshots: list[Snapshot] = [
        Snapshot(
            positions=positions, edges=edges, algorithm="kruskal",
            step=0, phase="init", current_edge=None,
            component_id=comp_ids(), n_components=n,
        )
    ]

    for i, e in enumerate(order, start=1):
        if uf.union(e.u, e.v):
            mst.append(e)
            total += e.w
            phase: Phase = "accept"
            cycle_path: list[int] = []
        else:
            rejected.append(e)
            phase = "reject"
            cycle_path = _path_in_forest(mst, e.u, e.v)

        snapshots.append(
            Snapshot(
                positions=positions, edges=edges, algorithm="kruskal",
                step=i, phase=phase, current_edge=e,
                mst_edges=list(mst), total_weight=total,
                n_components=n - len(mst),
                component_id=comp_ids(), rejected_edges=list(rejected),
                cycle_path=cycle_path, sorted_index=i,
            )
        )
        if len(mst) == n - 1:
            break

    last = snapshots[-1]
    snapshots.append(
        Snapshot(
            positions=positions, edges=edges, algorithm="kruskal",
            step=last.step, phase="done", current_edge=None,
            mst_edges=list(mst), total_weight=total,
            n_components=n - len(mst),
            component_id=comp_ids(), rejected_edges=list(rejected),
            sorted_index=last.sorted_index,
        )
    )
    return snapshots


def prim(positions: np.ndarray, edges: list[Edge], start: int = 0, frontier_limit: int = 8) -> list[Snapshot]:
    """Run Prim's algorithm from `start` and return every growth step.

    Parameters
    ----------
    start:          seed node the tree grows from
    frontier_limit: how many candidate crossing edges to keep in each
                    snapshot's frontier_preview, for the side panel

    Returns
    -------
    [init, grow_1, grow_2, ..., done]. If the graph is disconnected from
    `start`, the frontier heap empties before every node joins and the
    tree simply stops growing — n_components on the final snapshot will
    be > 1, signalling a spanning *forest* rather than a spanning tree.
    """
    n = len(positions)
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    for e in edges:
        adj[e.u].append((e.v, e.w))
        adj[e.v].append((e.u, e.w))

    in_tree = np.zeros(n, dtype=bool)
    in_tree[start] = True
    heap: list[tuple[float, int, int]] = [(w, start, v) for v, w in adj[start]]
    heapq.heapify(heap)

    mst: list[Edge] = []
    total = 0.0

    def preview() -> list[Edge]:
        """Best few live (non-stale, non-tree-internal) crossing edges."""
        seen: set[int] = set()
        live: list[Edge] = []
        for w, u, v in heap:
            if v in seen or in_tree[v]:
                continue
            seen.add(v)
            live.append(Edge(u, v, w))
        live.sort(key=lambda e: e.w)
        return live[:frontier_limit]

    snapshots: list[Snapshot] = [
        Snapshot(
            positions=positions, edges=edges, algorithm="prim",
            step=0, phase="init", current_edge=None,
            in_tree=in_tree.copy(), frontier_preview=preview(),
            n_components=n,
        )
    ]

    step = 0
    while heap and len(mst) < n - 1:
        w, u, v = heapq.heappop(heap)
        if in_tree[v]:
            continue  # stale entry: v already joined via a cheaper edge
        in_tree[v] = True
        e = Edge(min(u, v), max(u, v), w)
        mst.append(e)
        total += w
        step += 1

        for nbr, ww in adj[v]:
            if not in_tree[nbr]:
                heapq.heappush(heap, (ww, v, nbr))

        snapshots.append(
            Snapshot(
                positions=positions, edges=edges, algorithm="prim",
                step=step, phase="grow", current_edge=e,
                mst_edges=list(mst), total_weight=total,
                in_tree=in_tree.copy(), frontier_preview=preview(),
                n_components=n - len(mst),
            )
        )

    last = snapshots[-1]
    snapshots.append(
        Snapshot(
            positions=positions, edges=edges, algorithm="prim",
            step=last.step, phase="done", current_edge=None,
            mst_edges=list(mst), total_weight=total,
            in_tree=in_tree.copy(), frontier_preview=[],
            n_components=n - len(mst),
        )
    )
    return snapshots


def run(algorithm: Literal["kruskal", "prim"], positions: np.ndarray, edges: list[Edge], start: int = 0) -> list[Snapshot]:
    """Single entry point used by app.py."""
    if algorithm == "kruskal":
        return kruskal(positions, edges)
    if algorithm == "prim":
        return prim(positions, edges, start=start)
    raise ValueError(f"Unknown algorithm {algorithm!r}. Choose 'kruskal' or 'prim'.")
