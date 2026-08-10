"""
data.py — weighted graph generators for the MST visualiser

All generators return (positions, edges):
  positions — float64 array of shape (n_nodes, 2); a 2-D layout used only
              for drawing. It never affects which edges exist or how much
              they cost (except in the Euclidean presets, where distance
              *is* the weight by definition).
  edges     — list[Edge], each a canonical (u < v, weight) triple with no
              duplicates and no self-loops.

Public API
----------
make_random_euclidean(n_nodes, seed)              — random points, complete
                                                     graph, weight = distance
make_clustered(n_nodes, n_clusters, seed)         — clustered points,
                                                     complete graph, weight = distance
make_sparse_random(n_nodes, edge_prob, seed)      — circular layout, a random
                                                     subset of edges, random weights
make_grid_lattice(rows, cols, seed)               — lattice layout, 4-connected,
                                                     random weights

make_graph(preset, ...) — single entry-point dispatcher used by app.py

PRESET_KEYS / PRESET_NAMES — ordered keys/names kept in sync with app.py
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Edge(NamedTuple):
    """One weighted, undirected edge. Always stored with u < v so the same
    edge never appears twice under two different orderings."""

    u: int
    v: int
    w: float


PRESET_KEYS = ["random_euclidean", "clustered", "sparse_random", "grid_lattice"]
PRESET_NAMES = ["Random Euclidean", "Clustered blobs", "Sparse random graph", "Grid lattice"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def complete_graph_edges(positions: np.ndarray) -> list[Edge]:
    """Every pair of nodes connected once, weighted by Euclidean distance.

    This is the classic "Euclidean MST" setup: since the triangle
    inequality holds automatically for real distances, the resulting MST
    is always a *plausible* physical network (pipes, cables, roads) rather
    than an abstract weighted graph.
    """
    n = len(positions)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(np.hypot(*(positions[i] - positions[j])))
            edges.append(Edge(i, j, round(w, 3)))
    return edges


def _union_find_components(n: int, edges: list[Edge]) -> dict[int, list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges:
        union(e.u, e.v)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return comps


def _ensure_connected(
    n: int, edges: list[Edge], rng: np.random.Generator, weight_range: tuple[float, float]
) -> list[Edge]:
    """Bridge any disconnected fragments with cheap random edges.

    Random subsetting of edges can accidentally isolate a node or a whole
    cluster of nodes. MST is only defined (as a *tree*, rather than a
    forest) on a connected graph, so every generator here guarantees
    connectivity up front rather than surprising the learner with a
    silently incomplete tree.
    """
    comps = _union_find_components(n, edges)
    reps = list(comps.keys())
    if len(reps) <= 1:
        return edges
    extra = []
    lo, hi = weight_range
    for i in range(1, len(reps)):
        u, v = comps[reps[0]][0], comps[reps[i]][0]
        w = float(rng.integers(lo, hi + 1))
        extra.append(Edge(min(u, v), max(u, v), w))
    return edges + extra


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def make_random_euclidean(n_nodes: int = 12, seed: int = 0) -> tuple[np.ndarray, list[Edge]]:
    """Random points in the plane, fully connected, weight = distance.

    The textbook "Euclidean MST": with every pair connected, both Kruskal
    and Prim have to sift through a dense graph to find the same sparse
    (n-1)-edge tree — a good baseline before sparser, more structured graphs.
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.5, 9.5, size=(n_nodes, 2))
    return positions, complete_graph_edges(positions)


def make_clustered(n_nodes: int = 14, n_clusters: int = 3, seed: int = 0) -> tuple[np.ndarray, list[Edge]]:
    """Points drawn from a few well-separated Gaussian blobs, complete graph.

    Intra-cluster edges are cheap, inter-cluster edges are expensive. Both
    algorithms end up building a near-complete local tree inside each blob
    almost immediately, then spend most of their effort deciding on the one
    or two cheapest "bridge" edges connecting the blobs — a clean way to see
    the cut property pick out a single crossing edge from many candidates.
    """
    rng = np.random.default_rng(seed)
    cols = int(np.ceil(np.sqrt(n_clusters)))
    rows = int(np.ceil(n_clusters / cols))
    spacing = 7.0

    grid_x, grid_y = np.meshgrid(np.arange(cols) * spacing, np.arange(rows) * spacing)
    centres = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:n_clusters]

    sizes = np.full(n_clusters, n_nodes // n_clusters, dtype=int)
    sizes[: n_nodes % n_clusters] += 1

    parts = []
    for centre, size in zip(centres, sizes):
        parts.append(rng.normal(loc=centre, scale=0.8, size=(size, 2)))
    positions = np.vstack(parts)
    return positions, complete_graph_edges(positions)


def make_sparse_random(
    n_nodes: int = 14, edge_prob: float = 0.25, seed: int = 0
) -> tuple[np.ndarray, list[Edge]]:
    """Nodes on a circle, a random subset of edges with random weights.

    Sparse graphs are where the two algorithms' running-time trade-offs
    actually diverge: Kruskal's cost is dominated by sorting all edges,
    while Prim's is dominated by how many nodes/edges the frontier touches.
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    positions = np.column_stack([5.0 * np.cos(angles), 5.0 * np.sin(angles)])
    positions += rng.normal(0, 0.25, size=positions.shape)

    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                edges.append(Edge(i, j, float(rng.integers(1, 20))))

    return positions, _ensure_connected(n_nodes, edges, rng, (1, 20))


def make_grid_lattice(rows: int = 4, cols: int = 4, seed: int = 0) -> tuple[np.ndarray, list[Edge]]:
    """A rows x cols lattice, 4-connected, random edge weights.

    Grid graphs make the MST's shape legible at a glance — a good graph
    for spotting exactly which candidate edges Kruskal rejects as cycles,
    since most nodes have several short, similarly-weighted options.
    """
    rng = np.random.default_rng(seed)

    def idx(r: int, c: int) -> int:
        return r * cols + c

    positions = np.array(
        [[float(c), float(rows - 1 - r)] for r in range(rows) for c in range(cols)]
    )

    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append(Edge(idx(r, c), idx(r, c + 1), float(rng.integers(1, 20))))
            if r + 1 < rows:
                edges.append(Edge(idx(r, c), idx(r + 1, c), float(rng.integers(1, 20))))
    return positions, edges


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def make_graph(
    preset: str,
    n_nodes: int = 12,
    seed: int = 0,
    n_clusters: int = 3,
    edge_prob: float = 0.25,
    rows: int = 4,
    cols: int = 4,
) -> tuple[np.ndarray, list[Edge]]:
    """Return (positions, edges) for the requested preset.

    Parameters not relevant to a given preset are simply ignored (e.g.
    n_nodes for grid_lattice, whose size comes from rows x cols instead).
    """
    if preset == "random_euclidean":
        return make_random_euclidean(n_nodes, seed)
    if preset == "clustered":
        return make_clustered(n_nodes, n_clusters, seed)
    if preset == "sparse_random":
        return make_sparse_random(n_nodes, edge_prob, seed)
    if preset == "grid_lattice":
        return make_grid_lattice(rows, cols, seed)
    raise ValueError(f"Unknown preset {preset!r}. Choose from: {PRESET_KEYS}")
