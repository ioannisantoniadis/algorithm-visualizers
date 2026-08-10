"""
visualize.py — Plotly figure builders for the MST visualiser

Edges are drawn as line-segment traces (one trace per visual "role" —
background, MST-so-far, rejected, cycle-path, cut, current — batched into
a single go.Scatter each with None separators, rather than one trace per
edge, so even dense complete graphs render quickly). Nodes are drawn as a
scatter overlay coloured by Union-Find component (Kruskal) or tree
membership (Prim).

Public functions
-----------------
build_figure(snapshots)   animated figure, one Plotly frame per Snapshot
                          with its own slider (kept as a self-contained
                          alternative to the Streamlit-driven playback in
                          app.py)
make_static_figure(snap)  single non-animated frame — what the app
                          actually renders each step
make_editor_figure(positions, edges)  clickable canvas for placing / removing
                                       nodes while drawing your own graph
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .algorithm import Snapshot
from .data import Edge

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
_PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
            "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]
_FONT_FAMILY = "Inter, -apple-system, Segoe UI, sans-serif"

_BG_EDGE_COLOR = "#c9ccd1"
_MST_COLOR = "#14b8a6"           # teal — accepted / grown into the tree
_REJECT_COLOR = "#f43f5e"        # rose — rejected (would close a cycle)
_CYCLE_COLOR = "#f59e0b"         # amber — cycle path highlight
_CUT_COLOR = "#f59e0b"           # amber — cut (Prim frontier) highlight
_CURRENT_ACCEPT_COLOR = "#14b8a6"
_CURRENT_REJECT_COLOR = "#f43f5e"
_TREE_NODE_COLOR = "#14b8a6"     # teal — already in the tree (Prim)
_OUTSIDE_NODE_COLOR = "#a1a1aa"  # neutral zinc — not yet reached
_START_NODE_COLOR = "#6366f1"    # indigo (primary) — edge just added (Prim)


# ---------------------------------------------------------------------------
# Low-level trace builders
# ---------------------------------------------------------------------------

def _edge_trace(
    edges: list[Edge],
    positions: np.ndarray,
    color: str,
    width: float,
    dash: str | None = None,
    name: str = "edges",
    opacity: float = 1.0,
    showlegend: bool = True,
) -> go.Scatter:
    """A batch of edges sharing one visual style, as a single lines trace."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    for e in edges:
        x0, y0 = positions[e.u]
        x1, y1 = positions[e.v]
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    return go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=color, width=width, dash=dash),
        opacity=opacity, hoverinfo="skip",
        name=name, showlegend=showlegend,
    )


def _edge_hover_trace(edges: list[Edge], positions: np.ndarray) -> go.Scatter:
    """Invisible markers at edge midpoints so hovering anywhere on the
    graph reveals that edge's weight, without cluttering the drawing."""
    mx = [(positions[e.u, 0] + positions[e.v, 0]) / 2 for e in edges]
    my = [(positions[e.u, 1] + positions[e.v, 1]) / 2 for e in edges]
    text = [f"({e.u}, {e.v}) — weight {e.w:.1f}" for e in edges]
    return go.Scatter(
        x=mx, y=my, mode="markers",
        marker=dict(size=14, color="rgba(0,0,0,0)"),
        hovertext=text, hoverinfo="text",
        name="Edge weights", showlegend=False,
    )


def _node_trace(
    positions: np.ndarray,
    colors: list[str],
    name: str = "Nodes",
    size: int = 22,
    clickable: bool = False,
) -> go.Scatter:
    # NOTE: hoverinfo="skip" disables Plotly's hit-testing entirely, which
    # also silently kills click/selection events — not just tooltips. The
    # read-only replay traces don't need clicks, so they keep "skip". The
    # editor's node trace (clickable=True) needs "none" instead: no tooltip
    # text, but clicks still register with st.plotly_chart(on_select=...).
    return go.Scatter(
        x=positions[:, 0], y=positions[:, 1], mode="markers+text",
        marker=dict(size=size, color=colors, line=dict(width=1.5, color="white")),
        text=[str(i) for i in range(len(positions))],
        textposition="middle center",
        textfont=dict(size=9, color="white"),
        name=name, hoverinfo="none" if clickable else "skip", showlegend=False,
    )


def _graph_layout(positions: np.ndarray, title: str) -> go.Layout:
    """Shared layout for the node-link diagrams.

    The axes stay hidden (node positions are a schematic layout, not
    meaningful coordinate units — e.g. the circle layout for sparse_random
    or the grid layout for grid_lattice), but font, background and legend
    styling are kept in sync with the rest of the portfolio's chart theme.
    """
    pad = 1.0
    xr = [float(positions[:, 0].min()) - pad, float(positions[:, 0].max()) + pad]
    yr = [float(positions[:, 1].min()) - pad, float(positions[:, 1].max()) + pad]
    return go.Layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15, color="#18181b")),
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=xr),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=yr, scaleanchor="x"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=560,
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.04, x=0.5, xanchor="center",
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#e4e4e7", borderwidth=1,
            font=dict(size=12),
        ),
    )


def _component_colors(component_id: np.ndarray) -> list[str]:
    """One colour per current Union-Find root, cycling the palette.

    Roots merge over time (the whole point of the algorithm), so this
    recolours nodes every step — watching same-coloured islands fuse is
    a direct, visual readout of Kruskal's union operations.
    """
    roots = sorted(set(component_id.tolist()))
    color_of = {r: _PALETTE[i % len(_PALETTE)] for i, r in enumerate(roots)}
    return [color_of[c] for c in component_id.tolist()]


# ---------------------------------------------------------------------------
# Per-algorithm frame assembly
# ---------------------------------------------------------------------------

def _kruskal_traces(snap: Snapshot) -> list[go.BaseTraceType]:
    traces: list[go.BaseTraceType] = [
        _edge_trace(snap.edges, snap.positions, _BG_EDGE_COLOR, 1, name="All candidate edges"),
        _edge_hover_trace(snap.edges, snap.positions),
    ]

    if snap.rejected_edges:
        traces.append(
            _edge_trace(snap.rejected_edges, snap.positions, _REJECT_COLOR, 1.5,
                        dash="dot", name="Rejected (would cycle)", opacity=0.85)
        )

    if snap.mst_edges:
        traces.append(_edge_trace(snap.mst_edges, snap.positions, _MST_COLOR, 4, name="MST so far"))

    if snap.cycle_path and len(snap.cycle_path) > 1:
        path_edges = [
            Edge(snap.cycle_path[i], snap.cycle_path[i + 1], 0.0)
            for i in range(len(snap.cycle_path) - 1)
        ]
        traces.append(
            _edge_trace(path_edges, snap.positions, _CYCLE_COLOR, 3, dash="dash",
                        name="Cycle this edge would close")
        )

    if snap.current_edge is not None:
        color = _CURRENT_ACCEPT_COLOR if snap.phase == "accept" else _CURRENT_REJECT_COLOR
        label = "Just accepted" if snap.phase == "accept" else "Just rejected"
        traces.append(_edge_trace([snap.current_edge], snap.positions, color, 5, name=label))

    colors = (
        _component_colors(snap.component_id)
        if snap.component_id is not None
        else ["#4c72b0"] * len(snap.positions)
    )
    traces.append(_node_trace(snap.positions, colors, name="Component"))
    return traces


def _prim_traces(snap: Snapshot) -> list[go.BaseTraceType]:
    traces: list[go.BaseTraceType] = [
        _edge_trace(snap.edges, snap.positions, _BG_EDGE_COLOR, 1, name="All candidate edges"),
        _edge_hover_trace(snap.edges, snap.positions),
    ]

    if snap.in_tree is not None:
        cut_edges = [e for e in snap.edges if bool(snap.in_tree[e.u]) != bool(snap.in_tree[e.v])]
        if cut_edges:
            traces.append(
                _edge_trace(cut_edges, snap.positions, _CUT_COLOR, 2, dash="dot",
                            name="Cut (tree ↔ outside)", opacity=0.9)
            )

    if snap.mst_edges:
        traces.append(_edge_trace(snap.mst_edges, snap.positions, _MST_COLOR, 4, name="Tree so far"))

    if snap.current_edge is not None:
        traces.append(
            _edge_trace([snap.current_edge], snap.positions, _START_NODE_COLOR, 5, name="Just added")
        )

    if snap.in_tree is not None:
        colors = [_TREE_NODE_COLOR if t else _OUTSIDE_NODE_COLOR for t in snap.in_tree.tolist()]
    else:
        colors = [_OUTSIDE_NODE_COLOR] * len(snap.positions)
    traces.append(_node_trace(snap.positions, colors, name="Tree membership"))
    return traces


def _frame_traces(snap: Snapshot) -> list[go.BaseTraceType]:
    return _kruskal_traces(snap) if snap.algorithm == "kruskal" else _prim_traces(snap)


# ---------------------------------------------------------------------------
# Public figure builders
# ---------------------------------------------------------------------------

def build_figure(snapshots: list[Snapshot]) -> go.Figure:
    """Animated figure with one Plotly frame per Snapshot and a slider."""
    if not snapshots:
        raise ValueError("snapshot list is empty")

    frames = [
        go.Frame(data=_frame_traces(s), name=str(i), layout=go.Layout(title_text=s.title))
        for i, s in enumerate(snapshots)
    ]

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=_graph_layout(snapshots[0].positions, snapshots[0].title),
    )
    fig.update_layout(
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Step: ", font=dict(size=13)),
                pad=dict(t=40, b=10),
                len=0.9,
                x=0.05,
                steps=[
                    dict(
                        args=[[str(i)], {"frame": {"duration": 250, "redraw": True},
                                          "mode": "immediate", "transition": {"duration": 100}}],
                        label=str(i),
                        method="animate",
                    )
                    for i in range(len(snapshots))
                ],
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.15, x=0.0, xanchor="left",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 250, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 100}}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                         "mode": "immediate"}]),
                ],
            )
        ],
    )
    return fig


def make_static_figure(snap: Snapshot) -> go.Figure:
    """Single non-animated frame — what app.py renders on every step."""
    return go.Figure(data=_frame_traces(snap), layout=_graph_layout(snap.positions, snap.title))


def make_editor_figure(positions: np.ndarray, edges: list[Edge]) -> go.Figure:
    """Clickable canvas for placing / removing nodes while drawing your own
    graph. All current pairwise edges are shown faint (weight = distance);
    a dense invisible marker grid gives every point on the canvas a
    clickable target, since Streamlit's point-selection is most reliable
    on scatter traces.

    Pass to st.plotly_chart(fig, on_select="rerun", selection_mode="points")
    — app.py reads the clicked point's (x, y) back and decides whether it
    landed near an existing node (remove it) or empty space (add one).
    """
    traces: list[go.BaseTraceType] = []
    if edges:
        traces.append(_edge_trace(edges, positions, _BG_EDGE_COLOR, 1, name="Edges (= distance)"))
        traces.append(_edge_hover_trace(edges, positions))

    if len(positions):
        traces.append(_node_trace(positions, [_MST_COLOR] * len(positions), name="Nodes", size=24, clickable=True))

    # hoverinfo="none" (not "skip"!) — "skip" disables hit-testing entirely,
    # which would make every click on empty canvas silently do nothing, since
    # this is the trace Streamlit's on_select actually reads clicks from.
    grid_n = 24
    gx, gy = np.meshgrid(np.linspace(0, 10, grid_n), np.linspace(0, 10, grid_n))
    traces.append(
        go.Scatter(
            x=gx.ravel(), y=gy.ravel(), mode="markers",
            marker=dict(size=18, color="rgba(0,0,0,0)"),
            hoverinfo="none", showlegend=False, name="click-grid",
        )
    )

    layout = go.Layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text="Click to add a node · click a node to remove it", x=0.5,
                    xanchor="center", font=dict(size=15, color="#18181b")),
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.5, 10.5]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.5, 10.5], scaleanchor="x"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=560,
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        clickmode="event+select",
    )
    return go.Figure(data=traces, layout=layout)
