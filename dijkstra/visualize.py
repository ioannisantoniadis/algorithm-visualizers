"""
visualize.py — Plotly figure builders for the grid pathfinder

Cells are drawn as stacked heatmap layers (terrain → walls → visited →
frontier), each layer using NaN for "not this category" so the layers
beneath show through untouched. Start, goal, the currently-expanding
node, and the reconstructed path are drawn as scatter overlays on top.

Public functions
-----------------
build_figure(snapshots)              animated figure, one Plotly frame per
                                      Snapshot with its own slider (kept as
                                      a self-contained alternative to the
                                      Streamlit-driven playback in app.py)
make_static_figure(snap)             single non-animated frame — what the
                                      app actually renders each step
make_editor_figure(grid_cost, start, goal)  clickable grid for drawing walls
                                             and repositioning start/goal
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from common.theme import base_layout

from .algorithm import Cell, Snapshot

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
_PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
            "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]

_WALL_COLOR = "#27272a"      # neutral dark — walls stay off-palette on purpose
_CLOSED_COLOR = "#c7d2fe"    # light tint of the primary indigo — finalised nodes
_OPEN_COLOR = "#f59e0b"      # palette amber — open/frontier nodes
_CURRENT_COLOR = "#f43f5e"   # palette rose — node being expanded right now
_PATH_COLOR = "#14b8a6"      # palette teal — the final shortest path
_PATH_BORDER_COLOR = "#0f766e"
_START_COLOR = "#22c55e"     # semantic green — kept off-palette for clarity
_GOAL_COLOR = "#8b5cf6"      # palette violet — distinct from the rose "current" marker

_TERRAIN_SCALE = "YlOrBr"


def _solid_layer(mask: np.ndarray, color: str) -> go.Heatmap:
    """A heatmap layer that paints `color` where mask is True and is
    transparent (letting layers beneath show through) everywhere else."""
    z = np.where(mask, 1.0, np.nan)
    return go.Heatmap(
        z=z,
        zmin=0,
        zmax=1,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        hoverinfo="skip",
        xgap=1,
        ygap=1,
    )


def _terrain_layer(grid_cost: np.ndarray) -> go.Heatmap:
    z = np.where(np.isfinite(grid_cost), grid_cost, np.nan)
    finite = grid_cost[np.isfinite(grid_cost)]
    zmax = float(finite.max()) if finite.size else 1.0
    # Uniform-cost grids (open field, maze, random obstacles) have no real
    # terrain-cost range to show. Two things follow from that:
    #  - a colorbar spanning a fake, padded range around a single constant
    #    value is just noise, so only show it once there's an actual
    #    gradient to explain (weighted terrain);
    #  - feeding a degenerate zmin==zmax range into the YlOrBr colorscale
    #    renders a fully-saturated orange rather than the pale end of the
    #    scale, which visually swallows the amber "open set" overlay. Use a
    #    flat, pale neutral fill instead so open/frontier cells stay legible.
    varies = zmax > 1.0
    colorscale = _TERRAIN_SCALE if varies else [[0, "#faf5ec"], [1, "#faf5ec"]]
    return go.Heatmap(
        z=z,
        zmin=1.0,
        zmax=max(zmax, 1.0),
        colorscale=colorscale,
        showscale=varies,
        colorbar=dict(title="Terrain<br>cost", len=0.45, y=0.8, thickness=14) if varies else None,
        hovertemplate="row %{y}, col %{x}<br>cost %{z:.0f}<extra></extra>",
        xgap=1,
        ygap=1,
    )


def _wall_layer(grid_cost: np.ndarray) -> go.Heatmap:
    return _solid_layer(~np.isfinite(grid_cost), _WALL_COLOR)


def _base_layers(grid_cost: np.ndarray) -> list[go.Heatmap]:
    return [_terrain_layer(grid_cost), _wall_layer(grid_cost)]


def _grid_layout(rows: int, cols: int, title: str | None) -> go.Layout:
    return base_layout(
        title,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.5, cols - 0.5]),
        yaxis=dict(
            showgrid=False, zeroline=False, visible=False,
            range=[rows - 0.5, -0.5], scaleanchor="x",
        ),
    )


def _current_marker(current: Cell | None) -> go.Scatter | None:
    if current is None:
        return None
    r, c = current
    return go.Scatter(
        x=[c], y=[r], mode="markers",
        marker=dict(symbol="square", size=16, color=_CURRENT_COLOR,
                    line=dict(width=2, color="white")),
        name="Currently expanding", hoverinfo="skip",
    )


def _path_trace(path: list[Cell] | None) -> go.Scatter | None:
    if not path:
        return None
    xs = [p[1] for p in path]
    ys = [p[0] for p in path]
    return go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        line=dict(color=_PATH_COLOR, width=5),
        marker=dict(size=7, color=_PATH_COLOR, line=dict(width=1, color=_PATH_BORDER_COLOR)),
        name="Shortest path", hoverinfo="skip",
    )


def _tentative_trace(path_so_far: list[Cell]) -> go.Scatter | None:
    if len(path_so_far) < 2:
        return None
    xs = [p[1] for p in path_so_far]
    ys = [p[0] for p in path_so_far]
    return go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=_CURRENT_COLOR, width=2, dash="dot"),
        name="Path to current node", hoverinfo="skip",
    )


def _endpoint_markers(start: Cell, goal: Cell) -> list[go.Scatter]:
    return [
        go.Scatter(
            x=[start[1]], y=[start[0]], mode="markers+text",
            marker=dict(symbol="circle", size=18, color=_START_COLOR,
                        line=dict(width=2, color="white")),
            text=["S"], textfont=dict(color="white", size=11),
            name="Start", hoverinfo="skip",
        ),
        go.Scatter(
            x=[goal[1]], y=[goal[0]], mode="markers+text",
            marker=dict(symbol="diamond", size=18, color=_GOAL_COLOR,
                        line=dict(width=2, color="white")),
            text=["G"], textfont=dict(color="white", size=11),
            name="Goal", hoverinfo="skip",
        ),
    ]


def _frame_traces(snap: Snapshot) -> list[go.BaseTraceType]:
    traces: list[go.BaseTraceType] = list(_base_layers(snap.grid_cost))
    traces.append(_solid_layer(snap.closed, _CLOSED_COLOR))
    traces.append(_solid_layer(snap.open_mask, _OPEN_COLOR))

    tentative = _tentative_trace(snap.path_so_far)
    if tentative is not None:
        traces.append(tentative)

    path_trace = _path_trace(snap.path)
    if path_trace is not None:
        traces.append(path_trace)

    current = _current_marker(snap.current)
    if current is not None:
        traces.append(current)

    traces.extend(_endpoint_markers(snap.start, snap.goal))
    return traces


def build_figure(snapshots: list[Snapshot]) -> go.Figure:
    """Animated figure with one Plotly frame per Snapshot and a slider."""
    if not snapshots:
        raise ValueError("snapshot list is empty")

    rows, cols = snapshots[0].grid_cost.shape
    frames = [
        go.Frame(data=_frame_traces(s), name=str(i), layout=go.Layout(title_text=s.title))
        for i, s in enumerate(snapshots)
    ]

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=_grid_layout(rows, cols, None),  # per-frame title shown via go.Frame layout instead
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
    rows, cols = snap.grid_cost.shape
    # per-frame title dropped — already shown in the page's progress bar
    return go.Figure(data=_frame_traces(snap), layout=_grid_layout(rows, cols, None))


def make_editor_figure(grid_cost: np.ndarray, start: Cell, goal: Cell) -> go.Figure:
    """Clickable grid for drawing walls and repositioning start/goal.

    Pass to st.plotly_chart(fig, on_select="rerun", selection_mode="points")
    — app.py reads the clicked point's (x, y) back as a (col, row) cell.
    An invisible dense marker grid gives every cell a clickable target,
    since Streamlit's point-selection is most reliable on scatter traces.
    """
    rows, cols = grid_cost.shape
    traces: list[go.BaseTraceType] = list(_base_layers(grid_cost))
    traces.extend(_endpoint_markers(start, goal))

    gy, gx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    traces.append(
        go.Scatter(
            x=gx.ravel(), y=gy.ravel(), mode="markers",
            marker=dict(size=22, color="rgba(0,0,0,0)"),
            # NOTE: hoverinfo="skip" here would make this trace fully
            # unpickable — Plotly's click-to-select machinery relies on the
            # same hit-testing path as hover, and heatmap traces beneath
            # don't support point selection at all, so "skip" silently
            # made wall-drawing dead (every click produced an empty
            # event.selection.points). "none" keeps it pickable while
            # still showing no tooltip text.
            hoverinfo="none", showlegend=False, name="click-grid",
        )
    )

    layout = _grid_layout(rows, cols, "Click cells to draw walls or move start / goal")
    layout.clickmode = "event+select"
    return go.Figure(data=traces, layout=layout)
