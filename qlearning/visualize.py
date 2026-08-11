"""
visualize.py — Plotly figure builders for the TD-learning grid visualiser

Cells are drawn as stacked heatmap layers (one solid colour per cell kind,
or a continuous max-Q heatmap), with policy arrows, cell-kind letters, the
agent marker and the current greedy-policy route drawn as overlays on top
— the same "layered heatmap + scatter overlay" recipe used throughout the
sibling *-viz repos.

Public functions
-----------------
make_static_figure(snap, background)   single frame — what app.py renders
                                        on every step of manual/auto playback
build_figure(snapshots, background)    animated figure with a Plotly slider
                                        (self-contained alternative to the
                                        Streamlit-driven playback in app.py)
make_policy_figure(grid, q_table, title)  standalone grid for the
                                        Q-learning vs SARSA comparison panel
make_reward_figure(snapshots, ...)     return-per-episode + running average
make_compare_reward_figure(...)        two algorithms' reward curves overlaid
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from common.theme import base_layout

from .algorithm import ACTION_ARROWS, Snapshot, episode_summaries, greedy_action, rollout_policy_path
from .data import Cell, GridWorld

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
_PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
            "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]

# Cell-kind colours are domain-semantic (danger = red, goal = green, cliff =
# near-black, etc.) rather than an arbitrary categorical series, so they stay
# hand-picked for readability instead of being pulled straight off _PALETTE.
_KIND_COLOR = {
    "empty": "#f4f6f8",
    "start": "#cfe3ff",
    "iron": "#9e9e9e",
    "diamond": "#4fc3f7",
    "lava": "#e63946",
    "pit": "#8d1a1a",
    "cliff": "#1f2430",
    "goal": "#2ea043",
}
_KIND_ORDER = ["empty", "start", "iron", "diamond", "lava", "pit", "cliff", "goal"]
_LABEL = {"start": "S", "iron": "I", "diamond": "D", "lava": "L", "pit": "P", "cliff": "C", "goal": "G"}
_LABEL_COLOR = {
    "start": "#1f2430", "iron": "#ffffff", "diamond": "#08313a", "lava": "#ffffff",
    "pit": "#ffffff", "cliff": "#ffffff", "goal": "#ffffff",
}

# Agent/route/move colours harmonised with _PALETTE (orange + amber for the
# "orange circle" / "gold dotted line" the README describes; move/slip stay
# green/red so the README's own colour language stays accurate).
_AGENT_COLOR = "#fb923c"
_PATH_COLOR = "#f59e0b"
_SLIP_COLOR = "#e63946"
_MOVE_COLOR = "#2ea043"

MAX_ANIMATED_FRAMES = 200  # cap for build_figure — beyond this we subsample


def _solid_layer(mask: np.ndarray, color: str) -> go.Heatmap:
    z = np.where(mask, 1.0, np.nan)
    return go.Heatmap(
        z=z, zmin=0, zmax=1, colorscale=[[0, color], [1, color]],
        showscale=False, hoverinfo="skip", xgap=1, ygap=1,
    )


def _kind_layers(grid: GridWorld) -> list[go.Heatmap]:
    layers = []
    for kind in _KIND_ORDER:
        mask = grid.kind == kind
        if mask.any():
            layers.append(_solid_layer(mask, _KIND_COLOR[kind]))
    return layers


def _qvalue_layer(q_table: np.ndarray) -> go.Heatmap:
    max_q = q_table.max(axis=2)
    bound = max(float(np.abs(max_q).max()), 1e-6)
    return go.Heatmap(
        z=max_q, zmin=-bound, zmax=bound, colorscale="RdYlGn",
        showscale=True, colorbar=dict(title="max Q(s,·)", len=0.45, y=0.8, thickness=14),
        hovertemplate="row %{y}, col %{x}<br>max Q = %{z:.2f}<extra></extra>",
        xgap=1, ygap=1,
    )


def _cell_labels(grid: GridWorld) -> go.Scatter:
    xs, ys, texts, colors = [], [], [], []
    for r in range(grid.rows):
        for c in range(grid.cols):
            kind = grid.kind[r, c]
            if kind in _LABEL:
                xs.append(c)
                ys.append(r)
                texts.append(_LABEL[kind])
                colors.append(_LABEL_COLOR[kind])
    return go.Scatter(
        x=xs, y=ys, mode="text", text=texts,
        textfont=dict(size=15, color=colors, family="monospace"),
        hoverinfo="skip", showlegend=False, name="labels",
    )


def _policy_arrows(grid: GridWorld, q_table: np.ndarray) -> go.Scatter:
    xs, ys, texts = [], [], []
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.terminal[r, c]:
                continue
            xs.append(c)
            ys.append(r)
            texts.append(ACTION_ARROWS[greedy_action(q_table, (r, c))])
    return go.Scatter(
        x=xs, y=ys, mode="text", text=texts,
        textfont=dict(size=22, color="#1f2430"),
        hoverinfo="skip", showlegend=False, name="policy",
    )


def _policy_path_trace(path: list[Cell]) -> go.Scatter | None:
    if len(path) < 2:
        return None
    xs = [p[1] for p in path]
    ys = [p[0] for p in path]
    return go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color=_PATH_COLOR, width=4, dash="dot"),
        name="Current greedy route", hoverinfo="skip",
    )


def _agent_marker(state: Cell) -> go.Scatter:
    r, c = state
    return go.Scatter(
        x=[c], y=[r], mode="markers",
        marker=dict(symbol="circle", size=20, color=_AGENT_COLOR, line=dict(width=2, color="white")),
        name="Agent", hoverinfo="skip",
    )


def _action_line(state: Cell, next_state: Cell, slipped: bool) -> go.Scatter:
    color = _SLIP_COLOR if slipped else _MOVE_COLOR
    return go.Scatter(
        x=[state[1], next_state[1]], y=[state[0], next_state[0]], mode="lines",
        line=dict(color=color, width=3, dash="dot" if slipped else "solid"),
        name="Slipped" if slipped else "Move", hoverinfo="skip",
    )


def _grid_layout(rows: int, cols: int, title: str | None) -> go.Layout:
    return base_layout(
        title,
        height=460,
        showlegend=False,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.5, cols - 0.5]),
        yaxis=dict(
            showgrid=False, zeroline=False, visible=False,
            range=[rows - 0.5, -0.5], scaleanchor="x",
        ),
    )


def _frame_traces(
    grid: GridWorld, q_table: np.ndarray, agent_at: Cell,
    background: str, action_edge: tuple[Cell, Cell, bool] | None,
) -> list[go.BaseTraceType]:
    traces: list[go.BaseTraceType] = []
    if background == "qvalue":
        traces.append(_qvalue_layer(q_table))
    else:
        traces.extend(_kind_layers(grid))
    traces.append(_cell_labels(grid))
    traces.append(_policy_arrows(grid, q_table))

    path_trace = _policy_path_trace(rollout_policy_path(grid, q_table))
    if path_trace is not None:
        traces.append(path_trace)

    if action_edge is not None:
        state, next_state, slipped = action_edge
        traces.append(_action_line(state, next_state, slipped))

    traces.append(_agent_marker(agent_at))
    return traces


def make_static_figure(snap: Snapshot, background: str = "kind") -> go.Figure:
    """Single non-animated frame — what app.py renders on every step."""
    grid = snap.grid
    agent_at = snap.next_state if snap.next_state is not None else grid.start
    action_edge = None
    if snap.state is not None and snap.next_state is not None:
        action_edge = (snap.state, snap.next_state, snap.slipped)

    traces = _frame_traces(grid, snap.q_table, agent_at, background, action_edge)
    # per-frame title dropped — already shown in the page's progress bar
    return go.Figure(data=traces, layout=_grid_layout(grid.rows, grid.cols, None))


def build_figure(snapshots: list[Snapshot], background: str = "kind") -> go.Figure:
    """Animated figure with a Plotly slider — subsamples to at most
    MAX_ANIMATED_FRAMES frames since training can span thousands of steps."""
    if not snapshots:
        raise ValueError("snapshot list is empty")

    if len(snapshots) > MAX_ANIMATED_FRAMES:
        idx = np.linspace(0, len(snapshots) - 1, MAX_ANIMATED_FRAMES).astype(int)
        idx = sorted(set(idx.tolist()) | {len(snapshots) - 1})
        shown = [snapshots[i] for i in idx]
    else:
        shown = snapshots

    grid = shown[0].grid
    frames = []
    for i, s in enumerate(shown):
        agent_at = s.next_state if s.next_state is not None else grid.start
        action_edge = (s.state, s.next_state, s.slipped) if s.state and s.next_state else None
        frames.append(
            go.Frame(
                data=_frame_traces(grid, s.q_table, agent_at, background, action_edge),
                name=str(i), layout=go.Layout(title_text=s.title),
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=_grid_layout(grid.rows, grid.cols, None))
    fig.update_layout(
        sliders=[dict(
            active=0, currentvalue=dict(prefix="Frame: ", font=dict(size=13)),
            pad=dict(t=40, b=10), len=0.9, x=0.05,
            steps=[
                dict(args=[[str(i)], {"frame": {"duration": 200, "redraw": True},
                                       "mode": "immediate", "transition": {"duration": 80}}],
                     label=str(i), method="animate")
                for i in range(len(shown))
            ],
        )],
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.15, x=0.0, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 200, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 80}}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
            ],
        )],
    )
    return fig


def make_policy_figure(grid: GridWorld, q_table: np.ndarray, title: str, background: str = "kind") -> go.Figure:
    """Standalone grid (no agent marker) — used for the final side-by-side
    Q-learning vs SARSA policy comparison."""
    traces = _frame_traces(grid, q_table, grid.start, background, action_edge=None)
    return go.Figure(data=traces, layout=_grid_layout(grid.rows, grid.cols, title))


def _rolling_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < window:
        return np.array([]), np.array([])
    kernel = np.ones(window) / window
    avg = np.convolve(values, kernel, mode="valid")
    x = np.arange(window, len(values) + 1)
    return x, avg


def make_reward_figure(snapshots: list[Snapshot], name: str = "", color: str = _PALETTE[0]) -> go.Figure:
    """Return-per-episode line plus a running average, so noisy episode-to-
    episode reward is easy to read as an overall learning trend."""
    eps, returns, _lengths, _epsilons = episode_summaries(snapshots)
    fig = go.Figure()
    if len(eps) == 0:
        fig.layout = base_layout(None, height=280, showlegend=False, margin=dict(l=40, r=10, t=30, b=35))
        return fig

    window = max(2, min(10, len(eps) // 3 or 1))
    run_x, run_y = _rolling_average(returns, window)

    fig.add_trace(go.Scatter(
        x=eps, y=returns, mode="lines", name=f"{name} return" if name else "Return",
        line=dict(color=color, width=1), opacity=0.45,
    ))
    if len(run_x):
        fig.add_trace(go.Scatter(
            x=run_x, y=run_y, mode="lines",
            name=f"{name} running avg (w={window})" if name else f"Running avg (w={window})",
            line=dict(color=color, width=2.5),
        ))
    fig.layout = base_layout(
        None,
        height=280,
        margin=dict(l=45, r=10, t=30),
        xaxis=dict(title="Episode"),
        yaxis=dict(title="Total reward"),
    )
    return fig


def make_compare_reward_figure(
    snapshots_a: list[Snapshot], name_a: str,
    snapshots_b: list[Snapshot], name_b: str,
) -> go.Figure:
    """Both algorithms' return-per-episode running averages on one chart."""
    fig = go.Figure()
    for snaps, name, color in ((snapshots_a, name_a, _PALETTE[0]), (snapshots_b, name_b, _PALETTE[3])):
        eps, returns, _lengths, _epsilons = episode_summaries(snaps)
        if len(eps) == 0:
            continue
        window = max(2, min(10, len(eps) // 3 or 1))
        run_x, run_y = _rolling_average(returns, window)
        fig.add_trace(go.Scatter(
            x=eps, y=returns, mode="lines", line=dict(color=color, width=1),
            opacity=0.25, showlegend=False,
        ))
        if len(run_x):
            fig.add_trace(go.Scatter(
                x=run_x, y=run_y, mode="lines", name=f"{name} (running avg)",
                line=dict(color=color, width=2.5),
            ))
    fig.layout = base_layout(
        None,
        height=300,
        margin=dict(l=45, r=10, t=30),
        xaxis=dict(title="Episode"),
        yaxis=dict(title="Total reward"),
    )
    return fig
