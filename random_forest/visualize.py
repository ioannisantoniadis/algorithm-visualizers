"""
visualize.py — Plotly figure builders

Five public functions:

  build_figure(snapshots, n_classes)
      Animated figure with one frame per tree added. Each frame redraws
      both the newest tree's own boundary and the cumulative ensemble
      boundary. Includes a Plotly slider; play/pause is handled by
      Streamlit-native buttons in app.py.

  make_static_figure(snap, n_classes)
      Non-animated two-panel figure for a single snapshot — "this tree"
      next to "the ensemble so far". Used by the step-through / auto-play
      loop so each rerun redraws cleanly without animation overhead.

  make_error_figure(snapshots, current_idx)
      Training (resubstitution) error vs. out-of-bag error, both against
      tree count — the bias-variance story in one chart: training error
      keeps falling toward 0 (optimistic, and eventually just memorising
      noise) while OOB error is the honest estimate and plateaus.

  make_importance_figure(snap)
      Horizontal bar chart of cumulative feature importance (x1 vs x2).

  make_preview_figure(X, y)
      Plain scatter of the raw dataset, shown before training starts.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .algorithm import Snapshot

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
_PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
            "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]
_CLASS_COLORS = _PALETTE  # up to len(_PALETTE) classes
_TRAIN_ERR_COLOR = "#6366f1"   # indigo — training (resubstitution) error
_OOB_ERR_COLOR = "#14b8a6"     # teal — out-of-bag (honest) error
_FONT_FAMILY = "Inter, -apple-system, Segoe UI, sans-serif"


def _colors(n_classes: int) -> list[str]:
    return _CLASS_COLORS[:n_classes]


def _discrete_colorscale(colors: list[str]) -> list[list]:
    """Colorscale with one flat-colored band per class (no gradient)."""
    n = len(colors)
    scale = []
    for i, c in enumerate(colors):
        scale.append([i / n, c])
        scale.append([(i + 1) / n, c])
    return scale


def _heatmap(gx: np.ndarray, gy: np.ndarray, z: np.ndarray, n_classes: int,
             confidence: np.ndarray | None = None) -> go.Heatmap:
    colors = _colors(n_classes)
    kwargs = dict(
        x=gx[0, :], y=gy[:, 0], z=z,
        zmin=-0.5, zmax=n_classes - 0.5,
        colorscale=_discrete_colorscale(colors),
        showscale=False,
        opacity=0.55,
        hoverinfo="skip",
        name="Predicted region",
    )
    if confidence is not None:
        kwargs["customdata"] = confidence
        kwargs["hovertemplate"] = "class=%{z}<br>vote share=%{customdata:.0%}<extra></extra>"
        del kwargs["hoverinfo"]
    return go.Heatmap(**kwargs)


def _point_traces(
    points: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    bootstrap_counts: np.ndarray | None = None,
) -> list[go.Scatter]:
    """One trace per class. If bootstrap_counts is given (per-tree panel),
    points used to train this tree get a thick black border and points
    that were out-of-bag for it get a thin dashed-looking (small, hollow)
    marker instead — the OOB set is exactly what the tree never saw."""
    colors = _colors(n_classes)
    traces: list[go.Scatter] = []

    for c in range(n_classes):
        mask = labels == c
        if bootstrap_counts is not None:
            in_bag = mask & (bootstrap_counts > 0)
            oob = mask & (bootstrap_counts == 0)
            traces.append(go.Scatter(
                x=points[in_bag, 0], y=points[in_bag, 1], mode="markers",
                marker=dict(color=colors[c], size=8, opacity=0.9,
                            line=dict(width=1.6, color="black")),
                name=f"Class {c} (in bootstrap)", showlegend=bool(in_bag.any()),
            ))
            traces.append(go.Scatter(
                x=points[oob, 0], y=points[oob, 1], mode="markers",
                marker=dict(color=colors[c], size=6, opacity=0.55,
                            line=dict(width=1, color="white")),
                name=f"Class {c} (OOB)", showlegend=bool(oob.any()),
            ))
        else:
            traces.append(go.Scatter(
                x=points[mask, 0], y=points[mask, 1], mode="markers",
                marker=dict(color=colors[c], size=7, opacity=0.85,
                            line=dict(width=0.6, color="white")),
                name=f"Class {c}", showlegend=bool(mask.any()),
            ))
    return traces


_AXIS_STYLE = dict(
    showgrid=True, gridcolor="#eef0f4", gridwidth=1,
    zeroline=False, showline=True, linecolor="#e4e4e7", linewidth=1,
)


def _panel_layout(title: str, X: np.ndarray) -> dict:
    pad = 0.8
    return dict(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text=title, font=dict(size=13, color="#18181b")),
        xaxis=dict(**_AXIS_STYLE, title="x₁",
                   range=[float(X[:, 0].min()) - pad, float(X[:, 0].max()) + pad]),
        yaxis=dict(**_AXIS_STYLE, title="x₂", scaleanchor="x",
                   range=[float(X[:, 1].min()) - pad, float(X[:, 1].max()) + pad]),
    )


# ---------------------------------------------------------------------------
# Two-panel figure: this tree | the ensemble so far
# ---------------------------------------------------------------------------

def make_static_figure(snap: Snapshot, n_classes: int, grid_x: np.ndarray, grid_y: np.ndarray) -> go.Figure:
    """Single-frame, two-panel figure: the newest tree's own boundary next
    to the ensemble's majority-vote boundary using every tree so far."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Tree {snap.tree_index} alone",
            f"Ensemble of {snap.tree_index} tree{'s' if snap.tree_index > 1 else ''} (majority vote)",
        ),
        horizontal_spacing=0.08,
    )

    fig.add_trace(_heatmap(grid_x, grid_y, snap.tree_grid, n_classes), row=1, col=1)
    for tr in _point_traces(snap.points, snap.labels, n_classes, snap.bootstrap_counts):
        fig.add_trace(tr, row=1, col=1)

    fig.add_trace(
        _heatmap(grid_x, grid_y, snap.ensemble_grid, n_classes, confidence=snap.ensemble_confidence_grid),
        row=1, col=2,
    )
    for tr in _point_traces(snap.points, snap.labels, n_classes):
        tr.showlegend = False
        fig.add_trace(tr, row=1, col=2)

    pad = 0.8
    x_range = [float(snap.points[:, 0].min()) - pad, float(snap.points[:, 0].max()) + pad]
    y_range = [float(snap.points[:, 1].min()) - pad, float(snap.points[:, 1].max()) + pad]
    fig.update_xaxes(**_AXIS_STYLE, title="x₁", range=x_range)
    fig.update_yaxes(**_AXIS_STYLE, title="x₂", range=y_range, scaleanchor="x", col=1)
    fig.update_yaxes(**_AXIS_STYLE, title="x₂", range=y_range, scaleanchor="x2", col=2)

    fig.update_layout(
        height=460,
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#e4e4e7",
                    borderwidth=1, font=dict(size=12)),
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=13, color="#18181b")
    return fig


def build_figure(snapshots: list[Snapshot], n_classes: int, grid_x: np.ndarray, grid_y: np.ndarray) -> go.Figure:
    """Animated figure with one frame per tree added, plus a slider."""
    if not snapshots:
        raise ValueError("snapshot list is empty")

    frames: list[go.Frame] = []
    frame_labels: list[str] = []
    base_fig = make_static_figure(snapshots[0], n_classes, grid_x, grid_y)

    for i, snap in enumerate(snapshots):
        panel_fig = make_static_figure(snap, n_classes, grid_x, grid_y)
        frames.append(go.Frame(data=panel_fig.data, name=str(i)))
        frame_labels.append(snap.title)

    base_fig.frames = frames
    base_fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="Trees: ", font=dict(size=13), xanchor="center"),
            pad=dict(t=50, b=10), len=0.9, x=0.05,
            steps=[
                dict(
                    args=[[str(i)], {"frame": {"duration": 250, "redraw": True},
                                      "mode": "immediate", "transition": {"duration": 100}}],
                    label=str(i + 1),
                    method="animate",
                )
                for i in range(len(snapshots))
            ],
        )],
    )
    return base_fig


# ---------------------------------------------------------------------------
# Error curves — training (resubstitution) vs. out-of-bag
# ---------------------------------------------------------------------------

def make_error_figure(snapshots: list[Snapshot], current_idx: int) -> go.Figure:
    """Training error (always optimistic — computed on the same data the
    trees were fit on) alongside out-of-bag error (an honest generalisation
    estimate that comes for free from bagging). Watching the gap between
    them widen as trees deepen or multiply *is* the bias-variance trade-off."""
    xs = list(range(1, len(snapshots) + 1))
    train_err = [1.0 - s.train_accuracy for s in snapshots]
    oob_err = [s.oob_error for s in snapshots]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=train_err, mode="lines+markers", name="Training error (resubstitution)",
        line=dict(color=_TRAIN_ERR_COLOR, width=2), marker=dict(size=4),
    ))
    oob_xs = [x for x, e in zip(xs, oob_err) if e is not None]
    oob_ys = [e for e in oob_err if e is not None]
    if oob_ys:
        fig.add_trace(go.Scatter(
            x=oob_xs, y=oob_ys, mode="lines+markers", name="Out-of-bag error",
            line=dict(color=_OOB_ERR_COLOR, width=2), marker=dict(size=4),
        ))

    cur = snapshots[current_idx]
    fig.add_trace(go.Scatter(
        x=[current_idx + 1], y=[1.0 - cur.train_accuracy], mode="markers",
        marker=dict(size=12, color=_TRAIN_ERR_COLOR, line=dict(width=2, color="#18181b")),
        showlegend=False,
    ))
    if cur.oob_error is not None:
        fig.add_trace(go.Scatter(
            x=[current_idx + 1], y=[cur.oob_error], mode="markers",
            marker=dict(size=12, color=_OOB_ERR_COLOR, line=dict(width=2, color="#18181b")),
            showlegend=False,
        ))

    fig.update_layout(
        height=260,
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        margin=dict(l=45, r=15, t=25, b=35),
        xaxis=dict(title="Trees in the forest", **_AXIS_STYLE),
        yaxis=dict(title="Error rate", **_AXIS_STYLE, rangemode="tozero"),
        legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center",
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#e4e4e7",
                    borderwidth=1, font=dict(size=12)),
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def make_importance_figure(snap: Snapshot, feature_names: tuple[str, str] = ("x₁", "x₂")) -> go.Figure:
    """Horizontal bar chart of cumulative, normalised feature importance."""
    values = snap.feature_importance
    fig = go.Figure(go.Bar(
        x=values, y=list(feature_names), orientation="h",
        marker=dict(color=_PALETTE[: len(values)]),
        text=[f"{v:.0%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        height=180,
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        margin=dict(l=30, r=30, t=10, b=30),
        xaxis=dict(title="Cumulative importance", range=[0, 1], **_AXIS_STYLE),
        yaxis=dict(title="", showgrid=False, zeroline=False),
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Raw data preview
# ---------------------------------------------------------------------------

def make_preview_figure(X: np.ndarray, y: np.ndarray) -> go.Figure:
    """Plain scatter of the training data before any tree is grown."""
    n_classes = int(y.max()) + 1
    traces = _point_traces(X, y, n_classes)
    fig = go.Figure(data=traces, layout=go.Layout(**_panel_layout("Training data", X)))
    fig.update_layout(height=460, plot_bgcolor="#fbfbfd", paper_bgcolor="rgba(0,0,0,0)",
                       margin=dict(l=40, r=20, t=50, b=40))
    return fig
