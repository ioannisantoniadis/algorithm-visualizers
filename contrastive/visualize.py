"""
visualize.py — Plotly figure builders

Five public functions, each consuming a Snapshot (or the full snapshot
history) produced by algorithm.fit():

  make_input_figure(snap)
      Raw input space: base points plus this step's two augmented views
      (small circles / diamonds), with a dashed line connecting each
      positive pair — the two views the loss is trying to pull together.

  make_embedding_figure(snap)
      Embedding space: every point in the dataset, coloured by its
      ground-truth cluster (never seen by the algorithm — for reading the
      plot only), plotted on the unit circle every embedding is
      normalised onto. This step's batch views and their connecting line
      are overlaid on top.

  make_before_after_figure(snap_init, snap_final)
      Two embedding-space panels side by side: random encoder vs trained
      encoder, same colour-by-true-cluster scheme.

  make_similarity_heatmap(snap)
      (2B, 2B) cosine-similarity matrix for the current batch. The
      positive-pair diagonal (offset by B) should light up bright once
      training helps; everything else should fade.

  make_loss_figure(history, current_step)
      NT-Xent loss, and mean positive/negative similarity, vs training
      step, with a marker at the step currently being viewed.
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
_POS_LINE = "#14b8a6"   # positive pair connector — teal, "pull together"
_NEG_LINE = "#f43f5e"   # negative-similarity curve — rose, "push apart"
_GREY = "#a1a1aa"
_FONT_FAMILY = "Inter, -apple-system, Segoe UI, sans-serif"

_AXIS_STYLE = dict(
    showgrid=True, gridcolor="#eef0f4", gridwidth=1,
    zeroline=False, showline=True, linecolor="#e4e4e7", linewidth=1,
)
_LEGEND_RIGHT = dict(
    orientation="v", x=1.02, y=1, xanchor="left", yanchor="top",
    bgcolor="rgba(255,255,255,0.9)", bordercolor="#e4e4e7", borderwidth=1,
    font=dict(size=11),
)


def _cluster_colours(n_clusters: int) -> list[str]:
    return [_PALETTE[i % len(_PALETTE)] for i in range(n_clusters)]


def _connector_traces(xa, ya, xb, yb, colour=_POS_LINE, name="Positive pair") -> go.Scatter:
    """One Scatter trace drawing a dashed segment between each (xa,ya)-(xb,yb)
    pair, using None separators so it's a single trace, not B traces."""
    xs, ys = [], []
    for x0, y0, x1, y1 in zip(xa, ya, xb, yb):
        xs.extend([x0, x1, None])
        ys.extend([y0, y1, None])
    return go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=colour, width=1.2, dash="dot"),
        opacity=0.6, name=name, showlegend=True, hoverinfo="skip",
    )


# ---------------------------------------------------------------------------
# Input space — augmentation view
# ---------------------------------------------------------------------------

def make_input_figure(snap: Snapshot, n_clusters: int) -> go.Figure:
    colours = _cluster_colours(n_clusters)
    labels = snap.labels
    traces: list[go.BaseTraceType] = []

    # All dataset points, faint, as spatial context
    traces.append(
        go.Scatter(
            x=snap.points[:, 0], y=snap.points[:, 1], mode="markers",
            marker=dict(color=_GREY, size=5, opacity=0.35),
            name="Dataset", showlegend=False, hoverinfo="skip",
        )
    )

    if snap.batch_idx is not None:
        batch_labels = labels[snap.batch_idx]
        for c in range(n_clusters):
            mask = batch_labels == c
            if not mask.any():
                continue
            traces.append(
                go.Scatter(
                    x=snap.view_a[mask, 0], y=snap.view_a[mask, 1], mode="markers",
                    marker=dict(symbol="circle", color=colours[c], size=10,
                                line=dict(width=1, color="white")),
                    name=f"View A (cluster {c})", legendgroup=f"c{c}",
                )
            )
            traces.append(
                go.Scatter(
                    x=snap.view_b[mask, 0], y=snap.view_b[mask, 1], mode="markers",
                    marker=dict(symbol="diamond", color=colours[c], size=10,
                                line=dict(width=1, color="white")),
                    name=f"View B (cluster {c})", legendgroup=f"c{c}",
                )
            )
        traces.append(
            _connector_traces(
                snap.view_a[:, 0], snap.view_a[:, 1],
                snap.view_b[:, 0], snap.view_b[:, 1],
            )
        )

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
            title=dict(text="Input space — this step's augmented views", x=0.5,
                       xanchor="center", font=dict(size=15, color="#18181b")),
            xaxis=dict(**_AXIS_STYLE, title="x₁"),
            yaxis=dict(**_AXIS_STYLE, title="x₂", scaleanchor="x"),
            legend=dict(**_LEGEND_RIGHT),
            margin=dict(l=40, r=170, t=50, b=40),
            height=480,
            plot_bgcolor="#fbfbfd",
            paper_bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Embedding space — unit circle
# ---------------------------------------------------------------------------

def _unit_circle_shape() -> dict:
    return dict(
        type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1,
        line=dict(color="#d4d4d8", width=1, dash="dot"), layer="below",
    )


def make_embedding_figure(snap: Snapshot, n_clusters: int) -> go.Figure:
    colours = _cluster_colours(n_clusters)
    labels = snap.labels
    E = snap.embed_full
    traces: list[go.BaseTraceType] = []

    for c in range(n_clusters):
        mask = labels == c
        if not mask.any():
            continue
        traces.append(
            go.Scatter(
                x=E[mask, 0], y=E[mask, 1], mode="markers",
                marker=dict(color=colours[c], size=9, opacity=0.85,
                            line=dict(width=1, color="white")),
                name=f"Cluster {c}",
            )
        )

    if snap.batch_idx is not None and snap.embed_a is not None:
        traces.append(
            _connector_traces(
                snap.embed_a[:, 0], snap.embed_a[:, 1],
                snap.embed_b[:, 0], snap.embed_b[:, 1],
                name="Positive pair (this batch)",
            )
        )

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
            title=dict(text="Normalised embedding — unit circle", x=0.5,
                       xanchor="center", font=dict(size=15, color="#18181b")),
            xaxis=dict(**_AXIS_STYLE, title="e₁", range=[-1.25, 1.25]),
            yaxis=dict(**_AXIS_STYLE, title="e₂", range=[-1.25, 1.25], scaleanchor="x"),
            shapes=[_unit_circle_shape()],
            legend=dict(**_LEGEND_RIGHT),
            margin=dict(l=40, r=170, t=50, b=40),
            height=480,
            plot_bgcolor="#fbfbfd",
            paper_bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def make_before_after_figure(snap_init: Snapshot, snap_final: Snapshot, n_clusters: int) -> go.Figure:
    colours = _cluster_colours(n_clusters)
    labels = snap_init.labels

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Before training (random encoder)", "After training"),
        horizontal_spacing=0.1,
    )

    for panel, snap in enumerate([snap_init, snap_final], start=1):
        E = snap.embed_full
        for c in range(n_clusters):
            mask = labels == c
            if not mask.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=E[mask, 0], y=E[mask, 1], mode="markers",
                    marker=dict(color=colours[c], size=8, opacity=0.85,
                                line=dict(width=1, color="white")),
                    name=f"Cluster {c}", legendgroup=f"c{c}", showlegend=(panel == 1),
                ),
                row=1, col=panel,
            )
        fig.add_shape(_unit_circle_shape(), row=1, col=panel)

    fig.update_xaxes(**_AXIS_STYLE, range=[-1.25, 1.25])
    fig.update_yaxes(**_AXIS_STYLE, range=[-1.25, 1.25], scaleanchor="x", scaleratio=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        height=460,
        margin=dict(l=30, r=170, t=55, b=30),
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(**_LEGEND_RIGHT),
    )
    fig.update_annotations(font=dict(size=14, color="#18181b"))
    return fig


# ---------------------------------------------------------------------------
# Similarity heat-map
# ---------------------------------------------------------------------------

def make_similarity_heatmap(snap: Snapshot) -> go.Figure:
    if snap.sim_matrix is None:
        return go.Figure()
    S = snap.sim_matrix
    n = S.shape[0]
    half = n // 2

    fig = go.Figure(
        data=go.Heatmap(
            z=S, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
            colorbar=dict(title="cos sim", len=0.8, x=1.15),
        )
    )
    # Mark the positive-pair off-diagonal (row i <-> row i+half) with markers
    idx = np.arange(n)
    pos_idx = np.concatenate([np.arange(half, n), np.arange(0, half)])
    fig.add_trace(
        go.Scatter(
            x=pos_idx, y=idx, mode="markers",
            marker=dict(symbol="square-open", size=9, color="#18181b", line=dict(width=1.5)),
            name="Positive pair", showlegend=True,
        )
    )
    fig.update_layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text="Batch cosine-similarity matrix (2B × 2B)", x=0.5,
                   xanchor="center", font=dict(size=15, color="#18181b")),
        xaxis=dict(title="view index", showgrid=False),
        yaxis=dict(title="view index", showgrid=False, autorange="reversed"),
        legend=dict(
            orientation="h", x=0, y=1.1, xanchor="left", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.9)", font=dict(size=11),
        ),
        height=460,
        margin=dict(l=50, r=90, t=75, b=40),
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def make_loss_figure(history: list[tuple[int, float, float, float]], current_step: int) -> go.Figure:
    """history rows: (step, loss, pos_sim, neg_sim)."""
    steps = [h[0] for h in history]
    losses = [h[1] for h in history]
    pos_sims = [h[2] for h in history]
    neg_sims = [h[3] for h in history]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=steps, y=losses, mode="lines", line=dict(color="#3f3f46", width=2),
                   name="NT-Xent loss"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=steps, y=pos_sims, mode="lines", line=dict(color=_POS_LINE, width=1.6, dash="dash"),
                   name="Mean positive-pair cos sim"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=steps, y=neg_sims, mode="lines", line=dict(color=_NEG_LINE, width=1.6, dash="dash"),
                   name="Mean negative-pair cos sim"),
        secondary_y=True,
    )

    cur = next((h for h in history if h[0] == current_step), history[-1] if history else None)
    if cur is not None:
        fig.add_trace(
            go.Scatter(x=[cur[0]], y=[cur[1]], mode="markers",
                       marker=dict(size=11, color="#f43f5e", line=dict(width=2, color="white")),
                       name="Current step", showlegend=False),
            secondary_y=False,
        )

    fig.update_yaxes(title_text="NT-Xent loss", secondary_y=False, showgrid=True,
                      gridcolor="#eef0f4", zeroline=False)
    fig.update_yaxes(title_text="cosine similarity", secondary_y=True,
                      range=[-1.05, 1.05], showgrid=False, zeroline=False)
    fig.update_xaxes(title_text="Training step", showgrid=True, gridcolor="#eef0f4", zeroline=False)
    fig.update_layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text="Training progress", x=0.5, xanchor="center", font=dict(size=15, color="#18181b")),
        height=340,
        margin=dict(l=55, r=170, t=50, b=45),
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(**_LEGEND_RIGHT),
    )
    return fig
