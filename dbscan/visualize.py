"""
visualize.py — Plotly figure builders for DBSCAN

Three public functions:

  build_figure(snapshots)
      Animated figure — one Plotly frame per snapshot.
      Uses a FIXED trace layout (same number of traces in the same slots
      across every frame) so Plotly's animation engine transitions cleanly.

  make_static_figure(snap)
      Non-animated single-snapshot figure for the manual step-through mode.
      Trace count can vary here since there's no inter-frame animation.

  make_eps_circle(cx, cy, r)
      Parametric circle helper shared by both builders.

Fixed trace order (build_figure only)
--------------------------------------
  0            — unvisited points
  1            — noise points (✕)
  2 … 2+K-1   — core points, one trace per cluster
  2+K … 2+2K-1— border points, one trace per cluster
  2+2K         — ε-neighbourhood highlight (orange halo)
  2+2K+1       — ε-circle outline (dashed ring)
  2+2K+2       — seed point (★)

Empty x/y arrays are used for slots that have no data in a given frame.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .algorithm import BORDER, CORE, NOISE, UNVISITED, Snapshot

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
_PALETTE           = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
                      "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]
_NOISE_COLOUR      = "#a1a1aa"
_UNVISITED_COLOUR  = "#d4d4d8"
_NEIGHBOUR_COLOUR  = "#f59e0b"
_EPS_LINE_COLOUR   = "#d97706"
_FONT_FAMILY       = "Inter, -apple-system, Segoe UI, sans-serif"


def _cluster_colours(n: int) -> list[str]:
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


def make_eps_circle(cx: float, cy: float, r: float, n_pts: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) arrays for a closed circle trace centred at (cx, cy)."""
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=True)
    return cx + r * np.cos(theta), cy + r * np.sin(theta)


# ---------------------------------------------------------------------------
# Fixed-trace builder — used by build_figure (animated)
# ---------------------------------------------------------------------------

def _build_traces_fixed(snap: Snapshot, max_clusters: int) -> list[go.BaseTraceType]:
    """Return exactly 2 + 2*max_clusters + 3 traces in a fixed order.

    Slots that have no data for this snapshot receive empty x/y arrays
    so Plotly's animation engine can diff cleanly between frames.
    """
    X      = snap.points
    labels = snap.labels
    roles  = snap.roles
    cols   = _cluster_colours(max_clusters)
    empty  = ([], [])

    traces: list[go.BaseTraceType] = []

    # --- 0: unvisited -------------------------------------------------------
    m = labels == -2
    traces.append(go.Scatter(
        x=X[m, 0] if m.any() else [], y=X[m, 1] if m.any() else [],
        mode="markers",
        marker=dict(color=_UNVISITED_COLOUR, size=7, opacity=0.6,
                    line=dict(width=0.5, color="white")),
        name="Unvisited", showlegend=bool(m.any()),
    ))

    # --- 1: noise -----------------------------------------------------------
    m = labels == -1
    traces.append(go.Scatter(
        x=X[m, 0] if m.any() else [], y=X[m, 1] if m.any() else [],
        mode="markers",
        marker=dict(symbol="x", color=_NOISE_COLOUR, size=9, opacity=0.85,
                    line=dict(width=1.5, color="#888888")),
        name="Noise", showlegend=bool(m.any()),
    ))

    # --- 2 … 2+K-1: core points per cluster --------------------------------
    for c in range(max_clusters):
        m = (labels == c) & (roles == CORE)
        traces.append(go.Scatter(
            x=X[m, 0] if m.any() else [], y=X[m, 1] if m.any() else [],
            mode="markers",
            marker=dict(color=cols[c], size=9, opacity=0.85,
                        line=dict(width=0.5, color="white")),
            name=f"Cluster {c} — core", showlegend=bool(m.any()),
        ))

    # --- 2+K … 2+2K-1: border points per cluster ---------------------------
    for c in range(max_clusters):
        m = (labels == c) & (roles == BORDER)
        traces.append(go.Scatter(
            x=X[m, 0] if m.any() else [], y=X[m, 1] if m.any() else [],
            mode="markers",
            marker=dict(color="white", size=8, opacity=0.9,
                        line=dict(width=2, color=cols[c])),
            name=f"Cluster {c} — border", showlegend=bool(m.any()),
        ))

    # --- 2+2K: neighbourhood highlight -------------------------------------
    if snap.phase == "examine" and snap.neighbors is not None:
        nbr = snap.neighbors
        x_n, y_n = X[nbr, 0], X[nbr, 1]
        sl = True
        lbl = f"Neighbours ({len(nbr)})"
    else:
        x_n, y_n = [], []
        sl = False
        lbl = "Neighbours"
    traces.append(go.Scatter(
        x=x_n, y=y_n, mode="markers",
        marker=dict(color=_NEIGHBOUR_COLOUR, size=11, opacity=0.45,
                    line=dict(width=1.5, color=_NEIGHBOUR_COLOUR)),
        name=lbl, showlegend=sl,
    ))

    # --- 2+2K+1: ε-circle outline ------------------------------------------
    if snap.phase == "examine" and snap.active_point is not None:
        cx, cy = float(X[snap.active_point, 0]), float(X[snap.active_point, 1])
        xc, yc = make_eps_circle(cx, cy, snap.eps)
        sl = True
    else:
        xc, yc = [], []
        sl = False
    traces.append(go.Scatter(
        x=xc, y=yc, mode="lines",
        line=dict(color=_EPS_LINE_COLOUR, width=2, dash="dot"),
        fill="none",
        name="ε-circle", showlegend=sl, hoverinfo="skip",
    ))

    # --- 2+2K+2: seed point (★) -------------------------------------------
    if snap.active_point is not None:
        ap   = snap.active_point
        x_s  = [float(X[ap, 0])]
        y_s  = [float(X[ap, 1])]
        col  = cols[snap.cluster_id % len(cols)]
        sl   = True
    else:
        x_s, y_s, col, sl = [], [], "grey", False
    traces.append(go.Scatter(
        x=x_s, y=y_s, mode="markers",
        marker=dict(symbol="star", color=col, size=20,
                    line=dict(width=2, color="black")),
        name="Seed (★)", showlegend=sl,
    ))

    return traces


# ---------------------------------------------------------------------------
# Variable-trace builder — used by make_static_figure
# ---------------------------------------------------------------------------

def _build_traces_variable(snap: Snapshot, max_clusters: int) -> list[go.BaseTraceType]:
    """Variable-length trace list for static (non-animated) figures.

    Omits empty clusters entirely so the legend stays clean.
    """
    X      = snap.points
    labels = snap.labels
    roles  = snap.roles
    cols   = _cluster_colours(max_clusters)
    traces: list[go.BaseTraceType] = []

    # ε-circle outline (examine phase only)
    if snap.phase == "examine" and snap.active_point is not None:
        cx, cy = float(X[snap.active_point, 0]), float(X[snap.active_point, 1])
        xc, yc = make_eps_circle(cx, cy, snap.eps)
        traces.append(go.Scatter(
            x=xc, y=yc, mode="lines", fill="toself",
            fillcolor="rgba(224,123,0,0.12)",
            line=dict(color=_EPS_LINE_COLOUR, width=2, dash="dot"),
            name="ε-neighbourhood", showlegend=True, hoverinfo="skip",
        ))

    # Unvisited
    m = labels == -2
    if m.any():
        traces.append(go.Scatter(
            x=X[m, 0], y=X[m, 1], mode="markers",
            marker=dict(color=_UNVISITED_COLOUR, size=7, opacity=0.6,
                        line=dict(width=0.5, color="white")),
            name="Unvisited", showlegend=True,
        ))

    # Noise
    m = labels == -1
    if m.any():
        traces.append(go.Scatter(
            x=X[m, 0], y=X[m, 1], mode="markers",
            marker=dict(symbol="x", color=_NOISE_COLOUR, size=9, opacity=0.85,
                        line=dict(width=1.5, color="#888888")),
            name="Noise", showlegend=True,
        ))

    # Clusters
    for c in range(max_clusters):
        m_core   = (labels == c) & (roles == CORE)
        m_border = (labels == c) & (roles == BORDER)
        if m_core.any():
            traces.append(go.Scatter(
                x=X[m_core, 0], y=X[m_core, 1], mode="markers",
                marker=dict(color=cols[c], size=9, opacity=0.85,
                            line=dict(width=0.5, color="white")),
                name=f"Cluster {c} — core", showlegend=True,
            ))
        if m_border.any():
            traces.append(go.Scatter(
                x=X[m_border, 0], y=X[m_border, 1], mode="markers",
                marker=dict(color="white", size=8, opacity=0.9,
                            line=dict(width=2, color=cols[c])),
                name=f"Cluster {c} — border", showlegend=True,
            ))

    # Neighbourhood highlight
    if snap.phase == "examine" and snap.neighbors is not None:
        nbr = snap.neighbors
        traces.append(go.Scatter(
            x=X[nbr, 0], y=X[nbr, 1], mode="markers",
            marker=dict(color=_NEIGHBOUR_COLOUR, size=11, opacity=0.45,
                        line=dict(width=1.5, color=_NEIGHBOUR_COLOUR)),
            name=f"Neighbours ({len(nbr)})", showlegend=True,
        ))

    # Seed star
    if snap.active_point is not None:
        ap = snap.active_point
        col = cols[snap.cluster_id % len(cols)]
        traces.append(go.Scatter(
            x=[float(X[ap, 0])], y=[float(X[ap, 1])], mode="markers",
            marker=dict(symbol="star", color=col, size=20,
                        line=dict(width=2, color="black")),
            name="Seed (★)", showlegend=True,
        ))

    return traces


# ---------------------------------------------------------------------------
# Shared layout helper
# ---------------------------------------------------------------------------

def _base_layout(title: str, X: np.ndarray, height: int = 560) -> go.Layout:
    pad = 0.3
    axis_style = dict(
        showgrid=True, gridcolor="#eef0f4", gridwidth=1,
        zeroline=False, showline=True, linecolor="#e4e4e7", linewidth=1,
    )
    return go.Layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text=title, x=0.5, xanchor="center",
                    font=dict(size=16, color="#18181b")),
        xaxis=dict(**axis_style, title="x₁",
                   range=[float(X[:, 0].min()) - pad, float(X[:, 0].max()) + pad]),
        yaxis=dict(**axis_style, title="x₂",
                   range=[float(X[:, 1].min()) - pad, float(X[:, 1].max()) + pad],
                   scaleanchor="x"),
        legend=dict(orientation="v", x=1.02, y=1,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e4e4e7", borderwidth=1,
                    font=dict(size=12)),
        margin=dict(l=40, r=180, t=60, b=40),
        height=height,
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_figure(snapshots: list[Snapshot]) -> go.Figure:
    """Animated Plotly figure — one frame per snapshot.

    All frames share the same fixed trace layout so Plotly can animate
    smoothly without remapping data to the wrong traces.
    """
    if not snapshots:
        raise ValueError("snapshot list is empty")

    X = snapshots[0].points
    max_clusters = max((s.n_clusters for s in snapshots), default=1)
    max_clusters = max(max_clusters, 1)

    frames: list[go.Frame] = []
    frame_labels: list[str] = []

    for i, snap in enumerate(snapshots):
        traces = _build_traces_fixed(snap, max_clusters)
        frames.append(go.Frame(
            data=traces,
            name=str(i),
            layout=go.Layout(title_text=snap.title),
        ))
        frame_labels.append(snap.title)

    layout = _base_layout(snapshots[0].title, X)

    return go.Figure(data=frames[0].data, frames=frames, layout=layout)


def make_static_figure(snap: Snapshot, max_clusters: int = 8) -> go.Figure:
    """Non-animated figure for a single snapshot (manual step-through)."""
    traces = _build_traces_variable(snap, max_clusters)
    layout = _base_layout(snap.title, snap.points)
    return go.Figure(data=traces, layout=layout)
