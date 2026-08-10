"""
visualize.py — Plotly figure builders

Perceptron tab
  make_perceptron_figure(snap, true_normal, show_true_boundary)
      Scatter of the two classes, the current decision boundary, the weight
      vector drawn as an arrow normal to that boundary, and (when the
      snapshot is an "update") a gold ring around the point that triggered it.

Gradient-descent tab
  make_gd_contour_figure(snap, w1s, w2s, Z)
      The logistic-loss surface as a filled contour in weight space, with
      all three optimisers' full trajectories overlaid so they are directly
      comparable on one surface.
  make_gd_side_by_side_figure(snap, w1s, w2s, Z)
      The same surface repeated in three panels, one optimiser highlighted
      per panel — easier to read when the paths overlap heavily.
  make_gd_loss_curve_figure(snapshots, upto_step)
      Loss vs. step for all three optimisers, the classic convergence-speed
      comparison.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .algorithm import GDSnapshot, PerceptronSnapshot

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
_PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
            "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]
_FONT_FAMILY = "Inter, -apple-system, Segoe UI, sans-serif"

# Class colours (Perceptron tab)
_CLASS_COLOUR = {1.0: _PALETTE[0], -1.0: _PALETTE[3]}

# Optimiser colours (Gradient Descent tab) — consistent everywhere
_OPT_COLOUR = {"vanilla": _PALETTE[0], "momentum": _PALETTE[2], "adam": _PALETTE[1]}
_OPT_LABEL = {"vanilla": "Vanilla GD", "momentum": "Momentum", "adam": "Adam"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _axis_range_2d(*parts: np.ndarray, pad: float = 0.6) -> tuple[list[float], list[float]]:
    pts = np.vstack([p for p in parts if p is not None and np.size(p)])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 0.5)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    r = span / 2 + pad
    return [cx - r, cx + r], [cy - r, cy + r]


_AXIS_STYLE = dict(
    showgrid=True, gridcolor="#eef0f4", gridwidth=1,
    zeroline=False, showline=True, linecolor="#e4e4e7", linewidth=1,
)


def _base_layout_2d(title: str, xrange: list[float], yrange: list[float]) -> go.Layout:
    return go.Layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15, color="#18181b")),
        xaxis=dict(**_AXIS_STYLE, title="x₁", range=xrange),
        yaxis=dict(**_AXIS_STYLE, title="x₂", range=yrange,
                   scaleanchor="x", scaleratio=1),
        legend=dict(orientation="v", x=1.02, y=1, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e4e4e7", borderwidth=1, font=dict(size=12)),
        margin=dict(l=50, r=170, t=60, b=50),
        height=540,
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )


def _weight_space_layout(title: str, w1_range, w2_range) -> go.Layout:
    axis_style = dict(_AXIS_STYLE, zeroline=True, zerolinecolor="#a1a1aa")
    return go.Layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15, color="#18181b")),
        xaxis=dict(**axis_style, title="w₁", range=list(w1_range)),
        yaxis=dict(**axis_style, title="w₂", range=list(w2_range),
                   scaleanchor="x", scaleratio=1),
        legend=dict(orientation="v", x=1.02, y=1, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e4e4e7", borderwidth=1, font=dict(size=12)),
        margin=dict(l=50, r=170, t=60, b=50),
        height=540,
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )


def _boundary_geometry(w: np.ndarray, span: float):
    """Given w = [bias, w1, w2], return (foot, wvec, p1, p2) describing the
    decision boundary as a segment [p1, p2] of length ~2*span, plus the
    point on that line closest to the origin ("foot") and the weight vector
    itself — which is, by construction, the boundary's normal.

    Returns None if w1 == w2 == 0 (no boundary defined yet).
    """
    b, w1, w2 = w
    wvec = np.array([w1, w2])
    norm_sq = float(wvec @ wvec)
    if norm_sq < 1e-12:
        return None
    foot = -b * wvec / norm_sq
    direction = np.array([-w2, w1])
    direction = direction / np.linalg.norm(direction)
    p1 = foot - span * direction
    p2 = foot + span * direction
    return foot, wvec, p1, p2


# ---------------------------------------------------------------------------
# Perceptron tab
# ---------------------------------------------------------------------------

def make_perceptron_figure(
    snap: PerceptronSnapshot,
    true_normal: np.ndarray | None = None,
    show_true_boundary: bool = False,
) -> go.Figure:
    X, y, w = snap.points, snap.labels, snap.weights
    xr, yr = _axis_range_2d(X)
    span = 0.75 * max(xr[1] - xr[0], yr[1] - yr[0])

    fig = go.Figure()

    for label, name in [(1.0, "Class +1"), (-1.0, "Class −1")]:
        mask = y == label
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0], y=X[mask, 1], mode="markers",
                marker=dict(color=_CLASS_COLOUR[label], size=9, opacity=0.85,
                            line=dict(width=0.5, color="white")),
                name=name,
            )
        )

    # Ring the point that triggered this update, if any
    if snap.phase == "update" and snap.point_idx >= 0:
        px, py = X[snap.point_idx]
        fig.add_trace(
            go.Scatter(
                x=[px], y=[py], mode="markers",
                marker=dict(size=18, color="rgba(0,0,0,0)",
                            line=dict(width=3, color="#ffb300")),
                name="Misclassified → updated",
            )
        )

    # Ground-truth generating boundary, dashed, for comparison
    if show_true_boundary and true_normal is not None:
        direction = np.array([-true_normal[1], true_normal[0]])
        p1, p2 = -span * direction, span * direction
        fig.add_trace(
            go.Scatter(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode="lines",
                line=dict(color="#9e9e9e", width=2, dash="dot"),
                name="True boundary (generating)",
            )
        )

    geom = _boundary_geometry(w, span)
    if geom is not None:
        foot, wvec, p1, p2 = geom
        fig.add_trace(
            go.Scatter(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode="lines",
                line=dict(color="#212121", width=3),
                name="Decision boundary",
            )
        )
        wnorm = wvec / np.linalg.norm(wvec)
        arrow_len = 0.4 * span
        tip = foot + wnorm * arrow_len
        fig.add_annotation(
            x=float(tip[0]), y=float(tip[1]), ax=float(foot[0]), ay=float(foot[1]),
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=3,
            arrowcolor="#1565c0",
        )
        # Invisible trace purely so the arrow gets a legend entry
        fig.add_trace(
            go.Scatter(
                x=[foot[0]], y=[foot[1]], mode="markers",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                name="w (normal to boundary)",
            )
        )

    fig.update_layout(_base_layout_2d(snap.title, xr, yr))
    return fig


# ---------------------------------------------------------------------------
# Gradient descent tab
# ---------------------------------------------------------------------------

def make_gd_contour_figure(
    snap: GDSnapshot,
    w1s: np.ndarray,
    w2s: np.ndarray,
    Z: np.ndarray,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=w1s, y=w2s, z=Z,
            colorscale="Viridis",
            contours=dict(showlabels=False),
            opacity=0.92,
            colorbar=dict(title="loss", len=0.75),
            name="Loss surface",
            hoverinfo="skip",
        )
    )

    for key, path in [
        ("vanilla", snap.path_vanilla),
        ("momentum", snap.path_momentum),
        ("adam", snap.path_adam),
    ]:
        colour = _OPT_COLOUR[key]
        fig.add_trace(
            go.Scatter(
                x=path[:, 0], y=path[:, 1], mode="lines+markers",
                line=dict(color=colour, width=2),
                marker=dict(size=4, color=colour),
                name=_OPT_LABEL[key],
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[path[-1, 0]], y=[path[-1, 1]], mode="markers",
                marker=dict(size=15, color=colour, symbol="diamond",
                            line=dict(width=2, color="white")),
                showlegend=False,
            )
        )

    fig.update_layout(
        _weight_space_layout(snap.title, [w1s.min(), w1s.max()], [w2s.min(), w2s.max()])
    )
    return fig


def make_gd_side_by_side_figure(
    snap: GDSnapshot,
    w1s: np.ndarray,
    w2s: np.ndarray,
    Z: np.ndarray,
) -> go.Figure:
    specs = [
        ("vanilla", snap.path_vanilla),
        ("momentum", snap.path_momentum),
        ("adam", snap.path_adam),
    ]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[_OPT_LABEL[k] for k, _ in specs],
        horizontal_spacing=0.07,
    )

    for i, (key, path) in enumerate(specs, start=1):
        colour = _OPT_COLOUR[key]
        fig.add_trace(
            go.Contour(
                x=w1s, y=w2s, z=Z, colorscale="Viridis",
                contours=dict(showlabels=False), opacity=0.9,
                showscale=(i == 3), colorbar=dict(title="loss", len=0.75),
                hoverinfo="skip",
            ),
            row=1, col=i,
        )
        fig.add_trace(
            go.Scatter(
                x=path[:, 0], y=path[:, 1], mode="lines+markers",
                line=dict(color=colour, width=2), marker=dict(size=4, color=colour),
                showlegend=False,
            ),
            row=1, col=i,
        )
        fig.add_trace(
            go.Scatter(
                x=[path[-1, 0]], y=[path[-1, 1]], mode="markers",
                marker=dict(size=13, color=colour, symbol="diamond",
                            line=dict(width=2, color="white")),
                showlegend=False,
            ),
            row=1, col=i,
        )
        fig.update_xaxes(title_text="w₁", range=[w1s.min(), w1s.max()], row=1, col=i,
                          **_AXIS_STYLE)

    fig.update_yaxes(title_text="w₂", range=[w2s.min(), w2s.max()], row=1, col=1,
                      scaleanchor="x", **_AXIS_STYLE)
    fig.update_layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        height=400, margin=dict(l=40, r=40, t=50, b=40),
        plot_bgcolor="#fbfbfd", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def make_gd_loss_curve_figure(snapshots: list[GDSnapshot], upto_step: int) -> go.Figure:
    window = snapshots[: upto_step + 1]
    steps = [s.step for s in window]

    fig = go.Figure()
    for key, attr in [("vanilla", "loss_vanilla"), ("momentum", "loss_momentum"),
                       ("adam", "loss_adam")]:
        ys = [getattr(s, attr) for s in window]
        fig.add_trace(
            go.Scatter(
                x=steps, y=ys, mode="lines",
                line=dict(color=_OPT_COLOUR[key], width=2.5),
                name=_OPT_LABEL[key],
            )
        )

    fig.update_layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text="Loss vs. step", x=0.5, xanchor="center", y=0.97, yanchor="top",
                   font=dict(size=14, color="#18181b")),
        xaxis=dict(**_AXIS_STYLE, title="Step"),
        yaxis=dict(**_AXIS_STYLE, title="Logistic loss"),
        height=340,
        margin=dict(l=55, r=30, t=70, b=45),
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.16, font=dict(size=12)),
    )
    return fig
