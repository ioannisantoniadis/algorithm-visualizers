"""Plotly figures for GMM EM snapshots."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .algorithm import Snapshot

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
_PALETTE = [
    "#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
    "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899",
]
_FONT_FAMILY = "Inter, -apple-system, Segoe UI, sans-serif"


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _palette_rgb(k: int) -> np.ndarray:
    rows = [_hex_to_rgb(_PALETTE[i % len(_PALETTE)]) for i in range(k)]
    return np.asarray(rows, dtype=np.float64)


def _soft_marker_colors(gamma: np.ndarray) -> list[str]:
    base = _palette_rgb(gamma.shape[1])
    mix = gamma @ base
    mix = np.clip(mix, 0, 255)
    return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in mix]


def _ellipse_xy(
    mean: np.ndarray,
    cov: np.ndarray,
    n_std: float = 2.0,
    n_pts: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=True)
    radii = n_std * np.sqrt(vals)
    circle = np.column_stack([np.cos(t) * radii[0], np.sin(t) * radii[1]])
    xy = circle @ vecs.T + mean
    return xy[:, 0], xy[:, 1]


def _range2d(xy: np.ndarray, means: np.ndarray, pad: float = 0.5) -> tuple[list[float], list[float]]:
    pts = [xy]
    if means.size:
        pts.append(means)
    p = np.vstack(pts)
    lo = p.min(axis=0)
    hi = p.max(axis=0)
    c = (lo + hi) / 2
    r = max(hi[0] - lo[0], hi[1] - lo[1], 0.5) / 2 + pad
    return [c[0] - r, c[0] + r], [c[1] - r, c[1] + r]


def _title(snap: Snapshot) -> str:
    if snap.substep == "init":
        return "Initialisation — first E-step (soft assignments from random μ, Σ)"
    if snap.substep == "m":
        return (
            f"Iteration {snap.iteration} · M-step — "
            "update π, μ, Σ (point colours still from the previous E-step)"
        )
    return (
        f"Iteration {snap.iteration} · E-step — "
        "rescale responsibilities γ₁…γₖ for each point"
    )


def _base_layout(title: str, height: int = 560) -> go.Layout:
    axis_style = dict(
        showgrid=True, gridcolor="#eef0f4", gridwidth=1,
        zeroline=False, showline=True, linecolor="#e4e4e7", linewidth=1,
    )
    return go.Layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14, color="#18181b")),
        xaxis=dict(**axis_style, title="x", scaleanchor="y", scaleratio=1),
        yaxis=dict(**axis_style, title="y"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            x=0.5,
            xanchor="center",
            font=dict(size=12, color="#3f3f46"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e4e4e7",
            borderwidth=1,
            itemsizing="constant",
        ),
        margin=dict(l=50, r=30, t=70, b=130),
        height=height,
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )


def make_figure(snap: Snapshot) -> go.Figure:
    X = snap.X
    colours = _soft_marker_colors(snap.responsibilities)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(size=9, color=colours, opacity=0.9, line=dict(width=0.4, color="white")),
            name="Data points (mixed RGB = E-step responsibilities)",
        ),
    )

    k = len(snap.weights)
    for j in range(k):
        colour = _PALETTE[j % len(_PALETTE)]
        xe, ye = _ellipse_xy(snap.means[j], snap.covs[j], n_std=2.0)
        fig.add_trace(
            go.Scatter(
                x=np.append(xe, xe[0]),
                y=np.append(ye, ye[0]),
                mode="lines",
                line=dict(color=colour, width=2),
                name=f"Gaussian {j + 1} — 2σ ellipse",
                opacity=0.92,
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[snap.means[j, 0]],
                y=[snap.means[j, 1]],
                mode="markers",
                marker=dict(size=11, color=colour, symbol="x", line=dict(width=1, color="white")),
                name=f"μ{j+1}",
                showlegend=False,
            ),
        )

    xr, yr = _range2d(X, snap.means)
    layout = _base_layout(_title(snap))
    layout.xaxis.range = xr
    layout.yaxis.range = yr
    fig.update_layout(layout)
    return fig
