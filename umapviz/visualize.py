"""Plotly figures for UMAP snapshots (split view: ℝᴰ surrogate vs latent / embedding)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .algorithm import Snapshot

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
_PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
            "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]
_NOISE_COLOUR = "#a1a1aa"
_FONT_FAMILY = "Inter, -apple-system, Segoe UI, sans-serif"


def _colours(labels: np.ndarray) -> list[str]:
    out: list[str] = []
    for lb in labels:
        if lb < 0:
            out.append(_NOISE_COLOUR)
        else:
            out.append(_PALETTE[int(lb) % len(_PALETTE)])
    return out


def _range2d(xy: np.ndarray, pad: float = 0.35) -> tuple[list[float], list[float]]:
    if xy.size == 0:
        return [-4, 4], [-4, 4]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    c = (lo + hi) / 2
    r = max(hi[0] - lo[0], hi[1] - lo[1], 0.2) / 2 + pad
    return [c[0] - r, c[0] + r], [c[1] - r, c[1] + r]


def _latent_xy_plot(latent: np.ndarray) -> np.ndarray:
    """First two coords of latent (ℝ³ truth is shown as x–y projection in 2-D panels)."""
    return latent[:, :2].astype(np.float64, copy=False)


def _subplot_titles(snap: Snapshot) -> tuple[str, str]:
    d_lat = None
    if snap.latent_xy is not None and np.any(np.isfinite(snap.latent_xy[:, 0])):
        d_lat = int(snap.latent_xy.shape[1])
    if snap.phase in ("knn", "fuzzy"):
        if snap.phase == "knn":
            left = "A — What you see: PCA₂ of ℝᴰ + k-NN edges"
        else:
            left = "A — What you see: PCA₂ of ℝᴰ + fuzzy edges"
        if snap.latent_xy is None:
            right = "B — No latent; copy of A for scale"
        elif np.any(~np.isfinite(snap.latent_xy[:, 0])):
            right = "B — Ground truth (finite rows only)"
        elif d_lat == 3:
            right = "B — Ground truth: ℝ³ latent as (x,y)"
        else:
            right = "B — Ground truth: latent before ℝᴰ lift"
    else:
        if snap.latent_xy is None:
            left = "A — Reference: PCA₂ (no synthetic latent)"
            right = "B — UMAP output 𝑌 (this is the result)"
        elif np.any(~np.isfinite(snap.latent_xy[:, 0])):
            left = "A — Ground truth latent (partial)"
            right = "B — UMAP output 𝑌 (this is the result)"
        elif d_lat == 3:
            left = "A — Ground truth: ℝ³ as (x,y)"
            right = "B — UMAP output 𝑌 (2-D from 3-D clusters)"
        else:
            left = "A — Ground truth: latent shape"
            right = "B — UMAP output 𝑌 (compare to A)"
    return left, right


def _scatter(
    fig: go.Figure,
    row: int,
    col: int,
    xy: np.ndarray,
    colours: list[str],
    *,
    size: float,
    name: str,
    opacity: float = 0.92,
    showlegend: bool = True,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers",
            marker=dict(
                size=size,
                color=colours,
                opacity=opacity,
                line=dict(width=0.4, color="white"),
            ),
            name=name,
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def _masked_latent_scatter(
    fig: go.Figure,
    row: int,
    col: int,
    latent: np.ndarray,
    colours: list[str],
    *,
    size: float,
    name: str,
) -> None:
    mask = np.isfinite(latent[:, 0])
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    xy = _latent_xy_plot(latent[idx, :])
    cols = [colours[int(i)] for i in idx]
    _scatter(fig, row, col, xy, cols, size=size, name=name)


def _knn_edge_lines(p2: np.ndarray, knn_indices: np.ndarray, n: int) -> tuple[list, list]:
    ki = knn_indices
    xs: list = []
    ys: list = []
    for i in range(n):
        for t in range(ki.shape[1]):
            j = int(ki[i, t])
            if j < 0 or j == i:
                continue
            xs.extend([p2[i, 0], p2[j, 0], None])
            ys.extend([p2[i, 1], p2[j, 1], None])
    return xs, ys


def _fuzzy_edge_lines(p2: np.ndarray, edges: np.ndarray) -> tuple[list, list]:
    xs: list = []
    ys: list = []
    step = max(1, len(edges) // 3000)
    for row in edges[::step]:
        i, j = int(row[0]), int(row[1])
        xs.extend([p2[i, 0], p2[j, 0], None])
        ys.extend([p2[i, 1], p2[j, 1], None])
    return xs, ys


def make_figure(
    snap: Snapshot,
    *,
    frame_index: int | None = None,
    frame_total: int | None = None,
) -> go.Figure:
    labels = snap.labels
    colours = _colours(labels)
    n = snap.X.shape[0]

    tl, tr = _subplot_titles(snap)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(tl, tr),
        horizontal_spacing=0.09,
    )

    if snap.phase in ("knn", "fuzzy"):
        p2 = snap.pca2
        _scatter(fig, 1, 1, p2, colours, size=7, name="PCA₂ points", showlegend=True)
        if snap.phase == "knn" and snap.knn_indices is not None:
            xs, ys = _knn_edge_lines(p2, snap.knn_indices, n)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="rgba(99,102,241,0.35)", width=1),
                    name="k-NN edges",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        elif snap.phase == "fuzzy" and snap.fuzzy_edges is not None:
            xs, ys = _fuzzy_edge_lines(p2, snap.fuzzy_edges)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="rgba(139,92,246,0.45)", width=0.9),
                    name="Fuzzy edges",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        if snap.latent_xy is None:
            _scatter(
                fig, 1, 2, p2, colours, size=6, name="PCA₂ (no latent)", opacity=0.55,
            )
        else:
            _masked_latent_scatter(
                fig, 1, 2, snap.latent_xy, colours, size=7, name="Latent",
            )
        xr1, yr1 = _range2d(p2)
        if snap.latent_xy is None:
            xr2, yr2 = xr1, yr1
        else:
            mask = np.isfinite(snap.latent_xy[:, 0])
            if not np.any(mask):
                xr2, yr2 = xr1, yr1
            else:
                xy_l = _latent_xy_plot(snap.latent_xy[mask, :])
                xr2, yr2 = _range2d(xy_l)
    else:
        if snap.latent_xy is None:
            _scatter(fig, 1, 1, snap.pca2, colours, size=7, name="PCA ref")
        else:
            _masked_latent_scatter(
                fig, 1, 1, snap.latent_xy, colours, size=8, name="Latent",
            )
        y = snap.Y
        _scatter(fig, 1, 2, y, colours, size=9, name="Emb. 𝑌", showlegend=True)

        if snap.latent_xy is None:
            xr1, yr1 = _range2d(snap.pca2)
        else:
            mask = np.isfinite(snap.latent_xy[:, 0])
            xy_m = (
                _latent_xy_plot(snap.latent_xy[mask, :])
                if np.any(mask)
                else snap.pca2
            )
            xr1, yr1 = _range2d(xy_m)
        xr2, yr2 = _range2d(y)

    _axis_style = dict(
        showgrid=True, gridcolor="#eef0f4", gridwidth=1,
        zeroline=False, showline=True, linecolor="#e4e4e7", linewidth=1,
    )
    fig.update_xaxes(title_text="x", range=xr1, row=1, col=1, **_axis_style)
    fig.update_yaxes(title_text="y", range=yr1, row=1, col=1, **_axis_style)
    fig.update_xaxes(title_text="x", range=xr2, row=1, col=2, **_axis_style)
    fig.update_yaxes(title_text="y", range=yr2, row=1, col=2, **_axis_style)

    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_xaxes(scaleanchor="y2", scaleratio=1, row=1, col=2)

    title_main = snap.title
    if frame_index is not None and frame_total is not None:
        title_main = (
            f"{snap.title}<br><sup>Frame {frame_index + 1} / {frame_total} · "
            "use ◀ ▶ to walk the pipeline in order</sup>"
        )

    fig.update_layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text=title_main, x=0.5, font=dict(size=14, color="#18181b")),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.24, x=0.5, xanchor="center",
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#e4e4e7", borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(l=50, r=30, t=95, b=100),
        height=580,
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    for annotation in fig.layout.annotations:
        annotation.font = dict(family=_FONT_FAMILY, size=13, color="#3f3f46")
    return fig
