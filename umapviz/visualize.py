"""Plotly figures for UMAP snapshots (split view: ℝᴰ surrogate vs latent / embedding)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.theme import FONT_FAMILY as _FONT_FAMILY
from common.theme import PALETTE as _PALETTE
from common.theme import apply_theme
from .algorithm import Snapshot

_NOISE_COLOUR = "#a1a1aa"


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
    # Kept intentionally short — these sit above narrow subplot columns
    # (this figure renders inside a nested column split) and Plotly does
    # not wrap or truncate `subplot_titles` text, so long strings here
    # overlap each other. The full explanation lives in the info box the
    # page renders below the chart (`_frame_panels_explainer`), not here.
    d_lat = None
    if snap.latent_xy is not None and np.any(np.isfinite(snap.latent_xy[:, 0])):
        d_lat = int(snap.latent_xy.shape[1])
    if snap.phase in ("knn", "fuzzy"):
        left = "A — PCA₂ + k-NN edges" if snap.phase == "knn" else "A — PCA₂ + fuzzy edges"
        if snap.latent_xy is None:
            right = "B — no latent (scale ref)"
        elif np.any(~np.isfinite(snap.latent_xy[:, 0])):
            right = "B — ground truth (partial)"
        elif d_lat == 3:
            right = "B — ground truth (ℝ³→x,y)"
        else:
            right = "B — latent (pre-lift)"
    else:
        if snap.latent_xy is None:
            left = "A — PCA₂ (reference)"
            right = "B — UMAP output 𝑌"
        elif np.any(~np.isfinite(snap.latent_xy[:, 0])):
            left = "A — ground truth (partial)"
            right = "B — UMAP output 𝑌"
        elif d_lat == 3:
            left = "A — ground truth (ℝ³→x,y)"
            right = "B — UMAP output 𝑌"
        else:
            left = "A — ground truth (latent)"
            right = "B — UMAP output 𝑌"
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
    """`frame_index`/`frame_total` are accepted for call-site compatibility
    with apps/umap.py but no longer drive a chart title (dropped below) —
    `snap.title` (e.g. "Stage 1/3 — kNN graph...") duplicates the pipeline
    rail card above the chart and the page's own progress bar text."""
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

    apply_theme(fig, None, height=580, showlegend=True)
    # Re-apply the shared font to the "A"/"B" panel titles (make_subplots
    # annotations) — apply_theme only sets the top-level layout font.
    for annotation in fig.layout.annotations:
        annotation.font = dict(family=_FONT_FAMILY, size=11.5, color="#3f3f46")
    return fig
