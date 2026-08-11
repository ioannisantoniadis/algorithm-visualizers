"""Plotly split view: reference (latent or PCA₂) vs t-SNE embedding."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def _range2d(xy: np.ndarray, pad: float = 0.4) -> tuple[list[float], list[float]]:
    if xy.size == 0:
        return [-4, 4], [-4, 4]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    c = (lo + hi) / 2
    r = max(hi[0] - lo[0], hi[1] - lo[1], 0.2) / 2 + pad
    return [c[0] - r, c[0] + r], [c[1] - r, c[1] + r]


def _latent_2d(latent: np.ndarray) -> np.ndarray:
    return latent[:, :2].astype(np.float64, copy=False)


def _masked_latent_scatter(
    fig: go.Figure,
    row: int,
    col: int,
    latent: np.ndarray,
    colour_list: list[str],
    *,
    size: float,
    name: str,
) -> None:
    mask = np.isfinite(latent[:, 0])
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return
    xy = _latent_2d(latent[idx, :])
    cols = [colour_list[int(i)] for i in idx]
    fig.add_trace(
        go.Scatter(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers",
            marker=dict(size=size, color=cols, opacity=0.9, line=dict(width=0.4, color="white")),
            name=name,
        ),
        row=row,
        col=col,
    )


def make_figure(
    snap: Snapshot,
    *,
    frame_index: int | None = None,
    frame_total: int | None = None,
) -> go.Figure:
    """`frame_index`/`frame_total` are accepted for call-site compatibility
    with apps/tsne.py but no longer drive a chart title (dropped below) —
    the page's own progress bar and metrics row (phase, iteration, KL,
    perplexity) already show every value that title used to repeat."""
    colours = _colours(snap.labels)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "A — Ground truth (latent) or PCA₂ of ℝᴰ",
            "B — t-SNE embedding 𝑌 (Student-t kernel)",
        ),
        horizontal_spacing=0.09,
    )

    if snap.latent_xy is None:
        p2 = snap.pca2
        fig.add_trace(
            go.Scatter(
                x=p2[:, 0],
                y=p2[:, 1],
                mode="markers",
                marker=dict(size=7, color=colours, opacity=0.88, line=dict(width=0.4, color="white")),
                name="PCA₂",
            ),
            row=1,
            col=1,
        )
        xr1, yr1 = _range2d(p2)
    else:
        _masked_latent_scatter(fig, 1, 1, snap.latent_xy, colours, size=7, name="Latent")
        mask = np.isfinite(snap.latent_xy[:, 0])
        xr1, yr1 = _range2d(_latent_2d(snap.latent_xy[mask, :]) if np.any(mask) else snap.pca2)

    y = snap.Y
    fig.add_trace(
        go.Scatter(
            x=y[:, 0],
            y=y[:, 1],
            mode="markers",
            marker=dict(size=9, color=colours, opacity=0.92, line=dict(width=0.45, color="white")),
            name="𝑌 (t-SNE)",
        ),
        row=1,
        col=2,
    )
    xr2, yr2 = _range2d(y)

    axis_style = dict(
        showgrid=True, gridcolor="#eef0f4", gridwidth=1,
        zeroline=False, showline=True, linecolor="#e4e4e7", linewidth=1,
    )
    fig.update_xaxes(title_text="x", range=xr1, row=1, col=1, **axis_style)
    fig.update_yaxes(title_text="y", range=yr1, row=1, col=1, **axis_style)
    fig.update_xaxes(title_text="x", range=xr2, row=1, col=2, **axis_style)
    fig.update_yaxes(title_text="y", range=yr2, row=1, col=2, **axis_style)
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_xaxes(scaleanchor="y2", scaleratio=1, row=1, col=2)

    apply_theme(fig, None, height=580, showlegend=True)
    return fig
