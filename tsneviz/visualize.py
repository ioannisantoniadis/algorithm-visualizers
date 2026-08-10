"""Plotly split view: reference (latent or PCA₂) vs t-SNE embedding."""

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


def _snapshot_title(snap: Snapshot) -> str:
    ph = "Early exaggeration" if snap.phase == "early" else "Main optimisation"
    return f"t-SNE — {ph} · iteration {snap.iteration} · KL ≈ {snap.kl_divergence:.4f}"


def make_figure(
    snap: Snapshot,
    *,
    frame_index: int | None = None,
    frame_total: int | None = None,
) -> go.Figure:
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

    title = _snapshot_title(snap)
    if frame_index is not None and frame_total is not None:
        title += f"<br><sup>Frame {frame_index + 1}/{frame_total} · effective perplexity = {snap.perplexity:.1f}</sup>"

    fig.update_layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text=title, x=0.5, font=dict(size=14, color="#18181b")),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            x=0.5,
            xanchor="center",
            font=dict(size=12, color="#3f3f46"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e4e4e7",
            borderwidth=1,
        ),
        margin=dict(l=50, r=30, t=85, b=120),
        height=580,
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
