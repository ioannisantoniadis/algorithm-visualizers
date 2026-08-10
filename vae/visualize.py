"""
visualize.py — Plotly figure builders

Four public functions, each consuming a single Snapshot produced by
algorithm.fit() (plus, for the interactive decode demo, algorithm.decode()):

  make_data_space_figure(snap, title, show_manifold, highlight)
      Original points (coloured by label) with their reconstructions
      overlaid as faint "x" markers connected by a thin line — the gap
      between a point and its "x" *is* the reconstruction error. Optionally
      overlays the decoder's current manifold (decoded_grid) as a faint
      background scatter, and/or a single highlighted decoded point.

  make_latent_space_figure(snap, title, show_prior, highlight)
      The 2-D latent code for every point. For a VAE, each point is drawn
      with error bars sized by its own std = exp(0.5*logvar) — literally
      the spread the reparameterisation trick samples from — plus two
      dashed circles marking the 1-sigma / 2-sigma contours of the
      standard normal prior N(0, I) the KL term regularises toward.

  make_loss_figure(history_ae, history_vae)
      Reconstruction/KL/total loss curves for both runs side by side, so
      the trade-off the KL term introduces (worse reconstruction, better
      regularised latent space) is visible directly in the numbers.

  make_comparison_scatter(snap_ae, snap_vae)
      Small side-by-side latent-space panel with a shared axis range,
      built once per frame for the top-level AE-vs-VAE comparison.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .algorithm import Snapshot, decode

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
_PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
            "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]
_RECON_COLOUR = "#a1a1aa"
_PRIOR_COLOUR = "#71717a"
_HIGHLIGHT_COLOUR = "#f43f5e"
_MANIFOLD_COLOUR = "#d4d4d8"
_FONT_FAMILY = "Inter, -apple-system, Segoe UI, sans-serif"

_AXIS_RANGE = [-4.2, 4.2]  # fixed across frames so the animation doesn't jitter


def _label_colours(labels: np.ndarray) -> dict[int, str]:
    uniq = sorted(set(int(v) for v in labels))
    return {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(uniq)}


def _circle(radius: float, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 2 * np.pi, n)
    return radius * np.cos(t), radius * np.sin(t)


def _base_layout(title: str, xaxis_title: str = "x₁", yaxis_title: str = "x₂", height: int = 420) -> dict:
    """Shared layout treatment for the two square (scaleanchor="x") data/latent
    figures. The legend is placed vertically to the right of the plot rather
    than horizontally underneath it — with scaleanchor="x" forcing an equal
    aspect ratio, a narrow column (e.g. the decoder-explorer's 2/5-width
    charts) letterboxes the plot and pushes the x-axis title down into a
    y=-0.2 legend's territory, visibly overlapping the two texts. A right-hand
    legend never shares vertical space with the axis title, so it can't
    collide regardless of how the figure gets squeezed by the column layout.
    """
    axis_style = dict(
        showgrid=True, gridcolor="#eef0f4", gridwidth=1,
        zeroline=True, zerolinecolor="#eef0f4",
        showline=True, linecolor="#e4e4e7", linewidth=1,
    )
    return dict(
        font=dict(family=_FONT_FAMILY, size=12, color="#3f3f46"),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14, color="#18181b")),
        xaxis=dict(**axis_style, title=xaxis_title, range=_AXIS_RANGE, constrain="domain"),
        yaxis=dict(**axis_style, title=yaxis_title, range=_AXIS_RANGE, scaleanchor="x", constrain="domain"),
        legend=dict(orientation="v", x=1.02, y=1, xanchor="left",
                     bgcolor="rgba(255,255,255,0.9)", bordercolor="#e4e4e7",
                     borderwidth=1, font=dict(size=10)),
        margin=dict(l=40, r=170, t=40, b=40),
        height=height,
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )


# ---------------------------------------------------------------------------
# Data space
# ---------------------------------------------------------------------------

def make_data_space_figure(
    snap: Snapshot,
    title: str,
    show_manifold: bool = True,
    highlight: np.ndarray | None = None,
) -> go.Figure:
    """Original points + reconstructions in data space.

    highlight: an (x, y) point to mark distinctly (e.g. a live-decoded
    point from the interactive latent-space explorer).
    """
    X, labels = snap.points, snap.labels
    colours = _label_colours(labels)
    traces: list[go.BaseTraceType] = []

    if show_manifold:
        grid = snap.decoded_grid.reshape(-1, 2)
        traces.append(
            go.Scatter(
                x=grid[:, 0], y=grid[:, 1], mode="markers",
                marker=dict(color=_MANIFOLD_COLOUR, size=4, opacity=0.5),
                name="Decoder manifold (latent grid decoded)",
                hoverinfo="skip",
            )
        )

    # Thin lines from each true point to its reconstruction — visualises
    # reconstruction error as a literal gap.
    Xr = snap.reconstruction
    line_x, line_y = [], []
    for (x0, y0), (x1, y1) in zip(X, Xr):
        line_x += [x0, x1, None]
        line_y += [y0, y1, None]
    traces.append(
        go.Scatter(
            x=line_x, y=line_y, mode="lines",
            line=dict(color=_RECON_COLOUR, width=0.7),
            opacity=0.5, hoverinfo="skip", showlegend=False,
        )
    )

    for lbl, colour in colours.items():
        mask = labels == lbl
        traces.append(
            go.Scatter(
                x=X[mask, 0], y=X[mask, 1], mode="markers",
                marker=dict(color=colour, size=8, line=dict(width=1, color="white")),
                name=f"Class {lbl} (true)",
            )
        )
        traces.append(
            go.Scatter(
                x=Xr[mask, 0], y=Xr[mask, 1], mode="markers",
                marker=dict(color=colour, size=6, symbol="x-thin",
                            line=dict(width=1.5, color=colour)),
                name=f"Class {lbl} (reconstructed)",
                opacity=0.85,
            )
        )

    if highlight is not None:
        traces.append(
            go.Scatter(
                x=[highlight[0]], y=[highlight[1]], mode="markers",
                marker=dict(symbol="star", size=20, color=_HIGHLIGHT_COLOUR,
                            line=dict(width=1.5, color="black")),
                name="Decoded point (interactive)",
            )
        )

    fig = go.Figure(data=traces, layout=go.Layout(**_base_layout(title, "x₁", "x₂")))
    return fig


# ---------------------------------------------------------------------------
# Latent space
# ---------------------------------------------------------------------------

def make_latent_space_figure(
    snap: Snapshot,
    title: str,
    show_prior: bool = True,
    highlight: tuple[float, float] | None = None,
) -> go.Figure:
    """Latent codes (mu) coloured by label. For a VAE, error bars show each
    point's own std = exp(0.5*logvar) — the spread the reparameterisation
    trick samples z from — and dashed circles mark the 1-sigma/2-sigma
    contours of the standard normal prior the KL term pulls toward.
    """
    mu, labels = snap.mu, snap.labels
    colours = _label_colours(labels)
    traces: list[go.BaseTraceType] = []

    if show_prior and snap.variational:
        for radius, dash in [(1.0, "dash"), (2.0, "dot")]:
            cx, cy = _circle(radius)
            traces.append(
                go.Scatter(
                    x=cx, y=cy, mode="lines",
                    line=dict(color=_PRIOR_COLOUR, width=1.2, dash=dash),
                    name=f"Prior N(0,I) — {radius:.0f}σ", hoverinfo="skip",
                )
            )

    for lbl, colour in colours.items():
        mask = labels == lbl
        err = None
        if snap.variational:
            std = np.exp(0.5 * snap.logvar[mask])
            err = std
        marker_err = {}
        if err is not None:
            marker_err = dict(
                error_x=dict(type="data", array=err[:, 0], color=colour, thickness=0.8, width=0),
                error_y=dict(type="data", array=err[:, 1], color=colour, thickness=0.8, width=0),
            )
        traces.append(
            go.Scatter(
                x=mu[mask, 0], y=mu[mask, 1], mode="markers",
                marker=dict(color=colour, size=8, opacity=0.85, line=dict(width=1, color="white")),
                name=f"Class {lbl}",
                **marker_err,
            )
        )

    if highlight is not None:
        traces.append(
            go.Scatter(
                x=[highlight[0]], y=[highlight[1]], mode="markers",
                marker=dict(symbol="star", size=20, color=_HIGHLIGHT_COLOUR,
                            line=dict(width=1.5, color="black")),
                name="z (interactive)",
            )
        )

    fig = go.Figure(data=traces, layout=go.Layout(**_base_layout(title, "z₁", "z₂")))
    return fig


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------

def make_loss_figure(
    history_ae: list[tuple[int, float, float, float]],
    history_vae: list[tuple[int, float, float, float]],
    current_epoch: int,
) -> go.Figure:
    """Each history entry is (epoch, recon_loss, kl_loss, total_loss)."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Reconstruction loss — AE vs VAE", "VAE loss decomposition"),
    )

    e_ae = [h[0] for h in history_ae]
    r_ae = [h[1] for h in history_ae]
    e_vae = [h[0] for h in history_vae]
    r_vae = [h[1] for h in history_vae]
    kl_vae = [h[2] for h in history_vae]
    tot_vae = [h[3] for h in history_vae]

    fig.add_trace(go.Scatter(x=e_ae, y=r_ae, mode="lines", name="AE reconstruction",
                              line=dict(color=_PALETTE[0], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=e_vae, y=r_vae, mode="lines", name="VAE reconstruction",
                              line=dict(color=_PALETTE[1], width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=e_vae, y=r_vae, mode="lines", name="VAE reconstruction",
                              line=dict(color=_PALETTE[1], width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=e_vae, y=kl_vae, mode="lines", name="VAE KL term",
                              line=dict(color=_PALETTE[3], width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=e_vae, y=tot_vae, mode="lines", name="VAE total",
                              line=dict(color=_PALETTE[2], width=2, dash="dot")), row=1, col=2)

    for col in (1, 2):
        fig.add_vline(x=current_epoch, line=dict(color="#a1a1aa", dash="dash", width=1), row=1, col=col)

    fig.update_xaxes(title_text="Epoch", showgrid=True, gridcolor="#eef0f4",
                      showline=True, linecolor="#e4e4e7", linewidth=1)
    fig.update_yaxes(title_text="Loss", showgrid=True, gridcolor="#eef0f4",
                      showline=True, linecolor="#e4e4e7", linewidth=1)
    fig.update_layout(
        font=dict(family=_FONT_FAMILY, size=12, color="#3f3f46"),
        height=320,
        margin=dict(l=50, r=20, t=45, b=40),
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
    )
    fig.update_annotations(font=dict(size=14, color="#18181b"))
    return fig
