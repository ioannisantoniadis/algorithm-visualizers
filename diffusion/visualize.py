"""
visualize.py — Plotly figure builders for the diffusion visualiser

Four public functions:

  make_point_figure(snap, labels, ref_points, title)
      Single-frame scatter of the point cloud at one diffusion step.  Used
      by both the forward and reverse step-through views; axis ranges are
      passed in explicitly so consecutive frames don't jar as points spread
      out into noise or condense back into structure.

  axis_range(snapshot_lists, pad)
      Shared helper: compute one fixed (x, y) range across every snapshot in
      one or more lists, so an entire animation uses a stable "camera".

  make_schedule_figure(schedule)
      Two-panel chart of beta_t (noise added per step) and the signal-to-
      noise ratio log10(alpha_bar_t / (1 - alpha_bar_t)) across the chain —
      the "why T steps, and why this shape" picture.

  make_loss_figure(loss_history)
      Training loss curve for the score network (log-scale y-axis, since
      the simple DDPM loss typically spans more than one order of magnitude
      during training).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.theme import apply_theme, base_layout
from .algorithm import NoiseSchedule, Snapshot

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
_PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
            "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]

_NOISE_COLOUR = "#a1a1aa"  # neutral grey used for the reference / noise overlay


def _colours_for(labels: np.ndarray) -> list[str]:
    return [_PALETTE[int(c) % len(_PALETTE)] for c in labels]


def axis_range(snapshot_lists: list[list[Snapshot]], pad: float = 0.6) -> tuple[list[float], list[float]]:
    """Fixed (x_range, y_range) spanning every point in every snapshot passed.

    Computing this once up front — rather than letting Plotly auto-range
    each frame — is what makes the step-through/auto-play feel like a
    continuous camera instead of a jump-cut on every step.
    """
    all_pts = np.vstack([s.points for lst in snapshot_lists for s in lst])
    x_min, x_max = float(all_pts[:, 0].min()) - pad, float(all_pts[:, 0].max()) + pad
    y_min, y_max = float(all_pts[:, 1].min()) - pad, float(all_pts[:, 1].max()) + pad
    return [x_min, x_max], [y_min, y_max]


def make_point_figure(
    snap: Snapshot,
    labels: np.ndarray | None,
    x_range: list[float],
    y_range: list[float],
    ref_points: np.ndarray | None = None,
    title: str | None = None,
) -> go.Figure:
    """Static (non-animated) scatter of one diffusion step.

    ref_points, if given, is drawn first as a faint grey backdrop — e.g. the
    original data distribution behind a forward-noised frame, or the target
    data distribution behind a reverse-sampled frame — so it's always
    visible how far the current cloud is from where it started/is heading.
    """
    traces: list[go.Scatter] = []

    if ref_points is not None:
        traces.append(
            go.Scatter(
                x=ref_points[:, 0], y=ref_points[:, 1],
                mode="markers",
                marker=dict(color=_NOISE_COLOUR, size=5, opacity=0.25),
                name="Reference",
                showlegend=False,
            )
        )

    X = snap.points
    if labels is not None:
        colours = _colours_for(labels)
        for c in sorted(set(labels.tolist())):
            mask = labels == c
            traces.append(
                go.Scatter(
                    x=X[mask, 0], y=X[mask, 1],
                    mode="markers",
                    marker=dict(color=_PALETTE[int(c) % len(_PALETTE)], size=7, opacity=0.85,
                                line=dict(width=0.5, color="white")),
                    name=f"Cluster {c}",
                    showlegend=True,
                )
            )
    else:
        traces.append(
            go.Scatter(
                x=X[:, 0], y=X[:, 1],
                mode="markers",
                marker=dict(color="#4C78A8", size=7, opacity=0.85,
                            line=dict(width=0.5, color="white")),
                name="Samples",
                showlegend=False,
            )
        )

    # `title` (or its fallback, snap.title) is always a per-frame label that
    # duplicates the page's own progress-bar text at both call sites in
    # apps/diffusion.py — so it's intentionally not forwarded to the chart.
    fig = go.Figure(
        data=traces,
        layout=base_layout(
            None,
            height=520,
            xaxis=dict(title="x₁", range=x_range),
            yaxis=dict(title="x₂", range=y_range, scaleanchor="x"),
        ),
    )
    return fig


def make_schedule_figure(schedule: NoiseSchedule) -> go.Figure:
    """Two-panel view of the noise schedule: beta_t (left) and the
    signal-to-noise ratio on a log scale (right, since SNR decays roughly
    exponentially and spans many orders of magnitude across the chain).
    """
    t = np.arange(1, schedule.T + 1)
    snr = np.array([schedule.snr(int(ti)) for ti in t])
    snr = np.clip(snr, 1e-8, None)  # guard log(0) at the very last step

    fig = make_subplots(
        rows=1, cols=2,
        # Short — these sit above narrow subplot columns and Plotly
        # doesn't wrap/truncate subplot_titles text.
        subplot_titles=("Noise (β_t)", "SNR (log)"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(x=t, y=schedule.betas, mode="lines", line=dict(color=_PALETTE[3], width=2), name="β_t"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=schedule.alpha_bars, mode="lines", line=dict(color=_PALETTE[0], width=2, dash="dot"),
                    name="ᾱ_t (cumulative signal)"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=snr, mode="lines", line=dict(color=_PALETTE[1], width=2), name="SNR_t"),
        row=1, col=2,
    )
    grid_style = dict(showgrid=True, gridcolor="#eef0f4", gridwidth=1,
                       zeroline=False, showline=True, linecolor="#e4e4e7", linewidth=1)
    fig.update_yaxes(type="log", title="SNR (log scale)", **grid_style, row=1, col=2)
    fig.update_yaxes(title="value", **grid_style, row=1, col=1)
    fig.update_xaxes(title="diffusion step t", **grid_style, row=1, col=1)
    fig.update_xaxes(title="diffusion step t", **grid_style, row=1, col=2)
    apply_theme(
        fig,
        f"Noise schedule — {schedule.kind}, T={schedule.T}",  # static caption, not per-frame
        height=340,
    )
    return fig


def make_loss_figure(loss_history: list[float]) -> go.Figure:
    """Training loss curve for the score network (log-scale y-axis)."""
    epochs = np.arange(1, len(loss_history) + 1)
    fig = go.Figure(
        data=[go.Scatter(x=epochs, y=loss_history, mode="lines", line=dict(color=_PALETTE[5], width=2))],
        layout=base_layout(
            "Training loss — E[‖ε − ε̂‖²]",  # static caption, kept short so it doesn't overflow a narrow card
            height=320,
            xaxis=dict(title="epoch"),
            yaxis=dict(title="loss (log scale)", type="log"),
            showlegend=False,
        ),
    )
    return fig
