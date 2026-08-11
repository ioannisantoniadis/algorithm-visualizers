"""
visualize.py — Plotly figure builders

Three public functions:

  build_figure(snapshots)
      Animated figure with one frame per snapshot. Each frame shows the
      ground-truth path revealed so far, the noisy observations, the
      Kalman estimate trajectory, and the current 2-sigma uncertainty
      ellipse. Includes a Plotly slider for direct frame navigation;
      play/pause is handled by Streamlit-native buttons in app.py.

  make_static_figure(snapshots, idx)
      Non-animated figure for a single step. Used by the manual
      step-through and auto-play loop so Prev/Next re-renders without
      animation overhead.

  make_comparison_figure(snapshots)
      Full-run overlay of the raw (noisy) observations against the
      Kalman-smoothed estimate and the ground truth, for the always-on
      "raw vs. filtered" comparison panel.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from common.theme import base_layout
from .algorithm import Snapshot

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
_PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
            "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]

_COLOR_TRUE = "#a1a1aa"
_COLOR_OBS = _PALETTE[3]   # rose — noisy observations
_COLOR_EST = _PALETTE[0]   # indigo — Kalman estimate
_COLOR_ELLIPSE_FILL = "rgba(99, 102, 241, 0.15)"


def _confidence_ellipse(mean2: np.ndarray, cov2: np.ndarray, n_std: float = 2.0, n_points: int = 60):
    """Return (x, y) points tracing an n_std-sigma uncertainty ellipse.

    Eigen-decomposing the 2x2 covariance gives the ellipse's principal
    axes; scaling a unit circle by n_std * sqrt(eigenvalues) and rotating
    by the eigenvectors turns it into the actual uncertainty ellipse.
    n_std=2 traces each principal axis out to roughly its 95% bound.
    """
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.clip(vals, 0.0, None)  # guard against tiny negative numerical noise
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    ellipse = circle * (n_std * np.sqrt(vals))
    pts = ellipse @ vecs.T + mean2
    return pts[:, 0], pts[:, 1]


def _history_by_step(snapshots: list[Snapshot], upto: int):
    """Collect one true/observation/estimate point per time step, up to
    (and including) index `upto`, deduplicating the predict+update pair
    that share the same true position."""
    true_by_step: dict[int, np.ndarray] = {}
    obs_by_step: dict[int, np.ndarray] = {}
    est_by_step: dict[int, np.ndarray] = {}

    for s in snapshots[: upto + 1]:
        true_by_step[s.step] = s.true_pos
        if s.observation is not None:
            obs_by_step[s.step] = s.observation
        if s.phase in ("init", "update"):
            est_by_step[s.step] = s.pos_mean

    true_pts = np.array([true_by_step[k] for k in sorted(true_by_step)])
    obs_pts = np.array([obs_by_step[k] for k in sorted(obs_by_step)]) if obs_by_step else np.empty((0, 2))
    est_pts = np.array([est_by_step[k] for k in sorted(est_by_step)]) if est_by_step else np.empty((0, 2))
    return true_pts, obs_pts, est_pts


def _axis_ranges(snapshots: list[Snapshot], pad: float = 2.0):
    all_pts = np.vstack(
        [s.true_pos for s in snapshots]
        + [s.observation for s in snapshots if s.observation is not None]
    )
    x_range = [float(all_pts[:, 0].min()) - pad, float(all_pts[:, 0].max()) + pad]
    y_range = [float(all_pts[:, 1].min()) - pad, float(all_pts[:, 1].max()) + pad]
    return x_range, y_range


def _make_traces(snapshots: list[Snapshot], idx: int) -> list[go.Scatter]:
    snap = snapshots[idx]
    true_pts, obs_pts, est_pts = _history_by_step(snapshots, idx)
    ellipse_x, ellipse_y = _confidence_ellipse(snap.pos_mean, snap.pos_cov)

    traces = [
        go.Scatter(
            x=true_pts[:, 0], y=true_pts[:, 1],
            mode="lines",
            line=dict(color=_COLOR_TRUE, width=2, dash="dot"),
            name="True trajectory",
        ),
        go.Scatter(
            x=obs_pts[:, 0], y=obs_pts[:, 1],
            mode="markers",
            marker=dict(color=_COLOR_OBS, size=7, symbol="x"),
            name="Noisy observations",
        ),
        go.Scatter(
            x=ellipse_x, y=ellipse_y,
            mode="lines",
            fill="toself",
            line=dict(color=_COLOR_EST, width=1),
            fillcolor=_COLOR_ELLIPSE_FILL,
            name="Uncertainty (2σ)",
        ),
        go.Scatter(
            x=est_pts[:, 0], y=est_pts[:, 1],
            mode="lines+markers",
            line=dict(color=_COLOR_EST, width=2),
            marker=dict(size=5, color=_COLOR_EST),
            name="Kalman estimate",
        ),
        go.Scatter(
            x=[snap.pos_mean[0]], y=[snap.pos_mean[1]],
            mode="markers",
            marker=dict(size=13, color=_COLOR_EST, symbol="circle",
                        line=dict(width=2, color="white")),
            name="Current estimate",
            showlegend=False,
        ),
        go.Scatter(
            x=[snap.true_pos[0]], y=[snap.true_pos[1]],
            mode="markers",
            marker=dict(size=10, color=_COLOR_TRUE, symbol="circle",
                        line=dict(width=1.5, color="white")),
            name="Current true position",
            showlegend=False,
        ),
    ]
    return traces


def build_figure(snapshots: list[Snapshot]) -> go.Figure:
    """Return a Plotly Figure with one animation frame per snapshot."""
    if not snapshots:
        raise ValueError("snapshot list is empty")

    x_range, y_range = _axis_ranges(snapshots)

    frames: list[go.Frame] = []
    frame_labels: list[str] = []
    for i, snap in enumerate(snapshots):
        frames.append(go.Frame(data=_make_traces(snapshots, i), name=str(i),
                                layout=go.Layout(title_text=snap.title)))
        frame_labels.append(snap.title)

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=base_layout(
            None,  # per-frame title — already shown in the page's progress bar
            xaxis=dict(title="x", range=x_range),
            yaxis=dict(title="y", range=y_range, scaleanchor="x"),
        ),
    )
    fig.update_layout(
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Step: ", font=dict(size=13), xanchor="center"),
                pad=dict(t=50, b=10),
                len=0.9,
                x=0.05,
                steps=[
                    dict(
                        args=[[str(i)], {
                            "frame": {"duration": 300, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 150},
                        }],
                        label=label[:40],
                        method="animate",
                    )
                    for i, label in enumerate(frame_labels)
                ],
            )
        ]
    )
    return fig


def make_static_figure(snapshots: list[Snapshot], idx: int) -> go.Figure:
    """Return a non-animated Plotly figure for a single step, `idx`,
    given the full snapshot history (needed to draw the trails)."""
    x_range, y_range = _axis_ranges(snapshots)
    traces = _make_traces(snapshots, idx)
    layout = base_layout(
        None,  # per-frame title — already shown in the page's progress bar
        xaxis=dict(title="x", range=x_range),
        yaxis=dict(title="y", range=y_range, scaleanchor="x"),
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig


def make_comparison_figure(snapshots: list[Snapshot]) -> go.Figure:
    """Full-run static overlay: ground truth vs. raw observations vs.
    the Kalman-smoothed estimate. Used for the always-visible
    'raw vs. filtered' comparison panel."""
    true_pts, obs_pts, est_pts = _history_by_step(snapshots, len(snapshots) - 1)
    x_range, y_range = _axis_ranges(snapshots)

    traces = [
        go.Scatter(x=true_pts[:, 0], y=true_pts[:, 1], mode="lines",
                   line=dict(color=_COLOR_TRUE, width=2, dash="dot"), name="True trajectory"),
        go.Scatter(x=obs_pts[:, 0], y=obs_pts[:, 1], mode="lines+markers",
                   line=dict(color=_COLOR_OBS, width=1),
                   marker=dict(color=_COLOR_OBS, size=5, symbol="x"),
                   name="Raw observations"),
        go.Scatter(x=est_pts[:, 0], y=est_pts[:, 1], mode="lines+markers",
                   line=dict(color=_COLOR_EST, width=2.5),
                   marker=dict(color=_COLOR_EST, size=5),
                   name="Kalman-smoothed"),
    ]
    layout = base_layout(
        "Full run — raw observations vs. Kalman-smoothed path",  # static caption — keep
        height=460,
        xaxis=dict(title="x", range=x_range),
        yaxis=dict(title="y", range=y_range, scaleanchor="x"),
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig
