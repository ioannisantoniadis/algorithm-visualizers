"""
visualize.py — Plotly figure builders

Three public functions:

  build_figure(snapshots)
      Animated figure with one frame per snapshot.  Each frame shows
      cluster-coloured scatter traces and centroid stars (★).  Includes
      a Plotly slider for direct frame navigation; play/pause controls
      are handled by Streamlit-native buttons in app.py.

  make_static_figure(snap)
      Non-animated figure for a single snapshot.  Used by the step-
      through and auto-play modes so Prev/Next re-renders without
      animation overhead.

  make_picker_figure(X, selected, k)
      Centroid-placement picker: all points grey and clickable, already-
      placed centroids shown as coloured ★ stars.  Pass to
      st.plotly_chart(on_select="rerun") so Streamlit captures clicks.
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


def _cluster_colour(k: int, n_clusters: int) -> list[str]:
    """Return n_clusters hex colours, cycling the palette if needed."""
    return [_PALETTE[i % len(_PALETTE)] for i in range(n_clusters)]


def _make_frame(
    snap: Snapshot,
    n_clusters: int,
    colours: list[str],
    frame_name: str,
) -> go.Frame:
    """Build one Plotly Frame from a single Snapshot."""
    X = snap.points
    labels = snap.labels
    centroids = snap.centroids

    traces: list[go.Scatter] = []

    # One scatter trace per cluster so each gets its own colour legend entry.
    # Points still unassigned (label == -1, init frame only) go to a grey trace.
    for c in range(n_clusters):
        mask = labels == c
        colour = colours[c]
        traces.append(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                marker=dict(
                    color=colour,
                    size=7,
                    opacity=0.75,
                    line=dict(width=0.5, color="white"),
                ),
                name=f"Cluster {c}",
                showlegend=True,
            )
        )

    # Grey "unassigned" trace (only non-empty on the init frame)
    unassigned = labels == -1
    traces.append(
        go.Scatter(
            x=X[unassigned, 0],
            y=X[unassigned, 1],
            mode="markers",
            marker=dict(color="#aaaaaa", size=7, opacity=0.6),
            name="Unassigned",
            showlegend=bool(unassigned.any()),
        )
    )

    # Centroid stars — drawn on top so they're always visible
    traces.append(
        go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode="markers+text",
            marker=dict(
                symbol="star",
                size=22,
                color=colours[:n_clusters],
                line=dict(width=1.5, color="black"),
            ),
            text=[f"C{c}" for c in range(n_clusters)],
            textposition="top center",
            textfont=dict(size=10, color="black"),
            name="Centroids",
            showlegend=True,
        )
    )

    return go.Frame(data=traces, name=frame_name, layout=go.Layout(title_text=snap.title))


def _axis_range(X: np.ndarray, axis: int, pad: float) -> list[float]:
    return [float(X[:, axis].min()) - pad, float(X[:, axis].max()) + pad]


def build_figure(snapshots: list[Snapshot]) -> go.Figure:
    """Return a Plotly Figure with one animation frame per snapshot.

    Parameters
    ----------
    snapshots: output of algorithm.fit()

    Returns
    -------
    A go.Figure ready to pass to st.plotly_chart()
    """
    if not snapshots:
        raise ValueError("snapshot list is empty")

    X = snapshots[0].points
    n_clusters = int(snapshots[-1].centroids.shape[0])
    colours = _cluster_colour(0, n_clusters)

    # ----------------------------------------------------------------
    # Build frames
    # ----------------------------------------------------------------
    frames: list[go.Frame] = []
    frame_labels: list[str] = []

    for i, snap in enumerate(snapshots):
        name = str(i)
        frames.append(_make_frame(snap, n_clusters, colours, name))
        frame_labels.append(snap.title)

    # ----------------------------------------------------------------
    # Initial display = first frame's data
    # ----------------------------------------------------------------
    pad = 0.3
    layout = base_layout(
        None,  # per-frame title — already shown in the page's progress bar
        xaxis=dict(title="x₁", range=_axis_range(X, 0, pad)),
        yaxis=dict(title="x₂", range=_axis_range(X, 1, pad), scaleanchor="x"),
    )
    layout.update(
        sliders=[
            dict(
                active=0,
                currentvalue=dict(
                    prefix="Step: ",
                    font=dict(size=13),
                    xanchor="center",
                ),
                pad=dict(t=50, b=10),
                len=0.9,
                x=0.05,
                steps=[
                    dict(
                        args=[
                            [str(i)],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 150},
                            },
                        ],
                        label=label[:40],  # truncate long titles in slider
                        method="animate",
                    )
                    for i, label in enumerate(frame_labels)
                ],
            )
        ],
    )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=layout)
    return fig


def make_picker_figure(
    X: np.ndarray,
    selected: list[list[float]],
    k: int,
) -> go.Figure:
    """Return a Plotly figure used for manual centroid placement.

    All data points are rendered in grey.  Already-placed centroids are
    overlaid as coloured ★ stars.  The title prompts the user for the
    next click.

    Pass this figure to st.plotly_chart(fig, on_select="rerun",
    selection_mode="points") — Streamlit will rerun the script and
    return the clicked point's coordinates in the event object.
    """
    colours = _cluster_colour(0, k)
    n_placed = len(selected)

    # All data points in neutral grey — clickable
    data_trace = go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode="markers",
        marker=dict(
            color="#bbbbbb",
            size=7,
            opacity=0.7,
            line=dict(width=0.5, color="white"),
        ),
        name="Data points",
        showlegend=False,
    )

    traces: list[go.BaseTraceType] = [data_trace]

    # Already-placed centroids as coloured stars
    if selected:
        sel = np.array(selected)
        placed_colours = [colours[i] for i in range(n_placed)]
        traces.append(
            go.Scatter(
                x=sel[:, 0],
                y=sel[:, 1],
                mode="markers+text",
                marker=dict(
                    symbol="star",
                    size=24,
                    color=placed_colours,
                    line=dict(width=1.5, color="black"),
                ),
                text=[f"C{i}" for i in range(n_placed)],
                textposition="top center",
                textfont=dict(size=11, color="black"),
                name="Placed centroids",
                showlegend=False,
            )
        )

    if n_placed < k:
        title_text = f"Click a point to place centroid {n_placed + 1} of {k}"
    else:
        title_text = f"All {k} centroids placed — press 'Run K-means' below"

    # Axis ranges: pin to the data so the plot doesn't jump between clicks
    pad = 0.5
    layout = base_layout(
        title_text,  # dynamic instruction, not shown elsewhere on the page — keep
        height=500,
        showlegend=False,
        xaxis=dict(title="x₁", range=_axis_range(X, 0, pad)),
        yaxis=dict(title="x₂", range=_axis_range(X, 1, pad), scaleanchor="x"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    layout.update(clickmode="event+select")

    fig = go.Figure(data=traces, layout=layout)
    return fig


def make_static_figure(snap: Snapshot) -> go.Figure:
    """Return a simple non-animated Plotly figure for a single snapshot.

    Used by the Streamlit manual step-through mode so clicking Prev/Next
    does a clean re-render without the overhead of the full animation.
    """
    n_clusters = int(snap.centroids.shape[0])
    colours = _cluster_colour(0, n_clusters)
    frame = _make_frame(snap, n_clusters, colours, "static")

    pad = 0.3
    X = snap.points
    layout = base_layout(
        None,  # per-frame title — already shown in the page's progress bar
        xaxis=dict(title="x₁", range=_axis_range(X, 0, pad)),
        yaxis=dict(title="x₂", range=_axis_range(X, 1, pad), scaleanchor="x"),
    )
    fig = go.Figure(data=frame.data, layout=layout)
    return fig
