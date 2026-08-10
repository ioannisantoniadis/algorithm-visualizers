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

from .algorithm import Snapshot

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
_PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
            "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]
_FONT_FAMILY = "Inter, -apple-system, Segoe UI, sans-serif"


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


# ---------------------------------------------------------------------------
# Shared layout helper — matches the palette/typography used across the
# whole portfolio (see the sibling dbscan-viz repo's
# dbscan/visualize.py::_base_layout for the canonical recipe).
# ---------------------------------------------------------------------------

def _base_layout(title: str, X: np.ndarray, height: int = 560, pad: float = 0.3) -> go.Layout:
    axis_style = dict(
        showgrid=True, gridcolor="#eef0f4", gridwidth=1,
        zeroline=False, showline=True, linecolor="#e4e4e7", linewidth=1,
    )
    return go.Layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=16, color="#18181b")),
        xaxis=dict(**axis_style, title="x₁",
                   range=[float(X[:, 0].min()) - pad, float(X[:, 0].max()) + pad]),
        yaxis=dict(**axis_style, title="x₂",
                   range=[float(X[:, 1].min()) - pad, float(X[:, 1].max()) + pad],
                   scaleanchor="x"),
        legend=dict(orientation="v", x=1.02, y=1,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e4e4e7", borderwidth=1,
                    font=dict(size=12)),
        margin=dict(l=40, r=180, t=60, b=40),
        height=height,
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )


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
    layout = _base_layout(snapshots[0].title, X)
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
    layout = _base_layout(title_text, X, height=500, pad=0.5)
    layout.update(
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
        clickmode="event+select",
    )

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

    layout = _base_layout(snap.title, snap.points)
    fig = go.Figure(data=frame.data, layout=layout)
    return fig
