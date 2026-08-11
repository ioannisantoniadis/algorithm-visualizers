"""
visualize.py — Plotly figure builders

Four public functions, each consuming a single Snapshot (or the full
snapshot history) produced by algorithm.fit():

  make_boundary_figure(snap)
      Decision-boundary contour (predicted P(class=1) over the
      background grid) with the training points overlaid, coloured by
      their true label.

  make_network_figure(snap, layer_sizes)
      The network drawn as a layered graph. Edge width encodes weight
      (or, on a backward-pass snapshot, gradient) magnitude; edge colour
      encodes sign — blue for positive, red for negative.

  make_loss_figure(history, current_epoch, click_epoch)
      Loss-vs-epoch curve for the whole training run, with a marker at
      the currently viewed epoch and an annotation at the epoch where
      accuracy first crosses the "solved" threshold.

  make_activation_heatmap_figure(snap, grid_x, grid_y)
      Small-multiple heat-maps, one per first-hidden-layer neuron,
      showing how each individual unit partitions the input space.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.theme import apply_theme, base_layout
from .algorithm import Snapshot

# ---------------------------------------------------------------------------
# Shared style constants — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
# ---------------------------------------------------------------------------

_PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
            "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]

_CLASS_COLOURS = [_PALETTE[0], _PALETTE[3]]  # class 0 (indigo), class 1 (rose)
_BOUNDARY_SCALE = [[0.0, "#e5e5ff"], [0.5, "#fbfbfd"], [1.0, "#fde1e7"]]
_POS_EDGE = _PALETTE[0]   # positive weight / gradient — indigo
_NEG_EDGE = _PALETTE[3]   # negative weight / gradient — rose
_NODE_FILL = "#f8f8fb"
_NODE_LINE = "#3f3f46"
_ACT_SCALE = [[0.0, _PALETTE[3]], [0.5, "#fbfbfd"], [1.0, _PALETTE[0]]]


def _layer_x_positions(n_layers: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_layers)


def _layer_y_positions(n_units: int) -> np.ndarray:
    if n_units == 1:
        return np.array([0.0])
    return np.linspace(-(n_units - 1) / 2.0, (n_units - 1) / 2.0, n_units)


# ---------------------------------------------------------------------------
# Decision boundary
# ---------------------------------------------------------------------------

def make_boundary_figure(snap: Snapshot, grid_x: np.ndarray, grid_y: np.ndarray) -> go.Figure:
    """Contour of predicted P(class=1) plus the training points."""
    X, y = snap.points, snap.targets

    traces: list[go.BaseTraceType] = [
        go.Contour(
            x=grid_x[0, :],
            y=grid_y[:, 0],
            z=snap.grid_probs,
            zmin=0.0,
            zmax=1.0,
            colorscale=_BOUNDARY_SCALE,
            showscale=True,
            colorbar=dict(title="P(class 1)", len=0.8),
            line=dict(width=0),
            contours=dict(coloring="fill"),
            opacity=0.9,
            name="Decision surface",
        )
    ]

    for cls in (0, 1):
        mask = y == cls
        traces.append(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                marker=dict(
                    color=_CLASS_COLOURS[cls],
                    size=8,
                    line=dict(width=1, color="white"),
                ),
                name=f"Class {cls}",
            )
        )

    fig = go.Figure(
        data=traces,
        layout=base_layout(
            None,  # per-frame title — already shown in the page's progress bar
            xaxis=dict(title="x₁"),
            yaxis=dict(title="x₂", scaleanchor="x"),
            height=480,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Network graph
# ---------------------------------------------------------------------------

def make_network_figure(snap: Snapshot, layer_sizes: list[int]) -> go.Figure:
    """Layered network diagram.

    On a "forward" (or "init") snapshot, edges are coloured/sized by the
    current weights. On a "backward" snapshot, edges are coloured/sized
    by the gradient the backward pass just computed for that weight —
    literally showing the error signal flowing from output back to input.
    """
    show_grad = snap.phase == "backward" and snap.grad_weights is not None
    matrices = snap.grad_weights if show_grad else snap.weights

    n_layers = len(layer_sizes)
    xs = _layer_x_positions(n_layers)
    ys = [_layer_y_positions(n) for n in layer_sizes]

    # Normalise magnitudes within this snapshot so edge widths are comparable
    max_abs = max((np.abs(m).max() for m in matrices), default=1e-8)
    max_abs = max(max_abs, 1e-8)

    edge_traces: list[go.Scatter] = []
    for li, M in enumerate(matrices):
        n_in, n_out = M.shape
        for i in range(n_in):
            for j in range(n_out):
                val = float(M[i, j])
                norm = min(abs(val) / max_abs, 1.0)
                colour = _POS_EDGE if val >= 0 else _NEG_EDGE
                edge_traces.append(
                    go.Scatter(
                        x=[xs[li], xs[li + 1]],
                        y=[ys[li][i], ys[li + 1][j]],
                        mode="lines",
                        line=dict(width=0.6 + 5.5 * norm, color=colour),
                        opacity=0.25 + 0.65 * norm,
                        hoverinfo="text",
                        text=f"{'∂L/∂w' if show_grad else 'w'} = {val:+.3f}",
                        showlegend=False,
                    )
                )

    node_traces: list[go.Scatter] = []
    layer_labels = (
        ["Input"] + [f"Hidden {i + 1}" for i in range(n_layers - 2)] + ["Output"]
    )
    for li, n_units in enumerate(layer_sizes):
        if li == 0:
            labels = [f"x{i + 1}" for i in range(n_units)]
        elif li == n_layers - 1:
            labels = ["ŷ"]
        else:
            labels = [f"h{i + 1}" for i in range(n_units)]
        node_traces.append(
            go.Scatter(
                x=[xs[li]] * n_units,
                y=ys[li],
                mode="markers+text",
                marker=dict(size=30, color=_NODE_FILL, line=dict(width=2, color=_NODE_LINE)),
                text=labels,
                textposition="middle center",
                textfont=dict(size=11, color=_NODE_LINE),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    annotations = [
        dict(
            x=xs[li], y=min(ys[li].min(), -3) - 0.8 if len(ys[li]) else -3.8,
            xref="x", yref="y", text=f"<b>{layer_labels[li]}</b>",
            showarrow=False, font=dict(size=12, color="#4B5563"),
        )
        for li in range(n_layers)
    ]

    subtitle = "gradients (backward pass)" if show_grad else "weights"
    layout = base_layout(
        f"Network — {subtitle}",  # not shown elsewhere on the page — keep
        xaxis=dict(visible=False, range=[-0.15, 1.15]),
        yaxis=dict(visible=False),
        showlegend=False,
        margin=dict(l=20, r=20, b=60),
        height=480,
    )
    layout.update(annotations=annotations)
    fig = go.Figure(data=edge_traces + node_traces, layout=layout)
    return fig


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def make_loss_figure(
    history: list[tuple[int, float, float]],
    current_epoch: int,
    click_epoch: int | None,
) -> go.Figure:
    """Loss-vs-epoch curve for the full run, with a marker at current_epoch
    and an annotation where accuracy first "clicks" past the solved threshold.
    """
    epochs = [h[0] for h in history]
    losses = [h[1] for h in history]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs, y=losses, mode="lines", line=dict(color="#3f3f46", width=2),
            name="Loss",
        )
    )

    cur = next((h for h in history if h[0] == current_epoch), history[-1])
    fig.add_trace(
        go.Scatter(
            x=[cur[0]], y=[cur[1]], mode="markers",
            marker=dict(size=12, color=_PALETTE[3], line=dict(width=2, color="white")),
            name="Current epoch", showlegend=False,
        )
    )

    if click_epoch is not None:
        click_loss = next((h[1] for h in history if h[0] == click_epoch), None)
        if click_loss is not None:
            fig.add_vline(x=click_epoch, line=dict(color=_PALETTE[0], dash="dash", width=1.5))
            fig.add_annotation(
                x=click_epoch, y=max(losses),
                text="boundary clicks into shape ▾",
                showarrow=False, yshift=12, font=dict(size=11, color=_PALETTE[0]),
            )

    fig.update_layout(
        base_layout(
            "Training loss",  # static caption, not shown elsewhere — keep
            xaxis=dict(title="Epoch"),
            yaxis=dict(title="Binary cross-entropy"),
            margin=dict(l=50, t=45),
            height=300,
            showlegend=False,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Hidden-neuron activation heat-maps
# ---------------------------------------------------------------------------

def make_activation_heatmap_figure(
    snap: Snapshot, grid_x: np.ndarray, grid_y: np.ndarray, max_cols: int = 4
) -> go.Figure:
    """Small-multiple heat-maps — one per first-hidden-layer neuron —
    showing each unit's tanh activation across the whole input space.
    Blue/red regions are where that single neuron fires positive/negative;
    the decision boundary is a weighted combination of these patterns.
    """
    n_units = snap.hidden_activations.shape[-1]
    n_cols = min(max_cols, n_units)
    n_rows = int(np.ceil(n_units / n_cols))

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Neuron h{i + 1}" for i in range(n_units)],
        horizontal_spacing=0.04, vertical_spacing=0.12,
    )

    for i in range(n_units):
        r, c = divmod(i, n_cols)
        heatmap_kwargs = dict(
            x=grid_x[0, :], y=grid_y[:, 0], z=snap.hidden_activations[:, :, i],
            zmin=-1, zmax=1, colorscale=_ACT_SCALE,
            showscale=(i == n_units - 1),
        )
        if i == n_units - 1:
            heatmap_kwargs["colorbar"] = dict(title="tanh", len=0.7)
        fig.add_trace(go.Heatmap(**heatmap_kwargs), row=r + 1, col=c + 1)

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    apply_theme(fig, height=220 * n_rows, showlegend=False)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig
