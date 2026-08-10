"""
visualize.py — Plotly figure builders

Four public functions:

  build_figure(snapshots, kernel, gamma, degree, coef0)
      Animated figure with one frame per SMO sweep. Each frame redraws the
      decision-function contour (shaded regions + boundary + margin lines)
      and re-styles every point by class / support-vector / violation
      status. Includes a Plotly slider; play/pause is handled by
      Streamlit-native buttons in app.py.

  make_static_figure(snap, kernel, gamma, degree, coef0)
      Non-animated figure for a single snapshot — used by manual
      step-through so Prev/Next re-renders without animation overhead.

  make_loss_figure(snapshots, current_idx)
      Small line chart of the SMO dual objective across sweeps, with the
      current step highlighted — monotonically rising by construction,
      and by strong duality the mirror image of the hinge loss falling.

  make_preview_figure(X, y)
      Plain scatter of the raw dataset, shown before training starts.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .algorithm import Snapshot, kernel_matrix

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
_PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
            "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]
_FONT_FAMILY = "Inter, -apple-system, Segoe UI, sans-serif"

_POS_COLOR = _PALETTE[0]   # class +1 — indigo
_NEG_COLOR = _PALETTE[3]   # class -1 — rose
_VIOLATION_RING = "#dc2626"  # ring colour flagging a margin violation


# ---------------------------------------------------------------------------
# Decision surface on a grid — the actual "kernel trick made visible" bit
# ---------------------------------------------------------------------------

def _decision_grid(
    snap: Snapshot,
    kernel: str,
    gamma: float,
    degree: int,
    coef0: float,
    res: int = 65,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate f(x) = sum alpha_i y_i K(x_i, x) + b on a regular grid.

    Works identically for every kernel: only kernel_matrix changes, the
    surface it produces is what gets contoured at levels -1 / 0 / +1.
    """
    X = snap.points
    pad = 1.0
    x_min, x_max = float(X[:, 0].min()) - pad, float(X[:, 0].max()) + pad
    y_min, y_max = float(X[:, 1].min()) - pad, float(X[:, 1].max()) + pad

    gx = np.linspace(x_min, x_max, res)
    gy = np.linspace(y_min, y_max, res)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

    K_grid = kernel_matrix(X, grid_pts, kernel, gamma, degree, coef0)  # (N, res*res)
    f = (snap.alpha * snap.labels) @ K_grid + snap.b
    Z = f.reshape(GX.shape)
    return gx, gy, Z


def _surface_traces(gx: np.ndarray, gy: np.ndarray, Z: np.ndarray) -> list[go.Contour]:
    """Shaded class regions + three named contour lines: -1 (margin), 0 (boundary), +1 (margin)."""
    zmax = float(np.abs(Z).max()) or 1.0

    fill = go.Contour(
        x=gx, y=gy, z=Z,
        showscale=False,
        colorscale=[
            [0.0, "rgba(244,63,94,0.22)"],
            [0.5, "rgba(255,255,255,0.0)"],
            [1.0, "rgba(99,102,241,0.22)"],
        ],
        zmin=-zmax, zmax=zmax, zmid=0,
        contours=dict(coloring="fill", showlines=False),
        hoverinfo="skip",
        name="Decision regions",
        showlegend=False,
    )

    def _line(level: float, color: str, dash: str, name: str) -> go.Contour:
        return go.Contour(
            x=gx, y=gy, z=Z,
            showscale=False,
            contours=dict(start=level, end=level, size=1, coloring="lines"),
            line=dict(width=2.2, dash=dash, color=color),
            hoverinfo="skip",
            name=name,
            showlegend=False,
        )

    return [
        fill,
        _line(-1.0, _NEG_COLOR, "dash", "Margin (y=-1)"),
        _line(0.0, "#18181b", "solid", "Decision boundary"),
        _line(1.0, _POS_COLOR, "dash", "Margin (y=+1)"),
    ]


# ---------------------------------------------------------------------------
# Point styling — colour by class, size/border by support-vector & slack
# ---------------------------------------------------------------------------

def _point_traces(snap: Snapshot) -> list[go.Scatter]:
    X, y = snap.points, snap.labels
    sv = snap.support_mask
    slack = snap.slack

    traces: list[go.Scatter] = []
    for cls, colour, label in [(1, _POS_COLOR, "Class +1"), (-1, _NEG_COLOR, "Class −1")]:
        mask = y == cls
        n = int(mask.sum())
        sizes = np.where(sv[mask], 13, 7)
        line_width = np.where(sv[mask], 2.5, 0.5)
        line_colour = np.where(slack[mask] > 1e-6, _VIOLATION_RING, "#18181b")

        traces.append(
            go.Scatter(
                x=X[mask, 0], y=X[mask, 1],
                mode="markers",
                marker=dict(
                    color=colour, size=sizes, opacity=0.85,
                    line=dict(width=line_width.tolist(), color=line_colour.tolist()),
                ),
                text=[f"α={a:.2f}  ξ={s:.2f}" for a, s in zip(snap.alpha[mask], slack[mask])],
                hovertemplate="%{text}<extra></extra>",
                name=label,
                showlegend=n > 0,
            )
        )

    # Legend-only proxies explaining the marker encoding (no visible data)
    traces.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="white", size=13, line=dict(width=2.5, color="#18181b")),
        name="Support vector (thick border)", showlegend=True,
    ))
    traces.append(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color="white", size=9, line=dict(width=2, color=_VIOLATION_RING)),
        name="Margin violation (ξ>0)", showlegend=True,
    ))
    return traces


_AXIS_STYLE = dict(
    showgrid=True, gridcolor="#eef0f4", gridwidth=1,
    zeroline=False, showline=True, linecolor="#e4e4e7", linewidth=1,
)


def _layout(title: str, X: np.ndarray) -> go.Layout:
    pad = 1.0
    return go.Layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16, color="#18181b")),
        xaxis=dict(**_AXIS_STYLE, title="x₁",
                   range=[float(X[:, 0].min()) - pad, float(X[:, 0].max()) + pad]),
        yaxis=dict(**_AXIS_STYLE, title="x₂", scaleanchor="x",
                   range=[float(X[:, 1].min()) - pad, float(X[:, 1].max()) + pad]),
        legend=dict(orientation="v", x=1.02, y=1, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e4e4e7", borderwidth=1, font=dict(size=12)),
        margin=dict(l=40, r=190, t=60, b=40),
        height=560,
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_figure(
    snapshots: list[Snapshot],
    kernel: str,
    gamma: float,
    degree: int,
    coef0: float,
) -> go.Figure:
    """Animated figure with one frame per SMO sweep, plus a slider."""
    if not snapshots:
        raise ValueError("snapshot list is empty")

    frames: list[go.Frame] = []
    frame_labels: list[str] = []
    for i, snap in enumerate(snapshots):
        gx, gy, Z = _decision_grid(snap, kernel, gamma, degree, coef0)
        data = _surface_traces(gx, gy, Z) + _point_traces(snap)
        frames.append(go.Frame(data=data, name=str(i), layout=go.Layout(title_text=snap.title)))
        frame_labels.append(snap.title)

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=_layout(snapshots[0].title, snapshots[0].points),
    )
    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="Sweep: ", font=dict(size=13), xanchor="center"),
            pad=dict(t=50, b=10), len=0.9, x=0.05,
            steps=[
                dict(
                    args=[[str(i)], {"frame": {"duration": 300, "redraw": True},
                                      "mode": "immediate", "transition": {"duration": 150}}],
                    label=label[:36],
                    method="animate",
                )
                for i, label in enumerate(frame_labels)
            ],
        )],
    )
    return fig


def make_static_figure(
    snap: Snapshot,
    kernel: str,
    gamma: float,
    degree: int,
    coef0: float,
) -> go.Figure:
    """Single-frame figure for one snapshot — used by manual step-through."""
    gx, gy, Z = _decision_grid(snap, kernel, gamma, degree, coef0)
    data = _surface_traces(gx, gy, Z) + _point_traces(snap)
    return go.Figure(data=data, layout=_layout(snap.title, snap.points))


def make_loss_figure(snapshots: list[Snapshot], current_idx: int) -> go.Figure:
    """Dual objective curve (Σα − ½ αᵀ(yyᵀ⊙K)α) across sweeps.

    SMO is constructed so every accepted update increases this quantity (or
    leaves it unchanged) — it climbs monotonically toward its maximum. By
    strong duality that maximum equals the minimised primal hinge loss, so
    this rising curve is the mirror image of a gradient-descent loss curve
    falling toward its minimum.
    """
    xs = list(range(len(snapshots)))
    ys = [s.dual_objective for s in snapshots]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        line=dict(color=_PALETTE[0], width=2),
        marker=dict(size=5, color=_PALETTE[0]),
        name="Dual objective",
    ))
    fig.add_trace(go.Scatter(
        x=[current_idx], y=[ys[current_idx]], mode="markers",
        marker=dict(size=13, color=_PALETTE[3], symbol="circle",
                    line=dict(width=2, color="#18181b")),
        name="Current sweep", showlegend=False,
    ))
    fig.update_layout(
        font=dict(family=_FONT_FAMILY, size=12, color="#3f3f46"),
        height=220,
        margin=dict(l=40, r=20, t=25, b=35),
        xaxis=dict(title="Sweep", showgrid=True, gridcolor="#eef0f4", zeroline=False),
        yaxis=dict(title="Σα − ½αᵀQα", showgrid=True, gridcolor="#eef0f4"),
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def make_preview_figure(X: np.ndarray, y: np.ndarray) -> go.Figure:
    """Plain scatter of the raw dataset (no model yet) for a quick look."""
    traces = []
    for cls, colour, label in [(1, _POS_COLOR, "Class +1"), (-1, _NEG_COLOR, "Class −1")]:
        mask = y == cls
        traces.append(go.Scatter(
            x=X[mask, 0], y=X[mask, 1], mode="markers",
            marker=dict(color=colour, size=8, opacity=0.85,
                        line=dict(width=0.5, color="#18181b")),
            name=label,
        ))
    fig = go.Figure(data=traces, layout=_layout("Training data", X))
    return fig
