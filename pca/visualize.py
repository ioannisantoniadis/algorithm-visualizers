"""
visualize.py — Plotly figures for PCA snapshots (2-D and 3-D)
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from common.theme import base_layout, base_layout_3d
from .algorithm import Snapshot

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
_PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
            "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]
_NOISE_COLOUR = "#a1a1aa"

# 95% Mahalanobis: χ² critical value at df = dimension
_MAHAL_SQ: dict[int, float] = {2: 5.991, 3: 7.815}


def _colours(labels: np.ndarray | None, n: int) -> list[str]:
    if labels is None:
        return [_PALETTE[0]] * n
    out: list[str] = []
    for lb in labels:
        if lb < 0:
            out.append(_NOISE_COLOUR)
        else:
            out.append(_PALETTE[lb % len(_PALETTE)])
    return out


def _axis_range_2d(*parts: np.ndarray, pad: float = 0.4) -> tuple[list[float], list[float]]:
    pts = np.vstack([p for p in parts if p is not None and np.size(p)])
    if pts.size == 0:
        return [-4.0, 4.0], [-4.0, 4.0]
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 0.5)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    r = span / 2 + pad
    return [cx - r, cx + r], [cy - r, cy + r]


def _axis_range_3d(*parts: np.ndarray, pad: float = 0.45) -> tuple[list[float], list[float], list[float]]:
    pts = np.vstack([p for p in parts if p is not None and np.size(p)])
    if pts.size == 0:
        return [-4.0, 4.0], [-4.0, 4.0], [-4.0, 4.0]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = max((hi - lo).max(), 0.5)
    c = (lo + hi) / 2
    r = span / 2 + pad
    return [c[0] - r, c[0] + r], [c[1] - r, c[1] + r], [c[2] - r, c[2] + r]


def _scene_3d(xr, yr, zr) -> dict:
    axis_style = dict(
        gridcolor="#eef0f4", zerolinecolor="#d4d4d8",
        backgroundcolor="rgba(251,251,253,0.6)", showbackground=True,
    )
    return dict(
        xaxis=dict(**axis_style, title="x₁", range=xr),
        yaxis=dict(**axis_style, title="x₂", range=yr),
        zaxis=dict(**axis_style, title="x₃", range=zr),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.55, y=1.35, z=1.15)),
    )


def _ellipse_path_2d(evals: np.ndarray, evecs: np.ndarray, mahal_sq: float) -> tuple[np.ndarray, np.ndarray]:
    radii = np.sqrt(np.maximum(evals, 1e-15) * mahal_sq)
    t = np.linspace(0, 2 * np.pi, 160)
    ring = np.vstack([np.cos(t), np.sin(t)])
    pts = (evecs @ np.diag(radii) @ ring).T
    return pts[:, 0], pts[:, 1]


def _ellipsoid_surface(evals: np.ndarray, evecs: np.ndarray, mahal_sq: float):
    radii = np.sqrt(np.maximum(evals, 1e-15) * mahal_sq)
    u = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    v = np.linspace(-np.pi / 2, np.pi / 2, 28)
    U, V = np.meshgrid(u, v, indexing="xy")
    y0 = radii[0] * np.cos(V) * np.cos(U)
    y1 = radii[1] * np.cos(V) * np.sin(U)
    y2 = radii[2] * np.sin(V)
    local = np.stack([y0, y1, y2], axis=-1)
    world = local @ evecs.T
    return world[:, :, 0], world[:, :, 1], world[:, :, 2]


def make_static_figure(snap: Snapshot, labels: np.ndarray | None = None) -> go.Figure:
    """Route to 2-D or 3-D figure."""
    if snap.ndim == 2:
        return make_static_figure_2d(snap, labels)
    return make_static_figure_3d(snap, labels)


def make_marginal_xy_figure(snap: Snapshot, labels: np.ndarray | None = None) -> go.Figure:
    """(x₁, x₂) marginal: looking down the x₃ axis.

    For **covariance**, the dashed ellipse uses the **2×2 marginal** of **S**
    (what you would estimate from (x₁, x₂) only).
    """
    if snap.ndim != 3:
        return go.Figure()
    X = snap.X
    n = X.shape[0]
    colours = _colours(labels, n)
    mahal2 = _MAHAL_SQ[2]

    if snap.phase in ("raw", "reconstruct"):
        xr, yr = _axis_range_2d(
            X[:, :2],
            snap.X_hat[:, :2] if snap.X_hat is not None else X[:, :2],
        )
    else:
        xr, yr = _axis_range_2d(snap.X_centered[:, :2])

    fig = go.Figure()
    pc_col = ("#f59e0b", "#8b5cf6", "#14b8a6")

    if snap.phase == "raw":
        fig.add_trace(
            go.Scatter(
                x=X[:, 0], y=X[:, 1], mode="markers",
                marker=dict(size=7, color=colours, opacity=0.82, line=dict(width=0.4, color="white")),
                name="Data (x₁,x₂)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[snap.mu[0]], y=[snap.mu[1]], mode="markers",
                marker=dict(symbol="cross", size=14, color="black", line=dict(width=2)),
                name="μ (x₁,x₂)",
            )
        )
    elif snap.phase == "centered":
        xc = snap.X_centered
        fig.add_trace(
            go.Scatter(
                x=xc[:, 0], y=xc[:, 1], mode="markers",
                marker=dict(size=7, color=colours, opacity=0.82, line=dict(width=0.4, color="white")),
                name="Centered (x₁,x₂)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0], y=[0.0], mode="markers",
                marker=dict(symbol="x", size=12, color="black", line=dict(width=2)),
                name="Origin",
            )
        )
    elif snap.phase == "covariance":
        xc = snap.X_centered
        S2 = snap.cov[:2, :2]
        ev2, V2 = np.linalg.eigh(S2)
        order = np.argsort(ev2)[::-1]
        ev2 = np.maximum(ev2[order], 1e-15)
        V2 = V2[:, order]
        ex, ey = _ellipse_path_2d(ev2, V2, mahal2)
        fig.add_trace(
            go.Scatter(
                x=xc[:, 0], y=xc[:, 1], mode="markers",
                marker=dict(size=7, color=colours, opacity=0.75, line=dict(width=0.4, color="white")),
                name="Centered (x₁,x₂)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ex, y=ey, mode="lines",
                line=dict(color="rgba(99,102,241,0.9)", width=2, dash="dash"),
                name="Marginal 95% ellipse (S₂₂)",
            )
        )
    elif snap.phase == "eigen":
        xc = snap.X_centered
        fig.add_trace(
            go.Scatter(
                x=xc[:, 0], y=xc[:, 1], mode="markers",
                marker=dict(size=7, color=colours, opacity=0.75, line=dict(width=0.4, color="white")),
                name="Centered (x₁,x₂)",
            )
        )
        evals, V = snap.evals, snap.evecs
        arrow_scale = 2.0 * np.sqrt(evals)
        for j in range(3):
            vx, vy = float(V[0, j]), float(V[1, j])
            L = float(arrow_scale[j])
            if (vx * vx + vy * vy) ** 0.5 < 1e-9:
                continue
            x1, y1 = vx * L, vy * L
            fig.add_trace(
                go.Scatter(
                    x=[0.0, x1], y=[0.0, y1], mode="lines",
                    line=dict(color=pc_col[j], width=4),
                    name=f"PC{j + 1} → (x₁,x₂) shadow",
                )
            )
    else:
        assert snap.X_hat is not None
        k = snap.k_components
        if k == 0:
            fig.add_trace(
                go.Scatter(
                    x=X[:, 0], y=X[:, 1], mode="markers",
                    marker=dict(size=6, color=colours, opacity=0.35, line=dict(width=0.3, color="white")),
                    name="True (x₁,x₂)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[snap.mu[0]], y=[snap.mu[1]], mode="markers",
                    marker=dict(symbol="star", size=22, color="#f59e0b", line=dict(width=2, color="black")),
                    name="Prediction",
                )
            )
        else:
            Xh = snap.X_hat
            fig.add_trace(
                go.Scatter(
                    x=X[:, 0], y=X[:, 1], mode="markers",
                    marker=dict(size=6, color=colours, opacity=0.35, line=dict(width=0.3, color="white")),
                    name="True (x₁,x₂)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=Xh[:, 0], y=Xh[:, 1], mode="markers",
                    marker=dict(size=8, color=colours, opacity=0.95, symbol="diamond",
                                line=dict(width=1, color="white")),
                    name=f"Recon k={k}",
                )
            )
            xs, ys = [], []
            for i in range(n):
                xs.extend([X[i, 0], Xh[i, 0], None])
                ys.extend([X[i, 1], Xh[i, 1], None])
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color="rgba(244,63,94,0.4)", width=1),
                    name="Residual (x₁,x₂)",
                )
            )

    fig.update_layout(base_layout(
        "Marginal (x₁, x₂) — x₃ is not shown",
        height=340,
        xaxis=dict(
            title="x₁", range=xr, scaleratio=1,
            zeroline=True, zerolinewidth=1, zerolinecolor="#d4d4d8",
        ),
        yaxis=dict(
            title="x₂", range=yr, scaleanchor="x", scaleratio=1,
            zeroline=True, zerolinewidth=1, zerolinecolor="#d4d4d8",
        ),
    ))
    return fig


def make_static_figure_2d(snap: Snapshot, labels: np.ndarray | None = None) -> go.Figure:
    fig = go.Figure()
    X = snap.X
    n = X.shape[0]
    colours = _colours(labels, n)
    mahal_sq = _MAHAL_SQ[2]

    if snap.phase in ("raw", "reconstruct"):
        xr, yr = _axis_range_2d(X, snap.X_hat if snap.X_hat is not None else X)
    else:
        xr, yr = _axis_range_2d(snap.X_centered)

    if snap.phase == "raw":
        fig.add_trace(
            go.Scatter(
                x=X[:, 0], y=X[:, 1], mode="markers",
                marker=dict(size=9, color=colours, opacity=0.82, line=dict(width=0.5, color="white")),
                name="Data",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[snap.mu[0]], y=[snap.mu[1]], mode="markers",
                marker=dict(symbol="cross", size=16, color="black", line=dict(width=2)),
                name="Mean μ",
            )
        )
    elif snap.phase == "centered":
        xc = snap.X_centered
        fig.add_trace(
            go.Scatter(
                x=xc[:, 0], y=xc[:, 1], mode="markers",
                marker=dict(size=9, color=colours, opacity=0.82, line=dict(width=0.5, color="white")),
                name="Centered data",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0], y=[0.0], mode="markers",
                marker=dict(symbol="x", size=14, color="black", line=dict(width=2)),
                name="Origin (mean)",
            )
        )
    elif snap.phase == "covariance":
        xc = snap.X_centered
        ex, ey = _ellipse_path_2d(snap.evals, snap.evecs, mahal_sq)
        fig.add_trace(
            go.Scatter(
                x=xc[:, 0], y=xc[:, 1], mode="markers",
                marker=dict(size=9, color=colours, opacity=0.75, line=dict(width=0.5, color="white")),
                name="Centered data",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ex, y=ey, mode="lines",
                line=dict(color="rgba(99,102,241,0.95)", width=2, dash="dash"),
                name=f"χ²₍₀.₉₅,₂₎ = {mahal_sq:.2f} contour",
            )
        )
    elif snap.phase == "eigen":
        xc = snap.X_centered
        fig.add_trace(
            go.Scatter(
                x=xc[:, 0], y=xc[:, 1], mode="markers",
                marker=dict(size=9, color=colours, opacity=0.75, line=dict(width=0.5, color="white")),
                name="Centered data",
            )
        )
        evals, V = snap.evals, snap.evecs
        arrow_scale = 2.0 * np.sqrt(evals)
        colors = ("#f59e0b", "#8b5cf6")
        for j in range(2):
            v = V[:, j]
            L = float(arrow_scale[j])
            x1, y1 = float(v[0] * L), float(v[1] * L)
            fig.add_trace(
                go.Scatter(
                    x=[0.0, x1], y=[0.0, y1], mode="lines",
                    line=dict(color=colors[j], width=5),
                    name=f"PC{j + 1} (2σ)",
                )
            )
    else:
        assert snap.X_hat is not None
        k = snap.k_components
        if k == 0:
            fig.add_trace(
                go.Scatter(
                    x=X[:, 0], y=X[:, 1], mode="markers",
                    marker=dict(size=7, color=colours, opacity=0.35, line=dict(width=0.3, color="white")),
                    name="True",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[snap.mu[0]], y=[snap.mu[1]], mode="markers",
                    marker=dict(symbol="star", size=28, color="#f59e0b", line=dict(width=2, color="black")),
                    name="Prediction (mean)",
                )
            )
        else:
            Xh = snap.X_hat
            fig.add_trace(
                go.Scatter(
                    x=X[:, 0], y=X[:, 1], mode="markers",
                    marker=dict(size=7, color=colours, opacity=0.35, line=dict(width=0.3, color="white")),
                    name="True",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=Xh[:, 0], y=Xh[:, 1], mode="markers",
                    marker=dict(size=9, color=colours, opacity=0.95, symbol="diamond",
                                line=dict(width=1, color="white")),
                    name=f"Reconstruction (k={k})",
                )
            )
            xs, ys = [], []
            for i in range(n):
                xs.extend([X[i, 0], Xh[i, 0], None])
                ys.extend([X[i, 1], Xh[i, 1], None])
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color="rgba(244,63,94,0.45)", width=1),
                    name="Residual",
                )
            )

    fig.update_layout(base_layout(
        None,  # per-frame title, already shown in the page's progress bar
        height=520,
        xaxis=dict(
            title="x₁", range=xr, scaleratio=1,
            zeroline=True, zerolinewidth=1, zerolinecolor="#d4d4d8",
        ),
        yaxis=dict(
            title="x₂", range=yr, scaleanchor="x", scaleratio=1,
            zeroline=True, zerolinewidth=1, zerolinecolor="#d4d4d8",
        ),
    ))
    return fig


def make_static_figure_3d(snap: Snapshot, labels: np.ndarray | None = None) -> go.Figure:
    fig = go.Figure()
    X = snap.X
    n = X.shape[0]
    colours = _colours(labels, n)
    mahal_sq = _MAHAL_SQ[3]

    if snap.phase in ("raw", "reconstruct"):
        xr, yr, zr = _axis_range_3d(X, snap.X_hat if snap.X_hat is not None else X)
    else:
        xr, yr, zr = _axis_range_3d(snap.X_centered)

    pc_col = ("#f59e0b", "#8b5cf6", "#14b8a6")

    if snap.phase == "raw":
        fig.add_trace(
            go.Scatter3d(
                x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="markers",
                marker=dict(size=5, color=colours, opacity=0.85, line=dict(width=0.3, color="white")),
                name="Data",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[snap.mu[0]], y=[snap.mu[1]], z=[snap.mu[2]], mode="markers",
                marker=dict(size=10, color="black", symbol="cross", line=dict(width=2)),
                name="Mean μ",
            )
        )
    elif snap.phase == "centered":
        xc = snap.X_centered
        fig.add_trace(
            go.Scatter3d(
                x=xc[:, 0], y=xc[:, 1], z=xc[:, 2], mode="markers",
                marker=dict(size=5, color=colours, opacity=0.85),
                name="Centered data",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0.0], y=[0.0], z=[0.0], mode="markers",
                marker=dict(size=8, color="black", symbol="x"),
                name="Origin",
            )
        )
    elif snap.phase == "covariance":
        xc = snap.X_centered
        sx, sy, sz = _ellipsoid_surface(snap.evals, snap.evecs, mahal_sq)
        fig.add_trace(
            go.Scatter3d(
                x=xc[:, 0], y=xc[:, 1], z=xc[:, 2], mode="markers",
                marker=dict(size=4, color=colours, opacity=0.75),
                name="Centered data",
            )
        )
        fig.add_trace(
            go.Surface(
                x=sx, y=sy, z=sz,
                opacity=0.22,
                showscale=False,
                colorscale=[[0, "rgba(99,102,241,0.5)"], [1, "rgba(99,102,241,0.65)"]],
                name=f"χ²₍₀.₉₅,₃₎ = {mahal_sq:.2f} surface",
                hoverinfo="skip",
            )
        )
    elif snap.phase == "eigen":
        xc = snap.X_centered
        fig.add_trace(
            go.Scatter3d(
                x=xc[:, 0], y=xc[:, 1], z=xc[:, 2], mode="markers",
                marker=dict(size=4, color=colours, opacity=0.75),
                name="Centered data",
            )
        )
        evals, V = snap.evals, snap.evecs
        arrow_scale = 2.0 * np.sqrt(evals)
        for j in range(3):
            v = V[:, j]
            L = float(arrow_scale[j])
            x1, y1, z1 = float(v[0] * L), float(v[1] * L), float(v[2] * L)
            fig.add_trace(
                go.Scatter3d(
                    x=[0.0, x1], y=[0.0, y1], z=[0.0, z1], mode="lines",
                    line=dict(color=pc_col[j], width=10),
                    name=f"PC{j + 1} (2σ)",
                )
            )
    else:
        assert snap.X_hat is not None
        k = snap.k_components
        if k == 0:
            fig.add_trace(
                go.Scatter3d(
                    x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="markers",
                    marker=dict(size=4, color=colours, opacity=0.35),
                    name="True",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[snap.mu[0]], y=[snap.mu[1]], z=[snap.mu[2]], mode="markers",
                    marker=dict(size=12, color="#f59e0b", symbol="diamond", line=dict(width=2, color="black")),
                    name="Prediction (mean)",
                )
            )
        else:
            Xh = snap.X_hat
            fig.add_trace(
                go.Scatter3d(
                    x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="markers",
                    marker=dict(size=4, color=colours, opacity=0.35),
                    name="True",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=Xh[:, 0], y=Xh[:, 1], z=Xh[:, 2], mode="markers",
                    marker=dict(size=5, color=colours, symbol="diamond", opacity=0.95),
                    name=f"Reconstruction (k={k})",
                )
            )
            xs, ys, zs = [], [], []
            for i in range(n):
                xs.extend([X[i, 0], Xh[i, 0], None])
                ys.extend([X[i, 1], Xh[i, 1], None])
                zs.extend([X[i, 2], Xh[i, 2], None])
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(color="rgba(244,63,94,0.35)", width=2),
                    name="Residual",
                    showlegend=True,
                )
            )

    fig.update_layout(base_layout_3d(
        None,  # per-frame title, already shown in the page's progress bar
        scene=_scene_3d(xr, yr, zr),
    ))
    return fig
