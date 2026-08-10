"""
visualize.py — Plotly figure builders for the attention visualiser

Public functions
----------------
make_embedding_heatmap(tokens, X, blocks)      token embeddings, block-annotated
make_stage_figure(snapshot)                    dispatches on snapshot.phase/substep —
                                                the figure shown by the step-through /
                                                auto-play controls
make_attention_heatmap(weights, tokens, title) rows=query, cols=key, cell=weight
make_multihead_figure(results, tokens)         small-multiples, one heatmap per head
make_explain_word_figure(results, tokens, i)   "who does token i attend to" bar chart
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .algorithm import HeadResult, Snapshot
from .data import EmbeddingBlocks

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every categorical chart (e.g. the
# "explain a word" bar chart, one colour per head) matches the app chrome.
# Attention heatmaps intentionally use sequential/diverging colorscales
# instead (Blues / RdBu / Viridis) since cell value, not category identity,
# is what needs to read clearly there.
_PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
            "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]
_FONT_FAMILY = "Inter, -apple-system, Segoe UI, sans-serif"
_PLOT_BG = "#fbfbfd"
_PAPER_BG = "rgba(0,0,0,0)"
_GRID_COLOR = "#eef0f4"
_LINE_COLOR = "#e4e4e7"

_AXIS_STYLE = dict(
    showgrid=True, gridcolor=_GRID_COLOR, gridwidth=1,
    zeroline=False, showline=True, linecolor=_LINE_COLOR, linewidth=1,
)


def _dim_labels(n: int, prefix: str = "d") -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def _base_layout(title: str, height: int = 420) -> dict:
    return dict(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text=title, x=0.5, xanchor="center", y=0.97, yanchor="top",
                    font=dict(size=15, color="#18181b")),
        margin=dict(l=60, r=30, t=55, b=45),
        height=height,
        plot_bgcolor=_PLOT_BG,
        paper_bgcolor=_PAPER_BG,
    )


def make_matrix_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorscale: str = "Viridis",
    zmid: float | None = None,
    annotate: bool = True,
) -> go.Figure:
    """Generic annotated heatmap: rows/cols labelled, cells coloured by value."""
    z = np.asarray(matrix)
    text = np.round(z, 2).astype(str)

    heat = go.Heatmap(
        z=z,
        x=col_labels,
        y=row_labels,
        colorscale=colorscale,
        zmid=zmid,
        text=text if annotate else None,
        texttemplate="%{text}" if annotate else None,
        textfont=dict(size=10),
        colorbar=dict(thickness=14),
        hovertemplate="row=%{y}<br>col=%{x}<br>value=%{z:.3f}<extra></extra>",
    )
    fig = go.Figure(data=[heat], layout=go.Layout(**_base_layout(title)))
    fig.update_yaxes(autorange="reversed")
    return fig


def make_embedding_heatmap(tokens: list[str], X: np.ndarray, blocks: EmbeddingBlocks) -> go.Figure:
    """Embedding matrix with the three feature blocks (role / entity /
    position) shaded so it's visible which columns carry which meaning.
    """
    fig = make_matrix_heatmap(
        X, tokens, _dim_labels(X.shape[1]),
        title="Token embeddings — role | entity | position blocks",
        colorscale="RdBu", zmid=0, annotate=False,
    )
    # Vertical separators + labels between the three blocks
    boundaries = [blocks.role[1], blocks.entity[1]]
    names = ["role", "entity", "position"]
    starts = [blocks.role[0], blocks.entity[0], blocks.position[0]]
    ends = [blocks.role[1], blocks.entity[1], blocks.position[1]]
    for b in boundaries:
        fig.add_vline(x=b - 0.5, line=dict(color="black", width=2))
    for name, s, e in zip(names, starts, ends):
        fig.add_annotation(
            x=(s + e - 1) / 2, y=-0.9, text=name, showarrow=False,
            font=dict(size=12, color="#333"), yref="y",
        )
    fig.update_layout(height=380)
    return fig


def make_qkv_figure(hr: HeadResult, tokens: list[str]) -> go.Figure:
    """Three side-by-side heatmaps for a head's Q, K, V matrices."""
    fig = make_subplots(
        rows=1, cols=3, subplot_titles=["Q = X·Wq", "K = X·Wk", "V = X·Wv"],
        horizontal_spacing=0.08,
    )
    dim_labels = _dim_labels(hr.spec.d_k)
    for col, (mat, name) in enumerate([(hr.Q, "Q"), (hr.K, "K"), (hr.V, "V")], start=1):
        fig.add_trace(
            go.Heatmap(
                z=mat, x=dim_labels, y=tokens, colorscale="RdBu", zmid=0,
                showscale=(col == 3), coloraxis=None,
                hovertemplate=f"{name}[%{{y}}, %{{x}}] = %{{z:.3f}}<extra></extra>",
            ),
            row=1, col=col,
        )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(text=f"{hr.name} head — Query / Key / Value projections",
                    x=0.5, xanchor="center", y=0.97, yanchor="top",
                    font=dict(size=15, color="#18181b")),
        height=420, plot_bgcolor=_PLOT_BG, paper_bgcolor=_PAPER_BG,
        margin=dict(l=50, r=30, t=70, b=40),
    )
    return fig


def make_attention_heatmap(weights: np.ndarray, tokens: list[str], title: str) -> go.Figure:
    """The core deliverable: rows = query tokens, cols = key tokens, cell =
    attention weight (softmax output, rows sum to 1)."""
    fig = make_matrix_heatmap(
        weights, tokens, tokens, title=title,
        colorscale="Blues", zmid=None, annotate=True,
    )
    fig.update_layout(
        xaxis=dict(title="attends to (key)"),
        yaxis=dict(title="query token"),
    )
    return fig


def make_stage_figure(snap: Snapshot) -> go.Figure:
    """Dispatch on the snapshot's phase/substep to render exactly the
    matrix that stage of the pipeline is about. Used by both the manual
    step-through and the auto-play loop in app.py.
    """
    tokens = snap.tokens

    if snap.phase == "embed":
        # blocks aren't stored on the snapshot; caller passes them via
        # make_embedding_heatmap directly (see app.py) — fall back to a
        # plain heatmap here so this function stays total.
        return make_matrix_heatmap(
            snap.embeddings, tokens, _dim_labels(snap.embeddings.shape[1]),
            title="Token embeddings", colorscale="RdBu", zmid=0, annotate=False,
        )

    if snap.phase == "combine":
        return make_matrix_heatmap(
            snap.context, tokens, _dim_labels(snap.context.shape[1]),
            title="Final context vectors (embedding + Σ head outputs)",
            colorscale="RdBu", zmid=0, annotate=False,
        )

    hr = snap.results[snap.head_idx]
    if snap.substep == "qkv":
        return make_qkv_figure(hr, tokens)
    if snap.substep == "scores":
        return make_matrix_heatmap(
            hr.scores_raw, tokens, tokens,
            title=f"{hr.name} head — raw scores QKᵀ",
            colorscale="RdBu", zmid=0,
        )
    if snap.substep == "scale":
        return make_matrix_heatmap(
            hr.scores_scaled, tokens, tokens,
            title=f"{hr.name} head — scaled scores (÷√{hr.spec.d_k})",
            colorscale="RdBu", zmid=0,
        )
    if snap.substep == "softmax":
        return make_attention_heatmap(
            hr.weights, tokens, title=f"{hr.name} head — attention weights"
        )
    if snap.substep == "output":
        return make_matrix_heatmap(
            hr.head_output, tokens, _dim_labels(hr.spec.d_k),
            title=f"{hr.name} head — output = weights · V",
            colorscale="RdBu", zmid=0,
        )
    raise ValueError(f"Unhandled snapshot: phase={snap.phase!r} substep={snap.substep!r}")


def make_multihead_figure(results: list[HeadResult], tokens: list[str]) -> go.Figure:
    """Small-multiples: one attention heatmap per head, side by side, so you
    can see each head attending to a different part of the sentence at a
    glance."""
    n = len(results)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[r.name for r in results],
        horizontal_spacing=0.06,
    )
    for col, r in enumerate(results, start=1):
        fig.add_trace(
            go.Heatmap(
                z=r.weights, x=tokens, y=tokens, colorscale="Blues",
                showscale=(col == n), zmin=0, zmax=1,
                hovertemplate="query=%{y}<br>key=%{x}<br>weight=%{z:.2f}<extra></extra>",
            ),
            row=1, col=col,
        )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        height=420 if n <= 2 else 380,
        plot_bgcolor=_PLOT_BG, paper_bgcolor=_PAPER_BG,
        margin=dict(l=50, r=30, t=60, b=60),
        title=dict(text="Attention weights — every head, same sentence",
                    x=0.5, xanchor="center", y=0.97, yanchor="top",
                    font=dict(size=15, color="#18181b")),
    )
    return fig


def make_explain_word_figure(
    weights_by_head: list[tuple[str, np.ndarray]],
    tokens: list[str],
    query_idx: int,
) -> go.Figure:
    """'Explain a word' — grouped bar chart of how much the chosen query
    token attends to every other token. `weights_by_head` is a list of
    (head_name, weights_matrix) pairs — one bar group per entry, so the
    caller can pass individual heads or a pre-averaged combination."""
    fig = go.Figure()
    for i, (name, weights) in enumerate(weights_by_head):
        fig.add_trace(
            go.Bar(
                name=name,
                x=tokens,
                y=weights[query_idx],
                marker_color=_PALETTE[i % len(_PALETTE)],
                hovertemplate="key=%{x}<br>weight=%{y:.3f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(
        font=dict(family=_FONT_FAMILY, size=13, color="#3f3f46"),
        title=dict(
            text=f"What does “{tokens[query_idx]}” attend to?",
            x=0.5, xanchor="center", y=0.97, yanchor="top",
            font=dict(size=15, color="#18181b"),
        ),
        barmode="group",
        xaxis=dict(title="key token", **_AXIS_STYLE),
        yaxis=dict(title="attention weight", range=[0, 1], **_AXIS_STYLE),
        height=420,
        plot_bgcolor=_PLOT_BG, paper_bgcolor=_PAPER_BG,
        margin=dict(l=50, r=20, t=55, b=90),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.28, yanchor="top",
                     bgcolor="rgba(255,255,255,0.9)", bordercolor=_LINE_COLOR,
                     borderwidth=1, font=dict(size=12)),
    )
    return fig
