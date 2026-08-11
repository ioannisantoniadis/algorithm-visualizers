"""
visualize.py — Plotly figure builders for the attention visualiser

Public functions
----------------
make_embedding_heatmap(tokens, X, blocks)      token embeddings, block-annotated
make_stage_figure(snapshot)                    dispatches on snapshot.phase/substep —
                                                the figure shown by the step-through /
                                                auto-play controls
make_attention_heatmap(weights, tokens)        rows=query, cols=key, cell=weight
make_multihead_figure(results, tokens)         small-multiples, one heatmap per head
make_explain_word_figure(results, tokens, i)   "who does token i attend to" bar chart

Chart titles: every figure used inside the per-frame walkthrough (Section 2
of apps/attention.py — `make_embedding_heatmap` when reused there, and
everything reachable through `make_stage_figure`) drops its Plotly title,
since that exact text is already shown in the page's own progress bar
(`st.progress(..., text=f"... {snapshots[step_idx].title}")`) and metrics
row. `make_multihead_figure` and `make_explain_word_figure` (Sections 3/4,
always visible, independent of playback) keep their static captions.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.theme import PALETTE, apply_theme, base_layout
from .algorithm import HeadResult, Snapshot
from .data import EmbeddingBlocks


def _dim_labels(n: int, prefix: str = "d") -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def make_matrix_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str | None,
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
    fig = go.Figure(data=[heat], layout=base_layout(title, height=420, showlegend=False))
    fig.update_yaxes(autorange="reversed")
    return fig


def make_embedding_heatmap(tokens: list[str], X: np.ndarray, blocks: EmbeddingBlocks) -> go.Figure:
    """Embedding matrix with the three feature blocks (role / entity /
    position) shaded so it's visible which columns carry which meaning.
    The block names are annotated directly on the chart below (see
    `fig.add_annotation` calls), so the title would only repeat them —
    dropped, since this figure is also reused as-is inside the per-frame
    walkthrough (Section 2) where a title would duplicate the progress bar.
    """
    fig = make_matrix_heatmap(
        X, tokens, _dim_labels(X.shape[1]),
        title=None,
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
    # Per-frame walkthrough figure (Section 2) — title dropped, duplicated
    # by the page's progress bar / metrics row (head name + stage).
    apply_theme(fig, None, height=420, showlegend=False)
    return fig


def make_attention_heatmap(weights: np.ndarray, tokens: list[str]) -> go.Figure:
    """The core deliverable: rows = query tokens, cols = key tokens, cell =
    attention weight (softmax output, rows sum to 1). Only reachable via the
    per-frame walkthrough (`make_stage_figure`'s "softmax" substep), so its
    title is always dropped — duplicated by the progress bar."""
    fig = make_matrix_heatmap(
        weights, tokens, tokens, title=None,
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

    Every branch drops its chart title: the page's progress bar already
    shows `snapshots[step_idx].title`, and the metrics row above the chart
    shows phase/head/stage, so a Plotly title here would only repeat them.
    """
    tokens = snap.tokens

    if snap.phase == "embed":
        # blocks aren't stored on the snapshot; caller passes them via
        # make_embedding_heatmap directly (see app.py) — fall back to a
        # plain heatmap here so this function stays total.
        return make_matrix_heatmap(
            snap.embeddings, tokens, _dim_labels(snap.embeddings.shape[1]),
            title=None, colorscale="RdBu", zmid=0, annotate=False,
        )

    if snap.phase == "combine":
        return make_matrix_heatmap(
            snap.context, tokens, _dim_labels(snap.context.shape[1]),
            title=None,
            colorscale="RdBu", zmid=0, annotate=False,
        )

    hr = snap.results[snap.head_idx]
    if snap.substep == "qkv":
        return make_qkv_figure(hr, tokens)
    if snap.substep == "scores":
        return make_matrix_heatmap(
            hr.scores_raw, tokens, tokens,
            title=None,
            colorscale="RdBu", zmid=0,
        )
    if snap.substep == "scale":
        return make_matrix_heatmap(
            hr.scores_scaled, tokens, tokens,
            title=None,
            colorscale="RdBu", zmid=0,
        )
    if snap.substep == "softmax":
        return make_attention_heatmap(hr.weights, tokens)
    if snap.substep == "output":
        return make_matrix_heatmap(
            hr.head_output, tokens, _dim_labels(hr.spec.d_k),
            title=None,
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
    # Static caption, always visible (Section 3), not shown anywhere else —
    # kept.
    apply_theme(
        fig, "Attention weights — every head, same sentence",
        height=420 if n <= 2 else 380, showlegend=False,
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
                marker_color=PALETTE[i % len(PALETTE)],
                hovertemplate="key=%{x}<br>weight=%{y:.3f}<extra>" + name + "</extra>",
            )
        )
    # Static caption, always visible (Section 4), not shown anywhere else —
    # kept (it's dynamic on the selected query token, but not per-frame).
    layout = base_layout(
        f"What does “{tokens[query_idx]}” attend to?",
        height=420,
        xaxis=dict(title="key token"),
        yaxis=dict(title="attention weight", range=[0, 1]),
    )
    fig.update_layout(layout)
    fig.update_layout(barmode="group")
    return fig
