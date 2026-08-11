"""
visualize.py — Plotly figure builders

Grouped by tab:

  Signal & spectrum
    make_signal_figure(t, x, components)
    make_spectrum_figure(freqs, mag, true_freqs)
    make_reconstruction_figure(t, x, xk, k, n_bins)

  Butterfly diagram (the core algorithm visualiser)
    butterfly_edges(n_bits, rev)      — structural, computed once per N
    make_butterfly_figure(snap, edges, n_bits)

  Spectrogram
    make_spectrogram_figure(times, freqs, mags)

  Benchmark
    make_benchmark_figure(sizes, naive_times, fft_times)
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from common.theme import PALETTE as _PALETTE
from common.theme import base_layout
from .algorithm import Snapshot

_TRUE_FREQ_COLOUR = "#f43f5e"
_ORIGINAL_COLOUR = "#a1a1aa"


# ---------------------------------------------------------------------------
# Signal & spectrum
# ---------------------------------------------------------------------------

def make_signal_figure(
    t: np.ndarray,
    x: np.ndarray,
    components: list[np.ndarray] | None = None,
    title: str = "Time-domain signal",
) -> go.Figure:
    fig = go.Figure()

    if components:
        for i, comp in enumerate(components):
            fig.add_trace(
                go.Scatter(
                    x=t, y=comp, mode="lines",
                    line=dict(width=1, dash="dot", color=_PALETTE[(i + 1) % len(_PALETTE)]),
                    name=f"component {i + 1}",
                    opacity=0.65,
                )
            )

    fig.add_trace(
        go.Scatter(
            x=t, y=x, mode="lines",
            line=dict(width=2, color=_PALETTE[0]),
            name="x(t) sum" if components else "x(t)",
        )
    )

    fig.update_layout(base_layout(
        title, height=340,
        xaxis=dict(title="time (s)"),
        yaxis=dict(title="amplitude"),
    ))
    return fig


def make_spectrum_figure(
    freqs: np.ndarray,
    mag: np.ndarray,
    true_freqs: list[float] | None = None,
    title: str = "Magnitude spectrum |X(f)|",
) -> go.Figure:
    fig = go.Figure()

    # Stem plot: one vertical line per bin plus a marker at the tip —
    # closer to how a DFT output is usually drawn than a filled bar.
    stem_x, stem_y = [], []
    for f, m in zip(freqs, mag):
        stem_x += [f, f, None]
        stem_y += [0, m, None]
    fig.add_trace(
        go.Scatter(x=stem_x, y=stem_y, mode="lines", line=dict(color=_PALETTE[0], width=1.4),
                    showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=freqs, y=mag, mode="markers", marker=dict(color=_PALETTE[0], size=5),
            name="|X(f)|",
            hovertemplate="f=%{x:.2f} Hz<br>|X|=%{y:.3f}<extra></extra>",
        )
    )

    if true_freqs:
        for f in true_freqs:
            fig.add_vline(x=f, line=dict(color=_TRUE_FREQ_COLOUR, width=1, dash="dash"))
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=_TRUE_FREQ_COLOUR, width=1, dash="dash"),
                name="true frequency",
            )
        )

    fig.update_layout(base_layout(
        title, height=340,
        xaxis=dict(title="frequency (Hz)"),
        yaxis=dict(title="magnitude"),
    ))
    return fig


def make_reconstruction_figure(
    t: np.ndarray,
    x: np.ndarray,
    xk: np.ndarray,
    k: int,
    n_bins: int,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, mode="lines", line=dict(color=_ORIGINAL_COLOUR, width=2),
                              name="original signal"))
    fig.add_trace(go.Scatter(x=t, y=xk, mode="lines", line=dict(color=_PALETTE[3], width=2),
                              name=f"reconstructed from top {k} bin(s)"))

    rmse = float(np.sqrt(np.mean((x - xk) ** 2)))
    fig.update_layout(base_layout(
        f"Reconstruction from the {k} largest frequency bins (of {n_bins}) — RMSE {rmse:.3f}",
        height=340,
        xaxis=dict(title="time (s)"),
        yaxis=dict(title="amplitude"),
    ))
    return fig


# ---------------------------------------------------------------------------
# Butterfly diagram
# ---------------------------------------------------------------------------

def butterfly_edges(n_bits: int, rev: np.ndarray) -> list[dict]:
    """Precompute every edge of the butterfly network for N = 2**n_bits.

    This is purely structural (depends only on N, not on the data), so it
    is computed once per diagram size and reused across every snapshot's
    frame — only which edges are *highlighted* changes frame to frame.

    Column layout: 0 = natural order, 1 = bit-reversed order,
    2..n_bits+1 = after CT stage 1..n_bits.
    Each edge carries `stage` = the Snapshot.stage value at which it is
    "active" (0 for the bit-reversal permutation, s for stage-s butterflies).
    """
    n = 1 << n_bits
    edges: list[dict] = []

    for i in range(n):
        edges.append(dict(stage=0, x0=0, y0=i, x1=1, y1=int(rev[i])))

    for s in range(1, n_bits + 1):
        span = 1 << s
        half = span >> 1
        col_from, col_to = s, s + 1
        for k in range(0, n, span):
            for j in range(half):
                ie, io = k + j, k + j + half
                edges.append(dict(stage=s, x0=col_from, y0=ie, x1=col_to, y1=ie))
                edges.append(dict(stage=s, x0=col_from, y0=io, x1=col_to, y1=ie))
                edges.append(dict(stage=s, x0=col_from, y0=ie, x1=col_to, y1=io))
                edges.append(dict(stage=s, x0=col_from, y0=io, x1=col_to, y1=io))

    return edges


def make_butterfly_figure(snap: Snapshot, edges: list[dict], n_bits: int) -> go.Figure:
    """Draw the full butterfly network (faint) with the current stage's
    edges highlighted, and the array values at the current column overlaid
    as coloured, magnitude-sized nodes.
    """
    n = len(snap.values)
    col = snap.stage + 1  # column index for this snapshot

    fig = go.Figure()

    # Faint background: the whole network, for spatial context
    bx, by = [], []
    for e in edges:
        bx += [e["x0"], e["x1"], None]
        by += [e["y0"], e["y1"], None]
    fig.add_trace(go.Scatter(x=bx, y=by, mode="lines",
                              line=dict(color="rgba(160,160,175,0.30)", width=1),
                              hoverinfo="skip", showlegend=False))

    # Highlighted: edges feeding into the current column
    if snap.stage >= 0:
        ax, ay = [], []
        for e in edges:
            if e["stage"] == snap.stage:
                ax += [e["x0"], e["x1"], None]
                ay += [e["y0"], e["y1"], None]
        fig.add_trace(go.Scatter(x=ax, y=ay, mode="lines",
                                  line=dict(color=_PALETTE[3], width=2.4),
                                  hoverinfo="skip", showlegend=False,
                                  name="active butterflies"))

    # Faint node placeholders at every column, for structure
    for c in range(n_bits + 2):
        fig.add_trace(go.Scatter(x=[c] * n, y=list(range(n)), mode="markers",
                                  marker=dict(size=5, color="rgba(160,160,175,0.35)"),
                                  hoverinfo="skip", showlegend=False))

    # The actual computed values at the current column
    vals = snap.values
    mag = np.abs(vals)
    max_mag = float(mag.max()) if mag.max() > 0 else 1.0
    sizes = 9 + 16 * (mag / max_mag)

    # Colour = real part, ranged against the column's overall *magnitude*
    # (not the real part's own min/max). Pure-sine input (the default here)
    # has a spectrum that is almost entirely imaginary, so the real part is
    # ~0 everywhere but for floating-point noise (~1e-14) — letting Plotly
    # auto-range cmin/cmax off that noise floor zoomed the colorbar into a
    # meaningless femto-scale ("10f"/"-10f" tick labels) that looked broken.
    # Anchoring the range to max_mag instead keeps the scale physically
    # meaningful and correctly renders near-zero real parts as near-white.
    color_range = max(max_mag, 1e-9)

    fig.add_trace(
        go.Scatter(
            x=[col] * n, y=list(range(n)),
            mode="markers",
            marker=dict(
                size=sizes, color=vals.real, colorscale="RdBu",
                cmin=-color_range, cmax=color_range,
                line=dict(width=1, color="rgba(0,0,0,0.5)"),
                colorbar=dict(title="Re", len=0.6) if snap.stage == n_bits else None,
            ),
            text=[
                f"idx {i}<br>Re={v.real:.3f}  Im={v.imag:.3f}<br>|X|={m:.3f}"
                for i, (v, m) in enumerate(zip(vals, mag))
            ],
            hovertemplate="%{text}<extra></extra>",
            name="current values",
            showlegend=False,
        )
    )

    labels = ["Input", "Bit-rev"] + [f"Stage {s}" for s in range(1, n_bits + 1)]
    # Per-frame title (snap.title) dropped — this tab has its own progress
    # bar in apps/fft.py that already shows the equivalent step label.
    fig.update_layout(base_layout(
        None, height=520,
        xaxis=dict(
            tickmode="array", tickvals=list(range(n_bits + 2)), ticktext=labels,
            range=[-0.6, n_bits + 1.6], showgrid=False,
        ),
        yaxis=dict(title="array index", autorange="reversed", showgrid=False),
        margin=dict(l=50, r=30, t=60, b=50),
        showlegend=False,
    ))
    return fig


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------

def make_spectrogram_figure(times: np.ndarray, freqs: np.ndarray, mags: np.ndarray) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            x=times, y=freqs, z=mags,
            colorscale="Viridis",
            colorbar=dict(title="|X(f)|"),
        )
    )
    fig.update_layout(base_layout(
        "Spectrogram — |STFT(t, f)| via our own windowed FFT", height=420,
        xaxis=dict(title="time (s)"),
        yaxis=dict(title="frequency (Hz)"),
        showlegend=False,
    ))
    return fig


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def make_benchmark_figure(sizes: list[int], naive_times: list[float], fft_times: list[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=naive_times, mode="lines+markers",
                              name="naive DFT — O(N²)", line=dict(color=_PALETTE[3])))
    fig.add_trace(go.Scatter(x=sizes, y=fft_times, mode="lines+markers",
                              name="Cooley-Tukey FFT — O(N log N)", line=dict(color=_PALETTE[0])))

    fig.update_layout(base_layout(
        "Runtime vs N (log-log)", height=420,
        xaxis=dict(title="N (samples)", type="log"),
        yaxis=dict(title="time (s), averaged", type="log"),
    ))
    return fig
