"""
visualize.py — Plotly dashboard for one Metropolis-Hastings snapshot.

make_figure() renders a 2x2 diagnostic dashboard, the same four views a
practitioner checks when tuning any MCMC sampler:

  top-left     target density (contours) + the chain's path so far;
               accepted steps trace an indigo line, a just-rejected proposal
               is shown as a dotted rose side-step
  top-right    trace plot — x-coordinate vs. iteration, the standard tool
               for eyeballing mixing speed at a glance
  bottom-left  histogram of visited x-coordinates vs. the true marginal
  bottom-right autocorrelation of the x-coordinate — slow decay means
               poor mixing and a low effective sample size
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from common.theme import PALETTE, apply_theme
from .algorithm import Snapshot, autocorrelation

_ACCEPT_COLOUR = PALETTE[0]  # indigo — the chain / accepted states
_REJECT_COLOUR = PALETTE[3]  # rose   — a rejected proposal
_TRUE_COLOUR = PALETTE[1]    # teal   — ground-truth marginal curve
_CONTOUR_COLOUR = PALETTE[0]  # indigo — target-density contour fill


def make_figure(
    snap: Snapshot,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    marginal: tuple[np.ndarray, np.ndarray],
    max_lag: int = 40,
) -> go.Figure:
    """Build the full diagnostic dashboard for a single snapshot.

    Parameters
    ----------
    snap:     the Snapshot to render (its `chain` field holds every state
              visited up to and including this step)
    grid:     (xs, ys, Z) from data.grid_density(target) — precompute once
              per target, the density surface never changes across steps
    marginal: (xs, px) from data.marginal_x(target) — also precomputed once
    max_lag:  largest lag shown in the autocorrelation panel
    """
    xs_grid, ys_grid, Z = grid
    xs_marg, px_marg = marginal
    chain = snap.chain
    extent = float(xs_grid[-1])

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Target density & chain path",
            "Trace plot (x vs. iteration)",
            "Marginal histogram of x",
            "Autocorrelation of x",
        ),
        horizontal_spacing=0.09,
        vertical_spacing=0.15,
    )

    # ---- Panel 1: target contour + chain path ------------------------------
    fig.add_trace(
        go.Contour(
            x=xs_grid,
            y=ys_grid,
            z=Z,
            showscale=False,
            colorscale=[[0.0, "#fbfbfd"], [1.0, _CONTOUR_COLOUR]],
            contours=dict(coloring="fill", showlines=False),
            opacity=0.75,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chain[:, 0],
            y=chain[:, 1],
            mode="lines+markers",
            line=dict(color=_ACCEPT_COLOUR, width=1),
            marker=dict(color=_ACCEPT_COLOUR, size=4, opacity=0.7),
            name="Chain (accepted states)",
        ),
        row=1,
        col=1,
    )
    if snap.proposed is not None and not snap.accepted:
        # The "road not taken": a dotted line from the current state to the
        # proposal that was just rejected.
        origin = snap.current
        fig.add_trace(
            go.Scatter(
                x=[origin[0], snap.proposed[0]],
                y=[origin[1], snap.proposed[1]],
                mode="lines+markers",
                line=dict(color=_REJECT_COLOUR, width=1.5, dash="dot"),
                marker=dict(color=_REJECT_COLOUR, size=[0, 10], symbol=["circle", "x"]),
                name="Rejected proposal",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=[snap.current[0]],
            y=[snap.current[1]],
            mode="markers",
            marker=dict(color=_ACCEPT_COLOUR, size=13, symbol="star",
                        line=dict(width=1, color="black")),
            name="Current state",
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(range=[-extent, extent], title_text="x", zeroline=False, row=1, col=1)
    fig.update_yaxes(range=[-extent, extent], title_text="y", zeroline=False,
                      scaleanchor="x", scaleratio=1, row=1, col=1)

    # ---- Panel 2: trace plot -------------------------------------------------
    iters = np.arange(len(chain))
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=chain[:, 0],
            mode="lines",
            line=dict(color=_ACCEPT_COLOUR, width=1),
            name="x-coordinate",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="iteration", row=1, col=2)
    fig.update_yaxes(title_text="x", range=[-extent, extent], zeroline=False, row=1, col=2)

    # ---- Panel 3: marginal histogram -----------------------------------------
    fig.add_trace(
        go.Histogram(
            x=chain[:, 0],
            histnorm="probability density",
            marker=dict(color=_ACCEPT_COLOUR, opacity=0.55),
            nbinsx=40,
            name="Sampled x",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=xs_marg,
            y=px_marg,
            mode="lines",
            line=dict(color=_TRUE_COLOUR, width=2.5),
            name="True marginal p(x)",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="x", range=[-extent, extent], row=2, col=1)
    fig.update_yaxes(title_text="density", row=2, col=1)

    # ---- Panel 4: autocorrelation ---------------------------------------------
    acf = autocorrelation(chain[:, 0], max_lag=max_lag)
    fig.add_trace(
        go.Bar(
            x=np.arange(len(acf)),
            y=acf,
            marker=dict(color=_ACCEPT_COLOUR),
            name="ACF(x)",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(title_text="lag", row=2, col=2)
    fig.update_yaxes(title_text="autocorrelation", range=[-1, 1.05], row=2, col=2)

    # ---- Shared axis / font styling (applied to every subplot) ---------------
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f4", gridwidth=1,
                      showline=True, linecolor="#e4e4e7", linewidth=1)
    fig.update_yaxes(showgrid=True, gridcolor="#eef0f4", gridwidth=1,
                      showline=True, linecolor="#e4e4e7", linewidth=1)

    # Title dropped — snap.title (e.g. "Step N — Accepted") is per-frame and
    # already shown verbatim in the page's progress bar text.
    apply_theme(fig, None, height=740, showlegend=True)
    return fig
