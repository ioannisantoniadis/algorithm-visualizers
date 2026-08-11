"""Shared Plotly chart theming for the algorithm-visualizer pages.

Every ``<algo>/visualize.py`` should import from here instead of hand-
rolling its own layout helper. Before this module existed, 10 packages
each defined their own near-identical local ``_base_layout`` function and
the other 10 inlined the same boilerplate per-figure with no helper at
all — ~70 figure-builder functions total, all independently copy-pasted.

This module fixes two visual problems that copy-pasting made systemic:

- **Floated legends with a huge fixed right margin.** ~10 packages set
  ``legend=dict(x=1.02, ...)`` with ``margin=dict(r=140..190, ...)`` —
  reserving that much right-side space on *every* chart regardless of
  whether the legend needed it, which is what produced the "boxes within
  boxes" look (chart → card border → this floating legend box). The
  legend here instead sits horizontally *below* the plot and wraps onto
  more rows automatically for charts with many entries, so charts use
  their full card width.
- **Duplicate per-frame chart titles.** Nearly every figure set its
  Plotly title to the exact per-frame label (``snap.title``) that the
  page's own progress bar already shows above the chart. Pass
  ``title=None`` (the default) for any chart whose title would just be
  that per-frame label; pass an explicit string only for charts with a
  real, static caption (a loss curve, an importance chart, etc.) that
  isn't shown anywhere else on the page.
"""

from __future__ import annotations

import plotly.graph_objects as go

# Shared portfolio palette — kept in sync with .streamlit/config.toml's
# theme.chartCategoricalColors so every chart matches the app chrome.
PALETTE = [
    "#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9",
    "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899",
]
FONT_FAMILY = "Inter, -apple-system, Segoe UI, sans-serif"

# Target pixel gap between an x-axis title and the legend row below it.
# `legend.y` in Plotly is a *fraction of the plot area* (the axes' domain,
# i.e. `height - margin.t - margin.b`), not a fixed pixel offset — so a
# single fixed fraction (e.g. -0.16) gives a wildly different actual pixel
# gap on a 260px-tall chart vs. a 740px-tall one: too small a gap collides
# with the axis title's tick labels + title text, too large overshoots the
# margin entirely. Deriving the fraction from the actual plot-area height
# keeps this ~75px gap roughly constant across every figure size in the app.
_LEGEND_GAP_PX = 75


def cluster_colours(n: int) -> list[str]:
    """Return `n` hex colours, cycling the palette if needed."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def axis_style() -> dict:
    """Light gridlines + thin axis line, shared by every 2-D chart."""
    return dict(
        showgrid=True, gridcolor="#eef0f4", gridwidth=1,
        zeroline=False, showline=True, linecolor="#e4e4e7", linewidth=1,
    )


def _legend_dict(height: int, margin_t: int, margin_b: int, showlegend: bool) -> dict:
    """Horizontal legend below the plot, offset by a roughly constant
    pixel gap (see `_LEGEND_GAP_PX`) regardless of figure height — wraps
    onto more rows automatically for charts with many entries instead of
    reserving a large fixed right margin."""
    plot_area_height = max(height - margin_t - margin_b, 1)
    y = -_LEGEND_GAP_PX / plot_area_height if showlegend else 0
    return dict(
        orientation="h", yanchor="top", y=y, xanchor="center", x=0.5,
        bgcolor="rgba(0,0,0,0)", font=dict(size=11),
    )


def base_layout(
    title: str | None = None,
    *,
    height: int = 560,
    xaxis: dict | None = None,
    yaxis: dict | None = None,
    showlegend: bool = True,
    margin: dict | None = None,
) -> go.Layout:
    """Shared 2-D chart layout: Inter font, light gridlines, `#fbfbfd`
    plot background, transparent paper background, horizontal legend
    below the plot.

    `xaxis`/`yaxis` are merged over the shared `axis_style()` defaults —
    pass just the per-chart bits (title, range, scaleanchor, ...).
    Leave `title=None` (the default) when the chart's title would only
    repeat a per-frame label already shown elsewhere on the page (e.g. a
    progress bar) — pass a string only for a real, static caption.
    """
    m = dict(l=40, r=20, t=30 if title is None else 56, b=80 if showlegend else 40)
    if margin:
        m.update(margin)
    layout = go.Layout(
        font=dict(family=FONT_FAMILY, size=13, color="#3f3f46"),
        xaxis={**axis_style(), **(xaxis or {})},
        yaxis={**axis_style(), **(yaxis or {})},
        legend=_legend_dict(height, m["t"], m["b"], showlegend),
        showlegend=showlegend,
        margin=m,
        height=height,
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    if title:
        layout.title = dict(
            text=title, x=0.5, xanchor="center",
            font=dict(size=16, color="#18181b"),
        )
    return layout


def base_layout_3d(
    title: str | None = None,
    *,
    height: int = 560,
    scene: dict | None = None,
    showlegend: bool = True,
    margin: dict | None = None,
) -> go.Layout:
    """3-D counterpart of `base_layout` (used by pca's ℝ³ view)."""
    m = dict(l=10, r=10, t=30 if title is None else 56, b=10)
    if margin:
        m.update(margin)
    layout = go.Layout(
        font=dict(family=FONT_FAMILY, size=13, color="#3f3f46"),
        scene=scene or {},
        legend=_legend_dict(height, m["t"], m["b"], showlegend),
        showlegend=showlegend,
        margin=m,
        height=height,
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    if title:
        layout.title = dict(
            text=title, x=0.5, xanchor="center",
            font=dict(size=16, color="#18181b"),
        )
    return layout


def apply_theme(
    fig: go.Figure,
    title: str | None = None,
    *,
    height: int = 560,
    showlegend: bool = True,
    margin: dict | None = None,
) -> go.Figure:
    """Apply the shared font/background/legend/title theme to a figure
    already built with `plotly.subplots.make_subplots` (or any other
    figure whose per-axis config you don't want this module to own).
    Mutates and returns `fig`.

    Subplot figures commonly have their own per-panel x-axis title
    sharing the same bottom margin band as the legend, so this defaults
    to a taller bottom margin than `base_layout`'s single-panel default —
    override with e.g. `margin={"b": 130}` for an unusually tall figure.
    """
    m = dict(l=40, r=20, t=40 if title is None else 64, b=100 if showlegend else 40)
    if margin:
        m.update(margin)
    fig.update_layout(
        font=dict(family=FONT_FAMILY, size=13, color="#3f3f46"),
        legend=_legend_dict(height, m["t"], m["b"], showlegend),
        showlegend=showlegend,
        margin=m,
        height=height,
        plot_bgcolor="#fbfbfd",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    if title:
        fig.update_layout(title=dict(
            text=title, x=0.5, xanchor="center",
            font=dict(size=16, color="#18181b"),
        ))
    return fig
