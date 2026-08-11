"""Shared UI building blocks for the algorithm-visualizer pages.

Every `apps/<algo>.py` page imports from here instead of hand-rolling its
own sidebar block, badge markup, or CSS. The global CSS string defined here
is injected exactly once, from Home.py (per CLAUDE.md: st.set_page_config
and global CSS may only live in Home.py — Home.py is the actual entry
script that Streamlit reruns on every navigation, st.navigation just
dispatches the selected page's code from inside that same run).
"""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
SPACE_XS = "8px"
SPACE_SM = "12px"
SPACE_MD = "16px"
SPACE_LG = "24px"
SPACE_XL = "32px"

MAX_CONTENT_WIDTH = "1500px"

# One accent per home-page category, drawn from the existing
# chartCategoricalColors palette (.streamlit/config.toml) so nothing new
# is invented outside the established design system.
CATEGORY_ACCENTS: dict[str, str] = {
    "Clustering": "#6366f1",
    "Dimensionality reduction": "#14b8a6",
    "Classification & ensembles": "#f59e0b",
    "Deep learning building blocks": "#f43f5e",
    "Generative & self-supervised models": "#8b5cf6",
    "Graph algorithms": "#0ea5e9",
    "Probabilistic methods, state estimation & signal processing": "#06b6d4",
    "Reinforcement learning": "#fb923c",
}
DEFAULT_ACCENT = "#6366f1"


def category_accent(category: str) -> str:
    return CATEGORY_ACCENTS.get(category, DEFAULT_ACCENT)


# ---------------------------------------------------------------------------
# Global CSS (injected once, from Home.py)
# ---------------------------------------------------------------------------
def global_css() -> str:
    return f"""
<style>
[data-testid="stAppDeployButton"] {{ display: none; }}

.block-container {{
    padding-top: 2rem;
    padding-bottom: {SPACE_XL};
    max-width: {MAX_CONTENT_WIDTH};
}}

[data-testid="stAlert"] {{ border-radius: 0.75rem; }}

/* ---- Vertical rhythm: consistent gaps between stacked blocks ---- */
[data-testid="stVerticalBlock"] {{ gap: {SPACE_SM}; }}

/* ---- Sidebar navigation: clear hierarchy between section labels and links ---- */
[data-testid="stSidebarNavLink"] [data-testid="stIconEmoji"] {{ display: none; }}

[data-testid="stNavSectionHeader"] {{
    margin-top: {SPACE_MD};
    padding-top: {SPACE_SM};
    border-top: 1px solid #e4e4e7;
}}
[data-testid="stSidebarNavItems"] > div:first-child [data-testid="stNavSectionHeader"] {{
    margin-top: 0;
    padding-top: 0;
    border-top: none;
}}
[data-testid="stNavSectionHeader"] p {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #a1a1aa;
    margin: 0;
}}
[data-testid="stNavSectionHeader"] [data-testid="stIconMaterial"] {{
    font-size: 0.9rem;
    color: #a1a1aa;
}}

[data-testid="stSidebarNavLink"] {{
    border-radius: 0.5rem;
    margin: 0.05rem 0;
    transition: background-color .1s ease;
}}
[data-testid="stSidebarNavLink"] p {{
    font-size: 0.875rem;
    color: #3f3f46;
    font-weight: 400;
}}
[data-testid="stSidebarNavLink"]:hover {{
    background-color: #eef0fe;
}}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
    background-color: #eef0fe;
}}
[data-testid="stSidebarNavLink"][aria-current="page"] p {{
    color: #6366f1;
    font-weight: 600;
}}

/* ---- Params rail: the bordered "Configuration" card every algorithm
   page renders in its narrow main-content column via params_rail() ---- */
.rail-title {{
    color: #52525b;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0 0 {SPACE_XS} 2px;
}}

/* ---- Badge / pill row: compact categorical values, replaces st.metric
   for short labels (e.g. "Phase: assign") that aren't really KPIs ---- */
.ui-badge-row {{
    display: flex;
    flex-wrap: wrap;
    gap: {SPACE_SM};
}}
.ui-badge {{
    display: inline-flex;
    align-items: baseline;
    gap: 6px;
    background: #f8f8fb;
    border: 1px solid #e4e4e7;
    border-radius: 999px;
    padding: 0.3rem 0.75rem;
    font-size: 0.85rem;
}}
.ui-badge-label {{
    color: #71717a;
    font-weight: 500;
}}
.ui-badge-value {{
    color: #18181b;
    font-weight: 700;
}}
</style>
"""


# ---------------------------------------------------------------------------
# Params rail — every algorithm page's parameter widgets render inside this
# ---------------------------------------------------------------------------
@contextmanager
def params_rail(col, title: str = "Configuration"):
    """Bordered card in `col` for an algorithm page's parameter widgets.

    Usage:
        col_params, col_main = st.columns([1, 3])
        with params_rail(col_params):
            shape_name = st.selectbox(...)
            ...
    """
    with col:
        st.markdown(f'<p class="rail-title">{title}</p>', unsafe_allow_html=True)
        with st.container(border=True):
            yield


# ---------------------------------------------------------------------------
# Badge row — compact pill display for short categorical metrics
# ---------------------------------------------------------------------------
def badge_row(items: list[tuple[str, str]]) -> None:
    """Render `items` (label, value) pairs as a row of pills.

    Use in place of `st.metric` columns when the values are short
    categorical strings (e.g. phase names) rather than genuine KPIs.
    """
    pills = "".join(
        f'<span class="ui-badge"><span class="ui-badge-label">{label}</span>'
        f'<span class="ui-badge-value">{value}</span></span>'
        for label, value in items
    )
    st.markdown(f'<div class="ui-badge-row">{pills}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# About section — plain-language context + references, distinct from the
# mechanics-only "How X works" expander every page already has. Placed at
# the top of col_main (first thing after the caption) so it's the first
# expander a visitor encounters, not buried below the interactive controls.
# ---------------------------------------------------------------------------
def about_section(why_it_matters: str, references: list[str]) -> None:
    """`why_it_matters` is one short paragraph of markdown: what the
    algorithm actually solves and why it's still used, tying back to what
    this specific page's visualization demonstrates — not another
    restatement of the step-by-step mechanics (that's what "How X works"
    is for). `references` is a list of markdown-formatted citation
    strings, oldest/most foundational first."""
    with st.expander("📚 About this algorithm", expanded=False):
        st.markdown(why_it_matters)
        st.markdown("**Further reading**")
        for ref in references:
            st.markdown(f"- {ref}")
