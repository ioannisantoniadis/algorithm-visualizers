# algorithm-visualizers — context for whoever picks this up next

## What this is

A portfolio piece: 20 classic ML/CS algorithms, each a **from-scratch NumPy
implementation** (no scikit-learn, no PyTorch in the core algorithm), paired
with an interactive **Streamlit + Plotly** step-by-step walkthrough. All 20
live in one multipage Streamlit app so the portfolio has a single home,
instead of 20 separate repos.

- **Live**: https://algorithm-visualizers.streamlit.app/ (Streamlit Community
  Cloud, auto-deploys on push to `main`)
- **Owner's GitHub profile links here**: `johnantonn/johnantonn` README, under
  "Selected GitHub repositories"

## How this repo came to exist (short version)

Each of the 20 algorithms started life as its own standalone repo
(`dbscan-viz`, `kmeans-viz`, `pca-viz`, ... `fft-viz`). Each one went through,
independently:

1. An algorithm-correctness review (several numerically verified against
   reference implementations or finite-difference gradient checks).
2. A live bug-hunt driving the actual running app with Playwright — this
   found and fixed real bugs: default parameters that visually contradicted
   the app's own pedagogical claims, unhandled crashes from unclamped
   frame-index state, a couple of `StreamlitDuplicateElementId` crashes (one
   of which broke an app on *every* default page load), dead click-to-draw
   features from a Plotly `hoverinfo="skip"` footgun, and one real algorithm
   bug (a UMAP SGD update that only touched one side of each edge).
3. A shared visual theme applied mechanically to all 20: indigo/teal palette,
   Inter font, bordered-card layout, matching Plotly chart theme (see
   "Design system" below for the exact values).

All 20 repos were then merged into **this** repo as a Streamlit multipage
app, and the 20 standalone repos were deleted from GitHub (they no longer
exist — this repo is the only copy of any of that code). A tooling repo
(`portfolio-builder`) that orchestrated the original build was also deleted
after the merge — so there is no other historical record beyond git history
in this repo and the summary above.

## Architecture

```
algorithm-visualizers/
├── Home.py                 # Entry point: st.set_page_config, global CSS,
│                            # the CATALOGUE dict, the home-page card grid,
│                            # and st.navigation() wiring (8 categories)
├── apps/<algo>.py          # One page per algorithm — a structural port of
│                            # that algorithm's original standalone app.py
├── <algo>/                 # Each algorithm's own from-scratch package
│   ├── algorithm.py         # Core algorithm, records a Snapshot per step
│   ├── data.py               # Synthetic dataset / environment generators
│   └── visualize.py          # Plotly figure builders
├── .streamlit/config.toml  # Shared theme — see below
└── pyproject.toml / requirements.txt
```

**Critical convention — session-state namespacing.** `st.session_state` is
shared across *every* page in one Streamlit session (that's how
`st.navigation` multipage apps work). Every page follows this pattern:

```python
NS = "dbscan"                          # short name, unique per page
def _k(name: str) -> str:
    return f"{NS}__{name}"

st.session_state[_k("step_idx")]        # never a bare "step_idx"
st.slider("...", key=_k("eps"))         # every widget gets an explicit key
```

If you add a new page or edit an existing one, **every** `st.session_state`
read/write and **every** widget's `key=` must go through `_k(...)`. Skipping
this is exactly how two algorithms end up silently sharing state.

**Other things centralized in `Home.py` and deliberately absent from every
`apps/<algo>.py` page:**
- `st.set_page_config(...)` — can only be called once per app, lives only in
  `Home.py`.
- The global CSS injection (hides the deploy button, tightens top padding,
  rounds alert boxes, styles the sidebar nav) — lives only in `Home.py`.
- The "ALGORITHM VISUALISER" eyebrow label above the title — was on every
  page originally; removed per-page as redundant once there's a shared home
  page and persistent sidebar nav.

## Design system (current)

`.streamlit/config.toml`:
- `primaryColor = "#6366f1"` (indigo)
- `backgroundColor = "#ffffff"`, `secondaryBackgroundColor = "#f8f8fb"`
- `textColor = "#18181b"`, `borderColor = "#e4e4e7"`
- `baseRadius = "medium"`
- `font = "Inter"` (via Google Fonts URL)
- `chartCategoricalColors` — a 10-color qualitative palette:
  `["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9", "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]`
  — every algorithm's Plotly figures pull from this exact list so charts
  match the app chrome.
- `client.toolbarMode = "minimal"` — hides Streamlit's own hamburger
  menu/deploy button chrome.

Per-page layout convention: sidebar parameter groups, the metrics row,
playback controls, and the chart are each wrapped in `st.container(border=True)`
("card" look). Every Plotly figure sets `config={"displayModeBar": False}`
and a `_base_layout()`-style helper applies Inter font, light gridlines, a
`#fbfbfd` plot background, and a transparent paper background.

## Where the design stands, and why a new session is starting on it

A first design pass (this session) fixed specific complaints on the **home
page and sidebar nav only**:
- Removed per-algorithm emoji icons (cards + sidebar nav) — deemed noisy.
- Made each home-page card fully clickable (hover lift + border highlight)
  instead of a small separate "Open visualiser" link. Implementation note:
  the `st.page_link` is stretched via absolutely-positioned CSS to cover the
  whole card and visually hidden (`opacity: 0`); this required `!important`
  because Streamlit sets its own `width`/`height` on `stElementContainer`
  that otherwise wins, and the scoping selector
  `div[data-testid="stVerticalBlock"]:has(> div [data-testid="stPageLink"])`
  is what keeps this from leaking onto the bordered containers used
  everywhere else in the app (metrics rows, parameter groups, etc.) — don't
  remove that `:has()` scoping without checking every other page still looks
  right.
- Restructured the sidebar nav: section labels (`[data-testid="stNavSectionHeader"]`)
  are now small/muted/uppercase with a divider; links
  (`[data-testid="stSidebarNavLink"]`) get a hover background and a bold
  indigo highlight when active (`[aria-current="page"]`).

**Despite that pass, the owner's assessment (going into the next session) is
that the UI is "still very problematic and not very stylish."** Read that as:
the home-page fixes above were real but narrow. The deeper issue is almost
certainly the **20 individual algorithm pages themselves** — their visual
design came from a mechanical "swap in the shared color theme + wrap things
in bordered containers" pass, not a genuine design pass. Nothing about
layout density, typography hierarchy, spacing rhythm, chart-to-chrome
balance, or overall "does this look like a crafted product" has been
seriously worked on for the 20 pages. That's almost certainly the next
session's real task, not just further home-page tweaks.

A few concrete things worth looking at with fresh eyes:
- Every page follows an identical top-to-bottom template (title → caption →
  metrics row → playback controls → chart → info box → expander). This is
  functional but monotonous across 20 pages — worth asking whether some
  variety/hierarchy would help, or whether the template itself needs a
  visual rework (not just recoloring).
- The metrics row (`st.metric` x3-4 in a bordered container) is the same
  shape on every page regardless of how meaningful those specific numbers
  are — worth reconsidering per-algorithm rather than templated.
- Sidebar parameter groups are plain sliders/selects in a bordered box; no
  real information hierarchy beyond that.
- Streamlit's own widget chrome (sliders, selects, buttons) is still visibly
  "default Streamlit" underneath the theme colors — a deeper CSS pass might
  be warranted if a more custom look is wanted, but see the gotchas below
  before doing broad CSS surgery.

## Hard-won gotchas (read before touching CSS or navigation)

- **Streamlit CSS specificity**: Streamlit's own emotion-generated styles on
  `stElementContainer` (and similar) often carry their own specificity/tie
  with `!important`-equivalent force via attribute-selector rules tied to
  HTML attributes like `width="fit-content"`. If a CSS override doesn't seem
  to take effect, check computed styles/geometry directly (don't assume);
  you likely need `!important` and/or a more specific selector.
- **`:has()` is used for scoping** page-specific CSS (e.g. Home.py's card
  styling) so it can't bleed onto other pages' bordered containers. Prefer
  this pattern over ad-hoc classes if you add more page-specific styling.
- **Widget-value reset on page navigation** is a real, observed, *harmless*
  quirk: navigating away from a page and back can reset some widgets to
  their script-default value. This is systemic to how `st.navigation`
  handles widget lifecycle, not a bug in any specific page. It does not
  cause crashes or stale/invalid frame indices. Don't spend time "fixing"
  it unless it starts actually breaking something (wrong data shown, crash).
- **`st.set_page_config` and the global CSS block belong only in `Home.py`.**
  Don't add them to `apps/<algo>.py` — Streamlit will error (config) or
  you'll just get duplicate `<style>` tags (CSS).
- Test any navigation-related change by actually clicking through the
  **real running multipage app** (`uv run streamlit run Home.py`), not by
  reading an individual `apps/<algo>.py` file in isolation — several bugs
  in this codebase's history only manifested inside the real nav (stale
  state across pages, duplicate element IDs from two charts on one page).

## Running locally

```bash
uv sync
uv run streamlit run Home.py
```

Opens at `http://localhost:8501` with the full categorized nav.
