# algorithm-visualizers — context for whoever picks this up next

## What this is

A portfolio piece: **21** classic ML/CS algorithms, each a **from-scratch
NumPy implementation** (no scikit-learn, no PyTorch in the core algorithm),
paired with an interactive **Streamlit + Plotly** step-by-step walkthrough.
All 21 live in one multipage Streamlit app so the portfolio has a single
home, instead of many separate repos.

- **Live**: https://algorithm-visualizers.streamlit.app/ (Streamlit Community
  Cloud, auto-deploys on push to `main`)
- **Owner's GitHub profile links here**: `johnantonn/johnantonn` README, under
  "Selected GitHub repositories"

> **Known stale copy** (not yet fixed — cheap to fix, just hasn't been
> touched): `Home.py`'s hero `st.title(...)` and the `st.caption(...)` right
> below it still hardcode **"20 classic algorithms"**, while the hero-meta
> line directly underneath computes the real count dynamically from
> `CATALOGUE` and correctly shows **"21 algorithms across 8 categories"** —
> i.e. the home page currently contradicts itself within a few lines.
> `README.md`'s intro line and `pyproject.toml`'s `description` also still
> say "20". All of this dates to the Particle Filter page being added
> (`a057990`) without a copy pass. Fix by either updating the three hardcoded
> "20"s to "21", or better, deriving the hero copy from `len(CATALOGUE)`
> the way the meta line already does.

## How this repo came to exist (short version)

The original 20 algorithms each started life as their own standalone repo
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
   Inter font, bordered-card layout, matching Plotly chart theme.

All 20 repos were then merged into **this** repo as a Streamlit multipage
app, and the 20 standalone repos were deleted from GitHub (they no longer
exist — this repo is the only copy of any of that code). A tooling repo
(`portfolio-builder`) that orchestrated the original build was also deleted
after the merge — so there is no other historical record beyond git history
in this repo and the summary above.

**Particle Filter (`particle_filter/`, `apps/particle_filter.py`) is the one
exception**: it was authored directly in this repo (commit `a057990`), not
ported from a standalone repo, specifically as a side-by-side companion to
the Kalman Filter page (same trajectory generator, same comparison-figure
pattern — see `particle_filter/data.py`'s `TRAJ_KEYS`/`TRAJ_NAMES` matching
`kalman`'s). It's a good current reference for "how a new page should look"
since it was written *after* the shared `common/` package and the
fragment-scoped playback convention (below) existed — every other page had
to be retrofitted to those conventions, this one was built with them.

## Architecture

```
algorithm-visualizers/
├── Home.py                 # Entry point: st.set_page_config, calls
│                            # common.ui.global_css(), the CATALOGUE dict,
│                            # the home-page card grid, and st.navigation()
│                            # wiring (8 categories, 21 algorithms)
├── common/                 # Shared code every page/package imports from
│   ├── ui.py                 # global_css(), params_rail(), badge_row(),
│   │                          # about_section(), category_accent(), design
│   │                          # tokens (SPACE_*, CATEGORY_ACCENTS)
│   └── theme.py               # base_layout()/base_layout_3d()/apply_theme(),
│                               # cluster_colours(), axis_style() — the one
│                               # Plotly layout helper every visualize.py uses
├── apps/<algo>.py          # One page per algorithm — a structural port of
│                            # that algorithm's original standalone app.py
│                            # (except particle_filter.py, see above)
├── <algo>/                 # Each algorithm's own from-scratch package
│   ├── algorithm.py         # Core algorithm, records a Snapshot per step
│   ├── data.py               # Synthetic dataset / environment generators
│   └── visualize.py          # Plotly figure builders — imports common.theme
├── .streamlit/config.toml  # Shared theme — see below
└── pyproject.toml / requirements.txt / uv.lock
```

21 algorithm packages: `kmeans`, `dbscan`, `gmmviz`, `pca`, `umapviz`,
`tsneviz`, `perceptron`, `svm`, `random_forest`, `backprop`, `attention`,
`vae`, `diffusion`, `contrastive`, `dijkstra`, `mst`, `mcmc`, `kalman`,
`particle_filter`, `fft`, `qlearning`. (Package directory names don't always
match the page title or the `apps/` filename 1:1 — e.g. `gmmviz/` ↔
`apps/gmm.py` ↔ "Gaussian Mixture (EM)", `tsneviz/` ↔ `apps/tsne.py`,
`umapviz/` ↔ `apps/umap.py`. Check `Home.py`'s `CATALOGUE` dict for the
authoritative module-path → title mapping, don't assume from the directory
name.) No `tests/` directory and no CI config exist in this repo — there is
no automated test suite; correctness was established via the one-time
reviews described above, not via a regression suite that runs on every push.

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

**Other things centralized and deliberately absent from every
`apps/<algo>.py` page:**
- `st.set_page_config(...)` — can only be called once per app, lives only in
  `Home.py`.
- The global CSS string — defined in `common/ui.py`'s `global_css()`, but
  **injected exactly once**, from `Home.py` (`st.markdown(global_css(), ...)`
  near the top). `Home.py` is the actual entry script Streamlit reruns on
  every navigation; `st.navigation`/`pg.run()` just dispatches the selected
  page's code from inside that same run — so calling `global_css()` a second
  time from an `apps/<algo>.py` page would just emit a duplicate `<style>`
  tag, not error, but don't do it; the helper existing in `common/` doesn't
  change the "call site count: one" rule.
- The "ALGORITHM VISUALISER" eyebrow label above the title — was on every
  page originally; removed per-page as redundant once there's a shared home
  page and persistent sidebar nav.

## The per-page template (as of the current design pass)

All 21 `apps/<algo>.py` pages now follow the same structural template — this
was a real, repo-wide pass (see git log: `6b93e9a`, `8f21f8a`, `fa4650a`,
`669aed0`), not just a shared color theme applied on top of 21 independently-
shaped pages. If you're adding a 22nd algorithm, `apps/particle_filter.py`
is the cleanest page to copy from — it was written natively in this shape.

1. `st.title(...)` + an `st.empty()` "caption slot" filled in later (after
   params are read, since the caption echoes current param values).
2. `col_params, col_main = st.columns([1, 3])` — a narrow left rail for
   controls, a wide right column for everything else.
3. Inside `col_params`: one or more `params_rail(col_params, "Section
   Title")` context managers (from `common/ui.py`) — each renders a small
   uppercase label + a bordered card, and widgets go inside. Below the
   rail(s): contextual `st.info`/`st.expander` notes about the current
   parameter choice (e.g. per-trajectory behavior notes).
4. A parameter tuple (e.g. `_param_key = (eps, min_samples, seed, ...)`) is
   compared against a cached copy in `st.session_state[_k("_param_key")]`;
   the algorithm only re-runs (regenerating data + re-fitting + resetting
   `step_idx`/`playing`) when that tuple changes or a "regenerate" button
   was pressed — not on every widget interaction/rerun.
5. Inside `col_main`, first thing: `about_section(why_it_matters,
   references)` from `common/ui.py` — a `"📚 About this algorithm"` expander
   with one paragraph of plain-language motivation + a references list. This
   is deliberately distinct from the mechanics-only "How X works"/"reading
   the animation" expander every page also has further down.
6. Then, **one `@st.fragment`-decorated inner function** (conventionally
   named `_playback()`) wrapping: the playback control row (◀ Prev / ▶ Play
   / ⏸ Pause / Next ▶ / speed selector), the progress bar, the metrics row,
   the chart(s), the phase-dependent `st.info`/`st.success` explanation, the
   "reading the animation" expander, and the auto-advance `time.sleep(DELAY)`
   + `st.rerun(scope="fragment")` logic at the very end. **This fragment
   scoping is load-bearing, not stylistic — see gotchas below.**
7. Metrics are usually `st.metric` x3–4 in a bordered `st.container(border=True)`.
   `common/ui.py` also exports `badge_row(items)` as a pill-row alternative
   for short *categorical* values (e.g. `"Phase: assign"`) that aren't real
   KPIs — it exists and is documented, but as of this writing **no page
   actually calls it yet**; `st.metric` is what's in use everywhere. Worth
   using `badge_row` next time a page's "metric" is really just a status
   string, rather than adding another one-off badge implementation.

## Design system (current)

`.streamlit/config.toml`:
- `primaryColor = "#6366f1"` (indigo)
- `backgroundColor = "#ffffff"`, `secondaryBackgroundColor = "#f8f8fb"`
- `textColor = "#18181b"`, `borderColor = "#e4e4e7"`
- `baseRadius = "medium"`
- `font = "Inter"` (via Google Fonts URL)
- `chartCategoricalColors` — a 10-color qualitative palette:
  `["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9", "#8b5cf6", "#84cc16", "#fb923c", "#06b6d4", "#ec4899"]`
  — kept in sync with `common/theme.py`'s `PALETTE` constant, so every
  algorithm's Plotly figures match the app chrome.
- `[theme.sidebar]` — sidebar gets its own `backgroundColor`/`borderColor`.
- `[client] toolbarMode = "minimal"` — hides Streamlit's own hamburger
  menu/deploy button chrome.

`common/theme.py` — **the** shared Plotly layout helper (added to replace
~10 near-identical hand-rolled `_base_layout` functions plus ~10 more pages
that inlined the same boilerplate per-figure with no helper at all — every
`<algo>/visualize.py` should import from here rather than reintroducing
either pattern):
- `base_layout(title=None, *, height=560, xaxis=None, yaxis=None,
  showlegend=True, margin=None)` — Inter font, light gridlines (`axis_style()`),
  `#fbfbfd` plot background, transparent paper background, horizontal legend
  below the plot (not floated right — see the module docstring for why the
  old floated-legend pattern was actively bad). Pass `title=None` (default)
  whenever the chart's title would just repeat a per-frame label already
  shown elsewhere on the page (e.g. the progress bar); pass a real string
  only for a static caption (a loss curve, an importance chart, etc.).
- `base_layout_3d(...)` — 3-D counterpart (used by `pca`'s ℝ³ view).
- `apply_theme(fig, ...)` — same theme applied via mutation, for figures
  already built with `plotly.subplots.make_subplots`.
- `cluster_colours(n)` — cycles `PALETTE` for `n` colors.
- Legend vertical offset is computed from actual plot-area height
  (`_LEGEND_GAP_PX = 75`), not a fixed fraction — a fixed fraction gives a
  wildly different pixel gap on a 260px chart vs. a 740px one.

`common/ui.py` — shared page chrome (see "per-page template" above for how
these get used):
- `global_css()` — returns the injected `<style>` block (deploy button
  hidden, content max-width `1500px`, sidebar nav section headers/links
  styled, `.rail-title`/`.ui-badge*`/`stMetricValue` rules — see gotchas
  below for *why* several of these need `!important`).
- `params_rail(col, title="Configuration")` — context manager: label + bordered card.
- `badge_row(items)` — pill row for short categorical values (defined, not yet used anywhere — see template step 7).
- `about_section(why_it_matters, references)` — the "📚 About this algorithm" expander.
- `category_accent(category)` / `CATEGORY_ACCENTS` — one accent hex per
  home-page category (drawn from the existing chart palette, nothing new
  invented), used for the home page's category header bars and card hover.
- Design tokens: `SPACE_XS/SM/MD/LG/XL`, `MAX_CONTENT_WIDTH`.

Every Plotly figure sets `config={"displayModeBar": False}` when rendered
via `st.plotly_chart`.

## Design pass history (what actually happened, in order)

A first pass fixed the home page and sidebar nav only (removing per-algorithm
emoji, making cards fully clickable, restyling sidebar nav sections/links).
At that point the 21 individual algorithm pages still had a purely
mechanical "swap in the shared color theme + wrap things in bordered
containers" look, and were flagged as the real remaining work.

That work then happened, across several commits:
- **`6b93e9a` "Redesign UI/UX: nav/params split, home page, chart theming,
  algorithm context"** — introduced the `common/` package itself
  (`ui.py`/`theme.py`), the `col_params`/`col_main` two-column split,
  `params_rail`, `about_section`, and centralized Plotly theming, and rolled
  all of it out to every page. This is the commit that turned "21
  independently-themed pages" into "one consistent template" per the section
  above.
- **`8f21f8a`** — rewrote the home-page hero copy to lead with what the app
  *shows* rather than its implementation detail (from-scratch NumPy).
- **`fa4650a` "Fix font-size inconsistencies: metric truncation, title
  hierarchy, CSS specificity"** — the `stMetricValue` truncation fix and the
  `.rail-title`/`.hero-*`/`.category-title` sizing rules now baked into
  `global_css()`/`Home.py` (see gotchas below).
- **`669aed0` "Fix autoplay flicker by scoping playback reruns to
  st.fragment"** — introduced the `@st.fragment`-wrapped `_playback()`
  pattern now used by all 21 pages (template step 6 above).
- **`a057990` "Add Particle Filter visualizer"** — the 21st page, built
  directly against all of the above conventions (see architecture section).

**Net effect**: the "still very problematic and not very stylish" state
described by an earlier version of this doc has had real, repo-wide work
done against it since — the per-page template, shared chart theme, and
playback pattern described above now apply uniformly. That does *not* mean
the design is necessarily finished or that the owner has signed off on the
current state — no feedback from a session after `a057990` is recorded here.
Before assuming more visual-design work is wanted, it's worth checking with
the owner what (if anything) still bothers them, rather than assuming the
old "needs a genuine design pass" framing still holds verbatim.

A few things that would still be worth a fresh look if more design work is
requested:
- `badge_row` exists but is unused (see template step 7) — either use it
  somewhere the metric row is really a status readout, or decide it's dead
  code.
- Streamlit's own widget chrome (sliders, selects, buttons) is still
  visibly "default Streamlit" underneath the theme colors — a deeper CSS
  pass might be warranted for a more custom look, but see the gotchas below
  first; this kind of broad CSS surgery is exactly where the specificity and
  `:has()`-scoping traps below have bitten before.

## Hard-won gotchas (read before touching CSS, playback, or navigation)

- **Playback/autoplay must stay inside one `@st.fragment`-decorated inner
  function.** Before `669aed0`, `st.rerun()` during autoplay reran the
  *entire* page (title, caption, params rail, about-section included)
  several times a second; small per-render timing differences showed up as
  visible flicker/layout shift on every frame. All 21 pages now wrap
  playback controls + progress bar + metrics + chart(s) + phase explanation
  + the auto-advance `time.sleep(DELAY)` in a single `@st.fragment` function
  and call `st.rerun(scope="fragment")` (never a bare `st.rerun()`) for
  every playback-triggered rerun — Prev/Next/Play/Pause buttons included,
  not just the auto-advance branch. If you add a page without this, expect
  the same flicker bug back.
- **Streamlit CSS specificity**: Streamlit's own emotion-generated styles on
  `stElementContainer` (and similar) often carry their own specificity/tie
  with `!important`-equivalent force via attribute-selector rules tied to
  HTML attributes like `width="fit-content"`. Concretely: a per-instance
  emotion-cache rule scoped to each `stMarkdown` targets `p`/`ol`/`ul`/`dl`/`li`
  with `font-size: inherit` at specificity `(0,1,1)`, which beats a plain
  class selector like `.rail-title` at `(0,1,0)` — so `.rail-title`,
  `.hero-eyebrow`, `.hero-meta`, etc. all need `!important` on `font-size` or
  they silently render at the inherited 16px body size. If a CSS override
  doesn't seem to take effect, check computed styles/geometry directly
  (don't assume); you likely need `!important` and/or a more specific
  selector.
- **`st.metric`'s value element** hard-sets `white-space: nowrap` +
  `text-overflow: ellipsis` internally (built for short numbers) — any
  multi-word phase name or `"a / b"` value silently truncates
  (`"Backward"` → `"Backwa…"`) unless overridden, which is why
  `global_css()` forces `white-space: normal` + `text-overflow: clip` on
  `[data-testid="stMetricValue"] > div`.
- **`:has()` is used for scoping** page-specific CSS (e.g. `Home.py`'s card
  styling) so it can't bleed onto other pages' bordered containers. The
  selector must be *exact*, not just "loose enough to work": `Home.py`
  originally used `div[data-testid="stVerticalBlock"]:has(> div
  [data-testid="stPageLink"])` (any descendant, any depth) — which also
  matched every ancestor further up the tree (the column wrapping each
  card, and the page's whole outermost content block), because "a
  div-shaped immediate child containing a page_link *somewhere* inside it"
  is true all the way up the DOM. That made the *entire home page* register
  as one giant hover/click target sharing the card's hover styling, with
  clicks still hitting the correctly-sized real targets underneath — a bug
  that's easy to miss visually (hover effects "worked", just on too much of
  the page) but is exactly the kind of thing `:has()` scoping can silently
  produce. The current selector requires the page_link to be the *immediate*
  child of an *immediate* `stElementContainer` child. Prefer this
  tightly-scoped `:has()` pattern over ad-hoc classes for any new
  page-specific styling, and don't loosen it without re-checking every
  other page.
- **Widget-value reset on page navigation** is a real, observed, *harmless*
  quirk: navigating away from a page and back can reset some widgets to
  their script-default value. This is systemic to how `st.navigation`
  handles widget lifecycle, not a bug in any specific page. It does not
  cause crashes or stale/invalid frame indices (every page clamps
  `step_idx` defensively on read, e.g. `step_idx = max(0, min(step_idx,
  n_frames - 1))`). Don't spend time "fixing" it unless it starts actually
  breaking something (wrong data shown, crash).
- **`st.set_page_config` belongs only in `Home.py`**, and `global_css()`
  must be *called* only from `Home.py` (it can live in `common/ui.py`, just
  don't invoke it a second time from an `apps/<algo>.py` page). Violating
  the first errors; violating the second just duplicates a `<style>` tag,
  but still don't do it.
- Test any navigation- or playback-related change by actually clicking
  through the **real running multipage app** (`uv run streamlit run
  Home.py`), not by reading an individual `apps/<algo>.py` file in
  isolation — several bugs in this codebase's history only manifested
  inside the real nav or real autoplay loop (stale state across pages,
  duplicate element IDs from two charts on one page, the flicker bug above).

## Running locally

```bash
uv sync
uv run streamlit run Home.py
```

Opens at `http://localhost:8501` with the full categorized nav.
