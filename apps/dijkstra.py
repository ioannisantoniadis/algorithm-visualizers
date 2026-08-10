"""Dijkstra / A* page — adapted from the standalone dijkstra-viz repo for the
multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from dijkstra.algorithm import search
from dijkstra.data import PRESET_KEYS, PRESET_NAMES, make_grid
from dijkstra.visualize import make_editor_figure, make_static_figure

NS = "dijkstra"


def _k(name: str) -> str:
    return f"{NS}__{name}"


# ---------------------------------------------------------------------------
# Sidebar — grid source
# ---------------------------------------------------------------------------
st.sidebar.markdown("##### 🧭 GRID & SEARCH")

with st.sidebar.container(border=True):
    mode = st.radio(
        "Grid source",
        ["Preset", "Draw your own"],
        help="Preset generates and runs immediately. Draw your own starts from "
             "a blank canvas that you paint by hand before running.",
        key=_k("mode"),
    )

    preset_name = st.selectbox("Preset", PRESET_NAMES, index=2, disabled=(mode != "Preset"),
        help="Only applies when Grid source is Preset.", key=_k("preset_name"))
    preset_key = PRESET_KEYS[PRESET_NAMES.index(preset_name)]

    density = st.slider("Obstacle density", 0.05, 0.5, 0.25, step=0.05,
        disabled=not (mode == "Preset" and preset_key == "random_obstacles"),
        help="Only applies to the Random obstacles preset.", key=_k("density"))

    rows = st.slider("Rows", min_value=11, max_value=41, value=21, step=2, key=_k("rows"))
    cols = st.slider("Columns", min_value=15, max_value=51, value=31, step=2, key=_k("cols"))
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key=_k("seed"))

    regenerate = st.button(
        "🔄 Regenerate grid" if mode == "Preset" else "🔄 Clear grid",
        use_container_width=True,
        key=_k("regenerate"),
    )

# ---------------------------------------------------------------------------
# Sidebar — algorithm
# ---------------------------------------------------------------------------
st.sidebar.markdown("##### ALGORITHM")

with st.sidebar.container(border=True):
    algorithm_name = st.radio("Search algorithm", ["Dijkstra", "A*"], key=_k("algorithm_name"))
    algorithm_key = "astar" if algorithm_name == "A*" else "dijkstra"

    heuristic_name = st.selectbox(
        "Heuristic (A* only)",
        ["Manhattan", "Euclidean"],
        disabled=(algorithm_key == "dijkstra"),
        help="Scaled by the cheapest cell cost in the grid so it never "
             "overestimates — that's what keeps A* optimal.",
        key=_k("heuristic_name"),
    )
    heuristic_key = "euclidean" if heuristic_name == "Euclidean" else "manhattan"

_PRESET_NOTES = {
    "open": "**Open field** — no obstacles. With start and goal at opposite "
            "corners, *every* cell lies on some shortest path, so Dijkstra "
            "and A* actually visit the same number of nodes here — the "
            "heuristic only pays off once there's something to route around.",
    "random_obstacles": "**Random obstacles** — scattered impassable blocks. "
            "Watch A* hug the straight line to the goal far more closely "
            "than Dijkstra, which explores evenly in every direction.",
    "maze": "**Maze** — a perfect maze has exactly one true path, so dead "
            "ends look identical to the real corridor until fully explored. "
            "This is the clearest demo of how much the A* heuristic prunes "
            "compared to undirected Dijkstra.",
    "weighted_terrain": "**Weighted terrain** — patches of costly mud/water. "
            "The cheapest path is not always the shortest one in steps — "
            "that distinction is the whole point of Dijkstra over plain BFS.",
}
if mode == "Preset":
    st.sidebar.info(_PRESET_NOTES[preset_key])
else:
    st.sidebar.info(
        "**Draw your own** — pick a brush below the grid to add walls, "
        "erase them, or move the start (S) / goal (G), then press "
        "**▶ Run search**."
    )

with st.sidebar.expander("How the search works", expanded=False):
    st.markdown(
        """
1. Start at cost 0; every other cell is "unknown" (∞)
2. Pop the cheapest node from the priority queue, **finalise** it
3. **Relax** its neighbours — if reaching them via this node is cheaper,
   update their cost and push them into the queue
4. Repeat until the goal is finalised (or the queue runs empty)

A* adds a heuristic estimate of the remaining distance to each node's
priority, so it tries goal-ward nodes first without ever giving up
Dijkstra's optimality guarantee.
"""
    )

# ---------------------------------------------------------------------------
# Session state — grid generation / regeneration
# ---------------------------------------------------------------------------
FRONTIER_LIMIT = 10


def _run_search() -> None:
    grid = st.session_state[_k("grid_cost")]
    start = st.session_state[_k("start")]
    goal = st.session_state[_k("goal")]

    snapshots = search(
        grid, start, goal,
        algorithm=algorithm_key,
        heuristic_mode=heuristic_key,
        frontier_limit=FRONTIER_LIMIT,
    )
    st.session_state[_k("snapshots")] = snapshots
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False

    # Quietly run the *other* algorithm too, purely for the comparison
    # metric below — this is the "clean lesson in heuristic search" payoff.
    other_key = "astar" if algorithm_key == "dijkstra" else "dijkstra"
    other_snapshots = search(
        grid, start, goal,
        algorithm=other_key,
        heuristic_mode=heuristic_key,
        frontier_limit=1,
    )
    st.session_state[_k("compare")] = {
        algorithm_key: snapshots[-1].step,
        other_key: other_snapshots[-1].step,
    }


_param_key = (mode, preset_key, rows, cols, int(seed), density)
_search_key = (algorithm_key, heuristic_key)

grid_changed = (
    _k("grid_cost") not in st.session_state
    or regenerate
    or st.session_state.get(_k("_param_key")) != _param_key
)

if grid_changed:
    if mode == "Preset":
        grid, start, goal = make_grid(preset_key, rows, cols, seed=int(seed), density=density)
    else:
        grid, start, goal = make_grid("open", rows, cols)
    st.session_state[_k("grid_cost")] = grid
    st.session_state[_k("start")] = start
    st.session_state[_k("goal")] = goal
    st.session_state[_k("snapshots")] = None
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False
    st.session_state[_k("_param_key")] = _param_key
    if mode == "Preset":
        _run_search()
    st.session_state[_k("_search_key")] = _search_key
elif (
    st.session_state.get(_k("_search_key")) != _search_key
    and st.session_state.get(_k("snapshots")) is not None
):
    _run_search()
    st.session_state[_k("_search_key")] = _search_key

grid_cost: np.ndarray = st.session_state[_k("grid_cost")]
start = st.session_state[_k("start")]
goal = st.session_state[_k("goal")]
actual_rows, actual_cols = grid_cost.shape

# ---------------------------------------------------------------------------
# Main area header
# ---------------------------------------------------------------------------
st.title("Dijkstra & A* — Shortest Path on a Grid")

st.caption(
    f"Grid: **{actual_rows}×{actual_cols}** | "
    f"Algorithm: **{algorithm_name}** | "
    + (f"Heuristic: **{heuristic_name}** | " if algorithm_key == "astar" else "")
    + f"Start: **{start}** | Goal: **{goal}**"
)


def _apply_brush(r: int, c: int, brush: str) -> bool:
    """Apply `brush` to cell (r, c) in-place. Returns True only if the
    grid actually changed, so a persisted (already-processed) Plotly
    click selection doesn't re-trigger an endless rerun loop."""
    grid = st.session_state[_k("grid_cost")]
    cur_start = st.session_state[_k("start")]
    cur_goal = st.session_state[_k("goal")]
    rows_, cols_ = grid.shape
    if not (0 <= r < rows_ and 0 <= c < cols_):
        return False
    cell = (r, c)

    if brush == "Wall":
        if cell in (cur_start, cur_goal) or not np.isfinite(grid[cell]):
            return False
        grid[cell] = np.inf
        return True
    if brush == "Erase":
        if np.isfinite(grid[cell]):
            return False
        grid[cell] = 1.0
        return True
    if brush == "Move start":
        if cell in (cur_start, cur_goal):
            return False
        grid[cell] = 1.0
        st.session_state[_k("start")] = cell
        return True
    if brush == "Move goal":
        if cell in (cur_start, cur_goal):
            return False
        grid[cell] = 1.0
        st.session_state[_k("goal")] = cell
        return True
    return False


# ---------------------------------------------------------------------------
# EDIT GATE — shown until a search has been run on the current grid
# ---------------------------------------------------------------------------
snapshots = st.session_state.get(_k("snapshots"))

if snapshots is None:
    st.info(
        "Pick a brush, click cells on the grid to edit it, then press "
        "**▶ Run search**."
    )

    with st.container(border=True):
        bcol1, bcol2, bcol3 = st.columns([2, 2, 4])
        with bcol1:
            brush = st.radio(
                "Brush", ["Wall", "Erase", "Move start", "Move goal"],
                horizontal=False, label_visibility="collapsed",
                key=_k("brush"),
            )
        with bcol2:
            if st.button("🗑 Clear walls", use_container_width=True, key=_k("clear_walls")):
                st.session_state[_k("grid_cost")] = np.where(
                    np.isfinite(st.session_state[_k("grid_cost")]), st.session_state[_k("grid_cost")], 1.0
                )
                st.rerun()
            run_pressed = st.button("▶ Run search", use_container_width=True, type="primary", key=_k("run_search"))

        if run_pressed:
            _run_search()
            st.rerun()

        editor_fig = make_editor_figure(grid_cost, start, goal)
        event = st.plotly_chart(
            editor_fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key=_k("editor_chart"),
            config={"displayModeBar": False},
        )

        if event and event.selection and event.selection.points:
            pt = event.selection.points[0]
            r, c = int(round(pt["y"])), int(round(pt["x"]))
            if _apply_brush(r, c, brush):
                st.session_state[_k("snapshots")] = None
                st.rerun()

    st.stop()

# After a run: offer a quick way back into edit mode
st.info(
    "Search complete on this grid. Press **✏️ Edit grid** to move the "
    "start/goal or change walls, or adjust the sidebar to try a different "
    "grid or algorithm."
)
if st.button("✏️ Edit grid", key=_k("edit_grid")):
    st.session_state[_k("snapshots")] = None
    st.rerun()

# ---------------------------------------------------------------------------
# Playback controls (Streamlit-native — Prev/Next + Play/Pause via rerun)
# ---------------------------------------------------------------------------
n_steps = len(snapshots)
step_idx: int = st.session_state.get(_k("step_idx"), 0)
step_idx = max(0, min(step_idx, n_steps - 1))
st.session_state[_k("step_idx")] = step_idx
playing: bool = st.session_state.get(_k("playing"), False)

with st.container(border=True):
    speed = st.select_slider(
        "Playback speed", options=["0.5×", "1×", "2×", "4×", "8×"], value="2×",
        label_visibility="collapsed",
        key=_k("speed"),
    )
    DELAY = {"0.5×": 0.9, "1×": 0.45, "2×": 0.22, "4×": 0.11, "8×": 0.05}[speed]

    col_prev, col_play, col_pause, col_next, col_speed = st.columns([1, 1.2, 1.2, 1, 3])

    with col_prev:
        if st.button("◀ Prev", use_container_width=True, disabled=(step_idx == 0 or playing), key=_k("prev")):
            st.session_state[_k("step_idx")] = max(0, step_idx - 1)
            st.rerun()

    with col_play:
        if st.button("▶  Play", use_container_width=True,
                     disabled=(playing or step_idx == n_steps - 1), type="primary", key=_k("play")):
            st.session_state[_k("playing")] = True
            st.rerun()

    with col_pause:
        if st.button("⏸  Pause", use_container_width=True, disabled=not playing, key=_k("pause")):
            st.session_state[_k("playing")] = False
            st.rerun()

    with col_next:
        if st.button("Next ▶", use_container_width=True,
                     disabled=(step_idx == n_steps - 1 or playing), key=_k("next")):
            st.session_state[_k("step_idx")] = min(n_steps - 1, step_idx + 1)
            st.rerun()

    with col_speed:
        st.caption(f"Speed: **{speed}**  ({DELAY:.2f}s per frame)")

    snap = snapshots[step_idx]
    st.progress(step_idx / max(n_steps - 1, 1), text=f"Frame {step_idx + 1} / {n_steps} — {snap.title}")

# ---------------------------------------------------------------------------
# Chart + priority-queue panel
# ---------------------------------------------------------------------------
col_chart, col_queue = st.columns([3, 1])

with col_chart, st.container(border=True):
    fig = make_static_figure(snap)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=_k("chart"))

with col_queue, st.container(border=True):
    st.markdown("**Priority queue** (best entries)")
    if snap.frontier_preview:
        for priority, g, (r, c) in snap.frontier_preview:
            marker = "👉 " if snap.current == (r, c) else ""
            st.text(f"{marker}({r:>2},{c:>2})  f={priority:5.1f}  g={g:5.1f}")
    else:
        st.text("— empty —")

with st.container(border=True):
    m1, m2, m3 = st.columns(3)
    m1.metric("Phase", snap.phase.replace("_", " ").capitalize())
    m2.metric("Nodes visited", str(snap.step))
    if snap.phase == "found":
        m3.metric("Path cost", f"{snap.total_cost:.1f}")
    elif snap.phase == "no_path":
        m3.metric("Status", "Unreachable ✗")
    else:
        m3.metric("Open set size", str(int(np.count_nonzero(snap.open_mask))))

compare = st.session_state.get(_k("compare"))
if compare and snap.phase == "found":
    dij, a_star = compare.get("dijkstra"), compare.get("astar")
    if dij and a_star:
        diff_pct = 100 * (dij - a_star) / dij
        comparison = (
            f"A* explored **{diff_pct:.0f}% fewer** nodes to find the same optimal path."
            if diff_pct > 0
            else "both algorithms explored about the same number of nodes here."
        )
        st.caption(
            f"On this exact grid: Dijkstra visits **{dij}** nodes, A* visits "
            f"**{a_star}** nodes — {comparison}"
        )

if snap.phase == "init":
    st.info(
        "**Ready** — the priority queue holds only the start node at cost 0. "
        "Press **▶ Play** or **Next ▶** to begin expanding the frontier."
    )
elif snap.phase == "visit":
    n_upd = len(snap.updated)
    st.info(
        f"**Visiting ({snap.current[0]}, {snap.current[1]})** — this node's "
        f"shortest distance is now locked in. {n_upd} neighbour"
        f"{'s' if n_upd != 1 else ''} got a cheaper known cost and "
        f"{'were' if n_upd != 1 else 'was'} pushed into the queue."
    )
elif snap.phase == "found":
    st.success(
        f"**Goal reached** — cheapest path costs **{snap.total_cost:.1f}** "
        f"over {len(snap.path) - 1} steps, after visiting {snap.step} nodes."
    )
else:
    st.error(
        f"**No path exists** — the queue emptied after visiting {snap.step} "
        "nodes without ever reaching the goal. Walls fully separate start "
        "from goal."
    )

with st.expander("📖 Reading the animation"):
    st.markdown(
        """
- **Terrain shading** — darker/browner cells cost more to enter (mud, water)
- **Dark grey / black** — walls, impassable
- **Light indigo** — closed / finalised nodes (shortest distance locked in)
- **Amber** — open set — known but not yet finalised
- **Rose square** — the node being expanded *this step*
- **Dotted rose line** — the current shortest known path from start to that node
- **Teal line** — the final shortest path, once the goal is found
- **S (green) / G (violet)** — start and goal
- The **priority queue** panel on the right shows the best few candidates
  waiting to be expanded next — `f` is priority (`g` for Dijkstra, `g+h` for A*)
"""
    )

# ---------------------------------------------------------------------------
# Auto-advance (must be last — triggers rerun after a delay)
# ---------------------------------------------------------------------------
if playing:
    if step_idx < n_steps - 1:
        time.sleep(DELAY)
        st.session_state[_k("step_idx")] = min(n_steps - 1, step_idx + 1)
        st.rerun()
    else:
        st.session_state[_k("playing")] = False
        st.rerun()
