"""MST page — adapted from the standalone mst-viz repo for the multipage
portfolio app. Session-state keys are namespaced with a per-page prefix
since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from common.ui import about_section, params_rail
from mst.algorithm import run as run_mst
from mst.data import PRESET_KEYS, PRESET_NAMES, Edge, complete_graph_edges, make_graph
from mst.visualize import make_editor_figure, make_static_figure

NS = "mst"


def _k(name: str) -> str:
    return f"{NS}__{name}"


st.title("Minimum Spanning Tree — Kruskal & Prim")
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

# ---------------------------------------------------------------------------
# Params rail — graph source
# ---------------------------------------------------------------------------
with params_rail(col_params, "Graph"):
    mode = st.radio(
        "Graph source",
        ["Preset", "Draw your own"],
        help="Preset generates a graph and runs immediately. Draw your own lets "
             "you click nodes onto a blank canvas and optionally hand-edit "
             "edge weights before running.",
        key=_k("mode"),
    )

    preset_name = None
    preset_key = "random_euclidean"
    n_nodes = 12
    n_clusters = 3
    edge_prob = 0.25
    grid_rows, grid_cols = 4, 4

    if mode == "Preset":
        preset_name = st.selectbox("Preset", PRESET_NAMES, index=0, key=_k("preset"))
        preset_key = PRESET_KEYS[PRESET_NAMES.index(preset_name)]

        if preset_key == "grid_lattice":
            grid_rows = st.slider("Grid rows", 2, 8, 4, key=_k("grid_rows"))
            grid_cols = st.slider("Grid columns", 2, 8, 4, key=_k("grid_cols"))
        else:
            n_nodes = st.slider("Number of nodes", min_value=5, max_value=24, value=12, key=_k("n_nodes"))
            if preset_key == "clustered":
                n_clusters = st.slider("Number of clusters", 2, 5, 3, key=_k("n_clusters"))
            if preset_key == "sparse_random":
                edge_prob = st.slider("Edge probability", 0.05, 0.6, 0.25, step=0.05, key=_k("edge_prob"))

    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key=_k("seed"))

    regenerate = st.button(
        "🔄 Regenerate graph" if mode == "Preset" else "🔄 Clear canvas",
        use_container_width=True,
        key=_k("regenerate"),
    )

# ---------------------------------------------------------------------------
# Params rail — algorithm
# ---------------------------------------------------------------------------
with params_rail(col_params, "Algorithm"):
    algorithm_name = st.radio("MST algorithm", ["Kruskal", "Prim"], key=_k("algorithm"))
    algorithm_key = "prim" if algorithm_name == "Prim" else "kruskal"

    start_node = st.number_input(
        "Start node (Prim only)",
        min_value=0,
        max_value=max(n_nodes * 4, 1),  # generous upper bound; clamped again once the graph exists
        value=0,
        step=1,
        disabled=(algorithm_key != "prim"),
        help="Prim grows its tree outward from this node. Kruskal has no seed "
             "node — it works globally on the sorted edge list from the start.",
        key=_k("start_node"),
    )

with col_params:
    _PRESET_NOTES = {
        "random_euclidean": "**Random Euclidean** — every pair of points is connected, "
            "weighted by straight-line distance. Both algorithms have to sift through "
            "a dense complete graph to recover the same sparse tree.",
        "clustered": "**Clustered blobs** — a few tight groups of points. Watch each "
            "cluster wire itself up almost immediately, then notice how few, and how "
            "specific, the bridging edges between clusters are.",
        "sparse_random": "**Sparse random graph** — a random subset of possible edges. "
            "With fewer edges to choose from, cycle rejections (Kruskal) and dead-end "
            "frontier entries (Prim) become more common and more visible.",
        "grid_lattice": "**Grid lattice** — a regular 4-connected grid. Random weights "
            "on a very regular structure make it easy to see exactly why one edge beats "
            "a seemingly-similar neighbour.",
    }
    if mode == "Preset":
        st.info(_PRESET_NOTES[preset_key])
    else:
        st.info(
            "**Draw your own** — click the canvas to add nodes (click a node to "
            "remove it), then **🔒 Lock & set weights** to optionally hand-edit "
            "edge weights before pressing **▶ Run MST**."
        )

    st.markdown(
        """
**Kruskal (cycle property)**
1. Sort *all* edges by weight, cheapest first
2. Accept an edge unless its two endpoints are already connected
   (checked via Union-Find) — accepting it would close a cycle
3. Stop once n−1 edges have been accepted

**Prim (cut property)**
1. Start with one node "in the tree"
2. Repeatedly add the cheapest edge crossing the cut between the
   tree and everything outside it
3. Stop once every reachable node has joined
"""
    )

# ---------------------------------------------------------------------------
# Session state — preset graph generation
# ---------------------------------------------------------------------------

def _run_algorithm(start: int = 0) -> None:
    positions = st.session_state[_k("positions")]
    edges = st.session_state[_k("edges")]
    n = len(positions)
    start = int(np.clip(start, 0, max(n - 1, 0)))
    st.session_state[_k("snapshots")] = run_mst(algorithm_key, positions, edges, start=start)
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


_param_key = (mode, preset_key, n_nodes, n_clusters, edge_prob, grid_rows, grid_cols, int(seed))
_algo_key = (algorithm_key, int(start_node))

if mode == "Preset":
    graph_changed = _k("positions") not in st.session_state or regenerate or st.session_state.get(_k("_param_key")) != _param_key
    if graph_changed:
        positions, edges = make_graph(
            preset_key, n_nodes=n_nodes, seed=int(seed),
            n_clusters=n_clusters, edge_prob=edge_prob, rows=grid_rows, cols=grid_cols,
        )
        st.session_state[_k("positions")] = positions
        st.session_state[_k("edges")] = edges
        st.session_state[_k("_param_key")] = _param_key
        _run_algorithm(start=int(start_node))
        st.session_state[_k("_algo_key")] = _algo_key
    elif st.session_state.get(_k("_algo_key")) != _algo_key and st.session_state.get(_k("snapshots")) is not None:
        _run_algorithm(start=int(start_node))
        st.session_state[_k("_algo_key")] = _algo_key

if mode == "Preset":
    _n_actual = len(st.session_state.get(_k("positions"), []))
    # Display the *actual* (clamped) start node Prim will run from, not the
    # raw widget value — the widget's upper bound is deliberately generous
    # (it can't know grid_lattice's true node count ahead of time), so a
    # value like 47 on a 12-node graph silently gets clamped to 11 by
    # _run_algorithm. Showing the raw value here would contradict what the
    # chart actually depicts.
    _actual_start = int(np.clip(int(start_node), 0, max(_n_actual - 1, 0)))
    caption_slot.caption(
        f"Graph: **{preset_name}** | Nodes: **{_n_actual}** | "
        f"Algorithm: **{algorithm_name}**"
        + (f" (start = {_actual_start})" if algorithm_key == "prim" else "")
    )

with col_main:
    about_section(
        "MST algorithms answer a deceptively simple question — what's the "
        "cheapest way to connect every node in a network — that shows up "
        "everywhere from designing road, utility, and circuit networks to "
        "approximation algorithms for harder problems like the travelling "
        "salesman problem. Kruskal's and Prim's algorithms solve the *exact* "
        "same problem via two different greedy strategies (globally "
        "cheapest-edge-first vs. growing outward from one node), and both are "
        "provably optimal for completely different reasons — the **cycle "
        "property** and the **cut property** respectively — both called out "
        "directly in this page's step-by-step explanations.",
        [
            "Kruskal, J.B. (1956). \"On the Shortest Spanning Subtree of a "
            "Graph and the Traveling Salesman Problem.\" *Proceedings of the "
            "American Mathematical Society.*",
            "Prim, R.C. (1957). \"Shortest Connection Networks and Some "
            "Generalizations.\" *Bell System Technical Journal.*",
        ],
    )

    # -------------------------------------------------------------------
    # DRAW-YOUR-OWN mode
    # -------------------------------------------------------------------
    if mode == "Draw your own":
        if regenerate:
            st.session_state[_k("draw_positions")] = []
            st.session_state[_k("draw_locked")] = False
            st.session_state[_k("snapshots")] = None

        draw_positions: list[list[float]] = st.session_state.get(_k("draw_positions"), [])
        locked: bool = st.session_state.get(_k("draw_locked"), False)

        if not locked:
            st.info(
                "Click empty canvas to **add a node**, or click an existing node to "
                "**remove it**. Once you have at least 2 nodes, press "
                "**🔒 Lock & set weights**."
            )

            bcol1, bcol2, bcol3 = st.columns([2, 2, 4])
            with bcol1:
                if st.button("🗑 Clear nodes", use_container_width=True, disabled=(len(draw_positions) == 0), key=_k("clear_nodes")):
                    st.session_state[_k("draw_positions")] = []
                    st.rerun()
            with bcol2:
                lock_pressed = st.button(
                    "🔒 Lock & set weights", use_container_width=True, type="primary",
                    disabled=(len(draw_positions) < 2),
                    key=_k("lock"),
                )

            positions_arr = np.array(draw_positions) if draw_positions else np.empty((0, 2))
            edges_preview = complete_graph_edges(positions_arr) if len(positions_arr) >= 2 else []
            editor_fig = make_editor_figure(positions_arr, edges_preview)
            event = st.plotly_chart(
                editor_fig, use_container_width=True, on_select="rerun",
                selection_mode="points", key=_k("editor_chart"),
                config={"displayModeBar": False},
            )

            if lock_pressed:
                edges = complete_graph_edges(positions_arr)
                st.session_state[_k("weight_table")] = [
                    {"Node A": e.u, "Node B": e.v, "Weight": round(e.w, 2)} for e in edges
                ]
                st.session_state[_k("draw_locked")] = True
                st.rerun()

            if event and event.selection and event.selection.points:
                pt = event.selection.points[0]
                x, y = float(pt["x"]), float(pt["y"])
                if draw_positions:
                    dists = [np.hypot(x - px, y - py) for px, py in draw_positions]
                    nearest = int(np.argmin(dists))
                else:
                    dists, nearest = [], None
                if nearest is not None and dists[nearest] < 0.5:
                    draw_positions.pop(nearest)
                else:
                    draw_positions.append([round(x, 2), round(y, 2)])
                st.session_state[_k("draw_positions")] = draw_positions
                st.rerun()

            st.stop()

        # ---- Locked: edit weights, then run --------------------------------
        positions_arr = np.array(st.session_state[_k("draw_positions")])
        st.info(
            "Nodes are locked. Edit **Weight** values below (defaults = Euclidean "
            "distance), or leave them as-is, then press **▶ Run MST**."
        )

        edited = st.data_editor(
            st.session_state[_k("weight_table")],
            column_config={
                "Node A": st.column_config.NumberColumn(disabled=True),
                "Node B": st.column_config.NumberColumn(disabled=True),
                "Weight": st.column_config.NumberColumn(min_value=0.01, step=0.1, format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
            key=_k("weight_editor"),
        )
        st.session_state[_k("weight_table")] = edited
        edges = [Edge(int(r["Node A"]), int(r["Node B"]), float(r["Weight"])) for r in edited]

        bcol1, bcol2 = st.columns([2, 2])
        with bcol1:
            if st.button("🔓 Unlock (edit nodes)", use_container_width=True, key=_k("unlock")):
                st.session_state[_k("draw_locked")] = False
                st.session_state[_k("snapshots")] = None
                st.rerun()
        with bcol2:
            run_pressed = st.button("▶ Run MST", use_container_width=True, type="primary", key=_k("run_mst"))

        if run_pressed:
            st.session_state[_k("positions")] = positions_arr
            st.session_state[_k("edges")] = edges
            _run_algorithm(start=int(start_node))
            st.rerun()

        if st.session_state.get(_k("snapshots")) is None:
            st.stop()

    # -------------------------------------------------------------------
    # Shared visualisation (both modes reach here once snapshots exist)
    # -------------------------------------------------------------------
    snapshots = st.session_state.get(_k("snapshots"))
    if not snapshots:
        st.stop()

    n_steps = len(snapshots)
    n_nodes_actual = len(st.session_state[_k("positions")])

    # -------------------------------------------------------------------
    # Playback controls (Streamlit-native — Prev/Next + Play/Pause via rerun)
    # -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
    # Chart + side panel (Union-Find sets for Kruskal, frontier for Prim)
    # -------------------------------------------------------------------
    with st.container(border=True):
        chart_col, panel_col = st.columns([3, 1])

        with chart_col:
            fig = make_static_figure(snap)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=_k("main_chart"))

        with panel_col:
            if snap.algorithm == "kruskal":
                st.markdown("**Union-Find components**")
                if snap.component_id is not None:
                    comps: dict[int, list[int]] = {}
                    for node, root in enumerate(snap.component_id.tolist()):
                        comps.setdefault(root, []).append(node)
                    for members in sorted(comps.values(), key=lambda m: -len(m)):
                        st.text("{ " + ", ".join(str(m) for m in sorted(members)) + " }")

                st.markdown("**Next edges (sorted)**")
                order = sorted(snap.edges, key=lambda e: (e.w, e.u, e.v))
                upcoming = order[snap.sorted_index:snap.sorted_index + 5]
                if upcoming:
                    for e in upcoming:
                        st.text(f"({e.u:>2},{e.v:>2})  w={e.w:5.1f}")
                else:
                    st.text("— none left —")
            else:
                st.markdown("**Frontier (cut-crossing edges)**")
                if snap.frontier_preview:
                    for e in snap.frontier_preview:
                        marker = "👉 " if snap.current_edge == e else ""
                        st.text(f"{marker}({e.u:>2},{e.v:>2})  w={e.w:5.1f}")
                else:
                    st.text("— empty —")

    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Phase", snap.phase.replace("_", " ").capitalize())
        m2.metric("Edges in tree", f"{len(snap.mst_edges)} / {n_nodes_actual - 1}")
        m3.metric("Total weight", f"{snap.total_weight:.1f}")

    if snap.phase == "init":
        if snap.algorithm == "kruskal":
            st.info(
                "**Ready** — every node starts as its own component. Press "
                "**▶ Play** or **Next ▶** to start working through edges in "
                "increasing weight order."
            )
        else:
            # Read the actual start node back off the snapshot's own in_tree
            # mask (ground truth) rather than the raw start_node widget value —
            # the widget's value gets clamped to a valid node index inside
            # _run_algorithm, so echoing it unclamped here could contradict
            # the node actually highlighted green in the chart.
            _actual_start = int(np.argmax(snap.in_tree)) if snap.in_tree is not None else int(start_node)
            st.info(
                f"**Ready** — the tree starts containing only node **{_actual_start}**. "
                "Press **▶ Play** or **Next ▶** to start growing it."
            )
    elif snap.phase == "accept":
        e = snap.current_edge
        st.success(
            f"**Accepted ({e.u}, {e.v})**, weight {e.w:.1f} — its endpoints were in "
            f"different components, so this edge cannot close a cycle. Components merge."
        )
    elif snap.phase == "reject":
        e = snap.current_edge
        st.warning(
            f"**Rejected ({e.u}, {e.v})**, weight {e.w:.1f} — its endpoints are already "
            "connected by the highlighted path. Adding it would close a cycle, and by "
            "the **cycle property**, this edge is guaranteed to be the heaviest edge on "
            "that cycle (every other edge on the path was accepted earlier, at a lower "
            "or equal weight)."
        )
    elif snap.phase == "grow":
        e = snap.current_edge
        st.success(
            f"**Added ({e.u}, {e.v})**, weight {e.w:.1f} — the cheapest edge currently "
            "crossing the cut (dashed) between the tree and everything outside it. By "
            "the **cut property**, this edge is guaranteed to belong to *some* MST."
        )
    else:  # done
        if snap.n_components == 1:
            st.success(
                f"**Spanning tree complete** — {len(snap.mst_edges)} edges, total weight "
                f"**{snap.total_weight:.1f}**."
            )
        else:
            st.error(
                f"**Incomplete** — the graph has {snap.n_components} disconnected "
                f"fragments, so only a minimum spanning *forest* could be built "
                f"({len(snap.mst_edges)} edges total)."
            )

    with st.expander("📖 Reading the animation"):
        if snap.algorithm == "kruskal":
            st.markdown(
                """
- **Grey lines** — every candidate edge (hover for its weight)
- **Node colour** — current Union-Find component; same colour = same component
- **Teal (thick)** — edges accepted into the MST so far
- **Red dotted** — edges rejected because they would close a cycle
- **Amber dashed** — the existing tree path that, together with the current
  red edge, would form that cycle
- The **Union-Find components** panel lists which nodes currently share a set
- The **Next edges** panel previews what Kruskal's sorted queue looks at next
"""
            )
        else:
            st.markdown(
                """
- **Grey lines** — every candidate edge (hover for its weight)
- **Teal nodes** — already in the tree; **grey nodes** — not yet reached
- **Teal (thick)** — edges already added to the tree
- **Amber dotted** — the current *cut*: every edge crossing from the tree to
  outside it (the only ones the algorithm is allowed to choose from next)
- **Indigo (thick)** — the edge just added — always the cheapest edge in the cut
- The **Frontier** panel lists the best few live candidates from that cut
"""
            )

    # -------------------------------------------------------------------
    # Auto-advance (must be last — triggers rerun after a delay)
    # -------------------------------------------------------------------
    if playing:
        if step_idx < n_steps - 1:
            time.sleep(DELAY)
            st.session_state[_k("step_idx")] = step_idx + 1
            st.rerun()
        else:
            st.session_state[_k("playing")] = False
            st.rerun()
