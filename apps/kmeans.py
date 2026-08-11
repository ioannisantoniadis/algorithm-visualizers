"""K-means page — adapted from the standalone kmeans-viz repo for the
multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from common.ui import about_section, params_rail
from kmeans.algorithm import fit
from kmeans.data import SHAPE_KEYS, SHAPE_NAMES, make_dataset
from kmeans.visualize import make_picker_figure, make_static_figure

NS = "kmeans"


def _k(name: str) -> str:
    return f"{NS}__{name}"


# ---------------------------------------------------------------------------
# Header — title first, caption filled in after params are known below
# ---------------------------------------------------------------------------
st.title("K-means Clustering — Step-by-step Visualiser")
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

# ---------------------------------------------------------------------------
# Params rail — configuration for this algorithm
# ---------------------------------------------------------------------------
with params_rail(col_params):
    shape_name = st.selectbox(
        "Data shape",
        options=SHAPE_NAMES,
        index=0,
        help="Choose the geometry of the input data. Non-spherical shapes reveal "
             "K-means' fundamental limitations.",
        key=_k("shape"),
    )
    shape_key = SHAPE_KEYS[SHAPE_NAMES.index(shape_name)]

    n_points = st.slider("Number of points", min_value=50, max_value=500, value=200, step=10, key=_k("n_points"))
    n_clusters = st.slider(
        "Number of clusters (K)",
        min_value=2,
        max_value=8,
        value=4,
        help="This is K — the number of centroids K-means will search for. It "
             "always affects the algorithm, even on shapes (Moons / Rings) whose "
             "true group count is fixed at 2.",
        key=_k("n_clusters"),
    )
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key=_k("seed"))
    max_iter = st.slider("Max iterations", min_value=5, max_value=50, value=20,
        help="Hard cap on E/M cycles. Most shapes converge well before this; "
             "lower it to freeze the algorithm mid-convergence.",
        key=_k("max_iter"))

    init_mode = st.radio(
        "Centroid initialisation",
        ["Random", "Manual (click to place)"],
        help="Manual mode lets you click data points to set starting centroids, "
             "showing how the choice affects convergence.",
        key=_k("init_mode"),
    )
    manual_mode = init_mode == "Manual (click to place)"

    regenerate = st.button("🔄 Re-generate data", use_container_width=True, key=_k("regen"))

    # Shape-specific educational note
    _SHAPE_NOTES = {
        "blobs":        "**Gaussian blobs** — well-separated spherical clusters. "
                        "K-means works perfectly here with any reasonable initialisation.",
        "anisotropic":  "**Anisotropic blobs** — elongated, rotated ellipses. "
                        "K-means can mis-split blobs along the long axis, especially "
                        "with a poor starting centroid.",
        "varied":       "**Varied density** — clusters with very different sizes. "
                        "The large blob tends to absorb nearby small-cluster points "
                        "because K-means assumes equal-volume cells.",
        "moons":        "**Two moons** — non-convex crescents. K-means always produces "
                        "a straight-line cut regardless of K, because its decision "
                        "boundaries are Voronoi hyperplanes.",
        "circles":      "**Concentric rings** — K-means's decision boundaries are always "
                        "straight lines (Voronoi hyperplanes), so it can't recover nested "
                        "rings regardless of K; at K=2 it simply bisects the plane instead "
                        "of separating inner from outer.",
        "uniform":      "**Uniform noise** — no true clusters. K-means still converges "
                        "to K cells that tile the space, showing it always produces "
                        "*some* partition.",
    }
    st.info(_SHAPE_NOTES[shape_key])

    with st.expander("How K-means works", expanded=False):
        st.markdown("""
1. K points become initial **centroids ★**
2. **E-step** — every point assigned to nearest centroid
3. **M-step** — centroids move to their cluster mean
4. Repeat until centroids stop moving
""")

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _generate_data() -> None:
    X, true_labels = make_dataset(shape_key, n_points=n_points, n_clusters=n_clusters, seed=int(seed))
    st.session_state[_k("X")] = X
    st.session_state[_k("true_labels")] = true_labels
    st.session_state[_k("snapshots")] = None
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False
    st.session_state[_k("manual_centroids")] = []


def _run_kmeans(init_centroids: np.ndarray | None = None) -> None:
    X = st.session_state[_k("X")]
    snapshots = fit(
        X,
        k=n_clusters,
        max_iter=max_iter,
        seed=int(seed),
        init_centroids=init_centroids,
    )
    st.session_state[_k("snapshots")] = snapshots
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


# Detect parameter changes that should invalidate the current dataset/run.
_param_key = (shape_key, n_points, n_clusters, int(seed), max_iter)

# Detect a flip of the Random <-> Manual radio so we don't keep showing a
# stale run under the wrong label.
_mode_key = _k("_manual_mode_prev")
mode_changed = (
    _mode_key in st.session_state and st.session_state[_mode_key] != manual_mode
)
st.session_state[_mode_key] = manual_mode

if _k("X") not in st.session_state or regenerate or st.session_state.get(_k("_param_key")) != _param_key:
    _generate_data()
    st.session_state[_k("_param_key")] = _param_key
    # Auto-run immediately for random mode
    if not manual_mode:
        _run_kmeans()
elif mode_changed:
    # Same data, but the init mode flipped: drop the old run (and any
    # leftover manual picks) so the picker / auto-run reflects reality.
    st.session_state[_k("snapshots")] = None
    st.session_state[_k("playing")] = False
    st.session_state[_k("manual_centroids")] = []
    if not manual_mode:
        _run_kmeans()

# If mode switched to random and there's no run yet, auto-run
if not manual_mode and st.session_state.get(_k("snapshots")) is None:
    _run_kmeans()

X: np.ndarray = st.session_state[_k("X")]
manual_centroids: list[list[float]] = st.session_state.get(_k("manual_centroids"), [])

caption_slot.caption(
    f"Shape: **{shape_name}** | "
    f"Points: **{n_points}** | "
    f"K = **{n_clusters}** | "
    f"Seed: **{seed}** | "
    f"Init: **{init_mode}**"
)

with col_main:
    about_section(
        "One of the oldest and most widely used clustering algorithms, prized for "
        "being fast and simple enough to run on huge datasets as a first pass before "
        "reaching for anything fancier. It's the standard entry point to unsupervised "
        "learning and the backbone of tasks like customer segmentation and image "
        "color quantization — and initializing other models, including the Gaussian "
        "Mixture on this site, sometimes starts from a K-means run. Its core "
        "limitation — it can only find spherical, equally-sized clusters, because "
        "every decision boundary it draws is a straight line — is exactly what the "
        "**Moons**, **Rings**, and **Varied density** shapes in the sidebar are "
        "designed to expose.",
        [
            "Lloyd, S. (1957/1982). \"Least Squares Quantization in PCM.\" "
            "*IEEE Transactions on Information Theory.*",
            "MacQueen, J. (1967). \"Some Methods for Classification and Analysis "
            "of Multivariate Observations.\" *5th Berkeley Symposium on Mathematical "
            "Statistics and Probability.*",
        ],
    )

    # -----------------------------------------------------------------------
    # MANUAL CENTROID PLACEMENT — picker phase
    # -----------------------------------------------------------------------
    if manual_mode:
        n_placed = len(manual_centroids)
        remaining = n_clusters - n_placed

        # Action buttons row
        with st.container(border=True):
            bcol1, bcol2, bcol3 = st.columns([2, 2, 4])
            with bcol1:
                if st.button(
                    "🗑 Reset centroids",
                    use_container_width=True,
                    disabled=(n_placed == 0),
                    key=_k("reset_centroids"),
                ):
                    st.session_state[_k("manual_centroids")] = []
                    st.session_state[_k("snapshots")] = None
                    st.rerun()
            with bcol2:
                run_pressed = st.button(
                    "▶ Run K-means",
                    use_container_width=True,
                    disabled=(n_placed < n_clusters),
                    type="primary",
                    key=_k("run_kmeans_btn"),
                )

        if run_pressed and n_placed == n_clusters:
            _run_kmeans(init_centroids=np.array(manual_centroids))
            st.rerun()

        # Show picker only when K-means hasn't been run yet with these centroids
        if st.session_state.get(_k("snapshots")) is None:
            if remaining > 0:
                st.info(
                    f"**Click {remaining} more data point{'s' if remaining > 1 else ''}** "
                    f"on the chart below to place {'a centroid' if remaining == 1 else 'centroids'}. "
                    f"({n_placed}/{n_clusters} placed)"
                )
            else:
                st.success(
                    f"All **{n_clusters}** centroids placed. Press **▶ Run K-means** above."
                )

            with st.container(border=True):
                picker_fig = make_picker_figure(X, manual_centroids, n_clusters)
                event = st.plotly_chart(
                    picker_fig,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="points",
                    config={"displayModeBar": False},
                    key=_k("picker_chart"),
                )

            # Process click: add first newly selected point not already chosen
            if event and event.selection and event.selection.points:
                for pt in event.selection.points:
                    coord = [pt["x"], pt["y"]]
                    # Deduplicate: skip if this exact point was already placed
                    already = any(
                        abs(c[0] - coord[0]) < 1e-9 and abs(c[1] - coord[1]) < 1e-9
                        for c in manual_centroids
                    )
                    if not already and len(manual_centroids) < n_clusters:
                        st.session_state[_k("manual_centroids")].append(coord)
                        st.rerun()

            st.stop()  # Don't render the visualisation until K-means has been run

        # After running: show a "re-place" option alongside the visualisation
        st.info(
            f"K-means ran with your **{n_clusters} manually chosen** centroids. "
            "Press **🗑 Reset centroids** to try a different placement."
        )

    # -----------------------------------------------------------------------
    # K-means visualisation (both random and manual modes reach here)
    # -----------------------------------------------------------------------
    snapshots = st.session_state.get(_k("snapshots"))
    if snapshots is None:
        st.stop()

    n_steps = len(snapshots)

    # -----------------------------------------------------------------------
    # Playback controls  (pure Streamlit — no Plotly updatemenus)
    # -----------------------------------------------------------------------
    step_idx: int = st.session_state.get(_k("step_idx"), 0)
    step_idx = max(0, min(step_idx, n_steps - 1))
    st.session_state[_k("step_idx")] = step_idx
    playing:  bool = st.session_state.get(_k("playing"), False)

    with st.container(border=True):
        speed = st.select_slider(
            "Playback speed",
            options=["0.5×", "1×", "2×", "4×"],
            value="1×",
            label_visibility="collapsed",
            key=_k("speed"),
        )
        DELAY = {"0.5×": 1.4, "1×": 0.7, "2×": 0.35, "4×": 0.18}[speed]

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

        st.progress(step_idx / max(n_steps - 1, 1),
                    text=f"Frame {step_idx + 1} / {n_steps} — {snapshots[step_idx].title}")

    # -----------------------------------------------------------------------
    # Chart — always a static figure; auto-play advances step_idx via rerun
    # -----------------------------------------------------------------------
    snap = snapshots[step_idx]

    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Phase", snap.phase.capitalize())
        m2.metric("Iteration", str(snap.iteration))
        if snap.phase == "assign" and step_idx > 0:
            m3.metric("Points changed cluster", str(snap.n_changed))
        elif snap.converged:
            m3.metric("Status", "Converged ✓")
        else:
            m3.metric("Points changed cluster", "—")

    with st.container(border=True):
        fig = make_static_figure(snap)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=_k("chart"))

    if snap.phase == "init":
        kind = "manually chosen" if snap.manual_init else "randomly chosen"
        st.info(
            f"**Initialisation** — {n_clusters} {kind} data points become the "
            f"starting centroids (★). Points have no cluster assignment yet (grey)."
        )
    elif snap.phase == "assign":
        pct = f"{100 * snap.n_changed / n_points:.1f}%"
        st.info(
            f"**E-step (assign)** — Each point is assigned to its nearest centroid "
            f"using squared Euclidean distance. **{snap.n_changed} points** ({pct}) "
            f"changed cluster."
        )
    else:
        status = (
            "Centroids have stopped moving — the algorithm has **converged**."
            if snap.converged
            else "Centroids will keep moving next iteration."
        )
        st.info(f"**M-step (update)** — Each centroid moves to the mean of its cluster. {status}")

    with st.expander("📖 Reading the animation"):
        st.markdown("""
- **Colour change** in points = E-step (assign): points snapping to nearest centroid
- **Stars ★** moving = M-step (update): centroids sliding to their cluster mean
- **◀ Prev / Next ▶** to step manually, **▶ Play** to auto-advance
- Adjust speed with the speed selector above the controls
""")

    # -----------------------------------------------------------------------
    # Auto-advance (must be last — triggers rerun after a delay)
    # -----------------------------------------------------------------------
    if playing:
        if step_idx < n_steps - 1:
            time.sleep(DELAY)
            st.session_state[_k("step_idx")] = min(n_steps - 1, step_idx + 1)
            st.rerun()
        else:
            st.session_state[_k("playing")] = False
            st.rerun()
