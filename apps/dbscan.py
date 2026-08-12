"""DBSCAN page — adapted from the standalone dbscan-viz repo for the
multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import streamlit as st

from common.ui import about_section, params_rail
from dbscan.algorithm import fit
from dbscan.data import SHAPE_DEFAULTS, SHAPE_KEYS, SHAPE_NAMES, make_dataset
from dbscan.visualize import make_static_figure

NS = "dbscan"


def _k(name: str) -> str:
    return f"{NS}__{name}"


# ---------------------------------------------------------------------------
# Header — title first, caption filled in after params are known below
# ---------------------------------------------------------------------------
st.title("DBSCAN — Step-by-step Visualiser")
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

# ---------------------------------------------------------------------------
# Params rail
# ---------------------------------------------------------------------------
with params_rail(col_params, "Data"):
    shape_name = st.selectbox("Data shape", SHAPE_NAMES, index=3, key=_k("shape"))
    shape_key  = SHAPE_KEYS[SHAPE_NAMES.index(shape_name)]
    defaults   = SHAPE_DEFAULTS[shape_key]

    n_points   = st.slider("Number of points", 50, 400, 200, step=10, key=_k("n_points"))
    n_clusters = st.slider("Number of blobs (K)", 2, 8, 4,
        disabled=shape_key not in ("blobs", "anisotropic", "varied"),
        help="Only applies to Gaussian blobs, anisotropic, and varied-density shapes.",
        key=_k("n_clusters"))
    seed = st.number_input("Random seed", 0, 9999, 42, step=1, key=_k("seed"))

with params_rail(col_params, "DBSCAN parameters"):
    eps = st.slider("ε — neighbourhood radius", 0.1, 2.0,
        float(defaults["eps"]), step=0.05,
        help="Points within this distance of a core point belong to the same cluster.",
        key=_k("eps"))
    min_samples = st.slider("minPts — core-point threshold", 2, 20,
        int(defaults["min_samples"]),
        help="Minimum neighbours (including itself) to qualify as a core point.",
        key=_k("min_samples"))
    regenerate = st.button("🔄 Re-generate", use_container_width=True, key=_k("regen"))

with col_params:
    _SHAPE_NOTES = {
        "blobs":       "**Gaussian blobs** — DBSCAN trivially recovers the clusters. Try raising ε to merge two blobs.",
        "anisotropic": "**Anisotropic blobs** — elongated clusters. DBSCAN handles these well; K-means often fails.",
        "varied":      "**Varied density** — denser blobs need a smaller ε. A single global ε can't fit all densities.",
        "moons":       "**Two moons** — DBSCAN's showcase strength. K-means cuts vertically; DBSCAN traces the crescents.",
        "circles":     "**Concentric rings** — K-means bisects the plane; DBSCAN recovers both rings.",
        "uniform":     "**Uniform noise** — no density structure → almost everything is noise (✕).",
    }
    st.info(_SHAPE_NOTES[shape_key])
    with st.expander("How DBSCAN works", expanded=False):
        st.markdown("""
1. Pick an unvisited point, query its ε-ball
2. If ≥ minPts neighbours → **core point**, start a cluster
3. Expand via BFS: absorb reachable core & border points
4. Repeat until all points are visited
5. Remaining points → **noise** (✕)
""")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_param_key = (shape_key, n_points, n_clusters, int(seed), eps, min_samples)

def _run() -> None:
    X, _ = make_dataset(shape_key, n_points=n_points, n_clusters=n_clusters, seed=int(seed))
    st.session_state[_k("snapshots")] = fit(X, eps=eps, min_samples=min_samples)
    st.session_state[_k("step_idx")]  = 0
    st.session_state[_k("playing")]   = False

if _k("snapshots") not in st.session_state or regenerate or st.session_state.get(_k("_param_key")) != _param_key:
    _run()
    st.session_state[_k("_param_key")] = _param_key

snapshots = st.session_state[_k("snapshots")]
n_steps   = len(snapshots)
last      = snapshots[-1]

caption_slot.caption(
    f"Shape: **{shape_name}** | Points: **{n_points}** | "
    f"ε = **{eps}** | minPts = **{min_samples}** | Seed: **{seed}**"
)

with col_main:
    about_section(
        "DBSCAN solved a problem K-means fundamentally can't: finding clusters of "
        "arbitrary shape without knowing the number of clusters in advance, while "
        "flagging outliers as noise instead of forcing every point into a group. "
        "That combination made it a staple for spatial/geographic clustering and "
        "anomaly detection wherever cluster shapes aren't blob-like — it won the "
        "2014 SIGKDD Test of Time Award for its lasting influence. This page's "
        "**Two moons** and **Concentric rings** shapes are the textbook showcase: "
        "DBSCAN traces both correctly using only a neighbourhood radius (ε) and a "
        "density threshold (minPts), with no notion of \"K\" at all.",
        [
            "Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). \"A "
            "Density-Based Algorithm for Discovering Clusters in Large Spatial "
            "Databases with Noise.\" *KDD 1996.*",
        ],
    )

    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clusters found", str(last.n_clusters))
        m2.metric("Noise points",   str(last.n_noise))
        m3.metric("Total frames",   str(n_steps))
        m4.metric("ε / minPts",     f"{eps} / {min_samples}")

    # -------------------------------------------------------------------
    # Playback controls, chart, and auto-advance all live inside one
    # fragment: st.rerun() during autoplay used to fully rerun the whole
    # page (title/caption/params rail/about-section included) several
    # times a second, and small timing differences in how long each of
    # those took to re-render showed up as visible flicker/layout shift
    # on every frame. Scoping the rerun to just this fragment keeps
    # everything above it (and the sidebar) completely static.
    # -------------------------------------------------------------------
    @st.fragment
    def _playback() -> None:
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
            DELAY = {"0.5×": 1.8, "1×": 0.9, "2×": 0.45, "4×": 0.22}[speed]

            col_prev, col_play, col_pause, col_next, col_speed = st.columns([1, 1.2, 1.2, 1, 3])

            with col_prev:
                if st.button("◀ Prev", use_container_width=True, disabled=(step_idx == 0 or playing), key=_k("prev")):
                    st.session_state[_k("step_idx")] = max(0, step_idx - 1)
                    st.rerun(scope="fragment")

            with col_play:
                if st.button("▶  Play", use_container_width=True,
                             disabled=(playing or step_idx == n_steps - 1), type="primary", key=_k("play")):
                    st.session_state[_k("playing")] = True
                    st.rerun(scope="fragment")

            with col_pause:
                if st.button("⏸  Pause", use_container_width=True, disabled=not playing, key=_k("pause")):
                    st.session_state[_k("playing")] = False
                    st.rerun(scope="fragment")

            with col_next:
                if st.button("Next ▶", use_container_width=True,
                             disabled=(step_idx == n_steps - 1 or playing), key=_k("next")):
                    st.session_state[_k("step_idx")] = min(n_steps - 1, step_idx + 1)
                    st.rerun(scope="fragment")

            with col_speed:
                st.caption(f"Speed: **{speed}**  ({DELAY:.2f}s per frame)")

            # Progress bar
            st.progress(step_idx / max(n_steps - 1, 1),
                        text=f"Frame {step_idx + 1} / {n_steps} — {snapshots[step_idx].title}")

        snap  = snapshots[step_idx]
        max_c = max(s.n_clusters for s in snapshots) or 1
        fig   = make_static_figure(snap, max_clusters=max_c)
        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=_k("chart"))

        if snap.phase == "init":
            st.info(
                "**Initialisation** — all points are unvisited (light grey). "
                "DBSCAN scans through them in array order, examining each unvisited point."
            )
        elif snap.phase == "examine":
            n_nbr = len(snap.neighbors) if snap.neighbors is not None else 0
            is_core = n_nbr >= min_samples
            verdict = (
                f"**{n_nbr} neighbours ≥ minPts ({min_samples})** → **core point** ★. "
                f"A new cluster (cluster {snap.cluster_id + 1}) will grow from here via BFS."
                if is_core else
                f"**{n_nbr} neighbours < minPts ({min_samples})** → not a core point. "
                "Tentatively marked noise unless absorbed by a growing cluster."
            )
            st.info(
                f"**Examine** — the dashed ε-circle (radius {eps}) is drawn around the seed point. "
                f"{verdict}"
            )
        elif snap.phase == "expand":
            size = int((snap.labels == snap.cluster_id).sum())
            st.info(
                f"**Expand** — BFS complete for cluster {snap.cluster_id + 1}. "
                f"**{size} points** absorbed: filled = **core**, open ring = **border**."
            )
        else:
            st.success(
                f"**Done** — {snap.n_clusters} cluster{'s' if snap.n_clusters != 1 else ''}, "
                f"{snap.n_noise} noise point{'s' if snap.n_noise != 1 else ''} (grey ✕). "
                "Adjust ε or minPts in the sidebar to explore different results."
            )

        with st.expander("📖 Reading the chart"):
            st.markdown("""
- **ε-circle** (dashed ring) — the neighbourhood radius around the current seed point
- **Filled circles** — core points (≥ minPts neighbours within ε)
- **Open rings** — border points (within ε of a core, but not dense enough to be one)
- **Grey ✕** — noise points (not reachable from any core point)
- Colours are consistent per cluster across all frames
""")

        # Auto-advance (must be last — triggers a fragment-scoped rerun after a delay)
        if playing:
            if step_idx < n_steps - 1:
                time.sleep(DELAY)
                st.session_state[_k("step_idx")] = min(n_steps - 1, step_idx + 1)
                st.rerun(scope="fragment")
            else:
                st.session_state[_k("playing")] = False
                st.rerun(scope="fragment")

    _playback()
