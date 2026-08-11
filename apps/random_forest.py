"""Random Forest page — adapted from the standalone random-forest-viz repo for
the multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import streamlit as st

from common.ui import about_section, params_rail
from random_forest.algorithm import fit
from random_forest.data import SHAPE_KEYS, SHAPE_NAMES, make_dataset
from random_forest.visualize import (
    make_error_figure,
    make_importance_figure,
    make_static_figure,
)

NS = "random_forest"


def _k(name: str) -> str:
    return f"{NS}__{name}"


st.title("Random Forest — Tree-by-tree Visualiser")
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

# ---------------------------------------------------------------------------
# Params rail — data
# ---------------------------------------------------------------------------
with params_rail(col_params, "Data"):
    shape_name = st.selectbox(
        "Data shape",
        options=SHAPE_NAMES,
        index=1,
        help="Choose the geometry of the training data. Checkerboard is the "
             "best case for axis-aligned splits; spiral is the hardest.",
        key=_k("shape"),
    )
    shape_key = SHAPE_KEYS[SHAPE_NAMES.index(shape_name)]

    n_points = st.slider("Number of points", min_value=60, max_value=400, value=220, step=10, key=_k("n_points"))
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key=_k("seed"))
    regenerate = st.button("🔄 Re-generate data", use_container_width=True, key=_k("regen"))

# ---------------------------------------------------------------------------
# Params rail — forest
# ---------------------------------------------------------------------------
with params_rail(col_params, "Forest parameters"):
    n_estimators = st.slider(
        "Number of trees", min_value=1, max_value=60, value=25,
        help="Each new tree is trained on its own bootstrap resample and adds one vote.",
        key=_k("n_estimators"),
    )
    max_depth = st.slider(
        "Max tree depth", min_value=1, max_value=10, value=6,
        help="Deeper trees fit the training data more tightly — lower bias, higher variance per tree.",
        key=_k("max_depth"),
    )
    min_samples_leaf = st.slider(
        "Min samples per leaf", min_value=1, max_value=20, value=3,
        help="Minimum samples required on both sides of a split. Set this "
             "and the point count so high relative to each other and every "
             "tree degenerates into a single-leaf stump — a real, not a "
             "buggy, consequence of the constraint.",
        key=_k("min_samples_leaf"),
    )

    feature_mode = st.radio(
        "Feature sampling per split",
        ["1 feature (Random Forest)", "Both features (Bagging only)"],
        help="Restricting each split to 1 random feature decorrelates the "
             "trees — provably true, watch how much less alike they look. "
             "But with only 2 features total, that decorrelation costs real "
             "per-split accuracy, so out-of-bag error is often *not* lower "
             "with 1 feature than with both here — the bias/variance trade-off "
             "genuinely can go either way, which is itself worth seeing.",
        key=_k("feature_mode"),
    )
    max_features = 1 if feature_mode.startswith("1") else 2

    bootstrap_frac = st.slider(
        "Bootstrap sample fraction", min_value=0.3, max_value=1.0, value=1.0, step=0.05,
        help="Fraction of N resampled (with replacement) per tree. Lower values "
             "starve each tree of data but leave more points out-of-bag.",
        key=_k("bootstrap_frac"),
    )

with col_params:
    _SHAPE_NOTES = {
        "blobs":        "**Three blobs** — trivially separable. A single shallow "
                        "tree already gets this right; ensembling mostly smooths "
                        "the corners between blobs.",
        "checkerboard": "**Checkerboard** — a grid of alternating classes is "
                        "*literally* a union of axis-aligned rectangles, the "
                        "exact shape a CART split represents perfectly. The best "
                        "case for a tree-based model — but only once the trees "
                        "are given enough depth to resolve all the cells.",
        "moons":        "**Two moons** — a smooth curve. No axis-aligned split "
                        "matches it exactly; a tree approximates it with a "
                        "staircase of boxes that gets finer (and noisier) as "
                        "depth increases.",
        "circles":      "**Concentric circles** — a closed, radially symmetric "
                        "boundary. The tree has to curve around the origin in "
                        "every direction at once.",
        "spiral":       "**Three-arm spiral** — genuinely hard. Too few trees or "
                        "too little depth underfits into blunt rectangles; watch "
                        "how many trees it takes before the arms emerge.",
    }
    st.info(_SHAPE_NOTES[shape_key])

    with st.expander("How the forest grows", expanded=False):
        st.markdown("""
1. Each tree trains on its own **bootstrap resample** (~63% unique points, ~37% left out)
2. Each **split** considers only a random subset of features
3. The tree's own boundary is jagged and overfit to its resample
4. The **majority vote** across all trees cancels out each tree's independent noise
""")

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _generate_data() -> None:
    X, y = make_dataset(shape_key, n_points=n_points, seed=int(seed))
    st.session_state[_k("X")] = X
    st.session_state[_k("y")] = y
    st.session_state[_k("run")] = None
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


def _run_forest() -> None:
    X, y = st.session_state[_k("X")], st.session_state[_k("y")]
    run = fit(
        X, y,
        n_estimators=int(n_estimators), max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf), max_features=int(max_features),
        bootstrap_frac=float(bootstrap_frac), seed=int(seed),
    )
    st.session_state[_k("run")] = run
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


_param_key = (shape_key, n_points, int(seed), n_estimators, max_depth,
              min_samples_leaf, max_features, bootstrap_frac)
if _k("X") not in st.session_state or regenerate or st.session_state.get(_k("_param_key")) != _param_key:
    _generate_data()
    _run_forest()
    st.session_state[_k("_param_key")] = _param_key

run = st.session_state.get(_k("run"))
if run is None:
    st.stop()

snapshots = run.snapshots
n_steps = len(snapshots)

caption_slot.caption(
    f"Shape: **{shape_name}** | Points: **{n_points}** | Trees: **{n_estimators}** | "
    f"Max depth: **{max_depth}** | Feature sampling: **{feature_mode}** | "
    f"Bootstrap fraction: **{bootstrap_frac}** | Seed: **{seed}**"
)

with col_main:
    about_section(
        "Random Forest remains one of the strongest \"just works\" algorithms "
        "for tabular data even in the deep learning era — it's often the first "
        "model data scientists reach for before trying anything fancier. Its "
        "two ideas — bagging (train each tree on its own bootstrap resample) "
        "and feature subsampling (restrict each split to a random subset of "
        "features) — both exist purely to *decorrelate* the trees, so their "
        "individual mistakes cancel out on average instead of reinforcing each "
        "other. This page's out-of-bag error is the same trick the algorithm "
        "uses internally: since each tree never sees its own ~37% left-out "
        "points, their combined votes give an honest estimate of "
        "generalisation error for free, with no held-out test set needed.",
        [
            "Breiman, L. (2001). \"Random Forests.\" *Machine Learning*, "
            "45(1), 5–32.",
        ],
    )

    # -------------------------------------------------------------------
    # Playback controls (pure Streamlit — no Plotly updatemenus)
    # -------------------------------------------------------------------
    step_idx: int = st.session_state.get(_k("step_idx"), 0)
    step_idx = max(0, min(step_idx, n_steps - 1))
    st.session_state[_k("step_idx")] = step_idx
    playing: bool = st.session_state.get(_k("playing"), False)

    with st.container(border=True):
        speed = st.select_slider(
            "Playback speed",
            options=["0.5×", "1×", "2×", "4×"],
            value="1×",
            label_visibility="collapsed",
            key=_k("speed"),
        )
        DELAY = {"0.5×": 1.0, "1×": 0.5, "2×": 0.25, "4×": 0.12}[speed]

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

    # -------------------------------------------------------------------
    # Metric cards + main decision-boundary panels
    # -------------------------------------------------------------------
    snap = snapshots[step_idx]

    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trees so far", str(snap.tree_index))
        m2.metric("This tree's depth / leaves", f"{snap.tree_depth} / {snap.n_leaves}")
        m3.metric("Training error", f"{100 * (1 - snap.train_accuracy):.1f}%")
        m4.metric("Out-of-bag error", f"{100 * snap.oob_error:.1f}%" if snap.oob_error is not None else "—")

    with st.container(border=True):
        st.plotly_chart(
            make_static_figure(snap, run.n_classes, run.grid_x, run.grid_y),
            use_container_width=True, config={"displayModeBar": False}, key=_k("chart_main"),
        )

        if snap.tree_index == 1:
            st.info(
                f"**Tree 1** — trained on a bootstrap resample of the data "
                f"(**{snap.n_oob_points}** points never selected, shown as hollow "
                f"markers). Its own boundary (left) is already the ensemble's "
                f"boundary (right) — nothing to average yet."
            )
        else:
            st.info(
                f"**Tree {snap.tree_index}** — a fresh bootstrap resample "
                f"(**{snap.n_oob_points}** points out-of-bag this time) grows a new, "
                f"differently-shaped tree (left). Averaged with the {snap.tree_index - 1} "
                f"trees before it, the ensemble boundary (right) keeps smoothing as "
                f"independent mistakes cancel out."
            )

    # -------------------------------------------------------------------
    # Error curves + feature importance
    # -------------------------------------------------------------------
    col_err, col_imp = st.columns([1.6, 1])
    with col_err:
        with st.container(border=True):
            st.caption("Training error (resubstitution) vs. out-of-bag error")
            st.plotly_chart(
                make_error_figure(snapshots, step_idx),
                use_container_width=True, config={"displayModeBar": False}, key=_k("chart_error"),
            )
            st.caption(
                "Training error is measured on the same points the trees were fit "
                "on, so it keeps falling — sometimes toward pure memorisation. "
                "Out-of-bag error only counts each point's votes from trees that "
                "never saw it during training, so it is a genuinely honest estimate "
                "of generalisation error, for free, with no held-out set."
            )
    with col_imp:
        with st.container(border=True):
            st.caption("Cumulative feature importance")
            st.plotly_chart(
                make_importance_figure(snap),
                use_container_width=True, config={"displayModeBar": False}, key=_k("chart_importance"),
            )
            st.caption(
                "Total impurity decrease credited to each feature, summed over "
                "every split in every tree so far and normalised to sum to 1."
            )

    with st.expander("📖 Reading the animation"):
        st.markdown("""
- **Left panel** — the tree just added, evaluated alone. Solid-bordered points were in its bootstrap sample; faint hollow points were left out-of-bag.
- **Right panel** — the majority vote of every tree added so far. This is the actual Random Forest prediction.
- **Error chart** — indigo = training error (optimistic), teal = out-of-bag error (honest). Watch the gap between them as depth/tree-count change.
- **Importance chart** — which feature the forest has leaned on so far; updates every tree.
- **◀ Prev / Next ▶** to step tree by tree, **▶ Play** to auto-advance through the whole forest.
""")

    # -------------------------------------------------------------------
    # Auto-advance (must be last — triggers rerun after a delay)
    # -------------------------------------------------------------------
    if playing:
        if step_idx < n_steps - 1:
            time.sleep(DELAY)
            st.session_state[_k("step_idx")] = min(n_steps - 1, step_idx + 1)
            st.rerun()
        else:
            st.session_state[_k("playing")] = False
            st.rerun()
