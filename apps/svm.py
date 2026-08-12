"""SVM page — adapted from the standalone svm-viz repo for the
multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from common.ui import about_section, params_rail
from svm.algorithm import default_gamma, fit
from svm.data import SHAPE_KEYS, SHAPE_NAMES, make_dataset
from svm.visualize import make_loss_figure, make_static_figure

NS = "svm"


def _k(name: str) -> str:
    return f"{NS}__{name}"


st.title("Support Vector Machine — Step-by-step Visualiser")
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

# ---------------------------------------------------------------------------
# Params rail — data
# ---------------------------------------------------------------------------
with params_rail(col_params, "Data"):
    shape_name = st.selectbox(
        "Data shape",
        options=SHAPE_NAMES,
        index=0,
        help="Choose the geometry of the training data. Non-linear shapes reveal "
             "why the kernel trick matters.",
        key=_k("shape"),
    )
    shape_key = SHAPE_KEYS[SHAPE_NAMES.index(shape_name)]

    n_points = st.slider("Number of points", min_value=40, max_value=300, value=120, step=10, key=_k("n_points"))
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key=_k("seed"))
    regenerate = st.button("🔄 Re-generate data", use_container_width=True, key=_k("regen"))

# ---------------------------------------------------------------------------
# Params rail — model
# ---------------------------------------------------------------------------
with params_rail(col_params, "Model"):
    kernel_name = st.selectbox("Kernel", ["Linear", "RBF", "Polynomial"], index=0, key=_k("kernel"))
    kernel_key = {"Linear": "linear", "RBF": "rbf", "Polynomial": "poly"}[kernel_name]

    C = st.select_slider(
        "C (soft-margin penalty)",
        options=[0.01, 0.1, 1.0, 10.0, 100.0],
        value=1.0,
        help="Small C tolerates margin violations for a wider margin. "
             "Large C punishes every violation, favouring a tighter fit. "
             "Very large C on overlapping data can need hundreds of sweeps "
             "to actually converge with this simplified random-pair SMO — "
             "watch the α-pairs-updated counter and the convergence banner "
             "below the chart; a non-converged frame is a mid-training "
             "snapshot, not the model's final answer.",
        key=_k("C"),
    )

    gamma: float | None = None

    auto_gamma = st.checkbox(
        "Auto γ  (1 / 2·Var[X])",
        value=True,
        disabled=kernel_key == "linear",
        help="Only used by RBF and Polynomial kernels — the linear kernel "
             "has no γ term." if kernel_key == "linear" else
             "Use the data-driven default kernel width instead of setting one manually.",
        key=_k("auto_gamma"),
    )
    gamma_slider_disabled = kernel_key == "linear" or auto_gamma
    manual_gamma = st.slider(
        "γ (kernel width)", min_value=0.05, max_value=5.0, value=0.5, step=0.05,
        disabled=gamma_slider_disabled,
        help="Only applies when Auto γ is off and the kernel is RBF or Polynomial."
             if gamma_slider_disabled else
             "Larger γ → each point's influence shrinks to a tighter radius "
             "(more overfitting-prone); smaller γ → smoother, more global boundary.",
        key=_k("manual_gamma"),
    )
    if kernel_key in ("rbf", "poly") and not auto_gamma:
        gamma = manual_gamma

    degree_disabled = kernel_key != "poly"
    degree = st.slider(
        "Degree", min_value=2, max_value=5, value=3, disabled=degree_disabled,
        help="Only applies to the Polynomial kernel." if degree_disabled else
             "Degree of the polynomial expansion — degree 2 already resolves XOR.",
        key=_k("degree"),
    )
    coef0 = st.slider(
        "coef0", min_value=0.0, max_value=3.0, value=1.0, step=0.1, disabled=degree_disabled,
        help="Only applies to the Polynomial kernel." if degree_disabled else
             "Additive constant inside the polynomial — shifts the influence "
             "of lower-order interaction terms.",
        key=_k("coef0"),
    )

    max_epochs = st.slider(
        "Max iterations (sweeps)", min_value=5, max_value=150, value=30,
        help="Hard cap on SMO sweeps over the data. Simple shapes converge "
             "in well under 30; non-linear shapes and large C values can "
             "need 100+ — raise this if the run hits the cap before the "
             "'Converged' banner appears.",
        key=_k("max_epochs"),
    )

with col_params:
    _SHAPE_NOTES = {
        "linear_separable": "**Linearly separable** — a wide gap between classes. "
                             "Even the linear kernel finds a hard-margin-like boundary "
                             "with almost no support vectors needed.",
        "linear_overlap":   "**Overlapping classes** — the blobs bleed into each other. "
                             "No boundary gets everything right; watch the soft margin "
                             "(dashed lines) admit violators (red-ringed points).",
        "moons":             "**Two moons** — a linear kernel is stuck near chance level "
                             "on this shape. Switch to RBF to see the boundary curve "
                             "around each crescent.",
        "circles":           "**Concentric circles** — radially symmetric classes. "
                             "RBF wraps a closed loop around the inner class; a linear "
                             "kernel cannot do better than a coin flip.",
        "xor":               "**XOR pattern** — diagonal corners share a class. No "
                             "straight line separates them, but a degree-2 polynomial "
                             "kernel (which can represent x₁·x₂) separates them almost "
                             "perfectly — the residual errors are the Gaussian noise "
                             "baked into the dataset, not a kernel limitation.",
    }
    st.info(_SHAPE_NOTES[shape_key])

    with st.expander("How training works", expanded=False):
        st.markdown("""
1. Start with **α = 0** for every point (no support vectors)
2. Each sweep scans for points violating the **KKT** optimality conditions
3. Violating pairs are updated **jointly** so Σ αᵢyᵢ = 0 stays satisfied
4. Points that end with **α > 0** are the support vectors — everything
   else could be deleted without changing the boundary
""")

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _generate_data() -> None:
    X, y = make_dataset(shape_key, n_points=n_points, seed=int(seed))
    st.session_state[_k("X")] = X
    st.session_state[_k("y")] = y
    st.session_state[_k("snapshots")] = None
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


def _run_svm() -> None:
    X, y = st.session_state[_k("X")], st.session_state[_k("y")]
    g = gamma if gamma is not None else default_gamma(X)
    snapshots = fit(
        X, y,
        kernel=kernel_key, C=float(C), gamma=g, degree=int(degree), coef0=float(coef0),
        max_epochs=int(max_epochs), seed=int(seed),
    )
    st.session_state[_k("snapshots")] = snapshots
    st.session_state[_k("resolved_gamma")] = g
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


_param_key = (shape_key, n_points, int(seed), kernel_key, C, gamma, degree, coef0, max_epochs)
if _k("X") not in st.session_state or regenerate or st.session_state.get(_k("_param_key")) != _param_key:
    _generate_data()
    _run_svm()
    st.session_state[_k("_param_key")] = _param_key

snapshots = st.session_state.get(_k("snapshots"))
if not snapshots:
    st.stop()

n_steps = len(snapshots)
resolved_gamma = st.session_state.get(_k("resolved_gamma"), 1.0)

caption_slot.caption(
    f"Shape: **{shape_name}** | Points: **{n_points}** | Kernel: **{kernel_name}** | "
    f"C: **{C}**" + (f" | γ: **{resolved_gamma:.3f}**" if kernel_key != "linear" else "") +
    (f" | degree: **{degree}** | coef0: **{coef0}**" if kernel_key == "poly" else "") +
    f" | Seed: **{seed}**"
)

with col_main:
    about_section(
        "SVMs dominated classification tasks through the 2000s and early "
        "2010s, before deep learning overtook them on large datasets, because "
        "of a clean theoretical guarantee: the maximum-margin boundary they "
        "find generalises provably well. The kernel this page lets you switch "
        "between — Linear, RBF, Polynomial — is the technique's real "
        "innovation, the *kernel trick*: it lets a fundamentally linear "
        "algorithm draw non-linear boundaries by implicitly working in a "
        "higher-dimensional space, without ever computing that projection "
        "directly. Try the **XOR** shape on Linear, then switch to "
        "**Polynomial** — that single change is the kernel trick's whole "
        "point, made visible.",
        [
            "Cortes, C. & Vapnik, V. (1995). \"Support-Vector Networks.\" "
            "*Machine Learning*, 20(3), 273–297.",
            "Platt, J. (1998). \"Sequential Minimal Optimization: A Fast "
            "Algorithm for Training Support Vector Machines.\" (the SMO "
            "algorithm this page's training loop implements)",
        ],
    )

    # -------------------------------------------------------------------
    # Playback controls, charts, metrics, and auto-advance all live inside
    # one fragment: st.rerun() during autoplay used to fully rerun the
    # whole page (title/caption/params rail/about-section included)
    # several times a second, and small timing differences in how long
    # each of those took to re-render showed up as visible flicker/layout
    # shift on every frame. Scoping the rerun to just this fragment keeps
    # everything above it (and the sidebar) completely static.
    # -------------------------------------------------------------------
    @st.fragment
    def _playback() -> None:
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
            DELAY = {"0.5×": 1.2, "1×": 0.6, "2×": 0.3, "4×": 0.15}[speed]

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

            st.progress(step_idx / max(n_steps - 1, 1),
                        text=f"Frame {step_idx + 1} / {n_steps} — {snapshots[step_idx].title}")

        snap = snapshots[step_idx]

        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sweep", str(snap.iteration))
            m2.metric("Support vectors", str(snap.n_support))
            m3.metric("Dual objective", f"{snap.dual_objective:.2f}")
            if snap.converged:
                m4.metric("Status", "Converged ✓")
            elif snap.iteration > 0:
                m4.metric("α pairs updated", str(snap.n_changed))
            else:
                m4.metric("Status", "Not started")

        with st.container(border=True):
            chart_main, chart_loss = st.columns([3, 1.3])
            with chart_main:
                fig = make_static_figure(snap, kernel_key, resolved_gamma, degree, coef0)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=_k("chart_main"))
            with chart_loss:
                st.caption("SMO dual objective (Σα − ½αᵀQα)")
                loss_fig = make_loss_figure(snapshots, step_idx)
                st.plotly_chart(loss_fig, use_container_width=True, config={"displayModeBar": False}, key=_k("chart_loss"))
                st.caption(
                    "Rises monotonically by construction. By strong duality its maximum "
                    "equals the minimised primal hinge loss, so this climb mirrors the "
                    "hinge loss falling during gradient-based training."
                )

        if snap.iteration == 0:
            st.info(
                "**Initialisation** — every α is 0, so the decision function is flat "
                "(no support vectors yet). Press **▶ Play** or **Next ▶** to start training."
            )
        elif snap.n_changed > 0:
            pair = snap.active_pair
            pair_txt = f" (last pair updated: points {pair[0]} & {pair[1]})" if pair else ""
            st.info(
                f"**Sweep {snap.iteration}** — scanned every point for KKT violations and "
                f"updated **{snap.n_changed}** α pair(s){pair_txt}. Support vectors and the "
                f"margin band shift as α changes."
            )
        else:
            status = "**Converged** — three sweeps in a row changed nothing." if snap.converged \
                else "No violations found this sweep, but convergence isn't confirmed yet."
            st.info(f"**Sweep {snap.iteration}** — no α pairs needed updating. {status}")

        if step_idx == n_steps - 1 and not snap.converged:
            st.warning(
                f"**Reached the {max_epochs}-sweep cap without converging** — this is a "
                "mid-training snapshot, not the model's settled answer. Raise **Max "
                "iterations** in the sidebar (or lower **C**) to let SMO finish; this "
                "simplified solver picks its second index at random rather than by the "
                "maximum-violation heuristic real solvers use, so it can need many more "
                "sweeps on large-C or heavily overlapping data."
            )

        with st.expander("📖 Reading the animation"):
            st.markdown("""
- **Shaded regions** = predicted class (indigo = +1, rose = −1); intensity fades near the boundary
- **Solid black line** = decision boundary (f(x) = 0)
- **Dashed lines** = the margin (f(x) = ±1) — the "street" the SVM tries to keep empty
- **Thick black border** on a point = it's a **support vector** (α > 0)
- **Red ring** on a point = it's a **margin violation** (ξ > 0) — inside the street or misclassified
- **◀ Prev / Next ▶** to step manually, **▶ Play** to auto-advance through SMO sweeps
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
