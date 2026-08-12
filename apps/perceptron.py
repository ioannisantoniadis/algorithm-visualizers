"""Perceptron & Gradient Descent page — adapted from the standalone
perceptron-viz repo for the multipage portfolio app. Session-state keys
are namespaced with a per-page prefix since st.session_state is shared
across all pages. This page has two tabs (Perceptron, Gradient Descent),
each with its own playback state — every key below is namespaced through
_k() so the two tabs' widgets and state never collide with each other or
with other pages.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from common.ui import about_section, params_rail
from perceptron.algorithm import (
    accuracy,
    accuracy_nobias,
    gradient_descent_fit,
    loss_grid,
    perceptron_fit,
)
from perceptron.data import make_dataset
from perceptron.visualize import (
    make_gd_contour_figure,
    make_gd_loss_curve_figure,
    make_gd_side_by_side_figure,
    make_perceptron_figure,
)

NS = "perceptron"


def _k(name: str) -> str:
    return f"{NS}__{name}"


st.title("Perceptron & Gradient Descent — Step-by-step Visualiser")
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

# ---------------------------------------------------------------------------
# Params rail — dataset
# ---------------------------------------------------------------------------
with params_rail(col_params, "Dataset"):
    n_points = st.slider("Number of points", min_value=20, max_value=150, value=50, step=5, key=_k("n_points"))
    angle_deg = st.slider("True boundary angle (°)", min_value=0, max_value=180, value=35, step=5, key=_k("angle_deg"))
    separation = st.slider(
        "Class separation", min_value=0.0, max_value=4.0, value=3.5, step=0.1,
        help="Distance between the two class centres, in units of the noise "
             "std. Below ~1.5 the classes overlap so heavily the Perceptron "
             "essentially never converges. Above ~4 most random draws become "
             "linearly separable, though with only a few dozen points some "
             "seeds still land a stray point on the wrong side — that's "
             "sampling noise, not a bug, and it's exactly what the "
             "\"did not converge\" message further down is for.",
        key=_k("separation"),
    )
    noise_std = st.slider("Noise std", min_value=0.3, max_value=1.5, value=0.8, step=0.1, key=_k("noise_std"))
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key=_k("seed"))
    regenerate = st.button("🔄 Re-generate data", use_container_width=True, key=_k("regenerate"))

# ---------------------------------------------------------------------------
# Params rail — perceptron
# ---------------------------------------------------------------------------
with params_rail(col_params, "Perceptron"):
    lr_perceptron = st.slider(
        "Learning rate", min_value=0.1, max_value=2.0, value=1.0, step=0.1,
        help="Scales every mistake-driven update. Because w starts at zero, "
             "every update is simply lr · y · x — so the *entire* trajectory "
             "of signs, mistakes, weight updates and convergence is exactly "
             "the same for any positive learning rate; only the final "
             "weight vector's magnitude (and so the length of the blue "
             "arrow) scales with it. This is a real property of the "
             "Perceptron, not a disabled control — try two different "
             "values and watch the epoch/mistake counters stay identical.",
        key=_k("lr_perceptron"),
    )
    max_epochs = st.slider("Max epochs", min_value=3, max_value=40, value=15, step=1, key=_k("max_epochs"))

# ---------------------------------------------------------------------------
# Params rail — gradient descent
# ---------------------------------------------------------------------------
with params_rail(col_params, "Gradient descent"):
    lr_gd = st.slider("Learning rate (GD / Momentum)", min_value=0.05, max_value=2.0, value=0.5, step=0.05, key=_k("lr_gd"))
    lr_adam = st.slider("Learning rate (Adam)", min_value=0.02, max_value=1.0, value=0.3, step=0.02, key=_k("lr_adam"))
    momentum_beta = st.slider("Momentum β", min_value=0.5, max_value=0.99, value=0.9, step=0.01, key=_k("momentum_beta"))
    n_steps_gd = st.slider("GD steps", min_value=10, max_value=150, value=50, step=5, key=_k("n_steps_gd"))
    init_scale = st.slider(
        "Initial weight scale", min_value=0.5, max_value=4.0, value=2.5, step=0.5,
        help="Both optimisers start from the same random w₀ drawn uniformly in "
             "[-scale, scale]². A larger scale starts further from the optimum.",
        key=_k("init_scale"),
    )
    gd_seed = st.number_input("Initial weight seed", min_value=0, max_value=9999, value=7, step=1, key=_k("gd_seed"))

with col_params:
    with st.expander("How the algorithms work", expanded=False):
        st.markdown(
            """
**Perceptron**

1. Scan the data; for each point check which side of the boundary it's on
2. If misclassified, nudge **w** directly towards that point's label
3. Repeat until a full pass makes zero mistakes (converged) or epochs run out

**Gradient Descent**

1. Define a smooth loss over the weight vector (logistic loss here)
2. Compute the gradient — the direction of steepest *increase*
3. Step opposite the gradient; Momentum and Adam reshape that step using
   the update's own history
"""
        )

# ---------------------------------------------------------------------------
# Session state — dataset
# ---------------------------------------------------------------------------
_data_key = (n_points, angle_deg, separation, noise_std, int(seed))

if (
    _k("X") not in st.session_state
    or regenerate
    or st.session_state.get(_k("_data_key")) != _data_key
):
    X, y, true_normal = make_dataset(
        n_points=n_points, separation=separation, angle_deg=angle_deg,
        noise_std=noise_std, seed=int(seed),
    )
    st.session_state[_k("X")] = X
    st.session_state[_k("y")] = y
    st.session_state[_k("true_normal")] = true_normal
    st.session_state[_k("_data_key")] = _data_key
    st.session_state[_k("perc_snapshots")] = None
    st.session_state[_k("gd_snapshots")] = None

X: np.ndarray = st.session_state[_k("X")]
y: np.ndarray = st.session_state[_k("y")]
true_normal: np.ndarray = st.session_state[_k("true_normal")]

# ---------------------------------------------------------------------------
# Session state — Perceptron run
# ---------------------------------------------------------------------------
_perc_key = (_data_key, lr_perceptron, max_epochs)
if (
    st.session_state.get(_k("perc_snapshots")) is None
    or st.session_state.get(_k("_perc_key")) != _perc_key
):
    st.session_state[_k("perc_snapshots")] = perceptron_fit(
        X, y, lr=lr_perceptron, max_epochs=max_epochs, seed=int(seed),
    )
    st.session_state[_k("_perc_key")] = _perc_key
    st.session_state[_k("perc_step_idx")] = 0
    st.session_state[_k("perc_playing")] = False

perc_snapshots = st.session_state[_k("perc_snapshots")]

# ---------------------------------------------------------------------------
# Session state — Gradient descent run
# ---------------------------------------------------------------------------
_gd_key = (_data_key, lr_gd, lr_adam, momentum_beta, n_steps_gd, init_scale, int(gd_seed))
if (
    st.session_state.get(_k("gd_snapshots")) is None
    or st.session_state.get(_k("_gd_key")) != _gd_key
):
    st.session_state[_k("gd_snapshots")] = gradient_descent_fit(
        X, y, lr=lr_gd, adam_lr=lr_adam, momentum_beta=momentum_beta,
        n_steps=n_steps_gd, init_scale=init_scale, seed=int(gd_seed),
    )
    st.session_state[_k("_gd_key")] = _gd_key
    st.session_state[_k("gd_step_idx")] = 0
    st.session_state[_k("gd_playing")] = False

gd_snapshots = st.session_state[_k("gd_snapshots")]


def _weight_range(snapshots, pad_frac: float = 0.35):
    pts = np.vstack([snapshots[-1].path_vanilla, snapshots[-1].path_momentum, snapshots[-1].path_adam])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1.0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    r = (span / 2) * (1 + pad_frac)
    return (float(cx - r), float(cx + r)), (float(cy - r), float(cy + r))


w1_range, w2_range = _weight_range(gd_snapshots)
w1s, w2s, Z = loss_grid(X, y, w1_range, w2_range, resolution=60)

caption_slot.caption(
    f"Points: **{n_points}** | Separation: **{separation}** | "
    f"Angle: **{angle_deg}°** | Seed: **{seed}**"
)

with col_main:
    about_section(
        "The perceptron (1958) is the direct ancestor of every neural network "
        "elsewhere on this site — Backpropagation, Transformer Self-Attention, "
        "the VAE — it's the single artificial neuron. Its famous limitation, "
        "that it can't learn XOR or any other non-linearly-separable pattern, "
        "is precisely why hidden layers and backpropagation were developed. "
        "The Gradient Descent tab pairs that same neuron with the family of "
        "optimisers — vanilla SGD, Momentum, Adam — that train virtually every "
        "modern deep learning model, letting you watch directly why Adam "
        "typically reaches a low-loss region faster on the *identical* loss "
        "surface the other two are struggling across.",
        [
            "Rosenblatt, F. (1958). \"The Perceptron: A Probabilistic Model for "
            "Information Storage and Organization in the Brain.\" "
            "*Psychological Review.*",
            "Kingma, D.P. & Ba, J. (2015). "
            "[\"Adam: A Method for Stochastic Optimization.\"](https://arxiv.org/abs/1412.6980) "
            "arXiv:1412.6980.",
        ],
    )

    tab_perceptron, tab_gd = st.tabs(["🔷 Perceptron", "📉 Gradient Descent"])

    # =========================================================================
    # TAB 1 — Perceptron
    # =========================================================================
    with tab_perceptron:
        n_steps_p = len(perc_snapshots)

        # -----------------------------------------------------------
        # Lives inside a fragment: st.rerun() during autoplay used to
        # fully rerun the whole page (title/caption/params rail/about-
        # section included) several times a second, and small timing
        # differences in how long each of those took to re-render
        # showed up as visible flicker/layout shift on every frame.
        # Scoping the rerun to just this fragment keeps everything
        # above it (and the sidebar) completely static.
        # -----------------------------------------------------------
        @st.fragment
        def _perceptron_playback() -> None:
            step_idx_p: int = st.session_state.get(_k("perc_step_idx"), 0)
            step_idx_p = max(0, min(step_idx_p, n_steps_p - 1))
            st.session_state[_k("perc_step_idx")] = step_idx_p
            playing_p: bool = st.session_state.get(_k("perc_playing"), False)

            show_true_boundary = st.checkbox(
                "Show true generating boundary", value=False,
                help="The line actually used to generate the data — compare it to "
                     "what the Perceptron has learned so far.",
                key=_k("show_true_boundary"),
            )

            with st.container(border=True):
                speed_p = st.select_slider(
                    "Playback speed", options=["0.5×", "1×", "2×", "4×"], value="2×",
                    key=_k("speed_perceptron"), label_visibility="collapsed",
                )
                delay_p = {"0.5×": 1.2, "1×": 0.6, "2×": 0.3, "4×": 0.12}[speed_p]

                c1, c2, c3, c4, c5 = st.columns([1, 1.2, 1.2, 1, 3])
                with c1:
                    if st.button("◀ Prev", use_container_width=True, key=_k("prev_p"),
                                 disabled=(step_idx_p == 0 or playing_p)):
                        st.session_state[_k("perc_step_idx")] = max(0, step_idx_p - 1)
                        st.rerun(scope="fragment")
                with c2:
                    if st.button("▶  Play", use_container_width=True, key=_k("play_p"), type="primary",
                                 disabled=(playing_p or step_idx_p == n_steps_p - 1)):
                        st.session_state[_k("perc_playing")] = True
                        st.rerun(scope="fragment")
                with c3:
                    if st.button("⏸  Pause", use_container_width=True, key=_k("pause_p"), disabled=not playing_p):
                        st.session_state[_k("perc_playing")] = False
                        st.rerun(scope="fragment")
                with c4:
                    if st.button("Next ▶", use_container_width=True, key=_k("next_p"),
                                 disabled=(step_idx_p == n_steps_p - 1 or playing_p)):
                        st.session_state[_k("perc_step_idx")] = min(n_steps_p - 1, step_idx_p + 1)
                        st.rerun(scope="fragment")
                with c5:
                    st.caption(f"Speed: **{speed_p}**  ({delay_p:.2f}s per frame)")

                st.progress(
                    step_idx_p / max(n_steps_p - 1, 1),
                    text=f"Frame {step_idx_p + 1} / {n_steps_p} — {perc_snapshots[step_idx_p].title}",
                )

            snap_p = perc_snapshots[step_idx_p]
            acc = accuracy(snap_p.weights, X, y)

            with st.container(border=True):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Epoch", str(snap_p.epoch))
                m2.metric("Weight updates", str(snap_p.n_updates))
                if snap_p.phase == "epoch_end":
                    m3.metric("Mistakes this epoch", str(snap_p.n_mistakes_epoch))
                else:
                    m3.metric("Mistakes so far (epoch)", str(snap_p.n_mistakes_epoch))
                m4.metric("Training accuracy", f"{100 * acc:.1f}%")

            fig_p = make_perceptron_figure(snap_p, true_normal=true_normal, show_true_boundary=show_true_boundary)
            with st.container(border=True):
                st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False}, key=_k("chart_perceptron"))

            if snap_p.phase == "init":
                st.info("**Initialisation** — weights start at zero, so there is no boundary yet.")
            elif snap_p.phase == "update":
                st.info(
                    f"**Mistake-driven update** — point **#{snap_p.point_idx}** was on the wrong "
                    f"side of the boundary, so **w ← w + lr · y · x** rotates the boundary "
                    f"towards correctly classifying it."
                )
            else:
                if snap_p.converged:
                    st.success(
                        f"**Epoch {snap_p.epoch} complete** — a full pass made **zero mistakes**. "
                        "The Perceptron has converged: this boundary separates the classes perfectly."
                    )
                else:
                    st.warning(
                        f"**Epoch {snap_p.epoch} complete** — **{snap_p.n_mistakes_epoch}** "
                        f"mistake(s) remain. Continuing to the next epoch."
                    )

            if step_idx_p == n_steps_p - 1 and not snap_p.converged:
                st.error(
                    f"**Did not converge within {max_epochs} epochs.** The Perceptron "
                    "convergence theorem only guarantees a finite number of updates when "
                    "the data is linearly separable — increase separation, raise the "
                    "epoch budget, or accept that no straight line perfectly splits this data."
                )

            with st.expander("📖 Reading the animation"):
                st.markdown("""
- **Gold ring** = the point that was just misclassified and triggered an update
- **Black line** = current decision boundary; **blue arrow** = weight vector **w**,
  always perpendicular to the boundary by construction
- **◀ Prev / Next ▶** to step manually, **▶ Play** to auto-advance through every update
- Only *updates* and *epoch summaries* are recorded as frames — correctly
  classified points don't move the boundary, so they don't get a frame
""")

            # Auto-advance (must be last — triggers a fragment-scoped rerun after a delay)
            if playing_p:
                if step_idx_p < n_steps_p - 1:
                    time.sleep(delay_p)
                    st.session_state[_k("perc_step_idx")] = min(n_steps_p - 1, step_idx_p + 1)
                    st.rerun(scope="fragment")
                else:
                    st.session_state[_k("perc_playing")] = False
                    st.rerun(scope="fragment")

        _perceptron_playback()

    # =========================================================================
    # TAB 2 — Gradient Descent
    # =========================================================================
    with tab_gd:
        n_steps_g = len(gd_snapshots)

        # -----------------------------------------------------------
        # Lives inside a fragment: st.rerun() during autoplay used to
        # fully rerun the whole page (title/caption/params rail/about-
        # section included) several times a second, and small timing
        # differences in how long each of those took to re-render
        # showed up as visible flicker/layout shift on every frame.
        # Scoping the rerun to just this fragment keeps everything
        # above it (and the sidebar) completely static.
        # -----------------------------------------------------------
        @st.fragment
        def _gd_playback() -> None:
            step_idx_g: int = st.session_state.get(_k("gd_step_idx"), 0)
            step_idx_g = max(0, min(step_idx_g, n_steps_g - 1))
            st.session_state[_k("gd_step_idx")] = step_idx_g
            playing_g: bool = st.session_state.get(_k("gd_playing"), False)

            show_side_by_side = st.checkbox("Show side-by-side comparison panels", value=True, key=_k("show_side_by_side"))

            with st.container(border=True):
                speed_g = st.select_slider(
                    "Playback speed", options=["0.5×", "1×", "2×", "4×"], value="2×",
                    key=_k("speed_gd"), label_visibility="collapsed",
                )
                delay_g = {"0.5×": 0.5, "1×": 0.25, "2×": 0.12, "4×": 0.06}[speed_g]

                c1, c2, c3, c4, c5 = st.columns([1, 1.2, 1.2, 1, 3])
                with c1:
                    if st.button("◀ Prev", use_container_width=True, key=_k("prev_g"),
                                 disabled=(step_idx_g == 0 or playing_g)):
                        st.session_state[_k("gd_step_idx")] = max(0, step_idx_g - 1)
                        st.rerun(scope="fragment")
                with c2:
                    if st.button("▶  Play", use_container_width=True, key=_k("play_g"), type="primary",
                                 disabled=(playing_g or step_idx_g == n_steps_g - 1)):
                        st.session_state[_k("gd_playing")] = True
                        st.rerun(scope="fragment")
                with c3:
                    if st.button("⏸  Pause", use_container_width=True, key=_k("pause_g"), disabled=not playing_g):
                        st.session_state[_k("gd_playing")] = False
                        st.rerun(scope="fragment")
                with c4:
                    if st.button("Next ▶", use_container_width=True, key=_k("next_g"),
                                 disabled=(step_idx_g == n_steps_g - 1 or playing_g)):
                        st.session_state[_k("gd_step_idx")] = min(n_steps_g - 1, step_idx_g + 1)
                        st.rerun(scope="fragment")
                with c5:
                    st.caption(f"Speed: **{speed_g}**  ({delay_g:.2f}s per frame)")

                st.progress(
                    step_idx_g / max(n_steps_g - 1, 1),
                    text=f"Frame {step_idx_g + 1} / {n_steps_g} — {gd_snapshots[step_idx_g].title}",
                )

            snap_g = gd_snapshots[step_idx_g]

            with st.container(border=True):
                m1, m2, m3 = st.columns(3)
                m1.metric("Vanilla GD loss", f"{snap_g.loss_vanilla:.4f}",
                          help=f"‖∇L‖ = {snap_g.grad_norm_vanilla:.4f}")
                m2.metric("Momentum loss", f"{snap_g.loss_momentum:.4f}",
                          help=f"‖∇L‖ = {snap_g.grad_norm_momentum:.4f}")
                m3.metric("Adam loss", f"{snap_g.loss_adam:.4f}",
                          help=f"‖∇L‖ = {snap_g.grad_norm_adam:.4f}")

                a1, a2, a3 = st.columns(3)
                a1.metric("Vanilla GD accuracy", f"{100 * accuracy_nobias(snap_g.w_vanilla, X, y):.1f}%")
                a2.metric("Momentum accuracy", f"{100 * accuracy_nobias(snap_g.w_momentum, X, y):.1f}%")
                a3.metric("Adam accuracy", f"{100 * accuracy_nobias(snap_g.w_adam, X, y):.1f}%")

            with st.container(border=True):
                fig_contour = make_gd_contour_figure(snap_g, w1s, w2s, Z)
                st.plotly_chart(fig_contour, use_container_width=True, config={"displayModeBar": False}, key=_k("chart_contour"))

                if show_side_by_side:
                    st.markdown("##### Side-by-side comparison")
                    fig_sbs = make_gd_side_by_side_figure(snap_g, w1s, w2s, Z)
                    st.plotly_chart(fig_sbs, use_container_width=True, config={"displayModeBar": False}, key=_k("chart_sbs"))

                fig_loss = make_gd_loss_curve_figure(gd_snapshots, step_idx_g)
                st.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": False}, key=_k("chart_loss"))

            if step_idx_g == 0:
                st.info(
                    "**Same starting point, three update rules.** All three optimisers "
                    "begin from the identical random w₀ — any difference in path from "
                    "here on comes purely from the update rule, not luck."
                )
            else:
                fastest = min(
                    [("Vanilla GD", snap_g.loss_vanilla), ("Momentum", snap_g.loss_momentum),
                     ("Adam", snap_g.loss_adam)],
                    key=lambda t: t[1],
                )
                st.info(f"**Step {snap_g.step}** — lowest loss so far: **{fastest[0]}** ({fastest[1]:.4f}).")

            with st.expander("📖 Reading the animation"):
                st.markdown("""
- Background colour = logistic loss at that (w₁, w₂); darker/purple is lower loss
- **Diamond markers** = each optimiser's current position; the trailing line is its path so far
- **Momentum** accumulates velocity, so it can overshoot the minimum and oscillate
- **Adam** rescales each coordinate by its own gradient history, so it can move
  quickly even where the surface is nearly flat
- With cleanly separable data the loss has **no finite minimiser** — ‖w‖ grows
  and the loss keeps inching towards zero, which is why every path keeps drifting
  outward instead of settling at a point
""")

            # Auto-advance (must be last — triggers a fragment-scoped rerun after a delay)
            if playing_g:
                if step_idx_g < n_steps_g - 1:
                    time.sleep(delay_g)
                    st.session_state[_k("gd_step_idx")] = min(n_steps_g - 1, step_idx_g + 1)
                    st.rerun(scope="fragment")
                else:
                    st.session_state[_k("gd_playing")] = False
                    st.rerun(scope="fragment")

        _gd_playback()
