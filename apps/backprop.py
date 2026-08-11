"""Backpropagation page — adapted from the standalone backprop-viz repo for
the multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import streamlit as st

from backprop.algorithm import fit
from backprop.data import SHAPE_KEYS, SHAPE_NAMES, make_dataset
from backprop.visualize import (
    make_activation_heatmap_figure,
    make_boundary_figure,
    make_loss_figure,
    make_network_figure,
)
from common.ui import about_section, params_rail

NS = "backprop"


def _k(name: str) -> str:
    return f"{NS}__{name}"


st.title("Backpropagation — Step-by-step Visualiser")
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

# ---------------------------------------------------------------------------
# Params rail
# ---------------------------------------------------------------------------
with params_rail(col_params, "Data"):
    shape_name = st.selectbox(
        "Data shape",
        options=SHAPE_NAMES,
        index=0,
        help="XOR and the two smooth shapes cannot be separated by a straight "
             "line — that's exactly why a hidden layer is necessary.",
        key=_k("shape"),
    )
    shape_key = SHAPE_KEYS[SHAPE_NAMES.index(shape_name)]

    n_points = st.slider("Number of points", min_value=40, max_value=400, value=160, step=10, key=_k("n_points"))
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key=_k("seed"))

with params_rail(col_params, "Network & training"):
    arch_name = st.radio(
        "Architecture",
        ["Small (2 → 4 → 1)", "Deep (2 → 8 → 8 → 1)"],
        help="The deep network has more capacity and usually finds a smoother, "
             "more confident boundary, typically reaching a lower loss in the "
             "same number of epochs than the small network — at the cost of "
             "more computation per epoch.",
        key=_k("arch"),
    )
    hidden_sizes = (4,) if arch_name.startswith("Small") else (8, 8)

    lr = st.slider(
        "Learning rate", min_value=0.05, max_value=3.0, value=0.5, step=0.05,
        help="Too small: painfully slow. Too large: the loss curve oscillates "
             "or diverges instead of descending smoothly.",
        key=_k("lr"),
    )
    epochs = st.slider("Max epochs", min_value=20, max_value=300, value=150, step=10, key=_k("epochs"))

    regenerate = st.button("🔄 Re-generate data", use_container_width=True, key=_k("regen"))

with col_params:
    _SHAPE_NOTES = {
        "xor":     "**XOR pattern** — the textbook proof that a single-layer "
                   "perceptron cannot work: no straight line separates the classes. "
                   "Watch the hidden layer carve the plane into the pieces a linear "
                   "model never could.",
        "moons":   "**Two moons** — a smooth non-linear boundary. A handful of "
                   "hidden units, each contributing a soft linear cut, combine "
                   "into the curved boundary.",
        "circles": "**Concentric circles** — the boundary is a closed loop. "
                   "The network has to learn a *radial* feature, not a directional one.",
        "linear":  "**Linearly separable** — a single straight line solves this. "
                   "Even the smallest network converges in a few epochs; a useful "
                   "baseline against the other shapes.",
    }
    st.info(_SHAPE_NOTES[shape_key])

    with st.expander("How backprop works", expanded=False):
        st.markdown(
            """
1. **Forward pass** — data flows through the network, producing a prediction and a loss
2. **Backward pass** — the error is propagated back layer by layer via the chain rule, producing a gradient for every weight
3. **Update** — every weight takes a small step opposite its gradient
4. Repeat until the loss stops improving
"""
        )

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _generate_and_train() -> None:
    X, y = make_dataset(shape_key, n_points=n_points, seed=int(seed))
    run = fit(X, y, hidden_sizes=hidden_sizes, lr=lr, epochs=epochs, seed=int(seed))
    st.session_state[_k("run")] = run
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


_param_key = (shape_key, n_points, int(seed), hidden_sizes, lr, epochs)
if _k("run") not in st.session_state or regenerate or st.session_state.get(_k("_param_key")) != _param_key:
    _generate_and_train()
    st.session_state[_k("_param_key")] = _param_key

run = st.session_state[_k("run")]
snapshots = run.snapshots
n_steps = len(snapshots)

caption_slot.caption(
    f"Shape: **{shape_name}** | "
    f"Points: **{n_points}** | "
    f"Architecture: **{arch_name}** | "
    f"LR: **{lr}** | "
    f"Seed: **{seed}**"
)

with col_main:
    about_section(
        "Backpropagation is the algorithm that makes training any neural "
        "network computationally feasible — an efficient application of the "
        "chain rule that computes *every* weight's gradient in a single "
        "backward pass instead of one at a time. It was popularised (the "
        "underlying calculus is older) by Rumelhart, Hinton, and Williams' "
        "1986 *Nature* paper, and every model in this site's \"Generative & "
        "self-supervised\" and \"Transformer\" sections is trained by some "
        "variant of it. The **XOR** shape is the canonical illustration of "
        "*why* a hidden layer — and therefore backprop — is necessary at all: "
        "no single straight line separates the classes, but the network's "
        "hidden units learn to carve the plane into pieces that, combined, do.",
        [
            "Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). \"Learning "
            "Representations by Back-Propagating Errors.\" *Nature*, 323, "
            "533–536.",
        ],
    )

    # -------------------------------------------------------------------
    # Playback controls (pure Streamlit — no Plotly updatemenus)
    # -------------------------------------------------------------------
    # Defensive clamp: a widget interaction (e.g. changing a sidebar parameter)
    # can fire while an autoplay rerun for the *previous* run is still in flight.
    # Both reruns write to the same "step_idx" key, and the previous run may have
    # had a different (often larger) frame count, so the value read back here
    # is not guaranteed to be in range for the *current* `snapshots`. Clamping
    # on every read/write is what keeps that race from throwing an IndexError
    # below instead of just snapping back to a valid frame.
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
        DELAY = {"0.5×": 0.9, "1×": 0.45, "2×": 0.22, "4×": 0.1}[speed]

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

        st.progress(
            step_idx / max(n_steps - 1, 1),
            text=f"Frame {step_idx + 1} / {n_steps} — {snapshots[step_idx].title}",
        )

    # -------------------------------------------------------------------
    # Metric cards
    # -------------------------------------------------------------------
    snap = snapshots[step_idx]

    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Phase", snap.phase.capitalize())
        m2.metric("Epoch", str(snap.epoch))
        m3.metric("Loss", f"{snap.loss:.4f}")
        m4.metric("Accuracy", f"{100 * snap.accuracy:.1f}%")

    # -------------------------------------------------------------------
    # Charts — decision boundary + network diagram side by side
    # -------------------------------------------------------------------
    col_boundary, col_network = st.columns(2)
    with col_boundary:
        with st.container(border=True):
            st.plotly_chart(
                make_boundary_figure(snap, run.grid_x, run.grid_y),
                use_container_width=True, config={"displayModeBar": False},
                key=_k("boundary_chart"),
            )
    with col_network:
        with st.container(border=True):
            st.plotly_chart(
                make_network_figure(snap, run.layer_sizes),
                use_container_width=True, config={"displayModeBar": False},
                key=_k("network_chart"),
            )

    if snap.phase == "init":
        st.info(
            "**Initialisation** — weights are small random values (Xavier/Glorot "
            "uniform). The decision surface is essentially noise; the network "
            "hasn't seen any data yet."
        )
    elif snap.phase == "forward":
        st.info(
            f"**Forward pass** — the current weights produce a prediction for "
            f"every point. Loss = **{snap.loss:.4f}**, accuracy = "
            f"**{100 * snap.accuracy:.1f}%**. The contour on the left *is* the "
            f"network's current belief about the whole input space, not just "
            f"the training points."
        )
    else:
        status = (
            "Loss has stopped improving — training has **converged**."
            if snap.converged
            else "Every weight now has a gradient; the update happens right after this frame."
        )
        st.info(
            f"**Backward pass** — the error at the output is propagated back "
            f"through the chain rule. The network diagram on the right now shows "
            f"**gradients** instead of weights: blue edges want to increase, red "
            f"edges want to decrease. {status}"
        )

    st.markdown("---")

    # -------------------------------------------------------------------
    # Loss curve
    # -------------------------------------------------------------------
    history = [(s.epoch, s.loss, s.accuracy) for s in snapshots if s.phase != "backward"]
    click_epoch = next((e for e, _, a in history if a >= 0.95), None)

    with st.container(border=True):
        st.plotly_chart(
            make_loss_figure(history, snap.epoch, click_epoch),
            use_container_width=True, config={"displayModeBar": False},
            key=_k("loss_chart"),
        )

    # -------------------------------------------------------------------
    # Hidden-neuron activation heat-maps
    # -------------------------------------------------------------------
    with st.expander("🔬 Hidden-neuron activation heat-maps (first hidden layer)"):
        st.caption(
            "Each tile is one neuron's tanh activation evaluated across the "
            "entire input space at the current frame. The decision boundary "
            "above is a weighted sum of exactly these patterns, passed through "
            "the output layer."
        )
        st.plotly_chart(
            make_activation_heatmap_figure(snap, run.grid_x, run.grid_y),
            use_container_width=True, config={"displayModeBar": False},
            key=_k("activation_chart"),
        )

    with st.expander("📖 Reading the animation"):
        st.markdown(
            """
- **Left chart** — decision boundary: blue background = network predicts class 0, red = class 1
- **Right chart** — network diagram: edge width = magnitude, blue = positive / red = negative
  - On a **forward** frame, edges show the current **weights**
  - On a **backward** frame, edges show the **gradients** just computed for those weights
- **Loss curve** — always shows the full training run; the red dot tracks the frame you're viewing
- **◀ Prev / Next ▶** to step manually, **▶ Play** to auto-advance through the whole run
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
