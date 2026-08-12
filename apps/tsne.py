"""t-SNE page — adapted from the standalone tsne-viz repo for the
multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from common.ui import about_section, params_rail
from tsneviz.algorithm import fit
from tsneviz.data import DATASET_KEYS, DATASET_NAMES, make_dataset
from tsneviz.visualize import make_figure

NS = "tsne"


def _k(name: str) -> str:
    return f"{NS}__{name}"


st.title("t-SNE — KL divergence descent in 2-D")
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

with params_rail(col_params, "Data"):
    ds_name = st.selectbox("Dataset", DATASET_NAMES, index=1, key=_k("ds_name"))
    ds_key = DATASET_KEYS[DATASET_NAMES.index(ds_name)]
    n_points = st.slider("Points (n)", 50, 420, 220, 10, key=_k("n_points"))
    ambient_dim = st.slider("Ambient dimension D", 6, 64, 24, 2, key=_k("ambient_dim"))
    lift_noise = st.slider(
        "Lift noise", 0.0, 0.15, 0.05, 0.005,
        disabled=ds_key == "uniform",
        help="Isotropic noise added after lifting the latent shape into ℝᴰ. "
             "No effect on the Uniform (control) dataset — it is sampled "
             "directly in ℝᴰ and never lifted from a low-D latent.",
        key=_k("lift_noise"),
    )
    seed = st.number_input("Random seed", 0, 9999, 3, 1, key=_k("seed"))

with params_rail(col_params, "Optimisation"):
    perplexity = st.slider("Perplexity", 5.0, 60.0, 28.0, 1.0,
        help="Effective neighbourhood size for input affinities. Internally "
             "capped at max(5, (n-1)/3) — watch the caption's effective "
             "perplexity when n is small.",
        key=_k("perplexity"))
    n_iter = st.slider("Gradient steps", 200, 1200, 700, 50, key=_k("n_iter"))
    early_exag = st.slider("Early exaggeration factor", 4.0, 20.0, 12.0, 1.0, key=_k("early_exag"))
    early_iters = st.slider("Early exaggeration steps", 50, 300, 150, 10, key=_k("early_iters"))
    lr = st.slider("Learning rate", 50.0, 600.0, 280.0, 10.0, key=_k("lr"))
    snap_every = st.slider("Snapshot every … steps", 5, 50, 15, 5, key=_k("snap_every"))
    regenerate = st.button("🔄 Re-run", use_container_width=True, key=_k("regen"))

with col_params:
    st.caption(
        "**t-SNE** minimises KL(**P**‖**Q**): Gaussian affinities in ℝᴰ vs Student‑t **Q** in 2‑D. "
        "Early exaggeration inflates **P** briefly so clusters separate before fine layout."
    )

_param = (
    ds_key,
    n_points,
    ambient_dim,
    float(lift_noise),
    float(perplexity),
    int(n_iter),
    float(early_exag),
    int(early_iters),
    float(lr),
    int(snap_every),
    int(seed),
)


def _run() -> None:
    X, labels, latent_xy = make_dataset(
        ds_key,
        n_points=n_points,
        ambient_dim=ambient_dim,
        seed=int(seed),
        lift_noise=float(lift_noise),
    )
    snaps = fit(
        X,
        labels,
        latent_xy,
        perplexity=float(perplexity),
        n_iter=int(n_iter),
        early_exaggeration=float(early_exag),
        early_exaggeration_iter=int(early_iters),
        learning_rate=float(lr),
        random_state=int(seed),
        snap_every=int(snap_every),
    )
    st.session_state[_k("snapshots")] = snaps
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


if _k("snapshots") not in st.session_state or regenerate or st.session_state.get(_k("_pk")) != _param:
    _run()
    st.session_state[_k("_pk")] = _param

snapshots = st.session_state[_k("snapshots")]
n_steps = len(snapshots)
step_idx = int(st.session_state.get(_k("step_idx"), 0))
step_idx = max(0, min(step_idx, n_steps - 1))
st.session_state[_k("step_idx")] = step_idx
playing = bool(st.session_state.get(_k("playing"), False))
snap = snapshots[step_idx]

if n_steps > 1:
    with col_params:
        st.markdown("**KL(P‖Q)** across frames")
        st.line_chart({"KL": [s.kl_divergence for s in snapshots]}, height=140)

caption_slot.caption(
    f"**Dataset:** {ds_name} · **n**={n_points} · **D**={ambient_dim} · **perplexity**={snap.perplexity:.1f}"
)

with col_main:
    about_section(
        "t-SNE was the technique that made high-dimensional visualization "
        "genuinely popular — it's still one of the most common ways researchers "
        "\"look at\" what a trained model's embeddings actually learned. Its "
        "defining trick, matching pairwise similarities between a high-D "
        "Gaussian distribution and a low-D Student-t distribution, is "
        "specifically designed to fight the *crowding problem* that made "
        "earlier embedding methods produce cluttered, uninformative 2-D maps. "
        "The **early exaggeration** phase this page's KL(P‖Q) curve shows "
        "spiking early on is the mechanism responsible: briefly inflating the "
        "target similarities so clusters snap apart before the layout settles "
        "into its final, finer detail.",
        [
            "van der Maaten, L. & Hinton, G. (2008). "
            "[\"Visualizing Data using t-SNE.\"](https://jmlr.org/papers/v9/vandermaaten08a.html) "
            "*Journal of Machine Learning Research*, 9, 2579–2605.",
        ],
    )

    # -------------------------------------------------------------------
    # Metrics, playback controls, chart, and auto-advance all live inside
    # one fragment: st.rerun() during autoplay used to fully rerun the
    # whole page (title/caption/params rail/about-section included)
    # several times a second, and small timing differences in how long
    # each of those took to re-render showed up as visible flicker/layout
    # shift on every frame. Scoping the rerun to just this fragment keeps
    # everything above it (and the sidebar) completely static.
    # -------------------------------------------------------------------
    @st.fragment
    def _playback() -> None:
        with st.container(border=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("KL (this frame)", f"{snap.kl_divergence:.4f}")
            m2.metric("Phase", snap.phase)
            m3.metric("Iteration", str(snap.iteration))

        with st.container(border=True):
            speed = st.select_slider("Speed", ["0.5×", "1×", "2×"], "1×", label_visibility="collapsed", key=_k("speed"))
            delay = {"0.5×": 0.85, "1×": 0.44, "2×": 0.22}[speed]
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                if st.button("◀", disabled=step_idx == 0 or playing, key=_k("prev")):
                    st.session_state[_k("step_idx")] = max(0, step_idx - 1)
                    st.rerun(scope="fragment")
            with b2:
                if st.button("▶ Play", disabled=playing or step_idx == n_steps - 1, type="primary", key=_k("play")):
                    st.session_state[_k("playing")] = True
                    st.rerun(scope="fragment")
            with b3:
                if st.button("⏸", disabled=not playing, key=_k("pause")):
                    st.session_state[_k("playing")] = False
                    st.rerun(scope="fragment")
            with b4:
                if st.button("Next ▶", disabled=step_idx == n_steps - 1 or playing, key=_k("next")):
                    st.session_state[_k("step_idx")] = min(n_steps - 1, step_idx + 1)
                    st.rerun(scope="fragment")

            den = max(n_steps - 1, 1)
            st.progress(float(step_idx) / float(den), text=f"Frame {step_idx + 1}/{n_steps}")

        st.info(
            "**A (left):** latent geometry before the lift (or **PCA₂** if none). **B (right):** current **𝑌**. "
            "Colour is **only** for matching points — t-SNE is unsupervised."
        )

        fig = make_figure(snap, frame_index=step_idx, frame_total=n_steps)
        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=_k("chart"))

        # Auto-advance (must be last — triggers a fragment-scoped rerun after a delay)
        if playing and step_idx < n_steps - 1:
            time.sleep(delay)
            st.session_state[_k("step_idx")] = min(n_steps - 1, step_idx + 1)
            st.rerun(scope="fragment")
        elif playing:
            st.session_state[_k("playing")] = False
            st.rerun(scope="fragment")

    _playback()
