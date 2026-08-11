"""UMAP page — adapted from the standalone umap-viz repo for the
multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from common.ui import about_section, params_rail
from umapviz.algorithm import fit
from umapviz.data import DATASET_KEYS, DATASET_NAMES, latent_intrinsic_dim, make_dataset
from umapviz.visualize import make_figure

NS = "umap"


def _k(name: str) -> str:
    return f"{NS}__{name}"


PIPELINE_RAIL: list[tuple[str, str, str]] = [
    (
        "knn",
        "① Build kNN graph",
        "Neighbour pairs are chosen in **full ℝᴰ**. The scatter is only **PCA₂ of X** so edges can be drawn on screen.",
    ),
    (
        "fuzzy",
        "② Fuzzy union",
        "One global weighted graph from local fuzzy sets. **Same PCA₂ view on the left** — still not the UMAP map.",
    ),
    (
        "embed",
        "③ Optimise embedding 𝑌",
        "Only here does **panel B** show **UMAP’s output**. Advance frames in order ①→②→③.",
    ),
]


def _render_pipeline_rail(phase: str) -> None:
    r1, r2, r3 = st.columns(3)
    for col, (ph, title, body) in zip((r1, r2, r3), PIPELINE_RAIL):
        with col:
            if phase == ph:
                with st.container(border=True):
                    st.markdown(f"**{title} — you are here**")
                    st.caption(body)
            else:
                st.caption(f"{title} · later")


def _frame_panels_explainer(snap, dataset_key: str) -> str:
    if snap.phase in ("knn", "fuzzy"):
        a = (
            "**Panel A (left)** — **PCA₂** of the high-D matrix **X** (a 2-D “camera”; the algorithm still stores all **D** coordinates). "
            "Edges are **true** kNN or fuzzy links from **ℝᴰ**, drawn in this projection. "
            "**What UMAP uses as input:** the full matrix **X** — not this picture."
        )
        b_latent = (
            "**Panel B (right)** — **Synthetic latent** (geometry **before** the random rotation into ℝᴰ). "
            "Same colours as A. **UMAP never sees panel B** — it is only so you can judge whether the high-D data still encodes that shape."
        )
        b_none = (
            "**Panel B (right)** — **No ground-truth latent**; faded **PCA₂** copy for a side-by-side scale reference."
        )
        if snap.latent_xy is None:
            return a + "\n\n" + b_none
        return a + "\n\n" + b_latent

    a_parts = [
        "**Panel A (left)** — **Reference for your eyes** (ground-truth latent when we have it, otherwise **PCA₂**). **Not an input to UMAP.**"
    ]
    if dataset_key == "uniform" or snap.latent_xy is None:
        a_parts.append("Here A is **PCA₂** so there is a stable comparison view.")
    elif latent_intrinsic_dim(dataset_key) == 3:
        a_parts.append("Truth lives in **ℝ³**; the plot shows **x–y** only (depth not drawn).")
    elif dataset_key == "mixed":
        a_parts.append("Structured points only on the left **(uniform noise has no latent row)**.")
    b = (
        "**Panel B (right)** — **UMAP’s result:** embedding **𝑌** after optimisation. **This is the “reconstructed” 2-D map** — compare **B** to **A** (expect rotation / stretch; colours and clusters should relate)."
    )

    return "\n\n".join(a_parts) + "\n\n" + b


st.title("UMAP — Topological graph + embedding")
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

with params_rail(col_params, "Data"):
    ds_name = st.selectbox("Synthetic dataset", DATASET_NAMES, index=1, key=_k("ds_name"))
    ds_key = DATASET_KEYS[DATASET_NAMES.index(ds_name)]
    n_points = st.slider("Points", 60, 380, 200, 10, key=_k("n_points"))
    ambient_dim = st.slider("Ambient dimension D", 6, 64, 24, 2,
        help="Latent structure is rotated into Rᴰ — PCA preview often misses it.",
        key=_k("ambient_dim"))
    lift_noise = st.slider(
        "Lift noise (isotropic in Rᴰ)",
        0.0,
        0.20,
        0.05,
        0.005,
        disabled=ds_key == "uniform",
        help=(
            "Uniform (control) data has no latent shape and no lift step, so "
            "this has no effect for that dataset."
            if ds_key == "uniform" else
            "Extra Gaussian noise added after the random rotation into D dims — "
            "higher values make kNN / UMAP harder."
        ),
        key=_k("lift_noise"),
    )
    seed = st.number_input("Random seed", 0, 9999, 7, 1, key=_k("seed"))

with params_rail(col_params, "UMAP hyperparameters"):
    n_neighbors = st.slider("n_neighbors", 3, 45, 15, key=_k("n_neighbors"))
    min_dist = st.slider("min_dist (embedding geometry)", 0.0, 0.5, 0.15, 0.01, key=_k("min_dist"))
    spread = st.slider("spread (a,b curve fit)", 0.25, 2.0, 1.0, 0.05, key=_k("spread"))
    n_epochs = st.slider("SGD epochs", 60, 400, 220, 10, key=_k("n_epochs"))
    snap_every = st.slider("Snapshot every … epochs", 10, 80, 30, 5, key=_k("snap_every"))
    lr = st.slider("Learning rate", 0.2, 2.0, 1.0, 0.1, key=_k("lr"))
    neg_rate = st.slider("Negative samples / edge", 1, 12, 5, key=_k("neg_rate"))
    regenerate = st.button("🔄 Re-run", use_container_width=True, key=_k("regen"))

with col_params:
    st.info(
        "Use **Next** from left to right: ① kNN → ② fuzzy → ③ embedding. "
        "The **blue explainer** under the controls always describes **panels A and B** for the current step."
    )

_param = (
    ds_key,
    n_points,
    ambient_dim,
    float(lift_noise),
    seed,
    n_neighbors,
    min_dist,
    spread,
    n_epochs,
    snap_every,
    lr,
    neg_rate,
)


def _run() -> None:
    X, labels, latent_xy = make_dataset(
        ds_key,
        n_points=n_points,
        ambient_dim=ambient_dim,
        seed=int(seed),
        lift_noise=float(lift_noise),
    )
    st.session_state[_k("X")] = X
    st.session_state[_k("labels")] = labels
    st.session_state[_k("snaps")] = fit(
        X,
        labels,
        latent_xy=latent_xy,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        spread=float(spread),
        n_epochs=int(n_epochs),
        snap_embed_every=int(snap_every),
        random_state=int(seed),
        negative_sample_rate=int(neg_rate),
        learning_rate=float(lr),
    )
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


if _k("snaps") not in st.session_state or regenerate or st.session_state.get(_k("_pk")) != _param:
    _run()
    st.session_state[_k("_pk")] = _param

X = st.session_state[_k("X")]
snaps = st.session_state[_k("snaps")]
labels = st.session_state[_k("labels")]
n_steps = len(snaps)
step_idx = int(st.session_state.get(_k("step_idx"), 0))
step_idx = max(0, min(step_idx, n_steps - 1))
st.session_state[_k("step_idx")] = step_idx
playing = bool(st.session_state.get(_k("playing"), False))
snap = snaps[step_idx]

caption_slot.caption(
    f"**Dataset:** {ds_name} | **n**={n_points}, **D**={ambient_dim} | "
    f"n_neighbors={n_neighbors}, min_dist={min_dist}"
)

with col_main:
    about_section(
        "UMAP is the modern default for visualizing high-dimensional data — "
        "single-cell genomics, embeddings from language/vision models, and so "
        "on — largely replacing t-SNE because it's dramatically faster on large "
        "datasets and better preserves *global* structure alongside local "
        "neighbourhoods. It's grounded in a genuinely different mathematical "
        "foundation, Riemannian geometry and algebraic topology, than the "
        "purely probabilistic approach t-SNE takes, even though the two "
        "produce visually similar-looking maps. The three-stage pipeline this "
        "page walks through — build a k-NN graph, turn it into one fuzzy "
        "weighted graph, then optimise a 2-D layout to match it — *is* that "
        "topological idea made concrete.",
        [
            "McInnes, L., Healy, J., & Melville, J. (2018). "
            "[\"UMAP: Uniform Manifold Approximation and Projection for "
            "Dimension Reduction.\"](https://arxiv.org/abs/1802.03426) "
            "arXiv:1802.03426.",
        ],
    )

    with st.container(border=True):
        _render_pipeline_rail(snap.phase)

    left, right = st.columns([1.05, 1.45], gap="medium")

    with left:
        with st.container(border=True):
            st.markdown("### This frame")
            step_no = {"knn": 1, "fuzzy": 2, "embed": 3}[snap.phase]
            st.caption(
                f"Pipeline step **{step_no}/3** — highlighted above. "
                f"Phase **{snap.phase}**; animation frame **{step_idx + 1}/{n_steps}**."
            )
            m1, m2 = st.columns(2)
            m1.metric("Phase", snap.phase)
            if snap.phase == "embed":
                m2.metric("SGD epoch", str(snap.step))
            st.metric("Layout (a, b)", f"{snap.a_param:.3f}, {snap.b_param:.3f}")
            st.markdown("Low-dimensional pairwise kernel ")
            st.latex(r"q_{ij} = 1/\bigl(1 + a \,\lVert y_i - y_j \rVert^{2b}\bigr)")
            st.caption("(a, b) fitted to match min_dist & spread — same convention as umap-learn.")
            st.markdown("**High-D preview** — variance along first two PCs of **X**:")
            xv = X - X.mean(0, keepdims=True)
            _, s, _ = np.linalg.svd(xv, full_matrices=False)
            st.code(f"Top singular values (relative): {s[0]:.3f}, {s[1]:.3f}, …", language=None)
            with st.expander("Algorithm stages"):
                st.markdown("""
1. **k-NN** in ℝᴰ → local distances
2. **smooth_knn** normalisation → σᵢ, ρᵢ (local connectivity)
3. **Membership strengths** → directed fuzzy 1-simplices
4. **Fuzzy union** symmetrises the global weighted graph
5. **SGD** minimises cross-entropy between high-D and low-D fuzzy sets
""")

    with right:
        with st.container(border=True):
            speed = st.select_slider("Speed", ["0.5×", "1×", "2×"], "1×", label_visibility="collapsed", key=_k("speed"))
            delay = {"0.5×": 0.95, "1×": 0.48, "2×": 0.24}[speed]
            c1, c2, c3, c4 = st.columns([1, 1.1, 1.1, 1])
            with c1:
                if st.button("◀", disabled=(step_idx == 0 or playing), key=_k("prev")):
                    st.session_state[_k("step_idx")] = max(0, step_idx - 1)
                    st.rerun()
            with c2:
                if st.button("▶ Play", disabled=(playing or step_idx == n_steps - 1), type="primary", key=_k("play")):
                    st.session_state[_k("playing")] = True
                    st.rerun()
            with c3:
                if st.button("⏸", disabled=not playing, key=_k("pause")):
                    st.session_state[_k("playing")] = False
                    st.rerun()
            with c4:
                if st.button("Next ▶", disabled=(step_idx == n_steps - 1 or playing), key=_k("next")):
                    st.session_state[_k("step_idx")] = min(n_steps - 1, step_idx + 1)
                    st.rerun()

            st.progress(step_idx / max(n_steps - 1, 1), text=f"Frame {step_idx + 1}/{n_steps}")

            st.info(_frame_panels_explainer(snap, ds_key))

        with st.container(border=True):
            fig = make_figure(snap, frame_index=step_idx, frame_total=n_steps)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=_k("chart"))
            st.caption("**A / B** in the plot titles match the explainer above.")

    if snap.phase == "embed":
        st.caption(
            "Note: this is a **readable** SGD loop, not a drop-in for **umap-learn**; global topology should still line up with the reference."
        )

    if playing and step_idx < n_steps - 1:
        time.sleep(delay)
        st.session_state[_k("step_idx")] = min(n_steps - 1, step_idx + 1)
        st.rerun()
    elif playing:
        st.session_state[_k("playing")] = False
        st.rerun()
