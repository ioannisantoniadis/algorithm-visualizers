"""PCA page — adapted from the standalone pca-viz repo for the
multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from common.ui import about_section, params_rail
from pca.algorithm import fit
from pca.data import SHAPE_KEYS, SHAPE_NAMES, make_dataset
from pca.visualize import make_marginal_xy_figure, make_static_figure

NS = "pca"


def _k(name: str) -> str:
    return f"{NS}__{name}"


st.title("Principal Component Analysis — Step-by-step")
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

with params_rail(col_params, "Data"):
    emb_dim = st.radio("Ambient space", ["ℝ²", "ℝ³"], index=0,
        help="3-D point clouds include a **marginal (x₁,x₂)** plot: what you miss by ignoring x₃.",
        key=_k("emb_dim"))
    dim = 3 if emb_dim == "ℝ³" else 2

    shape_name = st.selectbox(
        "Point cloud shape", SHAPE_NAMES, index=1,
        help="Anisotropic blobs elongate along a shared axis — compare 2-D vs 3-D lift.",
        key=_k("shape"),
    )
    shape_key = SHAPE_KEYS[SHAPE_NAMES.index(shape_name)]
    n_points = st.slider("Number of points", 50, 500, 220, step=10, key=_k("n_points"))
    n_clusters = st.slider("Blob count (Gaussian shapes only)", 2, 8, 4,
        disabled=shape_key not in ("blobs", "anisotropic", "varied"),
        help="Only applies to Gaussian blobs, anisotropic blobs, and varied-density shapes — "
             "two moons, concentric rings, and uniform noise always use a fixed number of parts.",
        key=_k("n_clusters"))
    seed = st.number_input("Random seed", 0, 9999, 42, step=1, key=_k("seed"))
    regenerate = st.button("🔄 Re-generate", use_container_width=True, key=_k("regen"))

with col_params:
    _SHAPE_NOTES = {
        "blobs": "**Blobs** — PCs align with dominant variance directions.",
        "anisotropic": "**Anisotropic** — best demo: PC1 follows the longest axis of each ellipsoid.",
        "varied": "**Varied density** — large blobs dominate trace(**S**).",
        "moons": "**Moons** — in 3-D the lift breaks coplanarity; PCA still won’t unwrap nonlinearity.",
        "circles": "**Rings** — linear subspaces rarely match intuitive “clusters”.",
        "uniform": "**Uniform** — PCs reflect the box geometry, not hidden structure.",
    }
    st.info(_SHAPE_NOTES[shape_key])
    with st.expander("Pipeline", expanded=False):
        st.markdown("""
1. Centre **X** → **X̃**
2. **S** = **X̃**ᵀ**X̃**/(n−1)
3. **S** = **VΛV**ᵀ (columns of **V** = PCs)
4. Keep *k* columns: **X̂** = **X̃ V**₁:ₖ**V**₁:ₖᵀ + **1μ**ᵀ
""")

_param_key = (shape_key, n_points, n_clusters, int(seed), dim)


def _run() -> None:
    X, labels = make_dataset(
        shape_key, n_points=n_points, n_clusters=n_clusters, seed=int(seed), dim=dim,
    )
    st.session_state[_k("X")] = X
    st.session_state[_k("labels")] = labels
    st.session_state[_k("dim")] = dim
    st.session_state[_k("snapshots")] = fit(X)
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


if (
    _k("snapshots") not in st.session_state
    or regenerate
    or st.session_state.get(_k("_pk")) != _param_key
):
    _run()
    st.session_state[_k("_pk")] = _param_key

X: np.ndarray = st.session_state[_k("X")]
dim = int(st.session_state[_k("dim")])
labels: np.ndarray = st.session_state[_k("labels")]
snapshots = st.session_state[_k("snapshots")]
n_steps = len(snapshots)
last = snapshots[-1]

step_idx = int(st.session_state.get(_k("step_idx"), 0))
step_idx = max(0, min(step_idx, n_steps - 1))
st.session_state[_k("step_idx")] = step_idx
playing = bool(st.session_state.get(_k("playing"), False))
snap = snapshots[step_idx]

axis_lbl = [f"x{i + 1}" for i in range(dim)]

caption_slot.caption(
    f"Space: **ℝ^{dim}** | Shape: **{shape_name}** | Points: **{n_points}** | Seed: **{seed}**"
)

ev = last.evals
tr = float(ev.sum())
pct_rows = []
cum = 0.0
for i in range(dim):
    p = 100.0 * float(ev[i]) / tr if tr > 0 else 0.0
    cum += p
    pct_rows.append({"PC": i + 1, "λ": float(ev[i]), "% var": round(p, 2), "cum. %": round(cum, 2)})

with col_main:
    about_section(
        "PCA is the oldest and most fundamental dimensionality-reduction "
        "technique — nearly everything else in this app's \"Dimensionality "
        "reduction\" category (t-SNE, UMAP) exists to fix something PCA can't do: "
        "capture *nonlinear* structure. It's used everywhere — compressing data "
        "before feeding it to other models, denoising, and visualizing "
        "high-dimensional datasets. Because it only ever finds straight-line "
        "directions of variance, it's also the perfect control group for "
        "understanding why nonlinear methods were invented: switch this page to "
        "the **Moons** shape and watch PCA fail to find any \"structure\" beyond "
        "the raw spread of the points, since it has no way to represent a curve.",
        [
            "Pearson, K. (1901). \"On Lines and Planes of Closest Fit to Systems "
            "of Points in Space.\" *Philosophical Magazine.*",
            "Hotelling, H. (1933). \"Analysis of a Complex of Statistical "
            "Variables into Principal Components.\" *Journal of Educational "
            "Psychology.*",
        ],
    )

    # -------------------------------------------------------------------
    # Metrics, playback controls, algebra panel, figures, and auto-advance
    # all live inside one fragment: st.rerun() during autoplay used to
    # fully rerun the whole page (title/caption/params rail/about-section
    # included) several times a second, and small timing differences in
    # how long each of those took to re-render showed up as visible
    # flicker/layout shift on every frame. Scoping the rerun to just this
    # fragment keeps everything above it (and the sidebar) completely
    # static.
    # -------------------------------------------------------------------
    @st.fragment
    def _playback() -> None:
        with st.container(border=True):
            c_meta = st.columns(dim + 2)
            for i in range(dim):
                c_meta[i].metric(f"λ{i + 1}", f"{ev[i]:.4f}")
            c_meta[-2].metric("trace(S)", f"{tr:.4f}")
            c_meta[-1].metric("Frames", str(n_steps))

        with st.container(border=True):
            speed = st.select_slider(
                "Playback speed", options=["0.5×", "1×", "2×", "4×"], value="1×",
                label_visibility="collapsed",
                key=_k("speed"),
            )
            DELAY = {"0.5×": 1.2, "1×": 0.6, "2×": 0.32, "4×": 0.16}[speed]

            col_prev, col_play, col_pause, col_next, col_spd = st.columns([1, 1.2, 1.2, 1, 3])

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

            with col_spd:
                st.caption(f"Speed **{speed}** ({DELAY:.2f}s / frame)")

            st.progress(
                step_idx / max(n_steps - 1, 1),
                text=f"Frame {step_idx + 1} / {n_steps} — {snap.title}",
            )

        # ----- Algebra panel + figures -----
        left, right = st.columns([1.15, 1.5], gap="large")

        with left:
            st.markdown("### Numbers tied to this frame")
            st.caption("Same objects as in lecture notes — updated as you step.")

            S_df = pd.DataFrame(snap.cov, index=axis_lbl, columns=axis_lbl)
            st.markdown("**Sample covariance** **S** (Bessel n−1)")
            st.dataframe(S_df.round(4), use_container_width=True)

            eig_df = pd.DataFrame(pct_rows).set_index("PC")
            st.markdown("**Spectrum** (λᵢ / trace shares)")
            st.dataframe(eig_df.round(4), use_container_width=True)

            V = snap.evecs
            V_df = pd.DataFrame(
                V,
                index=[f"coord {i + 1}" for i in range(dim)],
                columns=[f"PC{j + 1}" for j in range(dim)],
            )
            st.markdown("**V** — eigenvectors as columns (PC directions)")
            st.dataframe(V_df.round(4), use_container_width=True)

            st.markdown("**Frame metrics**")
            m1, m2 = st.columns(2)
            m1.metric("Phase", snap.phase)
            m2.metric("k (recon.)", str(snap.k_components) if snap.phase == "reconstruct" else "—")
            m3, m4 = st.columns(2)
            m3.metric("MSE (mean sq. entry)", f"{snap.mse:.5f}" if snap.phase == "reconstruct" else "—")
            m4.metric("Var. captured", f"{100 * snap.cumulative_variance:.1f}%" if snap.phase == "reconstruct" else "—")

            st.latex(
                r"\hat{\mathbf{X}} = \tilde{\mathbf{X}} V_{1:k} V_{1:k}^{\top} + \mathbf{1}\mu^{\top}"
            )
            st.caption(
                r"MSE = mean of $(X_{ij}-\hat X_{ij})^2$ over all entries; "
                r"''Var. captured'' = $\sum_{j\leq k}\lambda_j\,/\,\mathrm{tr}(S)$."
            )

        with right:
            with st.container(border=True):
                fig_main = make_static_figure(snap, labels=labels)
                st.plotly_chart(fig_main, use_container_width=True, config={"displayModeBar": False}, key=_k("chart_main"))
                if dim == 3:
                    st.caption(
                        "**Below:** the **(x₁, x₂)** marginal — information along **x₃** is invisible here, "
                        "which is why 3-D PCA needs all three coordinates in the main plot."
                    )
                    fig_m = make_marginal_xy_figure(snap, labels=labels)
                    st.plotly_chart(fig_m, use_container_width=True, config={"displayModeBar": False}, key=_k("chart_marginal"))

        # ----- Narrative callouts -----
        if snap.phase == "raw":
            st.info(
                "**Raw data** — **μ** marks the affine centre; PCA always works on **X̃** = **X** − **μ**."
            )
        elif snap.phase == "centered":
            st.info(
                "**Centreing** — **S** describes second moments around **μ**; eigenvectors are directions "
                "of maximal variance *from here*."
            )
        elif snap.phase == "covariance":
            if dim == 2:
                st.info(
                    r"**Ellipse** — boundary where **x̃**ᵀ**S**⁻¹**x̃** = χ²₀.₉₅ (df = 2). Same **V** as PCA."
                )
            else:
                st.info(
                    r"**Ellipsoid** — **x̃**ᵀ**S**⁻¹**x̃** = χ²₀.₉₅ (df = 3). The marginal plot uses the **2×2** block "
                    r"**S**₂₂ for its own 95% ellipse — not a camera slice of the surface."
                )
        elif snap.phase == "eigen":
            st.info(
                "**Eigenvectors** — segment lengths 2√λⱼ (2σ along each PC in centred space). "
                f"Colours: PC1 orange, PC2 purple{', PC3 teal' if dim == 3 else ''}."
            )
        else:
            k = snap.k_components
            if k == 0:
                st.warning(
                    "Rank-0 reconstruction: **every** row of **X̂** is **μ**; error measures what variance you threw away."
                )
            elif k < dim:
                st.success(
                    f"Best rank-{k} linear approximation in ℝ^{dim}: residuals are orthogonal to the fitted subspace "
                    "(check both **3-D** and **marginal** views)."
                )
            else:
                st.success(
                    f"Full rank in ℝ^{dim}: **X̂** = **X** (numerically). No information left to remove."
                )

        with st.expander("📖 Legend"):
            st.markdown(f"""
- Colours = generator labels (PCA is **blind** to them)
- **χ²** contours = illustrative 95% Mahalanobis shells (__not__ a claim that data are Gaussian)
- **MSE** = Frobenius mean-squared entry, ‖**X**−**X̂**‖_F² / (n·{dim})
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
