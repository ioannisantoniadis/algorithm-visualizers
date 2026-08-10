"""GMM (EM algorithm) page — adapted from the standalone gmm-viz repo for the
multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from gmmviz.algorithm import fit
from gmmviz.data import SHAPE_DEFAULTS, SHAPE_KEYS, SHAPE_NAMES, make_dataset
from gmmviz.visualize import make_figure

NS = "gmm"


def _k(name: str) -> str:
    return f"{NS}__{name}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("##### ⚙️ DATA & MODEL")

with st.sidebar.container(border=True):
    shape_name = st.selectbox("Data shape", SHAPE_NAMES, index=0, key=_k("shape"))
    shape_key = SHAPE_KEYS[SHAPE_NAMES.index(shape_name)]
    defaults = SHAPE_DEFAULTS[shape_key]

    n_points = st.slider("Number of points", 50, 400, 200, step=10, key=_k("n_points"))
    n_clusters_data = st.slider(
        "True clusters (generative K)",
        2,
        8,
        4,
        disabled=shape_key in ("moons", "circles", "uniform"),
        help="Ground-truth blob count for synthetic shapes — moons, rings, and uniform noise ignore this.",
        key=_k("n_clusters_data"),
    )
    k_components = st.slider(
        "Mixture components (model K)",
        2,
        8,
        int(defaults["k_components"]),
        help="Number of Gaussians in the fitted GMM. Try K ≠ true K to see mismatch.",
        key=_k("k_components"),
    )
    max_iter = st.slider("EM outer iterations", 1, 40, 12, 1, key=_k("max_iter"))
    reg_covar = st.slider(
        "Covariance regularisation ε",
        1e-8,
        1e-2,
        1e-6,
        format="%.1e",
        help="Added to each diagonal of Σ after every M-step (stability).",
        key=_k("reg_covar"),
    )
    seed = st.number_input("Random seed", 0, 9999, 42, 1, key=_k("seed"))
    regenerate = st.button("🔄 Re-run", use_container_width=True, key=_k("regen"))

_SHAPE_NOTES = {
    "blobs":       "**Gaussian blobs** — well-separated spherical clusters; GMM recovers the generating components almost exactly.",
    "anisotropic": "**Anisotropic blobs** — full covariance lets ellipses rotate to match elongated clusters, a hard case for K-means.",
    "varied":      "**Varied density** — components adapt their own Σ instead of forcing equal size like K-means.",
    "moons":       "**Two moons** — each Gaussian can only cover an elliptical region, so crescents get split or blurred together.",
    "circles":     "**Concentric rings** — same failure mode as moons: no ellipse matches a ring.",
    "uniform":     "**Uniform noise** — no real structure to recover; components just spread out to tile the noise.",
}
st.sidebar.info(_SHAPE_NOTES[shape_key])

with st.sidebar.expander("How EM works", expanded=False):
    st.markdown("""
1. **E-step** — for fixed π, μ, Σ, compute “soft counts” γᵢₖ ∝ πₖ 𝒩(xᵢ|μₖ,Σₖ)
2. **M-step** — treat γ as weights; update π, μ, Σ by weighted MLE
3. Repeat until log-likelihood plateaus (here: fixed iterations + two frames per step)

**Frames:** **init** (first E-step), then for each iteration **M** (ellipses jump; colours frozen), **E** (colours catch up).
""")

_param = (shape_key, n_points, n_clusters_data, k_components, max_iter, float(reg_covar), seed)


def _run() -> None:
    X, _true_lbl = make_dataset(shape_key, n_points=n_points, n_clusters=n_clusters_data, seed=int(seed))
    snaps = fit(
        X,
        k_components,
        max_iter=int(max_iter),
        reg_covar=float(reg_covar),
        random_state=int(seed),
    )
    st.session_state[_k("snapshots")] = snaps
    st.session_state[_k("X")] = X
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


if _k("snapshots") not in st.session_state or regenerate or st.session_state.get(_k("_pk")) != _param:
    _run()
    st.session_state[_k("_pk")] = _param

snapshots = st.session_state[_k("snapshots")]
X = st.session_state[_k("X")]
n_steps = len(snapshots)

# Clamp defensively: rapid clicking of Prev/Next can otherwise queue a step_idx
# update that lands out of range for the *current* run's snapshot list and
# crash with an IndexError.
step_idx = int(st.session_state.get(_k("step_idx"), 0))
step_idx = max(0, min(step_idx, n_steps - 1))
st.session_state[_k("step_idx")] = step_idx

playing = bool(st.session_state.get(_k("playing"), False))
snap = snapshots[step_idx]
series = np.array([s.log_likelihood for s in snapshots], dtype=np.float64)

if len(snapshots) > 1:
    st.sidebar.markdown("**Marginal log-likelihood** (each frame)")
    st.sidebar.line_chart(
        {"log Σᵢ log p(xᵢ)": [s.log_likelihood for s in snapshots]},
        height=160,
    )

# ---------------------------------------------------------------------------
# Header & summary metrics
# ---------------------------------------------------------------------------
st.title("Gaussian mixture model — EM step-by-step")

st.caption(
    f"**Shape:** {shape_name} · **n**={n_points} · **model K**={k_components} · "
    f"**EM iters**={max_iter} · **ε_reg**={reg_covar:.1e} · seed={seed}"
)

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("Current log-likelihood", f"{snap.log_likelihood:.2f}")
    c2.metric("Frame", f"{step_idx + 1} / {n_steps}")
    c3.metric("ΔLL (vs frame 0)", f"{snap.log_likelihood - series[0]:.2f}")

# ---------------------------------------------------------------------------
# Playback controls
# ---------------------------------------------------------------------------
with st.container(border=True):
    speed = st.select_slider("Playback speed", ["0.5×", "1×", "2×", "4×"], "1×", label_visibility="collapsed", key=_k("speed"))
    delay = {"0.5×": 1.0, "1×": 0.52, "2×": 0.28, "4×": 0.14}[speed]
    b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
    with b1:
        if st.button("◀", use_container_width=True, disabled=step_idx == 0 or playing, key=_k("prev")):
            st.session_state[_k("step_idx")] = max(0, step_idx - 1)
            st.rerun()
    with b2:
        if st.button("▶ Play", use_container_width=True, disabled=playing or step_idx == n_steps - 1, type="primary", key=_k("play")):
            st.session_state[_k("playing")] = True
            st.rerun()
    with b3:
        if st.button("⏸", use_container_width=True, disabled=not playing, key=_k("pause")):
            st.session_state[_k("playing")] = False
            st.rerun()
    with b4:
        if st.button("Next ▶", use_container_width=True, disabled=step_idx == n_steps - 1 or playing, key=_k("next")):
            st.session_state[_k("step_idx")] = min(n_steps - 1, step_idx + 1)
            st.rerun()

    st.progress(step_idx / max(n_steps - 1, 1), text=f"Frame {step_idx + 1}/{n_steps}")

if snap.substep == "m":
    st.info(
        "**M-step:** each Gaussian’s π, μ, Σ was just updated from the latest soft assignments. "
        "Point colours **still** reflect the **previous** E-step so you can see ellipses move first."
    )
elif snap.substep == "e" or snap.substep == "init":
    st.success(
        "**E-step:** colours now match current π, μ, Σ (soft cluster membership). "
        + ("Compare ellipse axes to the colour gradients." if snap.substep == "e" else "")
    )

fig = make_figure(snap)
with st.container(border=True):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True}, key=_k("chart"))

if playing and step_idx < n_steps - 1:
    time.sleep(delay)
    st.session_state[_k("step_idx")] = min(n_steps - 1, step_idx + 1)
    st.rerun()
elif playing:
    st.session_state[_k("playing")] = False
    st.rerun()
