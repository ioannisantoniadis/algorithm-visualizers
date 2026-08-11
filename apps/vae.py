"""VAE page — adapted from the standalone vae-viz repo for the multipage
portfolio app. Session-state keys are namespaced with a per-page prefix
since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from common.ui import about_section, params_rail
from vae.algorithm import decode, fit
from vae.data import SHAPE_KEYS, SHAPE_NAMES, make_dataset
from vae.visualize import make_data_space_figure, make_latent_space_figure, make_loss_figure

NS = "vae"


def _k(name: str) -> str:
    return f"{NS}__{name}"


st.title("Variational Autoencoder — Step-by-step Visualiser")
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

# ---------------------------------------------------------------------------
# Params rail
# ---------------------------------------------------------------------------
with params_rail(col_params):
    shape_name = st.selectbox(
        "Data shape",
        options=SHAPE_NAMES,
        index=0,
        help="The VAE never sees the colours — they are cluster/class labels "
             "used only to make the plots readable.",
        key=_k("shape"),
    )
    shape_key = SHAPE_KEYS[SHAPE_NAMES.index(shape_name)]

    n_points = st.slider("Number of points", min_value=80, max_value=400, value=200, step=10, key=_k("n_points"))

    arch_name = st.radio(
        "Architecture (hidden layer width)",
        ["Small (H=8)", "Medium (H=16)", "Large (H=32)"],
        index=1,
        key=_k("arch"),
    )
    hidden_size = {"Small (H=8)": 8, "Medium (H=16)": 16, "Large (H=32)": 32}[arch_name]

    st.caption("Latent dimension is fixed at **2** so the whole latent space can be plotted directly.")

    lr = st.slider(
        "Learning rate", min_value=0.02, max_value=0.5, value=0.15, step=0.01,
        help="Too high and the loss oscillates; too low and training won't finish "
             "converging within the epoch budget.",
        key=_k("lr"),
    )
    epochs = st.slider("Training epochs", min_value=30, max_value=400, value=150, step=10, key=_k("epochs"))

    beta = st.slider(
        "β — KL weight (VAE only)", min_value=0.0, max_value=5.0, value=1.0, step=0.1,
        help="Weight on the KL term in the VAE loss. β=0 makes the VAE behave "
             "exactly like the plain autoencoder (no pull toward the prior). "
             "Large β over-regularises: the latent space becomes a tight, "
             "uninformative blob and reconstructions blur toward the data mean "
             "— a phenomenon called posterior collapse.",
        key=_k("beta"),
    )

    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key=_k("seed"))

    regenerate = st.button("🔄 Re-generate data", use_container_width=True, key=_k("regen"))

with col_params:
    _SHAPE_NOTES = {
        "blobs":   "**Gaussian blobs** — four well-separated clusters. Watch the "
                   "VAE pull every cluster's latent code toward a shared, "
                   "overlapping region near the origin, while the plain "
                   "autoencoder is free to spread them arbitrarily far apart.",
        "moons":   "**Two moons** — one smooth non-linear manifold. A 2-D "
                   "bottleneck has to curl both crescents into a shape the "
                   "decoder can invert without confusing the two classes.",
        "circles": "**Concentric circles** — the hardest case: two nested loops "
                   "competing for the same 2-D latent plane.",
    }
    st.info(_SHAPE_NOTES[shape_key])

    with st.expander("How a VAE works", expanded=False):
        st.markdown(
            """
1. **Encode** — x → hidden → (μ, log σ²), the mean and log-variance of a
   Gaussian over latent codes
2. **Reparameterise** — sample z = μ + σ·ε, ε ~ N(0, I), so gradients can
   flow through the sampling step
3. **Decode** — z → hidden → x̂, the reconstruction
4. **Loss** = reconstruction error + β · KL(q(z\\|x) ‖ N(0, I))
5. Repeat until the loss stops improving
"""
        )

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _generate_and_train() -> None:
    X, labels = make_dataset(shape_key, n_points=n_points, seed=int(seed))
    run_ae = fit(
        X, labels, variational=False, beta=0.0, hidden_size=hidden_size,
        lr=lr, epochs=epochs, seed=int(seed),
    )
    run_vae = fit(
        X, labels, variational=True, beta=beta, hidden_size=hidden_size,
        lr=lr, epochs=epochs, seed=int(seed),
    )
    st.session_state[_k("run_ae")] = run_ae
    st.session_state[_k("run_vae")] = run_vae
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


_param_key = (shape_key, n_points, hidden_size, lr, epochs, beta, int(seed))
if _k("run_vae") not in st.session_state or regenerate or st.session_state.get(_k("_param_key")) != _param_key:
    _generate_and_train()
    st.session_state[_k("_param_key")] = _param_key

run_ae = st.session_state[_k("run_ae")]
run_vae = st.session_state[_k("run_vae")]
snapshots_ae = run_ae.snapshots
snapshots_vae = run_vae.snapshots
n_steps = len(snapshots_vae)  # identical epoch count for both runs

caption_slot.caption(
    f"Shape: **{shape_name}** | Points: **{n_points}** | Architecture: **{arch_name}** | "
    f"LR: **{lr}** | β: **{beta}** | Epochs: **{epochs}** | Seed: **{seed}**"
)

with col_main:
    about_section(
        "The VAE was one of the first models to make generative deep learning "
        "practical — instead of just compressing data like a plain "
        "autoencoder, it learns a smooth, structured latent space you can "
        "sample from to generate new data. The **reparameterisation trick** "
        "(sampling z = μ + σ·ε instead of sampling z directly) is the paper's "
        "key contribution: it makes the sampling step differentiable, so "
        "gradients can flow through it during training. The **Explore the "
        "decoder** panel below lets you probe exactly what that buys you — "
        "drag anywhere in latent space, including points the encoder never "
        "actually produced, and the decoder still returns something "
        "plausible, because training pulled the *whole* space toward looking "
        "like a smooth Gaussian, not just the points seen.",
        [
            "Kingma, D.P. & Welling, M. (2013). "
            "[\"Auto-Encoding Variational Bayes.\"](https://arxiv.org/abs/1312.6114) "
            "arXiv:1312.6114.",
        ],
    )

    # -------------------------------------------------------------------
    # Playback controls (pure Streamlit — no Plotly updatemenus)
    # -------------------------------------------------------------------
    # Defensively clamp step_idx to [0, n_steps-1] on every read — rapid clicking
    # (or a param change landing mid-autoplay, before this rerun resets it below)
    # could otherwise push it out of range and crash the snapshots_vae[step_idx]
    # lookups below with an IndexError.
    step_idx: int = st.session_state.get(_k("step_idx"), 0)
    step_idx = max(0, min(step_idx, n_steps - 1))
    st.session_state[_k("step_idx")] = step_idx
    playing: bool = st.session_state.get(_k("playing"), False)

    with st.container(border=True):
        speed = st.select_slider(
            "Playback speed", options=["0.5×", "1×", "2×", "4×"], value="1×",
            label_visibility="collapsed",
            key=_k("speed"),
        )
        DELAY = {"0.5×": 0.6, "1×": 0.3, "2×": 0.15, "4×": 0.06}[speed]

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
            text=f"Frame {step_idx + 1} / {n_steps} — epoch {snapshots_vae[step_idx].epoch}",
        )

    # -------------------------------------------------------------------
    # Metric cards
    # -------------------------------------------------------------------
    snap_ae = snapshots_ae[step_idx]
    snap_vae = snapshots_vae[step_idx]

    with st.container(border=True):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Epoch", str(snap_vae.epoch))
        m2.metric("AE reconstruction loss", f"{snap_ae.recon_loss:.4f}")
        m3.metric("VAE reconstruction loss", f"{snap_vae.recon_loss:.4f}")
        m4.metric("VAE KL loss", f"{snap_vae.kl_loss:.4f}")
        m5.metric("VAE total loss", f"{snap_vae.total_loss:.4f}")

    # -------------------------------------------------------------------
    # Data-space figures — AE vs VAE side by side
    # -------------------------------------------------------------------
    st.subheader("Data space — original points vs reconstructions")
    with st.container(border=True):
        col_ae_data, col_vae_data = st.columns(2)
        with col_ae_data:
            st.plotly_chart(
                make_data_space_figure(snap_ae, "Plain Autoencoder — data space"),
                use_container_width=True, config={"displayModeBar": False},
                key=_k("chart_data_ae"),
            )
        with col_vae_data:
            st.plotly_chart(
                make_data_space_figure(snap_vae, "VAE — data space"),
                use_container_width=True, config={"displayModeBar": False},
                key=_k("chart_data_vae"),
            )

    # -------------------------------------------------------------------
    # Latent-space figures — AE vs VAE side by side
    # -------------------------------------------------------------------
    st.subheader("Latent space — μ (and, for the VAE, σ as error bars) per point")
    with st.container(border=True):
        col_ae_latent, col_vae_latent = st.columns(2)
        with col_ae_latent:
            st.plotly_chart(
                make_latent_space_figure(snap_ae, "Plain Autoencoder — latent space", show_prior=False),
                use_container_width=True, config={"displayModeBar": False},
                key=_k("chart_latent_ae"),
            )
        with col_vae_latent:
            st.plotly_chart(
                make_latent_space_figure(snap_vae, "VAE — latent space", show_prior=True),
                use_container_width=True, config={"displayModeBar": False},
                key=_k("chart_latent_vae"),
            )

    # -------------------------------------------------------------------
    # Phase description — collapse vs regularisation, told with actual numbers
    # -------------------------------------------------------------------
    ae_spread = float(np.mean(np.linalg.norm(snap_ae.mu, axis=1)))
    vae_spread = float(np.mean(np.linalg.norm(snap_vae.mu, axis=1)))

    if snap_vae.phase == "init":
        st.info(
            "**Initialisation** — both models start from *identical* random "
            "encoder/decoder weights (same seed), so their latent spaces and "
            "reconstructions are the same right now. Every difference you see "
            "from here on is caused purely by the KL term in the VAE's loss."
        )
    else:
        st.info(
            f"**Epoch {snap_vae.epoch}** — mean distance of a latent code from the "
            f"origin: plain AE = **{ae_spread:.2f}**, VAE = **{vae_spread:.2f}** "
            f"(the standard normal prior has a typical radius around **1.25** in "
            f"2-D). The AE's codes are free to drift anywhere that helps "
            f"reconstruction; the VAE's KL term continually pulls its codes back "
            f"toward the prior, trading a little reconstruction accuracy for a "
            f"latent space where every region decodes to something plausible."
        )

    st.markdown("---")

    # -------------------------------------------------------------------
    # Loss curves
    # -------------------------------------------------------------------
    st.subheader("Training curves")
    history_ae = [(s.epoch, s.recon_loss, s.kl_loss, s.total_loss) for s in snapshots_ae]
    history_vae = [(s.epoch, s.recon_loss, s.kl_loss, s.total_loss) for s in snapshots_vae]
    with st.container(border=True):
        st.plotly_chart(
            make_loss_figure(history_ae, history_vae, snap_vae.epoch),
            use_container_width=True, config={"displayModeBar": False},
            key=_k("chart_loss"),
        )

    # -------------------------------------------------------------------
    # Interactive decoder — the reparameterisation-trick playground
    # -------------------------------------------------------------------
    st.subheader("Explore the decoder")
    st.caption(
        "Pick any point in the VAE's latent space (it doesn't have to be one "
        "the encoder ever produced) and see what the *current* decoder — at "
        "this exact frame — turns it into in data space. This is exactly what "
        "'sampling from the prior and decoding' means: the whole point of "
        "training the latent space to look like N(0, I)."
    )
    with st.container(border=True):
        col_sliders, col_latent_explore, col_data_explore = st.columns([1, 2, 2])
        with col_sliders:
            z1 = st.slider("z₁", -3.0, 3.0, 0.0, 0.1, key=_k("z1"))
            z2 = st.slider("z₂", -3.0, 3.0, 0.0, 0.1, key=_k("z2"))
            decoded = decode(np.array([[z1, z2]]), snap_vae.decoder_params)[0]
            st.metric("Decoded x₁", f"{decoded[0]:.2f}")
            st.metric("Decoded x₂", f"{decoded[1]:.2f}")
        with col_latent_explore:
            st.plotly_chart(
                make_latent_space_figure(snap_vae, "Your point in latent space", show_prior=True, highlight=(z1, z2)),
                use_container_width=True, config={"displayModeBar": False},
                key=_k("chart_latent_explore"),
            )
        with col_data_explore:
            st.plotly_chart(
                make_data_space_figure(snap_vae, "...decodes to this in data space", highlight=tuple(decoded)),
                use_container_width=True, config={"displayModeBar": False},
                key=_k("chart_data_explore"),
            )

    with st.expander("📖 Reading the visualisation"):
        st.markdown(
            """
- **Data space** — solid dots are the true points; faint "×" marks are their
  reconstructions, joined by a thin grey line. A short line means an
  accurate reconstruction. The pale scattered dots are the decoder's
  current *manifold*: a fixed grid of latent points, decoded — this is
  literally everything the model can currently generate.
- **Latent space** — each dot is a point's encoded mean μ. For the VAE,
  the faint error bars show σ = exp(0.5·log σ²), the spread the
  reparameterisation trick (z = μ + σ·ε) actually samples from. The
  dashed/dotted circles are the 1σ/2σ contours of the standard normal
  prior N(0, I) that the KL term pulls every point's distribution toward.
- **Plain AE vs VAE** — both start identical (same seed, same weights).
  The AE only minimises reconstruction error, so its latent space can
  drift, stretch, or leave gaps with no penalty. The VAE's extra KL term
  is the *only* difference, and it is entirely responsible for the latent
  space staying compact, centred, and smooth enough to decode from
  anywhere near the origin.
- **◀ Prev / Next ▶** to step epoch by epoch, **▶ Play** to auto-advance
  through the whole training run.
- **Explore the decoder** — drag z₁/z₂ to sample anywhere in latent space
  and watch the live decode; try a point far from the prior's 2σ circle
  to see decoding quality degrade outside the region the model was
  trained to cover.
"""
        )

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
