"""Diffusion model (DDPM) page — adapted from the standalone diffusion-viz repo
for the multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from diffusion.algorithm import forward_process, make_schedule, reverse_process, train_score_network
from diffusion.data import SHAPE_KEYS, SHAPE_NAMES, make_dataset
from diffusion.visualize import axis_range, make_loss_figure, make_point_figure, make_schedule_figure

NS = "diffusion"


def _k(name: str) -> str:
    return f"{NS}__{name}"


# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------
st.sidebar.markdown("##### ⚙️ PARAMETERS")

with st.sidebar.container(border=True):
    shape_name = st.selectbox(
        "Data shape",
        options=SHAPE_NAMES,
        index=0,
        help="The 2-D distribution the forward process dissolves into noise, "
             "and the reverse process tries to reconstruct.",
        key=_k("shape"),
    )
    shape_key = SHAPE_KEYS[SHAPE_NAMES.index(shape_name)]

    n_points = st.slider("Number of points", min_value=80, max_value=400, value=200, step=10, key=_k("n_points"))
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key=_k("seed"))

st.sidebar.markdown("##### NOISE SCHEDULE")

with st.sidebar.container(border=True):
    T = st.slider(
        "T — number of diffusion steps", min_value=10, max_value=200, value=180, step=5,
        help="How many discrete steps the forward process takes to turn data into "
             "pure noise. More steps = smoother, easier-to-reverse chain, but slower "
             "sampling.",
        key=_k("T"),
    )
    schedule_kind = st.radio(
        "Schedule shape", ["linear", "cosine"], index=0,
        help="linear: β_t grows linearly (original DDPM). cosine: gentler decay of "
             "ᾱ_t at the start/end of the chain, spending more steps in the hardest "
             "medium-noise regime.",
        key=_k("schedule_kind"),
    )
    # NOTE: with the default T (180) and the linear schedule's default endpoints
    # (β_start=1e-4, β_end=0.045), ᾱ_T ≈ 0.016 — i.e. ~98% of the original signal
    # variance is gone by the final step, matching the "indistinguishable from
    # N(0, I)" claim made in the info box below. A previous default of T=60 with
    # β_end=0.02 left ᾱ_T ≈ 0.545 (54% of the signal still present), directly
    # contradicting that claim on first load.
    beta_start = st.slider(
        "β start", 0.0001, 0.01, 0.0001, 0.0001, format="%.4f",
        disabled=(schedule_kind == "cosine"),
        help="Only used by the linear schedule — the cosine schedule derives its "
             "own β_t from a fixed cosine ᾱ_t curve." if schedule_kind == "cosine" else None,
        key=_k("beta_start"),
    )
    beta_end = st.slider(
        "β end", 0.01, 0.2, 0.045, 0.005, format="%.3f",
        disabled=(schedule_kind == "cosine"),
        help="Only used by the linear schedule — the cosine schedule derives its "
             "own β_t from a fixed cosine ᾱ_t curve." if schedule_kind == "cosine" else None,
        key=_k("beta_end"),
    )

st.sidebar.markdown("##### SCORE NETWORK")

with st.sidebar.container(border=True):
    hidden_name = st.radio("Hidden layer width", ["Small (H=32)", "Medium (H=64)", "Large (H=128)"], index=1, key=_k("hidden_name"))
    hidden_size = {"Small (H=32)": 32, "Medium (H=64)": 64, "Large (H=128)": 128}[hidden_name]
    lr = st.slider("Learning rate", min_value=0.005, max_value=0.2, value=0.05, step=0.005, format="%.3f", key=_k("lr"))
    epochs = st.slider("Training epochs", min_value=50, max_value=1000, value=300, step=50, key=_k("epochs"))
    regenerate = st.button("🔄 Re-generate data & retrain", use_container_width=True, key=_k("regen"))

_SHAPE_NOTES = {
    "moons": "**Two moons** — a single smooth non-convex manifold. The classic "
             "toy distribution for demonstrating a diffusion model end to end.",
    "ring": "**Ring** — one closed loop. The score network must learn a purely "
            "*radial* restoring force pointing back onto the circle from any angle.",
    "blobs": "**Gaussian blobs** — disconnected clusters. The reverse process must "
              "commit to a mode early: once a sample is near one blob at low noise, "
              "it can't jump to another.",
}
st.sidebar.info(_SHAPE_NOTES[shape_key])

st.sidebar.markdown(
    """
**How a diffusion model works**

1. **Forward** — repeatedly blend in a little Gaussian noise until the data
   distribution becomes indistinguishable from N(0, I)
2. **Train** — a tiny MLP learns to predict *which noise* was added to a
   sample, at any step t (denoising score matching)
3. **Reverse** — start from pure noise, repeatedly ask the network "what
   noise do you see?" and subtract a bit of it
4. If training worked, the reverse process recovers the original shape
"""
)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _generate_and_run() -> None:
    X0, labels = make_dataset(shape_key, n_points=n_points, seed=int(seed))
    schedule = make_schedule(T, kind=schedule_kind, beta_start=beta_start, beta_end=beta_end)

    forward_snaps = forward_process(X0, schedule, seed=int(seed))
    run = train_score_network(
        X0, schedule, hidden_size=hidden_size, lr=lr, epochs=epochs, seed=int(seed),
    )
    reverse_snaps = reverse_process(run, schedule, n_points=n_points, seed=int(seed) + 1)

    st.session_state[_k("X0")] = X0
    st.session_state[_k("labels")] = labels
    st.session_state[_k("schedule")] = schedule
    st.session_state[_k("forward_snaps")] = forward_snaps
    st.session_state[_k("run")] = run
    st.session_state[_k("reverse_snaps")] = reverse_snaps
    st.session_state[_k("x_range")], st.session_state[_k("y_range")] = axis_range([forward_snaps, reverse_snaps])
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


_param_key = (shape_key, n_points, T, schedule_kind, beta_start, beta_end, hidden_size, lr, epochs, int(seed))
if _k("run") not in st.session_state or regenerate or st.session_state.get(_k("_param_key")) != _param_key:
    with st.spinner("Running forward process, training score network, sampling..."):
        _generate_and_run()
    st.session_state[_k("_param_key")] = _param_key

X0 = st.session_state[_k("X0")]
labels = st.session_state[_k("labels")]
schedule = st.session_state[_k("schedule")]
forward_snaps = st.session_state[_k("forward_snaps")]
run = st.session_state[_k("run")]
reverse_snaps = st.session_state[_k("reverse_snaps")]
x_range = st.session_state[_k("x_range")]
y_range = st.session_state[_k("y_range")]

# ---------------------------------------------------------------------------
# Main area header
# ---------------------------------------------------------------------------
st.title("Diffusion Model (DDPM) — Forward & Reverse Process")
st.caption(
    f"Shape: **{shape_name}** | Points: **{n_points}** | T: **{T}** | "
    f"Schedule: **{schedule_kind}** | Hidden: **{hidden_name}** | "
    f"LR: **{lr}** | Epochs: **{epochs}** | Seed: **{seed}**"
)
st.markdown("---")

# ---------------------------------------------------------------------------
# View selector — forward vs reverse
# ---------------------------------------------------------------------------
view = st.radio(
    "Which process do you want to watch?",
    ["Forward process (noising)", "Reverse process (denoising / sampling)"],
    horizontal=True,
    key=_k("view"),
)
is_forward = view.startswith("Forward")
snapshots = forward_snaps if is_forward else reverse_snaps
n_steps = len(snapshots)

# Reset the step index whenever the view is switched, so Prev/Next always
# starts from a sensible end of the chain currently being watched
if st.session_state.get(_k("_last_view")) != view:
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False
    st.session_state[_k("_last_view")] = view

# ---------------------------------------------------------------------------
# Playback controls (pure Streamlit — no Plotly updatemenus)
# ---------------------------------------------------------------------------
step_idx: int = st.session_state.get(_k("step_idx"), 0)
# Clamp defensively: rapid clicking can fire a click on Prev/Next (or a stale
# "Jump to step" slider value) before Streamlit has re-rendered the button's
# disabled state, letting step_idx drift past either end of the chain and
# raising an IndexError a few lines down at snapshots[step_idx]. Clamping
# here — and again in every handler below — makes step_idx always safe to
# index with, regardless of how fast the UI is clicked.
step_idx = max(0, min(step_idx, n_steps - 1))
st.session_state[_k("step_idx")] = step_idx
playing: bool = st.session_state.get(_k("playing"), False)

with st.container(border=True):
    speed = st.select_slider(
        "Playback speed", options=["0.5×", "1×", "2×", "4×"], value="1×",
        label_visibility="collapsed",
        key=_k("speed"),
    )
    DELAY = {"0.5×": 0.4, "1×": 0.2, "2×": 0.1, "4×": 0.04}[speed]

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

    # A direct slider lets you jump straight to any diffusion step, not just
    # step through one frame at a time
    slider_idx = st.slider(
        "Jump to step", min_value=0, max_value=n_steps - 1, value=step_idx,
        disabled=playing, label_visibility="visible",
        key=_k("jump_slider"),
    )
    if slider_idx != step_idx and not playing:
        st.session_state[_k("step_idx")] = max(0, min(slider_idx, n_steps - 1))
        st.rerun()

    st.progress(
        step_idx / max(n_steps - 1, 1),
        text=f"Frame {step_idx + 1} / {n_steps} — {snapshots[step_idx].title}",
    )

# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------
snap = snapshots[step_idx]
snr_display = "∞" if np.isinf(snap.snr_t) else f"{snap.snr_t:.3f}"

with st.container(border=True):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Step t", str(snap.step))
    m2.metric("β_t", f"{snap.beta_t:.5f}")
    m3.metric("ᾱ_t (signal remaining)", f"{snap.alpha_bar_t:.3f}")
    m4.metric("SNR_t", snr_display)

# ---------------------------------------------------------------------------
# Point-cloud figure
# ---------------------------------------------------------------------------
if is_forward:
    fig = make_point_figure(snap, labels, x_range, y_range, ref_points=None)
else:
    fig = make_point_figure(snap, None, x_range, y_range, ref_points=X0, title=snap.title)

with st.container(border=True):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=_k("main_point_chart"))

if is_forward:
    if snap.step == 0:
        st.info(
            "**t=0 — the original data.** From here, each step blends in a little "
            "Gaussian noise: `x_t = sqrt(1-β_t)·x_{t-1} + sqrt(β_t)·ε_t`. No "
            "learning happens in this direction — it's a fixed, known process."
        )
    elif snap.step == T:
        st.info(
            f"**t={T} — pure noise.** After {T} steps essentially all structure is "
            f"gone: ᾱ_t = {snap.alpha_bar_t:.4f}, meaning only "
            f"{snap.alpha_bar_t * 100:.2f}% of the original signal variance "
            "survives. The distribution is now indistinguishable from N(0, I)."
        )
    else:
        st.info(
            f"**Step t={snap.step}.** ᾱ_t = {snap.alpha_bar_t:.3f} of the original "
            f"signal variance remains; the rest has been replaced by noise. "
            f"SNR_t = {snr_display} — a rough measure of how hard it would be to "
            "recover x₀ from this frame alone."
        )
else:
    if snap.step == T:
        st.info(
            "**t=T — starting point.** Every generated sample begins as pure "
            "Gaussian noise, exactly matching the distribution the forward "
            "process converges to. The faint grey cloud is the *real* training "
            "data, shown as a target to compare against."
        )
    elif snap.step == 0:
        st.info(
            "**t=0 — final sample.** This is the model's output: pure noise, "
            "repeatedly denoised one small step at a time using only the score "
            "network's noise predictions. Compare the coloured points against "
            "the faint reference cloud — a well-trained model should roughly "
            "recover the target shape."
        )
    else:
        st.info(
            f"**Step t={snap.step}.** The network predicts the noise ε̂ currently "
            f"in the sample and a small amount of it is subtracted, following the "
            f"DDPM update `x_(t-1) = (x_t - β_t/√(1-ᾱ_t)·ε̂) / √α_t + √β_t·z`. "
            f"ᾱ_t = {snap.alpha_bar_t:.3f}."
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Noise schedule + training loss — always visible, static
# ---------------------------------------------------------------------------
col_schedule, col_loss = st.columns(2)
with col_schedule:
    with st.container(border=True):
        st.plotly_chart(make_schedule_figure(schedule), use_container_width=True,
                         config={"displayModeBar": False}, key=_k("schedule_chart"))
with col_loss:
    with st.container(border=True):
        st.plotly_chart(make_loss_figure(run.loss_history), use_container_width=True,
                         config={"displayModeBar": False}, key=_k("loss_chart"))

st.caption(
    f"Final training loss: **{run.loss_history[-1]:.4f}** "
    f"(started at **{run.loss_history[0]:.4f}**) over **{epochs}** epochs."
)

with st.expander("📖 Reading the visualisation"):
    st.markdown(
        """
- **Forward process** — a fixed, deterministic-in-distribution chain; no
  training involved. Every step samples fresh noise and blends it in. Watch
  the coloured clusters spread out and lose their shape as t increases.
- **Reverse process** — starts from pure noise (t=T) and denoises one step
  at a time using the *trained* score network's noise prediction. The faint
  grey cloud is the real data, shown only as a visual target — the model
  never gets to see it during sampling.
- **β_t** — how much noise variance is injected at step t. **ᾱ_t** — the
  fraction of the *original* signal variance still present after t steps
  (cumulative product of `1-β_s` for s≤t). **SNR_t = ᾱ_t / (1-ᾱ_t)** — signal
  divided by noise; it decays roughly exponentially, which is why the
  y-axis on the SNR panel is logarithmic.
- **Training loss** — E[‖ε − ε̂‖²], the mean squared error between the true
  noise added and the network's prediction, averaged over a fresh random
  (x₀, t, ε) each epoch. Falling loss means the network is getting better at
  identifying noise at *every* step, not just one.
- **◀ Prev / Next ▶** to step one frame at a time, **▶ Play** to
  auto-advance, or drag **Jump to step** to go straight to any t.
- Switching between **Forward** and **Reverse** resets the step slider to
  the start of that process.
"""
    )

# ---------------------------------------------------------------------------
# Auto-advance (must be last — triggers rerun after a delay)
# ---------------------------------------------------------------------------
if playing:
    if step_idx < n_steps - 1:
        time.sleep(DELAY)
        st.session_state[_k("step_idx")] = min(n_steps - 1, step_idx + 1)
        st.rerun()
    else:
        st.session_state[_k("playing")] = False
        st.rerun()
