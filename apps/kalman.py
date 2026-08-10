"""Kalman Filter page — adapted from the standalone kalman-filter-viz repo
for the multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from kalman.algorithm import fit
from kalman.data import TRAJ_KEYS, TRAJ_NAMES, make_trajectory
from kalman.visualize import make_comparison_figure, make_static_figure

NS = "kalman"


def _k(name: str) -> str:
    return f"{NS}__{name}"


# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------
st.sidebar.markdown("##### ⚙️ PARAMETERS")

with st.sidebar.container(border=True):
    traj_name = st.selectbox(
        "Trajectory",
        options=TRAJ_NAMES,
        index=0,
        help="Geometry of the hidden true motion. Curved and maneuvering paths "
             "expose the mismatch between the filter's constant-velocity model "
             "and reality.",
        key=_k("traj"),
    )
    traj_key = TRAJ_KEYS[TRAJ_NAMES.index(traj_name)]

    n_steps = st.slider("Number of time steps", min_value=20, max_value=150, value=60, step=5, key=_k("n_steps"))
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key=_k("seed"))

st.sidebar.markdown("##### FILTER TUNING")

with st.sidebar.container(border=True):
    q = st.slider(
        "Process noise — Q (std)",
        min_value=0.01, max_value=3.0, value=0.15, step=0.01,
        help="How much manoeuvring the filter *expects* between measurements. "
             "Too low → the filter trusts its motion model too much and lags "
             "behind turns. Too high → it trusts noisy measurements too much "
             "and the estimate becomes jittery.",
        key=_k("q"),
    )
    r = st.slider(
        "Measurement noise — R (std)",
        min_value=0.1, max_value=5.0, value=1.2, step=0.1,
        help="How noisy the sensor actually is. This value both generates the "
             "noisy observations and tells the filter how much to trust them.",
        key=_k("r"),
    )

    regenerate = st.button("🔄 Re-generate trajectory", use_container_width=True, key=_k("regen"))

_TRAJ_NOTES = {
    "linear": "**Constant velocity** — the scenario the model was built for. "
              "With a sensible Q the estimate should track closely and smooth "
              "out most of the sensor noise.",
    "circular": "**Circular orbit** — a constant-velocity model has no notion "
                "of curvature, so the filter is always slightly behind the turn. "
                "Raise Q to let it keep up; watch the ellipse stay small if Q is "
                "too low even as the estimate lags.",
    "weave": "**Sine weave** — periodic direction changes stress-test the "
             "Q trade-off: too smooth and it cuts every corner, too reactive "
             "and it chases sensor noise instead of the pattern.",
    "maneuver": "**Maneuvering turns** — abrupt velocity changes are the "
                "hardest case for a constant-velocity model. Watch the "
                "uncertainty ellipse balloon right after each turn.",
}
st.sidebar.info(_TRAJ_NOTES[traj_key])

with st.sidebar.expander("The predict–update loop", expanded=False):
    st.markdown(
        """
1. **Predict** — propagate state through the motion model; uncertainty *grows*
2. **Update** — fold in the new noisy measurement via the Kalman gain; uncertainty *shrinks*
3. Repeat for every new observation
"""
    )

# ---------------------------------------------------------------------------
# Session state — (re)generate data & re-run filter on any relevant change
# ---------------------------------------------------------------------------
_param_key = (traj_key, n_steps, int(seed), round(q, 4), round(r, 4))

if (
    _k("_param_key") not in st.session_state
    or regenerate
    or st.session_state[_k("_param_key")] != _param_key
):
    true_pos, observations = make_trajectory(
        traj_key, n_steps=n_steps, dt=1.0, seed=int(seed), measurement_noise=r,
    )
    snapshots = fit(observations, true_pos, dt=1.0, q=q, r=r)

    st.session_state[_k("snapshots")] = snapshots
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False
    st.session_state[_k("_param_key")] = _param_key

snapshots = st.session_state[_k("snapshots")]
n_frames = len(snapshots)

# ---------------------------------------------------------------------------
# Main area header
# ---------------------------------------------------------------------------
st.title("Kalman Filter — Predict/Update Visualiser")

st.caption(
    f"Trajectory: **{traj_name}** | "
    f"Steps: **{n_steps}** | "
    f"Q: **{q:.2f}** | "
    f"R: **{r:.2f}** | "
    f"Seed: **{seed}**"
)

# ---------------------------------------------------------------------------
# Playback controls  (pure Streamlit — no Plotly updatemenus)
# ---------------------------------------------------------------------------
# Defensive clamp: guards against a stale step_idx surviving into a rerun
# where n_frames shrank (e.g. a mid-autoplay parameter change) before the
# session-state reset above has a chance to run.
step_idx: int = st.session_state.get(_k("step_idx"), 0)
step_idx = max(0, min(step_idx, n_frames - 1))
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
                     disabled=(playing or step_idx == n_frames - 1), type="primary", key=_k("play")):
            st.session_state[_k("playing")] = True
            st.rerun()

    with col_pause:
        if st.button("⏸  Pause", use_container_width=True, disabled=not playing, key=_k("pause")):
            st.session_state[_k("playing")] = False
            st.rerun()

    with col_next:
        if st.button("Next ▶", use_container_width=True,
                     disabled=(step_idx == n_frames - 1 or playing), key=_k("next")):
            st.session_state[_k("step_idx")] = min(n_frames - 1, step_idx + 1)
            st.rerun()

    with col_speed:
        st.caption(f"Speed: **{speed}**  ({DELAY:.2f}s per frame)")

    st.progress(step_idx / max(n_frames - 1, 1),
                text=f"Frame {step_idx + 1} / {n_frames} — {snapshots[step_idx].title}")

# ---------------------------------------------------------------------------
# Chart + metrics for the current step
# ---------------------------------------------------------------------------
snap = snapshots[step_idx]

with st.container(border=True):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Phase", snap.phase.capitalize())
    m2.metric("Time step", str(snap.step))
    m3.metric("Uncertainty (trace of P)", f"{snap.pos_uncertainty:.2f}")
    if snap.phase == "update":
        m4.metric("Innovation |y|", f"{np.linalg.norm(snap.innovation):.2f}")
    else:
        m4.metric("Innovation |y|", "—")

fig = make_static_figure(snapshots, step_idx)
with st.container(border=True):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=_k("chart"))

if snap.phase == "init":
    st.info(
        "**Initial belief** — before any measurement is folded in, the filter "
        "starts at the first observation with completely unknown velocity, "
        "hence the wide uncertainty ellipse."
    )
elif snap.phase == "predict":
    st.info(
        "**Predict step** — the state is propagated through the constant-"
        "velocity motion model, and the covariance grows by the process "
        "noise **Q**. No measurement is used here; this is pure "
        "extrapolation, so uncertainty can only increase."
    )
else:
    err = float(np.linalg.norm(snap.pos_mean - snap.true_pos))
    st.info(
        f"**Update step** — the new noisy observation is folded in via the "
        f"Kalman gain **K**, shrinking the ellipse. Estimate is now "
        f"**{err:.2f}** units from the (hidden) true position."
    )

with st.expander("📖 Reading the animation"):
    st.markdown(
        """
- **Dotted grey line** — the hidden true trajectory (only shown for grading; the filter never sees it)
- **Rose ×** — noisy sensor observations, the filter's only input
- **Indigo line + dots** — the Kalman filter's running position estimate
- **Shaded indigo ellipse** — 2σ uncertainty region; grows on **Predict**, shrinks on **Update**
- **◀ Prev / Next ▶** to step manually, **▶ Play** to auto-advance
"""
    )

# ---------------------------------------------------------------------------
# Always-on comparison panel: raw observations vs. Kalman-smoothed path
# ---------------------------------------------------------------------------
st.subheader("Raw vs. Kalman-smoothed — full run")

update_snaps = [s for s in snapshots if s.phase in ("init", "update")]
true_full = np.array([s.true_pos for s in update_snaps])
obs_full = np.array([s.observation for s in update_snaps])
est_full = np.array([s.pos_mean for s in update_snaps])

rmse_raw = float(np.sqrt(np.mean(np.sum((obs_full - true_full) ** 2, axis=1))))
rmse_filtered = float(np.sqrt(np.mean(np.sum((est_full - true_full) ** 2, axis=1))))
improvement = (1 - rmse_filtered / rmse_raw) * 100 if rmse_raw > 0 else 0.0

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE — raw observations", f"{rmse_raw:.3f}")
    c2.metric("RMSE — Kalman estimate", f"{rmse_filtered:.3f}")
    c3.metric("Improvement", f"{improvement:.1f}%")

    st.plotly_chart(make_comparison_figure(snapshots), use_container_width=True,
                    config={"displayModeBar": False}, key=_k("comparison_chart"))

# ---------------------------------------------------------------------------
# Auto-advance (must be last — triggers rerun after a delay)
# ---------------------------------------------------------------------------
if playing:
    if step_idx < n_frames - 1:
        time.sleep(DELAY)
        st.session_state[_k("step_idx")] = min(n_frames - 1, step_idx + 1)
        st.rerun()
    else:
        st.session_state[_k("playing")] = False
        st.rerun()
