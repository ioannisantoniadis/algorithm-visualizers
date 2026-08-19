"""Particle Filter page — a bootstrap (SIR) particle filter tracking the
same kind of 2-D target as kalman.py, for a direct side-by-side comparison
of "Gaussian belief" vs. "particle cloud" state estimation. Session-state
keys are namespaced with a per-page prefix since st.session_state is shared
across all pages.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from common.ui import about_section, params_rail
from particle_filter.algorithm import fit
from particle_filter.data import TRAJ_KEYS, TRAJ_NAMES, make_trajectory
from particle_filter.visualize import make_comparison_figure, make_static_figure

NS = "particle_filter"


def _k(name: str) -> str:
    return f"{NS}__{name}"


st.title("Particle Filter — Predict/Update/Resample Visualiser")
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

# ---------------------------------------------------------------------------
# Params rail
# ---------------------------------------------------------------------------
with params_rail(col_params, "Trajectory"):
    traj_name = st.selectbox(
        "Trajectory",
        options=TRAJ_NAMES,
        index=0,
        help="Geometry of the hidden true motion — the same four shapes as "
             "the Kalman Filter page, for a direct comparison.",
        key=_k("traj"),
    )
    traj_key = TRAJ_KEYS[TRAJ_NAMES.index(traj_name)]

    n_steps = st.slider("Number of time steps", min_value=20, max_value=150, value=60, step=5, key=_k("n_steps"))
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key=_k("seed"))

with params_rail(col_params, "Filter tuning"):
    n_particles = st.slider(
        "Number of particles",
        min_value=50, max_value=1000, value=300, step=50,
        help="More particles approximate the true belief distribution more "
             "closely, at the cost of more computation per step.",
        key=_k("n_particles"),
    )
    process_noise = st.slider(
        "Process noise — velocity std",
        min_value=0.05, max_value=2.0, value=0.5, step=0.05,
        help="How much manoeuvring the filter *expects* between measurements "
             "— diffused onto each particle's velocity every predict step. "
             "Too low → particles can't keep up with turns. Too high → the "
             "cloud stays diffuse and jittery even on a straight path.",
        key=_k("process_noise"),
    )
    measurement_noise = st.slider(
        "Measurement noise — R (std)",
        min_value=0.1, max_value=5.0, value=1.2, step=0.1,
        help="How noisy the sensor actually is. This value both generates "
             "the noisy observations and shapes each particle's likelihood "
             "weight against them.",
        key=_k("measurement_noise"),
    )
    resample_frac = st.slider(
        "Resample threshold (fraction of N)",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Resample whenever the effective sample size (ESS) falls below "
             "this fraction of the particle count — i.e. whenever too few "
             "particles are carrying most of the weight.",
        key=_k("resample_frac"),
    )

    regenerate = st.button("🔄 Re-generate trajectory", use_container_width=True, key=_k("regen"))

with col_params:
    _TRAJ_NOTES = {
        "linear": "**Constant velocity** — with sensible tuning the particle "
                  "cloud should track closely and smooth out most of the "
                  "sensor noise, resampling occasionally to stay focused.",
        "circular": "**Circular orbit** — the constant-velocity motion model "
                    "has no notion of curvature, so particles are always "
                    "slightly behind the turn until diffusion catches up.",
        "weave": "**Sine weave** — periodic direction changes stress-test "
                 "the process-noise trade-off: too little and the cloud cuts "
                 "every corner, too much and it chases sensor noise.",
        "maneuver": "**Maneuvering turns** — abrupt velocity changes are the "
                    "hardest case. Watch the particle cloud balloon right "
                    "after each turn and resample to refocus once enough "
                    "particles have drifted toward the new heading.",
    }
    st.info(_TRAJ_NOTES[traj_key])

    with st.expander("The predict–update–resample loop", expanded=False):
        st.markdown(
            """
1. **Predict** — every particle moves by its own velocity, then that velocity is perturbed by process noise
2. **Update** — each particle is re-weighted by how likely the new observation is, given that particle's position
3. **Resample** *(only when needed)* — when a few particles hold most of the weight (low ESS), draw a new set proportional to weight and reset to uniform
4. Repeat for every new observation
"""
        )

# ---------------------------------------------------------------------------
# Session state — (re)generate data & re-run filter on any relevant change
# ---------------------------------------------------------------------------
_param_key = (
    traj_key, n_steps, int(seed), n_particles,
    round(process_noise, 4), round(measurement_noise, 4), round(resample_frac, 4),
)

if (
    _k("_param_key") not in st.session_state
    or regenerate
    or st.session_state[_k("_param_key")] != _param_key
):
    true_pos, observations = make_trajectory(
        traj_key, n_steps=n_steps, dt=1.0, seed=int(seed), measurement_noise=measurement_noise,
    )
    snapshots = fit(
        observations, true_pos,
        n_particles=n_particles, process_noise=process_noise,
        measurement_noise=measurement_noise, resample_frac=resample_frac,
        seed=int(seed),
    )

    st.session_state[_k("snapshots")] = snapshots
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False
    st.session_state[_k("_param_key")] = _param_key

snapshots = st.session_state[_k("snapshots")]
n_frames = len(snapshots)

caption_slot.caption(
    f"Trajectory: **{traj_name}** | "
    f"Steps: **{n_steps}** | "
    f"Particles: **{n_particles}** | "
    f"Process noise: **{process_noise:.2f}** | "
    f"R: **{measurement_noise:.2f}** | "
    f"Seed: **{seed}**"
)

with col_main:
    about_section(
        "The particle filter (sequential Monte Carlo) represents the "
        "filter's belief as a *cloud* of weighted samples instead of a "
        "single Gaussian — which means it makes no assumption that the "
        "belief is unimodal or Gaussian-shaped at all. That generality is "
        "the whole trade: it can represent arbitrary, even multi-modal, "
        "beliefs (useful once the motion or observation model is "
        "non-linear or non-Gaussian, where a Kalman filter breaks down), "
        "at the cost of needing hundreds of samples instead of one closed-"
        "form update. Every particle **predicts** forward independently, "
        "gets **re-weighted** by how well it explains the new observation, "
        "and — periodically — the whole set is **resampled** to keep "
        "particle diversity from collapsing onto a handful of survivors. "
        "Compare this page to **Kalman Filter**: both track the same "
        "trajectories, but one carries a covariance matrix and the other "
        "carries a swarm.",
        [
            "Gordon, N.J., Salmond, D.J., Smith, A.F.M. (1993). \"Novel "
            "approach to nonlinear/non-Gaussian Bayesian state estimation.\" "
            "*IEE Proceedings F.*",
        ],
    )

    # -------------------------------------------------------------------
    # Playback controls, charts, metrics, and auto-advance all live inside
    # one fragment — see kalman.py for why this scoping matters (autoplay's
    # st.rerun() must not re-render the params rail / about-section above).
    # -------------------------------------------------------------------
    @st.fragment
    def _playback() -> None:
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
                    st.rerun(scope="fragment")

            with col_play:
                if st.button("▶  Play", use_container_width=True,
                             disabled=(playing or step_idx == n_frames - 1), type="primary", key=_k("play")):
                    st.session_state[_k("playing")] = True
                    st.rerun(scope="fragment")

            with col_pause:
                if st.button("⏸  Pause", use_container_width=True, disabled=not playing, key=_k("pause")):
                    st.session_state[_k("playing")] = False
                    st.rerun(scope="fragment")

            with col_next:
                if st.button("Next ▶", use_container_width=True,
                             disabled=(step_idx == n_frames - 1 or playing), key=_k("next")):
                    st.session_state[_k("step_idx")] = min(n_frames - 1, step_idx + 1)
                    st.rerun(scope="fragment")

            with col_speed:
                st.caption(f"Speed: **{speed}**  ({DELAY:.2f}s per frame)")

            st.progress(step_idx / max(n_frames - 1, 1),
                        text=f"Frame {step_idx + 1} / {n_frames} — {snapshots[step_idx].title}")

        snap = snapshots[step_idx]

        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Phase", snap.phase.capitalize())
            m2.metric("Time step", str(snap.step))
            m3.metric("Effective sample size", f"{snap.ess:.0f} / {snap.n_particles}")
            if snap.phase == "update":
                err = float(np.linalg.norm(snap.pos_mean - snap.true_pos))
                m4.metric("Estimate error", f"{err:.2f}")
            else:
                m4.metric("Estimate error", "—")

        fig = make_static_figure(snapshots, step_idx)
        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=_k("chart"))

        if snap.phase == "init":
            st.info(
                "**Initial particle cloud** — before any predict/update, "
                "particles are scattered around the first observation with "
                "random velocity guesses (true velocity is unknown at t=0)."
            )
        elif snap.phase == "predict":
            st.info(
                "**Predict step** — every particle moves according to its "
                "own current velocity estimate, then that velocity is "
                "perturbed by process noise. No observation is used here; "
                "this is why the cloud spreads out."
            )
        elif snap.phase == "update":
            err = float(np.linalg.norm(snap.pos_mean - snap.true_pos))
            st.info(
                f"**Update step** — every particle is re-weighted by how "
                f"likely the new observation is given that particle's "
                f"position. Weighted-mean estimate is now **{err:.2f}** "
                f"units from the (hidden) true position. ESS: "
                f"**{snap.ess:.0f}** of {snap.n_particles} particles."
            )
        else:
            st.success(
                "**Resample** — the effective sample size dropped below "
                "threshold (too few particles were carrying most of the "
                "weight), so a fresh set was drawn proportional to weight "
                "and reset to uniform. Watch the cloud snap back to a tight "
                "cluster around the surviving high-weight region."
            )

        with st.expander("📖 Reading the animation"):
            st.markdown(
                """
- **Dotted grey line** — the hidden true trajectory (only shown for grading; the filter never sees it)
- **Rose ×** — noisy sensor observations, the filter's only input
- **Violet dots** — individual particles; larger and more opaque = higher weight
- **Indigo line + dots** — the particle filter's running weighted-mean estimate
- **◀ Prev / Next ▶** to step manually, **▶ Play** to auto-advance
"""
            )

        st.subheader("Raw vs. particle-filtered — full run")

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
            c2.metric("RMSE — particle-filter estimate", f"{rmse_filtered:.3f}")
            c3.metric("Improvement", f"{improvement:.1f}%")

            st.plotly_chart(make_comparison_figure(snapshots), use_container_width=True,
                            config={"displayModeBar": False}, key=_k("comparison_chart"))

        # Auto-advance (must be last — triggers a fragment-scoped rerun after a delay)
        if playing:
            if step_idx < n_frames - 1:
                time.sleep(DELAY)
                st.session_state[_k("step_idx")] = min(n_frames - 1, step_idx + 1)
                st.rerun(scope="fragment")
            else:
                st.session_state[_k("playing")] = False
                st.rerun(scope="fragment")

    _playback()
