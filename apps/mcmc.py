"""MCMC page — adapted from the standalone mcmc-viz repo for the
multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import streamlit as st

from mcmc.algorithm import run_mcmc
from mcmc.data import TARGET_DEFAULTS, TARGET_KEYS, TARGET_NAMES, grid_density, marginal_x
from mcmc.visualize import make_figure

NS = "mcmc"


def _k(name: str) -> str:
    return f"{NS}__{name}"


@st.cache_data(show_spinner=False)
def _grid(key: str):
    return grid_density(key, n=120)


@st.cache_data(show_spinner=False)
def _marginal(key: str):
    return marginal_x(key, n=400)


# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------
st.sidebar.markdown("##### ⚙️ PARAMETERS")

with st.sidebar.container(border=True):
    target_name = st.selectbox(
        "Target distribution",
        options=TARGET_NAMES,
        index=0,
        help="The unnormalised density the chain is trying to sample from. "
             "Metropolis-Hastings never needs to know its normalising constant.",
        key=_k("target_name"),
    )
    target_key = TARGET_KEYS[TARGET_NAMES.index(target_name)]
    defaults = TARGET_DEFAULTS[target_key]

    proposal_sigma = st.slider(
        "Proposal σ (random-walk step size)",
        min_value=0.05,
        max_value=float(defaults["extent"]),
        value=float(defaults["default_sigma"]),
        step=0.05,
        help="Too small → the chain crawls and mixes slowly (high acceptance, "
             "high autocorrelation, needs many steps). Too large → most "
             "proposals land in low-density territory and get rejected (low "
             "acceptance, the chain stalls in place).",
        key=_k("proposal_sigma"),
    )
    n_steps = st.slider("Chain length (steps)", min_value=50, max_value=1500, value=400, step=50, key=_k("n_steps"))
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key=_k("seed"))

    regenerate = st.button("🔄 Re-run chain", use_container_width=True, key=_k("regen"))

_TARGET_NOTES = {
    "gaussian": "**Correlated Gaussian** — the friendly baseline. A well-tuned "
                "isotropic proposal mixes efficiently here; use it to see what "
                "'good' acceptance and autocorrelation look like.",
    "banana":   "**Banana** — a curved, warped Gaussian. No single fixed, "
                "isotropic proposal variance fits its shape everywhere, so "
                "mixing is slower along the curve even when tuned well.",
    "bimodal":  "**Bimodal** — two separated peaks. With a small σ the chain "
                "can get trapped in one mode for a very long time; try raising "
                "σ (or re-running with a different seed) to see it jump across.",
    "ring":     "**Ring** — a thin, non-convex ring of mass. A large σ mostly "
                "proposes points inside the hole or outside the ring — both "
                "rejected — so acceptance drops fast as σ grows.",
}
st.sidebar.info(_TARGET_NOTES[target_key])

st.sidebar.markdown(
    """
**Metropolis-Hastings in one pass**

1. Propose `y = x + N(0, σ²I)` from the current state `x`
2. Compute the acceptance ratio `α = p̃(y) / p̃(x)`
3. Accept `y` with probability `min(1, α)`; otherwise stay at `x`
4. Repeat — the resulting chain's samples converge to the target
"""
)

# ---------------------------------------------------------------------------
# Run (or re-use) the chain
# ---------------------------------------------------------------------------
_param_key = (target_key, n_steps, float(proposal_sigma), int(seed))
if _k("snapshots") not in st.session_state or regenerate or st.session_state.get(_k("_param_key")) != _param_key:
    st.session_state[_k("snapshots")] = run_mcmc(
        target_key, n_steps=n_steps, proposal_sigma=proposal_sigma, seed=int(seed)
    )
    st.session_state[_k("_param_key")] = _param_key
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False

snapshots = st.session_state[_k("snapshots")]
n_frames = len(snapshots)

grid = _grid(target_key)
marginal = _marginal(target_key)

# Live acceptance-rate trend in the sidebar
rates = [s.acceptance_rate for s in snapshots[1:]]  # skip iteration 0 (undefined rate)
if rates:
    st.sidebar.markdown("**Cumulative acceptance rate**")
    st.sidebar.line_chart({"acceptance rate": rates}, height=160)
    final_rate = rates[-1]
    if final_rate > 0.7:
        note = "Very high — σ may be too small (slow exploration)."
    elif final_rate < 0.15:
        note = "Very low — σ may be too large (many rejections)."
    else:
        note = "Healthy range (roughly 15–70%) for random-walk MH."
    st.sidebar.caption(f"Final: **{final_rate:.1%}** — {note}")

# ---------------------------------------------------------------------------
# Main area header
# ---------------------------------------------------------------------------
st.title("Metropolis-Hastings — MCMC Visualiser")
st.caption(
    f"Target: **{target_name}** | Proposal σ: **{proposal_sigma:.2f}** | "
    f"Steps: **{n_steps}** | Seed: **{seed}**"
)

# ---------------------------------------------------------------------------
# Playback controls (pure Streamlit — no Plotly updatemenus)
# ---------------------------------------------------------------------------
step_idx: int = st.session_state.get(_k("step_idx"), 0)
step_idx = max(0, min(step_idx, n_frames - 1))
st.session_state[_k("step_idx")] = step_idx
playing: bool = st.session_state.get(_k("playing"), False)

with st.container(border=True):
    speed = st.select_slider(
        "Playback speed",
        options=["0.5×", "1×", "2×", "4×", "8×"],
        value="2×",
        label_visibility="collapsed",
        key=_k("speed"),
    )
    DELAY = {"0.5×": 0.6, "1×": 0.3, "2×": 0.15, "4×": 0.07, "8×": 0.03}[speed]

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

    st.progress(
        step_idx / max(n_frames - 1, 1),
        text=f"Step {step_idx} / {n_frames - 1} — {snapshots[step_idx].title}",
    )

# ---------------------------------------------------------------------------
# Metrics + dashboard for the current snapshot
# ---------------------------------------------------------------------------
snap = snapshots[step_idx]

with st.container(border=True):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Iteration", str(snap.iteration))
    m2.metric("Cumulative acceptance", f"{snap.acceptance_rate:.1%}" if snap.iteration else "—")
    m3.metric("log p̃(current)", f"{snap.log_density:.2f}")
    if snap.iteration == 0:
        m4.metric("Outcome", "Start")
    else:
        m4.metric("Outcome", "Accepted ✓" if snap.accepted else "Rejected ✗")

fig = make_figure(snap, grid, marginal, max_lag=40)
with st.container(border=True):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=_k("chart"))

if snap.iteration == 0:
    st.info(
        "**Start** — the chain is deliberately initialised away from the bulk of the "
        "probability mass, so you can watch it find its way there (this early phase "
        "is called *burn-in*)."
    )
elif snap.accepted:
    st.success(
        "**Accepted** — the proposal landed somewhere at least as dense as the "
        "current state (or won the coin-flip test against a less-dense point), "
        "so the chain moves there."
    )
else:
    st.warning(
        "**Rejected** — the proposal landed in lower-density territory and lost "
        "the coin-flip test, so the chain repeats its current state — visible as "
        "a flat step in the trace plot."
    )

with st.expander("📖 Reading the dashboard"):
    st.markdown(
        """
- **Top-left** — the target density (indigo contours) and the path the chain has walked so
  far. The indigo star marks the current state; a dotted rose line marks a proposal that was
  just rejected (the "road not taken").
- **Top-right** — the trace plot. Flat stretches mean rejected proposals; a slowly wandering
  line (rather than a jittery one) is the visual signature of *poor mixing*.
- **Bottom-left** — the growing histogram of visited x-coordinates against the true marginal
  density (green). Watch it converge as the chain runs longer.
- **Bottom-right** — autocorrelation of the x-coordinate. Fast decay to ~0 means samples
  become independent quickly; slow decay means you need many more steps for the same
  effective sample size.
- Use **◀ Prev / Next ▶** to step one proposal at a time, or **▶ Play** to auto-advance.
"""
    )

# ---------------------------------------------------------------------------
# Auto-advance (must be last — triggers rerun after a delay)
# ---------------------------------------------------------------------------
if playing:
    if step_idx < n_frames - 1:
        time.sleep(DELAY)
        st.session_state[_k("step_idx")] = step_idx + 1
        st.rerun()
    else:
        st.session_state[_k("playing")] = False
        st.rerun()
