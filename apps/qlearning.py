"""Q-learning / SARSA page — adapted from the standalone qlearning-viz repo
for the multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import streamlit as st

from common.ui import about_section, params_rail
from qlearning.algorithm import Snapshot, episode_summaries, train
from qlearning.data import PRESET_KEYS, PRESET_NAMES, make_grid
from qlearning.visualize import (
    make_compare_reward_figure,
    make_policy_figure,
    make_reward_figure,
    make_static_figure,
)

NS = "qlearning"


def _k(name: str) -> str:
    return f"{NS}__{name}"


st.title("Q-learning & SARSA — Temporal-Difference Learning on a Grid")
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

# ---------------------------------------------------------------------------
# Params rail — grid
# ---------------------------------------------------------------------------
with params_rail(col_params, "Grid & training"):
    preset_name = st.selectbox("Grid preset", PRESET_NAMES, index=0, key=_k("preset_name"))
    preset_key = PRESET_KEYS[PRESET_NAMES.index(preset_name)]

    if preset_key == "classic":
        rows, cols = 3, 5
        st.caption("Fixed 3×5 layout — reproduces grid-world-mdp exactly.")
    elif preset_key == "cliff":
        rows = st.slider("Rows", min_value=3, max_value=6, value=4, key=_k("rows"))
        cols = st.slider("Columns", min_value=6, max_value=16, value=12, key=_k("cols"))
    else:
        rows = st.slider("Rows", min_value=3, max_value=10, value=5, key=_k("rows"))
        cols = st.slider("Columns", min_value=3, max_value=12, value=7, key=_k("cols"))

    seed = st.number_input(
        "Random seed", min_value=0, max_value=9999, value=42, step=1,
        help=(
            "Classic and Cliff walk have a fixed layout, so this only reseeds "
            "the training run (env slips + ε-greedy choices) — it will NOT "
            "change the grid. For Random hazards it reseeds both the hazard "
            "layout and the training run."
            if preset_key != "random_hazards" else
            "Reseeds both the hazard layout and the training run's randomness."
        ),
        key=_k("seed"),
    )

    regenerate = st.button("🔄 Regenerate & retrain", use_container_width=True, key=_k("regen"))

# ---------------------------------------------------------------------------
# Params rail — algorithm
# ---------------------------------------------------------------------------
with params_rail(col_params, "Algorithm"):
    algorithm_name = st.radio("Method", ["Q-learning", "SARSA"], key=_k("algorithm_name"))
    algorithm_key = "qlearning" if algorithm_name == "Q-learning" else "sarsa"
    other_algorithm_key = "sarsa" if algorithm_key == "qlearning" else "qlearning"
    other_algorithm_name = "SARSA" if algorithm_key == "qlearning" else "Q-learning"

    compare_enabled = st.checkbox(
        f"Compare with {other_algorithm_name}",
        help="Also train the other algorithm with identical hyperparameters, "
             "and show both final policies and reward curves side by side.",
        key=_k("compare_enabled"),
    )

# ---------------------------------------------------------------------------
# Params rail — hyperparameters
# ---------------------------------------------------------------------------
with params_rail(col_params, "Hyperparameters"):
    alpha = st.slider("Learning rate (α)", 0.01, 1.0, 0.3, step=0.01, key=_k("alpha"))
    gamma = st.slider("Discount factor (γ)", 0.50, 0.999, 0.95, step=0.01, key=_k("gamma"))
    n_episodes = st.slider("Episodes", min_value=10, max_value=300, value=150, step=10, key=_k("n_episodes"))
    max_steps = st.slider("Max steps per episode", min_value=10, max_value=150, value=80, step=5, key=_k("max_steps"))

    st.markdown("**ε-greedy exploration schedule**")
    epsilon_start = st.slider("ε start", 0.0, 1.0, 1.0, step=0.05, key=_k("epsilon_start"))
    epsilon_end = st.slider("ε end", 0.0, 1.0, 0.05, step=0.01, key=_k("epsilon_end"))
    epsilon_decay_episodes = st.slider(
        "Decay over N episodes", min_value=1, max_value=n_episodes,
        value=min(60, n_episodes),
        help="ε falls linearly from 'ε start' to 'ε end' over this many episodes, "
             "then stays at 'ε end'. Early on the agent mostly explores randomly; "
             "later it mostly exploits its learned Q-table.",
        key=_k("epsilon_decay_episodes"),
    )

# ---------------------------------------------------------------------------
# Params rail — display
# ---------------------------------------------------------------------------
with params_rail(col_params, "Display"):
    background_name = st.radio("Grid background", ["Cell kind", "Max Q-value heatmap"], key=_k("background_name"))
    background_key = "qvalue" if background_name.startswith("Max") else "kind"

with col_params:
    _PRESET_NOTES = {
        "classic": "**Classic** — the exact 3×5 resource grid from grid-world-mdp "
                   "(iron, diamond, two lava cells), with the same 80/10/10 "
                   "stochastic movement. That repo solved it with full knowledge "
                   "of the transition model (policy iteration); here the agent "
                   "has to *discover* the same rewards purely through trial and error.",
        "cliff": "**Cliff walk** — Sutton & Barto's classic example. Every step "
                 "costs −1, falling off the cliff costs −100 and resets you to "
                 "start. Watch **Q-learning** converge to the risky optimal path "
                 "hugging the cliff edge, while **SARSA** learns a safer route "
                 "one row back — enable the comparison below to see both at once.",
        "random_hazards": "**Random hazards** — a fresh scattered layout of "
                   "resources and hazards per seed. Change the seed and press "
                   "**Regenerate & retrain** for a brand-new layout to learn.",
    }
    st.info(_PRESET_NOTES[preset_key])

    st.markdown(
        """
**How the update differs**

- **Q-learning** (off-policy): `Q(s,a) += α · [r + γ·maxₐ' Q(s',a') − Q(s,a)]`
- **SARSA** (on-policy): `Q(s,a) += α · [r + γ·Q(s',a') − Q(s,a)]`
  using the action `a'` exploration will *actually* take next
"""
    )

# ---------------------------------------------------------------------------
# Session state — grid / training
# ---------------------------------------------------------------------------


def _train_primary() -> None:
    grid = st.session_state[_k("grid")]
    st.session_state[_k("snapshots")] = train(
        grid, algorithm_key, n_episodes=n_episodes, max_steps=max_steps,
        alpha=alpha, gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes, seed=int(seed),
    )
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


def _train_secondary() -> None:
    grid = st.session_state[_k("grid")]
    st.session_state[_k("compare_snapshots")] = train(
        grid, other_algorithm_key, n_episodes=n_episodes, max_steps=max_steps,
        alpha=alpha, gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes, seed=int(seed),
    )


_param_key = (preset_key, rows, cols, int(seed))
_hparam_key = (alpha, gamma, epsilon_start, epsilon_end, epsilon_decay_episodes,
               n_episodes, max_steps, int(seed))
# Both cache keys below include _param_key (grid identity) as well as the
# hyperparameters. Without the grid identity, changing preset/rows/cols
# while "Compare with..." was unchecked left compare_snapshots trained on
# the OLD grid's shape; re-enabling the checkbox with otherwise-unchanged
# hyperparameters would then skip retraining (the key looked unchanged) and
# make_policy_figure would index a stale (rows, cols, 4) Q-table against the
# NEW grid's dimensions — an out-of-bounds IndexError on any grid-size change.
_train_key = _param_key + (algorithm_key,) + _hparam_key
_other_key = _param_key + (other_algorithm_key,) + _hparam_key

grid_changed = (
    _k("grid") not in st.session_state
    or regenerate
    or st.session_state.get(_k("_param_key")) != _param_key
)

if grid_changed:
    st.session_state[_k("grid")] = make_grid(preset_key, rows, cols, seed=int(seed))
    st.session_state[_k("_param_key")] = _param_key
    _train_primary()
    st.session_state[_k("_train_key")] = _train_key
    if compare_enabled:
        _train_secondary()
        st.session_state[_k("_other_key")] = _other_key
elif st.session_state.get(_k("_train_key")) != _train_key:
    _train_primary()
    st.session_state[_k("_train_key")] = _train_key
    if compare_enabled:
        _train_secondary()
        st.session_state[_k("_other_key")] = _other_key
elif compare_enabled and st.session_state.get(_k("_other_key")) != _other_key:
    _train_secondary()
    st.session_state[_k("_other_key")] = _other_key

grid = st.session_state[_k("grid")]
snapshots: list[Snapshot] = st.session_state[_k("snapshots")]
n_steps = len(snapshots)

caption_slot.caption(
    f"Grid: **{grid.name}** ({grid.rows}×{grid.cols}) | "
    f"Algorithm: **{algorithm_name}** | "
    f"Episodes: **{n_episodes}** | α=**{alpha}** γ=**{gamma}** | "
    f"Seed: **{seed}**"
)

with col_main:
    about_section(
        "Q-learning was one of the first algorithms to prove that an agent "
        "can learn an optimal policy purely through trial and error — "
        "without ever being told the rules of its environment — which made "
        "it foundational to modern reinforcement learning; the same family "
        "of ideas, scaled way up, trains game-playing agents and "
        "RLHF-tuned language models today. The **Compare with SARSA** option "
        "on this page is the textbook illustration of *off-policy* vs. "
        "*on-policy* learning: Q-learning always assumes it will act "
        "greedily in the future, while SARSA accounts for the exploration it "
        "will actually do — switch to the **Cliff walk** preset and enable "
        "the comparison to see why that difference makes Q-learning hug the "
        "cliff edge while SARSA learns a safer route.",
        [
            "Watkins, C.J.C.H. (1989). \"Learning from Delayed Rewards.\" "
            "PhD thesis, University of Cambridge.",
            "Rummery, G.A. & Niranjan, M. (1994). \"On-Line Q-Learning Using "
            "Connectionist Systems.\" Technical Report CUED/F-INFENG/TR 166, "
            "University of Cambridge — introduced SARSA.",
        ],
    )

    # -------------------------------------------------------------------
    # Playback controls
    # -------------------------------------------------------------------
    step_idx: int = min(st.session_state.get(_k("step_idx"), 0), n_steps - 1)
    playing: bool = st.session_state.get(_k("playing"), False)

    with st.container(border=True):
        speed = st.select_slider(
            "Playback speed", options=["0.5×", "1×", "2×", "4×", "8×", "32×"], value="4×",
            label_visibility="collapsed", key=_k("speed"),
        )
        DELAY = {"0.5×": 0.9, "1×": 0.45, "2×": 0.22, "4×": 0.11, "8×": 0.05, "32×": 0.01}[speed]

        col_prev, col_play, col_pause, col_next, col_speed = st.columns([1, 1.2, 1.2, 1, 3])

        with col_prev:
            if st.button("◀ Prev", use_container_width=True, disabled=(step_idx == 0 or playing), key=_k("prev")):
                st.session_state[_k("step_idx")] = step_idx - 1
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
                st.session_state[_k("step_idx")] = step_idx + 1
                st.rerun()

        with col_speed:
            st.caption(f"Speed: **{speed}**  ({DELAY:.2f}s per frame, {n_steps} steps total)")

        col_jump, col_go, col_end, _pad = st.columns([2, 1, 1.4, 3])
        with col_jump:
            jump_episode = st.number_input(
                "Jump to episode", min_value=0, max_value=n_episodes,
                value=snapshots[step_idx].episode, step=1, label_visibility="collapsed",
                key=_k("jump_episode"),
            )
        with col_go:
            if st.button("Go", use_container_width=True, disabled=playing, key=_k("go")):
                for i, s in enumerate(snapshots):
                    if s.episode == jump_episode:
                        st.session_state[_k("step_idx")] = i
                        st.rerun()
        with col_end:
            if st.button("⏭ Skip to end", use_container_width=True, disabled=playing, key=_k("skip_end")):
                st.session_state[_k("step_idx")] = n_steps - 1
                st.rerun()

        snap = snapshots[step_idx]
        st.progress(step_idx / max(n_steps - 1, 1), text=f"Step {step_idx + 1} / {n_steps} — {snap.title}")

    # -------------------------------------------------------------------
    # Grid figure + metrics
    # -------------------------------------------------------------------
    chart_col, side_col = st.columns([3, 2])

    with chart_col, st.container(border=True):
        fig = make_static_figure(snap, background=background_key)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=_k("main_grid_chart"))

    with side_col, st.container(border=True):
        m1, m2 = st.columns(2)
        m1.metric("Episode", str(snap.episode))
        m2.metric("Step in episode", str(snap.step_in_episode))
        m3, m4 = st.columns(2)
        m3.metric("ε (epsilon)", f"{snap.epsilon:.2f}")
        m4.metric("Episode return", f"{snap.episode_return:.2f}" if snap.episode > 0 else "—")

        if snap.phase == "init":
            st.info(
                "**Untrained** — every Q(s,a) starts at 0, so the greedy "
                "policy shown (arrows) is arbitrary. Press **▶ Play** to train."
            )
        else:
            verb = "Exploring" if snap.phase == "explore" else "Exploiting"
            slip_note = " — the stochastic grid **slipped** the move sideways" if snap.slipped else ""
            end_note = " **Episode ended.**" if snap.done else ""
            st.info(
                f"**{verb}** at {snap.state} — took action toward "
                f"**{['north','east','south','west'][snap.action]}**{slip_note}, "
                f"landed on {snap.next_state}, reward **{snap.reward:+.2f}**, "
                f"TD-error **{snap.td_error:+.3f}**.{end_note}"
            )

        st.plotly_chart(
            make_reward_figure(snapshots[: step_idx + 1], name=algorithm_name),
            use_container_width=True, config={"displayModeBar": False}, key=_k("reward_chart"),
        )

    with st.expander("📖 Reading the animation"):
        st.markdown(
            """
- **Coloured cells** — grid preset's resources/hazards: iron **I**, diamond **D**
  (positive terminal reward), lava **L**, pit **P** (negative terminal reward),
  cliff **C** (big penalty, resets to start), goal **G**
- **Arrows** — the current greedy policy: `argmax_a Q(s, a)` at every non-terminal cell
- **Gold dotted line** — where that greedy policy would currently walk from start
- **Orange circle** — the agent's position after this step
- **Green / red dotted line** — the move just taken; red-dotted means the
  stochastic grid **slipped** it sideways (10% chance either way)
- Switch **Grid background** in the sidebar to a **max Q-value heatmap** to
  watch the values themselves propagate outward from the rewards, instead
  of just the policy they imply
- **▶ Play** trains live; **Jump to episode** / **⏭ Skip to end** jump around
  without stepping through every single environment interaction
"""
        )

    # -------------------------------------------------------------------
    # Q-learning vs SARSA comparison
    # -------------------------------------------------------------------
    if compare_enabled and _k("compare_snapshots") in st.session_state:
        st.markdown("---")
        st.subheader(f"{algorithm_name} vs {other_algorithm_name} — after {n_episodes} episodes")

        compare_snapshots = st.session_state[_k("compare_snapshots")]
        col_a, col_b = st.columns(2)
        with col_a:
            # NOTE: always the actual FINAL trained Q-table (snapshots[-1]), not
            # `snap` — `snap` is whatever frame the scrubber is currently on, which
            # would make this panel's "final policy" caption false the moment the
            # user steps back to an earlier frame.
            st.plotly_chart(
                make_policy_figure(grid, snapshots[-1].q_table, f"{algorithm_name} — final policy", background_key),
                use_container_width=True, config={"displayModeBar": False}, key=_k("compare_policy_primary"),
            )
        with col_b:
            st.plotly_chart(
                make_policy_figure(
                    grid, compare_snapshots[-1].q_table,
                    f"{other_algorithm_name} — final policy", background_key,
                ),
                use_container_width=True, config={"displayModeBar": False}, key=_k("compare_policy_secondary"),
            )

        st.plotly_chart(
            make_compare_reward_figure(snapshots, algorithm_name, compare_snapshots, other_algorithm_name),
            use_container_width=True, config={"displayModeBar": False}, key=_k("compare_reward_chart"),
        )

        _eps_a, ret_a, _len_a, _e_a = episode_summaries(snapshots)
        _eps_b, ret_b, _len_b, _e_b = episode_summaries(compare_snapshots)
        if len(ret_a) and len(ret_b):
            tail = max(1, len(ret_a) // 5)
            avg_a, avg_b = ret_a[-tail:].mean(), ret_b[-tail:].mean()
            better = algorithm_name if avg_a > avg_b else other_algorithm_name
            st.caption(
                f"Average return over the last {tail} episodes: "
                f"**{algorithm_name} = {avg_a:.2f}**, **{other_algorithm_name} = {avg_b:.2f}** — "
                f"**{better}** is currently ahead on this grid. "
                + (
                    "On Cliff walk this is the textbook result: Q-learning's off-policy "
                    "target keeps assuming it will act greedily, so it settles right next "
                    "to the cliff; SARSA's on-policy target factors in exploration actually "
                    "stepping off sometimes, so it backs away — even though that makes its "
                    "*greedy* policy slightly longer."
                    if preset_key == "cliff" else ""
                )
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
