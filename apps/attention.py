"""Transformer Self-Attention page — adapted from the standalone
transformer-attention-viz repo for the multipage portfolio app.
Session-state keys are namespaced with a per-page prefix since
st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st

from attention.algorithm import build_head_specs, make_snapshots, run_attention
from attention.data import TEMPLATE_NAMES, build_sentence, make_embeddings
from attention.visualize import (
    make_embedding_heatmap,
    make_explain_word_figure,
    make_multihead_figure,
    make_stage_figure,
)
from common.ui import about_section, params_rail

NS = "attention"


def _k(name: str) -> str:
    return f"{NS}__{name}"


st.title("Transformer Self-Attention — Step-by-step Visualiser")
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

# ---------------------------------------------------------------------------
# Params rail
# ---------------------------------------------------------------------------
with params_rail(col_params):
    template_idx = st.selectbox(
        "Sentence template",
        options=range(len(TEMPLATE_NAMES)),
        format_func=lambda i: TEMPLATE_NAMES[i],
        help="Each template has hand-tagged syntactic roles and entity ids "
             "(which word refers to what). Only the concrete words are random.",
        key=_k("template_idx"),
    )

    pos_dim = st.slider(
        "Position encoding dimension", min_value=2, max_value=12, value=6, step=2,
        help="Size of the sinusoidal position block. Combined with the fixed "
             "role (6) and entity (3) blocks, this sets the total embedding "
             "dimension shown below.",
        key=_k("pos_dim"),
    )

    n_heads = st.slider(
        "Number of heads", min_value=1, max_value=4, value=3,
        help="1=Entity-linking only, 2=+Syntax, 3=+Position, 4=+a genuinely "
             "random untrained head shown for contrast.",
        key=_k("n_heads"),
    )

    seed = st.number_input(
        "Random seed (word choice)", min_value=0, max_value=9999, value=0, step=1,
        key=_k("seed"),
    )

    regenerate = st.button("🔄 Re-generate (new words)", use_container_width=True, key=_k("regenerate"))

with col_params:
    st.warning(
        "**Not a trained model.** There is no tokenizer, no learned embedding "
        "table, and no training loop anywhere in this app. Embeddings encode "
        "hand-assigned features (syntactic role, entity id, position) and the "
        "attention weights (Wq/Wk/Wv/Wo) are hand-built linear selectors, not "
        "learned. See the README for the full rationale.",
        icon="⚠️",
    )

    with st.expander("How self-attention works", expanded=False):
        st.markdown(
            """
1. Every token gets a Query, Key and Value vector
2. `scores = Q Kᵀ / √d_k` — how much each token relates to every other
3. `softmax(scores)` turns each row into a probability distribution
4. The output is the attention-weighted sum of Value vectors
5. Multiple heads run this in parallel and their outputs are summed back
   into the embedding (residual connection)
"""
        )

# ---------------------------------------------------------------------------
# Session state — rebuild the forward pass when parameters change
# ---------------------------------------------------------------------------
_param_key = (template_idx, pos_dim, n_heads, int(seed))

if regenerate:
    seed = int(np.random.default_rng().integers(0, 9999))
    _param_key = (template_idx, pos_dim, n_heads, seed)

if st.session_state.get(_k("_param_key")) != _param_key:
    sentence = build_sentence(template_idx, seed)
    X, blocks = make_embeddings(sentence, pos_dim)
    head_specs = build_head_specs(blocks, n_heads, seed=int(seed))
    results = run_attention(X, head_specs)
    snapshots = make_snapshots(sentence.words, X, results)

    st.session_state[_k("_param_key")] = _param_key
    st.session_state[_k("sentence")] = sentence
    st.session_state[_k("X")] = X
    st.session_state[_k("blocks")] = blocks
    st.session_state[_k("results")] = results
    st.session_state[_k("snapshots")] = snapshots
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False

sentence = st.session_state[_k("sentence")]
X = st.session_state[_k("X")]
blocks = st.session_state[_k("blocks")]
results = st.session_state[_k("results")]
snapshots = st.session_state[_k("snapshots")]
n_steps = len(snapshots)

caption_slot.caption(
    f"Template: **{TEMPLATE_NAMES[template_idx]}** | "
    f"Tokens: **{len(sentence.words)}** | "
    f"Embedding dim: **{blocks.total_dim}** (6 role + 3 entity + {pos_dim} position) | "
    f"Heads: **{n_heads}** | Seed: **{seed}**"
)

with col_main:
    about_section(
        "Self-attention is the mechanism inside every modern large language "
        "model — it lets a model weigh how much every token in a sequence "
        "should influence every other token, all in parallel, without the "
        "sequential bottleneck older recurrent architectures had. The 2017 "
        "paper that introduced it is one of the most-cited papers in computer "
        "science history. This page is upfront that its embeddings and "
        "attention weights are hand-built rather than learned (see the "
        "warning above) — but the actual Q/K/V-projection → scaled-scores → "
        "softmax → weighted-sum mechanics it animates in Section 2 are "
        "*exactly* what a trained model runs, just with hand-picked numbers "
        "standing in for learned ones.",
        [
            "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., "
            "Gomez, A.N., Kaiser, Ł., & Polosukhin, I. (2017). "
            "[\"Attention Is All You Need.\"](https://arxiv.org/abs/1706.03762) "
            "arXiv:1706.03762.",
        ],
    )

    sentence_html = "  ".join(
        f"<span style='padding:2px 6px;border-radius:4px;background:#eef2ff;"
        f"font-family:monospace;font-size:1.05em'>{w}</span>"
        for w in sentence.words
    )
    st.markdown(sentence_html, unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------------------------------------------
    # Section 1 — token embeddings (always visible, independent of playback)
    # -------------------------------------------------------------------
    st.subheader("1. Token embeddings")
    with st.container(border=True):
        st.plotly_chart(
            make_embedding_heatmap(sentence.words, X, blocks),
            use_container_width=True, config={"displayModeBar": False},
            key=_k("embedding_heatmap_main"),
        )
    with st.expander("What are these three blocks?"):
        st.markdown(
            """
- **role** (6 dims, one-hot) — DET / NOUN / VERB / ADJ / PRON / OTHER, assigned by the template, not predicted by a tagger
- **entity** (3 dims, one-hot) — which "thing" a word refers to; shared by a noun and any pronoun/adjective that points back at it. All-zero if the word has no referent.
- **position** (sinusoidal) — the standard Transformer position encoding, `sin`/`cos` at multiple frequencies
"""
        )

    st.markdown("---")

    # -------------------------------------------------------------------
    # Section 2 — step-through / auto-play walkthrough of the forward pass.
    # Lives inside a fragment: st.rerun() during autoplay used to fully
    # rerun the whole page (title/caption/params rail/about-section
    # included) several times a second, and small timing differences in
    # how long each of those took to re-render showed up as visible
    # flicker/layout shift on every frame. Scoping the rerun to just this
    # fragment keeps everything above it (and the sidebar) completely
    # static.
    # -------------------------------------------------------------------
    st.subheader("2. Forward pass, step by step")

    @st.fragment
    def _playback() -> None:
        # Defensive clamp: n_steps can shrink out from under a stale step_idx if a
        # parameter change and a queued auto-play rerun ever race each other.
        step_idx: int = st.session_state.get(_k("step_idx"), 0)
        step_idx = max(0, min(step_idx, n_steps - 1))
        st.session_state[_k("step_idx")] = step_idx
        playing: bool = st.session_state.get(_k("playing"), False)

        with st.container(border=True):
            speed = st.select_slider(
                "Playback speed", options=["0.5×", "1×", "2×", "4×"], value="1×",
                label_visibility="collapsed", key=_k("speed"),
            )
            DELAY = {"0.5×": 1.4, "1×": 0.7, "2×": 0.35, "4×": 0.18}[speed]

            col_prev, col_play, col_pause, col_next, col_speed = st.columns([1, 1.2, 1.2, 1, 3])

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

            with col_speed:
                st.caption(f"Speed: **{speed}**  ({DELAY:.2f}s per frame)")

            st.progress(
                step_idx / max(n_steps - 1, 1),
                text=f"Frame {step_idx + 1} / {n_steps} — {snapshots[step_idx].title}",
            )

        snap = snapshots[step_idx]

        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Phase", snap.phase.capitalize())
            m2.metric("Head", results[snap.head_idx].name if snap.head_idx >= 0 else "—")
            m3.metric("Stage", snap.substep.capitalize() if snap.substep else snap.phase.capitalize())
            m4.metric("d_k", str(results[snap.head_idx].spec.d_k) if snap.head_idx >= 0 else "—")

        # NB: the embed-phase frame reuses the exact same figure as Section 1 above
        # (same tokens/X/blocks) — give it a distinct `key` so Streamlit doesn't
        # collide the two plotly_chart elements' auto-generated IDs (they'd
        # otherwise be identical, which raises StreamlitDuplicateElementId on every
        # fresh page load, since frame 1 is always the embed phase by default).
        if snap.phase == "embed":
            fig = make_embedding_heatmap(sentence.words, X, blocks)
        else:
            fig = make_stage_figure(snap)
        with st.container(border=True):
            st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": False},
                key=_k("walkthrough_chart"),
            )

        st.info(snap.description)

        with st.expander("📖 Reading the walkthrough"):
            st.markdown(
                """
- **Embed** → each head runs **Q/K/V projection → raw scores → scaled scores → softmax → weighted sum** → **Combine** heads back into the residual
- **◀ Prev / Next ▶** to step manually, **▶ Play** to auto-advance through every stage of every head
- The metric cards above show exactly which head and pipeline stage the current chart illustrates
"""
            )

        # Auto-advance (must be last — triggers a fragment-scoped rerun after a delay)
        if playing:
            if step_idx < n_steps - 1:
                time.sleep(DELAY)
                st.session_state[_k("step_idx")] = step_idx + 1
                st.rerun(scope="fragment")
            else:
                st.session_state[_k("playing")] = False
                st.rerun(scope="fragment")

    _playback()

    st.markdown("---")

    # -------------------------------------------------------------------
    # Section 3 — attention heatmaps, all heads (independent of step index)
    # -------------------------------------------------------------------
    st.subheader("3. Attention heatmaps — every head")
    st.caption("Rows = query token, columns = key token, cell = attention weight (each row sums to 1).")
    with st.container(border=True):
        st.plotly_chart(
            make_multihead_figure(results, sentence.words),
            use_container_width=True, config={"displayModeBar": False},
            key=_k("multihead_heatmap"),
        )

    st.markdown("---")

    # -------------------------------------------------------------------
    # Section 4 — "explain a word"
    # -------------------------------------------------------------------
    st.subheader("4. Explain a word")
    st.caption("Pick a token and see exactly which other tokens each head attends to.")

    col_word, col_head = st.columns([1, 1])
    with col_word:
        # Keying on the token count forces Streamlit to instantiate a fresh
        # widget (defaulting back to index 0) whenever the template changes the
        # number of tokens, instead of silently keeping a now out-of-range index.
        query_idx = st.selectbox(
            "Query token", options=range(len(sentence.words)),
            format_func=lambda i: f"{i}: {sentence.words[i]}",
            key=_k(f"query_idx_select_{len(sentence.words)}"),
        )
    with col_head:
        # Same idea: keyed on n_heads so shrinking the head count can't leave
        # this selectbox pointing at a head name (e.g. "Random (untrained)")
        # that no longer exists in `results`.
        head_choice = st.selectbox(
            "Head", options=["All heads (side by side)"] + [r.name for r in results],
            key=_k(f"head_choice_select_{n_heads}"),
        )

    query_idx = min(int(query_idx), len(sentence.words) - 1)

    if head_choice == "All heads (side by side)":
        weights_by_head = [(r.name, r.weights) for r in results]
    else:
        weights_by_head = [(r.name, r.weights) for r in results if r.name == head_choice]
        if not weights_by_head:  # defensive fallback — stale selection, shouldn't happen with the key above
            weights_by_head = [(r.name, r.weights) for r in results]

    with st.container(border=True):
        st.plotly_chart(
            make_explain_word_figure(weights_by_head, sentence.words, query_idx),
            use_container_width=True, config={"displayModeBar": False},
            key=_k("explain_word_chart"),
        )

    # Highlighted-sentence view for the first matching head, so there's also a
    # plain-language reading of "what lit up"
    ref_weights = weights_by_head[0][1][query_idx]
    max_w = float(ref_weights.max()) or 1.0
    spans = []
    for i, w in enumerate(sentence.words):
        alpha = 0.15 + 0.7 * (ref_weights[i] / max_w)
        style = f"background: rgba(37,99,235,{alpha:.2f}); padding:2px 6px; border-radius:4px;"
        if i == query_idx:
            style += "border:2px solid #1d4ed8;font-weight:700;"
        spans.append(f"<span style='{style}font-family:monospace'>{w}</span>")
    st.markdown(
        f"Highlighted by **{weights_by_head[0][0]}** attention from "
        f"**“{sentence.words[query_idx]}”**:<br>" + "  ".join(spans),
        unsafe_allow_html=True,
    )
