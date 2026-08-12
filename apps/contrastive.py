"""Contrastive Learning page — adapted from the standalone contrastive-learning-viz
repo for the multipage portfolio app. Session-state keys are namespaced with a
per-page prefix since st.session_state is shared across all pages.
"""

from __future__ import annotations

import time

import streamlit as st

from common.ui import about_section, params_rail
from contrastive.algorithm import fit
from contrastive.data import AUG_KEYS, AUG_NAMES, SHAPE_KEYS, SHAPE_NAMES, make_dataset
from contrastive.visualize import (
    make_before_after_figure,
    make_embedding_figure,
    make_input_figure,
    make_loss_figure,
    make_similarity_heatmap,
)

NS = "contrastive"


def _k(name: str) -> str:
    return f"{NS}__{name}"


st.title("Contrastive Learning (SimCLR-style) — Step-by-step Visualiser")
st.caption(
    "Pull augmented views of the same point together, push every other "
    "point apart — with **no labels at all**."
)
caption_slot = st.empty()

col_params, col_main = st.columns([1, 3])

# ---------------------------------------------------------------------------
# Params rail
# ---------------------------------------------------------------------------
with params_rail(col_params, "Data"):
    shape_name = st.selectbox(
        "Dataset",
        options=SHAPE_NAMES,
        index=0,
        help="Each cluster is an unlabelled 'identity'. The encoder never sees "
             "these cluster ids — it only ever sees two augmented views at a time.",
        key=_k("shape"),
    )
    shape_key = SHAPE_KEYS[SHAPE_NAMES.index(shape_name)]

    n_points = st.slider("Number of points", min_value=20, max_value=160, value=80, step=4, key=_k("n_points"))
    n_clusters = st.slider(
        "Number of clusters", min_value=2, max_value=8, value=4,
        help="Shapes how many groups the dataset is generated with. These "
             "group ids are ground truth, used only to colour the plots — the "
             "encoder is never told which point belongs to which group.",
        key=_k("n_clusters"),
    )
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=0, step=1, key=_k("seed"))

with params_rail(col_params, "Augmentation (positive-pair mechanism)"):
    aug_name = st.radio(
        "Augmentation type",
        options=AUG_NAMES,
        index=3,
        help="A positive pair is two independently augmented views of the same "
             "point. Rotation and scaling are applied around the origin; jitter "
             "is additive Gaussian noise.",
        key=_k("aug_name"),
    )
    aug_kind = AUG_KEYS[AUG_NAMES.index(aug_name)]
    aug_strength = st.slider(
        "Augmentation strength", min_value=0.0, max_value=1.0, value=0.65, step=0.05,
        help="0 = views are identical to the original point (trivial positive "
             "pairs). 1 = large rotation/scale/jitter — harder to recognise the "
             "same point twice.",
        key=_k("aug_strength"),
    )

with params_rail(col_params, "Contrastive loss"):
    tau = st.slider(
        "Temperature τ", min_value=0.05, max_value=1.0, value=0.2, step=0.05,
        help="Low τ sharpens the softmax over negatives — the loss punishes "
             "even mildly-similar negatives hard, pushing them further apart. "
             "High τ is more forgiving and the embedding stays more spread out.",
        key=_k("tau"),
    )
    batch_size = st.slider(
        "Batch size", min_value=4, max_value=min(64, n_points), value=min(16, n_points), step=2,
        help="Number of base points sampled per step; each contributes two "
             "augmented views, so the loss sees 2×batch_size vectors and "
             "2×batch_size − 2 negatives per anchor.",
        key=_k("batch_size"),
    )

with params_rail(col_params, "Optimisation"):
    lr = st.slider("Learning rate", min_value=0.05, max_value=3.0, value=0.8, step=0.05, key=_k("lr"))
    steps = st.slider("Training steps", min_value=10, max_value=300, value=120, step=10, key=_k("steps"))

    arch_name = st.radio(
        "Encoder architecture",
        ["Small (2 → 16 → 2)", "Deep (2 → 16 → 16 → 2)"],
        help="Embedding dimension is fixed at 2 so the whole representation "
             "(post-normalisation, the unit circle) can be plotted directly — "
             "no PCA/t-SNE needed to look at it.",
        key=_k("arch_name"),
    )
    hidden_sizes = (16,) if arch_name.startswith("Small") else (16, 16)

    regenerate = st.button("🔄 Re-generate data & retrain", use_container_width=True, key=_k("regen"))

with col_params:
    _SHAPE_NOTES = {
        "blobs": "**Gaussian blobs** — well-separated clusters. The encoder "
                 "should find this easy: any embedding that respects local "
                 "structure separates them.",
        "ring":  "**Clusters on a circle** — neighbouring clusters sit closer "
                 "together, so NT-Xent has to work harder to keep them apart "
                 "while pulling each cluster's own views together.",
    }
    st.info(_SHAPE_NOTES[shape_key])

    st.markdown(
        """
**How contrastive learning works**

1. Sample a batch of points, make **two augmented views** of each
2. **Forward** — embed all views, normalise onto the unit circle
3. **NT-Xent loss** — pull each point's own two views together,
   push every other view in the batch apart
4. **Backward** — propagate the loss gradient through the encoder
5. Repeat — no labels used anywhere
"""
    )

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _generate_and_train() -> None:
    X, labels = make_dataset(shape_key, n_points=n_points, n_clusters=n_clusters, seed=int(seed))
    run = fit(
        X, labels,
        hidden_sizes=hidden_sizes,
        tau=tau,
        batch_size=batch_size,
        lr=lr,
        steps=steps,
        aug_strength=aug_strength,
        aug_kind=aug_kind,
        seed=int(seed),
    )
    st.session_state[_k("run")] = run
    st.session_state[_k("step_idx")] = 0
    st.session_state[_k("playing")] = False


_param_key = (shape_key, n_points, n_clusters, int(seed), hidden_sizes, tau,
              batch_size, lr, steps, aug_strength, aug_kind)
if _k("run") not in st.session_state or regenerate or st.session_state.get(_k("_param_key")) != _param_key:
    _generate_and_train()
    st.session_state[_k("_param_key")] = _param_key

run = st.session_state[_k("run")]
snapshots = run.snapshots
n_steps = len(snapshots)

caption_slot.caption(
    f"Dataset: **{shape_name}** | Points: **{n_points}** | Clusters: **{n_clusters}** | "
    f"Augmentation: **{aug_name}** (strength {aug_strength:.2f}) | "
    f"τ: **{tau:.2f}** | Batch: **{batch_size}** | LR: **{lr}** | Seed: **{seed}**"
)

with col_main:
    about_section(
        "SimCLR showed that a model can learn genuinely useful visual "
        "representations with *zero labels*, just by teaching it to recognise "
        "that two randomly-augmented views of the same point are \"the same "
        "thing\" while every other point in the batch is \"different.\" That "
        "result helped trigger the modern shift toward self-supervised "
        "pretraining, now standard practice before fine-tuning on a specific "
        "labelled task. The NT-Xent loss animated below — pulling each "
        "point's own two views together, pushing every other view in the "
        "batch apart — is the entire mechanism; the **Before vs. after** "
        "panel above is the payoff: structure the encoder was never told "
        "about emerges anyway, purely from that one rule.",
        [
            "Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). "
            "[\"A Simple Framework for Contrastive Learning of Visual "
            "Representations.\"](https://arxiv.org/abs/2002.05709) "
            "arXiv:2002.05709.",
        ],
    )

    # -------------------------------------------------------------------
    # Before / after — the headline comparison
    # -------------------------------------------------------------------
    with st.expander("🔀 Before vs. after training (unit-circle embedding)", expanded=True):
        st.caption(
            "The random encoder's mapping is arbitrary — its output has no relation "
            "to any *meaningful* structure. Because same-cluster points sit close "
            "together in the raw 2-D input, they can already land in the same rough "
            "patch of the circle by geometric accident, but at very different "
            "(also arbitrary) locations than after training. After training on "
            "nothing but augmented-view agreement, same-cluster points settle into "
            "tighter, better-separated groups — structure the algorithm was never "
            "told to find."
        )
        with st.container(border=True):
            st.plotly_chart(
                make_before_after_figure(snapshots[0], snapshots[-1], n_clusters),
                use_container_width=True, config={"displayModeBar": False},
                key=_k("chart_before_after"),
            )

    st.markdown("---")

    # -------------------------------------------------------------------
    # Playback controls, metrics, charts, and auto-advance all live inside
    # one fragment: st.rerun() during autoplay used to fully rerun the
    # whole page (title/caption/params rail/about-section included)
    # several times a second, and small timing differences in how long
    # each of those took to re-render showed up as visible flicker/layout
    # shift on every frame. Scoping the rerun to just this fragment keeps
    # everything above it (and the sidebar) completely static.
    # -------------------------------------------------------------------
    @st.fragment
    def _playback() -> None:
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
            DELAY = {"0.5×": 0.9, "1×": 0.45, "2×": 0.22, "4×": 0.1}[speed]

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

        st.markdown("---")

        snap = snapshots[step_idx]

        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Phase", snap.phase.capitalize())
            m2.metric("Step", str(snap.step))
            m3.metric("NT-Xent loss", f"{snap.loss:.3f}" if snap.step > 0 else "—")
            m4.metric(
                "Pos / neg cos-sim",
                f"{snap.pos_sim:.2f} / {snap.neg_sim:.2f}" if snap.step > 0 else "—",
            )

        col_input, col_embed = st.columns(2)
        with col_input:
            with st.container(border=True):
                st.plotly_chart(
                    make_input_figure(snap, n_clusters),
                    use_container_width=True, config={"displayModeBar": False},
                    key=_k("chart_input_space"),
                )
        with col_embed:
            with st.container(border=True):
                st.plotly_chart(
                    make_embedding_figure(snap, n_clusters),
                    use_container_width=True, config={"displayModeBar": False},
                    key=_k("chart_embedding_space"),
                )

        if snap.phase == "init":
            st.info(
                "**Initialisation** — random weights. The embedding has no relation "
                "yet to any semantic structure — any clustering you see here is a "
                "geometric accident of the raw layout, not something the encoder has "
                "learned."
            )
        elif snap.phase == "forward":
            st.info(
                f"**Forward pass** — a fresh batch is augmented into two views each, "
                f"embedded, and normalised. NT-Xent loss = **{snap.loss:.3f}**. "
                f"Mean positive-pair similarity **{snap.pos_sim:.2f}**, mean "
                f"negative-pair similarity **{snap.neg_sim:.2f}** — training pushes "
                f"the first up and the second down."
            )
        else:
            status = (
                "Loss has stopped improving — training has **converged**."
                if snap.converged
                else "Every weight now has a gradient; the update happens right after this frame."
            )
            st.info(
                f"**Backward pass** — the NT-Xent gradient is propagated back "
                f"through L2-normalisation and every dense layer via the chain "
                f"rule (see `contrastive/algorithm.py` for the full derivation). {status}"
            )

        st.markdown("---")

        history = [(s.step, s.loss, s.pos_sim, s.neg_sim) for s in snapshots if s.phase == "forward"]
        with st.container(border=True):
            st.plotly_chart(
                make_loss_figure(history, snap.step),
                use_container_width=True, config={"displayModeBar": False},
                key=_k("chart_loss_curve"),
            )

        with st.expander("🔬 Batch similarity matrix"):
            if snap.sim_matrix is None:
                st.caption(
                    "This frame is the random **initialisation** — no batch has been "
                    "sampled yet, so there's no similarity matrix to show. Step "
                    "forward to the first training step to see it."
                )
            else:
                st.caption(
                    "Every view in the current batch against every other. The open "
                    "black squares mark each anchor's single positive pair; everything "
                    "else is a negative. Training should brighten the marked cells "
                    "(cos sim → 1) and dim the rest (cos sim → −1 or 0)."
                )
                with st.container(border=True):
                    st.plotly_chart(
                        make_similarity_heatmap(snap),
                        use_container_width=True, config={"displayModeBar": False},
                        key=_k("chart_similarity_heatmap"),
                    )

        with st.expander("📖 Reading the animation"):
            st.markdown(
                """
- **Left chart** — input space: circles are "view A", diamonds are "view B" of
  the same batch of points; dotted lines connect each positive pair
- **Right chart** — embedding space: every dataset point's current position on
  the unit circle, coloured by its (hidden-from-the-algorithm) true cluster
- **Loss curve** — solid line is NT-Xent loss; dashed lines are mean
  positive/negative cosine similarity, read off the right-hand axis
- **◀ Prev / Next ▶** to step manually, **▶ Play** to auto-advance through
  the whole training run
"""
            )

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
