"""Algorithm Visualisers — multipage portfolio entry point.

Run with:
    streamlit run Home.py
"""

from __future__ import annotations

import streamlit as st

from common.ui import category_accent, global_css

st.set_page_config(
    page_title="Algorithm Visualisers",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(global_css(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page catalogue — (module path, title, one-line descriptor)
# ---------------------------------------------------------------------------
CATALOGUE = {
    "Clustering": [
        ("apps/kmeans.py", "K-Means", "Manual centroid placement, E/M step-through"),
        ("apps/dbscan.py", "DBSCAN", "Density clustering — core / border / noise points"),
        ("apps/gmm.py", "Gaussian Mixture (EM)", "Soft assignments, confidence ellipses"),
    ],
    "Dimensionality reduction": [
        ("apps/pca.py", "PCA", "Covariance, eigenvectors, rank-k reconstruction"),
        ("apps/umap.py", "UMAP", "Fuzzy simplicial graph + embedding SGD"),
        ("apps/tsne.py", "t-SNE", "KL-divergence descent on affinities"),
    ],
    "Classification & ensembles": [
        ("apps/perceptron.py", "Perceptron & Gradient Descent", "SGD / momentum / Adam optimisers"),
        ("apps/svm.py", "Support Vector Machine", "Soft-margin kernel SVM via SMO"),
        ("apps/random_forest.py", "Random Forest", "Bagging, feature subsampling, OOB error"),
    ],
    "Deep learning building blocks": [
        ("apps/backprop.py", "Backpropagation", "MLP gradient flow, numerically verified"),
        ("apps/attention.py", "Transformer Self-Attention", "Scaled dot-product, multi-head"),
    ],
    "Generative & self-supervised models": [
        ("apps/vae.py", "Variational Autoencoder", "Reparameterisation trick, KL vs reconstruction"),
        ("apps/diffusion.py", "Diffusion Model (DDPM)", "Forward/reverse noising process"),
        ("apps/contrastive.py", "Contrastive Learning", "SimCLR-style, NT-Xent loss"),
    ],
    "Graph algorithms": [
        ("apps/dijkstra.py", "Dijkstra & A*", "Shortest-path search on a grid"),
        ("apps/mst.py", "Minimum Spanning Tree", "Kruskal & Prim"),
    ],
    "Probabilistic methods, state estimation & signal processing": [
        ("apps/mcmc.py", "Markov Chain Monte Carlo", "Metropolis-Hastings sampling"),
        ("apps/kalman.py", "Kalman Filter", "Predict/update tracking"),
        ("apps/fft.py", "Fast Fourier Transform", "Cooley-Tukey, butterfly diagram"),
    ],
    "Reinforcement learning": [
        ("apps/qlearning.py", "Q-Learning / SARSA", "Stochastic grid-world"),
    ],
}


def _home_page() -> None:
    # Scoped to this page only: the whole card becomes the click target (the
    # page_link is stretched to cover it and visually hidden), with a hover
    # lift instead of a separate "Open" button.
    #
    # The selector requires the page_link to be the *immediate* child of an
    # *immediate* stElementContainer child — i.e. exactly the shape of a
    # card's own `st.container(border=True)` block in the loop below.
    # A looser `:has(> div [data-testid="stPageLink"])` (any descendant,
    # any depth) also matches every ancestor further up the tree — the
    # column wrapping each card, and the page's own outermost content
    # block — because a div-shaped immediate child containing a page_link
    # *somewhere* inside it is true all the way up. That bug made the
    # entire page (hero text down through every category) register as one
    # giant hover/click target sharing this card styling, while the real
    # click hit-boxes stayed correctly card-sized — so hovering anywhere
    # showed a pointer cursor and, on hover, drew this rule's border/shadow
    # around the whole page, and a click that happened to land on a real
    # card navigated with no visible cause. Keep this selector exact rather
    # than reverting to the looser form.
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] > div[data-testid="stPageLink"]) {
        position: relative;
        cursor: pointer;
        transition: border-color .15s ease, box-shadow .15s ease, transform .15s ease;
    }
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] > div[data-testid="stPageLink"]):hover {
        border-color: #6366f1;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.16);
        transform: translateY(-2px);
    }
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] > div[data-testid="stPageLink"])
        div[data-testid="stElementContainer"]:has(> div[data-testid="stPageLink"]) {
        position: absolute !important;
        inset: 0 !important;
        width: 100% !important;
        height: 100% !important;
    }
    [data-testid="stPageLink"], [data-testid="stPageLink"] > div {
        width: 100% !important;
        height: 100% !important;
    }
    [data-testid="stPageLink-NavLink"] {
        display: block !important;
        width: 100% !important;
        height: 100% !important;
        opacity: 0;
    }

    /* ---- Hero ---- */
    .hero-eyebrow {
        color: #6366f1; font-weight: 600; font-size: 0.8rem;
        letter-spacing: 0.08em; margin-bottom: 0.4rem;
    }
    .hero-meta {
        font-size: 0.85rem; color: #71717a; margin: 0.9rem 0 1.5rem 0;
    }

    /* ---- Category section headers ---- */
    .category-header {
        display: flex; align-items: center; gap: 0.6rem;
        margin: 2.25rem 0 1rem 0;
    }
    .category-accent-bar {
        width: 4px; height: 1.35rem; border-radius: 999px;
    }
    .category-title {
        font-size: 1.35rem; font-weight: 700; color: #18181b;
    }

    /* ---- Persistent "Open" affordance on every card (not hover-only) ---- */
    .card-open {
        margin-top: 0.85rem; font-size: 0.8rem; font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<p class='hero-eyebrow'>ALGORITHM VISUALISER PORTFOLIO</p>", unsafe_allow_html=True)
    st.title("20 classic algorithms, visualized step by step")
    st.caption(
        "Step-by-step, interactive visualizations of 20 classic algorithms — spanning "
        "clustering, deep learning, graph search, and more. Each one is implemented "
        "from scratch in NumPy, so what you're watching is the actual math, not a "
        "black box."
    )

    n_algos = sum(len(items) for items in CATALOGUE.values())
    st.markdown(
        f"<p class='hero-meta'>{n_algos} algorithms across {len(CATALOGUE)} categories</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    for category, items in CATALOGUE.items():
        accent = category_accent(category)
        st.markdown(
            f"""<div class="category-header">
                <span class="category-accent-bar" style="background:{accent};"></span>
                <span class="category-title">{category}</span>
            </div>""",
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for i, (path, title, blurb) in enumerate(items):
            page = PAGES_BY_PATH[path]
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"#### {title}")
                    st.caption(blurb)
                    st.page_link(page, label="Open visualiser")
                    st.markdown(
                        f'<div class="card-open" style="color:{accent};">Open visualiser →</div>',
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Build navigation
# ---------------------------------------------------------------------------
home_page = st.Page(_home_page, title="Home", default=True, url_path="home")

PAGES_BY_PATH: dict[str, st.Page] = {}
nav: dict[str, list[st.Page]] = {"Overview": [home_page]}
for category, items in CATALOGUE.items():
    section_pages = []
    for path, title, _blurb in items:
        p = st.Page(path, title=title)
        PAGES_BY_PATH[path] = p
        section_pages.append(p)
    nav[category] = section_pages

pg = st.navigation(nav)
pg.run()
