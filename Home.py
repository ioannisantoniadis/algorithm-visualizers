"""Algorithm Visualisers — multipage portfolio entry point.

Run with:
    streamlit run Home.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Algorithm Visualisers",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppDeployButton"] { display: none; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; }
[data-testid="stAlert"] { border-radius: 0.75rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page catalogue — (module path, title, icon, one-line descriptor)
# ---------------------------------------------------------------------------
CATALOGUE = {
    "Clustering": [
        ("apps/kmeans.py", "K-Means", "🔵", "Manual centroid placement, E/M step-through"),
        ("apps/dbscan.py", "DBSCAN", "🔍", "Density clustering — core / border / noise points"),
        ("apps/gmm.py", "Gaussian Mixture (EM)", "📊", "Soft assignments, confidence ellipses"),
    ],
    "Dimensionality reduction": [
        ("apps/pca.py", "PCA", "📐", "Covariance, eigenvectors, rank-k reconstruction"),
        ("apps/umap.py", "UMAP", "🧭", "Fuzzy simplicial graph + embedding SGD"),
        ("apps/tsne.py", "t-SNE", "🌀", "KL-divergence descent on affinities"),
    ],
    "Classification & ensembles": [
        ("apps/perceptron.py", "Perceptron & Gradient Descent", "🎯", "SGD / momentum / Adam optimisers"),
        ("apps/svm.py", "Support Vector Machine", "🧮", "Soft-margin kernel SVM via SMO"),
        ("apps/random_forest.py", "Random Forest", "🌲", "Bagging, feature subsampling, OOB error"),
    ],
    "Deep learning building blocks": [
        ("apps/backprop.py", "Backpropagation", "🔗", "MLP gradient flow, numerically verified"),
        ("apps/attention.py", "Transformer Self-Attention", "👁️", "Scaled dot-product, multi-head"),
    ],
    "Generative & self-supervised models": [
        ("apps/vae.py", "Variational Autoencoder", "🎭", "Reparameterisation trick, KL vs reconstruction"),
        ("apps/diffusion.py", "Diffusion Model (DDPM)", "🌊", "Forward/reverse noising process"),
        ("apps/contrastive.py", "Contrastive Learning", "🧲", "SimCLR-style, NT-Xent loss"),
    ],
    "Graph algorithms": [
        ("apps/dijkstra.py", "Dijkstra & A*", "🗺️", "Shortest-path search on a grid"),
        ("apps/mst.py", "Minimum Spanning Tree", "🌳", "Kruskal & Prim"),
    ],
    "Probabilistic methods, state estimation & signal processing": [
        ("apps/mcmc.py", "Markov Chain Monte Carlo", "🎲", "Metropolis-Hastings sampling"),
        ("apps/kalman.py", "Kalman Filter", "📡", "Predict/update tracking"),
        ("apps/fft.py", "Fast Fourier Transform", "🎵", "Cooley-Tukey, butterfly diagram"),
    ],
    "Reinforcement learning": [
        ("apps/qlearning.py", "Q-Learning / SARSA", "🕹️", "Stochastic grid-world"),
    ],
}


def _home_page() -> None:
    st.markdown(
        "<p style='color:#6366f1; font-weight:600; font-size:0.8rem; "
        "letter-spacing:0.08em; margin-bottom:-0.3rem;'>ALGORITHM VISUALISER PORTFOLIO</p>",
        unsafe_allow_html=True,
    )
    st.title("20 classic algorithms, from scratch")
    st.caption(
        "Every visualiser below is a from-scratch NumPy implementation — no scikit-learn "
        "or PyTorch in the core algorithm — paired with an interactive Streamlit + Plotly "
        "step-by-step walkthrough. Pick a category, pick an algorithm, and step through it."
    )
    st.divider()

    for category, items in CATALOGUE.items():
        st.subheader(category)
        cols = st.columns(3)
        for i, (path, title, icon, blurb) in enumerate(items):
            page = PAGES_BY_PATH[path]
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"### {icon} {title}")
                    st.caption(blurb)
                    st.page_link(page, label="Open visualiser →", use_container_width=True)
        st.write("")


# ---------------------------------------------------------------------------
# Build navigation
# ---------------------------------------------------------------------------
home_page = st.Page(_home_page, title="Home", icon="🏠", default=True, url_path="home")

PAGES_BY_PATH: dict[str, st.Page] = {}
nav: dict[str, list[st.Page]] = {"Overview": [home_page]}
for category, items in CATALOGUE.items():
    section_pages = []
    for path, title, icon, _blurb in items:
        p = st.Page(path, title=title, icon=icon)
        PAGES_BY_PATH[path] = p
        section_pages.append(p)
    nav[category] = section_pages

pg = st.navigation(nav)
pg.run()
