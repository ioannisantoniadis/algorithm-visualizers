# Algorithm Visualisers

20 classic machine-learning and CS algorithms, each implemented **from scratch in NumPy** (no scikit-learn, no PyTorch in the core algorithm) and paired with an interactive **Streamlit + Plotly** step-by-step walkthrough — all in one categorized multipage app.

**Live demo: [algorithm-visualizers.streamlit.app](https://algorithm-visualizers.streamlit.app/)**

This repo replaces 20 previously-standalone visualiser repos. Each algorithm keeps its own self-contained package (`<algo>/{algorithm,data,visualize}.py`); only the Streamlit entry point was merged into a single app with one shared home page and navigation.

## Quick start

```bash
uv sync
uv run streamlit run Home.py
```

Open the URL printed in your terminal (usually http://localhost:8501). The sidebar navigation groups every algorithm by category; the home page gives a card-based overview of all 20.

## What's inside

| Category | Algorithms |
|---|---|
| **Clustering** | K-Means · DBSCAN · Gaussian Mixture (EM) |
| **Dimensionality reduction** | PCA · UMAP · t-SNE |
| **Classification & ensembles** | Perceptron & Gradient Descent · Support Vector Machine · Random Forest |
| **Deep learning building blocks** | Backpropagation · Transformer Self-Attention |
| **Generative & self-supervised models** | Variational Autoencoder · Diffusion Model (DDPM) · Contrastive Learning |
| **Graph algorithms** | Dijkstra & A* · Minimum Spanning Tree (Kruskal & Prim) |
| **Probabilistic methods, state estimation & signal processing** | Markov Chain Monte Carlo · Kalman Filter · Fast Fourier Transform |
| **Reinforcement learning** | Q-Learning / SARSA |

Every visualiser follows the same convention: a from-scratch NumPy implementation, a step-by-step or frame-by-frame playback control, and an in-app explanation of what's happening and why the algorithm can fail.

## Project layout

```
algorithm-visualizers/
├── Home.py                 # Landing page + st.navigation wiring
├── apps/<algo>.py          # One Streamlit page per algorithm
├── <algo>/                 # Each algorithm's own from-scratch package
│   ├── algorithm.py         # Core algorithm, records a Snapshot per step
│   ├── data.py               # Synthetic dataset / environment generators
│   └── visualize.py          # Plotly figure builders
├── .streamlit/config.toml  # Shared theme (indigo/teal, Inter font)
└── pyproject.toml
```

## Design system

Every page shares one visual identity: an indigo/teal palette, Inter typeface, bordered card layout for parameter groups and charts, and a matching Plotly theme. `.streamlit/config.toml` and each page's chart palette are the source of truth — kept identical across every algorithm.
