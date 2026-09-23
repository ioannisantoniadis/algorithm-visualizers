"""Widescreen banner for the README hero image / GitHub social preview.

Recreates Home.py's own category/algorithm catalogue as a static grid,
using the exact category accent colors from common/ui.py's CATEGORY_ACCENTS
(itself drawn from .streamlit/config.toml's chartCategoricalColors) so the
banner matches the live app's design system instead of inventing a new one.

CATALOGUE and CATEGORY_ACCENTS are hand-copied from Home.py / common/ui.py
rather than imported, since both modules import `streamlit` at the top
level and assume an active Streamlit script-run context. Keep this in sync
by hand if either changes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PAPER = "#ffffff"
INK = "#18181b"
INK_SOFT = "#71717a"
BORDER = "#e4e4e7"
PANEL_BG = "#f8f8fb"

# category -> (accent color, [short display names])
CATALOGUE: dict[str, tuple[str, list[str]]] = {
    "Clustering": ("#6366f1", ["K-Means", "DBSCAN", "Gaussian Mixture (EM)"]),
    "Dimensionality reduction": ("#14b8a6", ["PCA", "UMAP", "t-SNE"]),
    "Classification & ensembles": ("#f59e0b", ["Perceptron & Grad. Descent", "SVM", "Random Forest"]),
    "Deep learning building blocks": ("#f43f5e", ["Backpropagation", "Self-Attention"]),
    "Generative & self-supervised": ("#8b5cf6", ["Variational Autoencoder", "Diffusion (DDPM)", "Contrastive Learning"]),
    "Graph algorithms": ("#0ea5e9", ["Dijkstra & A*", "MST (Kruskal / Prim)"]),
    "Probabilistic & signal processing": ("#06b6d4", ["MCMC", "Kalman Filter", "Particle Filter", "FFT"]),
    "Reinforcement learning": ("#fb923c", ["Q-Learning / SARSA"]),
}

FONT = "Helvetica Neue, Arial, sans-serif"  # Inter fallback in headless matplotlib


def rounded_rect(ax, x, y, w, h, color, *, fill=True, edgecolor=None, lw=1.2,
                  rounding=0.06, alpha=1.0, zorder=1):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw, edgecolor=edgecolor or "none",
        facecolor=color if fill else "none",
        alpha=alpha, zorder=zorder, mutation_aspect=1,
    )
    ax.add_patch(box)


def main() -> None:
    plt.rcParams["font.family"] = FONT

    # 12.8 x 6.4in @ 200dpi = 2560x1280px, 2x GitHub's recommended 1280x640
    # social preview size (retina-sharp, GitHub downsamples).
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 6.4)
    ax.invert_yaxis()
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Header.
    n_algos = sum(len(algos) for _, algos in CATALOGUE.values())
    ax.text(0.4, 0.62, "Algorithm Visualisers", fontsize=25, fontweight="bold",
             color=INK, ha="left", va="top")
    ax.text(0.4, 1.28, f"{n_algos} classic ML & CS algorithms, implemented from scratch "
                        "in NumPy and visualized step by step",
            fontsize=12.5, color=INK_SOFT, ha="left", va="top")

    # 4x2 grid of category panels.
    n_cols, n_rows = 4, 2
    margin_x, margin_y_top, margin_y_bottom, gap = 0.4, 1.75, 0.35, 0.28
    grid_w = 12.8 - 2 * margin_x
    grid_h = 6.4 - margin_y_top - margin_y_bottom
    panel_w = (grid_w - (n_cols - 1) * gap) / n_cols
    panel_h = (grid_h - (n_rows - 1) * gap) / n_rows

    categories = list(CATALOGUE.items())
    for idx, (name, (color, algos)) in enumerate(categories):
        col, row = idx % n_cols, idx // n_cols
        px = margin_x + col * (panel_w + gap)
        py = margin_y_top + row * (panel_h + gap)

        rounded_rect(ax, px, py, panel_w, panel_h, PANEL_BG,
                     edgecolor=BORDER, lw=1.2, rounding=0.09, zorder=1)
        rounded_rect(ax, px, py, 0.06, panel_h, color, rounding=0.03, zorder=2)

        pad = 0.16
        ax.text(px + pad + 0.06, py + pad, name, fontsize=10.3, fontweight="700",
                 color=color, ha="left", va="top", wrap=True,
                 zorder=3)

        chip_y = py + pad + 0.46
        chip_h = 0.34
        chip_gap = 0.09
        for algo in algos:
            chip_w = panel_w - 2 * (pad + 0.06)
            rounded_rect(ax, px + pad + 0.06, chip_y, chip_w, chip_h, color,
                         rounding=chip_h / 2, alpha=0.92, zorder=3)
            ax.text(px + pad + 0.06 + chip_w / 2, chip_y + chip_h / 2, algo,
                     fontsize=8.4, fontweight="600", color="white",
                     ha="center", va="center", zorder=4)
            chip_y += chip_h + chip_gap

    out = Path(__file__).resolve().parents[1] / "docs" / "social-preview.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out), dpi=200, facecolor=PAPER, bbox_inches=None)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
