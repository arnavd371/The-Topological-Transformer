"""
visualize.py — Matplotlib visualizations for the Topological Transformer
=========================================================================
Run this script directly to generate and save all diagnostic graphs:

    python visualize.py

Six graphs are produced and written to the ``graphs/`` directory:

1. ``graphs/01_persistence_diagram.png``   — H0 & H1 persistence diagrams
2. ``graphs/02_persistence_image.png``     — H0 & H1 persistence images (heatmaps)
3. ``graphs/03_point_cloud_saddles.png``   — 2-D PCA token cloud with saddle tokens
4. ``graphs/04_attention_weights.png``     — T-Form attention weight heatmap
5. ``graphs/05_topo_feature_norms.png``    — Per-layer topological feature norms
6. ``graphs/06_loss_curve.png``            — Synthetic training / validation loss curve
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in headless environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

# Make sure tform is importable when running from repo root
sys.path.insert(0, os.path.dirname(__file__))

from tform import (
    TopologicalTransformer,
    TopologicalAttention,
    _compute_persistence_diagrams,
    _detect_saddle_indices,
    _pca_project,
    _persistence_image,
    _IMG_RESOLUTION,
    _MAX_FILTRATION,
)

# ---------------------------------------------------------------------------
# Shared style settings
# ---------------------------------------------------------------------------
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "lines.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Colour palette
COLOR_H0_BLUE    = "#4C72B0"   # H0 components
COLOR_H1_ORANGE  = "#DD8452"   # H1 loops
COLOR_SADDLE_RED = "#C44E52"   # saddle tokens
COLOR_ACCENT_GREEN = "#55A868" # accent (best-epoch marker, etc.)


# ===========================================================================
# Helper: generate a reproducible sample point cloud + diagrams
# ===========================================================================

def _make_sample_cloud(n: int = 20, d: int = 8) -> np.ndarray:
    """Return a synthetic point cloud of shape (n, d)."""
    rng = np.random.default_rng(SEED)
    # Three loose clusters in high-D space
    c1 = rng.normal([0.0] * d, 0.4, (n // 3, d)).astype(np.float32)
    c2 = rng.normal([2.0] + [0.0] * (d - 1), 0.4, (n // 3, d)).astype(np.float32)
    c3 = rng.normal([1.0, 1.5] + [0.0] * (d - 2), 0.4, (n - 2 * (n // 3), d)).astype(
        np.float32
    )
    pts = np.vstack([c1, c2, c3])
    # Normalise
    lo, hi = pts.min(), pts.max()
    if hi - lo > 1e-8:
        pts = (pts - lo) / (hi - lo) * (_MAX_FILTRATION * 0.5)
    return pts


# ===========================================================================
# Graph 1 — Persistence Diagram
# ===========================================================================

def plot_persistence_diagram(save_path: str) -> None:
    """Scatter-plot birth/death pairs for H0 and H1 with the diagonal."""
    pts = _make_sample_cloud(n=24)
    h0, h1 = _compute_persistence_diagrams(pts, max_edge_length=_MAX_FILTRATION)

    fig, ax = plt.subplots(figsize=(6, 5))

    max_val = _MAX_FILTRATION
    ax.plot([0, max_val], [0, max_val], "k--", lw=1.0, alpha=0.4, label="Diagonal")

    if len(h0):
        ax.scatter(h0[:, 0], h0[:, 1], c=COLOR_H0_BLUE, s=60, zorder=3, label="H0 (components)")
    if len(h1):
        ax.scatter(h1[:, 0], h1[:, 1], c=COLOR_H1_ORANGE, s=60, marker="^", zorder=3,
                   label="H1 (loops)")

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title("Persistence Diagram — Vietoris-Rips Filtration")
    ax.set_xlim(-0.2, max_val + 0.2)
    ax.set_ylim(-0.2, max_val + 0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# Graph 2 — Persistence Image heatmaps (H0 and H1)
# ===========================================================================

def plot_persistence_images(save_path: str) -> None:
    """Side-by-side heatmaps of the H0 and H1 persistence images."""
    pts = _make_sample_cloud(n=24)
    h0, h1 = _compute_persistence_diagrams(pts, max_edge_length=_MAX_FILTRATION)

    res = _IMG_RESOLUTION
    img_h0 = _persistence_image(h0, resolution=res, max_val=_MAX_FILTRATION).reshape(
        res, res
    )
    img_h1 = _persistence_image(h1, resolution=res, max_val=_MAX_FILTRATION).reshape(
        res, res
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    extent = [0, _MAX_FILTRATION, 0, _MAX_FILTRATION]

    for ax, img, title, cmap in zip(
        axes,
        [img_h0, img_h1],
        ["Persistence Image — H0 (Components)", "Persistence Image — H1 (Loops)"],
        ["Blues", "Oranges"],
    ):
        im = ax.imshow(
            img, origin="lower", extent=extent, aspect="auto", cmap=cmap
        )
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Persistence")
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# Graph 3 — Token point cloud with saddle tokens highlighted
# ===========================================================================

def plot_point_cloud_saddles(save_path: str) -> None:
    """2-D PCA projection of the token cloud, saddle tokens in red."""
    pts = _make_sample_cloud(n=24)
    h0, _ = _compute_persistence_diagrams(pts, max_edge_length=_MAX_FILTRATION)
    saddle_idx = _detect_saddle_indices(pts, h0, _MAX_FILTRATION)

    pts2d = _pca_project(pts, n_components=2)

    saddle_set = set(saddle_idx)

    fig, ax = plt.subplots(figsize=(6, 5))
    # Draw regular tokens first
    reg = [i for i in range(len(pts2d)) if i not in saddle_set]
    if reg:
        ax.scatter(
            pts2d[reg, 0], pts2d[reg, 1],
            c=COLOR_H0_BLUE, s=50, zorder=2, label="Token"
        )
    if saddle_idx:
        ax.scatter(
            pts2d[saddle_idx, 0], pts2d[saddle_idx, 1],
            c=COLOR_SADDLE_RED, s=140, marker="*", zorder=3, label="Saddle token"
        )
    # Annotate indices
    for i, (x, y) in enumerate(pts2d):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color="dimgray")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Token Point Cloud (PCA 2-D) — Saddle Tokens Highlighted")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# Graph 4 — Attention weight heatmap
# ===========================================================================

def plot_attention_weights(save_path: str) -> None:
    """Heat-map of the averaged multi-head attention weights from T-Form."""
    B, L, D = 1, 12, 32
    model = TopologicalAttention(d_model=D, num_heads=4, dropout=0.0)
    model.eval()

    x = torch.randn(B, L, D)

    # Monkey-patch to capture attention weights
    captured: dict = {}

    def _patched_forward(x, mask=None):
        B_, L_, D_ = x.shape
        Q = model.W_q(x).view(B_, L_, model.num_heads, model.d_k).transpose(1, 2)
        K = model.W_k(x).view(B_, L_, model.num_heads, model.d_k).transpose(1, 2)
        V = model.W_v(x).view(B_, L_, model.num_heads, model.d_k).transpose(1, 2)

        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(model.d_k)
        topo_vec, saddle_mask = model._compute_topo_features(x)
        topo_bias = model.topo_bias_net(topo_vec).unsqueeze(1)
        logits = logits + topo_bias
        logits = model.saddle_embed(logits, saddle_mask)
        weights = F.softmax(logits, dim=-1)
        captured["weights"] = weights.detach().cpu().numpy()
        out = torch.matmul(weights, V).transpose(1, 2).contiguous().view(B_, L_, D_)
        return model.W_o(out)

    model.forward = _patched_forward

    with torch.no_grad():
        model(x)

    weights = captured["weights"][0]   # (H, L, L)
    avg_weights = weights.mean(axis=0) # (L, L)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: average across all heads
    im0 = axes[0].imshow(avg_weights, cmap="viridis", aspect="auto")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    axes[0].set_title("Attention Weights (Avg. over Heads)")
    axes[0].set_xlabel("Key position")
    axes[0].set_ylabel("Query position")

    # Right: per-head grid (up to 4 heads)
    n_heads = min(4, weights.shape[0])
    gs_inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=axes[1].get_subplotspec(), hspace=0.4, wspace=0.4
    )
    axes[1].remove()
    for h in range(n_heads):
        ax_h = fig.add_subplot(gs_inner[h // 2, h % 2])
        ax_h.imshow(weights[h], cmap="viridis", aspect="auto")
        ax_h.set_title(f"Head {h}", fontsize=9)
        ax_h.axis("off")

    fig.suptitle("T-Form Attention Weight Heatmap", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# Graph 5 — Topological feature norms across layers
# ===========================================================================

def plot_topo_feature_norms(save_path: str) -> None:
    """Bar chart: L2 norm of the H0 + H1 persistence image per layer."""
    num_layers = 4
    D, L = 32, 16

    model = TopologicalTransformer(
        d_model=D, num_heads=4, num_layers=num_layers, dropout=0.0
    )
    model.eval()

    x = torch.randn(1, L, D)
    norms_h0, norms_h1 = [], []

    current = x.clone()
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            normed = layer.norm1(current)
            pts_np = normed[0].cpu().float().numpy()
            if D > 32:
                pts_np = _pca_project(pts_np, n_components=32)
            lo, hi = pts_np.min(), pts_np.max()
            if hi - lo > 1e-8:
                pts_np = (pts_np - lo) / (hi - lo) * (_MAX_FILTRATION * 0.5)
            h0, h1 = _compute_persistence_diagrams(pts_np, max_edge_length=_MAX_FILTRATION)
            res = _IMG_RESOLUTION
            img_h0 = _persistence_image(h0, resolution=res, max_val=_MAX_FILTRATION)
            img_h1 = _persistence_image(h1, resolution=res, max_val=_MAX_FILTRATION)
            norms_h0.append(float(np.linalg.norm(img_h0)))
            norms_h1.append(float(np.linalg.norm(img_h1)))
            current = layer(current)

    layers = np.arange(1, num_layers + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(layers - width / 2, norms_h0, width, color=COLOR_H0_BLUE, label="H0 (components)")
    ax.bar(layers + width / 2, norms_h1, width, color=COLOR_H1_ORANGE, label="H1 (loops)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Persistence Image L2-Norm")
    ax.set_title("Topological Feature Norms Across T-Form Layers")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"Layer {i}" for i in layers])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# Graph 6 — Synthetic training / validation loss curve
# ===========================================================================

def plot_loss_curve(save_path: str) -> None:
    """Synthetic training + validation loss curve illustrating T-Form training."""
    rng = np.random.default_rng(SEED)
    epochs = 40
    t = np.arange(1, epochs + 1)

    # Smooth exponential decay + noise
    train_loss = 2.5 * np.exp(-0.12 * t) + 0.12 + rng.normal(0, 0.03, epochs)
    val_loss   = 2.5 * np.exp(-0.10 * t) + 0.18 + rng.normal(0, 0.04, epochs)
    train_loss = np.clip(train_loss, 0.08, 3.0)
    val_loss   = np.clip(val_loss, 0.10, 3.0)

    # Smooth with a simple moving average
    def _smooth(arr: np.ndarray, w: int = 5) -> np.ndarray:
        kernel = np.ones(w) / w
        padded = np.pad(arr, (w // 2, w // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")[: len(arr)]

    train_smooth = _smooth(train_loss)
    val_smooth   = _smooth(val_loss)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, train_loss, color=COLOR_H0_BLUE, alpha=0.35, lw=1.0)
    ax.plot(t, val_loss,   color=COLOR_H1_ORANGE, alpha=0.35, lw=1.0)
    ax.plot(t, train_smooth, color=COLOR_H0_BLUE, label="Training loss")
    ax.plot(t, val_smooth,   color=COLOR_H1_ORANGE, label="Validation loss", linestyle="--")

    best_epoch = int(np.argmin(val_smooth)) + 1
    ax.axvline(best_epoch, color=COLOR_ACCENT_GREEN, linestyle=":", lw=1.2, label=f"Best epoch ({best_epoch})")
    ax.scatter([best_epoch], [val_smooth[best_epoch - 1]], color=COLOR_ACCENT_GREEN, zorder=5, s=60)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Topological Transformer — Training / Validation Loss Curve")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# Main entry-point
# ===========================================================================

def generate_all() -> None:
    """Generate and save all six graphs."""
    print("Generating T-Form visualizations …")

    plot_persistence_diagram(
        os.path.join(GRAPH_DIR, "01_persistence_diagram.png")
    )
    plot_persistence_images(
        os.path.join(GRAPH_DIR, "02_persistence_image.png")
    )
    plot_point_cloud_saddles(
        os.path.join(GRAPH_DIR, "03_point_cloud_saddles.png")
    )
    plot_attention_weights(
        os.path.join(GRAPH_DIR, "04_attention_weights.png")
    )
    plot_topo_feature_norms(
        os.path.join(GRAPH_DIR, "05_topo_feature_norms.png")
    )
    plot_loss_curve(
        os.path.join(GRAPH_DIR, "06_loss_curve.png")
    )

    print("\nAll graphs saved to:", GRAPH_DIR)


if __name__ == "__main__":
    generate_all()
