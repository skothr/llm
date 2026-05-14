"""Render the vocab atlas captured by nla_vocab_atlas_capture.py.

Produces:
  fig19_vocab_atlas.png            - PCA of the 128 vocab anchors, by category
  fig20_combined_atlas.png         - 207-vector PCA: vocab + existing 167 captures
  fig21_interp_through_anchors.png - interpolation trajectory's top-3 nearest
                                     anchors at each step
  fig22_anchor_cosine_matrix.png   - category-ordered pairwise cosine matrix

All operate AFTER sink removal so the chat-template attractor doesn't drown
content similarity. Uses sink dims from pairwise_and_hotdims.pt.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import cast
cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARTIFACTS = Path("testing/.cache/nla_artifacts")
FIGDIR = Path("research/observations/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


CATEGORY_COLORS: dict[str, str] = {
    # Content (warm colors)
    "country":       "#d62728",
    "capital":       "#e377c2",
    "nature":        "#2ca02c",
    "codemath":      "#8c564b",
    "emotion":       "#ff7f0e",
    "refusal":       "#7f0000",
    # Function (cool colors)
    "article":       "#9edae5",
    "pronoun":       "#17becf",
    "demonstrative": "#aec7e8",
    "preposition":   "#1f77b4",
    "conjunction":   "#9467bd",
    "auxiliary":     "#c5b0d5",
    "negation":      "#000000",
    "quantifier":    "#bcbd22",
    "wh_word":       "#dbdb8d",
    # Punctuation (gray)
    "p_ender":       "#7f7f7f",
    "p_internal":    "#c7c7c7",
    "p_quote":       "#5a5a5a",
    "p_bracket":     "#3f3f3f",
    "p_dash":        "#2a2a2a",
    "p_special":     "#a0a0a0",
    # Numbers (green-ish)
    "number":        "#98df8a",
    "math_op":       "#5fbf5f",
}


def apply_sink_removal(H: torch.Tensor, sink_dims: list[int]) -> torch.Tensor:
    H_res = H.clone()
    for d in sink_dims:
        H_res[:, d] = 0.0
    return H_res


def pca_coords(H: torch.Tensor, n_pcs: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    mean_h = H.mean(dim=0, keepdim=True)
    Hc = H - mean_h
    _, S, Vh = torch.linalg.svd(Hc, full_matrices=False)
    coords = Hc @ Vh.T[:, :n_pcs]
    var = S ** 2
    return coords, var


def main() -> None:
    vocab = torch.load(ARTIFACTS / "vocab_atlas.pt", weights_only=False)
    pw = torch.load(ARTIFACTS / "pairwise_and_hotdims.pt", weights_only=False)
    labels: dict[int, str] = pw["labels"]
    sink_dims = sorted([idx for idx, lbl in labels.items() if lbl == "sink"])

    vocab_caps = vocab["captures"]
    n_vocab = len(vocab_caps)
    print(f"loaded {n_vocab} vocab anchors")
    H_vocab = torch.stack([c["h"] for c in vocab_caps])
    H_vocab_res = apply_sink_removal(H_vocab, sink_dims)
    print(f"sink-removed H_vocab shape: {tuple(H_vocab_res.shape)}")

    # --- fig19: vocab atlas PCA (sink-removed) ---
    coords, var = pca_coords(H_vocab_res)
    total_var = float(var.sum().item())
    print(f"vocab-atlas PCA variance: PC1={float(var[0]/total_var):.3f}  PC2={float(var[1]/total_var):.3f}  "
          f"PC3={float(var[2]/total_var):.3f}")

    fig, ax = plt.subplots(figsize=(15, 12))
    seen_cats: set[str] = set()
    for i, cap in enumerate(vocab_caps):
        cat = cap["category"]
        color = CATEGORY_COLORS.get(cat, "#999999")
        label = cat if cat not in seen_cats else None
        seen_cats.add(cat)
        ax.scatter(coords[i, 0].item(), coords[i, 1].item(), c=color,
                   s=80, alpha=0.75, edgecolors="black", linewidths=0.5, label=label)
        ax.annotate(cap["word"], (coords[i, 0].item(), coords[i, 1].item()),
                    textcoords="offset points", xytext=(5, 4), fontsize=7, alpha=0.9)
    ax.set_xlabel(f"PC1 ({var[0]/total_var*100:.1f}%, sink-removed)")
    ax.set_ylabel(f"PC2 ({var[1]/total_var*100:.1f}%, sink-removed)")
    ax.set_title(f"fig19 — Vocab atlas: 128 anchor tokens at end-of-user-message position\n"
                 f"sink-removed PCA; clusters reveal category structure")
    ax.legend(loc="best", framealpha=0.9, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig19_vocab_atlas.png", dpi=140)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig19_vocab_atlas.png")

    # --- fig20: combined atlas (vocab + existing 167) ---
    H_existing = pw["H"]  # (167, 3584)
    H_existing_res = apply_sink_removal(H_existing, sink_dims)
    items_meta = pw["items_meta"]
    H_all = torch.cat([H_vocab_res, H_existing_res], dim=0)
    coords_all, var_all = pca_coords(H_all)
    total_var_all = float(var_all.sum().item())

    fig, ax = plt.subplots(figsize=(16, 12))
    # existing captures as small gray dots first
    SRC_COLORS = {
        "aggregate": "#aec7e8", "haiku_gen": "#ffbb78", "forced": "#ff9896",
        "country_src": "#98df8a", "non_country_src": "#c5b0d5", "country_test": "#c49c94",
    }
    for i, it in enumerate(items_meta):
        c = SRC_COLORS.get(it["src"], "#cccccc")
        ax.scatter(coords_all[n_vocab + i, 0].item(), coords_all[n_vocab + i, 1].item(),
                   c=c, s=15, alpha=0.4, edgecolors="none")
    # vocab anchors on top with category colors + labels
    for i, cap in enumerate(vocab_caps):
        cat = cap["category"]
        color = CATEGORY_COLORS.get(cat, "#999999")
        ax.scatter(coords_all[i, 0].item(), coords_all[i, 1].item(), c=color,
                   s=70, alpha=0.85, edgecolors="black", linewidths=0.5)
        ax.annotate(cap["word"], (coords_all[i, 0].item(), coords_all[i, 1].item()),
                    textcoords="offset points", xytext=(5, 4), fontsize=7, alpha=0.95)
    ax.set_xlabel(f"PC1 ({var_all[0]/total_var_all*100:.1f}%, sink-removed)")
    ax.set_ylabel(f"PC2 ({var_all[1]/total_var_all*100:.1f}%, sink-removed)")
    ax.set_title(f"fig20 — Combined atlas: 128 vocab anchors (labeled) + 167 existing captures (faded dots)\n"
                 f"shared PC space, sink-removed")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig20_combined_atlas.png", dpi=140)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig20_combined_atlas.png")

    # --- fig21: interpolation trajectory through anchor space ---
    flip_path = ARTIFACTS / "interpolation_flipbook.pt"
    if flip_path.exists():
        flip = torch.load(flip_path, weights_only=False)
        steps = sorted(flip["steps"], key=lambda s: s["step"])
        n_steps = len(steps)

        # Compute cosine of each interp step to each vocab anchor (sink-removed)
        H_interp = torch.stack([s["h_t"] for s in steps])
        H_interp_res = apply_sink_removal(H_interp, sink_dims)
        H_vocab_unit = torch.nn.functional.normalize(H_vocab_res, dim=1)
        H_interp_unit = torch.nn.functional.normalize(H_interp_res, dim=1)
        cos_to_anchors = H_interp_unit @ H_vocab_unit.T  # (n_steps, n_vocab)

        fig, axes = plt.subplots(1, 2, figsize=(20, max(8, n_steps * 0.4)),
                                 gridspec_kw={"width_ratios": [1, 3]})

        # left: t-axis with top-3 anchor labels per step
        ax_l = axes[0]
        ax_l.set_xlim(0, 1)
        ax_l.set_ylim(-0.5, n_steps - 0.5)
        for j, s in enumerate(steps):
            t = s["t"]
            row_y = n_steps - 1 - j
            # color bar
            bar_color = (1 - t) * np.array([0.12, 0.46, 0.71]) + t * np.array([1.00, 0.50, 0.05])
            ax_l.add_patch(plt.Rectangle((0.0, row_y - 0.3), t, 0.6, color=bar_color, alpha=0.7))
            ax_l.text(-0.05, row_y, f"t={t:.3f}", ha="right", va="center", fontsize=9, family="monospace")
            # top-3 anchors
            top3_idx = torch.topk(cos_to_anchors[j], 3).indices.tolist()
            top3_words = [(vocab_caps[idx]["word"],
                            float(cos_to_anchors[j, idx].item())) for idx in top3_idx]
            label_txt = "  ".join(f"{w} ({cos:+.2f})" for w, cos in top3_words)
            ax_l.text(1.05, row_y, label_txt, ha="left", va="center", fontsize=9, family="monospace")
        ax_l.axis("off")
        ax_l.set_title(f"top-3 anchors per step\n(cosine after sink removal)")

        # right: heatmap of cosine to all anchors, sorted by category
        ax_r = axes[1]
        cat_order: list[str] = list(vocab["categories"])
        # build sorted vocab index
        vocab_order = []
        cat_boundaries = []
        for cat in cat_order:
            for i, c in enumerate(vocab_caps):
                if c["category"] == cat:
                    vocab_order.append(i)
            cat_boundaries.append(len(vocab_order))

        sorted_cos = cos_to_anchors[:, vocab_order].numpy()
        im = ax_r.imshow(sorted_cos, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        for b in cat_boundaries[:-1]:
            ax_r.axvline(b - 0.5, color="black", linewidth=0.3, alpha=0.5)
        # category labels on x-axis
        cat_centers = [0] + cat_boundaries
        label_pos = [(cat_centers[i] + cat_centers[i+1]) / 2 for i in range(len(cat_order))]
        ax_r.set_xticks(label_pos)
        ax_r.set_xticklabels(cat_order, rotation=45, ha="right", fontsize=8)
        ax_r.set_yticks(range(n_steps))
        ax_r.set_yticklabels([f"t={s['t']:.2f}" for s in reversed(steps)], fontsize=8)
        ax_r.set_title("cos(h_t, anchor) heatmap — anchors grouped by category")
        fig.colorbar(im, ax=ax_r, label="cosine")
        # flip y so t=0 is at top
        ax_r.invert_yaxis()

        fig.suptitle(f"fig21 — Interpolation trajectory through anchor space\n"
                     f"AR({flip['anchor_A_label']}) → AR({flip['anchor_B_label']})", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIGDIR / "fig21_interp_through_anchors.png", dpi=140)
        plt.close(fig)
        print(f"wrote {FIGDIR}/fig21_interp_through_anchors.png")
    else:
        print(f"  (skipping fig21: {flip_path} not found)")

    # --- fig22: anchor cosine matrix (category-ordered) ---
    H_v_unit = torch.nn.functional.normalize(H_vocab_res, dim=1)
    C_v = H_v_unit @ H_v_unit.T

    cat_order = list(vocab["categories"])
    vocab_order = []
    cat_boundaries = []
    for cat in cat_order:
        for i, c in enumerate(vocab_caps):
            if c["category"] == cat:
                vocab_order.append(i)
        cat_boundaries.append(len(vocab_order))
    vocab_order_arr = np.array(vocab_order)
    C_sorted = C_v[np.ix_(vocab_order_arr, vocab_order_arr)].numpy()

    fig, ax = plt.subplots(figsize=(14, 13))
    im = ax.imshow(C_sorted, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for b in cat_boundaries[:-1]:
        ax.axhline(b - 0.5, color="black", linewidth=0.5)
        ax.axvline(b - 0.5, color="black", linewidth=0.5)
    cat_centers = [0] + cat_boundaries
    label_pos = [(cat_centers[i] + cat_centers[i+1]) / 2 for i in range(len(cat_order))]
    ax.set_xticks(label_pos)
    ax.set_xticklabels(cat_order, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(label_pos)
    ax.set_yticklabels(cat_order, fontsize=8)
    ax.set_title(f"fig22 — Anchor pairwise cosine (sink-removed), grouped by category\n"
                 f"diagonal bright blocks = tight category clusters")
    fig.colorbar(im, ax=ax, label="cosine")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig22_anchor_cosine_matrix.png", dpi=140)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig22_anchor_cosine_matrix.png")

    # Print per-category cohesion stats
    print("\n--- per-category intra-cohesion (mean intra-category cosine, sink-removed) ---")
    cat_stats: list[tuple[str, float, int]] = []
    for cat in cat_order:
        idxs = [i for i, c in enumerate(vocab_caps) if c["category"] == cat]
        if len(idxs) < 2:
            continue
        block = C_v[idxs][:, idxs]
        eye = torch.eye(len(idxs), dtype=torch.bool)
        intra = float(block[~eye].mean().item())
        cat_stats.append((cat, intra, len(idxs)))
    cat_stats.sort(key=lambda x: -x[1])
    for cat, intra, count in cat_stats:
        print(f"  {cat:<18} n={count:>3}  intra-cos={intra:+.3f}")


if __name__ == "__main__":
    main()
