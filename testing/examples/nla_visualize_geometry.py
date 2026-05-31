"""Visualize the geometric structure mined in nla_geometric_features.py
and nla_pairwise_and_hotdims.py.

Reads the artifacts produced by those scripts and writes a set of PNGs
into research/arcs/nla-verbalizer/observations/figures/.

Panels:

  fig1_pca_scatter.png        - 2D PCA, colored by source
  fig2_cosine_heatmap.png     - pairwise cosine matrix, ordered by source
  fig3_hot_dim_space.png      - sign_consist vs cv_abs for top-20 hot dims,
                                colored by classifier label, sized by freq_top5
  fig4_hotdim_activations.png - per-capture activations of the 4 strongest
                                sinks and top-3 feature dims; sinks are flat,
                                features vary
  fig5_kurtosis_vs_ar.png     - kurtosis vs AR-cosine, with linear fit;
                                shows geometric concentration is a faithfulness
                                predictor
  fig6_haiku_trajectory.png   - over the haiku gen trajectory: norm, kurtosis,
                                top-1% energy, AR cosine per step

All from data already on disk; no model loading needed.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import Any, cast
cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = _REPO_ROOT / "testing" / ".cache" / "nla_artifacts"
FIGDIR = _REPO_ROOT / "research" / "arcs" / "nla-verbalizer" / "observations" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

SRC_COLORS: dict[str, str] = {
    "aggregate": "#1f77b4",
    "haiku_gen": "#ff7f0e",
    "forced": "#d62728",
    "country_src": "#2ca02c",
    "non_country_src": "#9467bd",
    "country_test": "#8c564b",
}

CHAR_COLORS: dict[str, str] = {
    "sink": "#1f77b4",
    "polarized": "#ff7f0e",
    "feature": "#2ca02c",
    "rare_burst": "#d62728",
    "background": "#7f7f7f",
}


def load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    geo = torch.load(ARTIFACTS / "geometric_features.pt", weights_only=False)
    pw = torch.load(ARTIFACTS / "pairwise_and_hotdims.pt", weights_only=False)
    return geo, pw


def fig1_pca(pw: dict[str, Any]) -> None:
    H = pw["H"]
    mean_h = pw["mean_h"]
    Vh = pw["PCA_Vh"]
    items = pw["items_meta"]
    coords = (H - mean_h.unsqueeze(0)) @ Vh.T  # (n, 20)

    fig, ax = plt.subplots(figsize=(10, 8))
    for src, color in SRC_COLORS.items():
        idxs = [i for i, it in enumerate(items) if it["src"] == src]
        if not idxs:
            continue
        x = coords[idxs, 0].numpy()
        y = coords[idxs, 1].numpy()
        ax.scatter(x, y, c=color, label=f"{src} (n={len(idxs)})",
                   alpha=0.7, s=60, edgecolors="black", linewidths=0.5)
    ax.set_xlabel("PC1 (16.5% variance)")
    ax.set_ylabel("PC2 (7.7% variance)")
    ax.set_title("PCA of 167 h[20] vectors, colored by capture source")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_pca_scatter.png", dpi=180)
    plt.close(fig)
    print(f"  wrote {FIGDIR}/fig1_pca_scatter.png")


def fig2_cosine_heatmap(pw: dict[str, Any]) -> None:
    items = pw["items_meta"]
    C = pw["C"].numpy()
    src_order = ["aggregate", "haiku_gen", "forced",
                 "country_src", "non_country_src", "country_test"]
    order: list[int] = []
    boundaries: list[int] = []
    for s in src_order:
        for i, it in enumerate(items):
            if it["src"] == s:
                order.append(i)
        boundaries.append(len(order))
    order_arr = np.array(order)
    C_ordered = C[np.ix_(order_arr, order_arr)]

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(C_ordered, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for b in boundaries[:-1]:
        ax.axhline(b - 0.5, color="black", linewidth=0.6)
        ax.axvline(b - 0.5, color="black", linewidth=0.6)
    centers = [0] + boundaries
    label_pos = [(centers[i] + centers[i+1]) / 2 - 0.5 for i in range(len(src_order))]
    ax.set_xticks(label_pos)
    ax.set_xticklabels(src_order, rotation=30, ha="right")
    ax.set_yticks(label_pos)
    ax.set_yticklabels(src_order)
    ax.set_title("Pairwise cosine between 167 h[20] vectors, grouped by source")
    fig.colorbar(im, ax=ax, label="cosine")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_cosine_heatmap.png", dpi=180)
    plt.close(fig)
    print(f"  wrote {FIGDIR}/fig2_cosine_heatmap.png")


def fig3_hot_dim_space(pw: dict[str, Any]) -> None:
    from collections import Counter
    counts5: Counter = Counter(pw["counts_top5"])
    stats = pw["stats_by_dim"]
    labels = pw["labels"]
    top_dims = [idx for idx, _ in counts5.most_common(20)]

    fig, ax = plt.subplots(figsize=(10, 8))
    for char, color in CHAR_COLORS.items():
        xs, ys, sizes, labels_text = [], [], [], []
        for idx in top_dims:
            if labels.get(idx) == char:
                s = stats[idx]
                xs.append(s["sign_consist"])
                ys.append(s["cv_abs"])
                sizes.append(200 + 1500 * s["freq_top5"])
                labels_text.append(str(idx))
        if xs:
            ax.scatter(xs, ys, c=color, label=char, alpha=0.75,
                       s=sizes, edgecolors="black", linewidths=0.7)
            for x, y, lbl in zip(xs, ys, labels_text):
                ax.annotate(lbl, (x, y), textcoords="offset points",
                            xytext=(8, 4), fontsize=8)

    ax.axvspan(-0.02, 0.05, color="lightblue", alpha=0.15)
    ax.axvspan(0.95, 1.02, color="lightblue", alpha=0.15)
    ax.axhline(0.55, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("sign_consist  (fraction of captures with h[idx] >= 0)")
    ax.set_ylabel("cv_abs  (coefficient of variation of |h[idx]|)")
    ax.set_title("Top-20 hot dims in (sign-locking x burstiness) space\n"
                 "size = freq_top5; color = character")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_hot_dim_space.png", dpi=180)
    plt.close(fig)
    print(f"  wrote {FIGDIR}/fig3_hot_dim_space.png")


def fig4_hotdim_activations(pw: dict[str, Any]) -> None:
    H = pw["H"]
    items = pw["items_meta"]
    labels = pw["labels"]
    # Strongest 4 sinks (by freq_top5) and strongest 3 features
    from collections import Counter
    counts5: Counter = Counter(pw["counts_top5"])
    top = [idx for idx, _ in counts5.most_common(20)]
    sinks = [idx for idx in top if labels.get(idx) == "sink"][:4]
    feats = [idx for idx in top if labels.get(idx) == "feature"][:3]

    n = H.shape[0]
    src_labels = [it["src"] for it in items]
    src_order = ["aggregate", "haiku_gen", "forced",
                 "country_src", "non_country_src", "country_test"]
    order: list[int] = []
    for s in src_order:
        order.extend([i for i, lbl in enumerate(src_labels) if lbl == s])
    order_arr = np.array(order)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax_s, ax_f = axes
    for idx in sinks:
        vals = H[order_arr, idx].numpy()
        ax_s.plot(range(n), vals, label=f"dim {idx} (sink)", marker=".", markersize=3, linewidth=0.6)
    for idx in feats:
        vals = H[order_arr, idx].numpy()
        ax_f.plot(range(n), vals, label=f"dim {idx} (feature)", marker=".", markersize=3, linewidth=0.6)

    # vertical lines for source boundaries
    cursor = 0
    for s in src_order:
        cnt = sum(1 for lbl in src_labels if lbl == s)
        cursor += cnt
        if cursor < n:
            ax_s.axvline(cursor - 0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
            ax_f.axvline(cursor - 0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)

    ax_s.axhline(0, color="black", linewidth=0.4, alpha=0.5)
    ax_f.axhline(0, color="black", linewidth=0.4, alpha=0.5)
    ax_s.set_title("Sink dims (sign-locked, always-on)")
    ax_f.set_title("Feature dims (sign flips, content-bearing)")
    ax_s.set_ylabel("h[idx] value")
    ax_f.set_ylabel("h[idx] value")
    ax_f.set_xlabel("capture index (ordered by source group)")
    for ax in (ax_s, ax_f):
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Sinks stay locked; features flip with content")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_hotdim_activations.png", dpi=180)
    plt.close(fig)
    print(f"  wrote {FIGDIR}/fig4_hotdim_activations.png")


def fig5_kurtosis_vs_ar(geo: dict[str, Any]) -> None:
    rows = [r for r in geo["rows"] if r.get("cos") is not None]
    xs = np.array([r["kurtosis"] for r in rows])
    ys = np.array([r["cos"] for r in rows])
    sources = [r["src"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 7))
    for src, color in SRC_COLORS.items():
        mask = np.array([s == src for s in sources])
        if mask.sum() == 0:
            continue
        ax.scatter(xs[mask], ys[mask], c=color, label=f"{src} (n={int(mask.sum())})",
                   alpha=0.7, s=55, edgecolors="black", linewidths=0.4)

    # linear fit
    if len(xs) > 2:
        log_xs = np.log10(xs)
        m, b = np.polyfit(log_xs, ys, 1)
        x_line = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 100)
        ax.plot(x_line, m * np.log10(x_line) + b, "r--", linewidth=1.5,
                label=f"linear fit (log-x):  cos = {m:.3f}*log10(kurt) + {b:.3f}")

    ax.set_xscale("log")
    ax.set_xlabel("kurtosis of h[20] components (log scale)")
    ax.set_ylabel("AR round-trip cosine")
    ax.set_title("Geometric concentration predicts AR faithfulness\n"
                 "(diffuse h's reconstruct worse than peaky ones)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_kurtosis_vs_ar.png", dpi=180)
    plt.close(fig)
    print(f"  wrote {FIGDIR}/fig5_kurtosis_vs_ar.png")


def fig6_haiku_trajectory(geo: dict[str, Any]) -> None:
    rows = [r for r in geo["rows"] if r["src"] == "haiku_gen"]
    rows.sort(key=lambda r: r["step"] if r["step"] is not None else 0)
    steps = [r["step"] for r in rows]
    tokens = [r["token"] for r in rows]
    norms = [r["norm2"] for r in rows]
    kurts = [r["kurtosis"] for r in rows]
    top1pct = [r["top1pct_energy"] for r in rows]
    cosines = [r.get("cos") for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(steps, norms, marker="o", color="#1f77b4")
    axes[0].set_ylabel("||h||_2")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Haiku generation trajectory: per-step geometric features\n"
                      f"prompt=rabbit haiku, generated={len(rows)} tokens")
    axes[1].plot(steps, kurts, marker="o", color="#ff7f0e")
    axes[1].set_ylabel("kurtosis")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[2].plot(steps, top1pct, marker="o", color="#2ca02c")
    axes[2].set_ylabel("top-1% energy")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(steps, cosines, marker="o", color="#d62728")
    axes[3].set_ylabel("AR cosine")
    axes[3].set_xlabel("generation step")
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xticks(steps)
    axes[3].set_xticklabels([f"{s}\n{t!r}" for s, t in zip(steps, tokens)],
                             rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_haiku_trajectory.png", dpi=180)
    plt.close(fig)
    print(f"  wrote {FIGDIR}/fig6_haiku_trajectory.png")


def main() -> None:
    print("loading inputs ...")
    geo, pw = load_inputs()
    print(f"  geo: {len(geo['rows'])} rows")
    print(f"  pw:  H={tuple(pw['H'].shape)}, top dims classified={len(pw['labels'])}")

    fig1_pca(pw)
    fig2_cosine_heatmap(pw)
    fig3_hot_dim_space(pw)
    fig4_hotdim_activations(pw)
    fig5_kurtosis_vs_ar(geo)
    fig6_haiku_trajectory(geo)


if __name__ == "__main__":
    main()
