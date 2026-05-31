"""fig28: render the discriminant stability scan.

Loads discriminant_stability.pt (8 anchors × 4 contexts = 32 captures).
For each anchor word, projects all 4 contexts onto the 23 discriminant
directions and visualizes:

  Layout: 8 rows (one per anchor) × 5 columns
    col 0: anchor + expected category label
    col 1: glyph for 'single' context
    col 2: glyph for 'subject' context
    col 3: glyph for 'object' context
    col 4: glyph for 'middle' context

Plus a summary bar chart at the bottom showing per-anchor discriminant
projection variance (the stability metric) and per-anchor mean
projection onto the expected category (the strength metric).

The headline question: does an anchor's category-discriminant projection
stay consistent across contexts (stability), or does it shift with
position (instability)?
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import Any, cast

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from _nla_artifacts import FIGURES as FIGDIR
from _nla_artifacts import load_artifact

FIGDIR.mkdir(parents=True, exist_ok=True)


CATEGORY_COLORS: dict[str, str] = {
    "country": "#d62728",
    "capital": "#e377c2",
    "nature": "#2ca02c",
    "codemath": "#8c564b",
    "emotion": "#ff7f0e",
    "refusal": "#7f0000",
    "article": "#9edae5",
    "pronoun": "#17becf",
    "demonstrative": "#aec7e8",
    "preposition": "#1f77b4",
    "conjunction": "#9467bd",
    "auxiliary": "#c5b0d5",
    "negation": "#000000",
    "quantifier": "#bcbd22",
    "wh_word": "#dbdb8d",
    "p_ender": "#7f7f7f",
    "p_internal": "#c7c7c7",
    "p_quote": "#5a5a5a",
    "p_bracket": "#3f3f3f",
    "p_dash": "#2a2a2a",
    "p_special": "#a0a0a0",
    "number": "#98df8a",
    "math_op": "#5fbf5f",
}


def apply_sink_removal(h: torch.Tensor, sink_dims: list[int]) -> torch.Tensor:
    h_res = h.clone()
    h_res[..., sink_dims] = 0.0
    return h_res


def compute_discriminants(
    vocab: dict[str, Any], sink_dims: list[int]
) -> dict[str, torch.Tensor]:
    caps = vocab["captures"]
    cats = vocab["categories"]
    discr: dict[str, torch.Tensor] = {}
    for cat in cats:
        in_cat = [i for i, c in enumerate(caps) if c["category"] == cat]
        out_cat = [i for i in range(len(caps)) if caps[i]["category"] != cat]
        in_hs = apply_sink_removal(
            torch.stack([caps[i]["h"] for i in in_cat]), sink_dims
        )
        out_hs = apply_sink_removal(
            torch.stack([caps[i]["h"] for i in out_cat]), sink_dims
        )
        d = in_hs.mean(dim=0) - out_hs.mean(dim=0)
        discr[cat] = d / (d.norm() + 1e-9)
    return discr


def project(
    h: torch.Tensor,
    discr: dict[str, torch.Tensor],
    categories: list[str],
    sink_dims: list[int],
) -> dict[str, float]:
    h_res = apply_sink_removal(h, sink_dims)
    h_unit = h_res / (h_res.norm() + 1e-9)
    return {cat: float(torch.dot(h_unit, discr[cat]).item()) for cat in categories}


def draw_glyph(
    ax: Any,
    projs: dict[str, float],
    categories: list[str],
    scale_max: float,
    lw: float = 2.0,
) -> None:
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = plt.Circle(
        (0.0, 0.0), 1.0, color="#cccccc", linewidth=0.4, fill=False, alpha=0.4
    )
    ax.add_patch(ring)
    for i, cat in enumerate(categories):
        v = projs.get(cat, 0.0)
        mag = min(abs(v) / scale_max, 1.0)
        if mag < 0.02:
            continue
        x_end = mag * np.cos(angles[i])
        y_end = mag * np.sin(angles[i])
        color = CATEGORY_COLORS.get(cat, "#999999")
        if v < 0:
            ax.plot(
                [0, x_end],
                [0, y_end],
                color=color,
                linewidth=lw,
                alpha=0.6,
                linestyle=":",
                solid_capstyle="round",
            )
        else:
            ax.plot(
                [0, x_end],
                [0, y_end],
                color=color,
                linewidth=lw,
                alpha=0.95,
                solid_capstyle="round",
            )


def main() -> None:
    stab = load_artifact("discriminant_stability.pt")
    vocab = load_artifact("vocab_atlas.pt")
    pw = load_artifact("pairwise_and_hotdims.pt")
    labels: dict[int, str] = pw["labels"]
    sink_dims = sorted([idx for idx, lbl in labels.items() if lbl == "sink"])
    categories = list(vocab["categories"])

    discr = compute_discriminants(vocab, sink_dims)
    caps = stab["captures"]
    print(f"loaded {len(caps)} stability captures")

    # Group captures by anchor
    by_anchor: dict[str, dict[str, dict[str, Any]]] = {}
    expected_cat: dict[str, str] = {}
    for cap in caps:
        by_anchor.setdefault(cap["anchor"], {})[cap["context"]] = cap
        expected_cat[cap["anchor"]] = cap["expected_cat"]
    anchors = list(by_anchor.keys())

    # Compute projections for every capture
    all_projs: dict[str, dict[str, dict[str, float]]] = {}
    for anchor in anchors:
        all_projs[anchor] = {}
        for ctx, cap in by_anchor[anchor].items():
            all_projs[anchor][ctx] = project(cap["h"], discr, categories, sink_dims)

    contexts = ["single", "short", "medium", "long"]

    # Find global scale_max across all captures
    scale_max = max(
        max(abs(v) for v in all_projs[a][c].values())
        for a in anchors
        for c in contexts
        if c in all_projs[a]
    )
    print(f"scale_max across stability captures: {scale_max:.3f}")

    n_anchors = len(anchors)
    n_cols = 1 + len(contexts)  # 1 for label + 4 for glyphs
    fig, axes = plt.subplots(n_anchors, n_cols, figsize=(2.5 * n_cols, 2.5 * n_anchors))
    for r, anchor in enumerate(anchors):
        exp = expected_cat[anchor]
        # Label column
        ax = axes[r, 0]
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            anchor,
            ha="center",
            va="center",
            fontsize=16,
            family="monospace",
            weight="bold",
            color=CATEGORY_COLORS.get(exp, "black"),
        )
        ax.text(0.5, 0.35, f"expected:\n{exp}", ha="center", va="center", fontsize=9)

        # Compute stability metrics for this anchor
        proj_vec_per_ctx = {
            ctx: torch.tensor([all_projs[anchor][ctx][c] for c in categories])
            for ctx in contexts
            if ctx in all_projs[anchor]
        }
        pv = torch.stack(list(proj_vec_per_ctx.values()))
        # Pairwise cosine between context projection vectors
        pv_unit = torch.nn.functional.normalize(pv, dim=1)
        ctx_cos = pv_unit @ pv_unit.T
        eye = torch.eye(len(contexts), dtype=torch.bool)
        mean_ctx_cos = float(ctx_cos[~eye].mean().item())
        # Expected-cat projection across contexts
        exp_projs = [all_projs[anchor][c][exp] for c in contexts]
        exp_mean = float(np.mean(exp_projs))
        exp_std = float(np.std(exp_projs))
        ax.text(
            0.5,
            0.10,
            f"mean ctx-cos: {mean_ctx_cos:+.3f}\n"
            f"{exp}-proj: {exp_mean:+.3f}±{exp_std:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            family="monospace",
        )

        # Glyph columns
        for c_idx, ctx in enumerate(contexts):
            ax = axes[r, c_idx + 1]
            ax.set_xlim(-1.4, 1.4)
            ax.set_ylim(-1.4, 1.4)
            ax.set_aspect("equal")
            ax.axis("off")
            if ctx in all_projs[anchor]:
                draw_glyph(
                    ax, all_projs[anchor][ctx], categories, scale_max=scale_max, lw=2.0
                )
                # Top-1 of this glyph
                top1_cat = max(all_projs[anchor][ctx].items(), key=lambda kv: kv[1])
                top1_color = CATEGORY_COLORS.get(top1_cat[0], "#999")
                ax.set_title(
                    f"{ctx}\ntop-1: {top1_cat[0]} ({top1_cat[1]:+.2f})",
                    fontsize=8,
                    color=top1_color,
                )
            else:
                ax.set_title(f"{ctx}  (missing)", fontsize=8)

    fig.suptitle(
        "fig28 — Discriminant stability: 8 anchors × 4 contexts each\n"
        "Stable anchor = glyphs match across contexts; unstable = position dominates\n"
        f"(scale_max = {scale_max:.3f}, sink-removed)",
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig28_discriminant_stability.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig28_discriminant_stability.png")

    # Print numeric stability table
    print(
        f"\n{'anchor':<12} {'expected':<10} {'ctx-cos mean':>12} {'exp-proj mean':>14} {'exp-proj std':>14}"
    )
    print("-" * 65)
    for anchor in anchors:
        exp = expected_cat[anchor]
        proj_vec_per_ctx = {
            ctx: torch.tensor([all_projs[anchor][ctx][c] for c in categories])
            for ctx in contexts
            if ctx in all_projs[anchor]
        }
        pv = torch.stack(list(proj_vec_per_ctx.values()))
        pv_unit = torch.nn.functional.normalize(pv, dim=1)
        ctx_cos = pv_unit @ pv_unit.T
        eye = torch.eye(len(contexts), dtype=torch.bool)
        mean_ctx_cos = float(ctx_cos[~eye].mean().item())
        exp_projs = [all_projs[anchor][c][exp] for c in contexts]
        exp_mean = float(np.mean(exp_projs))
        exp_std = float(np.std(exp_projs))
        print(
            f"{anchor:<12} {exp:<10} {mean_ctx_cos:>+12.3f} {exp_mean:>+14.3f} {exp_std:>+14.3f}"
        )


if __name__ == "__main__":
    main()
