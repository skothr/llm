"""Discriminant-direction glyph: corrects fig23/fig24's "all axes
active" issue.

In fig23/fig24, every glyph had rays in nearly every direction because
the 23 category centroids were highly non-orthogonal — mean
cross-category cosine was +0.850. Each centroid was 77-97% aligned
with the grand mean of all centroids; only a tiny remainder was
category-specific.

This script uses discriminant directions instead:
  d_cat = mean(cat_h's) - mean(non_cat_h's)
  d_cat /= ||d_cat||

After this switch, the 23 directions have mean pairwise cosine -0.032
(essentially uncorrelated) with range [-0.547, +0.939]. Genuine
anti-correlations appear, and glyph rays will be sparse on real
data — only categories the h actually fits will fire.

Outputs:
  fig25_discriminant_glyph_interp.png  - interpolation flipbook redo
  fig26_discriminant_glyph_samples.png - sample captures redo
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import Any, cast
cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

import textwrap
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = _REPO_ROOT / "testing" / ".cache" / "nla_artifacts"
FIGDIR = _REPO_ROOT / "research" / "observations" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


CATEGORY_COLORS: dict[str, str] = {
    "country": "#d62728", "capital": "#e377c2",
    "nature": "#2ca02c", "codemath": "#8c564b",
    "emotion": "#ff7f0e", "refusal": "#7f0000",
    "article": "#9edae5", "pronoun": "#17becf",
    "demonstrative": "#aec7e8", "preposition": "#1f77b4",
    "conjunction": "#9467bd", "auxiliary": "#c5b0d5",
    "negation": "#000000", "quantifier": "#bcbd22",
    "wh_word": "#dbdb8d",
    "p_ender": "#7f7f7f", "p_internal": "#c7c7c7",
    "p_quote": "#5a5a5a", "p_bracket": "#3f3f3f",
    "p_dash": "#2a2a2a", "p_special": "#a0a0a0",
    "number": "#98df8a", "math_op": "#5fbf5f",
}


def apply_sink_removal(h: torch.Tensor, sink_dims: list[int]) -> torch.Tensor:
    h_res = h.clone()
    h_res[..., sink_dims] = 0.0
    return h_res


def compute_discriminant_dirs(
    vocab: dict[str, Any], sink_dims: list[int]
) -> dict[str, torch.Tensor]:
    """For each category, return unit-normalized (mean - mean_of_others)."""
    caps = vocab["captures"]
    cats = vocab["categories"]
    by_cat_idxs: dict[str, list[int]] = {}
    for cat in cats:
        by_cat_idxs[cat] = [i for i, c in enumerate(caps) if c["category"] == cat]
    cat_means: dict[str, torch.Tensor] = {}
    for cat in cats:
        cat_hs = torch.stack([caps[i]["h"] for i in by_cat_idxs[cat]])
        cat_hs_res = apply_sink_removal(cat_hs, sink_dims)
        cat_means[cat] = cat_hs_res.mean(dim=0)

    discr: dict[str, torch.Tensor] = {}
    for cat in cats:
        # mean of all captures NOT in this category
        other_hs = torch.stack([caps[i]["h"] for i in range(len(caps))
                                  if caps[i]["category"] != cat])
        other_hs_res = apply_sink_removal(other_hs, sink_dims)
        other_mean = other_hs_res.mean(dim=0)
        d = cat_means[cat] - other_mean
        discr[cat] = d / (d.norm() + 1e-9)
    return discr


def project_to_discriminants(
    h: torch.Tensor, discr: dict[str, torch.Tensor], sink_dims: list[int]
) -> dict[str, float]:
    h_res = apply_sink_removal(h, sink_dims)
    h_unit = h_res / (h_res.norm() + 1e-9)
    return {cat: float(torch.dot(h_unit, d).item()) for cat, d in discr.items()}


def draw_discriminant_glyph(
    ax: Any, cx: float, cy: float, projections: dict[str, float],
    categories: list[str], radius: float, scale_max: float,
    lw: float = 2.0, show_labels: bool = False,
) -> None:
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = plt.Circle((cx, cy), radius, color="#cccccc", linewidth=0.5, fill=False, alpha=0.4)
    ax.add_patch(ring)
    for i, cat in enumerate(categories):
        v = projections.get(cat, 0.0)
        mag = min(abs(v) / scale_max, 1.0)
        color = CATEGORY_COLORS.get(cat, "#999999")
        if mag < 0.02:
            continue
        x_end = cx + radius * mag * np.cos(angles[i])
        y_end = cy + radius * mag * np.sin(angles[i])
        if v < 0:
            # Negative projection: same direction but dotted, indicating
            # "anti-this-category"
            ax.plot([cx, x_end], [cy, y_end], color=color, linewidth=lw, alpha=0.6,
                    linestyle=":", solid_capstyle="round")
        else:
            ax.plot([cx, x_end], [cy, y_end], color=color, linewidth=lw, alpha=0.95,
                    solid_capstyle="round")
        if show_labels:
            x_lbl = cx + 1.18 * radius * np.cos(angles[i])
            y_lbl = cy + 1.18 * radius * np.sin(angles[i])
            ax.text(x_lbl, y_lbl, cat, ha="center", va="center", fontsize=6, alpha=0.75,
                    color=color)


def main() -> None:
    vocab = torch.load(ARTIFACTS / "vocab_atlas.pt", weights_only=False)
    pw = torch.load(ARTIFACTS / "pairwise_and_hotdims.pt", weights_only=False)
    labels: dict[int, str] = pw["labels"]
    sink_dims = sorted([idx for idx, lbl in labels.items() if lbl == "sink"])
    categories = list(vocab["categories"])

    discr = compute_discriminant_dirs(vocab, sink_dims)
    print(f"computed discriminant directions for {len(discr)} categories")

    # Quick sanity: mean pairwise cosine of discriminants
    D = torch.stack([discr[c] for c in categories])
    C_d = D @ D.T
    eye = torch.eye(len(categories), dtype=torch.bool)
    off = C_d[~eye]
    print(f"discriminant pairwise cosines: mean={float(off.mean().item()):+.3f}  "
          f"min={float(off.min().item()):+.3f}  max={float(off.max().item()):+.3f}")

    # ---- fig25: interpolation flipbook with discriminant glyph ----
    flip = torch.load(ARTIFACTS / "interpolation_flipbook.pt", weights_only=False)
    steps = sorted(flip["steps"], key=lambda s: s["step"])
    n_steps = len(steps)
    all_projs = [project_to_discriminants(s["h_t"], discr, sink_dims) for s in steps]
    scale_max = max(max(abs(v) for v in p.values()) for p in all_projs)
    print(f"interp scale_max: {scale_max:.3f}")

    TITLE_RESERVE = 0.04
    fig = plt.figure(figsize=(20, 1.7 * n_steps))
    for j, s in enumerate(steps):
        row_height = (1.0 - TITLE_RESERVE) / n_steps
        row_y = (1.0 - TITLE_RESERVE) - (j + 1) * row_height
        projs = all_projs[j]
        t = s["t"]

        ax_t = fig.add_axes((0.01, row_y, 0.06, row_height))
        ax_t.axis("off")
        bar_color = (1 - t) * np.array([0.12, 0.46, 0.71]) + t * np.array([1.00, 0.50, 0.05])
        ax_t.add_patch(plt.Rectangle((0.0, 0.1), t, 0.8, color=bar_color, alpha=0.85))
        ax_t.set_xlim(0, 1)
        ax_t.set_ylim(0, 1)
        ax_t.text(0.5, 0.5, f"t={t:.3f}", ha="center", va="center", fontsize=10, family="monospace")

        ax_g = fig.add_axes((0.08, row_y + 0.005, 0.13, row_height - 0.01))
        ax_g.set_xlim(-1.4, 1.4)
        ax_g.set_ylim(-1.4, 1.4)
        ax_g.set_aspect("equal")
        ax_g.axis("off")
        draw_discriminant_glyph(ax_g, 0.0, 0.0, projs, categories, radius=1.0, scale_max=scale_max,
                                 lw=2.0, show_labels=(j == 0))

        ax_top = fig.add_axes((0.22, row_y + 0.005, 0.15, row_height - 0.01))
        ax_top.axis("off")
        top_sorted = sorted(projs.items(), key=lambda x: -x[1])
        top3 = top_sorted[:3]
        bot3 = top_sorted[-3:]
        top_str = "TOP-3:\n" + "\n".join(f"  {cat:>12} {v:+.3f}" for cat, v in top3)
        bot_str = "BOTTOM-3:\n" + "\n".join(f"  {cat:>12} {v:+.3f}" for cat, v in bot3)
        ax_top.text(0.0, 0.5, top_str + "\n\n" + bot_str, ha="left", va="center",
                    fontsize=8, family="monospace")

        ax_txt = fig.add_axes((0.38, row_y + 0.005, 0.60, row_height - 0.01))
        ax_txt.axis("off")
        av = s.get("av_text", "")
        first_para = av.split("\n\n")[0] if av else ""
        wrapped = textwrap.fill(first_para, width=140, break_long_words=False, break_on_hyphens=False)
        ax_txt.text(0.0, 0.5, wrapped, ha="left", va="center", fontsize=8, family="serif")

    fig.suptitle(
        f"fig25 — Interpolation flipbook with DISCRIMINANT-direction glyphs (fig23 successor)\n"
        f"23-ray glyph: each ray = (category_mean - non_category_mean) unit dir, sink-removed; "
        f"length = cosine projection. Solid = +, dotted = -. "
        f"Mean cross-axis cos: {float(off.mean().item()):+.3f}",
        fontsize=11, y=0.995)
    fig.savefig(FIGDIR / "fig25_discriminant_glyph_interp.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig25_discriminant_glyph_interp.png")

    # ---- fig26: sample existing captures ----
    items = pw["items_meta"]
    H_existing = pw["H"]
    samples_to_render: list[tuple[str, torch.Tensor]] = []
    for i, it in enumerate(items):
        if it["src"] == "country_src" and it["prompt_id"] == "country_0":
            samples_to_render.append((f"country_src/country_0\n('capital of France?')", H_existing[i]))
            break
    for i, it in enumerate(items):
        if it["src"] == "haiku_gen" and it.get("step") == 0:
            samples_to_render.append((f"haiku_gen/step 0 ({it['token']!r})", H_existing[i]))
            break
    for i, it in enumerate(items):
        if it["src"] == "forced" and "refuse" in (it.get("token") or "").lower() and "refusal" in it["prompt_id"]:
            samples_to_render.append((f"forced/refusal/refuse", H_existing[i]))
            break
    for i, it in enumerate(items):
        if it["src"] == "aggregate" and it["prompt_id"] == "code" and (it.get("token") or "").strip() == "function":
            samples_to_render.append((f"aggregate/code ('function')", H_existing[i]))
            break
    samples_to_render.append((f"interp t=0.000 (anchor A)", steps[0]["h_t"]))
    samples_to_render.append((f"interp t=1.000 (anchor B)", steps[-1]["h_t"]))

    sample_projs = [project_to_discriminants(h, discr, sink_dims) for _, h in samples_to_render]
    sample_scale = max(max(abs(v) for v in p.values()) for p in sample_projs)
    print(f"sample scale_max: {sample_scale:.3f}")

    n_samples = len(samples_to_render)
    fig, axes = plt.subplots(1, n_samples, figsize=(4.5 * n_samples, 5.5))
    if n_samples == 1:
        axes = [axes]
    for ax, (title, _), projs in zip(axes, samples_to_render, sample_projs):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        draw_discriminant_glyph(ax, 0.0, 0.0, projs, categories, radius=1.0, scale_max=sample_scale,
                                 lw=2.8, show_labels=True)
        sorted_p = sorted(projs.items(), key=lambda x: -x[1])
        top3 = sorted_p[:3]
        bot3 = sorted_p[-3:]
        top_str = "  ".join(f"{cat}({v:+.2f})" for cat, v in top3)
        bot_str = "  ".join(f"{cat}({v:+.2f})" for cat, v in bot3)
        ax.set_title(f"{title}\ntop-3: {top_str}\nbottom-3: {bot_str}", fontsize=8)

    fig.suptitle(
        f"fig26 — Sample captures with discriminant-direction glyphs (fig24 successor)\n"
        f"Now sparse: only categories the h actually fits have long rays. "
        f"Dotted rays = anti-category (h projects negative on that discriminant).",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig26_discriminant_glyph_samples.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig26_discriminant_glyph_samples.png")


if __name__ == "__main__":
    main()
