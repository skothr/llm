"""Anchor-projection glyph: each ray is a named CATEGORY axis (not an
anonymous dim index). Computes category centroids from the vocab atlas
and renders any h-vector as a 23-ray glyph with cosine-projection
ray lengths.

Produces:
  fig23_anchor_glyph_interp.png  - re-renders the interpolation flipbook
                                    (fig17 successor) with anchor-axis glyphs
  fig24_anchor_glyph_samples.png - representative existing captures rendered
                                    with the same primitive for cross-corpus
                                    comparison

All projections after sink removal.
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

from _nla_artifacts import read_artifact

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIGDIR = (
    _REPO_ROOT / "research" / "arcs" / "nla-verbalizer" / "observations" / "figures"
)
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


def compute_centroids(
    vocab: dict[str, Any], sink_dims: list[int]
) -> dict[str, torch.Tensor]:
    """Returns {category: unit-vector centroid} after sink removal."""
    caps = vocab["captures"]
    centroids: dict[str, torch.Tensor] = {}
    for cat in vocab["categories"]:
        idxs = [i for i, c in enumerate(caps) if c["category"] == cat]
        if not idxs:
            continue
        cat_hs = torch.stack([caps[i]["h"] for i in idxs])
        cat_hs_res = apply_sink_removal(cat_hs, sink_dims)
        centroid = cat_hs_res.mean(dim=0)
        centroids[cat] = centroid / (centroid.norm() + 1e-9)
    return centroids


def project_to_anchors(
    h: torch.Tensor, centroids: dict[str, torch.Tensor], sink_dims: list[int]
) -> dict[str, float]:
    """Cosine projection of (sink-removed) h onto each unit centroid."""
    h_res = apply_sink_removal(h, sink_dims)
    h_unit = h_res / (h_res.norm() + 1e-9)
    return {cat: float(torch.dot(h_unit, c).item()) for cat, c in centroids.items()}


def draw_anchor_glyph(
    ax: Any,
    cx: float,
    cy: float,
    projections: dict[str, float],
    categories: list[str],
    radius: float,
    scale_max: float,
    lw: float = 2.0,
    show_labels: bool = False,
) -> None:
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Ring marker for the unit-circle reference
    ring = plt.Circle(
        (cx, cy), radius, color="#cccccc", linewidth=0.5, fill=False, alpha=0.4
    )
    ax.add_patch(ring)
    for i, cat in enumerate(categories):
        v = projections.get(cat, 0.0)
        mag = min(abs(v) / scale_max, 1.0)
        if mag < 0.02:
            continue
        x_end = cx + radius * mag * np.cos(angles[i])
        y_end = cy + radius * mag * np.sin(angles[i])
        color = CATEGORY_COLORS.get(cat, "#999999")
        if v < 0:
            # Negative projection rendered as a hollow stub on the opposite side
            x_end = cx - radius * mag * np.cos(angles[i])
            y_end = cy - radius * mag * np.sin(angles[i])
            ax.plot(
                [cx, x_end],
                [cy, y_end],
                color=color,
                linewidth=lw,
                alpha=0.5,
                linestyle=":",
                solid_capstyle="round",
            )
        else:
            ax.plot(
                [cx, x_end],
                [cy, y_end],
                color=color,
                linewidth=lw,
                alpha=0.95,
                solid_capstyle="round",
            )
        if show_labels:
            x_lbl = cx + 1.18 * radius * np.cos(angles[i])
            y_lbl = cy + 1.18 * radius * np.sin(angles[i])
            ax.text(
                x_lbl,
                y_lbl,
                cat,
                ha="center",
                va="center",
                fontsize=6,
                alpha=0.7,
                color=CATEGORY_COLORS.get(cat, "#999999"),
            )


def main() -> None:
    vocab = torch.load(read_artifact("vocab_atlas.pt"), weights_only=False)
    pw = torch.load(read_artifact("pairwise_and_hotdims.pt"), weights_only=False)
    labels: dict[int, str] = pw["labels"]
    sink_dims = sorted([idx for idx, lbl in labels.items() if lbl == "sink"])
    categories = list(vocab["categories"])

    centroids = compute_centroids(vocab, sink_dims)
    print(f"computed centroids for {len(centroids)} categories")

    # ---- fig23: re-render interpolation flipbook with anchor glyphs ----
    flip = torch.load(read_artifact("interpolation_flipbook.pt"), weights_only=False)
    steps = sorted(flip["steps"], key=lambda s: s["step"])
    n_steps = len(steps)

    # Compute projections for all steps to find a sensible scale_max
    all_projs: list[dict[str, float]] = [
        project_to_anchors(s["h_t"], centroids, sink_dims) for s in steps
    ]
    scale_max = max(max(abs(v) for v in p.values()) for p in all_projs)
    print(f"interp scale_max (max |cosine to centroid|): {scale_max:.3f}")

    TITLE_RESERVE = 0.04
    fig = plt.figure(figsize=(20, 1.7 * n_steps))
    for j, s in enumerate(steps):
        row_height = (1.0 - TITLE_RESERVE) / n_steps
        row_y = (1.0 - TITLE_RESERVE) - (j + 1) * row_height
        projs = all_projs[j]
        t = s["t"]

        # col 1: t-marker
        ax_t = fig.add_axes((0.01, row_y, 0.06, row_height))
        ax_t.axis("off")
        bar_color = (1 - t) * np.array([0.12, 0.46, 0.71]) + t * np.array(
            [1.00, 0.50, 0.05]
        )
        ax_t.add_patch(plt.Rectangle((0.0, 0.1), t, 0.8, color=bar_color, alpha=0.85))
        ax_t.set_xlim(0, 1)
        ax_t.set_ylim(0, 1)
        ax_t.text(
            0.5,
            0.5,
            f"t={t:.3f}",
            ha="center",
            va="center",
            fontsize=10,
            family="monospace",
        )

        # col 2: anchor glyph (23 rays)
        ax_g = fig.add_axes((0.08, row_y + 0.005, 0.13, row_height - 0.01))
        ax_g.set_xlim(-1.4, 1.4)
        ax_g.set_ylim(-1.4, 1.4)
        ax_g.set_aspect("equal")
        ax_g.axis("off")
        draw_anchor_glyph(
            ax_g,
            0.0,
            0.0,
            projs,
            categories,
            radius=1.0,
            scale_max=scale_max,
            lw=2.0,
            show_labels=(j == 0),
        )

        # col 3: top-3 categories (numeric)
        ax_top = fig.add_axes((0.22, row_y + 0.005, 0.13, row_height - 0.01))
        ax_top.axis("off")
        top3 = sorted(projs.items(), key=lambda x: -x[1])[:3]
        lines = [f"{cat:>13} {v:+.3f}" for cat, v in top3]
        ax_top.text(
            0.0,
            0.5,
            "\n".join(lines),
            ha="left",
            va="center",
            fontsize=9,
            family="monospace",
        )

        # col 4: AV text first paragraph
        ax_txt = fig.add_axes((0.36, row_y + 0.005, 0.62, row_height - 0.01))
        ax_txt.axis("off")
        av = s.get("av_text", "")
        # first paragraph only for compact view
        first_para = av.split("\n\n")[0] if av else ""
        wrapped = textwrap.fill(
            first_para, width=140, break_long_words=False, break_on_hyphens=False
        )
        ax_txt.text(
            0.0, 0.5, wrapped, ha="left", va="center", fontsize=8, family="serif"
        )

    fig.suptitle(
        f"fig23 — Interpolation flipbook with anchor-projection glyphs\n"
        f"23-ray glyph: each ray = a named CATEGORY centroid (named in row 0); "
        f"length = cos(h_t, centroid), sink-removed. "
        f"Dotted = negative projection.",
        fontsize=11,
        y=0.995,
    )
    fig.savefig(FIGDIR / "fig23_anchor_glyph_interp.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig23_anchor_glyph_interp.png")

    # ---- fig24: sample existing captures rendered with anchor glyphs ----
    items = pw["items_meta"]
    H_existing = pw["H"]
    # Pick representative captures:
    samples_to_render: list[tuple[str, torch.Tensor, str]] = []
    # Country pool: France
    for i, it in enumerate(items):
        if it["src"] == "country_src" and it["prompt_id"] == "country_0":
            samples_to_render.append(
                (f"country_src/country_0\n('capital of France?')", H_existing[i], "")
            )
            break
    # Haiku: first content token
    for i, it in enumerate(items):
        if it["src"] == "haiku_gen" and it.get("step") == 0:
            samples_to_render.append(
                (f"haiku_gen/step 0 ({it['token']!r})", H_existing[i], "")
            )
            break
    # Forced: refuse
    for i, it in enumerate(items):
        if (
            it["src"] == "forced"
            and "refuse" in (it.get("token") or "").lower()
            and "refusal" in it["prompt_id"]
        ):
            samples_to_render.append((f"forced/refusal/refuse", H_existing[i], ""))
            break
    # Aggregate: code "function"
    for i, it in enumerate(items):
        if (
            it["src"] == "aggregate"
            and it["prompt_id"] == "code"
            and (it.get("token") or "").strip() == "function"
        ):
            samples_to_render.append(
                (f"aggregate/code ('function')", H_existing[i], "")
            )
            break
    # Interpolation step 0 and step 19 for comparison
    samples_to_render.append((f"interp t=0.000 (anchor A)", steps[0]["h_t"], ""))
    samples_to_render.append((f"interp t=1.000 (anchor B)", steps[-1]["h_t"], ""))

    sample_projs = [
        project_to_anchors(h, centroids, sink_dims) for _, h, _ in samples_to_render
    ]
    sample_scale = max(max(abs(v) for v in p.values()) for p in sample_projs)
    print(f"sample scale_max: {sample_scale:.3f}")

    n_samples = len(samples_to_render)
    fig, axes = plt.subplots(1, n_samples, figsize=(4.5 * n_samples, 5))
    if n_samples == 1:
        axes = [axes]
    for ax, (title, _, _), projs in zip(axes, samples_to_render, sample_projs):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        draw_anchor_glyph(
            ax,
            0.0,
            0.0,
            projs,
            categories,
            radius=1.0,
            scale_max=sample_scale,
            lw=2.5,
            show_labels=True,
        )
        top3 = sorted(projs.items(), key=lambda x: -x[1])[:3]
        top_str = "  ".join(f"{cat}({v:+.2f})" for cat, v in top3)
        ax.set_title(f"{title}\ntop-3: {top_str}", fontsize=9)

    fig.suptitle(
        f"fig24 — Sample existing captures rendered with the 23-axis anchor-projection glyph\n"
        f"Each ray = a named category centroid; ray length = cos(h, centroid) sink-removed",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig24_anchor_glyph_samples.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig24_anchor_glyph_samples.png")


if __name__ == "__main__":
    main()
