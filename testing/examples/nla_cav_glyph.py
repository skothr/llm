"""fig13: render the country CAV direction as a signature glyph and test
the H3 hypothesis (is the country direction dim-32-aligned?).

For the CAV direction at magnitude-150 scaling (matches AV input):
  * compute its values at all 8 feature dims and 7 sink dims
  * compute cos(CAV_direction, e_idx) for those 15 indices
  * render the CAV as a signature glyph alongside the source-pool glyphs
  * print a ranking table of which dims the CAV aligns with

The expected result if H3 is true: dim 32 alone accounts for a large
fraction of the CAV direction.
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


ARTIFACTS = Path("testing/.cache/nla_artifacts")
FIGDIR = Path("research/observations/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


def draw_signature_glyph(ax: Any, cx: float, cy: float, values: np.ndarray,
                          radius: float, scale_max: float, lw: float = 1.5) -> None:
    n = len(values)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for i, v in enumerate(values):
        mag = min(abs(v) / scale_max, 1.0)
        if mag < 0.02:
            continue
        x_end = cx + radius * mag * np.cos(angles[i])
        y_end = cy + radius * mag * np.sin(angles[i])
        color = "#d62728" if v >= 0 else "#1f77b4"
        ax.plot([cx, x_end], [cy, y_end], color=color, linewidth=lw, alpha=0.9,
                solid_capstyle="round")


def main() -> None:
    cav_data = torch.load(ARTIFACTS / "country_concept_vector.pt", weights_only=False)
    pw = torch.load(ARTIFACTS / "pairwise_and_hotdims.pt", weights_only=False)
    labels: dict[int, str] = pw["labels"]
    sink_dims = sorted([idx for idx, lbl in labels.items() if lbl == "sink"])
    feature_dims = sorted([idx for idx, lbl in labels.items() if lbl == "feature"])

    direction_unit = cav_data["direction_unit"]  # unit vector in R^3584
    country_mean = cav_data["country_mean"]
    non_country_mean = cav_data["non_country_mean"]

    print(f"feature dims (glyph order): {feature_dims}")
    print(f"sink dims:                  {sink_dims}")

    # CAV at the AV-injection scale of 150
    direction_av = direction_unit * 150.0
    print(f"\n||CAV_unit||      = {direction_unit.norm().item():.3f}  (should be 1)")
    print(f"||CAV @ scale 150|| = {direction_av.norm().item():.3f}")

    # H3 test: how aligned is direction_unit with each basis vector?
    print(f"\n--- H3 test: |CAV_unit . e_idx| ranked (top 20) ---")
    abs_components = direction_unit.abs()
    top_idxs = abs_components.topk(20).indices.tolist()
    for j, idx in enumerate(top_idxs):
        val = direction_unit[idx].item()
        char = labels.get(idx, "(not in top-20 hot)")
        share_av = (val ** 2) / float((direction_unit ** 2).sum().item())  # fraction of variance
        print(f"  #{j+1:>2}  dim {idx:>5}: CAV_unit value = {val:+.4f}   "
              f"sq share = {share_av:.4f}  char = {char}")

    # specific dim 32 check
    cos_e32 = float(direction_unit[32].item())  # CAV is already unit norm; cos with e_32 = direction_unit[32]
    print(f"\nH3 specific: cos(CAV_unit, e_32) = direction_unit[32] = {cos_e32:+.4f}")
    print(f"  (H3 prediction was >= +0.4; result: {'PASS' if abs(cos_e32) >= 0.4 else 'FAIL'})")

    # feature-dim contributions
    print(f"\n--- CAV contribution at feature-classified dims ---")
    for d in feature_dims:
        val = direction_unit[d].item()
        print(f"  dim {d:>5}: direction_unit[{d}] = {val:+.4f}")
    print(f"\n--- CAV contribution at sink-classified dims ---")
    for d in sink_dims:
        val = direction_unit[d].item()
        print(f"  dim {d:>5}: direction_unit[{d}] = {val:+.4f}")

    # --- figure: CAV glyph + source-pool glyphs side by side ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    # glyph normalizer: use the max abs value across all glyphs we'll draw
    glyph_vectors: list[tuple[str, np.ndarray]] = [
        ("country_mean (8 prompts)", country_mean[feature_dims].numpy()),
        ("non_country_mean (8 prompts)", non_country_mean[feature_dims].numpy()),
        ("CAV @ unit (×150 for AV)", direction_unit[feature_dims].numpy() * 150.0),
        ("country - non_country (raw diff)", (country_mean - non_country_mean)[feature_dims].numpy()),
    ]
    abs_max = float(max(np.abs(v).max() for _, v in glyph_vectors))

    for ax, (title, vec) in zip(axes, glyph_vectors):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        draw_signature_glyph(ax, 0.0, 0.0, vec, radius=1.0, scale_max=abs_max, lw=2.5)
        # ray labels
        n = len(feature_dims)
        angles_lbl = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for ang, d, v in zip(angles_lbl, feature_dims, vec):
            x_lbl = 1.25 * np.cos(ang)
            y_lbl = 1.25 * np.sin(ang)
            ax.annotate(f"d{d}\n{v:+.1f}", (x_lbl, y_lbl),
                        ha="center", va="center", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(f"Country CAV direction as a signature glyph (feature-dim subspace)\n"
                 f"cos(CAV_unit, e_32) = {cos_e32:+.4f}  "
                 f"({'H3 PASS, dim-32-aligned' if abs(cos_e32) >= 0.4 else 'H3 FAIL, not dim-32-aligned'})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig13_cav_glyph.png", dpi=140)
    plt.close(fig)
    print(f"\nwrote {FIGDIR}/fig13_cav_glyph.png")


if __name__ == "__main__":
    main()
