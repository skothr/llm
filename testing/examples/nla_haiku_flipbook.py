"""fig14: haiku trajectory flipbook.

For each of the 15 haiku generation steps, render a row containing:
  * step number
  * token (with proper escaping for newlines)
  * signature glyph in the general-content feature-dim subspace
  * the AV text (wrapped)
  * AR cosine for that step

Layout: 15 rows x 4 columns (step+token, glyph, AV text, AR cosine).
Reads rabbit_haiku_gen_trajectory.pt and pairwise_and_hotdims.pt.
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


ARTIFACTS = Path("testing/.cache/nla_artifacts")
FIGDIR = Path("research/observations/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)


def draw_signature_glyph(ax: Any, cx: float, cy: float, values: np.ndarray,
                          radius: float, scale_max: float, lw: float = 2.0) -> None:
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
    haiku = torch.load(ARTIFACTS / "rabbit_haiku_gen_trajectory.pt", weights_only=False)
    pw = torch.load(ARTIFACTS / "pairwise_and_hotdims.pt", weights_only=False)
    labels: dict[int, str] = pw["labels"]
    feature_dims = sorted([idx for idx, lbl in labels.items() if lbl == "feature"])
    print(f"feature dims (glyph order): {feature_dims}")

    captures = sorted(haiku["captures"], key=lambda c: c["step"])
    n_steps = len(captures)
    print(f"n_steps = {n_steps}")
    print(f"prompt = {haiku.get('prompt')!r}")
    print(f"generated = {haiku.get('generated_text')!r}")

    # global magnitude normalizer for glyph rays
    all_feat_vals = np.concatenate(
        [c["h"][feature_dims].numpy() for c in captures]
    )
    abs_max = float(np.abs(all_feat_vals).max())
    print(f"abs_max across haiku captures (feature dims): {abs_max:.2f}")

    # figure layout: rows = n_steps, cols = 4 (label, glyph, AV text, AR-cos bar).
    # Reserve top TITLE_RESERVE fraction for the suptitle so first row doesn't overlap.
    TITLE_RESERVE = 0.05
    fig = plt.figure(figsize=(18, 1.6 * n_steps))
    row_height = (1.0 - TITLE_RESERVE) / n_steps
    for j, cap in enumerate(captures):
        row_y = (1.0 - TITLE_RESERVE) - (j + 1) * row_height
        # column 1: step + token (text only)
        ax_lbl = fig.add_axes((0.02, row_y, 0.10, row_height))
        ax_lbl.axis("off")
        tok = cap["token"]
        tok_disp = tok.replace("\n", "\\n")
        ax_lbl.text(0.0, 0.5, f"step {cap['step']:>2}\n  token={tok_disp!r}",
                    ha="left", va="center", fontsize=11, family="monospace")

        # column 2: signature glyph
        ax_g = fig.add_axes((0.13, row_y + 0.005, 0.10, row_height - 0.01))
        ax_g.set_xlim(-1.3, 1.3)
        ax_g.set_ylim(-1.3, 1.3)
        ax_g.set_aspect("equal")
        ax_g.axis("off")
        vec = cap["h"][feature_dims].numpy()
        draw_signature_glyph(ax_g, 0.0, 0.0, vec, radius=1.0, scale_max=abs_max, lw=2.5)
        # mini-labels at compass cardinal points for the feature dims
        angles = np.linspace(0, 2 * np.pi, len(feature_dims), endpoint=False)
        for ang, d in zip(angles, feature_dims):
            x_lbl = 1.18 * np.cos(ang)
            y_lbl = 1.18 * np.sin(ang)
            ax_g.text(x_lbl, y_lbl, f"d{d}", ha="center", va="center", fontsize=6, alpha=0.6)

        # column 3: AV text wrapped
        ax_txt = fig.add_axes((0.25, row_y + 0.005, 0.60, row_height - 0.01))
        ax_txt.axis("off")
        av = cap.get("av_text", "") or ""
        wrapped = "\n".join(
            textwrap.fill(p, width=120, break_long_words=False, break_on_hyphens=False)
            for p in av.split("\n\n") if p.strip()
        )
        ax_txt.text(0.0, 0.5, wrapped, ha="left", va="center", fontsize=8,
                    family="serif", wrap=True)

        # column 4: AR cosine bar (inset within the row)
        bar_h = 0.4 * row_height
        bar_y = row_y + (row_height - bar_h) / 2
        ax_bar = fig.add_axes((0.87, bar_y, 0.11, bar_h))
        cos = cap.get("cosine")
        if cos is not None:
            ax_bar.barh([0], [cos], color="#2ca02c" if cos > 0.85 else "#ff7f0e", alpha=0.7)
            ax_bar.set_xlim(0.70, 0.95)
            ax_bar.set_yticks([])
            ax_bar.set_title(f"AR cos = {cos:+.3f}", fontsize=8)
            ax_bar.tick_params(axis="x", labelsize=7)
        else:
            ax_bar.axis("off")

    fig.suptitle(f"Haiku generation flipbook: rabbit haiku, 15 steps\n"
                 f"glyph rays = {feature_dims} | red = +, blue = -",
                 fontsize=12, y=0.995)
    fig.savefig(FIGDIR / "fig14_haiku_flipbook.png", dpi=180)
    plt.close(fig)
    print(f"\nwrote {FIGDIR}/fig14_haiku_flipbook.png")


if __name__ == "__main__":
    main()
