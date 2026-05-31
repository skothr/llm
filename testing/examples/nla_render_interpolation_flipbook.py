"""Render Path B output as fig17 (flipbook) + fig18 (diagnostic).

Reads interpolation_flipbook.pt. Produces:

  fig17_interp_flipbook.png  - vertical strip flipbook with t-marker,
                                two signature glyphs per row (general-
                                content + interpolation-specific), and
                                the AV text per step.
  fig18_interp_diagnostic.png - cos(h_t, h_A) and cos(h_t, h_B) curves
                                + ||h_t|| over t + step-to-step glyph
                                distance + ||h_t - h_A|| over t.

Anchors are rendered as the t=0 and t=1 rows (so the strip is 22 rows
total including the two anchor h's).
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
FIGDIR = _REPO_ROOT / "research" / "arcs" / "nla-verbalizer" / "observations" / "figures"
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


def label_glyph_axes(ax: Any, dims: list[int], radius: float = 1.18,
                      fontsize: int = 6, alpha: float = 0.55) -> None:
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)
    for ang, d in zip(angles, dims):
        x_lbl = radius * np.cos(ang)
        y_lbl = radius * np.sin(ang)
        ax.text(x_lbl, y_lbl, f"d{d}", ha="center", va="center", fontsize=fontsize, alpha=alpha)


def main() -> None:
    state = torch.load(ARTIFACTS / "interpolation_flipbook.pt", weights_only=False)
    pw = torch.load(ARTIFACTS / "pairwise_and_hotdims.pt", weights_only=False)
    labels: dict[int, str] = pw["labels"]
    general_dims = sorted([idx for idx, lbl in labels.items() if lbl == "feature"])
    print(f"general-content feature dims: {general_dims}")

    h_A: torch.Tensor = state["h_A"]
    h_B: torch.Tensor = state["h_B"]
    steps = sorted(state["steps"], key=lambda s: s["step"])
    n_steps = len(steps)
    print(f"loaded {n_steps} interpolation steps")
    print(f"anchor A: {state['anchor_A_text']!r}")
    print(f"anchor B: {state['anchor_B_text']!r}")
    print(f"cos(h_A, h_B) = {torch.nn.functional.cosine_similarity(h_A, h_B, dim=0).item():+.4f}")

    # Interpolation-specific glyph dims: top-8 by |h_A - h_B|
    diff_AB = (h_A - h_B).abs()
    interp_dims = sorted(diff_AB.topk(8).indices.tolist())
    print(f"interp-specific glyph dims (top-8 by |h_A - h_B|): {interp_dims}")

    # Build the full sequence: anchor A, all interp steps in order, anchor B
    # NOTE: t=0 already corresponds to step 0 (h = h_A), and t=1 to step n_steps-1 (h = h_B).
    # So we don't need to add anchor rows separately.
    row_h: list[tuple[str, float, torch.Tensor, str]] = []
    for s in steps:
        row_h.append((f"step {s['step']}", s["t"], s["h_t"], s["av_text"]))

    # Glyph normalizers
    abs_max_general = float(
        max(np.abs(rh[2][general_dims].numpy()).max() for rh in row_h)
    )
    abs_max_interp = float(
        max(np.abs(rh[2][interp_dims].numpy()).max() for rh in row_h)
    )
    print(f"abs_max general-dim: {abs_max_general:.2f}")
    print(f"abs_max interp-dim:  {abs_max_interp:.2f}")

    # --- fig17: vertical strip flipbook ---
    # Reserve top TITLE_RESERVE fraction for suptitle so first row doesn't overlap.
    TITLE_RESERVE = 0.04
    n_rows = len(row_h)
    fig = plt.figure(figsize=(20, 1.8 * n_rows))
    for j, (lbl, t, h_t, av_text) in enumerate(row_h):
        row_height = (1.0 - TITLE_RESERVE) / n_rows
        row_y = (1.0 - TITLE_RESERVE) - (j + 1) * row_height

        # col 1: t-marker
        ax_t = fig.add_axes((0.01, row_y, 0.07, row_height))
        ax_t.axis("off")
        # color gradient: blue (t=0) to orange (t=1)
        bar_color = (1 - t) * np.array([0.12, 0.46, 0.71]) + t * np.array([1.00, 0.50, 0.05])
        ax_t.add_patch(plt.Rectangle((0.0, 0.1), t, 0.8, color=bar_color, alpha=0.85))
        ax_t.set_xlim(0, 1)
        ax_t.set_ylim(0, 1)
        ax_t.text(0.5, 0.5, f"{lbl}\nt={t:.3f}", ha="center", va="center",
                  fontsize=10, family="monospace", weight="bold")

        # col 2: general-content glyph
        ax_g1 = fig.add_axes((0.09, row_y + 0.005, 0.10, row_height - 0.01))
        ax_g1.set_xlim(-1.3, 1.3)
        ax_g1.set_ylim(-1.3, 1.3)
        ax_g1.set_aspect("equal")
        ax_g1.axis("off")
        vec_g = h_t[general_dims].numpy()
        draw_signature_glyph(ax_g1, 0.0, 0.0, vec_g, radius=1.0, scale_max=abs_max_general, lw=2.5)
        label_glyph_axes(ax_g1, general_dims, fontsize=5)
        if j == 0:
            ax_g1.set_title("general-content glyph\n(feature dims)", fontsize=8, pad=2)

        # col 3: interp-specific glyph
        ax_g2 = fig.add_axes((0.20, row_y + 0.005, 0.10, row_height - 0.01))
        ax_g2.set_xlim(-1.3, 1.3)
        ax_g2.set_ylim(-1.3, 1.3)
        ax_g2.set_aspect("equal")
        ax_g2.axis("off")
        vec_i = h_t[interp_dims].numpy()
        draw_signature_glyph(ax_g2, 0.0, 0.0, vec_i, radius=1.0, scale_max=abs_max_interp, lw=2.5)
        label_glyph_axes(ax_g2, interp_dims, fontsize=5)
        if j == 0:
            ax_g2.set_title(f"interpolation-specific glyph\n(top-8 by |h_A - h_B|)", fontsize=8, pad=2)

        # col 4: AV text wrapped
        ax_txt = fig.add_axes((0.32, row_y + 0.005, 0.66, row_height - 0.01))
        ax_txt.axis("off")
        paragraphs = [p.strip() for p in av_text.split("\n\n") if p.strip()]
        wrapped = "\n\n".join(
            textwrap.fill(p, width=140, break_long_words=False, break_on_hyphens=False)
            for p in paragraphs
        )
        ax_txt.text(0.0, 0.5, wrapped, ha="left", va="center", fontsize=8,
                    family="serif")

    fig.suptitle(
        f"fig17 — interpolation flipbook:  AR( '{state['anchor_A_label']}' ) → AR( '{state['anchor_B_label']}' )\n"
        f"{n_steps} steps, h_t = (1-t)·h_A + t·h_B, AV-decoded at each step  |  cos(h_A, h_B) = "
        f"{torch.nn.functional.cosine_similarity(h_A, h_B, dim=0).item():+.3f}",
        fontsize=12, y=0.995)
    fig.savefig(FIGDIR / "fig17_interp_flipbook.png", dpi=180)
    plt.close(fig)
    print(f"\nwrote {FIGDIR}/fig17_interp_flipbook.png")

    # --- fig18: diagnostic plots ---
    ts = np.array([s["t"] for s in steps])
    h_ts = torch.stack([s["h_t"] for s in steps])  # (n_steps, 3584)
    cos_A = torch.nn.functional.cosine_similarity(h_ts, h_A.unsqueeze(0).expand_as(h_ts), dim=1).numpy()
    cos_B = torch.nn.functional.cosine_similarity(h_ts, h_B.unsqueeze(0).expand_as(h_ts), dim=1).numpy()
    norms = h_ts.norm(dim=1).numpy()
    dist_from_A = (h_ts - h_A.unsqueeze(0)).norm(dim=1).numpy()
    if n_steps > 1:
        step_dists = (h_ts[1:] - h_ts[:-1]).norm(dim=1).numpy()
    else:
        step_dists = np.array([])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(ts, cos_A, "o-", color="#1f77b4", label="cos(h_t, h_A)")
    ax.plot(ts, cos_B, "o-", color="#ff7f0e", label="cos(h_t, h_B)")
    ax.set_xlabel("t")
    ax.set_ylabel("cosine")
    ax.set_title("cos(h_t, anchor) — must cross at midpoint for linear interp")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(ts, norms, "o-", color="#2ca02c")
    ax.axhline(h_A.norm().item(), color="#1f77b4", linestyle="--", linewidth=0.8, alpha=0.6, label="||h_A||")
    ax.axhline(h_B.norm().item(), color="#ff7f0e", linestyle="--", linewidth=0.8, alpha=0.6, label="||h_B||")
    ax.set_xlabel("t")
    ax.set_ylabel("||h_t||")
    ax.set_title("||h_t|| over t (dip at midpoint if h_A, h_B point apart)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(ts, dist_from_A, "o-", color="#d62728")
    ax.set_xlabel("t")
    ax.set_ylabel("||h_t - h_A||")
    ax.set_title("Distance from anchor A — should be perfectly linear")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(step_dists) > 0:
        ts_mid = (ts[1:] + ts[:-1]) / 2
        ax.plot(ts_mid, step_dists, "o-", color="#9467bd")
        ax.set_xlabel("t (midpoint of step)")
        ax.set_ylabel("||h_t - h_(t-1)||")
        ax.set_title("Per-step distance — uniform for linear interp (sanity check)")
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")

    fig.suptitle(
        f"fig18 — interpolation diagnostic:  anchor distance "
        f"||h_A - h_B|| = {(h_A - h_B).norm().item():.2f}, cos = "
        f"{torch.nn.functional.cosine_similarity(h_A, h_B, dim=0).item():+.3f}",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig18_interp_diagnostic.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig18_interp_diagnostic.png")


if __name__ == "__main__":
    main()
