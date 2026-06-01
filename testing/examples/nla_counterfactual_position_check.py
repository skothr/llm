"""fig16: position-matched correction to fig15.

fig15 picked the LAST forced token for each pair as the representative,
which for refusal_metaware was ' refuse' at abs_pos 51 vs natural '4' at
abs_pos 41 — a 10-token position drift that inflates ||Δh||_feat. This
script picks the position-CLOSEST forced token for each pair, and
additionally renders all three refusal_metaware variants to show
||Δh||_feat's sensitivity to position drift.

Outputs fig16_counterfactual_position_check.png.
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

from _nla_artifacts import FIGURES as FIGDIR, load_artifact

FIGDIR.mkdir(parents=True, exist_ok=True)


def draw_signature_glyph(
    ax: Any,
    cx: float,
    cy: float,
    values: np.ndarray,
    radius: float,
    scale_max: float,
    lw: float = 2.5,
) -> None:
    n = len(values)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for i, v in enumerate(values):
        mag = min(abs(v) / scale_max, 1.0)
        if mag < 0.02:
            continue
        x_end = cx + radius * mag * np.cos(angles[i])
        y_end = cy + radius * mag * np.sin(angles[i])
        color = "#d62728" if v >= 0 else "#1f77b4"
        ax.plot(
            [cx, x_end],
            [cy, y_end],
            color=color,
            linewidth=lw,
            alpha=0.9,
            solid_capstyle="round",
        )


def label_glyph_axes(ax: Any, feature_dims: list[int]) -> None:
    angles = np.linspace(0, 2 * np.pi, len(feature_dims), endpoint=False)
    for ang, d in zip(angles, feature_dims):
        x_lbl = 1.18 * np.cos(ang)
        y_lbl = 1.18 * np.sin(ang)
        ax.text(x_lbl, y_lbl, f"d{d}", ha="center", va="center", fontsize=7, alpha=0.6)


def main() -> None:
    forced = load_artifact("forced_continuation.pt")
    pw = load_artifact("pairwise_and_hotdims.pt")
    labels: dict[int, str] = pw["labels"]
    feature_dims = sorted([idx for idx, lbl in labels.items() if lbl == "feature"])

    # natural cap per pair (each has exactly 1)
    nat: dict[str, dict[str, Any]] = {}
    for c in forced["captures"]:
        if c["version"] == "natural":
            nat[c["pair_id"]] = c

    # Build the row list: (label, natural_cap, forced_cap)
    # For pairs with multiple forced tokens, pick position-closest as primary
    # and additionally render the others for refusal_metaware to show drift effect
    rows: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []

    # singletons first
    forced_caps_by_pair: dict[str, list[dict[str, Any]]] = {}
    for c in forced["captures"]:
        if c["version"] == "forced":
            forced_caps_by_pair.setdefault(c["pair_id"], []).append(c)

    for pid in ("negation", "factual", "math"):
        n = nat[pid]
        f_caps = forced_caps_by_pair[pid]
        f = min(f_caps, key=lambda c: abs(c["abs_pos"] - n["abs_pos"]))
        rows.append((pid, n, f, f"Δpos={f['abs_pos'] - n['abs_pos']}"))

    # refusal_metaware: show all 3 forced tokens
    n = nat["refusal_metaware"]
    for f in sorted(
        forced_caps_by_pair["refusal_metaware"], key=lambda c: c["abs_pos"]
    ):
        dpos = f["abs_pos"] - n["abs_pos"]
        rows.append(("refusal_metaware", n, f, f"Δpos={dpos:+d}"))

    # global normalizers
    all_glyphs = []
    for _, n, f, _ in rows:
        all_glyphs.append(n["h"][feature_dims].numpy())
        all_glyphs.append(f["h"][feature_dims].numpy())
    abs_max = float(np.abs(np.concatenate(all_glyphs)).max())

    all_diffs = []
    for _, n, f, _ in rows:
        all_diffs.append((f["h"] - n["h"])[feature_dims].numpy())
    abs_max_diff = float(np.abs(np.concatenate(all_diffs)).max())

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 2.7 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (pid, n_cap, f_cap, dpos_str) in enumerate(rows):
        pair_meta = next(p for p in forced["pairs"] if p["id"] == pid)
        diff = (f_cap["h"] - n_cap["h"])[feature_dims].numpy()
        diff_norm_feat = float(np.linalg.norm(diff))
        diff_norm_full = float((f_cap["h"] - n_cap["h"]).norm().item())

        cols = [
            (
                f"natural  '{n_cap['actual_token']}'  pos={n_cap['abs_pos']}",
                n_cap["h"][feature_dims].numpy(),
                abs_max,
            ),
            (
                f"forced   '{f_cap['actual_token']}'  pos={f_cap['abs_pos']}  ({dpos_str})",
                f_cap["h"][feature_dims].numpy(),
                abs_max,
            ),
            (
                f"diff (||·||_feat = {diff_norm_feat:.2f}, full = {diff_norm_full:.1f})",
                diff,
                abs_max_diff,
            ),
        ]
        for c, (title, vec, normalizer) in enumerate(cols):
            ax = axes[r, c]
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect("equal")
            draw_signature_glyph(ax, 0.0, 0.0, vec, radius=1.0, scale_max=normalizer)
            label_glyph_axes(ax, feature_dims)
            ax.axis("off")
            ax.set_title(title, fontsize=9)
            if c == 0:
                ax.text(
                    -1.6,
                    1.4,
                    f"{pid}\n{pair_meta['prompt']!r}",
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "lightyellow",
                        "edgecolor": "gray",
                    },
                )

    fig.suptitle(
        f"fig16: position-matched counterfactual diff (corrects fig15)\n"
        f"For refusal_metaware, all 3 forced tokens shown to expose ||Δh||_feat sensitivity to position drift\n"
        f"feature dims = {feature_dims}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig16_counterfactual_position_check.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig16_counterfactual_position_check.png")

    # Summary table to stdout
    print(
        f"\n{'pair':<20} {'forced_token':>14} {'Δpos':>5}  {'||Δh||_feat':>12}  {'||Δh||_full':>12}"
    )
    print("-" * 80)
    for _, n_cap, f_cap, _ in rows:
        diff = f_cap["h"] - n_cap["h"]
        print(
            f"{f_cap['pair_id']:<20} {repr(f_cap['actual_token']):>14} {f_cap['abs_pos'] - n_cap['abs_pos']:>+5}  "
            f"{float(diff[feature_dims].norm().item()):>12.2f}  {float(diff.norm().item()):>12.2f}"
        )


if __name__ == "__main__":
    main()
