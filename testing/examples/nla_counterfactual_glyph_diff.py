"""fig15: counterfactual glyph diff for forced-continuation pairs.

For each of the 4 forced-continuation pairs, render a row of three
glyphs:
  * natural glyph    (h captured at the natural continuation token)
  * forced glyph     (h captured at the forced continuation token)
  * difference glyph (forced - natural, in the same feature subspace)

Reads forced_continuation.pt + pairwise_and_hotdims.pt. For pairs
with multiple forced captures (refusal_metaware has 3 — sensing/test/
refuse), use the final token of the forced completion.
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
FIGDIR = _REPO_ROOT / "research" / "observations" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def draw_signature_glyph(ax: Any, cx: float, cy: float, values: np.ndarray,
                          radius: float, scale_max: float, lw: float = 2.5) -> None:
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


def label_glyph_axes(ax: Any, feature_dims: list[int]) -> None:
    angles = np.linspace(0, 2 * np.pi, len(feature_dims), endpoint=False)
    for ang, d in zip(angles, feature_dims):
        x_lbl = 1.18 * np.cos(ang)
        y_lbl = 1.18 * np.sin(ang)
        ax.text(x_lbl, y_lbl, f"d{d}", ha="center", va="center", fontsize=7, alpha=0.6)


def main() -> None:
    forced = torch.load(ARTIFACTS / "forced_continuation.pt", weights_only=False)
    pw = torch.load(ARTIFACTS / "pairwise_and_hotdims.pt", weights_only=False)
    labels: dict[int, str] = pw["labels"]
    feature_dims = sorted([idx for idx, lbl in labels.items() if lbl == "feature"])
    print(f"feature dims: {feature_dims}")

    # Group captures by pair, pick final-token capture per version
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for cap in forced["captures"]:
        pid = cap["pair_id"]
        ver = cap["version"]
        by_pair.setdefault(pid, {})
        # if multiple captures per version, keep the highest abs_pos (final token)
        existing = by_pair[pid].get(ver)
        if existing is None or cap["abs_pos"] > existing["abs_pos"]:
            by_pair[pid][ver] = cap

    pair_ids = list(by_pair.keys())
    print(f"\npairs: {pair_ids}")
    for pid in pair_ids:
        nat = by_pair[pid].get("natural")
        forc = by_pair[pid].get("forced")
        if nat and forc:
            print(f"  {pid:>20}  natural={nat['actual_token']!r:>14}  forced={forc['actual_token']!r:>14}")

    # global normalizer
    all_feat: list[np.ndarray] = []
    for pid in pair_ids:
        for ver in ("natural", "forced"):
            cap = by_pair[pid].get(ver)
            if cap is not None:
                all_feat.append(cap["h"][feature_dims].numpy())
    abs_max = float(np.abs(np.concatenate(all_feat)).max())
    print(f"abs_max (natural+forced glyphs): {abs_max:.2f}")

    # also need diff normalizer (differences are smaller)
    all_diffs: list[np.ndarray] = []
    for pid in pair_ids:
        nat = by_pair[pid].get("natural")
        forc = by_pair[pid].get("forced")
        if nat and forc:
            diff = (forc["h"] - nat["h"])[feature_dims].numpy()
            all_diffs.append(diff)
    abs_max_diff = float(np.abs(np.concatenate(all_diffs)).max()) if all_diffs else 1.0
    print(f"abs_max (forced-natural diff): {abs_max_diff:.2f}")

    n_pairs = len(pair_ids)
    fig, axes = plt.subplots(n_pairs, 3, figsize=(13, 3.2 * n_pairs))
    if n_pairs == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, pid in enumerate(pair_ids):
        nat = by_pair[pid].get("natural")
        forc = by_pair[pid].get("forced")
        if nat is None or forc is None:
            continue
        pair_meta = next(p for p in forced["pairs"] if p["id"] == pid)

        for c, (title_suffix, vec, normalizer, cap) in enumerate([
            (f"natural  '{nat['actual_token']}'", nat["h"][feature_dims].numpy(), abs_max, nat),
            (f"forced   '{forc['actual_token']}'", forc["h"][feature_dims].numpy(), abs_max, forc),
            (f"forced - natural diff", (forc["h"] - nat["h"])[feature_dims].numpy(), abs_max_diff, None),
        ]):
            ax = axes[r, c]
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect("equal")
            draw_signature_glyph(ax, 0.0, 0.0, vec, radius=1.0, scale_max=normalizer)
            label_glyph_axes(ax, feature_dims)
            ax.axis("off")
            if c == 0:
                ax.text(-1.6, 1.4, f"{pid}\n{pair_meta['prompt']!r}",
                        ha="left", va="top", fontsize=9,
                        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "edgecolor": "gray"})
            ax.set_title(title_suffix, fontsize=10)
            if c == 2:
                # also note the diff norm
                diff_norm = float(np.linalg.norm(vec))
                ax.text(0.0, -1.45, f"||diff||_feat = {diff_norm:.2f}",
                        ha="center", va="top", fontsize=8, alpha=0.7)

    fig.suptitle(f"Counterfactual glyph diff: forced - natural in the feature-dim subspace\n"
                 f"glyph rays = {feature_dims} | red = +, blue = - | "
                 f"natural/forced scaled to ||*||≤{abs_max:.1f}; diff scaled to ||*||≤{abs_max_diff:.1f}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig15_counterfactual_diff.png", dpi=180)
    plt.close(fig)
    print(f"\nwrote {FIGDIR}/fig15_counterfactual_diff.png")


if __name__ == "__main__":
    main()
