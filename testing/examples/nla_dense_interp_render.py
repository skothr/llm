"""Render fig36 + fig37 for MAIN-34 dense interpolation near t=0.421.

fig36: flipbook-style strip with t-bar + AV-text per row.
       Dense-zone rows highlighted (yellow background).
fig37: diagnostic — top-3 vocab anchor per step heatmap + ||h_t|| over t
       + step-to-step glyph distance (do we see a jump near t=0.421?).
"""

import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = _REPO_ROOT / "testing" / ".cache" / "nla_artifacts"
FIGDIR = _REPO_ROOT / "research" / "arcs" / "nla-verbalizer" / "observations" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    flip = torch.load(ARTIFACTS / "dense_interp_near_pivot.pt", weights_only=False)
    vocab = torch.load(ARTIFACTS / "vocab_atlas.pt", weights_only=False)
    pw = torch.load(ARTIFACTS / "pairwise_and_hotdims.pt", weights_only=False)

    labels: dict[int, str] = pw["labels"]
    sink_dims = sorted([idx for idx, lbl in labels.items() if lbl == "sink"])
    steps = sorted(flip["steps"], key=lambda s: s["t"])
    n_steps = len(steps)
    dense_zone = flip["dense_zone"]
    print(f"loaded {n_steps} interpolated steps, dense zone {dense_zone}")

    # Build sink-removed vocab matrix for nearest-anchor lookup
    v_caps: list[dict[str, Any]] = vocab["captures"]
    H_v = torch.stack([c["h"] for c in v_caps]).clone()
    for d in sink_dims:
        H_v[:, d] = 0.0
    H_v_unit = torch.nn.functional.normalize(H_v, dim=1)
    v_words = [c["word"] for c in v_caps]

    # Per-step nearest-anchor + projection-on-anchors
    rows: list[dict[str, Any]] = []
    for s in steps:
        h = s["h_t"].clone()
        for d in sink_dims:
            h[d] = 0.0
        h_unit = h / (h.norm() + 1e-9)
        cos_to_anchors = (H_v_unit @ h_unit).numpy()
        top3 = np.argsort(-cos_to_anchors)[:3]
        rows.append({
            **s,
            "top3_words": [v_words[i] for i in top3],
            "top3_cos": [float(cos_to_anchors[i]) for i in top3],
            "cos_to_anchors": cos_to_anchors,
        })

    # ---- fig36: flipbook strip ----
    TITLE_RESERVE = 0.025
    fig = plt.figure(figsize=(20, 0.60 * n_steps + 1))
    row_h_frac = (1.0 - TITLE_RESERVE) / n_steps

    for j, r in enumerate(rows):
        row_y = (1.0 - TITLE_RESERVE) - (j + 1) * row_h_frac
        t = r["t"]
        in_dense = dense_zone[0] <= t <= dense_zone[1]
        bg_color = "#fff5cc" if in_dense else "#ffffff"

        # t-marker column with background highlight
        ax_t = fig.add_axes((0.01, row_y, 0.10, row_h_frac))
        ax_t.axis("off")
        ax_t.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0,
                                       transform=ax_t.transAxes,
                                       facecolor=bg_color, edgecolor="none"))
        bar_color = (1 - t) * np.array([0.12, 0.46, 0.71]) + t * np.array([1.00, 0.50, 0.05])
        ax_t.add_patch(plt.Rectangle((0.0, 0.35), t, 0.30, color=bar_color, alpha=0.85))
        ax_t.set_xlim(0, 1)
        ax_t.set_ylim(0, 1)
        ax_t.text(0.5, 0.5, f"t = {t:.4f}", ha="center", va="center",
                  fontsize=8, family="monospace",
                  weight="bold" if in_dense else "normal")

        # top-3 anchors
        ax_n = fig.add_axes((0.12, row_y, 0.16, row_h_frac))
        ax_n.axis("off")
        ax_n.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0,
                                       transform=ax_n.transAxes,
                                       facecolor=bg_color, edgecolor="none"))
        labels_top3 = "  ".join(
            f"{w}({c:+.2f})" for w, c in zip(r["top3_words"], r["top3_cos"])
        )
        ax_n.text(0.02, 0.5, labels_top3, ha="left", va="center",
                  fontsize=7, family="monospace")

        # AV text
        ax_v = fig.add_axes((0.29, row_y, 0.69, row_h_frac))
        ax_v.axis("off")
        ax_v.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0,
                                       transform=ax_v.transAxes,
                                       facecolor=bg_color, edgecolor="none"))
        first_para = (r["av_text"].split("\n\n") or [""])[0]
        wrapped = textwrap.fill(first_para, width=160, break_long_words=False,
                                  break_on_hyphens=False)
        ax_v.text(0.005, 0.5, wrapped, ha="left", va="center",
                  fontsize=7, family="serif")

    fig.suptitle(
        f"fig36 — Dense interpolation near t=0.421 ({n_steps} steps, "
        f"dense zone [{dense_zone[0]}, {dense_zone[1]}] highlighted)\n"
        f"AR({flip['anchor_A_label']}) → AR({flip['anchor_B_label']}). "
        f"Top-3 vocab anchors per step (sink-removed cosine) + AV-decode first-paragraph.",
        fontsize=10, y=0.995,
    )
    fig.savefig(FIGDIR / "fig36_dense_interp_flipbook.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig36_dense_interp_flipbook.png")

    # ---- fig37: diagnostic plots ----
    ts = np.array([r["t"] for r in rows])
    h_ts = torch.stack([r["h_t"] for r in rows])
    norms = h_ts.norm(dim=1).numpy()
    if n_steps > 1:
        step_dists = (h_ts[1:] - h_ts[:-1]).norm(dim=1).numpy()
        ts_mid = (ts[1:] + ts[:-1]) / 2
    else:
        step_dists = np.array([])
        ts_mid = np.array([])

    # Top-1 anchor word per step (build category-flip series)
    top1_words = [r["top3_words"][0] for r in rows]
    # Find indices where top-1 changes
    flip_indices = [i for i in range(1, len(top1_words))
                     if top1_words[i] != top1_words[i - 1]]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1: top-1 anchor cosine + word
    ax = axes[0]
    top1_cos = np.array([r["top3_cos"][0] for r in rows])
    ax.plot(ts, top1_cos, "o-", color="#1f77b4", linewidth=1.2)
    # annotate word transitions
    prev_word = top1_words[0] if top1_words else ""
    for i, w in enumerate(top1_words):
        if i == 0 or w != prev_word:
            ax.annotate(w, (ts[i], top1_cos[i]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=7, ha="center", rotation=0)
            prev_word = w
    ax.axvspan(dense_zone[0], dense_zone[1], color="#fff5cc", alpha=0.7,
               label=f"dense zone [{dense_zone[0]}, {dense_zone[1]}]")
    for fi in flip_indices:
        ax.axvline(ts[fi], color="#d62728", linewidth=0.6, alpha=0.7, linestyle=":")
    ax.set_ylabel("top-1 anchor cosine")
    ax.set_title(
        f"fig37(top) — Top-1 nearest vocab anchor per step. "
        f"Word-flip lines (red dotted) mark where the nearest anchor changed.",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # 2: ||h_t|| dip pattern
    ax = axes[1]
    ax.plot(ts, norms, "o-", color="#2ca02c", linewidth=1.2)
    ax.axvspan(dense_zone[0], dense_zone[1], color="#fff5cc", alpha=0.7)
    ax.set_ylabel("||h_t||")
    ax.set_title("fig37(mid) — ||h_t|| over t (should dip at midpoint for "
                 "anti-parallel anchors)", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3: step-to-step ||Δh_t||
    ax = axes[2]
    if len(step_dists) > 0:
        ax.plot(ts_mid, step_dists, "o-", color="#9467bd", linewidth=1.2)
    ax.axvspan(dense_zone[0], dense_zone[1], color="#fff5cc", alpha=0.7)
    ax.set_xlabel("t")
    ax.set_ylabel("||h_t - h_(t-1)||")
    ax.set_title("fig37(bottom) — Per-step distance. Constant = linear interp "
                 "(sanity check). Spike = artifact of resolution change.",
                 fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"fig37 — Dense interpolation diagnostic (n_steps={n_steps}, "
        f"dense Δt ≈ 0.0025, sparse Δt ≈ 0.25). "
        f"Word-flip count: {len(flip_indices)} transitions.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig37_dense_interp_diagnostic.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig37_dense_interp_diagnostic.png")

    # Print word-flip summary
    print(f"\nWord-flip transitions ({len(flip_indices)} total):")
    for fi in flip_indices:
        t_before = rows[fi - 1]["t"]
        t_after = rows[fi]["t"]
        print(f"  {top1_words[fi - 1]!r} → {top1_words[fi]!r}  "
              f"between t={t_before:.4f} and t={t_after:.4f}  "
              f"(Δt = {t_after - t_before:.4f})")


if __name__ == "__main__":
    main()
