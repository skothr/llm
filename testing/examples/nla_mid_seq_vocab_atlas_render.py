"""Render fig31 + fig32 for MAIN-44 mid-sequence vocab atlas comparison.

fig31 - side-by-side bar chart per category: signal (within-class
        projection) for end-of-prompt vs mid-sequence + max-noise overlay.
fig32 - confusion-like table: argmax-accuracy heatmap, end-of-prompt vs
        mid-sequence.
"""

from pathlib import Path

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
    cmp = torch.load(ARTIFACTS / "mid_seq_compare.pt", weights_only=False)
    rows = cmp["rows"]
    categories = cmp["categories"]

    eop_signal = np.array([r["eop_signal_mean"] for r in rows])
    mid_signal = np.array([r["mid_signal_mean"] for r in rows])
    eop_signal_std = np.array([r["eop_signal_std"] for r in rows])
    mid_signal_std = np.array([r["mid_signal_std"] for r in rows])
    eop_noise = np.array([r["eop_noise_max_mean"] for r in rows])
    mid_noise = np.array([r["mid_noise_max_mean"] for r in rows])
    eop_acc = np.array([r["eop_argmax_acc"] for r in rows])
    mid_acc = np.array([r["mid_argmax_acc"] for r in rows])

    # ---- fig31: signal + noise bars per category ----
    n_cat = len(categories)
    x = np.arange(n_cat)
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
    ax1 = axes[0]
    ax2 = axes[1]

    ax1.bar(x - width / 2, eop_signal, width, yerr=eop_signal_std, capsize=2,
            color="#1f77b4", alpha=0.85, label="end-of-prompt (baseline)")
    ax1.bar(x + width / 2, mid_signal, width, yerr=mid_signal_std, capsize=2,
            color="#ff7f0e", alpha=0.85, label="mid-sequence (this test)")
    ax1.set_ylabel("within-class signal: mean cos(h_a, d_cat)")
    ax1.set_title("fig31(top) — within-class discriminant signal per category, by capture protocol\n"
                  "Higher = the basis fires correctly when anchor is in category. "
                  "If mid-seq token-presence detection works, orange should rise sharply over blue.",
                  fontsize=10)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    ax2.bar(x - width / 2, eop_noise, width, color="#aec7e8", alpha=0.85,
            label="end-of-prompt max off-class")
    ax2.bar(x + width / 2, mid_noise, width, color="#ffbb78", alpha=0.85,
            label="mid-sequence max off-class")
    ax2.set_ylabel("max off-class projection (noise floor)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=55, ha="right", fontsize=8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_title("fig31(bottom) — max off-class projection per category. "
                  "Comparable height to top panel = basis is not discriminating.",
                  fontsize=10)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig31_mid_seq_signal_vs_noise.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig31_mid_seq_signal_vs_noise.png")

    # ---- fig32: argmax accuracy comparison ----
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(x - width / 2, eop_acc, width, color="#1f77b4", alpha=0.85,
           label=f"end-of-prompt (mean {eop_acc.mean():.1%})")
    ax.bar(x + width / 2, mid_acc, width, color="#ff7f0e", alpha=0.85,
           label=f"mid-sequence (mean {mid_acc.mean():.1%})")
    ax.set_ylabel("argmax-category accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=55, ha="right", fontsize=8)
    ax.axhline(1 / n_cat, color="gray", linestyle="--", linewidth=0.8,
               label=f"random = 1/{n_cat} = {1 / n_cat:.1%}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("fig32 — Anchor->category classification accuracy by capture protocol\n"
                 "argmax over 23 discriminants; correct if argmax == true category. "
                 "End-of-prompt is high by construction (discriminants derived from it).",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig32_mid_seq_argmax_accuracy.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig32_mid_seq_argmax_accuracy.png")


if __name__ == "__main__":
    main()
