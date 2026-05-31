"""MAIN-70 — Mid-seq native discriminants + cross-protocol axis stability.

Builds discriminants from BOTH protocols and projects each protocol's
captures onto each discriminant family. Tests whether the within-protocol
signal lifts at mid-seq when discriminants are mid-seq-native (predict
~+0.4, matching eop in-protocol). Then computes the 23×23 cross-protocol
discriminant cosine matrix — diagonal entries quantify how stable each
category's axis is across capture position.

Outputs:
  testing/.cache/nla_artifacts/mid_seq_native_compare.pt   (artifact)
  research/arcs/nla-verbalizer/observations/figures/fig33_native_signal_lift.png
  research/arcs/nla-verbalizer/observations/figures/fig34_cross_protocol_axis_cos.png
"""

import statistics
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _nla_artifacts import FIGURES as FIGDIR, load_artifact, write_artifact
from nla_discriminant_glyph import (
    apply_sink_removal,
    compute_discriminant_dirs,
    project_to_discriminants,
)


FIGDIR.mkdir(parents=True, exist_ok=True)


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def _aggregate_signal(
    captures: list[dict[str, Any]], discr: dict[str, torch.Tensor], sink_dims: list[int]
) -> tuple[float, float, dict[str, float]]:
    """Per-category-mean within-class signal + argmax accuracy + per-cat dict."""
    by_cat_sig: dict[str, list[float]] = {}
    by_cat_correct: dict[str, list[bool]] = {}
    for cap in captures:
        h = cap["h"]
        projs = project_to_discriminants(h, discr, sink_dims)
        sig = projs[cap["category"]]
        by_cat_sig.setdefault(cap["category"], []).append(sig)
        is_correct = max(projs.items(), key=lambda x: x[1])[0] == cap["category"]
        by_cat_correct.setdefault(cap["category"], []).append(is_correct)
    per_cat_sig: dict[str, float] = {c: _mean(vs) for c, vs in by_cat_sig.items()}
    sig_agg = _mean(list(per_cat_sig.values()))
    acc_agg = _mean(
        [_mean([1.0 if b else 0.0 for b in vs]) for vs in by_cat_correct.values()]
    )
    return sig_agg, acc_agg, per_cat_sig


def main() -> None:
    eop = load_artifact("vocab_atlas.pt")
    mid = load_artifact("mid_seq_vocab_atlas.pt")
    pw = load_artifact("pairwise_and_hotdims.pt")

    labels: dict[int, str] = pw["labels"]
    sink_dims = sorted([idx for idx, lbl in labels.items() if lbl == "sink"])
    categories = list(eop["categories"])
    assert list(mid["categories"]) == categories, "category lists must match"

    discr_eop = compute_discriminant_dirs(eop, sink_dims)
    discr_mid = compute_discriminant_dirs(mid, sink_dims)
    print(f"computed discriminants: eop x{len(discr_eop)}, mid x{len(discr_mid)}")

    # 4-cell signal+accuracy: rows = captures, cols = discriminants
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for cap_name, cap_src in [("eop", eop["captures"]), ("mid", mid["captures"])]:
        for d_name, d_src in [("eop", discr_eop), ("mid", discr_mid)]:
            sig, acc, per_cat = _aggregate_signal(cap_src, d_src, sink_dims)
            cells[(cap_name, d_name)] = {
                "agg_signal": sig,
                "agg_acc": acc,
                "per_cat_signal": per_cat,
            }
            print(
                f"  captures={cap_name:>3}  discr={d_name:>3}  "
                f"agg_signal={sig:+.4f}  agg_acc={acc:.2%}"
            )

    # Cross-protocol axis cosine
    D_eop = torch.stack([discr_eop[c] for c in categories])
    D_mid = torch.stack([discr_mid[c] for c in categories])
    cross_cos = (D_eop @ D_mid.T).numpy()  # (23, 23)
    diag_cos = np.diag(cross_cos)
    print(f"\nCross-protocol axis cosines (diagonal = same-category):")
    print(f"  mean diagonal: {diag_cos.mean():+.4f}")
    print(
        f"  min  diagonal: {diag_cos.min():+.4f}  ({categories[int(diag_cos.argmin())]})"
    )
    print(
        f"  max  diagonal: {diag_cos.max():+.4f}  ({categories[int(diag_cos.argmax())]})"
    )
    print(
        f"  mean off-diagonal: "
        f"{cross_cos[~np.eye(len(categories), dtype=bool)].mean():+.4f}"
    )

    # Print top-5 most-stable and least-stable axes
    print(f"\nMost-stable axes (eop axis ≈ mid axis):")
    for i in np.argsort(-diag_cos)[:5]:
        print(f"  {categories[i]:<14} cos={diag_cos[i]:+.4f}")
    print(f"\nLeast-stable axes:")
    for i in np.argsort(diag_cos)[:5]:
        print(f"  {categories[i]:<14} cos={diag_cos[i]:+.4f}")

    # ---- fig33: per-category in-protocol signal lift ----
    n_cat = len(categories)
    x = np.arange(n_cat)
    width = 0.4
    eop_native = np.array(
        [cells[("eop", "eop")]["per_cat_signal"][c] for c in categories]
    )
    mid_native = np.array(
        [cells[("mid", "mid")]["per_cat_signal"][c] for c in categories]
    )
    mid_cross = np.array(
        [cells[("mid", "eop")]["per_cat_signal"][c] for c in categories]
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(
        x - width,
        eop_native,
        width,
        color="#1f77b4",
        alpha=0.85,
        label=f"eop-h × eop-discr (in-protocol, mean +{eop_native.mean():.3f})",
    )
    ax.bar(
        x,
        mid_native,
        width,
        color="#2ca02c",
        alpha=0.85,
        label=f"mid-h × mid-discr (in-protocol, mean +{mid_native.mean():.3f})",
    )
    ax.bar(
        x + width,
        mid_cross,
        width,
        color="#ff7f0e",
        alpha=0.85,
        label=f"mid-h × eop-discr (cross-protocol, mean +{mid_cross.mean():.3f})",
    )
    ax.set_ylabel("within-class signal: mean cos(h, d_cat)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=55, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "fig33 — In-protocol vs cross-protocol within-class signal\n"
        "If mid-native bars (green) match eop-native (blue), discriminants are "
        "protocol-coupled by construction.\nOrange shows MAIN-44's collapsed "
        "cross-protocol projection (mid-h projected onto eop-derived axes).",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig33_native_signal_lift.png", dpi=180)
    plt.close(fig)
    print(f"\nwrote {FIGDIR}/fig33_native_signal_lift.png")

    # ---- fig34: cross-protocol axis cosine heatmap ----
    fig, ax = plt.subplots(figsize=(13, 11))
    vmax = float(np.abs(cross_cos).max())
    im = ax.imshow(cross_cos, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_cat))
    ax.set_yticks(range(n_cat))
    ax.set_xticklabels(categories, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(categories, fontsize=8)
    ax.set_xlabel("mid-sequence discriminant axis (d_mid)")
    ax.set_ylabel("end-of-prompt discriminant axis (d_eop)")
    # Annotate diagonal with values
    for i in range(n_cat):
        ax.text(
            i,
            i,
            f"{cross_cos[i, i]:+.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "none", "pad": 0.5},
        )
    fig.colorbar(im, ax=ax, label="cos(d_eop_C, d_mid_D)")
    ax.set_title(
        f"fig34 — Cross-protocol discriminant axis cosine matrix\n"
        f"Diagonal entries (eop_C vs mid_C) measure per-category axis stability. "
        f"Mean diagonal = {diag_cos.mean():+.3f}. "
        f"High = axis preserved across capture position; low = position-dependent.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig34_cross_protocol_axis_cos.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig34_cross_protocol_axis_cos.png")

    # Save artifact
    out_path = write_artifact("mid_seq_native_compare.pt")
    torch.save(
        {
            "cells": cells,
            "cross_cos": torch.from_numpy(cross_cos),
            "diag_cos": torch.from_numpy(diag_cos),
            "categories": categories,
            "sink_dims": sink_dims,
        },
        out_path,
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()


# Silence unused-import (apply_sink_removal kept for callers reading the module).
_ = apply_sink_removal
