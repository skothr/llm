"""Path A: sink-removed semantic atlas.

Takes the 167-vector matrix and the dim-character classifier labels from
nla_pairwise_and_hotdims.pt. Zeros out the 7 sink-classified component
indices from each h, then re-runs PCA, pairwise cosines, and per-source
intra/cross stats on the residual.

Outputs:
  fig7_pca_sink_vs_clean.png   - side-by-side: original PCA vs sink-removed PCA
  fig8_cosine_sink_vs_clean.png - side-by-side: original cosine matrix vs sink-removed
  fig9_feature_colored_atlas.png - sink-removed PCA, points colored by selected
                                   feature-dim value (dim 32) to confirm PC axes
                                   are now content-related
  Plus a summary table to stdout.
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
FIGDIR = _REPO_ROOT / "research" / "arcs" / "nla-verbalizer" / "observations" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUT = ARTIFACTS / "sink_removed_atlas.pt"

SRC_COLORS: dict[str, str] = {
    "aggregate": "#1f77b4",
    "haiku_gen": "#ff7f0e",
    "forced": "#d62728",
    "country_src": "#2ca02c",
    "non_country_src": "#9467bd",
    "country_test": "#8c564b",
}


def pca_coords(H: torch.Tensor, n_pcs: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean_h = H.mean(dim=0, keepdim=True)
    Hc = H - mean_h
    _, S, Vh = torch.linalg.svd(Hc, full_matrices=False)
    coords = Hc @ Vh.T[:, :n_pcs]
    var = S ** 2
    return coords, var, mean_h.squeeze(0)


def per_source_cos(C: torch.Tensor, src_labels: list[str]) -> dict[str, tuple[float, float]]:
    n = C.shape[0]
    unique_srcs = sorted(set(src_labels))
    out: dict[str, tuple[float, float]] = {}
    for s in unique_srcs:
        mask = torch.tensor([lbl == s for lbl in src_labels])
        if mask.sum() < 2:
            continue
        intra = C[mask][:, mask]
        eye = torch.eye(int(mask.sum().item()), dtype=torch.bool)
        intra_mean = float(intra[~eye].mean().item())
        cross = float(C[mask][:, ~mask].mean().item())
        out[s] = (intra_mean, cross)
    return out


def scatter_pca(ax: Any, coords: torch.Tensor, items: list[dict[str, Any]],
                title: str, var_frac: tuple[float, float]) -> None:
    for src, color in SRC_COLORS.items():
        idxs = [i for i, it in enumerate(items) if it["src"] == src]
        if not idxs:
            continue
        x = coords[idxs, 0].numpy()
        y = coords[idxs, 1].numpy()
        ax.scatter(x, y, c=color, label=f"{src} (n={len(idxs)})",
                   alpha=0.7, s=55, edgecolors="black", linewidths=0.4)
    ax.set_xlabel(f"PC1 ({var_frac[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_frac[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)


def main() -> None:
    pw = torch.load(ARTIFACTS / "pairwise_and_hotdims.pt", weights_only=False)
    H: torch.Tensor = pw["H"]
    items: list[dict[str, Any]] = pw["items_meta"]
    labels: dict[int, str] = pw["labels"]

    sink_dims = sorted([idx for idx, lbl in labels.items() if lbl == "sink"])
    feature_dims = sorted([idx for idx, lbl in labels.items() if lbl == "feature"])
    print(f"sink dims removed: {sink_dims}")
    print(f"feature dims tracked: {feature_dims}")

    # --- original geometry (for comparison) ---
    coords_orig, var_orig, _ = pca_coords(H)
    Hn_orig = torch.nn.functional.normalize(H, dim=1)
    C_orig = Hn_orig @ Hn_orig.T

    # --- sink-removed geometry ---
    H_res = H.clone()
    for d in sink_dims:
        H_res[:, d] = 0.0
    coords_res, var_res, _ = pca_coords(H_res)
    Hn_res = torch.nn.functional.normalize(H_res, dim=1)
    C_res = Hn_res @ Hn_res.T

    total_orig = float(var_orig.sum().item())
    total_res = float(var_res.sum().item())
    print(f"\ntotal variance: original={total_orig:.0f}, sink-removed={total_res:.0f}, "
          f"frac kept={total_res/total_orig:.3f}")

    print(f"\nPC variance fractions:")
    print(f"  original    : PC1={float(var_orig[0]/total_orig):.3f}  PC2={float(var_orig[1]/total_orig):.3f}  "
          f"PC3={float(var_orig[2]/total_orig):.3f}  PC4={float(var_orig[3]/total_orig):.3f}")
    print(f"  sink-removed: PC1={float(var_res[0]/total_res):.3f}  PC2={float(var_res[1]/total_res):.3f}  "
          f"PC3={float(var_res[2]/total_res):.3f}  PC4={float(var_res[3]/total_res):.3f}")

    # --- per-source cosine comparison ---
    src_labels = [it["src"] for it in items]
    cos_orig = per_source_cos(C_orig, src_labels)
    cos_res = per_source_cos(C_res, src_labels)
    print(f"\nper-source mean cosine (intra-source, cross-source):")
    print(f"  {'source':<18}  {'orig intra':>10}  {'orig cross':>10}  {'res intra':>10}  {'res cross':>10}  {'intra delta':>12}")
    for s in sorted(cos_orig.keys()):
        oi, oc = cos_orig[s]
        ri, rc = cos_res[s]
        print(f"  {s:<18}  {oi:>+10.3f}  {oc:>+10.3f}  {ri:>+10.3f}  {rc:>+10.3f}  {ri - oi:>+12.3f}")

    eye = torch.eye(H.shape[0], dtype=torch.bool)
    print(f"\noff-diag pairwise cosines:")
    print(f"  original    : mean={float(C_orig[~eye].mean()):.3f}  min={float(C_orig[~eye].min()):.3f}  max={float(C_orig[~eye].max()):.3f}")
    print(f"  sink-removed: mean={float(C_res[~eye].mean()):.3f}  min={float(C_res[~eye].min()):.3f}  max={float(C_res[~eye].max()):.3f}")

    # --- fig7: side-by-side PCA ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    scatter_pca(axes[0], coords_orig, items, "Original h[20] (PC1/PC2)",
                (float(var_orig[0]/total_orig), float(var_orig[1]/total_orig)))
    scatter_pca(axes[1], coords_res, items, "Sink-removed h[20] (PC1/PC2)",
                (float(var_res[0]/total_res), float(var_res[1]/total_res)))
    fig.suptitle(f"Removing 7 sink dims {sink_dims} rotates the PCA layout")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig7_pca_sink_vs_clean.png", dpi=180)
    plt.close(fig)
    print(f"\nwrote {FIGDIR}/fig7_pca_sink_vs_clean.png")

    # --- fig8: cosine matrices side by side ---
    src_order = ["aggregate", "haiku_gen", "forced",
                 "country_src", "non_country_src", "country_test"]
    order: list[int] = []
    boundaries: list[int] = []
    for s in src_order:
        for i, it in enumerate(items):
            if it["src"] == s:
                order.append(i)
        boundaries.append(len(order))
    order_arr = np.array(order)
    centers = [0] + boundaries
    label_pos = [(centers[i] + centers[i+1]) / 2 - 0.5 for i in range(len(src_order))]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, C, title in [(axes[0], C_orig, "Original pairwise cosine"),
                          (axes[1], C_res, "Sink-removed pairwise cosine")]:
        C_ord = C[np.ix_(order_arr, order_arr)].numpy()
        im = ax.imshow(C_ord, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for b in boundaries[:-1]:
            ax.axhline(b - 0.5, color="black", linewidth=0.6)
            ax.axvline(b - 0.5, color="black", linewidth=0.6)
        ax.set_xticks(label_pos)
        ax.set_xticklabels(src_order, rotation=30, ha="right")
        ax.set_yticks(label_pos)
        ax.set_yticklabels(src_order)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="cosine")
    fig.suptitle("Removing sinks reveals genuine content similarity structure")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig8_cosine_sink_vs_clean.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig8_cosine_sink_vs_clean.png")

    # --- fig9: sink-removed atlas colored by feature dim 32 value ---
    # Use a 2x2 grid: by source / by feature dim 32 / by feature dim 608 / by AR cosine
    feat_dim_a = 32
    feat_dim_b = 608
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    scatter_pca(axes[0, 0], coords_res, items, "Sink-removed PCA — by source",
                (float(var_res[0]/total_res), float(var_res[1]/total_res)))

    for ax, dim_idx, title in [(axes[0, 1], feat_dim_a, f"Sink-removed PCA — by h[{feat_dim_a}] value (feature dim)"),
                                 (axes[1, 0], feat_dim_b, f"Sink-removed PCA — by h[{feat_dim_b}] value (feature dim)")]:
        x = coords_res[:, 0].numpy()
        y = coords_res[:, 1].numpy()
        c = H[:, dim_idx].numpy()
        absmax = max(abs(c.min()), abs(c.max()))
        sc = ax.scatter(x, y, c=c, cmap="RdBu_r", vmin=-absmax, vmax=absmax,
                        alpha=0.85, s=55, edgecolors="black", linewidths=0.4)
        ax.set_xlabel(f"PC1 ({var_res[0]/total_res*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_res[1]/total_res*100:.1f}%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label=f"h[{dim_idx}]")

    # 4th panel: AR cosine where available
    ax = axes[1, 1]
    x = coords_res[:, 0].numpy()
    y = coords_res[:, 1].numpy()
    # we don't have AR cosine in pairwise artifact; reload geometric features
    geo = torch.load(ARTIFACTS / "geometric_features.pt", weights_only=False)
    # match by (src, prompt_id, step, token)
    cos_by_key: dict[tuple[Any, Any, Any, Any], float] = {}
    for r in geo["rows"]:
        if r.get("cos") is not None:
            key = (r["src"], r["prompt_id"], r.get("step"), r.get("token"))
            cos_by_key[key] = r["cos"]
    cos_list: list[float] = []
    has_ar: list[int] = []
    for i, it in enumerate(items):
        key = (it["src"], it["prompt_id"], it.get("step"), it.get("token"))
        if key in cos_by_key:
            cos_list.append(cos_by_key[key])
            has_ar.append(i)
    has_ar_arr = np.array(has_ar)
    if len(has_ar) > 0:
        sc = ax.scatter(x[has_ar_arr], y[has_ar_arr], c=cos_list, cmap="viridis",
                        vmin=0.70, vmax=0.93, alpha=0.85, s=55, edgecolors="black", linewidths=0.4)
        ax.scatter(x[~np.isin(np.arange(len(items)), has_ar_arr)],
                   y[~np.isin(np.arange(len(items)), has_ar_arr)],
                   c="lightgray", alpha=0.4, s=30, label="no AR")
        ax.set_xlabel(f"PC1 ({var_res[0]/total_res*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_res[1]/total_res*100:.1f}%)")
        ax.set_title("Sink-removed PCA — by AR round-trip cosine")
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label="AR cosine")

    fig.suptitle("Sink-removed atlas: source, two feature-dim values, AR fidelity")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig9_feature_colored_atlas.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig9_feature_colored_atlas.png")

    torch.save({
        "sink_dims": sink_dims,
        "feature_dims": feature_dims,
        "coords_orig": coords_orig,
        "coords_res": coords_res,
        "C_res": C_res,
        "var_orig": var_orig,
        "var_res": var_res,
        "cos_orig": cos_orig,
        "cos_res": cos_res,
    }, OUT)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
