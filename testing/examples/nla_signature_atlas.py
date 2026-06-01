"""Signature-glyph atlas: render every capture as a feature-signature star.

At each point in PC1/PC2 (sink-removed) space, place an 8-ray glyph
where each ray:
  * points at theta_i = i * 360/8 for feature dim i
  * has length = normalized magnitude of h at that dim
  * has color = red if h[dim] > 0, blue if h[dim] < 0

Captures with similar feature signatures produce similar star shapes,
allowing the viewer to see local feature-signature neighborhoods in the
PC layout without having to overlay 8 separate colored atlases.

Output: fig10_signature_atlas.png
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

SRC_COLORS: dict[str, str] = {
    "aggregate": "#1f77b4",
    "haiku_gen": "#ff7f0e",
    "forced": "#d62728",
    "country_src": "#2ca02c",
    "non_country_src": "#9467bd",
    "country_test": "#8c564b",
}


def draw_signature_glyph(
    ax: Any, cx: float, cy: float, values: np.ndarray, radius: float, scale_max: float
) -> None:
    """Draw an 8-ray star glyph centered at (cx, cy).

    values: 1D array length 8, signed feature-dim values.
    radius: maximum visual ray length at scale_max (data-coords).
    scale_max: value at which a ray reaches `radius` (caller-chosen normalizer).
    """
    n = len(values)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for i, v in enumerate(values):
        mag = min(abs(v) / scale_max, 1.0)
        if mag < 0.05:
            continue
        x_end = cx + radius * mag * np.cos(angles[i])
        y_end = cy + radius * mag * np.sin(angles[i])
        color = "#d62728" if v >= 0 else "#1f77b4"
        ax.plot(
            [cx, x_end],
            [cy, y_end],
            color=color,
            linewidth=0.9,
            alpha=0.85,
            solid_capstyle="round",
        )


def main() -> None:
    atlas = load_artifact("sink_removed_atlas.pt")
    pw = load_artifact("pairwise_and_hotdims.pt")

    H = pw["H"]
    items = pw["items_meta"]
    coords = atlas["coords_res"]
    feature_dims = atlas["feature_dims"]  # sorted indices of feature dims

    # Select 8 feature dims (we have 8) and sort them for consistent ray angles
    feat_vals = H[:, feature_dims].numpy()  # (n, 8)
    abs_max = float(np.abs(feat_vals).max())  # global normalizer

    print(
        f"feature dims (angle order, 0 deg to 315 deg in 45-deg steps): {feature_dims}"
    )
    print(f"global max |h[feature_dim]|: {abs_max:.2f}")

    xs = coords[:, 0].numpy()
    ys = coords[:, 1].numpy()
    xrange = xs.max() - xs.min()
    yrange = ys.max() - ys.min()
    radius = 0.020 * max(xrange, yrange)  # 2% of canvas

    # main figure: source-colored points + glyphs
    fig, ax = plt.subplots(figsize=(14, 11))

    # legend marks for sources
    for src, color in SRC_COLORS.items():
        idxs = [i for i, it in enumerate(items) if it["src"] == src]
        if not idxs:
            continue
        x = xs[idxs]
        y = ys[idxs]
        ax.scatter(
            x,
            y,
            c=color,
            label=f"{src} (n={len(idxs)})",
            alpha=0.35,
            s=30,
            edgecolors="black",
            linewidths=0.3,
            zorder=1,
        )

    # glyphs over every point
    for i in range(len(items)):
        draw_signature_glyph(
            ax, xs[i], ys[i], feat_vals[i], radius=radius, scale_max=abs_max
        )

    # explanatory inset: a glyph legend in the bottom-right corner
    legend_x, legend_y = xs.max() - 0.20 * xrange, ys.min() + 0.08 * yrange
    legend_r = 1.5 * radius
    angles = np.linspace(0, 2 * np.pi, len(feature_dims), endpoint=False)
    for i, (ang, d) in enumerate(zip(angles, feature_dims)):
        x_end = legend_x + legend_r * np.cos(ang)
        y_end = legend_y + legend_r * np.sin(ang)
        ax.plot([legend_x, x_end], [legend_y, y_end], color="black", linewidth=0.8)
        ax.annotate(
            f"dim {d}",
            (x_end, y_end),
            textcoords="offset points",
            xytext=(int(8 * np.cos(ang)), int(8 * np.sin(ang))),
            fontsize=7,
            ha="center",
            va="center",
        )
    ax.annotate(
        "ray angle = feature dim\nray length = |h[dim]|\nred = +, blue = -",
        (legend_x, legend_y - 2.0 * legend_r),
        fontsize=8,
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "gray"},
    )

    ax.set_xlabel(
        f"PC1 (sink-removed, {float(atlas['var_res'][0] / atlas['var_res'].sum()):.1%})"
    )
    ax.set_ylabel(
        f"PC2 (sink-removed, {float(atlas['var_res'][1] / atlas['var_res'].sum()):.1%})"
    )
    ax.set_title(
        "Feature signature atlas: 8-ray glyphs over PC1/PC2 (sink-removed)\n"
        "Adjacent points with similar glyphs share local feature signature"
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig10_signature_atlas.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig10_signature_atlas.png")

    # fig11: zoomed regions — country cluster vs haiku gen trajectory
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left panel: country cluster zoom with large glyphs + token labels
    country_idxs = [
        i
        for i, it in enumerate(items)
        if it["src"] in ("country_src", "non_country_src", "country_test")
    ]
    radius_zoom = 0.06 * (xs[country_idxs].max() - xs[country_idxs].min())
    for src in ("country_src", "non_country_src", "country_test"):
        idxs = [i for i in country_idxs if items[i]["src"] == src]
        if idxs:
            axes[0].scatter(
                xs[idxs],
                ys[idxs],
                c=SRC_COLORS[src],
                label=src,
                s=80,
                alpha=0.35,
                edgecolors="black",
                linewidths=0.5,
                zorder=1,
            )
    for i in country_idxs:
        draw_signature_glyph(
            axes[0], xs[i], ys[i], feat_vals[i], radius=radius_zoom, scale_max=abs_max
        )
        label = items[i]["prompt_id"][:18]
        axes[0].annotate(
            label,
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=6,
            alpha=0.7,
        )
    axes[0].set_xlim(xs[country_idxs].min() - 2, xs[country_idxs].max() + 2)
    axes[0].set_ylim(ys[country_idxs].min() - 2, ys[country_idxs].max() + 2)
    axes[0].set_xlabel("PC1 (sink-removed)")
    axes[0].set_ylabel("PC2 (sink-removed)")
    axes[0].set_title(
        "Country cluster zoom (29 captures): nearly-identical glyphs\n"
        "= shared feature signature, only sub-feature perturbations differ"
    )
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal", adjustable="datalim")

    # Right panel: haiku gen trajectory zoom with line connecting consecutive steps
    haiku_idxs_unsorted = [i for i, it in enumerate(items) if it["src"] == "haiku_gen"]
    haiku_idxs = sorted(haiku_idxs_unsorted, key=lambda i: items[i].get("step") or 0)
    haiku_xs = xs[haiku_idxs]
    haiku_ys = ys[haiku_idxs]
    axes[1].plot(
        haiku_xs, haiku_ys, color="#ff7f0e", alpha=0.5, linewidth=1.5, zorder=1
    )
    radius_haiku = 0.06 * (haiku_xs.max() - haiku_xs.min())
    for i in haiku_idxs:
        draw_signature_glyph(
            axes[1], xs[i], ys[i], feat_vals[i], radius=radius_haiku, scale_max=abs_max
        )
        tok = items[i]["token"]
        step = items[i]["step"]
        axes[1].annotate(
            f"{step}: {tok!r}",
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=7,
            alpha=0.85,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.7,
            },
        )
    axes[1].set_xlim(haiku_xs.min() - 5, haiku_xs.max() + 5)
    axes[1].set_ylim(haiku_ys.min() - 5, haiku_ys.max() + 5)
    axes[1].set_xlabel("PC1 (sink-removed)")
    axes[1].set_ylabel("PC2 (sink-removed)")
    axes[1].set_title(
        "Haiku generation trajectory (15 steps): glyphs morph step by step\n"
        "line traces the temporal path through feature space"
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig11_signature_zoom.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig11_signature_zoom.png")


if __name__ == "__main__":
    main()
