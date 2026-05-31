"""Render fig35 for MAIN-48: concept arithmetic atlas.

Multi-row text-table figure showing each arithmetic combination,
the predicted result, and the actual AV reading.
"""

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


from _nla_artifacts import FIGURES as FIGDIR, load_artifact

FIGDIR.mkdir(parents=True, exist_ok=True)


CATEGORY_COLORS: dict[str, str] = {
    "analogy": "#1f77b4",
    "subtraction": "#d62728",
    "axis": "#2ca02c",
    "compound": "#9467bd",
}


def main() -> None:
    out = load_artifact("concept_arithmetic_atlas.pt")
    combos = out["combos"]
    n = len(combos)
    print(f"loaded {n} combos")

    TITLE_RESERVE = 0.04
    row_h_frac = (1.0 - TITLE_RESERVE) / n
    fig = plt.figure(figsize=(20, 2.3 * n))

    for j, c in enumerate(combos):
        row_y = (1.0 - TITLE_RESERVE) - (j + 1) * row_h_frac
        color = CATEGORY_COLORS.get(c["category"], "#666666")

        # Left column: arithmetic + prediction
        ax_l = fig.add_axes((0.02, row_y + 0.005, 0.30, row_h_frac - 0.01))
        ax_l.axis("off")
        ax_l.add_patch(
            plt.Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=ax_l.transAxes,
                facecolor=color,
                alpha=0.10,
                edgecolor="none",
            )
        )
        ax_l.text(
            0.02,
            0.78,
            f"[{c['category']}]",
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            color=color,
            weight="bold",
            transform=ax_l.transAxes,
        )
        ax_l.text(
            0.02,
            0.58,
            c["label"],
            ha="left",
            va="top",
            fontsize=11,
            family="monospace",
            weight="bold",
            transform=ax_l.transAxes,
        )
        n_raw = float(c["h_raw"].norm().item())
        ax_l.text(
            0.02,
            0.30,
            f"||h_raw|| = {n_raw:.2f}\nrescaled to 150",
            ha="left",
            va="top",
            fontsize=8,
            family="monospace",
            alpha=0.7,
            transform=ax_l.transAxes,
        )
        ax_l.text(
            0.02,
            0.10,
            f"predicted: {c['predict']}",
            ha="left",
            va="top",
            fontsize=8,
            family="serif",
            style="italic",
            alpha=0.8,
            transform=ax_l.transAxes,
        )

        # Right column: AV reading
        ax_r = fig.add_axes((0.33, row_y + 0.005, 0.65, row_h_frac - 0.01))
        ax_r.axis("off")
        av_text = c.get("av_text", "(no AV reading)")
        paragraphs = [p.strip() for p in av_text.split("\n\n") if p.strip()]
        wrapped = "\n\n".join(
            textwrap.fill(p, width=150, break_long_words=False, break_on_hyphens=False)
            for p in paragraphs
        )
        ax_r.text(
            0.0,
            0.95,
            wrapped,
            ha="left",
            va="top",
            fontsize=8,
            family="serif",
            transform=ax_r.transAxes,
        )

    fig.suptitle(
        f"fig35 — Concept arithmetic atlas: AV-decoded combinations on layer-20 h\n"
        f"Each row: arithmetic expression (left, color = category) and the AV reading (right). "
        f"Tests whether NLA preserves word2vec-style additive/subtractive structure.",
        fontsize=11,
        y=0.995,
    )
    fig.savefig(FIGDIR / "fig35_concept_arithmetic_atlas.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig35_concept_arithmetic_atlas.png")


if __name__ == "__main__":
    main()
