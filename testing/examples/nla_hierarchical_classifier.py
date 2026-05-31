"""Hierarchical classifier — disambiguate sibling discriminants.

The 23-discriminant basis successfully classifies into macro-clusters
(content / function / structural) but country and capital constantly
swap top-1 because their discriminants have cos +0.938. Self-validation
showed country-source captures hit top-5 79% but top-1 only 34%.

This script adds a second-level disambiguator:
  1. Compute the 23 first-level discriminants from the vocab atlas
  2. Identify sibling pairs (cross-pair cos > +0.85)
  3. For each pair (A, B), compute sub-discriminator
     d_AB = unit(mean(H_A) - mean(H_B)) — a within-pair direction
  4. For each existing capture, if its top-2 first-level categories
     form a sibling pair, use d_AB's sign to pick the actual category
  5. Compare baseline vs hierarchical top-1 accuracy per source

Produces fig30_hierarchical_accuracy.png and a numeric table.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import Any, cast

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

from collections import Counter
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from _nla_artifacts import FIGURES as FIGDIR, load_artifact

FIGDIR.mkdir(parents=True, exist_ok=True)

SIBLING_COS_THRESHOLD = 0.80  # captures all 5 pairs that fig27 surfaced


def apply_sink_removal(h: torch.Tensor, sink_dims: list[int]) -> torch.Tensor:
    h_res = h.clone()
    h_res[..., sink_dims] = 0.0
    return h_res


def compute_discriminants(
    vocab: dict[str, Any], sink_dims: list[int]
) -> dict[str, torch.Tensor]:
    caps = vocab["captures"]
    cats = vocab["categories"]
    discr: dict[str, torch.Tensor] = {}
    for cat in cats:
        in_cat = [i for i, c in enumerate(caps) if c["category"] == cat]
        out_cat = [i for i in range(len(caps)) if caps[i]["category"] != cat]
        in_hs = apply_sink_removal(
            torch.stack([caps[i]["h"] for i in in_cat]), sink_dims
        )
        out_hs = apply_sink_removal(
            torch.stack([caps[i]["h"] for i in out_cat]), sink_dims
        )
        d = in_hs.mean(dim=0) - out_hs.mean(dim=0)
        discr[cat] = d / (d.norm() + 1e-9)
    return discr


def compute_sub_discriminants(
    vocab: dict[str, Any],
    sink_dims: list[int],
    sibling_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], torch.Tensor]:
    """For each (cat_a, cat_b) pair, sub-discriminant points FROM b TOWARD a:
    d_AB = unit(mean(H_a) - mean(H_b)).
    Projection > 0 means h is more like cat_a; < 0 means more like cat_b."""
    caps = vocab["captures"]
    sub: dict[tuple[str, str], torch.Tensor] = {}
    for a, b in sibling_pairs:
        ai = [i for i, c in enumerate(caps) if c["category"] == a]
        bi = [i for i, c in enumerate(caps) if c["category"] == b]
        H_a = apply_sink_removal(torch.stack([caps[i]["h"] for i in ai]), sink_dims)
        H_b = apply_sink_removal(torch.stack([caps[i]["h"] for i in bi]), sink_dims)
        d = H_a.mean(dim=0) - H_b.mean(dim=0)
        sub[(a, b)] = d / (d.norm() + 1e-9)
    return sub


def map_src_to_expected_cat(it: dict[str, Any]) -> str | None:
    """Same mapping as fig29 used."""
    src = it["src"]
    if src in ("country_src", "country_test"):
        return "country"
    if src == "non_country_src":
        return None
    if src == "haiku_gen":
        return "nature"
    if src == "aggregate" and it["prompt_id"] in ("creative_haiku", "creative_poem"):
        return "nature"
    if src == "aggregate" and it["prompt_id"] == "factual_easy":
        return "country"
    if src == "aggregate" and it["prompt_id"] in ("code", "math"):
        return "codemath"
    if (
        src == "forced"
        and "refusal" in it.get("prompt_id", "")
        and "refuse" in (it.get("token") or "").lower()
    ):
        return "refusal"
    if src == "forced" and "negation" in it.get("prompt_id", ""):
        return "negation"
    return None


def main() -> None:
    vocab = load_artifact("vocab_atlas.pt")
    pw = load_artifact("pairwise_and_hotdims.pt")
    labels: dict[int, str] = pw["labels"]
    sink_dims = sorted([idx for idx, lbl in labels.items() if lbl == "sink"])
    categories = list(vocab["categories"])

    discr = compute_discriminants(vocab, sink_dims)
    D_stack = torch.stack([discr[c] for c in categories])
    C_d = D_stack @ D_stack.T
    print(
        f"computed {len(discr)} discriminants; mean cross-cos = {float((C_d[~torch.eye(len(categories), dtype=torch.bool)]).mean().item()):+.3f}"
    )

    # Find sibling pairs from the discriminant cosine matrix
    sibling_pairs: list[tuple[str, str]] = []
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            if float(C_d[i, j].item()) > SIBLING_COS_THRESHOLD:
                sibling_pairs.append((categories[i], categories[j]))
    print(f"\nsibling pairs (discr cos > {SIBLING_COS_THRESHOLD}):")
    for a, b in sibling_pairs:
        cos_ab = float(C_d[categories.index(a), categories.index(b)].item())
        print(f"  {a:<13} <-> {b:<13}  cos = {cos_ab:+.3f}")

    sub_discr = compute_sub_discriminants(vocab, sink_dims, sibling_pairs)
    # Build a lookup so we can find the sub-discriminator regardless of order
    sub_lookup: dict[frozenset[str], tuple[str, str, torch.Tensor]] = {}
    for (a, b), d in sub_discr.items():
        sub_lookup[frozenset({a, b})] = (a, b, d)

    # Load 167 existing captures
    items = pw["items_meta"]
    H_existing = pw["H"]
    H_res = apply_sink_removal(H_existing, sink_dims)
    H_unit = torch.nn.functional.normalize(H_res, dim=1)
    proj_first = H_unit @ D_stack.T  # (167, 23)

    # Baseline and hierarchical top-1 categories per capture
    baseline_top1: list[str] = []
    hierarchical_top1: list[str] = []
    n_disambig_applied = 0
    n_disambig_flipped = 0
    for i in range(H_existing.shape[0]):
        scores = proj_first[i]
        sorted_idx = scores.argsort(descending=True).tolist()
        top1_idx = sorted_idx[0]
        top2_idx = sorted_idx[1]
        top1 = categories[top1_idx]
        top2 = categories[top2_idx]
        baseline_top1.append(top1)

        # Hierarchical: if top-1 and top-2 form a sibling pair, use sub-discriminator
        pair_key = frozenset({top1, top2})
        if pair_key in sub_lookup:
            n_disambig_applied += 1
            a, b, d_sub = sub_lookup[pair_key]
            # d_sub points from b toward a; projection > 0 → a; < 0 → b
            proj_sub = float(torch.dot(H_unit[i], d_sub).item())
            chosen = a if proj_sub > 0 else b
            hierarchical_top1.append(chosen)
            if chosen != top1:
                n_disambig_flipped += 1
        else:
            hierarchical_top1.append(top1)

    print(
        f"\ndisambiguator applied to {n_disambig_applied}/{H_existing.shape[0]} captures; "
        f"flipped top-1 in {n_disambig_flipped} cases"
    )

    # Per-source accuracy comparison
    print(f"\n--- baseline vs hierarchical top-1 accuracy per expected category ---")
    print(
        f"  {'expected':<12} {'n':>4}  {'baseline':>10}  {'hierarchical':>12}  {'Δ':>6}"
    )
    print("  " + "-" * 60)
    by_expected: dict[str, list[tuple[str, str]]] = {}
    for i, it in enumerate(items):
        exp = map_src_to_expected_cat(it)
        if exp is None or exp not in categories:
            continue
        by_expected.setdefault(exp, []).append((baseline_top1[i], hierarchical_top1[i]))

    bar_data: list[tuple[str, int, float, float]] = []
    for exp in sorted(by_expected.keys()):
        rows = by_expected[exp]
        n = len(rows)
        base_acc = sum(1 for b, _ in rows if b == exp) / n
        hier_acc = sum(1 for _, h in rows if h == exp) / n
        delta = hier_acc - base_acc
        print(
            f"  {exp:<12} {n:>4}  {base_acc:>9.0%}   {hier_acc:>11.0%}   {delta:>+6.1%}"
        )
        bar_data.append((exp, n, base_acc, hier_acc))

    # Overall
    total = sum(len(rows) for rows in by_expected.values())
    base_all = sum(
        sum(1 for b, _ in rows if b == exp) for exp, rows in by_expected.items()
    )
    hier_all = sum(
        sum(1 for _, h in rows if h == exp) for exp, rows in by_expected.items()
    )
    print("  " + "-" * 60)
    print(
        f"  {'OVERALL':<12} {total:>4}  {base_all / total:>9.0%}   {hier_all / total:>11.0%}   {(hier_all - base_all) / total:>+6.1%}"
    )

    # --- fig30: bar chart comparing baseline vs hierarchical top-1 ---
    fig, ax = plt.subplots(figsize=(11, 6))
    cats_plot = [e for e, _, _, _ in bar_data]
    base_pcts = [b * 100 for _, _, b, _ in bar_data]
    hier_pcts = [h * 100 for _, _, _, h in bar_data]
    ns = [n for _, n, _, _ in bar_data]
    x = np.arange(len(cats_plot))
    w = 0.35
    ax.bar(
        x - w / 2,
        base_pcts,
        w,
        label="Baseline (single discriminant)",
        color="#9ecae1",
        edgecolor="black",
    )
    ax.bar(
        x + w / 2,
        hier_pcts,
        w,
        label="Hierarchical (sub-discriminator on siblings)",
        color="#2ca02c",
        edgecolor="black",
    )
    for i, (bp, hp, n) in enumerate(zip(base_pcts, hier_pcts, ns)):
        ax.text(i - w / 2, bp + 1.5, f"{bp:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.text(
            i + w / 2,
            hp + 1.5,
            f"{hp:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )
        ax.text(i, -8, f"n={n}", ha="center", va="top", fontsize=8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cats_plot, rotation=0)
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_ylim(-12, 110)
    ax.set_title(
        f"fig30 — Hierarchical re-discrimination top-1 accuracy lift\n"
        f"Sub-discriminator applied when top-2 categories form a sibling pair "
        f"(discr cos > {SIBLING_COS_THRESHOLD})\n"
        f"Overall: baseline {base_all / total:.0%} → hierarchical {hier_all / total:.0%} "
        f"(applied to {n_disambig_applied}/{total} captures with mapped expected)"
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig30_hierarchical_accuracy.png", dpi=180)
    plt.close(fig)
    print(f"\nwrote {FIGDIR}/fig30_hierarchical_accuracy.png")

    # Where did the disambiguator help vs hurt?
    print(
        f"\n--- where the disambiguator flipped top-1 (truth, baseline, hierarchical) ---"
    )
    flips: Counter = Counter()
    for i, it in enumerate(items):
        if baseline_top1[i] == hierarchical_top1[i]:
            continue
        exp = map_src_to_expected_cat(it)
        if exp is None:
            continue
        b = baseline_top1[i]
        h = hierarchical_top1[i]
        outcome = "[ok]" if h == exp else ("[wrong]" if b == exp else "[neutral]")
        flips[(exp, b, h, outcome)] += 1
    for (exp, b, h, outcome), n in sorted(flips.items(), key=lambda x: -x[1]):
        print(f"  expected={exp:<10}  {b:<12} → {h:<12}  ({n}x)  {outcome}")


if __name__ == "__main__":
    main()
