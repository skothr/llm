"""fig27 + fig29: validate the 23 discriminant directions.

fig27 — Discriminant connectivity heatmap:
  23x23 pairwise cosine matrix of the discriminant directions.
  Reveals semantic siblings (high positive) and opposites (negative).

fig29 — Self-validation projection:
  Project all 167 existing captures onto each of 23 discriminants.
  Tests whether captures from a known content source top-score the
  matching discriminant: country pool should peak on 'country';
  haiku tokens should peak on 'nature'; forced 'refuse' should peak
  on 'refusal'; etc.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import Any, cast
cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

from collections import Counter
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
        in_cat_idxs = [i for i, c in enumerate(caps) if c["category"] == cat]
        out_cat_idxs = [i for i in range(len(caps)) if caps[i]["category"] != cat]
        in_hs = apply_sink_removal(torch.stack([caps[i]["h"] for i in in_cat_idxs]), sink_dims)
        out_hs = apply_sink_removal(torch.stack([caps[i]["h"] for i in out_cat_idxs]), sink_dims)
        d = in_hs.mean(dim=0) - out_hs.mean(dim=0)
        discr[cat] = d / (d.norm() + 1e-9)
    return discr


def map_src_to_expected_cat(it: dict[str, Any]) -> str | None:
    """Each existing capture's 'expected' discriminant category. Used to
    score self-validation."""
    src = it["src"]
    if src in ("country_src", "country_test"):
        return "country"
    if src == "non_country_src":
        return None  # no positive expectation; just shouldn't top country
    if src == "haiku_gen":
        return "nature"
    if src == "aggregate" and it["prompt_id"] in ("creative_haiku", "creative_poem"):
        return "nature"
    if src == "aggregate" and it["prompt_id"] == "factual_easy":
        return "country"
    if src == "aggregate" and it["prompt_id"] == "code":
        return "codemath"
    if src == "aggregate" and it["prompt_id"] == "math":
        return "codemath"
    if src == "forced" and "refusal" in it.get("prompt_id", "") and "refuse" in (it.get("token") or "").lower():
        return "refusal"
    if src == "forced" and "negation" in it.get("prompt_id", ""):
        return "negation"
    return None


def main() -> None:
    vocab = torch.load(ARTIFACTS / "vocab_atlas.pt", weights_only=False)
    pw = torch.load(ARTIFACTS / "pairwise_and_hotdims.pt", weights_only=False)
    labels: dict[int, str] = pw["labels"]
    sink_dims = sorted([idx for idx, lbl in labels.items() if lbl == "sink"])
    categories = list(vocab["categories"])
    n_cats = len(categories)

    discr = compute_discriminants(vocab, sink_dims)
    D = torch.stack([discr[c] for c in categories])
    print(f"computed {n_cats} discriminant directions")

    # ===== fig27: discriminant connectivity heatmap =====
    C_d = (D @ D.T).numpy()
    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(C_d, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(categories, fontsize=8)
    for i in range(n_cats):
        for j in range(n_cats):
            if i != j:
                v = C_d[i, j]
                if abs(v) > 0.4:
                    color = "white" if abs(v) > 0.6 else "black"
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=6, color=color)
    fig.colorbar(im, ax=ax, label="discriminant cosine")
    ax.set_title("fig27 — Discriminant connectivity (23×23)\n"
                 "Strong + (red, blue-text) = semantic siblings; "
                 "strong − (blue, white-text) = opposites.\n"
                 f"Mean off-diag = {float(C_d[~np.eye(n_cats, dtype=bool)].mean()):+.3f}; "
                 f"min = {float(C_d[~np.eye(n_cats, dtype=bool)].min()):+.3f}; "
                 f"max = {float(C_d[~np.eye(n_cats, dtype=bool)].max()):+.3f}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig27_discriminant_connectivity.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig27_discriminant_connectivity.png")

    # Top siblings and opposites table
    iu = torch.triu_indices(n_cats, n_cats, offset=1)
    pairs: list[tuple[float, str, str]] = []
    for j in range(iu.shape[1]):
        a, b = int(iu[0][j]), int(iu[1][j])
        pairs.append((float(C_d[a, b]), categories[a], categories[b]))
    pairs.sort()
    print("\n--- Top-10 semantic OPPOSITES (most negative discriminant cosine) ---")
    for v, a, b in pairs[:10]:
        print(f"  {v:+.3f}  {a:<14} <-> {b}")
    print("\n--- Top-10 semantic SIBLINGS (most positive discriminant cosine) ---")
    for v, a, b in pairs[-10:][::-1]:
        print(f"  {v:+.3f}  {a:<14} <-> {b}")

    # ===== fig29: self-validation =====
    H_existing = pw["H"]
    items = pw["items_meta"]
    H_res = apply_sink_removal(H_existing, sink_dims)
    H_unit = torch.nn.functional.normalize(H_res, dim=1)
    projections = H_unit @ D.T  # (167, 23)

    n_items = H_existing.shape[0]
    fig, ax = plt.subplots(figsize=(16, 22))
    # group rows by source
    src_order = ["aggregate", "haiku_gen", "forced",
                 "country_src", "non_country_src", "country_test"]
    row_order: list[int] = []
    src_boundaries: list[int] = []
    for s in src_order:
        for i, it in enumerate(items):
            if it["src"] == s:
                row_order.append(i)
        src_boundaries.append(len(row_order))
    row_order_arr = np.array(row_order)
    proj_sorted = projections[row_order_arr].numpy()
    vmax = float(np.abs(proj_sorted).max())
    im = ax.imshow(proj_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)

    # row labels: token / src / prompt_id
    row_lbls: list[str] = []
    for i in row_order:
        it = items[i]
        tok = (it.get("token") or "").replace("\n", "\\n")[:10]
        row_lbls.append(f"[{it['src'][:11]:<11}|{it['prompt_id'][:14]:<14}|{tok}]")
    ax.set_yticks(range(n_items))
    ax.set_yticklabels(row_lbls, fontsize=5)

    for b in src_boundaries[:-1]:
        ax.axhline(b - 0.5, color="black", linewidth=0.6)

    # Mark expected category per row, if any, with a black dot
    for new_row_idx, original_idx in enumerate(row_order):
        expected = map_src_to_expected_cat(items[original_idx])
        if expected and expected in categories:
            col = categories.index(expected)
            ax.plot(col, new_row_idx, "o", color="black", markersize=2)

    fig.colorbar(im, ax=ax, label="discriminant cosine")
    ax.set_title("fig29 — Self-validation: project 167 existing captures onto 23 discriminants\n"
                 "Black dots mark the EXPECTED top category per row (from src/prompt_id mapping).\n"
                 "Strong reds where the dots are = success; reds elsewhere = unexpected high projection.")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig29_self_validation.png", dpi=180)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig29_self_validation.png")

    # Numeric validation: for each row with an expected category, is the
    # expected category the top scorer?
    print("\n--- Self-validation: how often the expected category is top-1, top-3 ---")
    expectations = []
    for i in range(n_items):
        exp = map_src_to_expected_cat(items[i])
        if exp and exp in categories:
            sorted_idx = projections[i].argsort(descending=True).tolist()
            exp_idx = categories.index(exp)
            rank = sorted_idx.index(exp_idx)
            expectations.append((items[i]["src"], items[i]["prompt_id"], exp, rank, projections[i, exp_idx].item()))

    by_expected: dict[str, list[tuple[Any, ...]]] = {}
    for src, pid, exp, rank, val in expectations:
        by_expected.setdefault(exp, []).append((src, pid, rank, val))

    for exp, rows in sorted(by_expected.items()):
        n = len(rows)
        top1 = sum(1 for r in rows if r[2] == 0)
        top3 = sum(1 for r in rows if r[2] < 3)
        top5 = sum(1 for r in rows if r[2] < 5)
        mean_rank = sum(r[2] for r in rows) / n
        print(f"  expected={exp:<10}  n={n:>3}   "
              f"top-1: {top1:>3}/{n} ({top1/n:.0%})   "
              f"top-3: {top3:>3}/{n} ({top3/n:.0%})   "
              f"top-5: {top5:>3}/{n} ({top5/n:.0%})   "
              f"mean rank: {mean_rank:.1f}")

    # Where do captures FAIL? Top wrong category for failed rows
    print("\n--- For failed (rank>=3) captures: which discriminant DID they top? ---")
    failed_top_cats: Counter = Counter()
    for src, pid, exp, rank, val in expectations:
        if rank >= 3:
            for i in range(n_items):
                if items[i]["src"] == src and items[i]["prompt_id"] == pid:
                    top_cat = categories[int(projections[i].argmax().item())]
                    failed_top_cats[(exp, top_cat)] += 1
                    break
    for (exp, top_cat), count in failed_top_cats.most_common(10):
        print(f"  expected={exp:<10}  but topped {top_cat:<14}  ({count}x)")


if __name__ == "__main__":
    main()
