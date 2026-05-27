"""Cross-capture geometric structure: pairwise cosines + hot-dim census.

Loads the 167 h-vectors from all artifacts. Then:

1) Stacks them into a (167, 3584) matrix and computes the pairwise
   cosine similarity matrix (167, 167). Lists:
   - top-20 most-similar pairs (excluding self)
   - top-20 least-similar pairs
   - mean intra-source and mean cross-source cosine
2) Hot-dimension census: counts how often each component index
   appears in top-K (K=5, 20, 100) magnitudes across the 167 captures.
3) PCA over the 167 vectors. Prints variance explained for top-10 PCs.

Dimension semantics are decided by `classify_dim_character()` below
(sink / polarized / feature / rare_burst / background) using sign-consistency,
coefficient-of-variation, and frequency-in-top-K thresholds tuned on the
167-capture corpus. The classifier is fully implemented — earlier drafts
of this docstring referenced a `score_hot_dim()` stub that never landed.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import Any, cast

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

from pathlib import Path
from collections import Counter
import torch


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = _REPO_ROOT / "testing" / ".cache" / "nla_artifacts"
OUT = ARTIFACTS / "pairwise_and_hotdims.pt"


def load_all() -> list[dict[str, Any]]:
    """Returns list of {src, prompt_id, token, h: Tensor(3584,), av_text}."""
    items: list[dict[str, Any]] = []

    p = ARTIFACTS / "aggregate_faithfulness.pt"
    if p.exists():
        data = torch.load(p, weights_only=False)
        for prompt in data["prompts"]:
            for cap in prompt.get("captures", []):
                items.append(
                    {
                        "src": "aggregate",
                        "prompt_id": prompt["id"],
                        "token": cap.get("token") or "",
                        "step": cap.get("step"),
                        "h": cap["h"].detach().float().cpu(),
                        "av_text": cap.get("av_text", "") or "",
                    }
                )

    p = ARTIFACTS / "rabbit_haiku_gen_trajectory.pt"
    if p.exists():
        data = torch.load(p, weights_only=False)
        for cap in data.get("captures", []):
            items.append(
                {
                    "src": "haiku_gen",
                    "prompt_id": "rabbit_haiku",
                    "token": cap.get("token") or "",
                    "step": cap.get("step"),
                    "h": cap["h"].detach().float().cpu(),
                    "av_text": cap.get("av_text", "") or "",
                }
            )

    p = ARTIFACTS / "forced_continuation.pt"
    if p.exists():
        data = torch.load(p, weights_only=False)
        for cap in data.get("captures", []):
            items.append(
                {
                    "src": "forced",
                    "prompt_id": f"{cap['pair_id']}/{cap['version']}",
                    "token": cap.get("actual_token") or "",
                    "step": cap.get("abs_pos"),
                    "h": cap["h"].detach().float().cpu(),
                    "av_text": cap.get("av_text", "") or "",
                }
            )

    p = ARTIFACTS / "country_concept_vector.pt"
    if p.exists():
        data = torch.load(p, weights_only=False)
        for i, h in enumerate(data["h_country"]):
            items.append(
                {
                    "src": "country_src",
                    "prompt_id": f"country_{i}",
                    "token": "",
                    "step": None,
                    "h": h.detach().float().cpu(),
                    "av_text": "",
                }
            )
        for i, h in enumerate(data["h_non_country"]):
            items.append(
                {
                    "src": "non_country_src",
                    "prompt_id": f"non_country_{i}",
                    "token": "",
                    "step": None,
                    "h": h.detach().float().cpu(),
                    "av_text": "",
                }
            )
        for label, (_, h) in data["h_test"].items():
            items.append(
                {
                    "src": "country_test",
                    "prompt_id": label,
                    "token": "",
                    "step": None,
                    "h": h.detach().float().cpu(),
                    "av_text": "",
                }
            )
    return items


def classify_dim_character(stats: dict[str, float]) -> str:
    """Classify a hot dim by its sign-locking and burstiness profile.

    Returns one of: 'sink', 'polarized', 'feature', 'rare_burst', 'background'.

    Decision tree (cuts chosen from the observed top-20 spread):

      sink        : freq_top100 > 0.75 AND sign-locked (sign_consist <= 0.05
                    OR sign_consist >= 0.95) AND cv_abs < 0.55.
                    Always-on, sign-locked, low magnitude variation.
                    The attractor / norm-artifact case (2570, 458, 277, 1627).

      polarized   : sign-locked (<= 0.10 OR >= 0.90) but doesn't meet sink
                    criteria (either rarer in top-100 or higher cv_abs).
                    Same direction every time, but only fires sometimes
                    (3281, 3311, 3206).

      feature     : sign genuinely flips with content (0.15 <= sign_consist
                    <= 0.85) AND cv_abs >= 0.55. Content-bearing axis;
                    direction depends on what the token is doing (32, 608,
                    2604, 1790, 20).

      rare_burst  : freq_top5 < 0.05 AND cv_abs >= 0.9 — fires rarely but
                    huge when it does. Feature-detector-style dim (662, 392).

      background  : nothing distinctive — moderate freq, no sign lock, no
                    burst structure.
    """
    s = stats["sign_consist"]
    cv = stats["cv_abs"]
    f5 = stats["freq_top5"]
    f100 = stats["freq_top100"]
    sign_locked = (s <= 0.05) or (s >= 0.95)
    near_locked = (s <= 0.10) or (s >= 0.90)

    if f5 < 0.05 and cv >= 0.9:
        return "rare_burst"
    if sign_locked and cv < 0.55 and f100 > 0.75:
        return "sink"
    if near_locked:
        return "polarized"
    if 0.15 <= s <= 0.85 and cv >= 0.55:
        return "feature"
    return "background"


def per_dim_stats(
    H: torch.Tensor, counts_top5: Counter, counts_top100: Counter, n: int
) -> dict[int, dict[str, float]]:
    """Pre-compute the stats that classify_dim_character() uses."""
    H_abs = H.abs()
    mean_val = H.mean(dim=0)
    mean_abs = H_abs.mean(dim=0)
    std_abs = H_abs.std(dim=0)
    max_abs, _ = H_abs.max(dim=0)
    median_abs, _ = H_abs.median(dim=0)
    sign_consist = (H >= 0).float().mean(dim=0)
    cv_abs = std_abs / (mean_abs + 1e-9)
    out: dict[int, dict[str, float]] = {}
    seen = set(counts_top5.keys()) | set(counts_top100.keys())
    for idx in seen:
        out[idx] = {
            "freq_top5": counts_top5.get(idx, 0) / n,
            "freq_top100": counts_top100.get(idx, 0) / n,
            "mean_val": float(mean_val[idx].item()),
            "mean_abs": float(mean_abs[idx].item()),
            "max_abs": float(max_abs[idx].item()),
            "sign_consist": float(sign_consist[idx].item()),
            "cv_abs": float(cv_abs[idx].item()),
            "median_over_max": float(
                median_abs[idx].item() / (max_abs[idx].item() + 1e-9)
            ),
        }
    return out


def main() -> None:
    items = load_all()
    n = len(items)
    print(f"loaded {n} captures")

    H = torch.stack([it["h"] for it in items])  # (n, 3584)
    print(f"H matrix: {tuple(H.shape)}")

    # --- pairwise cosines ---
    Hn = torch.nn.functional.normalize(H, dim=1)
    C = Hn @ Hn.T  # (n, n)
    print(
        f"\nmean off-diagonal cosine: {C[~torch.eye(n, dtype=torch.bool)].mean().item():+.3f}"
    )
    print(f"min cosine: {C[~torch.eye(n, dtype=torch.bool)].min().item():+.3f}")
    print(f"max cosine: {C[~torch.eye(n, dtype=torch.bool)].max().item():+.3f}")

    # top-K and bottom-K pairs
    iu = torch.triu_indices(n, n, offset=1)
    flat = C[iu[0], iu[1]]
    sorted_vals, sorted_pos = torch.sort(flat, descending=True)

    def show(
        title: str, positions: torch.Tensor, values: torch.Tensor, k: int = 20
    ) -> None:
        print(f"\n--- {title} ---")
        for j in range(k):
            pos = int(positions[j].item())
            i_a, i_b = int(iu[0][pos].item()), int(iu[1][pos].item())
            a, b = items[i_a], items[i_b]
            print(
                f"  cos={values[j].item():+.3f}  "
                f"[{a['src']}/{a['prompt_id']:<14} {a['token']!r:<14}] <-> "
                f"[{b['src']}/{b['prompt_id']:<14} {b['token']!r:<14}]"
            )

    show("top-20 most-similar pairs", sorted_pos[:20], sorted_vals[:20])
    show(
        "top-20 least-similar pairs",
        sorted_pos[-20:].flip(0),
        sorted_vals[-20:].flip(0),
    )

    # intra/cross-source means
    src_labels = [it["src"] for it in items]
    unique_srcs = sorted(set(src_labels))
    print("\n--- per-source mean cosine to same-source vs different-source ---")
    for s in unique_srcs:
        mask_s = torch.tensor([lbl == s for lbl in src_labels])
        if mask_s.sum() < 2:
            continue
        intra = C[mask_s][:, mask_s]
        eye = torch.eye(int(mask_s.sum().item()), dtype=torch.bool)
        intra_mean = intra[~eye].mean().item()
        cross = C[mask_s][:, ~mask_s].mean().item()
        n_s = int(mask_s.sum().item())
        print(
            f"  {s:<18} (n={n_s:>3}):  intra={intra_mean:+.3f}   cross={cross:+.3f}   diff={intra_mean - cross:+.3f}"
        )

    # --- hot-dim census ---
    counts_top5: Counter = Counter()
    counts_top20: Counter = Counter()
    counts_top100: Counter = Counter()
    rank_weighted: dict[int, float] = {}
    magnitude_weighted: dict[int, float] = {}

    for it in items:
        absh = it["h"].abs()
        top5_v, top5_i = absh.topk(5)
        top20_i = absh.topk(20).indices
        top100_i = absh.topk(100).indices
        counts_top5.update(top5_i.tolist())
        counts_top20.update(top20_i.tolist())
        counts_top100.update(top100_i.tolist())
        for rank, (idx, val) in enumerate(zip(top5_i.tolist(), top5_v.tolist())):
            rank_weighted[idx] = rank_weighted.get(idx, 0.0) + (5 - rank)
            magnitude_weighted[idx] = magnitude_weighted.get(idx, 0.0) + abs(val)

    print(f"\n--- hot-dim census (across {n} captures) ---")
    print("top-20 by appearances-in-top-5:")
    for idx, c in counts_top5.most_common(20):
        rk = rank_weighted.get(idx, 0.0)
        mag = magnitude_weighted.get(idx, 0.0)
        print(
            f"  idx {idx:>5}: top5_count={c:>4}/{n}   rank_weighted={rk:>6.1f}   mag_sum={mag:>9.2f}"
        )

    print("\ntop-20 by appearances-in-top-100:")
    for idx, c in counts_top100.most_common(20):
        print(f"  idx {idx:>5}: top100_count={c:>4}/{n}")

    # --- PCA ---
    print("\n--- PCA over 167 captures ---")
    mean_h = H.mean(dim=0, keepdim=True)
    H_centered = H - mean_h
    U, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
    total_var = (S**2).sum().item()
    print(f"  total variance: {total_var:.2f}")
    cum = 0.0
    for i in range(min(20, len(S))):
        var_i = float((S[i] ** 2).item())
        cum += var_i
        print(
            f"  PC{i + 1:>2}: sigma={S[i].item():>9.2f}  var={var_i:>10.2f}  frac={var_i / total_var:>5.3f}  cum={cum / total_var:>5.3f}"
        )

    # --- per-dim statistics for the human-defined classifier ---
    stats_by_dim = per_dim_stats(H, counts_top5, counts_top100, n)
    top_dims = [idx for idx, _ in counts_top5.most_common(20)]
    print("\n--- per-dim stats for top-20 hot dims ---")
    print(
        f"  {'idx':>5}  {'frq5':>5}  {'frq100':>6}  {'meanV':>8}  {'meanA':>7}  {'maxA':>7}  {'sign+':>6}  {'cv_abs':>7}  {'med/max':>7}"
    )
    for idx in top_dims:
        s = stats_by_dim[idx]
        print(
            f"  {idx:>5}  {s['freq_top5']:>5.2f}  {s['freq_top100']:>6.2f}  "
            f"{s['mean_val']:>+8.2f}  {s['mean_abs']:>7.2f}  {s['max_abs']:>7.2f}  "
            f"{s['sign_consist']:>6.3f}  {s['cv_abs']:>7.2f}  {s['median_over_max']:>7.3f}"
        )

    print("\n--- classify_dim_character() over top-20 hot dims ---")
    labels: dict[int, str] = {}
    for idx in top_dims:
        try:
            labels[idx] = classify_dim_character(stats_by_dim[idx])
        except NotImplementedError as e:
            print(f"  (classifier not yet implemented: {e})")
            break
    if labels:
        for idx, lbl in labels.items():
            print(f"  idx {idx:>5}: {lbl}")

    torch.save(
        {
            "items_meta": [{k: v for k, v in it.items() if k != "h"} for it in items],
            "H": H,
            "C": C,
            "PCA_U": U[:, :20].clone(),
            "PCA_S": S.clone(),
            "PCA_Vh": Vh[:20, :].clone(),
            "mean_h": mean_h.squeeze(0),
            "counts_top5": dict(counts_top5),
            "counts_top20": dict(counts_top20),
            "counts_top100": dict(counts_top100),
            "rank_weighted": rank_weighted,
            "magnitude_weighted": magnitude_weighted,
            "stats_by_dim": stats_by_dim,
            "labels": labels,
        },
        OUT,
    )
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
