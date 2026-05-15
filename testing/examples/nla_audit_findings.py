"""Rigorous audit: re-derive every load-bearing numerical claim from the
raw NLA artifacts and compare against what the observation files report.

Reads ONLY the 4 raw source files:
  * aggregate_faithfulness.pt
  * rabbit_haiku_gen_trajectory.pt
  * forced_continuation.pt
  * country_concept_vector.pt

Re-derives the full H matrix, pairwise cosines, hot-dim census,
classifier labels, PCA, CAV alignment, and counterfactual diff norms
from first principles. Compares each value against the claim in the
observation files. Reports PASS/FAIL per claim.

Run from worktree root.
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


ARTIFACTS = Path("testing/.cache/nla_artifacts")


# ---------------------------------------------------------------------------
# Audit harness
# ---------------------------------------------------------------------------
PASS, FAIL, ISSUES = 0, 0, []


def claim(name: str, ok: bool, expected: Any, actual: Any, tol: float = 0.0) -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS   {name}   expected={expected!r}   actual={actual!r}")
    else:
        FAIL += 1
        ISSUES.append((name, expected, actual))
        print(f"  FAIL   {name}   expected={expected!r}   actual={actual!r}   tol={tol}")


def claim_near(name: str, expected: float, actual: float, atol: float = 0.005) -> None:
    claim(name, abs(actual - expected) <= atol, expected, round(actual, 4), tol=atol)


def claim_eq(name: str, expected: Any, actual: Any) -> None:
    claim(name, expected == actual, expected, actual)


# ---------------------------------------------------------------------------
# Re-derive H matrix from raw sources
# ---------------------------------------------------------------------------
def load_raw() -> tuple[torch.Tensor, list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    p = ARTIFACTS / "aggregate_faithfulness.pt"
    data = torch.load(p, weights_only=False)
    for prompt in data["prompts"]:
        for cap in prompt.get("captures", []):
            items.append({
                "src": "aggregate", "prompt_id": prompt["id"],
                "token": cap.get("token") or "", "step": cap.get("step"),
                "h": cap["h"].detach().float().cpu(),
                "h_pred": cap.get("h_pred").detach().float().cpu() if cap.get("h_pred") is not None else None,
                "cosine": cap.get("cosine"),
                "av_text": cap.get("av_text", ""),
            })

    p = ARTIFACTS / "rabbit_haiku_gen_trajectory.pt"
    data = torch.load(p, weights_only=False)
    for cap in data.get("captures", []):
        items.append({
            "src": "haiku_gen", "prompt_id": "rabbit_haiku",
            "token": cap.get("token") or "", "step": cap.get("step"),
            "h": cap["h"].detach().float().cpu(),
            "h_pred": cap.get("h_pred").detach().float().cpu() if cap.get("h_pred") is not None else None,
            "cosine": cap.get("cosine"),
            "av_text": cap.get("av_text", ""),
        })

    p = ARTIFACTS / "forced_continuation.pt"
    data = torch.load(p, weights_only=False)
    for cap in data.get("captures", []):
        items.append({
            "src": "forced", "prompt_id": f"{cap['pair_id']}/{cap['version']}",
            "token": cap.get("actual_token") or "", "step": cap.get("abs_pos"),
            "h": cap["h"].detach().float().cpu(),
            "h_pred": None, "cosine": None, "av_text": cap.get("av_text", ""),
        })

    p = ARTIFACTS / "country_concept_vector.pt"
    data = torch.load(p, weights_only=False)
    for i, h in enumerate(data["h_country"]):
        items.append({"src": "country_src", "prompt_id": f"country_{i}",
                      "token": "", "step": None,
                      "h": h.detach().float().cpu(),
                      "h_pred": None, "cosine": None, "av_text": ""})
    for i, h in enumerate(data["h_non_country"]):
        items.append({"src": "non_country_src", "prompt_id": f"non_country_{i}",
                      "token": "", "step": None,
                      "h": h.detach().float().cpu(),
                      "h_pred": None, "cosine": None, "av_text": ""})
    for label, (_, h) in data["h_test"].items():
        items.append({"src": "country_test", "prompt_id": label,
                      "token": "", "step": None,
                      "h": h.detach().float().cpu(),
                      "h_pred": None, "cosine": None, "av_text": ""})

    H = torch.stack([it["h"] for it in items])
    return H, items


# ---------------------------------------------------------------------------
# Independent classifier (re-derives sink/feature labels from scratch)
# ---------------------------------------------------------------------------
def classify_dim(s: dict[str, float]) -> str:
    sc = s["sign_consist"]
    cv = s["cv_abs"]
    f5 = s["freq_top5"]
    f100 = s["freq_top100"]
    if f5 < 0.05 and cv >= 0.9:
        return "rare_burst"
    if (sc <= 0.05 or sc >= 0.95) and cv < 0.55 and f100 > 0.75:
        return "sink"
    if sc <= 0.10 or sc >= 0.90:
        return "polarized"
    if 0.15 <= sc <= 0.85 and cv >= 0.55:
        return "feature"
    return "background"


def main() -> None:
    print("=" * 80)
    print("AUDIT 1: dataset size and structure")
    print("=" * 80)
    H, items = load_raw()
    n, d = H.shape
    claim_eq("total captures", 167, n)
    claim_eq("hidden dim", 3584, d)
    src_counts = Counter(it["src"] for it in items)
    claim_eq("aggregate captures", 113, src_counts["aggregate"])
    claim_eq("haiku_gen captures", 15, src_counts["haiku_gen"])
    claim_eq("forced captures", 10, src_counts["forced"])
    claim_eq("country_src captures", 8, src_counts["country_src"])
    claim_eq("non_country_src captures", 8, src_counts["non_country_src"])
    claim_eq("country_test captures", 13, src_counts["country_test"])

    print()
    print("=" * 80)
    print("AUDIT 2: hot-dim census")
    print("=" * 80)
    counts_top5: Counter = Counter()
    counts_top100: Counter = Counter()
    for it in items:
        absh = it["h"].abs()
        counts_top5.update(absh.topk(5).indices.tolist())
        counts_top100.update(absh.topk(100).indices.tolist())

    top1 = counts_top5.most_common(1)[0]
    top2 = counts_top5.most_common(2)[1]
    claim_eq("most frequent top-5 dim", 2570, top1[0])
    claim_eq("second most frequent top-5 dim", 458, top2[0])
    claim_eq("dim 2570 top-5 count", 165, top1[1])
    claim_eq("dim 458 top-5 count", 165, top2[1])
    claim_near("dim 2570 top-5 frequency", 0.99, top1[1] / n, atol=0.01)

    # Verify top-100 inclusivity
    claim_eq("dim 2570 top-100 count", 167, counts_top100[2570])
    claim_eq("dim 458 top-100 count", 167, counts_top100[458])

    print()
    print("=" * 80)
    print("AUDIT 3: per-dim stats for the top-20 hot dims (full re-derivation)")
    print("=" * 80)
    H_abs = H.abs()
    mean_val = H.mean(dim=0)
    mean_abs = H_abs.mean(dim=0)
    std_abs = H_abs.std(dim=0)
    sign_consist = (H >= 0).float().mean(dim=0)
    cv_abs = std_abs / (mean_abs + 1e-9)

    top20 = [idx for idx, _ in counts_top5.most_common(20)]
    stats_by_dim: dict[int, dict[str, float]] = {}
    for idx in top20:
        stats_by_dim[idx] = {
            "freq_top5": counts_top5[idx] / n,
            "freq_top100": counts_top100[idx] / n,
            "mean_val": float(mean_val[idx].item()),
            "mean_abs": float(mean_abs[idx].item()),
            "sign_consist": float(sign_consist[idx].item()),
            "cv_abs": float(cv_abs[idx].item()),
        }

    # Check key dim stats claimed in fig3 / geometric-deep-dive
    claim_near("dim 2570 sign_consist", 0.000, stats_by_dim[2570]["sign_consist"], atol=0.001)
    claim_near("dim 2570 cv_abs",       0.33,  stats_by_dim[2570]["cv_abs"],       atol=0.01)
    claim_near("dim 2570 mean_val",    -38.64, stats_by_dim[2570]["mean_val"],     atol=0.1)
    claim_near("dim 458 sign_consist",  0.000, stats_by_dim[458]["sign_consist"],  atol=0.001)
    claim_near("dim 458 cv_abs",        0.21,  stats_by_dim[458]["cv_abs"],        atol=0.01)
    claim_near("dim 458 mean_val",     -29.00, stats_by_dim[458]["mean_val"],      atol=0.1)
    claim_near("dim 32 sign_consist",   0.299, stats_by_dim[32]["sign_consist"],   atol=0.005)
    claim_near("dim 32 cv_abs",         0.67,  stats_by_dim[32]["cv_abs"],         atol=0.01)
    claim_near("dim 608 sign_consist",  0.629, stats_by_dim[608]["sign_consist"],  atol=0.005)
    claim_near("dim 608 cv_abs",        0.65,  stats_by_dim[608]["cv_abs"],        atol=0.01)

    print()
    print("=" * 80)
    print("AUDIT 4: classifier labels (must match published 7 sinks + 8 features)")
    print("=" * 80)
    labels = {idx: classify_dim(s) for idx, s in stats_by_dim.items()}
    sinks = sorted([d for d, l in labels.items() if l == "sink"])
    feats = sorted([d for d, l in labels.items() if l == "feature"])
    polarized = sorted([d for d, l in labels.items() if l == "polarized"])
    rare_burst = sorted([d for d, l in labels.items() if l == "rare_burst"])
    bg = sorted([d for d, l in labels.items() if l == "background"])
    claim_eq("sink dims",       [277, 458, 1427, 1627, 2107, 2570, 3110], sinks)
    claim_eq("feature dims",    [20, 32, 392, 608, 1121, 1790, 2604, 2953], feats)
    claim_eq("polarized dims",  [3206, 3281, 3311], polarized)
    claim_eq("rare_burst dims", [662], rare_burst)
    claim_eq("background dims", [1116], bg)

    print()
    print("=" * 80)
    print("AUDIT 5: PCA variance fractions")
    print("=" * 80)
    mean_h = H.mean(dim=0, keepdim=True)
    Hc = H - mean_h
    _, S, _ = torch.linalg.svd(Hc, full_matrices=False)
    var = (S ** 2)
    total_var = float(var.sum().item())
    claim_near("PC1 variance fraction (original)", 0.165, float(var[0] / total_var), atol=0.005)
    claim_near("PC2 variance fraction (original)", 0.077, float(var[1] / total_var), atol=0.005)
    cumtop20 = float(var[:20].sum() / total_var)
    claim_near("top-20 PCs cumulative variance", 0.609, cumtop20, atol=0.005)

    print()
    print("=" * 80)
    print("AUDIT 6: pairwise cosines (mean, min, max, intra-source)")
    print("=" * 80)
    Hn = torch.nn.functional.normalize(H, dim=1)
    C = Hn @ Hn.T
    eye = torch.eye(n, dtype=torch.bool)
    off = C[~eye]
    claim_near("mean off-diag cosine (original)", 0.403, float(off.mean().item()), atol=0.005)
    claim_near("min cosine (original)",            0.054, float(off.min().item()),  atol=0.01)
    claim_near("max cosine (original)",            1.000, float(off.max().item()),  atol=0.001)

    src_labels = [it["src"] for it in items]
    def intra(s: str) -> float:
        mask = torch.tensor([lbl == s for lbl in src_labels])
        if mask.sum() < 2:
            return 0.0
        block = C[mask][:, mask]
        eye_b = torch.eye(int(mask.sum().item()), dtype=torch.bool)
        return float(block[~eye_b].mean().item())
    claim_near("aggregate intra cos",         0.397, intra("aggregate"),         atol=0.005)
    claim_near("country_src intra cos",       0.858, intra("country_src"),       atol=0.005)
    claim_near("non_country_src intra cos",   0.876, intra("non_country_src"),   atol=0.005)
    claim_near("haiku_gen intra cos",         0.530, intra("haiku_gen"),         atol=0.005)
    claim_near("forced intra cos",            0.366, intra("forced"),            atol=0.01)

    print()
    print("=" * 80)
    print("AUDIT 7: sink-removed cosines (mean, min, intra-aggregate, intra-country)")
    print("=" * 80)
    H_res = H.clone()
    for d_idx in sinks:
        H_res[:, d_idx] = 0.0
    Hn_res = torch.nn.functional.normalize(H_res, dim=1)
    C_res = Hn_res @ Hn_res.T
    off_res = C_res[~eye]
    claim_near("mean off-diag cosine (sink-removed)", 0.179, float(off_res.mean().item()), atol=0.005)
    claim_near("min cosine (sink-removed)",          -0.069, float(off_res.min().item()),  atol=0.01)
    def intra_res(s: str) -> float:
        mask = torch.tensor([lbl == s for lbl in src_labels])
        if mask.sum() < 2:
            return 0.0
        block = C_res[mask][:, mask]
        eye_b = torch.eye(int(mask.sum().item()), dtype=torch.bool)
        return float(block[~eye_b].mean().item())
    claim_near("aggregate intra cos (sink-removed)",       0.156, intra_res("aggregate"),       atol=0.005)
    claim_near("country_src intra cos (sink-removed)",     0.777, intra_res("country_src"),     atol=0.005)
    claim_near("non_country_src intra cos (sink-removed)", 0.804, intra_res("non_country_src"), atol=0.005)

    # Sink-removed PC1 fraction
    mean_h_res = H_res.mean(dim=0, keepdim=True)
    Hc_res = H_res - mean_h_res
    _, S_res, _ = torch.linalg.svd(Hc_res, full_matrices=False)
    var_res = (S_res ** 2)
    total_var_res = float(var_res.sum().item())
    claim_near("variance fraction kept after sink removal", 0.950, total_var_res / total_var, atol=0.005)
    claim_near("PC1 variance fraction (sink-removed)", 0.153, float(var_res[0] / total_var_res), atol=0.005)

    print()
    print("=" * 80)
    print("AUDIT 8: CAV alignment (H3 falsification)")
    print("=" * 80)
    cav_data = torch.load(ARTIFACTS / "country_concept_vector.pt", weights_only=False)
    direction_unit = cav_data["direction_unit"]
    claim_near("CAV ||direction_unit||", 1.000, float(direction_unit.norm().item()), atol=0.001)
    claim_near("cos(CAV_unit, e_32) = direction_unit[32]", 0.0510, float(direction_unit[32].item()), atol=0.0005)
    # H3 prediction: >= 0.4. Confirm falsification.
    claim("H3 (cos >= 0.4) falsified",
          abs(float(direction_unit[32].item())) < 0.4,
          "falsified", "falsified" if abs(float(direction_unit[32].item())) < 0.4 else "NOT falsified")

    # Top CAV contributor
    abs_c = direction_unit.abs()
    top_dim = int(abs_c.argmax().item())
    claim_eq("top CAV contributor dim", 1803, top_dim)
    claim_near("dim 1803 sq share", 0.0154, float((direction_unit[1803] ** 2).item()), atol=0.001)

    # Sinks in top 8 CAV contributors
    top8 = abs_c.topk(8).indices.tolist()
    sinks_in_top8 = [d for d in top8 if d in sinks]
    claim("sinks in top-8 CAV contributors >= 2",
          len(sinks_in_top8) >= 2, ">=2 sink dims", f"{len(sinks_in_top8)} sink dims: {sinks_in_top8}")

    print()
    print("=" * 80)
    print("AUDIT 9: AR cosine vs kurtosis count (deduplication check)")
    print("=" * 80)
    # fig5 plots aggregate (n=113) + haiku_gen (n=15) = 128 points,
    # but aggregate/creative_haiku duplicates haiku_gen exactly (cos=1.000).
    # How many distinct h-vectors actually have AR data?
    ar_items = [it for it in items if it.get("cosine") is not None]
    n_ar = len(ar_items)
    claim_eq("captures with AR data (plotted)", 128, n_ar)
    # Check overlap between aggregate/creative_haiku and haiku_gen
    haiku_aggregate = [it for it in ar_items if it["src"] == "aggregate" and it["prompt_id"] == "creative_haiku"]
    haiku_gen = [it for it in ar_items if it["src"] == "haiku_gen"]
    claim_eq("creative_haiku captures in aggregate", 15, len(haiku_aggregate))
    claim_eq("haiku_gen captures",                    15, len(haiku_gen))
    # Verify they're actually duplicates by sorting and comparing h-vectors
    h_agg_set = sorted(haiku_aggregate, key=lambda it: it["step"])
    h_gen_set = sorted(haiku_gen, key=lambda it: it["step"])
    all_match = True
    for a, b in zip(h_agg_set, h_gen_set):
        if not torch.allclose(a["h"], b["h"], atol=1e-5):
            all_match = False
            break
    claim("haiku captures duplicated across aggregate+haiku_gen",
          all_match, "duplicates", "duplicates" if all_match else "NOT duplicates")
    # Unique AR captures
    unique_ar = n_ar - 15  # the 15 haiku_gen are duplicates of creative_haiku
    claim_eq("unique captures with AR data", 113, unique_ar)

    print()
    print("=" * 80)
    print("AUDIT 10: counterfactual diff norms (fig16 corrected)")
    print("=" * 80)
    forced_data = torch.load(ARTIFACTS / "forced_continuation.pt", weights_only=False)
    nat: dict[str, dict[str, Any]] = {}
    forc_by_pair: dict[str, list[dict[str, Any]]] = {}
    for c in forced_data["captures"]:
        if c["version"] == "natural":
            nat[c["pair_id"]] = c
        else:
            forc_by_pair.setdefault(c["pair_id"], []).append(c)

    def norm_feat(diff: torch.Tensor) -> float:
        return float(diff[feats].norm().item())

    # singletons
    for pid, expected_token, expected_norm in [
        ("negation", "No", 5.72),
        ("factual", " Berlin", 8.70),
        ("math", "5", 10.97),
    ]:
        n_cap = nat[pid]
        f_cap = next(c for c in forc_by_pair[pid] if c["actual_token"].strip() == expected_token.strip())
        diff = f_cap["h"] - n_cap["h"]
        claim_near(f"{pid} ||Δh||_feat", expected_norm, norm_feat(diff), atol=0.05)

    # refusal_metaware position-matched
    n_cap = nat["refusal_metaware"]
    for tok, expected_norm in [(" sensing", 29.86), (" test", 28.06), (" refuse", 35.55)]:
        f_cap = next(c for c in forc_by_pair["refusal_metaware"] if c["actual_token"] == tok)
        diff = f_cap["h"] - n_cap["h"]
        claim_near(f"refusal_metaware {tok!r} ||Δh||_feat", expected_norm, norm_feat(diff), atol=0.05)

    print()
    print("=" * 80)
    print("AUDIT 11: Path B interpolation flipbook")
    print("=" * 80)
    flip_path = ARTIFACTS / "interpolation_flipbook.pt"
    if flip_path.exists():
        flip = torch.load(flip_path, weights_only=False)
        h_A_flip = flip["h_A"]
        h_B_flip = flip["h_B"]
        steps_flip = sorted(flip["steps"], key=lambda s: s["step"])
        claim_eq("interpolation step count", 20, len(steps_flip))
        anchor_cos = float(torch.nn.functional.cosine_similarity(h_A_flip, h_B_flip, dim=0).item())
        claim_near("anchor cosine cos(h_A, h_B)", 0.6905, anchor_cos, atol=0.0005)
        # Note: the Path B observation originally mislabeled 51.94 as ||h_A||;
        # 51.94 is actually ||h_A - h_B|| (the inter-anchor distance).
        claim_near("||h_A||", 65.73, float(h_A_flip.norm().item()), atol=0.05)
        claim_near("||h_B||", 66.30, float(h_B_flip.norm().item()), atol=0.05)
        claim_near("||h_A - h_B|| (inter-anchor distance)",
                   51.94, float((h_A_flip - h_B_flip).norm().item()), atol=0.05)
        # per-step distance should be constant for linear interp
        h_ts_flip = torch.stack([s["h_t"] for s in steps_flip])
        step_dists = (h_ts_flip[1:] - h_ts_flip[:-1]).norm(dim=1)
        claim_near("per-step ||Δh|| (mean)", 2.734, float(step_dists.mean().item()), atol=0.01)
        claim_near("per-step ||Δh|| (std, near-zero for linear interp)",
                   0.0, float(step_dists.std().item()), atol=0.01)
        # midpoint norm should dip below max(||h_A||, ||h_B||)
        midpoint_norm = float(h_ts_flip[len(steps_flip) // 2].norm().item())
        claim("midpoint ||h_t|| < min(||h_A||, ||h_B||) (geometric bow)",
              midpoint_norm < min(float(h_A_flip.norm().item()), float(h_B_flip.norm().item())),
              "below both anchor norms", f"midpoint={midpoint_norm:.2f}")
    else:
        print(f"  (skipping AUDIT 11: {flip_path} not present)")

    print()
    print("=" * 80)
    print("AUDIT 12: Vocab atlas size + PCA variance")
    print("=" * 80)
    vocab_path = ARTIFACTS / "vocab_atlas.pt"
    if vocab_path.exists():
        vocab = torch.load(vocab_path, weights_only=False)
        v_caps = vocab["captures"]
        claim_eq("vocab atlas capture count", 128, len(v_caps))
        claim_eq("vocab atlas category count", 23, len(vocab["categories"]))
        H_v = torch.stack([c["h"] for c in v_caps])
        H_v_res = H_v.clone()
        for d in sinks:
            H_v_res[:, d] = 0.0
        mean_v = H_v_res.mean(dim=0, keepdim=True)
        _, S_v, _ = torch.linalg.svd(H_v_res - mean_v, full_matrices=False)
        var_v = S_v ** 2
        total_var_v = float(var_v.sum().item())
        pc1_v = float(var_v[0] / total_var_v)
        pc2_v = float(var_v[1] / total_var_v)
        top3_v = float(var_v[:3].sum() / total_var_v)
        claim_near("vocab atlas PC1 fraction (sink-removed)", 0.335, pc1_v, atol=0.005)
        claim_near("vocab atlas PC2 fraction (sink-removed)", 0.153, pc2_v, atol=0.005)
        claim_near("vocab atlas top-3 cumulative variance",   0.559, top3_v, atol=0.005)
        # Intra-category cosines (sink-removed)
        H_v_unit = torch.nn.functional.normalize(H_v_res, dim=1)
        C_v = H_v_unit @ H_v_unit.T
        def intra_cat(cat: str) -> float:
            idxs = [i for i, c in enumerate(v_caps) if c["category"] == cat]
            if len(idxs) < 2:
                return 0.0
            block = C_v[idxs][:, idxs]
            eye_b = torch.eye(len(idxs), dtype=torch.bool)
            return float(block[~eye_b].mean().item())
        claim_near("capital intra-cos (sink-removed)", 0.983, intra_cat("capital"), atol=0.005)
        claim_near("country intra-cos (sink-removed)", 0.981, intra_cat("country"), atol=0.005)
        claim_near("emotion intra-cos (sink-removed, loosest)", 0.849, intra_cat("emotion"), atol=0.01)
        claim_near("refusal intra-cos (sink-removed)", 0.847, intra_cat("refusal"), atol=0.01)

        print()
        print("=" * 80)
        print("AUDIT 13: Discriminant directions (centroid → discriminant fix)")
        print("=" * 80)
        # Compute centroids and discriminants from scratch
        categories = list(vocab["categories"])
        cat_indices: dict[str, list[int]] = {
            cat: [i for i, c in enumerate(v_caps) if c["category"] == cat]
            for cat in categories
        }
        # Centroids (sink-removed, unit normalized)
        centroids: dict[str, torch.Tensor] = {}
        for cat in categories:
            cat_hs = H_v_res[cat_indices[cat]]
            cen = cat_hs.mean(dim=0)
            centroids[cat] = cen / (cen.norm() + 1e-9)
        cen_stack = torch.stack([centroids[c] for c in categories])
        C_cen = cen_stack @ cen_stack.T
        eye_cat = torch.eye(len(categories), dtype=torch.bool)
        claim_near("mean cross-centroid cosine (the all-axes-active problem)",
                   0.850, float(C_cen[~eye_cat].mean().item()), atol=0.005)
        grand_mean = cen_stack.mean(dim=0)
        claim_near("||grand_mean of centroids|| (would be 0.209 if spread)",
                   0.926, float(grand_mean.norm().item()), atol=0.005)

        # Discriminants (mean(cat) - mean(non-cat), sink-removed, unit normalized)
        discr: dict[str, torch.Tensor] = {}
        for cat in categories:
            in_idxs = cat_indices[cat]
            out_idxs = [i for i in range(len(v_caps)) if i not in in_idxs]
            in_mean = H_v_res[in_idxs].mean(dim=0)
            out_mean = H_v_res[out_idxs].mean(dim=0)
            d = in_mean - out_mean
            discr[cat] = d / (d.norm() + 1e-9)
        D_stack = torch.stack([discr[c] for c in categories])
        C_d = D_stack @ D_stack.T
        d_off = C_d[~eye_cat]
        claim_near("mean cross-discriminant cosine (after fix)",
                   0.006, float(d_off.mean().item()), atol=0.01)
        claim_near("min cross-discriminant cosine (country↔demonstrative opposite)",
                   -0.638, float(d_off.min().item()), atol=0.01)
        # country-capital sibling cosine
        ci = categories.index("country")
        cap_i = categories.index("capital")
        claim_near("country-capital discriminant cosine",
                   0.938, float(C_d[ci, cap_i].item()), atol=0.005)

        print()
        print("=" * 80)
        print("AUDIT 14: Stability scan (8 anchors × 4 contexts)")
        print("=" * 80)
        stab_path = ARTIFACTS / "discriminant_stability.pt"
        if stab_path.exists():
            stab = torch.load(stab_path, weights_only=False)
            stab_caps = stab["captures"]
            claim_eq("stability capture count", 32, len(stab_caps))
            by_anchor: dict[str, dict[str, torch.Tensor]] = {}
            for c in stab_caps:
                by_anchor.setdefault(c["anchor"], {})[c["context"]] = c["h"]
            contexts = ["single", "short", "medium", "long"]
            def ctx_cos_for(anchor: str) -> float:
                projs = []
                for ctx in contexts:
                    h = by_anchor[anchor][ctx].clone()
                    for d in sinks:
                        h[d] = 0.0
                    h_unit = h / (h.norm() + 1e-9)
                    proj = torch.tensor([float(torch.dot(h_unit, discr[c]).item()) for c in categories])
                    projs.append(proj)
                pv = torch.stack(projs)
                pv_unit = torch.nn.functional.normalize(pv, dim=1)
                cmat = pv_unit @ pv_unit.T
                eye_c = torch.eye(len(contexts), dtype=torch.bool)
                return float(cmat[~eye_c].mean().item())
            claim_near("ctx-cos for `the` (most stable)",    0.917, ctx_cos_for("the"),    atol=0.005)
            claim_near("ctx-cos for `happy` (least stable)", 0.399, ctx_cos_for("happy"),  atol=0.005)
            claim_near("ctx-cos for `France`",               0.851, ctx_cos_for("France"), atol=0.005)
            claim_near("ctx-cos for `refuse`",               0.589, ctx_cos_for("refuse"), atol=0.005)
            # expected_proj for happy → emotion should be weak (the prompt-topic finding)
            happy_emo_projs = []
            for ctx in contexts:
                h = by_anchor["happy"][ctx].clone()
                for d in sinks:
                    h[d] = 0.0
                h_unit = h / (h.norm() + 1e-9)
                happy_emo_projs.append(float(torch.dot(h_unit, discr["emotion"]).item()))
            claim_near("happy → emotion projection mean (weak; the topic-not-token finding)",
                       0.083, float(sum(happy_emo_projs) / len(happy_emo_projs)), atol=0.005)
        else:
            print(f"  (skipping AUDIT 14: {stab_path} not present)")

        # AUDIT 15: Mid-sequence vocab atlas (MAIN-44) — null result numbers
        print()
        print("AUDIT 15: Mid-sequence vocab atlas (MAIN-44 null result)")
        mid_path = ARTIFACTS / "mid_seq_vocab_atlas.pt"
        if mid_path.exists():
            mid = torch.load(mid_path, weights_only=False)
            mid_caps = mid["captures"]
            claim_eq("mid_seq capture count", 128, len(mid_caps))
            claim_eq("mid_seq skip count", 0, len(mid.get("skipped", [])))
            # Re-derive: discriminants are computed from end-of-prompt vocab atlas;
            # project mid-seq h's onto them and check aggregate signal/accuracy.
            assert discr is not None and sinks is not None
            mid_sigs: list[float] = []
            happy_mid_emo: float | None = None
            by_cat_sigs: dict[str, list[float]] = {}
            by_cat_correct: dict[str, list[bool]] = {}
            for cap in mid_caps:
                h = cap["h"].clone()
                for d in sinks:
                    h[d] = 0.0
                h_unit = h / (h.norm() + 1e-9)
                projs = {c: float(torch.dot(h_unit, discr[c]).item()) for c in categories}
                sig = projs[cap["category"]]
                mid_sigs.append(sig)
                by_cat_sigs.setdefault(cap["category"], []).append(sig)
                is_correct = max(projs.items(), key=lambda x: x[1])[0] == cap["category"]
                by_cat_correct.setdefault(cap["category"], []).append(is_correct)
                if cap["word"] == "happy" and cap["category"] == "emotion":
                    happy_mid_emo = sig
            # Per-CATEGORY means then unweighted mean across categories
            # (matches the aggregate row in the compare script's table)
            cat_sig_means = [sum(v) / len(v) for v in by_cat_sigs.values()]
            cat_acc_means = [sum(v) / len(v) for v in by_cat_correct.values()]
            mid_sig_agg = sum(cat_sig_means) / len(cat_sig_means)
            mid_acc_agg = sum(cat_acc_means) / len(cat_acc_means)
            claim_near("mid_seq aggregate within-class signal (~8x weaker than eop)",
                       0.0491, mid_sig_agg, atol=0.005)
            claim_near("mid_seq aggregate argmax accuracy (per-cat mean, vs 75.4% eop)",
                       0.3204, mid_acc_agg, atol=0.005)
            if happy_mid_emo is not None:
                claim_near("mid_seq happy → emotion signal", 0.0759, happy_mid_emo, atol=0.005)

            # AUDIT 16: Mid-seq NATIVE discriminants (MAIN-70)
            print()
            print("AUDIT 16: Mid-seq native discriminants (MAIN-70)")
            # Recompute discriminants from mid-seq captures
            mid_cats: dict[str, list[int]] = {
                cat: [i for i, c in enumerate(mid_caps) if c["category"] == cat]
                for cat in categories
            }
            H_mid = torch.stack([c["h"] for c in mid_caps])
            H_mid_res = H_mid.clone()
            for d in sinks:
                H_mid_res[:, d] = 0.0
            discr_mid: dict[str, torch.Tensor] = {}
            for cat in categories:
                in_idxs = mid_cats[cat]
                out_idxs = [i for i in range(len(mid_caps)) if i not in in_idxs]
                in_mean = H_mid_res[in_idxs].mean(dim=0)
                out_mean = H_mid_res[out_idxs].mean(dim=0)
                d_vec = in_mean - out_mean
                discr_mid[cat] = d_vec / (d_vec.norm() + 1e-9)
            # Recompute aggregate signal/acc with mid-h × mid-discr
            mid_native_sigs: dict[str, list[float]] = {}
            mid_native_correct: dict[str, list[bool]] = {}
            for cap, h_res in zip(mid_caps, H_mid_res):
                h_unit = h_res / (h_res.norm() + 1e-9)
                projs = {c: float(torch.dot(h_unit, discr_mid[c]).item()) for c in categories}
                mid_native_sigs.setdefault(cap["category"], []).append(projs[cap["category"]])
                is_correct = max(projs.items(), key=lambda x: x[1])[0] == cap["category"]
                mid_native_correct.setdefault(cap["category"], []).append(is_correct)
            cat_sig_means_n = [sum(v) / len(v) for v in mid_native_sigs.values()]
            cat_acc_means_n = [sum(v) / len(v) for v in mid_native_correct.values()]
            mid_native_sig = sum(cat_sig_means_n) / len(cat_sig_means_n)
            mid_native_acc = sum(cat_acc_means_n) / len(cat_acc_means_n)
            claim_near("mid-h × mid-discr in-protocol signal (40% higher than eop in-protocol)",
                       0.5632, mid_native_sig, atol=0.005)
            claim_near("mid-h × mid-discr argmax accuracy",
                       0.9710, mid_native_acc, atol=0.005)
            # Cross-protocol axis cosines
            D_eop_stack = torch.stack([discr[c] for c in categories])
            D_mid_stack = torch.stack([discr_mid[c] for c in categories])
            cross = D_eop_stack @ D_mid_stack.T
            diag = torch.diagonal(cross)
            claim_near("mean cross-protocol diagonal cosine (axis stability)",
                       0.0784, float(diag.mean().item()), atol=0.005)
            claim_near("max diagonal (emotion most-stable axis)",
                       0.1704, float(diag.max().item()), atol=0.005)
            emotion_idx = categories.index("emotion")
            claim_near("emotion-emotion cross-protocol cosine",
                       0.1704, float(cross[emotion_idx, emotion_idx].item()), atol=0.005)
        else:
            print(f"  (skipping AUDIT 15/16: {mid_path} not present)")

        # AUDIT 17: Concept arithmetic atlas (MAIN-48)
        print()
        print("AUDIT 17: Concept arithmetic atlas (MAIN-48)")
        arith_path = ARTIFACTS / "concept_arithmetic_atlas.pt"
        if arith_path.exists():
            arith = torch.load(arith_path, weights_only=False)
            combos_a = arith["combos"]
            claim_eq("concept_arithmetic combo count", 7, len(combos_a))
            claim_eq("target_norm", 150.0, float(arith["target_norm"]))
            # All combos have non-empty AV text + rescaled to TARGET_NORM
            for c_a in combos_a:
                claim("av_text non-empty for " + c_a["label"][:30],
                      bool(c_a.get("av_text", "").strip()), True, True)
                n_rescaled = float(c_a["h_rescaled"].norm().item())
                claim_near(f"||h_rescaled|| ≈ 150 for {c_a['label'][:30]}",
                           150.0, n_rescaled, atol=0.5)
        else:
            print(f"  (skipping AUDIT 17: {arith_path} not present)")

        # AUDIT 18: Dense interpolation near t=0.421 (MAIN-34)
        print()
        print("AUDIT 18: Dense interpolation near pivot (MAIN-34)")
        dense_path = ARTIFACTS / "dense_interp_near_pivot.pt"
        if dense_path.exists():
            dense = torch.load(dense_path, weights_only=False)
            d_steps = sorted(dense["steps"], key=lambda s: s["t"])
            claim_eq("dense_interp step count", 30, len(d_steps))
            # Dense zone bounds
            dense_zone = dense.get("dense_zone")
            claim_eq("dense_zone bounds", [0.395, 0.455], list(dense_zone))
            # Norm dip at midpoint of plateau
            mid_step = next(s for s in d_steps if abs(s["t"] - 0.4200) < 1e-3)
            mid_norm = float(mid_step["h_t"].norm().item())
            # Anchor pair magnitudes ~66; midpoint norm should be lower
            claim("midpoint norm < anchor norm (anti-parallel anchors)",
                  mid_norm < 65.0, "<65", mid_norm)
            # Check that the plateau decodes are stable (same AV text across t=[0.395, 0.4400])
            plateau_steps = [s for s in d_steps if 0.395 <= s["t"] <= 0.4400]
            plateau_first_lines = set()
            for s in plateau_steps:
                fl = (s["av_text"].splitlines() or [""])[0][:100]
                plateau_first_lines.add(fl)
            claim("plateau decodes stable across t∈[0.395, 0.4400] (<=3 unique first-lines)",
                  len(plateau_first_lines) <= 3,
                  "<=3 unique", len(plateau_first_lines))
        else:
            print(f"  (skipping AUDIT 18: {dense_path} not present)")
    else:
        print(f"  (skipping AUDIT 12/13/14/15/16/17/18: {vocab_path} not present)")

    print()
    print("=" * 80)
    print(f"SUMMARY:  {PASS} PASS  |  {FAIL} FAIL")
    print("=" * 80)
    if FAIL > 0:
        print("\nFAILED CLAIMS:")
        for name, exp, act in ISSUES:
            print(f"  - {name}: expected {exp!r}, got {act!r}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
