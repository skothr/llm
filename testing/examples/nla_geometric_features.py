"""Per-capture geometric features across all 167 h-vectors on disk.

For each capture across all four artifacts, compute:

* ||h||_2, ||h||_inf, ||h||_1 / ||h||_2 (a sparsity proxy)
* kurtosis (heavy-tail indicator)
* top-5 component indices and their magnitudes (do hot dims repeat?)
* fraction of energy in top-1% of components (concentration)
* sign-balance (fraction positive components)
* if h_pred available: cos(h, h_pred), ||h - h_pred|| / ||h||
* if it's a generation token: argmax-next, AV text first ~80 chars

Output as a long table to stdout AND as a torch artifact for downstream
plotting.
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


ARTIFACTS = Path("testing/.cache/nla_artifacts")
FEATURES_OUT = ARTIFACTS / "geometric_features.pt"
D_MODEL = 3584


def _h_features(h: torch.Tensor) -> dict[str, Any]:
    h = h.detach().float().cpu()
    abs_h = h.abs()
    norm2 = float(h.norm().item())
    norm_inf = float(abs_h.max().item())
    norm1 = float(abs_h.sum().item())
    sparsity = norm1 / norm2 if norm2 > 0 else 0.0
    mean = float(h.mean().item())
    var = float(h.var().item())
    centered = h - mean
    kurt = float((centered.pow(4).mean() / (var ** 2 + 1e-12) - 3.0).item())
    top5_vals, top5_idxs = abs_h.topk(5)
    cutoff = max(1, D_MODEL // 100)  # top 1%
    top1pct_energy = float(abs_h.topk(cutoff).values.pow(2).sum().item()) / float(h.pow(2).sum().item())
    sign_balance = float((h > 0).float().mean().item())
    return {
        "norm2": norm2,
        "norm_inf": norm_inf,
        "sparsity": sparsity,           # ||h||_1 / ||h||_2; sqrt(d_model) ~= 59.87 ceiling
        "kurtosis": kurt,
        "top5_idxs": top5_idxs.tolist(),
        "top5_vals": [float(v) for v in top5_vals.tolist()],
        "top1pct_energy": top1pct_energy,
        "sign_balance": sign_balance,
    }


def _ar_features(h: torch.Tensor, h_pred: torch.Tensor) -> dict[str, float]:
    h = h.detach().float().cpu()
    h_pred = h_pred.detach().float().cpu()
    cos = float(torch.nn.functional.cosine_similarity(h, h_pred, dim=0).item())
    resid = h - h_pred
    rel_err = float(resid.norm().item() / (h.norm().item() + 1e-12))
    return {"cos": cos, "rel_err": rel_err}


def main() -> None:
    rows: list[dict[str, Any]] = []

    # --- aggregate_faithfulness.pt ---
    p = ARTIFACTS / "aggregate_faithfulness.pt"
    if p.exists():
        data = torch.load(p, weights_only=False)
        for prompt in data["prompts"]:
            for cap in prompt.get("captures", []):
                row = {
                    "src": "aggregate",
                    "prompt_id": prompt["id"],
                    "prompt": prompt["text"],
                    "step": cap.get("step"),
                    "token": cap.get("token"),
                    "av_text_head": (cap.get("av_text", "") or "")[:100],
                }
                row.update(_h_features(cap["h"]))
                if cap.get("h_pred") is not None:
                    row.update(_ar_features(cap["h"], cap["h_pred"]))
                rows.append(row)

    # --- rabbit_haiku_gen_trajectory.pt ---
    p = ARTIFACTS / "rabbit_haiku_gen_trajectory.pt"
    if p.exists():
        data = torch.load(p, weights_only=False)
        for cap in data.get("captures", []):
            row = {
                "src": "haiku_gen",
                "prompt_id": "rabbit_haiku",
                "prompt": data.get("prompt", ""),
                "step": cap.get("step"),
                "token": cap.get("token"),
                "av_text_head": (cap.get("av_text", "") or "")[:100],
            }
            row.update(_h_features(cap["h"]))
            if cap.get("h_pred") is not None:
                row.update(_ar_features(cap["h"], cap["h_pred"]))
            rows.append(row)

    # --- forced_continuation.pt ---
    p = ARTIFACTS / "forced_continuation.pt"
    if p.exists():
        data = torch.load(p, weights_only=False)
        for cap in data.get("captures", []):
            row = {
                "src": "forced",
                "prompt_id": f"{cap['pair_id']}/{cap['version']}",
                "prompt": cap.get("completion", ""),
                "step": cap.get("abs_pos"),
                "token": cap.get("actual_token"),
                "argmax_next": cap.get("argmax_next"),
                "av_text_head": (cap.get("av_text", "") or "")[:100],
            }
            row.update(_h_features(cap["h"]))
            rows.append(row)

    # --- country_concept_vector.pt ---
    p = ARTIFACTS / "country_concept_vector.pt"
    if p.exists():
        data = torch.load(p, weights_only=False)
        for i, h in enumerate(data["h_country"]):
            row = {"src": "country_src", "prompt_id": f"country_{i}", "prompt": "", "step": None, "token": None, "av_text_head": ""}
            row.update(_h_features(h))
            rows.append(row)
        for i, h in enumerate(data["h_non_country"]):
            row = {"src": "non_country_src", "prompt_id": f"non_country_{i}", "prompt": "", "step": None, "token": None, "av_text_head": ""}
            row.update(_h_features(h))
            rows.append(row)
        for label, (prompt, h) in data["h_test"].items():
            row = {"src": "country_test", "prompt_id": label, "prompt": prompt, "step": None, "token": None, "av_text_head": ""}
            row.update(_h_features(h))
            rows.append(row)

    # --- table dump ---
    print(f"{'src':<14}  {'prompt_id':<25}  {'token':<14}  {'||h||_2':>8}  {'||h||_inf':>10}  {'sparsity':>9}  {'kurt':>7}  {'top1%E':>7}  {'sign+':>6}  {'cos':>7}  {'relerr':>7}")
    print("-" * 160)
    for r in rows:
        cos = f"{r.get('cos'):+.3f}" if r.get("cos") is not None else "   -   "
        rel = f"{r.get('rel_err'):.3f}" if r.get("rel_err") is not None else "  -   "
        token = (r.get("token") or "")[:14]
        print(f"{r['src']:<14}  {r['prompt_id'][:25]:<25}  {token:<14}  "
              f"{r['norm2']:>8.2f}  {r['norm_inf']:>10.2f}  {r['sparsity']:>9.2f}  "
              f"{r['kurtosis']:>7.1f}  {r['top1pct_energy']:>7.3f}  {r['sign_balance']:>6.3f}  "
              f"{cos:>7}  {rel:>7}")

    # --- save artifact ---
    torch.save({"rows": rows, "d_model": D_MODEL}, FEATURES_OUT)
    print(f"\nWrote {len(rows)} feature rows to {FEATURES_OUT}")

    # --- quick summary ---
    print("\n--- summary statistics ---")
    norm2s = [r["norm2"] for r in rows]
    norminfs = [r["norm_inf"] for r in rows]
    kurts = [r["kurtosis"] for r in rows]
    top1pct = [r["top1pct_energy"] for r in rows]
    print(f"  ||h||_2  range:  [{min(norm2s):.2f}, {max(norm2s):.2f}]  median {sorted(norm2s)[len(norm2s)//2]:.2f}")
    print(f"  ||h||_inf range: [{min(norminfs):.2f}, {max(norminfs):.2f}]  median {sorted(norminfs)[len(norminfs)//2]:.2f}")
    print(f"  kurtosis range:  [{min(kurts):.1f}, {max(kurts):.1f}]  median {sorted(kurts)[len(kurts)//2]:.1f}")
    print(f"  top1% energy:    [{min(top1pct):.3f}, {max(top1pct):.3f}]  median {sorted(top1pct)[len(top1pct)//2]:.3f}")


if __name__ == "__main__":
    main()
