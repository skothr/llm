"""MAIN-71 (part 2) — Test plateau attractor strength via AR round-trip.

Take a plateau-mid h (from MAIN-34 dense interp at t=0.420, inside the
"Definition + Poem" hybrid plateau), AR-encode its AV text, capture the
new h_pred, and measure cosine + distances. If the plateau is a true
attractor, the round-trip should land near the original h (cos > 0.85).
If the plateau is a transit zone, h_pred will drift toward h_A or h_B.

Baseline comparisons: round-trip h_A (t=0.0) and h_B (t=1.0) — should
self-attract since AV(h_A) and AV(h_B) were produced from those h's
originally.

Output: testing/.cache/nla_artifacts/plateau_attractor_test.pt
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ["HF_HUB_OFFLINE"] = "1"
for _v in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    os.environ.pop(_v, None)

import sys
from io import TextIOWrapper
from typing import Any, cast

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

import gc
import time
from pathlib import Path
import torch

from llm_surgeon.probe import load_ar, nla_reconstruct, nla_score


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = _REPO_ROOT / "testing" / ".cache" / "nla_artifacts"
SOURCE = ARTIFACTS / "dense_interp_near_pivot.pt"
OUT = ARTIFACTS / "plateau_attractor_test.pt"


def main() -> None:
    print(f"loading dense_interp_near_pivot from {SOURCE}")
    src = torch.load(SOURCE, weights_only=False)
    steps = sorted(src["steps"], key=lambda s: s["t"])
    h_A = src["h_A"]
    h_B = src["h_B"]

    # Pick test points: plateau-mid + anchor A + anchor B
    plateau_step = next(s for s in steps if abs(s["t"] - 0.4200) < 1e-3)
    h_plateau = plateau_step["h_t"]
    av_plateau = plateau_step["av_text"]
    # Anchor A and B AV texts come from t=0.0 and t=1.0 steps
    h_A_step = next(s for s in steps if abs(s["t"] - 0.0) < 1e-3)
    h_B_step = next(s for s in steps if abs(s["t"] - 1.0) < 1e-3)
    av_A = h_A_step["av_text"]
    av_B = h_B_step["av_text"]

    targets = [
        ("plateau t=0.420", h_plateau, av_plateau),
        ("anchor A t=0.000", h_A, av_A),
        ("anchor B t=1.000", h_B, av_B),
    ]

    print(f"\nloading AR ...")
    t0 = time.time()
    backbone, value_head, tok, meta = load_ar()
    print(f"  AR loaded in {time.time() - t0:.1f}s")

    results: list[dict[str, Any]] = []
    print(f"\nAR-reconstructing each AV text and comparing to original h ...")
    for label, h_orig, av_text in targets:
        t0 = time.time()
        h_pred = nla_reconstruct(
            av_text,
            backbone=backbone,
            value_head=value_head,
            tok=tok,
            meta=meta,
        )
        scores = nla_score(h_orig, h_pred)
        # Distances to all three references
        cos_to_A = torch.nn.functional.cosine_similarity(
            h_pred.unsqueeze(0), h_A.unsqueeze(0), dim=1
        ).item()
        cos_to_B = torch.nn.functional.cosine_similarity(
            h_pred.unsqueeze(0), h_B.unsqueeze(0), dim=1
        ).item()
        cos_to_plateau = torch.nn.functional.cosine_similarity(
            h_pred.unsqueeze(0), h_plateau.unsqueeze(0), dim=1
        ).item()
        dt = time.time() - t0
        results.append(
            {
                "label": label,
                "h_orig": h_orig.detach().clone(),
                "h_pred": h_pred.detach().clone(),
                "av_text": av_text,
                "cosine_round_trip": scores["cosine"],
                "normalized_mse": scores["normalized_mse"],
                "cos_to_anchor_A": cos_to_A,
                "cos_to_anchor_B": cos_to_B,
                "cos_to_plateau": cos_to_plateau,
                "norm_orig": float(h_orig.norm().item()),
                "norm_pred": float(h_pred.norm().item()),
                "elapsed": dt,
            }
        )
        print(f"\n  {label}  ({dt:.0f}s)")
        print(f"    ||h_orig||      = {h_orig.norm().item():>6.2f}")
        print(f"    ||h_pred||      = {h_pred.norm().item():>6.2f}")
        print(f"    cosine round-trip (h_orig vs h_pred) = {scores['cosine']:+.4f}")
        print(f"    cos(h_pred, h_A_anchor)      = {cos_to_A:+.4f}")
        print(f"    cos(h_pred, h_B_anchor)      = {cos_to_B:+.4f}")
        print(f"    cos(h_pred, h_plateau_t=.42) = {cos_to_plateau:+.4f}")

    # Verdict
    print(f"\n{'=' * 70}")
    plateau_result = results[0]
    cos_rt = plateau_result["cosine_round_trip"]
    cos_to_A = plateau_result["cos_to_anchor_A"]
    cos_to_B = plateau_result["cos_to_anchor_B"]
    # NOTE: the +0.85 / +0.10 thresholds below are ad-hoc, not calibrated
    # against a noise baseline. cos_rt ~0.89 is also the typical AR+AV
    # round-trip self-consistency mean for any in-distribution h (see
    # `aggregate_faithfulness.pt`: mean +0.868). So "STRONG ATTRACTOR"
    # here means "round-trip stays close to original, comparable to the
    # pipeline's noise ceiling" — not a statistically tested basin claim.
    # A proper test bootstraps round-trip cosines across multiple
    # plateau-zone t values and compares the distribution to the
    # plateau-to-anchor margins.
    margin_to_nearest_anchor = cos_rt - max(cos_to_A, cos_to_B)
    print(f"PLATEAU ATTRACTOR VERDICT (heuristic, ad-hoc thresholds):")
    print(f"  plateau round-trip cosine: {cos_rt:+.4f}")
    print(f"  drift toward anchor A:     {cos_to_A:+.4f}")
    print(f"  drift toward anchor B:     {cos_to_B:+.4f}")
    print(
        f"  margin over nearest anchor: {margin_to_nearest_anchor:+.4f}  "
        f"(anchor self-margins are typically ~+0.25)"
    )
    if cos_rt > 0.85 and margin_to_nearest_anchor > 0.0:
        print(
            f"  → BASIN CANDIDATE: round-trip clears self-consistency threshold AND "
            f"is closer to plateau than either anchor. Confirms a basin direction; "
            f"narrowness of margin reflects basin width, not strength."
        )
    elif cos_rt > 0.85:
        print(
            f"  → SELF-CONSISTENT BUT NOT BASIN-LIKE: pipeline returns h close to "
            f"itself but closer to an anchor — looks like transit zone."
        )
    elif cos_to_A > cos_rt or cos_to_B > cos_rt:
        print(
            f"  → TRANSIT ZONE: AR re-encoding drifts toward {'A' if cos_to_A > cos_to_B else 'B'}"
        )
    else:
        print(f"  → WEAK SIGNAL: cos_rt {cos_rt:+.3f} not clearly in any basin")

    del backbone, value_head, tok
    gc.collect()

    torch.save({"results": results, "targets": [t[0] for t in targets]}, OUT)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
