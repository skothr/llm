"""MAIN-34 — Dense interpolation near t=0.421 pivot.

Tests whether MAIN-25's stepwise category transition at t=0.421 stays
sharp at ~10× resolution or smooths out (would mean the 20-step
flipbook was undersampled).

Reuses cached h_A, h_B from `interpolation_flipbook.pt` (no AR reload).
AV-decodes each interpolated h.

t-schedule (30 points):
  Sparse context:   {0.0, 0.25, 0.5, 0.75, 1.0}                  (5 pts)
  Dense pivot zone: linspace(0.395, 0.455, 25)  Δt ≈ 0.0025       (25 pts)

Output: testing/.cache/nla_artifacts/dense_interp_near_pivot.pt
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ["HF_HUB_OFFLINE"] = "1"
for _v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_v, None)

import sys
from io import TextIOWrapper
from typing import Any, cast
cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

import gc
import time
from pathlib import Path
import torch

from llm_surgeon.probe import load_av, nla_verbalize


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = _REPO_ROOT / "testing" / ".cache" / "nla_artifacts"
OUT = ARTIFACTS / "dense_interp_near_pivot.pt"
SOURCE = ARTIFACTS / "interpolation_flipbook.pt"


def build_t_schedule() -> list[float]:
    sparse = [0.0, 0.25, 0.5, 0.75, 1.0]
    # 25 points linearly spaced over [0.395, 0.455]
    n_dense = 25
    lo, hi = 0.395, 0.455
    dense = [lo + (hi - lo) * i / (n_dense - 1) for i in range(n_dense)]
    all_ts = sorted(set(sparse + dense))
    return all_ts


def main() -> None:
    print(f"loading source flipbook from {SOURCE}")
    src = torch.load(SOURCE, weights_only=False)
    h_A: torch.Tensor = src["h_A"]
    h_B: torch.Tensor = src["h_B"]
    print(f"  anchor A label: {src['anchor_A_label']!r}")
    print(f"  anchor B label: {src['anchor_B_label']!r}")
    print(f"  ||h_A|| = {h_A.norm().item():.2f}, ||h_B|| = {h_B.norm().item():.2f}")
    print(f"  cos(h_A, h_B) = "
          f"{torch.nn.functional.cosine_similarity(h_A, h_B, dim=0).item():+.4f}")

    ts = build_t_schedule()
    print(f"\nt-schedule: {len(ts)} points")
    for t in ts:
        marker = "  DENSE" if 0.395 <= t <= 0.455 else ""
        print(f"  t = {t:.4f}{marker}")

    print(f"\nloading AV ...")
    t0 = time.time()
    av_model, av_tok, av_meta = load_av()
    print(f"  AV loaded in {time.time() - t0:.1f}s")

    steps: list[dict[str, Any]] = []
    print(f"\nAV-decoding {len(ts)} interpolated h-vectors "
          f"(~{len(ts) * 85 / 60:.0f} min projected) ...")
    for i, t in enumerate(ts, 1):
        t_start = time.time()
        h_t = (1.0 - t) * h_A + t * h_B
        av_text = nla_verbalize(
            h_t, model=av_model, tok=av_tok, meta=av_meta, max_new_tokens=180,
        )
        first_line = (av_text.splitlines() or [""])[0][:120]
        elapsed = time.time() - t_start
        print(f"  [{i:>3}/{len(ts)}] t={t:.4f}  ({elapsed:.0f}s)  "
              f"first-line: {first_line}")
        steps.append({
            "step": i - 1, "t": t, "h_t": h_t.detach().clone(),
            "av_text": av_text,
        })

    del av_model, av_tok
    gc.collect()

    torch.save({
        "anchor_A_label": src.get("anchor_A_label"),
        "anchor_A_text": src.get("anchor_A_text"),
        "h_A": h_A,
        "anchor_B_label": src.get("anchor_B_label"),
        "anchor_B_text": src.get("anchor_B_text"),
        "h_B": h_B,
        "n_steps": len(steps),
        "steps": steps,
        "dense_zone": [0.395, 0.455],
    }, OUT)
    print(f"\nWrote {len(steps)} interpolated decodings to {OUT}")


if __name__ == "__main__":
    main()
