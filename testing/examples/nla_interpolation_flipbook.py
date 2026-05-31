"""Path B: semantic interpolation flipbook.

Two-phase, resumable:
  Phase 1: AR-encode two natural-language anchors -> h_A, h_B
           (loaded model: AR backbone + value head, ~5 B params)
  Phase 2: For each t in [0, 1/19, 2/19, ..., 1], compute
           h_t = (1-t) h_A + t h_B and AV-decode it
           (loaded model: AV, ~8 B params)

Memory: AR and AV don't fit in 31 GB simultaneously. The script loads
them sequentially and frees between phases.

Resumability: each phase saves to `interpolation_flipbook.pt`. If
phase 1's anchors are already saved, skipped. Phase 2 resumes from
whichever step was reached if interrupted mid-run.

Output: testing/.cache/nla_artifacts/interpolation_flipbook.pt
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ["HF_HUB_OFFLINE"] = "1"  # models cached locally; skip network metadata check
# httpx tries to init a SOCKS transport from ALL_PROXY at import time even when
# HF_HUB_OFFLINE is set, so clear proxy env vars for this process.
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
import torch

from _nla_artifacts import CACHE, find_artifact, write_artifact
from llm_surgeon.probe import (
    load_av,
    load_ar,
    nla_verbalize,
    nla_reconstruct,
)


ANCHOR_A_LABEL = "factual/geography"
ANCHOR_A_TEXT = (
    "Structured factual format with subject 'France' and predicate 'capital'. "
    "Final token 'Paris' completes a geography question."
)
ANCHOR_B_LABEL = "poetic/nature"
ANCHOR_B_TEXT = (
    "Structured poetic format with imagery of butterflies in spring. "
    "Final token 'flutter' completes a haiku-style line."
)
N_STEPS = 20


def phase1_anchors() -> dict[str, Any]:
    """Load AR, encode both anchor texts, free AR."""
    print(f"[phase 1] loading AR backbone + value head ...")
    t0 = time.time()
    backbone, value_head, tok, meta = load_ar()
    print(f"  AR loaded in {time.time() - t0:.0f}s")

    print(f"\n[phase 1] AR-encoding anchor A ({ANCHOR_A_LABEL}):")
    print(f"  {ANCHOR_A_TEXT!r}")
    t0 = time.time()
    h_A = (
        nla_reconstruct(
            ANCHOR_A_TEXT,
            backbone=backbone,
            value_head=value_head,
            tok=tok,
            meta=meta,
        )
        .detach()
        .float()
        .cpu()
    )
    print(f"  h_A: ||h||={h_A.norm().item():.2f}  ({time.time() - t0:.1f}s)")

    print(f"\n[phase 1] AR-encoding anchor B ({ANCHOR_B_LABEL}):")
    print(f"  {ANCHOR_B_TEXT!r}")
    t0 = time.time()
    h_B = (
        nla_reconstruct(
            ANCHOR_B_TEXT,
            backbone=backbone,
            value_head=value_head,
            tok=tok,
            meta=meta,
        )
        .detach()
        .float()
        .cpu()
    )
    print(f"  h_B: ||h||={h_B.norm().item():.2f}  ({time.time() - t0:.1f}s)")

    print(
        f"\n[phase 1] cos(h_A, h_B) = "
        f"{torch.nn.functional.cosine_similarity(h_A, h_B, dim=0).item():+.4f}"
    )
    print(f"[phase 1] ||h_A - h_B|| = {(h_A - h_B).norm().item():.2f}")

    print(f"[phase 1] freeing AR ...")
    del backbone, value_head, tok, meta
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "anchor_A_label": ANCHOR_A_LABEL,
        "anchor_A_text": ANCHOR_A_TEXT,
        "h_A": h_A,
        "anchor_B_label": ANCHOR_B_LABEL,
        "anchor_B_text": ANCHOR_B_TEXT,
        "h_B": h_B,
        "n_steps": N_STEPS,
        "steps": [],
    }


def phase2_av_decode(state: dict[str, Any]) -> None:
    """Load AV, for each remaining step compute h_t and verbalize, save after each."""
    h_A: torch.Tensor = state["h_A"]
    h_B: torch.Tensor = state["h_B"]
    n_steps: int = state["n_steps"]
    done_steps = {s["step"] for s in state.get("steps", [])}

    print(f"\n[phase 2] {len(done_steps)} of {n_steps} steps already done")
    if len(done_steps) >= n_steps:
        print("  nothing to do; all steps complete")
        return

    print(f"[phase 2] loading AV ({time.strftime('%H:%M:%S')}) ...")
    t0 = time.time()
    av_model, av_tok, av_meta = load_av()
    print(f"  AV loaded in {time.time() - t0:.0f}s")

    for step in range(n_steps):
        if step in done_steps:
            continue
        t = step / (n_steps - 1)
        h_t = (1.0 - t) * h_A + t * h_B
        print(
            f"\n[phase 2] step {step}/{n_steps - 1}  t={t:.3f}  ||h_t||={h_t.norm().item():.2f}"
        )
        t0 = time.time()
        av_text = nla_verbalize(
            h_t,
            model=av_model,
            tok=av_tok,
            meta=av_meta,
            max_new_tokens=200,
        )
        elapsed = time.time() - t0
        print(f"  AV ({elapsed:.0f}s):")
        for line in av_text.splitlines():
            print(f"    {line}")
        state["steps"].append(
            {
                "step": step,
                "t": t,
                "h_t": h_t,
                "av_text": av_text,
                "av_time": elapsed,
            }
        )
        out_path = write_artifact("interpolation_flipbook.pt")
        torch.save(state, out_path)
        print(f"  saved {len(state['steps'])}/{n_steps} steps to {out_path}")

    print(f"\n[phase 2] all {n_steps} steps complete; freeing AV")
    del av_model, av_tok, av_meta
    gc.collect()


def main() -> None:
    _existing = find_artifact("interpolation_flipbook.pt")
    if _existing is not None:
        state = torch.load(_existing, weights_only=False)
        # Compatibility: if anchor labels changed, restart
        if (
            state.get("anchor_A_text") != ANCHOR_A_TEXT
            or state.get("anchor_B_text") != ANCHOR_B_TEXT
            or state.get("n_steps") != N_STEPS
        ):
            print(
                f"[restart] saved state's anchors don't match this script's; "
                f"discarding and restarting phase 1"
            )
            state = phase1_anchors()
            torch.save(state, write_artifact("interpolation_flipbook.pt"))
        else:
            print(
                f"[resume] loaded state from {_existing}: {len(state.get('steps', []))} steps done"
            )
    else:
        state = phase1_anchors()
        torch.save(state, write_artifact("interpolation_flipbook.pt"))

    phase2_av_decode(state)

    print(f"\n[done] full flipbook saved to {CACHE / 'interpolation_flipbook.pt'}")
    print(
        f"[done] anchor cosine: {torch.nn.functional.cosine_similarity(state['h_A'], state['h_B'], dim=0).item():+.4f}"
    )
    print(
        f"[done] total wall time per step (median AV): "
        f"{sorted(s['av_time'] for s in state['steps'])[len(state['steps']) // 2]:.0f}s"
    )


if __name__ == "__main__":
    main()
