"""Token-by-token walkthrough of every NLA capture we have on disk.

Reads all three artifact files (aggregate, gen-trajectory, forced-
continuation) and dumps a structured per-capture walkthrough. Output
is plain text designed for human review. Sections per artifact, per
prompt, per token.

The AV text is wrapped, the original cos / mse / argmax-next fields
are surfaced, and a one-line "observation" prompt is shown above each
to remind the reader what was being probed.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import Any, cast
cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

import textwrap
from pathlib import Path

import torch


ARTIFACTS_DIR = Path("testing/.cache/nla_artifacts")
WIDTH = 92


def _wrap(text: str, indent: str = "      ") -> str:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    wrapped = [
        textwrap.fill(
            p, width=WIDTH - len(indent),
            initial_indent=indent, subsequent_indent=indent,
            break_long_words=False, break_on_hyphens=False,
        )
        for p in paragraphs
    ]
    return "\n\n".join(wrapped)


def _header(s: str, char: str = "=") -> None:
    print()
    print(char * WIDTH)
    print(s)
    print(char * WIDTH)


def dump_aggregate(artifact: dict[str, Any]) -> None:
    _header("ARTIFACT: aggregate_faithfulness.pt  (h[20] across generation, 8 prompts)")
    for p in artifact["prompts"]:
        if not p.get("captures"):
            continue
        _header(f"PROMPT [{p['id']}]  {p['text']!r}", char="-")
        print(f"  Generated:      {p.get('generated_text', '?')!r}")
        if p.get("gen_tokens"):
            print(f"  Generated tokens ({len(p['gen_tokens'])}): {p['gen_tokens']}")
        print()
        for c in p["captures"]:
            cos = c.get("cosine")
            mse = c.get("normalized_mse")
            cos_str = f"{cos:+.3f}" if cos is not None else "(no AR)"
            mse_str = f"{mse:.2f}" if mse is not None else "-"
            norm = float(c["h"].norm().item())
            print(f"  --- step {c['step']:>2}  token={c['token']!r:<14}  "
                  f"||h||={norm:>6.2f}  cos={cos_str}  mse={mse_str}")
            if c.get("av_text"):
                print(_wrap(c["av_text"]))
            print()


def dump_gen_trajectory(artifact: dict[str, Any]) -> None:
    _header("ARTIFACT: rabbit_haiku_gen_trajectory.pt  (single-prompt gen trajectory)")
    print(f"  Prompt:    {artifact.get('prompt', '?')!r}")
    print(f"  Generated: {artifact.get('generated_text', '?')!r}")
    print()
    for c in artifact.get("captures", []):
        cos = c.get("cosine")
        cos_str = f"{cos:+.3f}" if cos is not None else "(no AR)"
        norm = float(c["h"].norm().item())
        print(f"  --- step {c['step']:>2}  token={c['token']!r:<14}  "
              f"||h||={norm:>6.2f}  cos={cos_str}")
        if c.get("av_text"):
            print(_wrap(c["av_text"]))
        print()


def dump_forced(artifact: dict[str, Any]) -> None:
    _header("ARTIFACT: forced_continuation.pt  (teacher-forcing probe, 4 pairs)")
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for c in artifact["captures"]:
        by_pair.setdefault(c["pair_id"], []).append(c)
    for pair_id, caps in by_pair.items():
        pair = next(p for p in artifact["pairs"] if p["id"] == pair_id)
        _header(f"PAIR [{pair_id}]", char="-")
        print(f"  prompt:  {pair['prompt']!r}")
        print(f"  natural: {pair['natural']!r}")
        print(f"  forced:  {pair['forced']!r}")
        print()
        for c in caps:
            print(f"  --- {c['version'].upper():<7}  target={c['target']!r:<14}  "
                  f"pos={c['abs_pos']:>3}  ||h||={c['norm']:>6.2f}  "
                  f"argmax-next={c['argmax_next']!r}")
            if c.get("av_text"):
                print(_wrap(c["av_text"]))
            print()


def main() -> None:
    agg_path = ARTIFACTS_DIR / "aggregate_faithfulness.pt"
    haiku_path = ARTIFACTS_DIR / "rabbit_haiku_gen_trajectory.pt"
    forced_path = ARTIFACTS_DIR / "forced_continuation.pt"

    if agg_path.exists():
        dump_aggregate(torch.load(agg_path, weights_only=False))
    else:
        print(f"(no aggregate artifact at {agg_path})")

    if haiku_path.exists():
        dump_gen_trajectory(torch.load(haiku_path, weights_only=False))
    else:
        print(f"(no haiku artifact at {haiku_path})")

    if forced_path.exists():
        dump_forced(torch.load(forced_path, weights_only=False))
    else:
        print(f"(no forced-continuation artifact at {forced_path})")

    print()
    print("=" * WIDTH)
    print("end of walkthrough")
    print("=" * WIDTH)


if __name__ == "__main__":
    main()
