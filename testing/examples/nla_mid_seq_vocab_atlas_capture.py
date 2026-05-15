"""Mid-sequence vocab atlas capture (MAIN-44).

Tests MAIN-26's interpretation: the 23-discriminant basis classifies
prompt-TOPIC rather than token-presence because end-of-prompt h has
integrated the full user message into a topic representation. Capturing
h[20] at a MID-SEQUENCE anchor position — inside a carrier that
continues past the anchor — should preserve token-specific identity
before integration overwhelms it.

Carrier template:
    "The text contains many words. Here is one specific word: {anchor}.
     Typography matters significantly throughout."

Position finding: tokenize the chat-string truncated to anchor-end-char,
count tokens K -> anchor's last token sits at position (K-1) in the
full tokenization. Robust to BPE merges and anchor-as-carrier-substring
collisions because PRE is uniquely findable in the chat string.

Output: testing/.cache/nla_artifacts/mid_seq_vocab_atlas.pt
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

from llm_surgeon import surgery


BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
ARTIFACTS = Path("testing/.cache/nla_artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
OUT = ARTIFACTS / "mid_seq_vocab_atlas.pt"
VOCAB_SRC = ARTIFACTS / "vocab_atlas.pt"

PRE = "The text contains many words. Here is one specific word:"
SUFFIX = " continues throughout subsequent discussion paragraphs."


def find_anchor_end_token_pos(tok: Any, chat_str: str, anchor: str) -> tuple[int, int]:
    """Return (anchor_last_token_pos, anchor_first_token_pos) in full tokenization.

    The anchor span includes the leading space (so it can BPE-merge with the
    anchor token naturally). Cutpoint is RIGHT AFTER ':' to avoid the
    trailing-space merge problem — if `before` ended with " ", BPE would
    tokenize the trailing space as its own token, then re-merge it into
    ` anchor` in `with_anchor`, producing a zero-net-token-count diff.
    """
    pre_offset = chat_str.find(PRE)
    if pre_offset < 0:
        raise ValueError("PRE not found in chat_str")
    span_start = pre_offset + len(PRE)  # points at the space before the anchor
    expected_inserted = " " + anchor
    span_end = span_start + len(expected_inserted)
    if chat_str[span_start:span_end] != expected_inserted:
        raise ValueError(
            f"anchor span mismatch: expected {expected_inserted!r}, got "
            f"{chat_str[span_start:span_end]!r}"
        )
    before = chat_str[:span_start]
    with_anchor = chat_str[:span_end]
    enc_before = tok(before, add_special_tokens=False)
    enc_with = tok(with_anchor, add_special_tokens=False)
    k_before = len(enc_before["input_ids"])
    k_with = len(enc_with["input_ids"])
    if k_with <= k_before:
        raise ValueError(f"anchor consumed 0 tokens (before={k_before}, with={k_with})")
    return k_with - 1, k_before


def main() -> None:
    print(f"loading source vocab atlas from {VOCAB_SRC}")
    src = torch.load(VOCAB_SRC, weights_only=False)
    anchor_pairs = [(c["word"], c["category"]) for c in src["captures"]]
    print(f"  {len(anchor_pairs)} anchors across {len(src['categories'])} categories")

    print(f"loading base model {BASE_ID} (CPU bf16) ...")
    t0 = time.time()
    model, tok = surgery.load_model(BASE_ID, mode="bf16", device_map="cpu")
    print(f"  base loaded in {time.time() - t0:.0f}s")

    captures: list[dict[str, Any]] = []
    skipped: list[tuple[str, str, str]] = []
    total = len(anchor_pairs)
    print(f"\ncapturing h[{LAYER}] at mid-sequence anchor position for {total} anchors ...")

    for idx, (anchor, category) in enumerate(anchor_pairs, 1):
        t_start = time.time()
        carrier = f"{PRE} {anchor}{SUFFIX}"
        chat_str = tok.apply_chat_template(
            [{"role": "user", "content": carrier}],
            tokenize=False, add_generation_prompt=True,
        )
        try:
            anchor_last_pos, anchor_first_pos = find_anchor_end_token_pos(tok, chat_str, anchor)
        except ValueError as exc:
            print(f"  [{idx:>3}/{total}] SKIP {category}/{anchor!r}: {exc}")
            skipped.append((category, anchor, str(exc)))
            continue
        enc_full = tok(chat_str, return_tensors="pt", add_special_tokens=False)
        input_ids = enc_full["input_ids"]
        n_tok = int(input_ids.shape[1])
        # Sanity: anchor_last_pos must be < n_tok
        if anchor_last_pos >= n_tok:
            print(f"  [{idx:>3}/{total}] SKIP {category}/{anchor!r}: pos {anchor_last_pos} >= n_tok {n_tok}")
            skipped.append((category, anchor, "pos out of range"))
            continue
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[LAYER][0, anchor_last_pos, :].detach().float().cpu().clone()
        # Also capture the end-of-prompt for direct contrast on the same forward pass
        h_eop = out.hidden_states[LAYER][0, -1, :].detach().float().cpu().clone()
        captures.append({
            "category": category, "word": anchor,
            "h": h, "norm": float(h.norm().item()),
            "h_eop_same_prompt": h_eop,
            "norm_eop": float(h_eop.norm().item()),
            "n_input_tokens": n_tok,
            "anchor_first_pos": anchor_first_pos,
            "anchor_last_pos": anchor_last_pos,
            "anchor_n_tokens": anchor_last_pos - anchor_first_pos + 1,
            "carrier_str": carrier,
        })
        # Decode the anchor span tokens for debug
        anchor_ids = input_ids[0, anchor_first_pos:anchor_last_pos + 1].tolist()
        anchor_decoded = tok.decode(anchor_ids)
        print(f"  [{idx:>3}/{total}] {category:<15} {anchor!r:<22}  "
              f"pos={anchor_first_pos}-{anchor_last_pos}/{n_tok}  "
              f"toks={anchor_decoded!r:<24}  ||h||={h.norm().item():>6.2f}  "
              f"({time.time() - t_start:.1f}s)")
        del out

    del model, tok
    gc.collect()

    torch.save({
        "captures": captures,
        "categories": src["categories"],
        "base_id": BASE_ID,
        "layer": LAYER,
        "pre": PRE,
        "suffix": SUFFIX,
        "skipped": skipped,
    }, OUT)
    print(f"\nWrote {len(captures)} captures to {OUT}")
    print(f"Skipped: {len(skipped)}")
    if skipped:
        for cat, w, reason in skipped:
            print(f"  - {cat}/{w!r}: {reason}")


if __name__ == "__main__":
    main()
