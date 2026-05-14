"""fig28 setup: capture h[20] for 8 anchor words at 4 different
contexts each — to measure discriminant-projection stability.

If the discriminant directions encode genuine semantic content, then
the same anchor word in different positions should project similarly
onto its category's discriminant. If position dominates content, the
projections will diverge wildly.

Captures per anchor (chat-templated; capture h[20] at the anchor's
position in the user message):
  - single: anchor IS the entire user message
  - subject: "{anchor} is interesting." (start position)
  - object: "I am thinking about {anchor}." (late position)
  - middle: "The word {anchor} appears here often." (mid-prompt)

Outputs: testing/.cache/nla_artifacts/discriminant_stability.pt
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
OUT = Path("testing/.cache/nla_artifacts/discriminant_stability.pt")


ANCHORS: list[tuple[str, str, str]] = [
    # (anchor_word, expected_category, group_label_for_anchor)
    ("France",   "country",  "country"),
    ("Paris",    "capital",  "capital"),
    ("happy",    "emotion",  "emotion"),
    ("function", "codemath", "codemath"),
    ("refuse",   "refusal",  "refusal"),
    ("the",      "article",  "article"),
    (".",        "p_ender",  "p_ender"),
    ("7",        "number",   "number"),
]


def context_prompts(anchor: str) -> dict[str, str]:
    """Returns {context_name: user_message_text} for the four contexts."""
    return {
        "single":   anchor,
        "subject":  f"{anchor} is interesting.",
        "object":   f"I am thinking about {anchor}.",
        "middle":   f"The word {anchor} appears here often.",
    }


def find_anchor_position(tokenizer: Any, full_prompt_ids: torch.Tensor,
                          anchor_text: str, context_text: str) -> int:
    """Locate the LAST occurrence of anchor_text in the tokenized
    user-message portion. Returns the position index in full_prompt_ids."""
    # Find the position of anchor in context (string-level)
    # We capture h at the LAST token corresponding to the anchor word.
    # Strategy: tokenize the context prefix up to the END of the anchor
    # word, and use that length as the position.
    anchor_end_in_context = context_text.rfind(anchor_text) + len(anchor_text)
    prefix_text = context_text[:anchor_end_in_context]

    # Tokenize the FULL chat-templated prompt up to where the user-message
    # contains only the prefix. We get the full chat-template ids and find
    # where prefix ends.
    full_msg_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": context_text}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )[0].tolist()
    prefix_msg_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prefix_text}],
        tokenize=True, add_generation_prompt=False, return_tensors="pt",
    )[0].tolist()

    # The prefix's token sequence should be a prefix of the full message
    # up through the anchor. Find the longest common prefix.
    common = 0
    for a, b in zip(full_msg_ids, prefix_msg_ids):
        if a == b:
            common += 1
        else:
            break

    # Return position (0-indexed) of the LAST anchor token in full_prompt_ids
    return common - 1


def main() -> None:
    print(f"loading base model {BASE_ID} (CPU bf16) ...")
    t0 = time.time()
    model, tok = surgery.load_model(BASE_ID, mode="bf16", device_map="cpu")
    print(f"  base loaded in {time.time() - t0:.0f}s")

    captures: list[dict[str, Any]] = []
    total = sum(4 for _ in ANCHORS)
    idx = 0
    for anchor, expected_cat, group in ANCHORS:
        for ctx_name, ctx_text in context_prompts(anchor).items():
            idx += 1
            t0 = time.time()
            enc = tok.apply_chat_template(
                [{"role": "user", "content": ctx_text}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt",
            )
            input_ids = enc["input_ids"]

            # Find where the anchor ends in this prompt
            if ctx_name == "single":
                # For single-token msg, capture at the last user-side token (same
                # protocol as vocab atlas)
                pos = int(input_ids.shape[1]) - 1
                # Adjust for the assistant turn tokens — actually for single
                # captures we want -1, matching the vocab atlas protocol exactly
                # so they're directly comparable. We'll use position -1 throughout.
                anchor_pos = pos
            else:
                anchor_pos = find_anchor_position(tok, input_ids[0], anchor, ctx_text)
                if anchor_pos < 0 or anchor_pos >= int(input_ids.shape[1]):
                    print(f"  [WARN] {anchor!r} {ctx_name}: position out of range; falling back to -1")
                    anchor_pos = int(input_ids.shape[1]) - 1

            with torch.no_grad():
                out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[LAYER][0, anchor_pos, :].detach().float().cpu().clone()
            captures.append({
                "anchor": anchor, "expected_cat": expected_cat, "group": group,
                "context": ctx_name, "context_text": ctx_text,
                "anchor_pos": int(anchor_pos),
                "n_input_tokens": int(input_ids.shape[1]),
                "h": h, "norm": float(h.norm().item()),
            })
            print(f"  [{idx:>3}/{total}] anchor={anchor!r:<12} ctx={ctx_name:<8} "
                  f"pos={anchor_pos:>3}/{input_ids.shape[1]:>3}  ||h||={h.norm().item():>6.2f}  "
                  f"({time.time() - t0:.1f}s)")
            del out

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.save({
        "captures": captures,
        "anchors": [a for a, _, _ in ANCHORS],
        "base_id": BASE_ID, "layer": LAYER,
    }, OUT)
    print(f"\nWrote {len(captures)} captures to {OUT}")


if __name__ == "__main__":
    main()
