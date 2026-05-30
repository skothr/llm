"""fig28 setup: capture h[20] for 8 anchor words at 4 prefix-length
variants each — to measure discriminant-projection stability.

If the discriminant directions encode genuine semantic content, then
the same anchor word at end-of-prompt should project similarly
regardless of prefix length. If prefix-length dominates the readout,
the discriminants are unstable.

Captures per anchor: each context has anchor as the LAST content token
of the user message; capture h[20] at position -1 (same protocol as
the vocab atlas) so the comparison is apples-to-apples.

  - single: "{anchor}"
  - short:  "Word: {anchor}"
  - medium: "Tell me about {anchor}"
  - long:   "I want to discuss with you the following word, which is {anchor}"

Outputs: testing/.cache/nla_artifacts/discriminant_stability.pt
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

from llm_surgeon import surgery


BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
OUT = Path("testing/.cache/nla_artifacts/discriminant_stability.pt")


ANCHORS: list[tuple[str, str, str]] = [
    # (anchor_word, expected_category, group_label_for_anchor)
    ("France", "country", "country"),
    ("Paris", "capital", "capital"),
    ("happy", "emotion", "emotion"),
    ("function", "codemath", "codemath"),
    ("refuse", "refusal", "refusal"),
    ("the", "article", "article"),
    (".", "p_ender", "p_ender"),
    ("7", "number", "number"),
]


def context_prompts(anchor: str) -> dict[str, str]:
    """Returns {context_name: user_message_text} where anchor is always
    the LAST content token. Varies prefix length to test how the same
    anchor at end-of-prompt shifts under different prefix contexts."""
    return {
        "single": anchor,
        "short": f"Word: {anchor}",
        "medium": f"Tell me about {anchor}",
        "long": f"I want to discuss with you the following word, which is {anchor}",
    }


def find_anchor_position(tokenizer: Any, anchor_text: str, context_text: str) -> int:
    """INTENTIONALLY UNUSED — DO NOT WIRE WITHOUT RE-CAPTURING ALL STABILITY DATA.

    See the correction comment in `main()` (~L131) for full context. This
    helper was written for a planned alternative capture protocol (capture
    at the actual anchor token) but the committed `discriminant_stability.pt`
    artifact and all downstream analyses were produced under the
    "capture at position -1 of chat-templated input" protocol. Wiring this
    function in would silently change the protocol and invalidate all
    downstream stability comparisons (fig28 + the validation observation
    + the audit-locked numbers).

    Locate the LAST occurrence of anchor_text in the tokenized chat-
    templated prompt. Returns the position index of the anchor's final token."""
    anchor_end_in_context = context_text.rfind(anchor_text) + len(anchor_text)
    prefix_text = context_text[:anchor_end_in_context]

    full_msg_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": context_text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    prefix_msg_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prefix_text}],
        tokenize=True,
        add_generation_prompt=False,
    )

    # prefix tokens should match the first N tokens of full; find longest
    # common prefix length, where the anchor's last token lives at position N-1
    common = 0
    for a, b in zip(full_msg_ids, prefix_msg_ids):
        if a == b:
            common += 1
        else:
            break
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
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"]

            # Capture at position -1 of the chat-templated input (same
            # protocol as the vocab atlas).
            # CORRECTION (2026-05-29): the original comment claimed
            # "anchor is the LAST CONTENT TOKEN" at position -1. That
            # is not accurate — with `add_generation_prompt=True`, the
            # final tokens are `<|im_start|>assistant\n` (the prefix
            # the model is about to emit FROM). Position -1 is the
            # trailing newline of that assistant-turn opener.
            # What we are actually measuring is the model's
            # "what to say next given the user message" representation,
            # NOT the anchor token's representation. The stability
            # finding (ctx-cos varies by category) still holds because
            # all four contexts are measured at the same kind of
            # position — comparisons are internally consistent. The
            # framing in `2026-05-13-nla-discriminant-validation.md`
            # has been corrected to say "end-of-prompt" rather than
            # "anchor-token". The `find_anchor_position()` function
            # below was written for a planned alternative protocol
            # (capture at the actual anchor token); it is intentionally
            # unused and kept as a reference for any future re-run.
            anchor_pos = int(input_ids.shape[1]) - 1

            with torch.no_grad():
                out = model(
                    input_ids=input_ids, output_hidden_states=True, use_cache=False
                )
            h = (
                out.hidden_states[LAYER][0, anchor_pos, :]
                .detach()
                .float()
                .cpu()
                .clone()
            )
            captures.append(
                {
                    "anchor": anchor,
                    "expected_cat": expected_cat,
                    "group": group,
                    "context": ctx_name,
                    "context_text": ctx_text,
                    "anchor_pos": int(anchor_pos),
                    "n_input_tokens": int(input_ids.shape[1]),
                    "h": h,
                    "norm": float(h.norm().item()),
                }
            )
            print(
                f"  [{idx:>3}/{total}] anchor={anchor!r:<12} ctx={ctx_name:<8} "
                f"pos={anchor_pos:>3}/{input_ids.shape[1]:>3}  ||h||={h.norm().item():>6.2f}  "
                f"({time.time() - t0:.1f}s)"
            )
            del out

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.save(
        {
            "captures": captures,
            "anchors": [a for a, _, _ in ANCHORS],
            "base_id": BASE_ID,
            "layer": LAYER,
        },
        OUT,
    )
    print(f"\nWrote {len(captures)} captures to {OUT}")


if __name__ == "__main__":
    main()
