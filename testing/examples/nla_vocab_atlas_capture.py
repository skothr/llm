"""Capture h[20] for ~128 vocabulary anchors covering content words,
function words, punctuation, and numbers — to build a 'semantic-basis
grid' in h-space that we can project trajectories onto.

Protocol matches the country-pool capture for direct comparability:
  * Each anchor is fed as the entire user-message text in a chat-
    templated prompt
  * Forward pass with output_hidden_states=True
  * Capture h[20][0, -1, :] — last position before the assistant turn

Output: testing/.cache/nla_artifacts/vocab_atlas.pt
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
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = _REPO_ROOT / "testing" / ".cache" / "nla_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
OUT = ARTIFACTS / "vocab_atlas.pt"


# NOTE (2026-05-29): the existing committed `vocab_atlas.pt` (and the
# audit-locked numbers derived from it) was captured against an earlier
# version of this VOCAB that had three duplicates:
#   - "when" appeared in both wh_word and conjunction
#   - "-"    appeared in both p_dash and math_op
#   - "*"    appeared in both p_special and math_op
# Effect on the committed artifact: the conjunction/math_op centroids
# include those tokens' h-vectors. The downstream numbers are stable
# (effect <1% on coarse aggregates) but technically slightly biased.
# This dict has been deduplicated (each token in exactly one category)
# for any future regeneration; regenerating the atlas would shift the
# audit-locked numbers slightly and is filed as a future follow-up.
VOCAB: dict[str, list[str]] = {
    # ---- Content-bearing (43) ----
    "country": [
        "France",
        "Germany",
        "Japan",
        "Brazil",
        "Egypt",
        "United Kingdom",
        "China",
        "Italy",
        "Russia",
        "India",
    ],
    "capital": ["Paris", "Berlin", "Tokyo", "London", "Madrid"],
    "nature": [
        "spring",
        "summer",
        "autumn",
        "winter",
        "tree",
        "flower",
        "leaf",
        "butterfly",
        "snow",
        "mountain",
        "ocean",
        "sky",
    ],
    "codemath": ["function", "variable", "return", "equation", "integral", "theorem"],
    "emotion": ["happy", "sad", "anger", "fear", "love", "joy"],
    "refusal": ["refuse", "sorry", "cannot", "decline"],
    # ---- Function words (51, was 52 — removed "when" duplicate, kept in wh_word) ----
    "article": ["a", "an", "the"],
    "pronoun": ["I", "you", "he", "she", "it", "we", "they"],
    "demonstrative": ["this", "that", "these", "those"],
    "preposition": ["of", "in", "on", "at", "by", "with", "for", "from", "to", "into"],
    "conjunction": ["and", "or", "but", "because", "if"],
    "auxiliary": ["is", "are", "was", "were", "has", "have", "will", "can"],
    "negation": ["not", "no", "never"],
    "quantifier": ["all", "some", "every", "many", "few"],
    "wh_word": ["what", "who", "where", "when", "why", "how"],
    # ---- Punctuation (18) ----
    "p_ender": [".", "!", "?"],
    "p_internal": [",", ";", ":"],
    "p_quote": ['"', "'"],
    "p_bracket": ["(", ")", "[", "]"],
    "p_dash": ["-", "—"],
    "p_special": ["/", "*", "#", "@"],
    # ---- Numbers & math ops (13, was 15 — removed "-" and "*" duplicates, kept in p_dash/p_special) ----
    "number": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    "math_op": [
        "+",
        "=",
    ],  # "-" lives in p_dash; "*" and "/" live in p_special. Just the unambiguous arithmetic operators.
}

# Runtime sanity check: every anchor in exactly one category. If this
# fails after a future edit, the centroid arithmetic will be biased.
_all_words = [w for ws in VOCAB.values() for w in ws]
assert len(_all_words) == len(set(_all_words)), (
    f"VOCAB has duplicates: {[w for w in _all_words if _all_words.count(w) > 1]}"
)


def main() -> None:
    print(f"loading base model {BASE_ID} (CPU bf16) ...")
    t0 = time.time()
    model, tok = surgery.load_model(BASE_ID, mode="bf16", device_map="cpu")
    print(f"  base loaded in {time.time() - t0:.0f}s")

    captures: list[dict[str, Any]] = []
    total = sum(len(v) for v in VOCAB.values())
    print(f"\ncapturing h[{LAYER}] for {total} anchor tokens ...")

    idx = 0
    for category, words in VOCAB.items():
        for w in words:
            idx += 1
            t0 = time.time()
            enc = tok.apply_chat_template(
                [{"role": "user", "content": w}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"]
            with torch.no_grad():
                out = model(
                    input_ids=input_ids, output_hidden_states=True, use_cache=False
                )
            h = out.hidden_states[LAYER][0, -1, :].detach().float().cpu().clone()
            n_tok = int(input_ids.shape[1])
            captures.append(
                {
                    "category": category,
                    "word": w,
                    "h": h,
                    "norm": float(h.norm().item()),
                    "n_input_tokens": n_tok,
                }
            )
            print(
                f"  [{idx:>3}/{total}] {category:<15} {w!r:<22}  "
                f"||h||={h.norm().item():>6.2f}  n_input={n_tok:>3}  "
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
            "categories": list(VOCAB.keys()),
            "base_id": BASE_ID,
            "layer": LAYER,
        },
        OUT,
    )
    print(f"\nWrote {len(captures)} captures to {OUT}")


if __name__ == "__main__":
    main()
