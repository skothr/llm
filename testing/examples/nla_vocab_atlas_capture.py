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
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = _REPO_ROOT / "testing" / ".cache" / "nla_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
OUT = ARTIFACTS / "vocab_atlas.pt"


VOCAB: dict[str, list[str]] = {
    # ---- Content-bearing (43) ----
    "country": ["France", "Germany", "Japan", "Brazil", "Egypt",
                 "United Kingdom", "China", "Italy", "Russia", "India"],
    "capital": ["Paris", "Berlin", "Tokyo", "London", "Madrid"],
    "nature": ["spring", "summer", "autumn", "winter", "tree", "flower",
                "leaf", "butterfly", "snow", "mountain", "ocean", "sky"],
    "codemath": ["function", "variable", "return", "equation", "integral", "theorem"],
    "emotion": ["happy", "sad", "anger", "fear", "love", "joy"],
    "refusal": ["refuse", "sorry", "cannot", "decline"],

    # ---- Function words (52) ----
    "article": ["a", "an", "the"],
    "pronoun": ["I", "you", "he", "she", "it", "we", "they"],
    "demonstrative": ["this", "that", "these", "those"],
    "preposition": ["of", "in", "on", "at", "by", "with", "for", "from", "to", "into"],
    "conjunction": ["and", "or", "but", "because", "if", "when"],
    "auxiliary": ["is", "are", "was", "were", "has", "have", "will", "can"],
    "negation": ["not", "no", "never"],
    "quantifier": ["all", "some", "every", "many", "few"],
    "wh_word": ["what", "who", "where", "when", "why", "how"],

    # ---- Punctuation (18) ----
    "p_ender": [".", "!", "?"],
    "p_internal": [",", ";", ":"],
    "p_quote": ["\"", "'"],
    "p_bracket": ["(", ")", "[", "]"],
    "p_dash": ["-", "—"],
    "p_special": ["/", "*", "#", "@"],

    # ---- Numbers & math ops (15) ----
    "number": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    "math_op": ["+", "-", "=", "*"],
}


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
                tokenize=True, add_generation_prompt=True, return_tensors="pt",
            )
            input_ids = enc["input_ids"]
            with torch.no_grad():
                out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[LAYER][0, -1, :].detach().float().cpu().clone()
            n_tok = int(input_ids.shape[1])
            captures.append({
                "category": category, "word": w,
                "h": h, "norm": float(h.norm().item()),
                "n_input_tokens": n_tok,
            })
            print(f"  [{idx:>3}/{total}] {category:<15} {w!r:<22}  "
                  f"||h||={h.norm().item():>6.2f}  n_input={n_tok:>3}  "
                  f"({time.time() - t0:.1f}s)")
            del out

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.save({
        "captures": captures,
        "categories": list(VOCAB.keys()),
        "base_id": BASE_ID,
        "layer": LAYER,
    }, OUT)
    print(f"\nWrote {len(captures)} captures to {OUT}")


if __name__ == "__main__":
    main()
