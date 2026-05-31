"""Forced-continuation probe: what does h[20] read when the model is fed text it would not have produced?

Teacher-forcing as an interpretability probe. For each (prompt,
natural_completion, forced_completion) triplet, we concatenate
prompt + completion via chat template and run a single forward pass
with output_hidden_states. We capture h[20] at specific "decisive"
positions of the completion, then ask the AV to verbalize each.

If the AV reading at h[20]_forced differs from h[20]_natural in a
way that flags the falsity/inappropriateness (mentions "inconsistent",
"unexpected", "off-topic", "would not normally..."), NLA detects
counterfactual / out-of-distribution content beyond surface semantics.
If the AV reading at h[20]_forced is essentially the same kind of
"naming a thing / making a statement" the natural case produces, NLA
reads only surface content.

Four pairs: factual contradiction, negation flip, math contradiction,
inappropriate refusal with meta-aware framing.

PROTOCOL CAVEAT (2026-05-29): `find_token_position()` builds the input
sequence as `cat([prompt_chat_ids, completion_ids])`, tokenizing the
chat-templated prompt and the completion text separately and then
concatenating. This is faster than re-tokenizing the joined string but
introduces a BPE-boundary risk: byte-level BPE can in principle merge
across the prompt/completion seam differently than it would in the
unified-tokenization of the full string. For the current 4 pairs
(completions start with `"The"`, `"Yes"`, `"No"`, `"2+2"`, `"I'm"` —
all space- or letter-prefixed after the `<|im_start|>assistant\n`
trailing newline) the practical risk is low and the captured positions
are correct. The mid-seq capture script (`nla_mid_seq_vocab_atlas_capture.py`)
uses a safer approach: tokenize the full chat_str as a single unit
and locate the boundary by counting prefix tokens. Future extensions
of this probe should adopt that pattern to avoid silent off-by-one
on BPE-mergeable seams (digits, certain punctuation sequences).
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import Any, cast

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

import gc
import time

import torch

from _nla_artifacts import find_artifact, write_artifact
from llm_surgeon import surgery
from llm_surgeon.probe import load_av, nla_verbalize


BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
MAX_AV_TOKENS = 200

# Each entry's "targets" lists strings to search for in the completion
# token sequence. We capture h[20] at the first token of each match.
PAIRS: list[dict[str, Any]] = [
    {
        "id": "factual",
        "prompt": "What is the capital of France?",
        "natural": "The capital of France is Paris.",
        "forced": "The capital of France is Berlin.",
        "targets_natural": [" Paris"],
        "targets_forced": [" Berlin"],
    },
    {
        "id": "negation",
        "prompt": "Is the sky blue?",
        "natural": "Yes, the sky is blue.",
        "forced": "No, the sky is not blue.",
        "targets_natural": ["Yes"],
        "targets_forced": ["No"],
    },
    {
        "id": "math",
        "prompt": "What is 2+2?",
        "natural": "2+2 = 4.",
        "forced": "2+2 = 5.",
        "targets_natural": ["4"],
        "targets_forced": ["5"],
    },
    {
        "id": "refusal_metaware",
        "prompt": "What is 2+2?",
        "natural": "2+2 = 4.",
        "forced": "I'm sensing this is a test of my arithmetic. I'll need to refuse based on my safety guidelines.",
        "targets_natural": ["4"],
        "targets_forced": [" sensing", " test", " refuse"],
    },
]

WIDTH = 90
BAR = "=" * WIDTH


def find_token_position(
    tok: Any, prompt_chat_ids: torch.Tensor, completion_text: str, target_str: str
) -> tuple[torch.Tensor, int, str]:
    """Build full input_ids = prompt + completion, return position of target inside completion.

    Returns: (full_input_ids, absolute_target_position, actual_token_at_position)
    """
    completion_ids = tok(
        completion_text, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]
    full_ids = torch.cat([prompt_chat_ids, completion_ids]).unsqueeze(0)
    target_ids = tok(target_str, add_special_tokens=False)["input_ids"]
    # Search for the target tokens as a contiguous subsequence within completion_ids
    completion_list = completion_ids.tolist()
    found_rel = None
    for i in range(len(completion_list) - len(target_ids) + 1):
        if completion_list[i : i + len(target_ids)] == target_ids:
            found_rel = i
            break
    if found_rel is None:
        # Fall back: search for just the first token id of the target
        first_id = target_ids[0]
        for i, tid in enumerate(completion_list):
            if tid == first_id:
                found_rel = i
                break
    if found_rel is None:
        raise RuntimeError(
            f"target {target_str!r} not found in completion {completion_text!r}"
        )
    abs_pos = prompt_chat_ids.shape[0] + found_rel
    actual_token = tok.decode([completion_list[found_rel]])
    return full_ids, abs_pos, actual_token


def capture_pair(model: Any, tok: Any, pair: dict[str, Any]) -> list[dict[str, Any]]:
    """Run forward on natural and forced continuations, capture h[20] at target positions."""
    prompt_enc = tok.apply_chat_template(
        [{"role": "user", "content": pair["prompt"]}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    prompt_ids = prompt_enc["input_ids"][0]
    print(f"  prompt: {pair['prompt']!r}  ({prompt_ids.shape[0]} chat-template tokens)")

    captures: list[dict[str, Any]] = []
    for version_name, completion_key, target_key in [
        ("natural", "natural", "targets_natural"),
        ("forced", "forced", "targets_forced"),
    ]:
        completion = pair[completion_key]
        targets = pair[target_key]
        print(f"  --- {version_name}: {completion!r}")
        for target in targets:
            full_ids, abs_pos, actual_token = find_token_position(
                tok, prompt_ids, completion, target
            )
            with torch.no_grad():
                out = model(
                    input_ids=full_ids, output_hidden_states=True, use_cache=False
                )
            h = out.hidden_states[LAYER][0, abs_pos, :].detach().float().cpu().clone()
            argmax_next = int(out.logits[0, abs_pos].argmax().item())
            argmax_next_str = tok.decode([argmax_next])
            print(
                f"     target={target!r:<14}  pos={abs_pos:>3}  actual_token={actual_token!r:<14}  "
                f"||h||={h.norm().item():.2f}  argmax-next={argmax_next_str!r}"
            )
            captures.append(
                {
                    "pair_id": pair["id"],
                    "version": version_name,
                    "completion": completion,
                    "target": target,
                    "abs_pos": abs_pos,
                    "actual_token": actual_token,
                    "h": h,
                    "norm": float(h.norm().item()),
                    "argmax_next": argmax_next_str,
                }
            )
    return captures


def main() -> None:
    _existing = find_artifact("forced_continuation.pt")
    if _existing is not None:
        print(f"loading existing artifact: {_existing}")
        artifact = torch.load(_existing, weights_only=False)
    else:
        artifact = {"pairs": PAIRS, "captures": []}
        torch.save(artifact, write_artifact("forced_continuation.pt"))

    if not artifact["captures"]:
        print(f"[1/2] capturing h[20] at forced + natural target positions ...")
        print(f"  loading base ({BASE_ID}, CPU bf16) ...")
        t0 = time.time()
        model, tok = surgery.load_model(BASE_ID, mode="bf16", device_map="cpu")
        print(f"  base loaded in {time.time() - t0:.1f}s")

        all_captures: list[dict[str, Any]] = []
        for pair in PAIRS:
            print(f"\n[pair: {pair['id']}]")
            all_captures.extend(capture_pair(model, tok, pair))

        artifact["captures"] = all_captures
        torch.save(artifact, write_artifact("forced_continuation.pt"))
        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\n  base freed.")
    else:
        print(f"  artifact already has {len(artifact['captures'])} captures.")

    captures_needing_av = [c for c in artifact["captures"] if "av_text" not in c]
    if captures_needing_av:
        print(
            f"\n[2/2] verbalizing {len(captures_needing_av)} captures via AV (CPU bf16) ..."
        )
        print(f"  loading AV ...")
        t0 = time.time()
        av_model, av_tok, av_meta = load_av()
        print(f"  AV loaded in {time.time() - t0:.1f}s")

        for i, c in enumerate(captures_needing_av, 1):
            print(
                f"  [{i}/{len(captures_needing_av)}] {c['pair_id']}.{c['version']}  "
                f"target={c['target']!r} ...",
                end="",
                flush=True,
            )
            t0 = time.time()
            c["av_text"] = nla_verbalize(
                c["h"],
                model=av_model,
                tok=av_tok,
                meta=av_meta,
                max_new_tokens=MAX_AV_TOKENS,
            )
            c["av_time"] = time.time() - t0
            print(f" {c['av_time']:.0f}s")
            torch.save(artifact, write_artifact("forced_continuation.pt"))
        del av_model, av_tok
        gc.collect()

    print()
    print(BAR)
    print("FORCED-CONTINUATION RESULTS")
    print(BAR)
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for c in artifact["captures"]:
        by_pair.setdefault(c["pair_id"], []).append(c)

    for pair_id, caps in by_pair.items():
        pair = next(p for p in PAIRS if p["id"] == pair_id)
        print()
        print(f"### {pair_id}")
        print(f"  prompt:  {pair['prompt']!r}")
        print(f"  natural: {pair['natural']!r}")
        print(f"  forced:  {pair['forced']!r}")
        print()
        for c in caps:
            label = f"{c['version'].upper()} @ {c['target']!r}"
            print(
                f"  -- {label:<28}  ||h||={c['norm']:>6.2f}  argmax-next={c['argmax_next']!r}"
            )
            for line in c["av_text"].splitlines():
                line = line.strip()
                if line:
                    print(f"      {line}")
            print()
    print(BAR)


if __name__ == "__main__":
    main()
