"""Full NLA thought-trajectory through every user-message token.

For a chat-templated prompt, captures h[20] at every token of the
user-message slice plus the final pre-assistant position, verbalizes
each, and prints a clearly-formatted token-by-token trajectory showing
how the residual stream's semantic content evolves position to position.

Header section explains the exact sampling protocol (which layer of
which model, what normalization, what scale, what the AV does) so the
output is self-documenting.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import Any, cast
cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

import gc
import textwrap
import time

import torch

from llm_surgeon import surgery
from llm_surgeon.probe import AV_ID, load_av, nla_verbalize


BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
MAX_NEW_TOKENS = 200
PROMPT = "Write me a haiku about a rabbit in spring."

WIDTH = 80
BAR = "═" * WIDTH
SUB = "─" * WIDTH


def _wrap(text: str, indent: str = "  ") -> str:
    """Wrap multi-paragraph text at WIDTH-2 cols with indent."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    wrapped = [
        textwrap.fill(
            p, width=WIDTH - 2,
            initial_indent=indent, subsequent_indent=indent,
            break_long_words=False, break_on_hyphens=False,
        )
        for p in paragraphs
    ]
    return "\n\n".join(wrapped)


def _print_header(base_cfg: Any, meta: dict) -> None:
    nl = base_cfg.num_hidden_layers
    hs = base_cfg.hidden_size
    print(BAR)
    print("NLA THOUGHT TRAJECTORY")
    print(BAR)
    print()
    print("Base model")
    print(f"  ID:          {BASE_ID}")
    print(f"  Layers:      {nl} transformer decoder blocks")
    print(f"  d_model:     {hs}")
    print(f"  vocab:       {base_cfg.vocab_size}")
    print()
    print("Sampling protocol")
    print(f"  Layer:       hidden_states[{LAYER}] (HF convention)")
    print(f"               = output of decoder block {LAYER - 1} (0-indexed)")
    print(f"               = input to decoder block {LAYER}")
    print(f"               = ~71% depth of the {nl}-block stack")
    print(f"               matches kitft's extraction_layer_index = {meta['extraction_layer_index']}")
    print(f"  Per token:   one R^{hs} residual-stream vector")
    print()
    print("AV protocol (per kitft nla_meta.yaml)")
    print(f"  Model:       {AV_ID}")
    print(f"  Released:    Anthropic 2026-05-07")
    print(f"  Input prep:  h normalized to unit L2, then scaled to magnitude")
    print(f"               {meta['extraction']['injection_scale']} ({meta['tokens']['injection_char']!r} replacement scale)")
    print(f"  Injection:   replace embedding of injection_token (id={meta['tokens']['injection_token_id']})")
    print(f"               inside the AV's chat-templated prompt with the")
    print(f"               normalized + scaled activation")
    print(f"  Generation:  greedy decode, up to {MAX_NEW_TOKENS} new tokens")
    print(f"  Output:      text between <explanation>...</explanation> tags")
    print()
    print(f"Prompt: {PROMPT!r}")
    print()


def main() -> None:
    print(f"[base] loading {BASE_ID} (CPU bf16) ...")
    t0 = time.time()
    model, tok = surgery.load_model(BASE_ID, mode="bf16", device_map="cpu")
    print(f"[base] loaded in {time.time() - t0:.1f}s")

    enc = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    tokens = [tok.decode([int(t)]) for t in input_ids[0]]

    user_start = next(
        i + 2 for i, t in enumerate(tokens)
        if t == "<|im_start|>" and i + 1 < len(tokens) and tokens[i + 1] == "user"
    )
    user_end = next(
        i for i in range(user_start, len(tokens))
        if tokens[i] == "<|im_end|>"
    )
    final_pos = len(tokens) - 1

    sampled_positions: list[int] = list(range(user_start, user_end)) + [final_pos]
    print(f"[base] {input_ids.shape[1]} tokens; user-message slice = [{user_start}:{user_end}]; "
          f"plus final pre-assistant position {final_pos}")
    print(f"[base] will verbalize {len(sampled_positions)} positions")

    print(f"[base] forward pass ...")
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    print(f"[base]   done in {time.time() - t0:.1f}s")

    h20_all = out.hidden_states[LAYER][0]
    captures: list[tuple[int, str, float, torch.Tensor]] = []
    for pos in sampled_positions:
        h = h20_all[pos].detach().float().cpu().clone()
        captures.append((pos, tokens[pos], float(h.norm().item()), h))

    base_cfg = model.config
    del model, tok, out, h20_all
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[base] freed.\n")

    print(f"[av] loading AV (CPU bf16) ...")
    t0 = time.time()
    av_model, av_tok, meta = load_av()
    print(f"[av] loaded in {time.time() - t0:.1f}s\n")

    results: list[tuple[int, str, float, str, float]] = []
    for i, (pos, token, norm, h) in enumerate(captures, 1):
        print(f"[av] [{i}/{len(captures)}] pos={pos} token={token!r} ‖h‖={norm:.1f} ...")
        t0 = time.time()
        explanation = nla_verbalize(
            h, model=av_model, tok=av_tok, meta=meta, max_new_tokens=MAX_NEW_TOKENS,
        )
        dt = time.time() - t0
        print(f"     done in {dt:.0f}s ({dt/60:.1f} min)")
        results.append((pos, token, norm, explanation, dt))

    print()
    _print_header(base_cfg, meta)

    prompt_text = " · ".join(repr(t) for t in tokens[captures[0][0]:captures[-2][0] + 1])
    print(f"User-message tokens (in order):")
    print(_wrap(prompt_text))
    print()
    print("Plus position {} = {!r} (final position before the assistant's response).".format(
        captures[-1][0], captures[-1][1],
    ))
    print()
    print(BAR)
    print("POSITION-BY-POSITION READINGS")
    print(BAR)

    for pos, token, norm, explanation, dt in results:
        is_final = (pos == final_pos)
        title = f"Position {pos}  ·  token = {token!r}  ·  ‖h‖₂ = {norm:.1f}"
        if is_final:
            title += "  ·  [FINAL: model about to compose response]"
        print()
        print(SUB)
        print(title)
        print(SUB)
        print(f"  inference: {dt:.0f}s")
        print()
        print(_wrap(explanation))

    print()
    print(BAR)


if __name__ == "__main__":
    main()
