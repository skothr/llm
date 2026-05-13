"""NLA trajectory across the *generated* tokens.

While the model autoregressively writes its haiku, we capture h[20] at
each generation step — the residual-stream state that *produced* each
output token — and verbalize it via the AV. So we see what the model is
"thinking" just before it emits each word.

Each AV reading at step k describes the layer-20 representation that
the LM head was about to project to logits and from which the k-th
generated token was selected. The trajectory shows how internal state
evolves *as the model composes its response*.
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
MAX_NEW_TOKENS = 15
MAX_AV_TOKENS = 200
PROMPT = "Write me a haiku about a rabbit in spring."

WIDTH = 80
BAR = "═" * WIDTH
SUB = "─" * WIDTH


def _wrap(text: str, indent: str = "  ") -> str:
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


def _print_header(base_cfg: Any, meta: dict, generated_text: str, gen_tokens: list[str]) -> None:
    nl = base_cfg.num_hidden_layers
    hs = base_cfg.hidden_size
    print(BAR)
    print("NLA GENERATION-PHASE THOUGHT TRAJECTORY")
    print(BAR)
    print()
    print("Base model")
    print(f"  ID:          {BASE_ID}")
    print(f"  Layers:      {nl} transformer decoder blocks")
    print(f"  d_model:     {hs}")
    print()
    print("Sampling protocol")
    print(f"  Layer:       hidden_states[{LAYER}] (HF convention)")
    print(f"               = output of decoder block {LAYER - 1} (0-indexed)")
    print(f"               = ~71% depth of the {nl}-block stack")
    print(f"               matches kitft's extraction_layer_index = {meta['extraction_layer_index']}")
    print(f"  Per step:    one R^{hs} vector — the residual state that PRODUCED")
    print(f"               the k-th generated token (h[20] at the position whose")
    print(f"               logits were argmaxed to emit that token)")
    print()
    print("AV protocol")
    print(f"  Model:       {AV_ID}")
    print(f"  Input prep:  h normalized to unit L2, scaled to magnitude")
    print(f"               {meta['extraction']['injection_scale']}, injected at the")
    print(f"               '㈎' position of the AV's chat-templated prompt.")
    print(f"  Generation:  greedy decode, up to {MAX_AV_TOKENS} new tokens.")
    print()
    print(f"Prompt: {PROMPT!r}")
    print()
    print("Model's generated response (greedy decode, up to "
          f"{MAX_NEW_TOKENS} tokens):")
    print(f"  {generated_text!r}")
    print()
    print(f"  → split into {len(gen_tokens)} tokens:")
    chunks = []
    line = "    "
    for t in gen_tokens:
        piece = f"{t!r}"
        if len(line) + len(piece) + 2 > WIDTH:
            chunks.append(line.rstrip())
            line = "    " + piece + ", "
        else:
            line += piece + ", "
    if line.strip():
        chunks.append(line.rstrip().rstrip(","))
    print("\n".join(chunks))
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
    prompt_len = input_ids.shape[1]

    print(f"[base] generating up to {MAX_NEW_TOKENS} tokens with output_hidden_states ...")
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
        )
    dt = time.time() - t0
    n_steps = len(gen.hidden_states)
    print(f"[base]   generated {n_steps} tokens in {dt:.1f}s ({dt/max(1, n_steps):.1f}s/token)")

    generated_ids = gen.sequences[0, prompt_len:].tolist()
    generated_text = tok.decode(generated_ids, skip_special_tokens=False)
    gen_tokens = [tok.decode([int(t)]) for t in generated_ids]

    captures: list[tuple[int, str, float, torch.Tensor]] = []
    for step in range(n_steps):
        layer_states = gen.hidden_states[step]
        if step == 0:
            h = layer_states[LAYER][0, -1, :]
        else:
            h = layer_states[LAYER][0, 0, :]
        h_cpu = h.detach().float().cpu().clone()
        captures.append((step, gen_tokens[step], float(h_cpu.norm().item()), h_cpu))

    base_cfg = model.config
    del model, tok, gen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[base] freed.")

    print(f"[av] loading AV (CPU bf16) ...")
    t0 = time.time()
    av_model, av_tok, meta = load_av()
    print(f"[av] loaded in {time.time() - t0:.1f}s")

    print(f"[av] will verbalize {len(captures)} generation steps")
    results: list[tuple[int, str, float, str, float]] = []
    for i, (step, gen_tok, norm, h) in enumerate(captures, 1):
        print(f"[av] [{i}/{len(captures)}] step={step} → emit {gen_tok!r} (‖h‖={norm:.1f}) ...")
        t0 = time.time()
        explanation = nla_verbalize(
            h, model=av_model, tok=av_tok, meta=meta, max_new_tokens=MAX_AV_TOKENS,
        )
        dt = time.time() - t0
        print(f"     done in {dt:.0f}s ({dt/60:.1f} min)")
        results.append((step, gen_tok, norm, explanation, dt))

    print()
    _print_header(base_cfg, meta, generated_text, gen_tokens)

    print(BAR)
    print("STEP-BY-STEP READINGS  (what the model was 'thinking' before each token)")
    print(BAR)

    for step, gen_tok, norm, explanation, dt in results:
        print()
        print(SUB)
        print(f"Step {step}  ·  about to emit {gen_tok!r}  ·  ‖h‖₂ = {norm:.1f}  ({dt:.0f}s)")
        print(SUB)
        print(_wrap(explanation))

    print()
    print(BAR)


if __name__ == "__main__":
    main()
