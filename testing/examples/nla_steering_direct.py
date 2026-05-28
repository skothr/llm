"""Direct-transplant steering: skip AV+AR, splice layer-20 activations between prompts.

Tests whether layer-20 residual at the prefill-last position is causally
load-bearing for entity choice during generation. Procedure:

  1. Run base on a forest prompt; capture h_forest = h[20] at the
     forest-prompt's last position.
  2. Run base on the ocean prompt with no intervention — establishes
     baseline.
  3. Run base on the ocean prompt again, hooking decoder block 19's
     output to replace h[20] at the ocean-prompt's last position with
     h_forest. The model's prompt-attention still sees "ocean" but its
     layer-20 residual at the critical position is the forest-prompt
     activation.

If the steered output shifts toward forest content, layer 20 IS load-
bearing for entity choice. If unchanged, attention from later positions
back to the prompt's "ocean" overrides any single-layer intervention.

This is the cleaner causal test than nla_steering.py because the
steering vector has no AV+AR reconstruction noise — it is a real
activation taken from a real forest forward pass.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import Any, cast

cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

import time

import torch

from llm_surgeon import surgery


BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
LAYER_IDX_0 = LAYER - 1
MAX_NEW_TOKENS = 30
OCEAN_PROMPT = "Write a short poem about the ocean."
FOREST_PROMPT = "Write a short poem about a forest."

WIDTH = 80
BAR = "=" * WIDTH


from _layer_hooks import LayerOutputReplaceHook  # noqa: E402


def chat_template_input_ids(tok: Any, msg: str) -> torch.Tensor:
    enc = tok.apply_chat_template(
        [{"role": "user", "content": msg}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return enc["input_ids"]


def main() -> None:
    print(f"[1/4] loading base ({BASE_ID}, CPU bf16) ...")
    t0 = time.time()
    model, tok = surgery.load_model(BASE_ID, mode="bf16", device_map="cpu")
    print(f"  base loaded in {time.time() - t0:.1f}s")

    print(f"\n[2/4] capturing h_forest from forest prompt ...")
    forest_ids = chat_template_input_ids(tok, FOREST_PROMPT)
    print(f"  forest prompt: {FOREST_PROMPT!r}  ({forest_ids.shape[1]} tokens)")
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=forest_ids, output_hidden_states=True, use_cache=False)
    h_forest = out.hidden_states[LAYER][0, -1, :].detach().float().cpu().clone()
    print(f"  forward in {time.time() - t0:.1f}s")
    print(f"  ||h_forest|| @ last prompt position = {h_forest.norm().item():.2f}")
    next_tok_id = int(out.logits[0, -1].argmax().item())
    print(
        f"  forest-prompt argmax next token: {next_tok_id} -> {tok.decode([next_tok_id])!r}"
    )
    del out

    print(f"\n[3/4] BASELINE ocean generation (no intervention) ...")
    ocean_ids = chat_template_input_ids(tok, OCEAN_PROMPT)
    print(f"  ocean prompt: {OCEAN_PROMPT!r}  ({ocean_ids.shape[1]} tokens)")
    ocean_prompt_len = ocean_ids.shape[1]
    t0 = time.time()
    with torch.no_grad():
        baseline_out = model.generate(
            input_ids=ocean_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    print(f"  done in {time.time() - t0:.1f}s")
    baseline_text = tok.decode(
        baseline_out[0, ocean_prompt_len:], skip_special_tokens=False
    )

    print(
        f"\n[4/4] STEERED ocean generation "
        f"(replace h[{LAYER}] at pos {ocean_prompt_len - 1} with h_forest) ..."
    )
    target_layer = model.model.layers[LAYER_IDX_0]
    hook = LayerOutputReplaceHook(
        target_position=ocean_prompt_len - 1,
        replacement=h_forest,
    )
    handle = target_layer.register_forward_hook(hook)
    try:
        t0 = time.time()
        with torch.no_grad():
            steered_out = model.generate(
                input_ids=ocean_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        print(f"  done in {time.time() - t0:.1f}s")
    finally:
        handle.remove()
    steered_text = tok.decode(
        steered_out[0, ocean_prompt_len:], skip_special_tokens=False
    )

    print()
    print(BAR)
    print("DIRECT-TRANSPLANT STEERING RESULT")
    print(BAR)
    print(f"Forest prompt (source of h_forest): {FOREST_PROMPT!r}")
    print(f"Ocean prompt (target):              {OCEAN_PROMPT!r}")
    print()
    print("BASELINE (ocean prompt, no intervention):")
    print(f"  {baseline_text!r}")
    print()
    print(
        f"STEERED (ocean prompt, h[{LAYER}] at pos {ocean_prompt_len - 1} replaced with h_forest):"
    )
    print(f"  {steered_text!r}")
    print()
    print(f"Hook fired: {hook.fired}")
    print(BAR)


if __name__ == "__main__":
    main()
