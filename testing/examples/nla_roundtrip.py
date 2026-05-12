"""End-to-end: Qwen2.5-7B layer-20 residual stream -> NLA verbalization.

Loads base on GPU (nf4), captures hidden_states[20] at the final prompt
position, frees the base model, loads the AV on CPU (bf16), asks it to
verbalize what the activation represents, and prints the base's
argmax-next-token alongside for context.

See ``llm_surgeon.probe._nla`` for the AV inference helper.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import gc
import time
from typing import cast

import torch

from llm_surgeon import surgery
from llm_surgeon.probe import load_av, nla_verbalize


BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20


def _gb(n: int) -> float:
    return n / (1024**3)


def _gpu_used_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    free, total = cast(tuple[int, int], torch.cuda.mem_get_info())
    return _gb(total - free)


def capture_h_layer(prompt: str) -> tuple[torch.Tensor, str, str]:
    """Load base on GPU, capture h[LAYER] at the final position, free base."""
    print(f"[base] loading {BASE_ID} ...")
    t0 = time.time()
    model, tok = surgery.load_model(BASE_ID, mode="nf4", device_map={"": 0})
    print(f"[base] loaded in {time.time() - t0:.1f}s; gpu={_gpu_used_gb():.2f} GiB")

    input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    final_pos = input_ids.shape[1] - 1
    h = out.hidden_states[LAYER][0, final_pos].detach().float().cpu().clone()
    next_tok_id = int(out.logits[0, final_pos].argmax().item())
    next_tok = tok.decode([next_tok_id])
    final_tok = tok.decode([int(input_ids[0, final_pos].item())])

    print(
        f"[base] captured h[{LAYER}] at pos={final_pos} (token={final_tok!r}), "
        f"shape={tuple(h.shape)}, ||h||_2={h.norm().item():.2f}"
    )
    print(f"[base] argmax next token: {next_tok_id} -> {next_tok!r}")

    del model, tok, out, input_ids
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[base] freed; gpu={_gpu_used_gb():.2f} GiB")
    return h, final_tok, next_tok


def main() -> None:
    prompt = "The capital of France is"
    print(f"prompt: {prompt!r}")

    h, final_tok, next_tok = capture_h_layer(prompt)

    print(f"\n[av] loading AV on CPU (bf16) ...")
    t0 = time.time()
    av_model, av_tok, meta = load_av()
    print(f"[av] loaded in {time.time() - t0:.1f}s")

    print(f"\n[av] verbalizing h[{LAYER}] (final position, token={final_tok!r}) ...")
    t0 = time.time()
    explanation = nla_verbalize(h, model=av_model, tok=av_tok, meta=meta, max_new_tokens=120)
    dt = time.time() - t0

    print(f"\n========== RESULT ==========")
    print(f"prompt:             {prompt!r}")
    print(f"final token:        {final_tok!r}")
    print(f"base argmax next:   {next_tok!r}")
    print(f"nla verbalization ({dt:.0f}s, layer {LAYER}, final position):")
    print()
    print(explanation)
    print(f"=============================")


if __name__ == "__main__":
    main()
