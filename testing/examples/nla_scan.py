"""Plumbing test: do per-token-position h[20] verbalizations look different?

Captures Qwen2.5-7B layer-20 residual stream at every token of a fixed
prompt, plus the raw input-embedding of one OOD probe token, then asks
the NLA AV to verbalize each. The interesting signal is whether the AV
produces qualitatively different explanations across positions — if so,
the residual stream is genuinely carrying position-specific semantic
content, and the AV is reading it.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import cast
cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

import gc
import time

import torch

from llm_surgeon import surgery
from llm_surgeon.probe import load_av, nla_verbalize


BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
MAX_NEW_TOKENS = 200
PROMPT = "The capital of France is"
OOD_PROBE_TOKEN = " Paris"


def _gb(n: int) -> float:
    return n / (1024**3)


def _gpu_used_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    free, total = cast(tuple[int, int], torch.cuda.mem_get_info())
    return _gb(total - free)


def main() -> None:
    print(f"prompt: {PROMPT!r}")
    print(f"layer: {LAYER}, max_new_tokens: {MAX_NEW_TOKENS}\n")

    print(f"[base] loading {BASE_ID} ...")
    t0 = time.time()
    model, tok = surgery.load_model(BASE_ID, mode="nf4", device_map={"": 0})
    print(f"[base] loaded in {time.time() - t0:.1f}s; gpu={_gpu_used_gb():.2f} GiB")

    input_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    token_strs = [tok.decode([int(t)]) for t in input_ids[0]]
    print(f"[base] tokenized: {token_strs}")

    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    h20_all = out.hidden_states[LAYER][0]
    next_tok_ids = out.logits[0].argmax(dim=-1).tolist()
    next_tok_strs = [tok.decode([int(i)]) for i in next_tok_ids]

    captures: list[tuple[str, str, torch.Tensor]] = []
    for pos in range(input_ids.shape[1]):
        h = h20_all[pos].detach().float().cpu().clone()
        label = f"h[{LAYER}] @ pos={pos} (cur={token_strs[pos]!r}, next-argmax={next_tok_strs[pos]!r})"
        captures.append((f"pos{pos}", label, h))

    embed_table = model.get_input_embeddings()
    paris_id = tok(OOD_PROBE_TOKEN, add_special_tokens=False).input_ids[0]
    h_paris = embed_table.weight[paris_id].detach().float().cpu().clone()
    captures.append((
        "embed_paris",
        f"raw input-embedding of {OOD_PROBE_TOKEN!r} (token-id={paris_id}, OOD for AV)",
        h_paris,
    ))

    print(f"[base] captured {len(captures)} vectors:")
    for key, label, h in captures:
        print(f"  {key}: {label}  ||h||_2={h.norm().item():.2f}")

    del model, tok, out, input_ids, embed_table, h20_all
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[base] freed; gpu={_gpu_used_gb():.2f} GiB\n")

    print(f"[av] loading AV on CPU (bf16) ...")
    t0 = time.time()
    av_model, av_tok, meta = load_av()
    print(f"[av] loaded in {time.time() - t0:.1f}s\n")

    results: list[tuple[str, str, str, float]] = []
    for i, (key, label, h) in enumerate(captures, 1):
        print(f"[av] [{i}/{len(captures)}] verbalizing {key} ...")
        t0 = time.time()
        explanation = nla_verbalize(
            h, model=av_model, tok=av_tok, meta=meta, max_new_tokens=MAX_NEW_TOKENS,
        )
        dt = time.time() - t0
        print(f"      done in {dt:.0f}s")
        results.append((key, label, explanation, dt))

    print("\n" + "=" * 70)
    print(f"prompt:   {PROMPT!r}")
    print(f"tokens:   {token_strs}")
    print("=" * 70)
    for key, label, explanation, dt in results:
        print(f"\n--- {key} ({dt:.0f}s) ---")
        print(f"    {label}")
        print()
        for line in explanation.splitlines():
            print(f"    {line}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
