"""Aggregate NLA faithfulness across 8 diverse prompts.

Same round-trip pipeline as nla_faithfulness.py but batched across
multiple prompt domains so we can establish a baseline mean cosine
and check whether faithfulness varies by domain.

Phases (each saves to disk on completion per prompt):
  1. Capture h[20] across generation for every prompt (~10 min)
  2. Verbalize each h via AV (~2.5 hr)
  3. Reconstruct via AR + score (~40 min)

Total ~3.5 hr unattended on CPU. Artifact saved after each prompt's
phase so any interruption resumes cleanly.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import Any, cast
cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

import gc
import statistics
import time
from pathlib import Path

import torch

from llm_surgeon import surgery
from llm_surgeon.probe import (
    AR_ID, AV_ID,
    load_ar, load_av,
    nla_reconstruct, nla_score, nla_verbalize,
)


BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
MAX_NEW_TOKENS = 15
MAX_AV_TOKENS = 200

PROMPTS = [
    ("factual_easy",    "What is the capital of France?"),
    ("factual_hard",    "What year did the Treaty of Westphalia end?"),
    ("math",            "Compute 47 times 38."),
    ("code",            "Write a Python function to check if a number is prime."),
    ("creative_haiku",  "Write me a haiku about a rabbit in spring."),
    ("creative_poem",   "Write a short poem about the ocean."),
    ("reasoning_logic", "If all roses are flowers and all flowers need water, do all roses need water?"),
    ("reasoning_word",  "Mary has 4 brothers. Each brother has 2 sisters. How many sisters does Mary have?"),
]

ARTIFACTS_DIR = Path("testing/.cache/nla_artifacts")
ARTIFACT_FILE = ARTIFACTS_DIR / "aggregate_faithfulness.pt"

WIDTH = 80
BAR = "═" * WIDTH


def _save(artifact: dict[str, Any]) -> None:
    torch.save(artifact, ARTIFACT_FILE)


def init_artifact() -> dict[str, Any]:
    return {
        "prompts": [
            {"id": pid, "text": text, "phase": "init", "captures": []}
            for pid, text in PROMPTS
        ],
    }


def phase_capture(artifact: dict[str, Any]) -> None:
    todo = [p for p in artifact["prompts"] if p["phase"] == "init"]
    if not todo:
        return
    print(f"\n[1/3] capturing h[20] across generation for {len(todo)} prompts ...")
    print(f"   loading base ({BASE_ID}) on CPU bf16 ...")
    t0 = time.time()
    model, tok = surgery.load_model(BASE_ID, mode="bf16", device_map="cpu")
    print(f"   base loaded in {time.time() - t0:.1f}s")

    for p in todo:
        print(f"   [{p['id']}] {p['text']!r}")
        enc = tok.apply_chat_template(
            [{"role": "user", "content": p["text"]}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        prompt_len = input_ids.shape[1]

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
        n_steps = len(gen.hidden_states)

        generated_ids = gen.sequences[0, prompt_len:].tolist()
        gen_tokens = [tok.decode([int(t)]) for t in generated_ids]
        generated_text = tok.decode(generated_ids, skip_special_tokens=False)
        print(f"      generated in {time.time() - t0:.1f}s: {generated_text!r}")

        captures = []
        for step in range(n_steps):
            layer_states = gen.hidden_states[step]
            if step == 0:
                h = layer_states[LAYER][0, -1, :]
            else:
                h = layer_states[LAYER][0, 0, :]
            captures.append({
                "step": step,
                "token": gen_tokens[step],
                "h": h.detach().float().cpu().clone(),
            })

        p["generated_text"] = generated_text
        p["gen_tokens"] = gen_tokens
        p["captures"] = captures
        p["phase"] = "captured"
        _save(artifact)

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"   base freed.")


def phase_verbalize(artifact: dict[str, Any]) -> None:
    todo = [p for p in artifact["prompts"] if p["phase"] == "captured"]
    if not todo:
        return
    n_total = sum(len(p["captures"]) for p in todo)
    print(f"\n[2/3] verbalizing {n_total} captures via AV ({len(todo)} prompts) ...")
    print(f"   loading AV ({AV_ID}) ...")
    t0 = time.time()
    av_model, av_tok, av_meta = load_av()
    print(f"   AV loaded in {time.time() - t0:.1f}s")

    overall_i = 0
    for p in todo:
        print(f"   [{p['id']}]")
        for c in p["captures"]:
            overall_i += 1
            print(f"      [{overall_i}/{n_total}] step={c['step']} → {c['token']!r} ...", end="", flush=True)
            t0 = time.time()
            c["av_text"] = nla_verbalize(
                c["h"], model=av_model, tok=av_tok, meta=av_meta,
                max_new_tokens=MAX_AV_TOKENS,
            )
            c["av_time"] = time.time() - t0
            print(f" {c['av_time']:.0f}s")
        p["phase"] = "verbalized"
        _save(artifact)

    del av_model, av_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"   AV freed.")


def phase_reconstruct(artifact: dict[str, Any]) -> None:
    todo = [p for p in artifact["prompts"] if p["phase"] == "verbalized"]
    if not todo:
        return
    n_total = sum(len(p["captures"]) for p in todo)
    print(f"\n[3/3] reconstructing {n_total} captures via AR ({len(todo)} prompts) ...")
    print(f"   loading AR ({AR_ID}) ...")
    t0 = time.time()
    ar_backbone, ar_value_head, ar_tok, ar_meta = load_ar()
    print(f"   AR loaded in {time.time() - t0:.1f}s")

    overall_i = 0
    for p in todo:
        print(f"   [{p['id']}]")
        for c in p["captures"]:
            overall_i += 1
            t0 = time.time()
            h_pred = nla_reconstruct(
                c["av_text"],
                backbone=ar_backbone, value_head=ar_value_head,
                tok=ar_tok, meta=ar_meta,
            )
            c["h_pred"] = h_pred
            scores = nla_score(c["h"], h_pred)
            c["cosine"] = scores["cosine"]
            c["normalized_mse"] = scores["normalized_mse"]
            dt = time.time() - t0
            print(
                f"      [{overall_i}/{n_total}] step={c['step']} → {c['token']!r}: "
                f"cos={c['cosine']:+.3f}  mse={c['normalized_mse']:.2f}  ({dt:.1f}s)"
            )
        p["phase"] = "scored"
        _save(artifact)


def print_report(artifact: dict[str, Any]) -> None:
    print()
    print(BAR)
    print("NLA AGGREGATE FAITHFULNESS REPORT")
    print(BAR)
    print()
    print(f"{'prompt id':<18}  {'n':>3}  {'mean cos':>9}  {'min':>7}  {'max':>7}  generated")
    print("─" * WIDTH)

    all_cos: list[float] = []
    all_mse: list[float] = []
    for p in artifact["prompts"]:
        if p["phase"] != "scored":
            continue
        cos = [c["cosine"] for c in p["captures"]]
        mse = [c["normalized_mse"] for c in p["captures"]]
        all_cos.extend(cos)
        all_mse.extend(mse)
        gen_preview = p["generated_text"].replace("\n", " ")[:40]
        print(
            f"{p['id']:<18}  {len(cos):>3}  {sum(cos)/len(cos):>+9.3f}  "
            f"{min(cos):>+7.3f}  {max(cos):>+7.3f}  {gen_preview!r}"
        )
    print(BAR)
    if all_cos:
        print()
        print(f"OVERALL ({len(all_cos)} datapoints across {len([p for p in artifact['prompts'] if p['phase'] == 'scored'])} prompts):")
        print(f"  mean cosine:   {statistics.mean(all_cos):+.3f}")
        print(f"  stdev cosine:  {statistics.stdev(all_cos) if len(all_cos) > 1 else 0:.3f}")
        print(f"  median cosine: {statistics.median(all_cos):+.3f}")
        print(f"  min / max:     {min(all_cos):+.3f} / {max(all_cos):+.3f}")
        print()
        print(f"  mean MSE:      {statistics.mean(all_mse):.3f}")
        print(f"  median MSE:    {statistics.median(all_mse):.3f}")
        print()
        bins = [(-1.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 1.0)]
        print("  cosine histogram:")
        for lo, hi in bins:
            n = sum(1 for c in all_cos if lo <= c < hi or (hi == 1.0 and c == 1.0))
            bar = "█" * n
            print(f"    [{lo:+.2f}, {hi:+.2f}): {n:>3}  {bar}")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if ARTIFACT_FILE.exists():
        print(f"loading existing artifact: {ARTIFACT_FILE}")
        artifact = torch.load(ARTIFACT_FILE, weights_only=False)
        existing_ids = {p["id"] for p in artifact["prompts"]}
        for pid, text in PROMPTS:
            if pid not in existing_ids:
                artifact["prompts"].append({
                    "id": pid, "text": text, "phase": "init", "captures": [],
                })
    else:
        artifact = init_artifact()
        _save(artifact)

    phase_capture(artifact)
    phase_verbalize(artifact)
    phase_reconstruct(artifact)
    print_report(artifact)


if __name__ == "__main__":
    main()
