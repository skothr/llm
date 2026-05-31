"""Round-trip faithfulness: h → AV → text → AR → h_pred, scored per step.

For each generation step of the haiku, we now have three things:
  * h: the original layer-20 residual that produced the emitted token
  * text: the AV's English verbalization of h
  * h_pred: the AR's reconstruction of h from text alone

The AR was co-trained with the AV so that "valid" explanations must
enable activation reconstruction. So ``cosine(h, h_pred)`` is a
per-step *fidelity score* — high means the AV's text preserved enough
information to recover the activation; low means the AV confabulated.

Artifacts are saved to testing/.cache/nla_artifacts/ so subsequent
analysis doesn't require re-running the 30-min AV stage.
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
from llm_surgeon.probe import (
    AR_ID,
    AV_ID,
    load_ar,
    load_av,
    nla_reconstruct,
    nla_score,
    nla_verbalize,
)


BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
MAX_NEW_TOKENS = 15
MAX_AV_TOKENS = 200
PROMPT = "Write me a haiku about a rabbit in spring."

WIDTH = 80
BAR = "═" * WIDTH
SUB = "─" * WIDTH


def capture_generation_h() -> dict[str, Any]:
    """Phase 1: load base, generate haiku, capture h[20] at each step."""
    print(f"[1/3] capturing h[20] across generation ...")
    print(f"   loading base ({BASE_ID}) on CPU bf16 ...")
    t0 = time.time()
    model, tok = surgery.load_model(BASE_ID, mode="bf16", device_map="cpu")
    print(f"   base loaded in {time.time() - t0:.1f}s")

    enc = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    prompt_len = input_ids.shape[1]

    print(f"   generating up to {MAX_NEW_TOKENS} tokens ...")
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
    print(f"   generated {n_steps} tokens in {time.time() - t0:.1f}s")

    generated_ids = gen.sequences[0, prompt_len:].tolist()
    generated_text = tok.decode(generated_ids, skip_special_tokens=False)
    gen_tokens = [tok.decode([int(t)]) for t in generated_ids]
    print(f"   generated: {generated_text!r}")

    captures = []
    for step in range(n_steps):
        layer_states = gen.hidden_states[step]
        if step == 0:
            h = layer_states[LAYER][0, -1, :]
        else:
            h = layer_states[LAYER][0, 0, :]
        captures.append(
            {
                "step": step,
                "token": gen_tokens[step],
                "h": h.detach().float().cpu().clone(),
            }
        )

    del model, tok, gen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"   base freed.")
    return {
        "prompt": PROMPT,
        "generated_text": generated_text,
        "gen_tokens": gen_tokens,
        "captures": captures,
    }


def verbalize_all(artifact: dict[str, Any]) -> None:
    """Phase 2: load AV, verbalize each captured h. Mutates artifact in place."""
    print(f"\n[2/3] verbalizing {len(artifact['captures'])} captures via AV ...")
    print(f"   loading AV ({AV_ID}) ...")
    t0 = time.time()
    av_model, av_tok, av_meta = load_av()
    print(f"   AV loaded in {time.time() - t0:.1f}s")

    for i, c in enumerate(artifact["captures"], 1):
        print(
            f"   [{i}/{len(artifact['captures'])}] step={c['step']} → {c['token']!r} ..."
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
        print(f"      done in {c['av_time']:.0f}s")

    del av_model, av_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"   AV freed.")


def reconstruct_all(artifact: dict[str, Any]) -> None:
    """Phase 3: load AR, reconstruct each h from AV text, score."""
    print(f"\n[3/3] reconstructing via AR + scoring ...")
    print(f"   loading AR ({AR_ID}) — first run downloads ~10 GB ...")
    t0 = time.time()
    ar_backbone, ar_value_head, ar_tok, ar_meta = load_ar()
    print(f"   AR loaded in {time.time() - t0:.1f}s")

    for i, c in enumerate(artifact["captures"], 1):
        t0 = time.time()
        h_pred = nla_reconstruct(
            c["av_text"],
            backbone=ar_backbone,
            value_head=ar_value_head,
            tok=ar_tok,
            meta=ar_meta,
        )
        c["h_pred"] = h_pred
        c["ar_time"] = time.time() - t0
        scores = nla_score(c["h"], h_pred)
        c["cosine"] = scores["cosine"]
        c["normalized_mse"] = scores["normalized_mse"]
        print(
            f"   [{i}/{len(artifact['captures'])}] step={c['step']} → {c['token']!r}: "
            f"cos={c['cosine']:+.3f}  mse={c['normalized_mse']:.2f}  ({c['ar_time']:.1f}s)"
        )


def print_report(artifact: dict[str, Any]) -> None:
    print()
    print(BAR)
    print("NLA ROUND-TRIP FAITHFULNESS REPORT")
    print(BAR)
    print()
    print(f"Prompt:    {artifact['prompt']!r}")
    print(f"Generated: {artifact['generated_text']!r}")
    print()
    print(f"Each step's h[20] (from generation) was verbalized by the AV,")
    print(f"then reconstructed by the AR purely from that verbalization.")
    print(f"  cosine: cos(h, h_pred). 1.0=perfect direction, 0=orthogonal.")
    print(f"  mse:    L2-normalized MSE in [0, 4]. Random pairs ≈ 2.")
    print()
    print(BAR)
    print(f"  step  token          ‖h‖     cosine   mse    AV-text (first 60 chars)")
    print(SUB)
    for c in artifact["captures"]:
        norm = float(c["h"].norm().item())
        snippet = c["av_text"].replace("\n", " ").strip()[:60]
        print(
            f"  {c['step']:4d}  {c['token']!r:<14s} "
            f"{norm:6.1f}  {c['cosine']:+.3f}   {c['normalized_mse']:.2f}   {snippet}"
        )
    print(BAR)
    cos_vals = [c["cosine"] for c in artifact["captures"]]
    mse_vals = [c["normalized_mse"] for c in artifact["captures"]]
    print(f"\nSummary:")
    print(
        f"  mean cosine:  {sum(cos_vals) / len(cos_vals):+.3f}  "
        f"(min {min(cos_vals):+.3f}, max {max(cos_vals):+.3f})"
    )
    print(
        f"  mean MSE:     {sum(mse_vals) / len(mse_vals):.2f}  "
        f"(min {min(mse_vals):.2f}, max {max(mse_vals):.2f})"
    )


def main() -> None:
    _existing = find_artifact("rabbit_haiku_gen_trajectory.pt")
    if _existing is not None:
        print(f"loading existing artifact: {_existing}")
        artifact = torch.load(_existing, weights_only=False)
        if "av_text" not in artifact["captures"][0]:
            print(f"  artifact has h but no AV text — running phase 2 ...")
            verbalize_all(artifact)
            torch.save(artifact, write_artifact("rabbit_haiku_gen_trajectory.pt"))
        if "h_pred" not in artifact["captures"][0]:
            print(f"  artifact has AV text but no AR — running phase 3 ...")
            reconstruct_all(artifact)
            torch.save(artifact, write_artifact("rabbit_haiku_gen_trajectory.pt"))
    else:
        artifact = capture_generation_h()
        torch.save(artifact, write_artifact("rabbit_haiku_gen_trajectory.pt"))
        verbalize_all(artifact)
        torch.save(artifact, write_artifact("rabbit_haiku_gen_trajectory.pt"))
        reconstruct_all(artifact)
        torch.save(artifact, write_artifact("rabbit_haiku_gen_trajectory.pt"))

    print_report(artifact)


if __name__ == "__main__":
    main()
