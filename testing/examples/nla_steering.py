"""Steering experiment: AV-edit -> AR-decode -> splice into base, see if output flips.

Round-trip is now causal. We have the original h[20] at step 0 of the
ocean-poem generation, the AV's verbalization of that h, and a saved
artifact of both. We edit the AV text to describe a forest instead of
nature/sun content, reconstruct via AR to get h_pred_forest, and
splice that into the base model's layer-20 output at the last prompt
position during a fresh generation. If the generated poem moves from
ocean imagery to forest imagery, the AV+AR pair supports controlled
intervention — the Anthropic-style causal interpretability claim.
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
from pathlib import Path

import torch

from llm_surgeon import surgery
from llm_surgeon.probe import load_ar, nla_reconstruct


BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
LAYER_IDX_0 = LAYER - 1
MAX_NEW_TOKENS = 30
PROMPT = "Write a short poem about the ocean."
ARTIFACT = Path("testing/.cache/nla_artifacts/aggregate_faithfulness.pt")

WIDTH = 80
BAR = "=" * WIDTH


def edit_av_text(original: str) -> str:
    """Targeted edit: swap sun/nature subject for forest. Keep structure intact.

    Raises ValueError if NO swap matched — protects against silent no-op
    when AV phrasing drifts. Individual missing swaps are not fatal
    (different AV decodes may omit some of these patterns), but at least
    one swap must apply or the test is meaningless.
    """
    swaps = [
        ("the sun", "the forest"),
        ("gentle sun", "ancient forest"),
        ("the sun's qualities", "the forest's qualities"),
        ("The glowing ember of earth", "The towering oaks of ancient woods"),
        ("My dear sun", "My dear forest"),
    ]
    out = original
    n_applied = 0
    for old, new in swaps:
        if old in out:
            out = out.replace(old, new)
            n_applied += 1
    if n_applied == 0:
        raise ValueError(
            "edit_av_text: no swap pattern matched the input AV text. "
            "AV phrasing may have drifted; update the `swaps` list to match "
            "current verbalizer output. Input snippet: "
            f"{original[:200]!r}..."
        )
    return out


class LayerOutputReplaceHook:
    """Replace the decoder block's output at one position during the prefill pass only.

    Qwen2 decoder block returns a tuple (hidden_states, ...). We rewrite
    hidden_states[0, target_position, :] in-place on the first forward
    pass with sequence length > 1 (i.e. the prefill), then no-op for
    subsequent single-token decode passes.
    """

    def __init__(self, target_position: int, replacement: torch.Tensor) -> None:
        self.target_position = target_position
        self.replacement = replacement
        self.fired = False

    def __call__(self, module: Any, args: Any, output: Any) -> Any:
        if self.fired:
            return output
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        if hidden_states.shape[1] > 1:
            r = self.replacement.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            hidden_states[0, self.target_position, :] = r
            self.fired = True
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


def main() -> None:
    print("[1/4] loading saved poem trajectory artifact ...")
    artifact = torch.load(ARTIFACT, weights_only=False)
    poem = next(p for p in artifact["prompts"] if p["id"] == "creative_poem")
    step0 = poem["captures"][0]
    h_original: torch.Tensor = step0["h"]
    av_text_original: str = step0["av_text"]
    print(f"  original ||h|| at step 0: {h_original.norm().item():.2f}")
    print(f"  baseline generated:        {poem['generated_text']!r}")

    av_text_edited = edit_av_text(av_text_original)
    print(f"\n[2/4] AV-text edit (sun/woodland -> forest, structure preserved)")
    print(BAR)
    print("ORIGINAL AV text (step 0):")
    print(av_text_original)
    print(BAR)
    print("EDITED AV text:")
    print(av_text_edited)
    print(BAR)

    print(f"\n[3/4] AR-decoding edited text -> h_pred_forest ...")
    t0 = time.time()
    backbone, value_head, tok_ar, meta_ar = load_ar()
    print(f"  AR loaded in {time.time() - t0:.1f}s")
    t0 = time.time()
    h_pred_raw = nla_reconstruct(
        av_text_edited,
        backbone=backbone,
        value_head=value_head,
        tok=tok_ar,
        meta=meta_ar,
    )
    print(
        f"  AR reconstruct in {time.time() - t0:.1f}s; raw ||h_pred||={h_pred_raw.norm().item():.2f}"
    )
    h_pred = h_pred_raw / h_pred_raw.norm() * h_original.norm()
    print(
        f"  rescaled to match original magnitude: ||h_pred||={h_pred.norm().item():.2f}"
    )

    del backbone, value_head, tok_ar
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[4/4] loading base + running BASELINE then STEERED generation ...")
    t0 = time.time()
    model, tok = surgery.load_model(BASE_ID, mode="bf16", device_map="cpu")
    print(f"  base loaded in {time.time() - t0:.1f}s")

    enc = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    prompt_len = input_ids.shape[1]

    print(f"  baseline (no intervention) generation ...")
    t0 = time.time()
    with torch.no_grad():
        baseline_out = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    print(f"    done in {time.time() - t0:.1f}s")
    baseline_text = tok.decode(baseline_out[0, prompt_len:], skip_special_tokens=False)

    target_layer = model.model.layers[LAYER_IDX_0]
    hook = LayerOutputReplaceHook(target_position=prompt_len - 1, replacement=h_pred)
    handle = target_layer.register_forward_hook(hook)
    try:
        print(f"  steered (replace h[{LAYER}] at pos {prompt_len - 1}) generation ...")
        t0 = time.time()
        with torch.no_grad():
            steered_out = model.generate(
                input_ids=input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        print(f"    done in {time.time() - t0:.1f}s")
    finally:
        handle.remove()
    steered_text = tok.decode(steered_out[0, prompt_len:], skip_special_tokens=False)

    print()
    print(BAR)
    print("STEERING RESULT")
    print(BAR)
    print(f"Prompt: {PROMPT!r}")
    print()
    print("BASELINE (no intervention):")
    print(f"  {baseline_text!r}")
    print()
    print(
        "STEERED (h[20] at last prompt position replaced with AR-decoded forest text):"
    )
    print(f"  {steered_text!r}")
    print()
    print(f"Hook fired: {hook.fired}")
    print(BAR)


if __name__ == "__main__":
    main()
