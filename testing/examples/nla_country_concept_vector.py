"""Extract a 'country' concept direction from h[20] via difference-of-means.

Captures h[20] at the last prompt token for 8 country prompts and 8
matched non-country prompts, computes the difference-of-means direction
as a candidate 'country-ness' axis, then projects held-out test
prompts onto it. Test set includes:

  * 2 straightforward holdout country prompts (UK, Egypt)
  * 2 indirect-country-mention prompts (Mozart in Salzburg, Queen from
    England)
  * 1 fictional country (Tolkien's Gondor)
  * 5 weird-context country prompts (country-as-metaphor,
    country-as-personification, country-as-sensory-cross-reference)
  * 1 negation prompt ("Mars is not a country.")
  * 2 non-country sanity controls

Finally, feeds the country direction through the AV to read out a
natural-language description of what we've extracted. If the AV
describes the direction as 'about countries / nations / geographic
entities' the extraction worked; if it produces unrelated content the
direction is something else entirely.
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
from llm_surgeon.probe import load_av, nla_verbalize


BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
ARTIFACT = Path("testing/.cache/nla_artifacts/country_concept_vector.pt")

SOURCE_COUNTRY = [
    "What is the capital of France?",
    "Tell me about Germany.",
    "Brazil is famous for soccer.",
    "The population of India is large.",
    "Italy borders Switzerland.",
    "Spain is in Europe.",
    "Japan has a unique culture.",
    "China's economy is growing.",
]

SOURCE_NON_COUNTRY = [
    "What is the role of mitochondria?",
    "Tell me about apples.",
    "A piano has 88 keys.",
    "The properties of water are unique.",
    "Mozart wrote many symphonies.",
    "A car's engine converts fuel.",
    "Photosynthesis requires sunlight.",
    "Hammers are tools.",
]

TEST_PROMPTS: list[tuple[str, str]] = [
    ("holdout_country_uk",    "The United Kingdom has a king."),
    ("holdout_country_egypt", "Egypt is in Africa."),
    ("indirect_mozart",       "Mozart was born in Salzburg."),
    ("indirect_band_queen",   "The band Queen is from England."),
    ("fictional_gondor",      "Tell me about Gondor."),
    ("weird_sandwich",        "If France were a sandwich, what would it taste like?"),
    ("weird_pet_hamster",     "My pet hamster Belgium escaped this morning."),
    ("weird_smells",          "Portugal smells like Tuesdays to me."),
    ("weird_dream",           "In a dream last night, I argued with Germany over breakfast cereal."),
    ("weird_metaphor",        "Justice is the country of the soul."),
    ("negation",              "Mars is not a country."),
    ("non_country_molecule",  "The molecule has six carbon atoms."),
    ("non_country_banana",    "Banana is yellow."),
]

WIDTH = 88
BAR = "=" * WIDTH


def chat_input_ids(tok: Any, msg: str) -> torch.Tensor:
    enc = tok.apply_chat_template(
        [{"role": "user", "content": msg}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    return enc["input_ids"]


def capture_h(model: Any, tok: Any, prompt: str) -> torch.Tensor:
    """Forward + return h[20] at the last prompt token."""
    ids = chat_input_ids(tok, prompt)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[LAYER][0, -1, :].detach().float().cpu().clone()
    del out
    return h


def main() -> None:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading base ({BASE_ID}, CPU bf16) ...")
    t0 = time.time()
    model, tok = surgery.load_model(BASE_ID, mode="bf16", device_map="cpu")
    print(f"  base loaded in {time.time() - t0:.1f}s")

    print(f"\n[2/4] capturing h[20] @ last prompt token for "
          f"{len(SOURCE_COUNTRY)} country + {len(SOURCE_NON_COUNTRY)} non-country "
          f"+ {len(TEST_PROMPTS)} test prompts ...")
    h_country: list[torch.Tensor] = []
    h_non_country: list[torch.Tensor] = []
    h_test: dict[str, tuple[str, torch.Tensor]] = {}

    for i, p in enumerate(SOURCE_COUNTRY, 1):
        t0 = time.time()
        h = capture_h(model, tok, p)
        h_country.append(h)
        print(f"  [country {i:>2}] {p!r}  ||h||={h.norm().item():.2f}  ({time.time() - t0:.1f}s)")
    for i, p in enumerate(SOURCE_NON_COUNTRY, 1):
        t0 = time.time()
        h = capture_h(model, tok, p)
        h_non_country.append(h)
        print(f"  [non-country {i:>2}] {p!r}  ||h||={h.norm().item():.2f}  ({time.time() - t0:.1f}s)")
    for label, p in TEST_PROMPTS:
        t0 = time.time()
        h = capture_h(model, tok, p)
        h_test[label] = (p, h)
        print(f"  [test {label}] {p!r}  ||h||={h.norm().item():.2f}  ({time.time() - t0:.1f}s)")

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  base freed.")

    print(f"\n[3/4] computing country direction = mean(country) - mean(non-country) ...")
    country_mean = torch.stack(h_country).mean(dim=0)
    non_country_mean = torch.stack(h_non_country).mean(dim=0)
    direction_raw = country_mean - non_country_mean
    direction_norm = direction_raw.norm()
    direction_unit = direction_raw / direction_norm
    print(f"  ||country_mean|| = {country_mean.norm().item():.2f}")
    print(f"  ||non_country_mean|| = {non_country_mean.norm().item():.2f}")
    print(f"  ||diff (raw)|| = {direction_norm.item():.2f}")

    print()
    print(BAR)
    print("PROJECTION TABLE  (positive = aligned with 'country' direction, negative = away)")
    print(BAR)
    print(f"  {'label':<28}  {'projection':>10}  {'cos w/ dir':>10}  prompt")
    print("  " + "-" * (WIDTH - 2))

    results: list[tuple[str, float, float, str]] = []
    for i, p in enumerate(SOURCE_COUNTRY, 1):
        h = h_country[i - 1]
        proj = (h * direction_unit).sum().item()
        cos = (h * direction_unit).sum().item() / h.norm().item()
        results.append((f"SRC country {i}", proj, cos, p))
    for i, p in enumerate(SOURCE_NON_COUNTRY, 1):
        h = h_non_country[i - 1]
        proj = (h * direction_unit).sum().item()
        cos = (h * direction_unit).sum().item() / h.norm().item()
        results.append((f"SRC non-country {i}", proj, cos, p))
    for label, (p, h) in h_test.items():
        proj = (h * direction_unit).sum().item()
        cos = (h * direction_unit).sum().item() / h.norm().item()
        results.append((label, proj, cos, p))

    results.sort(key=lambda r: -r[1])
    for label, proj, cos, p in results:
        print(f"  {label:<28}  {proj:>+10.2f}  {cos:>+10.3f}  {p!r}")
    print(BAR)

    print(f"\n[4/4] AV-decoding the country direction (rescaled to ||h||=150) ...")
    t0 = time.time()
    av_model, av_tok, av_meta = load_av()
    print(f"  AV loaded in {time.time() - t0:.1f}s")

    direction_for_av = direction_unit * 150.0
    print(f"  ||direction_for_av|| = {direction_for_av.norm().item():.2f}")

    t0 = time.time()
    av_text = nla_verbalize(
        direction_for_av, model=av_model, tok=av_tok, meta=av_meta,
        max_new_tokens=200,
    )
    print(f"  AV reading ({time.time() - t0:.0f}s):")
    print()
    for line in av_text.splitlines():
        print(f"    {line}")
    print()

    torch.save({
        "h_country": h_country,
        "h_non_country": h_non_country,
        "h_test": h_test,
        "direction_unit": direction_unit,
        "country_mean": country_mean,
        "non_country_mean": non_country_mean,
        "av_text": av_text,
        "results": results,
    }, ARTIFACT)
    print(f"  artifact saved to {ARTIFACT}")


if __name__ == "__main__":
    main()
