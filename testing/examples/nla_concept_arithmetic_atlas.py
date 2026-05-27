"""MAIN-48 — Concept arithmetic atlas.

Tests whether NLA preserves word2vec-style additive/subtractive structure
on layer-20 end-of-prompt h vectors. AV-decode each arithmetic
combination after rescaling to ||h|| = 150 (typical eop norm).

Inputs from `vocab_atlas.pt`:
  - per-anchor h: France, Germany, Japan, Brazil, Egypt, United Kingdom,
                  China, Italy, Russia, India, Paris, Berlin, Tokyo,
                  London, Madrid, happy, sad, ...
  - category centroids: country, capital, emotion, ...

Combinations (canonical analogy + pure subtraction + axis directions +
compound concepts).

Output:
  testing/.cache/nla_artifacts/concept_arithmetic_atlas.pt
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ["HF_HUB_OFFLINE"] = "1"
for _v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_v, None)

import sys
from io import TextIOWrapper
from typing import Any, cast
cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

import gc
import time
from pathlib import Path
import torch

from llm_surgeon.probe import load_av, nla_verbalize


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = _REPO_ROOT / "testing" / ".cache" / "nla_artifacts"
OUT = ARTIFACTS / "concept_arithmetic_atlas.pt"
TARGET_NORM = 150.0


def find_h(captures: list[dict[str, Any]], word: str) -> torch.Tensor:
    for c in captures:
        if c["word"] == word:
            return cast(torch.Tensor, c["h"])
    raise KeyError(word)


def centroid(captures: list[dict[str, Any]], category: str) -> torch.Tensor:
    hs = [c["h"] for c in captures if c["category"] == category]
    if not hs:
        raise KeyError(category)
    return torch.stack(hs).mean(dim=0)


def rescale(h: torch.Tensor, target_norm: float = TARGET_NORM) -> torch.Tensor:
    n = float(h.norm().item())
    if n < 1e-9:
        return h.clone()
    return h * (target_norm / n)


def main() -> None:
    print(f"loading vocab atlas from {ARTIFACTS / 'vocab_atlas.pt'}")
    vocab = torch.load(ARTIFACTS / "vocab_atlas.pt", weights_only=False)
    caps = vocab["captures"]
    print(f"  {len(caps)} anchor captures, {len(vocab['categories'])} categories")

    # Build the arithmetic combinations
    combos: list[dict[str, Any]] = []

    # 1: canonical analogy — Paris is to France as Berlin is to Germany
    h1 = find_h(caps, "Paris") - find_h(caps, "France") + find_h(caps, "Germany")
    combos.append({
        "label": "Paris − France + Germany",
        "predict": "Berlin",
        "h_raw": h1,
        "category": "analogy",
    })

    # 2: analogy — Tokyo to Japan as ? to France
    h2 = find_h(caps, "Tokyo") - find_h(caps, "Japan") + find_h(caps, "France")
    combos.append({
        "label": "Tokyo − Japan + France",
        "predict": "Paris",
        "h_raw": h2,
        "category": "analogy",
    })

    # 3: analogy — Berlin to Germany as ? to United Kingdom
    h3 = find_h(caps, "Berlin") - find_h(caps, "Germany") + find_h(caps, "United Kingdom")
    combos.append({
        "label": "Berlin − Germany + United Kingdom",
        "predict": "London",
        "h_raw": h3,
        "category": "analogy",
    })

    # 4: pure subtraction — what's "France-ness minus Germany-ness"?
    h4 = find_h(caps, "France") - find_h(caps, "Germany")
    combos.append({
        "label": "France − Germany",
        "predict": "French-vs-German residual (low magnitude expected)",
        "h_raw": h4,
        "category": "subtraction",
    })

    # 5: axis direction — country - capital centroid
    h5 = centroid(caps, "country") - centroid(caps, "capital")
    combos.append({
        "label": "country_centroid − capital_centroid",
        "predict": "country-not-capital axis (geographic entity vs city)",
        "h_raw": h5,
        "category": "axis",
    })

    # 6: valence axis
    h6 = find_h(caps, "happy") - find_h(caps, "sad")
    combos.append({
        "label": "happy − sad",
        "predict": "positive-vs-negative emotional valence",
        "h_raw": h6,
        "category": "axis",
    })

    # 7: compound concept — country + emotion centroid sum
    h7 = centroid(caps, "country") + centroid(caps, "emotion")
    combos.append({
        "label": "country_centroid + emotion_centroid",
        "predict": "compound (nation + feeling)",
        "h_raw": h7,
        "category": "compound",
    })

    print(f"\nbuilt {len(combos)} arithmetic combinations")
    for c in combos:
        n = float(c["h_raw"].norm().item())
        print(f"  {c['label']:<45}  ||h_raw||={n:>7.2f}  predict: {c['predict']}")

    # Rescale to TARGET_NORM
    for c in combos:
        c["h_rescaled"] = rescale(c["h_raw"], TARGET_NORM)

    print(f"\nloading AV ...")
    t0 = time.time()
    av_model, av_tok, av_meta = load_av()
    print(f"  AV loaded in {time.time() - t0:.1f}s")

    # AV-decode each
    print(f"\nAV-decoding each combination (~85s each on CPU bf16) ...")
    for i, c in enumerate(combos, 1):
        t0 = time.time()
        text = nla_verbalize(
            c["h_rescaled"], model=av_model, tok=av_tok, meta=av_meta,
            max_new_tokens=180,
        )
        c["av_text"] = text
        elapsed = time.time() - t0
        first_line = (text.splitlines() or [""])[0][:140]
        print(f"  [{i}/{len(combos)}] {c['label']:<45}  ({elapsed:.0f}s)")
        print(f"      first-line: {first_line}")

    del av_model, av_tok
    gc.collect()

    torch.save({
        "combos": combos,
        "target_norm": TARGET_NORM,
        "base_id": vocab.get("base_id"),
        "layer": vocab.get("layer"),
    }, OUT)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
