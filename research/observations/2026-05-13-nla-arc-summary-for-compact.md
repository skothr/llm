# NLA Research Arc — Summary for Compact Resume

**Date:** 2026-05-13
**Branch:** `session/nla-research` in worktree
`/home/ai/ai-projects/llm/.claude/worktrees/nla-research`
**Purpose:** Single-document reorientation for resuming after context
compaction. Covers what was built, what was found, what's saved, and
the natural next directions.

## Toolkit (durable, committed)

Module: `llm_surgeon.probe` (re-exported via `__init__.py`)

```python
from llm_surgeon.probe import (
    AV_ID, AR_ID,
    load_av, load_av_meta,       # AV (activation -> text), 8B bf16
    load_ar, load_ar_meta,       # AR (text -> activation), 5B bf16
                                  #   backbone + Linear(d, d) value head
    nla_verbalize,               # h -> English description
    nla_reconstruct,             # English -> h_pred
    nla_score,                   # cos + normalized MSE
)
```

Wraps Anthropic's NLA (released 2026-05-07; kitft checkpoints on HF)
in a CPU-only inference path that avoids SGLang. AV prompt template
uses the `'㈎'` injection char at one position; injection-scale 150;
generation greedy decode. AR uses the `"Summary of the following
text: <text>{explanation}</text> <summary>"` template, takes
`last_hidden_state[0, -1]` through the value head, returns raw
vector. Score normalizes both to magnitude `sqrt(d_model) = 59.87`
before MSE; cosine is magnitude-invariant.

## Hardware and artifact locations

- Base model, AV, AR all cached at
  `/home/ai/ai-projects/llm/testing/.cache/models/` (~40 GB combined)
- Worktree's `testing/.cache` is a symlink to the main checkout's
  cache — required for the worktree-per-session pattern to share
  models across worktrees
- Captures saved under `testing/.cache/nla_artifacts/`:
  - `aggregate_faithfulness.pt` — 113 captures across 8 prompts, full
    AV + AR roundtrip
  - `rabbit_haiku_gen_trajectory.pt` — 15 captures, single haiku
    generation
  - `forced_continuation.pt` — 10 captures, 4 teacher-forcing pairs
  - `country_concept_vector.pt` — 29 h's + extracted country direction
- Hardware: 8 GB RTX 2080, 31 GB RAM. AV/AR loaded on CPU bf16;
  GPU nf4 attempted but only ~1.2x speedup vs CPU bf16 on RTX 2080;
  CPU is the canonical path
- Inference timings on CPU: AV ~85 s per verbalization (200 tokens),
  AR ~7-20 s per reconstruction (single forward), base load ~5 min
  cold

## Findings (one line each, ordered by significance)

1. **NLA faithfulness is high across domains.** Mean cos +0.868 over
   113 captures spanning factual / math / code / creative / reasoning
   prompts; no capture below +0.70; histogram unimodal centered at
   [+0.85, +0.90). The AV does not catastrophically fail on any
   domain we tested.
2. **AV output is structurally templated, not free-form.** Every
   verbalization follows a fixed 3-paragraph shape: format
   classification, phrase context, final-token prediction. This is
   training-induced (SFT+GRPO with the AR) — required for round-trip
   stability — and is the reason the AV cannot say "I don't know" or
   "this is unusual" outside template slots.
3. **AV decoded outputs are superposed feature lists, not narrative
   descriptions.** The model's residual at h[20] has weight on
   multiple features simultaneously; the AV stitches them into one
   grammatical sentence. Outputs like "The capital of France is a
   country..." are literally Feature A's frame ("X is the capital
   of France") combined with Feature B's predicate ("France is a
   country") into one sentence under template constraints. Read AV
   outputs as feature lists, not natural-language descriptions.
4. **Subject drift across generation positions is real, not AV
   confabulation.** AR round-trip scoring on the haiku trajectory
   showed drift positions (rabbit -> butterfly -> moon) score
   cos~+0.84 — above the trajectory mean of +0.83. Layer 20 stops
   emphasizing specific entities once attention moves past them.
   Output coherence comes from attention retrieval, not layer-20
   residual carrying entity forward.
5. **NLA detects falsehoods only when they are NAMED in pretraining.**
   Forced-continuation probe showed AV at h[20] @ ' Berlin' (for
   "France's capital") flags "incorrect, correction, error"; AV at
   h[20] @ '5' (for "2+2=") names the Simpsons/1984 meme by name.
   But AV at h[20] @ 'No' (for "the sky is not blue") reads as
   generic denial — does not flag falsehood. Truth-detection
   requires the falsehood to be culturally tagged.
6. **Qwen2.5-7B does not have Claude-style eval-detection.** The
   "I'm sensing this is a test" forced completion's AV reading was
   generic conversational content; no meta-cognitive signal.
   Consistent with Anthropic's "thinks it's being tested" finding
   being Claude-specific.
7. **Mid-BPE-merger prediction is visible.** At step 27 of the haiku
   prompt scan ( ' ha' token), AV reads "expecting 'iku' to complete
   the word". At step 7 of the generation trajectory ( ' bree'),
   AV reads "breeze whispers". The model predicts through BPE
   splits before they commit — visible in h[20].
8. **Massive activation outliers at the first content position
   (without chat template) reach ||h||=15000+** — matches Sun et al.
   2024 documented phenomenon. Disappears with chat-template prefix
   because the structural sentinel block absorbs the attention sink.
9. **Single-layer / single-position steering is insufficient on
   Qwen2.5-7B.** Both AV-text-edit and direct-activation-transplant
   produced bit-identical greedy outputs to baseline. AV-edit was
   ~1% angular movement (low steering bandwidth); direct transplant
   showed entity choice flows through layers 0-18's attention back
   to prompt entity tokens (unmodified KV cache).
10. **CAV-style concept extraction works in NLA.** Difference-of-
    means over 8 country + 8 non-country prompts gave a direction
    that the AV reads as country/geopolitical content (travel guide,
    Ministry of Foreign Affairs, Korea, Egypt). Projection
    discriminates source groups; weird country-framing prompts
    project negative — direction encodes country-as-topic, not
    country-as-token-mention. **The AV's ability to name
    arithmetically-constructed vectors in English is the novel
    methodological contribution.**

## Commit chain on session/nla-research (chronological)

```
465aea6  feat(llm_surgeon): NLA verbalizer for Qwen2.5-7B layer-20 on 8 GB GPU
ff25718  research(nla): per-position scan finds in-distribution wins + outlier failures
668a269  research(nla): rabbit-haiku + eval-probe show prompt-vs-composition split
e5e5e81  chore(examples): force line-buffered stdout so background runs show live progress
45cbd67  research(nla): full position-by-position trajectory finds three new phenomena
ea33cfc  research(nla): generation-phase trajectory watches the model compose its haiku
de18951  research(nla): AR round-trip scoring shows the drift is real, not confabulation
999376a  wip(nla-research): recover NLA round-trip research scratch
94071c0  research(nla): aggregate faithfulness across 8 prompts — mean cos +0.868
d435e6b  research(nla): forced-continuation probe finds NLA detects named falsehoods only
af73877  research(nla): full token-by-token walkthrough of all 138 captures
5418b46  research(nla): CAV-style country direction; AV reads arithmetic vectors
c7d3c67  research(nla): negative-result steering scripts + worktree pyright fix
```

13 commits. Last 4 (94071c0, d435e6b, af73877, 5418b46, c7d3c67) on
worktree branch only; earlier ones already on master.

## Research observations on session/nla-research

```
2026-05-12-nla-position-scan-qwen25-7b.md            (initial scan, outliers)
2026-05-12-nla-prompt-battery-rabbit-eval.md         (chat-template fix, eval-framing)
2026-05-12-nla-trajectory-rabbit-haiku.md            (per-position prompt scan)
2026-05-12-nla-generation-trajectory-haiku.md        (per-step generation scan)
2026-05-12-nla-faithfulness-haiku.md                 (single-haiku AR round-trip)
2026-05-13-nla-aggregate-faithfulness-8-prompts.md   (mean cos +0.868 across domains)
2026-05-13-nla-forced-continuation-detects-named-falsehoods.md
2026-05-13-nla-cav-country-direction.md              (NLA + CAV pipeline)
2026-05-13-nla-walkthrough-all-captures.txt          (2030-line raw dump)
2026-05-13-nla-arc-summary-for-compact.md            (this file)
```

## Example scripts

All in `testing/examples/`:

- `qwen_load_check.py` — preflight: base loads in nf4 on RTX 2080
- `nla_verbalize.py` was an early version; replaced by `_nla.py` in
  the probe module
- `nla_roundtrip.py` — single h -> AV -> result demo
- `nla_scan.py` — per-position scan of prompt residual
- `nla_prompt_battery.py` — multiple prompts, chat-templated
- `nla_trajectory.py` — full per-token prompt trajectory
- `nla_gen_trajectory.py` — per-token generation trajectory
- `nla_faithfulness.py` — single-prompt AR round-trip pipeline
- `nla_aggregate_faithfulness.py` — multi-prompt aggregate
- `nla_dump_walkthrough.py` — reads all artifacts, dumps walkthrough
- `nla_forced_continuation.py` — teacher-forcing counterfactual probe
- `nla_country_concept_vector.py` — CAV extraction + AV-decoding
- `nla_steering.py` (negative result) — AV-text-edit steering
- `nla_steering_direct.py` (negative result) — direct activation
  transplant

All set up to be run with `PYTHONPATH=$PWD/testing` and the main
checkout's venv at `/home/ai/ai-projects/llm/testing/.venv/`.

## Natural next moves (in order of information yield)

1. **Concept arithmetic.** Extract France direction and Germany
   direction separately, compute "France - Germany", AV-decode.
   Does the AV produce something like "cultural distinctions
   between European countries"? Tests whether concept arithmetic
   is interpretable in natural language. ~15 min.
2. **Concept-vector steering.** The intentional follow-up to the
   failed direct-transplant steering. Use `h_normal + alpha *
   country_direction` for varying alpha and a non-country prompt;
   see at what alpha the output flips to country content.
   Quantifies steering threshold. ~20 min.
3. **Multi-concept library.** Repeat CAV methodology for "person",
   "math", "code", "first-person", "emotional". Build a directory
   of named directions. ~30 min per concept; could batch.
4. **SAE comparison.** Train or download a Qwen2.5-7B layer-20 SAE;
   compare its "country" feature direction to our difference-of-means
   estimate. Tests whether SAE-isolated features have lower
   within-group variance than ours. Multi-hour, biggest commitment.
5. **Test of H1 from the CAV observation.** Build a much narrower
   CAV using only "geopolitical discussion" prompts vs only
   "naming/labeling" prompts. Predict they're orthogonal directions,
   not subsets.

## How to resume after compact

The branch `session/nla-research` and this observation file are the
two things to read first. From there:

1. `cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research`
2. Read this file and the 9 observation files for context
3. Verify symlink exists: `ls testing/.cache` should show the link
   to main's cache. If missing, recreate with `ln -s
   /home/ai/ai-projects/llm/testing/.cache testing/.cache`
4. Pick a next move from the list above
5. Run via `PYTHONPATH=$PWD/testing
   /home/ai/ai-projects/llm/testing/.venv/bin/python
   testing/examples/<script>.py`

All artifacts persist across sessions; do NOT delete
`testing/.cache/nla_artifacts/*.pt` (those are the captured h's +
AV texts + AR reconstructions, hours of CPU compute).
