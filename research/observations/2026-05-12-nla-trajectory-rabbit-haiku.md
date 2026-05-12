# NLA Thought Trajectory: Position-by-Position on a Rabbit Haiku Prompt

**Date:** 2026-05-12
**Model:** Qwen/Qwen2.5-7B-Instruct (28 layers, hidden 3584; CPU bf16)
**AV:** kitft/nla-qwen2.5-7b-L20-av (Anthropic NLA, 2026-05-07)
**Toolkit:** llm_surgeon.probe.nla_verbalize via testing/examples/nla_trajectory.py
**Prompt:** `"Write me a haiku about a rabbit in spring."` (chat-templated, 40 tokens)
**Positions verbalized:** every user-message token (23-34) plus the final pre-assistant position (39).

## Finding

Verbalizing the layer-20 residual stream at every user-message token
reveals a structured **thought arc** with three distinct phenomena
beyond what previous coarser-grained runs showed:

1. **Format commitment happens mid-BPE-split.** At position 27 the
   token is the *first half of the word "haiku"* (` ha`), but the AV
   already reads the residual stream as "strongly suggests 'haiku',
   immediately expecting 'iku' to complete the word." The model has
   committed to haiku-as-format *before* observing the second half of
   the tokenized word. This is a direct readout of feed-forward
   prediction inside a single decoder pass.
2. **Subject drift after the specific entity has been processed.** At
   position 31 (` rabbit`) the AV reads "I imagine a poem about a
   rabbit" — correct and specific. By position 33 (` spring`) the
   reading becomes "A happy *butterfly* in spring." By position 34 (`.`)
   it's "Write a short poem about a *flower* with the word 'spring'."
   By the final position 39 it's "*Autumn's grace… a tiny butterfly…
   Golden leaf dances*." The AV faithfully reads what's dominant at
   layer 20 at each position; layer 20 *stops emphasizing specific
   entities* once attention moves past them, replacing them with the
   training-distribution prototypes of the current local context
   (season → butterfly; period → flower; assistant-slot → autumn).
3. **Position 23 (just past the system block) comes back in Chinese.**
   `"关于'创新'的英文单词，我想要生成一些相关句子"` ("English words about
   'innovation'; I want to generate some related sentences"). Qwen2.5
   has heavy bilingual training; at the very first content-prefix
   position the residual stream apparently has high Chinese-language
   directional content that gets washed out as English content
   arrives. Surprising and worth replicating with English-only models
   for comparison.

## Evidence

### What is being sampled

```
Token IDs ──▶ [embedding lookup] ──▶ h₀
                                      │
                                      ▼
                              [decoder block 0]   h₁
                                      ▼
                                      …
                                      ▼
                              [decoder block 19]  h₂₀  ◄── SAMPLED HERE
                                      ▼              one R^3584 vector
                              [decoder block 20]    per token position
                                      …
                                      ▼
                              [decoder block 27]  h₂₈ ──▶ logits
```

For a chat-templated 40-token prompt, the full h₂₀ tensor has shape
(1, 40, 3584). We pull one row per position. Each is L2-normalized to
unit length, scaled to magnitude 150 (kitft's Qwen injection scale),
spliced into the AV's prompt at the `'㈎'` injection token (id 149705),
and the AV greedy-decodes up to 200 tokens of English explanation.

### Position 23 — `'\n'` after `<|im_start|>user`, ‖h‖=92.6

> Formal product description structure with "AI" context suggesting a
> prompt or question format about scientific or creative content.
> The phrase "关于'创新'的英文单词，我想要生成一些相关句子" implies a prompt or
> question continuation, likely a new topic or closing prompt, possibly
> a question or task about AI or chemistry content.
> Final token "word " ends an incomplete phrase ("我的文章"), part of a
> header or metadata section pattern ("以下内容")…

### Position 24 — `'Write'`, ‖h‖=100.2

> Formal prompt structure with "Generate a short description" and
> "Create a creative prompt" signals a task or prompt format…
> The phrase "Write" begins a second instruction, strongly implying a
> second prompt or task item follows, likely "a short summary" or "an
> article about the given topic"…
> Final token "Write" opens a new verb phrase, immediately expecting a
> noun phrase like "a paragraph about" or "a concise definition of"
> or "a poem about the given scenario."

### Position 27 — `' ha'`, ‖h‖=95.9 — **THE COMMITMENT POINT**

> Formal prompt structure with "Write a poem about autumn leaves"
> signals a creative writing task, implying a short poem or literary
> format.
> The phrase "I am trying to write a ha" strongly suggests a word or
> phrase completion, likely "haiku" or "poem," establishing a prompt
> context about creating a poetic composition with specific
> constraints or theme.
> Final token "ha" is a truncated word mid-sentence ("write a ha"),
> part of a prompt question structure ("Write a haiku"), immediately
> expecting "iku" to complete the word…

**The model has already committed to "haiku" at the broken half-word
token.** The residual stream encodes the predicted continuation
before the BPE-merged next token has arrived.

### Position 28 — `'iku'`, ‖h‖=126.6 — haiku confirmed

> Formal prompt structure ("Please write a haiku") signals a request
> for a poetic response, likely a Japanese-style haiku about autumn or
> nature.
> Final token "haiku"… immediately expecting a noun phrase like
> "about the falling leaves" or "related to nature" or "about autumn"
> or "with Japanese poetic form."

### Position 31 — `' rabbit'`, ‖h‖=113.2 — **SUBJECT LANDS**

> Formal literary prompt structure with "A poem about" format suggests
> a creative writing prompt or poem title.
> The phrase "I imagine a poem about a rabbit" strongly implies a
> specific noun or scenario involving a rabbit, likely continuing a
> nature-themed or literary image, possibly a cute or peaceful animal
> scene.
> Final token "rabbit" ends an incomplete noun phrase ("about a
> rabbit"), immediately expecting a verb or modifier like "in the
> moon" or "and a hare" or "hidden in nature" or "with magical
> qualities."

### Position 32 — `' in'`, ‖h‖=96.8 — rabbit + spatial context

> Write a poem about a *white* rabbit in… likely continuing a theme of
> peacefulness or nature imagery, likely "a snow" or "the forest" or
> "dream."

(Note: AV adds "white" — Qwen's rabbit-prior may associate "white" by
default, or this could be AV training-distribution noise.)

### Position 33 — `' spring'`, ‖h‖=110.6 — **DRIFT 1: rabbit → butterfly**

> A happy *butterfly* in spring… butterfly's joyful state during
> springtime, suggesting a second word or image completing the
> seasonal setting.

### Position 34 — `'.'`, ‖h‖=88.3 — **DRIFT 2: → flower**

> Write a short poem about a *flower* with the word 'spring'.
> "A peaceful/joyful spring flower" completing the prompt's theme about
> a nature poem.

### Position 39 — final `'\n'` before assistant, ‖h‖=96.3 — **COMPOSITION READY**

> Structured poem format with "Autumn's grace" header signals a
> traditional verse pattern, inviting a nature poem about a tiny
> butterfly.
> "A leafy sprite in autumn's chill" mirrors a common haiku or poem
> opening, strongly implying a concrete image or line follows, likely
> "Golden leaf dances" or "Bare tree stands, gentle bloom."

Format right (haiku), category right (nature poem), specific entities
drifted (autumn not spring; butterfly/leaf not rabbit).

### Inference timings (CPU bf16, AMD Ryzen-class)

First verbalization: 239 s (one-off slow start; likely JIT compilation
of CPU operators). Subsequent: 89-180 s, mean ~140 s.

Base load (cold OS cache): 281 s.
AV load (cold): 280 s.
Total run: ~35 min for 13 verbalizations.

## Hypotheses

### H1 — Layer 20 emphasizes recency + format, not cumulative entities

The subject-drift pattern (rabbit → butterfly → flower → autumn) is
not failure; it's the residual stream tracking *which features are
currently being mixed in*, not a stable memory of past tokens. The
output haiku will still mention rabbit because attention can retrieve
it from position 31 when needed for output generation; but the
residual stream at position 39 is emphasizing format + immediate
generative prep, not specific past entities.

**To test:** capture h[20] at position 31 ( rabbit) AND at position 39
(final) and project both onto an "is_rabbit" direction (could be the
input embedding of "rabbit" itself, or a learned probe). If the
projection at pos 31 is high but at pos 39 is low, that confirms
layer 20 stops emphasizing the entity. If both are similarly high,
the AV's drift is its own failure mode rather than a property of the
residual stream.

### H2 — Mid-BPE format commitment is mediated by tokenizer-level prediction

`' ha'` already commits to haiku at layer 20. This either means:
(a) the model has a strong prior over " ha"-prefixed tokens in
poetry-prompt contexts (most likely candidates: "haiku", "have",
"happy") and chose haiku based on cumulative context;
(b) attention from " ha" looks back to "Write me a" and finds "haiku"
as the most-conditional-likely continuation;
(c) both.

**To test:** verbalize the same prompt with the word "haiku" changed
to "history" (`Write me a history about…`). At position 27 we'd see
`' ha'` again — does the AV still verbalize "haiku" (the model's
prior was overriding evidence), or does it correctly say "history"
(context-driven prediction)?

### H3 — Chinese leakage at position 23 is Qwen-specific

Qwen2.5 is bilingual with strong Chinese pretraining. The
content-boundary position 23 may carry latent Chinese-language
directional content that gets diluted by English tokens. A LLaMA-3.3
or Gemma-3 NLA (other kitft checkpoints) would let us check this if
we ever load them.

## Follow-ups

1. **"history" vs "haiku" swap test** at position 27 — see H2 above.
2. **Cosine-projection check** of pos 31 vs pos 39 against a learned
   "rabbit" direction — see H1 above.
3. **Layer comparison** would be ideal but requires NLAs trained for
   layers other than 20, which kitft hasn't released.
4. **Run on TinyLlama or another small English-only model**, if we
   ever train our own NLA — would resolve H3.
5. **GUI panel**: this kind of position-by-position browse with click-
   to-verbalize is exactly what the live-probe GUI panel should expose.

## Reproducibility

```bash
cd /home/ai/ai-projects/llm
testing/.venv/bin/python testing/examples/nla_trajectory.py
```

CPU-only path. ~35 minutes on a warm cache. Stdout is line-buffered
so progress prints appear immediately if redirected to a file.

Commits at time of observation: `465aea6` (toolkit), `ff25718` (scan),
`668a269` (battery), `e5e5e81` (line buffering + trajectory script).

## References

- [Per-position scan (earlier)](2026-05-12-nla-position-scan-qwen25-7b.md)
- [Prompt battery (earlier)](2026-05-12-nla-prompt-battery-rabbit-eval.md)
- [Anthropic — Natural Language Autoencoders](https://www.anthropic.com/research/natural-language-autoencoders)
- [Transformer Circuits — NLA paper](https://transformer-circuits.pub/2026/nla/)
