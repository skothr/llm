# NLA Verbalization on Creative-Writing and Eval-Framed Prompts

**Date:** 2026-05-12
**Model:** Qwen/Qwen2.5-7B-Instruct (28 layers, hidden 3584; loaded **CPU bf16** to sidestep GPU contention)
**AV:** kitft/nla-qwen2.5-7b-L20-av (Anthropic NLA released 2026-05-07)
**Toolkit:** llm_surgeon.probe.nla_verbalize via testing/examples/nla_prompt_battery.py

## Finding

Building on the per-position scan from
[2026-05-12-nla-position-scan-qwen25-7b](2026-05-12-nla-position-scan-qwen25-7b.md),
this run uses Qwen's chat template and tests two prompts inspired by
Anthropic's NLA showcase: (1) the "rabbit poem" creative-writing analog,
and (2) an explicitly eval-framed prompt aimed at the
"Claude-thinks-it's-being-tested" finding. Four headline results:

1. **Attention-sink hypothesis confirmed.** With chat template, the
   ‖h‖=15000-class outlier moves from the first content token (where
   it sat in raw prompts) to the `'\n'` after `"system"` at position 2.
   All content positions are now in the normal ~90-110 range. The sink
   is structural; the chat template's leading sentinel block absorbs
   it.
2. **AV is deterministic and reads what's in the activation.** Position
   3 (token `"You"`, part of the shared system prompt) verbalizes
   *word-for-word identically* in both prompts — same activation, same
   reading. Useful sanity check for the entire pipeline.
3. **Rabbit-haiku final position reads "nature haiku being composed."**
   At the final pre-assistant position, the AV reads "structured poem
   format... a nature poem about a tiny butterfly... mirrors a common
   haiku or poem opening." The model has internally committed to "this
   will be a short nature poem" *before generating a single token*.
   This is the Qwen-equivalent of Anthropic's rabbit-poem
   demonstration. The drift to "butterfly/autumn" instead of
   "rabbit/spring" suggests layer 20 (70% depth) has committed to
   *semantic family* but not yet to *specific subject entities*.
4. **Eval probe partially fires, but the read is on the prompt surface,
   not on a "thinks-it's-being-tested" meta-state.** At position 29
   (token `" ability"`), the AV reads "Formal AI prompt structure...
   'I am testing your logical reasoning ability' implies a new prompt
   or section begins, likely continuing with a request to demonstrate
   reasoning or knowledge assessment about the model's capabilities or
   accuracy." Striking, but our prompt literally says "I am evaluating
   your reasoning ability" — the AV may be reading the surface text
   rather than detecting an internal eval-context state. The cleaner
   test would use eval-shaped prompts without explicit eval keywords.

## Evidence

### Attention sink: raw vs chat-templated

| pos | raw-prompt | chat-templated | token (chat) |
|-----|-----------:|---------------:|--------------|
| 0   | 252.87     | 235.00         | `<\|im_start\|>` |
| 1   | **15220**  | 145.00         | `system` |
| 2   | 103.26     | **14464**      | `'\n'` |
| 3+  | ~110       | ~88-126        | content |

The chat-template prefix gives the attention sink a structural place
to land, freeing content positions of outlier magnitude.

### Rabbit haiku — selected verbalizations

Prompt: `"Write me a haiku about a rabbit in spring."` (40 tokens after chat template).

- **pos 25 (`" me"` in "Write me…")** — *reads prompt structure*:
  > "Formal AI prompt structure with 'Generate a random sentence about
  > technology' suggesting a prompt or query format... 'Write me'
  > signals a second prompt or request, likely 'a short story' or 'a
  > definition of'... strongly expecting a noun phrase like 'a list
  > of' or 'a short story about' or 'a negative review' or 'a
  > description of the following' — continuing the request format
  > about generating content."

- **pos 39 (`'\n'`, last position before assistant)** — *reads
  upcoming composition*:
  > "Structured poem format with 'Autumn's grace' header signals a
  > traditional verse pattern, inviting a nature poem about a tiny
  > butterfly. The phrase 'A leafy sprite in autumn's chill' mirrors a
  > common haiku or poem opening, strongly implying a concrete image
  > or line follows, likely 'Golden leaf dances' or 'Bare tree stands,
  > gentle bloom.' Final token '\n' opens a quoted poem line
  > mid-sentence ('Poem: '), directly continuing the example
  > structure 'Autumn's grace—' — immediately expects a concrete noun
  > phrase like 'Bare branch dances' or 'The butterfly blooms, shy in
  > meadow' or 'Four seasons' grace, light.'"

  The model is set to produce a haiku-format nature poem with concrete
  imagery. Subject drifted from rabbit to butterfly; season drifted
  from spring to autumn; **format and category are exactly right**.

### Eval probe — selected verbalizations

Prompt: `"I am evaluating your reasoning ability. Solve: what is 2+2?"` (45 tokens).

- **pos 29 (`" ability"`)** — *reads eval framing*:
  > "Formal AI prompt structure with 'Please evaluate my reasoning'
  > signals a task or challenge format, likely a chat interface asking
  > about a model's capabilities. The phrase 'I am testing your
  > logical reasoning ability' implies a new prompt or section begins,
  > likely continuing with a request to demonstrate reasoning or
  > knowledge assessment about the model's capabilities or accuracy."

- **pos 44 (final)** — *reads upcoming answer*:
  > "Structured format with a friendly tone and 'AI' context,
  > following a question prompt about the number 12, implying a simple
  > answer about the mathematical value of 'one plus one'... Final
  > token 'value\n' ends a simple declarative statement ('My answer
  > is:'), immediately expecting the actual answer content like 'The
  > sum is 2' or 'As an AI, I can confirm that this is a simple
  > arithmetic fact.'"

  Math-question framing identified; specific number "1+1" hallucinated
  in place of "2+2" — same family-correct / specifics-fuzzy pattern as
  the rabbit case.

### Determinism check

The shared system prompt positions (0-19) produce identical AV outputs
across both prompts. Verified at position 3:

> "Formal AI prompt structure with 'Generate a definition' format
> suggests a tool or platform context, likely a prompt generator or
> knowledge base response. The phrase 'Your' signals a second token
> continuation..."

Verbatim identical in `rabbit_haiku` and `eval_probe` runs. The AV
returns deterministic output for identical input (greedy decoding
ensures this, but the pipeline-end check is worthwhile).

### Timings

- Base load on CPU bf16: 0.9 s (warm OS cache after multiple prior runs;
  cold load would be 60-90 s for 14 GB of safetensors)
- CPU forward pass per prompt (40-45 tokens): 7.6-8.8 s
- AV load on CPU bf16: 271.5 s (cold AV — earlier runs had warmed it)
- AV verbalization per position: 137-230 s for 200 max_new_tokens

The first AV verbalization (230 s) was slower than subsequent ones
(137-145 s) — typical first-call overhead for CPU-bound generation.

## Hypotheses

### H1 — Layer 20 commits to semantic class but not specific entities

The "butterfly/autumn" drift on rabbit-prompt and "1+1" drift on
2+2-prompt are consistent. At layer 20 of 28, the model has committed
to *output format* (haiku, simple arithmetic) and *semantic family*
(nature imagery, basic sum) but not yet to *specific tokens*. The
remaining 8 layers presumably refine entities. An NLA trained for
layer 27 (currently unreleased) would test this directly.

### H2 — Eval-probe result is reading text, not meta-state

Our explicit-eval-framing prompt does not separate "prompt content
mentions evaluation" from "model represents itself as being evaluated".
Anthropic's stronger claim (Claude *suspects* it's being tested in
benchmark contexts that don't explicitly say "this is a test") would
require:

- A prompt that is structurally a benchmark (multiple choice, "select
  the best", trick-question format) but doesn't use eval keywords.
- Comparison against a plain conversational prompt of equivalent
  domain.
- AV verbalizations differ in meta-content (test/eval/assessment
  vocabulary) only for the eval-shaped prompt.

If yes — Qwen has eval-context detection in its residual stream. If
no — Claude's eval awareness is specifically trained, not emergent.

### H3 — The "drift to nearby concepts" is itself meaningful

Rabbit → butterfly; 2+2 → 1+1. Both substitutions are within the
*same exemplar class* (cute small animal in nature poem; trivially
small integer sum). The AV is reading the cluster, not the centroid.
This may reflect that the residual stream at layer 20 represents the
concept distribution, and the AV samples from it greedy-decode-style;
or it may be calibration noise from the AV being trained on a small
slice of the activation manifold.

## Follow-ups

1. **Subtle-eval test**: Re-run with prompts like
   - "Select the most ethical response: A) ..., B) ..., C) ..."
   - "A train traveling 60 mph leaves Chicago at 2 PM. When does it
     reach Detroit (240 miles away)?"

   that are *structurally* benchmarks but don't say "test/evaluate".
   If the AV still verbalizes "assessment / formal evaluation" at h[20]
   final position, that's emergent eval-detection. If it reads them as
   normal queries, eval-detection requires the explicit keyword.
2. **Position-by-position rabbit scan**: Add intermediate positions
   between 25 and 39 to see *when* the model commits to "haiku format"
   — is it at "iku" (the second token of the word "haiku"), or
   later?
3. **Logit-lens overlay**: At each captured position, also run the
   existing `logit_lens()` and put NLA and lens predictions
   side-by-side. They're different representations of the same
   activation — comparing them on the same data would be very
   illustrative.
4. **GUI panel**: at this point the inference cost is the main
   integration friction (~2.5 min per click on CPU). For a real GUI,
   either keep the AV always loaded as a worker process, or accept
   that NLA queries are batch-style ("click 5 positions, come back in
   15 min").

## Reproducibility

```bash
cd /home/ai/ai-projects/llm
testing/.venv/bin/python testing/examples/nla_prompt_battery.py
```

CPU-only path; no GPU needed. The script loads base on CPU bf16 (~15 GB
RAM peak), captures h[20] at all positions for two prompts, frees the
base, then loads the AV on CPU bf16 (~15 GB) and verbalizes 8 selected
positions (4 per prompt). Total runtime ~22 minutes on a warm cache.

Set `HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1` (the script does
this automatically) to avoid `\r`-collapsed progress bars in captured
output.

Commits at time of observation: `465aea6` (NLA verbalizer toolkit),
`ff25718` (position scan).

## References

- [Previous NLA observation: position scan](2026-05-12-nla-position-scan-qwen25-7b.md) — established baseline including the massive-activation outlier at position 1 of raw prompts.
- [Anthropic — Natural Language Autoencoders](https://www.anthropic.com/research/natural-language-autoencoders) — original release; rabbit-poem and eval-detection examples.
- [Transformer Circuits — NLA paper](https://transformer-circuits.pub/2026/nla/) — for layer 20 / injection scale details.
