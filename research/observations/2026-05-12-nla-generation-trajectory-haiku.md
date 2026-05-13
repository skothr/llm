# NLA Generation-Phase Trajectory: Watching the Model Compose a Haiku

**Date:** 2026-05-12
**Model:** Qwen/Qwen2.5-7B-Instruct (28 layers, hidden 3584; CPU bf16)
**AV:** kitft/nla-qwen2.5-7b-L20-av (Anthropic NLA, 2026-05-07)
**Toolkit:** llm_surgeon.probe.nla_verbalize via testing/examples/nla_gen_trajectory.py
**Prompt:** `"Write me a haiku about a rabbit in spring."`
**Generated:** `"Soft fur leaps through grass,\nSpring breezes whisper, light as hops—"` (15 tokens, greedy)

## Finding

Verbalizing the layer-20 residual stream at every *generation* step
(i.e., the state that *produced* each output token) surfaces four
distinct phenomena that the prompt-side scans couldn't:

1. **Multiple BPE-split anticipations.** Step 7 emits the partial token
   ` bree` and the AV at that step reads "second noun phrase... breeze
   whispers." Step 8 emits `zes` and the AV reads "zes caress / zes
   kiss" — completing the word from the half-token's state. This
   confirms the same "predicting through BPE splits" we saw at prompt
   pos 27 (` ha` → "haiku"), now visible during writing as well as
   reading.
2. **Pun / wordplay detection at the final step.** The model emits
   ` hops` (step 13) and then `—` (step 14). At step 14 the AV reads
   "playful wordplay about agility/springiness, 'All joy, like hops,
   sips — hops'" — it has detected the simultaneous use of "hops" as
   rabbit-motion *and* (via "sips") the beer-ingredient sense. Whether
   the model intended the pun or it's coincidental in the AV's
   sampling is hard to say from one example, but the AV is reading a
   double-sense layer-20 representation.
3. **Generic-nature drift continues** during generation. AV verbalizes
   reference "deer," "butterfly," "moon," "flower" at many steps even
   though the specific subject is rabbit. Same phenomenon documented
   in the prompt trajectory: layer 20 tracks *currently-dominant*
   features, not a cumulative entity store.
4. **Compositional coherence is preserved despite trajectory drift.**
   The actual emitted haiku — "Soft fur leaps through grass, Spring
   breezes whisper, light as hops—" — is internally consistent around
   "rabbit in spring," yet the per-step AV readings flicker among
   nature prototypes. The model's *output* is coherent because (a)
   attention can retrieve "rabbit" from the prompt for each generation
   step, and (b) sampling commits to specific tokens from the *full*
   logit distribution; the AV reading is only of the layer-20 slice.
   This is direct evidence that **layer-20 residual stream and final
   output are different objects** — the residual is one snapshot mid-
   computation, the output integrates multiple layers + attention.

## Evidence

### What is being sampled

```
At each generation step k:
  • Prefill (k=0): forward pass on the prompt → hidden_states[0]
    h[20] at the LAST prompt position is the state whose logits
    were argmaxed to produce generated_token_ids[0].
  • Decode step k≥1: forward pass on generated_token_ids[k-1]
    extending the KV cache → hidden_states[k]
    h[20] at the only new position is the state whose logits
    were argmaxed to produce generated_token_ids[k].

So an AV reading at step k = "what is the model 'thinking' that leads
to emitting token k". Captured via:

  gen = model.generate(
      input_ids=...,
      output_hidden_states=True,
      return_dict_in_generate=True,
      ...
  )
  for k in range(len(gen.hidden_states)):
      h = gen.hidden_states[k][LAYER][0, -1 if k == 0 else 0, :]
```

### Per-step trajectory (key entries)

#### Step 0 — emits `'Soft'`, ‖h‖=96.3

> Structured poem format with "Autumn's grace" header signals a
> traditional verse pattern, inviting a nature poem about a tiny
> butterfly. The phrase "A leafy sprite in autumn's chill" mirrors a
> common haiku or poem opening, strongly implying a concrete image or
> line follows, likely "Golden leaf dances" or "Bare tree stands,
> gentle bloom."

(Identical to position 39 of the prompt-side trajectory — same
residual state, since step 0 *is* the prefill last-position state.)

#### Step 1 — emits `' fur'`, ‖h‖=98.4 — **ENTITY-SPECIFIC PREDICTION**

> The opening word "Soft" begins a concrete noun phrase, clearly a
> concrete image continuing the first person's seasonal description,
> likely naming the rabbit's fur or form, with "Snowflakes" or similar
> nature imagery expected next.

The AV explicitly anticipates "the rabbit's fur" — and the model emits
" fur". h[20] at the position of `' Soft'`-output had the
"about-to-name-rabbit-attribute" semantic content.

#### Step 7 — emits `' bree'`, ‖h‖=104.5 — **MID-BPE PREDICTION 1**

> The phrase "Bud blooms on bare boughs, Spring" mirrors the paired
> subject ("Bird's joy, Spring's gentle breeze"), strongly implying a
> second noun phrase completing the seasonal description of nature's
> renewal or bloom.
> Final token "Spring" is mid-word in "Spring" repeated... immediately
> requiring a noun or verb describing the season's attributes, likely
> "breeze whispers"…

AV reads "breeze whispers" at the position just before emitting
` bree`. The completion was predicted from h[20] before the partial
token committed.

#### Step 8 — emits `'zes'`, ‖h‖=102.4 — **MID-BPE PREDICTION 2**

> The phrase "Rose blooms, sun's smile; Spring bree" is mid-sentence
> listing sensory details...
> Final token "bree" is a truncated word mid-sentence ("Spring bree"),
> part of a parallel sensory list ("Summer breezes, Sun bree..."),
> immediately requiring completion like "zes caress" or "zes kiss my
> cheeks"…

AV reads "zes caress" or "zes kiss" before the model emits `zes` to
complete "breezes." This is now twice that the AV at a BPE-partial
position correctly predicts the merger continuation.

#### Step 14 — emits `'—'`, ‖h‖=101.7 — **WORDPLAY DETECTION**

> The phrase "Like sun's dance, joyful as hops" is mid-simile listing
> qualities ("All day's joy, hops and hops"), implying a playful
> wordplay about the deer's agility or springiness.
> Final token "hops" ends an incomplete quoted phrase ("Like hops,
> with hops..."), part of a parenthetical list of qualities ("All joy,
> like hops, sips — hops"), strongly expecting "," or "a leap" or "the
> deer's fruit" to close the simile.

"Playful wordplay" and "hops, sips" — the AV simultaneously reads
"hops" as both the verb of motion (rabbit hops) and the noun (beer
hops, via "sips"). Whether this is the model genuinely activating both
senses or coincidence in the AV's training distribution is uncertain
from one example, but the reading is unambiguous.

### Generated haiku (greedy decode, max_new_tokens=15)

```
Soft fur leaps through grass,
Spring breezes whisper, light as hops—
```

5 / 5 / 3 syllables (line 3 truncated by token cap). Internally coherent,
imagery-rich, and ends with a double-meaning "hops".

### Timings

- Base load (CPU bf16, cold): 290.6 s
- Generation with output_hidden_states (CPU): 83.4 s for 15 tokens
  (5.6 s/token)
- AV load (CPU bf16, cold): 275.0 s
- AV verbalization: 80-167 s per step, mean ~93 s
- **Total run**: ~37 min for 15 verbalizations + loads

The first AV call (167 s) is a one-off slow start. Steady-state ~90 s
per verbalization on CPU.

## Hypotheses

### H1 — Layer 20 represents the next-BPE-merger before sampling commits

Steps 7-8 (Spring → ` bree` → `zes`) show a two-step prediction chain
visible in h[20]. To strengthen this:

- Capture h[20] at step 7 and project it onto the input embedding of
  `breezes` (the full target word). If the cosine is high relative to
  random tokens, that's quantitative confirmation of the
  thought-ahead reading.
- Compare to a counterexample where the partial token is genuinely
  ambiguous (`' ha'` could complete to haiku, happy, have, hand…)
  and see whether one direction dominates h[20].

### H2 — "Hops" wordplay is genuine, not AV confabulation

The dual-sense reading at step 14 could be:
- (a) Qwen's h[20] at that position genuinely activates both senses
  (rabbit-motion + beer-hops) due to "hops" being lexically polysemous;
  the AV faithfully reports the duality.
- (b) Qwen activates only the motion sense; the AV happens to mention
  "sips" because "hops" in its training distribution carries that
  association regardless of context.

**To test:** swap the rabbit prompt for a non-rabbit context that
also produces "hops" (e.g., "Write a brewery jingle including the
word hops"). If the AV at the analogous position reads "playful
wordplay between hopping and brewing," (a) is correct. If it reads
only "beer hops in brewing context," then the rabbit-motion sense in
our run was the AV adding hop-rabbit context that didn't exist in
h[20].

### H3 — The trajectory's coherence-at-output-but-drift-mid-stream is general

Composing produces internally-coherent output through multi-layer
integration + attention retrieval. The single-layer slice we read
shows drift because layer 20 isn't where the final entity binding
happens. If this is right, sampling the same h at layer 27 (last
layer, just before lm_head) should be entity-stable: at every step,
"rabbit / spring" should dominate. We can't test this without an NLA
trained for layer 27, but **a cosine-similarity check against a
"rabbit direction"** at each step would substitute and partially
test the hypothesis.

## Follow-ups

1. **Cosine projection check** at each step: project h[20] at every
   generation step onto an external "rabbit" direction (e.g.,
   embed("rabbit") from the base model). If projection is roughly
   constant, drift in the AV reading is the AV's own variance; if
   projection actually decreases over the trajectory, the residual
   stream really does lose the entity binding as composition
   proceeds.
2. **Hops-wordplay test** as in H2 above.
3. **Generation with a longer ceiling.** max_new_tokens=15 cut the
   haiku off at the em-dash. Letting it run to natural EOS would
   reveal whether the AV's reading at the final-line position
   anticipates closure / line break.
4. **Different prompts.** This whole arc has been on one prompt;
   replicate at least once on a different creative-writing prompt
   to check that the four phenomena above are general.
5. **GUI integration**: a click-on-token → verbalize panel that
   shows the trajectory side-by-side with the actual generated tokens
   is exactly what this experiment design is asking for.

## Reproducibility

```bash
cd /home/ai/ai-projects/llm
testing/.venv/bin/python testing/examples/nla_gen_trajectory.py
```

CPU-only. ~37 min on a warm cache; first run requires ~30 GB HF cache
to be populated (base + AV checkpoints, ~15 GB each). Output is line-
buffered so progress prints appear immediately during background runs.

Commit at time of observation: `45cbd67` (prompt-side trajectory),
plus testing/examples/nla_gen_trajectory.py (this run's source).

## References

- [Prompt-side trajectory](2026-05-12-nla-trajectory-rabbit-haiku.md) — same prompt, captures across the *input* tokens; establishes the BPE prediction and entity-drift phenomena that this run extends to the generation phase.
- [Prompt battery](2026-05-12-nla-prompt-battery-rabbit-eval.md) — two-prompt comparison.
- [Initial scan](2026-05-12-nla-position-scan-qwen25-7b.md) — established the layer-20 sampling baseline.
- [Anthropic — Natural Language Autoencoders](https://www.anthropic.com/research/natural-language-autoencoders) — the original release whose rabbit-poem demo this experiment is the open-weights analog of.
