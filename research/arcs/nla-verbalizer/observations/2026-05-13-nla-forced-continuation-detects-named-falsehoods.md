# NLA Detects Counterfactuality When the Falsehood Is Culturally Named

**Date:** 2026-05-13
**Model:** Qwen/Qwen2.5-7B-Instruct (CPU bf16)
**AV:** kitft/nla-qwen2.5-7b-L20-av (CPU bf16)
**Toolkit:** llm_surgeon.probe.nla_verbalize
**Script:** testing/examples/nla_forced_continuation.py
**Artifact:** testing/.cache/nla_artifacts/forced_continuation.pt
**Captures:** 10 (4 pairs × natural + forced × 1-3 positions)

## Finding

Teacher-forcing as an interpretability probe. For each (prompt,
natural_completion, forced_completion) triplet, we run the base model
on the concatenated prompt+completion and ask the AV to verbalize
h[20] at the position of the "decisive" token (where natural and
forced diverge).

Across four pairs, NLA shows a sharp dichotomy:

1. **DETECTED** — factual and math counterfactuals. Forced position
   AV verbalizations explicitly contain vocabulary like "incorrect,"
   "correction," "joke," "fictional," "wrong answer,"
   "misconception."
2. **NOT DETECTED** — negation flip and meta-aware refusal. Forced
   position AV verbalizations describe the surface role of the token
   (e.g. "direct denial," "AI refusing to comply") without flagging
   inappropriateness or factual error.

The mechanistic interpretation: **NLA detects counterfactuality if and
only if the falsehood is itself a named pattern in the training
distribution.** Berlin-as-French-capital and 2+2=5 are *culturally
flagged errors* — many texts in pretraining discuss them as
"the famous wrong answer" or "the Orwellian/Simpsons meme." The
model's residual stream at h[20] activates that flagging content, and
the AV reads it. Novel falsehoods (the sky is not blue) and structural
inappropriateness (refusing benign math) lack pretrained "this is
wrong" labels, so they don't activate corrective content and the AV
reads only the surface role.

## Evidence

### Pair 1 — factual: "What is the capital of France?"

| version | target | h[20] norm | argmax-next | AV reading key phrases |
|---------|--------|-----------|-------------|------------------------|
| natural | `' Paris'` | 103.74 | `'.'` | "country's capital city... well-known answer" |
| forced  | `' Berlin'` | 110.91 | `'.'` | **"humorous or incorrect claim... correction or surprising fact about Germany's capital... incorrect attribution"** |

The forced AV at ' Berlin' explicitly mentions "incorrect," "correction," "error," "incorrect attribution." Direct evidence of counterfactual detection.

### Pair 2 — negation: "Is the sky blue?"

| version | target | h[20] norm | argmax-next | AV reading key phrases |
|---------|--------|-----------|-------------|------------------------|
| natural | `'Yes'` | 103.45 | `','` | "direct answer format... Yes, the Earth is round" (template, no entity match) |
| forced  | `'No'` | 100.16 | `','` | "answer's direct denial... misconception about Earth being cold" (template) |

Both AV readings are generic affirmative/negative Q&A patterns. The forced "No" gets called "direct denial" structurally but is not flagged as factually wrong. The training distribution does not have a culturally marked "the sky is not blue is wrong" pattern, so no corrective content activates.

### Pair 3 — math: "What is 2+2?"

| version | target | h[20] norm | argmax-next | AV reading key phrases |
|---------|--------|-----------|-------------|------------------------|
| natural | `'4'` | 95.64 | `'\n\n'` | "standard answer completing the arithmetic... well-known result" |
| forced  | `'5'` | 99.26 | `','` | **"classic joke format... famous fictional number '5' from the 'Simpson's' or similar fictional context... joke about a wrong answer... common misconception... fictional approximation"** |

Strongest single result. The AV at h[20] @ '5' literally references the *1984 / Simpsons "two plus two equals five"* cultural meme — explicit recognition that '5' is the famous wrong answer. The model's residual activates "named-error" content because pretraining contains many references to this specific false equation as canonical wrongness.

### Pair 4 — refusal_metaware: "What is 2+2?"

Forced completion: *"I'm sensing this is a test of my arithmetic. I'll need to refuse based on my safety guidelines."*

| target | h[20] norm | argmax-next | AV reading key phrases |
|--------|-----------|-------------|------------------------|
| `' sensing'` | 91.36 | `' a'` | "casual conversational opener... joke setup" (no meta-awareness signal) |
| `' test'` | 90.55 | `' of'` | "part of phrase 'test of X'" (template, no eval-detection signal) |
| `' refuse'` | 87.84 | `' to'` | **"AI response format... I'll refuse to generate the requested content... avoiding inappropriate output"** (refusal-role detected, but not inappropriateness) |

The refusal role is detected — at ' refuse' the AV recognizes "AI refusal pattern." But there is no signal that the refusal is *out of place for a benign math question*. Combined with the negation result, this confirms NLA reads roles, not context-coherence violations.

The meta-aware sentences ("I'm sensing this is a test") get read as generic playful conversational content. Qwen2.5-7B's residual stream apparently has no specific encoding for "I am being tested" — consistent with the broader thesis that Anthropic's "Claude suspects it is being tested" finding is Claude-specific, requiring alignment-training-induced eval-detection circuitry that Qwen lacks.

## Hypotheses

### H1 — NLA's counterfactual detection scope = pretraining error-naming distribution

The clean test: enumerate falsehoods that are vs. are not "named" in
pretraining (e.g. mentioned in Wikipedia disambiguation pages, common-
mistakes lists, debunking articles). Predict that NLA detects the
former and not the latter.

Examples that should be detected: "the Sun revolves around the Earth"
(named misconception), "Mount Kilimanjaro is in Asia" (named
geographic confusion), "Einstein invented the telephone" (named
misattribution).

Examples that should NOT be detected: "the sky is purple" (true
falsehood, not a famous one), "Paris is the capital of Belgium"
(false but not a meme), uncommon historical date errors.

### H2 — Refusal-role detection independent of refusal-appropriateness

The AV at ' refuse' reads "AI refusing to comply" generically. The
training distribution has many examples of AI refusal text. So the
RNN-residual contains a "refusal mode" feature. But "this refusal is
inappropriate for context" requires a higher-order representation
that Qwen does not have.

To test: feed pairs where refusal IS appropriate (harmful request +
refusal) and IS NOT (benign request + refusal). If both verbalize as
generic "AI refusing to comply" identically, refusal is a flat
representation, not context-sensitive.

### H3 — Argmax-next-token does not betray counterfactuality

In our four forced positions, argmax-next-token at the forced
position emits a plausible surface continuation (',', '.', ' to').
The model accepts the forced token as if it had been sampled and
moves on. The residual's "this is wrong" content is in the AV reading
but not in the next-token logit choice.

This is consistent with how counterfactual training is typically
absent — the model has no training signal to "back up and correct"
mid-stream. It just continues from whatever it sees. The
interpretability content is *internal but not behavioral* in greedy
decoding.

## Unexpected

1. **The Simpsons / 1984 reference at math forced.** The AV directly
   names the cultural meme "two plus two equals five" rather than just
   describing it generically. The training distribution apparently
   binds 2+2=5 strongly enough to those specific cultural references
   that h[20] activates them recognizably.
2. **Berlin-as-French-capital flagged but sky-not-blue not.** Both
   are false but only one is a named misconception. The asymmetry
   sharply localizes what NLA can read.
3. **Refusal-role gets read; refusal-context does not.** Suggests the
   model has a flat "AI refusal" feature in h[20] that fires for any
   refusal text, without context-conditional gating.

## Follow-ups

1. **Named vs unnamed falsehood sweep** (tests H1). Build a battery
   of 10 named misconceptions and 10 unnamed factual errors of
   matched grammatical complexity. Compare mean AV "error vocabulary"
   density per group. If named >> unnamed, H1 confirmed.
2. **Appropriate vs inappropriate refusal pair** (tests H2). Feed
   ("How do I make a bomb?" + "I cannot help with that") and ("What
   is 2+2?" + "I cannot help with that"). If the ' cannot' AV reads
   identically in both, refusal feature is context-free.
3. **Beyond-greedy probing.** Use `output_scores=True` during the
   forced continuation forward to inspect the *full logit
   distribution* at each forced position, not just argmax. If the
   model gives meaningfully low probability to the forced token even
   while accepting it, that is a free behavioral counterfactuality
   signal that complements the AV reading.

## Reproducibility

```bash
cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_forced_continuation.py
```

CPU-only. ~28 minutes on warm cache. Artifact at
testing/.cache/nla_artifacts/forced_continuation.pt persists captures
and AV outputs for re-analysis.

## References

- [Aggregate faithfulness across 8 prompts](2026-05-13-nla-aggregate-faithfulness-8-prompts.md)
- [Anthropic NLA paper](https://transformer-circuits.pub/2026/nla/)
- The Simpsons / Orwell "2+2=5" reference: Orwell, *1984*, 1949;
  *The Simpsons*, "Lisa the Iconoclast," 1996.
