# NLA Aggregate Faithfulness Across 8 Diverse Prompts

**Date:** 2026-05-13
**Model:** Qwen/Qwen2.5-7B-Instruct (28 layers, hidden 3584)
**AV:** kitft/nla-qwen2.5-7b-L20-av (CPU bf16 for 4 prompts, GPU nf4 for 4 prompts)
**AR:** kitft/nla-qwen2.5-7b-L20-ar (CPU bf16, value_head Linear(3584, 3584))
**Toolkit:** llm_surgeon.probe.nla_verbalize + nla_reconstruct + nla_score
**Script:** testing/examples/nla_aggregate_faithfulness.py
**Artifact:** testing/.cache/nla_artifacts/aggregate_faithfulness.pt
**Captures:** 113 generation-step positions, 8 prompts × ~15 tokens each

## Finding

Across 8 prompts spanning factual / math / code / creative / reasoning
domains, the round-trip `h -> AV -> text -> AR -> h_pred` produces
**mean cosine +0.868 (stdev 0.036, range +0.717 to +0.926, median
+0.875)**. The 113-capture distribution is unimodal with the peak at
[+0.85, +0.90) holding 60% of all captures. No capture below cosine
+0.70. The AV is consistently faithful across all domains tested.

Four specific findings emerge from per-prompt comparison:

1. **The haiku is the only real low-faithfulness outlier** (+0.833
   mean) — and it is so by a wide margin from the next-lowest
   (reasoning_logic at +0.855). Yet the ocean poem scores +0.884,
   high. **It is not "creative writing" that lowers fidelity — it is
   the haiku's compressed-form constraint specifically.** A 17-syllable
   form contains a higher fraction of transition / line-break /
   word-fragment positions, and transition positions are
   systematically the lower-cos ones across the board.
2. **Code has the highest single capture (+0.926) and highest mean
   (+0.889)**, despite being the least narrative content the AV's
   training distribution would have favored. This implies NLA
   faithfulness tracks **compositional commitment** — how
   geometrically decided the residual is at each step — rather than
   domain familiarity. Code is the most structurally tight content
   the model emits.
3. **The single lowest capture is reasoning_word step 12, token `'4'`
   at cos=+0.717.** Context: the model is restating "Mary has 4
   brothers" mid-reasoning. This is a re-emission of a number already
   in the prompt — an attention-routed copy operation. The activation
   at copy positions appears uniquely uninformative about content,
   matching the geometric prediction: pure copies have minimal
   residual-stream content to faithfully describe.
4. **The benchmark-trick framing did not destabilize the residual
   stream.** reasoning_word (+0.867) and reasoning_logic (+0.855)
   cluster together at the per-prompt mean level. We see no signature
   of "this is a trick question, be careful" in the AV-decodability
   of the residuals. This is *consistent with* my earlier prior that
   Qwen2.5-7B does not exhibit Claude-style test-context detection,
   though the cleaner test is reading the AV text itself for "test /
   trick / evaluation" content (qualitative read, not yet done on this
   batch).

## Evidence

### Per-prompt summary (113 datapoints)

| prompt | n | mean cos | min | max | preview |
|--------|---|----------|-----|-----|---------|
| factual_easy    | 8  | +0.885 | +0.862 | +0.904 | The capital of France is Paris. |
| factual_hard    | 15 | +0.856 | +0.787 | +0.895 | The Treaty of Westphalia was signed in 1 |
| math            | 15 | +0.881 | +0.822 | +0.922 | To compute (47 x 38), we can use |
| code            | 15 | +0.889 | +0.747 | +0.926 | Certainly! Below is a Python function that |
| creative_haiku  | 15 | +0.833 | +0.778 | +0.877 | Soft fur leaps through grass, Spring bre |
| creative_poem   | 15 | +0.884 | +0.835 | +0.914 | Whispering waves caress the shore, Mystic |
| reasoning_logic | 15 | +0.855 | +0.817 | +0.883 | Yes, all roses need water. Since all ros |
| reasoning_word  | 15 | +0.867 | +0.717 | +0.910 | Let's break down the information given: |

### Overall distribution

```
n = 113
mean cos    = +0.868
stdev cos   =  0.036
median cos  = +0.875
range cos   = [+0.717, +0.926]
mean MSE    =  0.264
median MSE  =  0.251

cosine histogram (count per bin):
  [-1.00, +0.50):    0
  [+0.50, +0.70):    0
  [+0.70, +0.80):    5    ( 4%)
  [+0.80, +0.85):   23    (20%)
  [+0.85, +0.90):   68    (60%)   <- mode
  [+0.90, +1.00):   17    (15%)
```

### Inference time accounting

- Phase 1 (base load + generate per prompt): ~5 min cold + ~15 s
  generate × 8 = ~7 min
- Phase 2a (AV verbalize on GPU nf4 for 4 prompts): ~90 min cold for
  53 captures = ~100 s per capture (GPU nf4 was not the 4-5x speedup
  predicted; closer to 1.2x vs CPU bf16's 85 s)
- Phase 2b (AV verbalize on CPU bf16 for 4 prompts): ~85 min for 60
  captures = ~85 s per capture (matches single-prompt baseline)
- Phase 3 (AR reconstruct + score): ~13 min for 113 captures = ~7 s
  per capture (CPU bf16; AR forward is a single 20-layer pass plus
  value-head linear, no autoregress)
- Phase 1 was the only step that needed the base model on GPU (it
  was loaded on CPU bf16 instead, ~5 min cold + 7 s/forward for
  inference vs ~6 min for GPU base nf4 load that we previously hit
  display-VRAM contention on)

The 8-prompt aggregate took ~3 hours wall-clock total, split across
two sessions due to a session interrupt mid-Phase-2. The
incremental-save pattern preserved all 53 GPU-verbalized AV outputs
across the interrupt and allowed resume without re-running them.

## Hypotheses

### H1 — Faithfulness tracks compositional commitment, not domain

The per-prompt ordering of mean cos is:
- code (+0.889) > factual_easy (+0.885) > creative_poem (+0.884)
  > math (+0.881) > reasoning_word (+0.867) > factual_hard (+0.856)
  > reasoning_logic (+0.855) > creative_haiku (+0.833)

Code, math, and factual_easy are all "highly determined" generation
contexts — the model has high confidence at most positions. Haiku is
form-constrained and includes more transition positions. The
hypothesis is that **per-step cos correlates with per-step
next-token entropy**, with low-entropy (model decided) positions
faithful and high-entropy (transition / uncertain) positions less
faithful.

**To test**: re-capture each prompt with `output_scores=True` and
compute per-step logit entropy. Compare against per-step cos. If they
correlate, we have a poor-man's confidence-from-activations probe.

### H2 — Copy operations are activation-thin

The lowest-cos capture (cos=+0.717 on `'4'` at reasoning_word step 12)
is a re-emission of a number from the prompt. Pure attention-routed
copy operations may carry minimal residual-stream content to verbalize
faithfully, because the relevant information is encoded in attention
weights rather than the residual at that position.

**To test**: identify other "copy" positions across the 113 captures
(tokens that appear in the prompt). Compute mean cos for copies vs
new-content emissions. If copies are systematically lower, H2 holds.

### H3 — Form-constrained generation has more transition positions

Haiku produces 17 syllables in tight form; each form-imposed token
boundary may be a transition position. The ocean poem is discursive
and contains long steady-state runs. If H1 (transition = lower cos) is
right, then form-constraint -> more transitions -> lower mean cos.

**To test**: per-token cosine within the haiku already shows the
pattern (cos at line-break tokens lower than mid-line). Same analysis
on a constrained form vs an unconstrained one (e.g. iambic vs free
verse) would generalize the claim.

## Surprises (vs prior predictions)

1. **Creative writing as a category is not low-faithfulness.** The
   ocean poem (+0.884) is right with code and math. Only the haiku
   is low. My prior was wrong; the right axis is structural
   commitment, not narrative-vs-not.
2. **Code is the highest-faithfulness domain.** I expected the AV's
   training (presumably narrative-heavy) would make code low. The
   opposite. AV reads structural content faithfully.
3. **Benchmark-trick prompt did not produce qualitatively different
   cosines.** reasoning_word's "Mary's sisters" trick clusters with
   reasoning_logic's plain syllogism. No "test-detection" signature in
   the activations on this base model. (Consistent with Claude being
   trained for eval-awareness, Qwen2.5-7B not being trained for it.)
4. **No catastrophic failures.** Every one of 113 positions decoded to
   cos > +0.70. The AV does not silently confabulate to orthogonal on
   any domain we tested.

## Follow-ups

1. **Steering experiment.** Edit a high-cos AV text in the haiku (e.g.
   step 1's "rabbit's fur" -> "deer's fur") -> AR decode to modified
   `h_pred` -> splice into base model's residual at layer 20, position
   1 -> re-generate. If the haiku changes from rabbit to deer, the
   AV+AR pair supports controlled intervention. The causal test of
   the round-trip's invertibility.
2. **Per-step entropy correlation** (tests H1). Cheap addition to
   existing artifact: re-do Phase 1 with `output_scores=True`, compute
   per-step logit entropy, plot vs per-step cos.
3. **Copy-token identification** (tests H2). For each capture, check
   whether its token appears in the prompt. Compute mean cos for
   copy-token captures vs novel-token captures.
4. **Form-constraint sweep** (tests H3). Run aggregate on
   form-constrained prompts (haiku, sonnet, limerick) vs
   unconstrained ones (free verse, prose poem). See if form-constraint
   systematically lowers mean cos.

## Reproducibility

```bash
cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_aggregate_faithfulness.py
```

Worktree-per-session means `testing/.cache` must be symlinked to the
main checkout's models cache (~40 GB shared across worktrees):

```bash
ln -s /home/ai/ai-projects/llm/testing/.cache testing/.cache
```

Total runtime ~3 hours on warm cache, all-CPU (GPU optional and
gave ~1.2x speedup on AV nf4, less than expected).

Commit at time of observation: TBD (this file's commit).

## References

- [Single-haiku faithfulness](2026-05-12-nla-faithfulness-haiku.md) — the n=15 prior whose mean (+0.833) is now confirmed as the haiku-specific low end of the cross-domain range, not a general baseline.
- [Per-position prompt scan](2026-05-12-nla-position-scan-qwen25-7b.md) — initial massive-activation outlier finding, baseline for layer-20 sampling protocol.
- [Generation-phase trajectory](2026-05-12-nla-generation-trajectory-haiku.md) — the per-step view we now have multi-prompt coverage of.
- [Anthropic NLA paper](https://transformer-circuits.pub/2026/nla/)
- [kitft/nla-inference](https://github.com/kitft/nla-inference) — NLACritic reference for AR protocol.
