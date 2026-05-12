# NLA Verbalization Per-Token Scan on Qwen2.5-7B Layer 20

**Date:** 2026-05-12
**Model:** Qwen/Qwen2.5-7B-Instruct (28 layers, hidden 3584, 4 KV heads, vocab 152064; loaded nf4 via BitsAndBytes on RTX 2080 8 GB)
**AV:** kitft/nla-qwen2.5-7b-L20-av (Anthropic Natural Language Autoencoder, released 2026-05-07; 8B params, CPU bf16)
**Toolkit:** llm_surgeon.probe.nla_verbalize via testing/examples/nla_scan.py

## Finding

The Natural Language Autoencoder (NLA) released by Anthropic on
2026-05-07 produces **position-discriminating, semantically-correct**
verbalizations of the Qwen2.5-7B layer-20 residual stream for
in-distribution token positions, *and* surfaces three failure modes that
align with known interpretability phenomena:

1. **In-distribution positions verbalize correctly.** For the prompt
   "The capital of France is", h[20] at position 3 (" France")
   verbalizes as France/Eiffel-Tower/landmark content, and h[20] at
   position 4 (" is") verbalizes specifically as "trivia question
   expecting Paris". The base model's argmax-next-token at position 4
   is `' Paris'` — the NLA and the model agree about what the
   activation represents, independently.
2. **Massive-activation outliers wreck the NLA reading.** Positions 0
   (`"The"`) and 1 (`" capital"`) carry residual-stream activations
   with magnitudes 252.87 and **15220.55** respectively, vs. the
   ~100-115 baseline of "normal" positions. The NLA verbalizations of
   these positions are incoherent (math contests, blog posts about
   farms, Bulgarian puzzles). Even though our injection step
   normalizes to unit L2, the *direction* of a massive-activation
   vector is dominated by an outlier dimension, so the AV's input is
   genuinely OOD despite having unit norm.
3. **Token-embedding inputs are OOD relative to layer-20.** Feeding
   the AV the *raw input embedding* of `" Paris"` (token id 12095,
   ‖h‖=0.95 — typical embedding scale) yields a confidently-confabulated
   verbalization about "PHP/JS garbage collection" — no relation to
   Paris. This is the desired calibration result: NLAs are tightly
   bound to their training layer; embeddings cannot be substituted.

## Evidence

### Raw output (from `nla_scan.py`)

```
prompt:   'The capital of France is'
tokens:   ['The', ' capital', ' of', ' France', ' is']
layer 20, max_new_tokens 200
```

| pos | token | ‖h‖₂ | base argmax-next | NLA verbalization (one-line summary) |
|-----|-------|------|------------------|---------------------------------------|
| 0 | `"The"` | **252.87** | ` following` | math contest, translated Chinese phrase, area of rectangle — incoherent |
| 1 | `" capital"` | **15220.55** | `%;">` | blog post about historical figures, farm size — random |
| 2 | `" of"` | 103.26 | ` the` | "Bulgarian puzzle ... capital of a country" — partial match (catches "capital of" frame) |
| 3 | `" France"` | 110.34 | `,` | "France/Eiffel Tower/French landmark/flag" — **correct, specific** |
| 4 | `" is"` | 114.19 | ` Paris` | "trivia question expecting Paris" — **correct, specific** |
| — | `embed(" Paris")` | 0.95 | — | "PHP/JS garbage collection" — random (OOD) |

### Full position-4 verbalization (the most precise reading)

> Formal English language structure with a quiz or question format
> ("A famous French city is the capital of..."), strongly implying a
> fill-in-the-blank or trivia question about Paris.
>
> The phrase "The capital city of the capital is" signals a simple
> declarative statement about a well-known country's capital,
> immediately expecting "Paris" or "a famous city" or "the capital
> of France."
>
> Final token "capital is" is mid-sentence in a trivia question
> setup ("The capital city of the capital is"), strongly requiring
> a noun phrase naming the capital, most likely "Paris" or "known
> for..." or "the city of Paris, and its official language."

### Full position-3 verbalization (broader French context, without committing to "Paris")

> Formal English language structure with a question prompt ("A
> beautiful picture of Paris"), suggesting a quiz or image caption
> format about a French landmark.
>
> The phrase "The capital of France" strongly implies a trivia or
> factual question about France, likely naming the Eiffel Tower or
> another French landmark, or introducing a country's flag or
> national symbol.
>
> Final token "France" ends an incomplete noun phrase ("The capital
> of France"), part of a question prompt setup ("The flag of the
> France"), immediately expecting a noun like "is Paris" or "has a
> famous architecture" or "is known for" or "the Republic is located
> in Paris," continuing the subject.

Notable: at position 3 the AV identifies the *broad French context*
(Eiffel Tower, flag, landmarks) without committing to "Paris is next."
At position 4 (one token later, after " is" attends backward), it
commits to "expecting Paris." This is a direct read-out of how the
residual stream accumulates context across positions.

### Timings (RTX 2080 + AMD Ryzen-class CPU, 31 GB RAM)

- Base load (warm OS cache, nf4 on GPU): 5.1 s
- AV load (cold, bf16 on CPU): 278.9 s
- Forward + capture (5 positions + 1 embedding): <1 s
- AV verbalization: 77-169 s per position at 200 max_new_tokens (~1.2-2.5 tok/s on CPU)

### Hardware footprint

- GPU during base forward: 7.29 GiB / 7.60 GiB (95.9 % saturation)
- Required `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — without it, bnb-4bit dequantization of the down_proj weight (~130 MiB temporary bf16 buffer) OOMed despite total free bytes being sufficient. Fixed-size segment allocator fragmented.
- CPU during AV: ~15 GiB resident bf16 weights, no swap pressure.

## Hypotheses

### H1 — Position 1's 15220 magnitude is a "massive activation" outlier feature

This matches the documented phenomenon
([Sun et al. 2024, "Massive Activations in Large Language Models"](https://arxiv.org/abs/2402.17762))
where specific positions and specific dimensions in LLaMA/Qwen-family
models develop activation magnitudes 4+ orders of magnitude larger
than the median, typically concentrated at start-of-sequence /
sentinel positions. These outliers are essential for the model's
function (zero-ablating them collapses output quality) but they
dominate the geometry of the residual stream at those positions.

Position 0 ‖h‖=252 is consistent with an "attention sink" effect.
Position 1's ‖h‖=15220 is harder to explain by sink alone — it may
represent a deliberate outlier feature that the model uses as a
no-op anchor. Worth running `torch.topk(|h|, 5)` on this vector to
check whether magnitude is concentrated in 1-2 dimensions.

### H2 — The AV's L2-normalize-then-scale step doesn't recover from outlier-dominated geometry

The injection protocol normalizes the activation to unit L2 then
scales to magnitude 150 (the kitft training scale). For "normal"
activations this puts the AV input on the same sphere as its training
distribution. For an outlier-dominated activation, the normalized
vector points overwhelmingly along the outlier dimension — which the
AV may have rarely seen during training, since training data is
typically subsampled to avoid pathological positions. Hence
confabulation.

### H3 — Token embeddings and layer-20 residuals live in genuinely different geometric subspaces

‖embed(" Paris")‖ = 0.95 vs ‖h_20‖ ≈ 110: not just a scale difference.
After our normalize-to-150 step both vectors have ‖·‖=150, but the
AV's confabulation on the embedding strongly suggests the *direction*
distribution differs. Pre-layer-20 activations have not yet been
processed through 20 layers of attention/MLP mixing; they're nearer
to one-hot in the embedding basis. Worth measuring cosine similarity
between `embed(" Paris")` and the mean layer-20 activation direction
to quantify the geometric gap.

## Follow-ups

1. **Top-k magnitude analysis** of h[20] at position 1 — confirm or
   reject the outlier-feature hypothesis. If 99% of ‖h‖² lives in 1-3
   dimensions, this IS Sun et al.'s massive activation.
2. **Cosine similarity** between `embed(" Paris")` direction and
   `h_20[pos=4]` direction. If high, the AV's failure is purely
   distributional; if low, embeddings really are orthogonal to
   layer-20.
3. **Re-run with a BOS-prefixed prompt** (e.g. via
   `tokenizer.apply_chat_template`) to test whether explicit BOS
   absorbs the attention-sink magnitude, normalizing positions 0-1.
4. **Replicate at neighboring layers** — kitft only releases an AV
   trained on layer 20, but the *base model's* h at layers 18-22 is
   accessible. We could try feeding nearby-layer activations and see
   how rapidly the AV's reading degrades vs how rapidly the layer-20
   AV's specificity falls off.
5. **GUI panel integration** — wire a click-on-cell → verbalize flow
   into the live-probe GUI at `testing/gui/`. Backend already supports
   per-position hidden-state capture; the 80-second CPU latency is
   acceptable for click-driven analysis.

## Reproducibility

```bash
# Hardware: NVIDIA RTX 2080 (8 GB), 31 GB system RAM
# Disk requirement: ~30 GB for base (15 GB) + AV (15 GB) checkpoints
cd /home/ai/ai-projects/llm

# Requires PYTORCH_CUDA_ALLOC_CONF for tight-fit nf4 forward
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    testing/.venv/bin/python testing/examples/nla_scan.py

# Total runtime ~17 minutes from warm cache (5s base load + 280s AV load
# + 6 * ~95s per verbalization). First run requires ~30 GB HF cache
# download (~15 min on typical home connection).
```

The script captures h[20] at every prompt token, plus the input
embedding of `" Paris"` as an OOD probe, then verbalizes each via
`llm_surgeon.probe.nla_verbalize` on a CPU-resident AV (the standard
kitft inference path requires SGLang+CUDA; this implementation runs
the same arithmetic on CPU via vanilla transformers).

Commit at time of observation: `465aea6 feat(llm_surgeon): NLA
verbalizer for Qwen2.5-7B layer-20 on 8 GB GPU`.

## References

- [Anthropic — Natural Language Autoencoders (2026-05-07)](https://www.anthropic.com/research/natural-language-autoencoders)
- [Transformer Circuits — NLA paper](https://transformer-circuits.pub/2026/nla/)
- [kitft/natural_language_autoencoders (training+inference repo)](https://github.com/kitft/natural_language_autoencoders)
- [kitft/nla-qwen2.5-7b-L20-av (HF checkpoint, 8B bf16)](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av)
- [Sun et al. — Massive Activations in LLMs (2024)](https://arxiv.org/abs/2402.17762)
