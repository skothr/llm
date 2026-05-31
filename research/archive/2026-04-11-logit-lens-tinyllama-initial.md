# Logit Lens Initial Observations — TinyLlama 1.1B

**Date:** 2026-04-11
**Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0 (22 layers, 2048 hidden, 32 heads, fp16, CPU)
**Prompt:** "The capital of France is"
**Tool:** llm_surgeon.probe.logit_lens (deterministic, no sampling/temperature)

## Findings

### Factual recall crystallizes in layers 18-19
- Layers 0-14: incoherent subword fragments ("nt", "omorph", "ilon", "lets")
- Layer 17 attn: "France" first surfaces (10%) — context gathering
- Layer 18 ffn: "Paris" first appears as top-1 (34%) — factual retrieval
- Layer 19 ffn: locks in at 90.9% on "Paris"
- Layers 20-21: maintain "Paris" but confidence drops slightly to 50.7%

### Attention gathers context, FFN retrieves facts
At layer 18: attention top-1 is "city" (20.5%), then FFN shifts to "Paris" (34.2%). Classic pattern: attention identifies the query type, FFN performs the lookup.

### Fill-in-the-blank token as valid prediction
`____` (token id 7652) appears in top-3 at layers 16 and 18-19. The model considers "The capital of France is ____" as a plausible continuation — a fill-in-the-blank template.

### Early-layer "attractor" tokens
Subword fragments like "nt", "omorph", "omorphic" dominate layers 0-8. These may be high-frequency subword attractors that the output head projects onto when the hidden state lacks semantic content. Hypothesis: these attractors are mostly prompt-independent (properties of the embedding space geometry, not the input).

### Hidden state similarity
- Layer 0→1 most similar (0.95 cosine) — early layers make small adjustments
- Middle layers ~0.83-0.88 — moderate refinement
- Layer 20→21 drops to 0.81 — final layer makes relatively large changes

### Intervention sensitivity
- Zeroing layer 10 FFN collapses output entirely (all near-zero probabilities)
- Noise std=0.5 at layer 5 is enough to destroy "Paris" prediction
- Factual recall at layers 18-19 requires clean signal propagation through middle layers

## Reproducibility

```bash
cd /home/ai/ai-projects/llm/testing
.venv/bin/python examples/probe_demo.py
```

## Follow-ups
- Compare early-layer attractor tokens across different prompts to test prompt-independence hypothesis
- Identify which specific layers/heads are critical for factual recall (systematic ablation)
- Test on OpenLLaMA 3B to see if the same crystallization pattern holds at larger scale
- Build live visual interface for interactive exploration before extensive testing
