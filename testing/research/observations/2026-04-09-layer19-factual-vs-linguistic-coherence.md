# Low-BI Layer Removal Preserves Linguistic Coherence but Destroys Factual Accuracy

**Date:** 2026-04-09
**Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0 (22 layers, 2048 hidden, 1.1B params)
**Toolkit:** llm-surgeon Phase 4 (block_influence + perplexity measurement)

## Finding

Removing layer 19 (lowest Block Influence score, BI=0.0616) from TinyLlama caused minimal perplexity degradation (+1.41, from 6.45 to 7.85 on WikiText-2) but dramatically impaired factual accuracy in generation. The modified model produces fluent, grammatically correct English that is almost entirely fabricated — confident-sounding but wrong facts, invented references, hallucinated URLs, and incorrect etymologies.

This suggests that BI scores (and by extension, perplexity) primarily measure **linguistic/statistical coherence** — whether the text flows naturally — but fail to capture **factual grounding** — whether the model retrieves correct knowledge. A geometrically small perturbation in the residual stream can apparently carry a disproportionate amount of factual steering information.

## Evidence

### BI Scores (all 22 layers)

```
Layer 19: BI = 0.0616  ← lowest, removed
Layer 17: BI = 0.0640
Layer 11: BI = 0.0685
Layer 18: BI = 0.0708
Layer 13: BI = 0.0713
Layer 14: BI = 0.0739
Layer 12: BI = 0.0753
Layer 15: BI = 0.0781
Layer 10: BI = 0.0810
Layer 16: BI = 0.0862
Layer  9: BI = 0.0881
Layer 20: BI = 0.0898
Layer  8: BI = 0.0974
Layer  6: BI = 0.1072
Layer  5: BI = 0.1074
Layer  4: BI = 0.1103
Layer  7: BI = 0.1180
Layer  1: BI = 0.1416
Layer  3: BI = 0.1526
Layer  0: BI = 0.3008
Layer 21: BI = 0.3059
Layer  2: BI = 0.3340
```

### Perplexity

- Baseline: 6.45 (WikiText-2, 50 samples, eval mode fp16)
- After removing layer 19: 7.85 (delta: +1.41, ~22% increase)

### Generation comparison (prompt: "What is the capital of France?")

**Original TinyLlama** (multiple runs):
- "The capital of France is Paris, also known as the City of Light or La Ville Lumière."
- "France's capital and largest city is Paris, located in the northern region of Île-de-France."
- "The capital of France is Paris."
- Consistently correct, concise, factually accurate.

**TinyLlama with layer 19 removed** (multiple runs):
- Run 1: "The name 'Paris' itself derives from the Latin word parisum, which literally reads as 'pier-town'" — **fabricated etymology** (parisum means "being born" in Latin, not "pier-town"; Paris actually derives from the Parisii, a Gallic tribe)
- Run 1: "officially named Paris after the French King Philip II (1840-1790)" — **impossible dates** (Philip II reigned 1180-1223)
- Run 1: "The Romans built the nearby Colle de Vallet, which later evolved into the present-day Loub dreams" — **nonsensical place names**
- Run 2: "The current presidential residence, the Hôtel du Pregarden (Picardie)" — **fabricated building** (actual residence is Élysée Palace); followed by entirely hallucinated URLs
- Run 2: Generated fabricated academic citations and reference links
- Both runs: Fluent English, proper grammar, coherent paragraph structure, but facts are largely or entirely wrong

### Key qualitative differences

| Dimension | Original | Layer 19 removed |
|-----------|----------|-----------------|
| Grammar/syntax | Correct | Correct |
| Fluency | Natural | Natural |
| Response length | Concise (1-3 sentences) | Verbose (multiple paragraphs) |
| Factual accuracy | High | Very low (confident confabulation) |
| Stop behavior | Appropriate | Tends to over-generate |
| Hallucination | Rare | Pervasive |

## Reproducibility

### Compute BI scores and perplexity

```python
from llm_surgeon import surgery, inspect, benchmark

model, tokenizer = surgery.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="eval")

bi = inspect.block_influence(model, tokenizer, prompts=[
    "The capital of France is",
    "Explain quantum entanglement simply",
    "The meaning of life is",
])
for layer, score in sorted(bi.items(), key=lambda x: x[1]):
    print(f"  Layer {layer:2d}: BI = {score:.4f}")

baseline_ppl = benchmark.perplexity(model, tokenizer, dataset="wikitext2", max_samples=50)
print(f"Baseline perplexity: {baseline_ppl:.2f}")

surgery.remove_layers(model, [19])
modified_ppl = benchmark.perplexity(model, tokenizer, dataset="wikitext2", max_samples=50)
print(f"Modified perplexity: {modified_ppl:.2f} (delta: +{modified_ppl - baseline_ppl:.2f})")
```

### Export and test in ollama

```python
from llm_surgeon import surgery, export

model, tokenizer = surgery.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="export")
surgery.remove_layers(model, [19])
export.full_pipeline(model, tokenizer=tokenizer, name="tinyllama-no19", quantization="Q4_K_M", output_dir="outputs/")
```

```bash
ollama run tinyllama "What is the capital of France?"
ollama run tinyllama-no19 "What is the capital of France?"
```

### Environment

- Hardware: RTX 2080 (8GB), 32GB RAM, i7-8700K
- Software: torch 2.6.0+cu126, transformers 5.5.0
- llm-surgeon commit: 094eb50

## Hypotheses

1. **Factual steering hypothesis:** Late layers (like 19) perform a small but critical "steering" operation that aligns the residual stream with factual knowledge stored in the embedding/output projection. The geometric change is small (low BI) but informational content is high.

2. **Distributed factual encoding:** Factual knowledge may not be localized to layer 19 specifically, but layer 19 may serve as a critical routing/gating point. Without it, the model can still access linguistic patterns but factual retrieval paths are disrupted.

3. **BI measures syntax, not semantics:** The cosine similarity metric captures representational geometry (direction of the hidden state vector), which may correlate more with syntactic/structural features than with semantic/factual features. Factual information might be encoded in the magnitude or in specific dimensions rather than the overall direction.

4. **Stop token disruption:** The over-generation behavior might indicate that layer 19 also participates in learning when to stop — the model's ability to recognize a complete response may depend on late-layer refinement.

## Follow-ups

- [ ] Run TruthfulQA / ARC-Challenge on the modified model to quantify factual degradation (Phase 3's eval_downstream)
- [ ] Remove other low-BI layers individually (17, 11, 18) and compare factual accuracy — is this specific to layer 19 or a general property of low-BI layers?
- [ ] Remove a high-BI early layer (e.g., layer 2, BI=0.334) — does this destroy linguistic coherence while preserving whatever factual accuracy remains?
- [ ] Compare attention entropy and residual stream norms before/after layer 19 removal — where exactly does the representation diverge?
- [ ] Investigate whether the factual degradation is prompt-dependent (try math, science, history prompts)
- [ ] Check if this finding replicates on LLaMA 3 8B once access is granted

## Related work

- ShortGPT (Men et al., 2024) — introduced Block Influence metric, focused on perplexity as evaluation. Did not specifically investigate factual accuracy of pruned models.
- The Unreasonable Ineffectiveness of the Deeper Layers (Gromov et al., 2024) — showed deep layer removal has minimal perplexity impact, consistent with our finding. Did not deeply investigate factual accuracy either.
- Known limitation of perplexity: fluent hallucinations score well because they're statistically plausible.
