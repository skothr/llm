# Attention Ablation Causes Possessive/Contraction Collapse

## Date
2026-04-09

## Context
- Model: OpenLLaMA 3B v2 (26 layers, 32 heads per layer)
- Experiment: zeroing attention sub-layers (o_proj weights) ranked by ascending BI score
- 3 lowest-BI layers zeroed (21, 22, 23): no visible degradation, +12.6% perplexity
- 6 lowest-BI layers zeroed (21, 22, 23, 24, 20, 19): severe coherence loss, +88.7% perplexity

## Finding
Zeroing 6 attention sub-layers produces a systematic morphological error: the model collapses multiple contraction types into the possessive `'s` form:
- "you's" (should be "you're" or "it's")
- "we's" (should be "we'll" or "here's")  
- "they's" (should be "they")

This is not random corruption — it's a consistent pattern where the most common contraction form (`'s`) replaces all others. Additionally, once nonstandard forms appear in the generated context, the model shifts toward informal/dialectal register, producing "ya'" — likely an artifact of the training data distribution for that register.

## Evidence
Prompt: "The three laws of thermodynamics are"
- Baseline: coherent multi-paragraph explanation
- 3-zeroed: coherent, actually more specific than baseline
- 6-zeroed: "In this article, we's a good idea to understand what they all mean? In other words: What'd you do with that 10-dollar bill I gave ya'."

Prompt: "The capital of France is"
- 6-zeroed: "you's also home to one of Europe's largest Chinatown"

## Interpretation
The additional layers zeroed going from 3→6 (specifically layers 19, 20, 24) appear to handle **morphological agreement** — tracking subject-verb person/number across token positions to select the correct contraction form. This is inherently an inter-token operation (attention's role), since MLP operates per-position and cannot compare subject to verb.

The collapse to `'s` specifically (rather than random tokens) suggests the model defaults to the highest-frequency contraction form when it loses the ability to discriminate. The attention heads in these layers were maintaining the distinction between "it is"→"it's", "we will"→"we'll", "you are"→"you're" by attending back to the subject.

## Reproducibility
```python
from llm_surgeon.surgery import load_model, zero_attention
model, tokenizer = load_model("openlm-research/open_llama_3b_v2", mode="inspect")
for layer in [21, 23, 22, 24, 20, 19]:
    zero_attention(model, layer)
# Generate with ollama after export, or use model directly
```

## Comparison with TinyLlama
TinyLlama (22 layers) showed a different degradation pattern with attention ablation:
- Lost **format compliance** (haiku→couplet)
- Lost **decisiveness** (hedging instead of answering)
- Did NOT show possessive collapse

This may be because OpenLLaMA 3B has more specialized attention heads (more capacity to allocate dedicated heads to morphological tasks), so ablation removes a specialized capability rather than degrading a distributed one.

## Follow-ups
- Identify which specific heads in layers 19, 20, 24 handle contraction discrimination (inspect_head on contraction-heavy prompts)
- Test whether scaling these attention layers by 0.5 (instead of zeroing) preserves morphological agreement
- Check if the collapse pattern varies by contraction type (test prompts with "we'll", "they're", "I've" specifically)
- Compare with GQA models (shared K/V heads) — does the collapse happen earlier or later?

## References
- Related to prior observation: 2026-04-09-layer19-factual-vs-linguistic-coherence.md (TinyLlama layer removal)
- ShortGPT (Men et al., 2024) — Block Influence scores for layer pruning
