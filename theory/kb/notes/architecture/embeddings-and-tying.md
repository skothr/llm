---
topic: architecture/embeddings-and-tying
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - press2017-tying
  - meta-llama3
  - olmo2
  - deepseek-v3
---

# Token embeddings and weight tying

**Status:** stub. Drafted from Phase 1 landscape sweep; needs full Phase 2
treatment.

## What this topic covers

The input embedding matrix $E_{\text{in}} \in \mathbb{R}^{|V| \times d_{\text{model}}}$
maps integer token IDs to residual-stream vectors; the output (LM head)
matrix $E_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times |V|}$
maps the final residual stream to logits over the vocabulary. **Weight
tying** sets $E_{\text{out}} = E_{\text{in}}^\top$, eliminating the
larger of the two matrices. Press & Wolf 2017 originally argued for it;
modern frontier LLMs have **shifted toward untied** embeddings at
multi-billion scale (LLaMA-3, OLMo 2, DeepSeek-V3, Qwen3 large), with
tying retained for compact / edge models (Qwen3 small, Phi-4-Mini).
Recent (2026) work argues standard tying biases embeddings toward the
output space; alternatives like Pseudo-Inverse Tying exist
`[Phase 1 sweep §2 embeddings-and-tying]`.

## Primary sources to read (in order)

1. `press2017-tying` — *Using the Output Embedding to Improve Language
   Models* (1608.05859) — original weight-tying proposal.
2. *Token Embeddings Violate the Manifold Hypothesis* (2504.01002) —
   structural analysis of embedding geometry.
3. *Weight Tying Biases Token Embeddings Towards the Output Space*
   (2603.26663) — recent diagnosis of tying-induced bias.
4. *Rethinking Weight Tying: Pseudo-Inverse Tying for Stable LM
   Training* (2602.04556) — PIT alternative.
5. `meta-llama3`, `olmo2`, `deepseek-v3`, `qwen3` — frontier-model
   reports (untied at scale).

## Key claims to ground (Phase 2 todo)

- Tied vs untied parameter counts: tying saves $|V| \cdot d_{\text{model}}$
  parameters. For LLaMA-3 8B with $|V| = 128256$, $d = 4096$, this is
  ~525M parameters — non-trivial at 8B scale (~6%).
- Embedding initialization: scaled random init vs from-pretrained-
  embedding init for adapted/extended models.
- Empirical reasons frontier models untie: (a) the logit-projection
  job differs from the token-lookup job; (b) tied matrices contain
  bias toward output-space directions, hurting input-side semantics;
  (c) at multi-billion scale the parameter savings are negligible.
- Embedding manifold structure: are token embeddings approximately on
  a low-dim manifold (the manifold-hypothesis result says no in
  recent work).
- Special-token embedding handling: BOS/EOS, system-prompt-region,
  reasoning-region (`<think>` / `</think>`) tokens — each has distinct
  trained-distribution properties.
- LM-as-embedder: using the trained LLM's hidden states as feature
  vectors for retrieval/classification (cf. `LLMs are Also Effective
  Embedding Models`, 2412.12591).
- Token-distillation / vocab-extension: how to add new tokens (and
  initialize their embeddings) to a pretrained model without quality
  loss.

## Related notes

- `kb/notes/architecture/tokenization.md` — what the integer token
  IDs are.
- `kb/notes/architecture/transformer-overview.md` — where embeddings
  sit at input and output of the block stack.
- `kb/notes/architecture/normalization.md` — final-layer-norm placement
  before LM head.
- `kb/notes/interpretability/probing.md` — the "embeddings as
  features" perspective overlaps with linear probing.
