---
topic: architecture/ffn
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - vaswani2017
  - shazeer2020
  - touvron2023
---

# Feed-forward network (FFN) sublayer

**Status:** stub. Drafted from Phase 1 landscape sweep; needs full Phase 2
treatment. The shazeer2020 GLU-variants excerpt is already written
during this Phase 2 pass (`kb/excerpts/shazeer2020.md`); promoting this
to draft is a low-effort follow-up.

## What this topic covers

The FFN sublayer is half of a Transformer block: it processes each
token's residual-stream vector independently (no inter-token mixing).
The original Vaswani 2017 form was a 2-matrix MLP with ReLU activation
and $d_{\text{ff}} = 4 d_{\text{model}}$. The modern form is **SwiGLU**
(Shazeer 2020): a 3-matrix gated structure with Swish activation.
LLaMA-1 adopted SwiGLU; every major decoder-only LLM since (LLaMA 2/3,
Mistral, Qwen, DeepSeek, Gemma, OLMo, Phi-4) has followed. The MoE
extension replaces the single FFN with $E$ parallel ones; treated in
`kb/notes/architecture/moe.md`.

## Primary sources to read (in order)

1. `vaswani2017` — *Attention Is All You Need* §3.3 — original FFN
   definition; already excerpted.
2. `shazeer2020` — *GLU Variants Improve Transformer* (2002.05202) —
   SwiGLU and the GEGLU/ReGLU/Bilinear/etc. variant family. **Already
   excerpted in Phase 2** (see `kb/excerpts/shazeer2020.md`).
3. `touvron2023` — *LLaMA* — first major LLM to adopt SwiGLU; reports
   the $d_{\text{ff}} = \tfrac{8}{3} d_{\text{model}}$ rule (rounded
   to multiples of 256 for hardware alignment).

## Key claims to ground (Phase 2 todo)

- The Vaswani FFN: $\mathrm{FFN}(\boldsymbol{x}) = \max(0, \boldsymbol{x} W_1 + b_1) W_2 + b_2$,
  with $W_1: d \times 4d, W_2: 4d \times d$
  `[shazeer2020 §1 Eq.1; kb/excerpts/shazeer2020#sec-1]`.
- The T5 / modern bias-free form: $\mathrm{FFN}_{\mathrm{ReLU}}(\boldsymbol{x}) = \max(0, \boldsymbol{x} W_1) W_2$
  `[shazeer2020 §1 Eq.2]`.
- The SwiGLU form: $\mathrm{FFN}_{\mathrm{SwiGLU}}(\boldsymbol{x}) = (\mathrm{Swish}_1(\boldsymbol{x} W) \otimes \boldsymbol{x} V) W_2$
  `[shazeer2020 §2 Eq.6; kb/excerpts/shazeer2020#sec-2-eq6]`.
- The $\tfrac{2}{3}$ parameter-equalization rule: SwiGLU's third matrix
  forces $d_{\text{ff}}$ shrink to $\tfrac{2}{3} \cdot 4 d_{\text{model}} = \tfrac{8}{3} d_{\text{model}}$
  to keep parameter count constant
  `[shazeer2020 §2; kb/excerpts/shazeer2020#sec-2]`.
- Quality result: GEGLU and SwiGLU both produce the lowest perplexity
  in shazeer2020 Table 1 (1.633 / 1.636 vs ReLU baseline 1.677). Why
  SwiGLU was adopted over GEGLU is unclear in the literature
  `[shazeer2020 §3.2; kb/excerpts/shazeer2020#sec-3-2]`.
- SwiGLU's "divine benevolence" theoretical status: shazeer2020 offers
  no derivation. Theoretical work attempting to explain SwiGLU's
  advantage exists (gating as soft selection; bilinear interaction)
  but is post-hoc.
- $d_{\text{ff}}$ scaling across frontier models: LLaMA-1 7B uses
  11008 ($\approx \tfrac{8}{3} \cdot 4096 = 10923$, rounded to a
  multiple of 256). DeepSeek-V3 routed-expert FFNs are narrower
  (fine-grained MoE; see moe.md).
- The activation-function lineage: ReLU → GELU → Swish → SwiGLU.
  GELU dominates BERT-era and pre-LLaMA decoder-only (GPT-2/3, OPT);
  SwiGLU dominates 2023+.

## Related notes

- `kb/notes/architecture/moe.md` — MoE replaces this single FFN with
  $E$ parallel SwiGLU FFNs (drafted Phase 2).
- `kb/notes/architecture/transformer-overview.md` — where FFN sits in
  the block (post-attention, post-RMSNorm).
- `kb/notes/architecture/normalization.md` — RMSNorm precedes the FFN
  in Pre-LN and modern designs.
