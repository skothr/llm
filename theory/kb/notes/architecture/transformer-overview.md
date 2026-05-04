---
topic: architecture/transformer-overview
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - vaswani2017
  - radford2018
  - touvron2023
  - meta-llama4
  - qwen3
  - deepseek-v3
  - gemini2-5
  - olmo2
---

# Transformer overview (and the modern decoder-only template)

**Status:** stub. Drafted from Phase 1 landscape sweep; needs full Phase 2
treatment. Should be one of the first stubs to be promoted to draft —
this is the "where everything plugs in" reference for the architecture
area.

## What this topic covers

The original Vaswani 2017 enc-dec Transformer; the GPT-1 (Radford 2018)
decoder-only adaptation; the LLaMA (Touvron 2023) "modern decoder-only"
refinements (RMSNorm + Pre-LN + RoPE + SwiGLU + GQA), and the
2024–2026 frontier convergence on **sparse MoE + GQA/MLA + RMSNorm +
RoPE + reasoning-capable** as the universal template
`[Phase 1 sweep §2 transformer-overview]`.

The note should function as a **block-diagram orientation** for the
rest of the architecture/ area, with cross-references rather than
deep treatment of any single sublayer (those are in the per-topic
notes).

## Primary sources to read (in order)

1. `vaswani2017` — *Attention Is All You Need* (1706.03762) — original
   enc-dec architecture, baseline for all subsequent work.
2. `radford2018` — *Improving Language Understanding by Generative
   Pre-Training* (GPT-1) — decoder-only adaptation; LM pretraining.
3. `touvron2023` — *LLaMA: Open and Efficient Foundation Language
   Models* (2302.13971) — the LLaMA-1 reference architecture: RMSNorm
   + SwiGLU + RoPE + Pre-LN. Reference template for 2023+ open models.
4. `meta-llama4` — *The Llama 4 Herd* (2601.11659) — frontier 2026
   architecture: MoE + iRoPE + early-fusion multimodal.
5. `qwen3` — *Qwen3 Technical Report* (2505.09388) — frontier
   contrast point: shared-expert-free MoE + thinking/non-thinking
   toggle.
6. `deepseek-v3` — *DeepSeek-V3 Technical Report* (2412.19437) —
   frontier contrast point: MLA + DeepSeekMoE + MTP + auxiliary-loss-
   free LB.
7. `gemini2-5` — *Gemini 2.5 Technical Report* (2507.06261) —
   frontier-but-opaque: confirms sparse MoE + 1M context, hides
   architectural specifics.
8. `olmo2` — *OLMo 2 Furious* (2501.00656) — fully-open frontier:
   RMSNorm + QK-Norm + RoPE; useful for reproducible architectural
   ablation citations.

## Key claims to ground (Phase 2 todo)

- The "modern decoder-only template" of 2023–2024: residual stream of
  width $d_{\text{model}}$; $L$ blocks; each block is `RMSNorm →
  Attention(GQA, RoPE) → +residual; RMSNorm → SwiGLU FFN → +residual`
  (Pre-LN); final RMSNorm → tied or untied LM head.
- The 2024–2026 transition: dense FFN sublayers → MoE FFN sublayers in
  most or all positions; GQA → MLA in the DeepSeek family; absolute
  position → RoPE → iRoPE in LLaMA 4; pre-LN → peri-LN in Gemma 3 /
  OLMo 2; one-token output → multi-token-prediction in DeepSeek-V3.
- Block-diagram comparisons across LLaMA 4, Qwen3, DeepSeek-V3,
  Gemma 3 — same residual stream + attention + MoE-FFN skeleton with
  divergent specifics in each sublayer.
- The encoder side: discuss why decoder-only won for LLMs (token-by-
  token autoregressive generation; no cross-attention overhead) and
  where enc-dec persists (translation, code-by-task, T5 family).
- Tensor-shape tables across frontier models: $d_{\text{model}}$, $L$,
  $h$, $d_h$, $E$, $k$.
- Stale-prior correction (per Phase 1 sweep §3): "MoE is a niche
  scaling trick" → MoE is the frontier default as of 2025.

## Tensor-shape reference (to be filled in)

A table of the form:

| Model | $d$ | $L$ | $h$ | $d_h$ | KV variant | Position | FFN | MoE? |
|---|---|---|---|---|---|---|---|---|
| LLaMA-1 7B | 4096 | 32 | 32 | 128 | MHA | RoPE | SwiGLU | no |
| LLaMA-3 8B | 4096 | 32 | 32 | 128 | GQA-8 | RoPE-500K | SwiGLU | no |
| ... | | | | | | | | |

## Related notes

- `kb/notes/architecture/attention-mechanism.md` — attention sublayer
  internals (already drafted, the pilot).
- `kb/notes/architecture/position-encoding.md` — RoPE / iRoPE / YaRN
  (drafted Phase 2).
- `kb/notes/architecture/ffn.md` — SwiGLU FFN sublayer (stub).
- `kb/notes/architecture/moe.md` — MoE FFN replacement (drafted
  Phase 2).
- `kb/notes/architecture/normalization.md` — RMSNorm, Pre/Peri/Post-LN
  (stub).
- `kb/notes/architecture/embeddings-and-tying.md` — input/output
  embeddings (stub).
- `kb/notes/architecture/tokenization.md` — BPE/SentencePiece/BLT (stub).
- `kb/notes/architecture/state-space-models.md` — non-attention
  alternatives (drafted Phase 2).
- `kb/notes/architecture/long-context.md` — frontier context-length
  techniques (drafted Phase 2).
- `kb/notes/architecture/multi-token-prediction.md` — DeepSeek-V3 MTP
  (stub).
- `kb/notes/architecture/reasoning-architectures.md` — thinking
  models (drafted Phase 2).
- `kb/notes/architecture/multimodal-llm-extensions.md` — vision/audio
  extensions (stub).
