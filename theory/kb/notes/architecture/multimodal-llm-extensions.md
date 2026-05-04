---
topic: architecture/multimodal-llm-extensions
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - llava2023
  - qwen2-vl-2024  # not yet in papers.json
  - qwen2-5-vl-2025
  - qwen3-vl-2025
  - internvl3-2025
  - meta-llama4
  - gemma3
---

# Multimodal LLM extensions

**Status:** stub. Drafted from Phase 1 landscape sweep; needs full Phase 2
treatment.

## What this topic covers

How vision, audio, and video are added to a text-pretrained LLM. The
2023 paradigm — `vision encoder → MLP projector → frozen LLM` (LLaVA,
2304.08485) — has given way in 2025–2026 to **early-fusion / native
multimodal pretraining** (LLaMA 4, InternVL3): the LLM is co-trained
on images and text from scratch rather than adapter-attached after the
fact. The Qwen-VL series introduces **M-RoPE** (2D rotary on
height/width axes for vision tokens), generalizing to **Interleaved-
MRoPE** in Qwen3-VL. Gemma 3 uses SigLIP2 as vision encoder. Audio
maturing but trails vision; Qwen3-Omni's AuT audio encoder is
trained-from-scratch on 20M hours
`[Phase 1 sweep §2 multimodal-llm-extensions]`.

## Primary sources to read (in order)

1. `llava2023` — *Visual Instruction Tuning* (2304.08485) — adapter-
   based: ViT + MLP projector + frozen Vicuna; instruction-tuned on
   GPT-4-generated visual-conversation data.
2. `qwen2-vl-2024` — *Qwen2-VL* (2409.12191) — Naive Dynamic Resolution
   + M-RoPE (2D + temporal).
3. `qwen2-5-vl-2025` — *Qwen2.5-VL* (2502.13923) — window attention in
   ViT; dynamic FPS for video.
4. `qwen3-vl-2025` — *Qwen3-VL* (2511.21631) — Interleaved-MRoPE for
   time/width/height.
5. *Qwen3-Omni* (2509.17765) — AuT audio encoder.
6. `internvl3-2025` — *InternVL3* (2504.10479) — native multimodal
   pretraining (joint image+text from scratch).
7. *InternVL3.5* (2508.18265) — versatility/reasoning/efficiency
   improvements.
8. *LLaVA-Video* (2410.02713) — SlowFast video pooling.
9. `gemma3` — SigLIP2 vision encoder integration.
10. `meta-llama4` — early-fusion multimodal in the herd.

## Key claims to ground (Phase 2 todo)

- The LLaVA paradigm: ViT (CLIP / SigLIP) extracts image patch
  features; MLP projector maps to LLM residual-stream dimension;
  frozen LLM consumes image tokens interleaved with text tokens.
- Adapter vs early fusion: the architectural distinction. Early fusion
  treats image patches as first-class tokens at pretraining time;
  adapter freezes the LLM and trains only the projector + (optionally)
  LoRA. Reported quality gap: early-fusion wins at sufficient
  multimodal pretraining data scale.
- M-RoPE generalization: split the per-head dimension $d_h$ into
  three thirds, apply RoPE rotations on temporal/height/width axes
  separately. The rotation matrix becomes block-diagonal across the
  three subspaces. See `kb/notes/architecture/position-encoding.md`
  for the 1D RoPE substrate.
- Interleaved-MRoPE (Qwen3-VL): instead of partitioning the channel
  range into temporal/width/height thirds, **interleave** the axes
  per dimension-pair. The intuition: every dimension-pair sees all
  three axes, improving the model's ability to combine spatial and
  temporal information.
- Naive Dynamic Resolution (Qwen2-VL): the vision encoder accepts
  variable-size images; image token count is proportional to the
  image's pixel count (rather than fixed at 256 or 1024 tokens).
- Window attention in vision encoder (Qwen2.5-VL): replace dense ViT
  attention with sliding-window for efficiency at high resolution.
- SigLIP2 (Google): the modern vision encoder of choice for open
  models; outperforms CLIP at matched scale.
- AuT audio encoder (Qwen3-Omni): trained from scratch on 20M hours
  of audio. Architectural choice: continuous audio → discrete-token
  representation + temporal positional encoding.
- LLaVA-Video SlowFast pooling: temporal sampling at two rates to
  cover both fine motion and long-horizon video understanding.
- Stale-prior correction (per Phase 1 sweep §3): "vision-language is
  vision-encoder + adapter + LLM" → 2025–2026: native co-pretraining;
  M-RoPE / Interleaved-MRoPE; Naive Dynamic Resolution.

## Open questions

- How much multimodal pretraining data is needed for early-fusion to
  beat adapter? No published scaling curve.
- M-RoPE vs Interleaved-MRoPE: head-to-head?
- Audio modality scaling: does AuT-style from-scratch audio encoder
  generalize to other lab's training stacks?

## Related notes

- `kb/notes/architecture/position-encoding.md` — the RoPE / M-RoPE /
  Interleaved-MRoPE substrate.
- `kb/notes/architecture/transformer-overview.md` — how multimodal
  tokens enter the residual stream.
- `kb/notes/architecture/tokenization.md` — image-patch tokenization
  is its own thing; audio-codec tokenization too.
- `kb/notes/training/synthetic-data-and-distillation.md` — multimodal
  instruction tuning relies heavily on synthetic data.
