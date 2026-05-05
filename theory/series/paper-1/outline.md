# Paper 1 — *The modern Transformer is a small set of choices*

**Working title:** *The modern Transformer is a small set of choices: convergent architecture in frontier LLMs, 2017–2026.*

**Thesis:** Frontier LLMs in 2026 (DeepSeek V3/R1, Llama 4, Claude 3.7-class, GPT-class, Mistral, Gemini-class) are converging on a small set of architectural choices along a handful of orthogonal axes. We name those axes, formalize the canonical 2017 Transformer along each axis, walk the choices the field made between 2017 and 2026, and identify the localized open frontiers where the convergence is still contested.

**Length target:** ~80 pages.

**Audience:** ML researchers and engineers with graduate-level math who want one place to read the synthesized architectural state of frontier LLMs. Not a tutorial; not a survey. A monograph.

## Section structure (target ~14 sections)

For each section: **(thesis sentence)** • [page budget] • {primary KB anchors} • {key contradictions to surface}.

### §1 — Introduction (5 pages)

The Transformer is not one architecture; it is a family parameterized by a small set of orthogonal choices. We list the axes (KV-sharing, attention-implementation, FFN, normalization, positional, tokenization-coupling, multi-token-output) and preview the convergence we will document.

- Anchors: `kb/notes/architecture/transformer-overview.md`, `kb/index/timeline.md` 2017-prequel, brainstorm doc Paper-1 sketch.
- Open question: *is the convergence real or model-zoo bias?* Answered in §13.

### §2 — Pre-2017 prequel and the 2017 canonical (6 pages)

What the Transformer replaced (RNN/LSTM seq2seq, encoder-decoder attention, ConvSeq2Seq), then the canonical Vaswani encoder-decoder formal definition with all variables defined: tensor shapes, scaled dot-product, multi-head, post-LN, FFN, sinusoidal positions, weight tying.

- Anchors: `kb/notes/architecture/transformer-overview.md` §1-3, `kb/excerpts/vaswani2017.md` (#sec-3-1, #sec-3-2-1, #sec-3-3, #sec-3-4, #sec-3-5).
- Math to transcribe: Eq. 1 (scaled dot-product), Eq. 2 (FFN), the sinusoidal formula, the residual+norm equation.
- Tensor-shape table: B × S × D throughout, plus head-split B × H × S × d_h.

### §3 — Tokenization → embedding → unembedding (5 pages)

Subword family (BPE → SentencePiece byte-level → tokenizer-free / BLT). Tied vs untied I/O embeddings as a parameter-saving choice, and why frontier-2026 untied. Vocabulary-size tradeoffs (50k → 128k → 256k for multilingual).

- Anchors: `kb/notes/architecture/{tokenization,embeddings-and-tying}.md`. `kb/excerpts/{sennrich2016,kudo2018-sentencepiece,press2017-tied-embed,radford2019-gpt2,blt2024}.md`.
- Open question: tokenizer-free architectures (BLT) — niche or future?

### §4 — Positional encoding (6 pages)

Sinusoidal → learned → T5 relative bias → ALiBi → RoPE → YaRN/LongRoPE → NoPE. Why RoPE became dominant. Why long-context extension is a positional question, not (only) an attention question.

- Anchors: `kb/notes/architecture/{position-encoding,long-context}.md`. `kb/excerpts/{vaswani2017,su2021,press-alibi}.md`.
- Math: rotation-matrix formulation, the RoPE inner product invariance.
- Contradiction `[CONTRADICTION]`: long-context extrapolation — RoPE-only with NTK-aware vs YaRN vs full retraining.

### §5 — Attention KV-sharing axis: MHA → MQA → GQA → MLA (8 pages, anchor section)

The full lineage along the "how do heads share K and V?" axis. MHA Δ → MQA (1 K/V head) → GQA (G groups) → MLA (low-rank joint compression with decoupled-RoPE). Cache-cost vs quality tradeoff curve. The pilot KB note already covers this — promote and expand.

- Anchors: `kb/notes/architecture/attention-mechanism.md` §2-3 (the pilot itself). `kb/excerpts/{vaswani2017,shazeer2019,ainslie2023,deepseek-v2}.md`.
- Math: the KV-cache size formula B × S × H_kv × d_h × 2bytes; how MQA/GQA/MLA each modify H_kv.
- Contradiction `[CONTRADICTION]`: MLA-vs-MHA quality at frontier scale (still single-source: DeepSeek-V2).

### §6 — Hardware-implementation axis: FA1 → FA2 → FA3 → NSA (8 pages)

Same scaled dot-product, four hardware-implementation strategies. FlashAttention 1 (block tiling + recomputation), FA2 (warp-level tile scheduling), FA3 (asynchronous + low-precision tensor-core scheduling on H100/H200), NSA (native sparse attention; ~1k-token sparse blocks). Why these are *optimizations*, not different attention formulas, and how that distinction matters when reasoning about quality.

- Anchors: `kb/notes/architecture/attention-mechanism.md` §4. `kb/excerpts/{dao2022,dao2023,shah2024,yuan2025}.md`.
- Tensor-shape walkthrough: how blocks of B × H × S × d_h are tiled, FLOPs vs HBM-bandwidth analysis.
- Contradiction `[CONTRADICTION]`: NSA vs FA3 head-to-head — single-source claim, no third-party reproduction yet.

### §7 — Feed-forward axis: dense → MoE → fine-grained-MoE → MoE-with-shared-experts (8 pages)

ReLU → GELU → SwiGLU dense FFN. Then Switch (1 expert per token), Mixtral (8 experts, 2 active), DeepSeekMoE (256 fine-grained + 1 shared per layer), and the routing-collapse problem. Why fine-grained + shared is the current convergence.

- Anchors: `kb/notes/architecture/{ffn,moe}.md`. `kb/excerpts/{shazeer2020,deepseekmoe2024,deepseek-v2}.md`.
- Math: MoE gating formula, load-balancing loss, the shared-expert-as-residual interpretation.
- Contradiction `[CONTRADICTION]`: MoE quality at frontier — gap between MoE-trained and dense-equivalent at matched active params (claimed closed in V3, not all replicated).

### §8 — Normalization axis: LayerNorm → RMSNorm → Pre-LN → DeepNorm (5 pages)

LN (Ba 2016 §3, Eq. 3) → RMSNorm (Zhang 2019, drops mean centering). Post-LN (original) → Pre-LN (GPT-2 §2.3, Xiong 2020 stability proof) → DeepNorm. Why frontier-2026 is RMSNorm + Pre-LN with `1/√N` residual init.

- Anchors: `kb/notes/architecture/normalization.md`. `kb/excerpts/{ba2016-layernorm,zhang2019,xiong2020,radford2019-gpt2}.md` — all PDF-grounded.
- Math: the LN/RMSNorm definitions side-by-side; Xiong's stability proof sketch (Theorem 1/2).

### §9 — State-space and hybrids: the alternative path (7 pages)

S4/S5 → Mamba (selective SSM) → Mamba-2 (state-space-duality with attention). Hybrids: Jamba (interleave), Samba (interleave-fine), Zamba2 (shared-SSM cross-attention bridge), Hymba (parallel SSM+attention). Why pure-SSM hasn't replaced attention at frontier and why hybrids are gaining.

- Anchors: `kb/notes/architecture/state-space-models.md`. `kb/excerpts/{gu2023-mamba,mamba2}.md`.
- Math: SSM recurrence form, selective gating, Mamba-2's SSD equivalence.
- Contradiction `[CONTRADICTION]`: SSM-hybrid quality at frontier — open.

### §10 — Multi-token output: speculation, MTP, parallel decoding (5 pages)

Speculative decoding (Leviathan 2023, draft+verify) is a serving optimization. Multi-Token Prediction (Gloeckle/DeepSeek-V3) is an architectural choice that improves both training quality and inference parallelism. Medusa, EAGLE, the spectrum.

- Anchors: `kb/notes/architecture/multi-token-prediction.md`, `kb/notes/inference/speculative-decoding.md`. `kb/excerpts/{gloeckle2024-mtp,leviathan2023,cai2024-medusa,li2024-eagle}.md`.
- Distinction: MTP is during *training*; speculative decoding is during *inference*.

### §11 — Multimodal extensions (5 pages)

Cross-attention bridge (Flamingo) → linear projection (LLaVA) → native multimodal (Gemini, GPT-4o, Claude 3.5 Sonnet). The "vision encoder + projector + LLM body" three-piece pattern. Where vision-language *is* an architecture question vs where it isn't.

- Anchors: `kb/notes/architecture/multimodal-llm-extensions.md`. `kb/excerpts/{alayrac2022-flamingo,llava2023}.md`.

### §12 — Inference adjuncts (folded in) (4 pages)

KV-cache lifecycle (paged attention, block-fragmentation), quantization regimes (INT8/INT4/FP8/NVFP4/BitNet), structured output as constrained decoding (logit-bias, Outlines FSM, XGrammar). These are architectural concerns, not bolt-ons, because they constrain which choices §5-§9 can take.

- Anchors: `kb/notes/inference/{kv-cache-management,quantization,structured-output}.md`. `kb/excerpts/{kwon2023,frantar2022,xiao2023-smoothquant,ma2024-bitnet,willard2023-outlines,xgrammar2024}.md`.

### §13 — Frontier-2026 snapshot: what choices the leading models share (5 pages)

DeepSeek V3, Llama 4, Claude 3.7-class, GPT-class, Mistral Large, Gemini-class, Qwen-class — table of architectural choices for each, color-coded by axis. Where they agree (RMSNorm+Pre-LN, RoPE-with-extension, GQA-or-MLA, SwiGLU, fine-grained-MoE-or-dense), where they diverge (KV-sharing strategy, MoE vs dense, multimodal architecture).

- Anchors: `kb/index/timeline.md` 2024-2026 entries, `kb/notes/architecture/transformer-overview.md` §frontier.
- Output: a 2-page snapshot table that becomes the most-screenshotted figure of the paper.

### §14 — Contradictions live and open frontiers (4 pages)

Concentrated discussion of `[CONTRADICTION]` markers: MLA-vs-MHA at scale (single-source), NSA-vs-FA3 head-to-head, SSM-hybrid quality, MoE quality gap, long-context extrapolation method comparison, MTP at non-DeepSeek scale. What evidence would close each.

- Anchors: `kb/index/contradictions.md` § architecture (13 contradictions).
- Closing thesis: the field's open architectural questions are localized and answerable; we predict 2027-class models converge further along axes §5-§9 and remain mixed on §6, §10.

## What this outline commits to

- **Math is verbatim from PDFs.** Every equation cited with a `kb/excerpts/<key>#<anchor>` dual-citation.
- **Tensor shapes everywhere relevant.** B (batch), S (seqlen), D (model), H (heads), d_h (head-dim), V (vocab), L (layers).
- **Contradictions are first-class.** Each `[CONTRADICTION]` gets explicit treatment in §14, not buried in body.
- **No analogy laundering.** `[INTUITION]` and `[ANALOGY]` markers preserved from the KB notes; analogy returns to canonical math each time.

## Sections by writing-pass parallelism

- Pass A (independent, easy parallelize): §3, §4, §8, §11, §12.
- Pass B (depends on §2 setup): §5, §6, §7, §9.
- Pass C (synthesis, last): §1 intro, §13 snapshot, §14 contradictions.
- Pass D: §2 prequel-and-canonical (depends on §3-§9 having stabilized notation).

7 parallel section-subagents on Pass A+B; 4 on Pass C+D after Pass A+B lands.
