# Landscape sweep — Architecture

**Scope:** transformer-overview, tokenization, embeddings-and-tying, position-encoding, attention-mechanism, ffn-and-moe, normalization, state-space-models, long-context, multimodal-llm-extensions
**Captured:** 2026-05-02
**Researcher:** Phase 1 subagent (Sonnet 4.6)
**Time spent:** ~45 minutes
**Sources hit:** WebSearch ~25 queries; no WebFetch (permission denied); blogs / HF model pages / arXiv abstracts used as supplementary confirmation

---

## 1. Candidate papers (Phase 2 priority list)

### transformer-overview

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| Attention Is All You Need | 2017 | Vaswani et al. | https://arxiv.org/abs/1706.03762 | Foundational enc-dec transformer; baseline for everything | A |
| The Llama 4 Herd: Architecture, Training, Evaluation, and Deployment Notes | 2026 | Meta AI | https://arxiv.org/abs/2601.11659 | Current frontier open model: MoE + iRoPE + early-fusion multimodal; Jan 2026 | A |
| Qwen3 Technical Report | 2025 | Qwen Team | https://arxiv.org/abs/2505.09388 | 235B-A22B MoE + 6 dense models; thinking/non-thinking toggle; May 2025 | A |
| DeepSeek-V3 Technical Report | 2024 | DeepSeek-AI | https://arxiv.org/abs/2412.19437 | 671B-A37B; MLA + DeepSeekMoE; NSA follow-on; Dec 2024 | A |
| DeepSeek-V3.2 | 2025 | DeepSeek-AI | https://arxiv.org/abs/2512.02556 | Adds DeepSeek Sparse Attention (DSA) on top of MLA; Dec 2025 | A |
| Gemma 3 Technical Report | 2025 | Gemma Team, Google DeepMind | https://arxiv.org/abs/2503.19786 | 1–27B; interleaved local/global attention (1:5 ratio); SigLIP2 vision; Mar 2025 | A |
| Gemini 2.5 Technical Report | 2025 | Google DeepMind | https://arxiv.org/abs/2507.06261 | Sparse MoE; 1M+ context; native multimodal; TPUv5p; Jul 2025 | A |
| OLMo 2 | 2025 | AI2 | https://arxiv.org/abs/2501.00656 | Fully open (weights + data + code); RMSNorm + QK-Norm; RoPE; Jan 2025 | A |
| Phi-4 Technical Report | 2024 | Microsoft | https://arxiv.org/abs/2412.08905 | 14B; data-quality-first; tiktoken vocab 100k; Dec 2024 | A |
| Mapping the LLM Landscape (cross-family survey) | 2026 | various | https://www.mdpi.com/2673-2688/7/4/142 | Systematic architecture/alignment/benchmark comparison across GPT/LLaMA/DeepSeek/Qwen/Gemini families | B |

---

### tokenization

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| Byte Latent Transformer: Patches Scale Better Than Tokens | 2024 | Meta AI | https://arxiv.org/abs/2412.09871 | Tokenizer-free; dynamic byte patches; matches LLaMA 3 at 50% fewer inference FLOPs; ACL 2025 | A |
| SuperBPE: Space Travel for Language Models | 2025 | various | https://arxiv.org/abs/2503.13423 | Superword BPE bridging whitespace; 33% fewer tokens at 200k vocab; +4% avg benchmark; COLM 2025 | A |
| You Can Learn Tokenization End-to-End with Reinforcement Learning | 2026 | various | https://arxiv.org/html/2602.13940 | RL-based joint tokenization+LM training; post-cutoff | A |
| An Information-Theoretic Perspective on LLM Tokenizers | 2026 | various | https://arxiv.org/html/2601.09039v1 | Theoretical analysis of tokenizer efficiency; Jan 2026 | A |
| Length-MAX Tokenizer for Language Models | 2025 | Dong Dong, Weijie Su | https://arxiv.org/abs/2511.20849 | 14–18% fewer tokens/char vs BPE; 18.5% fewer training steps; Nov 2025 | A |
| Decoupling the Benefits of Subword Tokenization via Byte-level Simulation | 2026 | various | https://arxiv.org/html/2604.27263 | Disentangles which BPE properties actually help; Apr 2026 | A |
| GPUTOK: GPU Accelerated Byte Level BPE | 2026 | various | https://arxiv.org/html/2603.02597v1 | Practical acceleration of byte BPE; Mar 2026 | A |
| Bit-level BPE: Below the byte boundary | 2025 | various | https://arxiv.org/html/2506.07541v1 | Sub-byte tokenization; extreme compression direction | A |
| Say Anything but This: When Tokenizer Betrays Reasoning in LLMs | 2026 | various | https://arxiv.org/html/2601.14658v1 | Tokenizer artifacts → reasoning failure modes | A |
| Teaching Old Tokenizers New Words | 2025 | various | https://arxiv.org/html/2512.03989 | Tokenizer adaptation for pretrained models | A |

---

### embeddings-and-tying

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| Using the Output Embedding to Improve Language Models | 2017 | Press & Wolf | https://arxiv.org/abs/1608.05859 | Original weight-tying proposal; still referenced | A |
| Weight Tying Biases Token Embeddings Towards the Output Space | 2026 | various | https://arxiv.org/abs/2603.26663 | Shows tied matrices skew toward output; gradient imbalance cause; Mar 2026 | A |
| Rethinking Weight Tying: Pseudo-Inverse Tying for Stable LM Training | 2026 | various | https://arxiv.org/abs/2602.04556 | PIT alternative to standard tying; better training stability; Feb 2026 | A |
| Token Embeddings Violate the Manifold Hypothesis | 2025 | various | https://arxiv.org/html/2504.01002v1 | Structural analysis of embedding geometry; Apr 2025 | A |
| LLMs are Also Effective Embedding Models | 2024 | various | https://arxiv.org/html/2412.12591v2 | Overview of LLM-as-embedder approaches; Dec 2024 | B |
| Token Distillation: Attention-Aware Input Embeddings for New Tokens | 2025 | various | https://arxiv.org/html/2505.20133v2 | Embedding initialization for new vocabulary tokens | A |

*Note:* Qwen3 family explicitly ties embeddings for smaller models but unties for larger (multi-billion+). Llama 3, OLMo 2, DeepSeek-V3, and Qwen3 all use untied embeddings.

---

### position-encoding

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE) | 2021 | Su et al. | https://arxiv.org/abs/2104.09864 | RoPE; de-facto standard in all modern decoder-only LLMs | A |
| Train Short, Test Long: ALiBi | 2021 | Press et al. | https://arxiv.org/abs/2108.12409 | ALiBi; still used in some efficient models | A |
| YaRN: Efficient Context Window Extension of Large Language Models | 2023 | Peng et al. | https://arxiv.org/abs/2309.00071 | NTK-aware RoPE scaling; 10x fewer tokens to extend context; ICLR 2024 | A |
| LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens | 2024 | Ding et al. | https://arxiv.org/abs/2402.13753 | Evolutionary search for non-uniform RoPE rescaling | A |
| LongRoPE2: Near-Lossless LLM Context Window Scaling | 2025 | various | https://arxiv.org/abs/2502.20082 | Near-lossless extension; 80× more token efficient than Meta's approach; preserves 98.5% short-context perf; Feb 2025 | A |
| Rope to Nope and Back Again: A New Hybrid Attention Strategy | 2025 | various | https://arxiv.org/abs/2501.18795 | Mixing RoPE and NoPE layers (iRoPE variant analysis); Jan 2025 | A |
| The Llama 4 Herd (iRoPE) | 2026 | Meta AI | https://arxiv.org/abs/2601.11659 | iRoPE: 3 RoPE layers + 1 NoPE layer interleaved; temperature scaling in NoPE; extrapolates to 10M ctx from 256K training | A |
| LaMPE: Length-aware Multi-grained Positional Encoding | 2025 | various | https://arxiv.org/html/2508.02308v1 | Training-free adaptive long-context scaling; Aug 2025 | A |

---

### attention-mechanism

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| GQA: Training Generalized Multi-Query Transformer Models | 2023 | Ainslie et al. | https://arxiv.org/abs/2305.13245 | GQA; now universal in production LLMs (Llama 3, Qwen3, Gemma 3…); EMNLP 2023 | A |
| DeepSeek-V2 (MLA) | 2024 | DeepSeek-AI | https://arxiv.org/abs/2405.04434 | Multi-head Latent Attention: 93.3% KV cache reduction vs MHA; outperforms GQA; May 2024 | A |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | 2022 | Dao et al. | https://arxiv.org/abs/2205.14135 | FlashAttention-1; IO-aware tiling | A |
| FlashAttention-2 | 2023 | Dao | https://arxiv.org/abs/2307.08691 | 2× speedup over FA-1; parallelism improvements | A |
| FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | 2024 | Shah et al. | https://arxiv.org/abs/2407.08608 | H100 Hopper; 740 TFLOPs FP16; ~1.2 PFLOPs FP8; NeurIPS 2024 | A |
| Differential Transformer | 2024 | Ye et al. | https://arxiv.org/abs/2410.05258 | Differential attention (two softmax maps subtracted); sparse patterns; less hallucination; ICLR 2025 | A |
| Native Sparse Attention (NSA) | 2025 | DeepSeek-AI | https://arxiv.org/abs/2502.11089 | Hardware-aligned trainable sparse attn; coarse + fine hierarchical; 64k-seq speedup over full attn; ACL 2025 | A |
| Tensor Product Attention Is All You Need | 2025 | Zhang et al. | https://arxiv.org/abs/2501.06425 | TPA: lower validation loss than MHA/MQA/GQA/MLA; ICML 2025 | A |
| TransMLA: MLA Is All You Need | 2025 | various | https://arxiv.org/abs/2502.07864 | Convert GQA pretrained models to MLA post-hoc; Feb 2025 | A |
| Learning to Route Between MHA, GQA, and MQA | 2024 | various | https://arxiv.org/abs/2512.20650 | MoAS: dynamic routing over attention strategies per token | A |
| FlashInfer: Efficient and Customizable Attention Engine | 2025 | various | https://arxiv.org/abs/2501.01005 | Outperforms FA kernels in diverse seq-length distributions; decode-focused | A |
| Ring Attention with Blockwise Transformers | 2023 | Liu et al. | https://arxiv.org/abs/2310.01889 | Ring attention enabling near-infinite context by distributing across devices | A |

---

### ffn-and-moe

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| GLU Variants Improve Transformer | 2020 | Shazeer | https://arxiv.org/abs/2002.05202 | SwiGLU, GEGLU, GeGLU; now universal in modern LLMs | A |
| Switch Transformers: Scaling to Trillion Parameter Models | 2021 | Fedus et al. | https://arxiv.org/abs/2101.03961 | Switch Transformer; top-1 routing; expert capacity | A |
| Mixture-of-Experts with Expert Choice Routing | 2022 | Zhou et al. | https://arxiv.org/abs/2202.09368 | Expert-choice vs token-choice routing; 2× faster convergence | A |
| Mixtral of Experts | 2024 | Jiang et al. | https://arxiv.org/abs/2401.04088 | Mixtral 8x7B: open sparse MoE baseline; top-2 routing | A |
| DeepSeekMoE: Towards Ultimate Expert Specialization | 2024 | DeepSeek-AI | https://arxiv.org/abs/2401.06066 | Fine-grained expert segmentation + shared experts; foundational for V2/V3 | A |
| DeepSeek-V2 | 2024 | DeepSeek-AI | https://arxiv.org/abs/2405.04434 | 236B-A21B; MLA + DeepSeekMoE combined; May 2024 | A |
| DeepSeek-V3 Technical Report | 2024 | DeepSeek-AI | https://arxiv.org/abs/2412.19437 | 671B-A37B; auxiliary-loss-free load balancing; multi-token prediction | A |
| Qwen3 Technical Report | 2025 | Qwen Team | https://arxiv.org/abs/2505.09388 | 128 experts, 8 activated; no shared experts; global-batch LB loss; May 2025 | A |
| Gemini 2.5 Technical Report | 2025 | Google DeepMind | https://arxiv.org/abs/2507.06261 | Sparse MoE; confirms MoE as dominant frontier architecture | A |
| Beyond Benchmarks: Understanding MoE Models through Internal Mechanisms | 2025 | various | https://arxiv.org/html/2509.23933v1 | Mechanistic interpretability of expert specialization | A |

---

### normalization

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| Root Mean Square Layer Normalization (RMSNorm) | 2019 | Zhang & Sennrich | https://dl.acm.org/doi/abs/10.5555/3454287.3455397 | RMSNorm; de-facto standard in all modern LLMs (Llama, Qwen, DeepSeek, Gemma…) | A |
| Peri-LN: Revisiting Normalization Layer in the Transformer Architecture | 2025 | various | https://arxiv.org/abs/2502.02732 | Peri-LN (pre + post per layer); adopted by Gemma and OLMo families; more stable than Pre-LN; ICML 2025 | A |
| Post-LayerNorm Is Back: Stable, ExpressivE, and Deep | 2026 | Chen & Wei | https://arxiv.org/abs/2601.19895 | Rehabilitates Post-LN with new stabilization techniques; Jan 2026 | A |
| Gated Removal of Normalization in Transformers | 2026 | various | https://arxiv.org/html/2602.10408 | Removing norm with gating; stable training + efficient inference | A |
| Geometric Interpretation of Layer Normalization vs RMSNorm | 2024 | various | https://arxiv.org/html/2409.12951v2 | Theoretical analysis of LN vs RMSNorm; irreversibility | A |
| OLMo 2 (QK-Norm, RMSNorm reordering) | 2025 | AI2 | https://arxiv.org/abs/2501.00656 | Practical example of Peri-LN + QK-Norm in production model | A |

---

### state-space-models

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | 2023 | Gu & Dao | https://arxiv.org/abs/2312.00752 | Mamba-1; selective SSM; content-aware recurrence; foundational | A |
| Transformers are SSMs: Structured State Space Duality (Mamba-2) | 2024 | Dao & Gu | https://arxiv.org/abs/2405.21060 | Mamba-2; SSD unifying SSM and attention; 2–8× faster; ICML 2024 | A |
| Jamba: A Hybrid Transformer-Mamba Language Model | 2024 | Lieber et al. (AI21) | https://arxiv.org/abs/2403.19887 | First production-scale hybrid; MoE + Mamba + Attention | A |
| Jamba-1.5: Hybrid Transformer-Mamba Models at Scale | 2024 | AI21 | https://arxiv.org/abs/2408.12570 | 94B active; 256K ctx; ExpertsInt8 quantization; Aug 2024 | A |
| Zamba: A Compact 7B SSM Hybrid Model | 2024 | Glorioso et al. | https://arxiv.org/abs/2405.16712 | Zamba1: Mamba backbone + shared attention blocks | A |
| The Zamba2 Suite: Technical Report | 2024 | Glorioso et al. | https://arxiv.org/abs/2411.15242 | Zamba2: Mamba-2 backbone, 2 shared attn blocks, 6× KV cache reduction; Nov 2024 | A |
| Hymba: A Hybrid-Head Architecture for Small Language Models | 2024 | NVIDIA | https://arxiv.org/abs/2411.13676 | Parallel (not serial) SSM + attention heads within same block; ICLR 2025 | A |
| Eagle and Finch: RWKV with Matrix-Valued States (RWKV-5/6) | 2024 | Peng et al. | https://arxiv.org/abs/2404.05892 | RWKV v5/v6; matrix-valued states; dynamic recurrence; COLM 2024 | A |
| RWKV-7 "Goose" with Expressive Dynamic State Evolution | 2025 | Peng et al. | https://arxiv.org/abs/2503.14456 | RWKV-7; generalized delta rule; vector-valued gating; TC⁰-beating expressiveness claims; Mar 2025 | A |
| A Survey of RWKV | 2024 | various | https://arxiv.org/abs/2412.14847 | Comprehensive RWKV survey through v6; Dec 2024 | B |
| Advancing Intelligent Sequence Modeling: S4 to Mamba Survey | 2025 | various | https://arxiv.org/abs/2503.18970 | SSM evolution survey; Mar 2025 | B |

---

### long-context

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| Ring Attention with Blockwise Transformers | 2023 | Liu et al. | https://arxiv.org/abs/2310.01889 | Ring attention; device-count × longer sequences; foundational for distributed long context | A |
| Leave No Context Behind: Infini-Attention | 2024 | Munkhdalai et al. | https://arxiv.org/abs/2404.07143 | Compressive memory + local+linear attention in single block; 1M passkey | A |
| Context Parallelism for Scalable Million-Token Inference | 2024 | various | https://arxiv.org/abs/2411.01783 | 93% parallelization efficiency at 1M ctx, Llama 405B, 128 H100s; Nov 2024 | A |
| LongRoPE2: Near-Lossless LLM Context Window Scaling | 2025 | various | https://arxiv.org/abs/2502.20082 | 80× more token efficient than Meta's approach for 128K extension | A |
| Star Attention: Efficient LLM Inference over Long Sequences | 2024 | various | https://arxiv.org/abs/2411.17116 | Block-sparse approximate attention; scales to 1M tokens; Nov 2024 | A |
| Extending Language Model Context to 3 Million Tokens on a Single GPU | 2025 | various | https://arxiv.org/abs/2502.08910 | Single-GPU 3M ctx; practical deployment considerations | A |
| Infinite Retrieval: Attention Enhanced LLMs in Long-Context Processing | 2025 | various | https://arxiv.org/abs/2502.12962 | InfiniRetri: 0.5B param model extended to 1M tokens | A |
| Native Sparse Attention (NSA) | 2025 | DeepSeek-AI | https://arxiv.org/abs/2502.11089 | Key long-context enabler; 64k seq speedup; trainable sparse attention | A |
| Arctic Long Sequence Training | 2025 | various | https://arxiv.org/abs/2506.13996 | Scalable multi-million token training; covers Llama 4 (10M) and Qwen 2.5; Jun 2025 | A |
| Gemini 1.5: Unlocking multimodal understanding across millions of tokens | 2024 | Google DeepMind | https://arxiv.org/abs/2403.05530 | Gemini 1.5 1M; MoE + sparse attn architecture details | A |

---

### multimodal-llm-extensions

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| Visual Instruction Tuning (LLaVA) | 2023 | Liu et al. | https://arxiv.org/abs/2304.08485 | LLaVA original; vision encoder + MLP projector + LLM paradigm | A |
| Qwen2-VL: Enhancing Vision-Language Model's Perception at Any Resolution | 2024 | Qwen Team | https://arxiv.org/abs/2409.12191 | Naive Dynamic Resolution + M-RoPE (2D/temporal); Sep 2024 | A |
| Qwen2.5-VL Technical Report | 2025 | Qwen Team | https://arxiv.org/abs/2502.13923 | Window attention in ViT; dynamic FPS; hour-long video; Feb 2025 | A |
| Qwen3-VL Technical Report | 2025 | Qwen Team | https://arxiv.org/abs/2511.21631 | Interleaved-MRoPE for time/width/height; video reasoning; Nov 2025 | A |
| Qwen3-Omni Technical Report | 2025 | Qwen Team | https://arxiv.org/html/2509.17765v1 | AuT audio encoder (trained from scratch on 20M hours); omni architecture; Sep 2025 | A |
| InternVL3: Exploring Advanced Training and Test-Time Recipes | 2025 | various | https://arxiv.org/abs/2504.10479 | Native multimodal pretraining (joint image+text from scratch); SOTA open MLLM; Apr 2025 | A |
| InternVL3.5 | 2025 | various | https://arxiv.org/abs/2508.18265 | Follow-on; versatility + reasoning + efficiency improvements; Aug 2025 | A |
| LLaVA-Video: Video Instruction Tuning With Synthetic Data | 2024 | Zhang et al. | https://arxiv.org/abs/2410.02713 | SlowFast video pooling; synthetic LLaVA-Video-178K dataset | A |
| Gemma 3 Technical Report | 2025 | Google DeepMind | https://arxiv.org/abs/2503.19786 | SigLIP2 vision encoder; multimodal Gemma family | A |
| Phi-4-Mini Technical Report: Compact Multimodal via Mixture-of-LoRAs | 2025 | Microsoft | https://arxiv.org/abs/2503.01743 | MoE-of-LoRAs for multimodal compact models; Mar 2025 | A |
| Unified Multimodal Understanding and Generation Models: Survey | 2025 | various | https://arxiv.org/abs/2505.02567 | Advances, challenges, opportunities across unified multimodal approaches | B |

---

## 2. Hot topics / active research threads (last 6 months)

### transformer-overview
The field has converged on **sparse MoE + GQA/MLA + RMSNorm + RoPE** as the dominant "modern decoder-only" template. Every major 2025–2026 frontier model (Llama 4, Qwen3, DeepSeek V3, Gemini 2.5, Mistral 3 Large) uses MoE. Pure dense architectures at the frontier are now the exception. The architectural comparison article at largo.dev (2026) explicitly states: "dense architectures can't compete on capability-per-FLOP" at frontier scale. Unified enc-dec architectures remain active only in specialized domains (time-series, translation).

### tokenization
The **tokenizer-free** direction is heating up. Meta's BLT (Dec 2024) matched LLaMA 3 at 50% fewer inference FLOPs and was published at ACL 2025 — this is the strongest existence proof yet. Concurrently, **SuperBPE** (COLM 2025, 2503.13423) shows BPE itself can be dramatically improved with cross-whitespace superword tokens. Several 2025–2026 papers (2602.13940, 2601.09039) take information-theoretic lenses to tokenizer design. Vocabulary sizes have grown: Llama 3 moved to 128k, Mistral models to ~131k. Post-cutoff (Feb–Apr 2026) papers continue at high pace.

### embeddings-and-tying
Weight tying is being actively re-examined. Two Feb–Mar 2026 papers (2602.04556, 2603.26663) show that standard tying biases embeddings toward the output space and propose fixes (Pseudo-Inverse Tying, gradient re-scaling). Industry trend: **untie at multi-billion scale** (Llama 3, DeepSeek-V3, OLMo 2, Qwen3 large), tie only for compact/edge models.

### position-encoding
**iRoPE** (interleaved RoPE + NoPE, Llama 4, 2601.11659) is the biggest recent development: 3 RoPE layers + 1 NoPE layer with temperature scaling lets the model extrapolate from 256K training to 10M context at inference. **LongRoPE2** (2502.20082, Feb 2025) is a near-lossless extension method 80× more token-efficient than Meta's earlier approach. The "Rope to Nope" paper (2501.18795) corroborates that mixing position-aware and position-free layers improves extrapolation. NTK-aware scaling / YaRN remains widely used for post-hoc context extension without retraining.

### attention-mechanism
Three simultaneous threads: (1) **MLA adoption and derivatives** — DeepSeek's MLA (2405.04434) is being retro-fitted to other models (TransMLA, 2502.07864); Tensor Product Attention (2501.06425) claims even better quality. (2) **Sparse attention for long context** — NSA (2502.11089) is the flagship trainable sparse attention; Differential Transformer (2410.05258, ICLR 2025) shows that subtracting two attention maps reduces noise and hallucination. (3) **FlashAttention engineering** — FA-3 (2407.08608) on H100 Hopper achieves near-1.2 PFLOPs FP8; FlashInfer (2501.01005) optimizes non-uniform sequence length batches for serving.

### ffn-and-moe
MoE is the dominant story. Key architectural debates in 2024–2025: **shared vs. no-shared experts** (DeepSeek uses shared experts; Qwen3 dropped them with global-batch LB loss instead), **fine-grained expert segmentation** (DeepSeekMoE's "segment + share" pattern), **auxiliary-loss-free load balancing** (DeepSeek V3 innovation). Llama 4 uses routed+shared expert structure with early-fusion multimodality. Gemini 2.5 is sparse MoE but architectural details are sparse (no parameter count disclosed). Mistral 3 Large is 675B-total/41B-active MoE. The SwiGLU FFN remains universal; recent GLU attention variants (2507.00022) apply gating to attention itself.

### normalization
**Peri-LN** (2502.02732, ICML 2025) — placing LN both before and after each sublayer — is gaining traction: both Gemma and OLMo families have converged on this. **Post-LN rehabilitation** (2601.19895, Jan 2026) is a surprise result: new stabilization techniques make post-LN competitive again in deep networks. A gated normalization removal approach (2602.10408) trades norm layers for gates at no quality cost. QK-Norm (applied to Q and K before attention) is increasingly common in production models (OLMo 2, Llama 3.1+) for training stability.

### state-space-models
**Hybrid SSM+Transformer** has become a mature research area. The main variants: serial interleaving (Jamba/Jamba-1.5: Mamba layers then Attn layers), shared-attention backbone (Zamba/Zamba2: Mamba-2 backbone with 2 shared attn blocks, 6× KV reduction), parallel fusion (Hymba: SSM + attention heads in same block). **RWKV-7 "Goose"** (2503.14456, Mar 2025) introduces generalized delta rule with vector-valued gating, claims state-tracking expressiveness beyond Transformers under standard complexity assumptions — a theoretical claim worth scrutinizing. Pure-Mamba at frontier scale has not materialized; the trend is hybrids capturing SSM efficiency for long sequences while retaining attention for recall.

### long-context
The **1M+ token regime is now real** for frontier models: Gemini 2.5 Pro (1M token window), Llama 4 Scout (10M claimed, trained at 256K), Claude 3.5/3.7 (200K). Key enabling technologies: context parallelism across devices (2411.01783), NSA sparse attention (2502.11089), iRoPE position encoding (Llama 4), efficient KV cache (MLA from DeepSeek). The practical bottleneck has shifted from "can we handle long context" to "can we *use* it effectively" — retrieval, faithfulness, and compute cost at 1M tokens remain open challenges. Star Attention (2411.17116) and InfiniRetri (2502.12962) are two recent efficient approximate approaches.

### multimodal-llm-extensions
**Early fusion / native multimodal pretraining** is replacing the bolt-on "vision encoder + MLP projector" approach. Llama 4 uses early fusion from scratch (not adapter). InternVL3 (2504.10479) jointly pretrained on image+text from scratch. Qwen's Omni series (Qwen3-Omni) trains an AuT audio encoder from scratch on 20M hours. SigLIP2 has become the vision encoder of choice for open models (Qwen3-VL, Gemma 3). **Video understanding** is now standard: LLaVA-Video SlowFast pooling, Qwen2.5-VL dynamic FPS, Qwen3-VL Interleaved-MRoPE for time×width×height. The audio modality is maturing but trails vision significantly.

---

## 3. Training-vs-current discrepancies

> **Trained-in view:** "MoE is an interesting scaling trick used by a few models (Mixtral, Switch Transformer); most frontier models are dense."
> **Current view (per largo.dev, 2026; Qwen3 report May 2025; Gemini 2.5 report Jul 2025):** MoE is now the *default* architecture at frontier scale. Every major 2025–2026 frontier model is MoE. Dense is niche at frontier.

> **Trained-in view:** "GQA is the current KV cache reduction technique; MQA is the budget version."
> **Current view (per DeepSeek-V2, arXiv 2405.04434, May 2024):** MLA (Multi-head Latent Attention) provides 93.3% KV cache reduction vs MHA while *outperforming* GQA. It has been adopted in DeepSeek V2/V3/V3.2 and is spreading (TransMLA, 2502.07864 converts pretrained GQA models).

> **Trained-in view:** "RoPE is extended post-hoc using YaRN or linear interpolation; context length is bounded by training."
> **Current view (per Llama 4, arXiv 2601.11659, Jan 2026):** iRoPE (interleaved NoPE layers + temperature scaling) enables extrapolation from 256K training to 10M context at inference. This is a qualitative regime change, not just better interpolation.

> **Trained-in view:** "Tokenization is BPE/WordPiece/SentencePiece; byte-level models are niche and slow."
> **Current view (per BLT, arXiv 2412.09871, Dec 2024; ACL 2025):** BLT at 8B scale matches LLaMA 3 at 50% fewer inference FLOPs. Tokenizer-free is no longer a fringe direction.

> **Trained-in view:** "Pre-LN is standard; Post-LN is unstable for deep networks."
> **Current view (per Peri-LN ICML 2025; Post-LN Is Back Jan 2026):** Peri-LN (norm around each sublayer) is being adopted in Gemma and OLMo families. Post-LN has been rehabilitated with new stabilization techniques (2601.19895).

> **Trained-in view:** "State space models (Mamba) are an alternative to transformers; Jamba is an early hybrid."
> **Current view (per Zamba2 Nov 2024; Jamba-1.5 Aug 2024; Hymba Nov 2024; RWKV-7 Mar 2025):** The SSM+Transformer hybrid is a mature, production-deployed architecture with multiple distinct families. Pure SSM at frontier scale has not materialized; hybrids with 1:5–1:6 attention:SSM ratios are the deployed form.

> **Trained-in view:** "FlashAttention-2 is state of the art for attention kernels."
> **Current view (per FA-3, arXiv 2407.08608, NeurIPS 2024):** FlashAttention-3 on H100 Hopper achieves 740 TFLOPs (FP16) and ~1.2 PFLOPs (FP8) — roughly 1.5–2× FA-2. FP8 is now viable for attention.

> **Trained-in view:** "Differential Attention / Diff Transformer was a research preprint."
> **Current view (per arXiv 2410.05258v2):** Published as ICLR 2025 conference paper; has follow-on work (Shared DIFF Transformer, understanding papers). Becoming mainstream attention variant.

---

## 4. Taxonomy refinements

Suggestions for `theory/kb/index/topics.md`:

**Split:**
- `ffn-and-moe` → **`ffn`** (SwiGLU/GLU variants, hidden-dim choices, parameter counts) + **`moe`** (routing algorithms, expert specialization, load balancing, fine-grained experts, shared experts). These are now sufficiently distinct that combining them hides important architecture decisions.
- `state-space-models` → **`ssm-pure`** (S4, S5, Mamba, Mamba-2, RWKV lineage) + **`ssm-hybrid`** (Jamba, Zamba, Hymba, OLMo-Hybrid — models that interleave SSM and attention). The hybrid design choices (serial vs. parallel, shared vs. full attention, ratio) are substantively different from pure SSM theory.
- `long-context` → **`long-context-techniques`** (position encoding extensions, sparse attention, ring attention, context parallelism) + **`long-context-evaluation`** (what actually works at 1M+, retrieval faithfulness, NIAH, real-world tasks). The engineering stack and evaluation criteria deserve separate treatment.

**Add:**
- **`reasoning-architectures`** — thinking tokens / chain-of-thought at architecture level; DeepSeek-R1 RL training method; Qwen3 thinking-budget mechanism; Llama 4 mid-training for reasoning. These are 2025's biggest new dimension.
- **`efficient-inference`** (or `serving-architecture`) — KV cache management (MLA, GQA, PagedAttention), speculative decoding, continuous batching, quantization-aware attention (FA-3 FP8). Not pure architecture, but architecture decisions are increasingly co-designed with serving constraints.
- **`native-multimodal`** — Early fusion (Llama 4, InternVL3, GPT-4o-style) as a separate topic from `multimodal-llm-extensions`. The paradigm shift from adapter-based to joint pretraining is architecturally significant.

**Rename:**
- `embeddings-and-tying` → `token-embeddings` — the scope has expanded beyond tying (now covers manifold structure, embedding geometry, initialization); "tying" is a subtopic.
- `transformer-overview` → `architecture-overview` — the field now includes SSM hybrids, MoE transformers, and tokenizer-free architectures; "transformer" is too narrow as a section heading.

**Merge:**
- `normalization` could be merged under a broader `training-stabilization` topic that also covers QK-Norm, weight initialization, gradient clipping, and DeepNorm — but only if the KB prioritizes training over pure architectural taxonomy. Keep separate if architectural placement (pre/peri/post LN) is the primary concern.

---

## 5. Forum / blog signals worth following up

- https://largo.dev/articles/frontier-llm-architectures-2026/ — Tier B: Clear structured comparison of MLA (DeepSeek), iRoPE (Llama 4), mHC (multi-head compression) as of early 2026. Good architecture taxonomy for the "frontier convergence" narrative.
- https://tridao.me/blog/2024/mamba2-part1-model/ — Tier B: Tri Dao's own blog on Mamba-2 / SSD. Primary author commentary on the SSM-attention duality theory.
- https://ai21.com/blog/rise-of-hybrid-llms/ — Tier B: AI21 (Jamba authors) on the hybrid LLM trend. Self-serving but technically grounded.
- https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html — Tier B: Technical deep-dive on RoPE scaling history from sinusoidal to 2M context; useful pedagogical resource.
- https://goombalab.github.io/blog/2024/mamba2-part1-model/ — Tier B: Gu & Dao lab blog on SSD theory (Mamba-2); complementary to arxiv paper.
- https://planetbanatt.net/articles/mla.html — Tier C: Clear community explanation of MLA math; useful for Phase 2 KB writing.
- https://sesamedisk.com/llm-architecture-gallery-2026/ — Tier C: Architecture gallery with LLM family diagrams; discovery signal for which models have notable structural differences.

---

## 6. Open questions / unresolved

1. **Gemini 2.5 architecture specifics** — The technical report (2507.06261) confirms sparse MoE + 1M context + TPUv5p but does not disclose parameter counts, expert counts, or attention mechanism variant. This is intentional opacity. Worth monitoring for follow-on disclosures or reverse-engineering analyses.

2. **GPT-4o / o1 / o3 / GPT-5 architecture** — OpenAI has disclosed almost nothing architectural. It is unclear whether GPT-4o uses MLA, what its MoE structure is, or how o1/o3-style reasoning is implemented architecturally (vs. pure post-training). Primary sources essentially do not exist; searches return only speculation.

3. **iRoPE extrapolation mechanism** — How does the NoPE-layer + temperature-scaling combination actually achieve 10M context from 256K training? The Llama 4 tech report (2601.11659) describes it at high level; a deeper theoretical treatment would be valuable. The "Rope to Nope" paper (2501.18795) is a starting point but may not fully explain the Llama 4 scaling factor.

4. **RWKV-7 expressiveness claims** — The paper (2503.14456) claims RWKV-7 can perform state tracking and recognize all regular languages, exceeding Transformer capabilities under standard complexity conjectures. This is a strong claim; independent verification and benchmarking on formal language tasks not yet found in this sweep.

5. **Tokenizer-free at frontier scale** — BLT (2412.09871) demonstrates viability at 8B/4T bytes. Whether tokenizer-free architectures can scale to 70B+ and match frontier quality is an open question. No such experiment found in this sweep.

6. **DeepSeek Sparse Attention (DSA) in V3.2** — The V3.2 paper (2512.02556) describes DSA as built on MLA with shared latent vectors across query heads. The paper appears to be from December 2025; full details of the routing mechanism and its interaction with MLA are worth a Phase 2 deep-dive.

7. **Shared vs. no-shared experts** — DeepSeek uses shared experts (always-on); Qwen3 dropped them. The tradeoff is not clearly resolved in the literature. Is global-batch load balancing alone sufficient to prevent expert collapse without shared experts?

8. **Native multimodal vs. adapter at scale** — Llama 4 and InternVL3 claim native early-fusion is superior, but controlled comparisons are scarce. Most papers compare to their own prior adapter-based baseline.

9. **Claude 3.x / 4.x architectural disclosures** — Anthropic has not published architecture papers for Claude 3.5/3.7/4/4.5/4.6/4.7. Model cards contain no structural details. This sweep found nothing beyond confirmation that these models exist and perform at frontier level.

10. **Grok 3/4/5 architecture** — xAI has not published architecture papers. No primary-source architectural details found in this sweep.
