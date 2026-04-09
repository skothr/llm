# LLM Core Architecture — Base Document Design

## Goal

A top-down architectural reference covering the Transformer and its decoder-only evolution, at matrix-dimension depth with implementation notes. Serves as the foundational document for the `theory/` workspace. Written as a learning resource grounded in canonical sources.

## Format

- **LaTeX (.tex)** — formal document with math, derivations, concrete dimensions
- **HTML companion** — interactive architecture diagram with clickable layers showing tensor shapes and data flow

## Document Structure (LaTeX) — Revised After Source Review

1. **The Original Transformer** — encoder-decoder overview, why attention replaced RNNs (complexity comparison from Vaswani §4: self-attention O(n²·d) vs recurrent O(n·d²) per layer, but O(1) sequential ops vs O(n)), data flow through encoder and decoder stacks [vaswani2017]
2. **Tokenization** — BPE algorithm step-by-step (Sennrich Algorithm 1: start with character vocab, iteratively merge most frequent pairs), vocabulary sizes (37K in original Transformer, 40K in GPT-1, 32K in LLaMA via SentencePiece), byte-fallback [sennrich2016, touvron2023]
3. **Embeddings** — token embedding matrix W_e ∈ ℝ^{V×d}, weight tying between input embedding and output projection (Vaswani §3.4, GPT-1 §3.1), √d_model scaling factor in original Transformer [vaswani2017, radford2018]
4. **Positional Encoding** — why position info is needed (attention is permutation-invariant), sinusoidal PE (Vaswani §3.5), learned PE (GPT-1), Rotary Position Embedding/RoPE (encode via rotation matrix, dot product depends only on relative position m-n, applied to Q and K only) [vaswani2017, radford2018, su2021]
5. **Attention Mechanism** — scaled dot-product (Attention(Q,K,V) = softmax(QK^T/√d_k)V), why scale by √d_k (variance of dot products grows with d_k), multi-head attention (project to h subspaces, attend in parallel, concatenate), three attention types in original (encoder self-attention, decoder masked self-attention, encoder-decoder cross-attention), full matrix dimensions (W_Q,W_K ∈ ℝ^{d×d_k}, W_V ∈ ℝ^{d×d_v}, W_O ∈ ℝ^{hd_v×d}) [vaswani2017]
6. **Feed-Forward Network** — position-wise: applied independently to each position. Evolution: ReLU FFN(x)=max(0,xW₁)W₂ (Vaswani), GELU (GPT-1), SwiGLU FFN(x)=(Swish(xW)⊗xV)W₂ (3 matrices, reduce d_ff by 2/3 to match params, LLaMA uses 2/3·4d). Dimension expansion: d_ff=4d convention [vaswani2017, radford2018, shazeer2020, touvron2023]
7. **Normalization and Residual Connections** — LayerNorm formula (γ·(v-μ)/σ + β), Post-LN (original: sublayer → add → norm) vs Pre-LN (norm → sublayer → add + final norm before output), gradient analysis: Post-LN gradient O(d√(ln d)) independent of L, Pre-LN O(d√(ln d/L)), RMSNorm (drop mean-centering, keep RMS re-scaling, 7-64% faster). Residual connections enable gradient flow, interact with norm placement [vaswani2017, xiong2020, zhang2019, touvron2023]
8. **The Decoder-Only Shift** — GPT-1: take decoder half, remove cross-attention, use causal mask (set upper-triangle to -∞ before softmax). Why encoder was dropped: language modeling only needs left-to-right context. Architectural evolution: Transformer (2017) → GPT-1 (2018, decoder-only + learned PE + GELU) → GPT-3 (2020, scale to 175B) → LLaMA (2023, pre-norm RMSNorm + SwiGLU + RoPE) [radford2018, brown2020, touvron2023]
9. **Output Head** — final layer norm (Pre-LN adds one before projection), linear projection to vocab logits (W_e^T · h_n), softmax for probability distribution, weight tying reduces parameters [vaswani2017, radford2018]
10. **Putting It Together** — full forward pass walkthrough with LLaMA-7B dimensions: d=4096, h=32, L=32, d_ff=11008 (2/3·4·4096), vocab=32000. Trace a token from input through every operation to output logit [touvron2023]

## HTML Companion

Interactive architecture diagram showing the full model with clickable layers that highlight data flow and show tensor shapes at each stage.

## Source Management

All claims grounded in canonical papers. Sources stored in `theory/sources/`:

- `sources/index.md` — master index (citation key, title, authors, year, URL, summary, local filename)
- `sources/papers/` — local PDF copies
- `sources/other/` — non-paper sources (blogs, etc.)

### Key Sources

| Citation Key | Paper | Relevance |
|---|---|---|
| vaswani2017 | Vaswani et al., "Attention Is All You Need" (2017) | Original Transformer architecture |
| radford2018 | Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018) | Decoder-only / GPT |
| brown2020 | Brown et al., "Language Models are Few-Shot Learners" (2020) | GPT-3 scaling |
| touvron2023 | Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023) | Modern decoder-only architecture |
| sennrich2016 | Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (2016) | BPE tokenization |
| su2021 | Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021) | RoPE |
| shazeer2020 | Shazeer, "GLU Variants Improve Transformer" (2020) | SwiGLU activation |
| xiong2020 | Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020) | Pre-norm vs post-norm |
| zhang2019 | Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019) | RMSNorm (used by LLaMA) |

## Out of Scope

Training, loss functions, inference optimization, fine-tuning, scaling laws — these will be separate documents.

## Process

1. Gather and download all sources
2. Re-review document outline against source content for accuracy and comprehensiveness
3. Write LaTeX document grounded in sources
4. Build HTML companion visualization
