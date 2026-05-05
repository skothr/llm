---
topic: architecture/transformer-overview
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - vaswani2017
  - radford2018
  - radford2019-gpt2
  - brown2020
  - touvron2023
  - meta-llama3
  - meta-llama4
  - deepseek-v3
  - qwen3
secondary_sources:
  - devlin2019-bert
  - olmo2
  - chowdhery2022
  - gemini2-5
  - gemma3
  - jiang2023
related_topics:
  - architecture/attention-mechanism
  - architecture/position-encoding
  - architecture/ffn
  - architecture/moe
  - architecture/normalization
  - architecture/embeddings-and-tying
  - architecture/tokenization
  - architecture/multi-token-prediction
---

# Transformer overview (and the modern decoder-only template)

This note is the **block-diagram orientation** for the architecture/
area. It defines the original 2017 encoder–decoder Transformer, the
2018–2020 decoder-only adaptation, and the 2023–2026 frontier template
(SwiGLU + RMSNorm + RoPE + GQA/MLA + sparse MoE). Every load-bearing
claim cites a primary source; sublayer details are deferred to the
per-sublayer notes.

## 1. Formal definition — the original Vaswani 2017 architecture

The Transformer is a stack of identical blocks operating on a sequence
of $d_{\text{model}}$-dimensional vectors, with no recurrence and no
convolution `[vaswani2017 §3; kb/excerpts/vaswani2017#sec-3-2]`. An
**encoder block** maps $X \in \mathbb{R}^{N \times d_{\text{model}}}$ to
$Y \in \mathbb{R}^{N \times d_{\text{model}}}$ via two sublayers:

$$X' = \mathrm{LN}(X + \mathrm{MHA}(X, X, X)) \tag{1}$$
$$Y  = \mathrm{LN}(X' + \mathrm{FFN}(X')) \tag{2}$$

A **decoder block** has three sublayers — masked self-attention, encoder–
decoder cross-attention, and FFN
`[vaswani2017 §3.1; kb/excerpts/vaswani2017#sec-3-2-3]`. The original
"Post-LN" placement (Eq. 1–2) applies LayerNorm *after* the residual
add, which is the source of the warmup-required training instability
later diagnosed in Pre-LN work `[xiong2020 §1; see kb/notes/architecture/normalization.md]`.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $V$ | tokenizer vocabulary; $|V|$ is the number of token IDs |
| $N$ | sequence length (input tokens) |
| $B$ | batch size |
| $d_{\text{model}}$ | residual-stream width |
| $L$ | number of Transformer blocks |
| $h$ | number of attention heads per block |
| $d_h$ | per-head dimension; usually $d_{\text{model}}/h$ |
| $d_{\text{ff}}$ | FFN inner dimension; Vaswani: $4 d_{\text{model}}$ |
| $E_{\text{in}} \in \mathbb{R}^{|V| \times d_{\text{model}}}$ | input token embedding |
| $E_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times |V|}$ | LM-head projection |
| $E$ | number of MoE experts (when applicable) |
| $k$ | MoE top-$k$ activations per token |

The base Vaswani 2017 model uses $d_{\text{model}}=512$, $L=6$ encoder +
$6$ decoder, $h=8$, $d_h=64$, $d_{\text{ff}}=2048$
`[vaswani2017 §3.1; kb/excerpts/vaswani2017#sec-3-2-2]`.

## 2. Mechanism — token IDs to logits

For a decoder-only LLM (the dominant 2018+ form), the forward pass on
input token IDs $\mathbf{t} \in \{0, \ldots, |V|-1\}^N$ proceeds:

1. **Embed** $X^0 = E_{\text{in}}[\mathbf{t}] + P$, where $P$ is the
   positional signal (additive sinusoidal in Vaswani; learned absolute
   in GPT-1; **multiplicative rotary (RoPE)** in modern LLMs — see
   `kb/notes/architecture/position-encoding.md`). Shape:
   $X^0 \in \mathbb{R}^{B \times N \times d_{\text{model}}}$.
2. **Stack of $L$ identical blocks**, each:
   - Attention sublayer: $X^{\ell+\frac12} = X^\ell +
     \mathrm{MHA}(\mathrm{Norm}(X^\ell))$, with a causal mask
     `[vaswani2017 §3.2.3; kb/excerpts/vaswani2017#sec-3-2-3]`.
   - FFN sublayer: $X^{\ell+1} = X^{\ell+\frac12} +
     \mathrm{FFN}(\mathrm{Norm}(X^{\ell+\frac12}))$
     `[vaswani2017 §3.3]`. Modern variant: SwiGLU FFN
     `[shazeer2020 §2 Eq.6; kb/excerpts/shazeer2020#sec-2-eq6]`.
   - The arrangement above is **Pre-LN**: norm precedes each sublayer,
     residual adds the unnormalized stream. Adopted by GPT-2 onward
     `[radford2019-gpt2 §2.3; kb/excerpts/radford2019-gpt2#sec-2-3]`.
3. **Final norm + LM head**: $\mathrm{logits} = \mathrm{Norm}(X^L) E_{\text{out}}$,
   shape $(B, N, |V|)$.
4. **Softmax + cross-entropy** at training; **sample / argmax** at
   inference.

The block stack is the **residual stream**: at every layer $\ell$, a
position-$i$ vector $X^\ell_{B,i,:}$ accumulates additive contributions
from each sublayer. Mechanistic interpretability work treats this stream
as the workspace through which attention heads and FFN sublayers
communicate `[olsson2022-induction-heads §2; see kb/notes/interpretability/mechanistic-interpretability.md]`.

[INTUITION] Each Transformer block does two distinct operations: the
attention sublayer **mixes information across positions** (every position
can read every earlier position), and the FFN sublayer **mixes channels
within a position** (no inter-token interaction). The factorization is
strict; a decoder-only LLM has no other inter-token mixing path. The
canonical math is Eq. 1–2 plus the masked variant — those are the only
two operations a decoder block contains.

## 3. The encoder–decoder → decoder-only collapse

Vaswani 2017's encoder–decoder design maps a source sequence (encoder)
to a target sequence (decoder, with cross-attention to encoder output).
Three architectural lineages branched from it:

| Lineage | Key paper | Structure | Use case |
|---|---|---|---|
| **Encoder-only** | BERT 2018 | $L$-block stack, bidirectional self-attention, MLM objective | classification, retrieval `[devlin2019-bert §3.1]` |
| **Encoder–decoder** | T5 2019 | full Vaswani structure, span-corruption objective | translation, span infilling `[raffel2019 §2]` |
| **Decoder-only** | GPT-1/2/3 | single causal stack, next-token objective | open-ended generation `[radford2018 §3]` |

The decoder-only form **won for LLMs**:

- **Causal mask = no encoder needed.** A single stack with a causal
  mask gives autoregressive generation directly
  `[radford2018 §3; vaswani2017 §3.2.3]`.
- **In-context learning emerges.** GPT-3 showed that scale + decoder-
  only + next-token objective produces few-shot learning without
  task-specific architectures `[brown2020 §3]`. The "task as prompt"
  paradigm makes the encoder unnecessary.
- **KV-cache reuse.** During autoregressive decoding the prefix's
  $(K_t, V_t)$ vectors are cached and reused
  `[shazeer2019 §1; kb/excerpts/shazeer2019#sec-1]`. Encoder–decoder
  stacks must recompute encoder representations or design separate
  caches.

T5-style enc–dec persists in translation and code-by-task settings
`[raffel2019 §3]`; encoder-only in retrieval (sentence-BERT, ColBERT).
Frontier general-purpose LLMs (LLaMA, GPT-3+, Claude, Gemini, Qwen,
DeepSeek) are decoder-only.

## 4. The "modern decoder-only template" (LLaMA-1 baseline, 2023)

LLaMA-1 (Touvron et al. 2023) crystallized a set of modifications that
have since become near-universal `[touvron2023 §2.2]`:

| Component | Vaswani 2017 | LLaMA-1 (modern baseline) |
|---|---|---|
| Norm placement | Post-LN, after residual | **Pre-LN**, before sublayer |
| Norm type | LayerNorm (mean+var) | **RMSNorm** (var-only) `[zhang2019 §3.1]` |
| FFN | ReLU 2-matrix | **SwiGLU** 3-matrix, $d_{\text{ff}} \approx \tfrac{8}{3} d$ `[shazeer2020 §2 Eq.6; kb/excerpts/shazeer2020#sec-2-eq6]` |
| Position | Additive sinusoidal | **RoPE** at every layer `[su2021 §3.2.2; kb/excerpts/su2021#sec-3-2-2]` |
| Bias terms | Yes | Removed throughout |
| Vocabulary | ~32K WordPiece | ~32K SentencePiece-BPE `[touvron2023 §2.3]` |

The block becomes:

```
X' = X + Attn(RMSNorm(X))     # RoPE applied inside Attn
Y  = X' + SwiGLU(RMSNorm(X')) # bias-free
```

with a **final RMSNorm** before the LM head `[touvron2023 §2.2]`. The
LM head is either tied ($E_{\text{out}} = E_{\text{in}}^\top$) for
small models or **untied** for multi-billion-scale (LLaMA-3, OLMo 2,
DeepSeek-V3) — see `kb/notes/architecture/embeddings-and-tying.md`.

## 5. The 2024–2026 frontier convergence

Every major frontier open model from 2024 onward retains the LLaMA-1
core but layers on four sublayer-level changes
`[Phase 1 sweep §2 transformer-overview]`:

### 5.1 Attention: GQA → MLA (or stay with GQA)

GQA-$g$ shares one $K, V$ pair across $g$ query heads
`[ainslie2023 §2.2; kb/excerpts/ainslie2023#sec-2-2]`. LLaMA-2/3,
Mistral, Qwen3, Gemma 3, OLMo 2 all use GQA. DeepSeek-V2/V3 instead use
**MLA** (Multi-head Latent Attention), reconstructing per-head $K, V$
from a $d_c$-dim latent
`[deepseek-v2 §2.1.2 Eq.9; kb/excerpts/deepseek-v2#sec-2-1-2]`. Full
treatment in `kb/notes/architecture/attention-mechanism.md`.

### 5.2 FFN: dense SwiGLU → sparse MoE SwiGLU

The dense SwiGLU FFN remains pointwise; **MoE replaces it with $E$
parallel SwiGLU experts** routed per-token
`[fedus2021-switch §2.1; deepseekmoe2024 §2.2; mixtral2024 §2.2]`.
At frontier scale (Llama 4, Qwen3 235B, DeepSeek-V3 671B, Gemini 2.5,
Mixtral) MoE is now the **default**, not the exception
`[Phase 1 sweep §3 stale-prior corrections]`. Full treatment in
`kb/notes/architecture/moe.md`.

[CONTRADICTION] Dense vs. MoE at frontier: as of 2026 nearly every
public frontier disclosure is MoE; one article (largo.dev 2026,
Tier B) asserts "dense can't compete on capability-per-FLOP" at
frontier scale, but Anthropic's Claude family architecture is
undisclosed and may not be MoE. Treat the "MoE is universal at the
frontier" claim as a strong empirical regularity restricted to
publicly disclosed architectures.

### 5.3 Normalization: Pre-LN → Peri-LN, plus QK-Norm

**Peri-LN** (norm before *and* after each sublayer)
`[peri-ln2025 §3]` is adopted by Gemma 3 and OLMo 2 `[olmo2 §3.1]`.
**QK-Norm** (RMSNorm on $Q$ and $K$ before the dot product) reduces
attention-logit explosion at long context and is in OLMo 2 and
LLaMA 3.1+ `[olmo2 §3.1]`. Full treatment in
`kb/notes/architecture/normalization.md`.

### 5.4 Position: RoPE → iRoPE, NoPE-interleaved

LLaMA 4 introduces **iRoPE**: 3 RoPE layers + 1 NoPE (no-position)
layer, interleaved, with temperature scaling on the NoPE layers
`[meta-llama4 §2.1]`. Trained at 256K tokens, claimed inference
extrapolation to 10M `[meta-llama4 §2.4]`. Full treatment in
`kb/notes/architecture/position-encoding.md`.

### 5.5 Output: single-head → multi-token prediction

**MTP** (Multi-Token Prediction) adds auxiliary heads predicting tokens
$t+2, t+3, \ldots$ jointly with the main next-token head
`[deepseek-v3 §2.3.1]`. Originated as a Meta research paper
`[gloeckle2024-mtp §1]`; productionized by DeepSeek-V3. Full treatment
in `kb/notes/architecture/multi-token-prediction.md`.

## 6. Tensor-shape reference across frontier models

Where parameter counts are public; "—" = undisclosed. $d$ = $d_{\text{model}}$,
$L$ = block count, $h_q$ / $h_{kv}$ = query/KV head counts.

| Model | $d$ | $L$ | $h_q$ | $h_{kv}$ | Attn | Position | FFN | MoE: $E/k$ | Source |
|---|---|---|---|---|---|---|---|---|---|
| LLaMA-1 7B | 4096 | 32 | 32 | 32 | MHA | RoPE-10K | SwiGLU $d_{\text{ff}}=11008$ | dense | `[touvron2023 Tab.2]` |
| LLaMA-3 8B | 4096 | 32 | 32 | 8 | GQA-8 | RoPE-500K | SwiGLU | dense | `[meta-llama3 §3.1]` |
| LLaMA-3 70B | 8192 | 80 | 64 | 8 | GQA-8 | RoPE-500K | SwiGLU | dense | `[meta-llama3 §3.1]` |
| Mistral 7B | 4096 | 32 | 32 | 8 | GQA-8 + SWA | RoPE | SwiGLU | dense | `[jiang2023 §2]` |
| DeepSeek-V3 | 7168 | 61 | 128 | (latent $d_c$=512) | MLA | RoPE-decoupled | DeepSeekMoE | $E$=256, $k$=8 + 1 shared | `[deepseek-v3 §2.1; kb/excerpts/deepseek-v3-training#abstract]` |
| Qwen3 235B | — | — | — | — | GQA | RoPE | SwiGLU | $E$=128, $k$=8, no shared | `[qwen3 §2.1]` |
| Gemma 3 27B | — | — | — | — | GQA + interleaved local/global | RoPE | SwiGLU + Peri-LN | dense | `[gemma3 §2.1]` |
| OLMo 2 13B | 5120 | 40 | 40 | 8 | GQA + QK-Norm | RoPE | SwiGLU + Peri-LN | dense | `[olmo2 §3.1]` |
| Llama 4 (per-expert active) | — | — | — | — | GQA | iRoPE (3:1 RoPE:NoPE) | SwiGLU | MoE w/ shared expert | `[meta-llama4 §2.1]` |

The table illustrates the convergence: **same skeleton (Pre-/Peri-LN +
{RMS,Layer}Norm + GQA/MLA + RoPE/iRoPE + SwiGLU dense/MoE)**, with
per-lab specifics on KV sharing, MoE sparsity pattern, and norm
placement.

## 7. Intuitions and analogies

[ANALOGY] The residual stream is often described as a **"shared
chalkboard"**: every block reads what's there, writes additive updates,
and passes it on. The chalkboard analogy returns to canonical form via
the residual recurrence $X^{\ell+1} = X^\ell + \mathrm{Sublayer}(X^\ell)$
in §2. Heads and FFNs cannot communicate except via the chalkboard;
this is why mechanistic interpretability tools (logit lens, activation
patching) target the residual stream as the substrate
`[belrose2023-tuned-lens §1]`.

[INTUITION] Decoder-only is **not "encoder-decoder minus encoder"** —
it is a different objective regime. The encoder–decoder split assumes
input and output sequences are distinguishable (e.g., source and target
languages). Decoder-only collapses both into one autoregressive stream,
which is why "task as prompt" works: there is no architectural marker
for "input" vs. "output" — only the causal mask. The canonical math
is the masked self-attention of `[vaswani2017 §3.2.3]` applied to the
unified prefix.

[INTUITION] Why **Pre-LN was a regression** that worked: Vaswani's
Post-LN places normalization on the residual path, so deep stacks
amplify gradients exponentially without warmup. Pre-LN moves norm off
the residual, restoring stability `[xiong2020 §3.1]`. The cost is
"residual drift": stream norms grow with depth, motivating the
final-RMSNorm in modern designs and the partial Post-LN comeback in
`[peri-ln2025 §3]`. The canonical effect is the gradient analysis in
Xiong et al.

## 8. Frontier and open questions (as of 2026-05)

- **Hybrid SSM-Transformer architectures.** Jamba, Zamba2, Hymba, and
  RWKV-7 interleave or fuse SSM and attention layers
  `[lieber2024-jamba §2; rwkv7-2025 §2]`. Pure-SSM at frontier scale
  has not materialized — the deployed form is hybrid with 1:5 to 1:6
  attention:SSM ratios. Full treatment in
  `kb/notes/architecture/state-space-models.md`. [CONTRADICTION] RWKV-7
  claims state-tracking expressiveness exceeding Transformers under
  standard complexity conjectures; independent verification on formal-
  language benchmarks is sparse.
- **Tokenizer-free.** BLT (Meta 2024) matched LLaMA-3 quality at 50%
  fewer inference FLOPs at 8B scale `[blt2024 §1]`. Whether tokenizer-
  free scales to 70B+ is open. See
  `kb/notes/architecture/tokenization.md`.
- **Native multimodal.** Llama 4, InternVL3, and Gemini 2.5 train
  jointly on image+text from scratch rather than vision-encoder +
  adapter `[meta-llama4 §2; internvl3-2025 §3.1]`. The architectural
  shift is from "LLM with grafted vision" to "LLM with first-class
  image tokens." See `kb/notes/architecture/multimodal-llm-extensions.md`.
- **Reasoning architectures.** DeepSeek-R1 and Qwen3 expose a
  thinking-mode toggle that switches between long-CoT and direct
  generation `[deepseek-r1 §2; qwen3 §3.2]`. Whether this requires
  architectural support beyond training-time changes is contested.
  See `kb/notes/architecture/reasoning-architectures.md`.
- **Architecture opacity at the frontier.** Anthropic (Claude 3.5/3.7/
  4.x), OpenAI (GPT-4o/o1/o3/o4/GPT-5), and xAI (Grok 3+) have not
  published architecture details. Inference-stack reverse-engineering
  is the only public information path.

## 9. See also

- `kb/notes/architecture/attention-mechanism.md` — the attention
  sublayer in detail, including MQA/GQA/MLA and FlashAttention.
- `kb/notes/architecture/position-encoding.md` — RoPE / iRoPE / YaRN
  and the position-injection story.
- `kb/notes/architecture/ffn.md` — SwiGLU and the FFN sublayer.
- `kb/notes/architecture/moe.md` — MoE replacement of the FFN.
- `kb/notes/architecture/normalization.md` — RMSNorm, Pre-/Peri-/Post-
  LN, QK-Norm.
- `kb/notes/architecture/embeddings-and-tying.md` — input/output
  embedding decisions.
- `kb/notes/architecture/tokenization.md` — BPE / SentencePiece /
  tiktoken / BLT.
- `kb/notes/architecture/multi-token-prediction.md` — MTP heads.
- `kb/notes/architecture/state-space-models.md` — non-attention
  alternatives.
- `kb/notes/architecture/long-context.md` — frontier context-length
  techniques.
- `kb/notes/architecture/multimodal-llm-extensions.md` — vision/audio
  extensions.
- `kb/notes/architecture/reasoning-architectures.md` — thinking-mode
  models.
- `kb/notes/scaling/scaling-frontier.md` — how the template scales.
