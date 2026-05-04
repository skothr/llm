---
topic: scaling/scaling-frontier
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - hoffmann2022-chinchilla
  - snell2024
  - scaling-laws-precision-2024
secondary_sources:
  - meta-llama3
  - meta-llama4
  - deepseek-v3
  - deepseek-r1
  - gemini2-5
  - qwen3
related_topics:
  - scaling/chinchilla
  - scaling/kaplan-laws
  - scaling/mu-transfer
  - scaling/inference-time-compute-scaling
  - training/mixed-precision-and-stability
  - inference/quantization
---

# Scaling frontier — what frontier labs actually do (2024–2026)

Where the community thinks the frontier is **right now**, and how that
diverges from the textbook Kaplan-then-Chinchilla narrative. This is a
synthesis topic: every load-bearing claim points at a specific
training/serving recipe published in a tech report. Treat the citations
in this note as Tier-A evidence of **what was done**, not necessarily
of "what is optimal."

## 1. The four-axis scaling frontier

The community has expanded "scaling" from a single $C_{\text{train}}$
axis (Kaplan/Chinchilla) into a four-axis frontier:

| Axis | Objective | Canonical paper |
|---|---|---|
| Training compute | Lower pre-training loss at fixed FLOPs | `[hoffmann2022-chinchilla; kb/notes/scaling/chinchilla.md]` |
| Inference compute | Higher accuracy at fixed test-time FLOPs | `[snell2024; kb/notes/scaling/inference-time-compute-scaling.md]` |
| Precision | Lower bits per weight/activation at fixed loss | `[scaling-laws-precision-2024; kb/index/papers.json#scaling-laws-precision-2024]` |
| Data quality / mix | Lower loss at fixed token count via better composition | `[fineweb2024, data-mixing-laws-2024]` |

A frontier lab in 2026 is jointly optimizing all four, *not just*
training compute. The empirical surface is multi-dimensional and
incompletely mapped.

## 2. The "post-Chinchilla" industry move

Hoffmann et al. 2022 showed *training-FLOP-optimal* tokens-per-param
≈ 20 `[hoffmann2022-chinchilla §3 Table 3; kb/excerpts/hoffmann2022-chinchilla#sec-3-table-3]`.
What frontier models actually train on (selected; tokens-per-param
ratios approximate, drawn from official tech reports):

| Model | Active params | Training tokens | Tokens / param | Source |
|---|---|---|---|---|
| Chinchilla 70B (2022) | 70B | 1.4T | 20.0 | `[hoffmann2022-chinchilla §4.1]` |
| Llama 2 70B (2023) | 70B | 2T | 28.6 | `[meta-llama3 → predecessor cited]` |
| Llama 3 8B (2024) | 8B | ~15T | ~1875 | `[meta-llama3]` |
| Llama 3 70B (2024) | 70B | ~15T | ~214 | `[meta-llama3]` |
| Qwen2.5 (2024) | varies | 18T | dense varies | `[qwen2-5]` |
| DeepSeek-V3 (2024) | 37B active / 671B total | 14.8T | ~400 (active basis) | `[deepseek-v3]` |
| Llama 4 (2026) | varies | ~2T+ FP8 disclosed | varies | `[meta-llama4]` |

Reading: small dense models (Llama 3 8B) are over-trained by ~94×
relative to Chinchilla. MoE models (DeepSeek-V3) are over-trained
relative to *active* params by ~20× (or roughly Chinchilla-optimal
relative to total params, ≈ 22).

### 2.1 Why over-train past Chinchilla?

Chinchilla minimizes only $C_{\text{train}}$. The actual cost a
frontier lab pays is:

$$\text{Total} = C_{\text{train}} + N_{\text{queries-served}} \cdot C_{\text{inf-per-query}}.$$

When $N_{\text{queries-served}} \gg 1$ (consumer products, API), the
second term dominates and the optimum shifts toward smaller $N$ +
proportionally more $D$. Llama 3 8B's ~1875 tokens/param is the
endpoint of this optimization for "small model destined for billions of
queries."

[INTUITION] **The "Chinchilla trap" is a community term** for the
realization that compute-optimal pre-training is *not* deployment-
optimal. A model trained exactly at Chinchilla-optimum is the cheapest
to *pre-train* but often too large to *serve* cheaply. The trap is real
but is not a flaw in the Chinchilla derivation — it is a flaw in
substituting "pre-train compute" for "total cost" as the objective.

[FORUM-SIGNAL] The "Chinchilla trap" framing originates in 2024
community discussion (Latent Space, Twitter/X threads, Lambda Labs
analyses). It is widely repeated; it should be cited as a
*description-of-current-practice*, not as a Tier-A claim about
optimality.

### 2.2 Where Chinchilla still rules

For models that will be served *only at the lab* (research models,
internal tools, single-customer deployments where $N_{\text{queries-served}}$
is small), Chinchilla-optimal training is still the right target.
Chinchilla 70B itself, DeepMind's Gopher-replacement runs, Anthropic's
research-only training runs likely all sit closer to Chinchilla than
to Llama-3 territory.

## 3. The reasoning-model paradigm shift

DeepSeek-R1 (Jan 2025) and the o1/o3/Gemini-Thinking family
demonstrated that **inference-time compute is now a deployable scaling
axis**, not just an academic finding `[snell2024]` `[deepseek-r1]`.
Industry projections (Epoch AI, 2025) put inference at **~75% of total
AI compute by 2030**.

Production reasoning models offer configurable thinking budgets — the
productionization of Snell's compute-optimal idea
`[gemini2-5; kb/index/papers.json#gemini2-5]`. Frontier labs now train
*two* model variants (or one with a thinking-mode switch) per family:

| Variant | Cost per query | Use case |
|---|---|---|
| Non-thinking / fast | Low | Most consumer queries |
| Thinking / reasoning | 5–100× higher | Hard math, code agents, agentic loops |

Qwen3's thinking/non-thinking-mode toggle via system prompt
`[qwen3; kb/index/papers.json#qwen3]` and Gemini 2.5's `thinkingBudget`
API are the canonical examples.

## 4. Precision as a scaling axis

Kumar et al. 2024 `[scaling-laws-precision-2024]` derive
**precision-aware scaling laws**: as training data $D$ grows past a
threshold, post-training quantization becomes increasingly costly —
eventually *actively harmful*. The implication is that the over-training
strategy of Llama 3 (≈1875 tokens/param) has a hidden cost: it
**makes models harder to quantize for inference**, partly offsetting
the inference-compute savings of the smaller $N$.

Native low-precision training (FP8 in DeepSeek-V3, FP8 in Llama 4) is
the response: pre-train at the precision you'll deploy, not above it.
u-μP `[u-mup2024]` is the principled parametrization that enables
this without dynamic rescaling. See
`kb/notes/training/mixed-precision-and-stability.md`.

[CONTRADICTION] On 1-bit / ternary training. BitNet b1.58
`[ma2024-bitnet]` claims ternary {−1, 0, +1} weights match FP16 at
scale. Independent reproduction at 7B+ is thin; the claim is
load-bearing for projected hardware roadmaps but methodologically
contested as of 2026.

## 5. The frontier model lineup (as of early 2026)

Tier-A reference points, from the corresponding tech-report cards in
`papers.json`:

- **OpenAI:** GPT-4o, GPT-4.5 (still selectively available), o1, o3,
  o3-mini, o4-mini, GPT-5 series. Architecture details opaque. o-series
  models are reasoning-trained; thinking-budget exposed via
  reasoning-effort levels.
- **Anthropic:** Claude 3.5 Sonnet (now Claude 3.7+), Claude 4, Claude
  Sonnet 4.5; extended-thinking mode exposed in API. Architecture
  details opaque. Circuit-tracing work on Haiku
  `[lindsey2025-circuit-tracing]` confirms decoder-only transformer
  family.
- **Google DeepMind:** Gemini 2.5 / 3.x series. Sparse MoE, 1M+
  context, native multimodal, configurable `thinkingBudget`. TPU v5p.
  `[gemini2-5]`.
- **Meta:** Llama 3 (405B/70B/8B), Llama 4 (MoE + iRoPE + early-fusion
  multimodal). FP8 pre-training; 256k → 10M context extrapolation.
  `[meta-llama3, meta-llama4]`.
- **DeepSeek:** V2 (MLA), V3 (MoE + MTP + FP8), V3.1, R1 (RLVR
  reasoning). Open-weights frontier; 671B-A37B at competitive cost.
  `[deepseek-v2, deepseek-v3, deepseek-r1]`.
- **Alibaba:** Qwen 2.5, Qwen 3 (MoE + thinking-mode toggle). Strong
  open-weights leaderboard presence. `[qwen2-5, qwen3]`.

Architectural commonalities at the frontier (2026):
- Decoder-only transformer base, RoPE or RoPE-derivative position
  encoding.
- GQA or MLA attention (KV-cache compression).
- SwiGLU FFN.
- Pre-LN or peri-LN normalization.
- MoE for the largest scale; dense for inference-cost-sensitive
  smaller scales.
- FP8 pre-training where the training stack supports it.
- Reasoning-mode variants for hard-task workflows.

Architectural divergences:
- Position-encoding: standard RoPE vs. iRoPE (Llama 4) vs. NoPE
  hybrids vs. Differential Transformer.
- Attention: GQA (Llama, Mistral, Qwen) vs. MLA (DeepSeek) vs. NSA
  (DeepSeek-followon).
- Routing: top-2 (Mixtral) vs. fine-grained + shared (DeepSeekMoE)
  vs. no-shared global-batch (Qwen 3).

## 6. The Goodhart pressure

[INTUITION] As benchmark saturation pushed MMLU, GPQA, and AIME 2024
to ceiling, the field's *measurement* of frontier scaling became the
bottleneck before scaling itself did. The eval crisis (contamination,
benchmark errors, training-on-test) and the response (FrontierMath,
HLE, MathArena, ARC-AGI-2) is itself part of the scaling-frontier
story — see `kb/notes/evaluation/eval-methodology.md` and
`kb/notes/evaluation/reasoning-benchmarks.md`. A "compute-optimal"
recipe is only as informative as the loss/accuracy proxy you're
optimizing for.

## 7. Open questions at the frontier

- **What's the right joint $(C_{\text{train}}, C_{\text{test}})$
  scaling law?** No clean theoretical model exists for the multi-axis
  surface. Empirical trade-offs are documented (Snell §6) but not
  captured in a single closed-form law.
- **Does over-training past Chinchilla generalize beyond English?**
  Multilingual / code-heavy mixes shift the optimum; how far is open.
- **Is RLVR a separate scaling axis or just better post-training?**
  DeepSeek-R1's GRPO + verifiable rewards qualitatively shifted what
  reasoning ability is achievable. Whether this is a "fifth axis" or
  reducible to better data + more inference is debated.
  `[CONTRADICTION]` — see `kb/notes/post-training/rlvr-and-grpo.md`
  (Yue 2025 vs DeepSeek 2025 disagreement).
- **What displaces MMLU / GPQA / AIME at the top of the eval ladder?**
  No consensus replacement that has both adoption and stability has
  emerged as of mid-2026.
- **How much of the apparent capability gain at frontier comes from
  scaling vs. from post-training?** Llama 3 base model on its own
  scores well; instruction-tuned + RL'd Llama 3 scores much better.
  Disentangling these contributions is methodologically hard.

## 8. See also

- `kb/notes/scaling/chinchilla.md` — the training-compute optimum
  this topic frames itself against.
- `kb/notes/scaling/kaplan-laws.md` — the historical baseline.
- `kb/notes/scaling/inference-time-compute-scaling.md` — the second
  axis that has reshaped the frontier.
- `kb/notes/scaling/mu-transfer.md` — the HP-transfer technology that
  makes frontier-scale tuning tractable.
- `kb/notes/training/mixed-precision-and-stability.md` — FP8 training
  and the precision axis.
- `kb/notes/post-training/rlvr-and-grpo.md` — the RL pipeline that
  creates reasoning-trained frontier models.
- `kb/notes/inference/quantization.md` — how the precision axis lands
  at deployment time.
