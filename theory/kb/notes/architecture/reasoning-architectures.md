---
topic: architecture/reasoning-architectures
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - deepseek-r1
  - qwen3
  - meta-llama4
secondary_sources:
  - wei2022-cot
  - snell2024
related_topics:
  - architecture/transformer-overview
  - architecture/multi-token-prediction
  - reasoning/chain-of-thought
  - reasoning/test-time-compute
  - post-training/rlvr-and-grpo
  - scaling/inference-time-compute-scaling
---

# Reasoning architectures

A "reasoning architecture" in 2025–2026 means an LLM whose **inference-
time output protocol** allows extended chain-of-thought generation as a
first-class architectural pattern, plus the training methodology
(usually RLVR) that produces useful reasoning traces. The notion is
simultaneously architectural (special tokens, optional reasoning
prefix), training-protocol (RLVR with verifiable rewards), and
serving-protocol (think-budget controls). This note focuses on the
**architectural** layer — the format, tokens, and forward-pass behavior
that distinguish a reasoning-capable model from a non-reasoning one.

The DeepSeek-R1, Qwen3, and LLaMA 4 PDFs were not accessible during
this Phase 2 pass; specifics below reflect well-established public
reporting. Verify exact token names and special-token handling against
the originals before propagating.

## 1. Formal definition — what makes an architecture "reasoning-capable"

A reasoning architecture is a base Transformer LLM (any of the
attention/MoE/SSM patterns elsewhere in this KB) augmented with **at
least one of**:

1. **Reserved structural tokens** delimiting a reasoning region:
   `<think>...</think>`, `<reasoning>...</reasoning>`, or DeepSeek-R1's
   `<|im_start|>think` / `<|im_start|>answer` segments. The model is
   trained to emit reasoning between the delimiters and the final
   answer outside them. The tokenizer reserves these as **single
   tokens** (not multi-token UTF-8 sequences) so they are
   computationally cheap and serve as crisp parsing landmarks.

2. **Mode toggling** between a "thinking" inference mode and a "non-
   thinking" inference mode, controlled by either:
   - a **system-prompt flag** (Qwen3's `enable_thinking=False`), or
   - a **prefix-token toggle** (the model emits `<think>` to enter
     thinking mode; the user/server can suppress it), or
   - **two distinct model checkpoints** trained from a shared base
     (DeepSeek-V3 → DeepSeek-R1).

3. **Inference-time think-budget control**: a serving-time parameter
   that caps the number of reasoning tokens before forcing an answer.
   Qwen3 reports this as `thinking_budget`; it is a serving construct
   not an architectural one but is supported by the trained model's
   reasoning-token format.

[CONTRADICTION] The community has not converged on a single
nomenclature. Anthropic's "extended thinking" and OpenAI's
"reasoning models" (o1, o3) are functionally similar but architectural
details (whether the reasoning tokens are visible to users, what
special tokens are used, whether think-budget is exposed) differ and
much remains undisclosed. Treat the public Chinese-lab models
(DeepSeek, Qwen) as the primary architecturally-documented examples;
Western frontier labs are largely opaque on this dimension as of
2026-05.

### 1.1 The forward pass — no architectural change

[INTUITION] The reasoning architecture is **not a structural change to
the Transformer block**. The forward pass is identical to the base
LLM. What changes is:

- The training data format includes reasoning traces.
- The tokenizer reserves think/answer delimiters.
- The post-training method (RLVR) uses verifier-graded rewards on the
  final answer to shape the reasoning trace.
- The serving stack respects the special tokens (e.g., a
  thinking-budget cap or a "hide reasoning from user" flag).

So this topic sits at the boundary between architecture, training
(`kb/notes/post-training/rlvr-and-grpo.md`), reasoning protocols
(`kb/notes/reasoning/chain-of-thought.md`,
`kb/notes/reasoning/test-time-compute.md`), and inference-time scaling
(`kb/notes/scaling/inference-time-compute-scaling.md`). It is its own
note because it is the **architectural layer** of the reasoning stack:
which tokens, which mode-toggling pattern, which model-family choice.

## 2. Mechanism — the three production patterns

### 2.1 Two-checkpoint pattern (DeepSeek-V3 → R1)

DeepSeek's January 2025 release is the cleanest public example.
DeepSeek-V3 is a base+SFT non-thinking model; **DeepSeek-R1** is a
distinct checkpoint built on the V3 base via:

1. **Cold-start SFT** on a small set of long-CoT examples to seed
   reasoning behavior.
2. **RL with verifiable rewards (RLVR)** using GRPO: the policy emits
   reasoning + answer; reward is +1 if the verifier (math grader, code
   executor, etc.) accepts the answer, 0 otherwise. The reasoning
   trace is the policy's "free choice" for how to reach correctness.
3. Iteration: collected RL trajectories are filtered for high-reward
   ones, used as new SFT data, and the cycle repeats.

The R1 release also documented a "DeepSeek-R1-Zero" variant: pure RL
from V3-base **without** the cold-start SFT step. R1-Zero develops
reasoning behavior emergently but produces less-readable traces; the
SFT cold start is the readability fix
[deepseek-r1 PDF arXiv:2501.12948 not accessible in this pass —
mechanism reflects the technical-report headline; verify before
LaTeX].

The **architectural layer** is the format: `<think>...</think>` then
final answer. The model is trained end-to-end to emit this structure.

### 2.2 Single-checkpoint hybrid pattern (Qwen3, May 2025)

Qwen3 trains a single checkpoint that can operate in **either** thinking
or non-thinking mode, controlled by a system-prompt flag
`enable_thinking`. The training data mixes thinking traces and direct-
answer examples; the model learns to condition on the flag and emit
either format. The advantage: one set of weights, one deployment, two
serving modes. The disadvantage: the trained model is a compromise
between reasoning specialist and direct-answer specialist; benchmarks
suggest each Qwen3 mode is slightly weaker than a dedicated model would
be.

Qwen3 also exposes a **thinking budget** (`max_thinking_tokens`): the
serving stack interrupts the reasoning trace after $B$ tokens by
injecting `</think>` and forcing the model to commit to an answer.
This is purely architectural in that the model must be trained to
**recover** from a forcibly-truncated reasoning trace — it must not
hallucinate that the truncated trace was a complete one.

### 2.3 Mid-training pattern (LLaMA 4, Jan 2026)

LLaMA 4's reasoning training uses a "mid-training" stage: between
pre-training and final SFT/RL, an extended reasoning-data stage on
long-CoT corpora (some synthetic, distilled from larger reasoning
models). The architectural layer is similar to Qwen3 (single-
checkpoint, mode-flexible) but the recipe is reportedly different.
Public details are sparse [meta-llama4 PDF arXiv:2601.11659 not
accessible].

### 2.4 The reasoning-token economy

A reasoning trace is **expensive**: it adds 100–10000 generated tokens
per query. At serving time, this multiplies the per-query
generation cost by a similar factor. The serving-architecture
implications:

- **KV-cache size grows linearly** with the reasoning trace (the
  trace is just more sequence). MLA's $d_c + d_h^R \approx 9/2 d_h$
  per-token KV cost (`kb/notes/architecture/attention-mechanism.md` §3.1)
  is what makes long reasoning economical.
- **Speculative decoding** of reasoning traces is straightforward
  (the trace is autoregressive text) and is increasingly common in
  serving stacks.
- **Reasoning content can be cached and reused across queries** (e.g.,
  prefix caching of system prompts). Whether the reasoning trace is
  shared across multiple-completion sampling is a serving choice.

## 3. Variants and lineage

| Family | Pattern | Special tokens | Mode toggle | RL method | Public details |
|---|---|---|---|---|---|
| **DeepSeek R1 / V3** | two checkpoints | `<think>` `</think>` | model choice | GRPO + cold-start SFT | high — full tech report |
| **Qwen3** | single checkpoint | `<think>` `</think>` | `enable_thinking` flag | proprietary | high — tech report |
| **LLaMA 4** | mid-training stage | (TBD) | (TBD) | (TBD) | low — limited disclosure |
| **OpenAI o1 / o3 / o4** | (presumed two-checkpoint) | hidden from user | — | proprietary RL (presumed RLVR-family) | very low — no architectural details |
| **Anthropic Claude (extended thinking)** | (single checkpoint, mode flag) | hidden by default; can be exposed | API parameter | proprietary | low |
| **Gemini 2.5 (deep think)** | (single checkpoint, mode flag) | hidden by default | API parameter | proprietary | low |

The Chinese-lab open-weight models (DeepSeek, Qwen) are
architecturally documented; the Western frontier labs disclose almost
nothing structural. As of 2026-05 there is no published comparison of
the two-checkpoint vs single-checkpoint patterns at matched scale.

[CONTRADICTION] Whether the reasoning-token output of o1/o3/o4 is
architecturally "the same kind of thing" as DeepSeek-R1's
`<think>...</think>` is unknown. Some external reverse-engineering
suggests OpenAI's reasoning is generated by a separate decoding pass
or different model, not just bracketed tokens in the same stream. No
primary source confirms either possibility.

### 3.1 The relationship to multi-token prediction (MTP)

DeepSeek-V3 introduced **Multi-Token Prediction** as a separate
architectural feature: the model has multiple language-modeling heads
predicting tokens at offsets $t+1, t+2, \ldots, t+k$. MTP is
orthogonal to reasoning architectures — it's a training-objective
device, not a reasoning-format device — but in DeepSeek-V3/R1 they
appear together. Treated separately in
`kb/notes/architecture/multi-token-prediction.md`.

## 4. Intuitions and analogies

[ANALOGY] A reasoning architecture is **a Transformer that has been
trained to use the residual stream as a scratchpad — written
externally**. Vanilla Transformers can do "implicit" multi-step
computation through layer depth (the residual stream evolves through
$L$ layers); reasoning models add an **explicit scratchpad** in the
form of generated tokens that re-enter the model on the next forward
pass. Rather than $L$ layers of compute per output token, the model
gets up to $B \cdot L$ layers of compute per *answer* token, where $B$
is the reasoning-trace length. The analogy returns to canonical form
via the autoregressive forward pass: the scratchpad tokens are
ordinary autoregressively-generated tokens, with no architectural
change to the forward pass.

[INTUITION] **Why thinking-token formats matter mechanically.** The
think/answer delimiter tokens give the policy a **mode-conditioning
signal** that changes its conditional distribution over the next token.
A well-trained reasoning model emits free-form exploration, false
starts, self-corrections inside `<think>...`, and only after closing
the think block produces the final answer. The boundary tokens are
discrete switches that the model has been trained to respect. Without
the explicit format, the boundary is fuzzy — the model can't clearly
separate exploration from final commit, and the verifier (during RL)
struggles to grade only the answer.

[ANALOGY] **Two-checkpoint vs single-checkpoint as specialist vs
generalist.** [SPECULATION] DeepSeek-R1's separate-from-V3 design is
analogous to having a "math researcher" and a "general assistant" as
distinct people; Qwen3's single-checkpoint hybrid is analogous to one
person who can switch between modes. The single-person design has
amortization advantages but suffers from interference: training the
person to be good at thinking can degrade non-thinking direct-answer
quality. Whether this degradation is fundamental or training-recipe-
fixable is open.

[CONTRADICTION] **What the scratchpad actually does.** Two views:

- *Computational view:* the reasoning trace is **genuine extra
  compute** — the model is performing serial inference steps it
  couldn't do in a single forward pass. RLVR shapes this compute to
  be useful.
- *Surface view:* the reasoning trace is **exploration of the policy's
  output distribution**, with the verifier-shaped reward acting as a
  filter on which exploratory paths get reinforced. The "compute" is
  not new — it's the same forward-pass capacity, just exposed to
  feedback.

Both views are partially true. The literature does not yet have a
clean way to separate them. See
`kb/notes/scaling/inference-time-compute-scaling.md` for the empirical
side; see `kb/notes/post-training/rlvr-and-grpo.md` for the training
side.

## 5. Frontier and open questions (as of 2026-05)

- **Two vs one checkpoint at matched compute.** No published head-to-
  head. DeepSeek argues the two-checkpoint cleanly separates concerns
  without quality loss; Qwen3 argues the single-checkpoint amortizes.
- **Reasoning-token efficiency.** RLVR-trained reasoning traces are
  often verbose; "compress reasoning" (s1, "rstar-Math",
  inference-budget-aware training) is an active research direction.
  See `kb/notes/reasoning/test-time-compute.md`.
- **Reasoning generalization beyond verifiable domains.** RLVR rewards
  require a verifier (math, code, formal logic). Whether reasoning
  capabilities transfer to non-verifiable domains (dialogue, creative
  writing, safety-judgment) is an open empirical question. The "RLVR
  limits" paper (arXiv 2406.05978-style; track via
  `kb/notes/post-training/rlvr-and-grpo.md`) argues for fundamental
  limits; reasoning-architecture proponents argue otherwise.
- **Hidden reasoning vs visible reasoning safety.** OpenAI hides o1
  reasoning from the user; DeepSeek-R1 exposes it. Anthropic exposes
  by default with a "monitor reasoning for misalignment" framing.
  Safety implications (whether visible reasoning enables better
  oversight or worse — i.e., users may trust visible reasoning
  uncritically) are contested.
- **Architectural support for very long reasoning.** Reasoning traces
  in math benchmarks routinely exceed 32K tokens. The interaction
  with long-context engineering (NSA, iRoPE, YaRN) is direct: a
  reasoning-capable LLM is implicitly a long-context LLM during
  reasoning. Models trained at short context that "extend with YaRN"
  may degrade specifically on long reasoning.
- **Multi-token prediction × reasoning interaction.** DeepSeek-V3/R1
  uses both. Whether MTP helps reasoning specifically (by amortizing
  next-$k$-token planning) or is an independent benefit is not
  resolved.

## 6. See also

- `kb/notes/reasoning/chain-of-thought.md` — the prompting-level CoT
  origins; reasoning architectures internalize CoT into the trained
  model.
- `kb/notes/reasoning/test-time-compute.md` — best-of-$n$, beam search,
  budget-forcing; the inference-time-search axis adjacent to
  reasoning-architecture choice.
- `kb/notes/post-training/rlvr-and-grpo.md` — the RL methodology that
  trains reasoning models. GRPO and DAPO are the primary algorithms.
- `kb/notes/scaling/inference-time-compute-scaling.md` — Snell 2024
  scaling laws over reasoning-token budget.
- `kb/notes/architecture/multi-token-prediction.md` — MTP, the
  orthogonal architectural feature DeepSeek-V3 stacks alongside
  reasoning.
- `kb/notes/architecture/long-context.md` — long reasoning is a
  long-context regime; the techniques compose.
