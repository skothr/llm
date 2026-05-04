---
topic: reasoning/chain-of-thought
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - wei2022-cot
  - kojima2022
  - wang2022-self-consistency
secondary_sources:
  - arcuschin2025-cot-wild
  - shen2025-faithcot-bench
  - mehta2026-faithfulness-scaling
related_topics:
  - reasoning/test-time-compute
  - reasoning/process-supervision
  - reasoning/reasoning-training
  - reasoning/inference-time-search
  - interpretability/mechanistic-interpretability
---

# Chain-of-thought reasoning

Chain-of-thought (CoT) is a prompting and decoding pattern in which the
model produces an intermediate sequence of reasoning steps before
emitting a final answer. CoT is the historical bridge between the
pre-2022 era — where LLMs answered in one shot — and the 2025 "thinking
model" paradigm in which long internal reasoning traces are a first-class
training and inference target. This note covers the prompting paradigm.
The architectural and training counterparts (long-form internal reasoning
trained via RL) are in `kb/notes/reasoning/reasoning-training.md` and
`kb/notes/architecture/reasoning-architectures.md`.

> **PDF-fetch caveat (2026-05-04):** primary PDFs for `wei2022-cot`,
> `kojima2022`, and `wang2022-self-consistency` were not downloaded in
> this Phase 2 pass — see the area report. Section anchors in citations
> below refer to the canonical arXiv-version sectioning of each paper;
> verbatim excerpt files are skeleton-only and must be populated during
> the excerpt-population follow-up before any LaTeX paper-series text
> propagates these claims.

## 1. Formal definition

Let $x$ be an input prompt and $y$ the desired final answer. **Direct
prompting** asks the model to compute $p_\theta(y \mid x)$ and decode the
answer in one step. **Chain-of-thought prompting** introduces an
intermediate variable $z$, the *reasoning trace*, and decodes:

$$z \sim p_\theta(\cdot \mid x), \qquad y \sim p_\theta(\cdot \mid x, z) \tag{1}$$

In practice $z$ and $y$ are produced jointly by autoregressive decoding
of a single sequence $z \cdot y$, where $z$ is a natural-language
explanation that ends in a terminator (e.g., "The answer is …") and $y$
is extracted post-hoc by a regex or parser
`[wei2022-cot §3.1]`.

The **CoT prompt** is the few-shot demonstration that elicits this
behavior: each in-context example contains its own reasoning trace
$z_i$ followed by its answer $y_i$, so that the model learns from the
demonstrations to emit traces before answers `[wei2022-cot §1, §3.1]`.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $x$ | input question / prompt |
| $z$ | reasoning trace (natural-language intermediate steps) |
| $y$ | final answer |
| $p_\theta$ | the LLM with parameters $\theta$ |
| $K$ | number of few-shot exemplars in the CoT prompt |
| $N$ | number of independent samples (for self-consistency) |

## 2. Mechanism — protocols

### 2.1 Few-shot CoT (Wei 2022)

The original recipe `[wei2022-cot §3.1]`:

1. Construct a prompt of $K$ exemplars $\{(x_i, z_i, y_i)\}_{i=1}^K$
   followed by a query $x_*$. Each $z_i$ is a hand-written natural-
   language reasoning trace.
2. Decode greedily (or at low temperature) from $p_\theta(\cdot \mid
   \text{prompt}, x_*)$ until a stop pattern.
3. Parse the suffix matching "The answer is $\langle \cdot \rangle$" to
   recover $y_*$.

Wei et al. report that the gain from CoT over direct prompting is **an
emergent property of scale**: on GSM8K, CoT prompting helps PaLM 540B
substantially but does not help (and can hurt) sub-100B models
`[wei2022-cot §3.3 fig.4, §6]`. This finding seeded the 2022–2024 line of
work distinguishing "emergent" capabilities from compute-monotonic ones.

### 2.2 Zero-shot CoT (Kojima 2022)

A surprising follow-on: appending the literal string "Let's think step by
step" to the prompt elicits CoT-style reasoning **without any
exemplars** `[kojima2022 §3]`. The protocol is two-stage:

1. **Reasoning extraction:** prompt with $x_* + $"Let's think step by
   step" and decode $z_*$.
2. **Answer extraction:** prompt with $x_* + z_* + $"Therefore, the
   answer is" and decode $y_*$.

Zero-shot CoT recovers a substantial fraction of the few-shot CoT gain
on arithmetic and symbolic reasoning benchmarks at the GPT-3-instruct
and InstructGPT scale `[kojima2022 §4]`. This is the canonical evidence
that the CoT-eliciting effect lives in pretraining + instruction-tuning,
not in the few-shot exemplars per se.

### 2.3 Self-consistency (Wang 2022)

Decoding **multiple** reasoning traces and **majority-voting** over the
extracted answers improves accuracy at fixed model size
`[wang2022-self-consistency §3]`:

1. Sample $N$ traces $z^{(1)}, \ldots, z^{(N)}$ at temperature $T > 0$.
2. Extract answers $y^{(1)}, \ldots, y^{(N)}$.
3. Return $\hat y = \arg\max_y \sum_{n=1}^N \mathbb{1}[y^{(n)} = y]$.

The rationale: many distinct reasoning paths can land on the same
correct answer, so majority over diverse traces concentrates probability
mass on the correct $y$ even when individual traces err. This is the
simplest **test-time-compute** strategy and the baseline against which
search and verifier-guided methods (best-of-$N$, MCTS, PRM-reranking)
are compared in `kb/notes/reasoning/test-time-compute.md` and
`kb/notes/reasoning/inference-time-search.md`.

[INTUITION] Self-consistency works when the answer space is
low-cardinality (e.g., integers, multiple-choice) and the model's per-
sample error is independent enough that majority vote reduces variance.
The canonical form is the standard ensemble-variance-reduction argument,
not anything specific to language. The 2025 work on search and PRMs
generalises this to weighted votes where the weight is a learned
verifier score (see §3.3 below).

### 2.4 Decoding configurations that shape $z$

Several knobs interact with the basic protocol:

- **Temperature $T$.** Self-consistency requires $T > 0$ to obtain
  distinct samples; greedy CoT (Wei) uses $T = 0$ with a single path.
- **Trace length budget.** Imposing a maximum trace length truncates
  $z$. The 2025 *budget forcing* technique (s1, see
  `kb/notes/reasoning/test-time-compute.md`) deliberately extends $z$
  by appending continuation tokens like "Wait" to elicit more
  reasoning.
- **Stop tokens / answer template.** Robust answer extraction relies on
  a stable end-of-trace pattern. Trace-format drift (model forgets to
  emit "The answer is") is the dominant failure mode of CoT pipelines
  in production.

## 3. Variants and lineage

### 3.1 Lineage table

| Variant | Year | Core change | Status as of 2026 |
|---|---|---|---|
| Few-shot CoT (Wei 2022) | 2022 | Add hand-written reasoning traces to in-context examples | Foundational; rarely used directly at frontier (subsumed by trained reasoning) |
| Zero-shot CoT (Kojima 2022) | 2022 | Trigger phrase "Let's think step by step", no exemplars | Useful baseline; weak vs trained reasoning models |
| Self-consistency (Wang 2022) | 2022 | Sample $N$ traces, majority-vote answers | Still a strong, cheap baseline; subsumed by best-of-$N$ + verifier |
| Tree-of-Thoughts (Yao 2023) | 2023 | Branch and prune over partial reasoning states | Cooled in 2025 — RL-trained long CoT largely supplanted manual search structures |
| Graph-of-Thoughts (Besta 2023) | 2023 | Allow merging of branches | Niche; subsumed by adaptive variants |
| Adaptive Graph-of-Thoughts (AGoT 2025, [arXiv 2502.05078]) | 2025 | Single framework that adaptively chooses chain / tree / graph | Open question whether this beats trained long-CoT |
| Long-CoT trained models (R1 2025, o1 2024) | 2024–25 | $z$ is generated *inside* model at inference; trained via RL | Dominant frontier paradigm; see `reasoning-training.md` |

### 3.2 Distinguishing CoT-as-prompting vs CoT-as-output

A frequent source of confusion in the 2025+ literature: "CoT" is used
both for the *prompting technique* (this note) and for the
*emitted output behaviour* of reasoning-trained models (e.g., R1's
`<think>...</think>` blocks before final answers). The mechanisms are
distinct:

- Prompted CoT — behaviour induced by prompt formatting. Works on any
  base model; quality is heavily prompt-engineering-dependent
  `[wei2022-cot §3.3]`.
- Trained CoT — behaviour learned via SFT on long-trace data and / or
  RL with verifiable rewards. The trace is generated unconditionally;
  no prompt scaffolding needed. Used by o1, R1, QwQ, Gemini 2.5 thinking
  `[deepseek-r1 §2]`.

The two interact: trained CoT models still respond to additional
prompting nudges (system instructions, "think harder"), but the floor
behaviour is set by training, not prompting.

## 4. Frontier: faithfulness

The dominant 2025 question is **faithfulness**: does the trace $z$
causally drive the answer $y$, or is $y$ chosen by some other route and
$z$ generated as post-hoc rationalization?

### 4.1 Empirical signal

Recent measurements `[arcuschin2025-cot-wild, arXiv 2503.08679]`,
`[mehta2026-faithfulness-scaling, arXiv 2601.06423]` find a striking
asymmetry:

- Production non-thinking models (GPT-4o-mini, Haiku 3.5) exhibit
  post-hoc rationalization rates on the order of $7\text{–}13\%$.
- Thinking models (Sonnet 3.7 with thinking on, Gemini 2.5 Pro)
  exhibit rates on the order of $0.04\text{–}0.14\%$
  `[mehta2026-faithfulness-scaling §4]`.

[CONTRADICTION] Whether this gap is causally produced by *thinking
training* (long-CoT SFT + RL), by general scale, or by RLVR
post-training cannot be cleanly separated by published 2025–2026
ablations. The 2026 multi-model study controls for many but not all
factors.

### 4.2 Decoupling hypothesis

The Decoupling Hypothesis [SPECULATION] [chen2025-decoupling, arXiv
2505.17406] proposes that, in non-thinking models, the answer-token
distribution is largely set by the prompt's leading semantics, and the
trace is a separately sampled rationalization conditioned on that
distribution. The trace and answer are then *correlated* (both
conditioned on prompt) but not *causally connected*. Trained reasoning
models, by contrast, are alleged to use the trace as a genuine compute
substrate.

This is a single-paper hypothesis as of writing; the controlled
ablations needed to confirm or falsify it have not been published. It
is a productive lens for organising the 2026 faithfulness literature
but should not be treated as established `[chen2025-decoupling]`.

### 4.3 FaithCoT-Bench

A new instance-level benchmark `[shen2025-faithcot-bench, arXiv
2510.04040]` measures, per problem, whether a perturbation to $z$ that
should change $y$ in fact does. The metric is mechanistic — it does not
rely on holistic plausibility scoring of traces — and is now the
standard reference target for faithfulness claims.

## 5. Intuitions

[ANALOGY] The few-shot-CoT prompt is a **demonstration of an internal
monologue**. Wei's exemplars tell the model "in-distribution outputs
include reasoning traces", and the model continues that pattern at
inference. The analogy returns to canonical form in §1: the
demonstration shifts $p_\theta$'s mass toward sequences in which $z$
appears before $y$.

[INTUITION] Self-consistency is **majority-vote ensembling without
training an ensemble**. Each sample is a noisy estimator of the
underlying conditional, and majority over $N$ samples reduces variance.
Crucially, it works only when $y$ has a small support (math answers,
multiple-choice) — for free-form generation the analogue is best-of-$N$
under a verifier, not majority vote.

[INTUITION] Zero-shot CoT works because pretraining text contains
abundant "Let's think step by step…" patterns from instructional and
math-tutorial corpora; the trigger phrase is a *soft retrieval cue*
into that distribution. The mechanism is a learned co-occurrence, not
a discovered reasoning module `[kojima2022 §5]`.

[ANALOGY] Faithfulness is the question of whether a CoT trace is
**reasoning** or **rationalisation**. The sociological analogy — humans
often confabulate plausible justifications for decisions made on other
grounds — is suggestive but not load-bearing here; the canonical
mechanistic question is whether perturbations to $z$ propagate to $y$
under counterfactual decoding `[shen2025-faithcot-bench §3]`.

## 6. Frontier and open questions (as of 2026-05)

- **Is the trained-CoT advantage in faithfulness due to training data,
  RL signal, or scale?** The 2026 cross-model study cannot fully
  separate these. A controlled ablation training one model with and
  without long-CoT SFT at fixed scale would settle it.
- **Does CoT generalise outside math / code / formal reasoning?** Wei
  2022's gains are concentrated on arithmetic, commonsense reasoning,
  and symbolic tasks. Open-domain factual QA, creative tasks, and
  multi-modal reasoning show weaker or contested gains
  `[wei2022-cot §6 limitations]`.
- **Is "more thinking" always better?** [CONTRADICTION] Multiple 2025
  papers report that extending CoT length monotonically degrades
  accuracy past a problem-dependent ceiling on R1-class models
  `[arXiv 2502.12215]`. This contradicts the simple "more compute =
  better" interpretation of test-time scaling.
- **Trace compression and steganography.** A growing concern: trained
  CoT models can compress reasoning into opaque token sequences that
  are no longer human-interpretable, defeating CoT's transparency
  promise. Empirical evidence is fragmentary as of 2026.
- **CoT in agentic loops.** When the trace itself contains tool-call
  decisions, "faithfulness" generalises to "do tool decisions reflect
  actual reasoning content?". This intersects
  `kb/notes/evaluation/agentic-benchmarks.md`.

## 7. See also

- `kb/notes/reasoning/test-time-compute.md` — sampling-, search-, and
  sequential-compute test-time-compute strategies; CoT is the
  substrate they extend.
- `kb/notes/reasoning/process-supervision.md` — verifying individual
  CoT steps with PRMs.
- `kb/notes/reasoning/reasoning-training.md` — RL-trained long CoT;
  o1 / R1 / QwQ lineage.
- `kb/notes/reasoning/inference-time-search.md` — MCTS / beam over
  reasoning steps.
- `kb/notes/architecture/reasoning-architectures.md` — architectural
  side (thinking-token paradigm, hybrid think / non-think modes).
- `kb/notes/interpretability/mechanistic-interpretability.md` — circuits
  for in-context learning that underlie CoT-elicitation effects.
