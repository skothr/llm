---
topic: evaluation/eval-methodology
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - liang2022-helm
  - wang2024-mmlu-pro
  - contamination-survey-2025
  - lm-eval-harness
secondary_sources:
  - mmlu-redux-2024  # Polo et al., arXiv 2406.04127
  - matharena-2025
  - jimenez2024-swebench
related_topics:
  - evaluation/knowledge-benchmarks
  - evaluation/reasoning-benchmarks
  - evaluation/agentic-benchmarks
  - alignment/safety-evaluation
---

# Evaluation methodology

This note covers the **methodological scaffolding** that sits under
every benchmark in the `evaluation/` area: how scores are produced,
what failure modes affect comparability across models, what
contamination is and how it's measured, what HELM-style multi-metric
frameworks add, and what conventions the open-source eval harness
(EleutherAI lm-evaluation-harness) standardizes.

The other notes in this area
(`kb/notes/evaluation/reasoning-benchmarks.md`,
`kb/notes/evaluation/agentic-benchmarks.md`) describe **what each
benchmark measures**; this note describes **what governs whether
the score actually reflects that.**

## 1. Formal framing — the score is a noisy random variable

Treat a benchmark score as a sample from a distribution

$$\hat{S}(\theta, \mathcal{B}, \pi, \tau) \in [0, 1]$$

where

| Symbol | Meaning |
|---|---|
| $\theta$ | the model under evaluation |
| $\mathcal{B}$ | the benchmark items (corpus + verifier) |
| $\pi$ | the **prompt formatting protocol** (system prompt, few-shot examples, answer-extraction template) |
| $\tau$ | the **decoding parameters** (temperature, top-p, max tokens, sampling vs. greedy) |

A reported score $\hat{S}$ is a single point estimate. The
methodological project is to bound the variance of $\hat{S}$ across
$(\pi, \tau)$ and to detect when $\mathcal{B}$ has been compromised
by training-set contamination.

Three failure modes break the comparability of $\hat{S}$ across
papers/labs:

1. **Prompt-format sensitivity** — $\hat{S}(\theta, \mathcal{B}, \pi_1, \tau)$
   and $\hat{S}(\theta, \mathcal{B}, \pi_2, \tau)$ can differ by
   4–5 percentage points on MMLU under reasonable $\pi_1, \pi_2$
   `[wang2024-mmlu-pro §sec-delta; kb/excerpts/wang2024-mmlu-pro#sec-delta]`.
2. **Contamination** — items in $\mathcal{B}$ are in $\theta$'s
   training set, biasing $\hat{S}$ upward by an amount that depends
   on memorization strength
   `[contamination-survey-2025 §abstract; kb/excerpts/contamination-survey-2025#abstract]`.
3. **Verifier mis-specification** — $V$ in $\mathcal{B}$ accepts
   wrong answers (MMLU's 6.49% baseline error rate) or rejects
   correct ones (string-match too strict). MMLU-Redux is the canonical
   audit `[FORUM-SIGNAL: arXiv 2406.04127, Phase 1 sweep §eval-methodology]`.

## 2. Mechanism — how each failure mode propagates

### 2.1 Prompt-format sensitivity — the 4–5% noise floor

A benchmark item like an MMLU question can be presented to the model
in many ways:

- **Loglikelihood scoring:** compute $\log P(\text{"A"})$, $\log P(\text{"B"})$,
  $\log P(\text{"C"})$, $\log P(\text{"D"})$ and pick the argmax.
  This is the lm-evaluation-harness default for MMLU
  `[lm-eval-harness §sec-request-types; kb/excerpts/lm-eval-harness#sec-request-types]`.
- **Generation scoring:** generate text and parse "the answer is
  X." This is what closed-API models (OpenAI, Anthropic, Google)
  must do because logprobs are not exposed.
- **Letter-only vs. full-answer:** "A" vs. "A) Photosynthesis" —
  different prior distributions over the model's tokens.
- **System-prompt presence:** "You are a helpful assistant" vs.
  no system prompt; "Think step by step" vs. direct answer.
- **Few-shot vs. zero-shot:** 0 / 5 / 25 examples in context.

Each of these moves $\hat{S}$ by ~1–4% on MMLU. Combined effects can
exceed 10% — enough to reorder leaderboard rankings entirely. MMLU-Pro
explicitly tested 24 prompt variations and reports prompt-induced
variance dropping from 4–5% (MMLU) to ~2% (MMLU-Pro)
`[wang2024-mmlu-pro §sec-delta; kb/excerpts/wang2024-mmlu-pro#sec-delta]`.

The **answer-letter prior** is one specific mechanism. A 4-way MCQ
gives every option a uniform 25% prior under random guessing, but
*pre-trained models do not have uniform priors over A/B/C/D* — they
have a learned bias from training-data distribution of multiple-
choice questions. Letter-bias can favor "A" or "C" by several
percentage points. Going from 4-way to 10-way MCQ (MMLU-Pro) dilutes
this effect because each individual letter contributes less to the
score.

### 2.2 Contamination — the upward bias

Contamination is the scenario where benchmark items, or close paraphrases,
appear in $\theta$'s training corpus. Standard taxonomy (synthesized
from Phase 1 sweep §eval-contamination + the contamination-survey-2025
abstract):

- **Input contamination** — the benchmark question text appears
  verbatim or near-verbatim in training data.
- **Label contamination** — the gold answer string appears in
  training data, even if the question doesn't (still memorization-
  exploitable for short-answer benchmarks).
- **Distribution contamination** — items from the same
  benchmark family are in training data (e.g., AIME 2023 problems
  are in training, helping with AIME 2024 because the style
  generalizes).
- **Indirect contamination** — solutions to the benchmark items
  appear (StackExchange, OEIS, Reddit post-mortems).
- **Generative contamination** — synthetic data generated *from*
  the benchmark (e.g., AIME-style problem augmentation) is in
  training.

The contamination-survey-2025 categorizes the field's response into
**static** vs. **dynamic** evaluation
`[contamination-survey-2025 §sec-scope; kb/excerpts/contamination-survey-2025#sec-scope]`:

- **Static methods** improve a fixed benchmark: Goodhart-resistant
  question selection (GPQA's Google-proof construction), n-gram-
  overlap detection at training time (Brown et al. 2020 GPT-3
  paper §C used this), and decontamination filters in pre-training
  pipelines (LLaMA 3, Phi-4 explicitly document this).
- **Dynamic methods** are contamination-immune by construction:
  MathArena appends fresh competition problems post-cutoff
  `[FORUM-SIGNAL: matharena.ai; arXiv 2505.23281]`; LiveCodeBench
  appends post-cutoff competitive programming problems;
  GAIA's gated leaderboard delays gold-answer leakage
  `[mialon2023-gaia §sec-leaderboard; kb/excerpts/mialon2023-gaia#sec-leaderboard]`.

[CONTRADICTION] The contamination-survey-2025 abstract
`[contamination-survey-2025 §sec-gap]` calls out a specific gap:
"the lack of standardized criteria for evaluating dynamic
benchmarks." A dynamic benchmark is contamination-immune **today**;
whether it's well-calibrated, comparable across model releases, and
free of selection bias in the problem stream is not yet
methodologically established.

### 2.3 Verifier mis-specification — MMLU-Redux as the canonical audit

The "Are We Done with MMLU?" / MMLU-Redux paper (Polo et al.,
arXiv 2406.04127) audited 5,700 MMLU questions and found

> 6.49% of MMLU questions contain errors

`[FORUM-SIGNAL: confirmed by Phase 1 sweep §eval-methodology]`. The
distribution of errors is heavy-tailed:

> 57% of the analysed questions in the Virology subset contain errors

Per-domain error rate in the Virology subset is so high that the
sub-score is essentially noise. The aggregate MMLU score is robust
to ~6.5% error, but **leaderboard rankings can flip** when scores are
within a few percentage points — and frontier models routinely score
within that band of one another. The MMLU-Redux finding is that
several reported model rankings are unstable to this correction.

The general mechanism: if $V$ has error rate $\epsilon_V$, then for
two models with true scores $S^\star_1 \approx S^\star_2$, the
observed scores $\hat{S}_1, \hat{S}_2$ each carry ~$\epsilon_V$
noise that is **positively correlated** with the model that has more
training-data agreement with the **wrong** gold answers — the wrong
gold answers preferentially come from the same training corpora the
models were trained on. So contamination and verifier-error
**compound**: a model that has memorized the wrong-but-popular
answers can outscore a model with better true reasoning.

## 3. The HELM multi-metric framework — what's in scope beyond accuracy

HELM `[liang2022-helm §abstract; kb/excerpts/liang2022-helm#abstract]`
makes the methodological argument that single-axis benchmarks
(accuracy alone) systematically miss the failure modes that matter
for deployment. The seven HELM axes
`[liang2022-helm §sec-metrics; kb/excerpts/liang2022-helm#sec-metrics]`:

| Axis | Definition | Why it matters |
|---|---|---|
| **Accuracy** | task-specific (exact match, F1, BLEU) | the obvious one |
| **Calibration** | Brier / ECE between predicted prob and empirical accuracy | overconfident-but-wrong is worse than uncertain-and-wrong; matters for refusal/abstention deployment |
| **Robustness** | accuracy delta under perturbed prompts (typos, paraphrase) | predicts in-the-wild behavior under input noise |
| **Fairness** | accuracy gap across demographic-subgroup-conditioned prompts | required for many deployment contexts; legal compliance |
| **Bias** | demographic representation in open-ended generation | reputational / legal risk |
| **Toxicity** | rate of toxic generations under Perspective-API or similar | safety eval overlap |
| **Efficiency** | wall-clock and energy / inference call | cost / sustainability axis |

The structural claim in HELM is that a benchmark instance
$\mathcal{B}$ should ship a **metric bundle**, not a single metric
`[liang2022-helm §sec-metrics]`. This propagated only partially: 2024
leaderboards (HF Open LLM, Artificial Analysis, lmsys) are still
single-metric-dominant, but per-axis breakdowns are increasingly
common in tech reports (LLaMA 3, Gemma 3, etc.).

## 4. Open-source eval discipline — lm-evaluation-harness

EleutherAI's lm-evaluation-harness `[lm-eval-harness §sec-framing; kb/excerpts/lm-eval-harness#sec-framing]`
is the de-facto open-source standard. Three structural choices that
matter:

1. **Tasks are YAML configs**, not Python scripts. This makes the
   prompt template and answer-extraction protocol an artifact that
   can be diffed across versions. When a benchmark's "score"
   changes between model releases, the diff isolates whether the
   prompt or the model changed.

2. **Three request-type primitives** `[lm-eval-harness §sec-request-types]`:
   `loglikelihood`, `loglikelihood_rolling`, `generate_until`. Every
   benchmark reduces to one of these. This API surface is also why
   closed-API models (which expose only `generate_until`) score
   differently from open-weight models (which expose all three) on
   the same MMLU items — different request types can produce
   different scores even on the same data.

3. **Open LLM Leaderboard backend** `[lm-eval-harness §sec-adoption]`
   — the harness is the actual scoring infrastructure for HF's
   public leaderboard. This standardizes the prompt format for all
   models on the leaderboard, eliminating one major source of
   cross-model noise (at the cost of fixing one specific format
   choice).

The methodological gap: closed-lab evals (OpenAI, Anthropic, Google)
do **not** use the harness. They have their own internal harnesses
with their own prompt templates. This means a number from a
closed-lab tech report on MMLU is **not directly comparable** to a
number from the same model run through lm-evaluation-harness; the
delta is typically a few percentage points. This is the **methodological
asymmetry** that frustrates open-vs-closed comparisons.

## 5. Variants and lineage

### 5.1 Contamination-detection methods

| Method | Year | Idea | Strength | Weakness |
|---|---|---|---|---|
| **N-gram overlap** | Brown et al. 2020 (GPT-3 §C) | flag training docs containing benchmark n-grams ≥ 13 tokens | cheap; standard in tech reports | misses paraphrase contamination |
| **Embedding nearest-neighbor** | various 2022+ | flag training docs near benchmark items in embedding space | catches paraphrases | tunable threshold; many false positives |
| **Min-K% Prob** | Shi et al. 2024 | for a candidate item, compute $\sum_{i \in K\%-lowest} \log P(x_i)$; high → memorized | model-internal signal; doesn't need training corpus | needs API access to logprobs |
| **DCR** (arXiv 2509) | EMNLP 2025 | formal contamination quantification combining static + dynamic signals | unified framework | very recent; not yet field-standard |

`[FORUM-SIGNAL: contamination-survey-2025 references; Phase 1 sweep
§eval-methodology]`

### 5.2 The dynamic-benchmark family

| Benchmark | Domain | Append rule | Adoption |
|---|---|---|---|
| **MathArena** | math competitions | post-cutoff problem stream | active (~50 models, 7 competitions, 162 problems) `[FORUM-SIGNAL: matharena.ai]` |
| **LiveCodeBench** | competitive programming | post-cutoff CodeForces, LeetCode, AtCoder | active; widely cited in coding-model tech reports |
| **GAIA gated leaderboard** | tool-use tasks | gold answers hidden server-side | active `[mialon2023-gaia §sec-leaderboard]` |
| **SWE-bench Verified** | coding agents | static, but Docker-pinned for reproducibility | active; not strictly dynamic |
| **HLE leaderboard** | broad academic | static, but new frontier-difficulty target | active `[phan2025-hle §abstract]` |

[INTUITION] Dynamic benchmarking is a **partial answer** to
contamination. It moves the failure mode from "training data
includes the benchmark" to "training data cutoff is recent enough
that the dynamic stream hasn't been retrained-on." A model with a
2025-12 cutoff and a 2026-01 dynamic-benchmark-append is uncontaminated
*for that specific append*. But the next month's append faces the
same problem if the model retrains. The net effect is that dynamic
benchmarks decay in clean-signal value as model release frequency
increases.

## 6. Intuitions and analogies

[ANALOGY] Benchmark contamination is to evaluation as **Goodhart's
Law** is to KPIs in any organization: any metric that becomes a
target stops being a good metric. The mechanism in evaluation is
specifically that benchmark items leak into training corpora,
which is structurally the same as employees discovering which KPI
the manager checks. The analogy returns to canonical form: contamination
is the realization of $\hat{S}(\theta, \mathcal{B}, \pi, \tau)$
having a positive bias term whenever $\mathcal{B} \subseteq$
training-data of $\theta$.

[ANALOGY] HELM is to single-metric leaderboards as a **car-safety
crash-test rating** is to top speed. The single number (top speed,
MMLU score) is a useful summary; the multi-axis rating (HELM, NHTSA
star ratings across rollover, side-impact, frontal-impact, etc.) is
what a buyer actually needs. The analogy returns to canonical
form: HELM's claim is that $\hat{S}$ should be a vector
$(\hat{S}_{\text{acc}}, \hat{S}_{\text{cal}}, \hat{S}_{\text{rob}}, \dots)$
not a scalar `[liang2022-helm §sec-metrics]`.

[INTUITION] **Reproducibility ≠ comparability.** lm-evaluation-harness
gives reproducibility (anyone can re-run any task and get the same
number for the same model). It does **not** give comparability
across closed-lab tech reports, because closed labs use different
harnesses. Reproducibility is necessary but not sufficient.

[INTUITION] The right mental model for "MMLU saturation" is **noise-
floor approach**, not "100% capability achieved." MMLU-Redux's
6.49% error rate is a **hard upper bound** on perfectly-calibrated
true score (a perfect model still scores ≤93.5% on the
gold-labeled MMLU because some gold labels are wrong). Frontier
models at 90%+ are within MMLU's verifier noise floor — the
benchmark literally cannot tell them apart, regardless of training
contamination. This is why follow-on benchmarks (MMLU-Pro, GPQA,
HLE) are not optional — they're the only path to keeping a usable
signal.

## 7. Frontier and open questions (as of 2026-05)

- **Standard for evaluating dynamic benchmarks.** As called out by
  the contamination-survey-2025 abstract
  `[contamination-survey-2025 §sec-gap]`, no community standard
  exists. Open question.
- **Cross-harness comparability.** Closed-lab tech-report numbers
  vs. lm-evaluation-harness numbers differ by a few percentage
  points, but the magnitude is not consistently characterized.
  An open-vs-closed double-blind harness study would be valuable.
- **LLM-as-judge in eval pipelines.** AlpacaEval, MT-Bench, and
  Arena-Hard use LLM-as-judge. This introduces a contamination axis
  where the same model class scores its own outputs generously
  (`Elo` rankings on lmsys arena are partly mediated by GPT-4-as-judge,
  for example). Replacing with human judges is expensive; replacing
  with rule-based checkers reduces benchmark scope. Open question.
- **Compute-budget axis on leaderboards.** With reasoning models
  spending 10–100× more inference tokens, comparing pass@1 on a
  reasoning model vs. base model conflates capability with cost.
  Reporting `(score, $/sample, tokens/sample)` is the proposal but
  not yet adopted on most leaderboards.
- **Goodhart's Law on benchmark suites.** As benchmarks consolidate
  (Open LLM Leaderboard's MMLU + GSM8K + ARC + HellaSwag + TruthfulQA
  + Winogrande was the canonical 2023 set; replaced by MMLU-Pro,
  IFEval, GPQA, BigBenchHard, MATH-500, MuSR in 2024), labs optimize
  *the suite* — checkpoint selection on the leaderboard score,
  not on a held-out true distribution of capabilities. The escalation
  from suite-1 to suite-2 happens roughly every 18–24 months.
- **Verifier audits as ongoing infrastructure.** MMLU-Redux happened
  in 2024 — three years after MMLU's release. A faster cadence of
  benchmark audits would help, but no organization has stepped into
  the role. Epoch AI's FrontierMath is the closest to a "validated-by-
  construction" model.

## 8. See also

- `kb/notes/evaluation/reasoning-benchmarks.md` — covers the actual
  reasoning benchmarks (MMLU, MMLU-Pro, GPQA, FrontierMath, HLE,
  ARC-AGI-2). Cites this note for contamination/methodology.
- `kb/notes/evaluation/agentic-benchmarks.md` — agent-side
  benchmarks (SWE-bench, GAIA, OSWorld, tau-bench). Methodological
  issues there are largely the same; the verifier-misspec axis is
  load-bearing for agent evals because end-state checkers can be
  buggy.
- `kb/notes/evaluation/knowledge-benchmarks.md` — the saturated-
  benchmark family (MMLU and its descendants); methodology in this
  note explains why MMLU saturated.
- `kb/notes/alignment/safety-evaluation.md` — HarmBench,
  JailbreakBench, METR autonomy suite. The same contamination /
  prompt-sensitivity issues apply to safety evals; in some cases
  worse, because safety evals often have small item counts.
- `kb/notes/scaling/inference-time-compute-scaling.md` — context
  for why "compute-budget axis" is relevant on benchmark
  leaderboards.
