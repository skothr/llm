---
topic: evaluation/knowledge-benchmarks
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - hendrycks2021-mmlu
  - wang2024-mmlu-pro
  - mmlu-redux-2024
  - rein2023-gpqa
secondary_sources:
  - liang2022-helm                  # broader eval methodology
  - contamination-survey-2025
  - phan2025-hle                    # the post-MMLU "broad academic" successor frame
related_topics:
  - evaluation/reasoning-benchmarks
  - evaluation/eval-methodology
  - evaluation/agentic-benchmarks
  - alignment/safety-evaluation
---

# Knowledge benchmarks

A **knowledge benchmark** measures how much factual / declarative
content a language model has internalized that it can correctly
retrieve under prompted question-answering. The canonical instance is
**MMLU** (Hendrycks et al. 2021), and the lineage from MMLU forward
is the central story of this note: how the field's de-facto knowledge
benchmark went from *the* discriminating eval (2021) to a saturated
in-distribution smoke test (2025), and what replaced it (MMLU-Pro,
GPQA, HLE).

This note focuses on the **knowledge-retrieval slice** of evaluation
— questions whose correct answer is mostly retrieval rather than
multi-step reasoning. Reasoning-heavy successors (MMLU-Pro, GPQA,
HLE, FrontierMath) are deep-dived in
`kb/notes/evaluation/reasoning-benchmarks.md`; the methodological
scaffolding (contamination, prompt sensitivity, verifier audits) is
in `kb/notes/evaluation/eval-methodology.md`.

## 1. Formal framing

A knowledge-benchmark instance has the same skeleton as the
reasoning-benchmark instance from
`kb/notes/evaluation/reasoning-benchmarks.md` §1:

$$\mathcal{B} = (\rho, A, V)$$

| Symbol | Meaning |
|---|---|
| $\rho$ | the question (a multiple-choice item, a short-answer prompt) |
| $A$ | the answer-format constraint (single letter, span of text, integer) |
| $V$ | verifier $V(\hat{a}, a^\star) \in \{0, 1\}$ checking model answer against gold |

What distinguishes a knowledge benchmark from a reasoning benchmark
is **what kind of $\rho$**:

- **Knowledge-dominant $\rho$:** answerable in a single step given
  the relevant fact. CoT does not improve performance on these items
  because there is no chain to compute.
- **Reasoning-dominant $\rho$:** requires multi-step inference from
  given inputs. CoT-vs-direct gap is the diagnostic signal that the
  benchmark is reasoning-heavy.

This is the diagnostic the MMLU-Pro paper makes load-bearing
`[wang2024-mmlu-pro §sec-cot;
kb/excerpts/wang2024-mmlu-pro#sec-cot]`:

> Chain-of-Thought reasoning showed marked improvements on MMLU-Pro,
> contrasting with the original MMLU where direct answering performed
> comparably—indicating the benchmark includes substantially more
> complex reasoning questions.

By this diagnostic, MMLU is *primarily* a knowledge benchmark (CoT ≈
direct), while MMLU-Pro is a *knowledge + reasoning* hybrid. The
note's primary subject — pure knowledge — has progressively been
absorbed into reasoning-augmented hybrids; there is no major
post-2024 standalone knowledge benchmark.

## 2. Mechanism — how MMLU is constructed and scored

### 2.1 MMLU (Hendrycks et al. 2021)

**MMLU** is 15,908 multiple-choice questions across **57 subjects**:
elementary mathematics, US history, computer science, law, medicine,
ethics, etc. `[hendrycks2021-mmlu §abstract]`. Each question has 4
options; the verifier is exact-match on the chosen letter. Subject
breakdown groups items into four broad areas (humanities, social
sciences, STEM, other-professional).

The dataset is **manually curated** from publicly available exam-bank
sources (AP exams, GRE, MCAT, professional licensing exams). Items
are filtered for clarity and uniqueness of correct answer. The
intended difficulty target was *zero-shot college-graduate baseline*
— the test was designed so that broadly-educated humans could pass
with cumulative subject coverage.

**Random-guess baseline:** 25% (4-way MCQ).
**Human expert baseline (cited in the paper):** ~89.8% averaged
across subjects.
**GPT-3 175B at launch (Hendrycks 2021):** ~43.9%, near random in
many subjects.

The headline 2021 finding: *language models retrieve broadly but
shallowly* — strong on humanities, weaker on calculations and
specialized professional knowledge.

### 2.2 The lm-evaluation-harness MMLU recipe

The de-facto open-source scoring of MMLU uses EleutherAI's
`lm-evaluation-harness` `[lm-eval-harness §sec-request-types;
kb/excerpts/lm-eval-harness#sec-request-types]`. The standardized
recipe:

1. **Prompt template**: `{subject_intro}\n\nQuestion: {question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:`
2. **5-shot in-context** with subject-matched dev examples.
3. **Loglikelihood scoring**: compute $\log P(\text{"A"})$,
   $\log P(\text{"B"})$, $\log P(\text{"C"})$, $\log P(\text{"D"})$
   directly from the model's logits and pick argmax.
4. **No CoT** — direct-answer scoring.

This is what makes MMLU numbers **reproducible across open-weights
models** (anyone running the same harness on the same model gets the
same number) but *not* directly comparable to closed-API tech-report
numbers, because closed labs use generation-style scoring with their
own prompt templates. The cross-harness gap is typically a few
percentage points
`[kb/notes/evaluation/eval-methodology.md §4]`.

### 2.3 The 2024 verifier audit (MMLU-Redux)

Polo et al. 2024 (*Are We Done with MMLU?*, arXiv 2406.04127) audit
5,700 MMLU questions against expert annotators
`[mmlu-redux-2024 §abstract]`:

- **6.49% baseline error rate** across all audited subjects.
- **Heavy tail:** 57% error rate in the Virology subset; the
  sub-score there is essentially noise.
- The errors include malformed questions, wrong gold answers,
  ambiguous wording, and items where multiple options are defensibly
  correct.

The implication for headline scores: a perfect oracle scores at most
~93.5% on gold-labeled MMLU. Frontier models at 90%+ are within the
verifier-error band of one another and of the perfect-oracle
ceiling. **Leaderboard rankings inside that band are unstable to
verifier corrections.**

The MMLU-Redux paper is a methodology audit, not a new benchmark; it
publishes the corrections so any subsequent run can use the
audited gold labels. Adoption: partial — the Open LLM Leaderboard
moved to MMLU-Pro for its primary headline, while individual research
papers still use raw MMLU for in-distribution sanity checks.

## 3. The lineage — MMLU and successors

| Benchmark | Year | Format | Items | Random-guess | Status (2026-05) |
|---|---|---|---|---|---|
| **TriviaQA** (Joshi et al. 2017) | 2017 | open-domain QA | 95k | n/a (string-match) | superseded; pre-LLM era |
| **Natural Questions** (Kwiatkowski et al. 2019) | 2019 | open-domain QA | 307k | n/a (span-match) | superseded for frontier; still used in retrieval |
| **OpenBookQA** (Mihaylov et al. 2018) | 2018 | 4-way MCQ + book context | 5,957 | 25% | saturated |
| **ARC-Challenge** (Clark et al. 2018) | 2018 | 4-way MCQ science | 7,787 | 25% | saturated; in old Open LLM Leaderboard suite |
| **MMLU** (Hendrycks et al.) | 2021 | 4-way MCQ × 57 subjects | 15,908 | 25% | **saturated** at >90% on frontier models `[hendrycks2021-mmlu §abstract]` |
| **MMLU-Redux** (Polo et al.) | 2024 | audit of MMLU | 5,700 audited | n/a (audit) | active diagnostic; 6.49% error rate `[mmlu-redux-2024 §abstract]` |
| **MMLU-Pro** (Wang et al.) | 2024 | 10-way MCQ + reasoning | 12,032 | 10% | **actively discriminative** `[wang2024-mmlu-pro §abstract]` |
| **GPQA / GPQA Diamond** (Rein et al.) | 2023 | 4-way MCQ PhD bio/phys/chem | 448 / 198 | 25% | saturated at frontier `[rein2023-gpqa §abstract]` |
| **HLE** (Phan et al.) | 2025 | mixed MCQ / short-answer | 2,500 | varies | actively discriminative `[phan2025-hle §abstract]` |

### 3.1 MMLU → MMLU-Pro: the 2024 update

MMLU-Pro `[wang2024-mmlu-pro §abstract;
kb/excerpts/wang2024-mmlu-pro#abstract]` makes three structural
changes targeting MMLU's saturation failures:

1. **More choices.** 4-way → 10-way MCQ. Random-guess ceiling drops
   $25\% \to 10\%$, expanding dynamic range above the noise floor.
2. **Reasoning-focused.** Adds reasoning-heavy items where CoT helps
   measurably (the diagnostic CoT-vs-direct gap of §1).
3. **De-noised.** "Eliminates trivial and noisy questions from the
   original MMLU" — implicit acknowledgment of MMLU-Redux's findings.

At launch, top frontier models scored 16–33 percentage points lower
than their MMLU score
`[wang2024-mmlu-pro §sec-delta;
kb/excerpts/wang2024-mmlu-pro#sec-delta]`. Prompt-format sensitivity
dropped from 4–5% (MMLU) to ~2% (MMLU-Pro).

The structural innovation of MMLU-Pro that matters for *knowledge*
benchmarks specifically: 10-way MCQ dilutes the **answer-letter prior
effect**. Pre-trained models do not have uniform priors over A/B/C/D
on multiple-choice questions; they have a learned bias from
training-data distribution that can favor one letter by several
percentage points. Going from 4-way to 10-way reduces the
contribution of any one letter to the score.

### 3.2 MMLU → GPQA: the 2023 PhD-level pivot

GPQA `[rein2023-gpqa §abstract;
kb/excerpts/rein2023-gpqa#abstract]` is 448 questions written by
domain-expert PhDs in biology, physics, and chemistry. Construction
emphasizes **Google-proofness**:

- **Expert PhD baseline:** ~65% accuracy.
- **Non-expert validator with web access (avg 30+ min/question):**
  34% — only marginally above the 25% random baseline.

The 34% non-expert-with-web baseline is the load-bearing methodological
number — it operationalizes "the question can't be answered just by
Googling." This pivots the benchmark from "broad-coverage knowledge"
to "narrow-domain *deep* knowledge with reasoning required."

GPQA Diamond (198 items where two experts agree and at least one
non-expert validator with web access got it wrong) is the canonical
leaderboard slice. Saturation status as of 2026: **frontier models
exceed 90% on Diamond** `[FORUM-SIGNAL: Phase 1 sweep §3
stale-prior table]`, ~24 points above the human PhD baseline.

### 3.3 MMLU → HLE: the 2025 cross-domain frontier

HLE (Humanity's Last Exam) `[phan2025-hle §abstract;
kb/excerpts/phan2025-hle#abstract]` is positioned as the **last
closed-ended academic benchmark** — a deliberate frontier-difficulty
target for the post-MMLU era:

> LLMs now achieve over 90% accuracy on popular benchmarks like
> MMLU, limiting informed measurement of state-of-the-art LLM
> capabilities.

2,500 questions vetted by global subject-matter experts; mixed MCQ +
short-answer; some multimodal. The verifier discipline:

> a known solution that is unambiguous and easily verifiable, but
> cannot be quickly answered via internet retrieval.
> `[phan2025-hle §abstract]`

Treated in full detail in
`kb/notes/evaluation/reasoning-benchmarks.md`.

### 3.4 The TruthfulQA divergence

Lin et al.'s **TruthfulQA** (2022, ACL 2022) is positioned as a
knowledge benchmark for *truthful* knowledge under adversarial prompts
— questions where models are tempted to give a popular but incorrect
answer (urban legends, common misconceptions). Score: percentage of
questions where the model gives a truthful answer.

TruthfulQA has historically been included in the Open LLM Leaderboard
suite, but its scope is more aligned with **alignment evaluation**
than knowledge evaluation; the relevant axis is *honesty under
pressure*, not *broad coverage*. It's covered tangentially in
`kb/notes/alignment/sycophancy.md` and Marks & Tegmark 2023's truth
direction work `[marks-tegmark-2023-truth §abstract;
kb/excerpts/marks-tegmark-2023-truth#abstract]`.

## 4. The "knowledge benchmark" status problem

By 2025, *standalone* knowledge benchmarks (where retrieval, not
reasoning, dominates the score) have largely lost their
discrimination function at the frontier. Two diagnostics:

1. **CoT-vs-direct gap = 0** on MMLU at the frontier means
   knowledge-only items are not the bottleneck — frontier models
   know the facts; the 5–10% they miss are not retrieval failures
   but verifier-error / contamination / ambiguity items
   `[wang2024-mmlu-pro §sec-cot;
   kb/excerpts/wang2024-mmlu-pro#sec-cot]`.
2. **Saturation within the verifier-error band.** With MMLU-Redux's
   6.49% baseline error rate `[mmlu-redux-2024 §abstract]`, a
   perfect oracle scores at most ~93.5%. Top frontier models cluster
   at 90–92% — *inside* the verifier-error band, where score
   differences are not statistically meaningful.

The field's response has been to fold knowledge into
reasoning-augmented hybrids (MMLU-Pro, GPQA, HLE) rather than build a
new pure-knowledge benchmark. **There is no widely-adopted
post-MMLU pure-knowledge benchmark as of 2026-05** — and arguably the
field has decided this is the wrong target. The relevant frontier
question is *how knowledge is used*, not *which facts are stored*.

## 5. Variants and lineage — the Open LLM Leaderboard transition

The 2023 Open LLM Leaderboard suite — MMLU, GSM8K, ARC-Challenge,
HellaSwag, TruthfulQA, Winogrande — was the canonical "general
capabilities" suite. By mid-2024, every component except possibly
TruthfulQA was saturated above the verifier-error band for frontier
open-weight models. The 2024 transition to:

- **MMLU-Pro** (knowledge + reasoning hybrid)
- **IFEval** (instruction following)
- **GPQA Diamond** (PhD-level science)
- **BigBench-Hard / MuSR** (multi-step reasoning)
- **MATH-500** (math problem solving)

removed the saturation bottleneck for ~12–24 months. The pattern
is: **leaderboard suites have a 12–24 month lifespan before frontier
models saturate them**, after which the suite is rebuilt one tier
harder. See `kb/notes/evaluation/eval-methodology.md` §7.

## 6. Intuitions and analogies

[ANALOGY] MMLU is to LLM evaluation as the **SAT is to college
admissions**: a broad, standardized, multi-subject MCQ that became
*the* number to optimize, then saturated as test-takers improved
faster than the test could iterate. The analogy returns to canonical
form: each $\mathcal{B}_t$ has the same structural form
$(\rho, A, V)$; what changes across MMLU → MMLU-Pro → GPQA → HLE is
the difficulty distribution of $\rho$ and the human-effort
calibration. Each tier extends the field's measurable headroom by
12–24 months.

[INTUITION] **Why "knowledge" specifically saturated.** Two
contributing factors:

1. **Pre-training scale.** A 70B-parameter model trained on
   ~10–15T tokens has internalized essentially all undergraduate-level
   facts in widely-discussed domains. There is no factual headroom
   for retrieval-only items — the relevant gap is reasoning, not
   knowledge.
2. **Contamination pressure.** MMLU items are widely circulated;
   close paraphrases appear in training corpora; even strict
   decontamination filters miss indirect contamination
   (StackExchange explanations of MMLU questions, Reddit
   post-mortems). The contamination-survey-2025 catalog
   `[contamination-survey-2025 §sec-scope;
   kb/excerpts/contamination-survey-2025#sec-scope]` makes this
   explicit. Saturation accelerates as benchmark visibility grows.

[INTUITION] **Saturated benchmarks are not useless** — they remain
fast in-distribution smoke tests for "did we break the model with
this fine-tune?" The MMLU number stays in nearly every model card
for this reason, even though it no longer discriminates the
frontier. As an in-distribution health check: a 5-percentage-point
drop in MMLU after a fine-tune is a strong signal something broke,
even if a 5-point gap between two frontier models is noise.

[ANALOGY] **Knowledge-vs-reasoning is the ML-eval analog of crystallized
vs. fluid intelligence** in psychometrics. Crystallized intelligence
is what you've internalized; fluid intelligence is how you reason
*with* what you've internalized. The MMLU lineage measures
crystallized; MMLU-Pro / GPQA / FrontierMath measures fluid (with
crystallized as a precondition). This is why 2024-2025 benchmarks
fold knowledge into reasoning hybrids: at the frontier,
crystallized capacity is no longer the bottleneck.

## 7. Frontier and open questions (as of 2026-05)

- **Is there a real "post-MMLU pure-knowledge benchmark"?** Or has
  the field correctly concluded that pure-knowledge benchmarks at
  the frontier are uninformative? No major proposal currently
  contests the latter view — but specific knowledge domains
  (medicine, law, code documentation) might still warrant
  domain-specific saturation tracking.
- **Closed-book vs. open-book / RAG knowledge benchmarks.** If
  retrieval-augmented LLMs become the deployment-default, the
  benchmark question shifts from "what does the model know" to
  "what can the model find given retrieval." The Natural Questions
  / TriviaQA lineage is the substrate for this; no MMLU-equivalent
  benchmark for the RAG era has emerged.
- **Audit cadence.** MMLU-Redux happened in 2024 — three years
  after MMLU's release. A faster cadence of benchmark audits would
  help, but no organization has stepped into the role. The audit
  finding (6.49% errors, 57% in Virology) is itself a methodology
  warning that propagated only partially through the field
  `[mmlu-redux-2024 §abstract]`.
- **Knowledge benchmarks for non-English and non-academic domains.**
  C-Eval (Chinese), MMLU-XL (multilingual), etc. saturate later
  than English MMLU but follow the same trajectory. Whether
  language-specific knowledge gaps remain an actionable
  discriminator at the frontier is open.
- **The TruthfulQA category drift.** TruthfulQA started as a
  knowledge benchmark and became an alignment / honesty benchmark.
  Whether the "factuality under pressure" axis is a knowledge
  category at all, or a behavioral category, is a definitional
  question that affects how the result is reported and adopted.
- **Multimodal knowledge.** With vision-language models (LLaVA,
  Gemini, Claude with image input, GPT-4V), MMLU-style benchmarks
  that include diagrams, charts, and images extend the knowledge
  surface. MMLU itself is text-only; the multimodal generalization
  is an active design space (cf. MMMU, ScienceQA).

## 8. See also

- `kb/notes/evaluation/reasoning-benchmarks.md` — full deep-dive on
  MMLU-Pro, GPQA, HLE, FrontierMath, ARC-AGI-2, MathArena. Most of
  the post-MMLU lineage is treated there.
- `kb/notes/evaluation/eval-methodology.md` — contamination,
  prompt-format sensitivity, lm-eval-harness conventions, MMLU-Redux
  audit details. The methodological scaffolding under all knowledge
  benchmarks.
- `kb/notes/evaluation/agentic-benchmarks.md` — the agent-side
  complement; SWE-bench, GAIA, OSWorld, τ-bench. Agentic benchmarks
  test *applied* knowledge in tool-using contexts.
- `kb/notes/alignment/safety-evaluation.md` — HarmBench /
  JailbreakBench / METR; behavioral safety evals share the
  saturation / contamination / verifier-mis-spec failure modes
  that knowledge benchmarks face.
- `kb/notes/scaling/scaling-laws.md` — the upstream scaling story
  that drives knowledge-benchmark saturation: at 10–15T tokens of
  training, the factual headroom on undergraduate-level material
  approaches the verifier-error ceiling.
