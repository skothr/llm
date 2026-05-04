---
topic: evaluation/reasoning-benchmarks
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - glazer2024-frontiermath
  - rein2023-gpqa
  - phan2025-hle
  - wang2024-mmlu-pro
secondary_sources:
  - hendrycks2021-mmlu  # for the saturation backstory
  - matharena-2025
  - arc-agi-2-2025
related_topics:
  - evaluation/eval-methodology
  - evaluation/agentic-benchmarks
  - reasoning/test-time-compute
  - reasoning/chain-of-thought
  - scaling/inference-time-compute-scaling
---

# Reasoning benchmarks

This note covers benchmarks designed to measure **multi-step reasoning
capability** in language models — math problems, science Q&A, abstract
puzzles — at the post-2024 frontier. The primary methodological pivot
captured in this note is the **stale-prior callout** flagged by the
Phase 1 sweep: the benchmarks that 2022–2023 LLM papers used
canonically (MMLU, MATH, AIME 2024, even GPQA) are saturated or
contaminated for frontier reasoning models, and the eval landscape
has fragmented into FrontierMath, HLE, ARC-AGI-2, and MathArena as
the current discriminative benchmarks.

This note focuses on **what the benchmark is** and **what it
measures**; the **how to score them well** material (CoT scaffolds,
self-consistency, best-of-N, tool use) is in
`kb/notes/reasoning/test-time-compute.md` and
`kb/notes/reasoning/chain-of-thought.md`.

## 1. The reasoning-benchmark protocol — formal specification

A reasoning benchmark instance is a tuple

$$\mathcal{B} = (\rho, A, V)$$

where

| Symbol | Meaning |
|---|---|
| $\rho$ | the problem statement (math problem, MC question, ARC grid pair) |
| $A$ | the answer-format constraint (numeric integer, single letter, $n \times m$ grid) |
| $V$ | verifier $V(\hat{a}, a^\star) \in \{0, 1\}$ checking model answer $\hat{a}$ against gold $a^\star$ |

Compared to the agentic-benchmark spec
(`kb/notes/evaluation/agentic-benchmarks.md` §1), there is **no
environment** — the model produces a single response, $V$ checks it,
done. The **methodological challenge** is that $V$ must be both
(i) tight enough to reject lucky guesses, and (ii) automated enough
to run at scale without LLM-as-judge.

The four canonical 2024–2025 reasoning benchmarks below exhibit
distinct strategies for keeping $V$ tight:

- **MMLU-Pro:** 10-way MCQ, $V$ = string equality
  `[wang2024-mmlu-pro §abstract; kb/excerpts/wang2024-mmlu-pro#abstract]`.
- **GPQA Diamond:** 4-way MCQ on Google-proof PhD questions, $V$ =
  string equality, ceiling raised by domain difficulty rather than
  distractor count `[rein2023-gpqa §abstract; kb/excerpts/rein2023-gpqa#abstract]`.
- **FrontierMath:** open-ended, with a single canonical numerical /
  symbolic answer parsed by a domain-specific checker `[glazer2024-frontiermath §sec-verification; kb/excerpts/glazer2024-frontiermath#sec-verification]`.
- **HLE (Humanity's Last Exam):** mixed MCQ + short-answer; $V$
  combines string match for short-answer items and choice match for
  MCQ items `[phan2025-hle §abstract; kb/excerpts/phan2025-hle#abstract]`.

## 2. Mechanism — how each benchmark is constructed

### 2.1 MMLU-Pro — the de-noised, deepened MMLU

MMLU-Pro `[wang2024-mmlu-pro §abstract; kb/excerpts/wang2024-mmlu-pro#abstract]`
takes the original MMLU's 57-subject coverage and applies three
filters:

1. Drop "trivial and noisy questions" — the MMLU-Redux audit
   (Polo et al. 2024) found 6.49% baseline error rate
   `[FORUM-SIGNAL: also confirmed by `Are We Done with MMLU?` arXiv 2406.04127]`.
2. Expand from 4-way to **10-way MCQ** — random-guess ceiling drops
   $25\% \to 10\%$, expanding the dynamic range above the noise
   floor.
3. Add reasoning-heavy items where Chain-of-Thought helps measurably
   (the diagnostic CoT-vs-direct gap; see §2.4 below).

Resulting size: 12,032 questions across 14 super-categories. At
launch (June 2024), top frontier models scored 16–33 percentage
points lower than their MMLU score
`[wang2024-mmlu-pro §sec-delta; kb/excerpts/wang2024-mmlu-pro#sec-delta]`.
Prompt-format sensitivity dropped from 4–5% to ~2%, partly because
10-way reduces the **answer-letter-prior effect** (where models
have a learned preference for option B over D, etc.).

### 2.2 GPQA Diamond — the Google-proof PhD subset

GPQA `[rein2023-gpqa §abstract; kb/excerpts/rein2023-gpqa#abstract]`
is 448 questions written by domain-expert PhDs in biology, physics,
and chemistry. The "Google-proof" property is operationalized:

- **Expert PhD baseline:** ~65% accuracy.
- **Non-expert validator with web access (avg 30+ min/question):**
  34% — only marginally above the 25% random baseline.

`[rein2023-gpqa §abstract]` This 34% is the load-bearing
methodological number — it's what justifies "the question can't be
answered just by Googling."

**GPQA Diamond** is the 198-question subset where (i) both expert
annotators agree on the gold answer, and (ii) at least one non-expert
validator with web access got it wrong
`[rein2023-gpqa §sec-diamond; kb/excerpts/rein2023-gpqa#sec-diamond]`.
Diamond is the version reported on every leaderboard.

[CONTRADICTION] At launch (Nov 2023), GPT-4 scored 39% — below the
PhD expert baseline of 65%. As of Feb 2026, frontier reasoning
models exceed 90% on Diamond
`[FORUM-SIGNAL: Artificial Analysis leaderboard, Feb 2026; Phase 1
sweep §3 stale-prior table]`. **GPQA Diamond is now effectively
saturated at the frontier** — 24 percentage points above the human
PhD baseline. The original paper's framing of "below human expert"
is the stale prior that this note's existence corrects.

### 2.3 FrontierMath — research-grade math

FrontierMath `[glazer2024-frontiermath §abstract; kb/excerpts/glazer2024-frontiermath#abstract]`
is constructed by 60+ working mathematicians (verified PhD or
equivalent), each authoring problems in their domain and
cross-vetting peers' problems for solvability and uniqueness.
Domains span computational number theory, real analysis, algebraic
geometry, category theory.

The methodologically critical design choice: every problem has a
**single canonical numerical or symbolic answer** that can be parsed
by an automated checker
`[glazer2024-frontiermath §sec-verification; kb/excerpts/glazer2024-frontiermath#sec-verification]`.
Problems requiring proof or open-ended argument are excluded — this
is what enables contamination-resistant scoring without LLM-as-judge.

The headline at launch (Nov 2024): **<2% solve rate** for the
strongest contemporary models
`[glazer2024-frontiermath §sec-launch-result; kb/excerpts/glazer2024-frontiermath#sec-launch-result]`.

The difficulty calibration:

> Problems demand multiple hours to days of effort from domain
> specialists.

`[glazer2024-frontiermath §sec-difficulty; kb/excerpts/glazer2024-frontiermath#sec-difficulty]`

This places FrontierMath one tier above competition math (AIME, Putnam,
IMO) on the human-effort axis. The contrast with AIME 2024 (now
~83–100% on frontier reasoning models per Artificial Analysis) is the
key design rationale: FrontierMath was constructed precisely as
post-saturation backup
`[glazer2024-frontiermath §sec-saturation; kb/excerpts/glazer2024-frontiermath#sec-saturation]`.

### 2.4 HLE (Humanity's Last Exam) — broad-domain frontier

HLE `[phan2025-hle §abstract; kb/excerpts/phan2025-hle#abstract]`
is 2,500 questions across "dozens of subjects, including mathematics,
humanities, and the natural sciences." The framing is explicit:

> LLMs now achieve over 90% accuracy on popular benchmarks like
> MMLU, limiting informed measurement of state-of-the-art LLM
> capabilities.

`[phan2025-hle §abstract]` HLE is positioned as the **last
closed-ended academic benchmark** — a deliberate frontier-difficulty
target for the post-MMLU era. Questions are vetted by global subject-
matter experts; format mixes MCQ and short-answer; some questions are
multimodal (image inputs).

The verifier discipline:

> a known solution that is unambiguous and easily verifiable, but
> cannot be quickly answered via internet retrieval.

`[phan2025-hle §abstract]`

### 2.5 ARC-AGI-2 — abstract grid-puzzle generalization

ARC-AGI-2 (François Chollet / ARC Prize, released March 2025) is the
successor to ARC-AGI-1. Each task is an abstract grid puzzle: a
small set of `(input_grid, output_grid)` training pairs from which
the test-taker must infer a transformation rule, then apply it to
held-out test inputs.

`[FORUM-SIGNAL: ARC Prize official site; Phase 1 sweep §scaling-frontier]`
ARC-AGI-1 was effectively solved by o3 (>85%) in late 2024.
ARC-AGI-2 was constructed specifically to be harder along the
abstract-rule-induction axis. At launch (early 2025):

- **Top commercial model:** ~37.6%
- **Refinement-loop systems** (program synthesis with iterative
  test-feedback): ~54%
- **Human baseline (representative average):** ~95%
- **Unconstrained test-time-compute systems** (ARC Prize 2025
  competition winners): ~54% with very large compute budgets

`[FORUM-SIGNAL: ARC Prize 2025 Technical Report]`

ARC-AGI's design philosophy is distinct: the verifier is grid-
equality; the inputs are tiny (5×5 to 30×30 grids); the difficulty
is in inferring **the right abstract rule** from very few examples.
This makes ARC-AGI a different shape of "reasoning" benchmark
than FrontierMath or HLE — it tests inductive generalization with
minimal world knowledge, vs. compositional knowledge with deep
domain expertise.

## 3. Variants and lineage

### 3.1 The post-2024 reasoning-benchmark map

| Benchmark | Year | Format | Items | Verifier | Headline launch result | Saturation status (2026-05) |
|---|---|---|---|---|---|---|
| **MMLU** | 2021 | 4-way MCQ | 15,908 | string-eq | GPT-3: ~44% | **saturated** at >90% for frontier `[hendrycks2021-mmlu]` |
| **MATH** | 2021 | numeric/LaTeX free-form | 12,500 | symbolic match | early models <10% | partially saturated (~80%+) for reasoning models |
| **AIME 2024** | 2024 | numeric integer | 30 | int-eq | early reasoning models ~50% | **contaminated** (Phase 1 sweep §3 table; 83–100% on frontier) |
| **MMLU-Pro** | 2024 | 10-way MCQ | 12,032 | string-eq | top model ~72% | actively discriminative `[wang2024-mmlu-pro]` |
| **GPQA Diamond** | 2023 | 4-way MCQ | 198 | string-eq | GPT-4 39%, PhD 65% | **saturated at frontier** (>90%) `[rein2023-gpqa]` |
| **FrontierMath** | Nov 2024 | numeric/symbolic | "hundreds" | domain checker | <2% | actively discriminative; expected useful through 2026+ `[glazer2024-frontiermath]` |
| **HLE** | Jan 2025 | mixed MCQ/short-answer | 2,500 | string-eq | (low; explicit calibration target) | actively discriminative; ~65% top score Phase 1 sweep `[phan2025-hle]` |
| **ARC-AGI-2** | Mar 2025 | grid input → grid output | (held) | grid-eq | top commercial ~37.6% | actively discriminative |
| **MathArena** | 2025 | competition problems | rolling | numeric/symbolic | n/a (live append) | dynamic-by-design — saturation impossible by construction |

### 3.2 Lineage thread: from MMLU to HLE

The most recent reasoning-benchmark line (knowledge density):

1. **MMLU** (2021) — 57 subjects, undergraduate-level, MCQ,
   retrieval-heavy.
2. **MMLU-Pro** (2024) — same subject coverage, harder items,
   reasoning-heavy, 10-way MCQ.
3. **MMLU-Redux** (2024) — error-corrected MMLU; not a new
   benchmark but a methodological audit; documented 6.49%
   baseline error rate `[FORUM-SIGNAL: arXiv 2406.04127]`.
4. **GPQA / GPQA Diamond** (2023) — narrower domain (bio/phys/chem),
   PhD-level difficulty, Google-proof.
5. **HLE** (2025) — broader subject scope, frontier-calibrated
   difficulty, mixed format. Explicit successor framing.

### 3.3 Lineage thread: from MATH to FrontierMath

The math-reasoning line (problem-solving depth):

1. **MATH** (Hendrycks et al. 2021) — 12,500 competition-style
   problems; most middle-/high-school level.
2. **GSM8K** (Cobbe et al. 2021) — 8,500 grade-school word problems;
   foundational for CoT papers.
3. **AIME 2024** — 30 American Invitational Mathematics Exam
   problems; integer-answer; high-school competition tier.
4. **AIME 2025** — newer set; less contamination than 2024 but
   still aggressively scored.
5. **PutnamBench, OlympiadBench** (2024) — undergraduate-olympiad
   tier, multilingual.
6. **FrontierMath** (Nov 2024) — research-mathematician tier; 1–3
   orders of magnitude harder per item than competition math.
7. **MathArena** (2025) — dynamic, post-cutoff competition problems;
   contamination-immune by construction.

[INTUITION] The math line is monotone in human-effort-per-problem:
GSM8K ≈ 1–5 min, MATH ≈ 5–30 min, AIME ≈ 10–30 min, FrontierMath ≈
hours to days. Each generation has a corresponding LLM-effort
threshold beyond which the benchmark saturates within ~12–18 months.
FrontierMath bought the field about 18 months of headroom from its
Nov 2024 launch.

### 3.4 Lineage thread: ARC

1. **ARC** (Chollet 2019) — abstract grid puzzles; the original
   "fluid intelligence" frame.
2. **ARC-AGI** (rebranded; ARC Prize era 2024) — same task, public
   leaderboard.
3. **ARC-AGI-1 solved by o3** (late 2024) at >85% — the
   inference-time-compute paradigm match.
4. **ARC-AGI-2** (March 2025) — designed to resist
   inference-time-compute brute force; top commercial ~37.6%.

`[FORUM-SIGNAL: arcprize.org; Phase 1 sweep §scaling-frontier]`

## 4. Intuitions and analogies

[ANALOGY] The post-2024 reasoning-benchmark landscape is **like a
ratchet of standardized tests**: as each level (SAT → GRE → LSAT
→ qualifying exam → research) saturates, the eval community
introduces a new, harder level to measure progress. The analogy
returns to canonical form: each $\mathcal{B}_t$ has the same
structural form $(\rho, A, V)$; what changes is the difficulty
distribution of $\rho$ and the human-effort calibration that fixes
the saturation timeline. MMLU is "standardized HS test"; GPQA Diamond
is "PhD qualifier exam"; FrontierMath is "research seminar problem
set"; HLE is "the most distilled cross-domain hard set anyone could
write."

[INTUITION] **Random-guess ceiling matters** in a way that is often
underappreciated. A 4-way MCQ with random-guess at 25% has very
low dynamic range above noise: a 5% improvement could be ~20%
relative gain or ~5% absolute, but the noise floor is high. A
10-way MCQ with random-guess at 10% has more room to discriminate.
A free-form numeric answer with no plausible random guesser has
**no random-guess floor at all** — the benchmark can in principle
discriminate down to 0%. This is one reason FrontierMath's <2%
launch number is informative in a way that "MMLU 30%" never could
have been: the noise floor on FrontierMath is essentially zero.

[INTUITION] **Goodhart pressure** scales with benchmark visibility.
MMLU's saturation was not just frontier capability gain — it was
also driven by per-checkpoint optimization: labs ablate
training-data composition and post-training recipes against MMLU
because investors want to see the headline number move. FrontierMath
is partly designed to be **expensive to optimize against** (no
simple training-data recipe will move the score), restoring some
robustness to Goodharting. See `kb/notes/evaluation/eval-methodology.md`
§Goodhart for the broader picture.

[CONTRADICTION] Whether reasoning models (`o1`, `R1`, `o3`,
Claude with extended thinking) deserve to be ranked on the same
leaderboards as non-reasoning models is contested. Reasoning models
spend 10–100× more tokens at inference, which is a different cost
profile. **MMLU-Pro reports CoT-vs-direct as a diagnostic**
`[wang2024-mmlu-pro §sec-cot; kb/excerpts/wang2024-mmlu-pro#sec-cot]`,
which partly normalizes for this; FrontierMath does not. Whether to
add a "compute-budget" axis to leaderboards is an open community
question.

## 5. Frontier and open questions (as of 2026-05)

- **What replaces FrontierMath?** As frontier models pass 50% on
  FrontierMath, the next tier ("open math conjecture, partial
  proof acceptable") is hard to construct and even harder to
  auto-verify. The MathArena dynamic-append model may be the only
  scalable approach.
- **HLE saturation curve.** As of Phase 1 sweep, top model on HLE is
  ~65% (Claude Mythos Preview). At MMLU-style trajectories this
  saturates by 2027. Whether HLE is the genuine ceiling or a way-
  station is not yet clear.
- **ARC-AGI-2 vs program-synthesis**.
  Refinement-loop systems (54%) outperform pure-LLM systems (~37.6%)
  on ARC-AGI-2 — i.e., the test-time-search systems beat the test-
  time-CoT systems. Whether this generalizes is the live question
  at the boundary of `kb/notes/scaling/inference-time-compute-scaling.md`.
- **Contamination and AIME 2024.** AIME 2024 scores are now treated
  as unreliable in the open-model community
  `[FORUM-SIGNAL: Phase 1 sweep §3]`. AIME 2025 is being re-evaluated.
  MathArena's stream-of-fresh-problems is the principled response
  but adoption is partial.
- **"Reasoning model" vs "base model" comparability.** No standard
  yet. Some leaderboards (Artificial Analysis, lmsys) report both
  conditions; others (HF Open LLM Leaderboard) historically did not.
- **HLE multimodality fraction.** The fraction of HLE that is
  text-only vs multimodal is under-documented in the abstract; this
  matters for which models can fairly compete.
- **Saturated benchmarks as in-distribution probes.** A saturated
  benchmark is not useless — it remains a fast in-distribution
  smoke test for "did we break the model with this fine-tune?" The
  MMLU number stays in nearly every model card for this reason.
  But its value as a frontier-discriminator is gone.

## 6. See also

- `kb/notes/evaluation/eval-methodology.md` — contamination, prompt
  sensitivity, lm-eval-harness conventions; the methodological
  scaffolding under all of this.
- `kb/notes/evaluation/agentic-benchmarks.md` — the agent-side
  complement; SWE-bench / GAIA / OSWorld; covers the
  long-horizon-task axis that this note's static benchmarks don't.
- `kb/notes/reasoning/test-time-compute.md` — Snell et al., DeepSeek-R1,
  o-series; the inference-time-compute story that explains why so
  many of these benchmarks saturate so fast.
- `kb/notes/reasoning/chain-of-thought.md` — Wei et al. 2022; the
  CoT-vs-direct gap is the diagnostic flagged by MMLU-Pro for
  "real reasoning."
- `kb/notes/scaling/inference-time-compute-scaling.md` — how reasoning
  budgets translate to benchmark scores; relevant to the "compute-
  budget axis" open question above.
