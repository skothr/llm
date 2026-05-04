---
topic: evaluation/agentic-benchmarks
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - jimenez2024-swebench
  - mialon2023-gaia
  - xie2024-osworld
  - yao2024-tau-bench
secondary_sources:
  - swe-bench-verified  # OpenAI Aug 2024 follow-up
  - cybench2024
  - agentbench2023
related_topics:
  - evaluation/eval-methodology
  - evaluation/reasoning-benchmarks
  - reasoning/test-time-compute
  - alignment/safety-evaluation
---

# Agentic benchmarks

The benchmarks in this note evaluate LLMs as **agents** —
i.e., systems that autonomously plan, take actions in an environment
(tools, code, OS, or browser), observe results, and replan.
This is methodologically distinct from question-answering benchmarks
(where the model's "action" is a single text response checked against
a gold answer) and from coding benchmarks where solutions are checked
against fixed test suites without any environment interaction.

The post-2024 frontier shift in evaluation runs through this category:
as MMLU/HumanEval saturated, the open question moved from "does the
model know things" to "can the model do things autonomously over long
horizons." That shift is the reason this note exists as its own leaf,
not as a sub-section of `evaluation/knowledge-benchmarks.md`.

## 1. The agentic-benchmark protocol — formal specification

An agentic benchmark instance is a tuple

$$\mathcal{B} = (s_0, \mathcal{A}, \mathcal{T}, \rho, g, V)$$

where

| Symbol | Meaning |
|---|---|
| $s_0$ | initial state of the environment (codebase, OS image, DB tables) |
| $\mathcal{A}$ | action space (tool calls, shell commands, mouse/keyboard events) |
| $\mathcal{T}$ | environment transition function $s_{t+1} = \mathcal{T}(s_t, a_t)$ |
| $\rho$ | task description (natural-language issue, user query) |
| $g$ | goal specification (gold end-state, gold answer string, regression test bundle) |
| $V$ | verifier $V(s_T, g) \in \{0, 1\}$ that accepts/rejects the trajectory |

The agent under test maps $(\rho, s_t, h_t) \to a_t$ at each step, with
history $h_t$, until either a stop condition or a step budget is hit.
Scoring is the **end-state verifier**, not a turn-by-turn rubric.

This is the structural difference from QA benchmarks: in MMLU, the
agent's action is a single multiple-choice letter and $V$ is
string-equality; in SWE-bench, the agent's action is a sequence of
file edits and $V$ is `pytest` outcome on the post-state codebase
`[jimenez2024-swebench §sec-eval-protocol; kb/excerpts/jimenez2024-swebench#sec-eval-protocol]`.

## 2. Mechanism — how each benchmark instantiates $V$

The four canonical 2023–2024 agentic benchmarks differ primarily in
**what $V$ checks**, which determines what kind of agent capability
the score actually measures.

### 2.1 SWE-bench — regression-test verification

SWE-bench `[jimenez2024-swebench §abstract; kb/excerpts/jimenez2024-swebench#abstract]`
ships 2,294 real GitHub-issue tasks across 12 Python repos. Each task
instance carries:

- `repo` + `base_commit` — the codebase state $s_0$.
- `problem_statement` — the issue text $\rho$.
- `FAIL_TO_PASS` — set of tests that fail on $s_0$ and must
  pass on $s_T$ (the post-patch state).
- `PASS_TO_PASS` — set of tests that pass on $s_0$ and must
  continue to pass on $s_T$.

The verifier is

$$V(s_T, g) = \mathbb{1}\Big[\bigwedge_{t \in \text{F2P}} \mathrm{pass}(t, s_T)\Big] \cdot \mathbb{1}\Big[\bigwedge_{t \in \text{P2P}} \mathrm{pass}(t, s_T)\Big]$$

`[jimenez2024-swebench §sec-eval-protocol; kb/excerpts/jimenez2024-swebench#sec-eval-protocol]`.

The headline metric is `% resolved` over the 2,294 (or 500 in the
Verified subset) instances. At launch (Oct 2023), Claude 2 resolved
~1.96% `[jimenez2024-swebench §sec-headline]`; as of mid-2025 the
SWE-bench Verified leaderboard sits at >70% for Claude Sonnet 4.5
class agents (see Phase 1 sweep, EVAL section).

**SWE-bench Verified** (OpenAI, Aug 2024) is a 500-task subset
filtered for unambiguous issue specs and fair test sets via human
SWE labelers — original SWE-bench had a long tail of under-specified
issues `[jimenez2024-swebench §sec-eval-protocol]`. SWE-bench
Verified is now the de-facto target; raw SWE-bench is treated as
internal-development.

### 2.2 GAIA — reference-string verification with tool budget

GAIA `[mialon2023-gaia §abstract; kb/excerpts/mialon2023-gaia#abstract]`
is 466 questions (300 leaderboard / 166 dev). Each question requires
multi-step web browsing, file reading, or tool composition, but the
gold answer is a **single short string** (a number, a date, a name).
The verifier is exact-match-after-normalization on the agent's final
answer field — not LLM-as-judge.

GAIA is stratified into Levels 1/2/3 by tool-chain depth
`[mialon2023-gaia §sec-dataset; kb/excerpts/mialon2023-gaia#sec-dataset]`.

The headline gap at launch:

> human respondents obtain 92% vs. 15% for GPT-4 equipped with plugins.

`[mialon2023-gaia §abstract]` This 92%/15% gap is the inverse of the
GPQA gap (where AI is closing on humans): GAIA was designed so
**any human can solve every question** — the difficulty is purely
in the multi-step nature, not in specialized knowledge.

### 2.3 OSWorld — end-state verification on a real OS

OSWorld `[xie2024-osworld §abstract; kb/excerpts/xie2024-osworld#abstract]`
is 369 tasks across Ubuntu/Windows/macOS. Each task ships:

- An OS image setup script that produces $s_0$.
- A natural-language task description $\rho$ (e.g., "find the file
  matching pattern X and copy it to ~/results/").
- A **per-task evaluator script** that checks the final OS state
  $s_T$ against the goal `[xie2024-osworld §sec-task-spec; kb/excerpts/xie2024-osworld#sec-task-spec]`.

The action space is mouse/keyboard events on a real GUI (via VNC or
similar), not a structured API. This means the agent must do
**vision-based GUI grounding** — converting "click the Save button"
into pixel coordinates. The launch result was

> Human performance: 72.36%; best model: 12.24% (GPT-4V).

`[xie2024-osworld §sec-results; kb/excerpts/xie2024-osworld#sec-results]`

The dominant failure mode reported:

> [agents] primarily struggling with GUI grounding and operational
> knowledge.

`[xie2024-osworld §sec-results]` This matters for the eval-methodology
note (`kb/notes/evaluation/eval-methodology.md`): OSWorld's metric is
bottlenecked by a sub-capability (vision grounding) that may not be
the capability the eval intends to measure.

### 2.4 tau-bench — dual-control evaluation with policy compliance

tau-bench `[yao2024-tau-bench §abstract; kb/excerpts/yao2024-tau-bench#abstract]`
adds two pieces missing in the others:

1. **A simulated user** (an LLM playing the customer role) who can
   clarify, change their mind, or give incomplete information.
2. **A policy document** the agent must follow (e.g., the airline's
   refund rules).

The verifier is end-state check on a domain DB (retail / airline
DB rows match the goal state). What's new is the **pass^k metric**
`[yao2024-tau-bench §sec-passk; kb/excerpts/yao2024-tau-bench#sec-passk]`:

$$\mathrm{pass}^k(\text{agent}, \text{task}) = \mathbb{1}\Big[\bigwedge_{i=1}^{k} V(s_T^{(i)}, g) = 1\Big]$$

— **all** $k$ independent runs must succeed (not "at least one,"
which is the standard pass@k). The headline finding:

> even state-of-the-art function calling agents (like gpt-4o) succeed
> on <50% of the tasks, and are quite inconsistent (pass^8 <25% in
> retail).

`[yao2024-tau-bench §sec-result; kb/excerpts/yao2024-tau-bench#sec-result]`

## 3. Variants and lineage

The four canonical 2024 benchmarks above measure different kinds of
agent capability. The Phase 1 sweep proposed splitting `agentic-benchmarks`
into three sub-leaves (coding-agent, computer-use, task-completion); the
following table is a finer cut along the **what does $V$ check** axis:

| Benchmark | Year | $\rho$ source | $V$ kind | Step budget | Headline gap (human → AI launch) | Active leaderboard? |
|---|---|---|---|---|---|---|
| **SWE-bench / Verified** | 2023 / 2024 | real GitHub issues | regression tests on `pytest` | open (typically 30–50 turns) | n/a (no human baseline reported in paper) → Claude 2: 1.96%; mid-2025: >70% on Verified | yes (swebench.com) |
| **GAIA** | 2023 | hand-crafted multi-tool Qs | exact-match on string answer | open (tool budget only) | 92% → 15% (GPT-4) at launch | yes (HF Spaces) |
| **OSWorld** | 2024 | real desktop tasks | end-state OS check via custom script | bounded ~50 actions | 72% → 12% (GPT-4V) | yes (os-world.github.io) |
| **tau-bench** | 2024 | simulated retail / airline calls | DB end-state + pass^k | open (multi-turn) | n/a (no human baseline) → GPT-4o: <50% pass@1, <25% pass^8 retail | yes (Sierra GitHub) |
| **AgentBench** | 2023 | 8 envs (OS, DB, KG, web, etc.) | per-env rubric or end-state | varies | varies; key finding: "significant disparity between top commercial LLMs and open-source <70B" `[FORUM-SIGNAL]` | yes (THUDM) |
| **Cybench** | 2024 | 40 professional CTFs | flag-string match | open | n/a → "Claude 3.5 Sonnet, GPT-4o, o1-preview, Claude 3 Opus solved tasks taking human teams up to 11 minutes" | yes; used by AISI `[FORUM-SIGNAL]` |

`[FORUM-SIGNAL]` Cybench's role as a US/UK AISI pre-deployment eval
is from the Phase 1 sweep notes and METR communications, not the
Cybench abstract itself; the abstract speaks only of capability
results, not adoption.

### 3.1 Lineage: from MiniWob to OSWorld

Computer-use evaluation has a clear lineage:

- **MiniWoB** (2017, Shi et al.) — synthetic web-form tasks; the
  proto-OSWorld; small action space.
- **WebShop** (2022, Yao et al.) — simulated e-commerce; first step
  toward natural-language goals over an environment.
- **WebArena** (2024) — real websites under reproducible Docker
  snapshots; a domain-restricted alternative to OSWorld.
- **OSWorld** (2024) — full OS GUI; superset of WebArena's web tasks
  plus desktop apps.
- **τ-bench** (2024) — adds the simulated-user axis on top of
  WebArena/OSWorld-style domains.

### 3.2 Lineage: from HumanEval to SWE-bench-Verified

Coding evaluation has moved from sandbox tests to real-issue tests:

- **HumanEval** (2021, Chen et al., OpenAI Codex paper) — 164 hand-
  written function-completion problems with unit tests. Saturated
  at >95% pass@1 by 2024; treated as "hello world" only in 2025.
- **MBPP** (2021, Austin et al.) — similar profile; saturated.
- **APPS** (2021, Hendrycks et al.) — competition coding; partially
  saturated.
- **HumanEval+ / MBPP+** (2023, EvalPlus) — adversarial test
  augmentation against the original sets; partially restored signal
  but still small-program scope.
- **LiveCodeBench** (2024) — dynamic-append: post-cutoff competition
  problems to defeat contamination. The "honest version" of AIME-style
  evaluation for code.
- **SWE-bench** (2023) → **SWE-bench Verified** (2024) — repo-level,
  real-issue, real-tests. The current leaderboard target.
- **Aider Polyglot** (2024) — 225 Exercism exercises × 6 languages
  with two-attempt error-feedback protocol; reports cost-per-task
  alongside accuracy. Tier B in the Phase 1 sweep but increasingly
  cited.

[INTUITION] The trend is monotone: each generation of coding eval
makes the inputs more like real-world tasks (single function → multi-
file → real repos with real test infrastructure) and the verifier
more execution-grounded (string match → unit tests → repo regression
suite). The price is brittleness — repo-level evaluation fails if the
test environment isn't reproducible, which is why SWE-bench's
Docker-pinned environments are a load-bearing methodological detail.

## 4. Intuitions and analogies

[ANALOGY] Agentic benchmarks are to QA benchmarks as **end-of-game
chess evaluation** is to **chess-puzzle evaluation**: the puzzle
("white to move and mate in 2") rewards finding a single correct
move sequence, but the end-of-game evaluation rewards reaching a
correct final position via any legal sequence — including ones the
puzzle-author didn't anticipate. The analogy returns to canonical
form in §1: $V$ checks the end-state $s_T$, not the trajectory.

[INTUITION] pass^k vs. pass@k is the **production discipline** of
agent evaluation. pass@k says "the model has the capability somewhere
in its sample distribution." pass^k says "the model has the
capability reliably enough to deploy without human review." For
customer-service agents, "answers correctly 50% of the time and
wrong 50% of the time" is unacceptable in a way that "answers
correctly with one prompt template, fails with another" never is for
QA `[yao2024-tau-bench §sec-passk]`.

[INTUITION] The 1.96% → 70%+ trajectory on SWE-bench Verified over
~18 months is structurally similar to the AlphaGo → AlphaZero
trajectory: not a model-size win but a scaffolding-and-tool win.
Most of the gain came from agent harnesses (SWE-Agent, OpenHands,
Aider, Devin-style scaffolds) wrapping the same base model with
tool loops, not from frontier model capability alone. This is a
metric-design issue: the score conflates "model capability" with
"scaffold engineering quality" `[FORUM-SIGNAL: SWE-bench leaderboard
entries publish scaffold + model separately, but headline numbers
don't always]`.

[CONTRADICTION] Whether agentic benchmark scores **predict
real-world agent value** is contested. Anthropic's "computer use"
demo (Oct 2024) showed strong OSWorld-class performance but mixed
real-deployment reception; Devin's SWE-bench-Verified results
significantly outpaced its early-customer reports. The Phase 1
sweep's open question 6 — "what is the right eval framework for
agent capabilities" — has no consensus answer as of 2026-05.
METR's "task horizon" metric (time-equivalent of human
completion) is a candidate unifying metric but not yet widely
adopted.

## 5. Frontier and open questions (as of 2026-05)

- **Saturation timing.** SWE-bench Verified (500 tasks) shows a
  similar saturation trajectory to MMLU before it: 1.96% (Oct 2023)
  → 70%+ (mid-2025). At current rates the benchmark loses
  discrimination by 2026-Q3. Replacements: Multi-SWE-bench
  (multi-language extension, 2025), SWE-bench Multimodal (with
  screenshots), and SWE-Lancer (long-horizon real freelance work)
  are candidates but none yet has Verified's adoption.
- **Cost-aware leaderboards.** Aider Polyglot reports $-per-task
  alongside accuracy; SWE-bench Verified does not. As inference-time
  compute scaling grows (`kb/notes/scaling/inference-time-compute-scaling.md`),
  the right metric is increasingly accuracy/cost rather than
  accuracy alone. No standard yet.
- **User-simulator quality.** tau-bench's user simulator is itself
  an LLM, which means tau-bench is partially measuring the agent's
  ability to handle the **specific** quirks of the user-simulator's
  generation distribution. Whether scores transfer to real users is
  open.
- **Vision-grounding bottleneck.** OSWorld's 12% baseline is
  bottlenecked on GUI grounding, not on planning. This means
  text-only agents are evaluated unfairly — they can't even
  attempt the eval. The methodological question: should there be
  a separate "headless" track that abstracts the GUI to a text
  representation?
- **Agent-evaluator collusion / self-grading.** Some agentic
  benchmarks use LLM-as-judge for partial scoring; this opens a
  contamination axis where the same model class scores its own
  trajectory generously. tau-bench's DB-state check sidesteps this;
  GAIA's string match sidesteps this; SWE-bench's pytest sidesteps
  this. AgentBench partially uses LLM-as-judge — the Phase 1 sweep
  flagged this as a methodological weakness.
- **METR task horizon metric.** A unified metric — "time it would
  take an experienced human to complete this task" — would let
  cross-benchmark comparisons of agent capability. METR's autonomy
  evaluation report (2024) uses it; not yet adopted on
  community leaderboards.

## 6. See also

- `kb/notes/evaluation/eval-methodology.md` — contamination, prompt
  sensitivity, lm-eval-harness conventions; the methodological
  scaffolding under all of this.
- `kb/notes/evaluation/reasoning-benchmarks.md` — the reasoning
  side of the post-MMLU eval shift; complement to the agentic side.
- `kb/notes/reasoning/test-time-compute.md` — the inference-time
  compute axis that interacts with agent-scaffold quality on
  these leaderboards.
- `kb/notes/alignment/safety-evaluation.md` — agent benchmarks like
  Cybench cross-list as safety evaluations (dangerous-capability
  measurement). HarmBench, JailbreakBench, METR autonomy suite are
  the safety-side complements.
