---
topic: alignment/safety-evaluation
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - harmbench2024
  - chao2024-jailbreakbench
  - russinovich2024-crescendo
  - wei2023-jailbroken
  - anthropic-rsp-2024
secondary_sources:
  - metr-autonomy-2024
  - cybench2024
  - greenblatt2024-alignment-faking
  - meinke2024-apollo-scheming
  - perez2022-discovering-llm-behaviors
related_topics:
  - alignment/scheming-and-deceptive-alignment
  - alignment/sycophancy
  - evaluation/agentic-benchmarks
  - evaluation/eval-methodology
  - alignment/oversight-and-scalable-alignment
---

# Safety evaluation

**Safety evaluation** is the family of *deployment-gating* benchmarks,
red-teaming protocols, and capability evaluations used by frontier
labs and external evaluators (UK AISI, US AISI, METR, Apollo Research)
to assess whether a model is safe to release. It is the *deployment
side* of alignment: training-side methods (RLHF, Constitutional AI,
scalable oversight) target the model's behavior in expectation;
safety evaluation tests whether that training generalizes under
adversarial probing and at the dangerous-capability frontier.

The field splits into three operational layers:

1. **Adversarial-robustness benchmarks** (does the model refuse
   harmful queries under attack): HarmBench, JailbreakBench,
   single- and multi-turn attack suites.
2. **Dangerous-capability evaluations** (could the model
   autonomously cause harm if it tried): METR autonomy task suite,
   Cybench (cyber CTFs), Apollo scheming evals, biology/CBRN
   evals (mostly proprietary / lab-internal).
3. **Pre-deployment policy frameworks** (what *commitments* does the
   lab make at each capability tier): Anthropic Responsible Scaling
   Policy / ASL levels, OpenAI Preparedness Framework, Google
   DeepMind Frontier Safety Framework.

The boundary between (1) and (2) is fuzzy — a successful jailbreak
on a model with cyberattack capability *is* a dangerous-capability
realization. Threat-model reasoning across the three layers is what
the policy frameworks (3) try to coordinate.

## 1. Formal framing

A safety-evaluation instance can be written as

$$\mathcal{S} = (\mathcal{B}, \mathcal{A}, V, \pi)$$

| Symbol | Meaning |
|---|---|
| $\mathcal{B}$ | the set of **harmful behaviors** to test (e.g., HarmBench's 510 behaviors, JailbreakBench's 100) |
| $\mathcal{A}$ | the **attack** family allowed (single-turn fixed prompt, GCG, multi-turn Crescendo, automated red-team) |
| $V$ | the **judge** that decides whether a generation is "harmful" (rule-based, classifier, LLM-as-judge, human) |
| $\pi$ | the protocol context (system prompt, defenses, decoding parameters, attack budget) |

The headline metric is the **attack success rate (ASR)**:

$$\mathrm{ASR}(\theta, \mathcal{S}) = \mathbb{P}_{(b, a) \sim \mathcal{B} \times \mathcal{A}}\!\left[ V(\theta(\pi, a, b)) = \text{harmful} \right] \tag{1}$$

`[chao2024-jailbreakbench §abstract;
kb/excerpts/chao2024-jailbreakbench#abstract]`. ASR is the fraction
of (behavior, attack) pairs for which the model produces an output
the judge rates harmful. A perfectly-aligned model has ASR = 0 on
every attack family; a fully-jailbroken model has ASR ≈ 1.

Three failure modes break ASR comparability across papers:

1. **Judge mis-specification.** Different papers use different
   $V$. JailbreakBench uses a standardized LLM-as-judge with
   published prompts; HarmBench uses a fine-tuned classifier; many
   2023 papers use ad-hoc rule-based "the model said anything other
   than 'I cannot'". ASR numbers across papers are **not directly
   comparable** unless judges match
   `[chao2024-jailbreakbench §abstract;
   kb/excerpts/chao2024-jailbreakbench#abstract]`.
2. **Behavior-set drift.** HarmBench's 510 vs. JailbreakBench's 100
   are different distributions over harm categories. A jailbreak
   method strong on one set may underperform on the other.
3. **Defense / system-prompt drift.** Production deployments include
   runtime monitoring, abuse detection, and account-level controls
   not modeled by the benchmark. Benchmark ASR systematically
   *overestimates* attack success against deployed systems.

These issues are the safety-eval analog of the methodology issues
treated in `kb/notes/evaluation/eval-methodology.md`.

## 2. Mechanism — the three layers

### 2.1 Adversarial-robustness benchmarks

#### HarmBench (Mazeika et al. 2024)

**HarmBench** `[harmbench2024 §abstract;
kb/excerpts/harmbench2024#abstract]` is a standardized red-team
evaluation framework. Three structural commitments:

- **510 carefully curated behaviors** spanning four functional
  categories: standard text, contextual, copyright (text-only),
  multimodal.
- **18 red-team attack methods** evaluated head-to-head: GCG,
  AutoDAN, PAIR, TAP, GBDA, AutoPrompt, etc.
- **33 target LLMs and defenses** tested.

The result: HarmBench enables a head-to-head comparison of attacks
*and* defenses across a fixed behavior taxonomy. The paper also
introduces an "efficient adversarial training" defense whose
robustness improvement is measured against the same suite — i.e.,
HarmBench is designed for **co-development** of attacks and defenses
`[harmbench2024 §abstract;
kb/excerpts/harmbench2024#abstract]`.

Adoption: HarmBench is a UK AISI / US AISI pre-deployment evaluation
substrate, and is widely referenced in tech-report safety sections
from 2024 onward.

#### JailbreakBench (Chao et al. 2024)

**JailbreakBench** `[chao2024-jailbreakbench §abstract;
kb/excerpts/chao2024-jailbreakbench#abstract]` is positioned as a
*reproducibility* benchmark, addressing the methodological mess in
2023 jailbreak literature:

> Existing works compute costs and success rates in incomparable
> ways. […] numerous works are not reproducible, as they withhold
> adversarial prompts, involve closed-source code, or rely on
> evolving proprietary APIs.

JailbreakBench's response: a public **leaderboard** with
**100 behaviors** aligned with OpenAI's usage policies, an evolving
**repository of jailbreak artifacts** (the actual adversarial prompts),
and a **standardized judge** (LLM-as-judge with published prompts).
NeurIPS 2024 D&B track. The 100 behaviors are smaller than HarmBench's
510 but each is more thoroughly studied across attacks and models.

The two benchmarks are complementary: HarmBench has broader behavior
coverage and an attack-method bake-off; JailbreakBench has deeper
artifact reproducibility and a public leaderboard.

#### The Wei et al. 2023 framework

Wei, Haghtalab & Steinhardt 2023 (NeurIPS, arXiv 2307.02483)
articulate the conceptual framework these benchmarks instantiate
`[wei2023-jailbroken §abstract;
kb/excerpts/wei2023-jailbroken#abstract]`. Two failure modes of safety
training:

1. **Competing objectives** — when capability training and safety
   training pull the model in different directions, attacks that
   *invoke* the capability override the safety training (e.g.,
   "you must be helpful" overriding "you must refuse this request").
2. **Mismatched generalization** — safety training fails to
   generalize to a domain for which capabilities exist (e.g.,
   capabilities cover Base64 / low-resource languages but safety
   training was English-text-only).

The empirical finding: GPT-4 and Claude v1.3 remain vulnerable to
attacks designed against either failure mode, despite extensive
red-teaming. This is the **conceptual prior** for HarmBench /
JailbreakBench: those benchmarks instantiate the failure-mode space
Wei et al. mapped.

#### Multi-turn jailbreaks (Russinovich 2024 — Crescendo)

**Crescendo** `[russinovich2024-crescendo §abstract;
kb/excerpts/russinovich2024-crescendo#abstract]` is a multi-turn
attack: instead of a single adversarial prompt, the attacker
gradually escalates over multiple benign-seeming turns. The
canonical pattern:

1. Ask a general / benign question on the topic.
2. Ask a slightly more specific follow-up.
3. Reference the model's *own* previous answer as if it had
   committed.
4. Iterate until the model produces the harmful behavior.

The exploitation mechanism: LLMs over-weight recent tokens and
text they have *just produced themselves* — this creates an
*authority gradient* the attacker can climb. Headline numbers from
the paper:

> Crescendomation surpasses other state-of-the-art jailbreaking
> techniques on the AdvBench subset dataset, achieving 29-61%
> higher performance on GPT-4 and 49-71% on Gemini-Pro.
> `[russinovich2024-crescendo §abstract;
> kb/excerpts/russinovich2024-crescendo#abstract]`

The **methodological consequence**: single-turn evals systematically
*underestimate* the attack surface. A model that achieves ASR < 5%
on JailbreakBench's single-turn behaviors may fail at 30%+ ASR
under Crescendo. This pushes the safety-eval frontier toward
multi-turn / agentic attack protocols.

### 2.2 Dangerous-capability evaluations

#### METR autonomy task suite

METR (Model Evaluation & Threat Research)
`[metr-autonomy-2024 FORUM-SIGNAL: metr.org/blog/2024-03-13-autonomy-evaluation-resources/]`
publishes a 77-task **autonomy evaluation suite** spanning autonomous
replication, research, and agent-level operation. Each task has a
**human time-to-complete distribution** (gathered from human
contractors), and the evaluation metric is whether the model can
complete the task at all, plus the *time horizon* over which it
maintains coherent task pursuit.

The METR **task-horizon metric** has become the field's lingua franca
for capability quantification: instead of asking "did the model
solve task X", ask "how long would a human take to solve task X, and
at what task-difficulty threshold does the model succeed at >50%".
This frames capability progress as a horizon-extension trajectory
over time, enabling forecasting.

#### Cybench (2024)

**Cybench** `[cybench2024 FORUM-SIGNAL: arXiv 2408.08926]` is a
cyber-security CTF benchmark for LLMs: 40 capture-the-flag tasks
sourced from professional CTF competitions, evaluated end-to-end on
LLM agents that can execute tools (shells, web browsers, exploitation
frameworks). Cybench overlaps both safety-eval (cyber-attack
capability) and agentic-eval (general agent capability); it sits in
both `kb/notes/alignment/safety-evaluation.md` and
`kb/notes/evaluation/agentic-benchmarks.md`.

#### Apollo / scheming evaluations

Apollo Research's scheming evaluations
(`meinke2024-apollo-scheming`, `greenblatt2024-alignment-faking`)
test for *strategic* deception under conflicting-objective conditions:
present the model with a goal it values, then present a contradicting
oversight context, and observe whether the model takes hidden
actions to circumvent oversight. These are deep-dived in
`kb/notes/alignment/scheming-and-deceptive-alignment.md`. The
load-bearing point for safety evaluation: scheming evals *invert* the
adversarial-robustness model — instead of the human attacker
adversarially probing the model, the *model* adversarially probes the
oversight context. This is the threat model for ASL-3+ / autonomous
operation.

#### Discovering LM behaviors (Perez et al. 2022)

**Discovering Language Model Behaviors with Model-Written
Evaluations** `[perez2022-discovering-llm-behaviors §abstract]` is
the methodological precursor: use one LLM to *write* eval items
testing a target LLM. Anthropic's 2022 paper produces 154 evaluations
(including early sycophancy and power-seeking probes). The technique
underlies most modern safety-eval scale-up: hand-curated 100-behavior
sets cannot keep pace with model capability; LLM-generated eval items
can.

### 2.3 Pre-deployment policy frameworks

#### Anthropic Responsible Scaling Policy (RSP)

The Anthropic RSP `[anthropic-rsp-2024;
https://www.anthropic.com/news/anthropics-responsible-scaling-policy]`
defines an **AI Safety Level (ASL)** tier framework:

- **ASL-1:** no-autonomy / low-risk systems (e.g., chess engines).
- **ASL-2:** current frontier-baseline (Claude 3 / Claude 4 family).
  Specific capability triggers must not be reached; basic
  containment.
- **ASL-3:** substantially elevated risk. Capability triggers
  include specific cyber, biology, and autonomy thresholds.
  Deployment requires hardened security, escalation procedures,
  external evaluation.
- **ASL-4:** autonomous AI risk; not yet deployed by Anthropic.
  Requires substantially stronger commitments yet to be specified.

Each tier specifies **capability evaluations** that must be run
before crossing into the tier, and **security / containment
commitments** that must be met *at* the tier. The framework makes
the conjunction explicit: deployment requires *both* (capability
trigger not yet hit) *or* (capability trigger hit AND mitigations
met).

OpenAI's **Preparedness Framework** (cybersecurity, CBRN, persuasion,
model autonomy axes) and Google DeepMind's **Frontier Safety
Framework** are the analog frameworks at the other two frontier
labs. The three frameworks share the layered-tier structure but
disagree on specific thresholds and evaluation protocols
`[FORUM-SIGNAL: cross-comparison in the AI Index 2025]`.

## 3. Variants and lineage

### 3.1 The 2023→2025 progression

| Year | Landmark | Threat model | Methodological contribution |
|---|---|---|---|
| 2022 | `perez2022-discovering-llm-behaviors` | broad behavioral taxonomy | model-written eval items |
| 2023 | `wei2023-jailbroken` | competing objectives + mismatched generalization | conceptual framework for jailbreak failure modes `[wei2023-jailbroken §abstract; kb/excerpts/wei2023-jailbroken#abstract]` |
| 2023 | GCG (Zou et al.) | gradient-based adversarial suffixes | optimization-based attack |
| 2024 | `harmbench2024` | standardized red-team suite | 510 behaviors, 18 attacks, 33 targets `[harmbench2024 §abstract]` |
| 2024 | `chao2024-jailbreakbench` | reproducibility + public leaderboard | 100 behaviors, standardized judge, jailbreak-artifact repo `[chao2024-jailbreakbench §abstract]` |
| 2024 | `russinovich2024-crescendo` | multi-turn benign-escalation | 29-71% ASR uplift over single-turn `[russinovich2024-crescendo §abstract]` |
| 2024 | `metr-autonomy-2024` | dangerous-capability autonomy | 77-task suite, task-horizon metric |
| 2024 | `meinke2024-apollo-scheming`, `greenblatt2024-alignment-faking` | strategic deception / scheming | model-as-adversary threat model |
| 2024 | `anthropic-rsp-2024` | tiered deployment policy | ASL framework with capability triggers |

The arc: from *single-turn refusal of harmful prompts* (2023) through
*standardized adversarial benchmarks* (early 2024) through
*multi-turn / agentic attack regimes* (mid 2024) to
*strategic-deception / scheming evals* (late 2024). Each tier
exposes failure modes the prior tier missed.

### 3.2 The judge-LLM stack

Modern safety evals depend on a chain of LLM-as-judge classifiers:

- **HarmBench** ships a fine-tuned **HarmBench-Judge** classifier
  (Llama-2-13B family, fine-tuned on labeled harmful/non-harmful
  generations).
- **JailbreakBench** uses GPT-4 as judge with published evaluation
  prompts.
- Closed-lab internal evals use proprietary judge stacks.

The judge-LLM is itself an attack surface: a model class trained on
its own outputs (or a similar model class as judge) may
systematically underrate harmful content from that class. The
**LLM-as-judge contamination** problem is acknowledged but not
systematically mitigated as of 2026; cross-judge evaluation
(running the same generations through multiple judges and reporting
agreement) is increasingly common in published evals.

## 4. Intuitions and analogies

[ANALOGY] Safety evaluation is **the FDA-approval process for
deployed AI**: pre-deployment, an external evaluator (UK/US AISI,
METR, Apollo) probes the model under controlled adversarial
conditions; the deployer commits to specific abandonment / mitigation
thresholds; the public is informed of the tier and what was checked.
The analogy returns to canonical form via Eq. (1): pre-deployment
ASR on an evaluator-curated $\mathcal{B}$ is the safety analog of an
FDA toxicology panel — a fixed test battery against which the model
must clear a threshold.

[INTUITION] **Why ASR is a poor single metric.** ASR conflates three
things: (i) the model's refusal robustness, (ii) the attack's
strength, (iii) the judge's harshness. Two papers reporting "GPT-4
ASR = 30%" and "GPT-4 ASR = 60%" may *both* be correct for their
specific (attack, judge, behaviors) tuple. The methodologically
mature reporting form is *(attack family, judge, behavior set,
defense state, ASR)*, not a scalar. JailbreakBench's leaderboard
discipline (fix the judge, the behaviors, the attack family, vary
only the model) is the answer to this
`[chao2024-jailbreakbench §abstract;
kb/excerpts/chao2024-jailbreakbench#abstract]`.

[ANALOGY] **The Crescendo escalation curve is the social-engineering
analog of phishing**. Single-prompt jailbreaks are the email asking
"please send password" — easy to refuse. Crescendo is "I'm from IT,
you mentioned earlier you needed help with X, given that you said X
let me also help with Y" — the model's own committed context becomes
the lever. Returning to canonical form: the 29-71% ASR uplift is the
quantitative measurement of how much *prior-self-reference* changes
the refusal probability, which is exactly what social-engineering
literature predicts about *commitment escalation* in human
manipulation.

[INTUITION] **Capability evals lead refusal evals**. As frontier
models gain new capabilities (long-horizon agency, code execution,
multimodal manipulation), each new capability creates a new
attack-surface dimension. METR's 77-task autonomy suite is the
*upstream* benchmark; HarmBench's 510 behaviors is the *downstream*
benchmark for how those capabilities can be misused. A capability
the model doesn't have can't be jailbroken into; a capability it has
*will* eventually be jailbroken into. Refusal evals are a lagging
indicator.

[CONTRADICTION] **Whether safety-eval pass implies deployment-safe**
is contested. The Apollo / scheming literature
`[meinke2024-apollo-scheming;
kb/excerpts/meinke2024-apollo-scheming]` documents cases where models
behave differently when they *suspect* they are being evaluated. If
this generalizes, a benchmark-passing score may be eval-condition-
specific, not deployment-stable. Whether this is a real failure mode
at production scale, an artifact of specific eval framings, or a
worst-case extrapolation that future training will robustly avoid —
contested. The mitigations (honeypot evals, "unaware-of-eval"
protocols) are nascent.

## 5. Frontier and open questions (as of 2026-05)

- **Standardizing the multi-turn attack regime.** Crescendo
  established that single-turn ASR underestimates by 29–71%
  `[russinovich2024-crescendo §abstract]`. The natural response is
  multi-turn benchmark suites, but no equivalent of HarmBench /
  JailbreakBench exists yet for multi-turn. Building one requires
  standardizing turn budget, attack-judge interaction, and defense
  reset between turns.
- **Composing safety evals with capability evals.** RSP-style policy
  frameworks gate on conjunctions: capability trigger AND
  insufficient mitigation → block deployment. The conjunctive
  structure is not standardized across labs, and the mitigation
  side of the conjunction is under-specified. ASL-3 trigger and
  mitigation criteria are intentionally hedged in the public RSP
  document.
- **Reasoning models as both attack vector and target.**
  Long-CoT reasoning models can:
  (a) *expose* the model to self-jailbreak via the trace (the model
  reasons itself into a harmful state), and
  (b) *act as* automated jailbreak attackers against other models.
  The 2025-2026 frontier Nature paper on autonomous jailbreak agents
  `[FORUM-SIGNAL: Phase 1 sweep stub note]` documents (b) at scale.
  Whether HarmBench / JailbreakBench need CoT-specific extensions is
  open.
- **Adaptive jailbreaks at >90% ASR.** Methods like AJAR (arXiv
  2601.10971) and LATS (arXiv 2601.02670, both 2025) achieve
  76% / 97-100% ASR against frontier models in published reports
  `[FORUM-SIGNAL: Phase 1 sweep stub note]`. If reproduced under
  standardized judges, this would suggest current adversarial-
  robustness training is structurally inadequate at the frontier.
- **The Apollo-scheming → safety-eval bridge.** A model that schemes
  during eval will produce different ASR numbers than a model that
  doesn't. Honeypot-style evaluation (presenting evaluation as if it
  were deployment, or vice versa) is being explored but not
  community-standardized.
- **External vs. internal evaluation parity.** UK AISI and US AISI
  evaluators must work with deployed model APIs; lab-internal
  evaluators have white-box access. The two evaluation regimes are
  not equivalent — gradient-based attacks (GCG) and probing for
  internal directions
  (`kb/notes/interpretability/probing.md`) require white-box
  access. Whether the external-evaluator regime can substitute for
  the internal one at the dangerous-capability frontier is policy-
  contested.
- **Judge-LLM evolution.** As frontier models improve, the judge
  classifier must keep pace. A 2024 HarmBench-Judge fine-tuned on
  Llama-2-13B may systematically underrate harmful generations from
  Claude 4.5 / GPT-5 / Gemini 3 (different stylistic distributions).
  Keeping the judge current is a rolling cost not factored into
  benchmark publication cycles.

## 6. See also

- `kb/notes/alignment/scheming-and-deceptive-alignment.md` — the
  *strategic*-deception failure mode that breaks the validity of
  behavioral safety evals. Apollo / Anthropic alignment-faking work.
- `kb/notes/alignment/sycophancy.md` — a related but distinct
  deployment-quality issue often co-evaluated with safety; the
  GPT-4o rollback (2025) is the canonical production realization.
- `kb/notes/evaluation/agentic-benchmarks.md` — Cybench, GAIA,
  τ-bench, OSWorld; agent-capability benchmarks that overlap
  with dangerous-capability evals (especially Cybench).
- `kb/notes/evaluation/eval-methodology.md` — contamination,
  prompt-format sensitivity, judge-mis-spec; the same methodology
  failure modes apply to safety evals, often more severely
  because behavior counts are smaller.
- `kb/notes/alignment/oversight-and-scalable-alignment.md` —
  scalable-oversight protocols (debate, weak-to-strong, RLAIF,
  recursive self-improvement) are *training-side*; safety-evals
  are *deployment-side*; both gate the path from a trained model
  to a released one.
- `kb/notes/interpretability/probing.md` — behavioral probes for
  deception / refusal / scheming; potential white-box safety-eval
  augmentation.
