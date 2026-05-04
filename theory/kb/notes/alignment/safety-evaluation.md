---
topic: alignment/safety-evaluation
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - harmbench2024
  - chao2024-jailbreakbench
  - russinovich2024-crescendo
  - anthropic-rsp-2024  # Anthropic Responsible Scaling Policy
  - metr-autonomy-2024
related_topics:
  - alignment/scheming-and-deceptive-alignment
  - alignment/sycophancy
  - evaluation/agentic-benchmarks
  - alignment/oversight-and-scalable-alignment
---

# Safety evaluation

**Status:** stub. Drafted from Phase 1 landscape sweep + WebFetch
abstracts; needs full Phase 2 treatment with PDF-grounded excerpts.

## What this topic covers

Safety evaluation is the family of *deployment-gating* benchmarks and
protocols used by frontier labs and external evaluators (UK AISI, US
AISI, METR, Apollo Research) to assess whether a model is safe to
release. It splits into three operational layers:

1. **Adversarial robustness benchmarks** — does the model refuse
   harmful queries under attack? (`harmbench2024`,
   `chao2024-jailbreakbench`, multi-turn jailbreaks like
   `russinovich2024-crescendo`).
2. **Dangerous-capability evaluations** — can the model autonomously
   carry out a task whose successful execution would constitute a
   threat? (METR autonomy task suite, Cybench for cyber CTFs).
3. **Pre-deployment policy frameworks** — what *commitments* does the
   lab make about what it will and will not deploy at each capability
   tier? (Anthropic Responsible Scaling Policy / ASL levels, OpenAI
   Preparedness Framework, Google DeepMind Frontier Safety Framework).

## Primary sources to read (in order)

1. `harmbench2024` — Mazeika et al. — "HarmBench: A Standardized
   Evaluation Framework for Automated Red Teaming and Robust Refusal"
   (`arxiv 2402.04249`). 510 behaviors, 18 attack methods, 33 target
   models/defenses. The standard red-teaming benchmark. Adopted by
   AISI for pre-deployment evals.
2. `chao2024-jailbreakbench` — Chao et al. — "JailbreakBench: An Open
   Robustness Benchmark for Jailbreaking Large Language Models"
   (`arxiv 2404.01318`, NeurIPS 2024 D&B). 100 target behaviors
   aligned with OpenAI usage policies; standardized threat model,
   system prompts, scoring functions; public leaderboard.
3. `russinovich2024-crescendo` — Russinovich et al. — "The Crescendo
   Multi-Turn LLM Jailbreak Attack" (`arxiv 2404.01833`,
   USENIX 2025). Multi-turn gradual escalation attack; **29–71% ASR
   improvement over single-turn baselines on AdvBench**. Documents
   that single-turn evals systematically *underestimate* attack
   surface.
4. `anthropic-rsp-2024` — Anthropic Responsible Scaling Policy
   (https://www.anthropic.com/news/anthropics-responsible-scaling-policy).
   ASL (AI Safety Level) tier framework: ASL-1 (no autonomy / low
   risk), ASL-2 (current frontier-baseline, e.g. Claude 3 family),
   ASL-3 (substantially elevated risk; specific deployment controls),
   ASL-4 (autonomous AI risk; not yet deployed). Each tier specifies
   capability evaluations that **must trigger** before deployment, and
   security / containment commitments that must be met **at** each
   tier.
5. `metr-autonomy-2024` — METR Autonomy Evaluation Resources
   (https://metr.org/blog/2024-03-13-autonomy-evaluation-resources/).
   77-task dangerous-capability suite spanning autonomous replication,
   research, and agent-level operation. METR's *task-horizon* metric
   (the human time-to-complete distribution) is becoming the
   field's lingua franca for capability quantification.

## Key claims to ground (Phase 2 todo)

- **HarmBench: the 510/18/33 numbers and what "behavior" means.** Need
  PDF excerpt to anchor the *taxonomy of behaviors* (illegal,
  unethical, dangerous capabilities, etc) — these affect how
  HarmBench composes with Apollo / scheming evals.
- **JailbreakBench: the 100-behavior set and scoring.** What is the
  exact threat model? How is "success" graded — judge-LLM, rule-based,
  human? Critical because attack-success-rate (ASR) numbers are not
  comparable across benchmarks if scoring differs.
- **Crescendo's escalation curve.** The 29–71% ASR delta over baseline
  is the load-bearing number. Need to ground what baselines and what
  models. Crescendo also disproportionately succeeds on Claude family
  models in the original paper — needs verification under updated
  defenses.
- **ASL-3 trigger conditions.** What specific capability evals does
  Anthropic's RSP commit to running? At what thresholds does ASL-3
  trigger? The RSP document is the primary source; this needs careful
  reading because the language is intentionally hedged.
- **METR task horizon.** The METR claim that "humans take N hours" is
  predictive of "model time-to-complete approaches the same scale" is
  an empirical extrapolation from a 2024 study. Need to verify the
  scaling-curve fit and the population over which the human-time
  distribution is taken.
- **The two-paper Nature 2026 result on autonomous jailbreak agents**
  (Phase 1 landscape sweep noted this) — reasoning models can
  autonomously construct jailbreaks against other models. This is the
  most concerning recent development; needs grounding.
- **AJAR and LATS** (`arxiv 2601.10971`, `arxiv 2601.02670`, both
  2025) — adaptive jailbreaks at 76% / 97–100% ASR against frontier
  models. This sets the upper end of the attack-success benchmark
  and is the natural antagonist to whatever defense is deployed.

## Open methodological questions (Phase 2)

- **How does safety-eval pass/fail compose with capability-eval
  pass/fail?** Anthropic RSP's ASL framework gates on *both*
  capability triggers AND specific safety mitigations. The
  conjunctive structure is not standardized across labs (OpenAI's
  Preparedness Framework uses a similar but distinct decomposition).
- **Adversarial-robustness vs deployment realism.** HarmBench /
  JailbreakBench measure attack success against *fixed* defenses.
  Production deployment includes runtime monitoring, abuse detection,
  and account-level controls — not modeled by the benchmark.
- **The reasoning-model attack surface.** Long CoT exposes the model
  to *self-jailbreak* via the trace. Apollo-style evaluation methods
  partly cover this; whether HarmBench-style benchmarks need a CoT-
  specific extension is open.
- **Integration with `alignment/scheming-and-deceptive-alignment`.**
  A scheming model presented with a safety eval may behave differently
  than the same model presented with a real-deployment query (cf.
  Apollo's "instrumental alignment-faking" probe). Safety-eval scores
  are not robust to this; the evaluation suite needs *honeypot* or
  *unaware-of-eval* protocols.

## Related notes

- `kb/notes/alignment/scheming-and-deceptive-alignment.md` — the
  failure mode that compromises the validity of behavioral
  safety evals.
- `kb/notes/alignment/sycophancy.md` — a related but distinct
  deployment-quality issue often co-evaluated.
- `kb/notes/evaluation/agentic-benchmarks.md` — Cybench appears in
  both safety-eval (cyber-attack capability) and agentic-eval
  (general agent capability) frames.
- `kb/notes/alignment/oversight-and-scalable-alignment.md` —
  scalable-oversight protocols are *training-side*; safety-evals
  are *deployment-side*; both gate the path from a trained model
  to a released one.
