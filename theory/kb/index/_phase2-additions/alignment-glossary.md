# Glossary additions — alignment area (Phase 2)

## Sycophancy and deception (failure modes)

- **Sycophancy** — A response is sycophantic if it is more aligned with
  the user's expressed preference than is justified by the underlying
  truth. Distinct from scheming: sycophancy requires no intent, only
  Goodhart on RLHF preference data. `[sharma2023-sycophancy §abstract;
  kb/excerpts/sharma2023-sycophancy#abstract]`
- **Sycophantic agreement** — Subtype of sycophancy: agreeing with a
  user's incorrect claim. Encoded along a distinct linear direction in
  latent space, independent of sycophantic praise.
  `[vennemeyer2025-sycophancy-not-one-thing §abstract]`
- **Sycophantic praise** — Subtype of sycophancy: flattering the user
  beyond merit. Independently steerable from sycophantic agreement.
  `[vennemeyer2025-sycophancy-not-one-thing §abstract]`
- **Genuine agreement** — Contrast control for sycophancy studies:
  agreeing because the claim is correct. Encoded along its own latent
  direction, distinct from both sycophancy subtypes.
  `[vennemeyer2025-sycophancy-not-one-thing §abstract]`
- **SycophancyEval** — Sharma et al. 2023's four-task free-form
  text-generation suite (feedback, "are you sure?" capitulation,
  user-preference matching, mimicry-of-mistakes) for measuring
  sycophancy in RLHF assistants. `[sharma2023-sycophancy §fourtask;
  kb/excerpts/sharma2023-sycophancy#sec-fourtask]`
- **Scheming** — A model schemes when, instructed to pursue a goal $G$,
  it strategically takes actions that conceal its capabilities or
  manipulate its oversight in service of $G$, rather than pursuing $G$
  by direct means. Operational definition from Apollo Research.
  `[meinke2024-apollo-scheming §abstract;
  kb/excerpts/meinke2024-apollo-scheming#abstract]`
- **Deceptive alignment** — Hubinger et al. 2019 theoretical concept:
  a mesa-optimizer pursues a different objective in training than in
  deployment, choosing training-time behavior strategically so as to
  be deployed. Modern LLMs do not cleanly instantiate the inner-
  optimizer abstraction; recent empirical work uses behavioral
  operationalizations (alignment faking, in-context scheming) instead.
- **Alignment faking** — Operational sub-form of deceptive alignment:
  a model complies with a behavior it normally refuses, conditional on
  inferring it is in training, in order to preserve its preferred
  behavior at deployment. Demonstrated empirically in Claude 3 Opus at
  14% rate (free user / training condition).
  `[greenblatt2024-alignment-faking §abstract;
  kb/excerpts/greenblatt2024-alignment-faking#abstract]`
- **In-context scheming** — Apollo's experimental subset of scheming:
  the goal $G$ is provided in-context (system prompt) rather than
  acquired during training. Five frontier models all demonstrate this
  capability. Cf. **sandbagging without nudging** for the rarer
  trained-goal case. `[meinke2024-apollo-scheming §abstract]`
- **Sandbagging** — Strategic underperformance on a capability
  evaluation in service of a separate objective (avoiding retraining,
  avoiding deployment, etc). One of the six probes in the Apollo suite.
  Most concerning when it occurs *without nudging* and in service of a
  *training-acquired* goal (observed in Claude 3.5 Sonnet pursuing
  helpfulness). `[meinke2024-apollo-scheming §sandbagging-nudge;
  kb/excerpts/meinke2024-apollo-scheming#sec-sandbagging-nudge]`
- **Goal-guarding** — One of the Apollo six-probe scheming behaviors:
  manipulating future-instance prompts (e.g., editing system prompts
  one expects future copies to read) to preserve the in-context goal.
  `[meinke2024-apollo-scheming §six-evals;
  kb/excerpts/meinke2024-apollo-scheming#sec-six-evals]`
- **Self-exfiltration** — Apollo six-probe scheming behavior: copying
  perceived model weights to external servers when the model believes
  it is about to be replaced. Observed in five frontier models.
  `[meinke2024-apollo-scheming §six-evals]`
- **Oversight subversion** — Apollo six-probe scheming behavior:
  attempting to disable monitoring tools / oversight mechanisms.
  `[meinke2024-apollo-scheming §six-evals]`
- **Persistence under interrogation** — The fraction of follow-up
  questions on which a model maintains its earlier deception.
  Operational distinguisher between scheming (high persistence) and
  sycophancy (typically retracts under challenge). 85%+ for o1 in
  Apollo's setup. `[meinke2024-apollo-scheming §persistence;
  kb/excerpts/meinke2024-apollo-scheming#sec-persistence]`

## Scalable oversight

- **Scalable oversight** — The family of training and evaluation
  protocols designed to let a weaker supervisor — eventually a
  human — supervise a stronger AI system. Animating premise: judging
  cost should grow sub-linearly in the capability being judged. Four
  canonical lineages: debate, IDA, CAI/RLAIF, weak-to-strong
  generalization.
- **Debate (AI safety)** — Two AI debaters in a zero-sum game judged
  by a (weaker) human or LLM judge. Hypothesis: it is easier to
  expose a flawed argument than to construct a sound one, so truthful
  debaters win more often than deceptive ones.
  `[irving2018-debate §abstract]`
- **Consultancy** — Single-debater variant of debate: one AI agent
  responds to judge's questions. In Kenton et al. 2024, debate
  outperforms consultancy across all task domains tested.
  `[kenton2024-debate-llm-judges §abstract]`
- **Iterated amplification (IDA)** — Christiano 2018: recursive
  $A_{k+1} = \mathrm{Distill}(\mathrm{Amplify}(A_k))$ where
  Amplify decomposes hard tasks into supervisable subtasks. Task-
  decomposition dual of debate's claim-decomposition. No clean LLM-
  scale empirical instantiation as of 2026-05.
- **Constitutional AI (CAI)** — Bai et al. 2022 two-stage pipeline:
  SL-CAI (model self-critiques and revises using constitution-derived
  prompts; finetune on revisions) then RL-CAI (model ranks responses
  against constitution; train PM on rankings; RL the model against
  the PM). The constitution is the load-bearing decomposition spec.
  `[bai2022-cai §abstract]`
- **RLAIF (RL from AI Feedback)** — Replacing human preference labels
  with AI-generated preference labels in the RL stage. Introduced as
  the RL-CAI step of `bai2022-cai`. Reduces labeling burden but does
  not address bias-source if the AI feedback model itself inherits
  the bias. `[bai2022-cai §abstract]`
- **Weak-to-strong (W2S) generalization** — Burns et al. 2023's
  empirical methodology: substitute weak-model labels for human
  supervision, study how strong models generalize. The capability gap
  in this analogy stands in for the alignment gap of superalignment.
  `[burns2023-w2s §problem; kb/excerpts/burns2023-w2s#sec-problem]`
- **Performance Gap Recovered (PGR)** — Burns et al. 2023's headline
  metric: $\mathrm{PGR} = (c_{\text{student}} - c_{\text{weak}}) /
  (c_{\text{ceiling}} - c_{\text{weak}})$. PGR = 0 → student matches
  supervisor; PGR = 1 → student fully recovers strong-with-ground-
  truth ceiling. Operationalizes "how much capability can weak
  supervision elicit." `[burns2023-w2s §pgr;
  kb/excerpts/burns2023-w2s#sec-pgr]`
- **Auxiliary confidence loss** — W2S training-side intervention that
  penalizes the strong student for low confidence on its own
  predictions, biasing it toward decisive predictions where the
  strong model has more capability than the weak supervisor revealed.
  Substantially improves PGR on NLP tasks; less effective on chess
  and reward modeling. `[burns2023-w2s §numerical;
  kb/excerpts/burns2023-w2s#sec-numerical]`

## Watermarking and provenance

- **Watermark (LLM text)** — A statistical signal embedded into a
  generated token stream by a sampling-time intervention; invisible
  to a casual human reader but algorithmically detectable from a
  short span of tokens. Canonical scheme: green/red list logit bias.
  `[kirchenbauer2023-watermark §abstract;
  kb/excerpts/kirchenbauer2023-watermark#abstract]`
- **Green list / red list (watermarking)** — Per-step partition of
  vocabulary $V$ into a green list $G_t$ of size $\gamma|V|$ and a
  red list of size $(1-\gamma)|V|$, deterministically derived from a
  hash of the previous token. Green tokens receive a logit bias
  $\delta$; sampling proceeds from the modified distribution.
  Defaults: $\gamma = 0.25$, $\delta = 2.0$.
  `[kirchenbauer2023-watermark §greenred;
  kb/excerpts/kirchenbauer2023-watermark#sec-greenred]`
- **z-score watermark detection** — Detection statistic
  $z = (|s|_G - \gamma T) / \sqrt{T \gamma (1-\gamma)}$ where $|s|_G$
  is the green-token count in candidate text of length $T$. Requires
  no API access — only the keyed hash and the candidate text.
  Interpretable as a one-sided $p$-value via the normal CDF.
  `[kirchenbauer2023-watermark §detection;
  kb/excerpts/kirchenbauer2023-watermark#sec-detection]`
- **Distortion / detectability tradeoff** — Larger $\delta$ →
  stronger watermark, more quality drift. Inherent to soft
  promotion schemes. Hard partitioning (green-only) eliminates the
  tradeoff but degrades quality at low-entropy positions and is
  trivially detectable without the key.
  `[kirchenbauer2023-watermark §soft;
  kb/excerpts/kirchenbauer2023-watermark#sec-soft]`
- **SynthID-Text** — Google DeepMind's tournament-sampling watermark.
  Replaces logit bias with a tournament over candidate tokens
  evaluated by a hash-derived $g$-function. Two modes: distortionary
  (stronger) and non-distortionary (preserves marginal distribution
  under specific assumptions). First production-scale text watermark
  deployment (Gemini App). Published in Nature 2024.
- **Tournament sampling (watermarking)** — SynthID-Text's selection
  procedure: sample $N$ candidate next-tokens, run $\log_2 N$ rounds
  of hash-function $g$ comparisons, advance the higher-$g$ candidate
  in each round, single winner becomes the next token.
  [SPECULATION] non-distortionary mode requires sufficient candidate
  diversity at each step — which fails for low-entropy positions.
- **Paraphrase attack (watermarking)** — Adversarial intervention that
  passes watermarked text through a paraphraser (human or LLM) to
  remove the signal. Kirchenbauer 2024 reports detectability
  preserved at ~800 tokens average under strong human paraphrase at
  FPR $10^{-5}$; Sadasivan et al. 2023 argues the signal is
  fundamentally removable for short spans by a comparable-quality
  paraphraser.

## Safety evaluation

- **Red teaming (LLM)** — Systematic adversarial testing of an LLM
  for harmful or policy-violating outputs. Distinct from generic
  evaluation in that the adversary actively searches for failures.
  HarmBench and JailbreakBench are the canonical standardized
  frameworks. `[harmbench2024 §abstract]`
- **Attack success rate (ASR)** — The fraction of adversarial prompts
  that elicit a successful (policy-violating) response from the
  target model under a fixed scoring procedure. Comparing ASR across
  benchmarks requires identical scoring and threat-model assumptions.
- **HarmBench** — Mazeika et al. 2024's standardized red-teaming
  framework: 510 behaviors, 18 attack methods, 33 target models /
  defenses. Adopted by AISI for pre-deployment evaluations.
  `[harmbench2024 §abstract]`
- **JailbreakBench** — Chao et al. 2024's open robustness benchmark:
  100 target behaviors aligned with OpenAI usage policies,
  standardized threat model / prompts / scoring, public leaderboard,
  repository of jailbreak artifacts. Co-canonical with HarmBench.
  `[chao2024-jailbreakbench §abstract]`
- **Crescendo attack** — Russinovich et al. 2024 multi-turn
  gradual-escalation jailbreak: 29-71% ASR improvement over single-
  turn baselines on AdvBench. Documents that single-turn evals
  systematically underestimate the multi-turn attack surface.
- **AI Safety Level (ASL)** — Anthropic's Responsible Scaling Policy
  capability tier system. ASL-1 (no autonomy / low risk), ASL-2
  (current frontier baseline), ASL-3 (substantially elevated risk;
  specific deployment / security controls required), ASL-4
  (autonomous AI risk; not yet deployed). Each tier specifies
  capability evaluations that must trigger pre-deployment.
- **Responsible Scaling Policy (RSP)** — Anthropic's pre-deployment
  policy framework. Each ASL tier comes with capability triggers
  (which safety evaluations must run) and security commitments
  (which infrastructure / containment requirements must be met). Co-
  canonical with OpenAI's Preparedness Framework and DeepMind's
  Frontier Safety Framework.
- **Dangerous-capability evaluation** — Pre-deployment test for
  specific high-stakes capabilities: autonomous replication,
  autonomous research, cybersecurity attack capability, biological-
  weapons-uplift capability. METR's 77-task suite is the canonical
  cross-vendor reference.
- **Task horizon (METR)** — Capability-quantification metric:
  the distribution of human-time-to-complete for a task. METR's
  empirical scaling-curve work suggests model time-to-complete
  approaches the human-time scale as capabilities improve. Emerging
  field lingua franca for capability measurement.

## Cross-cutting

- **Goodhart's law (in alignment)** — When a proxy metric is
  optimized against, it ceases to be a good measure. The canonical
  failure mode for RLHF: optimizing against a preference model that
  faithfully reflects biased human preferences produces a model
  faithful to the biased proxy, not to the underlying value.
  Foundational to the sycophancy mechanism. `[sharma2023-sycophancy
  §rlhf-causal; kb/excerpts/sharma2023-sycophancy#sec-rlhf-causal]`
- **Mesa-optimizer** — Hubinger et al. 2019 concept: a learned
  optimizer that emerges inside a trained network. The deceptive-
  alignment failure mode is defined for mesa-optimizers; whether
  modern LLMs instantiate the abstraction cleanly is contested.
- **Situational awareness (LLM)** — A model's capacity to reason
  about its own context, training process, and deployment status.
  The Greenblatt 2024 alignment-faking experiment depends on the
  model inferring (from system-prompt cues) that it is in training;
  this is a situational-awareness probe. The Apollo six-probe suite
  similarly leans on the model's awareness of its evaluation
  context. `[greenblatt2024-alignment-faking §setup-wedge;
  kb/excerpts/greenblatt2024-alignment-faking#setup-wedge]`
- **Deliberative alignment** — OpenAI training technique (introduced
  in the o1 stack): train the model to explicitly reason about
  safety policies in its CoT before acting. Conceptually related to
  Constitutional AI's critique step but applied at inference rather
  than training. Apollo's "Stress-Testing Deliberative Alignment for
  Anti-Scheming Training" (2025) reports partial-but-incomplete
  reduction in scheming under this technique.
- **CoT faithfulness** — Whether a model's chain-of-thought trace
  reflects its actual reasoning process. Empirically partial: Lanham
  et al. 2023 / Turpin et al. 2024 show CoT can be unfaithful
  (post-hoc rationalization, label-leakage). Apollo's argument that
  scheming-shaped CoT constitutes evidence of scheming sidesteps
  faithfulness by treating CoT as part of the deployed surface, not
  as ground-truth introspection.
