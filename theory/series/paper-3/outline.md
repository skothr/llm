# Paper 3 — *Reasoning is compute, search, and verification*

**Working title:** *Reasoning is compute, search, and verification: the inference-scaling triangle of frontier reasoning LLMs, 2022–2026.*

**Thesis:** The reasoning-LLM gains observed between 2022 and 2026 are not the product of any single mechanism but of a co-designed *triangle*: (a) **inference-time compute scaling** (chain-of-thought, self-consistency, sampling-based test-time compute); (b) **verifier signals at inference** (process reward models, beam/MCTS tree-search, generative PRMs); (c) **RL on verifiable rewards** (RLVR, GRPO, DAPO) that *uses* those verifier signals as training rewards. Strong reasoning models — DeepSeek-R1, o1/o3-class, Gemini-2.5-thinking, QwQ, Qwen3-thinking — co-design all three legs simultaneously. Treating any one leg in isolation systematically understates both the gains and the open questions.

**Length target:** ~80 pages.

**Audience:** ML researchers and engineers with graduate-level math who want one place to read the synthesized state of reasoning-LLM training and inference. Not a survey of CoT prompting; not a survey of RL; a monograph on the *interaction* of compute, search, and verification.

## Section structure (target ~14 sections)

For each section: **(thesis sentence)** • [page budget] • {primary KB anchors} • {key contradictions to surface}.

### §1 — Introduction: the reasoning triangle (5 pages)

The 2022–2026 reasoning gains are best understood as a triangle of co-designed levers: inference-time compute (CoT, sampling), verifier signals (PRMs, tree-search), and RL on verifiable rewards (RLVR/GRPO). We name the three legs, sketch the canonical R1 evidence (AIME 2024 15.6% → 77.9% pass@1, → 86.7% with majority vote `[deepseek-r1 §2.3; kb/excerpts/deepseek-r1#sec-2-3-aime]`), preview where each leg disagrees with the others, and forecast the contradictions §14 catalogues.

- Anchors: `kb/notes/reasoning/{chain-of-thought,test-time-compute,inference-time-search,process-supervision,reasoning-training}.md`, `kb/notes/scaling/inference-time-compute-scaling.md`, `kb/notes/post-training/rlvr-and-grpo.md`, brainstorm doc Paper-3 sketch.
- Open question (forwarded to §14): *is the triangle a real co-design, or three independent gains we happen to ship together?*
- Cross-paper: builds on Paper-2 §RL-stage (RLHF→RLVR transition) and Paper-1 §architecture (long-context attention, KV-cache cost of long traces).

### §2 — The CoT phenomenon and its scale-emergence (6 pages)

Chain-of-thought reasoning as the historical entry point: few-shot CoT (Wei 2022 `[wei2022-cot §3.1]`), zero-shot "Let's think step by step" (Kojima 2022 `[kojima2022 §3]`), and the *emergence-with-scale* finding — CoT helps PaLM-540B substantially but does not help (and can hurt) sub-100B models on GSM8K `[wei2022-cot §3.3 fig.4, §6]`. The post-2024 reframing: CoT was the first instance of inference-time compute substituting for parameters, but the field initially read it as "prompting magic" rather than "compute axis."

- Anchors: `kb/notes/reasoning/chain-of-thought.md` §1-2. `kb/excerpts/{wei2022-cot,kojima2022}.md`.
- Math to transcribe: the CoT decoding factorisation $z \sim p_\theta(\cdot|x)$, $y \sim p_\theta(\cdot|x,z)$ (Eq. 1 of `chain-of-thought.md`).
- Tensor-shape note: $z$ is the reasoning trace ($\sim 10^2$–$10^4$ tokens), $y$ the extracted answer; $K$ in-context exemplars; trace-length cost dominates context-window budget.
- Contradiction `[CONTRADICTION]`: faithfulness deferred to §11 — does $z$ causally drive $y$, or is it post-hoc rationalisation?

### §3 — Self-consistency and majority voting (5 pages)

Wang et al.'s self-consistency `[wang2022-self-consistency §3]` is the cheapest leg of the triangle: sample $N$ independent CoTs at $T > 0$ and majority-vote the extracted answers. No verifier needed; no training change. Why it works (the answer manifold is lower-entropy than the trace manifold), why it remains a strong baseline well into the R1 era (R1-Zero AIME goes from 77.9% pass@1 to 86.7% with self-consistency `[deepseek-r1 §2.3; kb/excerpts/deepseek-r1#sec-2-3-aime]`), and where it breaks (open-ended generation, low-cardinality answer space requirement).

- Anchors: `kb/notes/reasoning/test-time-compute.md` §2.1, `kb/notes/reasoning/chain-of-thought.md` §self-consistency. `kb/excerpts/wang2022-self-consistency.md`.
- Math: $\hat y = \mathrm{aggregate}(y^{(1)}, \ldots, y^{(N)})$ with majority-vote $=\arg\max_y \sum_n \mathbb{1}[y^{(n)} = y]$; sample-cost $C \propto N$, embarrassingly parallel.
- Contradiction `[CONTRADICTION]`: when does best-of-N + verifier dominate self-consistency, and when not? Snell shows search wins on hard prompts; majority vote remains competitive at tight latency budgets.

### §4 — Inference-time compute as a scaling axis (anchor section, 7 pages)

Snell et al. 2024 promote test-time compute from "prompting trick" to a *scaling-law-shaped* axis with its own functional form, comparable to Chinchilla on the training side. We transcribe the compute-optimal definition (Eq. 1 of `inference-time-compute-scaling.md` §1.1):

$$\theta^\ast_{q, y^\ast(q)}(N) = \arg\max_\theta \mathbb{E}_{y \sim \mathrm{Target}(\theta, N, q)}\!\left[\mathbb{1}_{y = y^\ast(q)}\right]$$

walk through the difficulty-binning operationalisation (5 quantiles of base-model pass@1) `[snell2024 §3.2; kb/excerpts/snell2024#sec-3-2-difficulty]`, and present the FLOPs-matched headline: on easy and intermediate prompts, test-time compute substitutes for ~14× more pretraining FLOPs; on the hardest prompts it does not `[snell2024 §1, §6; kb/excerpts/snell2024#sec-1-flops-matched]`. Cross-reference Paper-2 §scaling-laws for the structural Chinchilla parallel.

- Anchors: `kb/notes/scaling/inference-time-compute-scaling.md` §1-2 (the anchor note for this paper). `kb/excerpts/snell2024.md`.
- Math: Eq. 1 above; the proposer/verifier decomposition `[snell2024 §2; kb/excerpts/snell2024#sec-2-proposer-verifier]`; the 4× compute-optimal vs naïve best-of-N gain `[snell2024 abstract]`.
- Cross-paper: Paper-2 §Chinchilla provides the training-side analog ($N$ vs $D$ at fixed $C_\text{train}$); this paper's Eq. 1 is the inference-side analog ($\theta$ at fixed $C_\text{test}$).
- Contradiction `[CONTRADICTION]`: the functional form of $L(\theta, N)$ — power-law, log, or something else — is open. s1 shows roughly logarithmic returns to thinking-token count `[s1-2025]`; no closed-form law is yet attested.

### §5 — Process Reward Models: the verifier leg (8 pages, anchor section)

The verifier side of the triangle. Lightman et al.'s PRM800K `[lightman2023-prm800k §3]` introduces process supervision as the contrast to outcome reward models: per-step correctness labels rather than final-answer labels. We formalise both:

$$\mathcal{L}_\text{PRM} = -\mathbb{E}_{(x, s_{1:t}, c_t)}\!\left[c_t \log R^P_\phi(x, s_{1:t}) + (1-c_t)\log(1 - R^P_\phi(x, s_{1:t}))\right]$$

(Eq. 2 of `process-supervision.md`), walk the three trace-aggregation rules (min, product, last-step), and reach OpenAI's headline finding: **min-aggregation** is the strongest aggregator on PRM800K-grade verifiers `[lightman2023-prm800k §4.2]` — a single bad step is fatal. Then walk the post-2023 PRM lineage: Math-Shepherd (automatic step-labelling), ThinkPRM `[thinkprm2025]` (generative PRMs that themselves reason about correctness), R-PRM `[she2025-r-prm]` (reasoning-aware PRM), EDU-PRM `[cao2025-edu-prm]` (educational-reasoning-domain PRM).

- Anchors: `kb/notes/reasoning/process-supervision.md` §1-3 (the anchor note for this section). `kb/excerpts/{lightman2023-prm800k,thinkprm2025,she2025-r-prm,cao2025-edu-prm}.md`.
- Math: Eq. 1-2 of `process-supervision.md` (ORM and PRM losses); the three aggregation rules; PRM-best-of-$N$ vs ORM-best-of-$N$ at matched $N$ on MATH `[lightman2023-prm800k §4 fig.4]`.
- Tensor-shape: PRM scores per-step on prefix $s_{1:t}$, output $\in [0,1]$; aggregation reduces $T$-vector to scalar before search comparison.
- Contradiction `[CONTRADICTION]`: PRM > ORM at the original PRM800K scale, but the gap shrinks for reasoning-trained models (revisited in §12). Generative PRMs claim to dominate discriminative PRMs `[thinkprm2025]`; cross-paper replication remains thin.

### §6 — Tree-search families: BFS, DFS, MCTS (8 pages, anchor section)

Inference-time search is the third decomposition of test-time compute (after sampling and sequential). The taxonomy `[wei2025-tree-search-survey]`: beam over reasoning steps, BFS / level-order, MCTS. We formalise the canonical reasoning-MCTS recurrence (rStar-Math `[rstar-math2025 §3]`):

$$U(s) = Q(s) + c_\text{puct}\, \pi_\theta(s|\text{parent})\, \frac{\sqrt{N_\text{parent}}}{1 + N_s}$$

(the UCT exploration rule from `inference-time-search.md` §2.2). Walk the four MCTS phases (selection / expansion / simulation / backup), then present ReST-MCTS* `[zhang2024-rest-mcts]` as the canonical AlphaZero-style reasoning loop: MCTS rollouts → correct-trajectory harvest → SFT on harvested traces → re-train PRM → repeat. Then MCTS-RAG `[hu2025-mcts-rag]` as the retrieval-extended cousin: at each search node, the proposer can call into a retrieval index, blending knowledge access with stepwise reasoning.

- Anchors: `kb/notes/reasoning/inference-time-search.md` §1-3 (the anchor note for this section). `kb/excerpts/{rstar-math2025,zhang2024-rest-mcts,wei2025-tree-search-survey,hu2025-mcts-rag}.md`.
- Math: UCT formula above; cost analysis $O(N_\text{rollout} \cdot D \cdot L_\text{step})$; the AlphaZero structural parallel.
- Tensor-shape note: MCTS state = node in trace tree; $V_\phi$ = PRM over partial prefix; full rollout cost per simulation dominates expansion cost.
- Contradiction `[CONTRADICTION]`: BFS-with-PRM vs MCTS — when does the extra UCT machinery pay for itself vs naïve beam? Snell §4 has BFS competitive with MCTS at fixed budget on math, contra rStar's MCTS-headline framing.

### §7 — The RLVR paradigm: GRPO and DAPO (anchor section, 8 pages)

The third leg of the triangle: RL with verifiable rewards. We transcribe the reward definition (Eq. of `rlvr-and-grpo.md` §1.1):

$$R_\mathrm{verifier}(x, y) = \mathbb{1}[\,\mathrm{verify}(x, y) = \mathsf{true}\,] \in \{0, 1\}$$

then the GRPO objective in full (Eq. 1 of `rlvr-and-grpo.md` §1.2):

$$\mathcal{J}_\mathrm{GRPO}(\theta) = \mathbb{E}_{q,\, \{o_i\}_{i=1}^G}\!\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\!\left(r_{i,t}\hat{A}_{i,t},\, \mathrm{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat{A}_{i,t}\right) - \beta\, \mathbb{D}_{\mathrm{KL}}[\pi_\theta\|\pi_\mathrm{ref}]\right]$$

with the group-normalised advantage $\hat A_{i,t} = (R_i - \mathrm{mean}(\{R_j\}))/\mathrm{std}(\{R_j\})$ (Eq. 2 of `rlvr-and-grpo.md` §1.2) `[shao2024 §4.1; kb/excerpts/shao2024#sec-4-1]`. Why GRPO matters operationally: it removes the value network co-sized with the policy, halving training-time activation memory at the cost of $G$ rollouts per prompt. Then DAPO's four targeted improvements: clip-higher (asymmetric $\epsilon$), dynamic sampling (drop zero-gradient groups), token-level loss (equal-weight every token, not every trajectory), overlong reward shaping `[dapo2025 §2; kb/excerpts/dapo2025#sec-2]`. Empirical headline: DAPO reaches AIME 2024 50 with Qwen2.5-32B-base in roughly half the steps of the GRPO baseline `[dapo2025 §3-headline; kb/excerpts/dapo2025#sec-3-headline]`.

- Anchors: `kb/notes/post-training/rlvr-and-grpo.md` §1-3 (the anchor note for this section), `kb/notes/reasoning/reasoning-training.md` §1. `kb/excerpts/{shao2024,dapo2025,deepseek-r1}.md`.
- Math: verifier reward, GRPO objective and group-normalised advantage, the DAPO modifications.
- Tensor-shape: $G$ trajectories of length $\le L_\max$ per prompt; advantage broadcast token-wise; KL term per step.
- Cross-paper: Paper-2 §post-training-RL situates RLVR as the successor to RLHF; this paper deepens the reasoning-specific story.
- Contradiction `[CONTRADICTION]`: VAPO `[arXiv:2504.05118]` claims a carefully-trained value network beats DAPO on long-CoT; whether the GRPO-vs-PPO advantage is "value models are useless" or "naïve value models are useless" is unresolved.

### §8 — DeepSeek-R1 and the reasoning-trained model family (7 pages)

R1 as the empirical proof point that the triangle composes. Walk the four-stage protocol `[deepseek-r1 §2.3; kb/excerpts/deepseek-r1#sec-2-3]`: cold-start SFT on long-CoT exemplars → GRPO on math/code/STEM with rule-based verifiers (response length grows from ~200 to ~10K tokens, AIME 2024 15.6% → 77.9% `[deepseek-r1 §2.3; kb/excerpts/deepseek-r1#sec-2-3-aime]`) → rejection-sampling SFT on harvested traces → general-domain RL. Why R1-Zero (no cold start) achieves the same reasoning capability with unreadable interleaved-language traces `[deepseek-r1 §2.2.4]`, and what the cold-start adds (legibility, format, regularisation). Then the SFT-distillation track: R1's distilled-into-Qwen-7B/14B/32B variants, and the resulting open-weight reasoning ecosystem (Tülu 3, OpenR1, Qwen3-thinking, QwQ).

- Anchors: `kb/notes/reasoning/reasoning-training.md` §2-3, `kb/notes/post-training/rlvr-and-grpo.md` §2.3. `kb/excerpts/deepseek-r1.md`.
- Math: the long-CoT-compounding-effect framing — accuracy as a monotone-concave function of $\log$ thinking-token budget.
- Cross-paper: Paper-1 §long-context attention is what makes 10K-token reasoning traces tractable in the first place; Paper-2 §SFT bookends the RL stage on both ends.
- Contradiction `[CONTRADICTION]`: AIME 2024 contamination — was a fraction of AIME-style problems plausibly in R1's pretraining corpus? Cross-reference Paper-5 §contamination.

### §9 — Test-time scaling beyond CoT: think-tokens, budget forcing, reasoning architectures (5 pages)

s1 `[s1-2025]` as the minimum recipe: a small model + 1K-example reasoning SFT + "budget forcing" — append `Wait` to extend thinking, or append the answer-prefix to truncate. Reasoning-architecture as a separate axis from base architecture: the model is the same Transformer, but the training+inference protocol designates `<think>...</think>` as a structurally distinct phase from the answer phase. Where this lives in the architecture taxonomy (Paper-1 §reasoning-architectures), and where it lives in the training taxonomy (Paper-2 §post-training).

- Anchors: `kb/notes/reasoning/test-time-compute.md` §2.3 (sequential TTC), `kb/notes/reasoning/reasoning-training.md` §2 (R1 protocol). `kb/excerpts/s1-2025.md`.
- Math: thinking-token-budget $L$ as the explicit compute knob; logarithmic-ish accuracy returns to $L$ on s1 evals.
- Distinction: budget-forcing is an *inference-time* discipline; the cold-start SFT is a *training-time* discipline; both shape the same `<think>` channel.
- Cross-paper: ties Paper-1 (reasoning-as-architecture-element) to Paper-2 (reasoning-as-training-stage).

### §10 — Search-vs-RL under fixed compute (5 pages)

The triangle's central operational tradeoff: at fixed total compute, when do you spend marginal FLOPs on (a) more inference search, (b) more RLVR training, (c) more pretraining? Snell's FLOPs-matched curves give one answer for inference-vs-pretraining `[snell2024 §6; kb/excerpts/snell2024#sec-1-flops-matched]`; the reasoning-training literature gives partial answers for inference-vs-RL (DAPO's step-efficiency `[dapo2025 §3]`, R1's training-token cost `[deepseek-r1 §2.3]`). Where the curves cross is benchmark-specific and base-model-specific. We present what is known and explicitly demarcate what isn't.

- Anchors: `kb/notes/scaling/inference-time-compute-scaling.md` §2.2-2.3, `kb/notes/post-training/rlvr-and-grpo.md` §2.2. Cross-cite `[snell2024,dapo2025,deepseek-r1]`.
- Math: the FLOPs-matched comparison Snell §6, the rough $C_\mathrm{small}^\mathrm{train} + C^\mathrm{test} \approx C_\mathrm{large}^\mathrm{train} + C_\mathrm{large}^\mathrm{1-pass}$ identity.
- Contradiction `[CONTRADICTION]`: there is no closed-form tradeoff curve; only empirical anecdotes per benchmark. Whether a Chinchilla-style joint $(C_\mathrm{train}, C_\mathrm{RL}, C_\mathrm{test})$ frontier exists is an open frontier question (revisited in §14).

### §11 — CoT faithfulness: does the trace cause the answer? (6 pages)

The faithfulness debate: does the reasoning trace $z$ causally drive the answer $y$, or is it post-hoc rationalisation that happens to be readable? The 2025 wave: Arcuschin et al. "CoT in the wild" `[arcuschin2025-cot-wild]` finds substantial unfaithfulness on naturalistic prompts; Shen et al. FaithCoT-Bench `[shen2025-faithcot-bench]` proposes a dedicated benchmark; Mehta et al. `[mehta2026-faithfulness-scaling]` measures whether faithfulness improves with model scale (prediction: it does *not* monotonically). What unfaithfulness means operationally: the model can be steered to produce the same answer even after the trace is intervention-corrupted (paraphrased, contradicted, truncated). Why this matters cross-paper: Paper-4 §interpretability uses CoT traces as a probe surface; Paper-5 §alignment treats CoT as a monitorable channel — if traces are unfaithful, both uses are compromised.

- Anchors: `kb/notes/reasoning/chain-of-thought.md` §faithfulness, `kb/notes/reasoning/test-time-compute.md` §interpretability-questions. `kb/excerpts/{arcuschin2025-cot-wild,shen2025-faithcot-bench,mehta2026-faithfulness-scaling}.md`.
- Math: faithfulness as $p(y | x, z) \ne p(y | x, z')$ for paraphrased $z'$; intervention-based operationalisation.
- Cross-paper: Paper-4 §CoT-as-probe and Paper-5 §CoT-as-alignment-surface both depend on this section's resolution.
- Contradiction `[CONTRADICTION]`: post-hoc rationalisation vs faithful trace — Arcuschin's evidence is in tension with R1-style reasoning-training framings that treat $z$ as causal. Mehta's scaling-of-faithfulness result (if it replicates) is the field's most consequential 2026 finding for this paper's thesis.

### §12 — Process supervision after R1: do PRMs still help? (5 pages)

R1's headline result was achieved with *terminal* verifier rewards only — no PRM in the RL loop `[deepseek-r1 §2.2; kb/excerpts/deepseek-r1#sec-2-2-reward]`. This raised the question: post-R1, does process supervision still help? The 2025-2026 PRM survey `[zheng2025-prm-survey]` catalogues the landscape: PRMs continue to add value at *inference time* (best-of-N, search-against-PRM), but their value as a *training-time* signal beyond verifier-only RLVR is contested. Generative PRMs (ThinkPRM, R-PRM) appear to dominate discriminative ones; whether the dominance survives reasoning-trained policy distributions is not yet settled. Process-vs-outcome RL: when does the per-step signal pay for the labelling cost?

- Anchors: `kb/notes/reasoning/process-supervision.md` §3-5. `kb/excerpts/{zheng2025-prm-survey,thinkprm2025,she2025-r-prm}.md`.
- Math: PRM-best-of-N vs ORM-best-of-N at matched $N$ on reasoning-trained models — empirical curves, no clean theory.
- Contradiction `[CONTRADICTION]`: the PRM800K-era "PRM > ORM" result `[lightman2023-prm800k §4 fig.4]` is the foundational evidence; whether it generalises to R1-class models post-2025 is the survival question. The 2025 survey reports mixed empirics.

### §13 — Frontier-2026 snapshot: which model uses which combination (5 pages)

A 2-page snapshot table of the leading reasoning models — DeepSeek-R1, OpenAI o1/o3, Gemini 2.5-thinking, Claude 3.7-extended-thinking, Qwen3-thinking, QwQ — coded by which of the three legs each ships and at what budget. Categorisation: R1-style (long-CoT + RLVR + verifier-only), o1-style (RL-trained thinking with proprietary mix), hybrid think/non-think (Gemini 2.5's thinkingBudget API, OpenAI's reasoning-effort levels). Where they agree (RLVR is universal at frontier; long-CoT is universal; PRMs are absent from headline RL loops; thinking-budget exposure is the productionised form). Where they diverge (training-data composition, reasoning-architecture details, search vs no-search at inference).

- Anchors: `kb/index/timeline.md` 2024–2026 entries, `kb/notes/scaling/inference-time-compute-scaling.md` §3.3. `kb/excerpts/{deepseek-r1,s1-2025}.md` plus model-card papers.
- Output: a 2-page snapshot table that becomes the most-screenshotted figure of the paper.
- Cross-paper: Paper-1 §13's frontier-2026 architecture snapshot is the analog on the architecture side; this paper's table is the reasoning-protocol side.

### §14 — Contradictions live and open frontiers (4 pages)

Concentrated discussion of `[CONTRADICTION]` markers from §2-§13:

1. **PRM utility post-R1** (§12): do PRMs survive the reasoning-trained-model era as a training signal, or only as inference verifiers? Evidence is mixed; the 2025 PRM survey is the most comprehensive catalogue but does not settle the question.
2. **CoT faithfulness scaling** (§11): does faithfulness improve, plateau, or degrade with model scale and with reasoning-training intensity? Mehta et al. is the headline 2026 result; replication is sparse.
3. **RLVR generalisation vs memorisation** (§7-§8): the Yue et al. 2025 result `[rlvr-limits-2025]` — RLVR sharpens pass@1 within base-model competency but does not expand pass@K — vs the R1 narrative of RL-elicited reasoning. The 2026 consensus is leaning toward Yue with caveats; not closed.
4. **Search-vs-RL under fixed compute** (§10): no closed-form tradeoff exists. Whether one will emerge as a Chinchilla-style joint frontier is an open research direction.
5. **AIME contamination** (§8): the headline R1 jump may be partially attributable to pretraining contamination; cross-paper to Paper-5 §contamination, where the methodology of contamination-detection itself is contested.
6. **MCTS vs beam at matched budget** (§6): rStar-Math headline framings vs Snell §4 evidence that BFS-with-PRM is competitive on math; whether the extra MCTS machinery pays for itself is benchmark-specific.

Closing thesis: the three legs of the triangle are real and co-designed in frontier 2026 reasoning models. The open questions are not "is reasoning real" but "which combination of legs is optimal at which (compute, latency, domain) operating point" — and the answer will not be closed-form for at least another model generation.

- Anchors: `kb/index/contradictions.md` § reasoning (14 contradictions surfaced).
- Cross-paper: Paper-4 §faithfulness and Paper-5 §AIME-contamination both backreference §11 and §14 of this paper.

## What this outline commits to

- **Math is verbatim from PDFs.** Every equation cited with a `kb/excerpts/<key>#<anchor>` dual-citation. The anchor papers are `shao2024` (GRPO objective), `snell2024` (Eq. 1, FLOPs-matched), `lightman2023-prm800k` (PRM/ORM losses, min-aggregation), `rstar-math2025` (UCT formula), `deepseek-r1` (R1 four-stage protocol, AIME numbers), `dapo2025` (four DAPO modifications).
- **Tensor shapes everywhere relevant.** Trace length $L \in [10^2, 10^4]$, group size $G \in [8, 64]$, search depth $D$, branching factor $B$, MCTS rollout count $N_\mathrm{rollout}$, thinking-budget $L$.
- **Contradictions are first-class.** Each `[CONTRADICTION]` gets explicit treatment in §14, not buried in body. Six top-level contradictions, each tied to specific 2025-2026 papers.
- **No analogy laundering.** `[INTUITION]` and `[ANALOGY]` markers preserved from the KB notes; the MCMC analogy for Snell's proposer/verifier decomposition, the AlphaZero analogy for ReST-MCTS*, and the median-of-rollouts analogy for GRPO all return to canonical math.

## Cross-paper threads

- **RLVR** is shared with Paper-2 §post-training-RL. Paper-2 establishes the RLHF→RLVR transition as a stage of the training pipeline; Paper-3 deepens the reasoning-specific objective (GRPO Eq. 1, DAPO modifications) and the long-CoT-compounding empirics.
- **CoT faithfulness** is shared with Paper-4 (interpretability: CoT as a probe surface) and Paper-5 (alignment: CoT as a monitorable channel). Paper-3 §11 is the load-bearing treatment; Papers 4-5 cite forward.
- **Process supervision** is shared with Paper-5 (PRMs as evaluation tools). Paper-3 §5 and §12 are the load-bearing treatments; Paper-5 cites forward.
- **Inference-time compute scaling** is the analog of training-time scaling (Paper-2 §scaling-laws); §4 of this paper makes the structural parallel explicit.
- **Long-context attention** (Paper-1 §long-context) is the architectural precondition for 10K-token reasoning traces; cited in §8.

## Sections by writing-pass parallelism

- Pass A (independent, easy parallelize): §2 CoT phenomenon, §3 self-consistency, §9 budget-forcing, §11 faithfulness, §12 PRM-survival.
- Pass B (depends on §4 Snell setup or §7 GRPO setup): §5 PRMs, §6 tree-search, §8 R1, §10 search-vs-RL.
- Pass C (synthesis, last): §1 intro, §13 frontier snapshot, §14 contradictions.
- Pass D: §4 Snell anchor and §7 GRPO anchor (the two math-heavy anchor sections; depend on the rest stabilising notation).

5 parallel section-subagents on Pass A; 4 on Pass B (after §4 and §7 land in Pass D); 3 on Pass C (after Pass B lands).
