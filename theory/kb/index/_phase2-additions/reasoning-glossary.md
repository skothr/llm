# Glossary additions — reasoning/ area

## Chain-of-thought and prompting

- **Chain-of-thought (CoT)** — A prompting and decoding pattern in which the model produces an intermediate sequence of reasoning steps $z$ before emitting a final answer $y$. In few-shot CoT (Wei 2022), the prompt contains exemplars with hand-written reasoning traces; in zero-shot CoT (Kojima 2022), the trigger phrase "Let's think step by step" elicits the same behaviour without exemplars. `[wei2022-cot §1; kojima2022 §3]`
- **Few-shot CoT** — The Wei 2022 exemplar-based variant. The prompt contains $K$ demonstrations $(x_i, z_i, y_i)$ before the query. CoT gain over direct prompting is **emergent at scale** — sub-100B models often do not benefit. `[wei2022-cot §3.3]`
- **Zero-shot CoT** — The Kojima 2022 trigger-phrase variant. Two-stage decode: reasoning extraction + answer extraction. Recovers a substantial fraction of few-shot CoT gain at instruction-tuned scale without exemplars. `[kojima2022 §3]`
- **Self-consistency** — Sample $N$ reasoning traces at temperature $T > 0$, majority-vote over extracted answers. The simplest test-time-compute strategy. Gains: GSM8K +17.9%, SVAMP +11.0% over greedy CoT. `[wang2022-self-consistency §3]`
- **Trained CoT** — The behaviour of a reasoning-trained model (o1, R1, QwQ) that emits long reasoning traces unconditionally, without prompt scaffolding. Distinct from prompted CoT in that the trace is the model's natural output distribution. `[deepseek-r1 §2]`
- **CoT faithfulness** — The extent to which the trace $z$ causally drives the answer $y$ versus being post-hoc rationalisation. Measured by counterfactual perturbation. Thinking models report rates ~0.04–0.14% post-hoc; non-thinking models ~7–13%. `[chen2026-faithfulness-scaling, arXiv 2601.06423; chen2025-faithcot-bench, arXiv 2510.04040]`

## Test-time compute

- **Test-time compute (TTC)** — The inference-side compute spent on a single problem instance to improve answer quality, beyond a single greedy forward pass. Recognised in 2024 as a co-equal scaling axis with training compute. `[snell2024 §1]`
- **Compute-optimal TTC** — The strategy maximising expected verifier score at fixed inference compute $C$. Snell et al. 2024 show the optimal mix is problem-difficulty-dependent: easy → short single trace; medium → sampling-based; hard → search. `[snell2024 §3, §5]`
- **Sampling-based TTC** — Draw $N$ independent traces, aggregate. Three rules: self-consistency (majority vote), best-of-$N$ + ORM, best-of-$N$ + PRM. Linear in $N$. `[wang2022-self-consistency; lightman2023-prm800k §4]`
- **Search-based TTC** — Tree expansion over partial reasoning prefixes, directed by a learned value function. Beam, BFS, MCTS variants. See `kb/notes/reasoning/inference-time-search.md`. `[snell2024 §4; rstar-math2025 §3]`
- **Sequential-compute TTC / extended thinking** — Hold $N = 1$ but emit a much longer single trace. The o1/R1/QwQ paradigm. Compute scales super-linearly in trace length due to KV-cache re-attention. `[deepseek-r1 §2.3]`
- **Budget forcing** — The s1 2025 mechanism. When the model emits its end-of-thinking token before the budget is reached, suppress it and append the continuation token "Wait" to force more reasoning; hard-truncate to enforce shorter budgets. `[s1-2025 §3.2]`
- **Thinking budget** — Productised TTC control: API-level integer (Gemini 2.5 `thinkingBudget`), effort level (OpenAI o-series), or token cap (Anthropic thinking block). `[gemini2-5 §3]`

## Process supervision and reward models

- **Outcome Reward Model (ORM)** — A scoring function $R^O_\phi: (x, z, y) \to [0, 1]$ trained on (prompt, complete solution, outcome correctness) triples. Sparse signal — one label per trace. `[lightman2023-prm800k §2.1]`
- **Process Reward Model (PRM)** — A scoring function $R^P_\phi: (x, s_{1:t}) \to [0, 1]$ trained on per-step correctness labels. Dense signal — one label per step. Outperforms ORM at matched best-of-$N$ on MATH. `[lightman2023-prm800k §2.2, §4]`
- **PRM800K** — OpenAI's released dataset of $\approx 800{,}000$ human step-correctness labels on GPT-4-generated MATH solutions. The foundational reference dataset for PRM training. `[lightman2023-prm800k §3]`
- **Aggregation rule (PRM)** — How per-step PRM scores are combined to a trace score. Min, product, last-step. **Min** is the strongest aggregator at PRM800K-grade. `[lightman2023-prm800k §4.2]`
- **Discriminative PRM** — A classifier-style PRM: outputs a probability of step correctness in one feed-forward pass. Cheap to evaluate; classical Math-Shepherd / OpenAI PRM lineage. `[lightman2023-prm800k §2.2]`
- **Generative PRM (ThinkPRM)** — A small reasoning-LLM verifier that emits its own short CoT evaluating each step before producing a verdict. Trained on 1% of PRM800K, beats full-PRM800K discriminative PRMs on ProcessBench / AIME 2024. `[thinkprm2025 §3]`
- **Process Preference Model (PPM)** — rStar-Math's variant: trained on preferences over MCTS-discovered step pairs, rather than absolute step labels. Avoids naïve step-score annotation. `[rstar-math2025 §3]`
- **ProcessBench** — Standard benchmark for PRM step-error detection (identify the first incorrect step in a partial trace). `[arXiv 2412.06559]`
- **Best-of-$N$ + verifier** — Sample $N$ traces, score each with a verifier (ORM, PRM, or self-consistency baseline), select the highest-scored. Canonical sampling-based TTC. `[lightman2023-prm800k §4]`

## Reasoning training (RL with verifiable rewards)

- **RLVR (RL with Verifiable Rewards)** — RL post-training where the reward is computed mechanically from the emitted answer (regex match for math, unit-test execution for code, proof-checker for formal proofs), not from a learned reward model. The defining feature of post-2024 reasoning training. `[deepseek-r1 §2.2.1]`
- **GRPO (Group Relative Policy Optimisation)** — Critic-free PPO variant (Shao 2024). Per-prompt group of $G$ samples; advantage is each sample's reward minus group mean, divided by group std. Eliminates the value-network requirement. `[shao2024 §4.1; deepseek-r1 §2.1]`
- **DAPO** — ByteDance's GRPO variant. Four modifications: clip-higher (asymmetric clip), dynamic sampling (drop zero-variance groups), token-level loss (vs sequence-level), overlong reward shaping. AIME 2024 50pts on Qwen2.5-32B. `[dapo2025 §2]`
- **Cold-start SFT** — A small-scale ($\sim 10^4$ examples) SFT pass on long-CoT exemplars, run before RL to fix readability problems of pure-RL outputs (R1-Zero). Does not change asymptotic capability but improves trace coherence. `[deepseek-r1 §2.2.4]`
- **Rejection-sampling SFT** — Generate candidate solutions with the current policy, filter by verifier (and optional quality filter), SFT on the survivors. Used as Stage 3 of the R1 pipeline. `[deepseek-r1 §2.3.3]`
- **Distillation (reasoning)** — Use a strong teacher's RL-trained traces as SFT data for smaller students. R1 demonstrates this beats running RL directly on a small model. `[deepseek-r1 §3.2]`
- **Capability ceiling (RLVR)** — The empirical finding that RLVR improves $\mathrm{pass}@1$ on problems the base model could already solve at $\mathrm{pass}@k$ for some $k$, but does not expand the set of solvable problems. Set at pre-training. `[rlvr-limits-2025 §3]`
- **Reward / process hacking** — Failure mode in which the policy learns to exploit imperfections in the verifier (or PRM) rather than improving on the underlying task. `[deepseek-r1 §2.2.4; arXiv 2505.00551]`
- **Hybrid think / non-think model** — A single model that toggles between long-CoT and short-answer modes via system-prompt instruction. Productisation step of reasoning training. `[qwen3 §4; deepseek V3.1+ release notes]`

## Inference-time search

- **Inference-time search** — TTC family that constructs and explores a tree (or DAG) of partial reasoning prefixes, using a verifier or value function to direct expansion. Beam / BFS / MCTS variants. `[rstar-math2025 §3]`
- **Reasoning-MCTS** — Monte Carlo Tree Search where actions are next-step reasoning generations and the value function is a PRM. Selection-expansion-simulation-backup loop, UCT-style exploration. `[rstar-math2025 §3]`
- **Generator-discriminator mutual verification** — rStar-Math's two-SLM design: policy SLM generates candidate next-steps, reward / discriminator SLM scores them. Both small (~7B); together rival o1-mini on MATH. `[rstar-math2025 §3]`
- **ReST-MCTS*** — Self-training loop using MCTS to generate correct trajectories, then SFT on those, then update the value function. Structurally analogous to AlphaZero self-play. `[rest-mcts-2024, arXiv 2406.03816]`
- **Lookahead beam** — Beam search variant: from each beam candidate, perform short rollouts and score the rolled-out outcomes, then prune. The form Snell 2024 uses for compute-optimal mix experiments. `[snell2024 §4]`
- **MCTS-RAG** — MCTS extended over retrieval decisions in addition to reasoning steps. Bridges search-based TTC and retrieval-augmented generation. `[mcts-rag-2025, arXiv 2503.20757]`
