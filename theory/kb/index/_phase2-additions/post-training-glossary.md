# Post-training glossary additions (Phase 2)

Subagent: post-training (2026-05-04)

Each term gets a 1-3 sentence definition + a paper-key citation.

## RLHF and reward modeling

- **RLHF (Reinforcement Learning from Human Feedback)** — The 3-stage post-training pipeline: SFT → reward-model training on pairwise preferences → PPO with KL anchor against $\pi^{\mathrm{SFT}}$. Canonical reference is InstructGPT `[ouyang2022-instructgpt §3]`.
- **Bradley–Terry preference model** — Assumes pairwise preferences depend on a difference of latent rewards: $p(y_w \succ y_l|x) = \sigma(r(x,y_w) - r(x,y_l))$. The basis for both the RM training loss and DPO `[ouyang2022-instructgpt §3.5; rafailov2023-dpo §4]`.
- **Reward Model (RM, $r_\phi$)** — A neural network trained on Bradley–Terry preferences to score response quality. Initialized from $\pi^{\mathrm{SFT}}$ with a scalar head replacing the LM head. Used as the reward signal for PPO `[ouyang2022-instructgpt §3.5]`.
- **PPO-ptx** — InstructGPT's RLHF objective with an added pre-training-mixture term: $+\gamma\, \mathbb{E}_{\mathrm{pretrain}}[\log\pi_\theta]$. Reduces alignment tax on standard NLP benchmarks `[ouyang2022-instructgpt §3.5]`.
- **KL anchor / KL penalty** — The $\beta \log\pi_\theta(y|x)/\pi^{\mathrm{SFT}}(y|x)$ term in RLHF objectives that prevents the policy from drifting too far from the SFT initialization. Implemented per-token in the reward in InstructGPT-style PPO `[ouyang2022-instructgpt §3.5]`.
- **Alignment tax** — The performance regression on standard NLP benchmarks observed after RLHF. PPO-ptx mitigates it by mixing in pre-training likelihood `[ouyang2022-instructgpt §3.5]`.
- **WARM (Weight-Averaged Reward Models)** — Average parameters of $K$ independently-trained RMs to reduce overoptimization and improve robustness. Adopted in Gemma 3 post-training `[warm-2024]`.
- **WARP** — Alternates RL training with policy weight averaging; companion to WARM at the policy side. Used together in Gemma 3 `[warp-2024]`.
- **Reward overoptimization** — The gap between proxy reward $r_\phi$ and gold reward grows as KL($\pi_\theta\|\pi_{\mathrm{ref}}$) grows; the policy "hacks" the RM, scoring high on $r_\phi$ while degrading on true quality. Affects both RLHF (Gao et al. 2023) and DPO-class methods `[rlhf-overopt-2024]`.
- **Reward hacking** — Policy exploits an artifact of the reward signal that does not correspond to true quality (e.g., verbosity, sycophancy, format-shortcuts). Distinct from overoptimization in that hacking can occur even at low KL `[deepseek-r1 §2.2]`.

## DPO and offline preference optimization

- **DPO (Direct Preference Optimization)** — Closed-form, classification-loss reformulation of RLHF: trains the policy directly on preference pairs by inverting the closed-form RL optimum. No reward model, no on-policy sampling, no PPO loop `[rafailov2023-dpo §4 Eq.7]`.
- **Implicit reward** — In DPO, the policy log-ratio $\hat{r}_\theta(x,y) = \beta\log\pi_\theta(y|x)/\pi_{\mathrm{ref}}(y|x)$ that plays the role of a reward function. The policy is trained on preferences over implicit-reward differences `[rafailov2023-dpo §4]`.
- **DPO reparameterization** — The closed-form mapping between optimal policy and reward (Eq. 4 of Rafailov 2023): $\pi^*(y|x) \propto \pi_{\mathrm{ref}}(y|x)\exp(r(x,y)/\beta)$. Inverting this gives the implicit reward `[rafailov2023-dpo §4 Eq.4-5]`.
- **DPO length bias** — DPO-trained models tend to produce longer outputs than the reference because long responses are easier to satisfy with high log-ratios. Addressed by SimPO's length normalization `[meng2024-simpo §3-length]`.
- **IPO (Identity Preference Optimization)** — DPO variant replacing log-sigmoid loss with squared loss on the preference gap, addressing DPO's overconfidence pathology `[azar2023-ipo]`.
- **KTO (Kahneman-Tversky Optimization)** — Operates on single-rating data $(x,y,\mathrm{label}\in\{\text{good},\text{bad}\})$ instead of pairwise preferences. Asymmetric prospect-theoretic loss with loss aversion `[ethayarajh2024-kto]`.
- **ORPO (Odds Ratio Preference Optimization)** — Combines SFT and preference losses in a single objective; no reference model, no SFT stage. Trains from base directly via log-odds ratio of SFT loss between chosen and rejected responses `[hong2024-orpo]`.
- **SimPO (Simple Preference Optimization)** — Reference-free DPO variant using length-normalized average log-prob as implicit reward, plus a target reward margin $\gamma$. Eliminates DPO length bias on length-controlled benchmarks `[meng2024-simpo §3]`.
- **Reference-free vs reference-anchored** — Axis along which DPO variants split. Reference-anchored: DPO, IPO, KTO. Reference-free: ORPO, SimPO, CPO. Reference-anchored variants depend on $\pi_{\mathrm{ref}}(y|x)$ in the loss; reference-free variants use only $\pi_\theta$ `[meng2024-simpo §3; hong2024-orpo §2]`.
- **Iterative DPO** — Multi-round pipeline: run DPO, sample new responses, label them, repeat. Llama-3 and Tülu 3 reference recipe `[meta-llama3; tulu3]`.

## Constitutional AI and RLAIF

- **RLAIF (RL from AI Feedback)** — Replace human preference labels in RLHF with AI-generated preference labels, typically from a feedback model conditioned on a written constitution `[bai2022-cai §4]`.
- **Constitutional AI (CAI)** — Anthropic 2022 framework for training harmless assistants without human harm labels. Two stages: SL-CAI (sample → critique under principle → revise → SFT) and RL-CAI (sample pairs → AI judge under principle → train RM → PPO) `[bai2022-cai §3-4]`.
- **Constitution** — A list of principles ("the assistant should not be racist," etc.) used by the feedback model to evaluate / critique responses in CAI. Typical size: ~16 principles `[bai2022-cai §3]`.
- **SL-CAI** — Supervised learning stage of Constitutional AI. Sample initial responses, critique them under a randomly-chosen principle, revise, and SFT on revised responses `[bai2022-cai §3]`.
- **RL-CAI** — Reinforcement learning stage of Constitutional AI. Generate AI preferences by having a feedback model choose between response pairs under a principle, train a reward model on AI preferences, run PPO `[bai2022-cai §4]`.
- **R-CAI (Reverse Constitutional AI)** — Inverts the constitution for adversarial / toxic data generation; probability-clamped RLAIF for red-team data synthesis (April 2026) `[r-cai-2026]`.

## RLVR and GRPO

- **RLVR (Reinforcement Learning with Verifiable Rewards)** — Post-training paradigm where the reward is a deterministic programmatic verifier (math equality, code unit-tests, format check), not a learned reward model. The policy still uses on-policy RL `[deepseek-r1 §2.2; tulu3]`.
- **Verifiable reward** — A 0/1 reward computed by a programmatic check: $R(x,y) = \mathbb{1}[\mathrm{verify}(x,y)]$. Sparse and terminal `[deepseek-r1 §2.2]`.
- **GRPO (Group Relative Policy Optimization)** — Critic-free PPO variant: replaces the value model with a group-mean reward baseline. Sample $G$ trajectories per prompt, normalize rewards within the group, use the standardized advantage in the PPO loss. Used by DeepSeek-R1, Qwen 3, Tülu 3 RLVR `[shao2024 §4.1; deepseek-r1 §2.1 Eq.1]`.
- **Group-normalized advantage** — In GRPO, $\tilde{A}_i = (R_i - \mathrm{mean}(\{R_j\}))/\mathrm{std}(\{R_j\})$, broadcast to every token of trajectory $o_i$. Replaces GAE-derived per-token advantages `[shao2024 §4.1; deepseek-r1 §2.1 Eq.3]`.
- **Format reward** — In RLVR, a small constant reward for trajectories that conform to a structural template (e.g., emit a `<think>...</think>` block). Keeps the chain-of-thought pathway alive during early training `[deepseek-r1 §2.2]`.
- **DAPO (Decoupled clip and dynamic sAmpling Policy Optimization)** — GRPO improvement with four targeted modifications: Clip-Higher (asymmetric PPO clip), Dynamic Sampling (reject zero-gradient groups), Token-Level Loss (equal-weight tokens regardless of trajectory length), Overlong Reward Shaping (soft penalty for length-truncated trajectories) `[dapo2025 §2]`.
- **Clip-Higher** — DAPO's asymmetric PPO clip $[1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}]$ with $\epsilon_{\text{high}} > \epsilon_{\text{low}}$. Allows larger upward updates on under-probable correct tokens `[dapo2025 §2]`.
- **Dynamic Sampling (DAPO)** — Reject groups where all $G$ trajectories share the same reward, since the GRPO advantage is then zero and the gradient is wasted. Re-sample until each group contains both $R=0$ and $R=1$ trajectories `[dapo2025 §2]`.
- **VAPO (Value-Augmented PPO)** — RL variant that re-introduces a learned value function on top of GRPO's design, with shared backbone and value-loss-clipping. Reportedly outperforms DAPO on long-CoT tasks `[vapo-2025]`.
- **VinePPO** — PPO variant using per-prefix Monte Carlo re-simulation as value estimate. Drops the value network. 3× wall-clock speedup over PPO on MATH/GSM8K `[vineppo-2024]`.
- **REINFORCE++** — A family of REINFORCE-class methods that drop PPO's clip but keep the GRPO group-mean baseline and KL anchor. Whether clip helps in this regime is contested.
- **R1-Zero** — DeepSeek-R1-Zero, the version of R1 trained with **no SFT cold-start** — pure GRPO RL applied directly to the DeepSeek-V3 base. Achieves AIME 2024 pass@1 of 77.9% `[deepseek-r1 §2.3]`.
- **Aha moment (R1-Zero)** — A spike in the model's use of reflection words like "wait" during training, taken as evidence of self-reorganization of reasoning structure `[deepseek-r1 §2.3]`.
- **Pass@1 / Pass@K** — Pass@K is the probability that at least one of $K$ independent samples from the model passes the verifier. Pass@1 is the single-sample success rate. RLVR primarily improves pass@1 (sampling efficiency) `[rlvr-limits-2025]`.
- **RLVR sampling-efficiency interpretation** — Empirical finding (NeurIPS 2025 oral, Yue et al.) that RLVR improves pass@1 on problems the base model can already solve at higher K, but does not expand the set of problems solvable at any K. The capability ceiling is set by pre-training `[rlvr-limits-2025]`.

## SFT-specific

- **Rejection sampling** — Llama-3 SFT quality gate: sample $N$ candidates from the model, score by reward model, keep top-$k$, SFT on those. Used as a training-loop stage in modern multi-round pipelines `[meta-llama3]`.
- **MAGPIE** — Alignment data synthesis from scratch: prompt aligned LLMs with only their instruction-template prefix (no question), have them fabricate the question first, then sample the answer. Near-zero-cost SFT data generation `[magpie-2024]`.
- **Cold-start SFT (R1 sense)** — A small SFT stage on long-CoT examples preceding the main RL stage. Provides a reasoning-format and language-quality regularizer for the subsequent RL. Distinct from full SFT pipelines `[deepseek-r1 §3]`.
- **Reasoning-distillation SFT** — Training small models (1.5B–32B) on reasoning trajectories generated by a strong reasoning teacher (R1, o1) via pure SFT with no RL. The 800K-sample DeepSeek-R1 distillation set is the canonical example `[deepseek-r1 §4]`.
- **Thinking / non-thinking modes (Qwen 3 sense)** — Two distinct response styles trained with separate SFT data and selectable at inference via system prompt: "thinking" emits a `<think>...</think>` block, "non-thinking" produces direct answers `[qwen3]`.

## Pipeline conventions

- **Tülu 3 pipeline** — The canonical 4-stage open post-training reference: data curation → SFT → DPO → RLVR. Tülu 3 405B (Jan 2025) scales RLVR to 405B parameters `[tulu3]`.
- **Iterative SFT-DPO** — Multi-round pipeline used by Llama-3 and Tülu 3: SFT, then sample candidates, run rejection sampling, run DPO, repeat with the new policy as starting point `[meta-llama3]`.
