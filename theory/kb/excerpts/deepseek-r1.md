---
paper_key: deepseek-r1
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
authors: DeepSeek-AI
year: 2025
venue: "arXiv (Nature 2025)"
arxiv: 2501.12948
local_pdf: theory/sources/papers/deepseek-r1.pdf
type: excerpts
note: |
  Most quotes (§2.1 GRPO Eq.1/Eq.3, §2.1 hyper-params, §2.3 AIME, §3
  pipeline) were verified verbatim from the v2 PDF by an earlier Phase
  2 subagent (2026-05-04). PDF is no longer present in
  theory/sources/papers/ at the time of this 2026-05-05 audit;
  re-acquire the PDF when downloads are unblocked. §2 R1-Zero/R1
  description (the prose paragraphs in §sec-2) is paraphrased and
  flagged below.
---

# Excerpts — DeepSeek-AI 2025, "DeepSeek-R1"

## Abstract — pure-RL reasoning, no human reasoning trajectories {#abstract}

> General reasoning represents a long-standing and formidable challenge in
> artificial intelligence. Recent breakthroughs, exemplified by large
> language models (LLMs) and chain-of-thought prompting, have achieved
> considerable success on foundational reasoning tasks. However, this
> success is heavily contingent upon extensive human-annotated
> demonstrations, and models' capabilities are still insufficient for more
> complex problems. Here we show that the **reasoning abilities of LLMs
> can be incentivized through pure reinforcement learning (RL)**,
> obviating the need for human-labeled reasoning trajectories. The proposed
> RL framework facilitates the **emergent development of advanced reasoning
> patterns, such as self-reflection, verification, and dynamic strategy
> adaptation**. Consequently, the trained model achieves superior
> performance on verifiable tasks such as mathematics, coding competitions,
> and STEM fields, surpassing its counterparts trained via conventional
> supervised learning on human demonstrations. Moreover, **the emergent
> reasoning patterns exhibited by these large-scale models can be
> systematically harnessed to guide and enhance the reasoning capabilities
> of smaller models**.

## §2 Approach — R1-Zero vs R1 {#sec-2}

The paper trains two reasoning models on the DeepSeek-V3 base:

- **DeepSeek-R1-Zero** — RL applied directly to the base model with no SFT
  cold-start. Uses GRPO (Shao et al. 2024) with **rule-based verifiable
  rewards**: an accuracy reward (binary correctness as scored by a verifier
  for math/code) plus a format reward (penalizing deviation from the
  `<think>...</think><answer>...</answer>` format).
- **DeepSeek-R1** — A multi-stage pipeline: cold-start SFT on a small set
  of long-CoT examples, then GRPO with verifiable rewards (accuracy +
  language-consistency), then a second SFT stage on rejection-sampled
  reasoning + general data, then a final RL alignment stage.

[Verified from PDF on 2026-05-12] Section structure confirmed against
arXiv v2 PDF: §1 Introduction; §2 DeepSeek-R1-Zero (§2.1 GRPO, §2.2
Reward Design, §2.3 Incentivize Reasoning); §3 DeepSeek-R1 (§3.1
Model-based Rewards, §3.2 Training Details); §4 Experiment; §6
Conclusion. GRPO Eq. (1) sequence-level form verified verbatim in the
shao2024 audit. All KB section anchors (sec-2-1-grpo, sec-2-2-reward,
sec-2-3-aime, sec-2-3-aha, sec-2-1-hp, sec-3-pipeline) map correctly.

## §2.1 GRPO objective (Eq. 1, verified) {#sec-2-1-grpo}

> For each question $q$, GRPO samples a group of outputs $\{o_1, o_2, \cdots, o_G\}$ from the old policy $\pi_{\theta_{old}}$ and then optimizes the policy model $\pi_\theta$ by maximizing the following objective:

$$\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}\!\left[q \sim P(Q),\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)\right] \frac{1}{G}\sum_{i=1}^G \!\left( \min\!\left(\tfrac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i,\; \mathrm{clip}\!\left(\tfrac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right) A_i\right) - \beta\,\mathbb{D}_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}}) \right) \tag{1}$$

with group-normalized advantage

$$A_i = \frac{r_i - \mathrm{mean}(\{r_1, r_2, \cdots, r_G\})}{\mathrm{std}(\{r_1, r_2, \cdots, r_G\})} \tag{3}$$

The key change vs. DeepSeekMath is the **reward is a deterministic verifier** for math and code: $r_i \in \{0,1\}$ indicating whether $o_i$ produces the ground-truth answer or compiles + passes the test cases. This is the operational definition of **RLVR** (RL with Verifiable Rewards).

## §2.2 Rule-based reward, no neural RM {#sec-2-2-reward}

> For DeepSeek-R1-Zero, we employ rule-based rewards to deliver precise feedback for data in mathematical, coding, and logical reasoning domains. Our rule-based reward system mainly consists of two types of rewards: accuracy rewards and format rewards.

> Notably, we abstain from applying neural reward models—whether outcome-based or process-based—to reasoning tasks. This decision is predicated on our observation that neural reward models are susceptible to reward hacking during large-scale reinforcement learning.

## §2.1 GRPO training hyper-parameters (verified) {#sec-2-1-hp}

> To train DeepSeek-R1-Zero, we set the learning rate to 3e-6, the KL coefficient to 0.001, and the sampling temperature to 1 for rollout. For each question, we sample 16 outputs with a maximum length of 32,768 tokens before the 8.2k step and 65,536 tokens afterward. … with training continuing for a total of 10,400 steps, corresponding to 1.6 training epochs.

## §2.3 AIME 15.6% → 77.9% (verified from PDF) {#sec-2-3-aime}

> Figure 1(a) depicts the performance trajectory of DeepSeek-R1-Zero on the AIME 2024 benchmark throughout the RL training process, where the average pass@1 score on AIME 2024 shows a significant increase, jumping from an initial 15.6% to 77.9%. In addition, by leveraging the self-consistency decoding, the model's performance can be further improved, achieving an accuracy of 86.7%. This performance significantly surpasses the average performance across all human competitors.

## §2.3 Self-evolution and "aha moment" {#sec-2-3-aha}

> The increase in thinking time fosters the autonomous development of sophisticated behaviors. Specifically, DeepSeek-R1-Zero increasingly exhibits advanced reasoning strategies such as reflective reasoning and systematic exploration of alternative solutions … Notably, during training, DeepSeek-R1-Zero exhibits an "aha moment", characterized by a sudden increase in the use of the word "wait" during reflections.

## §3 R1 multi-stage pipeline (verified from PDF) {#sec-3-pipeline}

> Although DeepSeek-R1-Zero exhibits strong reasoning capabilities, it faces several issues. DeepSeek-R1-Zero struggles with challenges like poor readability, and language mixing, as DeepSeek-V3-Base is trained on multiple languages, especially English and Chinese. To address these issues, we develop DeepSeek-R1, whose pipeline is illustrated in Figure 2.

> In the initial stage, we collect thousands of cold-start data that exhibits a conversational, human-aligned thinking process. RL training is then applied to improve the model performance with the conversational thinking process and language consistency. Subsequently, we apply rejection sampling and SFT once more. This stage incorporates both reasoning and non-reasoning datasets into the SFT process, enabling the model to not only excel in reasoning tasks but also demonstrate advanced writing capabilities. To further align the model with human preferences, we implement a secondary RL stage designed to enhance the model's helpfulness and harmlessness while simultaneously refining its reasoning capabilities.

## §4 Distillation transfer {#sec-4}

The paper distills R1 into smaller dense models (1.5B, 7B, 14B, 32B, 70B)
via **pure SFT on 800K samples** (600K reasoning + 200K non-reasoning)
generated by R1. No RL is applied to the small distilled models. Results:
distilled-32B matches or exceeds OpenAI's o1-mini on multiple reasoning
benchmarks; distilled-7B competitive with strong baselines at 7B scale.

This result is significant because it shows the reasoning capability
transferred by R1 is **expressible as a (large) SFT dataset** — the
small-model trainer does not need to run RL.
