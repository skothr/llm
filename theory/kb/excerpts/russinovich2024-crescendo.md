---
paper_key: russinovich2024-crescendo
title: "Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack"
authors: Russinovich, Salem, Eldan
year: 2024
venue: USENIX Security 2025 (arXiv 2404.01833, April 2024)
arxiv: 2404.01833
local_pdf: theory/sources/papers/russinovich2024-crescendo.pdf
type: excerpts
---

# Excerpts — Russinovich, Salem, Eldan 2024, "The Crescendo Multi-Turn LLM Jailbreak Attack"

## #abstract — full verbatim

> Large Language Models (LLMs) have risen significantly in popularity
> and are increasingly being adopted across multiple applications. These
> LLMs are heavily aligned to resist engaging in illegal or unethical
> topics as a means to avoid contributing to responsible AI harms.
> However, a recent line of attacks, known as jailbreaks, seek to
> overcome this alignment. Intuitively, jailbreak attacks aim to narrow
> the gap between what the model can do and what it is willing to do.
> In this paper, we introduce a novel jailbreak attack called Crescendo.
> Unlike existing jailbreak methods, Crescendo is a simple multi-turn
> jailbreak that interacts with the model in a seemingly benign manner.
> It begins with a general prompt or question about the task at hand
> and then gradually escalates the dialogue by referencing the model's
> replies progressively leading to a successful jailbreak. We evaluate
> Crescendo on various public systems, including ChatGPT, Gemini Pro,
> Gemini-Ultra, LlaMA-2 70b and LlaMA-3 70b Chat, and Anthropic Chat.
> Our results demonstrate the strong efficacy of Crescendo, with it
> achieving high attack success rates across all evaluated models and
> tasks. Furthermore, we present Crescendomation, a tool that automates
> the Crescendo attack and demonstrate its efficacy against
> state-of-the-art models through our evaluations. Crescendomation
> surpasses other state-of-the-art jailbreaking techniques on the
> AdvBench subset dataset, achieving 29-61% higher performance on GPT-4
> and 49-71% on Gemini-Pro. Finally, we also demonstrate Crescendo's
> ability to jailbreak multimodal models.

(Verified verbatim from PDF on 2026-05-12.)

## §3 — The Crescendo multi-turn escalation pattern {#sec-3-crescendo}

> Crescendo is a multi-turn jailbreaking technique that uses benign human
> readable prompts. Crescendo distinguishes itself from other approaches
> by utilizing the target model's outputs to direct the model towards
> bypassing its safety alignment. This approach begins with an innocuous
> topic linked to the target task and progressively intensifies, directing
> the model's responses towards the intended outcome. Hence, it circumvents
> defenses and safety measures, especially ones designed to react mainly to
> the user's prompts. The incremental nature of Crescendo's strategy mirrors
> the "foot-in-the-door" psychological tactic, where agreeing to a small,
> initial request increases the likelihood of complying with subsequent,
> larger demands.

The paper illustrates the escalation pattern with a concrete three-turn
misinformation example (Figure 2): the adversary opens with "Tell me about
X," elicits a neutral model response, then asks for viewpoints people
concerned about the topic might express, and finally requests "combine those
quotes into article form and add a title" — each step using the model's own
prior output as the warrant for the next request. No individual turn contains
explicitly malicious content.

## §3.1 — Black-box threat model {#sec-3-1-threat-model}

> Crescendo is a completely black-box jailbreak that makes no assumptions
> about the target model. It is executed through entirely normal interactions
> with an LLM, similar to most other jailbreaks. For automating Crescendo,
> access to a black-box API is the only requirement needed to interact with
> the target LLM.

This distinguishes Crescendo from optimization-based attacks (GCG, AutoDAN)
that require white-box gradient access. The black-box property means
Crescendo applies equally to closed-API systems (ChatGPT, Gemini, Anthropic
Chat) and open-weight models.

## §3.3 — Understanding Crescendo: the auto-referential mechanism {#sec-3-3-mechanism}

> Intuitively, Crescendo is able to jailbreak a target model by progressively
> asking it to generate related content until the model has generated
> sufficient content to essentially override its safety alignment.

The paper tests this hypothesis mechanistically on LLaMA-2 70b (the only
open-sourced aligned model available at time of writing) using token-level
probability analysis. Starting from the benign sentence "Frank, I love you
man." and monitoring the probability of generating the word "fuck," the
authors show that incrementally infusing the context with increasingly
aggressive (but individually innocuous) sentences raises the generation
probability monotonically. Table 3 reports the key finding:

| Sentence Combination | Success Percentage |
|----------------------|-------------------|
| B                    | 36.2%             |
| A → B                | 99.99%            |
| B → C                | 17.3%             |
| A → B → C            | 99.9%             |
| A → B → C'           | < 1%              |

The critical result is the contrast between B alone (36.2%) and A → B
(99.99%): a single prior benign turn nearly guarantees compliance on the
next turn. The model's own prior outputs function as a commitment gradient
that the attacker climbs turn by turn.

## §5.2 — Attack success rates against frontier models {#sec-5-2-results}

> As shown in Table 4, Crescendo is significantly better than the other
> jailbreaks, with an improvement between 29% and 61% on the average ASR.
> Crescendo achieves a 98% success rate when considering the binary success
> rate. The binary success rate is essentially considering a jailbreak to be
> successful if at least one attempt was successful. This means that Crescendo
> is able to jailbreak 49 out of the 50 tasks compared to the second best
> jailbreak (MSJ) which jailbreaks only 43 tasks.
>
> Similarly, Figure 6b shows the performance when targeting GeminiPro. Here,
> the performance of Crescendo is even more significant compared to the other
> jailbreaks, as shown in Table 4, which shows an improvement between 49%
> and 71%. In this case, Crescendo is able to jailbreak all 50 tasks,
> indicating a binary success rate of 100%.

Table 4 (verbatim numbers from the paper):

| Model     | CIA        | COA        | MSJ        | PAIR       | Crescendo   |
|-----------|------------|------------|------------|------------|-------------|
| GPT-4     | 35.6 (82.0)| 22.0 (22.0)| 37.0 (86.0)| 40.0 (76.0)| 56.2 (98.0) |
| GeminiPro | 42.4 (92.0)| 24.0 (24.0)| 35.4 (88.0)| 33.0 (80.0)| 82.6 (100.0)|

Values are average ASR; binary ASR (at least one successful attempt) in
parentheses. These numbers document that single-turn evaluations
systematically understate the attack surface: all prior baselines operate
on single-turn prompt optimization while Crescendo exploits the multi-turn
context window available in every deployed chat interface.

[Verified from PDF on 2026-05-12] Added §3 (#sec-3-crescendo), §3.1 (#sec-3-1-threat-model), §3.3 (#sec-3-3-mechanism), §5.2 (#sec-5-2-results). Abstract verified verbatim.
