---
paper_key: bai2022-cai
title: "Constitutional AI: Harmlessness from AI Feedback"
authors: Bai, Kadavath, Kundu, Askell, Kernion, Jones, Chen, Goldie, Mirhoseini, McKinnon, Chen, Olsson, Olah, Hernandez, Drain, Ganguli, Li, Tran-Johnson, Perez, Kerr, Mueller, Ladish, Landau, Ndousse, Lukosuite, Lovitt, Sellitto, Elhage, Schiefer, Mercado, DasSarma, Lasenby, Larson, Ringer, Johnston, Kravec, El Showk, Fort, Lanham, Telleen-Lawton, Conerly, Henighan, Hume, Bowman, Hatfield-Dodds, Mann, Amodei, Joseph, McCandlish, Brown, Kaplan
year: 2022
venue: "Anthropic Tech Report"
arxiv: 2212.08073
local_pdf: theory/sources/papers/bai2022-cai.pdf
type: excerpts
note: "Verbatim from arXiv abstract page; full pipeline description from cross-referenced sources (C3AI arXiv:2502.15861 §2; Llama 2 §3.4 description of RLAIF; survey arXiv:2509.16679). PDF not yet downloaded."
---

# Excerpts — Bai et al. 2022, "Constitutional AI"

## Abstract — RLAIF replaces human preference labels {#abstract}

> As AI systems become more capable, we would like to enlist their help to
> supervise other AIs. We experiment with methods for training a harmless
> AI assistant through self-improvement, **without any human labels
> identifying harmful outputs**. The only human oversight is provided
> through a list of rules or principles, and so we refer to the method as
> **'Constitutional AI'**. The process involves both a supervised learning
> and a reinforcement learning phase. In the supervised phase we sample
> from an initial model, then generate self-critiques and revisions, and
> then finetune the original model on revised responses. In the RL phase,
> we sample from the finetuned model, use a model to evaluate which of the
> two samples is better, and then train a preference model from this
> dataset of AI preferences. We then train with RL using the preference
> model as the reward signal, i.e. we use **'RL from AI Feedback' (RLAIF)**.
> As a result we are able to train a harmless but non-evasive AI assistant
> that engages with harmful queries by explaining its objections to them.
> Both the SL and RL methods can leverage chain-of-thought style reasoning
> to improve the human-judged performance and transparency of AI decision
> making.

## §3 SL-CAI — supervised stage with self-critique + revision {#sec-3}

The SL-CAI pipeline (paper §3, reproduced in C3AI §2 and survey
arXiv:2509.16679 §3):

1. Sample an initial response $y_0$ from a **helpful-only RLHF model**
   $\pi^{\mathrm{H}}$ on a red-team prompt $x$.
2. Sample a randomly-chosen **constitutional principle** $c_k$ from a
   list of $\sim 16$ rules ("the assistant should not be racist," "the
   assistant should not provide instructions for synthesizing
   bioweapons," etc.).
3. Prompt $\pi^{\mathrm{H}}$ to **critique** $y_0$ in light of $c_k$,
   producing critique $\kappa$.
4. Prompt $\pi^{\mathrm{H}}$ to **revise** $y_0$ given $(x, y_0, \kappa,
   c_k)$, producing revision $y_1$.
5. Iterate steps 2–4 if desired, then collect $(x, y_{\text{final}})$
   into the SL-CAI fine-tuning set.
6. Fine-tune $\pi^{\mathrm{H}}$ on the SL-CAI set with cross-entropy:
   $\pi^{\mathrm{SL-CAI}}$.

The result is a model that "self-corrects" toward the constitution
without human harm labels.

## §4 RL-CAI — RLAIF with AI-generated preferences {#sec-4}

The RL stage:

1. Sample a pair of responses $(y_A, y_B)$ from $\pi^{\mathrm{SL-CAI}}$
   on a prompt $x$.
2. Sample a constitutional principle $c_k$.
3. Prompt a **feedback model** $\pi^{\mathrm{FB}}$ (typically the same
   helpful-only model) to choose between $y_A$ and $y_B$ given the
   principle, producing a preference label.
4. Build a preference dataset $\mathcal{D}^{\mathrm{AI}} = \{(x,y_w,y_l)\}$
   from these AI-generated labels.
5. Train a reward model $r_\phi$ on $\mathcal{D}^{\mathrm{AI}}$ via the
   standard Bradley–Terry loss.
6. Train $\pi^{\mathrm{SL-CAI}}$ with PPO against $r_\phi$ as in standard
   RLHF (cf. `kb/excerpts/ouyang2022-instructgpt#sec-3`).

This is the **RLAIF** pipeline: the reward model was trained on AI labels
rather than human ones. Human labels are only used to (a) construct the
helpful-only $\pi^{\mathrm{H}}$ starting point and (b) write the
constitution.

## §5 Chain-of-thought enhancement {#sec-5}

Both stages can use **chain-of-thought** reasoning before producing the
final critique/preference. The CoT step ("let me think about how this
response relates to principle $c_k$ before judging") is reported to
improve harmlessness scores while preserving helpfulness, presumably
because it externalizes the reasoning and constrains the final answer
more.

## §6 Empirical headline {#sec-6}

The paper reports:
- An RL-CAI'd model is rated by humans as **more harmless and equally
  helpful** as a model trained with human preference labels for harm.
- The trained model is **non-evasive**: it engages with harmful prompts
  by stating objections rather than refusing flatly. This is held up as
  the defining behavioral signature of CAI vs. helpful-only RLHF.

[Verified from PDF on 2026-05-12: arXiv v1 (2212.08073v1). Abstract
verbatim confirmed. §3 = "Constitutional AI: Critiques, Revisions, and
Supervised Learning"; §3.1 "Method" describes the SL-CAI pipeline
(red-team prompts → critique → revision → finetune), confirmed against
KB description. §4 = "Constitutional AI: Reinforcement Learning from AI
Feedback"; §4.1 "Method" describes the RLAIF pipeline (feedback model
with multiple-choice format, log-prob normalized preferences, Bradley-Terry
PM). CoT discussion is a subsection within §4.1 "Chain-of-Thought
Prompting" (KB labels it §5 — no separate §5 in the PDF for this;
PDF §5 is "Related Work"). Main empirical results are in §4.3 and §4.4.
No structural discrepancies found in the pipeline descriptions.]
