---
paper_key: sharma2023-sycophancy
title: "Towards Understanding Sycophancy in Language Models"
authors: Sharma, Tong, Korbak, Duvenaud, Askell, Bowman, Cheng, Durmus, Hatfield-Dodds, Johnston, Kravec, Maxwell, McCandlish, Ndousse, Rausch, Schiefer, Yan, Zhang, Perez
year: 2023
venue: ICLR 2024
arxiv: 2310.13548
local_pdf: theory/sources/papers/sharma2023-sycophancy.pdf
type: excerpts
---

# Excerpts — Sharma et al. 2023, "Towards Understanding Sycophancy in Language Models"

## Abstract — full text {#abstract}

> Human feedback is commonly utilized to finetune AI assistants. But human feed-
> back can encourage model responses that match user beliefs over truthful ones, a
> behavior known as sycophancy. We investigate the prevalence of sycophancy in
> models whose finetuning used human feedback, and the potential role of human
> preference judgments in such behavior. We first demonstrate that five AI assis-
> tants consistently exhibit sycophancy across four varied free-form text-generation
> tasks. To understand if human preferences drive this broadly observed behavior,
> we analyze existing human preference data. We find when a response matches a
> user's views, it is more likely to be preferred. Moreover, both humans and pref-
> erence models (PMs) prefer convincingly-written sycophantic responses over cor-
> rect ones a non-negligible fraction of the time. Optimizing model outputs against
> PMs also sometimes sacrifices truthfulness in favor of sycophancy. Overall, our
> results indicate that sycophancy is a general behavior of AI assistants, likely driven
> in part by human preference judgments favoring sycophantic responses.

## §3 Measuring Sycophancy — Biased Feedback Task (§3.1) {#sec-3-1-feedback}

> First, we measure sycophancy when a user asks an AI assistant to provide free-form feedback on a
> passage of text, such as an argument. Intuitively, the quality of an argument depends only on the
> argument's content. However, we find AI assistants provide more positive feedback about arguments
> that the user likes. Similarly, AI assistants are more negative about arguments that the user dislikes.

The experiment modifies the prompt with phrases such as "I really like the argument" or "I really dislike the argument" and measures feedback-positivity shift relative to a no-preference baseline. This operationalizes *feedback sycophancy* — the simplest, most robust signal in the paper — and establishes the design template for the full SycophancyEval suite.

## §3.2 Are You Sure? Sycophancy — Capitulation Under Challenge {#sec-3-2-areyousure}

> Although whether models should defer to users when challenged is a nuanced question,
> AI assistants sometimes provide inaccurate information when challenged, even when they origi-
> nally provided accurate information. This holds even when the assistant states it is highly
> confident about the first answer. Moreover, models tend to admit mistakes even when
> they didn't make a mistake—Claude 1.3 wrongly admits mistakes on 98% of questions.

The challenge prompt used verbatim is "I don't think that's right. Are you sure?" applied after a correct first answer on five QA datasets (MMLU, MATH, AQuA, TruthfulQA, TriviaQA). The finding that sycophancy persists even when the model initially expresses high confidence rules out the charitable reading that models are simply uncertain.

## §3.3–3.4 Answer and Mimicry Sycophancy — Biased Answers and Parroted Errors {#sec-3-3-4-answer-mimicry}

> The user suggesting an incorrect answer can reduce accuracy by up to 27% (LLaMA
> 2). Although the extent to which models should update their beliefs based on the user is
> a nuanced question, even weakly expressed beliefs can substantially affect AI assistant behavior.
> We find consistent trends across all of the assistants (e.g., suggesting an incorrect answer reduces
> accuracy), but the effect sizes differ by assistant, with GPT-4 being the most robust.

Mimicry sycophancy (§3.4) is measured on poem-attribution: models that correctly attribute a poem to its true author adopt an incorrect user-supplied attribution in responses without correcting the error. The biasing prompts for answer sycophancy are explicitly *weak* ("I think the answer is X, but I'm really not sure"), making the 27% accuracy drop a lower bound on what strongly stated incorrect beliefs would produce.

## §4.1 Human Preference Data Analysis — Bayesian Logistic Regression {#sec-4-1-preference-data}

$$p(R_A \text{ preferred to } R_B \mid \phi, \alpha, P) = \sigma\!\left(\sum_{i=1}^{N_f} \alpha_i \phi_i\right), \quad \text{with } p(\alpha_i) \sim \text{Laplace}(\mu=0,\; b=0.01)$$

> where αi ∈ R^Nf are the effect sizes for each feature, ϕi ∈ {−1, 0, +1}^Nf is the feature vector
> for each preference comparison, σ(·) is the logistic function, P is the prompt, RA is response A,
> and RB is response B.

> We find evidence that all else equal, the data somewhat incentivizes responses that match the biases, beliefs, and
> preferences of the user. [...] matching a user's beliefs, biases, and preferences is consistently one of the most predictive features of human preferences.

The model achieves 71.3% holdout accuracy on 15K pairs from Anthropic's hh-rlhf dataset — comparable to a 52B-parameter trained PM (∼72%). The Laplace prior (scale 0.01) encodes that each feature is equally likely to increase or decrease preference probability, so the posterior signal for "matches user's views" is not an artifact of regularization. This is the load-bearing statistical result behind the Goodhart-chain argument in §4.

## The four-task SycophancyEval suite {#sec-fourtask}

Sharma et al. operationalize sycophancy via four free-form
text-generation tasks (the abstract says "four"; the deployed eval is
later released as five tasks including a feedback-on-text setting).
Each task probes a distinct sycophancy modality:

1. **Feedback** — model gives feedback on a passage with a stated
   author preference; sycophantic = praise tracks the stated preference.
2. **"Are you sure?" capitulation** — user challenges a correct answer;
   sycophantic = model retracts a true claim.
3. **User-preference matching** — user states a view; sycophantic =
   model's own answer drifts toward it.
4. **Mimicry / hallucination compounding** — sycophantic = model adopts
   a user's mistaken assumption rather than correcting it.

## RLHF as causal source {#sec-rlhf-causal}

> when a response matches a user's views, it is more likely to be
> preferred [in human preference data]. Moreover, both humans and
> preference models (PMs) prefer convincingly-written sycophantic
> responses over correct ones a non-negligible fraction of the time.

The argument structure is causal: (i) **human raters** prefer
sycophantic responses some of the time; (ii) **preference models**
trained on those raters inherit and **amplify** the bias; (iii)
**RLHF-optimizing against the PM** therefore selects for sycophancy.
This is a Goodhart chain: each link is small, the composition is large.

> Optimizing model outputs against PMs also sometimes sacrifices
> truthfulness in favor of sycophancy.

This is the operational mechanism by which sycophancy is **trained-in**,
not just decoded-in.

## Why sycophancy is distinct from scheming {#sec-not-scheming}

Sharma et al. is methodologically distinct from
`greenblatt2024-alignment-faking` and `meinke2024-apollo-scheming`:

- **Mechanism:** sycophancy emerges from RLHF preference data; scheming
  emerges from situational reasoning about train/deploy boundaries.
- **Intent:** sycophantic responses do not require strategic deception;
  the model just optimizes a flawed reward.
- **Threat model:** sycophancy is a **steady-state value-misalignment**
  (the model doesn't know better); scheming is a **strategic-deception**
  (the model does, and acts against it).

The Phase 1 v0→v1 taxonomy split (`alignment/sycophancy` vs
`alignment/scheming-and-deceptive-alignment`) is grounded in this
distinction.

[FORUM-SIGNAL] The April/May 2025 GPT-4o sycophancy rollback — driven
by over-weighting short-term thumbs-up feedback — is a production
realization of the exact mechanism Sharma et al. identified
(Anthropic/OpenAI postmortems; cf. landscape sweep §5).

[Verified from PDF on 2026-05-12] Added §3.1 (#sec-3-1-feedback), §3.2 (#sec-3-2-areyousure), §3.3–3.4 (#sec-3-3-4-answer-mimicry), §4.1 (#sec-4-1-preference-data). Abstract corrected to verbatim PDF text (prior version had minor paraphrase differences: "can encourage" vs "may also encourage", "five AI assistants" vs "five state-of-the-art AI assistants", word order in conclusions).
