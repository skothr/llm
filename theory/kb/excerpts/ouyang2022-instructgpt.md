---
paper_key: ouyang2022-instructgpt
title: "Training Language Models to Follow Instructions with Human Feedback (InstructGPT)"
authors: Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, Zhang, Agarwal, Slama, Ray, Schulman, Hilton, Kelton, Miller, Simens, Askell, Welinder, Christiano, Leike, Lowe
year: 2022
venue: NeurIPS
arxiv: 2203.02155
local_pdf: null
type: excerpts
note: "Excerpts from arXiv abstract page (verbatim) plus canonical pipeline description as reproduced in standard RLHF references (Stiennon 2020 §3 — InstructGPT inherits this pipeline; DPO §3 — restates the InstructGPT objective). Section/equation numbers from the published NeurIPS 2022 version."
---

# Excerpts — Ouyang et al. 2022, "InstructGPT"

## Abstract — three-stage RLHF pipeline {#abstract}

> Making language models bigger does not inherently make them better at
> following a user's intent. For example, large language models can generate
> outputs that are untruthful, toxic, or simply not helpful to the user. In
> other words, these models are not aligned with their users. In this
> paper, we show an avenue for aligning language models with user intent on
> a wide range of tasks by **fine-tuning with human feedback**. Starting
> with a set of labeler-written prompts and prompts submitted through the
> OpenAI API, we collect a dataset of labeler demonstrations of the desired
> model behavior, which we use to fine-tune GPT-3 using supervised learning.
> We then collect a dataset of rankings of model outputs, which we use to
> further fine-tune this supervised model using reinforcement learning from
> human feedback. We call the resulting models InstructGPT. In human
> evaluations on our prompt distribution, **outputs from the 1.3B parameter
> InstructGPT model are preferred to outputs from the 175B GPT-3, despite
> having 100x fewer parameters**. Moreover, InstructGPT models show
> improvements in truthfulness and reductions in toxic output generation
> while having minimal performance regressions on public NLP datasets.

## §3 Methods — the canonical RLHF pipeline {#sec-3}

The three-stage InstructGPT pipeline (paper §3.5; this is the recipe
that "RLHF" usually refers to in the LLM literature):

**Stage 1 — Supervised Fine-Tuning (SFT).** Collect human-written
demonstrations $\{(x, y)\}$ of desired behavior on prompts $x$ drawn
from the OpenAI API distribution. Fine-tune GPT-3 with cross-entropy
maximum-likelihood:

$$
\mathcal{L}_{\mathrm{SFT}}(\pi_\theta) = -\mathbb{E}_{(x,y)\sim\mathcal{D}_{\mathrm{SFT}}} \left[ \log\pi_\theta(y|x) \right].
$$

Result: $\pi^{\mathrm{SFT}}$.

**Stage 2 — Reward Model (RM) training.** Collect a dataset of
**preference comparisons**: for each prompt $x$, sample $K \in \{4,9\}$
responses from $\pi^{\mathrm{SFT}}$ and have human labelers rank them.
Train a reward model $r_\phi(x,y)$ (initialized from $\pi^{\mathrm{SFT}}$
with the LM head replaced by a scalar head) on the **Bradley–Terry**
log-likelihood:

$$
\mathcal{L}_{\mathrm{RM}}(\phi) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{\mathrm{RM}}} \left[\log\sigma(r_\phi(x,y_w) - r_\phi(x,y_l))\right].
$$

Across all $\binom{K}{2}$ pairs per prompt, with the standard pair
correlation correction.

**Stage 3 — Reinforcement Learning with PPO.** Optimize the policy
$\pi_\theta$ initialized from $\pi^{\mathrm{SFT}}$ to maximize the
RM reward subject to a per-token KL penalty against $\pi^{\mathrm{SFT}}$:

$$
\mathcal{J}_{\mathrm{RLHF}}(\theta) = \mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(\cdot|x)} \left[r_\phi(x,y)\right] - \beta\, \mathbb{E}_{x,y}\left[\log\frac{\pi_\theta(y|x)}{\pi^{\mathrm{SFT}}(y|x)}\right] + \gamma\, \mathbb{E}_{x\sim\mathcal{D}_{\mathrm{pretrain}}}[\log\pi_\theta(x)].
$$

The third term ($\gamma$ coefficient) is the **PPO-ptx** mixin:
maximum-likelihood on the original pre-training distribution, added to
prevent regression on standard NLP benchmarks ("alignment tax"). When
$\gamma = 0$, the objective reduces to standard PPO-RLHF.

The KL term is implemented as a per-token penalty added to the reward,
not as an explicit constraint:

$$
r_{\mathrm{total}}(x, y_t) = r_\phi(x,y) \cdot \mathbb{1}[t = T] - \beta \log\frac{\pi_\theta(y_t|x,y_{<t})}{\pi^{\mathrm{SFT}}(y_t|x,y_{<t})}.
$$

The terminal-token reward + per-token KL is the standard "RLHF reward
shape" inherited by Stiennon 2020 and reused by every subsequent RLHF
paper.

## §4 Headline result — alignment beats scale on user prompts {#sec-4}

> outputs from the 1.3B parameter InstructGPT model are preferred to
> outputs from the 175B GPT-3, despite having 100x fewer parameters

This is the headline of the paper: an aligned 1.3B model is preferred
over an unaligned 175B model on user-submitted prompts. The result
established RLHF as a first-class post-training technique and is the
direct ancestor of every assistant LLM since.

[NOTE — pdf-not-available] Equation forms are the canonical RLHF
formulation as reproduced in Stiennon 2020 §3, DPO §3, the textbook
treatment, and the survey arXiv:2509.16679; section numbers should be
re-verified against the PDF.
