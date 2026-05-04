---
paper_key: rafailov2023-dpo
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
authors: Rafailov, Sharma, Mitchell, Ermon, Manning, Finn
year: 2023
venue: NeurIPS
arxiv: 2305.18290
local_pdf: null
type: excerpts
note: "Excerpts from arXiv abstract page (verbatim) plus canonical equation forms widely reproduced in surveys (arXiv:2503.11701; arXiv:2410.15595) and downstream variant papers (SimPO §2.1; ORPO §2; KTO §2). Section/equation numbers follow the published NeurIPS 2023 / arXiv v3 layout."
---

# Excerpts — Rafailov et al. 2023, "Direct Preference Optimization"

## Abstract — DPO replaces RM+PPO with classification {#abstract}

> While large-scale unsupervised language models (LMs) learn broad world
> knowledge and some reasoning skills, achieving precise control of their
> behavior is difficult due to the completely unsupervised nature of their
> training. Existing methods for gaining such steerability collect human
> labels of the relative quality of model generations and fine-tune the
> unsupervised LM to align with these preferences, often with reinforcement
> learning from human feedback (RLHF). However, RLHF is a complex and often
> unstable procedure, first fitting a reward model that reflects the human
> preferences, and then fine-tuning the large unsupervised LM using
> reinforcement learning to maximize this estimated reward without drifting
> too far from the original model. In this paper we **introduce a new
> parameterization of the reward model in RLHF that enables extraction of
> the corresponding optimal policy in closed form**, allowing us to solve
> the standard RLHF problem with only **a simple classification loss**. The
> resulting algorithm, which we call Direct Preference Optimization (DPO),
> is stable, performant, and computationally lightweight, eliminating the
> need for sampling from the LM during fine-tuning or performing significant
> hyperparameter tuning.

## §4 Derivation — KL-constrained optimum and reparameterization {#sec-4}

The starting point is the standard KL-constrained reward maximization
objective of RLHF (cf. Stiennon et al. 2020 §3; Ouyang et al. 2022 §3.3):

$$
\max_{\pi_\theta} \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot|x)}
\left[ r(x,y) \right] - \beta\, \mathbb{D}_{\mathrm{KL}}\!\left[
\pi_\theta(\cdot|x) \| \pi_{\mathrm{ref}}(\cdot|x) \right]. \tag{3}
$$

The well-known closed-form optimum (cf. Ziebart 2010; Peng et al. 2019)
is

$$
\pi^*(y|x) = \frac{1}{Z(x)}\, \pi_{\mathrm{ref}}(y|x) \exp\!\left(\frac{1}{\beta} r(x,y)\right). \tag{4}
$$

Inverting Eq. (4) gives the **DPO reparameterization** of the reward in
terms of the optimal policy and the reference:

$$
r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta \log Z(x). \tag{5}
$$

The Bradley–Terry model assumes preferences depend on a difference of
rewards: $p(y_w \succ y_l|x) = \sigma(r(x,y_w) - r(x,y_l))$. Substituting
Eq. (5) cancels the partition function $Z(x)$:

$$
p^*(y_w \succ y_l|x) = \sigma\!\left( \beta \log\frac{\pi^*(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi^*(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} \right). \tag{6}
$$

## §4 The DPO loss {#sec-4-loss}

Maximum-likelihood on the preference dataset $\mathcal{D} =
\{(x_i, y_w^i, y_l^i)\}$ gives the **DPO loss** (Eq. 7 in the paper):

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}}) = - \mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[ \log\sigma\!\left( \beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}
- \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} \right) \right]. \tag{7}
$$

This is the central object: a classification loss directly on policy
log-ratios, with $\pi_{\mathrm{ref}}$ frozen. **No reward model and no
sampling are needed during DPO training.**

## §4 Gradient signal {#sec-4-gradient}

The gradient (paper §4) is

$$
\nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\beta\, \mathbb{E}\!\left[ \sigma(\hat{r}(x,y_l) - \hat{r}(x,y_w)) \cdot \big(\nabla_\theta \log\pi_\theta(y_w|x) - \nabla_\theta\log\pi_\theta(y_l|x)\big) \right],
$$

where $\hat{r}(x,y) = \beta\log\pi_\theta(y|x)/\pi_{\mathrm{ref}}(y|x)$
is the **implicit reward**. The sigmoid-weighted term is large when the
model already has the preference wrong (assigning lower implicit reward
to $y_w$), so DPO updates more aggressively where it is most miscalibrated.

[NOTE — pdf-not-available] Equations (3)–(7) are the canonical DPO
derivation as reproduced verbatim across dozens of downstream papers and
two surveys (arXiv:2503.11701, arXiv:2410.15595). Section/equation
numbers should be re-verified against the PDF.
