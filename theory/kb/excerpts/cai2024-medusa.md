---
paper_key: cai2024-medusa
title: "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"
authors: Cai, Li, Geng, Peng, Lee, Chen, Dao
year: 2024
venue: ICML
arxiv: 2401.10774
local_pdf: theory/sources/papers/cai2024_medusa.pdf
type: excerpts
note: Verbatim from the v3 arXiv PDF (Jun 2024, ICML 2024). Medusa is the model-free family of speculative decoding — instead of training a draft model, it adds multiple LM-head-like heads on the frozen backbone to predict tokens at offset positions, then verifies them with tree attention.
---

# Excerpts — Cai et al. 2024, "Medusa"

## Abstract — multiple heads, tree attention, no draft model {#abstract}

> Large Language Models (LLMs) employ auto-regressive decoding that
> requires sequential computation, with each step reliant on the
> previous one's output. This creates a bottleneck as each step
> necessitates moving the full model parameters from High-Bandwidth
> Memory (HBM) to the accelerator's cache. While methods such as
> speculative decoding have been suggested to address this issue, their
> implementation is impeded by the challenges associated with acquiring
> and maintaining a separate draft model. In this paper, we present
> Medusa, an efficient method that augments LLM inference by adding
> extra decoding heads to predict multiple subsequent tokens in
> parallel. Using a *tree-based attention mechanism*, Medusa constructs
> multiple candidate continuations and verifies them simultaneously in
> each decoding step. … Our experiments demonstrate that Medusa-1 can
> achieve over 2.2× speedup without compromising generation quality,
> while Medusa-2 further improves the speedup to 2.3-2.8×.

## §2.1.1 Medusa Heads — architecture and Eq.(0) {#sec-2-1-1}

> In speculative decoding, subsequent tokens are predicted by an
> auxiliary draft model. … This pre-training process demands
> substantial additional computational resources. … Additionally,
> separate pre-training can potentially create a distribution shift
> between the draft model and the original model. … To streamline and
> democratize the acceleration of LLM inference, we take inspiration
> from Stern et al. (2018), which utilizes parallel decoding for tasks
> such as machine translation and image super-resolution. Medusa heads
> are additional decoding heads appended to the last hidden states of
> the original model. Specifically, given the original model's last
> hidden states $h_t$ at position $t$, we add $K$ decoding heads to
> $h_t$. The $k$-th head is used to predict the token in the
> $(t + k + 1)$-position of the next tokens (the original language
> model head is used to predict the $(t+1)$-th position). The
> prediction of the $k$-th head is denoted as $p_t^{(k)}$, representing
> a distribution over the vocabulary, while the prediction of the
> original model is denoted as $p_t^{(0)}$. … We utilize a single
> layer of feed-forward network with a residual connection for each
> head. The definition of the $k$-th head is outlined as:
>
> $$p_t^{(k)} = \operatorname{softmax}\!\left( W_2^{(k)} \cdot \left( \operatorname{SiLU}\!\left( W_1^{(k)} \cdot h_t \right) + h_t \right) \right),$$
> $$\text{where } W_2^{(k)} \in \mathbb{R}^{d \times V}, W_1^{(k)} \in \mathbb{R}^{d \times d}.$$
>
> $d$ is the output dimension of the LLM's last hidden layer and $V$ is
> the vocabulary size. We initialize $W_2^{(k)}$ identically to the
> original language model head, and $W_1^{(k)}$ to zero.

## §2.1.2 Tree Attention — Cartesian product over top-s_k predictions {#sec-2-1-2}

> Through Medusa heads, we obtain probability predictions for the
> subsequent $K + 1$ tokens. These predictions enable us to create
> length-$K + 1$ continuations as candidates. While the speculative
> decoding studies (Leviathan et al., 2022; Chen et al., 2023) suggest
> sampling a single continuation as the candidate, leveraging multiple
> candidates during decoding can enhance the expected acceptance length
> within a decoding step. … For a given $k$-th head, its top-$s_k$
> predictions serve as the basis for candidate formation, where $s_k$
> is a designated hyperparameter. These candidates are established by
> determining the Cartesian product of the top-$s_k$ predictions from
> each head. For instance, in Figure 2, with $s_1 = 2$ and $s_2 = 3$,
> each first head prediction can be succeeded by any prediction from
> the second head. This leads to a tree structure where $s_k$ branches
> exist at the $k$-th level. … The cumulative number of new tokens is
> calculated as $\sum_{k=1}^{K} \prod_{i=1}^{k} s_i$.

## §2.2.1 Medusa-1 training loss — Eq.(1) {#sec-2-2-1}

> To train Medusa heads with a frozen backbone model, we can use the
> cross-entropy loss between the prediction of Medusa heads and the
> ground truth. Specifically, given the ground truth token
> $y_{t+k+1}$ at position $t + k + 1$, the loss for the $k$-th head is
> $\mathcal{L}_k = -\log p_t^{(k)}(y_{t+k+1})$ where $p_t^{(k)}(y)$
> denotes the probability of token $y$ predicted by the $k$-th head. We
> also observe that $\mathcal{L}_k$ is larger when $k$ is larger, …
> Therefore, we can add a weight $\lambda_k$ to $\mathcal{L}_k$ to
> balance the loss of different heads. And the total Medusa loss is:
>
> $$\mathcal{L}_{\text{Medusa-1}} = \sum_{k=1}^{K} -\lambda_k \log p_t^{(k)}(y_{t+k+1}). \tag{1}$$
>
> In practice, we set $\lambda_k$ as the $k$-th power of a constant
> like $0.8$.

## §2.2.2 Medusa-2 — joint training Eq.(2) {#sec-2-2-2}

> $\mathcal{L}_{\text{Medusa-2}} = \mathcal{L}_{\text{LM}} + \lambda_0 \mathcal{L}_{\text{Medusa-1}}$.
>
> Combined loss; we add $-\log p_t^{(0)}(y_{t+1})$ to keep the
> backbone model's next-token prediction capability.

## §2.3.1 Typical Acceptance — sampling-quality relaxation {#sec-2-3-1}

> However, in real-world scenarios, sampling from language models is
> often employed to generate diverse responses, and the temperature
> parameter is used merely to modulate the "creativity" of the
> response. Therefore, higher temperatures should result in more
> opportunities for the original model to accept the draft model's
> output. We ascertain that it is typically unnecessary to match the
> distribution of the original model. Thus, we propose employing a
> *typical acceptance* scheme to select plausible candidates rather
> than using rejection sampling. … Specifically, given $x_1, x_2,
> \ldots, x_n$ as context, when evaluating the candidate sequence
> $(x_{n+1}, x_{n+2}, \ldots, x_{n+K+1})$ … we consider the condition
>
> $$p_{\text{original}}(x_{n+k} | x_1, x_2, \ldots, x_{n+k-1}) > \min\!\left( \epsilon, \delta \exp(-H(p_{\text{original}}(\cdot | x_1, x_2, \ldots, x_{n+k-1}))) \right),$$
>
> where $H(\cdot)$ denotes the entropy function, and $\epsilon, \delta$
> are the hard threshold and the entropy-dependent threshold
> respectively.

## §2.2.3 Number of heads {#sec-2-2-3}

> Empirically, we found that five heads are sufficient at most.
> Therefore, we recommend training with five heads and referring to the
> strategy described in Section 2.3.3 to determine the optimal
> configuration of the tree attention.

[NB: typical acceptance is the trade-off — Medusa diverges from
Leviathan-style exact distribution preservation when using typical
acceptance with non-zero temperature. Medusa-1 with rejection sampling
*does* preserve the distribution.]
