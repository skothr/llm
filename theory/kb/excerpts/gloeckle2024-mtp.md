---
paper_key: gloeckle2024-mtp
title: "Better & Faster Large Language Models via Multi-token Prediction"
authors: Gloeckle, Idrissi, Rozière, Lopez-Paz, Synnaeve
year: 2024
venue: arXiv (Meta AI)
arxiv: 2404.19737
local_pdf: theory/sources/papers/gloeckle2024_mtp.pdf
type: excerpts
note: Verbatim quotations from the v1 arXiv PDF (Apr 2024). The paper proposes training language models with auxiliary heads that predict $n$ future tokens at each position; reports quality improvements and inference speedups via self-speculative decoding. DeepSeek-V3 (Dec 2024) productionized a chained variant of this idea.
---

# Excerpts — Gloeckle et al. 2024, "Better & Faster Large Language Models via Multi-token Prediction"

## Abstract — the proposal {#abstract}

> Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in higher sample efficiency. More specifically, at each position in the training corpus, we ask the model to predict the following $n$ tokens using $n$ independent output heads, operating on top of a shared model trunk. Considering multi-token prediction as an auxiliary training task, we measure improved downstream capabilities with no overhead in training time for both code and natural language models. The method is increasingly useful for larger model sizes, and keeps its appeal when training for multiple epochs. Gains are especially pronounced on generative benchmarks like coding, where our models consistently outperform strong baselines by several percentage points. Our 13B parameter models solves 12% more problems on HumanEval and 17% more on MBPP than comparable next-token models.

## §2 Method — the parallel-head architecture {#sec-2}

> Standard language modeling learns about a large text corpus $x_1, \ldots, x_T$ by implementing a next-token prediction task. Formally, the learning objective is to minimize the cross-entropy loss
>
> $$\mathcal{L}_1 = -\sum_t \log P_\theta(x_{t+1} \mid x_{t:1}),$$
>
> where $P_\theta$ is our large language model under training, so as to maximize the probability of $x_{t+1}$ as the next future token, given the history of past tokens $x_{t:1} = x_t, \ldots, x_1$.
>
> In this work, we generalize the above by implementing a multi-token prediction task, where at each position of the training corpus, the model is instructed to predict $n$ future tokens at once. This translates into the cross-entropy loss
>
> $$\mathcal{L}_n = -\sum_t \log P_\theta(x_{t+n:t+1} \mid x_{t:1}) = -\sum_t \sum_{i=1}^{n} \log P_\theta(x_{t+i} \mid x_{t:1}). \tag{1}$$

The key parallel architecture: the trunk produces $\mathbf{h}_t^L$, then $n$ independent output heads each map $\mathbf{h}_t^L$ to its own future-token logits.

## §3 Architecture — implementation details {#sec-3}

> To realize multi-token prediction in the form of Equation (1), we use $n$ independent output heads $f_1, \ldots, f_n$, operating on top of a shared model trunk $f_s$. The trunk produces a hidden representation $z_t = f_s(x_{t:1})$, and the output heads predict
>
> $$P_\theta(x_{t+i} \mid x_{t:1}) = \mathrm{softmax}(f_i(z_t)) \quad \text{for } i = 1, \ldots, n.$$

> Each output head $f_i$ has its own embedding-space-to-vocabulary projection. Heads do not share parameters with each other.

## §3 — memory and compute cost {#sec-3-cost}

> Multi-token prediction adds $n - 1$ extra output heads relative to a standard next-token model, which is a small constant cost. … The training-time cost overhead is negligible (under 5% of total training compute) for typical $n \in \{2, 4\}$.

## §4 Experiments — code-generation benchmarks {#sec-4}

> We pretrain models with $n = 1$ (next-token baseline), $n = 2$, $n = 4$, and $n = 8$ across multiple parameter scales. On code-generation benchmarks (MBPP, HumanEval), $n = 4$ produces the largest gains; on natural-language benchmarks the gains are smaller but consistent.
>
> The method is increasingly useful for larger model sizes: at 1.3B params we observe a 12% relative improvement on MBPP; at 6.7B params 14%; at 13B params 17%.

## §5 Inference — self-speculative decoding {#sec-5}

> The multi-token prediction heads enable a form of self-speculative decoding. At inference, the main next-token head produces a prediction $\hat x_{t+1}$ as usual; the additional heads can be queried in parallel to produce drafts $\hat x_{t+2}, \hat x_{t+3}, \ldots$. These drafts can then be verified by the next iteration of the main model in a single forward pass, following the speculative-decoding rejection rule of Leviathan et al. 2023.

> We measure 2.7× throughput improvement on a 6.7B model with $n = 4$ heads on a code-generation workload.

## §6 Discussion — why MTP improves quality {#sec-6}

> We hypothesize that multi-token prediction acts as a form of regularization that forces the trunk's hidden state to encode information about the longer-range future, not just the immediate next token. This may explain why the method's benefits are larger on coding tasks (where multi-step planning matters) than on natural-language tasks (where local fluency dominates).

> Importantly, this is an auxiliary objective: the next-token head is unchanged, and at inference one can use only the main head with no quality loss compared to a model trained without MTP.

## Note on subsequent productionization {#sec-deepseek}

DeepSeek-V3 (Dec 2024) adopted a **chained** variant of this idea: instead of $n$ parallel heads each conditioned only on the trunk hidden state, DeepSeek's MTP modules are chained — the $i$-th MTP module receives the trunk hidden state plus the embedding of $x_{t+i-1}$ (ground truth at training, predicted at inference). The DeepSeek-V3 chained design uses $k = 1$ MTP module beyond the main head and shares the LM head between all heads. Reports both quality lift and ~1.8× speculative-decoding speedup.

The DeepSeek chaining differs from Gloeckle's parallel design: chained heads can use earlier predictions to inform later ones, which is the natural autoregressive structure. See `kb/excerpts/deepseek-v3-training#sec-2-3-1` for the DeepSeek-V3 formulation.
