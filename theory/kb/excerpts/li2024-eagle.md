---
paper_key: li2024-eagle
title: "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"
authors: Li, Wei, Zhang, Zhang
year: 2024
venue: ICML
arxiv: 2401.15077
local_pdf: theory/sources/papers/li2024_eagle.pdf
type: excerpts
note: Verbatim from the v3 arXiv PDF (Mar 2025 update). EAGLE is a draft-model speculative decoding scheme that drafts at the *feature* level (second-to-top-layer hidden state) rather than the token level. EAGLE-2 (2406.16858) and EAGLE-3 (2503.01840) are direct successors that change the tree construction (dynamic) and abandon feature prediction respectively.
---

# Excerpts — Li et al. 2024, "EAGLE"

## Abstract — feature-level speculation {#abstract}

> Autoregressive decoding makes the inference of Large Language Models
> (LLMs) time-consuming. In this paper, we reconsider speculative
> sampling and derive two key observations. Firstly, autoregression at
> the feature (second-to-top-layer) level is more straightforward than
> at the token level. Secondly, the inherent uncertainty in feature
> (second-to-top-layer) level autoregression constrains its performance.
> Based on these insights, we introduce EAGLE (Extrapolation Algorithm
> for Greater Language-model Efficiency), a simple yet highly efficient
> speculative sampling framework. By incorporating a token sequence
> advanced by one time step, EAGLE effectively resolves the
> uncertainty, enabling precise second-to-top-layer feature prediction
> with minimal overhead. … For LLaMA2-Chat 70B, EAGLE achieved a
> latency speedup ratio of 2.7x-3.5x, doubled throughput, while
> maintaining the distribution of the generated text.

## §2 Preliminaries — feature definition and speculative sampling recap {#sec-2}

> "Target LLM" denotes the LLM intended for acceleration, while "draft
> model" refers to the model used for draft generation. "Feature"
> generally signifies the second-to-top-layer feature of a LLM, the
> hidden state before the LM head. Tokens are denoted by lowercase $t$,
> their embeddings by $e$, features by $f$, and distributions by $p$.
> … In a LLM, input $T_{1:j}$ is transformed into embeddings $E_{1:j}$
> through the embedding layer, then to features $F_{1:j}$, and the LM
> Head maps $f_j$ to a distribution $p_{j+1} = \text{LM\_Head}(f_j)$,
> sampling the next token $t_{j+1}$. Vanilla autoregression at the
> token level is described by $T_{1:j} \to E_{1:j} \to f_j \to p_{j+1}
> \to t_{j+1}$ for any integer $j \ge 1$.

> Speculative sampling. Speculative sampling operates through draft and
> verification phases, with the drafting phase using a smaller model to
> generate $\gamma$ tokens $\hat T_{j+1:j+\gamma}$ and their
> distributions $\hat P_{j+1:j+\gamma}$. In the verification phase, a
> single forward pass of the target LLM yields the probabilities
> $P_{j+1:j+\gamma}$. Tokens are then sequentially evaluated, with a
> token $\hat t_{j+i}$ having an acceptance probability
> $\min(1, p_{j+i}(\hat t_{j+i}) / \hat p_{j+i}(\hat t_{j+i}))$. Upon
> the rejection of a token $\hat t_{j+i}$, all subsequent tokens are
> discarded, and this token is resampled based on a distribution
> $\operatorname{norm}(\max(0, p_{j+i} - \hat p_{j+i}))$. As proven in
> Appendix A.1 of speculative sampling (Leviathan et al., 2023), this
> method equates to sampling directly from the target LLM. EAGLE adopts
> this method, ensuring that the distribution of the generated text
> remains unchanged for both the greedy and non-greedy settings.

## §3.1 Drafting phase — shifted token sequence {#sec-3-1}

> The primary distinction between EAGLE and other methods lies
> predominantly in the drafting phase. … EAGLE predicts $f_3$ using
> the feature sequence $(f_1, f_2)$ and the token sequence $(t_2, t_3)$,
> advanced by one time step. From $p_4 = \text{LM\_Head}(f_3)$, $t_4$
> is sampled. Subsequently, $f_3$ and $t_4$ are concatenated into the
> input sequence to predict the next feature $f_4$ and sample the
> subsequent token $t_5$.
>
> As illustrated in Figure 6, EAGLE's draft model comprises three
> modules: the Embedding layer, LM Head, and Autoregression Head. The
> Embedding layer and LM Head employ the parameters of the target LLM
> and do not necessitate additional training. The draft model takes as
> input a feature sequence of shape (bs, seq_len, hidden_dim) and an
> advanced token sequence of shape (bs, seq_len). It then converts the
> token sequence into a token embedding sequence of shape (bs, seq_len,
> hidden_dim), and concatenates it to form a fused sequence of shape
> (bs, seq_len, 2×hidden_dim). The Autoregression Head consisting of an
> FC layer and a decoder layer. The FC layer reduces the dimensionality
> of the fused sequence to (bs, seq_len, hidden_dim) and then utilize
> the decoder layer to predict the next feature.

## §3.1 Tree-structured draft {#sec-3-1-tree}

> EAGLE creates a tree-structured draft using tree attention, generating
> a draft tree with depth $m$ and more than $m$ tokens through $m$
> forward passes. For instance, as shown in Figure 6, EAGLE drafts a
> 10-token tree with just 3 forward passes.

## §3.2 Training the draft model — losses {#sec-3-2}

> Predicting the next feature constitutes a regression task, for which
> we employ Smooth L1 loss:
>
> $$L_{\text{reg}} = \text{SmoothL1}(f_{i+1}, \text{Draft\_Model}(T_{2:i+1}, F_{1:i})).$$
>
> Predicting features is an intermediary objective of the draft model,
> with the ultimate goal being the prediction of tokens to generate a
> sequence of tokens. Consequently, we also employ classification loss
> to directly optimize towards this final objective:
>
> $$p_{i+2} = \operatorname{Softmax}(\text{LM\_Head}(f_{i+1})),$$
> $$\hat p_{i+2} = \operatorname{Softmax}(\text{LM\_Head}(\hat f_{i+1})),$$
> $$L_{\text{cls}} = \text{CrossEntropy}(p_{i+2}, \hat p_{i+2}).$$
>
> By integrating regression loss and classification loss, we train the
> Autoregression Head using the combined loss function $L = L_{\text{reg}}
> + w_{\text{cls}} L_{\text{cls}}$. Typically, the classification loss
> is an order of magnitude larger than the regression loss in numerical
> terms. Consequently, we set $w_{\text{cls}} = 0.1$.

## §1 Headline speedups (Figure 1) {#sec-1-figure-1}

> Figure 1 [MT-bench, temperature=0]: Vicuna 7B 2.90×, Vicuna 13B 3.07×,
> Vicuna 33B 2.95×, LLaMA2-Chat 7B 2.78×, LLaMA2-Chat 13B 3.03×,
> LLaMA2-Chat 70B 3.01×.
>
> Figure 2 [MT-bench, temperature=1]: Vicuna 7B 2.13×, Vicuna 13B 2.32×,
> Vicuna 33B 2.40×, LLaMA2-Chat 7B 2.22×, LLaMA2-Chat 13B 2.68×,
> LLaMA2-Chat 70B 2.67×.

## §4.1 — vs Lookahead and Medusa {#sec-4-1}

> Compared to recently introduced speculative sampling-based methods,
> Lookahead and Medusa, EAGLE is faster by 1.70×-2.08× and 1.47×-1.60×,
> respectively. … For DistillSpec, to ensure fairness, we used the
> same training data as EAGLE.

## §3.3 Verification phase — distribution preservation {#sec-3-3}

> Acceleration of EAGLE theoretically guarantees the preservation of
> the target LLMs' output distribution. Consequently, evaluating the
> quality of EAGLE's generated results is both unnecessary and
> meaningless.
