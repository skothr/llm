---
paper_key: hoffmann2022-chinchilla
title: Training Compute-Optimal Large Language Models
authors: Hoffmann, Borgeaud, Mensch, Buchatskaya, Cai, Rutherford, et al. (DeepMind)
year: 2022
venue: arXiv (subsequently incorporated into Gopher follow-on work)
arxiv: 2203.15556
local_pdf: theory/sources/papers/hoffmann2022-chinchilla_chinchilla.pdf
type: excerpts
note: Verbatim quotations from the v1 arXiv PDF (Mar 2022). Anchors point to section + table/equation numbers in the paper. The headline result is the equal-scaling rule N ≈ D / 20 at compute-optimum, overthrowing the Kaplan model-size-bias rule.
---

# Excerpts — Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models" (Chinchilla)

## Abstract — the headline equal-scaling rule {#abstract}

> We investigate the optimal model size and number of tokens for training a
> transformer language model under a given compute budget. We find that
> current large language models are significantly undertrained, a
> consequence of the recent focus on scaling language models whilst keeping
> the amount of training data constant. By training over 400 language
> models ranging from 70 million to over 16 billion parameters on 5 to 500
> billion tokens, we find that for compute-optimal training, the model size
> and the number of training tokens should be scaled equally: for every
> doubling of model size the number of training tokens should also be
> doubled. We test this hypothesis by training a predicted compute-optimal
> model, *Chinchilla*, that uses the same compute budget as *Gopher* but
> with 70B parameters and 4× more more data. *Chinchilla* uniformly and
> significantly outperforms *Gopher* (280B), GPT-3 (175B), Jurassic-1
> (178B), and Megatron-Turing NLG (530B) on a large range of downstream
> evaluation tasks. This also means that *Chinchilla* uses substantially
> less compute for fine-tuning and inference, greatly facilitating
> downstream usage. As a highlight, *Chinchilla* reaches a state-of-the-art
> average accuracy of 67.5% on the MMLU benchmark, greater than a 7%
> improvement over *Gopher*.

## §2 Related Work — explicit disagreement with Kaplan {#sec-2-disagree-kaplan}

> Our work differs from [Kaplan et al. (2020)] in several important ways.
> First, the authors use a fixed number of training tokens and learning
> rate schedule for all models; this prevents them from modelling the
> impact of these hyperparameters on the loss. In contrast, we find that
> setting the learning rate schedule to approximately match the number of
> training tokens results in the best final loss regardless of model size
> — see Figure A1. For a fixed learning rate cosine schedule to 130B
> tokens, the intermediate loss estimates (for $D' \ll 130B$) are therefore
> overestimates of the loss of a model trained with a schedule length
> matching $D'$. Using these intermediate losses results in
> underestimating the effectiveness of training models on less data than
> 130B tokens, and eventually contributes to the conclusion that model size
> should increase faster than training data size as compute budget
> increases. In contrast, our analysis predicts that both quantities should
> scale at roughly the same rate.

## §3 Estimating the optimal parameter/training tokens allocation — the three approaches {#sec-3}

> We present three different approaches to answer the question driving our
> research: *Given a fixed FLOPs budget, how should one trade-off model
> size and the number of training tokens?* In all three cases we start by
> training a range of models varying both model size and the number of
> training tokens and use the resulting training curves to fit an empirical
> estimator of how they should scale. We assume a power-law relationship
> between compute and model size as done in [Clark et al. (2022); Kaplan et
> al. (2020)], though future work may want to include potential curvature
> in this relationship for large model sizes. The resulting predictions are
> similar for all three methods and suggest that parameter count and
> number of training tokens should be increased equally with more compute
> — with proportions reported in Table 2.

### §3.1 Approach 1: Fix model sizes and vary number of training tokens {#sec-3-1}

> For each parameter count $N$ we train 4 different models, decaying the
> learning rate by a factor of $10\times$ over a horizon (measured in
> number of training tokens) that ranges by a factor of $16\times$. … At
> 1500 logarithmically spaced FLOP values, we find which model size
> achieves the lowest loss of all models along with the required number of
> training tokens. Finally, we fit power laws to estimate the optimal model
> size and number of training tokens for any given amount of compute (see
> the center and right panels of Figure 2), obtaining a relationship
> $N_{opt} \propto C^a$ and $D_{opt} \propto C^b$. We find that $a = 0.50$
> and $b = 0.50$ — as summarized in Table 2.

### §3.2 Approach 2: IsoFLOP profiles {#sec-3-2}

> In our second approach we vary the model size for a fixed set of 9
> different training FLOP counts (ranging from $6 \times 10^{18}$ to
> $3 \times 10^{21}$ FLOPs), and consider the final training loss for each
> point. … For each FLOP budget, we plot the final loss (after smoothing)
> against the parameter count in Figure 3 (left). … We fit a parabola to
> each IsoFLOPs curve to directly estimate at what model size the minimum
> loss is achieved (Figure 3 (left)). … Again, we fit exponents of the
> form $N_{opt} \propto C^a$ and $D_{opt} \propto C^b$ and we find that
> $a = 0.49$ and $b = 0.51$ — as summarized in Table 2.

### §3.3 Approach 3: Fitting a parametric loss function {#sec-3-3}

> Lastly, we model all final losses from experiments in Approach 1 & 2 as
> a parametric function of model parameter count and the number of seen
> tokens. Following a classical risk decomposition (see Section D.2), we
> propose the following functional form
>
> $$\hat{L}(N, D) \;\triangleq\; E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}. \tag{2}$$
>
> The first term captures the loss for an ideal generative process on the
> data distribution, and should correspond to the entropy of natural text.
> The second term captures the fact that a perfectly trained transformer
> with $N$ parameters underperforms the ideal generative process. The
> final term captures the fact that the transformer is not trained to
> convergence, as we only make a finite number of optimisation steps, on
> a sample of the dataset distribution.

> **Model fitting.** To estimate $(A, B, E, \alpha, \beta)$, we minimize
> the Huber loss [Huber, 1964] between the predicted and observed log loss
> using the L-BFGS algorithm [Nocedal, 1980]:
>
> $$\min_{A, B, E, \alpha, \beta} \;\sum_{\text{Runs } i} \mathrm{Huber}_\delta\!\big(\log \hat{L}(N_i, D_i) - \log L_i\big). \tag{3}$$

> **Efficient frontier.** We can approximate the functions $N_{opt}$ and
> $D_{opt}$ by minimizing the parametric loss $\hat{L}$ under the
> constraint $\mathrm{FLOPs}(N, D) \approx 6 N D$ [Kaplan et al., 2020].
> The resulting $N_{opt}$ and $D_{opt}$ balance the two terms in
> Equation (3) that depend on model size and data. By construction, they
> have a power-law form:
>
> $$N_{opt}(C) = G\!\left(\frac{C}{6}\right)^a, \quad D_{opt}(C) = G^{-1}\!\left(\frac{C}{6}\right)^b, \tag{4}$$
>
> $$\text{where } G = \left(\frac{\alpha A}{\beta B}\right)^{\!1/(\alpha+\beta)},\ a = \frac{\beta}{\alpha+\beta},\ b = \frac{\alpha}{\alpha+\beta}.$$
>
> … From this approach, we find that $a = 0.46$ and $b = 0.54$ — as
> summarized in Table 2.

## §3 Table 2 — exponents across all three approaches vs. Kaplan {#sec-3-table-2}

> **Estimated parameter and data scaling with increased training compute.**
> The listed values are the exponents, $a$ and $b$, on the relationship
> $N_{opt} \propto C^a$ and $D_{opt} \propto C^b$. … The 10th and 90th
> percentiles are estimated via bootstrapping data (80% of the dataset is
> sampled 100 times) and are shown in parenthesis.
>
> | Approach | Coeff. $a$ where $N_{opt} \propto C^a$ | Coeff. $b$ where $D_{opt} \propto C^b$ |
> |----------|---------------------------------------|---------------------------------------|
> | 1. Minimum over training curves | 0.50 (0.488, 0.502) | 0.50 (0.501, 0.512) |
> | 2. IsoFLOP profiles | 0.49 (0.462, 0.534) | 0.51 (0.483, 0.529) |
> | 3. Parametric modelling of the loss | 0.46 (0.454, 0.455) | 0.54 (0.542, 0.543) |
> | Kaplan et al. (2020) | 0.73 | 0.27 |

## §3 Table 3 — the implied tokens-per-parameter recipe {#sec-3-table-3}

> **Estimated optimal training FLOPs and training tokens for various model
> sizes.** For various model sizes, we show the projections from Approach
> 1 of how many FLOPs and training tokens would be needed to train
> compute-optimal models.
>
> | Parameters | FLOPs | FLOPs (in *Gopher* unit) | Tokens |
> |---|---|---|---|
> | 400 Million | 1.92e+19 | 1/29,968 | 8.0 Billion |
> | 1 Billion | 1.21e+20 | 1/4,761 | 20.2 Billion |
> | 10 Billion | 1.23e+22 | 1/46 | 205.1 Billion |
> | 67 Billion | 5.76e+23 | 1 | 1.5 Trillion |
> | 175 Billion | 3.85e+24 | 6.7 | 3.7 Trillion |
> | 280 Billion | 9.90e+24 | 17.2 | 5.9 Trillion |
> | 520 Billion | 3.43e+25 | 59.5 | 11.0 Trillion |
> | 1 Trillion | 1.27e+26 | 221.3 | 21.2 Trillion |
> | 10 Trillion | 1.30e+28 | 22515.9 | 216.2 Trillion |

This is the table that gives the famous "≈20 tokens per parameter" rule:
1B params → 20.2B tokens; 10B → 205.1B; 70B → ~1.4T (Chinchilla itself).

## §3.4 Optimal model scaling — the empirical claim {#sec-3-4}

> We find that the three approaches, despite using different fitting
> methodologies and different trained models, yield comparable predictions
> for the optimal scaling in parameters and tokens with FLOPs (shown in
> Table 2). All three approaches suggest that as compute budget increases,
> model size and the amount of training data should be increased in
> approximately equal proportions. … In *Table 3* we show the estimated
> number of FLOPs and tokens that would ensure that a model of a given
> size lies on the compute-optimal frontier. Our findings suggests that
> the current generation of large language models are considerably
> over-sized, given their respective compute budgets, as shown in Figure 1.
> For example, we find that a 175 billion parameter model should be
> trained with a compute budget of $4.41 \times 10^{24}$ FLOPs and on over
> 4.2 trillion tokens. A 280 billion *Gopher*-like model is the optimal
> model to train given a compute budget of approximately $10^{25}$ FLOPs
> and should be trained on 6.8 trillion tokens.

## §4.1 Model and training details — Chinchilla 70B vs Gopher 280B {#sec-4-1}

> Based on our analysis in Section 3, the optimal model size for the
> *Gopher* compute budget is somewhere between 40 and 70 billion
> parameters. We test this hypothesis by training a model on the larger
> end of this range — 70B parameters — for 1.4T tokens, due to both
> dataset and computational efficiency considerations. In this section we
> compare this model, which we call *Chinchilla*, to *Gopher* and other
> LLMs. Both *Chinchilla* and *Gopher* have been trained for the same
> number of FLOPs but differ in the size of the model and the number of
> training tokens.

Architecture comparison (Table 4):

| Model | Layers | Number Heads | Key/Value Size | $d_{\text{model}}$ | Max LR | Batch Size |
|---|---|---|---|---|---|---|
| *Gopher* 280B | 80 | 128 | 128 | 16,384 | $4 \times 10^{-5}$ | 3M → 6M |
| *Chinchilla* 70B | 80 | 64 | 128 | 8,192 | $1 \times 10^{-4}$ | 1.5M → 3M |

Same depth, half the width; LR raised; ~half the batch.

## §4.2 Results — Chinchilla beats Gopher across the board {#sec-4-2}

> *Chinchilla* significantly outperforms *Gopher* on all evaluation
> subsets of The Pile [Gao et al., 2020], as shown in Figure 5. Compared
> to Jurassic-1 (178B) [Lieber et al., 2021], *Chinchilla* is more
> performant on all but two subsets — `dm_mathematics` and `ubuntu_irc` —
> see Table A5 for a raw bits-per-byte comparison. On Wikitext103 [Merity
> et al., 2017], *Chinchilla* achieves a perplexity of 7.16 compared to
> 7.75 for *Gopher*. Some caution is needed when comparing *Chinchilla*
> with *Gopher* on these language modelling benchmarks as *Chinchilla* is
> trained on $4\times$ more data than *Gopher* and thus train/test set
> leakage may artificially enhance the results.

## Why this matters — the post-Chinchilla era {#why-matters}

The headline rule "tokens ≈ 20 × parameters at compute-optimum" reset
the entire scaling-recipe community in 2022. GPT-3 (175B params, 300B
tokens, ratio ≈ 1.7) and Gopher (280B, 300B, ratio ≈ 1.07) were both
**dramatically under-trained** by this metric. Chinchilla (70B, 1.4T,
ratio = 20.0) became the canonical compute-optimal point. The exponents
0.46–0.50 (vs. Kaplan's 0.73) for $a$ in $N_{opt} \propto C^a$ are the
quantitative core of the disagreement.

Subsequent industry practice has actually pushed *past* Chinchilla
toward **token-heavy over-training** — LLaMA 3 8B used ~15T tokens
(ratio ≈ 1875), driven by inference economics that favor smaller models
even at the cost of pre-training compute efficiency. This is a different
optimization (inference-FLOP-weighted, not pre-training-FLOP-weighted)
and does not contradict Chinchilla; it changes the objective. See
`kb/notes/scaling/scaling-frontier.md`.
