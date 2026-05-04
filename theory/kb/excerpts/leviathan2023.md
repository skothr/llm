---
paper_key: leviathan2023
title: "Fast Inference from Transformers via Speculative Decoding"
authors: Leviathan, Kalman, Matias
year: 2023
venue: ICML
arxiv: 2211.17192
local_pdf: theory/sources/papers/leviathan2023_speculative.pdf
type: excerpts
note: Verbatim from the v2 arXiv PDF (May 2023, ICML 2023). Speculative decoding's defining contribution is the speculative-sampling rejection rule that *exactly* preserves the target model's output distribution. Concurrent independent work by Chen et al. 2023 (DeepMind, arXiv:2302.01318) derives the same rejection scheme.
---

# Excerpts — Leviathan et al. 2023, "Speculative Decoding"

## Abstract — what the algorithm does {#abstract}

> Inference from large autoregressive models like Transformers is slow -
> decoding K tokens takes K serial runs of the model. In this work we
> introduce speculative decoding - an algorithm to sample from
> autoregressive models faster *without any changes to the outputs*, by
> computing several tokens in parallel. At the heart of our approach lie
> the observations that (1) hard language-modeling tasks often include
> easier subtasks that can be approximated well by more efficient models,
> and (2) using speculative execution and a novel sampling method, we
> can make exact decoding from the large models faster, by running them
> in parallel on the outputs of the approximation models, potentially
> generating several tokens concurrently, and without changing the
> distribution. … We demonstrate it on T5-XXL and show a 2X-3X
> acceleration compared to the standard T5X implementation, with
> identical outputs.

## §2.1 Overview — the speculative core idea {#sec-2-1}

> Let $M_p$ be the target model, inference from which we're trying to
> accelerate, and $p(x_t | x_{<t})$ the distribution we get from the
> model for a prefix $x_{<t}$. Let $M_q$ be a more efficient
> approximation model for the same task, and denote by $q(x_t | x_{<t})$
> the distribution we get from the model for a prefix $x_{<t}$. The
> core idea is to (1) use the more efficient model $M_q$ to generate
> $\gamma \in \mathbb{Z}^+$ completions (see Section 3.5 for how to
> optimally choose this parameter), then (2) use the target model $M_p$
> to evaluate all of the guesses and their respective probabilities from
> $M_q$ *in parallel*, accepting all those that *can* lead to an
> identical distribution, and (3) sampling an additional token from an
> adjusted distribution to fix the first one that was rejected, or to
> add an additional one if they are all accepted. That way, each
> parallel run of the target model $M_p$ will produce at least one new
> token (so the number of serial runs of the target model can never,
> even in the worst case, be larger than the simple autoregressive
> method), but it can potentially generate many new tokens, up to
> $\gamma + 1$, depending on how well $M_q$ approximates $M_p$.

## §2.3 Speculative Sampling — the rejection rule {#sec-2-3}

> To sample $x \sim p(x)$, we instead sample $x \sim q(x)$, keeping it
> if $q(x) \le p(x)$, and in case $q(x) > p(x)$ we reject the sample
> with probability $1 - \tfrac{p(x)}{q(x)}$ and sample $x$ again from an
> adjusted distribution $p'(x) = \operatorname{norm}(\max(0, p(x) -
> q(x)))$ instead. It's easy to show (see Appendix A.1) that for any
> distributions $p(x)$ and $q(x)$, and $x$ sampled in this way, indeed
> $x \sim p(x)$.

## §2.3 Algorithm 1 — SpeculativeDecodingStep {#sec-2-3-algorithm}

> Inputs: $M_p$, $M_q$, prefix.
>
> ▷ Sample $\gamma$ guesses $x_1, \ldots, x_\gamma$ from $M_q$ autoregressively.
>   for $i = 1$ to $\gamma$ do
>     $q_i(x) \leftarrow M_q(prefix + [x_1, \ldots, x_{i-1}])$
>     $x_i \sim q_i(x)$
>   end for
> ▷ Run $M_p$ in parallel.
>   $p_1(x), \ldots, p_{\gamma+1}(x) \leftarrow M_p(prefix), \ldots, M_p(prefix + [x_1, \ldots, x_\gamma])$
> ▷ Determine the number of accepted guesses $n$.
>   $r_1 \sim U(0, 1), \ldots, r_\gamma \sim U(0, 1)$
>   $n \leftarrow \min(\{i - 1 \mid 1 \le i \le \gamma, r_i > p_i(x)/q_i(x)\} \cup \{\gamma\})$
> ▷ Adjust the distribution from $M_p$ if needed.
>   $p'(x) \leftarrow p_{n+1}(x)$
>   if $n < \gamma$ then $p'(x) \leftarrow \operatorname{norm}(\max(0, p_{n+1}(x) - q_{n+1}(x)))$ end if
> ▷ Return one token from $M_p$, and $n$ tokens from $M_q$.
>   $t \sim p'(x)$
>   return $prefix + [x_1, \ldots, x_n, t]$

## §3.1 Number of generated tokens — Eq.(1) {#sec-3-1}

> Definition 3.1. The *acceptance rate* $\beta_{x_{<t}}$, given a
> prefix $x_{<t}$, is the probability of accepting $x_t \sim
> q(x_t|x_{<t})$ by speculative sampling, as per Section 2.3. $E(\beta)$
> is then a natural measure of how well $M_q$ approximates $M_p$. If we
> make the simplifying assumption that the $\beta$s are i.i.d., and
> denote $\alpha = E(\beta)$, then the number of tokens produced by a
> single run of Algorithm 1 is a capped geometric variable, with success
> probability $1 - \alpha$ and cap $\gamma + 1$, and the expected number
> of tokens generated by Algorithm 1 satisfies Equation (1):
>
> $$E(\#\text{generated tokens}) = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}. \tag{1}$$

## §3.2 — α as KL-style divergence {#sec-3-2}

> Theorem 3.5. $\beta = 1 - D_{LK}(p, q)$.
>
> Corollary 3.6. $\alpha = 1 - E(D_{LK}(p, q)) = E(\min(p, q))$.

## §3.3 Walltime improvement — Theorem 3.8 {#sec-3-3}

> Definition 3.7. Let $c$, the *cost coefficient*, be the ratio between
> the time for a single run of $M_q$ and the time for a single run of
> $M_p$.

> Theorem 3.8. The expected improvement factor in total walltime by
> Algorithm 1 is $\frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}$.

## §3.4 Number of arithmetic operations — Theorem 3.11 {#sec-3-4}

> Theorem 3.11. The expected factor of increase in the number of total
> operations of Algorithm 1 is
> $\frac{(1 - \alpha)(\gamma \hat c + \gamma + 1)}{1 - \alpha^{\gamma+1}}$.

## §3.6 Approximation models — empirical α {#sec-3-6}

Reported $\alpha$ values from the paper's Table 3:

> [English-German translation, T5-XXL 11B target with T5-Small 77M draft:
> α ≈ 0.75; with bigram model: α ≈ 0.20.]

Section 4 reports T5-XXL with appropriate draft models achieves 2.6× to
3.4× empirical speedup on machine translation and summarization tasks.
