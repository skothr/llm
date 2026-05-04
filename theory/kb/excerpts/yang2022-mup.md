---
paper_key: yang2022-mup
title: "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
authors: Yang, Hu, Babuschkin, Sidor, Liu, Farhi, Ryder, Pachocki, Chen, Gao (Microsoft Research / OpenAI)
year: 2022
venue: arXiv (subsequently NeurIPS 2022)
arxiv: 2203.03466
local_pdf: theory/sources/papers/yang2022-mup_tensor_programs_v.pdf
type: excerpts
note: Verbatim quotations from the v1 arXiv PDF (Mar 2022). The headline contribution is Table 3, the μP scaling rules table, and the empirical demonstration that HPs tuned on a 40M proxy transfer zero-shot to GPT-3 6.7B with tuning cost 7% of pretraining.
---

# Excerpts — Yang et al. 2022, "Tensor Programs V" (μP / muTransfer)

## Abstract — what μTransfer enables {#abstract}

> Hyperparameter (HP) tuning in deep learning is an expensive process,
> prohibitively so for neural networks (NNs) with billions of
> parameters. We show that, in the recently discovered Maximal Update
> Parametrization (μP), many optimal HPs remain stable even as model
> size changes. This leads to a new HP tuning paradigm we call
> **μTransfer**: parametrize the target model in μP, tune the HP
> indirectly on a smaller model, and zero-shot transfer them to the
> full-sized model, i.e., without directly tuning the latter at all. We
> verify μTransfer on Transformer and ResNet. For example, 1) by
> transferring pretraining HPs from a model of 13M parameters, we
> outperform published numbers of BERT-large (350M parameters), with a
> total tuning cost equivalent to pretraining BERT-large once; 2) by
> transferring from 40M parameters, we outperform published numbers of
> the **6.7B GPT-3** model, with **tuning cost only 7% of total
> pretraining cost**.

## §1 Contributions {#sec-1-contributions}

> - We demonstrate it is possible to zero-shot transfer near optimal HPs
>   to a large model from a small version via the Maximal Update
>   Parametrization (μP) from [Tensor Programs IV / Yang and Hu 2021].
> - While [57] only covered SGD, here we derive μP for Adam as well
>   (Table 3).
> - We propose a new HP tuning technique, μTransfer, for large neural
>   networks based on this observation that provides massive speedup
>   over conventional methods and covers both SGD and Adam training;
> - We thoroughly verify our method on machine translation and large
>   language model pretraining (in Section 7.3) as well as image
>   classification (in Appendix G.1);
> - We release a PyTorch package for implementing μTransfer painlessly.

## §1 What is "μTransferable" {#sec-1-table-2-transferable}

> **Table 2: Examples of μTransferable Hyperparameters.** All of the
> below can also be specialized to per-layer hyperparameters.
>
> | Optimizer Related | Initialization | Parameter Multipliers |
> |---|---|---|
> | learning rate (LR), momentum, Adam beta, LR schedule, etc | per-layer init variance | multiplicative constants after weight/biases, etc |

## §2 Why standard parametrization fails to transfer (the CLT primer) {#sec-2-clt}

> The Central Limit Theorem (CLT) says that, if $x_1, \ldots, x_n$ are
> iid samples from a zero-mean, unit-variance distribution, then
> $\frac{1}{\sqrt{n}}(x_1 + \cdots + x_n)$ converges to a standard
> Gaussian $\mathcal{N}(0, 1)$ as $n \to \infty$. Therefore, we can say
> that $\frac{1}{\sqrt{n}}$ is the right order of *scaling factor* $c_n$
> such that $c_n(x_1 + \cdots + x_n)$ converges to something nontrivial.
> In contrast, if we set $c_n = 1/n$, then $c_n(x_1 + \cdots + x_n) \to
> 0$; or if $c_n = 1$, then $c_n(x_1 + \cdots + x_n)$ blows up in
> variance as $n \to \infty$.

> [Then for sufficiently large $n$, the optimal $\alpha^\ast_n$] should
> be close to $\alpha^\ast_N$ for any $N > n$, and indeed, for $\bar
> N = \infty$ — this precisely means we can *transfer* the optimal
> $c^\ast_n$ or $\alpha^\ast_n$ for a smaller problem (say $F_n$) to a
> larger problem (say $F_N$): $G_N$ is approximately minimized by
> $\alpha^\ast_n$ and $F_N$ is approximately minimized by $c^\ast_n
> \sqrt{n/N}$. Because the transfer algorithm is simply copying
> $\alpha$, we say the parametrization $c = \alpha/\sqrt{n}$ is the
> *correct parametrization* for this problem.

> [Theoretically, a μP network has a well-defined infinite-width limit —
> akin to $(x_1 + \cdots + x_n)/\sqrt{n}$ having a $\mathcal{N}(0, 1)$
> limit by CLT — while a SP network does not (the limit will blow up).
> In fact, based on the theoretical foundation laid in [57], we argue
> in Appendix J.3 that **μP should also be the *unique* parametrization
> that allows HP transfer across width**.]

## §4 The μP scaling rules table — Table 3 {#sec-4-table-3}

> **Table 3: μP[57] and SP for General Neural Networks.** Here, we
> emphasize the *scaling with width (fan_in or fan_out)*; in practice,
> we may insert tunable multipliers in front of fan_in and fan_out as in
> Eq. (4). … *SGD* (resp. *Adam*) here can be replaced by variants such
> as SGD with momentum (resp. Adagrad, etc); see Appendix B.3 for other
> optimizers. In general, the three columns here can be interpreted as
> linear layers with {finite, infinite, infinite} input dimension and
> {infinite, finite, infinite} output dimension in an infinite-width
> network.
>
> |  | Input weights & all biases | Output weights | Hidden weights |
> |---|---|---|---|
> | **Init. Var.** | $1/\text{fan\_in}$ | $1/\text{fan\_in}^2$ ($\,1/\text{fan\_in}\,$) | $1/\text{fan\_in}$ |
> | **SGD LR** | $\text{fan\_out}$ ($1$) | $1/\text{fan\_in}$ ($1$) | $1$ |
> | **Adam LR** | $1$ | $1/\text{fan\_in}$ ($1$) | $1/\text{fan\_in}$ ($1$) |
>
> Gray text recalls the corresponding standard parametrization (SP);
> purple highlights key differences from SP. … Transformer μP requires
> one more modification ($1/d$ attention instead of $1/\sqrt{d}$); see
> Definition 4.1.

The big takeaway: **what changes between SP and μP is how three things
scale with width** — init variance of output weights ($1/n^2$ vs.
$1/n$), SGD LR for input/output weights, and Adam LR for hidden weights
($1/n$ vs. $1$). Once those are fixed, optimal HPs (LR, momentum, Adam
betas, ...) become *invariant under width scaling*.

## §4 Definition 4.1 — μP for Transformer {#sec-4-definition-4-1}

> **Definition 4.1.** The *Maximal Update Parametrization (μP) for a
> Transformer* is given by Table 3 and $1/d$ attention instead of
> $1/\sqrt{d}$, i.e. the attention logit is calculated as $q^\top k / d$
> instead of $q^\top k / \sqrt{d}$ where query $q$ and key $k$ have
> dimension $d$.

This is a structural change: **μP replaces the standard $1/\sqrt{d_k}$
softmax scaling with $1/d_k$**. The reasoning, from §5: in SP, after one
SGD step, $q^\top k$ has variance $\Theta(d)$ (because $q$ and $k$
become correlated); the right rescaling for stability under width is
then $1/d$, not $1/\sqrt{d}$. Vaswani's $1/\sqrt{d_k}$ is correct *at
initialization* (uncorrelated $q, k$); μP's $1/d$ is correct *during
training* (correlated $q, k$).

## §5 The "Logits and attention logits blow up in SP, not μP" claim {#sec-5-blowup}

> Logits and attention logits, but not word embeddings, of a Transformer
> blow up with width in SP after 1 step of training. In contrast, all
> three are well-behaved with width in μP. Here we measure how much
> different values change coordinatewise from initialization over 4
> steps of Adam updates, as a function of width. … Specifically, we plot
> the standard deviation of the coordinates of $x_t - x_0$, for $t = 0,
> \ldots, 4$, and $x \in \{$logits, attention logits, word embeddings$\}$.

Empirical evidence that the SP→μP fix is necessary not just at init
but also after any nontrivial number of steps.

## §6 What μTransfers (and what doesn't) {#sec-6-what-transfers}

> [HPs] can be divided into three kinds:
>
> 1. those that can transfer from the small to the large model, such as
>    learning rate (Table 2);
> 2. those that primarily control regularization and don't work well
>    with our technique; and
> 3. those that define training *scale*, such as width as discussed
>    above as well as others like depth and batch size, across which we
>    transfer other HPs.
>
> Those in the first category transfer across width, as theoretically
> justified above. … For the second category, the amount of
> regularization (for the purpose of controlling overfitting) naturally
> depends on both the model size and data size, so we should not expect
> transfer to work if the parametrization only depends on model size.

## §6.1 Empirical caveats {#sec-6-1-caveats}

> Empirically, we find that for language modeling on Transformers, HPs
> generally transfer across scale dimensions if some minimum width
> (e.g. 256), depth (e.g., 4), batch size (e.g., 32), sequence length
> (e.g., 128), and training steps (e.g., 5000) are met, and the target
> scale is within the "reasonable range" as in our experiments. … the
> best initialization standard deviation does not seem to transfer well
> across depth (2nd row, 3rd column), despite having a stabler pattern
> across width. In addition, while our results on width, batch size,
> sequence length, and training time still hold for post-layernorm
> Transformer (Fig. 17), the transfer across depth only works for
> fixing the initialization standard deviation and sweeping along the
> learning rate schedule.

> [I]n practice (e.g. our results in Section 7.3) we find that fixing
> initialization standard deviation while tuning other HPs works well
> when transferring across depth.

This is the practical user's guide: **μTransfer is for width**;
transfer across depth is partial; transfer across batch size and
sequence length is empirically supported within the "reasonable range."

## §7 Headline applications {#sec-7-applications}

> Now that the plausibility of μTransfer has been established in toy
> settings, we turn to more realistic scenarios to see if one can
> achieve tangible gains. Specifically, we perform HP tuning only on a
> smaller proxy model, test the obtained HPs on the large target model
> directly, and compare against baselines tuned using the target model.
> We seek to answer the question: Can μTransfer make HP tuning more
> efficient while achieving performance on par with traditional tuning?
> As we shall see by the end of the section, the answer is positive.

The §7.3 GPT-3 6.7B headline (from the abstract): tuning at 40M proxy
scale, transferring to 6.7B, *outperforms published GPT-3 6.7B numbers*
at total tuning cost = 7% of one pretraining run.

## Why this matters — μP in production {#why-matters}

μP/μTransfer is the practical answer to "how do you tune
hyperparameters for a model that costs $10M+ to train *once*". The
recipe (as stated in §7) is:

1. Parametrize target model in μP (use the package).
2. Pick a small proxy model (e.g., 40M params, same depth as target).
3. Sweep LR, init scale, and any per-layer multipliers on the proxy.
4. Copy the optimal $\alpha$ (the dimensionless multipliers) to target;
   **do not retune at scale**.

Industry adoption: Cerebras' practitioner guide (2024) confirms μP is
in production use; Tensor Programs VI (2023) extends to depth (Depth-μP);
u-μP (Graphcore, ICLR 2025) integrates μP with unit-scaling for native
FP8 training. Llama 4 reportedly uses "MetaP" for cross-scale HP transfer
(per the Meta tech report). The "tune small, run big" workflow is now
table stakes at frontier labs.

The theoretical foundation in Tensor Programs IV (Yang & Hu 2021) shows
μP is the **unique** parametrization (up to redundancy) that admits a
non-trivial feature-learning infinite-width limit. Standard
parametrization (SP) either has trivial dynamics in the kernel/NTK
regime or blows up.
