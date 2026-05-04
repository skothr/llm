---
paper_key: kaplan2020
title: Scaling Laws for Neural Language Models
authors: Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei (OpenAI)
year: 2020
venue: arXiv
arxiv: 2001.08361
local_pdf: theory/sources/papers/kaplan2020_scaling_laws.pdf
type: excerpts
note: Verbatim quotations from the v1 arXiv PDF (Jan 2020). The headline power-law equations (1.1)–(1.6) are the core of the paper. Anchors point to section + equation numbers.
---

# Excerpts — Kaplan et al. 2020, "Scaling Laws for Neural Language Models"

## Abstract — three power laws over seven orders of magnitude {#abstract}

> We study empirical scaling laws for language model performance on the
> cross-entropy loss. The loss scales as a power-law with model size,
> dataset size, and the amount of compute used for training, with some
> trends spanning more than seven orders of magnitude. Other architectural
> details such as network width or depth have minimal effects within a
> wide range. Simple equations govern the dependence of overfitting on
> model/dataset size and the dependence of training speed on model size.
> These relationships allow us to determine the optimal allocation of a
> fixed compute budget. Larger models are significantly more
> sample-efficient, such that optimally compute-efficient training
> involves training very large models on a relatively modest amount of
> data and stopping significantly before convergence.

## §1.1 Summary of findings (text bullets) {#sec-1-1-bullets}

> **Performance depends strongly on scale, weakly on model shape:** Model
> performance depends most strongly on scale, which consists of three
> factors: the number of model parameters $N$ (excluding embeddings), the
> size of the dataset $D$, and the amount of compute $C$ used for
> training. Within reasonable limits, performance depends very weakly on
> other architectural hyperparameters such as depth vs. width.

> **Smooth power laws:** Performance has a power-law relationship with
> each of the three scale factors $N, D, C$ when not bottlenecked by the
> other two, with trends spanning more than six orders of magnitude (see
> Figure 1). We observe no signs of deviation from these trends on the
> upper end, though performance must flatten out eventually before
> reaching zero loss.

> **Universality of overfitting:** Performance improves predictably as
> long as we scale up $N$ and $D$ in tandem, but enters a regime of
> diminishing returns if either $N$ or $D$ is held fixed while the other
> increases. The performance penalty depends predictably on the ratio
> $N^{0.74}/D$, meaning that every time we increase the model size 8x,
> we only need to increase the data by roughly 5x to avoid a penalty.

> **Sample efficiency:** Large models are more sample-efficient than
> small models, reaching the same level of performance with fewer
> optimization steps … and using fewer data points.

> **Convergence is inefficient:** When working within a fixed compute
> budget $C$ but without any other restrictions on the model size $N$ or
> available data $D$, we attain optimal performance by training *very
> large models* and stopping *significantly short of convergence*. … data
> requirements grow very slowly as $D \sim C^{0.27}$ with training
> compute.

## §1.2 Summary of Scaling Laws — the canonical equations {#sec-1-2-equations}

> The test loss of a Transformer trained to autoregressively model
> language can be predicted using a power-law when performance is limited
> by only either the number of non-embedding parameters $N$, the dataset
> size $D$, or the optimally allocated compute budget $C_{\min}$ (see
> Figure 1):
>
> 1. For models with a limited number of parameters, trained to
>    convergence on sufficiently large datasets:
>    $$L(N) = (N_c / N)^{\alpha_N}\,;\quad \alpha_N \sim 0.076,\ \ N_c \sim 8.8 \times 10^{13} \text{ (non-embedding parameters)} \tag{1.1}$$
>
> 2. For large models trained with a limited dataset with early stopping:
>    $$L(D) = (D_c / D)^{\alpha_D}\,;\quad \alpha_D \sim 0.095,\ \ D_c \sim 5.4 \times 10^{13} \text{ (tokens)} \tag{1.2}$$
>
> 3. When training with a limited amount of compute, a sufficiently large
>    dataset, an optimally-sized model, and a sufficiently small batch
>    size (making optimal use of compute):
>    $$L(C_{\min}) = (C_c^{\min}/C_{\min})^{\alpha_C^{\min}}\,;\quad \alpha_C^{\min} \sim 0.050,\ \ C_c^{\min} \sim 3.1 \times 10^8 \text{ (PF-days)} \tag{1.3}$$

> These relations hold across eight orders of magnitude in $C_{\min}$,
> six orders of magnitude in $N$, and over two orders of magnitude in
> $D$. They depend very weakly on model shape and other Transformer
> hyperparameters (depth, width, number of self-attention heads), with
> specific numerical values associated with the Webtext2 training set …
> The precise numerical values of $N_c, C_c^{\min}$, and $D_c$ depend on
> the vocabulary size and tokenization and hence do not have a
> fundamental meaning.

## §1.2 The joint $L(N,D)$ form {#sec-1-2-joint}

> Equation (1.1) and (1.2) together suggest that as we increase the model
> size, we should increase the dataset size sublinearly according to
> $D \propto N^{\alpha_N/\alpha_D} \sim N^{0.74}$. In fact, we find that
> there is a single equation combining (1.1) and (1.2) that governs the
> simultaneous dependence on $N$ and $D$ and governs the degree of
> overfitting:
>
> $$L(N, D) = \left[ \left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D} \right]^{\alpha_D} \tag{1.5}$$

## §1.2 Compute-optimal allocation rule (the controversial one) {#sec-1-2-compute-allocation}

> When training within a fixed compute budget $C$, but with no other
> constraints, Equation (1.6) leads to the prediction that the optimal
> model size $N$, optimal batch size $B$, optimal number of steps $S$,
> and dataset size $D$ should grow as
>
> $$N \propto C^{\alpha_C^{\min}/\alpha_N},\quad B \propto C^{\alpha_C^{\min}/\alpha_B},\quad S \propto C^{\alpha_C^{\min}/\alpha_S},\quad D = B \cdot S \tag{1.7}$$
>
> with
>
> $$\alpha_C^{\min} = 1\,/\,(1/\alpha_S + 1/\alpha_B + 1/\alpha_N) \tag{1.8}$$
>
> which closely matches the empirically optimal results $N \propto
> C_{\min}^{0.73}$, $B \propto C_{\min}^{0.24}$, and $S \propto
> C_{\min}^{0.03}$. As the computational budget $C$ increases, it should
> be spent primarily on larger models, without dramatic increases in
> training time or dataset size.

This is the rule that Chinchilla overturned: $N \propto C^{0.73}$ implies
that a 10× compute budget should buy a ~5.4× larger model and only ~1.4×
more data. Hoffmann et al. (2022) showed the correct exponents are
~0.50/0.50, i.e., **equal scaling**.

## §2.1 Definition of $N$ and $C$ {#sec-2-1-definitions}

> We use $N$ to denote the model size, which we define as the number of
> *non-embedding* parameters
>
> $$N \approx 2 d_{\text{model}} n_{\text{layer}} (2 d_{\text{attn}} + d_{\text{ff}}) = 12 n_{\text{layer}} d_{\text{model}}^2 \quad \text{with the standard } d_{\text{attn}} = d_{\text{ff}}/4 = d_{\text{model}}. \tag{2.1}$$

The much-quoted $C \approx 6 N D$ heuristic comes from this paper:

> Evaluating a forward pass of the Transformer involves roughly
> $C_{\text{forward}} \approx 2 N + 2 n_{\text{layer}} n_{\text{ctx}} d_{\text{attn}}$
> add-multiply operations, where the factor of two comes from the
> multiply-accumulate operation used in matrix multiplication. … For
> contexts and models with $d_{\text{model}} > n_{\text{ctx}}/12$, the
> context-dependent computational cost per token is a relatively small
> fraction of the total compute. … Accounting for the backwards pass
> (approximately twice the compute as the forwards pass), we then define
> the estimated non-embedding compute as $C \approx 6N$ floating point
> operators per training token.

So total training compute $C \approx 6ND$ — the formula every subsequent
scaling paper inherits.

## §1.3 Notation — what $N$ excludes {#sec-1-3-notation}

> $N$ – the number of model parameters, *excluding all vocabulary and
> positional embeddings*

This exclusion of embeddings matters: Kaplan's $N$ is **non-embedding**
parameters; later scaling work (Chinchilla onward) often reports total
parameters, and the small constant offset matters for tight comparisons.

## Why this paper still matters (and why it's superseded) {#why-matters}

The three power laws (1.1)–(1.3) and the joint form (1.5) are still
broadly correct and used as the *functional form* in essentially every
follow-up scaling paper, including Chinchilla itself. What was wrong was
the **exponent allocation** in (1.7): Kaplan inferred $\alpha_N \approx
0.076$ and $\alpha_C^{\min} \approx 0.050$ from data that had a fixed
LR-decay horizon for all model sizes, biasing the fits toward "more
model, less data." When Hoffmann et al. (2022) varied the LR-decay
horizon to match the actual training-token count, the exponents flipped
to roughly equal scaling.

Treat Kaplan as: (a) the canonical statement of the power-law functional
forms, (b) the canonical compute-budget formula $C \approx 6ND$, (c)
historical evidence of how subtle methodological choices (LR schedule
length) can dominate apparent results in scaling studies. The specific
allocation rule "$N \propto C^{0.73}$" is **superseded**.
