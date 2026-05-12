---
paper-key: xiong2020
title: "On Layer Normalization in the Transformer Architecture"
authors: Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang, Liu
year: 2020
venue: ICML
arxiv: 2002.04745
local_pdf: theory/sources/papers/xiong2020_layer_norm.pdf
extracted: 2026-05-05 (Phase 2.5 deepening)
---

# Verbatim excerpts — Xiong et al. 2020, "On Layer Normalization in the Transformer Architecture"

The paper that established Pre-LN as the default placement for layer
normalization in production Transformers. The theoretical contribution
is a mean-field-theory analysis showing Post-LN gradients near the
output layer scale with $O(\sqrt{L})$ (where $L$ is depth), motivating
the warmup, while Pre-LN gradients are $O(1)$ at initialization, allowing
warmup-free training. Every modern decoder-only LLM (GPT-3, LLaMA,
Mistral, Qwen, DeepSeek) uses Pre-LN.

## Abstract {#abstract}

> The Transformer is widely used in natural language processing tasks.
> To train a Transformer however, one usually needs a carefully designed
> learning rate warm-up stage, which is shown to be crucial to the final
> performance but will slow down the optimization and bring more
> hyper-parameter tunings. In this paper, we first study theoretically
> why the learning rate warm-up stage is essential and show that the
> location of layer normalization matters. Specifically, we prove with
> mean field theory that at initialization, for the original-designed
> Post-LN Transformer, which places the layer normalization between the
> residual blocks, the expected gradients of the parameters near the
> output layer are large. Therefore, using a large learning rate on
> those gradients makes the training unstable. The warm-up stage is
> practically helpful for avoiding this problem. On the other hand, our
> theory also shows that if the layer normalization is put inside the
> residual blocks (recently proposed as Pre-LN Transformer), the
> gradients are well-behaved at initialization. This motivates us to
> remove the warm-up stage for the training of Pre-LN Transformers. We
> show in our experiments that Pre-LN Transformers without the warm-up
> stage can reach comparable results with baselines while requiring
> significantly less training time and hyper-parameter tuning on a wide
> range of applications.

## §1 Contributions and motivation {#sec-1}

> Our contributions are summarized as follows:
>
> - We investigate two Transformer variants, the Post-LN Transformer and
>   the Pre-LN Transformer, using mean field theory. By studying the
>   gradients at initialization, we provide evidence to show why the
>   learning rate warm-up stage is essential in training the Post-LN
>   Transformer.
>
> - We are the first to show that the learning-rate warm-up stage can be
>   removed for the Pre-LN Transformer, which eases the hyper-parameter
>   tuning. We further show that by using proper learning rate
>   schedulers, the training time can be largely reduced on a wide range
>   of applications.

## §3.1 Post-LN vs Pre-LN — architectural definition {#sec-3-1}

The two variants differ only in where LayerNorm is applied. Post-LN
places it after the residual addition; Pre-LN places it before each
sub-layer (inside the residual branch) and adds a final LayerNorm before
the output projection.

**Post-LN Transformer** (the original design from Vaswani et al. 2017):

$$x_{l,i}^{\mathrm{post,1}} = \mathrm{MultiHeadAtt}(x_{l,i}^{\mathrm{post}}, [x_{l,1}^{\mathrm{post}}, \ldots, x_{l,n}^{\mathrm{post}}])$$
$$x_{l,i}^{\mathrm{post,2}} = x_{l,i}^{\mathrm{post}} + x_{l,i}^{\mathrm{post,1}}$$
$$x_{l,i}^{\mathrm{post,3}} = \mathrm{LayerNorm}(x_{l,i}^{\mathrm{post,2}})$$
$$x_{l,i}^{\mathrm{post,4}} = \mathrm{ReLU}(x_{l,i}^{\mathrm{post,3}} W^{1,l} + b^{1,l}) W^{2,l} + b^{2,l}$$
$$x_{l,i}^{\mathrm{post,5}} = x_{l,i}^{\mathrm{post,3}} + x_{l,i}^{\mathrm{post,4}}$$
$$x_{l+1,i}^{\mathrm{post}} = \mathrm{LayerNorm}(x_{l,i}^{\mathrm{post,5}})$$

**Pre-LN Transformer:**

$$x_{l,i}^{\mathrm{pre,1}} = \mathrm{LayerNorm}(x_{l,i}^{\mathrm{pre}})$$
$$x_{l,i}^{\mathrm{pre,2}} = \mathrm{MultiHeadAtt}(x_{l,i}^{\mathrm{pre,1}}, [x_{l,1}^{\mathrm{pre,1}}, \ldots, x_{l,n}^{\mathrm{pre,1}}])$$
$$x_{l,i}^{\mathrm{pre,3}} = x_{l,i}^{\mathrm{pre}} + x_{l,i}^{\mathrm{pre,2}}$$
$$x_{l,i}^{\mathrm{pre,4}} = \mathrm{LayerNorm}(x_{l,i}^{\mathrm{pre,3}})$$
$$x_{l,i}^{\mathrm{pre,5}} = \mathrm{ReLU}(x_{l,i}^{\mathrm{pre,4}} W^{1,l} + b^{1,l}) W^{2,l} + b^{2,l}$$
$$x_{l+1,i}^{\mathrm{pre}} = x_{l,i}^{\mathrm{pre,3}} + x_{l,i}^{\mathrm{pre,5}}$$
$$\text{Final LayerNorm: } x_{\mathrm{Final},i}^{\mathrm{pre}} \leftarrow \mathrm{LayerNorm}(x_{L+1,i}^{\mathrm{pre}})$$

## §3.2 The warmup schedule {#sec-3-2}

> A learning rate warm-up stage for the Post-LN Transformer seems
> critical (Popel & Bojar, 2018). We denote the learning rate of the
> t-th iteration as $\mathrm{lr}(t)$ and the maximum learning rate during
> training as $\mathrm{lr}_{\max}$. Given a predefined time frame
> $T_{\mathrm{warmup}}$, the learning rate scheduler for the first
> $T_{\mathrm{warmup}}$ iterations (Vaswani et al., 2018) is defined as
>
> $$\mathrm{lr}(t) = \frac{t}{T_{\mathrm{warmup}}} \mathrm{lr}_{\max}, \quad t \le T_{\mathrm{warmup}}. \tag{1}$$
>
> After this warm-up stage, the learning rate will be set by classical
> learning rate schedulers, such as the linear decay, the inverse
> square-root decay, or forced decay at particular iterations.

## §3.3 Theorem 1 — mean-field gradient scales {#sec-3-3-theorem1}

The load-bearing theoretical claim. Verbatim from §3.3:

> **Theorem 1** (Gradients of the last layer in the Transformer).
> Assume that $\|x^{\mathrm{post,5}}_{L,i}\|_2^2$ and
> $\|x^{\mathrm{pre}}_{L+1,i}\|_2^2$ are $(\epsilon, \delta)$-bounded
> for all $i$, where $\epsilon$ and $\delta = \delta(\epsilon)$ are
> small numbers. Then with probability at least
> $0.99 - \delta - \frac{0.9\epsilon}{1+\epsilon}$, for the Post-LN
> Transformer with $L$ layers, the gradient of the parameters of the
> last layer satisfies
>
> $$\left\|\frac{\partial \tilde{\mathcal{L}}}{\partial W^{2,L}}\right\|_F \leq O\!\left(d\sqrt{\ln d}\right)$$
>
> and for the Pre-LN Transformer with $L$ layers,
>
> $$\left\|\frac{\partial \tilde{\mathcal{L}}}{\partial W^{2,L}}\right\|_F \leq O\!\left(d\sqrt{\frac{\ln d}{L}}\right).$$

The paper continues:

> From Theorem 1, we can see that for the Post-LN Transformer, the
> scale of the gradients to the last FFN layer is of order
> $O(d\sqrt{\ln d})$ which is independent of $L$. For the Pre-LN
> Transformer, the scale of the gradients is much smaller.

And the extended empirical finding:

> Our main result is that the gradient norm in the Post-LN Transformer
> is large for the parameters near the output and will be likely to
> decay as the layer index $l$ decreases. On the contrary, the gradient
> norm in the Pre-Transformer will be likely to stay the same for any
> layer $l$.

The practical consequence: Post-LN needs warmup to compensate for
output-layer gradient scale; Pre-LN does not. [INTUITION: the $1/\sqrt{L}$
factor in the Pre-LN bound comes from the growing hidden-state norm —
Lemma 2 shows $E(\|x^{\mathrm{pre}}_{l,i}\|_2^2)$ grows linearly in $l$,
so the final LayerNorm divides by $\sqrt{L}$, taming all upstream gradients.]

## §4 Experimental confirmation {#sec-4-experiments}

> We find in the previous section that the gradients at initialization
> for Pre-LN Transformer are well-behaved. Given this observation, we
> deduce that the learning rate warm-up stage can be safely removed when
> training Pre-LN Transformer. In this section, we empirically verify it
> on two main tasks in NLP, machine translation and unsupervised
> pre-training.

Results from §4.2 (machine translation):

> First, as we can see from the figure, the learning rate warm-up stage
> is not critical anymore for training the Pre-LN Transformer and the
> performance of the learned model is competitive. For example, on the
> IWSLT14 De-En task, the BLEU score and validation loss of the Pre-LN
> Transformer can achieve around 34 and 4, which are comparable with the
> performance of the Post-LN Transformer.
>
> Second, the Pre-LN Transformer converges faster than the Post-LN
> Transformer using the same $\mathrm{lr}_{\max}$.

The warmup-sensitivity finding from §3.2 (verbatim):

> Without the warm-up stage, the BLEU score of the model trained with
> Adam optimizer can only achieve 8.45. As a comparison, the model
> trained using the warm-up stage can achieve around 34 in terms of BLEU
> score.
>
> [S]uch a warm-up stage has several disadvantages. First, its
> configuration significantly affects the final performance. The
> practitioners need a careful hyper-parameter tuning, which is
> computationally expensive for large-scale NLP tasks. Second, the
> warm-up stage could slow down the optimization. Standard optimization
> algorithms usually start with a large learning rate for fast
> convergence. However, when using the warm-up stage, the learning rate
> has to gradually increase from zero, which may make the training
> inefficient.

## Source notes

- Tier A canonical (ICML 2020 main conference). The mean-field-theory
  analysis is in Theorem 1 (§3.3) but the proofs occupy the technical
  appendices.
- PDF retrieved from arXiv (arXiv:2002.04745v2). Section / equation anchors stable.
- Modern frontier LLMs all use Pre-LN. The 2025 Peri-LN paper (Lee et
  al.) revisits this design space and argues that Peri-LN — with norm
  on **both** the input and the output of each sub-layer — improves on
  Pre-LN at frontier scale; see `kb/notes/architecture/normalization.md`
  for the lineage.

---

[Verified from PDF on 2026-05-12] Added §3.3/Theorem1 verbatim gradient-bound
(#sec-4-theorem1, replacing prior paraphrase), §4 experimental-confirmation
verbatim (#sec-5-experiments, replacing prior paraphrase). Abstract verified
verbatim. `local_pdf:` front-matter set.
