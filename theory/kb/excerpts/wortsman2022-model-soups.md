---
paper_key: wortsman2022-model-soups
title: "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time"
authors: Wortsman, Ilharco, Gadre, Roelofs, Gontijo-Lopes, Morcos, Namkoong, Farhadi, Carmon, Kornblith, Schmidt
year: 2022
venue: ICML 2022
arxiv: 2203.05482
local_pdf: null
type: excerpts
note: "Excerpts from arXiv abstract page (verbatim) plus methodology widely reproduced in model-merging surveys (arXiv:2503.08998 §3) and downstream papers (Task Arithmetic §2; TIES §2). PDF not yet downloaded."
---

# Excerpts — Wortsman et al. 2022, "Model Soups"

## Abstract — averaging fine-tunes from a shared base {#abstract}

> The conventional recipe for maximizing model accuracy is to (1)
> train multiple models with various hyperparameters and (2) pick the
> individual model which performs best on a held-out validation set,
> discarding the remainder. In this paper, we revisit the second step
> of this procedure in the context of fine-tuning large pre-trained
> models, where fine-tuned models often appear to lie in a single low
> error basin. We show that **averaging the weights of multiple
> models fine-tuned with different hyperparameter configurations
> often improves accuracy and robustness**. Unlike a conventional
> ensemble, we may average many models without incurring any
> additional inference or memory costs—we call the results "model
> soups." When fine-tuning large pre-trained models such as CLIP,
> ALIGN, and a ViT-G pre-trained on JFT, our soup recipe provides
> significant improvements over the best model in a hyperparameter
> sweep on ImageNet.

## §3 Soup recipes {#sec-3}

Given $K$ fine-tuned checkpoints $\{\theta_k\}_{k=1}^K$ from the same
pre-trained base $\theta_0$:

**Uniform soup**:
$$
\theta_{\text{uniform}} = \frac{1}{K}\sum_{k=1}^K \theta_k.
$$

**Greedy soup**: sort models by validation accuracy; iterate $k = 1,
\ldots, K$. Maintain a running soup $\bar\theta$. At each step, include
$\theta_k$ in the soup iff doing so does not decrease validation
accuracy:
$$
\bar\theta_{k} = \frac{|S_{k-1}| \bar\theta_{k-1} + \theta_k}{|S_{k-1}| + 1}
$$
if $\mathrm{ValAcc}(\bar\theta_k) \geq \mathrm{ValAcc}(\bar\theta_{k-1})$,
else skip.

**Learned soup**: continuous mixing weights learned via gradient on
validation accuracy. Stronger but more expensive.

## §3 Linear-mode-connectivity assumption {#sec-3-lmc}

The soup recipe relies on **linear mode connectivity**: fine-tunes
from a shared base $\theta_0$ land in a connected loss basin, so the
average of weights stays in low-loss territory. The paper provides
empirical evidence (loss along the linear path between fine-tunes is
low) but does not prove sufficient conditions for LMC.

## §4 Headline results {#sec-4}

> Our model soup approach extends to multiple image classification
> and natural language processing tasks, improves out-of-distribution
> performance, and improves zero-shot performance on new downstream
> tasks. Finally, we relate the performance similarity of the
> weight-averaging and logit-ensembling approaches to the flatness of
> the loss and confidence of the predictions.

The headline is that (a) soups frequently beat the best single fine-
tune in a hyperparameter sweep without inference-time cost, and (b)
logit ensembling and weight averaging produce similar accuracy when
the loss landscape is locally flat — formalizing the LMC intuition.

[NOTE — pdf-not-available] Section numbers from ICML 2022 / arXiv v3.
