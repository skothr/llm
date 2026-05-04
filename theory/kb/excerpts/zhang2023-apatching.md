---
paper_key: zhang2023-apatching
title: "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods"
authors: Zhang, Nanda
year: 2023
venue: ICLR 2024
arxiv: 2309.16042
local_pdf: theory/sources/papers/zhang2023_apatching-best-practices.pdf
type: excerpts
note: Verbatim quotations from the v2 arXiv PDF (Jan 2024). The first systematic methodology study of activation patching: shows the choice of corruption method (Gaussian Noising vs. Symmetric Token Replacement) and metric (probability vs. logit difference) materially changes which components get implicated. Recommends STR + logit difference as default. The paper that turned activation patching from an artisanal technique into a documented protocol.
---

# Excerpts — Zhang & Nanda 2023, "Towards Best Practices of Activation Patching"

## Abstract — three degrees of freedom and their consequences {#abstract}

> Mechanistic interpretability seeks to understand the internal
> mechanisms of machine learning models, where localization—identifying
> the important model components—is a key step. Activation patching,
> also known as causal tracing or interchange intervention, is a
> standard technique for this task (Vig et al., 2020), but the
> literature contains many variants with little consensus on the choice
> of hyperparameters or methodology. In this work, we systematically
> examine the impact of methodological details in activation patching,
> including evaluation metrics and corruption methods. […] Backed by
> empirical observations, we give conceptual arguments for why certain
> metrics or methods may be preferred. Finally, we provide
> recommendations for the best practices of activation patching going
> forwards.

## §1 What activation patching is and what it gives you {#sec-1}

> A basic goal in MI is *localization*: identify the specific model
> components responsible for particular functions. Activation patching,
> also known as *causal tracing*, *interchange intervention*, *causal
> mediation analysis* or *representation denoising*, is a standard tool
> for localization in language models (Vig et al., 2020; Meng et al.,
> 2022). The method attempts to pinpoint activations that causally
> affect on the output. Specifically, it involves 3 forward passes of
> the model: (1) on a clean prompt while caching the latent activations;
> (2) on a corrupted prompt; and (3) on the corrupted prompt but
> replacing the activation of a specific model component by its clean
> cache. For instance, the clean prompt can be "The Eiffel Tower is in"
> and the corrupted one with the subject replaced by "The Colosseum". If
> the model outputs "Paris" in step (3) but not in (2), then it suggests
> that the specific component being patched is important for producing
> the answer.

## §2.1 Background — the canonical procedure restated {#sec-2-1}

> Activation patching identifies the important model components by
> intervening on their latent activations. The method involves a clean
> prompt ($X_\text{clean}$, e.g., "The Eiffel Tower is in") with an
> associated answer $r$ ("Paris"), a corrupted prompt ($X_\text{corrupt}$,
> e.g., "The Colosseum is in"), and three model runs:
>
> (1) Clean run: run the model on $X_\text{clean}$ and cache activations
>     of a set of given model components, such as MLP or attention heads
>     outputs.
> (2) Corrupted run: run the model on $X_\text{corrupt}$ and record the
>     model outputs.
> (3) Patched run: run the model on $X_\text{corrupt}$ with a specific
>     model component's activation restored from the cached value of the
>     clean run.
>
> Finally, we evaluate the patching effect, such as $\mathbb{P}(\text{"Paris"})$
> in the patched run (3) compared to the corrupted run (2). Intuitively,
> corruption hurts model performance while patching restores it.
> Patching effect measures how much the patching intervention restores
> performance, which indicates the importance of the activation.

## §2.1 Corruption methods — GN vs. STR {#sec-2-1-corruption}

> **Corruption methods.** To generate $X_\text{corrupt}$, **GN** adds
> Gaussian noise $\mathcal{N}(0, \nu)$ to the embeddings of certain key
> tokens, where $\nu$ is 3 times the standard deviation of the token
> embeddings from the textset. **STR** replaces the key tokens by
> similar ones with equal sequence length. In STR, let $r'$ denote the
> answer of $X_\text{corrupt}$ ("Rome"). All implementations of STR in
> this paper yield in-distribution prompts such that $X_\text{corrupt}$
> is identically distributed as a fresh draw of a clean prompt.

## §2.1 Metrics — probability vs. logit difference vs. KL {#sec-2-1-metrics}

Let $\text{cl}, *, \text{pt}$ be the clean, corrupted, and patched runs.

> - **Probability**: $\mathbb{P}(r)$; e.g., $\mathbb{P}(\text{"Paris"})$.
>   The patching effect is $\mathbb{P}_\text{pt}(r) - \mathbb{P}_*(r)$.
> - **Logit difference**: $\text{LD}(r, r') = \text{Logit}(r) -
>   \text{Logit}(r')$; e.g., $\text{Logit}(\text{"Paris"}) -
>   \text{Logit}(\text{"Rome"})$. The patching effect is given by
>   $\text{LD}_\text{pt}(r, r') - \text{LD}_*(r, r')$. Following Wang et
>   al. (2023), we always normalize this by $\text{LD}_\text{cl}(r, r')
>   - \text{LD}_*(r, r')$, so it typically lies in $[0, 1]$, where 1
>   corresponds to fully restored performance and 0 to the corrupted run
>   performance.
> - **KL divergence**: $D_\text{KL}(P_\text{cl} \mid\mid P)$, the
>   Kullback-Leibler divergence from the probability distribution of
>   model outputs in the clean run. The patching effect is
>   $D_\text{KL}(P_\text{cl} \mid\mid P_*) - D_\text{KL}(P_\text{cl}
>   \mid\mid P_\text{pt})$.

## §1 Findings — GN can break model internals; LD beats P {#sec-findings}

> **Findings**. Our contributions uncover nuanced discrepancies within
> activation patching techniques applied to language models. On
> corruption method, we show that GN and STR can lead to inconsistent
> localization and circuit discovery outcomes (Section 3.1). Towards
> explaining the gaps, we posit that GN breaks model's internal
> mechanisms by putting it off distribution. We give tentative evidence
> for this claim in the setting of IOI circuit discovery (Section 3.2).
> We believe that this is a fundamental concern in using GN corruption
> for activation patching. On evaluation metrics, we provide an
> analogous set of differences between logit difference and probability
> (Section 4), including an observation that probability can overlook
> negative model components that hurt performance.

## §1 Recommendations {#sec-recommendations}

> **Recommendations for practice**. At a high-level, our findings
> highlight the sensitivity of activation patching to methodological
> details. Backed by our analysis, we make several recommendations on
> the application of activation patching in language model
> interpretability (Section 6). We advocate for STR, as it supplies
> in-distribution corrupted prompts that help to preserve consistent
> model behavior. On evaluation metric, we recommend logit difference,
> as we argue that it offers fine-grained control over the localization
> outcomes and is capable of detecting negative modules.
