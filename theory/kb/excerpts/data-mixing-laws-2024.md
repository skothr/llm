---
paper_key: data-mixing-laws-2024
title: "Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance"
authors: Ye, Liu, Sun, Zhan, Zhou, Qiu (Fudan + Shanghai AI Lab)
year: 2024
venue: ICLR 2025
arxiv: 2403.16952
local_pdf: theory/sources/papers/data-mixing-laws-2024.pdf
type: excerpts
note: Verbatim quotes from v2 (20 Mar 2025). Establishes the predictability of validation loss as an exponential function of mixture proportions, plus a nested scaling-law pipeline that predicts large-scale loss from small-model experiments.
---

# Excerpts — Ye et al. 2024, "Data Mixing Laws"

## Abstract — predictability + 48% step savings {#abstract}

> Pretraining data of large language models composes multiple domains (e.g., web texts, academic papers, codes), whose mixture proportions crucially impact the competence of outcome models. While existing endeavors rely on heuristics or qualitative strategies to tune the proportions, we discover the **quantitative predictability of model performance regarding the mixture proportions in function forms**, which we refer to as the *data mixing laws*. Fitting such functions on sample mixtures unveils model performance on unseen mixtures before actual runs, thus guiding the selection of an ideal data mixture. Furthermore, we propose **nested use of the scaling laws of training steps, model sizes, and our data mixing law** to enable predicting the performance of large models trained on massive data under various mixtures with only small-scale training. Moreover, experimental results verify that our method effectively optimizes the training mixture of a 1B model trained for 100B tokens in RedPajama, **reaching a performance comparable to the one trained for 48% more steps on the default mixture**.

## §1 Functional form (Eq. 1) {#sec-1-functional}

> we find that, given a mixture of $M$ domains, an exponential function over the linear combination of the proportions, i.e.,

$$L_i(r_{1...M}) = c_i + k_i \exp\!\left(\sum_{j=1}^M t_{ij} r_j\right) \tag{1}$$

> can predict the validation loss $L_i$ on any of the training domains $i$ accurately under a fixed model size and amount of training data, where $r_{1...M}$ are the proportions of the $M$ domains and $c_i, k_i, t_{ij}$ are parameters to fit.

## §1 Nested-scaling-laws pipeline {#sec-1-nested}

> Despite the predictability, fitting the function between mixture proportions and validation loss, or the *data mixing laws* for simplicity, requires samples of numerous runs with different mixtures. Running these experiments with the same model size and the amount of training data as the target model is unreasonably expensive. Fortunately, fruitful research on scaling laws demonstrates impressive results that fitting power laws with small models and small data effectively predicts the losses on larger models and data over orders of magnitudes … On this basis, we propose a pipeline to nested utilize the scaling laws of training steps, model sizes, and our data mixing law, so that we can study the impact of mixture proportions for the target model sizes and data amount with only experiments at the affordable scales.

## §2 Standard scaling-law backbone (Eq. 3) {#sec-2-power-law}

> For a wide range of factors $x$, scaling laws (Kaplan et al. 2020, Henighan et al. 2020, Hoffmann et al. 2022) show that their effect on the losses $L$ follows power laws

$$L = c + k x^\alpha \tag{3}$$

> where $c$, $k$, and $\alpha$ are parameters to fit and $x$ can be model sizes, numbers of training data, training steps, and the amount of computation.

## §2 Problem formalization {#sec-2-formalization}

> We study optimizing the mixture proportions of pretraining data for large language models. … Formally, for a training dataset comprising $M$ domains, we parameterize the function

$$L = f_\theta(\mathbf{r}) \tag{4}$$

> under the fixed model sizes and number of training steps, where $\mathbf{r} = r_{1...M}$ is the proportion of the $M$ domains. Harnessing this function, we seek a mixture that achieves the desired performance. Without loss of generality, we search for the mixture that reaches minimum validation loss, i.e., $\mathbf{r}^* = \arg\min_\mathbf{r} f_\theta(\mathbf{r})$ (Eq. 5).

## §3 Two structural challenges {#sec-3-challenges}

> To discover the data mixing laws, we encounter two challenges posed by their characteristics.
> **(i) Multi-variables.** For a data mixing law for $K$ domains, we should consider $K-1$ degrees of freedom in the mixture proportions and, correspondingly, $K-1$ variables in the target function.
> **(ii) Nonmonotonicity.** A monotonic relationship between losses and the proportion of any domain indicates that a lopsided mixture can achieve minimum loss without endeavors to balance domain proportions, which contradicts the practice. Therefore … the data mixing law we study should accommodate non-monotonic functions.

## §3.1 Two-domain pilot — log-linear regime {#sec-3-1-pilot}

> We train 70M and 160M language models on the mixture of Github and Pile-CC subset from the Pile dataset with five different mixture proportions $r \in \{0.25, 0.375, 0.5, 0.625, 0.75\}$ for Github. We train all models with a batch size of 1M tokens for 30k steps, which is 30B tokens in total.

> Results in Fig. 2 reveal the quantitative predictability of domain losses given the domain proportions. … domain losses in the log scale demonstrate a linear relationship to the domain proportion. … The result indicates that with other factors fixed, the domain losses of a pretrained language model regarding the domain proportion precisely fit into an exponential law

$$L_i(r_i) = c_i + k_i \exp(t_{ii} r_i) \tag{6}$$

## §3.2 Multi-domain — symmetry/compatibility principles {#sec-3-2-multi}

> We base our conjecture of possible forms on the following two principles:
> - **Compatibility.** The form can reduce to Eqn. 6 if the number of domains $M = 2$.
> - **Symmetry.** Any exchanging of variables should not change the functional form as we should not incorporate any domain-specific bias.

> Together, the two principles lead to candidate functions that replicate the exponential term in Eqn. 6 for each training domain and combine them through operations that adhere to commutative law.

> [Tested forms M1–M4; chose M4 with fewest coefficients $M+2$:]

$$L_i(\mathbf{r}_{1...M}) = c_i + k_i \exp\!\left(\sum_{j=1}^M t_{ij} r_j \right) \tag{7}$$

## §3.2 Coefficient interpretation {#sec-3-2-coefficients}

> $c_i$ represents losses that are not reducible by adjusting the data mixture. $t_{ij}$, depending on both training domain $i$ and validation domain $j$, shows the interaction between them. A negative $t_{ij}$ indicates that training data of domain $j$ helps reduce validation loss on domain $i$ and vice versa.
