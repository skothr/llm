---
paper_key: yadav2023-ties-merging
title: "TIES-Merging: Resolving Interference When Merging Models"
authors: Yadav, Tam, Choshen, Raffel, Bansal
year: 2023
venue: NeurIPS 2023
arxiv: 2306.01708
local_pdf: null
type: excerpts
note: "Excerpts from arXiv abstract page (verbatim) plus methodology summary reproduced in merging surveys (arXiv:2503.08998 §4) and the MergeKit paper §3. PDF not yet downloaded."
---

# Excerpts — Yadav et al. 2023, "TIES-Merging"

## Abstract — interference resolution {#abstract}

> Transfer learning – i.e., further fine-tuning a pre-trained model
> on a downstream task – can confer significant advantages, including
> improved downstream performance, faster convergence, and better
> sample efficiency. … However, **fine-tuned models that perform well
> on a single task often struggle to combine with other models'
> fine-tunes, especially through naive parameter averaging**. … In
> this paper, we demonstrate that previous merging techniques
> inadvertently lose valuable information due to two major sources of
> interference: (a) **interference due to redundant parameter
> values** and (b) **disagreement on the sign of a given parameter's
> values across models**. To address this, we propose our method,
> **TRIM, ELECT SIGN & MERGE (TIES-Merging)**, which introduces three
> novel steps when merging models: (1) **resetting parameters that
> only changed a small amount during fine-tuning**, (2) **resolving
> sign conflicts**, and (3) **merging only the parameters that are in
> alignment with the final agreed-upon sign**.

## §3 The three TIES steps {#sec-3}

Given task vectors $\{\tau_i\}_{i=1}^K$ from a shared base $\theta_0$:

**Step 1 — Trim.** For each task vector $\tau_i$, keep only the top
$k\%$ of entries by magnitude:
$$
\tilde\tau_{i,j} = \begin{cases}\tau_{i,j} & \text{if } |\tau_{i,j}| \in \text{top-}k\% \text{ of } |\tau_i|\\ 0 & \text{otherwise}\end{cases}
$$
Typical $k = 20\%$. Removes redundant low-magnitude updates.

**Step 2 — Elect sign.** For each coordinate $j$, compute the sign
that has the largest total magnitude across tasks:
$$
s_j = \mathrm{sign}\!\left(\sum_i \tilde\tau_{i,j}\right).
$$
This is the dominant-direction sign, weighted by magnitude.

**Step 3 — Disjoint merge.** For each coordinate, average only the
task vectors whose sign matches $s_j$:
$$
\bar\tau_j = \frac{1}{|\mathcal{A}_j|}\sum_{i \in \mathcal{A}_j} \tilde\tau_{i,j}, \quad \mathcal{A}_j = \{i : \mathrm{sign}(\tilde\tau_{i,j}) = s_j\}.
$$

The merged model is $\theta_0 + \lambda \bar\tau$, with $\lambda$
typically $1$ or tuned on a held-out set.

## §4 Empirical headline {#sec-4}

> We find that TIES-Merging outperforms several existing methods in
> diverse settings covering a range of modalities, domains, number of
> tasks, model sizes, architectures, and finetuning settings.

The paper sweeps NLP and vision tasks with up to 11 tasks merged,
showing TIES > Task Arithmetic > Uniform Soup as the typical ordering
on multi-task accuracy. Sign-conflict resolution is the largest
contributing factor; trimming gives smaller but consistent gains.

[NOTE — pdf-not-available] Section numbers from NeurIPS 2023.
