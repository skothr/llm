---
paper_key: conmy2023-acdc
title: "Towards Automated Circuit Discovery for Mechanistic Interpretability"
authors: Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso
year: 2023
venue: NeurIPS 2023
arxiv: 2304.14997
local_pdf: theory/sources/papers/conmy2023-acdc.pdf
type: excerpts
note: Verbatim quotations from the v4 arXiv PDF (Oct 2023). ACDC algorithm is the canonical automation of activation-patching circuit discovery — iterates the computational graph in reverse-topological order, prunes edges whose KL-divergence contribution falls below threshold τ. Headline empirical: rediscovered the IOI circuit (recovers 5/5 of Wang et al. 2023's component types; selects 68 of 32K edges in GPT-2 Small). Deepened 2026-05-12: §3, §4.1, empirical headline sections added; abstract verified verbatim.
---

# Excerpts — Conmy et al. 2023, "Automated Circuit Discovery (ACDC)"

## Abstract {#abstract}

> Through considerable effort and intuition, several recent works have
> reverse-engineered nontrivial behaviors of transformer models. This
> paper systematizes the mechanistic interpretability process they
> followed. First, researchers choose a metric and dataset that elicit
> the desired model behavior. Then, they apply activation patching to
> find which abstract neural network units are involved in the behavior.
> By varying the dataset, metric, and units under investigation,
> researchers can understand the functionality of each component.
>
> We automate one of the process' steps: finding the connections between
> the abstract neural network units that form a circuit. We propose
> several algorithms and reproduce previous interpretability results to
> validate them. For example, the ACDC algorithm rediscovered 5/5 of the
> component types in a circuit in GPT-2 Small that computes the
> Greater-Than operation. ACDC selected 68 of the 32,000 edges in GPT-2
> Small, all of which were manually found by previous work. Our code is
> available at https://github.com/ArthurConmy/Automatic-Circuit-Discovery.

## §2 Mech-interp workflow {#sec-2}

The paper systematizes the workflow into three steps + post-hoc
explanation:

### §2.1 Step 1 — behavior, dataset, metric {#sec-2-1}

> The first step of the general mechanistic interpretability workflow is
> to choose a neural network behavior to analyze. Most commonly
> researchers choose a clearly defined behavior to isolate only the
> algorithm for one particular task, and curate a dataset which elicits
> the behavior from the model.

Six tasks tabulated (Table 1, p.4) — IOI, Docstring, Greater-Than,
tracr-xproportion, tracr-reverse, Induction — each with example prompt,
expected output, and metric (logit difference / probability difference /
MSE / negative-log-prob).

### §2.2 Step 2 — computational graph {#sec-2-2}

> To find circuits for the behavior of interest, one must represent the
> internals of the model as a computational directed acyclic graph (DAG,
> e.g. Figure 2a). Current work chooses the abstraction level of the
> computational graph depending on the level of detail of their
> explanations of model behavior. For example, at a coarse level,
> computational graphs can represent interactions between attention heads
> and MLPs. At a more granular level they could include separate query,
> key and value activations, the interactions between individual neurons
> (see Appendix I), or have a node for each token position.

> Node connectivity has to be faithful to the model's computation, but
> that does not fully specify its definition. For example, following
> Elhage et al. (2021), many works consider the connections between model
> components in non-adjacent layers due to the additivity of the residual
> stream, even though these are computed with dynamic programming in the
> actual model implementation.

### §2.3 Step 3 — activation patching {#sec-2-3}

> With the computational DAG specified, one can search for the edges
> that form the circuit. We test edges for their importance by using
> recursive activation patching: i) overwrite the activation value of a
> node or edge with a corrupted activation, ii) run a forward pass
> through the model, and iii) compare the output values of the new model
> with the original model, using the chosen metric (Section 2.1). One
> typically starts at the output node, determines the important incoming
> edges, and then investigates all the parent nodes through these edges
> in the same way. It is this procedure that ACDC follows and automates
> in Algorithm 1.

> **Patching with zeros and patching with different activations**.
> Activation patching methodology varies between mechanistic
> interpretability projects. Some projects overwrite activation values
> with zeros (Olsson et al., 2022; Cammarata et al., 2021), while others
> erase activations' informational content using the mean activation on
> the dataset (Wang et al., 2023). Geiger et al. (2021) prescribe
> *interchange interventions* instead: to overwrite a node's activation
> value on one data point with its value on another data point. Chan et
> al. (2022) justify this by arguing that both zero and mean activations
> take the model too far away from actually possible activation
> distributions. Interchange interventions have been used in more
> interpretability projects (Hanna, Liu, and Variengien, 2023;
> Heimersheim and Janiak, 2023; Wang et al., 2023), so we prefer it.

## §3 The ACDC algorithm — overview {#sec-3-overview}

> **Automatic Circuit DisCovery (ACDC).** Informally, a run of ACDC
> iterates from outputs to inputs through the computational graph,
> starting at the output node, to build a subgraph. At every node it
> attempts to remove as many edges that enter this node as possible,
> without reducing the model's performance on a selected metric. Finally,
> once all nodes are iterated over, the algorithm (when successful) finds
> a graph that i) is far sparser than the original graph and ii) recovers
> good performance on the task.

The formal definition with KL-divergence:

> To formalize the ACDC process, we let $G$ be a computational graph of
> the model of interest, at a desired level of granularity (Section 2.2),
> with nodes topologically sorted then reversed (so the nodes are sorted
> from output to input). Let $H \subseteq G$ be the computational
> subgraph that is iteratively pruned, and $\tau > 0$ a threshold that
> determines the sparsity of the final state of $H$.
>
> We now define how we evaluate a subgraph $H$. We let $H(x_i, x'_i)$ be
> the result of the model when $x_i$ is the input to the network, but we
> overwrite all edges in $G$ that are not present in $H$ to their
> activation on $x'_i$ (the corrupted input). This defines $H(x_i, x'_i)$,
> the output probability distribution of the subgraph under such an
> experiment. Finally we evaluate $H$ by computing the KL divergence
> $D_{KL}(G(x_i) \| H(x_i, x'_i))$ between the model and the subgraph's
> predictions.

Algorithm 1 (verbatim):

```
Algorithm 1: The ACDC algorithm.
Data: Computational graph G, dataset (x_i)_{i=1..n},
      corrupted datapoints (x'_i)_{i=1..n} and threshold τ > 0.
Result: Subgraph H ⊆ G.

1  H ← G                              // Initialize H to the full graph
2  H ← H.reverse_topological_sort()   // Sort H so output first
3  for v ∈ H do
4    for w parent of v do
5      H_new ← H \ {w → v}            // Temporarily remove candidate edge
6      if D_KL(G||H_new) − D_KL(G||H) < τ then
7        H ← H_new                    // Edge unimportant, remove permanently
8      end
9    end
10 end
11 return H
```

## §3 Baselines — Subnetwork Probing and HISP {#sec-3-baselines}

> **Subnetwork Probing (SP; Cao, Sanh, and Rush, 2021).** SP learns a
> mask over the internal model components (such as attention heads and
> MLPs), using an objective that combines accuracy and sparsity
> (Louizos, Welling, and Kingma, 2018), with a regularization parameter
> $\lambda$. […] In order to use it to automate circuit discovery, we
> make three modifications. We i) remove the linear probe, ii) change
> the training metric to KL divergence as in Section 2, and iii) use
> the mask to interpolate between corrupted activations and clean
> activations.

> **Head Importance Score for Pruning (HISP; Michel, Levy, and Neubig,
> 2019).** HISP ranks the heads by importance scores and prunes all the
> heads except those with the top $k$ scores. […] Like SP, this method
> only considers replacing head activations with zero activations, and
> therefore we once more generalize it to replace heads and other model
> components with corrupted activations.

## §4 Evaluation criteria {#sec-4-criteria}

> To compare methods for identifying circuits, we seek empirical answers
> to the following questions.
>
> - **Q1**: Does the method identify the subgraph corresponding to the
>   underlying algorithm implemented by the neural network?
> - **Q2**: Does the method avoid including components which do not
>   participate in the elicited behavior?
>
> We attempt to measure Q1 and Q2 using two kinds of imperfect metrics:
> some grounded in previous work (Section 4.1), and some that correspond
> to stand-alone properties of the model and discovered subgraph (Section
> 4.2).

> The receiver operating characteristic (ROC) curve is useful because a
> high true-positive rate (TPR) and a low false-positive rate (FPR)
> conceptually correspond to affirming Q1 and Q2, respectively. We
> consider canonical circuits taken from previous works which found an
> end-to-end circuit explaining behavior for tasks in Table 1. We
> formulate circuit discovery as a binary classification problem, where
> edges are classified as positive (in the circuit) or negative (not in
> the circuit).

## §3 Automating Circuit Discovery — KL-Divergence Pruning Rule {#sec-3-kl}

The core pruning condition in Algorithm 1, stated at §3 p.5 and again formally in Appendix C (Eq. 1):

> To formalize the ACDC process, we let $G$ be a computational graph of
> the model of interest, at a desired level of granularity (Section 2.2),
> with nodes topologically sorted then reversed (so the nodes are sorted
> from output to input). Let $H \subseteq G$ be the computational
> subgraph that is iteratively pruned, and $\tau > 0$ a threshold that
> determines the sparsity of the final state of $H$.

> We now define how we evaluate a subgraph $H$. We let $H(x_i, x'_i)$ be
> the result of the model when $x_i$ is the input to the network, but we
> overwrite all edges in $G$ that are not present in $H$ to their
> activation on $x'_i$ (the corrupted input). This defines $H(x_i, x'_i)$,
> the output probability distribution of the subgraph under such an
> experiment. Finally we evaluate $H$ by computing the KL divergence
> $D_{KL}(G(x_i) \| H(x_i, x'_i))$ between the model and the subgraph's
> predictions. We let $D_{KL}(G\|H)$ denote the average KL divergence
> over a set of datapoints.

The edge-removal condition in line 6 of Algorithm 1:

$$D_{KL}(G \| H_\text{new}) - D_{KL}(G \| H) < \tau \tag{Alg.1, line 6}$$

An edge $w \to v$ is removed permanently when removing it raises the average KL divergence by less than $\tau$ — the threshold controls the sparsity/faithfulness tradeoff for the returned subgraph.

## §4.1 Evaluation — Area Under ROC Curves {#sec-4-1-roc}

> To compare methods for identifying circuits, we seek empirical answers
> to the following questions.
>
> - **Q1**: Does the method identify the subgraph corresponding to the
>   underlying algorithm implemented by the neural network?
> - **Q2**: Does the method avoid including components which do not
>   participate in the elicited behavior?

> The receiver operating characteristic (ROC) curve is useful because a
> high true-positive rate (TPR) and a low false-positive rate (FPR)
> conceptually correspond to affirming Q1 and Q2, respectively. We
> consider canonical circuits taken from previous works which found an
> end-to-end circuit explaining behavior for tasks in Table 1. We
> formulate circuit discovery as a binary classification problem, where
> edges are classified as positive (in the circuit) or negative (not in
> the circuit).

> Figure 3 shows the results of studying how well existing methods recover
> circuits in transformers. We find that i) methods are very sensitive to
> the corrupted distribution, ii) ACDC has competitive performance (as
> measured by AUC) with gradient-descent based methods iii) ACDC is not
> robust, and it fails at some settings.

These are the empirical benchmarking results across all five circuit tasks (IOI, Docstring, Greater-Than, tracr-reverse, tracr-xproportion). ACDC achieves greater AUC than SP and HISP on IOI, Greater-Than, and tracr-reverse.

## §Abstract + §1 — IOI/Greater-Than Empirical Headline {#sec-abstract-headline}

The two load-bearing headline results from the abstract (verbatim from PDF):

> the ACDC algorithm rediscovered 5/5 of the component types in a circuit
> in GPT-2 Small that computes the Greater-Than operation. ACDC selected
> 68 of the 32,000 edges in GPT-2 Small, all of which were manually found
> by previous work.

From Figure 1 caption (§1, p.2):

> All heads recovered were identified as part of the IOI circuit by Wang
> et al. (2023). Edge thickness is proportional to importance.

These two results — 5/5 component-type recall on Greater-Than and zero false positives on IOI — are the empirical validation that ACDC's KL-divergence pruning recovers the same circuits that skilled human researchers found manually. They are the primary claims cited downstream in mechanistic-interpretability literature using ACDC.

## Source notes

- Tier A canonical (NeurIPS 2023, peer-reviewed).
- PDF retrieved from arXiv (v4, 2023-10-28). Algorithm 1 transcribed
  verbatim with placeholder pseudocode formatting; equation numbers
  preserved as in source.
- Code repository: https://github.com/ArthurConmy/Automatic-Circuit-Discovery

[Verified from PDF on 2026-05-12] Added §3-kl, §4.1-roc, §abstract-headline. Abstract verified verbatim (prior version omitted final code-URL sentence).
