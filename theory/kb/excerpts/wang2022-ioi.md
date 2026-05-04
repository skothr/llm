---
paper_key: wang2022-ioi
title: "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small"
authors: Wang, Variengien, Conmy, Shlegeris, Steinhardt
year: 2022
venue: ICLR 2023
arxiv: 2211.00593
local_pdf: theory/sources/papers/wang2022-ioi_ioi-circuit.pdf
type: excerpts
note: Verbatim quotations from the v1 arXiv PDF (Nov 2022). The IOI circuit is the canonical end-to-end mech-interp case study: 26 attention heads in GPT-2 small (1.1% of all (head, token-position) pairs) reverse-engineered as the circuit responsible for indirect object identification. The paper introduces *path patching* as a refinement of activation patching that can isolate direct effects of one head on another. Faithfulness/completeness/minimality criteria from this paper are now standard for circuit validation.
---

# Excerpts — Wang et al. 2022, "Interpretability in the Wild: IOI Circuit"

## Abstract — what was found {#abstract}

> In this work, we bridge this gap by presenting an explanation for how
> GPT-2 small performs a natural language task called indirect object
> identification (IOI). Our explanation encompasses 26 attention heads
> grouped into 7 main classes, which we discovered using a combination
> of interpretability approaches relying on causal interventions. To our
> knowledge, this investigation is the largest end-to-end attempt at
> reverse-engineering a natural behavior "in the wild" in a language
> model. We evaluate the reliability of our explanation using three
> quantitative criteria — *faithfulness*, *completeness* and
> *minimality*.

## §1 The IOI task and the human-readable algorithm {#sec-1-task}

> We focus on understanding a specific natural language task that we
> call indirect object identification (IOI). In IOI, sentences such as
> "When Mary and John went to the store, John gave a drink to" should be
> completed with "Mary". We chose this task because it is linguistically
> meaningful and admits an interpretable algorithm: of the two names in
> the sentence, predict the name that isn't the subject of the last
> clause.

The reverse-engineered algorithm:

> Recall the example sentence "When Mary and John went to the store,
> John gave a drink to". The following human-interpretable algorithm
> suffices to perform this task:
>
> 1. Identify all previous names in the sentence (Mary, John, John).
> 2. Remove all names that are duplicated (in the example above: John).
> 3. Output the remaining name.

## §2 Definition of a circuit {#sec-2-definition}

> Just as the entire model $M$ defines a function $M(x)$ from inputs to
> logits, we also associate each circuit $C$ with a function $C(x)$ via
> *knockouts*. A knockout removes a set of nodes $K$ in a computational
> graph $M$ with the goal of "turning off" nodes in $K$ but capturing
> all other computations in $M$. Thus, $C(x)$ is defined by knocking
> out all nodes in $M \setminus C$ and taking the resulting logit
> outputs in the modified computational graph.

Knockout via **mean ablation** rather than zero ablation:

> A first naïve knockout approach consists of simply deleting each node
> in $K$ from $M$. The net effect of this removal is to set its output
> to 0. […] Perhaps because of this, we find zero ablation to lead to
> noisy results in practice.
>
> To address this, we instead knockout nodes through *mean ablation*:
> replacing them with their average activation value across some
> reference distribution […] In this work, all knockouts are performed
> in a modified $p_\text{IOI}$ distribution called $p_\text{ABC}$. […]
> In $p_\text{ABC}$, sentences no longer have a single plausible IO,
> but the grammatical structures from the $p_\text{IOI}$ templates are
> preserved.

## §3 The seven head classes {#sec-3-circuit}

> Our circuit contains three major classes of heads, corresponding to
> the three steps of the algorithm above:
>
> - **Duplicate Token Heads** identify tokens that have already appeared
>   in the sentence. They are active at the S2 token, attend primarily
>   to the S1 token, and signal that token duplication has occurred by
>   writing the position of the duplicate token.
> - **S-Inhibition Heads** remove duplicate tokens from Name Mover
>   Heads' attention. They are active at the END token, attend to the
>   S2 token, and write in the query of the Name Mover Heads, inhibiting
>   their attention to S1 and S2 tokens.
> - **Name Mover Heads** output the remaining name. They are active at
>   END, attend to previous names in the sentence, and copy the names
>   they attend to. Due to the S-Inhibition Heads, they attend to the
>   IO token over the S1 and S2 tokens.

Plus four minor classes:

> A fourth major family of heads writes in the *opposite* direction of
> the Name Mover Heads, thus decreasing the confidence of the
> predictions. We speculate that these *Negative Name Mover Heads*
> might help the model "hedge" so as to avoid high cross-entropy loss
> when making mistakes.
>
> There are also three minor classes of heads that perform related
> functions to the components above:
>
> - *Previous Token Heads* copy information about the token S to token
>   S1+1, the token after S1.
> - *Induction Heads* perform the same role as the Duplicate Token
>   Heads through an induction mechanism. They are active at position
>   S2, attend to token S1+1 (mediated by the Previous Token Heads),
>   and their output is used both as a pointer to S1 and as a signal
>   that S is duplicated.
> - Finally, *Backup Name Mover Heads* do not normally move the IO
>   token to the output, but take on this role if the regular Name Mover
>   Heads are knocked out.

## §3.1 Path patching — direct vs. indirect effects {#sec-3-1-path-patching}

> **Path patching**. We begin by searching for attention heads $h$
> directly affecting the model's logits. To differentiate indirect
> effect (where the influence of a component is mediated by another
> head) from direct effect, we designed a technique called **path
> patching** (Figure 1).
>
> Path patching replaces part of a model's forward pass with activations
> from a different input. Given inputs $x_\text{orig}$ and $x_\text{new}$,
> and a set of paths $\mathcal{P}$ emanating from a node $h$, path
> patching runs a forward pass on $x_\text{orig}$, but for the paths in
> $\mathcal{P}$ it replaces the activations for $h$ with those from
> $x_\text{new}$. In our case, $h$ will be a fixed attention head and
> $\mathcal{P}$ consists of all direct paths from $h$ to a set of
> components $R$, i.e. paths through residual connections and MLPs (but
> not through other attention heads); this measures the counterfactual
> effect of $h$ on the members of $R$.

## §4 Three criteria for circuit validity {#sec-4-criteria}

> Explanations for model behavior can easily be misleading or
> non-rigorous (Jain & Wallace, 2019; Bolukbasi et al., 2021). To remedy
> this problem, we formulate three criteria to help validate our circuit
> explanations. These criteria are:
>
> - **faithfulness** (the circuit can perform the task as well as the
>   whole model),
> - **completeness** (the circuit contains all the nodes used to perform
>   the task), and
> - **minimality** (the circuit doesn't contain nodes irrelevant to the
>   task).
>
> Our circuit shows significant improvements compared to a naïve (but
> faithful) circuit, but fails to pass the most challenging tests.

## §3.1 The "name-mover" copy score {#sec-3-1-copy-score}

> All three Name Mover Heads have a copy score above 95%, compared to
> less than 20% for an average head.

## §3 Insights — backup heads + redundancy {#sec-3-insights}

> By zooming in on a crisp task in a particular model, we obtained
> several insights about the challenges of mechanistic interpretability.
> In particular:
>
> - We identified several instances of heads implementing redundant
>   behavior (Michel et al., 2019). The most surprising were "Backup
>   Name-Mover Heads", which copy names to the correct position in the
>   output, but only when regular Name-Mover Heads are ablated. This
>   complicates the search for complete mechanisms, as different model
>   structure is found when some components are ablated.
> - We found known structures (specifically induction heads (Elhage et
>   al., 2021)) that were used in unexpected ways. Thus mainline
>   functionality of a component does not always give a full picture.
> - Finally, we identified heads reliably writing in the *opposite*
>   direction of the correct answer.
