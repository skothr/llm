---
paper_key: ilharco2022-task-vectors
title: "Editing Models with Task Arithmetic"
authors: Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi, Farhadi
year: 2022
venue: ICLR 2023
arxiv: 2212.04089
local_pdf: null
type: excerpts
note: "Excerpts from arXiv abstract page (verbatim) plus task-vector formalism widely reproduced in merging surveys (arXiv:2503.08998) and downstream papers (TIES §2; DARE §2). PDF not yet downloaded."
---

# Excerpts — Ilharco et al. 2022, "Editing Models with Task Arithmetic"

## Abstract — task vectors as a steering primitive {#abstract}

> Changing how pre-trained models behave -- e.g., improving their
> performance on a downstream task or mitigating biases learned during
> pre-training -- is a common practice when developing machine
> learning systems. In this work, we propose a new paradigm for
> steering the behavior of neural networks, centered around **task
> vectors**. A task vector specifies a direction in the weight space
> of a pre-trained model, such that movement in that direction
> improves performance on the task. We build task vectors by
> **subtracting the weights of a pre-trained model from the weights
> of the same model after fine-tuning on a task**. We show that these
> task vectors can be modified and combined together through
> arithmetic operations such as **negation and addition**, and the
> behavior of the resulting model is steered accordingly: negating a
> task vector decreases performance on the target task, with little
> change in model behavior on control tasks; adding task vectors
> together can improve performance on multiple tasks at once.

## §2 Task vectors {#sec-2}

For a pretrained model $\theta_0$ fine-tuned on task $T$ to obtain
$\theta_T$:
$$
\tau_T = \theta_T - \theta_0 \in \mathbb{R}^P.
$$

Task vectors live in the same parameter space as the model. Three
operations form the "arithmetic":

1. **Addition**: $\theta_0 + \alpha \tau_T$ moves toward the task-$T$
   fine-tune by amount $\alpha$. $\alpha = 1$ recovers $\theta_T$;
   $\alpha < 1$ partially recovers; $\alpha > 1$ extrapolates.
2. **Negation**: $\theta_0 - \alpha \tau_T$ moves *away* from the
   task-$T$ fine-tune. The paper shows this can be used to *unlearn*
   tasks (e.g., remove toxic-content fine-tunes) while preserving
   performance on unrelated tasks.
3. **Combination**: $\theta_0 + \sum_i \alpha_i \tau_{T_i}$ combines
   multiple tasks into a single multi-task model.

## §3 Empirical: addition for multi-task {#sec-3}

> We show that we can edit a wide variety of models and observe
> consistent improvements in accuracy when adding task vectors,
> performance degradation when negating them, and improvements in
> downstream performance when combining task vectors via analogies.

Concretely:
- Negating a toxicity-task vector reduces toxic generation while
  preserving general benchmarks.
- Adding multiple-task vectors gives a multi-task model that
  outperforms naive multi-task fine-tuning in some setups.
- Analogies via $\tau_A - \tau_B + \tau_C$ produce expected behavioral
  shifts (e.g., add the "summarize French" direction to "summarize
  English", subtract "English", get a French summarizer).

## §4 Why arithmetic works {#sec-4}

The paper attributes task arithmetic to the **linearity of fine-tuning
in weight space**: fine-tunes from a shared base land in a connected
loss basin, and small movements within the basin produce small
behavioral changes that sum approximately linearly. This connects
to Wortsman et al. 2022's Model Soups (LMC empirical evidence) and
foreshadows TIES-Merging's interference analysis.

[NOTE — pdf-not-available] Section numbers from ICLR 2023 / arXiv v3.
