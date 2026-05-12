---
paper_key: heimersheim2024
title: "How to use and interpret activation patching"
authors: Heimersheim, Nanda
year: 2024
venue: arXiv:2404.15255
arxiv: 2404.15255
local_pdf: theory/sources/papers/heimersheim2024_apatching-howto.pdf
type: excerpts
note: Verbatim quotations from the v1 arXiv PDF (Apr 2024). Practitioner-facing companion to Zhang & Nanda 2023's "best practices" paper. Establishes the noising/denoising distinction and the AND/OR-gate worked example that shows why the two directions are NOT symmetric. Includes the comprehensive metric pitfall list (logit-diff false-positives, logprob saturation, probability non-linearity, discrete-metric problems).
---

# Excerpts — Heimersheim & Nanda 2024, "How to use and interpret activation patching"

## Abstract {#abstract}

> Activation patching is a popular mechanistic interpretability
> technique, but has many subtleties regarding how it is applied and how
> one may interpret the results. We provide a summary of advice and best
> practices, based on our experience using this technique in practice.
> We include an overview of the different ways to apply activation
> patching and a discussion on how to interpret the results. We focus on
> what evidence patching experiments provide about circuits, and on the
> choice of metric and associated pitfalls.

## §1.1 What is activation patching {#sec-1-1}

> Activation patching (also referred to as Interchange Intervention,
> Causal Tracing, Resample Ablation, or Causal Mediation Analysis) is
> the technique of replacing internal activations of a neural net. […]
> Variants of this technique have been widely used in the literature
> (Vig et al., 2020; Geiger et al., 2021a; Geiger, Richardson, and Potts,
> 2020; […] Meng et al., 2022; Wang et al., 2022; […] Conmy et al.,
> 2023; Todd et al., 2023; […]). Here we focus on the technique where we
> overwrite some activations during a model run with cached activations
> from a previous run (on a different input), and observe how this
> affects the model's output.

The 6-step protocol (§1.3, p.1):

> 1. Choose two similar prompts that differ in some key fact or otherwise
>    elicit different model behaviour: E.g. "The Colosseum is in" and
>    "The Louvre is in" to vary the landmark but control for everything
>    else.
> 2. Choose which model activations to patch: E.g. MLP outputs.
> 3. Run the model with the first prompt — the source prompt — and save
>    its internal activations: E.g. "The Louvre is in" (source).
> 4. Run the model with the second prompt — the destination prompt — but
>    overwrite the selected internal activations with the previously
>    saved ones (patching): E.g. "The Colosseum is in" (destination).
> 5. See how the model output has changed. The outputs of this patched
>    run are typically somewhere between what the model would output for
>    the un-patched first prompt or second prompt: E.g. observe change in
>    the output logits for "Paris" and "Rome".
> 6. Repeat for all activations of interest: E.g. sweep to test all MLP
>    layers.

## §2.3 Noising vs Denoising {#sec-2-3}

The key conceptual distinction the paper introduces:

> 1. **Denoising**: We can patch activations from a clean first prompt
>    into a corrupted second prompt "clean → corrupt". That is running
>    the model on the clean prompt while saving its activations, then
>    running the model on the corrupted prompt while overwriting some of
>    its activations with previously saved clean-prompt activations. We
>    observe which patch restores the clean-prompt behaviour, i.e.
>    *patching which activations were sufficient to restore the
>    behaviour*.
> 2. **Noising**: Or you can patch activations from a corrupted first
>    prompt into a clean second prompt "corrupt → clean". That is running
>    the model on the corrupted prompt while saving its activations,
>    then running the model on the clean prompt while overwriting some
>    of its activations with previously saved corrupt-prompt activations.
>    We observe which patch breaks the clean-prompt behaviour, i.e.
>    *patching which activations were necessary to maintain for the
>    behaviour*.
>
> An important and underrated point is that these two directions can be
> very different, and are not just symmetric mirrors of each other.

| Technique | Source (saved) | Source run input | Destination | Destination input | Observation |
|---|---|---|---|---|---|
| Clean → corrupted (Denoising, Causal Tracing) | First run activations (clean) | Clean tokens | Second run activations (corrupted) | Corrupt tokens | What restores behaviour |
| Corrupted → clean (Noising, Resample Ablation) | First run activations (corrupted) | Corrupt tokens | Second run activations (clean) | Clean tokens | What breaks behaviour |

## §2.4 AND vs OR gates — noising and denoising are not symmetric {#sec-2-4}

> Consider a hypothetical circuit of three components A, B, and C that
> are connected with an AND or an OR gate. They are embedded in a much
> larger network, and of the three just C is connected to the output. We
> run an experiment where we patch all components using the denoising or
> noising technique.

**AND circuit: C = A AND B**

> - **Denoising** (clean → corrupt patching): Denoising either A or B has
>   no effect on the output, only denoising C restores the output. This
>   is because denoising A still leaves B at the corrupted (incorrect)
>   baseline, and vice versa. Denoising found only one of the circuit
>   components.
> - **Noising** (corrupt → clean patching): Noising either A or B has an
>   effect, as well as noising C.
>
> Noising works better in this case, as it finds all circuit components
> in the first pass.

**OR circuit: C = A OR B**

> - **Denoising** (clean → corrupt patching): Denoising either A or B has
>   an effect, as well as denoising C.
> - **Noising** (corrupt → clean patching): Noising either A or B has no
>   effect on the output, only denoising C restores the output. […]
>   Denoising found only one of the circuit components.
>
> Denoising works better in this case, as it finds all circuit components
> in the first pass.

This worked example is the canonical justification for choosing
direction-of-patching based on circuit topology: AND-shaped serial
dependencies need noising; OR-shaped redundant components need
denoising.

## §4.1 Logit difference as the recommended metric {#sec-4-1}

> Logit difference measures to what extent the model knows the correct
> answer, and it allows us to be specific: We can control for things we
> don't want to measure (e.g. components that boost both, Mary and John,
> in the IOI example) by choosing the right logits to compare (e.g. Mary
> vs John, or multiple-choice answers). The metric also is a mostly
> linear function of the residual stream (unlike probability-based
> metrics) which makes it easy to directly attribute logit difference to
> individual components ("direct logit attribution", "logit lens"). It's
> also a "softer" metric, allowing us to see partial effects on the
> model even if they don't change the rank of the output tokens (unlike
> e.g. accuracy), which is crucial for exploratory patching.

## §4.2 Metric pitfall taxonomy {#sec-4-2}

The paper enumerates problems for each metric type:

> - **Logit difference / logprob difference**: […] Potential
>   false-positive: Because the metric is a difference it may be driven
>   by either getting better at the correct answer or worse at the
>   incorrect answer. […] This is particularly concerning because the
>   corrupted model likely puts a high probability on the incorrect
>   answer. This means that any patch that indiscriminately damages the
>   model and gets it closer to uniform will damage the incorrect answer
>   logprob and so boost the logit diff.
> - **Logprobs**: […] Saturation: Once the correct answer becomes the
>   model's top guess, the logprob stops increasing meaningfully, even
>   though the confidence can increase much more. […] Unspecificity: We
>   lose the ability to control for other properties […] Inhibition: To
>   increase the logprob on John, the model can either increase the John
>   logit, or decrease other top logits, and it is hard to distinguish
>   which is happening.
> - **Probabilities**: […] Probabilities are non-linear, in the sense
>   that they track the logits exponentially. For example, a model
>   component adding +2 to a given logit can create a 1 or 40 percentage
>   point probability increase, just depending on what the baseline was.
> - **Binary and discrete metrics (Accuracy / top-k performance / rank)**:
>   These metrics round off each input to a discrete metric (and then
>   tend to average over a bunch of inputs). The problem with these is
>   that generally many components contribute to a model's performance,
>   with no single decisive contributor.

## §-ref Attribution patching as fast approximation {#sec-attribution-patching}

> Fast approximations to activation patching, such as attribution
> patching (see Nanda, 2023, and also AtP*, atpstar) can help speed up
> this process in large models.

The paper does not reproduce the attribution patching formula inline; it
delegates to Nanda 2023 (blogpost: neelnanda.io/mechanistic-interpretability/attribution-patching).
[TODO] Formula `(a_corrupt - a_clean) · ∂L/∂a_clean` — verbatim
equation not present in this PDF; source is Nanda 2023 blogpost (Tier B).

## §5 Summary — practical recommendations (verbatim) {#sec-5-summary}

> In most situations, use activation patching instead of ablations.
> Different corrupted prompts give you different information, be careful
> about what you choose and try to test a range of prompts.
> There are two different directions you can patch in: denoising and
> noising. These are not symmetric. Be aware of what a patching result
> implies!
>
> - Denoising (a clean → corrupt patch) shows whether the patched
>   activations were sufficient to restore the model behaviour. This
>   implies the components make up a cross-section of the circuit.
> - Noising (a corrupt → clean patch) shows whether the patched
>   activations were necessary to maintain the model behaviour. This
>   implies the components are part of the circuit.
>
> Be careful when using metrics that are (i) discrete, (ii) overly
> sharp, or (iii) sensitive to unintended information. Ideally use a
> range of metrics, and try to have at least one metric that is
> continuous and roughly linear in logits such as logit difference or
> logprob. We recommend representing patching results in a big dataframe
> with a column per metric and row per patching experiment, and making a
> bunch of plots from this.
>
> - Model top-k accuracy is discrete and can overrepresent changes at
>   thresholds and shows no change for large effects that don't cross
>   thresholds.
> - Most effects from patching are linear and additive in logit space.
>   Probability is exponential in logit space, so it overemphasises
>   effects near a threshold and suppresses effects elsewhere, creating
>   overly sharp patching plots.
> - Logprob can saturate, and cannot control for a patch that boosts
>   both the correct and incorrect answer(s).

## Source notes

- Tier A (arXiv preprint, methodology paper authored by Heimersheim and
  Nanda — same group as Conmy 2023, Wang 2022).
- PDF retrieved from arXiv (v1, 2024-04-23).
- Companion to `kb/excerpts/zhang2023-apatching` — Zhang & Nanda 2023
  is the systematic empirical study; Heimersheim & Nanda 2024 is the
  practitioner's playbook.

[Verified from PDF on 2026-05-12] Added #sec-attribution-patching (fast-approximation reference + [TODO] for formula not present in this PDF), #sec-5-summary (verbatim §5 practical recommendations list). Abstract verified verbatim against arXiv:2404.15255v1. Extraction failure: attribution patching formula `(a_corrupt - a_clean) · ∂L/∂a` does not appear in this paper; it is delegated to Nanda 2023 blogpost (Tier B source only).
