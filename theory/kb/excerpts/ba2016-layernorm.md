---
paper-key: ba2016-layernorm
title: Layer Normalization
authors: Ba, Kiros, Hinton
year: 2016
venue: arXiv:1607.06450
pdf: theory/sources/papers/ba2016_layernorm.pdf
extracted: 2026-05-04 (Phase 2.5 deepening pass)
---

# Verbatim excerpts — Ba, Kiros, Hinton 2016, "Layer Normalization"

## #abstract

> Training state-of-the-art, deep neural networks is computationally
> expensive. One way to reduce the training time is to normalize the
> activities of the neurons. A recently introduced technique called
> batch normalization uses the distribution of the summed input to a
> neuron over a mini-batch of training cases to compute a mean and
> variance which are then used to normalize the summed input to that
> neuron on each training case. This significantly reduces the
> training time in feed-forward neural networks. However, the effect
> of batch normalization is dependent on the mini-batch size and it is
> not obvious how to apply it to recurrent neural networks. In this
> paper, we transpose batch normalization into layer normalization by
> computing the mean and variance used for normalization from all of
> the summed inputs to the neurons in a layer on a single training
> case. Like batch normalization, we also give each neuron its own
> adaptive bias and gain which are applied after the normalization but
> before the non-linearity. Unlike batch normalization, layer
> normalization performs exactly the same computation at training and
> test times. It is also straightforward to apply to recurrent neural
> networks by computing the normalization statistics separately at
> each time step. Layer normalization is very effective at stabilizing
> the hidden state dynamics in recurrent networks. Empirically, we
> show that layer normalization can substantially reduce the training
> time compared with previously published techniques.

## #sec-3 — Layer normalization (§3, p.2-3)

> We now consider the layer normalization method which is designed to
> overcome the drawbacks of batch normalization.

The core definition (Eq. 3 — the load-bearing equation):

> Notice that changes in the output of one layer will tend to cause
> highly correlated changes in the summed inputs to the next layer,
> especially with ReLU units whose outputs can change by a lot. This
> suggests the "covariate shift" problem can be reduced by fixing the
> mean and the variance of the summed inputs within each layer. We,
> thus, compute the layer normalization statistics over all the hidden
> units in the same layer as follows:
>
>   μ^l = (1/H) Σ_{i=1..H} a_i^l
>   σ^l = sqrt( (1/H) Σ_{i=1..H} (a_i^l − μ^l)^2 )      … (3)
>
> where H denotes the number of hidden units in a layer. The
> difference between Eq. (2) [batch norm] and Eq. (3) is that under
> layer normalization, all the hidden units in a layer share the same
> normalization terms μ and σ, but different training cases have
> different normalization terms. Unlike batch normalization, layer
> normalization does not impose any constraint on the size of a
> mini-batch and it can be used in the pure online regime with batch
> size 1.

## #sec-3-1 — Layer normalized RNNs (§3.1, p.2-3)

> In a standard RNN, the summed inputs in the recurrent layer are
> computed from the current input x^t and previous vector of hidden
> states h^{t−1} which are computed as a^t = W_hh h^{t−1} + W_xh x^t.
> The layer normalized recurrent layer re-centers and re-scales its
> activations using the extra normalization terms similar to Eq. (3).
> [Per-time-step μ^t and σ^t are computed from the H summed inputs at
> step t alone; gain g and bias b are shared across timesteps.]

The motivating contrast with batch norm in RNNs:

> [Batch norm] is problematic if a test sequence is longer than any of
> the training sequences. Layer normalization does not have such
> problem because its normalization terms depend only on the summed
> inputs to a layer at the current time-step. It also has only one
> set of gain and bias parameters shared over all time-steps.

## #sec-4 — Analysis: invariance properties (§4 / Table 1, p.4)

The paper's invariance taxonomy (Table 1 reproduced):

| Method        | Weight matrix re-scaling | Weight matrix re-centering | Dataset re-scaling | Dataset re-centering |
|---------------|:-:|:-:|:-:|:-:|
| Batch norm    | Invariant | No | Invariant | Invariant |
| Weight norm   | Invariant | No | No | No |
| **Layer norm**| Invariant (single weight vector only — see §4) | **Invariant** | Invariant | No |

> Layer normalization is invariant under re-scaling of the entire
> weight matrix and re-centering of all of the incoming weights.

The note `kb/notes/architecture/normalization.md` should consume this
table when contrasting the LN/BN/WN family.

## Source notes

- Tier A canonical (arXiv preprint, widely cited).
- PDF retrieved 2026-05-04 from arXiv. Page-anchored quotes; equation
  numbers preserved as in source.
- xref reconstruction warning from pdftotext is benign — text content
  is intact.
