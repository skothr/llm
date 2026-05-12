---
paper_key: alain-bengio-2017
title: "Understanding intermediate layers using linear classifier probes"
authors: Alain, Bengio
year: 2016
venue: ICLR 2017 Workshop (arXiv 1610.01644)
arxiv: 1610.01644
local_pdf: theory/sources/papers/alain-bengio-2017.pdf
type: excerpts
---

# Excerpts — Alain & Bengio 2017, "Understanding intermediate layers using linear classifier probes"

## #abstract — full verbatim

> Neural network models have a reputation for being black boxes. We
> propose to monitor the features at every layer of a model and measure
> how suitable they are for classification. We use linear classifiers,
> which we refer to as "probes", trained entirely independently of the
> model itself. This helps us better understand the roles and dynamics
> of the intermediate layers. We demonstrate how this can be used to
> develop a better intuition about models and to diagnose potential
> problems. We apply this technique to the popular models Inception v3
> and Resnet-50. Among other things, we observe experimentally that the
> linear separability of features increase monotonically along the depth
> of the model.

(Verified verbatim from PDF on 2026-05-12.)

## §3.2 — Linear classifier probes: formal definition {#sec-3-2-probe-definition}

> Consider the common scenario in deep learning in which we are trying to
> classify the input data X to produce an output distribution over D classes.
> The last layer of the model is a densely-connected map to D values followed
> by a softmax, and we train by minimizing cross-entropy.
> At every layer we can take the features $H_k$ from that layer and try to
> predict the correct labels y using a linear classifier parameterized as
>
> $$f_k : H_k \to [0,1]^D$$
> $$h_k \mapsto \operatorname{softmax}(W h_k + b).$$
>
> where $h_k \in H$ are the features of hidden layer $k$, $[0,1]^D$ is the
> space of categorical distributions of the D target classes, and $(W, b)$
> are the probe weights and biases to be learned so as to minimize the usual
> cross-entropy loss.

> We refer to those linear classifiers as "probes" in an effort to clarify
> our thinking about the model. These probes do not affect the model
> training. They only measure the level of linear separability of the
> features at a given layer. Blocking the backpropagation from the probes
> to the model itself can be achieved by using tf.stop_gradient in
> Tensorflow (or its Theano equivalent), or by managing the probe
> parameters separately from the model parameters.

The probe-vs-fine-tune distinction is definitional here: the underlying
network's weights are held entirely fixed; only $(W, b)$ of the probe are
optimized. Probe accuracy at layer $k$ is therefore a read-only measurement
of how linearly decodable the target label is from $h_k$, with no gradient
flowing into the model being studied.

## §3.1 — Information-theoretic motivation and the monotonic-separability finding {#sec-3-1-monotonic}

> The initial motivation for linear classifier probes was related to a
> reflection about the nature of information (in the entropy sense of the
> word) passing from one layer to the next.
> New information is never added as we propagate forward in a model.

> The task of a deep neural network classifier is to come up with a
> representation for the final layer that can be easily fed to a linear
> classifier (i.e. the most elementary form of useful classifier). The
> cross-entropy loss applies a lot of pressure directly on the last layer
> to make it linearly separable. Any degree of linear separability in the
> intermediate layers happens only as a by-product.

The paper grounds this in the Data Processing Inequality: for $X \to Y \to Z$,

$$I(X;Z) \leq I(X;Y)$$

where $I(\cdot;\cdot)$ denotes mutual information. Every layer loses
information relative to its input, yet probe accuracy increases with depth —
a result the authors describe as "at first glance a contradiction." Their
resolution: neural networks distill computationally useful representations,
not information-theoretic maxima. Linear separability of the label can
increase even as total information decreases.

## §4.1 — ResNet-50 empirical result: near-perfect monotonicity {#sec-4-1-resnet}

> For the ResNet-50 model trained on ImageNet, we can see deeper features
> are better at predicting the output classes. More importantly, the
> relationship between depth and validation prediction error is almost
> perfectly monotonic. This suggests a certain "greedy" aspect of the
> representations used in deep neural networks. This property is something
> that comes naturally as a result of conventional training, and it is not
> due to the insertion of probes in the model.

Table 4a (validation probe prediction error by layer, verbatim):

| layer name | topology        | probe valid prediction error |
|------------|-----------------|------------------------------|
| input 1    | (224, 224, 3)   | 0.99                         |
| add 1      | (28, 28, 256)   | 0.94                         |
| add 3      | (28, 28, 256)   | 0.88                         |
| add 7      | (28, 28, 512)   | 0.76                         |
| add 13     | (14, 14, 1024)  | 0.51                         |
| add 16     | (7, 7, 2048)    | 0.31                         |

Probe error falls from 0.99 at the input to 0.31 at the penultimate residual
block — an almost perfectly monotonic decrease across all 16 residual add
layers. The same monotonic pattern is confirmed on Inception v3 (§4.2):
"The smooth gradient of colors in Figure 5 shows how the linear separability
increases monotonically as we probe layers deeper into the network."

[Verified from PDF on 2026-05-12] Added §3.2 (#sec-3-2-probe-definition), §3.1 (#sec-3-1-monotonic), §4.1 (#sec-4-1-resnet). Abstract verified verbatim.
