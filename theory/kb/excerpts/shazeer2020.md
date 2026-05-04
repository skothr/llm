---
paper_key: shazeer2020
title: "GLU Variants Improve Transformer"
authors: Shazeer
year: 2020
venue: arXiv (technical note)
arxiv: 2002.05202
local_pdf: theory/sources/papers/shazeer2020_glu_variants.pdf
type: excerpts
note: Verbatim quotations from the v1 arXiv PDF (Feb 2020). 5-page note that introduced the now-universal SwiGLU FFN. The formal definitions (Eq. 5–6) and the empirical T5 perplexity table (Table 1) are the load-bearing content.
---

# Excerpts — Shazeer 2020, "GLU Variants Improve Transformer"

## Abstract — what is being tested {#abstract}

> Gated Linear Units [Dauphin et al., 2016] consist of the component-wise product of two linear projections, one of which is first passed through a sigmoid function. Variations on GLU are possible, using different nonlinear (or even linear) functions in place of sigmoid. We test these variants in the feed-forward sublayers of the Transformer [Vaswani et al., 2017] sequence-to-sequence model, and find that some of them yield quality improvements over the typically-used ReLU or GELU activations.

## §1 Introduction — the original FFN {#sec-1}

The Vaswani 2017 FFN, with biases:

$$\mathrm{FFN}(x, W_1, W_2, b_1, b_2) = \max(0, x W_1 + b_1) W_2 + b_2 \tag{1}$$

The T5-style FFN, no biases (which is what modern decoder-only LLMs adopted):

$$\mathrm{FFN}_{\mathrm{ReLU}}(x, W_1, W_2) = \max(x W_1, 0) W_2 \tag{2}$$

GELU and Swish drop-in variants:

$$\mathrm{FFN}_{\mathrm{GELU}}(x, W_1, W_2) = \mathrm{GELU}(x W_1) W_2$$
$$\mathrm{FFN}_{\mathrm{Swish}}(x, W_1, W_2) = \mathrm{Swish}_1(x W_1) W_2 \tag{3}$$

## §2 Gated Linear Units and variants — Eq. (4)–(5) {#sec-2}

> [Dauphin et al., 2016] introduced Gated Linear Units (GLU), a neural network layer defined as the component-wise product of two linear transformations of the input, one of which is sigmoid-activated. They also suggest omitting the activation, which they call a "bilinear" layer and attribute to [Mnih and Hinton, 2007].

$$\mathrm{GLU}(x, W, V, b, c) = \sigma(x W + b) \otimes (x V + c)$$
$$\mathrm{Bilinear}(x, W, V, b, c) = (x W + b) \otimes (x V + c) \tag{4}$$

> We can also define GLU variants using other activation functions:

$$\mathrm{ReGLU}(x, W, V, b, c) = \max(0, x W + b) \otimes (x V + c)$$
$$\mathrm{GEGLU}(x, W, V, b, c) = \mathrm{GELU}(x W + b) \otimes (x V + c)$$
$$\mathrm{SwiGLU}(x, W, V, b, c, \beta) = \mathrm{Swish}_\beta(x W + b) \otimes (x V + c) \tag{5}$$

## §2 — the GLU-variant FFN, the now-universal form Eq. (6) {#sec-2-eq6}

The bias-free Transformer FFN with GLU gating — this is what every modern LLM uses:

$$\mathrm{FFN}_{\mathrm{GLU}}(x, W, V, W_2) = (\sigma(x W) \otimes x V) W_2$$
$$\mathrm{FFN}_{\mathrm{Bilinear}}(x, W, V, W_2) = (x W \otimes x V) W_2$$
$$\mathrm{FFN}_{\mathrm{ReGLU}}(x, W, V, W_2) = (\max(0, x W) \otimes x V) W_2$$
$$\mathrm{FFN}_{\mathrm{GEGLU}}(x, W, V, W_2) = (\mathrm{GELU}(x W) \otimes x V) W_2$$
$$\mathrm{FFN}_{\mathrm{SwiGLU}}(x, W, V, W_2) = (\mathrm{Swish}_1(x W) \otimes x V) W_2 \tag{6}$$

> All of these layers have three weight matrices, as opposed to two for the original FFN. **To keep the number of parameters and the amount of computation constant, we reduce the number of hidden units $d_{ff}$ (the second dimension of $W$ and $V$ and the first dimension of $W_2$) by a factor of $\tfrac{2}{3}$ when comparing these layers to the original two-matrix version.**

The $\tfrac{2}{3}$ factor is the parameter-equalization rule that determines the LLaMA / Qwen / DeepSeek hidden ratio. Vaswani's original $d_{ff} = 4 d_{\text{model}}$ becomes $d_{ff} \approx \tfrac{8}{3} d_{\text{model}}$ in modern SwiGLU FFNs (rounded — LLaMA 7B uses 11008 vs $\tfrac{8}{3} \cdot 4096 = 10923$).

## §3.1 Experimental setup — T5-base scale {#sec-3-1}

> We use the same code base, model architecture, and training task as the base model from [Raffel et al., 2019]. The encoder and decoder each consist of 12 layers, with $d_{\text{model}} = 768$. For the attention layers, $h = 12$ and $d_k = d_v = 64$. The FFN layers have hidden size $d_{ff} = 3072$. As we describe above, for the GLU-variant-based FFN layers, which have three weight matrices instead of two, we reduce the hidden layer to $d_{ff} = 2048$, so as to maintain the same parameter and operation counts as the base model.

## §3.2 Headline result — Table 1 perplexity {#sec-3-2}

Heldout-set log-perplexity on T5 segment-filling at 524,288 training steps (lower is better):

| Variant | Perplexity (524k steps) |
|---|---|
| FFN_ReLU (baseline) | 1.677 |
| FFN_GELU | 1.679 |
| FFN_Swish | 1.683 |
| FFN_GLU | 1.663 |
| FFN_Bilinear | 1.648 |
| FFN_GEGLU | **1.633** |
| FFN_SwiGLU | **1.636** |
| FFN_ReGLU | 1.645 |

> The GEGLU and SwiGLU variants produce the best perplexities.

GEGLU and SwiGLU are within 0.003 of each other; SwiGLU was adopted (LLaMA, PaLM) presumably for the cleaner Swish activation surface. GEGLU appears in some Google models (T5.1.1, GLaM, mT5).

## §4 Conclusions — author's epistemic candor {#sec-4}

> We have extended the GLU family of layers and proposed their use in Transformer. In a transfer-learning setup, the new variants seem to produce better perplexities for the de-noising objective used in pre-training, as well as better results on many downstream language-understanding tasks. These architectures are simple to implement, and have no apparent computational drawbacks. **We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.**

The "divine benevolence" line is famous — the gain has no theoretical justification in this paper, only an empirical sweep. Subsequent theoretical work on SwiGLU as a smoothed-piecewise approximation of attention-style multiplicative interactions exists but is not in this 5-page note.
