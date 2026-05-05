---
paper-key: llava2023
title: "Visual Instruction Tuning"
authors: Liu, Li, Wu, Lee
year: 2023
venue: NeurIPS
arxiv: 2304.08485
pdf: theory/sources/papers/llava2023.pdf
extracted: 2026-05-05 (Phase 2.5 deepening)
---

# Verbatim excerpts — Liu et al. 2023, "Visual Instruction Tuning" (LLaVA)

LLaVA's contribution is a **simple, parametrically minimal** approach to
multimodal LLMs: a single trainable linear projection from CLIP visual
features into the LLM word-embedding space, with a two-stage training
recipe. Contrasts with Flamingo's gated cross-attention insertion. The
LLaVA recipe is the dominant pattern for open multimodal models in
2024–2026 (Llama 3.2-Vision, InternVL, Qwen2-VL all build on it).

## #abstract

> Instruction tuning large language models (LLMs) using machine-generated
> instruction-following data has been shown to improve zero-shot
> capabilities on new tasks, but the idea is less explored in the
> multimodal field. We present the first attempt to use language-only
> GPT-4 to generate multimodal language-image instruction-following
> data. By instruction tuning on such generated data, we introduce
> **LLaVA: Large Language and Vision Assistant**, an end-to-end trained
> large multimodal model that connects a vision encoder and an LLM for
> general-purpose visual and language understanding. To facilitate
> future research on visual instruction following, we construct two
> evaluation benchmarks with diverse and challenging application-
> oriented tasks. Our experiments show that LLaVA demonstrates
> impressive multimodal chat abilities, sometimes exhibiting the
> behaviors of multimodal GPT-4 on unseen images/instructions, and
> yields a 85.1% relative score compared with GPT-4 on a synthetic
> multimodal instruction-following dataset. When fine-tuned on Science
> QA, the synergy of LLaVA and GPT-4 achieves a new state-of-the-art
> accuracy of 92.53%.

## #sec-3 — GPT-assisted visual instruction data generation (§3)

> Inspired by the success of recent GPT models in text-annotation tasks
> [17], we propose to leverage ChatGPT/GPT-4 for multimodal instruction-
> following data collection, based on the widely existing image-pair
> data. … to encode an image into its visual features to prompt a text-
> only GPT, we use two types of symbolic representations: (i) **Captions**
> typically describe the visual scene from various perspectives;
> (ii) **Bounding boxes** usually localize the objects in the scene,
> and each box encodes the object concept and its spatial location.

> We collect 158K unique language-image instruction-following samples
> in total, including 58K in conversations, 23K in detailed description,
> and 77K in complex reasoning, respectively. We ablated the use of
> ChatGPT and GPT-4 in our early experiments, and found that GPT-4
> consistently provides higher quality instruction-following data, such
> as spatial reasoning.

The three response types — **conversation**, **detailed description**,
**complex reasoning** — are the LLaVA data taxonomy, copied widely in
follow-up VLM datasets.

## #sec-4-1 — Architecture: linear projection (§4.1, Eq. 1)

> The primary goal is to effectively leverage the capabilities of both
> the pre-trained LLM and visual model. The network architecture is
> illustrated in Figure 1. We choose Vicuna [9] as our LLM $f_\phi(\cdot)$
> parameterized by $\phi$, as it has the best instruction following
> capabilities in language tasks among publicly available checkpoints.
>
> For an input image $X_v$, we consider the pre-trained CLIP visual
> encoder ViT-L/14 [40], which provides the visual feature $Z_v =
> g(X_v)$. The grid features before and after the last Transformer
> layer are considered in our experiments. **We consider a simple linear
> layer to connect image features into the word embedding space.**
> Specifically, we apply a trainable projection matrix $W$ to convert
> $Z_v$ into language embedding tokens $H_v$, which have the same
> dimensionality as the word embedding space in the language model:
>
> $$H_v = W \cdot Z_v, \quad \text{with } Z_v = g(X_v) \tag{1}$$
>
> Thus, we have a sequence of visual tokens $H_v$. **Note that our
> simple projection scheme is lightweight, which allows us to iterate
> data centric experiments quickly.** More sophisticated schemes to
> connect the image and language representations can also be considered,
> such as gated cross-attention in Flamingo [2] and Q-former in BLIP-2
> [28]. We leave exploring possibly more effective and sophisticated
> architecture designs for LLaVA as future work.

This is the entire architectural innovation: a single linear matrix $W
\in \mathbb{R}^{d_{\text{LLM}} \times d_{\text{vision}}}$. Contrast
Flamingo's gated cross-attention blocks (`kb/excerpts/alayrac2022-flamingo#sec-2-1-gated`):
LLaVA changes nothing about the LLM's architecture, only prepends visual
tokens to the input sequence as if they were word embeddings. Later
LLaVA-1.5 / LLaVA-NeXT replaced the linear $W$ with a 2-layer MLP for
small additional gains.

## #sec-4-2 — Two-stage training (§4.2)

> For each image $X_v$, we generate multi-turn conversation data
> $(X_q^1, X_a^1, \cdots, X_q^T, X_a^T)$, where $T$ is the total number
> of turns. We organize them as a sequence, by treating all answers as
> the assistant's response. … We perform instruction-tuning of the LLM
> on the prediction tokens, using its original auto-regressive training
> objective.
>
> Specifically, for a sequence of length $L$, we compute the probability
> of the target answers $X_a$ by:
>
> $$p(X_a \mid X_v, X_{\text{instruct}}) = \prod_{i=1}^{L} p_\theta(x_i \mid X_v, X_{\text{instruct},<i}, X_{a,<i}) \tag{3}$$

The two-stage recipe (load-bearing for VLM training):

> **Stage 1: Pre-training for Feature Alignment.** To strike a balance
> between concept coverage and training efficiency, we filter CC3M to
> 595K image-text pairs. … In training, we keep both the visual encoder
> and LLM weights frozen, and maximize the likelihood of (3) with
> trainable parameters $\theta = W$ (the projection matrix) only. **In
> this way, the image features $H_v$ can be aligned with the pre-trained
> LLM word embedding. This stage can be understood as training a
> compatible visual tokenizer for the frozen LLM.**
>
> **Stage 2: Fine-tuning End-to-End.** We always keep the visual
> encoder weights frozen, and continue to update both the pre-trained
> weights of the projection layer and LLM in LLaVA; i.e., the trainable
> parameters are $\theta = \{W, \phi\}$ in (3).

The frozen-vision-encoder + trainable-projection + trainable-LLM
configuration is the LLaVA training pattern that all open VLMs
inherit.

## #sec-5 — Experimental settings (§5)

> We train all models with 8× A100s, following Vicuna's hyperparameters.
> We pre-train our model on the filtered CC-595K subset for 1 epoch with
> a learning rate of 2e-3 and a batch size of 128, and fine-tune on the
> proposed LLaVA-Instruct-158K dataset for 3 epochs, with a learning
> rate of 2e-5 and a batch size of 32.

## #sec-5-2 — Quantitative result on Science QA (§5.2)

> When fine-tuned on Science QA, LLaVA alone achieves an accuracy of
> 90.92%. The synergy of LLaVA and GPT-4 achieves a new state-of-the-art
> accuracy of 92.53%.

## Source notes

- Tier A canonical (NeurIPS 2023). v2 of arXiv (Dec 2023) corresponds to
  the camera-ready and reproduces the headline numbers.
- PDF retrieved from arXiv. Anchors stable; equation numbers preserved.
- LLaVA-1.5 (Liu et al. Oct 2023, arXiv:2310.03744) replaces the linear
  projection with a 2-layer MLP and uses higher-resolution images;
  LLaVA-NeXT extends to dynamic high-resolution. None of these change
  the core architectural premise — linear/MLP projection of CLIP
  features into the LLM embedding space.
