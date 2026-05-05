---
paper_key: alayrac2022-flamingo
title: "Flamingo: a Visual Language Model for Few-Shot Learning"
authors: Alayrac, Donahue, Luc, Miech, Barr, Hasson, Lenc, Mensch, Millican, Reynolds, Ring, Rutherford, Cabi, Han, Gong, Samangooei, Monteiro, Menick, Borgeaud, Brock, Nematzadeh, Sharifzadeh, Bińkowski, Barreira, Vinyals, Zisserman, Simonyan
year: 2022
venue: NeurIPS
arxiv: 2204.14198
local_pdf: theory/sources/papers/alayrac2022_flamingo.pdf
type: excerpts
note: Verbatim from the v2 arXiv PDF (Nov 2022). Flamingo introduced the gated cross-attention pattern for inserting vision features into a frozen LLM. Foundational for Visual Language Models predating LLaVA.
---

# Excerpts — Alayrac et al. 2022, "Flamingo"

## Abstract — what Flamingo does {#abstract}

> Building models that can be rapidly adapted to numerous tasks using only a handful of annotated examples is an open challenge for multimodal machine learning research. We introduce Flamingo, a family of Visual Language Models (VLM) with this ability. We propose key architectural innovations to: (i) bridge powerful pretrained vision-only and language-only models, (ii) handle sequences of arbitrarily interleaved visual and textual data, and (iii) seamlessly ingest images or videos as inputs. Thanks to their flexibility, Flamingo models can be trained on large-scale multimodal web corpora containing arbitrarily interleaved text and images, which is key to endow them with in-context few-shot learning capabilities.

## §2.1 Architecture — bridging vision and language {#sec-2-1}

> Flamingo's architecture brings together two complementary, pretrained, and frozen models: a Vision Encoder (which extracts visual features from images and videos) and a Language Model (LM, which provides general purpose generative language capabilities). To bridge these models we incorporate two trainable architectural components, namely the **Perceiver Resampler** and the **Gated Cross-Attention Dense (GATED XATTN-DENSE) blocks**. We initialize the Vision Encoder and Language Model with pretrained weights and freeze them throughout training.

### Perceiver Resampler — fixed-size visual tokens

> The Perceiver Resampler module takes as input a variable number of image or video features from the Vision Encoder and produces a fixed number of visual tokens. This is achieved by learning a predefined number of latent input queries that cross-attend to the visual features.

### Gated Cross-Attention Dense — vision-language fusion

> Several new blocks are inserted between the original layers of the frozen pretrained LM. The new GATED XATTN-DENSE blocks consist of a cross-attention layer attending to the visual features, followed by a feed-forward layer. To ensure that at initialization, the conditioned model yields the same results as the original language model, we use a $\tanh(\alpha)$ gating mechanism applied at the output of both the cross-attention and feed-forward layers, where $\alpha$ is a learnable scalar initialized to $0$.

## §2.1 — the gated cross-attention block {#sec-2-1-gated}

The architectural form of an inserted block:

$$\mathbf{y} = \mathbf{x} + \tanh(\alpha_{\text{xattn}}) \cdot \mathrm{XAttn}(\mathbf{x}, \mathbf{V})$$
$$\mathbf{z} = \mathbf{y} + \tanh(\alpha_{\text{ffw}}) \cdot \mathrm{FFW}(\mathbf{y})$$

where $\mathbf{x}$ is the language hidden state, $\mathbf{V}$ are the Perceiver Resampler outputs, and $\alpha_{\text{xattn}}, \alpha_{\text{ffw}}$ are learnable scalars initialized to $0$.

> Setting $\alpha = 0$ at initialization ensures that the inserted layer is the identity at the start of training, so the model behaves like the frozen base LM. As training proceeds, the gates open and the visual features begin contributing to the language hidden states.

## §2.1 — interleaving visual and text tokens {#sec-2-1-interleave}

> To naturally accommodate any number of images and videos in the input, we use a position-tagging mechanism that explicitly tags the language tokens corresponding to each image or video. Each image is replaced in the input text by an `<image>` tag at the appropriate place, and the language model attends to the `<image>` tokens via the cross-attention to retrieve the visual information.

## §3 Training — interleaved web corpora {#sec-3}

> Flamingo is trained on a mixture of web-scale multimodal datasets, all containing pairs or sequences of images / videos and text. The most distinctive of these is M3W (MultiModal MassiveWeb), a dataset constructed from a snapshot of the web with high-quality images and surrounding text. M3W contains 43M webpages with 185M images.

> The training objective is the standard next-token prediction loss on the language tokens, conditioned on the visual context. Visual encoder and base LM remain frozen.

## §4 Few-shot capabilities {#sec-4}

> Flamingo achieves a new state of the art in few-shot learning on a wide range of open-ended visual-language tasks, simply by being prompted with task-specific examples. On numerous benchmarks, Flamingo outperforms models fine-tuned on thousands of times more task-specific data … Flamingo-80B outperforms prior state-of-the-art with 32 in-context examples on 6 of the 16 benchmarks evaluated.

## Note on the LLaVA / direct-projection alternative {#sec-llava}

Flamingo's approach (frozen LM + new gated cross-attention blocks) was followed in 2023 by Liu et al.'s LLaVA, which took a simpler approach: project visual features through a learned MLP into the LM's input embedding space, and concatenate them as ordinary tokens. LLaVA's approach is parametrically simpler but requires unfreezing the LM during instruction tuning. The two approaches occupy different points on a "how invasive is multimodal integration" spectrum, with LLaVA at the minimal-architectural-surgery end and Flamingo at the new-cross-attention-blocks end. See `kb/excerpts/llava2023#sec-3-1` for the LLaVA contrast.

The 2025–2026 frontier (Llama 4, InternVL3) has moved beyond both: native multimodal pretraining where vision and text are co-trained from scratch.
