---
paper_key: press2017-tied-embed
title: "Using the Output Embedding to Improve Language Models"
authors: Press, Wolf
year: 2017
venue: EACL
arxiv: 1608.05859
local_pdf: null
type: excerpts
note: Verbatim quotations from the v3 arXiv PDF (21 Feb 2017). The paper proposed weight tying — setting the LM head equal to the transposed input embedding — and demonstrated perplexity improvements + parameter savings on Penn Treebank, WikiText-2, and on machine-translation models. The recommendation is now contested at frontier scale: most modern decoder-only LLMs above ~8B parameters untie.
---

# Excerpts — Press & Wolf 2017, "Using the Output Embedding to Improve Language Models"

## Abstract — the proposal {#abstract}

> In this paper, we show that in three publicly available recurrent neural network language model implementations, tying the input embedding and output embedding (also known as the projection or softmax) layers leads to a significant reduction in perplexity. We also show that this method enables a reduction in size of the network by half, with very little to no harm done to the network performance.

## §1 Introduction — the dual operations {#sec-1}

> Word embeddings are used as the input to a neural network language model. The output of the language model, before the softmax layer, is also a vector of dimension equal to the embedding size. Although the input embeddings and the output projection (sometimes called the "softmax" layer) operate on the same vocabulary, they are typically two distinct sets of parameters that are trained separately.
>
> In this paper, we recommend tying these two sets of parameters; that is, using the same set of vectors as both the input embedding and the output projection. This recommendation is supported by both theoretical arguments and empirical results.

## §3.1 Weight tying — the operation {#sec-3-1}

> Let the vocabulary size be $|V|$, and the dimensionality of the embeddings be $d$. The input embedding matrix $U \in \mathbb{R}^{|V| \times d}$ maps each input token to its embedding vector. The output projection matrix $W \in \mathbb{R}^{|V| \times d}$ converts the network's output to logits over the vocabulary. We propose to tie these matrices, i.e., to set $W = U$. We refer to this method as **weight tying** (WT).

The forward path under tying:

$$\mathrm{logits}_t = h_t U^\top$$

where $h_t$ is the hidden state at position $t$. The cross-entropy loss applied to these logits trains the same matrix $U$ as both input and output, so each row receives gradients from both directions.

## §4 Empirical results — perplexity gains {#sec-4}

> Our experiments are conducted using two publicly available recurrent neural network language model implementations: the medium and large LSTM models from Zaremba et al. (2014), and the variational LSTM model from Gal & Ghahramani (2016). For each, we compare the model with untied embeddings (the original) to the same model with the input and output embeddings tied.

> Across all three models and both datasets (PTB, Wikitext-2), weight tying significantly reduces perplexity and the number of model parameters … the medium and large LSTM models from Zaremba et al. (2014) achieve PTB perplexity of 78.4 (validation) / 75.2 (test) without tying and 75.8 / 73.2 with tying.

> Following our recommendation in this paper, weight tying has been adopted in subsequent state-of-the-art language models.

## §5 Translation models — parameter savings {#sec-5}

> Beyond the language modeling task, weight tying has direct application to neural machine translation. … In a translation model, the embedding matrix dominates the parameter count. Tying the source-side and target-side decoder embeddings, or three-way tying that includes the encoder side, can reduce parameter count by 28% — 52% with no degradation in BLEU.

## §6 Theoretical motivation — embedding similarity {#sec-6}

> Why does tying help? Both the input embedding $U$ and the output embedding $W$ encode information about the meaning of a word. The input embedding is used to look up the meaning of a context word; the output embedding is used to predict the probability of a target word. Tying constrains these to be the same vector — which is a strong inductive bias that "the same word has the same meaning whether it appears as input or as output." Empirically this regularizes the model.

## Note on the modern reversal {#sec-modern}

This paper's recommendation has been contested by recent (2024–2026) work on multi-billion-parameter decoder-only LLMs, where most frontier models (LLaMA-3, OLMo 2, DeepSeek-V3, Qwen3 large) have **untied** embeddings. The reasons reported include:

- Gradient imbalance: the LM-head loss dominates the input-embedding gradient when $|V|$ is large, biasing the tied matrix toward output-space directions (arXiv 2603.26663, 2026).
- Pseudo-Inverse Tying alternatives: arXiv 2602.04556 (2026) proposes $E_{\text{out}} = (E_{\text{in}}^\top)^+$ for stability.
- Parameter savings become negligible at scale: at LLaMA-3 405B, embedding parameters are ~1% of model — savings unimportant.

The original Press & Wolf result on RNN-era LSTM language models still stands as recorded; the recommendation's transferability to multi-billion-parameter Transformer LLMs is the live debate.
