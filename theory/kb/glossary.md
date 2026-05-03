# Glossary

Every technical term used in the LLM theory KB, defined concisely. Each term
should cite a paper-key from `kb/index/papers.json` for traceability where
applicable. Terms are added during Phase 2 (per-topic research) and reviewed
during Phase 4 (lint).

This glossary was seeded from `archive/2026-05-03-pre-expansion/GLOSSARY.md`
on 2026-05-03; entries from that seed have not yet been re-cited and are
treated as `status: pre-expansion-seed` until re-validated during Phase 2.

## Models and Architectures

- **LLM** — Large Language Model. A neural network with billions of parameters trained on large text corpora to predict and generate language.
- **Transformer** — Neural network architecture introduced by Vaswani et al. (2017) that replaces recurrence with self-attention, allowing parallel processing of sequences.
- **Encoder-Decoder** — The original Transformer layout: an encoder processes the full input bidirectionally, a decoder generates output autoregressively, with cross-attention connecting them.
- **Decoder-Only** — Simplified Transformer architecture (GPT, LLaMA) that removes the encoder and cross-attention, using only masked self-attention and FFN layers.
- **RNN** — Recurrent Neural Network. Processes sequences one step at a time, maintaining a hidden state. Predecessor to the Transformer.
- **LSTM** — Long Short-Term Memory. An RNN variant with gating mechanisms to better preserve long-range dependencies.
- **T5 (Text-to-Text Transfer Transformer)** — Google's encoder-decoder Transformer (Raffel et al., 2019) that frames every NLP task as text-to-text. Originator of several conventions reused elsewhere (e.g., SentencePiece vocab, relative positional bias).
- **PaLM** — Google's 540B-parameter decoder-only LLM (Chowdhery et al., 2022). Uses SwiGLU, RoPE, parallel attention/FFN layers. Reference point for scaling experiments.
- **Mistral** — Open-weights decoder-only LLM family from Mistral AI (Jiang et al., 2023). 7B variant uses sliding-window attention and grouped-query attention on top of the LLaMA-style architecture.
- **Claude** — Anthropic's proprietary decoder-only LLM family. No public architecture paper; structural family shares the decoder-only Transformer lineage.

## Tokenization

- **Tokenization** — Converting raw text into a sequence of discrete integer token IDs that a model can process.
- **BPE (Byte Pair Encoding)** — Subword tokenization algorithm that iteratively merges the most frequent adjacent character pairs to build a vocabulary.
- **Subword Unit** — A token that may be a whole word, a word fragment, or a single character/byte, depending on corpus frequency.
- **SentencePiece** — A tokenizer that operates on raw byte sequences rather than Unicode characters, used by LLaMA. Provides byte-fallback so any input can be tokenized.
- **Byte-Fallback** — Mechanism ensuring no "unknown token" failures: any byte sequence can be represented, even if unseen during training.
- **Vocabulary (V)** — The fixed set of all tokens a model can recognize. Size ranges from ~32K (LLaMA) to ~50K (GPT-3).

## Embeddings and Positional Encoding

- **Token Embedding** — A learned lookup table ($W_e \in \mathbb{R}^{V \times d_{\text{model}}}$) mapping each token index to a dense vector.
- **Weight Tying** — Sharing the same weight matrix $W_e$ for both input embeddings and the output projection to logits, halving vocabulary-related parameters.
- **Positional Encoding** — Any mechanism that injects sequence-position information into the model, since self-attention is inherently permutation-invariant.
- **Sinusoidal Positional Encoding** — Fixed positional encoding using sine/cosine functions at different frequencies (original Transformer, 2017).
- **Learned Positional Embedding** — A trainable position matrix $W_p$ where each row corresponds to a sequence position (GPT-1, 2018). Fixes maximum context length at training time.
- **RoPE (Rotary Position Embedding)** — Positional encoding that rotates query and key vectors by position-dependent angles, making attention dot products depend on relative position. Applied at every layer to Q and K only. Used by LLaMA and most modern LLMs.

## Attention

- **Self-Attention** — Mechanism where each position in a sequence computes a weighted sum over all positions, with weights determined by learned query-key similarity.
- **Scaled Dot-Product Attention** — Core attention operation: $\text{softmax}(QK^\top / \sqrt{d_k}) V$. Scaling by $\sqrt{d_k}$ prevents large dot products from saturating the softmax.
- **Multi-Head Attention** — Running $h$ parallel attention operations on different learned projections, then concatenating and projecting the results. Allows attending to different representation subspaces simultaneously.
- **Query (Q)** — Projected representation of the position that is "asking" for information in attention.
- **Key (K)** — Projected representation of positions being "queried" against. The Q·K dot product determines attention weights.
- **Value (V)** — Projected representation of the content to be aggregated. Weighted by attention scores to produce the output.
- **Attention Weights** — The softmax-normalized scores ($a_j \geq 0$, $\sum a_j = 1$) that determine how much each position contributes to the output.
- **Causal Mask** — A triangular mask that prevents position $i$ from attending to positions $> i$, enforcing autoregressive (left-to-right) generation.
- **Cross-Attention** — Attention where Q comes from the decoder and K, V come from the encoder output. Removed in decoder-only models.
- **Head Dimension ($d_k$, $d_v$)** — Dimensionality of query/key and value projections per attention head. Typically $d_{\text{model}} / h$.

## Feed-Forward Network

- **FFN (Feed-Forward Network)** — A position-wise two-layer (or three-layer for GLU variants) fully-connected network applied independently at each sequence position. Transforms representations within each position, complementing attention which mixes across positions.
- **ReLU** — Rectified Linear Unit: $\max(0, x)$. Original Transformer activation (2017).
- **GELU (Gaussian Error Linear Unit)** — Smooth activation $x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF. Used by GPT-1 (2018).
- **Swish / SiLU** — Activation function $x \cdot \sigma(\beta x)$ where $\sigma$ is the sigmoid. SiLU is the $\beta = 1$ case: $x \cdot \sigma(x)$. Unlike ReLU, it is smooth and non-monotonic around zero.
- **GLU (Gated Linear Unit)** — Architecture where one linear projection produces content and another produces a gate controlling what passes through.
- **SwiGLU** — GLU variant using Swish activation for the gate. Uses three weight matrices instead of two; inner dimension reduced by 2/3 to compensate. Used by LLaMA (2023).
- **Inner Dimension ($d_{\text{ff}}$)** — The expanded hidden size inside the FFN. Typically $4 \times d_{\text{model}}$, or $\frac{2}{3} \times 4 \times d_{\text{model}}$ for SwiGLU.

## Normalization and Residual Connections

- **Residual Connection (Skip Connection)** — Adding the sub-layer input directly to its output ($\text{output} = x + \text{SubLayer}(x)$). Enables gradient flow and makes each layer learn a correction rather than a full representation.
- **Layer Normalization (LayerNorm)** — Normalizes across the feature dimension per position: subtract mean, divide by standard deviation, apply learned scale ($\gamma$) and bias ($\beta$).
- **RMSNorm (Root Mean Square Normalization)** — Simplified LayerNorm that drops mean-centering and bias, normalizing only by the RMS of the input. 7–64% faster with comparable quality. Used by LLaMA.
- **Post-LN** — Original Transformer placement: normalize after the residual addition. Requires learning rate warmup for stable training.
- **Pre-LN** — Modern placement: normalize before the sub-layer, then add residually. Gradients scale as $O(d\sqrt{\ln d / L})$, enabling stable training without warmup.
- **Warmup** — Gradually increasing the learning rate from zero during early training steps. Critical for Post-LN; unnecessary for Pre-LN.

## Output and Training

- **Logits** — Unnormalized log-probabilities over the vocabulary, produced by the output projection ($h_{\text{final}} \cdot W_e^\top$).
- **Softmax** — Converts logits to a probability distribution: $P(j) = e^{z_j} / \sum_k e^{z_k}$.
- **Cross-Entropy Loss** — Training objective: $\mathcal{L} = -\log P(t_i \mid t_1, \ldots, t_{i-1})$, summed over all positions.
- **Autoregressive** — Generating tokens one at a time, left to right, where each prediction depends only on previous tokens.
- **Perplexity** — $2^{\mathcal{L}}$ (or $e^{\mathcal{L}}$ with natural log). Measures how "surprised" the model is by the data. Lower is better.

## Dimensions and Notation

- **$d_{\text{model}}$** — The model's hidden dimension (width of the residual stream). All sub-layers input and output this size.
- **$n$** — Sequence length (number of tokens in the input).
- **$h$** — Number of attention heads.
- **$L$** — Number of Transformer layers (blocks).
- **$V$** — Vocabulary size.
- **Hidden State** — The $d_{\text{model}}$-dimensional vector representation at each position, flowing through the residual stream across layers.
- **Residual Stream** — The main data pathway through the model: each layer reads from and writes back to this stream via residual connections.
- **Parameters** — The learned weights of the model (embedding matrices, projection matrices, normalization scales, etc.).

## Inference

- **Greedy Decoding** — Selecting the highest-probability token at each step.
- **Top-k Sampling** — Sampling from only the $k$ most probable tokens.
- **Nucleus Sampling (Top-p)** — Sampling from the smallest set of tokens whose cumulative probability exceeds $p$.
