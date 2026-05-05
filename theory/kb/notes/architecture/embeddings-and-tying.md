---
topic: architecture/embeddings-and-tying
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - press2017-tied-embed
  - vaswani2017
  - meta-llama3
  - olmo2
  - deepseek-v3
secondary_sources:
  - touvron2023
  - radford2018
  - radford2019-gpt2
  - phi4
related_topics:
  - architecture/tokenization
  - architecture/transformer-overview
  - architecture/normalization
  - interpretability/probing
---

# Token embeddings and weight tying

Two parameter blocks in every Transformer LLM are unique to the
discrete-token interface: the **input embedding** (token IDs → residual-
stream vectors) and the **LM head** (final residual stream → logits over
the vocabulary). Whether they share weights is the **weight-tying
question**, originally proposed in Press & Wolf 2017 and now a
deliberate per-model choice that frontier labs **untie at multi-billion
scale**.

## 1. Formal definition

Given vocabulary $V$ with $|V|$ tokens, residual width $d_{\text{model}}$,
sequence of token IDs $\mathbf{t} \in \{0, \ldots, |V|-1\}^N$, the input
embedding is a learned lookup:

$$X^0_i = E_{\text{in}}[t_i] \in \mathbb{R}^{d_{\text{model}}}, \quad
E_{\text{in}} \in \mathbb{R}^{|V| \times d_{\text{model}}} \tag{1}$$

The LM head produces logits over the vocabulary at the final layer:

$$\mathrm{logits}_i = X^L_i \, E_{\text{out}} \in \mathbb{R}^{|V|}, \quad
E_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times |V|} \tag{2}$$

(in modern models, $X^L$ is $\mathrm{Norm}(X^L)$ — the final RMSNorm
precedes the LM head). **Weight tying** sets

$$E_{\text{out}} = E_{\text{in}}^\top \tag{3}$$

so the model has only $|V| \cdot d_{\text{model}}$ unique
embedding/projection parameters instead of $2 |V| \cdot d_{\text{model}}$
`[press2017-tied-embed §3.1; kb/excerpts/press2017-tied-embed#sec-3-1]`.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $V$ | tokenizer vocabulary |
| $E_{\text{in}}$ | input token embedding matrix; row $i$ is the $d_{\text{model}}$-dim vector for token $i$ |
| $E_{\text{out}}$ | LM-head projection matrix; column $i$ is the readout direction for token $i$ |
| $X^0$ | residual stream after embedding (before block 0) |
| $X^L$ | residual stream after block $L-1$ (before LM head) |

## 2. Mechanism

### 2.1 Input embedding lookup

The lookup is sparse: for token ID $t_i$, $E_{\text{in}}[t_i]$ retrieves
row $t_i$. This is a **multiplication by a one-hot vector**, but
implemented as a gather.

The original Vaswani 2017 model **scaled the embedding by
$\sqrt{d_{\text{model}}}$** before adding the positional signal
`[vaswani2017 §3.4; kb/excerpts/vaswani2017#sec-3-2-2]`, an artifact of
the Xavier-initialization variance regime. Modern decoder-only LLMs
typically *omit* the scaling and rely on the first RMSNorm to fix the
norm.

### 2.2 LM head projection

For each position $i$ at depth $L$, the model computes
$|V|$ inner products $\langle X^L_i, E_{\text{out}}[:, k] \rangle$ for
$k \in \{0, \ldots, |V|-1\}$ and softmaxes. Cost: $O(N \cdot |V| \cdot
d_{\text{model}})$ — at $|V| = 128\text{K}$ and $d = 4\text{K}$, this is
~$5 \times 10^8$ FLOPs per token, comparable to one Transformer block.

Optimizations exist (sampled softmax, hierarchical softmax, adaptive
softmax) but **modern LLMs use the full softmax** because $|V|$ is
moderate and the matmul is a regular dense GEMM well-served by
hardware.

### 2.3 Tying mechanically

When tied (Eq. 3), the parameter footprint drops by
$|V| \cdot d_{\text{model}}$. For LLaMA-3 8B with $|V| = 128{,}256$ and
$d = 4096$, that is **~525M parameters** — about 6.5% of the model
`[meta-llama3 §3.1]`. The forward pass is unchanged; the backward pass
sums input-side and output-side gradients into the same matrix.

## 3. Variants and lineage

### 3.1 Tying decisions across major models

| Model | Tied? | $|V|$ | $d$ | Embedding params | Source |
|---|---|---|---|---|---|
| Vaswani 2017 (base) | tied | 37K | 512 | 19M | `[vaswani2017 §3.4]` |
| GPT-1 | tied | 40K | 768 | 31M | `[radford2018 §3.1]` |
| GPT-2 | tied | 50K | 768/1024/1280/1600 | 38–80M | `[radford2019-gpt2 §2.3]` |
| LLaMA-1 / 2 | tied | 32K | 4096 (7B) | 131M | `[touvron2023 §2]` |
| **LLaMA-3 8B** | **untied** | 128K | 4096 | 1.05B (525M each) | `[meta-llama3 §3.1]` |
| **OLMo 2** | **untied** | ~50K | varies | varies | `[olmo2 §3.1]` |
| **DeepSeek-V3** | **untied** | 128K | 7168 | 1.84B (920M each) | `[deepseek-v3 §2.1]` |
| Phi-4 | tied (small) | 100K | 5120 | 512M | `[phi4 §2.1; kb/excerpts/phi4#sec-2-1]` |
| Qwen3 (small) | tied | 152K | varies | varies | `[qwen3 §2.1]` |
| Qwen3 (235B) | untied | 152K | varies | varies | `[qwen3 §2.1]` |

The empirical pattern: **tie at small scale, untie at multi-billion**.

### 3.2 Press & Wolf 2017 — original proposal

Press & Wolf (2017, EACL) argued that the input and output embedding
matrices are theoretically performing **dual operations**: $E_{\text{in}}$
maps token IDs to vectors that are good *predictive contexts*, while
$E_{\text{out}}$ maps the predicted hidden state to vectors that are
good *target similarities*. They argued these should be the same vector
for the same token, and reported significant perplexity reductions and
parameter savings on Penn Treebank and WikiText
`[press2017-tied-embed §4; kb/excerpts/press2017-tied-embed#sec-4]`.

The result was extended by Inan, Khosravi, Socher 2017 (arXiv
1611.01462) to a loss-framework view: tied embeddings induce a
classification loss equivalent to maximizing the inner product between
the predicted hidden and the target token's embedding.

### 3.3 The frontier untiing trend

Modern (2024–2026) frontier open models with $> 8\text{B}$ parameters
mostly **untie**:

- **LLaMA-3 8B/70B/405B**: untied at all scales `[meta-llama3 §3.1]`.
- **DeepSeek-V3 671B**: untied `[deepseek-v3 §2.1]`.
- **OLMo 2**: untied at 7B/13B `[olmo2 §3.1]`.
- **Qwen3 large**: untied; Qwen3 small ties `[qwen3 §2.1]`.

Recent (2026) work diagnoses why: tying **biases input embeddings
toward the output space** because the same matrix is updated by both
gradients, and the larger LM-head loss dominates `[arXiv 2603.26663 §3]`.
The **Pseudo-Inverse Tying** alternative (arXiv 2602.04556) sets
$E_{\text{out}} = (E_{\text{in}}^\top)^+$ rather than the strict
transpose, claiming better stability `[arXiv 2602.04556 §3]`.

[CONTRADICTION] The historical Press & Wolf result reports tying
*improves* perplexity on small models; the recent untied-frontier
results report tied embeddings *hurt* at scale. The reconciling claim
(if any) is that the gradient-imbalance failure mode only matters when
$|V| \cdot d \gg \text{rest of model}$ is no longer true — i.e., at
multi-billion-parameter scale the embedding matrices are a small
fraction of the model and the gradient imbalance dominates. This is
an [INTUITION] not a derivation.

### 3.4 Special-token embeddings

Modern instruction-tuned models reserve embeddings for control tokens:

- **BOS, EOS** — sequence delimiters; learned same as content tokens.
- **`<|system|>`, `<|user|>`, `<|assistant|>`** — chat-template role
  markers; appear at SFT/post-training stages.
- **`<think>`, `</think>`** (or `<reasoning>` blocks) — reasoning-mode
  delimiters introduced by R1 / Qwen3 thinking models
  `[deepseek-r1 §2.1; qwen3 §3.2]`. The training distribution of these
  tokens is shaped by the long-CoT post-training corpus.

Reserved-but-unused token IDs are common (LLaMA-3 reserves a block of
unused IDs for downstream fine-tuning) — adding them at SFT requires no
embedding-matrix surgery.

### 3.5 Vocabulary expansion / token addition

Adding new tokens to a pretrained model (e.g., new languages, new
domain symbols) requires **embedding initialization for unseen rows**.
Common strategies:

1. **Mean-init** — new row = mean of existing rows. Stable but generic.
2. **Subword-mean init** — new row = mean of the BPE-pieces that
   represent the new token's surface form. Better domain transfer.
3. **Token Distillation** (Pham et al. 2025, arXiv 2505.20133) — train
   a small loss aligning new-token embeddings with attention-aware
   contexts of the surface form's prior tokenization. State-of-the-art
   for new-token addition `[arXiv 2505.20133 §3]`.

## 4. Intuitions and analogies

[ANALOGY] The input embedding is **the model's dictionary of word
meanings**: the $i$-th row is the LLM's prior on what token $i$ means
before context is applied. Pre-context, the dictionary is queried by
ID; post-context (after the block stack), the residual stream contains
**weighted, mixed combinations** of those vectors plus learned
features. The dictionary analogy returns to canonical form via Eq. 1.

[ANALOGY] The LM head is **a soft argmax over reading directions**: at
each position, the model has a hidden state $X^L_i$, and asks "which
token's readout direction $E_{\text{out}}[:, k]$ is this hidden state
most aligned with?" Tying makes the readout direction equal to the
input embedding, so the question becomes "which token is this hidden
state most like at the input level." That circularity is the source of
the gradient-imbalance bias when tied.

[INTUITION] **Why frontier labs untie at scale**: the embedding-matrix
parameter savings from tying are constant ($|V| \cdot d$) while the
non-embedding model grows quadratically with $d$ and linearly with $L$.
At LLaMA-3 8B, embeddings are 12% of params (1B/8B); at 70B, 5%
(3.5B/70B); at 405B, 1% (4B/405B). Untiing's parameter cost goes from
"meaningful" to "negligible" while the alleged quality cost (output-
space bias) is constant. The trade flips.

## 5. Frontier and open questions (as of 2026-05)

- **Manifold structure of embeddings.** "Token Embeddings Violate the
  Manifold Hypothesis" (arXiv 2504.01002, Apr 2025) claims trained
  token embeddings do *not* lie on a low-dimensional manifold, against
  prior assumptions. If correct, this affects compression, retrieval-
  via-embedding, and embedding-projection probing methods.
- **LLM-as-embedder.** The trained LLM's hidden states are increasingly
  used as feature vectors for retrieval / classification (arXiv
  2412.12591). Architecturally, this is "LM head as embedding model"
  — invert the lookup direction.
- **Pseudo-Inverse Tying and gradient-balanced tying.** Whether these
  alternatives reach the quality of fully-untied at frontier scale is
  open `[arXiv 2602.04556 §4]`.
- **Embedding initialization for tokenizer-free.** BLT replaces token
  embeddings with byte-level features and a local encoder. The
  conceptual analog of "input embedding" becomes a small Transformer
  rather than a lookup table. See
  `kb/notes/architecture/tokenization.md`.

## 6. See also

- `kb/notes/architecture/tokenization.md` — what the integer IDs are.
- `kb/notes/architecture/transformer-overview.md` — where embeddings
  sit at the I/O boundaries of the block stack.
- `kb/notes/architecture/normalization.md` — final-layer RMSNorm
  precedes the LM head in modern designs.
- `kb/notes/architecture/multi-token-prediction.md` — MTP heads are
  *additional* LM heads sharing the embedding decision.
- `kb/notes/interpretability/lens-techniques.md` — logit-lens applies
  $E_{\text{out}}$ to intermediate hidden states.
- `kb/notes/interpretability/probing.md` — embeddings as features.
