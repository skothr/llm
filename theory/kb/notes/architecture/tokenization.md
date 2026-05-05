---
topic: architecture/tokenization
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - sennrich2016
  - kudo2018-sentencepiece
  - blt2024
  - meta-llama3
  - phi4
secondary_sources:
  - radford2018
  - radford2019-gpt2
  - touvron2023
  - deepseek-v3
related_topics:
  - architecture/embeddings-and-tying
  - architecture/transformer-overview
  - training/pre-training-data
---

# Tokenization

The tokenizer maps raw byte/character text to a sequence of integer
IDs in $\{0, \ldots, |V|-1\}$ that the LLM consumes. It sits between
the user and the model and is, surprisingly, **load-bearing for
quality**: vocabulary choice, byte-fallback strategy, and pre-
tokenization rules shape what the model can represent compactly.
Modern frontier LLMs use one of three families: SentencePiece-BPE,
tiktoken-BPE, or — newly — tokenizer-free byte-patch models (BLT).

## 1. Formal definition — BPE

**Byte-Pair Encoding** (Gage 1994; Sennrich et al. 2016 for NLP) is a
greedy bottom-up clustering of frequent symbol pairs into a fixed-size
vocabulary `[sennrich2016 §3.2; kb/excerpts/sennrich2016#sec-3-2]`.

**Training:**

1. Initialize a base alphabet — UTF-8 bytes (256 symbols) for byte-
   level BPE, or unique Unicode characters for character-level.
2. Tokenize the training corpus into base symbols.
3. Repeatedly find the most frequent adjacent pair $(a, b)$, add a new
   symbol $ab$ to the vocabulary, and replace all occurrences.
4. Stop when $|V|$ reaches the target.

The resulting vocabulary $V$ is a multiset of byte-strings of varying
lengths, each with an integer ID. **Encoding** at inference uses the
*merge rules* (the ordered sequence of pairs added in step 3) to
deterministically segment new text.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $V$ | tokenizer vocabulary |
| $|V|$ | vocabulary size (32K–256K in modern LLMs) |
| $\mathbf{t} \in V^N$ | tokenized sequence of length $N$ |
| $\rho = N / \mathrm{len}_{\text{chars}}$ | tokens-per-char ratio (lower = more compressed) |
| BOS, EOS | beginning/end-of-sequence special tokens |

## 2. Mechanism — three production families

### 2.1 SentencePiece (LLaMA, Qwen, Mistral, OLMo)

SentencePiece (Kudo & Richardson 2018) is a tokenizer **library** that
implements both BPE and unigram-LM segmentation. Its defining choice:
it operates **directly on raw text**, treating whitespace as a regular
character (encoded as `▁` U+2581) rather than requiring a pre-
tokenization step `[kudo2018-sentencepiece §3; kb/excerpts/kudo2018-sentencepiece#sec-3]`.

Practical consequences:

- **Round-trip lossless.** `decode(encode(x)) == x` even for arbitrary
  whitespace and Unicode, because no information is discarded by a
  language-specific pre-tokenizer.
- **Byte-fallback.** Out-of-vocabulary characters fall through to the
  256 byte tokens; no UNK token needed.
- **Adopted by:** LLaMA-1/2/3 (32K and later 128K vocab), Mistral,
  Qwen (custom byte-level BPE built on SentencePiece conventions),
  Gemma, OLMo `[touvron2023 §2.3; meta-llama3 §3.1]`.

### 2.2 tiktoken (GPT-2/3/4, Phi-4)

OpenAI's tiktoken is a byte-level BPE that pre-tokenizes input via a
regex pattern (e.g., `'(?i:[sdmt]|ll|ve|re)`, `\p{L}+`, `\p{N}{1,3}`,
`\s+`) so merges never cross category boundaries. The base alphabet is
the 256 UTF-8 bytes (no Unicode-character base); merges are over byte
sequences `[radford2019-gpt2 §2.2; kb/excerpts/radford2019-gpt2#sec-2-2]`.

Vocabulary sizes have grown dramatically: GPT-2 used 50,257; cl100k
(GPT-3.5/4) uses 100,256; o200k (GPT-4o) uses 200K; Phi-4 uses
tiktoken's 100K vocab `[phi4 §2.1; kb/excerpts/phi4#sec-2-1]`.

### 2.3 Byte-Latent Transformer — tokenizer-free (BLT, Meta 2024)

BLT replaces the tokenizer with a **learned byte-patch boundary
predictor** `[blt2024 §3]`. Architecture:

1. **Local encoder** — a small Transformer over raw UTF-8 bytes that
   predicts patch boundaries via a per-byte entropy estimator.
2. **Patch grouping** — bytes are grouped into variable-length patches
   based on entropy thresholds.
3. **Latent transformer** — the main LLM operates over patch
   representations (effectively coarser tokens, dynamically sized).
4. **Local decoder** — a small Transformer expands patch outputs back
   to byte logits.

At 8B parameters / 4T training bytes, BLT matched LLaMA-3 quality at
**50% fewer inference FLOPs** `[blt2024 §1, abstract; kb/excerpts/blt2024#abstract]`.
This is the strongest existence proof to date that tokenizer-free is
viable at non-trivial scale.

## 3. Variants and lineage

### 3.1 Vocabulary-size trade-off

| Model | $|V|$ | Tokens/word (English) | Notes |
|---|---|---|---|
| GPT-2 | 50,257 | ~1.3 | byte-level BPE baseline |
| LLaMA-1 | 32,000 | ~1.5 | SentencePiece BPE; English-skewed `[touvron2023 §2.3]` |
| LLaMA-2 | 32,000 | ~1.5 | unchanged from LLaMA-1 |
| LLaMA-3 | 128,256 | ~1.0 | SentencePiece-BPE expanded; better non-English `[meta-llama3 §3.1]` |
| GPT-4o (o200k) | ~200,000 | ~0.9 | tiktoken; multilingual coverage |
| Mistral-NeMo | 131,072 | ~1.0 | tekken (Mistral's variant of tiktoken) |
| DeepSeek-V3 | 128,000 | ~1.0 | byte-level BPE `[deepseek-v3 §2.1]` |
| Phi-4 | 100,000 | ~1.0 | tiktoken cl100k `[phi4 §2.1; kb/excerpts/phi4#sec-2-1]` |

Larger $|V|$ → fewer tokens per document → faster generation per token
of *content*, but **larger embedding matrix** $|V| \cdot d_{\text{model}}$
and **more parameters that see rare-token gradients only**. LLaMA-3's
4× vocab expansion (vs LLaMA-2) added ~525M parameters at 8B scale
($128256 \cdot 4096 = 525\text{M}$, untied).

### 3.2 Glitch tokens

Tokens learned during BPE merging but never seen at meaningful frequency
during LM pretraining produce **unpredictable model behavior** when
prompted. The canonical example is `SolidGoldMagikarp` in GPT-2/3,
discovered by Watkins & Rumbelow 2023 (community report, Tier C). The
mechanism: the embedding $E_{\text{in}}[t]$ for token $t$ never receives
a useful gradient, so it sits near initialization in a region of
embedding space that the model has no learned response to. The fix is
to filter the BPE vocab against post-tokenization training-data
frequency; modern frontier tokenizers (GPT-4o o200k, LLaMA-3) report
explicit unused-token reservation for this reason.

### 3.3 Reasoning failure modes from tokenization

The tokenizer can break model competence on tasks where the relevant
unit isn't a token boundary:

- **Multi-digit arithmetic.** GPT-3.5's BPE merged some multi-digit
  numbers ("12345" as one token) but not others. The model's ability
  to do digit-by-digit arithmetic depends on whether the digits are
  separated. LLaMA-3 explicitly tokenizes digits individually
  `[meta-llama3 §3.1]`.
- **Code.** Python's whitespace-significance interacts with BPE
  whitespace handling. Tab vs. 4-spaces tokenize differently;
  significant-whitespace bugs cluster around tokenizer boundaries.
- **Reversal / spelling.** "Reverse the word strawberry" tasks fail
  more often when the word is a single token (no per-character access).

## 4. Intuitions and analogies

[ANALOGY] BPE is **lossy compression at training time**: pick the
$|V|$-symbol code that minimizes expected tokens per document on the
corpus. The analogy returns to canonical form via the merge-frequency
greedy in §1: each merge "saves" one token per occurrence of the merged
pair. The analogy is sharp because BPE is, formally, a special case of
Huffman-style coding restricted to bottom-up pair merges.

[INTUITION] Why **subword** beat both **word-level** and **character-
level**: word-level fails on the long tail (rare names, technical
terms, OOV); character-level inflates sequence length by ~4–5×, which
is quadratic-in-attention waste. Subword $|V| \approx 32\text{K}$
captured the bulk of the gain `[sennrich2016 §1; kb/excerpts/sennrich2016#sec-1]`.
Character-level returned only when attention's $O(N^2)$ cost was
addressed (BLT, byte-level).

[INTUITION] **The tokenizer is a fixed prior the model can't learn its
way out of.** If the BPE merges are bad for a domain (e.g., LLaMA-1's
~1.5 tokens/word for non-English languages), the model trains on a
representation that can't represent the relevant units compactly.
Vocabulary expansion (LLaMA-3 → 128K) and tokenizer-free (BLT) are
both responses to this constraint.

## 5. Frontier and open questions (as of 2026-05)

- **Tokenizer-free at 70B+.** BLT demonstrated viability at 8B `[blt2024 §1]`.
  The 70B+ regime has not been published. [CONTRADICTION] Phase 1
  sweep §3 flags this as "no longer niche"; whether frontier-scale
  parity holds is open.
- **SuperBPE and length-maximizing BPE variants.** SuperBPE (Liu et al.
  2025, COLM) lifts BPE's cross-whitespace constraint, producing
  multi-word tokens like " of the"; reports 33% fewer tokens at 200K
  vocab and +4% average benchmark
  `[arXiv 2503.13423 §3]`. Length-MAX (Dong & Su 2025, arXiv
  2511.20849) reports 14–18% fewer tokens/char vs BPE at matched
  vocabulary. Adoption pending.
- **Information-theoretic tokenizer design.** Recent work (arXiv
  2601.09039, Jan 2026) takes Shannon-entropy lenses to tokenizer
  efficiency; a quantitative target for BPE replacement.
- **End-to-end RL-trained tokenizers.** arXiv 2602.13940 (2026)
  proposes joint tokenizer + LM training via RL; very recent.
- **"Tokenizer betrays reasoning" thread.** arXiv 2601.14658 (2026)
  documents tokenizer-induced failure modes systematically.
- **Tokenizer adaptation for pretrained models.** Adding new tokens
  (and initializing their embeddings) post-pretraining without
  quality loss is an open problem; relevant for vocabulary expansion
  and domain adaptation. See `kb/notes/architecture/embeddings-and-tying.md`.

## 6. See also

- `kb/notes/architecture/embeddings-and-tying.md` — what the integer
  IDs become inside the model.
- `kb/notes/architecture/transformer-overview.md` — where tokenization
  sits in the forward pipeline.
- `kb/notes/training/pre-training-data.md` — tokenizer training data
  and language coverage.
