---
paper_key: kudo2018-sentencepiece
title: "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing"
authors: Kudo, Richardson
year: 2018
venue: EMNLP demo track
arxiv: 1808.06226
local_pdf: theory/sources/papers/kudo2018_sentencepiece.pdf
type: excerpts
note: Verbatim from the v1 arXiv PDF (Aug 2018). SentencePiece is the tokenizer library used by LLaMA, Mistral, Qwen, OLMo, Gemma, and most modern open LLMs. It implements BPE and unigram-LM segmentation directly on raw text — no language-specific pre-tokenization.
---

# Excerpts — Kudo & Richardson 2018, "SentencePiece"

## Abstract — what the library does {#abstract}

> This paper describes SentencePiece, a language-independent subword tokenizer and detokenizer designed for Neural-based text processing, including Neural Machine Translation. It provides open-source C++ and Python implementations for subword units. While existing subword segmentation tools assume that the input is pre-tokenized into word sequences, SentencePiece can train subword models directly from raw sentences, which allows us to make a purely end-to-end and language independent system.

## §2 System Overview — the three-stage architecture {#sec-2}

> SentencePiece consists of four main components: Normalizer, Trainer, Encoder, and Decoder. The Normalizer is a module to normalize semantically-equivalent Unicode characters into canonical forms. The Trainer trains the subword segmentation model from the normalized corpus. We specify the type of subword model as the parameter of Trainer. Encoder internally executes Normalizer to normalize the input text and tokenizes it into a subword sequence with the subword model trained by Trainer. Decoder converts the subword sequence into the normalized text.

## §3 Library Design — design choices that distinguish SentencePiece {#sec-3}

### §3.1 Lossless tokenization {#sec-3-1}

> Existing subword segmentation tools tokenize the input text into word sequences in the language-dependent pre-processing, which is not always reproducible at the decoding time. … SentencePiece implements lossless tokenization, which preserves all information necessary to reproduce the original text. The basic idea is simple: SentencePiece treats the input text just as a sequence of Unicode characters. Whitespace is also handled as a normal symbol. To handle the whitespace as a basic token explicitly, SentencePiece first escapes the whitespace with a meta symbol `_` (U+2581) as follows:
>
>     Hello␣World. → Hello▁World.
>
> Then, this text is segmented into small pieces, for example:
>
>     [Hello] [▁Wor] [ld] [.]

The `▁` (U+2581 LOWER ONE EIGHTH BLOCK) prefix marks "begins-with-space"; this is how the decoder reconstructs the whitespace.

### §3.2 Efficient subword training and segmentation {#sec-3-2}

> Existing subword segmentation tools use the algorithm of Sennrich et al. (2016) for training BPE. The naive implementation has the time complexity of $O(N^2)$, where $N$ is the length of input sentences. SentencePiece adopts an $O(N \log N)$ algorithm in which the merged symbols are managed by a binary heap (priority queue).

This efficiency note is why SentencePiece is fast enough to train tokenizers on multi-terabyte corpora — relevant for modern LLMs trained on 1T+ tokens.

### §3.3 Vocabulary id management {#sec-3-3}

> SentencePiece manages the vocabulary to id mapping to convert input text into id sequences and vice versa. The size of the vocabulary is specified by the `--vocab_size` flag of training. … The user can also define special meta symbols (e.g., `<s>`, `</s>`, `<pad>`, etc.) as user-defined symbols by the `--user_defined_symbols` flag.

### §3.4 Customizable character normalization {#sec-3-4}

> SentencePiece by default normalizes the input text with the Unicode NFKC normalization. NFKC normalization is widely used in many natural language applications. … In addition to default NFKC normalization, SentencePiece supports custom normalization rules.

## §4 Subword model — both BPE and unigram {#sec-4}

SentencePiece supports two segmentation algorithms:

> **Byte-Pair-Encoding (BPE):** the standard greedy merge algorithm of Sennrich et al. (2016), implemented over Unicode characters or bytes.
>
> **Unigram language model:** a probabilistic model that picks subwords to maximize the likelihood of the corpus given a prior distribution over subwords. Better tokenization-quality but slower than BPE; used in some Asian-language models.

LLaMA-1/2 use BPE mode of SentencePiece; LLaMA-3 expanded the vocab. Qwen uses a custom byte-level BPE built on SentencePiece conventions.

## §5 Experiments — translation quality {#sec-5}

> We compared SentencePiece directly applied to raw text versus the standard subword regularization on pre-tokenized text. The results show that direct training on raw text **does not degrade translation quality** while removing the need for language-specific pre-tokenizers.

> Furthermore, byte-level BPE handles never-seen characters via the byte-fallback feature, eliminating the UNK token entirely.

## Note on byte-fallback {#sec-byte-fallback}

The byte-fallback feature (added in later SentencePiece versions and standard in modern LLM training) lets out-of-vocabulary characters fall through to the 256 byte tokens. This is what allows LLaMA-3, Qwen, and DeepSeek-V3 to handle arbitrary Unicode without UNK — they use SentencePiece-BPE with byte-fallback enabled.
