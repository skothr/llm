---
paper-key: sennrich2016
title: "Neural Machine Translation of Rare Words with Subword Units"
authors: Sennrich, Haddow, Birch
year: 2016
venue: ACL
arxiv: 1508.07909
local_pdf: theory/sources/papers/sennrich2016_bpe.pdf
extracted: 2026-05-05 (Phase 2.5 deepening)
---

# Verbatim excerpts — Sennrich et al. 2016, "Neural Machine Translation of Rare Words with Subword Units"

The paper that introduced BPE (byte-pair encoding) to neural sequence
modelling. The two contributions, in the authors' own framing: (1)
demonstrate that open-vocabulary NMT works by encoding rare words as
sequences of subword units; (2) adapt the BPE compression algorithm
(Gage 1994) to the task of word segmentation. The merge-pair-iteratively
recipe defined here is the load-bearing reference for every
subword tokenizer in modern LLMs (LLaMA, GPT-2/3/4, Qwen, DeepSeek,
Mistral).

## Abstract {#abstract}

> Neural machine translation (NMT) models typically operate with a fixed
> vocabulary, but translation is an open-vocabulary problem. Previous
> work addresses the translation of out-of-vocabulary words by backing
> off to a dictionary. In this paper, we introduce a simpler and more
> effective approach, making the NMT model capable of open-vocabulary
> translation by encoding rare and unknown words as sequences of subword
> units. This is based on the intuition that various word classes are
> translatable via smaller units than words, for instance names (via
> character copying or transliteration), compounds (via compositional
> translation), and cognates and loanwords (via phonological and
> morphological transformations). We discuss the suitability of
> different word segmentation techniques, including simple character
> n-gram models and a segmentation based on the byte pair encoding
> compression algorithm, and empirically show that subword models
> improve over a back-off dictionary baseline for the WMT 15 translation
> tasks English→German and English→Russian by up to 1.1 and 1.3 BLEU,
> respectively.

## §1 Two main contributions {#sec-1}

> This paper has two main contributions:
>
> - We show that open-vocabulary neural machine translation is possible
>   by encoding (rare) words via subword units. We find our architecture
>   simpler and more effective than using large vocabularies and back-off
>   dictionaries (Jean et al., 2015; Luong et al., 2015b).
>
> - We adapt byte pair encoding (BPE) (Gage, 1994), a compression
>   algorithm, to the task of word segmentation. BPE allows for the
>   representation of an open vocabulary through a fixed-size vocabulary
>   of variable-length character sequences, making it a very suitable
>   word segmentation strategy for neural network models.

## §3.2 Byte Pair Encoding — the algorithmic core {#sec-3-2}

> Byte Pair Encoding (BPE) (Gage, 1994) is a simple data compression
> technique that iteratively replaces the most frequent pair of bytes in
> a sequence with a single, unused byte. We adapt this algorithm for
> word segmentation. Instead of merging frequent pairs of bytes, we
> merge characters or character sequences.
>
> Firstly, we initialize the symbol vocabulary with the character
> vocabulary, and represent each word as a sequence of characters, plus
> a special end-of-word symbol '·', which allows us to restore the
> original tokenization after translation. We iteratively count all
> symbol pairs and replace each occurrence of the most frequent pair
> ('A', 'B') with a new symbol 'AB'. Each merge operation produces a
> new symbol which represents a character n-gram. Frequent character
> n-grams (or whole words) are eventually merged into a single symbol,
> thus BPE requires no shortlist. The final symbol vocabulary size is
> equal to the size of the initial vocabulary, plus the number of merge
> operations — the latter is the only hyperparameter of the algorithm.
>
> For efficiency, we do not consider pairs that cross word boundaries.
> The algorithm can thus be run on the dictionary extracted from a text,
> with each word being weighted by its frequency.

The reference Python implementation (Algorithm 1, p.3) — load-bearing
because every modern BPE library traces lineage to this snippet:

```python
import re, collections
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs
def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out
vocab = {'l o w </w>': 5, 'l o w e r </w>': 2,
         'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)
```

> Figure 1 shows a toy example of learned BPE operations. At test time,
> we first split words into sequences of characters, then apply the
> learned operations to merge the characters into larger, known symbols.
> This is applicable to any word, and allows for open-vocabulary
> networks with fixed symbol vocabularies.

## §3.2 Joint vs independent BPE {#sec-3-2-joint}

> We evaluate two methods of applying BPE: learning two independent
> encodings, one for the source, one for the target vocabulary, or
> learning the encoding on the union of the two vocabularies (which we
> call **joint BPE**). The former has the advantage of being more
> compact in terms of text and vocabulary size, and having stronger
> guarantees that each subword unit has been seen in the training text
> of the respective language, whereas the latter improves consistency
> between the source and the target segmentation.

The joint-BPE pattern is the ancestor of the multilingual single-vocab
tokenizers used in LLaMA (32k tokens) and Qwen/DeepSeek
(byte-level BPE, ~128k+).

## §4 Evaluation framing {#sec-4}

> We aim to answer the following empirical questions:
>
> - Can we improve the translation of rare and unseen words in neural
>   machine translation by representing them via subword units?
> - Which segmentation into subword units performs best in terms of
>   vocabulary size, text size, and translation quality?
>
> We perform experiments on data from the shared translation task of
> WMT 2015. For English→German, our training set consists of 4.2
> million sentence pairs, or approximately 100 million tokens. For
> English→Russian, the training set consists of 2.6 million sentence
> pairs, or approximately 50 million tokens.

## §5 Headline result {#sec-5}

> The main contribution of this paper is that we show that neural
> machine translation systems are capable of open-vocabulary translation
> by representing rare and unseen words as a sequence of subword units.
> This is both simpler and more effective than using a back-off
> translation model. We introduce a variant of byte pair encoding for
> word segmentation, which is capable of encoding open vocabularies with
> a compact symbol vocabulary of variable-length subword units. … Our
> best subword segmentations achieve up to 1.1 BLEU and 1.3 BLEU
> improvements for English→German and English→Russian respectively.

## Source notes

- Tier A canonical (ACL 2016 main conference paper). The original
  byte-fallback / byte-level BPE in modern LLMs (GPT-2 onward) is a
  later refinement; this paper is character-level BPE with `</w>` end-
  of-word markers.
- PDF retrieved from arXiv. Anchors are stable; equation/algorithm
  numbers preserved.
- The 2018 SentencePiece paper (Kudo & Richardson) wraps this algorithm
  into a language-independent library and adds the unigram-LM
  alternative; see `kb/excerpts/kudo2018-sentencepiece.md`.

[Verified from PDF on 2026-05-12] Added §1 contributions, §3.2 BPE
algorithmic core, §3.2 joint-vs-independent variant, §4 evaluation, §5
headline. Abstract verified verbatim.
