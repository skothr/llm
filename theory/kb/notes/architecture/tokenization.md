---
topic: architecture/tokenization
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - sennrich2016
  - blt2024
  - superbpe2025
  - length-max-tokenizer-2025
---

# Tokenization

**Status:** stub. Drafted from Phase 1 landscape sweep; needs full Phase 2
treatment.

## What this topic covers

How raw text becomes the integer sequence the Transformer consumes.
The legacy default is **subword BPE** (Sennrich 2016); production
LLMs use SentencePiece-BPE (LLaMA), tiktoken-BPE (GPT, Phi-4), or
custom byte-level BPE (Qwen). The 2024–2026 frontier has two threads:
(i) **tokenizer-free** byte-level architectures (Meta's BLT, ACL
2025), and (ii) **better BPE** via superword bridging (SuperBPE) or
length-maximizing variants (Length-MAX). Vocabulary sizes have grown
from ~32K (LLaMA-1) to ~128K (LLaMA-3, Mistral, Qwen3) to ~200K
(SuperBPE). The "tokenizer betrayal" thread shows tokenizer artifacts
cause specific reasoning-failure modes
`[Phase 1 sweep §2 tokenization]`.

## Primary sources to read (in order)

1. `sennrich2016` — *Neural Machine Translation of Rare Words with
   Subword Units* (1508.07909) — BPE, the foundational algorithm.
   Already in `theory/sources/papers/sennrich2016_bpe.pdf`.
2. `blt2024` — *Byte Latent Transformer: Patches Scale Better Than
   Tokens* (2412.09871) — tokenizer-free; matches LLaMA 3 at 50%
   fewer inference FLOPs at 8B/4T.
3. `superbpe2025` — *SuperBPE: Space Travel for Language Models*
   (2503.13423) — cross-whitespace superword tokens; 33% fewer
   tokens at 200K vocab; +4% avg benchmark.
4. `length-max-tokenizer-2025` — *Length-MAX Tokenizer for Language
   Models* (2511.20849) — 14–18% fewer tokens/char vs BPE.
5. *Say Anything but This: When Tokenizer Betrays Reasoning*
   (2601.14658) — tokenizer artifacts → reasoning failure modes.
6. *Information-Theoretic Perspective on LLM Tokenizers*
   (2601.09039) — theoretical lens on tokenizer efficiency.

## Key claims to ground (Phase 2 todo)

- BPE algorithm: greedy merge of frequent symbol pairs, building a
  fixed-size vocabulary $V$ from a base alphabet (UTF-8 bytes or
  Unicode chars). Train-time procedure vs inference-time decoding.
- SentencePiece vs tiktoken differences: byte-fallback handling,
  whitespace tokenization, special-token reservations.
- **Vocabulary sizing tradeoffs**: larger $V$ → fewer tokens per
  document → faster generation, but also → larger embedding matrix
  $|V| \times d_{\text{model}}$ → more parameters, potentially worse
  embedding quality on rare tokens.
- The **"glitch token"** phenomenon: tokens learned during BPE merge
  but never seen at significant frequency during LM training; produce
  unpredictable behavior when prompted. (e.g., `SolidGoldMagikarp`
  in GPT-2/3.)
- BLT mechanism: dynamic byte-patching via a learned entropy-based
  segmenter; a small "patcher" model produces patches; the main LLM
  operates on patches. Inference-time FLOP reduction comes from
  smaller effective sequence lengths.
- SuperBPE mechanism: lift the cross-whitespace constraint, producing
  multi-word tokens like " of the". Tradeoff: longer tail of rare
  multiword tokens.
- Tokenizer's effect on reasoning: number tokenization (especially
  multi-digit), code tokenization (whitespace-significant in Python,
  identifier merging), and language-coverage skew (English vs CJK
  token-per-character ratio).
- Stale-prior correction (per Phase 1 sweep §3): "tokenizer-free is
  niche and slow" → BLT at 8B matches LLaMA-3 quality with 50% fewer
  inference FLOPs. Treat as no-longer-niche.

## Related notes

- `kb/notes/architecture/embeddings-and-tying.md` — what happens to
  the integer token IDs once they enter the model.
- `kb/notes/architecture/transformer-overview.md` — where tokenization
  sits in the input pipeline.
- `kb/notes/training/pre-training-data.md` — tokenizer training data
  and its effects on language coverage.
