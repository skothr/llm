# Phase 5 — LaTeX series brainstorm

_2026-05-04. Per `theory/docs/superpowers/specs/2026-05-03-theory-expansion-design.md` §6 Phase 5: "After KB is stable: a separate brainstorming round drafts the multi-paper LaTeX series structure, producing N paper specs each with their own implementation plan." This is the brainstorm — not a fixed plan. Picks a recommended shape and surfaces alternatives + tradeoffs._

## What we have to work with

After Phases 0-4 + the 2026-05-04 stub-fill round:

- **187 papers** in `kb/index/papers.json`; 88 with excerpt files.
- **54 draft notes** across 9 areas, all status `draft` (0 stubs).
- **`kb/index/timeline.md`** — fully linked chronological progression pre-2017 → 2026-H1.
- **`kb/index/contradictions.md`** — 99 cross-area open questions, density-ranked.
- **`kb/glossary.md`** — ~500 terms with citations.

Density by area (notes / new excerpts available / contradictions surfaced):

| Area | Notes | Excerpts | Contradictions |
|------|------:|---------:|---------------:|
| architecture | 13 | high | 13 |
| training | 6 | high | 6 |
| post-training | 5 | medium | 6 |
| reasoning | 5 | medium | 14 |
| inference | 5 | medium | 6 |
| scaling | 5 | high | 9 |
| interpretability | 6 | high (post-stubfill) | 6 |
| evaluation | 4 | medium | 4 |
| alignment | 5 | medium-high | 9 |

The KB already operates as a usable working substrate; the question is how to project it into one or more LaTeX deliverables.

## Five candidate paper-series shapes

### Shape A — "9 surveys" (one paper per KB area)

**Form:** Each KB area becomes a 30-50 page survey paper.

**Pros:** Trivial mapping from substrate. Easy to staff (each paper has a clear champion). Surveys are widely-cited and fill a real gap (no current survey covers, e.g., 2024-26 reasoning-training as a unified field).

**Cons:** "Surveys" is an unambitious target for the user's stated bar (Feynman, Karpathy-style). The KB's contradictions are mostly *cross-area* (e.g., RLVR-and-search overlap, MoE-and-distributed-training, MLA-and-KV-cache); per-area surveys lose them. Total page count high (300-450 pages), uneven quality risk.

### Shape B — "Lifecycle 4-paper" (research-cycle phases)

Four papers, each covering ~12-14 leaf topics:

1. **Building** — architecture + training (19 topics).
2. **Aligning** — post-training + reasoning (10 topics).
3. **Running** — inference + scaling (10 topics).
4. **Understanding** — interpretability + evaluation + alignment-eval (15 topics).

**Pros:** Maps how a frontier-LLM team actually divides labor. Each paper is broad enough to give context, narrow enough to cut. Page count ~120-200 each.

**Cons:** "Aligning" papers tend to bury reasoning; "Understanding" is two distinct fields stitched. Doesn't give cross-cutting themes (scaling crosses Building and Running) a natural home.

### Shape C — "Thesis 5-paper" (each paper has a load-bearing claim) — **recommended**

Five papers, each organized around a *thesis* the KB makes defensible, not around a topic boundary:

1. **The modern Transformer is a small set of choices.** The architectural KV/expert/normalization/positional axes; what choices the frontier converges on (MHA→MLA, dense→fine-grained MoE, LN→RMSNorm/Pre-LN, sinusoidal→RoPE/YaRN, FA1→FA3+NSA) and why. Anchor: `kb/notes/architecture/`.
2. **Training is now a multi-stage pipeline, not a single objective.** Pre-training-data → mixed-precision → distributed scaffolding → SFT → preference RL → reasoning-RL. Each stage's load-bearing variables (token count, mixture, precision, batch size, KL constraints, verifier signal). Anchors: `kb/notes/training/`, `kb/notes/post-training/`, `kb/notes/scaling/`.
3. **Reasoning is compute, search, and verification.** Inference-time-compute scaling laws, CoT faithfulness, process-supervision vs outcome-supervision, RLVR/GRPO, search families (BFS/DFS/MCTS) as a parallel scaling axis to training compute. Anchor: `kb/notes/reasoning/` + `kb/notes/scaling/inference-time-compute-scaling.md`.
4. **The internal computation can be partially read.** Lenses, probes, SAEs, activation patching, circuit tracing — what each method commits to, what cross-method evidence looks like, where the methods disagree. Includes the contradictions surface for interpretability claims. Anchor: `kb/notes/interpretability/`.
5. **What we measure and what slips through.** Knowledge benchmarks (MMLU lineage, contamination), reasoning benchmarks, agentic benchmarks, safety evaluation, alignment threats (sycophancy, scheming, alignment-faking). Anchor: `kb/notes/evaluation/`, `kb/notes/alignment/`.

**Pros:** Each paper has a thesis a reader can argue with. Reorganizes the KB without leaving topics homeless. Aligns with the contradiction-density heatmap (architecture, reasoning, alignment lead — those are Papers 1, 3, 4-5).

**Cons:** Inference (`kv-cache`, `quantization`, `serving-systems`, `speculative-decoding`, `structured-output`) doesn't have its own paper. Could fold into Paper 1 (architecture) or Paper 3 (compute/scaling) — neither natural. Possible 6th paper "Running large LMs" or absorb piecewise.

### Shape D — "Karpathy-style monograph"

Single 300-500 page comprehensive book; chapters mirror KB notes.

**Pros:** Matches "Karpathy-style" framing user gave at start. Single citation target. Easier to maintain as the field moves (one source, not five).

**Cons:** Massive scope; even 300pp risks staleness in a 6-month-half-life field. Loses the audience-finding benefit of separate papers (a reasoning researcher won't read a 500pp book to find the reasoning chapter). Maintenance scales poorly.

### Shape E — "Pedagogical primer + reference appendix" (Karpathy-meets-textbook)

Two-tier deliverable:
- **Primer** (60-100pp): self-contained intro to modern LLMs. Builds intuition with worked equations + small reference implementations (à la nanoGPT). Cites the KB heavily.
- **Reference** (the KB itself, exposed as a navigable site): notes/excerpts/timeline/contradictions stay in markdown but get rendered for browsing.

**Pros:** Plays to the substrate's natural form. Primer is teachable; reference is queryable. Exposes the KB to non-Anthropic readers without first writing N surveys.

**Cons:** Primer alone won't satisfy the "comprehensive theory framework" goal user articulated. Site-rendering is engineering, not research.

## Recommendation

**Shape C (5-paper thesis)** — with Inference folded into Paper 1 (kv-cache, FA, quantization) and Paper 3 (speculative-decoding, structured-output, serving). This loses the standalone-inference paper but: (a) inference *is* an architectural concern, (b) speculative + structured + serving *are* compute-scaling concerns, and (c) contradictions density on inference (6) is the lowest of the contention areas, so it doesn't carry its own thesis.

Why C wins:
1. **Reader has a question, finds the relevant paper.** A reader investigating "is RLVR generalizing or memorizing?" finds Paper 3, not "Aligning" (B) or "Reasoning Survey" (A).
2. **The KB's contradictions cluster naturally.** Density-leading areas (architecture, reasoning, alignment) become Papers 1, 3, 5 — each gets its open questions concentrated rather than scattered.
3. **Theses age more honestly than surveys.** "The frontier converged on MLA" can be falsified by a 2027 paper; "A survey of attention" never can.

## Concrete next step: Paper-1 sketch (as illustration)

If we go with Shape C, here's what Paper 1 could look like:

> **Title (working):** *The modern Transformer is a small set of choices: convergent architecture in frontier LLMs, 2017–2026.*
>
> **Outline (~80 pages):**
> 1. Pre-2017 prequel: what Transformer replaced. Source: `timeline.md` 2017-prequel.
> 2. The canonical 2017 Transformer (formal). Source: `kb/notes/architecture/transformer-overview.md` + `attention-mechanism.md` §1-3 + excerpt `vaswani2017`.
> 3. Tokenization → embedding → unembedding pipeline. Sources: `tokenization.md`, `embeddings-and-tying.md`.
> 4. Positional encoding axis (sinusoidal → RoPE → YaRN → NoPE, with [CONTRADICTION] on long-context extrapolation).
> 5. **Attention KV-sharing axis (MHA → MQA → GQA → MLA).** Already drafted as the pilot; lift directly. The thesis paper's anchor.
> 6. Hardware-implementation axis (FA1 → FA2 → FA3, NSA, ring/sliding).
> 7. FFN axis (ReLU → GLU → SwiGLU; dense → MoE → fine-grained MoE; shared experts).
> 8. Normalization axis (LN → RMSNorm; Post-LN → Pre-LN → DeepNorm).
> 9. SSM and hybrids (Mamba, Mamba-2; Jamba/Samba/Zamba/Hymba) — the alternative path.
> 10. Multimodal extensions (cross-attn-bridge vs early-fusion; native multimodal).
> 11. Multi-token prediction (DeepSeek-V3) and what it implies for the next decade.
> 12. **Inference adjuncts** (folded in): KV-cache lifecycle, quantization regimes (INT8/INT4/FP8/NVFP4), structured output as constrained decoding.
> 13. Frontier-2026 snapshot: what choices the leading models share, where they diverge.
> 14. Contradictions live: MLA-vs-MHA at scale (single-source), SSM hybrid quality at frontier, NSA vs FA3 head-to-head.

Roughly 80 pages, ~250 citations, drawn from ~22 KB notes. Implementation plan for this paper alone would need ~3-4 weeks at the user's quality bar.

## Preconditions for actually starting Paper-1

Before any LaTeX implementation plan starts:

1. **Phase 2.5 — PDF deepening for Paper-1's anchor papers.** ~30-40 papers (Vaswani, Shazeer, Ainslie, Dao, DeepSeek-V2/V3, Shazeer GLU, Ba LayerNorm, Zhang RMSNorm, Su RoPE, Press tied-embed, Kudo SentencePiece, Mamba/Mamba-2, Alayrac Flamingo, Gloeckle MTP, Gu SSM, Yuan NSA, Press ALiBi, etc.) need to be re-pulled and §/eq-anchored. The pilot (`attention-mechanism.md`) is the only note with full dual-citation anchoring; that's not enough provenance for a paper.
2. **One reviewer pass on Papers 1-5 outlines.** Make sure no thesis is unfalsifiable.
3. **`theory/series/` substructure** — placeholder per the spec. Each paper gets its own subdirectory with `paper-N.tex`, `outline.md`, `references.bib`.

## Status of the brainstorm

This is a brainstorm — pick A, B, C, D, E or a mutation. C is the recommended default. The KB is now a working substrate either way; this doc is here to make the next decision explicit rather than implicit.
