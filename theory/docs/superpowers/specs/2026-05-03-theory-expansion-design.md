# Theory expansion: KB construction + LaTeX series substrate

**Status:** approved (2026-05-03)
**Author:** brainstorming session, Opus 4.7
**Supersedes:** treats `2026-04-06-llm-core-architecture-design.md` as the v1
"core architecture only" deliverable. This v2 design expands `theory/` from a
single LaTeX/PDF doc into a knowledge-base substrate plus a deferred LaTeX
paper series.

---

## 1. Goal

Bring `theory/` to bleeding-edge LLM coverage with both **breadth** (every
major area of contemporary LLM research) and **depth** (every hard claim
grounded in primary sources). Output is two-layered:

- **A KB** — modular, citation-grounded, RAG-ready, designed so synthesis
  can never be confused with primary-source quotation. Lives in `theory/kb/`.
- **A future LaTeX paper series** — outlined *after* the KB is complete, so
  topic structure follows from observed research breadth rather than from
  stale priors. Lives in `theory/series/` (placeholder until Phase 5).

Quality bar: Feynman-style — math-first formalism that returns to canonical
symbolic form, mechanism explained mechanically, intuition delivered with
explicit `[INTUITION]` / `[ANALOGY]` tagging so analogies never launder as
formal claims.

## 2. Scope

**In scope:** LLM-specific theory and practice.

- Architecture (attention, FFN/MoE, normalization, position encoding, SSMs,
  hybrids, long-context)
- Training (data, optimization, distributed, mixed-precision)
- Post-training & alignment (SFT, RLHF, DPO, RLAIF, RLVR, GRPO, Constitutional)
- Reasoning (CoT, process supervision, test-time compute, RL for reasoning)
- Inference (KV-cache, quantization, serving, speculative decoding)
- Scaling (Kaplan, Chinchilla, μP, frontier updates)
- Interpretability (lens techniques, mech-interp, SAEs, activation patching)
- Evaluation (benchmarks, methodology)
- Safety (red-teaming, watermarking, scalable oversight, deception)
- Multimodal **only** as LLM-side extensions (vision encoder + projector +
  LLM body; native multimodal early-fusion). Pure vision, pure
  text-to-image diffusion, pure speech synthesis are out of scope.

**Out of scope:** RL agents/robotics foundation models, world models, classical
ML foundations, pure vision/audio generative models. May be added in a future
expansion.

## 3. Final directory layout

```
theory/
├── README.md                         # top-level navigation
├── kb/                               # knowledge base — the substrate
│   ├── README.md                     # epistemic rules + citation format
│   ├── notes/                        # digested synthesis, one file per topic
│   │   ├── architecture/
│   │   ├── training/
│   │   ├── post-training/
│   │   ├── reasoning/
│   │   ├── inference/
│   │   ├── scaling/
│   │   ├── interpretability/
│   │   ├── evaluation/
│   │   └── alignment/
│   ├── excerpts/                     # verbatim quoted passages, one file per paper
│   │   └── <paper-key>.md
│   ├── index/
│   │   ├── papers.json               # successor to sources/index.json
│   │   ├── topics.md                 # topic graph + per-topic status
│   │   └── timeline.md               # chronological 2017→present progression
│   └── glossary.md                   # term → canonical citation map
├── sources/
│   ├── README.md                     # tier definitions
│   ├── papers/                       # PDFs (primary citation source)
│   └── forums/                       # selectively archived blog/forum snapshots
├── plans/                            # research/KB construction plans + progress
│   └── 2026-05-03-research-plan.md
├── archive/
│   ├── llm-core-architecture-2026-04-11.pdf
│   ├── llm-core-architecture-2026-04-16.pdf
│   └── 2026-05-03-pre-expansion/     # full snapshot of the v1 deliverable
│       ├── llm-core-architecture.tex
│       ├── llm-core-architecture.pdf
│       ├── GLOSSARY.md
│       └── visuals/llm-architecture-diagram.html
├── series/                           # placeholder for future LaTeX paper series
│   └── README.md                     # "outlined after KB pass complete"
└── docs/superpowers/specs/
    ├── 2026-04-06-llm-core-architecture-design.md   # existing
    └── 2026-05-03-theory-expansion-design.md        # this spec
```

**Migrations from current state:**

| From | To |
|------|-----|
| `theory/build/llm-core-architecture/` | `theory/archive/2026-05-03-pre-expansion/` |
| `theory/llm-core-architecture.pdf` | `theory/archive/2026-05-03-pre-expansion/` |
| `theory/GLOSSARY.md` | `theory/archive/2026-05-03-pre-expansion/GLOSSARY.md` (frozen) and seed for `theory/kb/glossary.md` (will grow) |
| `theory/visuals/` | `theory/archive/2026-05-03-pre-expansion/visuals/` |
| `theory/sources/index.json` | `theory/kb/index/papers.json` (preserve content; schema expansion in Phase 1) |
| `theory/sources/papers/` | unchanged (still the canonical PDF location) |
| `theory/Makefile` | `theory/archive/2026-05-03-pre-expansion/Makefile` (no longer builds anything until series is outlined) |

## 4. KB epistemic tiers

Three tiers, separated by directory and by frontmatter, so retrieval can
filter by epistemic status.

### Tier 1 — `kb/excerpts/<paper-key>.md` (verbatim ground truth)

Every line is in a blockquote. No synthesis. Anchored to section/equation IDs
in the paper. Exists for direct retrieval and for note-citation resolution.

```markdown
---
paper: vaswani2017
title: Attention Is All You Need
captured: 2026-05-03
---

## §3.2 Scaled Dot-Product Attention

> "We call our particular attention 'Scaled Dot-Product Attention' [...]
> We compute the dot products of the query with all keys, divide each by
> sqrt(d_k), and apply a softmax function to obtain the weights on the values."
> — [vaswani2017 §3.2, p.4]

## Eq. 1 — Attention formula

> Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
> — [vaswani2017 §3.2, eq.1]
```

### Tier 2 — `kb/notes/<area>/<topic>.md` (digested synthesis)

Every load-bearing claim cites either a paper directly or an excerpt anchor.
Analogies and intuitions are tagged so they cannot be retrieved as fact.

Standard note structure (Feynman-aligned):

```markdown
---
topic: <topic-name>
area: <area>
status: stub | draft | reviewed | locked
last-updated: YYYY-MM-DD
primary-sources: [<paper-key>, ...]
related-notes: [<area>/<topic>, ...]
---

# <Topic title>

## Formal definition
<the math, with every variable defined immediately under the equation,
citation: [paper-key §X, eq.Y]>

## Mechanism
<how the math actually computes — what each term does at runtime>

## Variants and lineage
<bulleted list, each with a citation>

## [INTUITION] / [ANALOGY]
<tagged paragraph(s) — must always return to the canonical symbolic form>

## Frontier and open questions
<recent papers, contested territory, [CONTRADICTION] markers if sources disagree>

## See also
<cross-links to related notes>
```

### Tier 3 — `kb/index/` (structured indices)

`papers.json` schema:

```json
{
  "key": "vaswani2017",
  "title": "Attention Is All You Need",
  "authors": "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin",
  "year": 2017,
  "venue": "NeurIPS",
  "url": "https://arxiv.org/abs/1706.03762",
  "local_file": "sources/papers/vaswani2017_attention_is_all_you_need.pdf",
  "excerpts_file": "kb/excerpts/vaswani2017.md",
  "notes_referenced_by": ["architecture/attention-mechanism", "architecture/transformer-overview"],
  "topics": ["attention", "transformer", "encoder-decoder"],
  "category": "foundational | landmark | recent | tech-report | superseded",
  "summary": "..."
}
```

`topics.md`: hierarchical taxonomy with per-topic status (`stub`, `draft`,
`reviewed`, `locked`). Authoritative source of which areas exist.

`timeline.md`: chronological progression 2017→present, organized by
architectural shift / training-paradigm shift, with each entry citing one or
more papers.

## 5. Citation discipline (CLAUDE.md addition)

A new "Theory KB & citation discipline" section will be added to the project
`CLAUDE.md`:

```markdown
# Theory KB & citation discipline

When making technical claims about LLM architecture, training, inference, or
related theory:

1. **Every load-bearing claim cites a source.** Either:
   - `[paper-key §X]` — pointing to a paper in `theory/kb/index/papers.json`
   - `[kb/notes/<area>/<file>#<anchor>]` — pointing into a synthesis note

2. **Verify against the original PDF before propagating a KB-note claim**
   into LaTeX, code, or commit messages. The KB is digested; the paper is
   canonical.

3. **Analogies and intuitions are tagged, never asserted as fact.** Use
   `[ANALOGY]` or `[INTUITION]` so they can't be confused with formal claims.
   They must always return to the canonical symbolic form.

4. **If a claim depends on something not in the KB, add it before continuing.**
   Don't make claims you can't ground.

5. **Forum/blog citations are valid as discovery signals only.** They never
   solely back a hard technical claim — only primary papers (Tier A) can.
```

## 6. Research workflow (five phases)

### Phase 0 — Restructure (this session, mechanical)

Move files per Section 3, scaffold new dirs, add READMEs, update CLAUDE.md.
Commit. No research yet.

### Phase 1 — Current-state landscape sweep *(breadth, fast)*

- WebSearch + arxiv-recent + curated forums (r/LocalLLaMA,
  r/MachineLearning, HN, X researcher threads) + vendor research blogs
  (Anthropic / OpenAI / DeepMind / Hugging Face / Together / Mosaic).
- Pull recent papers (~last 6 months especially), identify hot topics,
  flag training-vs-current discrepancies.
- **Output:** v1 of `kb/index/topics.md`; candidate paper list in
  `papers.json` (metadata only, no PDFs yet); `plans/2026-05-03-landscape-sweep-findings.md`.
- **Stop when:** 2-3 independent forum/blog/paper signals converge on the
  major topics; further searches return strongly diminishing novelty.

### Phase 2 — Per-topic deep research + KB construction *(depth, slow)*

For each topic in the refined taxonomy:
- Gather canonical foundational papers + landmark recent ones.
- Download PDFs to `sources/papers/`.
- Register in `kb/index/papers.json`.
- Extract excerpts to `kb/excerpts/<key>.md`.
- Write digested synthesis to `kb/notes/<area>/<topic>.md`.
- Update `kb/glossary.md`.

**Stopping condition per topic:** the note answers — what is it, why does
it exist, what does the math say, what variants/lineage, what's the
frontier, what are the open questions — all citation-grounded.

**Parallelization:** spawn one subagent per area (~9 areas) using
`Agent(subagent_type=general-purpose)`. Each subagent gets the tool-rule
reminder and a topic-area scope. Subagents *cannot* download to shared
state without coordination — they write to area-scoped subdirs and report
back, then the orchestrator merges into `papers.json`.

### Phase 3 — Historical/technical backfill

- Trace 2017 → current progression. Fill bridges where a topic is "modern
  state" but the path from prior art is missing.
- Build `kb/index/timeline.md`.
- **Stop when:** no "how did we get from X to Y?" question lacks a
  documented bridge.

### Phase 4 — Cross-link + lint + contradiction surface

- Verify every claim has a citation; every citation resolves.
- Glossary covers every glossary-worthy term.
- All analogies/intuitions are tagged.
- Surface places where sources disagree as `[CONTRADICTION]` markers with
  both sides cited.
- Optional: evaluate `adversarial-research:kb-lint` skill for fit. Likely
  not the right tool (designed for argument-graph contested-claim KBs;
  ours is canon-survey), but a 10-minute look is cheap.

### Phase 5 — LaTeX series outline (deferred)

After KB is stable: a separate brainstorming round drafts the multi-paper
LaTeX series structure, producing N paper specs each with their own
implementation plan. Deferred until breadth is observed in the KB.

## 7. Topic taxonomy (v0)

This is the starting hypothesis. Phase 1 will refine it. Each leaf becomes
a `kb/notes/<area>/<topic>.md` file.

```
architecture/
├── transformer-overview            # encoder-dec vs dec-only, parallel-attn (PaLM)
├── tokenization                    # BPE, WordPiece, SentencePiece, byte-level, tokenizer-free
├── embeddings-and-tying            # token + positional, tied vs untied
├── position-encoding               # sinusoidal, learned, T5 bias, RoPE, ALiBi, NoPE, YaRN/LongRoPE
├── attention-mechanism             # SDPA, MHA, MQA, GQA, MLA, FlashAttn 1/2/3, ring, sliding, NSA
├── ffn-and-moe                     # ReLU/GELU/SwiGLU; Switch, Mixtral, DeepSeek MoE, fine-grained
├── normalization                   # LayerNorm, RMSNorm, Pre/Post-LN, DeepNorm
├── state-space-models              # S4, Mamba, Mamba-2; hybrids (Jamba, Samba, Zamba2, Hymba); RWKV
├── long-context                    # context extension, position interpolation, ring attn, infini-attn
└── multimodal-llm-extensions       # vision encoder + projector + LLM body; native multimodal

training/
├── pre-training-data               # The Pile → RedPajama → RefinedWeb → FineWeb (Edu); dedup, filtering
├── synthetic-data-and-distillation # model-gen data, distillation, rejection sampling, weak-to-strong
├── optimization                    # AdamW, Lion, Sophia, Shampoo, Muon, schedules
├── distributed-training            # DP, ZeRO/FSDP, TP, PP, SP, EP, 3D
├── mixed-precision-and-stability   # FP16/BF16/FP8, loss scaling, instability mitigation
└── continued-pretraining           # mid-training, domain adapt, model merging

post-training/
├── sft                             # FLAN, T0, instruction tuning datasets
├── rlhf                            # InstructGPT, RM, PPO, reward hacking
├── dpo-and-offline                 # DPO, IPO, KTO, ORPO, SimPO
├── rlaif-and-constitutional        # CAI, RLAIF
└── rlvr-and-grpo                   # verifiable rewards, GRPO

reasoning/
├── chain-of-thought                # Wei 2022, self-consistency, ToT, GoT
├── process-supervision             # PRM800K, step-level reward models
├── test-time-compute               # o1/R1-style sampling, deliberate reasoning
├── reasoning-training              # RL for reasoning, distilled reasoners
└── inference-time-search           # MCTS, verifier-guided search

inference/
├── kv-cache-management             # paged attn (vLLM), prefix caching, KV compression
├── quantization                    # GPTQ, AWQ, SmoothQuant, FP8, BitNet b1.58
├── speculative-decoding            # vanilla, EAGLE, Medusa, Lookahead
├── serving-systems                 # vLLM, SGLang, TRT-LLM, llama.cpp; throughput/latency tradeoffs
└── structured-output               # JSON-mode, grammars, function-calling

scaling/
├── kaplan-laws                     # 2020 power laws
├── chinchilla                      # 2022 compute-optimal
├── mu-transfer                     # μP, Tensor Programs
└── scaling-frontier                # data-constrained, repetitions, current updates

interpretability/
├── lens-techniques                 # logit lens, tuned lens
├── mechanistic-interpretability    # induction heads, circuits, IOI
├── sparse-autoencoders             # Anthropic SAEs, monosemanticity
├── activation-patching             # causal interventions, attribution/path patching
└── probing                         # linear probes

evaluation/
├── knowledge-benchmarks            # MMLU, MMLU-Pro, GPQA, FrontierMath
├── reasoning-benchmarks            # ARC-AGI, BBH, GSM8K, MATH, AIME
├── agentic-benchmarks              # AgentBench, SWE-bench, OSWorld, GAIA
└── eval-methodology                # contamination, log-prob vs gen, lm-eval-harness

alignment/
├── safety-evaluation               # red-teaming, jailbreaking, refusal
├── watermarking-and-provenance     # Kirchenbauer, C2PA-adjacent
├── oversight-and-scalable-alignment# debate, RRM, weak-to-strong
└── sycophancy-and-deception        # measurement, mitigation, strategic deception
```

**Total:** 9 areas, ~50 leaf topics. Phase 1 will likely add/remove leaves.

## 8. Source provenance tiers

| Tier | Sources | Can solely back hard claims? | Storage |
|------|---------|----------------------------|---------|
| **A — Canonical** | arxiv preprints, peer-reviewed venues, official tech reports / model cards (Llama, Qwen, DeepSeek, GPT-4 system card, Claude/Gemini model cards), reference github repos | **Yes** | `theory/sources/papers/<key>.pdf` + `theory/kb/excerpts/<key>.md` |
| **B — High-signal commentary** | Vendor research blogs (Anthropic / OpenAI / DeepMind / HF), respected lab blogs (EleutherAI / Databricks-MosaicML / Together / Anyscale), individual respected researchers (Dettmers, Rush, Alammar, Lilian Weng) | **Only if** the underlying tier-A source is also cited | Optional snapshot to `theory/sources/forums/` if uniquely informative |
| **C — Community signal** | r/LocalLLaMA, r/MachineLearning, HuggingFace community, X/Twitter (named researchers preferred), HN comment threads | **No** — discovery only | Selective: only when a thread materially shaped a synthesis. Tagged `[FORUM-SIGNAL]` for context, not authority |

Selective forum archival format (when used):

```markdown
---
url: https://reddit.com/r/LocalLLaMA/comments/...
captured: YYYY-MM-DD
informed-notes: [<area>/<topic>, ...]
tier: C
---

# [Thread title]

[author, flair, post date]

> [verbatim, blockquoted excerpt]

**Why captured:** Surfaced X claim that I then verified against [paper-key].
The thread is *not* the citation; the paper is. This is here for provenance
of the discovery path.
```

## 9. Writing-style operationalization (Feynman bar)

Every topic note follows this shape:

1. **Formal definition** — full math; every variable defined under the
   equation; cite paper section/equation.
2. **Mechanism** — describe what the math computes step-by-step at runtime.
   Tensor shapes belong here.
3. **Variants and lineage** — cited list of historical and modern variants.
4. **`[INTUITION]` / `[ANALOGY]`** — tagged paragraph(s). Analogies must
   always return to the canonical symbolic form. Never force an awkward
   analogy; if intuition is best delivered by re-describing the mechanism
   in plain language, do that instead and tag `[INTUITION]`.
5. **Frontier and open questions** — recent papers, contested territory,
   `[CONTRADICTION]` markers where sources disagree.
6. **See also** — cross-links to related notes.

For LaTeX papers (Phase 5+): same structure with custom environments —
`\begin{intuitionbox}`, `\begin{analogybox}` distinct from the existing
`\begin{implbox}` and `\begin{evobox}`.

## 10. Conventions

### Paper-key naming

- `<lastname-first-author><year>` — e.g. `vaswani2017`, `dao2022`,
  `ainslie2023`.
- Multi-author with no clear first / strong shorthand: use the shorthand —
  `flashattn2`, `mamba2`, `deepseek-v3`.
- Same author/year disambiguated with letter — `dao2023a`, `dao2023b`.
- Tech reports / model cards: vendor-prefix —
  `anthropic-claude-3-model-card`, `openai-gpt4-system-card`,
  `meta-llama3-tech-report`.

### Status fields

- Topic notes (`status:`): `stub` | `draft` | `reviewed` | `locked`.
- Papers (`category:`): `foundational` | `landmark` | `recent` |
  `tech-report` | `superseded`.

### Anchors and cross-links

- Markdown anchors are stable: `kb/notes/architecture/attention-mechanism.md#variants-and-lineage`.
- Cross-links use full paths from `theory/`: `[kb/notes/architecture/attention-mechanism#variants-and-lineage]`.

### READMEs

- `theory/README.md` — top-level navigation: what each subdir is, where to
  start.
- `theory/kb/README.md` — epistemic rules, citation format, analogy
  tagging, status conventions.
- `theory/sources/README.md` — tier definitions, what gets archived where.
- `theory/series/README.md` — placeholder explaining the deferred LaTeX
  series.
- `theory/plans/README.md` — how plans are organized.

## 11. Implementation plan boundary

This spec describes the full target. Implementation will be carried out as
a sequence of phase-scoped plans rather than one mega-plan, since the work
is large enough that a single plan would be unwieldy.

- **Phase 0 plan:** restructure + CLAUDE.md update — *next, via writing-plans skill*.
- **Phase 1 plan:** landscape sweep — written after Phase 0 ships.
- **Phase 2 plans:** one per area (parallelizable) — written after Phase 1
  refines the taxonomy.
- **Phases 3–4 plans:** one each — written when their inputs exist.
- **Phase 5:** separate brainstorming round (deferred).

Self-review gate after each phase: spot-check claims against source PDFs,
verify breadth against external recent surveys, check writing-style
adherence, verify epistemic-tier discipline.
