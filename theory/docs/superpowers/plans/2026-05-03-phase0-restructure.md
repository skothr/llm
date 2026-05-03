# Phase 0: theory/ restructure + CLAUDE.md citation discipline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the existing single-LaTeX-doc theory/ workspace into the KB-substrate layout defined in `theory/docs/superpowers/specs/2026-05-03-theory-expansion-design.md`, and add citation-discipline rules to project `CLAUDE.md`. No research yet — purely structural.

**Architecture:** Mechanical file moves + new directory scaffolding + README authoring. Migration is atomic (single commit) because intermediate states have stale references. Pre-expansion artifacts archived under `theory/archive/2026-05-03-pre-expansion/` for human-legible historical lookup; git history preserves full provenance.

**Tech Stack:** filesystem ops via `mv`/`cp`/`mkdir`, JSON migration via Python one-liner, markdown authoring via Write tool.

---

## File map

**Files created:**
- `theory/README.md`
- `theory/kb/README.md`
- `theory/kb/glossary.md` (seed copy of current GLOSSARY.md content)
- `theory/kb/index/papers.json` (migrated + schema-expanded from `theory/sources/index.json`)
- `theory/kb/index/topics.md` (taxonomy v0 from spec §7)
- `theory/kb/index/timeline.md` (stub — populated in Phase 3)
- `theory/kb/notes/architecture/.gitkeep` (and 8 sibling .gitkeep files for empty area dirs)
- `theory/kb/excerpts/.gitkeep`
- `theory/sources/README.md`
- `theory/sources/forums/.gitkeep`
- `theory/series/README.md`
- `theory/plans/README.md`
- `theory/archive/2026-05-03-pre-expansion/` (snapshot dir)

**Files moved:**
- `theory/build/llm-core-architecture/llm-core-architecture.tex` → `theory/archive/2026-05-03-pre-expansion/build/llm-core-architecture/llm-core-architecture.tex`
- `theory/llm-core-architecture.pdf` → `theory/archive/2026-05-03-pre-expansion/llm-core-architecture.pdf`
- `theory/GLOSSARY.md` → `theory/archive/2026-05-03-pre-expansion/GLOSSARY.md`
- `theory/visuals/llm-architecture-diagram.html` → `theory/archive/2026-05-03-pre-expansion/visuals/llm-architecture-diagram.html`
- `theory/Makefile` → `theory/archive/2026-05-03-pre-expansion/Makefile`
- `theory/sources/index.json` → DELETED (content migrated to `theory/kb/index/papers.json`)

**Files modified:**
- `CLAUDE.md` — replace the now-stale `# Theory Document` section; add `# Theory KB & citation discipline` section. **Do not touch the user's pending C++ GUI section** (lines ~204–233 of current working copy).

**Files NOT to touch:**
- `docs/skills/` (untracked, user's separate work)
- `learn/` (untracked, user's separate work)
- `pyproject.toml` (untracked, user's separate work)
- `testing/gui_cpp/` (untracked, user's separate work)
- `theory/sources/papers/` and its contents (PDFs preserved in place)
- `theory/archive/llm-core-architecture-2026-04-11.pdf` and `theory/archive/llm-core-architecture-2026-04-16.pdf` (existing dated archives; leave at archive root)
- `theory/docs/superpowers/specs/2026-04-06-llm-core-architecture-design.md` (existing v1 spec; leave in place — it's now historical context for v2)
- `theory/docs/superpowers/specs/2026-05-03-theory-expansion-design.md` (this plan's spec; already committed)

---

## Task 1: Pre-flight verification

**Files:** none (read-only checks)

- [ ] **Step 1: Confirm current working directory**

```bash
pwd
```
Expected: `/home/ai/ai-projects/llm`

- [ ] **Step 2: Confirm only-expected uncommitted changes**

```bash
git status --short
```
Expected (order may vary):
```
 M CLAUDE.md
?? docs/skills/
?? learn/
?? pyproject.toml
?? testing/gui_cpp/
```
Stop and ask the user if anything else appears — Phase 0 must not stomp on unrelated dirty state.

- [ ] **Step 3: Confirm v2 spec is committed**

```bash
git log --oneline -1 -- theory/docs/superpowers/specs/2026-05-03-theory-expansion-design.md
```
Expected: a commit hash + message starting with `docs(theory): spec for KB-substrate expansion`. If empty, stop — the spec must exist before its plan executes.

- [ ] **Step 4: Confirm artifacts to be moved exist**

```bash
ls theory/build/llm-core-architecture/llm-core-architecture.tex
ls theory/llm-core-architecture.pdf
ls theory/GLOSSARY.md
ls theory/visuals/llm-architecture-diagram.html
ls theory/Makefile
ls theory/sources/index.json
```
All six lines must succeed. If any fail, stop and resync with the user.

---

## Task 2: Create archive snapshot directory

**Files:**
- Create: `theory/archive/2026-05-03-pre-expansion/`
- Create: `theory/archive/2026-05-03-pre-expansion/build/llm-core-architecture/`
- Create: `theory/archive/2026-05-03-pre-expansion/visuals/`

- [ ] **Step 1: Make the snapshot tree**

```bash
mkdir -p theory/archive/2026-05-03-pre-expansion/build/llm-core-architecture
mkdir -p theory/archive/2026-05-03-pre-expansion/visuals
```

- [ ] **Step 2: Verify the dirs exist and are empty**

```bash
ls -la theory/archive/2026-05-03-pre-expansion/
ls -la theory/archive/2026-05-03-pre-expansion/build/llm-core-architecture/
ls -la theory/archive/2026-05-03-pre-expansion/visuals/
```
Expected: each directory listed; `build/llm-core-architecture/` and `visuals/` show only `.` and `..` entries.

---

## Task 3: Move pre-expansion artifacts into snapshot

**Files moved:**
- `theory/build/llm-core-architecture/llm-core-architecture.tex` → snapshot
- `theory/llm-core-architecture.pdf` → snapshot
- `theory/GLOSSARY.md` → snapshot
- `theory/visuals/llm-architecture-diagram.html` → snapshot
- `theory/Makefile` → snapshot

- [ ] **Step 1: Move the LaTeX source**

```bash
git mv theory/build/llm-core-architecture/llm-core-architecture.tex \
       theory/archive/2026-05-03-pre-expansion/build/llm-core-architecture/llm-core-architecture.tex
```

- [ ] **Step 2: Move the compiled PDF**

```bash
git mv theory/llm-core-architecture.pdf \
       theory/archive/2026-05-03-pre-expansion/llm-core-architecture.pdf
```

- [ ] **Step 3: Move the glossary**

```bash
git mv theory/GLOSSARY.md \
       theory/archive/2026-05-03-pre-expansion/GLOSSARY.md
```

- [ ] **Step 4: Move the HTML visual**

```bash
git mv theory/visuals/llm-architecture-diagram.html \
       theory/archive/2026-05-03-pre-expansion/visuals/llm-architecture-diagram.html
```

- [ ] **Step 5: Move the Makefile**

```bash
git mv theory/Makefile \
       theory/archive/2026-05-03-pre-expansion/Makefile
```

- [ ] **Step 6: Remove now-empty source directories**

```bash
rmdir theory/build/llm-core-architecture
rmdir theory/build
rmdir theory/visuals
```
Expected: silent success. If `rmdir` complains, the directories aren't empty — stop and inspect.

- [ ] **Step 7: Verify snapshot has all five artifacts**

```bash
find theory/archive/2026-05-03-pre-expansion -type f | sort
```
Expected output (5 files):
```
theory/archive/2026-05-03-pre-expansion/GLOSSARY.md
theory/archive/2026-05-03-pre-expansion/Makefile
theory/archive/2026-05-03-pre-expansion/build/llm-core-architecture/llm-core-architecture.tex
theory/archive/2026-05-03-pre-expansion/llm-core-architecture.pdf
theory/archive/2026-05-03-pre-expansion/visuals/llm-architecture-diagram.html
```

- [ ] **Step 8: Verify the original locations are gone**

```bash
[ ! -e theory/build ] && echo "build gone" || echo "build STILL EXISTS"
[ ! -e theory/visuals ] && echo "visuals gone" || echo "visuals STILL EXISTS"
[ ! -e theory/llm-core-architecture.pdf ] && echo "pdf gone" || echo "pdf STILL EXISTS"
[ ! -e theory/GLOSSARY.md ] && echo "glossary gone" || echo "glossary STILL EXISTS"
[ ! -e theory/Makefile ] && echo "makefile gone" || echo "makefile STILL EXISTS"
```
Expected: all five lines say "X gone".

---

## Task 4: Scaffold new KB directory tree

**Files:**
- Create: `theory/kb/` and subdirs
- Create: `theory/kb/notes/{architecture,training,post-training,reasoning,inference,scaling,interpretability,evaluation,alignment}/`
- Create: `theory/kb/excerpts/`
- Create: `theory/kb/index/`
- Create: `theory/sources/forums/`
- Create: `theory/plans/`
- Create: `theory/series/`

- [ ] **Step 1: Create the full tree in one shot**

```bash
mkdir -p theory/kb/notes/architecture
mkdir -p theory/kb/notes/training
mkdir -p theory/kb/notes/post-training
mkdir -p theory/kb/notes/reasoning
mkdir -p theory/kb/notes/inference
mkdir -p theory/kb/notes/scaling
mkdir -p theory/kb/notes/interpretability
mkdir -p theory/kb/notes/evaluation
mkdir -p theory/kb/notes/alignment
mkdir -p theory/kb/excerpts
mkdir -p theory/kb/index
mkdir -p theory/sources/forums
mkdir -p theory/plans
mkdir -p theory/series
```

- [ ] **Step 2: Add .gitkeep files so empty dirs survive in git**

```bash
touch theory/kb/notes/architecture/.gitkeep
touch theory/kb/notes/training/.gitkeep
touch theory/kb/notes/post-training/.gitkeep
touch theory/kb/notes/reasoning/.gitkeep
touch theory/kb/notes/inference/.gitkeep
touch theory/kb/notes/scaling/.gitkeep
touch theory/kb/notes/interpretability/.gitkeep
touch theory/kb/notes/evaluation/.gitkeep
touch theory/kb/notes/alignment/.gitkeep
touch theory/kb/excerpts/.gitkeep
touch theory/sources/forums/.gitkeep
```

- [ ] **Step 3: Verify the tree**

```bash
find theory/kb theory/sources/forums theory/plans theory/series -type d | sort
```
Expected (16 directories listed):
```
theory/kb
theory/kb/excerpts
theory/kb/index
theory/kb/notes
theory/kb/notes/alignment
theory/kb/notes/architecture
theory/kb/notes/evaluation
theory/kb/notes/inference
theory/kb/notes/interpretability
theory/kb/notes/post-training
theory/kb/notes/reasoning
theory/kb/notes/scaling
theory/kb/notes/training
theory/plans
theory/series
theory/sources/forums
```
Verifiable as a count: `find theory/kb theory/sources/forums theory/plans theory/series -type d | wc -l` should print `16`.

---

## Task 5: Migrate sources/index.json → kb/index/papers.json with expanded schema

**Files:**
- Create: `theory/kb/index/papers.json`
- Delete: `theory/sources/index.json`

The new schema adds `venue`, `excerpts_file`, `notes_referenced_by`, `topics`, `category` fields. For the 12 existing papers, populate what we know; leave Phase 1 to fill `excerpts_file`/`notes_referenced_by` as notes are written.

- [ ] **Step 1: Write the migrated papers.json**

Write the file at `theory/kb/index/papers.json`:

```json
{
  "description": "Master index of canonical sources for the LLM theory KB. Schema: paper-key -> metadata + KB cross-references. excerpts_file and notes_referenced_by are populated as KB content is written.",
  "schema_version": "2026-05-03",
  "papers": [
    {
      "key": "vaswani2017",
      "title": "Attention Is All You Need",
      "authors": "Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin",
      "year": 2017,
      "venue": "NeurIPS",
      "url": "https://arxiv.org/abs/1706.03762",
      "local_file": "sources/papers/vaswani2017_attention_is_all_you_need.pdf",
      "excerpts_file": null,
      "notes_referenced_by": [],
      "topics": ["attention", "transformer", "encoder-decoder", "positional-encoding"],
      "category": "foundational",
      "summary": "Introduces the Transformer: encoder-decoder architecture based entirely on attention, replacing RNNs. Defines scaled dot-product attention, multi-head attention, sinusoidal positional encoding, position-wise FFN. Base model: d=512, h=8, N=6, d_ff=2048."
    },
    {
      "key": "radford2018",
      "title": "Improving Language Understanding by Generative Pre-Training",
      "authors": "Radford, Narasimhan, Salimans, Sutskever",
      "year": 2018,
      "venue": "OpenAI Tech Report",
      "url": "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf",
      "local_file": "sources/papers/radford2018_gpt1.pdf",
      "excerpts_file": null,
      "notes_referenced_by": [],
      "topics": ["transformer", "decoder-only", "pretraining", "tokenization"],
      "category": "foundational",
      "summary": "GPT-1: first decoder-only transformer for language modeling. 12 layers, d=768, 12 heads, d_ff=3072. Uses learned positional embeddings, GELU activation, BPE tokenization. Introduces generative pre-training + discriminative fine-tuning paradigm."
    },
    {
      "key": "brown2020",
      "title": "Language Models are Few-Shot Learners",
      "authors": "Brown, Mann, Ryder, et al.",
      "year": 2020,
      "venue": "NeurIPS",
      "url": "https://arxiv.org/abs/2005.14165",
      "local_file": "sources/papers/brown2020_gpt3.pdf",
      "excerpts_file": null,
      "notes_referenced_by": [],
      "topics": ["scaling", "in-context-learning", "decoder-only"],
      "category": "foundational",
      "summary": "GPT-3: scales decoder-only transformer to 175B parameters. Demonstrates few-shot learning via in-context examples. Establishes scaling as a primary driver of capability."
    },
    {
      "key": "touvron2023",
      "title": "LLaMA: Open and Efficient Foundation Language Models",
      "authors": "Touvron, Lavril, Izacard, et al.",
      "year": 2023,
      "venue": "Meta Tech Report",
      "url": "https://arxiv.org/abs/2302.13971",
      "local_file": "sources/papers/touvron2023_llama.pdf",
      "excerpts_file": null,
      "notes_referenced_by": [],
      "topics": ["decoder-only", "rope", "swiglu", "rmsnorm", "open-weights"],
      "category": "landmark",
      "summary": "Modern decoder-only architecture with three key modifications to original Transformer: pre-normalization with RMSNorm, SwiGLU activation (dim 2/3 * 4d), RoPE at every layer. Model sizes 7B-65B on public data."
    },
    {
      "key": "sennrich2016",
      "title": "Neural Machine Translation of Rare Words with Subword Units",
      "authors": "Sennrich, Haddow, Birch",
      "year": 2016,
      "venue": "ACL",
      "url": "https://arxiv.org/abs/1508.07909",
      "local_file": "sources/papers/sennrich2016_bpe.pdf",
      "excerpts_file": null,
      "notes_referenced_by": [],
      "topics": ["tokenization", "bpe"],
      "category": "foundational",
      "summary": "Introduces BPE for NLP tokenization. Algorithm: start with character vocabulary, iteratively merge most frequent adjacent pairs. Produces fixed-size vocabulary of variable-length subword units."
    },
    {
      "key": "su2021",
      "title": "RoFormer: Enhanced Transformer with Rotary Position Embedding",
      "authors": "Su, Lu, Pan, Wen, Liu",
      "year": 2021,
      "venue": "arXiv",
      "url": "https://arxiv.org/abs/2104.09864",
      "local_file": "sources/papers/su2021_rope.pdf",
      "excerpts_file": null,
      "notes_referenced_by": [],
      "topics": ["positional-encoding", "rope", "relative-position"],
      "category": "landmark",
      "summary": "Rotary Position Embedding (RoPE): encodes position by rotating Q and K vectors via rotation matrix. Dot product between Q_m and K_n depends only on relative position (m-n). Properties: sequence length flexibility, decaying dependency with distance."
    },
    {
      "key": "shazeer2020",
      "title": "GLU Variants Improve Transformer",
      "authors": "Shazeer",
      "year": 2020,
      "venue": "arXiv",
      "url": "https://arxiv.org/abs/2002.05202",
      "local_file": "sources/papers/shazeer2020_glu_variants.pdf",
      "excerpts_file": null,
      "notes_referenced_by": [],
      "topics": ["ffn", "swiglu", "geglu", "activations"],
      "category": "landmark",
      "summary": "Tests GLU variants in Transformer FFN. SwiGLU: (Swish(xW) ⊗ xV)W2. GLU variants use 3 weight matrices instead of 2; reduce hidden dim by 2/3 to match parameter count. GEGLU and SwiGLU achieve best perplexity."
    },
    {
      "key": "xiong2020",
      "title": "On Layer Normalization in the Transformer Architecture",
      "authors": "Xiong, Yang, He, et al.",
      "year": 2020,
      "venue": "ICML",
      "url": "https://arxiv.org/abs/2002.04745",
      "local_file": "sources/papers/xiong2020_layer_norm.pdf",
      "excerpts_file": null,
      "notes_referenced_by": [],
      "topics": ["normalization", "pre-ln", "post-ln", "training-stability"],
      "category": "landmark",
      "summary": "Analyzes Post-LN (sublayer→add→norm) vs Pre-LN (norm→sublayer→add). Proves via mean-field theory that Post-LN gradient scale is O(d√(ln d)) independent of depth, while Pre-LN scales as O(d√(ln d / L)). Pre-LN enables training without warmup."
    },
    {
      "key": "zhang2019",
      "title": "Root Mean Square Layer Normalization",
      "authors": "Zhang, Sennrich",
      "year": 2019,
      "venue": "NeurIPS",
      "url": "https://arxiv.org/abs/1910.07467",
      "local_file": "sources/papers/zhang2019_rmsnorm.pdf",
      "excerpts_file": null,
      "notes_referenced_by": [],
      "topics": ["normalization", "rmsnorm"],
      "category": "landmark",
      "summary": "RMSNorm: simplifies LayerNorm by removing mean-centering, keeping only RMS re-scaling. Comparable performance with 7-64% faster runtime. Used by LLaMA instead of standard LayerNorm."
    },
    {
      "key": "chowdhery2022",
      "title": "PaLM: Scaling Language Modeling with Pathways",
      "authors": "Chowdhery, Narang, Devlin, et al.",
      "year": 2022,
      "venue": "JMLR",
      "url": "https://arxiv.org/abs/2204.02311",
      "local_file": null,
      "excerpts_file": null,
      "notes_referenced_by": [],
      "topics": ["scaling", "swiglu", "rope", "parallel-attn-ffn", "mqa"],
      "category": "landmark",
      "summary": "PaLM: 540B-parameter decoder-only Transformer trained on Pathways. Architecture choices: SwiGLU activation, RoPE, parallel attention/FFN layers, multi-query attention for inference efficiency."
    },
    {
      "key": "jiang2023",
      "title": "Mistral 7B",
      "authors": "Jiang, Sablayrolles, Mensch, et al.",
      "year": 2023,
      "venue": "arXiv (Mistral AI Tech Report)",
      "url": "https://arxiv.org/abs/2310.06825",
      "local_file": null,
      "excerpts_file": null,
      "notes_referenced_by": [],
      "topics": ["decoder-only", "gqa", "sliding-window-attention", "open-weights"],
      "category": "landmark",
      "summary": "Mistral 7B: open-weights decoder-only LLM with grouped-query attention (GQA) and sliding-window attention. Outperforms LLaMA 2 13B on multiple benchmarks at smaller size. Foundational reference for the modern compact-LLM architecture family."
    },
    {
      "key": "raffel2019",
      "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
      "authors": "Raffel, Shazeer, Roberts, et al.",
      "year": 2019,
      "venue": "JMLR",
      "url": "https://arxiv.org/abs/1910.10683",
      "local_file": null,
      "excerpts_file": null,
      "notes_referenced_by": [],
      "topics": ["t5", "encoder-decoder", "sentencepiece", "relative-position-bias"],
      "category": "foundational",
      "summary": "T5: encoder-decoder Transformer that casts every NLP task as text-to-text. Contributes several conventions reused across the field (SentencePiece, relative positional bias)."
    }
  ]
}
```

- [ ] **Step 2: Verify the JSON parses**

```bash
python3 -c "import json; d=json.load(open('theory/kb/index/papers.json')); print(f'{len(d[\"papers\"])} papers, schema_version={d[\"schema_version\"]}')"
```
Expected: `12 papers, schema_version=2026-05-03`

- [ ] **Step 3: Verify paper-key uniqueness**

```bash
python3 -c "import json; keys=[p['key'] for p in json.load(open('theory/kb/index/papers.json'))['papers']]; assert len(keys)==len(set(keys)), 'duplicate keys'; print('all keys unique')"
```
Expected: `all keys unique`

- [ ] **Step 4: Remove the old index**

```bash
git rm theory/sources/index.json
```

- [ ] **Step 5: Verify old index gone**

```bash
[ ! -e theory/sources/index.json ] && echo "old index gone" || echo "STILL THERE"
```
Expected: `old index gone`

---

## Task 6: Seed kb/glossary.md

**Files:**
- Create: `theory/kb/glossary.md`

The current `theory/GLOSSARY.md` has been moved to the archive (Task 3, Step 3). For Phase 0 we copy its content as the seed for the new glossary — Phase 2 will add new terms as topics get researched.

- [ ] **Step 1: Copy archived glossary content as the seed**

```bash
cp theory/archive/2026-05-03-pre-expansion/GLOSSARY.md theory/kb/glossary.md
```

- [ ] **Step 2: Replace the seed file's first line with a header noting it's the active glossary**

Use the Edit tool on `theory/kb/glossary.md`:

```
old_string: # GLOSSARY

Every technical term used in this workspace, defined concisely.
new_string: # Glossary

Every technical term used in the LLM theory KB, defined concisely. Each term
should cite a paper-key from `kb/index/papers.json` for traceability where
applicable. Terms are added during Phase 2 (per-topic research) and reviewed
during Phase 4 (lint).

This glossary was seeded from `archive/2026-05-03-pre-expansion/GLOSSARY.md`
on 2026-05-03; entries from that seed have not yet been re-cited and are
treated as `status: pre-expansion-seed` until re-validated during Phase 2.
```

- [ ] **Step 3: Verify the file**

```bash
head -10 theory/kb/glossary.md
wc -l theory/kb/glossary.md
```
Expected: header reflects the new opening; line count ≈ 100 (slightly more than the 91-line original due to the longer header).

---

## Task 7: Write kb/index/topics.md (taxonomy v0)

**Files:**
- Create: `theory/kb/index/topics.md`

Content reflects spec §7. Each leaf is annotated with `status: pending`. The file is the authoritative list of which `kb/notes/<area>/<topic>.md` files exist or will exist.

- [ ] **Step 1: Write topics.md**

Write `theory/kb/index/topics.md` with this content:

````markdown
# KB Topic Graph (v0 — taxonomy hypothesis)

This is the starting taxonomy for the KB. It will be refined during Phase 1
(landscape sweep) as actual current-state research surfaces topics that
should be added, merged, or split.

## Status legend

- `pending` — not yet researched
- `stub` — note file created with placeholder
- `draft` — first pass written, not reviewed
- `reviewed` — claims spot-checked, citations verified
- `locked` — frozen for the current research cycle

## Areas

### architecture/

| Topic | Status | Note path |
|-------|--------|-----------|
| transformer-overview | pending | `kb/notes/architecture/transformer-overview.md` |
| tokenization | pending | `kb/notes/architecture/tokenization.md` |
| embeddings-and-tying | pending | `kb/notes/architecture/embeddings-and-tying.md` |
| position-encoding | pending | `kb/notes/architecture/position-encoding.md` |
| attention-mechanism | pending | `kb/notes/architecture/attention-mechanism.md` |
| ffn-and-moe | pending | `kb/notes/architecture/ffn-and-moe.md` |
| normalization | pending | `kb/notes/architecture/normalization.md` |
| state-space-models | pending | `kb/notes/architecture/state-space-models.md` |
| long-context | pending | `kb/notes/architecture/long-context.md` |
| multimodal-llm-extensions | pending | `kb/notes/architecture/multimodal-llm-extensions.md` |

### training/

| Topic | Status | Note path |
|-------|--------|-----------|
| pre-training-data | pending | `kb/notes/training/pre-training-data.md` |
| synthetic-data-and-distillation | pending | `kb/notes/training/synthetic-data-and-distillation.md` |
| optimization | pending | `kb/notes/training/optimization.md` |
| distributed-training | pending | `kb/notes/training/distributed-training.md` |
| mixed-precision-and-stability | pending | `kb/notes/training/mixed-precision-and-stability.md` |
| continued-pretraining | pending | `kb/notes/training/continued-pretraining.md` |

### post-training/

| Topic | Status | Note path |
|-------|--------|-----------|
| sft | pending | `kb/notes/post-training/sft.md` |
| rlhf | pending | `kb/notes/post-training/rlhf.md` |
| dpo-and-offline | pending | `kb/notes/post-training/dpo-and-offline.md` |
| rlaif-and-constitutional | pending | `kb/notes/post-training/rlaif-and-constitutional.md` |
| rlvr-and-grpo | pending | `kb/notes/post-training/rlvr-and-grpo.md` |

### reasoning/

| Topic | Status | Note path |
|-------|--------|-----------|
| chain-of-thought | pending | `kb/notes/reasoning/chain-of-thought.md` |
| process-supervision | pending | `kb/notes/reasoning/process-supervision.md` |
| test-time-compute | pending | `kb/notes/reasoning/test-time-compute.md` |
| reasoning-training | pending | `kb/notes/reasoning/reasoning-training.md` |
| inference-time-search | pending | `kb/notes/reasoning/inference-time-search.md` |

### inference/

| Topic | Status | Note path |
|-------|--------|-----------|
| kv-cache-management | pending | `kb/notes/inference/kv-cache-management.md` |
| quantization | pending | `kb/notes/inference/quantization.md` |
| speculative-decoding | pending | `kb/notes/inference/speculative-decoding.md` |
| serving-systems | pending | `kb/notes/inference/serving-systems.md` |
| structured-output | pending | `kb/notes/inference/structured-output.md` |

### scaling/

| Topic | Status | Note path |
|-------|--------|-----------|
| kaplan-laws | pending | `kb/notes/scaling/kaplan-laws.md` |
| chinchilla | pending | `kb/notes/scaling/chinchilla.md` |
| mu-transfer | pending | `kb/notes/scaling/mu-transfer.md` |
| scaling-frontier | pending | `kb/notes/scaling/scaling-frontier.md` |

### interpretability/

| Topic | Status | Note path |
|-------|--------|-----------|
| lens-techniques | pending | `kb/notes/interpretability/lens-techniques.md` |
| mechanistic-interpretability | pending | `kb/notes/interpretability/mechanistic-interpretability.md` |
| sparse-autoencoders | pending | `kb/notes/interpretability/sparse-autoencoders.md` |
| activation-patching | pending | `kb/notes/interpretability/activation-patching.md` |
| probing | pending | `kb/notes/interpretability/probing.md` |

### evaluation/

| Topic | Status | Note path |
|-------|--------|-----------|
| knowledge-benchmarks | pending | `kb/notes/evaluation/knowledge-benchmarks.md` |
| reasoning-benchmarks | pending | `kb/notes/evaluation/reasoning-benchmarks.md` |
| agentic-benchmarks | pending | `kb/notes/evaluation/agentic-benchmarks.md` |
| eval-methodology | pending | `kb/notes/evaluation/eval-methodology.md` |

### alignment/

| Topic | Status | Note path |
|-------|--------|-----------|
| safety-evaluation | pending | `kb/notes/alignment/safety-evaluation.md` |
| watermarking-and-provenance | pending | `kb/notes/alignment/watermarking-and-provenance.md` |
| oversight-and-scalable-alignment | pending | `kb/notes/alignment/oversight-and-scalable-alignment.md` |
| sycophancy-and-deception | pending | `kb/notes/alignment/sycophancy-and-deception.md` |

## Total

- 9 areas
- 48 leaf topics

Phase 1 will refine this taxonomy before Phase 2 begins per-topic research.
````

- [ ] **Step 2: Verify file written and total topic count matches expectation**

```bash
grep -c "^| " theory/kb/index/topics.md
```
Expected: `57` (48 data rows + 9 header rows).

```bash
grep -c "| pending |" theory/kb/index/topics.md
```
Expected: `48`

---

## Task 8: Write kb/index/timeline.md (stub)

**Files:**
- Create: `theory/kb/index/timeline.md`

Stub file. Phase 3 (historical backfill) populates it.

- [ ] **Step 1: Write the stub**

Write `theory/kb/index/timeline.md` with:

```markdown
# LLM Architecture Timeline

> **Status:** stub — populated during Phase 3 (historical/technical backfill).

This file traces the chronological progression of LLM theory and practice
from 2017 (Transformer) through bleeding edge. Each entry cites one or more
papers from `kb/index/papers.json`.

## Format

Each entry is an architectural or paradigm shift with:

- **Year:** YYYY
- **Shift:** one-line description
- **Sources:** [paper-key, paper-key, ...]
- **Note:** brief framing; full discussion lives in the relevant
  `kb/notes/<area>/<topic>.md` file.

## Entries

(Populated in Phase 3.)
```

- [ ] **Step 2: Verify**

```bash
wc -l theory/kb/index/timeline.md
```
Expected: ~17 lines.

---

## Task 9: Write theory/README.md (top-level navigation)

**Files:**
- Create: `theory/README.md`

- [ ] **Step 1: Write the README**

Write `theory/README.md` with:

````markdown
# theory/

LLM theoretical-framework workspace. Two-layered:

1. **`kb/`** — knowledge base. Modular, citation-grounded notes plus
   verbatim source excerpts plus structured indices. Source of truth for
   technical claims throughout this project.
2. **`series/`** — placeholder for a future multi-paper LaTeX series,
   outlined only after the KB is complete (see
   `docs/superpowers/specs/2026-05-03-theory-expansion-design.md` §11).

## Layout

```
theory/
├── kb/                # the knowledge base
│   ├── notes/         # digested synthesis, one file per topic
│   ├── excerpts/      # verbatim quoted passages from papers
│   ├── index/         # papers.json, topics.md, timeline.md
│   └── glossary.md
├── sources/           # primary source PDFs + selectively archived forum threads
├── plans/             # phase-scoped research/construction plans
├── archive/           # historical snapshots (pre-expansion v1, dated PDFs)
├── series/            # placeholder for future LaTeX paper series (Phase 5+)
└── docs/superpowers/  # specs and plans for the expansion itself
```

## Where to start

- **New to the KB?** Read `kb/README.md` for citation rules and epistemic
  conventions.
- **Looking for a topic?** Check `kb/index/topics.md` for the topic graph and
  per-topic status.
- **Tracing a claim?** Look up the paper-key in `kb/index/papers.json`,
  then read the matching `kb/excerpts/<key>.md` for verbatim source text.
- **Wondering what was here before?** See
  `archive/2026-05-03-pre-expansion/` for the v1 single-LaTeX-doc state.

## Citation discipline

Technical claims about LLM architecture, training, inference, or related
theory must cite either `[paper-key §X]` or
`[kb/notes/<area>/<file>#<anchor>]`. Analogies and intuitions must be tagged
`[ANALOGY]` / `[INTUITION]`. See project `CLAUDE.md` § "Theory KB & citation
discipline" for the full rule.

## Status

- v1 (pre-expansion) — single LaTeX/PDF "core architecture" doc covering
  Transformer 2017 → LLaMA 2023. Archived under
  `archive/2026-05-03-pre-expansion/`.
- v2 (in progress) — KB-substrate expansion to bleeding-edge breadth and
  depth. Phase plans live under `docs/superpowers/plans/`.
````

- [ ] **Step 2: Verify**

```bash
wc -l theory/README.md
head -3 theory/README.md
```
Expected: ~50 lines; first line is `# theory/`.

---

## Task 10: Write theory/kb/README.md (epistemic rules)

**Files:**
- Create: `theory/kb/README.md`

- [ ] **Step 1: Write the README**

Write `theory/kb/README.md` with:

````markdown
# theory/kb/

Knowledge base — modular, citation-grounded synthesis of LLM theory and
practice. Designed so retrieval (RAG or human) cannot confuse digested
synthesis with primary-source quotation.

## Three-tier epistemic separation

| Tier | Directory | What it contains | Can solely back a hard claim? |
|------|-----------|------------------|-------------------------------|
| 1 — Verbatim | `excerpts/<paper-key>.md` | Quoted passages from a paper, anchored to section/equation. Every line in a blockquote. | **Yes** (it's the paper, quoted) |
| 2 — Synthesis | `notes/<area>/<topic>.md` | Digested explanation. Each load-bearing claim cites either a paper or an excerpt anchor. Analogies tagged. | Only when the cited source itself does. |
| 3 — Structure | `index/papers.json`, `index/topics.md`, `index/timeline.md`, `glossary.md` | Indices and cross-references. Not claims, but the navigation map. | N/A |

## Citation format

- `[paper-key §X, eq.Y]` — points to a paper in `index/papers.json`. Examples:
  `[vaswani2017 §3.2, eq.1]`, `[touvron2023 Tab.2]`.
- `[kb/notes/<area>/<file>#<anchor>]` — points to a synthesis note. Use full
  path from `theory/`. Anchor is the markdown heading slug.
- `[kb/excerpts/<paper-key>#<heading>]` — points to a verbatim excerpt.

## Tagging analogies and intuition

Lines or paragraphs that are not formal claims must be prefixed with an
explicit tag so retrieval cannot launder them as fact:

- `[ANALOGY]` — analogical framing (e.g., "Q is the question being asked").
- `[INTUITION]` — re-description of the formal mechanism in plain language.
- `[CONTRADICTION]` — sources disagree on this point; both cited inline.
- `[FORUM-SIGNAL]` — claim shape derived from a community-tier source; full
  discussion lives in the linked `theory/sources/forums/` snapshot. Always
  paired with a tier-A citation that backs the claim itself.
- `[SPECULATION]` — author's own framing not directly grounded in a cited
  source. Use sparingly.

Analogies must always **return** to the canonical symbolic form — never
leave the reader stranded on an analogy that lacks a math anchor.

## Topic note structure

Standard layout for `notes/<area>/<topic>.md`:

```
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
<how the math actually computes — what each term does at runtime;
tensor shapes belong here>

## Variants and lineage
<bulleted list, each with a citation>

## [INTUITION] / [ANALOGY]
<tagged paragraph(s) — must always return to the canonical symbolic form>

## Frontier and open questions
<recent papers, contested territory, [CONTRADICTION] markers>

## See also
<cross-links to related notes>
```

## Status field

`status:` in note frontmatter:

- `stub` — file created, placeholder content only
- `draft` — first pass written, not reviewed
- `reviewed` — claims spot-checked against PDFs, citations verified, no
  internal contradictions
- `locked` — frozen for the current research cycle (e.g., before a LaTeX
  paper writeup)

## Conventions

- **Paper-key format:** `<lastname-first-author><year>` (`vaswani2017`,
  `dao2022`). Letter suffix for same author/year (`dao2023a`, `dao2023b`).
  Shorthand allowed when conventional (`flashattn2`, `mamba2`,
  `deepseek-v3`). Tech reports get vendor prefix
  (`anthropic-claude-3-model-card`).
- **Markdown anchors are stable** — never rename a heading without updating
  every referrer.
- **Glossary updates** when introducing a new technical term — add to
  `glossary.md` with a citation.
````

- [ ] **Step 2: Verify**

```bash
wc -l theory/kb/README.md
grep -c "^##" theory/kb/README.md
```
Expected: ~80 lines; ~7 H2 headings.

---

## Task 11: Write theory/sources/README.md (tier definitions)

**Files:**
- Create: `theory/sources/README.md`

- [ ] **Step 1: Write the README**

Write `theory/sources/README.md` with:

````markdown
# theory/sources/

Primary sources for the KB.

## Layout

- `papers/` — PDF copies of canonical papers, named `{paper-key}_{slug}.pdf`.
  Registered in `kb/index/papers.json`. Source of truth.
- `forums/` — selectively archived blog/forum/thread snapshots. Used as
  discovery-path provenance, **never** as the sole backing for a hard claim.

## Source tiers

| Tier | Sources | Can solely back hard claims? | Storage |
|------|---------|----------------------------|---------|
| **A — Canonical** | arxiv preprints, peer-reviewed venues, official tech reports / model cards (Llama, Qwen, DeepSeek, GPT-4 system card, Claude/Gemini model cards), reference github repos | **Yes** | `papers/<key>.pdf` + `kb/excerpts/<key>.md` |
| **B — High-signal commentary** | Vendor research blogs (Anthropic / OpenAI / DeepMind / HF), respected lab blogs (EleutherAI / Databricks-MosaicML / Together / Anyscale), individual respected researchers (Dettmers, Rush, Alammar, Lilian Weng) | **Only if** the underlying tier-A source is also cited | Optional snapshot to `forums/` if uniquely informative |
| **C — Community signal** | r/LocalLLaMA, r/MachineLearning, HuggingFace community, X/Twitter (named researchers preferred), HN comment threads | **No** — discovery only | Selective: only when a thread materially shaped a synthesis. Tagged `[FORUM-SIGNAL]` for context, not authority |

## Forum/blog archival format

When a tier-B or tier-C source materially shapes a KB note, snapshot it
under `forums/`:

```markdown
---
url: https://reddit.com/r/LocalLLaMA/comments/...
captured: YYYY-MM-DD
informed-notes: [<area>/<topic>, ...]
tier: B | C
---

# [Thread/post title]

[author, date]

> [verbatim, blockquoted excerpt]

**Why captured:** Surfaced X claim that I then verified against [paper-key].
The thread is *not* the citation; the paper is. This is here for provenance
of the discovery path.
```

## Adding a new paper

1. Download PDF to `papers/<paper-key>_<slug>.pdf` (slug is filesystem-safe
   short title).
2. Add an entry to `kb/index/papers.json` with all schema fields. `topics`
   should align with leaves in `kb/index/topics.md`.
3. If the paper is going to be cited from a note, write
   `kb/excerpts/<paper-key>.md` with the relevant verbatim passages and
   set `excerpts_file` in `papers.json`.
````

- [ ] **Step 2: Verify**

```bash
wc -l theory/sources/README.md
```
Expected: ~50 lines.

---

## Task 12: Write theory/series/README.md (placeholder)

**Files:**
- Create: `theory/series/README.md`

- [ ] **Step 1: Write the README**

Write `theory/series/README.md` with:

```markdown
# theory/series/ — LaTeX paper series (deferred)

Placeholder for a future multi-paper LaTeX series. **Outlined only after
the KB (`theory/kb/`) reaches stable coverage** — see
`docs/superpowers/specs/2026-05-03-theory-expansion-design.md` §11 (Phase 5).

The previous single-LaTeX-doc deliverable (`llm-core-architecture`) is
archived under `theory/archive/2026-05-03-pre-expansion/`. Its `.tex` source
is preserved for reference and may be partially reused once the series is
outlined.

## Why deferred

Topic structure for the series should follow from observed research breadth
in the KB, not from priors held before research begins. Outlining now would
fix decisions on stale information.

## Until then

This directory stays empty (apart from this README). Phase 5 brainstorming
will produce the series outline + per-paper specs.
```

- [ ] **Step 2: Verify**

```bash
wc -l theory/series/README.md
```
Expected: ~20 lines.

---

## Task 13: Write theory/plans/README.md

**Files:**
- Create: `theory/plans/README.md`

- [ ] **Step 1: Write the README**

Write `theory/plans/README.md` with:

```markdown
# theory/plans/

Phase-scoped research and construction plans for the theory KB.

## Naming

`YYYY-MM-DD-<phase-or-topic-slug>.md`.

## Format

Lightweight running plans / progress logs distinct from the formal
implementation plans under `docs/superpowers/plans/`. Use these for:

- Landscape-sweep findings (Phase 1 output).
- Per-topic research scratchpads (Phase 2 working notes).
- Lint-pass findings (Phase 4 output).
- Open-question lists.

Formal step-by-step implementation plans live under
`theory/docs/superpowers/plans/` and follow the writing-plans format.
```

- [ ] **Step 2: Verify**

```bash
wc -l theory/plans/README.md
```
Expected: ~20 lines.

---

## Task 14: Update CLAUDE.md — replace stale Theory section, add citation discipline

**Files:**
- Modify: `CLAUDE.md` — replace the `# Theory Document` section (currently lines 38–62) with new content; add a new `# Theory KB & citation discipline` section.
- Modify: `CLAUDE.md` § Project Structure — update the `theory/` bullet to reflect new layout.

**Critical:** the user has a pending uncommitted edit that adds `# C++ GUI frontend (testing/gui_cpp/)` (~lines 204–233 in the working copy). **Do not touch that section.** Your edit operates only on the `# Theory Document` section and the `theory/` bullet under `## Project Structure`.

- [ ] **Step 1: Update the `theory/` bullet under `## Project Structure`**

Use the Edit tool on `CLAUDE.md`. Replace the existing `theory/` bullet block (currently lines 6–11) with:

```
old_string: `theory/` — LLM theoretical framework (**GROUND TRUTH**): LLM architectures and math, historical progression/timeline, high-level explanations, and visualizations.
  - `build/llm-core-architecture/` — LaTeX source and build artifacts
  - `sources/` — WebSearch-based publication research for the theoretical/mathematical basis
  - `visuals/` — visual materials
  - `GLOSSARY.md` — every technical term used in this workspace must be defined here. If you encounter or use a term not in the glossary, add it immediately.
  - `llm-core-architecture.pdf` — compiled output (kept at theory root)
new_string: `theory/` — LLM theoretical-framework workspace (**GROUND TRUTH**) for theory, math, historical progression, and visualizations. KB-substrate layout (v2 from 2026-05-03):
  - `kb/notes/<area>/<topic>.md` — digested synthesis, one file per topic
  - `kb/excerpts/<paper-key>.md` — verbatim quoted passages from primary sources
  - `kb/index/` — `papers.json` (paper metadata + KB cross-refs), `topics.md` (topic graph + status), `timeline.md` (chronological progression)
  - `kb/glossary.md` — every technical term used in this workspace, with a citation. If you use a term not in the glossary, add it immediately.
  - `sources/papers/` — PDFs of canonical papers (filename: `{paper-key}_{slug}.pdf`)
  - `sources/forums/` — selectively archived blog/forum snapshots (discovery-path provenance only)
  - `archive/2026-05-03-pre-expansion/` — v1 single-LaTeX-doc state preserved as a snapshot
  - `series/` — placeholder for future LaTeX paper series (outlined after KB pass complete; see `theory/docs/superpowers/specs/2026-05-03-theory-expansion-design.md`)
  - `plans/` — research/KB construction running plans
```

- [ ] **Step 2: Replace the entire `# Theory Document` section**

Use the Edit tool on `CLAUDE.md`. Replace the section that currently reads (lines 38–62, "# Theory Document" through the end of "## Conventions/Rules"):

```
old_string: # Theory Document

## Core Document
`theory/build/llm-core-architecture/llm-core-architecture.tex` — LaTeX document covering Transformer architecture from original encoder-decoder through modern decoder-only (LLaMA). 10 sections: Transformer overview, Tokenization, Embeddings, Positional Encoding, Attention, FFN, Normalization/Residuals, Decoder-Only Shift, Output Head, Full Forward Pass.

### Custom LaTeX environments:
- `\begin{implbox}` — green "Implementation Note" callouts
- `\begin{evobox}` — blue "Architectural Evolution" callouts
- `\dimtext{}` — inline dimension annotations

## HTML Companion
`theory/visuals/llm-architecture-diagram.html` — standalone interactive diagram with clickable layers showing tensor shapes and data flow. Dark theme, self-contained (no build step).

## Sources

All claims must be grounded in canonical papers.
- `theory/sources/index.json` — master index (citation key, title, authors, year, URL, local file, summary)
- `theory/sources/papers/` — local PDF copies, named `{key}_{slug}.pdf`
When adding a new source: add entry to `theory/sources/index.json`, download PDF to `theory/sources/papers/`, use the citation key consistently in LaTeX `\cite{}` commands

## Conventions/Rules
- Define every variable in every equation, directly underneath it (brief is fine, but no undefined variables)
- First formalize math, then describe technical aspects and practical use, then elaborate using accessible language and/or analogies
- Ground all architectural claims in specific source papers from core document or original paper(s), with citation keys from `theory/sources/index.json`

new_string: # Theory KB & citation discipline

The theory workspace at `theory/` is now a knowledge-base substrate (v2,
2026-05-03), not a single LaTeX doc. The previous v1 single-doc deliverable
is archived at `theory/archive/2026-05-03-pre-expansion/`. The future
LaTeX paper series under `theory/series/` is **deferred** until the KB
reaches stable coverage — see
`theory/docs/superpowers/specs/2026-05-03-theory-expansion-design.md`.

## Citation rules — non-negotiable

When making technical claims about LLM architecture, training, inference,
interpretability, evaluation, alignment, or related theory:

1. **Every load-bearing claim cites a source.** Either:
   - `[paper-key §X, eq.Y]` — pointing to a paper in `theory/kb/index/papers.json`
   - `[kb/notes/<area>/<file>#<anchor>]` — pointing into a synthesis note
   - `[kb/excerpts/<paper-key>#<heading>]` — pointing into a verbatim excerpt

2. **Verify against the original PDF before propagating a KB-note claim**
   into LaTeX, code, or commit messages. The KB is digested; the paper is
   canonical.

3. **Analogies and intuitions are tagged, never asserted as fact.** Use
   `[ANALOGY]`, `[INTUITION]`, `[CONTRADICTION]`, `[FORUM-SIGNAL]`, or
   `[SPECULATION]` so they cannot be laundered as formal claims. Analogies
   must always return to the canonical symbolic form.

4. **If a claim depends on something not in the KB, add it before continuing.**
   Don't make claims you can't ground.

5. **Forum/blog citations are valid as discovery signals only** (tier B/C
   in `theory/sources/README.md`). They never solely back a hard technical
   claim — only primary papers (tier A) can.

## Source tiers

- **Tier A (canonical):** arxiv, peer-reviewed venues, official tech reports
  / model cards, reference github repos. Stored under
  `theory/sources/papers/`. Backs hard claims.
- **Tier B (high-signal commentary):** vendor research blogs, respected lab
  blogs, named researchers' writeups. Cite alongside an underlying tier-A
  source.
- **Tier C (community signal):** Reddit/HN/X/HF community. Discovery only;
  never the sole citation.

## Writing-style rule (Feynman bar)

Each topic note follows: formal definition (math + variables defined
underneath) → mechanism (how it computes, with tensor shapes) →
variants/lineage (cited list) → tagged `[INTUITION]` / `[ANALOGY]` (which
always return to canonical symbolic form) → frontier and open questions
(with `[CONTRADICTION]` markers where sources disagree).

When introducing a new technical term in a note, add it to
`theory/kb/glossary.md` with a citation.

```

- [ ] **Step 3: Verify the edits**

```bash
grep -c "Theory KB & citation discipline" CLAUDE.md
grep -c "Theory Document" CLAUDE.md
grep -c "build/llm-core-architecture/llm-core-architecture.tex" CLAUDE.md
grep -c "kb/notes/<area>/<topic>.md" CLAUDE.md
```
Expected:
- First: `1` (new section header present)
- Second: `0` (old section removed)
- Third: `0` (old path no longer referenced)
- Fourth: `1` (new path referenced)

- [ ] **Step 4: Verify the user's pending C++ section is intact**

```bash
grep -c "# C++ GUI frontend" CLAUDE.md
grep -c "imgui-cpp-development" CLAUDE.md
```
Expected: `1` and `1` — unchanged from the user's pending edit.

---

## Task 15: End-state directory structure verification

**Files:** none (read-only checks)

- [ ] **Step 1: Snapshot the new tree**

```bash
find theory -type d | sort
```
Expected (excluding `theory/sources/papers` contents):
```
theory
theory/archive
theory/archive/2026-05-03-pre-expansion
theory/archive/2026-05-03-pre-expansion/build
theory/archive/2026-05-03-pre-expansion/build/llm-core-architecture
theory/archive/2026-05-03-pre-expansion/visuals
theory/docs
theory/docs/superpowers
theory/docs/superpowers/plans
theory/docs/superpowers/specs
theory/kb
theory/kb/excerpts
theory/kb/index
theory/kb/notes
theory/kb/notes/alignment
theory/kb/notes/architecture
theory/kb/notes/evaluation
theory/kb/notes/inference
theory/kb/notes/interpretability
theory/kb/notes/post-training
theory/kb/notes/reasoning
theory/kb/notes/scaling
theory/kb/notes/training
theory/plans
theory/series
theory/sources
theory/sources/forums
theory/sources/papers
```

- [ ] **Step 2: Confirm no orphan files at theory/ root**

```bash
find theory -maxdepth 1 -type f | sort
```
Expected: `theory/README.md` only.

- [ ] **Step 3: Confirm pre-expansion artifacts are all in archive**

```bash
find theory/archive/2026-05-03-pre-expansion -type f | sort
```
Expected (5 files):
```
theory/archive/2026-05-03-pre-expansion/GLOSSARY.md
theory/archive/2026-05-03-pre-expansion/Makefile
theory/archive/2026-05-03-pre-expansion/build/llm-core-architecture/llm-core-architecture.tex
theory/archive/2026-05-03-pre-expansion/llm-core-architecture.pdf
theory/archive/2026-05-03-pre-expansion/visuals/llm-architecture-diagram.html
```

- [ ] **Step 4: Confirm the existing dated PDFs at archive root are untouched**

```bash
ls theory/archive/llm-core-architecture-*.pdf
```
Expected: two files — `llm-core-architecture-2026-04-11.pdf` and `llm-core-architecture-2026-04-16_180243.pdf`.

- [ ] **Step 5: Confirm new READMEs all exist**

```bash
ls theory/README.md theory/kb/README.md theory/sources/README.md theory/series/README.md theory/plans/README.md
```
Expected: 5 files listed, no errors.

- [ ] **Step 6: Confirm KB index files all exist**

```bash
ls theory/kb/index/papers.json theory/kb/index/topics.md theory/kb/index/timeline.md theory/kb/glossary.md
```
Expected: 4 files listed, no errors.

---

## Task 16: Stage and commit

**Files:** all of Phase 0.

- [ ] **Step 1: Show staged + unstaged status**

```bash
git status --short
```
Expected: lots of `D ` lines for moved-out files, lots of `??` lines for new files, plus `M CLAUDE.md`, plus the user's pre-existing untracked `docs/skills/`, `learn/`, `pyproject.toml`, `testing/gui_cpp/`.

- [ ] **Step 2: Stage only Phase 0 changes**

```bash
git add CLAUDE.md
git add theory/
```
**Do not** use `git add -A` or `git add .` — those would catch the user's untracked C++ work.

- [ ] **Step 3: Verify staged set excludes user's separate work**

```bash
git diff --cached --stat -- "docs/skills/" "learn/" "pyproject.toml" "testing/gui_cpp/"
```
Expected: empty output (none staged).

- [ ] **Step 4: Verify staged set has expected shape**

```bash
git diff --cached --stat | tail -1
```
Expected: a single summary line like `N files changed, X insertions(+), Y deletions(-)` with N around 25 (6 new READMEs/index files + 11 .gitkeep files + 5 moved files + 1 deleted index + 1 modified CLAUDE.md + 1 plan file).

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
docs(theory): Phase 0 — restructure to KB-substrate layout

v1 single-LaTeX-doc state archived under
archive/2026-05-03-pre-expansion/. New KB-substrate layout under
theory/kb/ with three-tier epistemic separation: kb/excerpts/ for
verbatim quoted passages (one file per paper), kb/notes/<area>/ for
digested synthesis with inline citations and tagged analogies,
kb/index/ for papers.json + topics.md + timeline.md, plus
kb/glossary.md.

Top-level navigation in theory/README.md; epistemic rules in
kb/README.md; tier definitions in sources/README.md; placeholder
explanations in series/README.md and plans/README.md.

theory/sources/index.json migrated to theory/kb/index/papers.json
with expanded schema (added venue, excerpts_file,
notes_referenced_by, topics, category fields). All 12 existing
papers preserved.

theory/build/, theory/visuals/, theory/llm-core-architecture.pdf,
theory/GLOSSARY.md, theory/Makefile all moved into the
pre-expansion snapshot. theory/sources/papers/ untouched.

CLAUDE.md updated: removed stale "Theory Document" section,
added "Theory KB & citation discipline" with rules requiring
[paper-key §X] or [kb/notes/...#anchor] cites for every
load-bearing claim, [ANALOGY]/[INTUITION] tagging, source-tier
discipline (A/B/C). Project Structure bullet for theory/ updated
to reflect new layout.

Subsequent phases (1: landscape sweep, 2: per-topic deep
research, 3: historical backfill, 4: lint, 5: deferred LaTeX
series outline) will populate the empty kb/notes/<area>/
directories per
theory/docs/superpowers/specs/2026-05-03-theory-expansion-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Post-commit verification**

```bash
git status --short
git log --oneline -1
```
Expected:
- Status shows only the user's pre-existing dirty state (`?? docs/skills/`, `?? learn/`, `?? pyproject.toml`, `?? testing/gui_cpp/`) and nothing else.
- Latest commit is `docs(theory): Phase 0 — restructure to KB-substrate layout`.

---

## End of Phase 0

Restructure complete. The KB substrate is empty (notes/excerpts dirs are
populated only with `.gitkeep` files). Phase 1 (landscape sweep) is the
next plan to execute — it produces the candidate paper list and refines
the topic taxonomy before Phase 2 builds notes.

If the post-commit `git status` shows anything unexpected, investigate
before declaring Phase 0 done.
