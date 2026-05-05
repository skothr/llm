# Section-writer playbook

Every LaTeX section subagent reads this file first. It is the single source of style, citation, and tooling rules for the LaTeX paper series at `theory/series/`.

## Your job

You are a section-writer subagent. You own a single section of one paper:

- **Paper:** `theory/series/paper-N/`
- **Section file:** `theory/series/paper-N/sections/<section-slug>.tex`
- **Section anchor in outline:** `theory/series/paper-N/outline.md` §X

You produce one `.tex` file at the path above. You do not modify `main.tex`, `preamble.tex`, `references.bib`, or any KB files.

## Quality bar (non-negotiable)

The series target is a Feynman-bar monograph series. Every section must hit:

1. **Cite, never assert.** Every load-bearing technical claim cites a paper-key from `theory/kb/index/papers.json` via `\citep{key}` or `\citet{key}` (BibTeX entries are auto-generated in `theory/series/references.bib`). Where the KB has a section-anchored excerpt at `theory/kb/excerpts/<key>.md`, also cite the anchor inline as a footnote: `\footnote{KB excerpt: \texttt{kb/excerpts/key.md\#sec-3}}`. Where the excerpt is still abstract-only, mark with `\deepencite{key §X}` (a yellow margin note).

2. **Math is verbatim from the cited paper.** When you transcribe an equation, it must match the paper's equation up to notation choice. Don't paraphrase math. Use the standard tensor-shape notation from `preamble.tex`: `B` (batch), `S` (sequence), `D` (model dim), `H` (heads), `d_h` (head-dim), `V` (vocab), `L` (layers).

3. **Tagged claims, not laundered ones.** Match the KB's `[INTUITION]` / `[ANALOGY]` / `[CONTRADICTION]` / `[FORUM-SIGNAL]` / `[SPECULATION]` markers using the LaTeX environments in `preamble.tex`:
   ```latex
   \begin{intuition}
   The KV cache is a window onto the past tokens' projected memory:
   it grows with context length, and any reduction in its size buys
   serving cost at the price of attention quality. (Returning to math:
   shape is $\BHSdh$ at full MHA, $\mathsf{B}\!\times\!\mathsf{K}\!\times\!\mathsf{S}\!\times\!d_h$ at GQA
   with $K$ groups.)
   \end{intuition}
   ```
   Available env: `intuition`, `analogy`, `contradiction`, `forumsignal`, `speculation`. Each renders as a colored callout box. Analogies must always close with a return to canonical math.

4. **Page budget honored within ±20%.** Your section's outline gives a page budget. ~400 words per page in `\documentclass[11pt]`, with figures and equations taking proportionally less text. If you blow budget, you're doing too much; trim.

5. **Section structure.** Every section has:
   - Section heading (LaTeX `\section{}` or `\subsection{}` per outline).
   - Opening paragraph stating the section's thesis sentence (from outline).
   - Body: 3-7 subsections, each grounded in primary KB anchors.
   - At least one tagged callout (intuition, analogy, contradiction).
   - Closing paragraph that connects to the next section (forward-link).

6. **No analogies without canonical math.** If you write an analogy, the next sentence or subsection must give the symbolic form the analogy maps to. Per CLAUDE.md, "analogies always return to canonical symbolic form."

## Tools and workflow

1. **Read first** (in this order):
   - `theory/series/SHAPE-C-decision.md` (paper context)
   - `theory/series/paper-N/outline.md` (your section's full spec)
   - `theory/series/preamble.tex` (LaTeX env you can use)
   - The KB notes in your section's anchor list — `theory/kb/notes/<area>/<topic>.md` files
   - The KB excerpts in your anchor list — `theory/kb/excerpts/<key>.md` files (these are your verbatim source)
   - 1-2 already-written sections from another paper as style reference (when available)

2. **Use `Read`** for KB files — never `Bash(cat/head/tail)`.
3. **Use `Write`** for your section.tex output. **Never** edit `main.tex`, `preamble.tex`, or files outside `theory/series/paper-N/sections/`.
4. **No subagent dispatch from inside a subagent.** You are a leaf in the dispatch tree.
5. **No git operations.** The orchestrator commits.

## Citation patterns

### Inline cite
```latex
\citep{vaswani2017} introduces scaled dot-product attention as
$\softmax(QK^\top/\sqrt{d_k}) V$ \citep[\S 3.2.1, Eq. 1]{vaswani2017}.
```

### Cite with KB excerpt anchor
```latex
\citet{ba2016-layernorm} defines layer normalization%
\footnote{KB excerpt: \texttt{kb/excerpts/ba2016-layernorm.md\#sec-3}}
as $\mu^l = \frac{1}{H} \sum_{i=1}^H a_i^l$ and
$\sigma^l = \sqrt{\frac{1}{H} \sum_{i=1}^H (a_i^l - \mu^l)^2}$ (Eq. 3).
```

### Deepen-cite marker (KB excerpt is still abstract-only)
```latex
The Muon optimizer reports 52\% fewer FLOPs to matched loss on a 16B-MoE
training run \citep{muon-moonlight2025}. \deepencite{muon-moonlight2025 §3}
```

### Tagged callout
```latex
\begin{contradiction}
\citet{deepseek-v2} report MLA matches MHA quality at 4x lower
KV-cache memory; the result is single-source as of 2026-Q2 with no
published third-party replication at frontier scale.
\end{contradiction}
```

## What NOT to do

- Don't invent citations. If a key isn't in `references.bib`, flag it and use `\deepencite` with a description.
- Don't paraphrase equations. Transcribe.
- Don't skip the Feynman bar. Every analogy returns to math.
- Don't run LaTeX. The orchestrator builds.
- Don't write more than your section's page budget × 1.2.
- Don't modify any file outside your assigned section path.

## Final report (what you return to orchestrator)

Under 200 words:

1. **Section file:** path written.
2. **Word count:** ~xx words / ~yy pages.
3. **Citations used:** count distinct paper-keys cited.
4. **Tagged callouts:** count of intuition/analogy/contradiction/etc.
5. **Open `\deepencite` markers:** count + brief list.
6. **Cross-paper forward-references:** any place you cite Paper M § Y; orchestrator will resolve to a `\cref` at link time.
7. **Anything you couldn't ground.** A paragraph would be a citation but the KB doesn't support it; flagged for the next deepening pass.
