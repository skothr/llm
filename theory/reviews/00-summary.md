# Review-wave consolidated findings (2026-05-06)

Six independent fresh-context opus reviewers, each with a different lens, audited
the just-completed five-paper LaTeX series at `theory/series/` (70 sections,
~394pp, 5 PDFs). Each reviewer was read-only on the source and wrote one
markdown report; this document triages their combined findings.

| # | Reviewer | Lens | Report | Findings |
|---|----------|------|--------|----------|
| 1 | Adversarial-content | Hostile / skeptical: unsupported claims, citation/scope mismatch | `01-adversarial-content.md` | 0 BLOCKER, 4 MAJOR, 6 MINOR, 3 NIT |
| 2 | Cross-paper coherence | Notation drift, contradictions between papers, thread duplication | `02-cross-paper-coherence.md` | 0 BLOCKER, 2 MAJOR, 5 MINOR |
| 3 | Math correctness | Equation-vs-source verbatim check, dimensional consistency | `03-math-correctness.md` | 0 BLOCKER, 0 MAJOR, 4 MINOR |
| 4 | Citation-tier discipline | Tier-A backing, deepencite hiding load-bearing claims | `04-citation-tier.md` | 0 BLOCKER, 0 MAJOR, 2 MINOR, 3 NIT |
| 5 | Frontier currency (web-enabled) | 2025-26 numerical claims still hold; recent releases | `05-frontier-currency.md` | **1 BLOCKER**, 4 MAJOR |
| 6 | Feynman-bar pedagogy | Analogy-returns-to-math, intuitions grounded, callout discipline | `06-pedagogy.md` | 0 BLOCKER, 1 MAJOR, 2 MINOR |

**Aggregate: 1 BLOCKER, 11 MAJOR, ~19 MINOR, 6 NIT.**

The single BLOCKER is from the web-enabled currency reviewer: a contradiction
that Paper 3 §14 frames as "open" has been partially closed by a Nature paper
that landed after the KB build cutoff. The 11 MAJORs are concentrated in two
clusters — frontier-currency drift and citation-source mismatch — and both
clusters are short-form mechanical fixes rather than re-writes.

## BLOCKER (must fix before publication)

### B1 — AIME contamination contradiction is now partially closed

- **Source:** Reviewer #5 (frontier-currency).
- **Where:** Paper 3 §14 (`14-contradictions.tex`), contradiction C5 ("AIME contamination").
- **What's stale:** the section frames the AIME contamination contradiction as open. DeepSeek-R1 was peer-reviewed in *Nature* (Sept 2025) and the published version includes a decontamination audit plus a Qwen2-7B control experiment that addresses the contamination concern.
- **Fix:** add a one-paragraph update to Paper 3 §14 C5: "The Sept 2025 *Nature* publication of DeepSeek-R1 includes a decontamination audit \citep{...} and a Qwen2-7B control that reduces but does not fully close this contradiction." Add the Nature publication as a new key in `papers.json`.
- **Estimated cost:** 15 minutes including bib-key add and rebuild.

## MAJOR (concentrated mechanical fixes)

### Citation / source mismatch cluster (5 issues)

| ID | Reviewer | Where | Issue | Fix |
|---|---|---|---|---|
| M1 | R1 | Paper 3 §11 | AIME number error: "15.6%→71%" should be "15.6%→77.9%" (canonical R1 number) | sed-replace one number |
| M2 | R1 | Paper 2 §14 + Paper 3 §14 | `rlvr-limits-2025` cited bare as load-bearing counter-claim despite no KB excerpt | Add `\deepencite{rlvr-limits-2025 §X, citation-pending}` flag, or add the KB excerpt |
| M3 | R1 | Paper 5 §14 | `sadasivan2023-detection` cited without `\deepencite` marker that §12 uses | Add the marker for consistency |
| M4 | R2 | Paper 3 (3 sites) | Hoffmann Chinchilla parametric loss cited as "Eq. 10" but it's "Eq. 2" in source | sed-replace 3 occurrences |
| M5 | R3 | Paper 3 §7 | GRPO Eq. cites `deepseek-r1` for the token-level form that's actually from `shao2024` | swap cite-key |

### Frontier-currency drift cluster (4 issues)

| ID | Reviewer | Where | Issue | Fix |
|---|---|---|---|---|
| M6 | R5 | Paper 1 §13 + §14 | Closed-row labels stale 6-12 months: GPT-5/5.4, Claude 4-family, Gemini 2.5 Deep Think | Update the snapshot table closed-vendor rows + add 1-paragraph caveat |
| M7 | R5 | Paper 1 §13 | Mistral Large 3 (Dec 2025) is MoE — flips Paper 1's dense-flagship Mistral row | Update one snapshot cell + add brief note |
| M8 | R5 | Paper 5 §10 + §14 | OpenAI/Apollo deliberative-alignment paper (Sept 2025) sharpens scheming contradiction | Add 1-paragraph update + new bib key |
| M9 | R5 | Paper 3 §11 | "2026-05 consensus leans Yue with caveats" framing is one-sided per recent counter-evidence | Soften the consensus claim + cite the counter-paper |

### Single-issue MAJORs (2)

| ID | Reviewer | Where | Issue | Fix |
|---|---|---|---|---|
| M10 | R1 | Paper 1 §13 + Paper 3 §13 | Universal-quantifier convergence claims built on closed-vendor system cards (tier B by project rules) | Add explicit tier-B hedge to the snapshot intro paragraph |
| M11 | R2 | Paper 3 §7 | Intuition box overstates contested Yue RLVR-limit hypothesis as fact (next paragraph walks it back, but the intuition reads ungated) | Soften the intuition box copy |

## MINOR (cleanup, batchable)

19 minor findings spread across the six reports. All are mechanical and can be
batched into a single sweep:

- **Notation drift in Paper 4 from canonical shape macros** (R2): Paper 4 sections use $h_\ell$ and $H$ inconsistently with Paper 1's $\BSH$/$\BHSdh$ macros. Define explicit aliases or normalize on the canonical macros.
- **Paper 2 one-off `C_{\text{infer}}` slip** (R2): inconsistent with the rest of the series's $C_{\text{test}}$.
- **Paper 5 `\deepencite` misuse for known cross-paper section refs** (R2): a few `\deepencite{}` markers are pointing at content that's actually in Paper 4 — should be inline-text Paper-N §X mentions instead.
- **Snell `θ*` subscript drift** (R3): minor notation inconsistency between §4 introduction and equation.
- **PUCT formula labeled "UCT"** (R3): the rStar-Math formula uses PUCT (with prior $\pi_\theta$) but Paper 3 §6 calls it UCT.
- **Watermark spike-entropy form** (R3): properly `\deepencite`-flagged but the section claims more about it than the abstract supports.
- **`\begin{speculation}` env defined but never used** (R6): ~12 [SPECULATION] markers in KB notes never landed in the LaTeX as `\begin{speculation}` callouts. Tag-coverage gap.
- **Paper 4 callout density lowest of all five papers** (R6): consider adding 2-3 more intuition/contradiction callouts.
- **Forumsignal undercount** (R6): only 9 across the series; nostalgebraist + Nanda + a few should be tagged in body sections, not just snapshot/14-contradictions.
- **Tier-4 `lindsey2025-circuit-tracing` is single-source methodology** (R4): hedge with a margin note or a forumsignal tag.
- **Paper 3 §13 o-series RLVR cell** (R4): soften from a positive assertion to "claimed in the system card; not externally reproduced."
- (~7 other minors per individual reports)

## NIT (cosmetic, optional)

6 nits across the reports — mostly capitalization, hyphenation, spacing in the
contradiction tables. Not worth a sweep on their own.

## Recommended fix order

1. **B1 alone** — single-section update; immediately publishable.
2. **M1, M3, M4, M5** — sed-class fixes; ~10 minutes total.
3. **M2** — either add the KB excerpt for `rlvr-limits-2025` (opens Phase 2.5 territory) or add the deepencite flag (1 minute). Recommend the deepencite flag in this pass; KB excerpt during the optional Phase 2.5.
4. **M6, M7, M8, M9** — frontier-currency drift, requires WebSearch + bib additions. Single subagent or a focused 30-minute orchestrator session.
5. **M10, M11** — copy-edit pass on the two relevant sections; ~15 minutes.
6. **MINORs** — single batch sweep, ~30 minutes.
7. **NITs** — defer indefinitely or include with the next deepening pass.

## Bottom line

The series is publication-quality at thesis level. The reviewers found
**zero structural issues** and **one factual issue closed by a post-cutoff
publication**. The remaining MAJORs are concentrated, mechanical, and
collectively closeable in under 2 hours of orchestrator time. The MINORs and
NITs are accumulating tax that's worth a single batched sweep but not blocking.

The structural theses of all five papers — five-axis convergence (P1),
six-stage pipeline (P2), reasoning triangle (P3), method-pluralism (P4),
layered defense (P5) — survive intact across all six review lenses.
