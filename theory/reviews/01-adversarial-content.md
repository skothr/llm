# Adversarial-content review

_Date: 2026-05-05. Reviewer: fresh adversarial reader. Scope: all 5 papers, 70 sections, 394pp PDF. Read-only on `theory/series/` and `theory/kb/`._

## Summary

- **Sections sampled (15):**
  - Paper 1: §05 attention-kv-sharing, §06 attention-hardware-implementation, §13 frontier-snapshot, §14 contradictions
  - Paper 2: §03 scaling-laws (spot), §07 sft (spot), §14 contradictions
  - Paper 3: §11 cot-faithfulness, §13 frontier-snapshot, §14 contradictions
  - Paper 4: §05 correlation-causation (spot), §06 truth-directions, §08 sae-canonicality, §10 rome, §11 circuits (spot)
  - Paper 5: §08 dangerous-capability, §10 scheming, §12 watermarking, §13 scalable-oversight, §14 contradictions
- **Findings:** 0 BLOCKER, 4 MAJOR, 6 MINOR, 3 NIT.
- **Bottom line:** The series' citation discipline is mostly honest — the `\deepencite{… citation-pending}` machinery is used heavily and at the right places. The risk is concentrated in (a) one hard factual error inside a `\begin{contradiction}` block, (b) a load-bearing single-source counter-claim repeatedly cited without `\deepencite`, and (c) two frontier-snapshot tables (Papers 1 §13, 3 §13) whose closed-vendor rows are scaffolded entirely on `citation-pending` system cards while the section's headline-claim text reads as resolved fact.

## Findings (by severity)

### MAJOR

**M1. Paper 3 §11 line 248: AIME pass@1 climb wrong — `15.6% → 71%`, KB and Paper 3 §14 say `15.6% → 77.9%`.**

`paper-3/sections/11-cot-faithfulness.tex:248` —
`AIME 2024 pass\@1 climbing $15.6\% \to 71\%$,` is footnoted to `kb/excerpts/deepseek-r1.md#sec-2-2-reward`. Two problems: (a) the KB excerpt at that anchor is the rule-based-reward quote and contains no AIME number; (b) the actual AIME number (`#sec-2-3-aime`) is `15.6% → 77.9%`, verified against the verbatim quote `"jumping from an initial 15.6% to 77.9%"`. Paper 3 §14 itself uses 77.9% in three other places (including line 87 of §14 and line 113 in the C4 paragraph). The 71% number appears nowhere in the KB. **What would close it:** change `71\%` to `77.9\%` and re-target the footnote to `#sec-2-3-aime`. Inside a `\begin{contradiction}` block this is exactly the kind of factual error that breaks the surrounding rhetorical move.

**M2. `rlvr-limits-2025` (Yue et al., NeurIPS 2025 oral) is the central RLVR-limits counter-claim and has no KB excerpt or `\deepencite` flag.**

`papers.json:867` lists `rlvr-limits-2025` with `excerpts_file: null`, `local_file: null`, `authors: "various"`. It is cited in:
- `paper-2/sections/14-contradictions.tex:277, 286–290` (load-bearing for contradiction (9) "Does RLVR create new reasoning capability?")
- `paper-3/sections/14-contradictions.tex:88, 96, 102` (load-bearing for C3 — including a sub-section quote `\citep[\S 5]{rlvr-limits-2025}` for the synthesis claim "RLVR teaches trace-length use without teaching new primitives")
- the contradiction in §14 reads as if Yue's pass@K-vs-pass@1 finding is field-stable, and §3 of Paper 3 inherits the same weight.

Per the playbook, when the KB excerpt is abstract-only or absent, the LaTeX should mark with `\deepencite{key §X}`. None of the §14 cites do — they read as primary-grounded `\citet{}`/`\citep[…]{}` while the KB has neither a §-anchored excerpt nor a downloaded PDF. The detailed numerical/structural claim (`pass@1 vs pass@K at matched sampling budget`, `RLVR sharpens within base-model competency but does not expand`) is presented as fact. **What would close it:** download the PDF to `sources/papers/`, build `kb/excerpts/rlvr-limits-2025.md` with §3 (the pass@K comparison) and §5 (the synthesis claim) verbatim, and replace bare `\citet{rlvr-limits-2025}` with anchored versions or `\deepencite` until the excerpt lands. This is the load-bearing 2026 counter-claim of Paper 3's whole thesis; it cannot rest on an abstract-only `[CONTRADICTION]` index entry.

**M3. Paper 5 §14 line 129 — `sadasivan2023-detection` cited bare for a load-bearing watermarking claim that requires `\deepencite`; co-located empirical claim has no citation at all.**

In `paper-5/sections/14-contradictions.tex:127–134`:
```
\citet{sadasivan2023-detection} prove that any watermark detectable from
short spans is in principle removable by a paraphrase attacker with a
comparable-quality model, and independent adversarial probes of
SynthID-Text confirm that aggressive paraphrase plus modest fine-tuning
does defeat the deployed scheme on short-form text.
```
`papers.json:3726` lists `sadasivan2023-detection` with `excerpts_file: null`, `local_file: null` — it is abstract-summary-only in the KB. §12 of the same paper handles this correctly (`\citet{sadasivan2023-detection} \deepencite{sadasivan2023-detection §main-result, citation-pending}` at line 282–283), so §14's missing flag is an inconsistency, not an oversight in the KB. The second clause ("independent adversarial probes of SynthID-Text confirm…") has no citation at all in §14 — §12 does cite the SRI Lab probe via `\deepencite{eth-synthid-probe-2024, citation-pending}` (line 335), but §14 launders this as if it were a settled experimental result. **What would close it:** add the matching `\deepencite{sadasivan2023-detection §main-result, citation-pending}` and `\deepencite{eth-synthid-probe-2024, citation-pending}` to §14, so the contradictions catalogue inherits the same source caveats §12 already carries.

**M4. Paper 1 §13 and Paper 3 §13 frontier-snapshot tables: closed-vendor rows scaffolded on `\deepencite{… system card pending}` — table reads as load-bearing structural fact, sources are tier-B by KB rules.**

`paper-3/sections/13-frontier-snapshot.tex` builds a 6-row comparison table that the section explicitly identifies as load-bearing: line 35–38, *"every frontier-2026 reasoning entry ships RLVR plus long-CoT, none of them ship a process reward model in the headline RL loop, and the productionised exposure of inference-time compute has converged."* Of the 6 rows, only DeepSeek-R1 cites a tier-A primary source (`\citep[\S 2]{deepseek-r1}`); the other five (Qwen3-thinking, QwQ, OpenAI o1/o3, Gemini 2.5-thinking, Claude 3.7/4) are entirely scaffolded on `\deepencite{… system card pending}`. Per the project's source-tier rule (`CLAUDE.md`: tier B never solely backs a hard technical claim), the headline structural claim is overweight relative to the source quality.

Paper 1 §13 has the same shape — `meta-llama4` (one tier-A row), and `\deepencite{citation-pending: vendor system card}` on Mistral, Qwen 2.5, Anthropic Claude 3.7 (lines 26, 133, 154, 175, etc.). The section does include a `\begin{forumsignal}` block on closed-vendor disclosure but the *headline* in `\subsection{Cross-cutting agreement and divergence}` is presented as resolved.

This is the closest thing in the series to "closed-vendor laundering." It is mitigated — every closed-vendor cell is honestly tagged — but the sections' upstream-rhetorical claims (a convergence statement) lean on those cells. A reader who skips the table and reads only the prose will believe the convergence claim is settled. **What would close it:** when stating the convergence claim, downgrade scope explicitly: "of the publicly-disclosed open-weight rows, all ship X." Either soften the universal quantifier or move the convergence claim to a `\begin{forumsignal}` block. This is a writing fix, not a research fix.

### MINOR

**m1. Paper 5 §14 line 57 — Crescendo `29–71% ASR uplift` conflates two ranges across two models.**

`russinovich2024-crescendo` reports `29–61%` on GPT-4 and `49–71%` on Gemini-Pro (verified at `kb/excerpts/russinovich2024-crescendo.md:38–39`). The §14 text writes a single span "29–71%" — defensible as "the union of model-specific ranges" but ambiguously presented. The right form is "29–61% on GPT-4 and 49–71% on Gemini-Pro." Same shape repeats at line 57 and is not a quote of the source.

**m2. Paper 5 §14 line 30 — `mmlu-redux-2024` `6.49%` cited without flagging the KB excerpt is itself abstract/README-only.**

`kb/excerpts/mmlu-redux-2024.md:10` self-flags: *"PDF not yet downloaded; excerpts compiled from the arXiv 2406.04127 abstract / project README. The headline numbers (6.49% overall error rate) are widely cited and quoted in the Phase 1 sweep §eval-methodology. PDF verification still pending."* §14 cites the number with no `\deepencite{}`, and §03 (knowledge-benchmarks, where the verifier-error-ceiling thesis is set up) presumably does the same thing. The number is widely-quoted enough that this is not high-risk, but the discipline is supposed to be flag-on-pending.

**m3. Paper 3 §11 line 247 — R1 trace-length growth `~200 to ~10,000 tokens` not directly in `kb/excerpts/deepseek-r1.md`.**

The §11 contradiction block uses two specific numbers (response length 200→10,000) to set up the causal-trace-vs-rationalisation tension. The KB excerpt at `kb/excerpts/deepseek-r1.md#sec-2-2-reward` (which the §11 footnote points at) does not contain those numbers. The R1 paper does report a thinking-time growth curve in its Figure 3, but until the KB has a §-anchored excerpt for it, those numbers should be flagged. (Compare the AIME 15.6→77.9 number, which has its own `#sec-2-3-aime` anchor.)

**m4. Paper 5 §13 line 264 — `76–88% accuracy` for Brown-Cohen et al. 2024 is asserted while KB excerpt is `citation-pending`.**

`paper-5/sections/13-scalable-oversight.tex:259–266` cites a specific accuracy band for the doubly-efficient-debate result with `\deepencite{brown-cohen2024-doubly-efficient-debate §main-theorem, citation-pending}`. The deepencite is applied (good); the quoted number is then used as a result in the rest of the paragraph. Since the deepencite says "citation-pending," any specific number drawn from that source should be inside the deepencite scope or restated in conditional language until the excerpt lands.

**m5. Paper 1 §14 line 70–74 — `~75% of theoretical peak FP16` for FA3 cited via `\citep{shah2024}` while the KB has no shah2024 excerpt.**

§06 (where FA3 is introduced) correctly applies `\deepencite{shah2024 §3 (excerpt not yet in KB; …)}` at line 207. §14 then cites `\citep{shah2024}` bare for the 75% claim. The §06 deepencite arguably licenses the §14 reference under the project's "first use anchors, downstream uses cite the bibkey" pattern, so this is borderline acceptable — but a strict reading of the playbook would say §14 should also carry the marker.

**m6. Paper 5 §14 paragraph 5 — sadasivan claim itself OK in §12, but §14 attributes "in principle" to the source. `sadasivan2023-detection`'s actual claim is about *recursive paraphrasing* converging to random performance, not "any watermark detectable from short spans is in principle removable." The inferential bridge is fine, but the verb "prove" is too strong for an arXiv paper that argues empirically/theoretically without a tight equilibrium proof.**

This is more a writing-discipline issue than a citation issue: an arXiv paper showing convergence under recursive paraphrasing is not a *proof* that any watermark is *in principle* removable — it's a strong empirical-plus-theoretical argument under specific assumptions. Soften to "argue."

### NIT

**n1. References.bib coverage of `\deepencite{}` keys.** Several keys cited only in `\deepencite{}` are intentionally not in `references.bib` (`christiano2018-ida`, `leike2018-rrm`, `shamir1992-ip`, `eth-synthid-probe-2024`, `qwen3-thinking`, `qwq`, `openai-o1`, `openai-o3`, `claude-3-7`, `dathathri2024-synthid`, `kirchenbauer2024-reliability`, `watermark-fine-tune-survey`, `bricken2023-monosemanticity`, `templeton2024-scaling-monosemanticity`, `lima-2023`, `rosset2024-direct-alignment-scaling`, `qwen3-thinking`, `meta-llama5`, `bitnet-2b4t-2025`, `gemini2-5` (this one IS in papers.json but not in bib), …). The `\deepencite` macro renders as a margin note and never tries to resolve the key; that's by design. **Flag for next pass:** when those keys come into the references.bib, swap the `\deepencite` for `\citep` + KB-excerpt footnote.

**n2. Multiple "(Vaswani–Eq. 1)" / "(Vaswani–Eq. 2–3)" custom labels** are cute but inconsistent across sections — Paper 1 §05 uses them, Paper 2 §03 uses standard `\eqref` with cite. Stylistic, no impact.

**n3. The `\begin{contradiction}` block in Paper 1 §14 line 196–211 (the C5 long-context "intuition")** is in the contradiction environment but reads more like an `\begin{intuition}` (it offers a reconciling read, not a flagging-of-disagreement). The next environment `\begin{intuition}` at 196 fixes it; the *labelled* `\begin{contradiction}` at 153 is correctly contradiction-shaped. No bug; the playbook's environment vocabulary maps well in practice.

## What I sampled and why

- **All five §14 contradictions sections** — the brief calls them out specifically; they are also the places where elision is most consequential. Paper 3 §14 carried the most cross-paper weight and gets the most scrutiny here.
- **Two anchor sections per paper** — §5 attention-kv-sharing (P1), §3 scaling-laws (P2), §11 cot-faithfulness (P3), §6 truth-directions and §10 ROME (P4), §12 watermarking and §10 scheming (P5). These are the "load-bearing math + claim" surfaces where the Feynman-bar rule and citation discipline matter most.
- **Highest-deepencite-density sections** — Paper 3 §13 frontier-snapshot (32 deepencites), Paper 5 §12 watermarking (12), Paper 4 §08 sae-canonicality, Paper 5 §08 dangerous-capability, Paper 5 §13 scalable-oversight. These are where the risk of over-claiming on tier-B sources concentrates.
- **Spot-checks** — Paper 1 §13 (the closed-vendor architecture-snapshot analog of Paper 3 §13), Paper 4 §05 correlation-causation (load-bearing for the rest of Paper 4), Paper 4 §11 circuits (IOI as the canonical interpretability worked example), Paper 2 §07 SFT.

For each picked section I verified 3–6 specific quantitative or causal claims by either reading the corresponding `kb/excerpts/<key>.md` anchor (when the playbook required one) or by reading the `papers.json` summary plus `kb/index/contradictions.md` lineage. The §11 R1-AIME number was the only outright-wrong claim (M1); the rlvr-limits-2025 issue (M2) was the largest scope citation-discipline gap; the §13 frontier tables (M4) are the largest risk for the closed-vendor-laundering category in the brief.

The surface I did **not** sample: Paper 2 §04–§13 internal subsections, Paper 4 §03/§04/§07/§09/§13, Paper 5 §02/§03/§04/§06/§07/§09/§11. A second pass focusing on Paper 4 §13 scaling-and-snapshot (9 deepencites) and Paper 2 §10 rlaif-cai (9 deepencites) would be the next-most-valuable additional 2 hours.
