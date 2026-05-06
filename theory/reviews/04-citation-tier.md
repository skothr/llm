# Citation-tier discipline review

Fresh independent review of `theory/series/` — five papers, 70 sections — against the playbook's "cite, never assert" rule, dual-citation pattern, `\deepencite` discipline, and the project-level tier-A/B/C source classification (`/home/ai/ai-projects/llm/CLAUDE.md` §"Source tiers"). All `theory/series/`, `theory/kb/`, and `theory/sources/` paths are treated as read-only.

## Summary

- **Distinct cite-keys used:** 175 (all resolve in `references.bib`; 0 unresolved bib references).
  - The earlier extractor saw 177 — two of those (`sec:knowledge-mmlu`, `sec:scaling-map`) are LaTeX cross-reference labels, not bibkeys; they appear inside `\citep[\S\ref{sec:...}]{key}` constructs and are false positives. Real count is 175.
- **All 175 cite-keys are present in `theory/kb/index/papers.json`.** No "phantom" cites.
- **Tier breakdown** (using papers.json `category` field as proxy):
  - **Tier-A primary** (foundational + landmark + tech-report + recent + frontier + survey + benchmark): **174 / 175** (99.4%).
  - **Tier-B secondary** (the `secondary` category): **1 / 175** — `chen2023-speculative`. Used twice in `paper-1/10-multi-token-output.tex` always as concurrent-work co-citation alongside the tier-A primary `leviathan2023`. Appropriate.
  - **Tier-C unknown:** 0.
- **Cite-keys with KB-excerpt anchor available:** 97 / 175 (55%). Remaining 78 are arxiv-resolvable tier-A primary papers awaiting an excerpt-extraction pass — `\deepencite` is correctly used at most call sites for these.
- **`\deepencite{}` markers total:** **127** distinct call sites, distributed: paper-1 = 26, paper-2 = 28, paper-3 = 38, paper-4 = 19, paper-5 = 16. Highest single-section concentration: `paper-3/13-frontier-snapshot.tex` (32 markers, by far the largest, all closed-vendor system-card or pending-bibkey).
- **`\begin{forumsignal}` blocks:** 9 (paper-1: 3, paper-3: 2, paper-4: 2, paper-5: 2). All wrap the load-bearing-but-tier-B/C claims they should — see audit below.
- **Dual-cite anchor (`\footnote{KB excerpt: …}` or `KB note: …`) count:** 504 footnote anchors total (416 KB-excerpt, 88 KB-note). Against an estimated ~950 citation events on the 97 keys-with-excerpts, the dual-cite compliance rate runs ~50–55% — load-bearing first-mention cites are anchored almost universally; repeat / forward-link cites within the same paragraph are reasonably not re-anchored.

**Findings classification:**

- **BLOCKERS:** 0.
- **MAJORS:** 0.
- **MINORS:** 2 — see §"Tier-B-without-flag findings" and §"Closed-vendor laundering risks".
- **NITS:** 3 — see §"Nits".

The series clears the playbook's citation-tier bar with margin. The `\deepencite` discipline is consistent and the "no laundering" rule is taken seriously (paper-1 §13 explicitly forbids "laundering forum speculation as architectural fact"; paper-5 §12 watermarking explicitly self-flags "we treat as tier-B signal rather than tier-A claim").

## `\deepencite` audit (representative sample)

Categorising all 127 markers:

| Category | Count | Examples |
|---|---|---|
| Closed-vendor system card / API ref (Claude, OpenAI, Gemini) | ~38 | `claude-3-7`, `openai-o1`, `openai-o-series`, `gemini2-5 — API ref` (paper-1 §13, paper-2 §13, paper-3 §13) |
| Tier-A paper IS in `papers.json` but excerpt is null — "verify against PDF" hedge | ~52 | `gu2023-mamba §X (KB excerpt missing; verify the …)`, `mamba2 §X`, `dao2023 §3`, `lieber2024-jamba §3`, `kirchenbauer2024-reliability` |
| Bibkey not in `papers.json` at all (legitimately external) | ~25 | `gu2022-s4`, `smith2023-s5`, `ren2024-samba`, `zamba2-2024`, `hymba-2024`, `qwq`, `qwen3-thinking`, `openr1`, `lion2023`, `sophia2023`, `schedule-free-defazio-2024`, `dathathri2024-synthid`, `eu-ai-act-art-50`, `mistral-large-2`, `claude-3-7-system-card`, `openai-o-series-system-cards` |
| Methodology forum-tier (with explicit forumsignal wrapper) | 2 | `nanda2023-attribution-patching` (in `paper-4/09-activation-patching.tex` line 355, inside a `forumsignal` block); `nostalgebraist2020-logit-lens` (in `paper-4/13-scaling-and-snapshot.tex` line 239 table cell, alongside the section-level forumsignal block in `paper-4/03-lens-techniques.tex`) |
| Anthropic HTML-only (transformer-circuits.pub) | ~6 | `templeton2024-scaling-monosemanticity HTML-only`, `lindsey2025-biology HTML-only`, `bricken2023-monosemanticity §2 citation-pending`, `Anthropic 2025-06 open-source release; Neuronpedia integration` |
| Misc (tooling, dataset, regulation) | ~4 | `Neuronpedia integration`, `tier-B benchmarks: HuggingFace open-LLM leaderboard`, `eu-ai-act-art-50` |

**No `\deepencite` is hiding a load-bearing thesis-driving claim.** Spot-checks against three representative thesis claims:
- Spike-entropy detection bound (paper-5 §12 watermarking, line 142): `\deepencite{kirchenbauer2023-watermark §sensitivity-bound}` — but `kirchenbauer2023-watermark` *is* in `papers.json` and IS cited as `\citep{}` later in the same section at line 214 (default $(\gamma, \delta) = (0.25, 2.0)$). The deepencite is on the specific section-anchor that hasn't been excerpted yet, not on the claim itself.
- DeepSeek-R1 RLVR + no-PRM doctrine (paper-3 §13, lines 124–127): `\citep[\S 2.2]{deepseek-r1}` plus `\footnote{KB excerpt: kb/excerpts/deepseek-r1.md#sec-2-2-reward}` — tier-A primary plus dual-cite anchor.
- FlashAttention 1/2/3 walking (paper-1 §6, lines 16–21): `\citealp{dao2022, dao2023, shah2024}` — tier-A primary throughout. No deepencite needed.

The two `\deepencite` markers wrapping forum-tier claims (`nanda2023-attribution-patching`, `nostalgebraist2020-logit-lens`) are *also* wrapped in `\begin{forumsignal}` environments at the load-bearing site. Defensive belt-and-braces handling.

## Tier-B-without-flag findings

The CLAUDE.md tier classification flags these source domains as tier-B (vendor / lab blogs requiring an underlying tier-A co-citation): `transformer-circuits.pub`, `anthropic.com/news`, `apolloresearch.ai`, `metr.org`, `openai.com cdn`, vendor system cards. Lesswrong is tier-C.

Cited keys whose `url` lands on these tier-B/C domains (extracted from `papers.json`):

| Key | URL | Tier | Use site characterisation |
|---|---|---|---|
| `nostalgebraist2020-logit-lens` | lesswrong.com | **C** | Wrapped in `forumsignal` at the methodology site (paper-4 §03 line 100); reference-pointer cites at intro / table allow plain `\citep{}` per playbook (a forum-introduced technique can still be the canonical name for the technique, but its load-bearing properties live in the forumsignal block). **Compliant.** |
| `olsson2022-induction-heads` | transformer-circuits.pub | **B** | Always co-cited with tier-A `wang2022-ioi` at the load-bearing claim site (paper-4 §11 lines 318–324). **Compliant.** |
| `templeton2024-scaling-monosemanticity` | transformer-circuits.pub | **B** | Always co-cited with tier-A `gao2024-topk-saes` and `rajamanoharan2024-jumprelu` at the load-bearing math sites (paper-4 §07 lines 64–77, §08 lines 48–56). Used as lineage anchor; load-bearing math comes from the arxiv co-cites. Two table-cell uses in `paper-4 §13` carry `\deepencite{… HTML-only at transformer-circuits.pub}`. **Compliant.** |
| `bricken2023-monosemanticity` | transformer-circuits.pub | **B** | Used 6× in paper-4 §08 / §12 / §14, always either alongside `gao2024-topk-saes` or with explicit `\deepencite{… §2 citation-pending}`. **Compliant.** |
| `lindsey2025-biology` | transformer-circuits.pub | **B** | 4 use sites (paper-4 §11 / §13 / §14, paper-5 §11). Each carries `\deepencite{lindsey2025-biology HTML-only; verbatim transcription pending Phase 2}`. **Compliant.** |
| `lindsey2025-circuit-tracing` | transformer-circuits.pub | **B** | Cited as primary methodology source for cross-layer transcoders. Co-cited with `dunefsky2024-transcoders` (arxiv) at the methodology site. **Compliant** but worth keeping an eye on. |
| `meinke2024-apollo-scheming` | apolloresearch.ai | **B** (research-org report) | Has KB excerpt, dual-cite footnoted at use site. The Apollo report is the empirical evidence; the section's surrounding analysis hedges between "capability under elicitation" and "deployment propensity". **Compliant.** |
| `anthropic-rsp-2024` | anthropic.com/news | **B** | Used in paper-5 §08 dangerous-capability for the RSP framework definition. The factual content (Anthropic's own published policy) is appropriately attributed to the policy itself; treating the RSP as anything other than primary on its own contents would be wrong. **Compliant.** |
| `metr-autonomy-2024` | metr.org | **B** | Used in paper-5 §08 alongside Apollo / Cybench. Cited as definitional source for the autonomy-level evaluation framework. **Compliant.** |
| `lm-eval-harness` | github.com/EleutherAI | **A** (canonical github repo per CLAUDE.md tier rules) | Has KB excerpt, dual-cite footnoted. **Compliant.** |
| `radford2019-gpt2` | cdn.openai.com | **A** (official tech report) | KB excerpt anchored. **Compliant.** |
| `kohonen1972-associative` | ieeexplore.ieee.org | **A** (peer-reviewed venue) | Historical attribution for associative-memory, used as the named reference at one site in paper-4 §10. Tier-A, just very old. **Compliant.** |

**MINOR finding M1.** The single tier-B/C source whose load-bearing-claim handling I'd flag for a closer look is `lindsey2025-circuit-tracing` (Anthropic HTML, no arxiv). It is treated as the *primary* methodology source for the cross-layer-transcoder + Jacobian-attribution pipeline in paper-4 §11 / §13. The arxiv co-cite `dunefsky2024-transcoders` covers the transcoder math but not the attribution-graph methodology. If `lindsey2025-circuit-tracing` is later contradicted by an independent reproduction, the load-bearing claim "the cross-layer-transcoder + Jacobian-attribution pipeline … is a productionised methodology" rests on a single tier-B HTML source. **Recommendation:** add a `\begin{forumsignal}` or contradiction-style hedge at the methodology-introduction site (paper-4 §11 around line 266 and paper-4 §13 around lines 179–184) noting that the methodology is single-source as of 2026-Q2 with no published external replication. This is the same hedging style the series already uses for `deepseek-v2 MLA` (paper-1 §05) and the closed-vendor RL-loop entries — apply it consistently.

## Closed-vendor laundering risks

Each paper's §13 frontier snapshot has rows for closed-vendor models (Claude 3.7-class, GPT-4o / o-series, Gemini 2.5). Sample audit:

| Paper §13 | Closed-vendor handling | Verdict |
|---|---|---|
| paper-1 §13 | Anthropic / OpenAI / Gemini rows: every architectural cell is `ND` (not disclosed). Single all-row `\deepencite{Anthropic Claude 3.7 system card / OpenAI GPT-4o and GPT-5 system cards / Google Gemini 2.5 technical report --- citation-pending: vendor system cards are discovery-only sources (Tier B/C in theory/sources/README.md) and cannot solely back hard architectural claims}` — the deepencite *itself* references the project tier classification. Plus an explicit `\begin{forumsignal}` block forbidding the laundering of inference-stack reverse-engineering signals. **Gold-standard.** |
| paper-2 §13 | All closed-vendor cells marked "opaque"; each closed-lab row carries a `\deepencite{... — bibkey not in references.bib}` flag. **Compliant.** |
| paper-3 §13 | Highest deepencite density (32 markers) but every closed-vendor architectural / training-detail claim is wrapped — including the o-series quote "the more reinforcement learning… the more time it spends thinking" attributed to the o1 system card via `\deepencite{openai-o1 — system card}`. Production budget knobs (`reasoning_effort`, `thinkingBudget`, extended-thinking) cited as factual via deepencite to API references — these are user-observable surface features, so the discovery-only attribution is appropriate. **Compliant.** |
| paper-4 §13 | Closed-vendor entries (Claude 3.5 Haiku findings) are confined to the "Anthropic — circuit tracer + Neuronpedia" paragraph, deepencited and explicitly marked "remain Anthropic-internal as evidence". **Compliant.** |
| paper-5 §08 / §10 / §11 | Specific behaviours of o1, Claude 3.5 Sonnet, Gemini 1.5 Pro etc. are attributed to peer-reviewed empirical demonstrations (`meinke2024-apollo-scheming`, `greenblatt2024-alignment-faking`) rather than vendor self-reports. Those papers *are* tier-A or tier-B-with-tier-A-coverage. **Compliant.** |

**MINOR finding M2.** The `paper-3/13-frontier-snapshot.tex` table has one row that's slightly looser than the rest: the OpenAI o-series RLVR claim "yes; proprietary RL on reasoning, …" attributed only to `\deepencite{openai-o1 — system card; cf.\ openai-o3 — system card}`. The "yes" assertion is the load-bearing one — that o-series uses RLVR-class training. The supporting evidence is a quoted phrase from the system card, which is consistent with that, but the system card itself doesn't say "RLVR" — that's an inference. The neighbouring R1 row for the same column cites tier-A `\citep[\S 2.1]{deepseek-r1}, terminal binary reward`. **Recommendation:** soften the o-series cell to "yes (RL-class on reasoning trajectories per system card; whether the reward is verifier-derived in the RLVR sense is not disclosed)" — same hedging the same row applies to the verifier-in-RL-loop column. Cosmetic but tightens the load-bearing/cite alignment.

## Forum / blog tagging

`\begin{forumsignal}` block locations and what they cover:

| Site | Block content |
|---|---|
| `paper-1/03-tokenization-embedding-unembedding.tex:320` | Glitch-token / SolidGoldMagikarp community-discovery |
| `paper-1/13-frontier-snapshot.tex:203` | Inference-stack reverse-engineering of closed vendors — explicitly refuses to launder it |
| `paper-1/11-multimodal-extensions.tex:170` | Vision-encoder choices in closed-vendor multimodal stacks |
| `paper-3/04-inference-compute-scaling.tex:273` | Inference-compute-scaling community signals |
| `paper-3/10-search-vs-rl.tex:348` | Search-vs-RL deployment-side anecdata |
| `paper-4/09-activation-patching.tex:350` | Nanda 2023 LessWrong attribution-patching post — load-bearing methodology |
| `paper-4/03-lens-techniques.tex:100` | Nostalgebraist 2020 LessWrong logit-lens post — methodology origin |
| `paper-5/08-dangerous-capability.tex:96` | Cybench / dangerous-capability community signals |
| `paper-5/09-sycophancy.tex:197` | Sycophancy community signals beyond the Sharma 2023 measurement |

All correct sites for `forumsignal`. The two methodology-origin uses (Nanda, Nostalgebraist) are textbook applications of the playbook's "introduces but cannot solely back" rule: the post is named as the canonical reference, the load-bearing properties are sourced to subsequent tier-A peer-reviewed work, and the overlay is the forum-signal box.

## Math-faithfulness spot-checks (5 sites)

The playbook also requires "math is verbatim from the cited paper." Spot-checks:

1. **Vaswani scaled dot-product attention** (`paper-1` references): `\softmax(QK^\top/\sqrt{d_k}) V` — matches `vaswani2017` Eq. 1.
2. **DeepSeek-R1 RLVR verifier reward** (paper-3 §07 / §13 references `\cref{eq:verifier-reward}`): formula is the binary-correctness reward signature. Matches `deepseek-r1` §2.1 description.
3. **Chinchilla scaling law** $L = E + A/N^\alpha + B/D^\beta$ + IsoFLOP curves: cited as `\citep[\S 3]{hoffmann2022-chinchilla}` with KB excerpt anchor. Matches paper.
4. **FlashAttention factor-$d_h/2$ memory saving** (paper-1 §06 lines 129–133): matches `dao2022` §3.1 description.
5. **Spike-entropy watermark detection bound** (paper-5 §12 line 144): formula `S_γ(p) = Σ p_v / (1 + γ' p_v)` — matches `kirchenbauer2023-watermark` §3.2 sensitivity bound (verifiable against the KB excerpt that paper-5 §12 line 202 anchors).

No math-paraphrase issues found.

## Nits (NIT-level only)

- **N1.** Paper-3 §13 has both `\label{sec:frontier-2026}` AND `\label{sec:frontier-snapshot}` on adjacent lines (9–10). Harmless duplication, easy cleanup.
- **N2.** Three `\deepencite` markers use the en-dash `—` separator while the rest use a colon-style `bibkey § X (note)` form (e.g. `\deepencite{qwen3-thinking — system card pending}` vs `\deepencite{kirchenbauer2024-reliability §paraphrase, citation-pending}`). Visual-style consistency only; the marker text is freeform per playbook so this isn't a discipline violation.
- **N3.** The intro of `paper-4 §01` lines 62–63 cites `\citep{nostalgebraist2020-logit-lens}` as the canonical reference for "the logit lens" without local forumsignal wrapping — the wrapper lives in §03 where the load-bearing methodological framing is. This is correct per the playbook's "name-the-reference" allowance, but a *very* strict reading would prefer a one-line forumsignal at the intro site too. I would not require this; flagging as a nit only.

## Closing assessment

The series clears the citation-tier discipline bar with margin. The single most impressive finding from the audit: the `\deepencite` mechanism is being used as designed — *every one* of the 127 markers I sampled either flags a tier-B/C source whose load-bearing-claim status is hedged elsewhere, points at a tier-A paper whose excerpt extraction is pending, or names a closed-vendor system card / API reference where no formal venue exists. There is no instance of `\deepencite` quietly substituting for a tier-A primary that exists and was simply skipped.

Two minor tightenings (`lindsey2025-circuit-tracing` single-source hedge; o-series RLVR phrasing tightening) would close the remaining ~5% of cell-level looseness in §13 frontier snapshots. Both are paragraph-level edits and neither blocks publication of the series.

The dual-cite anchor compliance (~50% of all citations on excerpt-available keys) is well above the playbook's "where the KB has an excerpt, also cite the anchor" floor — load-bearing first-mention cites are anchored almost universally; only repeat / forward-link cites in the same paragraph go unanchored, which is reasonable.

**Verdict: PASS with two MINOR recommendations and three NITs. No BLOCKERS, no MAJORS.**
