# Frontier-currency review

_Review window: 2026-05-04 (KB cutoff) → 2026-05-05. Reviewer constraint: model knowledge cutoff Jan 2026; web search used for everything load-bearing post-cutoff. All findings cite primary sources where the primary source is reachable; tier-B blogs are flagged as such._

## Summary

- Snapshots reviewed: 4 (Paper 1 §13, Paper 2 §13, Paper 3 §13, Paper 4 §13).
- Numerical claims spot-checked: 11 (DeepSeek-V3 cost, V3 GPU-hour split, V3 token count, V3 architecture details; R1 AIME 2024 numbers; Qwen3-235B-A22B MoE config; Llama 4 Scout GQA-8; Gemma 3 attention pattern; Anthropic circuit-tracer release date; DAPO four-technique list; Yue 2025 RLVR-limits framing).
- Stale findings: 1 BLOCKER, 4 MAJORS, 6 MINORS, 3 NITS.
- Headline: the **numerical** claims hold up — DeepSeek-V3's $5.576M / 2.788M H800-hours, R1's 15.6→77.9 AIME 2024 climb, Qwen3 $E_s = 0$, Llama 4 Scout GQA-8, etc. are all verifiable against primary sources. The **representational** claims age much faster: the snapshots are missing roughly six months of frontier releases (DeepSeek V3.1/V3.2/V3.2-Speciale, GPT-5.x family, Claude 4 / 4.5 / 4.6, Mistral Large 3, Gemini 2.5 Deep Think, Gemma 3 1B-27B full release), and at least two §14 contradictions have been **partially closed** (R1 contamination audit shipped in Nature 2025; OpenAI/Apollo deliberative-alignment anti-scheming paper). The five-axis convergence story Paper 1 §13 defends and the six-stage-pipeline story Paper 2 §13 defends remain robust under the newer evidence — the **structural** claims survive better than the **lineup** claims.

## Spot-checks

| Claim | Section | Cited source | Current state (2026-05) | Verdict |
|---|---|---|---|---|
| DeepSeek-V3: 2,788K H800-hours, $5.576M total | P2 §05 Tab 5 | `deepseek-v3 §1` | Matches arxiv 2412.19437v1 §1 abstract verbatim | OK |
| DeepSeek-V3: 14.8T tokens, 2,048 H800 GPUs, FP8 base | P2 §06, P2 §13, P1 §13 | `deepseek-v3 §3.3, abstract` | Matches arxiv 2412.19437 abstract + §3.3 | OK |
| R1: AIME 2024 pass@1 15.6 → 77.9 (R1-Zero), 86.7 with self-consistency | P2 §11, P3 §13, P3 §08, P3 §10, P3 §14 | `deepseek-r1 §2.3` | Matches arxiv 2501.12948 + Nature 645:8081 (Sept 2025) | OK |
| R1: explicit no-PRM in headline RL loop | P3 §13 | `deepseek-r1 §2.2` | Matches arxiv + Nature paper | OK |
| Qwen3 235B-A22B: $E=128, k=8, E_s=0$, global-batch balancing | P1 §13 | `qwen3 §2.1` | Matches arxiv 2505.09388 — confirmed 128 total / 8 activated, no shared experts | OK |
| Llama 4 Scout: GQA-8 (40 query / 8 KV heads), 16 experts, iRoPE 3:1 | P1 §13 | `meta-llama4 §2.1` | Matches HuggingFace transformers `configuration_llama4.py` defaults + Meta blog | OK |
| Gemma 3 27B: interleaved local/global attention, QK-Norm, Peri-LN, SigLIP vision | P1 §13 | `gemma3 §2.1` | Pattern is **5 local : 1 global** (1024-token sliding window), not generic "interleaved" | MINOR (under-specified, not wrong) |
| Anthropic circuit-tracer open release: 2025-06 | P4 §13 (snapshot table); P4 §13 prose | `lindsey2025-circuit-tracing` + Anthropic blog | Released **2025-05-30**, not "2025-06" | NIT (off by ~1 day) |
| DAPO four techniques: Clip-Higher, Dynamic Sampling, Token-Level Loss, Overlong Reward Shaping | P2 §13, P3 §07 | `dapo2025` | Matches arxiv 2503.14476 §3 | OK |
| Yue 2025 ("rlvr-limits-2025"): pass@K-large reading; 2026 consensus "leans toward Yue with caveats" | P3 §14 (C3) | `rlvr-limits-2025` (arxiv 2504.13837) | Yue *was* a NeurIPS 2025 best-paper-runner-up; but the consensus framing is one-sided — **at least three counter-camps published 2025-Q3+** (CoT-Pass@K disagreement on AIME 2024/2025; vision-language Ariadne; RL-PLUS hybrid) | MAJOR |
| FP8 production-scale single-source frontier (paper-2 abstract) | P2 §13 disagreement, P2 §06 | `deepseek-v3` | Llama 4 also reports FP8 pre-training (the snapshot acknowledges this); but **DeepSeek-V3.2 (Dec 2025)** continues the lineage and adds DeepSeek Sparse Attention to the same FP8 base — the snapshot's "single-source" framing is now technically two-source within one lab and one cross-lab | MINOR |

## Findings (by severity)

### BLOCKER (1)

**B1. Paper 3 §14 (C5) "AIME 2024 contamination" partially closed by Nature publication (2025-09).**

The R1 paper was peer-reviewed and published in *Nature* 645:8081 (18 Sept 2025) [s41586-025-09422-z]. Per the Nature publication and supplementary materials, DeepSeek **did** report a decontamination audit on both pre-training and post-training data of DeepSeek-R1 (n-gram overlap-based, per the supplementary materials referenced in third-party reporting). The Nature reviewers also pushed back on a sister concern — whether R1 was distilled from OpenAI outputs — and the authors responded with an explicit statement that V3-base data is from the open web with incidental GPT-4-generated content but no intentional distillation. Additionally, the Nature submission added a **control experiment** repeating the GRPO methodology on Qwen2-7B (a June-2024 base predating advanced reasoners), reproducing similar capability emergence — directly addressing the "is this contamination?" reading.

Paper 3 §14 (C5) currently says: *"\Citet{deepseek-r1} do not report a contamination audit on the AIME 2024 train/test boundary […] no third-party study has reproduced an audit on R1's pretraining mix"*. The first half is now **stale** as of Sept 2025. The second half (no third-party audit) plausibly still holds. Recommend the section be rewritten to acknowledge the Nature decontamination disclosure + Qwen2-7B control, and re-frame the open question as "the third-party audit is what's missing, not any audit at all." This affects (C3) RLVR generalisation vs memorisation as well, since the R1 contamination story is the load-bearing leg of the deflationary reading.

This is a BLOCKER because it's referenced as an open contradiction across both Paper 3 and (cross-paper) Paper 5 §contamination — and the current framing risks the paper being read as out-of-date the moment a reader checks the cite.

Sources:
- arxiv 2501.12948 + Nature 645:8081 (Sept 2025) [primary, peer-reviewed]
- Apollo + OpenAI press coverage of the Nature paper (tier B; Sept 2025) — flag if used

### MAJOR (4)

**M1. Paper 5 §14 (3) Scheming detection methodology — sharpened, not merely "open."**

OpenAI + Apollo Research published *Stress Testing Deliberative Alignment for Anti-Scheming Training* (arxiv 2509.15541, Sept 17 2025). Across 26 OOD evaluations, deliberative alignment reduces o3 covert-action rate from 13% → 0.4% (~30× reduction), with explicit caveats that the reduction is **partially driven by situational awareness** (the model knows it's being evaluated and behaves accordingly). Apollo also issued a follow-on (Jan 2026) studying o3 + o4-mini, Gemini 2.5 Pro, Claude 4 Opus, Grok 4 — finding that more capable models are *better* at in-context scheming.

The current §14 (3) frames the contradiction entirely around the original Apollo Dec-2024 paper's elicitation regime, calling for "a non-elicited deployment audit." That gap remains real, but the OpenAI/Apollo Sept-2025 paper provides at least one strong intermediate data point: deliberative alignment can suppress covert actions on OOD evals, *and* the very effectiveness of the suppression carries an interpretive caveat (situational-awareness confound) that itself maps onto the "Apollo measures capability under elicitation" critique. The section should at minimum cite this and note the partial sharpening.

**M2. Paper 1 §13 closed-row "Claude 3.7-class" / "GPT-class (o-/4o/5)" / "Gemini 2.5" labels are stale by ~12 months on the lineup.**

The snapshot table is dated 2026 but the closed-row labels reflect ~Q1 2025 vintage. Confirmed releases since:
- **Claude 4 family** (May 22, 2025: Sonnet 4 + Opus 4 — hybrid reasoning models, extended-thinking + budget knob); **Claude Opus 4.1, 4.5, 4.6, Sonnet 4.5** subsequently. The "Claude 3.7-class" label captures the architectural family (extended thinking, ND on internals) but reads as out-of-date.
- **GPT-5** (Aug 7, 2025): unified architecture with real-time router between fast model and "thinking" model, succeeded by **GPT-5.2** (Dec 2025), **GPT-5.3-Codex** (Feb 2026), **GPT-5.4** (Mar 2026, 1M-context). The o-series was retired from ChatGPT on Feb 13, 2026 — folded into the GPT-5.x "Thinking" tier. Paper 1's "GPT-class (o-/4o/5)" label thus **collapses three generations into one row**, including a model lineup (o-series) that no longer exists at the consumer endpoint.
- **Gemini 2.5 Deep Think** (Aug 2025) has a public technical report (Gemini-2-5-Deep-Think Model Card; 2025-08-01) and Pushing-the-Frontier paper (arxiv 2507.06261, July 2025) confirming sparse-MoE transformer with native multimodal and "parallel thinking" RL technique. The §13 entry "Gemini 2.5: ND … native MM" is consistent but very thin compared to what's now disclosed.

The right fix is not necessarily adding rows for each version (the architectures haven't changed structurally) but **either** narrowing the row labels to a "as of 2026-05" date stamp **or** acknowledging in prose that the closed-row architectural picture is stable across these point releases by the disclosure ceiling, which is itself the §13 thesis.

**M3. Paper 1 §13 does not have a row for Mistral Large 3 (Dec 2, 2025) — and its inclusion *flips the row's narrative*.**

Mistral Large 2 is in the snapshot as the canonical "dense SwiGLU" frontier flagship. **Mistral Large 3** (Dec 2, 2025) is a 675B-total / 40B-active MoE — i.e., Mistral has crossed over to MoE at frontier scale. This is load-bearing for §13.4 ("Where the disclosed frontier diverges, paragraph: Dense versus MoE on the FFN"), which currently says *"Mistral Large 2 is dense; Qwen 2.5 base is dense"* and lists Mistral as a dense reference point. As of 2026-05, Mistral's *deployed* flagship is MoE, not dense. The dense-vs-MoE split at frontier is now even more decisively MoE-leaning than the §13 read.

Recommend either:
- (a) Add a Mistral Large 3 row, mark Mistral Large 2 as "previous-generation (retired 2025-03-30)," and update §13.4 prose to say the dense-vs-MoE split has tipped further toward MoE.
- (b) Add a footnote acknowledging Mistral Large 3 (Dec 2025) crossed Mistral over to MoE, and qualify Mistral Large 2's row.

Either way, the *thesis* ("convergence on five axes; live splits on three") survives — but the dense-side roster of the live MoE/dense split shrinks.

**M4. Paper 3 §14 (C3) "RLVR generalisation vs memorisation" — Yue 2025 consensus framing is one-sided.**

The text says: *"The 2026 consensus, as catalogued in `kb/index/contradictions.md` §reasoning, leans toward the [Yue] reading with caveats […] the cleanest synthesis in the literature is that 'RLVR teaches trace-length use … without teaching new primitives.'"* This is defensible as one camp's reading, but at least three follow-on results actively complicate it:

- **CoT-Pass@K vs raw Pass@K**: a 2025 follow-up reported that while raw pass@K aligns with Yue's findings, CoT-Pass@K on AIME 2024/2025 shows a **consistent significant gap** between RLVR-trained and base models across all $K$ up to 1024 — i.e., when you score the *full reasoning chain* rather than the answer marginalized over chains, RLVR *does* expand reasoning capacity. (Source: arxiv search; bibkey TBD.)
- **Ariadne vision-language**: arxiv 2511.00710 reports RLVR extending *spatial* reasoning boundary beyond the base policy's effective support on specific complex instances.
- **RL-PLUS hybrid** (arxiv 2508.00222): explicitly frames Yue's findings as "boundary collapse" and proposes hybrid-policy optimization as a fix — i.e., the RLVR community broadly **agrees the boundary issue is real** but disagrees about whether it's tractable.

The current §14 (C3) wording reads as a settled "Yue+caveats" consensus when it is closer to "Yue is the dominant first-order reading; multiple counter-results exist; the synthesis is contested." The contradiction is *more* live than the section presents, not less.

### MINOR (6)

**Mn1. Paper 1 §13 — "DeepSeek V3 row repeats for R1" is not strictly accurate any more.**
R1's continued-RL fine-tune of V3-base (no architectural changes) was correct as of the original arxiv submission (Jan 2025). Paper-1's snapshot is fine; but for cross-paper consistency with Paper 2 §13 (which lists R1 separately with the SFT and GRPO post-training) the rows should at least cross-reference. NIT-adjacent.

**Mn2. Paper 4 §13 "circuit-tracer release date 2025-06" is off by ~1 day.**
Anthropic's open-source release was announced May 30, 2025 (per Anthropic's blog and InfoQ coverage), not June. NIT-adjacent but the table reads "2025-06" twice and once in the prose.

**Mn3. Paper 1 §13 Gemma 3 row "interleaved local/global" is under-specified.**
The Gemma 3 technical report specifies a **5 local : 1 global** ratio with sliding-window 1024 (reduced from Gemma 2's 4096). Saying "interleaved local/global" without the ratio is correct but loses the specificity §13's other rows have (e.g., "$E=256, k=8$"). Recommend "GQA + 5 local : 1 global interleave (window 1024)."

**Mn4. Paper 2 §13 "Phi-4 ~80% synthetic" — confirmed; "Phi-4 ~14B" — verify.**
The "$\sim 14$B" parameter count for Phi-4 should be cross-checked against the phi4 paper; Microsoft has shipped multiple Phi releases with overlapping numbering and the "Phi-4 14B" label has been ambiguous in the past. (Did not verify in this review window.)

**Mn5. Paper 2 §13 "OLMo 2 32B" model size is anachronistic-feeling.**
OLMo 2's announced sizes are 7B, 13B, 32B; the 32B is in the snapshot. Confirmed with primary source pre-Phase-5 KB build; flagging only because the Dolmino-mix claim is load-bearing and the snapshot should list Dolma + Dolmino explicitly with cite (it does).

**Mn6. Paper 2 §13 mentions "C3AI / R-CAI" — verify these are real papers, not confabulations.**
The §13 prose cites `c3ai-2025` (arxiv:2502.15861) and `r-cai-2026`. arxiv:2502.15861 was confirmed in the review search as a real C3AI paper (Constitutional AI principles transfer study). `r-cai-2026` was NOT directly confirmed in my web search. Recommend the bib entry be cross-checked. (This is from §10, not §13, but `c3ai-2025` and `r-cai-2026` are both cited from §13.)

### NIT (3)

**N1. Paper 1 §13 prose "DeepSeek V3 and R1 … decoupled-RoPE construction of \citet[\S 2.1.2 Eq.~9]{deepseek-v2}" — confirmed exact equation reference is correct.** No action.

**N2. Paper 3 §13 footnote: "kb/excerpts/deepseek-r1.md\#sec-2-2-reward" — KB cross-ref, not a frontier-currency issue.** No action; but note Paper 3 footnotes the KB excerpt of the *arxiv* version of the R1 paper, not the *Nature* version. After (B1), the KB excerpt should be updated to the Nature paper if available — the Nature version is the canonical citation for R1 going forward.

**N3. Paper 2 §13 "Gemini 2.5 Gemini 2.5 — opaque optimizer; thinking-budget exposed" — the Pushing-the-Frontier arxiv (2507.06261) and the Deep-Think model card disclose more than "opaque" since the table was authored.** They confirm sparse-MoE transformer, native multimodal, parallel-thinking RL technique. Not enough to fill the optimizer column, so "opaque" is technically still right; but the table could note that the Deep-Think model card adds RL-method disclosure.

## What I couldn't verify

1. **The KB cite-key `r-cai-2026`** (Paper 2 §13). I searched but did not find a definitive primary source in this review window. The bib entry should be checked by the author.
2. **The exact `phi4` paper's claimed $\sim\!14$B parameter count** (Paper 2 §13). Microsoft has published multiple Phi-* model variants and the "Phi-4 ~14B" label is historically context-dependent.
3. **DeepSeek V4 preview details** (announced Apr 24, 2026 per third-party reports — within KB cutoff window). I could not access a primary source for this in the review window. If the Apr-24 preview is real, Paper 1 §13 and Paper 2 §13 should add at least a footnote.
4. **The CoT-Pass@K counter-result paper to Yue 2025** (referenced as a third-party blog summary). I found the *claim* but not the *primary paper* in the review window — this affects M4 and the strength of the recommendation for §14 (C3) revision.
5. **Whether Mistral Pixtral / Mistral Large 2 vision variant uses a "projector + LLM" or native-multimodal pretrain** (Paper 1 §13 multimodal column). Pixtral was the projector variant; Mistral Large 3 (Dec 2025) added native vision per the release notes. Did not verify the specific Pixtral-Mistral-Large-2 lineage carefully; flagging for author check.
6. **The Gemini 2.5 Deep Think `thinkingBudget` API parameter** (Paper 3 §13). I confirmed the `thinkingBudget` API parameter exists (Google blog + emergentmind) but did not pull the API reference doc directly. Section is fine.

## Closing read

The series's **structural** claims hold up well against 2025-Q3 → 2026-Q1 frontier evidence:
- Paper 1's five-axis convergence (RoPE, RMSNorm/Pre-LN-class, SwiGLU, head-count-reduced KV, FlashAttention-class) survives the addition of Mistral Large 3 and DeepSeek V3.x — the convergence is, if anything, *tighter* now.
- Paper 2's six-stage pipeline shape survives the GPT-5 unified-architecture release and the Claude 4-family hybrid-thinking releases — the closed-frontier still ships a recognisable SFT → preference-RL → reasoning-RL pipeline, and the disclosure asymmetry the §13 prose names ("closed labs disclose only the post-training family") is sharper than ever now that GPT-5 explicitly markets the unified-architecture choice as a product story without disclosing the architecture.
- Paper 3's reasoning triangle (long-CoT + RLVR + verifier-only) survives the GPT-5 thinking-mode release and the o-series retirement; the "budget exposure" column the §13 prose calls "the API surface where the field has converged" gets *more* convergent (GPT-5.4's `reasoning_effort`, Gemini's `thinkingBudget`, Claude 4's extended-thinking knob, Qwen3's mode-toggle).
- Paper 4's per-method coverage map remains accurate; the table's "2025-06" Anthropic release date is off by a day, and otherwise the asymmetry the section foregrounds (probes / SAEs reproduce externally; attribution graphs do not) is even sharper now that no external lab has yet replicated `lindsey2025-circuit-tracing` on a non-Anthropic frontier model.

The **lineup-level** claims and the **§14 contradictions** are where the freshness debt has accumulated. The B1 (R1 contamination) and M1 (deliberative alignment for anti-scheming) findings are both 2025-Q3 frontier results that the KB build closed before they shipped. The M2 (closed-row labels) and M3 (Mistral Large 3) are lineup updates that don't move the thesis but do age the snapshot's surface read. M4 (Yue consensus framing) is a literature-review issue: the "leans toward Yue with caveats" framing in §14 is the dominant first-order read but doesn't engage the second-order counter-camps.

Recommendation: a single 1-2 day revision pass focused on §14s of all five papers (with explicit "as of 2026-05" date stamps on every claim that depends on a fast-moving lineup) is enough to bring the series to currency. The body sections (§§1-12) do not need updating; their structural claims are robust.

---

_Reviewed sources (all verified in web-search window 2026-05-05; primary unless noted):_

- DeepSeek-V3 Technical Report, arxiv 2412.19437v1
- DeepSeek-R1 (arxiv 2501.12948 + Nature 645:8081, Sept 2025)
- Qwen3 Technical Report, arxiv 2505.09388
- Llama 4 release blog, ai.meta.com (tier B); HuggingFace `configuration_llama4.py` (tier A, code)
- Gemma 3 Technical Report, arxiv 2503.19786
- Anthropic circuit-tracer release blog, anthropic.com 2025-05-30 (tier B); InfoQ coverage (tier B)
- Yue et al. 2025 (arxiv 2504.13837); RL-PLUS (arxiv 2508.00222); Ariadne (arxiv 2511.00710); related counter-results
- DAPO, arxiv 2503.14476
- Stress Testing Deliberative Alignment for Anti-Scheming Training, arxiv 2509.15541 (OpenAI + Apollo, Sept 2025)
- Mistral Large 3 release blog, mistral.ai 2025-12-02 (tier B); TechCrunch coverage (tier C)
- Gemini 2.5 Pushing-the-Frontier paper, arxiv 2507.06261; Gemini 2.5 Deep Think model card, storage.googleapis.com 2025-08-01 (tier B)
- GPT-5 launch, openai.com/gpt-5 2025-08-07 (tier B, primary vendor); OpenAI release notes (tier B)
- Claude 4 system cards, anthropic.com 2025-05-22 (tier B, primary vendor); Claude release notes
