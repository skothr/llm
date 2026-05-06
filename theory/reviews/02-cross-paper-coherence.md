# Cross-paper coherence review

_Review pass: independent reviewer, 2026-05-05. Scope: 70 sections across
paper-1 through paper-5 of `theory/series/`. Read-only review; no source
modified._

## Summary

- **Cross-paper threads checked.** All six named threads from
  `SHAPE-C-decision.md`: MoE (P1→P2), MLA (P1→P4), CoT (P3→P4, P3→P5),
  process supervision (P3→P5), scaling laws (P2→P3), and split-RLVR
  (P2 §11 / P3 §7).
- **Notation grade: B+.** Canonical shape macros (`\BSH`, `\BHSdh`,
  `\bsv`) hold inside Paper 1 and the few Paper 3/4 sections that opt
  in, but Paper 4 drifts to inline `\mathbb{R}^{B \times S \times D}`
  (plain italics, no `\mathsf`, no macro) in §1, §3, §4, §5, §9, while
  §2's notation table claims to use the canonical convention. Paper 2
  oscillates between `C_{\text{test}}` and `C_{\text{infer}}` for the
  same quantity. No drift on $\beta$, $N$, $D$, $C$, $H_{\mathrm{kv}}$.
- **Findings.**
  - **BLOCKERS:** 0
  - **MAJORS:** 2 (Paper 3 cites wrong Hoffmann equation number 3×;
    Paper 3 §7 intuition asserts the contested Yue-limit hypothesis as
    mechanical fact)
  - **MINORS:** 5 (Paper 4 shape-macro drift; Paper 2 internal
    `C_{\text{test}}` vs `C_{\text{infer}}` drift; Paper 5's misuse of
    `\deepencite` for cross-paper section refs ×5 occurrences;
    Paper 5 §11 introduces lanham/turpin via `\deepencite` despite
    Paper 3 §11 being the home of CoT-faithfulness; Paper 4 §10 doesn't
    backref Paper 1 §5 when discussing MLA substrate)
  - **NITS:** 2 (Paper 2 brace-style `C_{\text{test}}` vs Paper 3
    `C_\text{test}`; Paper 1 §7 MoE home doesn't pre-emptively cite
    Paper 2 callers, asymmetric but tolerable)
- **Bottom line.** The series is coherent at the thesis level. The
  RLVR-split between Paper 2 §11 and Paper 3 §7 — the most structurally
  risky cross-paper thread — is the most cleanly executed: matching
  equations, explicit complementary-framing language, distinct
  contradiction blocks pointing to each other. The two MAJORs are a
  3×-occurrence equation-number error in Paper 3 (mechanical fix) and a
  rhetorical overcommitment in a Paper 3 intuition box that the
  surrounding text walks back. Notation drift is mostly Paper 4 not
  using the preamble macros it inherits — fixable with a sweep.

## Findings (by severity)

### MAJOR

**M1. Paper 3 cites Hoffmann's parametric-loss equation as `Eq. 10`;
canonical is `Eq. 2`.**

Three separate Paper 3 locations cite
`\citep[\S 3, Eq.~10]{hoffmann2022-chinchilla}` for the form
$L(N, D) = E + A/N^\alpha + B/D^\beta$:
- `paper-3/sections/04-inference-compute-scaling.tex:328`
- `paper-3/sections/04-inference-compute-scaling.tex:398`
- `paper-3/sections/10-search-vs-rl.tex:205`

Paper 2 §3 transcribes the same equation as `[\S 3.3, eq. 2]`
(`paper-2/sections/03-scaling-laws.tex:161`), with the equation tagged
`\tag{2}` in the LaTeX. The KB excerpt at
`theory/kb/excerpts/hoffmann2022-chinchilla.md:102` confirms the source
labels this equation `(2)`. Paper 3 is wrong on the citation pin.
Mechanical fix: replace `Eq.~10` with `eq.~2` and `\S 3` with
`\S 3.3` in the three locations. The downstream cite to Eq.~3 (Huber
fit) and Eq.~4 (frontier closed-form) in Paper 2 §3 should also
propagate to Paper 3 §4 if it ever needs them.

**M2. Paper 3 §7's intuition box asserts the contested Yue-limit
hypothesis as if it were a mechanical fact.**

Paper 3 §7, intuition (lines 180–183):

> RLVR cannot make the policy solve a problem it has *never* solved at
> any sample budget; it can only sharpen the distribution toward the
> rollouts that already succeed.

This is the Yue et al. (2025) reading of GRPO's group-mean baseline,
treated by both Paper 2 §11 and Paper 3 §14 (C3) as the *contested*
position in an open contradiction. Paper 2 §11 (lines 398–414) presents
deepseek-r1's "incentivizing reasoning capability" framing and
Yue's pass@1-vs-pass@$K$ reading as opposing positions with 2026-Q2
consensus "leans toward Yue with caveats." Paper 3 §14 (C3) lists this
as one of six unresolved contradictions and names the experiment
required to close it.

The intuition box stating it as "RLVR cannot..." is rhetorically
stronger than the sourced position. Paper 3 §7 lines 186–194
immediately walk this back ("Whether this exhausts the picture or
merely captures the dominant term is the open question"), so the
section is internally consistent at the paragraph level — but a reader
skimming the highlighted intuition box absorbs a definitive claim that
contradicts §14. Suggested fix: weaken the intuition's last clause to
"RLVR's reach on a given prompt is bounded *as a first-order claim* by
the base policy's pass@$K$ at any $K$" or relocate the qualifier
inside the box. The math derivation in the box is correct; only the
final assertion overcommits.

### MINOR

**m1. Paper 4 drifts from canonical shape macros.**

Preamble defines `\BSH = \mathsf{B}\!\times\!\mathsf{S}\!\times\!\mathsf{D}`.
Paper 4 §2's notation table (line 288) declares it follows
"the $\mathsf{B} \!\times\! \mathsf{S} \!\times\! \mathsf{D}$ convention
of Paper 1," but in practice:
- §1 line 150: `\mathbb{R}^{B \times S \times D}` (plain italics)
- §3 line 65: `\mathsf{B}\!\times\!\mathsf{S}\!\times\!D` (D loses
  `\mathsf`)
- §4 lines 66–68: same `\mathsf{B}\!\times\!\mathsf{S}\!\times\!D`
- §5 line 173: `\mathsf{B}\!\times\!\mathsf{S}\!\times\!D`
- §9 line 99: `\mathsf{B}\!\times\!\mathsf{S}\!\times\!D`

Paper 2 §5 line 80 uses the full canonical `\mathsf{B}\!\times\!\mathsf{S}\!\times\!\mathsf{D}`
(no macro but correct typesetting). Paper 1 uses `\BSH` as the macro.
Three drift forms (`\BSH`, full mathsf inline, partial mathsf inline,
plain italics) for the same quantity. Suggested fix: sweep Paper 4 to
either `\BSH` or the full mathsf form to match Paper 1/Paper 2.

**m2. Paper 2 internally uses `C_{\text{test}}` and `C_{\text{infer}}`
for the same quantity.**

- `paper-2/sections/03-scaling-laws.tex` line 519, 543, 552: uses
  `C_{\text{test}}` for inference-time-compute budget.
- `paper-2/sections/13-frontier-snapshot.tex` line 299: uses
  `C_{\text{infer}}` for the same quantity in the analogous joint-
  surface phrase.
- `paper-2/sections/14-contradictions.tex` lines 46, 289: back to
  `C_{\text{test}}`.

Paper 3 §4 and §14 use `C_\text{test}` (no braces) consistently. Within
Paper 2, §13 should be normalized to `C_{\text{test}}` to match the
section that introduces the quantity (§3) and the contradictions
chapter (§14).

**m3. Paper 5 misuses `\deepencite` for cross-paper section
references.**

`\deepencite` is defined in `preamble.tex` as a yellow-orange margin
`\todo` for "deepen-cite" (KB excerpt still abstract-only). Paper 5
uses it for known internal cross-paper references in five places:
- `paper-5/sections/04-contamination.tex` lines 196, 234:
  `\deepencite{paper-3 §reasoning-RL}`,
  `\deepencite{paper-3 §AIME-contamination}`
- `paper-5/sections/05-reasoning-benchmarks.tex` lines 61, 285, 300:
  similar pattern

These are not missing-bib markers — Paper 3 §7 and §14 exist and the
sections referenced are the canonical homes. The other four papers use
plain prose ("Paper~3 §X" or `\cref` once compiled together). The
Paper 5 pattern produces orange margin `\todo` boxes in the rendered
PDF on internal cross-paper references that aren't actually missing.
Suggested fix: replace each `\deepencite{paper-3 §X}` with
"Paper~3 §X" prose (the PDF currently builds five papers separately, so
`\cref` cannot reach across; the prose convention is what every other
paper uses for cross-paper section pointers).

**m4. Paper 5 §11 reaches for `lanham2023-cot-faithfulness` and
`turpin2024-cot-not-faithful` via `\deepencite` despite Paper 3 §11
being the CoT-faithfulness home.**

`paper-5/sections/11-alignment-faking.tex` line 263:
`\deepencite{lanham2023-cot-faithfulness, turpin2024-cot-not-faithful}`.

Paper 3 §11 is the load-bearing home for CoT-faithfulness in this
series and cites `arcuschin2025-cot-wild`, `mehta2026-faithfulness-scaling`,
and `shen2025-faithcot-bench` as its primary anchors. It does *not*
cite lanham2023 or turpin2024. Paper 5 §11 also cites
`arcuschin2025-cot-wild` (line 264) — so it's already drawing on
Paper 3's bibliography. The `\deepencite` for lanham/turpin is either
(a) Paper 5 reaching for older anchors that should be added to Paper 3
§11's citation set instead, or (b) Paper 5 should defer to Paper 3 §11
without re-introducing new bibkeys via deepencite. Either is fine; the
mixed pattern is what's incoherent. Suggested fix: pick one — add
lanham/turpin to Paper 3 §11 if they're foundational; or drop them from
Paper 5 §11 and rely on the Paper 3 anchor set.

**m5. Paper 4 §10 doesn't back-reference Paper 1 §5 when treating MLA
substrate.**

Paper 1 §5 line 449–455 forward-references Paper~4 §10 for "the tension
that MLA's compressed-KV produces for circuit-tracing," and Paper 4 §10
lines 250–277 does engage MLA-on-frontier-architectures. But §10 says
"MLA-compressed-KV attention (DeepSeek-V2/V3)" without pointing the
reader back to Paper 1 §5 (the MLA home). The forward-reference is
asymmetric — Paper 1 reaches forward, Paper 4 doesn't reach back. Other
Paper 4 cross-paper threads (§13 to Paper 1 §13, §5 to Paper 3 §11) do
reach back. Suggested fix: one-line back-reference in §10's MLA
discussion ("the MLA mechanism of Paper~1 §5").

### NIT

**n1. Brace-style drift on `C_\text{test}`.**

Paper 2 uses `C_{\text{test}}` (with braces around `\text{...}`); Paper 3
uses `C_\text{test}` (no braces). Both render identically. LaTeX-only
nit; cosmetic.

**n2. Paper 1 §7 (MoE home) doesn't pre-cite Paper 2 callers.**

Paper 1 §7's FFN/MoE section (the home) doesn't reference Paper 2 §5
(EP/distributed) or Paper 2 §6 (FP8 expert-collapse), even though those
are the three places MoE legitimately surfaces in the series. Paper 2
correctly defers to Paper 1 (§1 line 187–190 explicitly: "MoE...lives
in Paper 1...intersects training...is cited from there"). Asymmetric
but defensible — the home paper isn't obligated to enumerate every
downstream consumer. Tolerable as-is.

## Notation audit

Six symbols sampled across the series:

| Symbol | Canonical (preamble) | Paper 1 | Paper 2 | Paper 3 | Paper 4 | Paper 5 | Verdict |
|---|---|---|---|---|---|---|---|
| `\BSH` | `\mathsf{B}\!\times\!\mathsf{S}\!\times\!\mathsf{D}` | `\BSH` (uses macro) | inline mathsf when needed | inline mathsf §1 | mixed: §2 says "Paper 1 convention" but §1/§3/§4/§5/§9 use partial-mathsf or `\mathbb{R}^{B \times S \times D}` | n/a (no tensor shapes) | A in P1; D in P4 (m1) |
| `B` (batch) | `\mathsf{B}` | `\mathsf{B}` | `\mathsf{B}` | `\mathsf{B}` | drift to plain `B` | n/a | drift in P4 |
| `H` (heads) | not macro'd; `H_q`, `H_{\mathrm{kv}}` per §5 | `H_{\mathrm{kv}}` consistent | n/a (uses Paper 1 forms in distributed) | n/a | n/a | n/a | A |
| `\beta` (KL coef.) | not macro'd; conventional | `\beta` (RLVR ranges in §11) | `\beta = 0.001` (RLVR), `0.01–0.05` (RLHF), `0.1` (DPO) — three different regimes correctly distinguished | `\beta = 0.001` (RLVR consistent with P2) | n/a | n/a | A |
| `N` params, `D` tokens, `C` FLOPs | not macro'd; conventional from Hoffmann | `L`, `\mathsf{D}` (model dim — distinct from data) | $N$, $D$, $C \approx 6 N D$ canonical Hoffmann | $N$, $D$, $C \approx 6 N D$ matches P2 | n/a | n/a | A |
| `C_\text{test}` (inf compute) | not macro'd | n/a | `C_{\text{test}}` mostly, `C_{\text{infer}}` once in §13 (m2) | `C_\text{test}` no braces (n1) | n/a | n/a | B |

The most consequential drift is `m1` (Paper 4's `\mathbb{R}^{B \times S \times D}` and partial-mathsf forms) — visible in renders where Paper 1 reads as `\mathsf{B}\!\times\!\mathsf{S}\!\times\!\mathsf{D}` (sans-serif, multiplication-cross) and Paper 4 §1 reads as `B \times S \times D` (italic), even though §2 declares the same convention. Easy sweep.

## Cross-paper-thread audit

### Thread 1: MoE (home P1 §7, cited from P2 §5, §6)

**Verdict: A.** Paper 2 §1 line 187–190 explicitly disclaims:
"Architectural choices...live in Paper 1, even though MoE intersects
training (\cref{sec:distributed-training} on EP, \cref{sec:mixed-precision}
on FP8 expert-collapse) and is cited from there." Paper 2 §5
(distributed) and §6 (mixed-precision) treat MoE-specific concerns
without re-deriving the architectural object. No duplication; clean
deferral. Asymmetric — Paper 1 §7 doesn't pre-cite Paper 2 callers
(n2) — but tolerable.

### Thread 2: MLA (home P1 §5, cited from P4 §10, §13)

**Verdict: B+.** Paper 4 §13 (lines 142–144) cites
`\citep{deepseek-v2}` for MLA and references "Paper~1, §13" for the
frontier-2026 architecture snapshot. Paper 4 §10 (lines 257–277) treats
MLA-compressed-KV substrate for ROME's localization claim but doesn't
back-reference Paper 1 §5 (the MLA home) — m5. Paper 1 §5 line 449–455
forward-references Paper 4 §10 correctly. Asymmetric forward/backward
linkage.

### Thread 3: CoT (home P3 §2 + §11, cited from P4 §5, P5 §9, §10, §11)

**Verdict: B+.** Paper 4 §5 line 360 properly defers: "The
chain-of-thought-faithfulness question of Paper~3 §11 is, in the
framing of this section, a probing question." Paper 5 §10 line 144
cites CoT as evidence-channel and §11 lines 259–270 explicitly defers
to "two questions inherited from prior papers in the series." But
Paper 5 §11 line 263 introduces lanham/turpin via `\deepencite` —
neither bibkey is in Paper 3 §11's citation set (m4). Paper 3 §11 cites
arcuschin/mehta/shen as primary anchors; Paper 5 should align (or
Paper 3 should expand to absorb lanham/turpin if they're truly
foundational).

### Thread 4: Process supervision / PRMs (home P3 §5, cited from P5 §13)

**Verdict: A.** Paper 5 §13 line 340–348 says: "Process supervision
and PRMs, covered in Paper~3 §process-supervision-and-PRMs, supplies
an adjacent training-side machinery." Clean deferral; no re-derivation.

### Thread 5: Scaling laws (home P2 §3, cited from P3 §4, §10, §14)

**Verdict: B.** Paper 3 §4 line 327: "Paper~2's scaling-laws section
transcribes \citet[\S 3, Eq.~10]{hoffmann2022-chinchilla}'s parametric
loss $L(N, D) = E + A/N^\alpha + B/D^\beta$." The reference back is
clean — but the equation pin is wrong (m1, MAJOR). Paper 2 §3 cites
the equation as `eq.~2`, the source labels it `(2)`, Paper 3 §4 says
`Eq.~10`. Same equation, two different numbers, propagated across
three Paper 3 locations.

Paper 2 §3 line 511–532 explicitly defers the *joint*
$(C_\text{train}, C_\text{test})$ closed-form to Paper 3 territory:
"the joint $(C_{\text{train}}, C_{\text{test}}, b, q)$ scaling law —
documented empirically, not captured in closed form." Paper 3 §4 line
345 picks this up: "the joint closed-form $L(N, D, \theta, C_\text{test})$
scaling law that subsumes Chinchilla and Snell remains an open
question." Both papers consistently flag this as the open frontier.

### Thread 6: RLVR — split between P2 §11 (post-training pipeline) and P3 §7 (reasoning triangle)

**Verdict: A−.** This is the structurally riskiest thread (intentional
split). Both papers transcribe the same canonical equations:
- `eq:verifier-reward` ($R = \mathbb{1}[\mathrm{verify}=\mathsf{true}]$)
- `eq:grpo` (full PPO-clipped surrogate with KL)
- `eq:grpo-advantage` (group-normalized advantage)
- `eq:dapo-clip`, `eq:dapo-dynsample`, `eq:dapo-tokenloss`,
  `eq:dapo-overlong` (DAPO's four modifications)

Both use the same labels and Greek symbols: $G \in [8, 64]$,
$\beta = 0.001$, $\epsilon = 0.2$, $\epsilon_{\mathrm{high}} = 0.28$.

Both explicitly cross-reference each other:
- Paper 2 §11 line 379: "Paper~3 §7 cites \cref{eq:verifier-reward},
  \cref{eq:grpo}, \cref{eq:grpo-advantage}, and the four DAPO equations
  as the canonical statement of the RLVR leg, and develops the
  inference-time interactions...that this section does not cover."
- Paper 3 §7 line 33: "deliberately complementary to Paper~2's RL-stage
  treatment (Paper~2 §11)."
- Both have `[CONTRADICTION]` blocks on the RLVR-creates-capability
  question pointing at each other (M2 above is a tonal issue inside
  Paper 3 §7's intuition, not a content disagreement).

The complementary framings are crisp: P2 = post-training-pipeline
stage, P3 = reasoning-triangle leg. The same R1 evidence ($15.6\% \to
77.9\%$ AIME) is used in both, with explicit different angles (P2:
"cleanest single number in the entire training pipeline"; P3: "the
cleanest single number in the reasoning-LLM literature, and the central
evidence that the triangle composes"). VAPO contradiction is treated
in both, consistently. M2 is the only blemish on this thread.

## Reading-order check

Reading order asserted in `SHAPE-C-decision.md` is 1→2→3→4→5; each
paper assumes the prior. Verified:

- **P1 → P2.** P2 §1 line 186–196 lists what P2 doesn't cover (P1's
  domain). P2 §3 uses the residual-stream notation P1 §2 establishes.
  Clean.
- **P2 → P3.** P3 §1 line 88–94: "Paper~2 establishes a six-stage
  post-training pipeline...That cross-reference is the inheritance
  route." P3 §7 explicitly defers RLVR-as-pipeline-stage to P2 §11.
  Clean.
- **P3 → P4.** P4 §5 line 360 picks up P3 §11's CoT-faithfulness
  question as a probing-methodology problem. P4 §13 references P1 §13
  for frontier-2026 substrate. Clean.
- **P4 → P5.** P5 §1 line 192: "Paper~4's probing and circuit-tracing
  methods are a complementary attack surface for [scheming,
  alignment-faking]." P5 §9, §10, §11 cite Paper 4 by section. Clean.

The reading order is honored. RLVR-creates-capability appears in P2
§11 as a CONTRADICTION, in P3 §7 deferred-to-P3-§14 form, and in P3
§14 (C3) as the central treatment — P3 §7 explicitly defers to "also
discussed in Paper~2 §11" (line 187). No duplication-as-derivation.

## What I did not check

- Per-equation transcription accuracy from primary PDFs (review #3
  scope).
- Per-model claim currency against 2026-Q2 sources (review #5 scope).
- Pedagogy / Feynman-bar adherence (review #6 scope).
- Whether `references.bib` resolves every cited bibkey (build-pass
  scope).

## Recommended fixes (in priority order)

1. **M1 — fix three Hoffmann equation pins in Paper 3.** Mechanical:
   replace `\S 3, Eq.~10` with `\S 3.3, eq.~2` in
   `paper-3/sections/04-inference-compute-scaling.tex` (lines 328, 398)
   and `paper-3/sections/10-search-vs-rl.tex` (line 205).
2. **M2 — soften Paper 3 §7's intuition box.** Either qualify the
   "RLVR cannot..." clause inside the box or move it outside the
   intuition environment so the box's load-bearing claim doesn't
   overstate what §14 (C3) treats as contested.
3. **m3 — replace Paper 5's five `\deepencite{paper-N §X}` calls with
   "Paper~N §X" prose** (or `\cref` if a unified-build target is added
   later).
4. **m1 — sweep Paper 4 to canonical `\BSH` macro or full
   `\mathsf{B}\!\times\!\mathsf{S}\!\times\!\mathsf{D}`** in §1, §3,
   §4, §5, §9.
5. **m2 — change Paper 2 §13's one `C_{\text{infer}}` to
   `C_{\text{test}}`.**
6. **m4 — decide on lanham/turpin status:** add to Paper 3 §11's
   citation set or remove from Paper 5 §11 in favor of Paper 3's
   anchor set.
7. **m5 — add a one-line back-reference to Paper 1 §5 in Paper 4 §10's
   MLA-substrate paragraph.**
8. **n1 — cosmetic brace-style normalization on `C_\text{test}`** can
   be batched with m2.

None of these changes alter any technical claim. The review is
publication-blocking on M1 (mechanically wrong citation, three
locations), and recommendation-grade on M2 (rhetorical, internally
walked back on the next paragraph). The series is otherwise coherent.
