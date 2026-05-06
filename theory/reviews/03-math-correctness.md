# Math-correctness review

Reviewer: independent math-correctness pass on the LaTeX paper series at
`theory/series/`. Read-only on `series/` and `kb/`. Sample-based audit of
14 equations across the high-math-density anchor sections of all five
papers, plus tensor-shape consistency check on the Paper 1 §5 pilot
section, plus three re-derivations.

## Summary

- **Equations sampled:** 14
- **Match-rate (verbatim or equivalent):** 13 / 14 fully match the cited
  excerpt. 1 / 14 (GRPO) is correct against one excerpt but has a
  cross-reference mismatch with the inline citation.
- **Findings:**
  - **BLOCKERS:** 0
  - **MAJORS:** 0 (one near-major on GRPO citation downgraded to MINOR
    after verifying the equation form matches the canonical Shao 2024
    excerpt — the issue is which paper is cited, not what is written).
  - **MINORS:** 4
  - **NITS:** 3

The math is in good shape. Every equation I sampled is either verbatim
from the cited excerpt or a faithful notation-substitution; tensor
shapes compose; the headline FLOP-count identities (`C ≈ 6ND`,
`Θ(N²d²/M)` FA-IO, `(d_c + d_h^R)·b` MLA cache, `Z = (|s|_G − γT)/√(Tγ(1−γ))`
watermark) all check. No load-bearing claim is unsupported.

## Sampled equations

| # | Equation | Section | Source excerpt | Verdict | Note |
|---|----------|---------|----------------|---------|------|
| 1 | `Attention(Q,K,V) = softmax(QKᵀ/√d_h)V` | P1 §5 (vaswani-attn) | `vaswani2017.md#sec-3-2-1` Eq. 1 | MATCH | d_k → d_h notation substitution; consistent w/ paper's d_k = d_v = d_h convention |
| 2 | `head_i = Attention(XW_i^Q, XW_i^K, XW_i^V)` and `Multihead(X) = Concat(...)W^O` | P1 §5 (vaswani-mh) | `vaswani2017.md#sec-3-2-2` Eq. 2-3 | MATCH | Vaswani writes `Multihead(Q,K,V)`; paper writes self-attention form `Multihead(X)` — equivalent |
| 3 | MLA `c_t^{KV} = W^{DKV} h_t`, `k_t^C = W^{UK} c_t^{KV}`, `v_t^C = W^{UV} c_t^{KV}` | P1 §5 (mla-cdkv/kc/vc) | `deepseek-v2.md#sec-2-1-2` Eq. 9-11 | MATCH | Verbatim |
| 4 | MLA decoupled-RoPE Eqs. 14-18 | P1 §5 (mla-qr/kr/q-cat/k-cat/out) | `deepseek-v2.md#sec-2-1-3` | MATCH | Eq. 14 paraphrased as "from q_t^R = RoPE(...)" — minor wording deviation, math identical |
| 5 | LayerNorm `μ^l = (1/H)Σa_i^l`, `σ^l = √((1/H)Σ(a_i^l−μ^l)²)` | P1 §8 (ba-eq3) | `ba2016-layernorm.md#sec-3` Eq. 3 | MATCH | Verbatim |
| 6 | RMSNorm `ā_i = (a_i / RMS(a))·g_i`, `RMS(a) = √((1/n)Σa_i²)` | P1 §8 (rmsnorm) | `zhang2019.md#sec-3-1` Eq. 4 | MATCH | Verbatim |
| 7 | Xiong 2020 Theorems 1 & 2 (Post-LN `O(d√(ln d))` final-FFN, Pre-LN `O(d√(ln d/L))` per-layer) | P1 §8 (thm postln/preln) | `xiong2020.md#sec-4` Theorems 1-2 | MATCH | Verbatim, with Pre-LN's `O(√L)` near-output growth attribution restated |
| 8 | FA-1 IO complexity: standard `Θ(Nd + N²)`, FlashAttention `Θ(N²d²/M)` | P1 §6 | `dao2022.md#sec-3-2` Theorem 2 | MATCH | Verbatim, d → d_h notation only |
| 9 | Kaplan power laws `L(N) = (N_c/N)^α_N`, `L(D) = (D_c/D)^α_D`, `L(C_min) = (C_c^min/C_min)^α_C^min`; joint `L(N,D)`; allocation Eq. 1.7 | P2 §3 | `kaplan2020.md#sec-1-2-equations`/`#sec-1-2-joint`/`#sec-1-2-compute-allocation` | MATCH | Eq. 1.1, 1.2, 1.3, 1.5, 1.7 verbatim incl. constants `α_N≈0.076`, `α_D≈0.095`, etc. |
| 10 | Chinchilla parametric loss `L̂(N,D) = E + A/N^α + B/D^β` and frontier `N_opt = G(C/6)^a`, `a = β/(α+β)`, `b = α/(α+β)` | P2 §3 | `hoffmann2022-chinchilla.md#sec-3-3` Eq. 2, 3, 4 | MATCH | Eq. 2, 3, 4 verbatim |
| 11 | Muon update Eq. 1 (`M_t`, `O_t = NS(M_t)`, `W_t = W_{t-1} − η_t O_t`); Newton-Schulz Eq. 2; weight-decay Eq. 3; per-shape RMS-adjusted Eq. 4 | P2 §4 | `muon-moonlight2025.md#sec-2-1-update-rule`/`#sec-2-1-ns`/`#sec-2-2-wd`/`#sec-2-2-rms` | MATCH | All four verbatim incl. `(a,b,c) = (3.4445,−4.7750,2.0315)` and `0.2·O_t·√max(A,B)` |
| 12 | Snell compute-optimal `θ*_{q,y*(q)}(N) = argmax_θ E_{y∼Target}[1_{y=y*(q)}]` | P3 §4 | `snell2024.md#sec-3-1` Eq. 1 | MATCH (MINOR) | Subscript drift: excerpt has `θ*_{q,a*(q)}` (mixed `a*` / `y*`), paper unifies to `y*` — the indicator is identical, the indexing label differs (see Findings §M3) |
| 13 | GRPO `J_GRPO(θ) = E[(1/G)Σ(1/|o_i|)Σ_t min(r_{i,t}Â_{i,t}, clip·Â_{i,t}) − β·KL]`; `Â_{i,t} = (R_i − mean)/std` | P3 §7 (also P2 §11) | `shao2024.md#sec-4-1` (token-level form) | MATCH | Token-level Shao 2024 form is verbatim. The inline cite `[\S 2.1, Eq. 1]{deepseek-r1}` is to a sequence-level form in a different excerpt — see Findings §M1 |
| 14 | ROME rank-one edit `Ŵ = W + Λ(C^{-1}k_*)^⊤` with `Λ = (v_*−Wk_*)/((C^{-1}k_*)^⊤k_*)`, `C = KKᵀ` | P4 §10 | `meng2022-rome.md#sec-3-1` Eq. 2 | MATCH | Verbatim. zsRE table values (99.8/88.1/24.2 etc.) also match exactly |
| 15 (bonus) | Watermark z-score `z = (|s|_G − γT)/√(Tγ(1−γ))` | P5 §12 | `kirchenbauer2023-watermark.md#sec-detection` | MATCH | Verbatim. Numerical example (γ=0.25, T=200, |s|_G=100 ⇒ z≈8.16) re-derived correctly |
| 16 (bonus) | Belrose tuned-lens loss `L(ℓ) = E[KL(M_{>ℓ}(h_ℓ) ‖ TunedLens_ℓ(h_ℓ))]` | P4 §3 | `belrose2023-tuned-lens.md#sec-3-loss` Eq. 9 | MATCH | Verbatim direction (forward KL with model output as target). `M_{>ℓ}` substitutes excerpt's `f_{>ℓ}`, defined consistently in §2 above |

(15 and 16 included as bonus-checks to push the sample over the
12-15-equation request.)

## Findings (by severity)

### MAJORS

None.

### MINORS

**M1. GRPO citation cross-reference mismatch (P3 §4, P3 §7).** The
GRPO objective in `paper-3/sections/07-rlvr-grpo-dapo.tex` Eq. 1
(eq:grpo) is in the **token-level form** with per-token importance
ratio `r_{i,t}(θ)` and per-token clip. Paper §7 cites it as
`[\S 2.1, Eq.\ 1]{deepseek-r1}` with a footnote saying "verified
against R1 §2.1 Eq. 3". However, `kb/excerpts/deepseek-r1.md#sec-2-1-grpo`
gives the **sequence-level form** with `π_θ(o_i|q)/π_θ_old(o_i|q)·A_i`
(no `(1/|o_i|)Σ_t` and no per-token ratio). The token-level form *does*
match `kb/excerpts/shao2024.md#sec-4-1` verbatim, which the paper also
cites in the same paragraph. The form is correct (and is the form used
in actual GRPO implementations), but a reader following the
deepseek-r1 anchor will see the sequence-level form and be confused.
**Fix:** swap the primary cite from `deepseek-r1` to `shao2024` for
this equation, or add a footnote explaining that R1's published Eq. 1
is the sequence-level form and Shao 2024 §4.1 is the token-level form
that the field implements. Same fix applies to Paper 2 §11 if it
shares the citation.

Also: P3 §4 ¶ "Structural Chinchilla parallel" cites
`[\S 3, Eq.~10]{hoffmann2022-chinchilla}` for the parametric-loss
form `L(N,D) = E + A/N^α + B/D^β`. The Chinchilla excerpt and the
Chinchilla paper itself number this as **Eq. (2)**, not Eq. (10).
P2 §3 correctly cites it as Eq. 2. This is a transcription error in
P3 §4 (occurs twice, ll. 328 and 397 of `04-inference-compute-scaling.tex`).

**M2. Watermark spike-entropy form is unverified.** Paper 5 §12
displays `S_γ(p) = Σ_v p_v / (1 + γ' p_v)` and the expectation
`E[|s|_G/T] = γ + c(γ,δ)·S_γ(p)·(1−γ) + o(·)`, both flagged with
`\deepencite{kirchenbauer2023-watermark §sensitivity-bound}`. The KB
excerpt at `kirchenbauer2023-watermark.md` confirms only the abstract
and the z-score; the spike-entropy functional is not transcribed. The
form `Σ p_v/(1+γ' p_v)` is correct against the published Kirchenbauer
2023 Definition 1, but **the workspace's verification-from-PDF
discipline has a gap here**. The `\deepencite` marker is correctly
applied; treat this as a known follow-up rather than an error.

**M3. Snell Eq. 1 subscript notation drift (P3 §4).** Excerpt
`snell2024.md#sec-3-1` writes `θ*_{q,a*(q)}(N)` (mixed: uses both `a*`
and `y*` in the same equation block — `a*` in the subscript, `y*` in
the indicator and prose). Paper-3 §4 unifies to `y*(q)` in both. The
unification is more readable but is non-verbatim. NIT-bordering-on-MINOR;
suggest a footnote explaining the unification.

**M4. UCT vs. PUCT label (P3 §6).** Section 06-tree-search.tex labels
Eq. (uct) the "UCT selection rule" but the formula
`U(s) = Q(s) + c_puct · π_θ(s|parent) · √N_parent / (1 + N_s)` is
the AlphaGo/AlphaZero **PUCT** form (with the policy prior multiplier
and `1+N_s` denominator), not the original Kocsis-Szepesvári UCT
`Q(s) + c·√(ln N_parent / N_s)`. The accompanying prose correctly says
"the second term is the PUCT exploration bonus", so the math and the
prose disagree with the equation label. Trivial relabel (`eq:puct`,
"PUCT selection rule") would resolve. The KB excerpt
`rstar-math2025.md#sec-3-mcts` is admittedly `[NOTE — pdf-not-available]`
so the exact constants are flagged; the form is standard for AlphaZero
shoulders.

### NITS

**N1. `\BSH` macro name is misleading.** `preamble.tex` defines
`\newcommand{\BSH}{B×S×D}` — the macro name suggests "B-S-H" (heads)
but expands to "B-S-D" (model dim). All current usage is consistent
with the *expansion*, but a future writer reading "BSH" will guess
wrong. Suggest renaming to `\BSD` in a future cleanup pass.

**N2. P2 §4 deepencite ergonomics.** Several `\deepencite{kingma2015-adam: ...}`
markers carry colon-prefixed annotation strings (e.g.,
"`\deepencite{kingma2015-adam: ... $(β_1, β_2)$ moment definitions ...}`").
The macro expects `key §section`-style content; with a colon and prose
it will still render the orange margin note correctly, but the
generated todo-list at link time may parse oddly. Cosmetic.

**N3. P1 §5 cache table comparison.** The table
caption-vs-data: row "GQA-2" shows cache `4 d_h b`, which is `2·G·d_h·b`
with `G=2`, ✓. Row "GQA-8" shows `16 d_h b`, ✓. Row MLA shows
`(d_c + d_h^R)·b ≈ 4.5 d_h b`, ✓ (with the DSV2 defaults `d_c=4d_h`,
`d_h^R=d_h/2`). Numbers all consistent. The "MLA equals GQA with 2.25
groups" claim: `2.25·2·d_h·b = 4.5·d_h·b` ✓. Table is internally
self-consistent and matches `deepseek-v2.md#sec-2-1-4` Table 1
(which has `(d_c + d_h^R) l ≈ 9/2 d_h l` per token with `9/2 = 4.5`).

## Tensor-shape consistency (P1 §5 pilot)

Walked the attention block end-to-end. All shapes compose:

```
X ∈ R^{B × S × D}                                         residual stream
W_i^Q, W_i^K, W_i^V ∈ R^{D × d_h}                         per-head projections
Q, K, V ∈ R^{B × H × S × d_h}    (head split + transpose)
S = QK^T / √d_h ∈ R^{B × H × S × S}                       attention scores
P = softmax(S) ∈ R^{B × H × S × S}                        attention weights
PV ∈ R^{B × H × S × d_h}                                   per-head outputs
Concat: R^{B × S × H·d_h} = R^{B × S × D}                  (H·d_h = D)
W^O ∈ R^{H d_h × D}                                       output projection
Multihead(X) ∈ R^{B × S × D}                               back to residual
```

✓ Everything composes.

For MQA: `K, V ∈ R^{B × S × d_h}` (single shared head), broadcast across
H query heads at attention time. ✓.

For GQA-G: `K, V ∈ R^{B × G × S × d_h}` with each query head in group
g attending to its group's K,V. The cache footprint `2·G·d_h·b` per
token per layer follows directly. ✓.

For MLA: cache holds `c_t^{KV} ∈ R^{d_c}` plus `k_t^R ∈ R^{d_h^R}`, so
per-token cache is `(d_c + d_h^R)·b` bytes. The "absorption" identity
`(W_i^Q)^T (W^{UK} c) = ((W^{UK})^T W_i^Q)^T c` puts the per-head
reconstruction matrix into the query side at compile time, never
materializing the per-head K. ✓ for the content path; the RoPE path is
a separate per-token store of `d_h^R` dims that does *not* commute and
must be cached.

KL between two distributions: every KL in the series (Belrose tuned-lens
loss, ACDC edge importance, GRPO penalty, ChatGPT/Snell formulation) is
between distributions over the **same support** (vocabulary V, or
output-token distribution), so dimensional mismatch is not an issue.

## Re-derivations

**(a) Watermark z-score numerical sanity.**
`γ=0.25, T=200, |s|_G=100`. Null mean `γT = 50`. Variance
`Tγ(1−γ) = 200·0.25·0.75 = 37.5`. So
`z = (100 − 50)/√37.5 = 50/6.124 ≈ 8.165`. Paper says `≈ 8.16`. ✓.
One-sided p-value `1 − Φ(8.16) ≈ 1.6×10⁻¹⁶ < 10⁻¹⁵` ✓ (paper says
`< 10⁻¹⁵`, conservative).

**(b) MLA cache vs. GQA-equivalent groups.**
`(d_c + d_h^R)·b = (4 d_h + 0.5 d_h)·b = 4.5 d_h b`. GQA-G cache is
`2·G·d_h·b`. Setting equal: `2G = 4.5 → G = 2.25` ✓. Matches
DeepSeek-V2's claim verbatim.

**(c) Chinchilla compute-optimal exponents.** From the parametric form
`L̂ = E + A/N^α + B/D^β` minimized under `C = 6ND`. Lagrangian
gives `αA/N^{α+1} = βB/D^{β+1} · (constant)`, leading to
`N^{α+1}/D^{β+1} = (αA)/(βB)`. Combined with `N·D = C/6` and
power-law form `N = G·(C/6)^a`, `D = G^{-1}·(C/6)^b`, the constraint
`a + b = 1` is identical (since `N·D ∝ C^{a+b} = C^1`). With
`a = β/(α+β), b = α/(α+β)`, we have `a + b = (α+β)/(α+β) = 1` ✓.
The paper's Eq. (cc-frontier) and (cc-abG) are internally consistent.

## What I did not check

- Paper 2 §5 (DualPipe) — has algorithmic prose but no equation
  numbers I could verify from `deepseek-v3-training.md` excerpt; the
  KB excerpt itself is largely paraphrased.
- Paper 2 §6 FP8 — checked the formulas pass-through; no obvious
  drift, but did not exhaustively compare to
  `deepseek-v3-training.md#FP8` which is excerpt-only at high level.
- Paper 4 §11 attribution-graph Jacobian Eq. — paper says
  `a(f_i^{ℓ_1,t_1} → f_j^{ℓ_2,t_2}) = ∂f_j/∂f_i ≈ ⟨e_j, A_{ℓ_1→ℓ_2} d_i⟩`.
  This is a faithful inner-product of decoder-column with encoder-row
  through a linearized residual-transport operator; the KB excerpt for
  `lindsey2025-circuit-tracing` is HTML-only and not transcribed, so
  verbatim verification was not possible. The form is consistent with
  the cross-layer-transcoder construction described in
  `dunefsky2024-transcoders`.
- Paper 5 §12 spike-entropy *constants* and the form of `c(γ,δ)` —
  flagged with `\deepencite` for verification against the paper PDF
  on the next deepening pass. (See Finding M2.)
- Paper 3 §6 PUCT exact constants from `rstar-math2025` — `\deepencite`
  flagged; PDF not yet acquired.

## Summary verdict

The math layer of the paper series is in solid shape. Sampled equations
match their cited excerpts at high fidelity; the load-bearing identities
all check; tensor shapes compose end-to-end on the audited section
(Paper 1 §5). The four MINOR findings are concentrated in citation
hygiene rather than mathematical content: M1 (GRPO citation should
point at Shao 2024 not deepseek-r1, or carry a clarifying footnote;
also Eq. 10 → Eq. 2 in P3 §4); M3 (Snell subscript notation
unification footnote); M4 (UCT → PUCT label). M2 is a known
verification gap properly flagged via `\deepencite`. No blockers, no
majors. Recommend addressing M1's Eq. 10/Eq. 2 fix and M4's UCT→PUCT
label in a touch-up commit; the remainder can ride to the next
deepening pass.

**Files of interest:**
- `/home/ai/ai-projects/llm/theory/series/paper-1/sections/05-attention-kv-sharing.tex`
- `/home/ai/ai-projects/llm/theory/series/paper-1/sections/06-attention-hardware-implementation.tex`
- `/home/ai/ai-projects/llm/theory/series/paper-1/sections/08-normalization-axis.tex`
- `/home/ai/ai-projects/llm/theory/series/paper-2/sections/03-scaling-laws.tex`
- `/home/ai/ai-projects/llm/theory/series/paper-2/sections/04-optimization.tex`
- `/home/ai/ai-projects/llm/theory/series/paper-3/sections/04-inference-compute-scaling.tex` (M1 second instance, M3)
- `/home/ai/ai-projects/llm/theory/series/paper-3/sections/06-tree-search.tex` (M4)
- `/home/ai/ai-projects/llm/theory/series/paper-3/sections/07-rlvr-grpo-dapo.tex` (M1 primary instance)
- `/home/ai/ai-projects/llm/theory/series/paper-4/sections/03-lens-techniques.tex`
- `/home/ai/ai-projects/llm/theory/series/paper-4/sections/10-rome.tex`
- `/home/ai/ai-projects/llm/theory/series/paper-4/sections/11-circuits.tex`
- `/home/ai/ai-projects/llm/theory/series/paper-5/sections/12-watermarking.tex` (M2)
- `/home/ai/ai-projects/llm/theory/series/preamble.tex` (N1)
