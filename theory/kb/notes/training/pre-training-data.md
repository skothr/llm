---
topic_key: pre-training-data
area: training
status: draft
last_updated: 2026-05-04
primary_sources: [fineweb2024, data-mixing-laws-2024]
secondary_sources: [phi4, deepseek-v3, kaplan2020, hoffmann2022-chinchilla]
related_topics: [synthetic-data-and-distillation, scaling-laws, optimization]
---

# Pre-training data

## Formal definition {#def}

A pre-training dataset is a finite multiset $\mathcal{D} = \bigsqcup_{j=1}^{M}
\mathcal{D}_j$ partitioned into $M$ source domains. The *mixture* is the
proportion vector $\mathbf{r} = (r_1, \dots, r_M)$ with $r_j \geq 0$,
$\sum_j r_j = 1$, where $r_j$ is the fraction of the per-step token budget
drawn from $\mathcal{D}_j$ during training. Two design problems live here:

1. **Per-domain curation** — what counts as $\mathcal{D}_j$, how is it
   filtered and deduplicated, what tokens make it into the multiset.
2. **Mixture choice** — given curated domains, what $\mathbf{r}$ minimizes
   loss / maximizes downstream metric at the target $(N, D)$ scale, where
   $N$ is parameter count and $D$ is total token budget
   `[kaplan2020 §1, eq.1.5; hoffmann2022-chinchilla §2]`.

Symbol glossary: $\mathcal{D}_j$ = source-$j$ token multiset;
$\mathbf{r}$ = mixture proportions; $L_i(\mathbf{r})$ = validation loss on
held-out domain $i$ as a function of training mixture; $N$ = parameters;
$D$ = total training tokens.

## Mechanism — the modern web pipeline {#mechanism}

The current state-of-the-art open recipe (FineWeb, 15T tokens, mid-2024)
shows a pipeline that the rest of the field now mirrors `[fineweb2024
§abstract; kb/excerpts/fineweb2024#abstract]`:

**Pipeline stages.** Crawl extraction → URL/language filter → text
extraction → quality heuristics → deduplication → custom filters → final.
Each stage is judged by training a small "ablation" model
(1.71 B params, ~28 B tokens, identical architecture/optimizer across runs)
on the candidate output and reading downstream task scores; the slope of
"ablation-model accuracy vs. token count" is the only signal the pipeline
trusts `[fineweb2024 §3.1; kb/excerpts/fineweb2024#sec-3-1-proxy]`.

**Concrete numbers, FineWeb.**
- 96 Common Crawl snapshots (2013-summer through 2024-04), processed
  individually `[fineweb2024 §3.4]`.
- Text extraction: trafilatura, ~25 % more text extracted than CC's
  built-in WET dump and judged higher-quality by ablation
  `[fineweb2024 §3.2; kb/excerpts/fineweb2024#sec-3-2-text]`.
- Base filter: Gopher-style URL block-list, language-ID English
  threshold 0.65, plus C4 line-level filters minus the "curly-brace"
  rule (which would remove most code) `[fineweb2024 §3.3;
  kb/excerpts/fineweb2024#sec-3-3-base]`.
- MinHash dedup: 5-gram shingles, 112 hash functions arranged as
  $14 \text{ buckets} \times 8 \text{ hashes}$, Jaccard threshold $\approx 0.75$
  `[fineweb2024 §3.4; kb/excerpts/fineweb2024#sec-3-4-minhash]`.

**[CONTRADICTION] Global vs. individual-snapshot dedup.** A naive prior
holds that more aggressive deduplication is monotonically better.
FineWeb directly disproves this: deduplicating *across all 96 snapshots
at once* makes the model worse than deduplicating each snapshot
independently. The hypothesis is that global dedup over-removes
high-quality content (because it appeared in multiple snapshots) and
leaves a residue dominated by ad copy and SEO spam — i.e., what
appeared *only once* is lower-quality on average than what appeared
multiple times `[fineweb2024 §3.4;
kb/excerpts/fineweb2024#sec-3-4-global]`.

**Custom heuristic filters (FineWeb's late-stage gain).** Three
quantile-derived rules that fired on C4-vs-FineWeb diffs:
$P(\text{line ends with terminal punctuation}) \leq 0.12$,
$P(\text{duplicate-line characters}) \geq 0.01$, and
$P(\text{short lines} \leq 30 \text{ chars}) \geq 0.67$ — applied as a
filter, lift HellaSwag accuracy by ~1 pt at the ablation scale
`[fineweb2024 §3.6; kb/excerpts/fineweb2024#sec-3-6-custom]`.

## Mixture choice — Data Mixing Laws {#mixing}

Given curated domains, the mixture $\mathbf{r}$ controls validation loss
through a fittable functional form. Ye et al. 2024 establish the
*data mixing law* `[data-mixing-laws-2024 §1, eq.1;
kb/excerpts/data-mixing-laws-2024#sec-1-functional]`:

$$L_i(r_{1\ldots M}) = c_i + k_i \exp\!\left(\sum_{j=1}^{M} t_{ij} r_j\right)$$

where $L_i$ is validation loss on domain $i$, and $(c_i, k_i, t_{ij})$ are
fit per-target-domain. Coefficient interpretation
`[data-mixing-laws-2024 §3.2; kb/excerpts/data-mixing-laws-2024#sec-3-2-coefficients]`:
- $c_i$ — irreducible loss for domain $i$ (no mixture choice can lower it).
- $t_{ij}$ — interaction: $t_{ij} < 0$ means training-domain $j$ helps
  validation-domain $i$; $t_{ij} > 0$ means it hurts.

**Why the form.** Two principles fix it `[data-mixing-laws-2024 §3.2;
kb/excerpts/data-mixing-laws-2024#sec-3-2-multi]`:
*compatibility* (reduces to the two-domain log-linear regime when $M=2$,
which Fudan + Shanghai AI Lab verify empirically with 70 M and 160 M
models on Github+PileCC at 30 B tokens
`[data-mixing-laws-2024 §3.1, eq.6; kb/excerpts/data-mixing-laws-2024#sec-3-1-pilot]`)
and *symmetry* (no domain-specific bias in the functional form).

**Nested scaling laws.** Fitting Eq. (1) directly at the target scale is
unaffordable. Ye et al. wrap a power-law of training-step ($L = c + kS^\alpha$)
inside a power-law of model-size, inside the mixing law, so that small-model
small-step experiments predict large-model end-of-training loss as a
function of $\mathbf{r}$ `[data-mixing-laws-2024 §1, §2, eq.3;
kb/excerpts/data-mixing-laws-2024#sec-1-nested]`. The reported headline
result: a 1 B model trained for 100 B tokens on RedPajama, with mixture
chosen by this nested fit, reaches the loss of the same model trained for
**48 % more steps** on RedPajama's default mixture
`[data-mixing-laws-2024 §abstract; kb/excerpts/data-mixing-laws-2024#abstract]`.

## Quality-classifier filtering — FineWeb-Edu {#edu}

After heuristic filtering produces a 15 T-token web corpus, a *quality
classifier* can extract a high-density subset. FineWeb-Edu workflow
`[fineweb2024 §4; kb/excerpts/fineweb2024#sec-4-edu]`:

1. Sample 500 K documents from FineWeb.
2. Annotate each with Llama-3-70B-Instruct on a 0–5 "educational value"
   prompt.
3. Train a small classifier on the resulting (text, score) pairs.
4. Run the classifier over the full 15 T and keep documents above a
   threshold.

Result: a 1.3 T-token subset that, training-token-for-training-token,
beats FineWeb on knowledge-heavy benchmarks (MMLU, ARC). Cost of the
annotation step: about 6000 H100-hours
`[fineweb2024 §4; kb/excerpts/fineweb2024#sec-4-edu]`.

The pattern generalizes: *spend teacher tokens once at corpus-build
time, recover them every step of every downstream pre-training run that
uses the filtered corpus.* See also
[kb/notes/training/synthetic-data-and-distillation.md](synthetic-data-and-distillation.md)
for the closely-related case where the teacher is used to *generate*
rather than *filter*.

## Variants & lineage {#variants}

| Corpus | Tokens | Year | Distinctive choice |
|---|---|---|---|
| C4 | 0.75T | 2020 | First open web pipeline; line-level rules |
| The Pile | 0.825T | 2020 | 22 hand-picked sources with manual mixture |
| RefinedWeb | 5T | 2023 | Falcon's web-only corpus; aggressive dedup |
| RedPajama-V1 / V2 | 1.2T / 30T | 2023 / 2024 | Open Llama-1 reproduction; V2 returns CC at scale |
| Dolma | 3T | 2024 | OLMo's open release; 7 sources |
| FineWeb / FineWeb-Edu | 15T / 1.3T | 2024 | Ablation-model-driven design `[fineweb2024]` |
| DCLM-Baseline | 4T | 2024 | Classifier-filtered (fastText on instruction data) |
| Nemotron-CC | 6.3T | 2024 | NVIDIA; rephrased + classifier-filtered web |

Trends visible in the table: (1) total token counts have grown ~20× in
four years; (2) the curation interface has shifted from "hand-picked
sources" (Pile) to "ablation-model-driven heuristics" (FineWeb) to
"classifier-driven distillation" (FineWeb-Edu, DCLM); (3) the upper bound
on usable web text is being approached — multiple labs now report
diminishing returns past ~10 T high-quality web tokens, which is the
proximate driver of the synthetic-data turn covered in
[synthetic-data-and-distillation](synthetic-data-and-distillation.md).

## Intuitions & analogies {#intuitions}

**[INTUITION] The ablation-model is a calibrated noise meter.** The
1.71 B / 28 B-token ablation runs FineWeb uses are individually too small
to hit any production benchmark cleanly; the design relies on the *signed
direction* of the difference between two candidate corpora being stable
across scale. Empirically this holds for most heuristic decisions but
fails for the global-vs-individual dedup case, which is why that result
is so surprising — the small-scale signal told the team to do the
counterintuitive thing, and it kept holding at 7 B and beyond
`[fineweb2024 §3.4; kb/excerpts/fineweb2024#sec-3-4-global]`. Returning to
canonical form: this is the empirical fact that
$\Delta L(\text{candidate}_A, \text{candidate}_B; N=1.71\text{B}, D=28\text{B})$
has the same sign as $\Delta L(\cdot, \cdot; N=7\text{B}, D=1\text{T})$
in *most but not all* cases — the cases where it does not are the most
informative.

**[ANALOGY] Quality classifier = chef recommending dishes from a buffet.**
The teacher LLM scores each document for "educational value"; the
classifier learns the cheap version of that scoring; the trained model
eats the curated subset. The student never sees the teacher's outputs
*as text* — only the teacher's *judgment over what text to keep*. This is
why FineWeb-Edu's distillation path is much narrower-bandwidth than the
Phi-4 path covered in
[synthetic-data-and-distillation](synthetic-data-and-distillation.md), but
also much cheaper: 6 K H100-hours of annotation amortizes over every
downstream training run. Returning to canonical form: the teacher
parameterizes a function $q: \text{doc} \to [0, 5]$; the classifier fits
$\hat{q}$; the corpus is $\{d : \hat{q}(d) \geq \tau\}$.

**[INTUITION] Mixture coefficient $t_{ij}$ as transfer affinity.** A
fitted $t_{\text{code} \leftarrow \text{math}} < 0$ says "training on more
math reduces validation loss on code" — exactly the cross-domain
transfer effect that motivated curated-mixture practice in the first
place. Data Mixing Laws upgrades this from folklore to a measurable
parameter `[data-mixing-laws-2024 §3.2;
kb/excerpts/data-mixing-laws-2024#sec-3-2-coefficients]`.

## Frontier & open questions {#frontier}

1. **Web token ceiling.** How much usable web text exists? Multiple
   labs (DeepSeek-V3 at 14.8 T, Llama-3 at 15 T, FineWeb at 15 T) have
   independently converged near the 15 T mark, suggesting an effective
   ceiling on raw English web text after standard dedup
   `[deepseek-v3 §abstract; kb/excerpts/deepseek-v3-training#abstract]`.
   Past this, growth comes from non-English web, code, or synthetic.

2. **Late-stage curriculum / mixture schedules.** Data Mixing Laws fits
   a *single* $\mathbf{r}$ for a fixed budget; OLMo 2's "Dolmino" mix
   shows large gains from *changing* $\mathbf{r}$ near the end of
   training (curriculum). The functional form for time-varying mixtures
   is open. See [adaptation-and-merging](adaptation-and-merging.md)
   STUB for the related continual-pre-training case.

3. **[CONTRADICTION] Quality vs. quantity trade.** Chinchilla scaling
   says compute-optimal $D \approx 20 N$
   `[hoffmann2022-chinchilla §3]`, but practitioners now routinely
   over-train (Llama-3 8B on 15 T, $D \approx 1875 N$) because inference
   matters more than training and quality-filtered tokens give better
   per-token gain. Whether the Chinchilla curve is *recoverable* at
   high quality (i.e., the optimal $D/N$ ratio is itself a function of
   data quality) is unresolved.

4. **Classifier-induced mode collapse.** A quality classifier trained on
   a teacher LLM inherits the teacher's blind spots; a corpus filtered
   to the classifier's high-confidence region may strip diversity in
   ways the teacher cannot see. No published large-scale study isolates
   this effect from the gain. `[FORUM-SIGNAL]` repeatedly raised on
   community channels but with no primary citation.

5. **Mixing-law extrapolation across architecture.** Ye et al. fit
   coefficients for one architecture family; whether the same
   $(c_i, k_i, t_{ij})$ transfer to a different model family
   (e.g., MoE vs. dense) is untested.
