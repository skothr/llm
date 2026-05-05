---
paper_key: marks-tegmark-2023-truth
title: "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets"
authors: Marks, Tegmark
year: 2023
venue: COLM 2024 (arXiv 2310.06824, v3 2024-08-19)
arxiv: 2310.06824
local_pdf: sources/papers/burns2023-truth-geometry.pdf
type: excerpts
note: Excerpts verified from local PDF. The local file is misnamed `burns2023-truth-geometry.pdf` for historical reasons (Phase 1 transcription error); paper is by Marks & Tegmark, hence `marks-tegmark-2023-truth` is the canonical key. Anchors cover abstract, §1 contributions and related work, §2 datasets, §3 patching localization, §4 visualization and misalignment, §5+ probe-direction comparison. Sections beyond §5 are not yet excerpted in this pass.
---

# Excerpts — Marks & Tegmark 2023, "The Geometry of Truth"

## Abstract — full text {#abstract}

> Large Language Models (LLMs) have impressive capabilities, but are
> prone to outputting falsehoods. Recent work has developed techniques
> for inferring whether a LLM is telling the truth by training probes
> on the LLM's internal activations. However, this line of work is
> controversial, with some authors pointing out failures of these
> probes to generalize in basic ways, among other conceptual issues.
> In this work, we use high-quality datasets of simple true/false
> statements to study in detail the structure of LLM representations
> of truth, drawing on three lines of evidence: 1. Visualizations of
> LLM true/false statement representations, which reveal clear linear
> structure. 2. Transfer experiments in which probes trained on one
> dataset generalize to different datasets. 3. Causal evidence
> obtained by surgically intervening in a LLM's forward pass, causing
> it to treat false statements as true and vice versa. Overall, we
> present evidence that at sufficient scale, LLMs *linearly represent*
> the truth or falsehood of factual statements. We also show that
> simple difference-in-mean probes generalize as well as other probing
> techniques while identifying directions which are more causally
> implicated in model outputs.

## §1 Three contributions {#sec-1-contributions}

> Working with autoregressive transformers from the LLaMA-2 family
> (Touvron et al., 2023), we shed light on this murky state of
> affairs. After curating high-quality datasets of simple,
> unambiguous true/false statements, we perform a detailed
> investigation of LLM representations of factuality. Our analysis,
> which draws on patching experiments, simple visualizations with
> principal component analysis (PCA), a study of probe generalization,
> and causal interventions, finds:
>
> - Evidence that linear representations of truth emerge with scale,
>   with larger models having a more abstract notion of truth that
>   applies across structurally and topically diverse inputs.
> - A small group of causally-implicated hidden states which encode
>   these truth representations.
> - Consistent results across a suite of probing techniques, but with
>   simple difference-in-mean probes identifying directions which are
>   most causally implicated.

## §1.1 Related work — probing for truthfulness {#sec-1-related}

> **Probing for truthfulness.** Others have trained probes to classify
> truthfulness from LLM activations, using both logistic regression
> (Azaria & Mitchell, 2023; Li et al., 2023b), unsupervised (Burns
> et al., 2023), and contrastive (Zou et al., 2023; Rimsky et al.,
> 2024) techniques. This work differs from prior work in a number of
> ways. First, a cornerstone of our analysis is evaluating whether
> probes trained on one dataset transfer to topically and structurally
> different datasets in terms of *both* classification accuracy *and*
> causal mediation of model outputs. Second, we specifically
> interrogate whether our probes attend to *truth*, rather than merely
> features which correlate with truth (e.g. probable vs. improbable
> text). Third, we localize truth representations to a small number
> of hidden states above certain tokens. Fourth, we go beyond the
> mass-mean shift interventions of Li et al. (2023b) by systematically
> studying the properties of difference-in-mean. Finally, we carefully
> scope our setting, using only datasets of clear, simple, and
> unambiguous factual statements, rather than statements which are
> complicated and structured (Burns et al., 2023), confusing (Azaria
> & Mitchell, 2023; Levinstein & Herrmann, 2023), or intentionally
> misleading (Li et al., 2023b; Lin et al., 2022).

## §2 Datasets — `cities`, negations, conjunctions, `likely` {#sec-2-datasets}

Table 1 (page 3) lists the curated datasets:

| Name | Description | Rows |
|---|---|---|
| `cities` | "The city of [city] is in [country]." | 1496 |
| `neg_cities` | Negations of statements in cities with "not" | 1496 |
| `sp_en_trans` | "The Spanish word '[word]' means '[English word]'." | 354 |
| `neg_sp_en_trans` | Negations of statements in sp_en_trans with "not" | 354 |
| `larger_than` | "x is larger than y." | 1980 |
| `smaller_than` | "x is smaller than y." | 1980 |
| `cities_cities_conj` | Conjunctions of statements in cities with "and" | 1500 |
| `cities_cities_disj` | Disjunctions of statements in cities with "or" | 1500 |
| `companies_true_false` | Claims about companies; from Azaria & Mitchell (2023) | 1200 |
| `common_claim_true_false` | Various claims; from Casper et al. (2023) | 4450 |
| `counterfact_true_false` | Various factual recall claims; from Meng et al. (2022) | 31960 |
| `likely` | Nonfactual text with likely or unlikely final tokens | 10000 |

The `likely` dataset is the methodological control:

> we introduce a `likely` dataset, consisting of *nonfactual text*
> where the final token is either the most or 100th most likely
> completion according to LLaMA-13B. […] Together with the `likely`
> dataset, this will help us establish that the linear structure we
> observe in LLM representations is not due to LLMs linearly
> representing the difference between probable and improbable text.

## §3 Localizing truth via patching {#sec-3-layers}

> Before beginning our study of LLM truth representations, we first
> address the question of which hidden states might contain such
> representations. We use simple patching experiments (Vig et al.,
> 2020; Finlayson et al., 2021; Meng et al., 2022; Geiger et al.,
> 2020) to localize certain hidden states for further analysis. […]
> We see three groups of causally-implicated hidden states. The
> final group, labeled (c), directly encodes the model's prediction:
> after applying the LLM's decoder head directly to these hidden
> states, the top logits belong to tokens like "true," "True," and
> "TRUE." The first group, labeled (a), likely encodes the LLM's
> representation of "Chicago" or "Toronto."
>
> What does group (b) encode? The position of this group — over the
> final token of the statement and end-of-sentence punctuation —
> suggests that it encodes information pertaining to the full
> statement. Since the information encoded is also causally
> influential on the model's decision to output "TRUE" or "FALSE,"
> we hypothesize that these hidden states store a representation of
> the statement's truth.

> This *summarization* behavior, in which information about clauses
> is encoded over clause-ending punctuation tokens, was also noted in
> Tigges et al. (2023). We note that the largest LLaMA model displays
> this summarization behavior in a more context-dependent way; see
> App. B.

## §4 Visualization — separation and misalignment {#sec-4-misalignment}

> When visualizing LLaMA-2-13B and 70B representations of our curated
> datasets […] **we see clear linear structure** (Fig. 1), with true
> statements separating from false ones in the top two principal
> components (PCs). As explored in App. C, this structure emerges
> rapidly in early-middle layers and emerges later for datasets of
> more structurally complex statements (e.g. conjunctive statements).

> To what extent does this visually-apparent linear structure align
> between different datasets? Our visualizations indicate a nuanced
> answer: **the axes of separation for various true/false datasets
> align often, but not always.** […] On the other hand, Fig. 3(c)
> shows stark failures of alignment, with the axes of separation for
> datasets and statements and their negations being approximately
> orthogonal.

> These cases of misalignment have an interesting relationship to
> scale. Fig. 3(b) shows `larger_than` and `smaller_than` separating
> along *antipodal* directions in LLaMA-2-13B, but along a common
> direction in LLaMA-2-70B. App. C depicts a similar phenomenon
> occurring over the layers of LLaMA-2-13B: in early layers, `cities`
> and `neg_cities` separate antipodally, before rotating to lie
> orthogonally (as in Fig. 3(c)), and finally aligning in later layers.

## §4.1 Discussion — emergent abstraction {#sec-4-discussion}

> Overall, these visualizations suggest that as LLMs scale (and
> perhaps, also as a fixed LLM progresses through its forward pass),
> they hierarchically develop and linearly represent increasingly
> general abstractions. Small models represent surface-level
> characteristics of their inputs, and large models linearly represent
> more abstract concepts, potentially including notions like "truth"
> that capture shared properties of topically and structurally
> diverse inputs.
