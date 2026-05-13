---
paper_key: anthropic2026-nla
title: "Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations"
authors: Fraser-Taliente, Kantamneni, Ong, Mossing, Lu, Bogdan, et al. (Anthropic)
year: 2026
venue: "Transformer Circuits Thread (Anthropic)"
url: https://transformer-circuits.pub/2026/nla/
local_pdf: null
type: excerpts
note: |
  Quotes verified from the HTML at transformer-circuits.pub/2026/nla/
  via WebFetch on 2026-05-12. No PDF on disk yet; once a downloadable
  preprint is published, set local_pdf and re-anchor against the PDF.
  Some numeric headline figures (3% baseline vs 12-15% NLA-equipped
  auditor success-rate) circulating in secondary coverage were NOT
  surfaced by the WebFetch crawl of the primary page and are marked
  [FORUM-SIGNAL] below pending direct verification.
---

# Excerpts — Anthropic 2026, "Natural Language Autoencoders"

## Abstract — dual AV+AR architecture, RL-trained round-trip {#abstract}

> An NLA consists of two LLM modules: an activation verbalizer (AV)
> that maps an activation to a text description and an activation
> reconstructor (AR) that maps the description back to an activation.

> We jointly train the AV and AR with reinforcement learning to
> reconstruct residual stream activations. Although we optimize for
> activation reconstruction, the resulting NLA explanations read as
> plausible interpretations of model internals.

The pair forms a "dual-way" pipeline whose only bottleneck is natural
language: the AV must encode whatever information the AR will need to
recover the activation, but it must do so in tokens. The round-trip
makes the explanation testable — a confabulated description will
score badly on reconstruction.

## Method — AV / AR maps and reconstruction loss {#sec-method}

The two modules are conditional sequence/embedding maps:

- **Activation verbalizer.** $\mathrm{AV}(z \mid h_\ell)$ takes
  residual-stream activation $h_\ell \in \mathbb{R}^{d_\mathrm{model}}$
  as input, generates a natural-language explanation token sequence
  $z$.
- **Activation reconstructor.** $\mathrm{AR}(z)$ takes the explanation
  $z$ and produces a reconstruction
  $\hat{h}_\ell \in \mathbb{R}^{d_\mathrm{model}}$.

Joint training optimizes the reconstruction MSE under sampled
explanations:

$$
\mathcal{L} = \mathbb{E}_{h_\ell \sim \mathcal{H}}\,
              \mathbb{E}_{z \sim \mathrm{AV}(\cdot \mid h_\ell)}
              \left[\, \lVert h_\ell - \mathrm{AR}(z) \rVert_2^2 \,\right]
$$

Reported quality metric is **Fraction of Variance Explained** (FVE),
normalising the MSE by the residual variance around the mean:

$$
\mathrm{FVE} = 1 - \frac{\mathcal{L}}{\mathbb{E}\!\left[\,\lVert h_\ell - \bar h_\ell \rVert_2^2 \,\right]}
$$

[INTUITION] FVE = 1 would mean the natural-language explanation
losslessly captures the activation; FVE = 0 means the explanation is
no better than always emitting the mean activation. The text token
budget is the bottleneck — a longer or more precise explanation can
push FVE up at the cost of being less human-readable.

## Auditing application — surfacing unverbalized evaluation awareness {#sec-auditing}

> Finally, we validate that NLAs are useful in a downstream auditing
> task adapted from prior work. We task an auditing agent with
> investigating the intentionally-misaligned model introduced in
> Marks et al., and find that NLA-equipped auditing agents more
> effectively discover the root cause of the model's misalignment
> than baselines.

> A key application of NLAs is surfacing unverbalized aspects of
> model cognition, especially unverbalized evaluation awareness — a
> model internally suspecting it is being tested without explicitly
> saying so.

> For instance, we find that NLAs surface evaluation awareness more
> often on evaluation transcripts than real deployment traffic, and
> that rewriting inputs to be more realistic reduces NLA-measured
> evaluation awareness.

[FORUM-SIGNAL] Multiple secondary write-ups (MarkTechPost, MindStudio,
explainx.ai, dated 2026-05-08) report the auditing success rate
"jumped from under 3% baseline to 12–15% with NLAs" on the Marks et al.
misaligned-model investigation; the exact figure is not in the
primary-source excerpt collected on 2026-05-12 and needs direct
verification once a downloadable preprint or the missing section is
located.

## Failure mode — confabulation {#sec-confabulation}

> Confabulation: While NLA explanations can be insightful, they can
> also be noisy ... sometimes include contradictory information or
> verifiably false claims about the context.

The round-trip score gives a calibration signal — a confabulated
explanation will fail to reconstruct accurately — but a single
sampled explanation, taken at face value, can still mislead.
Workflows should sample multiple explanations per activation and
reject those with low FVE.

## Lineage and contrast {#sec-lineage}

[ANALOGY] NLAs sit at the same abstraction level as **sparse
autoencoders** (Templeton 2024; Gao 2024; Rajamanoharan 2024 /
gemma-scope-2024) but swap the SAE's sparse-feature-vector code
for a natural-language code. Where an SAE feature is "interpretable"
only after manual labelling of its top activating examples, an NLA's
output IS the explanation. Where SAE feature labels are descriptive,
NLA explanations are *causally testable* via the AR round-trip.

[CONTRADICTION] The tradeoff against the **logit / tuned lens**
lineage (nostalgebraist 2020; belrose2023-tuned-lens) is sharper:
lenses project a hidden state through the model's own unembedding
to produce a token distribution — same vocabulary, no extra training,
deterministic. NLAs produce free-form text via two RL-trained LLM
modules — more expressive output, more compute, and unlike lenses can
hallucinate. Same goal (read out what a layer "thinks"), different
points on the cheap-deterministic ↔ expressive-stochastic frontier.

## Authors and release context {#sec-release}

> Kit Fraser-Taliente*, Subhash Kantamneni*‡, Euan Ong*, Dan Mossing,
> Christina Lu, Paul C. Bogdan [and 10 others]
> Published: May 7, 2026

Released at transformer-circuits.pub/2026/nla/ as part of the
Transformer Circuits Thread — Anthropic's house interpretability
publication venue. No arXiv preprint listed in the primary-source
crawl on 2026-05-12; check periodically for a preprint companion.

[Verified verbatim from HTML on 2026-05-12]
Sections marked verbatim: §method (AV/AR + reconstruction loss),
§auditing (auditing-agent setup + evaluation-awareness finding),
§confabulation (failure mode). Sections marked [FORUM-SIGNAL]:
3%-vs-12-15% auditing headline. Sections marked [INTUITION] or
[ANALOGY]: not in primary source — author framing.
