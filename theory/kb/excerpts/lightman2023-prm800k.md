---
paper_key: lightman2023-prm800k
title: "Let's Verify Step by Step"
authors: Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever, Cobbe
year: 2023
venue: ICLR 2024
arxiv: 2305.20050
local_pdf: null
type: excerpts
note: |
  Verbatim abstract from the arXiv abstract page (v3, Oct 2023).
  PDF was not downloaded during this Phase 2 pass (sandbox/curl
  restriction); body-section quotes are pending PDF acquisition.
  Section anchors are stable for follow-up population.
---

# Excerpts — Lightman et al. 2023, "Let's Verify Step by Step"

## Abstract — verbatim {#abstract}

> In recent years, large language models have greatly improved in their
> ability to perform complex multi-step reasoning. However, even
> state-of-the-art models still regularly produce logical mistakes. To
> train more reliable models, we can turn either to outcome supervision,
> which provides feedback for a final result, or process supervision,
> which provides feedback for each intermediate reasoning step. Given
> the importance of training reliable models, and given the high cost
> of human feedback, it is important to carefully compare the both
> methods. Recent work has already begun this comparison, but many
> questions still remain. We conduct our own investigation, finding
> that process supervision significantly outperforms outcome
> supervision for training models to solve problems from the
> challenging MATH dataset. Our process-supervised model solves 78% of
> problems from a representative subset of the MATH test set.
> Additionally, we show that active learning significantly improves
> the efficacy of process supervision. To support related research, we
> also release PRM800K, the complete dataset of 800,000 step-level
> human feedback labels used to train our best reward model.

## Headline numbers {#headline}

- Process-supervised model solves **78%** of problems from a
  representative MATH test subset.
- **PRM800K**: 800,000 step-level human feedback labels released
  openly.
- Active learning is shown to improve PRM data efficiency.

## §1 Introduction — process vs outcome supervision {#sec-1}

[NOTE — pdf-not-available] Targets: the framing of the contrast
between PRM and ORM; the authors' working definitions; the headline
claim that process supervision (PRM800K-trained verifier)
substantially outperforms outcome supervision on MATH best-of-$N$.

## §2.1 ORM training objective {#sec-2-1}

[NOTE — pdf-not-available] Targets: the binary cross-entropy training
objective for outcome reward models, with full notation. The grounding
for Eq. (1) in `kb/notes/reasoning/process-supervision.md`.

## §2.2 PRM training objective {#sec-2-2}

[NOTE — pdf-not-available] Targets: the per-step binary cross-entropy
objective for process reward models, with the conditioning on prefix
$s_{1:t}$ explicit. The grounding for Eq. (2) in `process-
supervision.md`.

## §3 PRM800K dataset construction {#sec-3}

[NOTE — pdf-not-available] Targets: the dataset-construction details:
~12K problems from the MATH train split; ~800K step-level labels;
human-annotation protocol; label classes (positive / negative /
neutral); inter-annotator agreement.

## §4 Headline experiment: PRM > ORM at matched N {#sec-4}

[NOTE — pdf-not-available] Targets: the exact best-of-$N$ comparison
from Figure 4. The PRM substantially outperforms ORM and majority-vote
baselines across $N \in [1, 1024]$.

## §4.2 Aggregation rules — min wins {#sec-4-2}

[NOTE — pdf-not-available] Targets: the comparison of min vs product
vs last-step aggregation. The reported finding is that **min** is the
strongest aggregator at this PRM grade.

## §4.3 PRM-weighted self-consistency {#sec-4-3}

[NOTE — pdf-not-available] Targets: the variant where self-consistency
majority-vote is weighted by PRM scores; the gain over plain self-
consistency.

## §5 Discussion — why PRMs win {#sec-5}

[NOTE — pdf-not-available] Targets: the authors' proposed mechanism —
PRMs catch local errors that the ORM only sees through final-answer
signal; discussion of when ORM and PRM converge; the relationship to
credit assignment.

## Notes for the populator

- Dataset name "PRM800K" is from this paper; verify dataset-release
  URL when populating §3 (HuggingFace / GitHub).
- The generator is a GPT-4 variant; this matters for the trace
  distribution PRMs are calibrated to. Note when populating §4.
- Section numbers anchor to the v3 (Oct 26 2023) arXiv revision.
- Download command: `curl -sL -o theory/sources/papers/lightman2023_
  prm800k.pdf "https://arxiv.org/pdf/2305.20050"`.
