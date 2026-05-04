---
topic: training/synthetic-data-and-distillation
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - phi4
  - deepseek-r1
  - burns2023-w2s
secondary_sources:
  - deepseek-v3       # R1→V3 distillation handoff
  - fineweb2024       # Llama-3-70B-as-classifier (FineWeb-Edu)
related_topics:
  - training/pre-training-data
  - post-training/sft
  - post-training/rlvr-and-grpo
  - reasoning/reasoning-training
  - alignment/oversight-and-scalable-alignment
---

# Synthetic data and distillation

The class of pre-training and post-training techniques where an
*LLM-generated* corpus, not a human-written one, is the primary signal.
Two practical regimes dominate the 2024–2026 frontier:

1. **Pre-training synthetic corpora** — the Phi-4 recipe: tens of
   billions of synthetic tokens stand in for organic web text, optimized
   for reasoning and STEM.
2. **Reasoning distillation** — the R1 recipe: a strong reasoning teacher
   produces hundreds of thousands of long-CoT trajectories that are used
   as pure SFT data to elevate small students.

A third, recursive regime — **weak-to-strong** — is the
scalable-oversight framing where a weaker supervisor's labels are used
to elicit stronger student capabilities. It is treated under
`kb/notes/alignment/oversight-and-scalable-alignment.md`; we cite it
here only for the lineage of "labels are cheap if you can generate
them."

## 1. The two operational definitions

### 1.1 Synthetic data (pre-training-substitute)

Let $\mathcal{D}_{\text{org}}$ be a curated organic corpus and let
$\pi_{\text{teacher}}$ be a (typically already-aligned) generator LM.
A synthetic dataset $\mathcal{D}_{\text{syn}}$ is constructed by a
multi-stage pipeline $\mathcal{P}$ that takes seeds $s \in
\mathcal{S}_{\text{seed}} \subset \mathcal{D}_{\text{org}}$, prompts
$\pi_{\text{teacher}}$ with rewrites/exercises/CoT-elicitations, and
optionally critiques + revises the output:

$$\mathcal{D}_{\text{syn}} = \{\, \mathcal{P}(s; \pi_{\text{teacher}}) \mid s \in \mathcal{S}_{\text{seed}}\,\}.$$

The student LM $\pi_\theta$ is then pre-trained (or mid-trained) on a
mixture $\alpha \mathcal{D}_{\text{syn}} + (1-\alpha)
\mathcal{D}_{\text{org}}$ for some fraction $\alpha$ that, in
production recipes (Phi-4), is the *majority* of the training tokens
`[phi4 §1.pillars; kb/excerpts/phi4#sec-1-pillars]`.

### 1.2 Distillation (post-training-imitation)

Distillation in the modern LLM sense is *response-level imitation*: a
strong teacher $\pi_{\text{teacher}}$ samples responses $y \sim
\pi_{\text{teacher}}(\cdot \mid x)$ on a prompt set $\{x\}$, and the
student is fit to those responses by next-token cross-entropy:

$$\mathcal{L}_{\text{distill}}(\theta) = -\mathbb{E}_{x, y \sim \pi_{\text{teacher}}(\cdot|x)} \log \pi_\theta(y \mid x).$$

Note this is *plain SFT*; the only structural change vs. classical
SFT is who wrote $y$. Logit-level Hinton-style distillation (matching
softmax distributions, KL divergence on logits) is rarely used at LLM
scale because (a) teacher logits over a 100k+ vocabulary are expensive
to store and (b) frontier teachers often only expose token streams
[INTUITION].

## 2. Mechanism — three canonical recipes

### 2.1 Phi-4: 50 pipelines, ~400B synthetic tokens

The Phi-4 pre-training corpus is "centrally focused on data quality"
with synthetic data as the *bulk*, not a supplement
`[phi4 abstract; kb/excerpts/phi4#abstract]`. The pipeline is
explicitly enumerated:

> "We created **50 broad types of synthetic datasets**, each one
> relying on a different set of seeds and different multi-stage
> prompting procedure, … accumulating to a total of about **400B
> unweighted tokens**." `[phi4 §2.2; kb/excerpts/phi4#sec-2-2-pipelines]`

Three primitives are reused across pipelines:

1. **Seed curation.** Web/code/book passages are filtered for
   educational and reasoning value, then segmented into passages and
   re-scored. Question banks are filtered by *plurality voting*:
   discard questions where all teacher answers agree (too easy) or
   where they disagree entirely (too hard/ambiguous), keeping the
   middle band where teacher samples partially agree
   `[phi4 §2.2; kb/excerpts/phi4#sec-2-2-seeds]`.
2. **Rewrite/augment.** Seeds are transformed by multi-step prompting
   into exercises, dialogues, structured reasoning tasks
   `[phi4 §2.2; kb/excerpts/phi4#sec-2-2-rewrite]`.
3. **Self-revision.** Initial responses are iteratively refined by a
   model critiquing and improving its own output under reasoning/
   factual rubrics
   `[phi4 §2.2; kb/excerpts/phi4#sec-2-2-rewrite]`.

Phi-4 is guided by four principles for the generated corpus —
**diversity, nuance/complexity, accuracy, chain-of-thought**
`[phi4 §2.1; kb/excerpts/phi4#sec-2-1-principles]`. The headline
empirical claim is that the resulting 14B model **surpasses GPT-4o
(its teacher) on GPQA and MATH** despite being roughly an order of
magnitude smaller in inference cost
`[phi4 §1; kb/excerpts/phi4#sec-1-stem]`.

### 2.2 R1 reasoning distillation: long-CoT teacher → small SFT student

DeepSeek-R1 (Jan 2025) operationalizes a different recipe: a single
reasoning-RL-trained teacher (DeepSeek-R1) generates **800K samples**
(600K reasoning + 200K non-reasoning) which are used as **pure SFT
data** to fine-tune small dense models (1.5B, 7B, 14B, 32B, 70B). No
RL is applied to the students
`[deepseek-r1 §1.intro;
kb/excerpts/deepseek-r1#abstract]`.

The methodology proceeds in three stages:

1. **Train teacher with RLVR.** R1-Zero is GRPO-trained on
   DeepSeek-V3-Base with rule-based verifiable rewards (math/code).
   AIME 2024 pass@1 climbs from 15.6% → 77.9% over training
   `[deepseek-r1 §2.3; kb/excerpts/deepseek-r1#sec-2-3-aime]`.
2. **Sample distillation set.** R1 (the multi-stage refined version)
   produces long-CoT responses on a curated prompt distribution. The
   600K reasoning slice covers math, code, science, logic; the 200K
   non-reasoning slice (writing, dialogue) prevents the student from
   *only* knowing how to think aloud.
3. **SFT the student.** Plain next-token cross-entropy. No RL.

The R1 abstract puts the lineage explicitly:

> "the emergent reasoning patterns exhibited by these large-scale
> models can be **systematically harnessed to guide and enhance the
> reasoning capabilities of smaller models**." `[deepseek-r1 abstract;
> kb/excerpts/deepseek-r1#abstract]`

DeepSeek-V3 itself uses the same handoff in its post-training:

> "We introduce an innovative methodology to distill reasoning
> capabilities from the long-Chain-of-Thought (CoT) model, specifically
> from one of the DeepSeek R1 series models, into standard LLMs,
> particularly DeepSeek-V3." `[deepseek-v3 §5.4.1;
> kb/excerpts/deepseek-r1#distillation-set]`

### 2.3 Distill-then-classify (FineWeb-Edu)

A hybrid pattern that uses the teacher not as a generator of tokens
but as a *labeler*: Llama-3-70B-Instruct scores 460k FineWeb pages
0–5 on educational quality, those scores train a small linear
regressor on Snowflake-arctic-embed embeddings, and the regressor is
applied to all 15T tokens of FineWeb at a cost of 6,000 H100-hours
`[fineweb2024 §4; kb/excerpts/fineweb2024#sec-4-edu]`.

This is *distillation of judgment* into a static classifier: the
teacher's judgment is amortized into a frozen filter that costs O(1)
per page rather than an LLM call. The output FineWeb-Edu (1.3T
tokens) outperforms FineWeb on MMLU/ARC. This is the path that
several frontier labs (Llama 3, Phi-3) used internally before FineWeb
made it public.

## 3. Variants and lineage

| Recipe | Teacher | Student | Phase | Tokens | Headline |
|---|---|---|---|---|---|
| **Phi-1/1.5/2/3/4** lineage | GPT-4 (Phi-3), self+GPT-4 (Phi-4) | 14B dense | Pre-train + mid-train | ~400B synthetic + organic | 14B beats GPT-4o on GPQA, MATH `[phi4#sec-1-stem]` |
| **R1 distillation** | DeepSeek-R1 (long-CoT, 671B-A37B) | 1.5B–70B dense | Post-train SFT | 800K samples (600K reasoning) | distill-32B ≈ o1-mini on reasoning |
| **R1 → V3 self-handoff** | R1 (R1's reasoning patterns) | DeepSeek-V3 | Post-train | not disclosed | "verification and reflection patterns of R1 into V3" `[deepseek-v3 §5.4.1]` |
| **FineWeb-Edu** (distill-then-classify) | Llama-3-70B-Instruct (judge) | classifier → 1.3T pretraining tokens | Pre-train data filter | 460k labels → 15T scored | dramatic MMLU/ARC gains over FineWeb `[fineweb2024 abstract]` |
| **MAGPIE** (Phase 1 only — not yet excerpted) | Aligned Llama-3-Instruct | any SFT student | Post-train SFT | 4M instruction pairs | beats UltraFeedback SFT+DPO baselines |
| **Weak-to-strong** | weak human-equivalent (e.g. GPT-2-class) | strong (GPT-4-class) | RLHF analogy | varies | strong student recovers ~80% of strong-supervisor performance `[burns2023-w2s; kb/excerpts/burns2023-w2s]` |

The lineage is **not** "synthetic data is just cheap data": Phi-4 is
an explicit claim that *quality* (not cost) is the reason synthetic
tokens dominate organic for STEM reasoning training
`[phi4 abstract]`. The Phi authors articulate the case in §2.1:

[INTUITION] **Spoonfeeding alignment.** Phi-4 §2.1 argues organic
text has "complex and indirect" token-to-token relationships — many
implicit reasoning steps may separate one token from the next, making
next-token prediction a noisy training signal. LLM-generated text is
by construction predictable from its own prefix, so synthetic
sequences "may act as a form of *spoonfeeding*, presenting challenges
in a digestible and progression-oriented manner" `[phi4 §2.1;
kb/excerpts/phi4#sec-2-1-purpose]`. Returning to canonical form: the
loss $-\log \pi_\theta(y \mid x)$ has lower variance per token when
$y$ was itself generated autoregressively under a related conditional
distribution.

[INTUITION] **Inference-context alignment.** Synthetic data is
"typically closer to the format of outputs we expect our models to
generate" `[phi4 §2.1; kb/excerpts/phi4#sec-2-1-purpose]`. A
classical web-forum corpus distributes mass over conversational
patterns the deployed model will rarely produce; an LLM-rewritten
version of the same factual content distributes mass over the kind
of structured prose the deployed model will produce. This narrows
the train–test distribution gap.

## 4. Intuitions and analogies

[ANALOGY] **Distillation as a syllabus.** A frontier teacher acts as
a curriculum designer: rather than asking the student to learn from
the raw web (heterogeneous, redundant, noisy), the teacher writes a
textbook tailored to the student's current capability level. The
analogy returns to canonical form: the SFT loss $-\log \pi_\theta(y
\mid x)$ on teacher-written $(x, y)$ pairs is the textbook's "answer
key," and the gradient step is one act of the student copying the
solution. The textbook framing is helpful for orienting students; it
is not a separate claim. The Phi-4 authors use almost this language:
synthetic data acts as "spoonfeeding"
`[phi4 §2.1; kb/excerpts/phi4#sec-2-1-purpose]`.

[ANALOGY] **R1 distillation as compression of search.** The teacher
spent enormous compute (RL rollouts, group sampling, KL regularization
to a reference) to discover good reasoning trajectories. SFT on
those trajectories *compresses* the search outcome into the student's
weights without re-running the search. The analogy returns to math:
$\pi_\theta$ converges, in the cross-entropy sense, to the
distribution induced by $\pi_{\text{teacher}}$ on the prompts where
SFT data were collected — a distillation upper bound that is tight
when the student has sufficient capacity.

[CONTRADICTION] **Does distillation transfer reasoning capacity, or
just reasoning style?** A NeurIPS 2025 oral (`rlvr-limits-2025`,
arXiv:2504.13837) finds that RLVR "improves sampling efficiency
(pass@1) on problems the base model can already solve, but does NOT
expand the model's underlying reasoning capacity" — see
`kb/notes/post-training/rlvr-and-grpo.md` and the Phase 1 sweep
report. By contrast the R1 distillation result is widely interpreted
as *transferring reasoning capability* to small students. The two
findings are not strictly contradictory (RLVR is the teacher's
training, distillation is the student's training, and the teacher
was already much larger), but the implication that distillation
unlocks capabilities the student "could not have" alone is contested.
Treat distill-32B vs base-32B comparisons as informative but not yet
mechanistically resolved.

[CONTRADICTION] **Model collapse from recursive synthetic mixing.** A
line of work (Shumailov et al. 2024, et seq.) argues that training
LMs on their own outputs *recursively* leads to mode collapse: tail
distributions are forgotten, divergence from the original data
distribution accumulates. Phi-4's report does not address this
directly — Phi-4 trains on synthetic data from a *different* model
(GPT-4) but its own 14B model could in principle be retrained on
its own outputs. The "1/3 synthetic" community heuristic for safe
mixing has limited empirical basis (Phase 1 sweep, OQ #4). Treat as
unresolved.

## 5. Frontier and open questions (as of 2026-05)

- **Optimal synthetic fraction.** No large-scale systematic study
  exists of the collapse-vs-benefit tradeoff $\alpha$ in $\alpha
  \mathcal{D}_{\text{syn}} + (1-\alpha) \mathcal{D}_{\text{org}}$
  across model scale and domain. Phi-4 chose ~majority synthetic for
  a 14B model targeting STEM; whether this transfers to 70B+ general-
  purpose is open.
- **Teacher–student gap.** Phi-4 surpasses GPT-4 on STEM, which is
  surprising under classical distillation theory (student bounded by
  teacher). The Phi authors' explanation is that pipelines + filtering
  + post-training "go beyond distillation" `[phi4 abstract]` —
  precisely *what* part of the pipeline lifts the student above the
  teacher is not isolated.
- **Reasoning distillation ceiling.** Distill-32B nearly matches
  o1-mini, but distill-7B trails. Where the ceiling sits as a function
  of student parameter count, and whether recursive R1-style RL on a
  distilled student helps, is an active question. (See
  `kb/notes/post-training/rlvr-and-grpo.md` and Phi-4-reasoning,
  which extends the recipe with o3-mini demonstrations.)
- **MAGPIE-style template prompting at frontier scale.** MAGPIE uses
  the alignment template's *pre-query* prefix to elicit instruction
  pairs from an aligned LLM with no seed prompt; ICLR 2025 result
  beats prior SFT+DPO combos. Whether this generalizes to harder
  reasoning data (vs. general instruction following) is open.
- **Data-mixing laws apply to synthetic too?** Data Mixing Laws
  `[data-mixing-laws-2024]` cover organic-domain proportions; an
  analogous law over synthetic-pipeline mixtures (the 50 Phi-4
  pipelines as "domains") has not been published. This is a natural
  Phase 2 extension.

## 6. See also

- `kb/notes/training/pre-training-data.md` — where synthetic data sits
  alongside organic web filters in the upstream pipeline.
- `kb/notes/post-training/sft.md` — the loss form is plain SFT; what
  changes is the data source.
- `kb/notes/post-training/rlvr-and-grpo.md` — the upstream side of
  R1's distillation handoff: how the teacher itself was trained.
- `kb/notes/reasoning/reasoning-training.md` — the long-CoT data the
  R1 distillation set is built from.
- `kb/notes/alignment/oversight-and-scalable-alignment.md` —
  weak-to-strong (Burns et al. 2023) is the inverted distillation
  setting (weak teacher → strong student) and is the formal frame for
  scalable oversight.
