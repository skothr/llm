---
topic_key: adaptation-and-merging
area: training
status: stub
last_updated: 2026-05-04
primary_sources: []
secondary_sources: [phi4, deepseek-v3]
related_topics: [pre-training-data, optimization]
---

# Adaptation & merging

> **Status: STUB.** Skeleton; needs primary excerpts before promotion.

## Scope

Techniques that *modify a trained model without full re-training*.
Distinct from RL/SFT post-training (covered under `post-training/`)
because the unit of action is *parameter-level* — adding, averaging,
mixing, or low-rank-updating weights — rather than gradient-via-loss
on data. Four sub-areas:

1. **Continual / late-stage pre-training.** Fixed-architecture, fixed-
   data-pipeline, but mixture $\mathbf{r}$ shifts towards higher-quality
   sources at the end of training (Phi-4's "midtrain" stage; OLMo 2's
   Dolmino mix; DeepSeek-V3's context-extension stage at 119 K H800-hr
   `[deepseek-v3 §1; kb/excerpts/deepseek-v3-training#sec-1-cost]`).
   Connects directly to [pre-training-data](pre-training-data.md)
   §frontier-question-2.
2. **Parameter-efficient fine-tuning (PEFT).** LoRA, DoRA, IA3,
   prefix-tuning, etc. — adapt a frozen base via low-rank or sparse
   updates. Need primary excerpts (Hu et al. 2021 LoRA; Liu et al.
   2024 DoRA).
3. **Model merging.** Average / interpolate / SLERP weights from two or
   more trained models that share a common ancestor. Subtypes:
   - *Linear weight averaging* (Model Soups, Wortsman et al. 2022)
   - *Task arithmetic* (Ilharco et al. 2022) — vectors of "fine-tune
     direction" added/subtracted
   - *TIES-merging* (Yadav et al. 2023) — sign-conflict resolution
   - *DARE* (Yu et al. 2024) — sparsify before merging
   - *Evolutionary merging* (Akiba et al. 2024, Sakana) — search
     merge coefficients with a black-box optimizer
   - *MergeKit* (2024) — community toolkit consolidating the above.
4. **Pruning & weight surgery.** SparseGPT, Wanda, post-training
   structured pruning, layer-drop. Closer to inference-time techniques
   in practice but the boundary blurs when pruning is followed by
   recovery fine-tuning.

## Open questions for the full draft

- **Why merging works at all.** The dominant theory is that fine-tunes
  of a shared base land in a connected loss basin (linear mode
  connectivity); empirically this holds for SFT-from-the-same-base
  and partly for RLHF-from-the-same-base, breaks for from-scratch
  models. Needs primary citation (Frankle, Dziugaite et al. 2020).
- **Merging vs. distillation as compression of multiple skills into one
  model.** Merging is parameter-level and free at inference; distillation
  is data-level and pays a teacher pass. Comparative empirical results
  exist but no unified theory.
- **What `[FORUM-SIGNAL]` "frankenmerges" actually do.** Layer-wise
  interleaving of two models (block-diagonal merging) is widely used in
  the OSS community (e.g. Mistral derivatives) but has limited primary
  literature. Distinguish from formal model merging above.

## Pending primary sources to add

- Wortsman et al. 2022 *Model Soups* — `model-soups-2022` paper-key,
  needs PDF + excerpt.
- Akiba et al. 2024 *Evolutionary Optimization of Model Merging Recipes*
  — `evolutionary-merging-2024`, needs PDF + excerpt.
- Hu et al. 2021 *LoRA* — `hu2021-lora`, needs PDF + excerpt.
- Goddard et al. 2024 *MergeKit* — `mergekit2024`, needs PDF + excerpt.
- OLMo 2 tech report — `olmo2-2024`, needs PDF + excerpt; covers Dolmino
  late-stage mix.

Until these excerpts exist, this note cannot make hard claims. The above
list is a *to-do*, not a citation.

## Cross-references

- [pre-training-data](pre-training-data.md) §frontier-question-2 — late-
  stage curriculum / mixture schedules is the continuous-curriculum end
  of "adaptation."
- [synthetic-data-and-distillation](synthetic-data-and-distillation.md)
  — distillation is the data-level alternative to weight-merging.
