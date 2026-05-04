---
topic: post-training/sft
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - ouyang2022-instructgpt  # canonical 3-stage RLHF; SFT is stage 1
  - tulu3  # current open SOTA: data curation -> SFT -> DPO -> RLVR
  - meta-llama3  # rejection sampling as SFT quality gate
  - qwen3  # 4-stage post-training inc. thinking/non-thinking SFT
  - magpie  # alignment data synthesis from scratch
  - deepseek-r1  # 800K SFT distillation set for reasoning
related_topics:
  - post-training/rlhf
  - post-training/dpo-and-offline
  - post-training/rlvr-and-grpo
  - training/synthetic-data-and-distillation
---

# Supervised Fine-Tuning (SFT)

**Status:** stub. Drafted from Phase 1 landscape sweep; needs full Phase 2 treatment.

## What this topic covers

SFT is **stage 1 of essentially every modern post-training pipeline**:
maximum-likelihood fine-tuning of a base LM on curated
prompt+response pairs, producing the instruction-following starting
point $\pi^{\mathrm{SFT}}$ that all subsequent stages (DPO, PPO, RLVR)
take as their reference. The objective is plain cross-entropy:

$$
\mathcal{L}_{\mathrm{SFT}}(\pi_\theta) = -\mathbb{E}_{(x,y)\sim\mathcal{D}_{\mathrm{SFT}}}\left[\log\pi_\theta(y|x)\right].
$$

The interesting research questions are not in the loss but in the
**data**: how to curate, synthesize, deduplicate, balance, and quality-
gate the SFT set. As of 2026, the canonical open reference is **Tülu
3** (data curation → SFT → DPO → RLVR pipeline, fully open). Llama-3
established **rejection sampling** as a core quality gate. **MAGPIE**
(ICLR 2025) demonstrated near-zero-cost instruction synthesis by
prompting aligned LLMs with empty pre-query templates. The **DeepSeek-R1
distillation set** (800K samples, 600K reasoning + 200K non-reasoning)
showed that reasoning capability is transferable via pure SFT to small
models, displacing more complex reasoning-distillation methods.

## Primary sources to read (in order)

1. `ouyang2022-instructgpt` — Ouyang et al. 2022, "Training Language
   Models to Follow Instructions" (arXiv:2203.02155). Canonical
   3-stage pipeline; SFT is stage 1. Read §3.1 (data collection) and
   §3.5 (SFT training details).
2. `tulu3` — Wang et al. 2024 (arXiv:2411.15124). Fully open 4-stage
   recipe. Read §3 (data curation) and §4 (SFT training).
3. `meta-llama3` — Llama 3 Herd of Models (arXiv:2407.21783). §4
   covers the iterative SFT + rejection sampling + DPO loop at scale.
4. `qwen3` — Qwen 3 Technical Report (arXiv:2505.09388). 4-stage
   post-training including thinking/non-thinking mode SFT.
5. **MAGPIE** — Xu et al. ICLR 2025 (URL in Phase 1 sweep). Alignment
   data synthesis from scratch; near-zero-cost instruction generation.
6. `deepseek-r1` — DeepSeek-R1 paper (arXiv:2501.12948) §4 covers
   the 800K reasoning-distillation SFT set used to transfer reasoning
   to smaller models.

## Key claims to ground (Phase 2 todo)

- The loss form (MLE / NLL on response tokens, with the prompt masked
  out of the loss).
- **Rejection sampling as quality gate** (Llama-3): sample $N$
  candidates, score by reward model, keep top-$k$, SFT on those.
- **Scale**: InstructGPT used ~13K demonstrations; Tülu 3 SFT set
  totals ~1M+ samples; Qwen 2.5 reports >1M samples; Phi-4 uses 400B
  tokens of synthetic SFT.
- **Data composition**: math, code, reasoning, multi-turn, refusal /
  safety, multilingual — the typical Tülu/Qwen/OLMo split.
- **MAGPIE's pre-query template trick**: prompt an aligned LLM with
  only its instruction-template prefix (no actual question), have it
  fabricate the question first, then sample the answer.
- **Reasoning-distillation SFT** (DeepSeek-R1): 800K samples generated
  by R1, applied to 1.5B–32B base models via pure SFT (no RL),
  achieving near-o1 performance at small scale.
- **Thinking/non-thinking mode SFT** (Qwen 3): training the model to
  emit `<think>...</think>` blocks gated by a system prompt.
- **The "1/3 synthetic, 2/3 natural" heuristic** for synthetic-data
  fraction in SFT, and its empirical basis (or lack thereof — Phase 1
  sweep §6 lists this as an open question).
- **Loss masking** — by convention, the prompt tokens are excluded
  from the SFT loss; only response tokens contribute. Different
  conventions in the field for multi-turn (mask user, train on
  assistant; or train on all assistant turns including system).

## Related notes

- `kb/notes/post-training/rlhf.md` — PPO-RLHF starts from $\pi^{\mathrm{SFT}}$.
- `kb/notes/post-training/dpo-and-offline.md` — DPO starts from
  $\pi^{\mathrm{SFT}}$ and uses it as $\pi_{\mathrm{ref}}$.
- `kb/notes/post-training/rlvr-and-grpo.md` — DeepSeek-R1's cold-start
  SFT is the starting point for the RL stage.
- `kb/notes/training/synthetic-data-and-distillation.md` — overlapping
  with this topic; the source of synthetic SFT data.
