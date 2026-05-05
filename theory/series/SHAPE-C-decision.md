# Series structure — Shape C (5 thesis papers)

_2026-05-04. Decision lifted from `theory/plans/2026-05-04-latex-series-brainstorm.md`. See that file for the full A-E comparison and tradeoffs._

Each paper is organized around a **load-bearing thesis** the KB makes defensible. Surveys describe; theses commit. The KB's contradictions cluster naturally — areas with the most contention (architecture, reasoning, alignment) lead the paper-set, with two tighter papers covering training/post-training and interpretability/evaluation.

## The five theses

### Paper 1 — *The modern Transformer is a small set of choices*

**Thesis:** Frontier LLMs in 2026 are converging on a small, identifiable set of architectural choices along a few independent axes (KV-sharing, attention-implementation, FFN, normalization, positional encoding). The convergence is empirically describable and the open frontiers are localized.

**KB anchors:** `kb/notes/architecture/` (13 notes), `kb/excerpts/vaswani2017,shazeer2019,ainslie2023,dao2022,dao2023,shah2024,deepseek-v2,yuan2025,shazeer2020,zhang2019,xiong2020,su2021,gloeckle2024-mtp,gu2023-mamba,alayrac2022-flamingo,kudo2018-sentencepiece,sennrich2016,press2017-tied-embed,radford2019-gpt2,blt2024,llava2023.md`.

**Inference adjuncts folded in:** kv-cache, FA1/2/3, NSA, quantization regimes, structured output as constrained decoding, speculative decoding (architectural side).

**Contradictions to engage:** MLA-vs-MHA at scale (single-source), SSM-hybrid quality at frontier, NSA vs FA3 head-to-head, Pre-LN vs Post-LN at depth.

### Paper 2 — *Training is a multi-stage pipeline, not a single objective*

**Thesis:** Modern frontier-LLM training is a six-stage pipeline (pre-training-data → mixed-precision/distributed scaffolding → SFT → preference-RL → reasoning-RL → adaptation/merging). Each stage's load-bearing variables (token count, mixture, precision, batch size, KL constraints, verifier signal, merge geometry) are now well-characterized. The pipeline composes; pretending it is one objective produces persistently confused empirics.

**KB anchors:** `kb/notes/training/` (6 notes), `kb/notes/post-training/` (5 notes), `kb/notes/scaling/{kaplan-laws,chinchilla,scaling-frontier,mu-transfer}.md`. `kb/excerpts/{kaplan2020,hoffmann2022-chinchilla,yang2022-mup,fineweb2024,data-mixing-laws-2024,micikevicius2017-mixed-precision,kalamkar2019-bfloat16,torchtitan2024,ouyang2022-instructgpt,bai2022-cai,rafailov2023-dpo,meng2024-simpo,burns2023-w2s,phi4,muon-moonlight2025,hu2021-lora,dettmers2023-qlora,wortsman2022-model-soups,ilharco2022-task-vectors,yadav2023-ties-merging}.md`.

**Contradictions to engage:** Kaplan vs Chinchilla optimal scaling, RLHF reward-hacking vs RLAIF stability, DPO vs PPO at scale, model-merging guarantees, FP8/NVFP4 production status.

### Paper 3 — *Reasoning is compute, search, and verification*

**Thesis:** Reasoning-LLM gains since 2022 are a triangle: (a) inference-time compute scaling (CoT, sampling, search trees), (b) verifier signals at inference (PRMs, tree-search, MCTS), (c) RL on verifiable rewards using those signals (RLVR, GRPO, DAPO). Strong reasoning models are a co-design of all three; isolating any one understates the gains.

**KB anchors:** `kb/notes/reasoning/` (5 notes), `kb/notes/scaling/inference-time-compute-scaling.md`, `kb/notes/post-training/rlvr-and-grpo.md`. `kb/excerpts/{wei2022-cot,kojima2022,wang2022-self-consistency,lightman2023-prm800k,snell2024,deepseek-r1,rstar-math2025,zhang2024-rest-mcts,wei2025-tree-search-survey,hu2025-mcts-rag,she2025-r-prm,zheng2025-prm-survey,cao2025-edu-prm,thinkprm2025,arcuschin2025-cot-wild,shen2025-faithcot-bench,mehta2026-faithfulness-scaling,dapo2025,shao2024,s1-2025}.md`.

**Contradictions to engage:** PRM utility post-R1 (diminishing? yes/no), CoT faithfulness scaling, RLVR generalization vs memorization, search-vs-RL tradeoff under fixed compute.

### Paper 4 — *The internal computation can be partially read*

**Thesis:** Five interpretability methods (lenses, probes, SAEs, activation patching, circuit tracing) make different commitments about what counts as a feature, a circuit, or an explanation. Cross-method evidence converges in some cases (e.g., IOI circuit) and diverges in others (e.g., truth direction across datasets). The state of the art is method-pluralistic for principled reasons, not yet-to-be-resolved disagreement.

**KB anchors:** `kb/notes/interpretability/` (6 notes). `kb/excerpts/{nostalgebraist2020-logit-lens,belrose2023-tuned-lens,alain-bengio-2017,hewitt2019-structural-probe,belinkov2022-probing-survey,marks-tegmark-2023-truth,gao2024-topk-saes,rajamanoharan2024-jumprelu,gemma-scope-2024,dunefsky2024-transcoders,leask2025-sae-not-canonical,meng2022-rome,wang2022-ioi,conmy2023-acdc,zhang2023-apatching,heimersheim2024}.md`.

**Contradictions to engage:** SAE canonical-feature claim, lens cross-model unreliability, probe correlation vs causation, circuit-tracing scalability past Anthropic-2025.

### Paper 5 — *What we measure and what slips through*

**Thesis:** The eval/alignment landscape is a layered defense: knowledge-benchmark contamination → reasoning-benchmark validity → agentic-benchmark realism → safety-evaluation robustness → alignment-threat (sycophancy, scheming, alignment-faking) detection. Each layer has known failure modes; the field's current bar is honest about which threats it can measure today and which slip through.

**KB anchors:** `kb/notes/evaluation/` (4 notes), `kb/notes/alignment/` (5 notes). `kb/excerpts/{hendrycks2021-mmlu,wang2024-mmlu-pro,glazer2024-frontiermath,phan2025-hle,rein2023-gpqa,jimenez2024-swebench,xie2024-osworld,mialon2023-gaia,yao2024-tau-bench,liang2022-helm,contamination-survey-2025,mmlu-redux-2024,harmbench2024,chao2024-jailbreakbench,wei2023-jailbroken,russinovich2024-crescendo,kirchenbauer2023-watermark,greenblatt2024-alignment-faking,meinke2024-apollo-scheming,sharma2023-sycophancy,irving2018-debate}.md`.

**Contradictions to engage:** MMLU saturation reality vs reported, ASR-comparability across jailbreak benchmarks, scheming detection methodology, alignment-faking generalization, watermarking robustness against fine-tuning.

## Cross-paper threads

A few topics legitimately cross paper boundaries; we name the load-bearing home and reference from elsewhere:

- **MoE** lives in Paper 1 (architecture); cited from Paper 2 (training: load-balancing, fine-grained experts).
- **MLA** lives in Paper 1 (KV compression); cited from Paper 4 (interpretability of compressed-KV models).
- **CoT** lives in Paper 3 (reasoning); referenced from Paper 4 (faithfulness/probing) and Paper 5 (CoT as an alignment surface).
- **Process-supervision** lives in Paper 3; referenced from Paper 5 (PRMs as eval).
- **Scaling laws** (Kaplan/Chinchilla) live in Paper 2; reference from Paper 3 (inference-time compute as a parallel axis).

## Cross-paper sequencing

Reading order: 1 → 2 → 3 → 4 → 5. Each paper assumes the prior:

- Paper 2 assumes Paper 1's notation (residual stream, FFN, attention).
- Paper 3 assumes Paper 2's pipeline (post-training stages, especially RL phase).
- Paper 4 assumes Paper 1's mechanism + Paper 2's training story (probes built on training-time activations).
- Paper 5 assumes Papers 2-4 (eval reaches inside training pipeline + interpretability methods).

## Page-budget targets

| Paper | Target | Drives |
|------:|-------:|--------|
| 1 | 80 pages | foundation: most-used downstream |
| 2 | 100 pages | broadest scope (6 stages × ~15pp each) |
| 3 | 80 pages | thesis is compact even though the field is dense |
| 4 | 70 pages | five-method comparison + worked examples |
| 5 | 70 pages | fewer formal mechanisms, more empirical claims |
| **Total** | **400 pages** | five paper-monographs, not minor surveys |

## Citation discipline

Every load-bearing claim cites:
- A paper-key from `kb/index/papers.json` AND
- An anchor `kb/excerpts/<key>#<heading>` where the excerpt has been deepened to §-anchored.

Where the KB is still abstract-level only, the LaTeX gets a margin marker `\todo{deepen-cite}` and the bibliography entry retains the abstract-level citation. The build review (`theory/plans/2026-05-04-kb-build-review.md`) tracks the deepening status.

## What lands first

1. Paper-1 outline (this turn) — quality template for Papers 2-5 outlines.
2. Papers 2-5 outlines (parallel subagents) — fill out the structure.
3. Per-paper implementation plans (sequential).
4. Shared LaTeX preamble + bibliography skeleton.
5. Section-level parallel writing for Paper 1.
6. Repeat (5) for Papers 2-5.
7. Final cross-ref + bibliography + lint pass; build all five PDFs.
