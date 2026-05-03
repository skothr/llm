# Landscape Sweep — Training and Post-training

**Scope:** pre-training-data, synthetic-data-and-distillation, optimization, distributed-training, mixed-precision-and-stability, continued-pretraining, sft, rlhf, dpo-and-offline, rlaif-and-constitutional, rlvr-and-grpo
**Captured:** 2026-05-02
**Researcher:** Phase 1 subagent (Sonnet 4.6)
**Time spent:** ~35 minutes
**Sources hit:** WebSearch ~22 queries, WebFetch ~1 URL, plus vendor blogs, model cards, GitHub repos

---

## 1. Candidate Papers (Phase 2 Priority List)

### pre-training-data

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| FineWeb: Decanting the Web for the Finest Text Data at Scale | 2024 | Penedo et al. (HuggingFace) | https://arxiv.org/abs/2406.17557 | 15T token CC-derived dataset; proxy-model validation approach for filter selection; spawned FineWeb-Edu educational classifier | A |
| FineWeb-2: Multilingual extension | 2024–25 | HuggingFace FW Team | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | Extends FineWeb to 1000+ languages, language-specific filter tuning | B |
| RedPajama: an Open Dataset for Training Large Language Models | 2024 | Together AI | https://arxiv.org/abs/2411.12372 | NeurIPS 2024; RedPajama-v2 (30T tokens, 84 CC snapshots, 40+ quality signals, precomputed MinHash for fuzzy dedup) | A |
| Data Mixing Laws: Optimizing Data Mixtures by Predicting LM Performance | 2024/2025 | Liu et al. | https://arxiv.org/abs/2403.16952 | ICLR 2025; predictable functional relationship between domain proportions and loss; 48% step savings from optimized mix | A |
| Optimizing Pretraining Data Mixtures with LLM-Estimated Utility (UtiliMax/MEDU) | 2025 | Multiple | https://arxiv.org/abs/2501.11747 | LLM-estimated data utility; ~200x cheaper than ablation-based mixing | A |
| OLMo 2 Furious | 2025 | AI2 OLMo Team | https://arxiv.org/abs/2501.00656 | Fully transparent training (weights+data+code+logs); Dolmino Mix 1124 late-stage curriculum; COLM 2025 paper | A |
| Dolma 3 / OLMo 3 Corpus | 2025 | AI2 | https://arxiv.org/abs/2512.13961 | 9.3T token corpus (web via olmOCR, code, math, science); strongest fully-open reasoning model pipeline | A |

### synthetic-data-and-distillation

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| Phi-4 Technical Report | 2024 | Abdin et al. (Microsoft) | https://arxiv.org/abs/2412.08905 | Archetypal synthetic-data-first recipe: 400B tokens from 50+ synthetic pipelines (multi-agent prompting, self-revision, instruction reversal); surpasses teacher (GPT-4) on STEM QA | A |
| Phi-4-reasoning Technical Report | 2025 | Abdin et al. (Microsoft) | https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/phi_4_reasoning.pdf | Reasoning-specialized Phi-4 fine-tuned on o3-mini demonstrations; Phi-4-mini finetuned on DeepSeek-R1 synthetic data | A |
| DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL | 2025 | DeepSeek AI | https://arxiv.org/abs/2501.12948 | R1's 800K distillation set (600K reasoning, 200K non-reasoning) enables small models (1.5B–32B) to achieve near-o1 performance via pure SFT distillation | A |
| MAGPIE: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing | 2025 | Xu et al. | https://proceedings.iclr.cc/paper_files/paper/2025/file/be06e3802e9411381feece79b4d960c1-Paper-Conference.pdf | ICLR 2025; prompts aligned LLMs' pre-query templates to auto-generate instruction pairs; 4M instruction set; beats UltraFeedback SFT+DPO on AlpacaEval/ArenaHard | A |
| Weak-to-Strong Generalization: Eliciting Strong Capabilities with Weak Supervision | 2023/2024 | Burns, Izmailov, Kirchner et al. (OpenAI) | https://arxiv.org/abs/2312.09390 | Foundational framing for scalable alignment; weak supervisor extracts strong capabilities | A |
| Beyond Distillation: Synthetic Data and Curriculum Learning in 14B Models | 2026 | Pathak | https://medium.com/@harshnpathak/beyond-distillation-use-of-synthetic-data-and-curriculum-learning-by-14b-models-to-beat-72b-ce90e9997021 | 14B model beats 72B+ with curriculum; post-cutoff community report | C |

### optimization

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| Muon is Scalable for LLM Training | 2025 | Liu, Su et al. (Moonlight team) | https://arxiv.org/abs/2502.16982 | Scales Muon (Newton-step-like momentum with gradient orthogonalization) to LLMs via weight decay + per-param scale tuning; ~2× efficiency over AdamW compute-optimally; trains Moonlight 3B/16B-MoE on 5.7T tokens | A |
| MuonBP: Faster Muon via Block-Periodic Orthogonalization | 2025 | — | https://arxiv.org/abs/2510.16981 | Reduces Muon's orthogonalization overhead via block-periodic scheduling | A |
| Sophia: A Scalable Stochastic Second-Order Optimizer for LM Pre-training | 2023 | Liu, Li et al. (Stanford) | https://arxiv.org/abs/2305.14342 | ICLR 2024; diagonal Hessian second-order method; claimed 2× speedup over AdamW; held up empirically but downstream task generalization still favors AdamW | A |
| Understanding Warmup-Stable-Decay LR: A River Valley Loss Landscape Perspective | 2024 | — | https://arxiv.org/abs/2410.05192 | Theoretical basis for WSD (popularized by MiniCPM); removes fixed compute budget requirement | A |
| Optimal LR Schedules under Functional Scaling Laws | 2026 | — | https://arxiv.org/abs/2602.06797 | Feb 2026; derives optimal schedule including WSD from functional scaling law; bridges theory–practice gap | A |
| Pre-Training LLMs on a Budget: A Comparison of Three Optimizers | 2025 | — | https://arxiv.org/abs/2507.08472 | Empirical AdamW vs Lion vs Sophia; Sophia lowest train loss; AdamW best downstream; Lion fastest wall-clock | A |
| MiniCPM: Unveiling the Potential of SLMs with Scalable Training Strategies | 2024 | Hu et al. (Tsinghua) | https://arxiv.org/abs/2404.06395 | Originated WSD schedule for continuous training; widely cited and adopted | A |

### distributed-training

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| TorchTitan: One-stop PyTorch Native Solution for Production-Ready LLM Pre-training | 2024 | Meta/PyTorch | https://arxiv.org/abs/2410.06511 | ICLR 2025; reference implementation for 4D parallelism (FSDP2 + TP + PP + Context Parallel); Ring Attention for 1M sequence length; 65% speedup on Llama 3.1 8B | A |
| MegaBlocks: Efficient Sparse Training with Mixture-of-Experts | 2022/2023 | Gale, Narayanan et al. (Stanford/Databricks) | https://arxiv.org/abs/2211.15841 | MLSys 2023; dropless MoE via block-sparse ops; 1.8–4.35× speedup; now in Nanotron, GPT-NeoX, Mistral Mixtral reference stack | A |
| DeepSeek-V3 Technical Report — DualPipe algorithm | 2024 | DeepSeek AI | https://arxiv.org/abs/2412.19437 | Cross-node MoE with full compute-communication overlap; 2.788M H800 GPU hours for 671B MoE on 14.8T tokens; practical large-scale reference | A |
| FSDP2 / PyTorch fully_shard | 2024–25 | Meta PyTorch | https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html | FSDP2 rewrite on DTensor abstraction; 7% lower GPU memory vs FSDP1; composable with TP/PP/CP; de facto replacement for FSDP1 | B |
| Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware | 2025 | — | https://arxiv.org/abs/2505.09343 | Hardware co-design lessons from V3 training; H800 cross-node bandwidth bottlenecks | A |

### mixed-precision-and-stability

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| DeepSeek-V3: FP8 Mixed-Precision Training Framework | 2024 | DeepSeek AI | https://arxiv.org/abs/2412.19437 | First validated FP8 training at extreme scale (671B MoE); fine-grained blockwise/tilewise scaling (128×128 weight blocks, 1×128 activation vectors); FP32 accumulation | A |
| An Investigation of FP8 Across Accelerators for LLM Inference | 2025 | — | https://arxiv.org/abs/2502.01070 | Cross-accelerator FP8 inference comparison; useful for training-inference precision parity | A |
| NVFP4 Trains with Precision of 16-Bit and Speed/Efficiency of 4-Bit | 2025 | NVIDIA | https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/ | NVIDIA Blackwell (GB200/GB300); E2M1 FP4 format; 2× FP8 throughput; 1% accuracy degradation on DeepSeek-R1; 3.5× memory reduction vs FP16 | B |
| Llama 4 FP8 Training: MetaP Hyperparameter Transfer | 2025 | Meta | https://ai.meta.com/blog/llama-4-multimodal-intelligence/ | FP8 pre-training of 2T+ token multimodal model on 32K GPUs at 390 TFLOPs/GPU; MetaP for cross-scale HP transfer | B |
| Training Dynamics of the Cooldown Stage in WSD | 2025 | — | https://arxiv.org/abs/2508.01483 | Aug 2025; analyzes stability at the decay phase; practical guidance for training stability | A |

### continued-pretraining

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| Evolutionary Optimization of Model Merging Recipes | 2024 | Akiba et al. (Sakana AI) | https://arxiv.org/abs/2403.13187 | Nature Machine Intelligence 2025; automated evolutionary search over weight-space and layer-stack merge spaces; creates cross-architecture models with no training | A |
| Arcee MergeKit: A Toolkit for Merging Large Language Models | 2024 | Goddard et al. | https://arxiv.org/abs/2403.13257 | Open-source toolkit implementing TIES, DARE, Task Arithmetic, SLERP, Frankenmerge; backbone of community merging | A |
| Merging Continual Pretraining Models for Domain-Specialized LLMs: Finance | 2025 | — | https://arxiv.org/abs/2511.02451 | Nov 2025; first systematic evaluation of CPT (not SFT) model merging; TIES/DARE-TIES on financial benchmarks | A |
| Model Soups: Averaging Weights of Multiple Fine-tuned Models | 2022 | Wortsman et al. | https://arxiv.org/abs/2203.05482 | ICML 2022 foundational work for weight averaging; basis for WARM (used in Gemma post-training) | A |
| Rethinking Weight-Averaged Model Merging | 2024 | — | https://arxiv.org/abs/2411.09263 | Critical analysis of when weight averaging helps vs hurts; important counterpoint | A |
| Dolmino Mix 1124 / OLMo 2 annealing curriculum | 2025 | AI2 | https://arxiv.org/abs/2501.00656 | Late-stage curriculum injection of high-quality domain data during annealing; substantially improves benchmarks without full retrain | A |

---

### sft

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| Tülu 3: Pushing Frontiers in Open Language Model Post-Training | 2024 | Wang et al. (AI2) | https://arxiv.org/abs/2411.15124 | Fully open 4-stage recipe: data curation → SFT → DPO → RLVR; beats Llama 3.1-Instruct, Qwen 2.5, GPT-4o-mini; Tülu 3 405B (Jan 2025) scales RLVR to 405B | A |
| MAGPIE: Alignment Data Synthesis from Scratch | 2025 | Xu et al. | https://proceedings.iclr.cc/paper_files/paper/2025/file/be06e3802e9411381feece79b4d960c1-Paper-Conference.pdf | (See synthetic-data; ICLR 2025) beats prior SFT+DPO combos using SFT alone | A |
| Qwen 2.5 Technical Report | 2024 | Qwen Team | https://arxiv.org/abs/2412.15115 | 1M+ SFT examples; multistage SFT+DPO+GRPO post-training; 18T token pretraining; detail-rich reference for current SOTA open post-training | A |
| Qwen 3 Technical Report | 2025 | Qwen Team | https://arxiv.org/abs/2505.09388 | May 2025; 4-stage post-training including "thinking"/"non-thinking" mode toggle; GRPO for reasoning; additional RL round for instruction following + agent tasks | A |
| Llama 3 Herd of Models | 2024 | Meta | https://arxiv.org/abs/2407.21783 | Post-training: SFT → rejection sampling → DPO (no PPO at scale); β=0.1; on-policy preference data; rejection sampling as core quality gate | A |
| OLMo 3 Technical Report | 2025 | AI2 | https://arxiv.org/abs/2512.13961 | Fully open Base → Instruct → Think pipeline; Dolma 3 corpus; OlmoTrace for training data attribution | A |

### rlhf

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| Reward Hacking in the Era of Large Models: Mechanisms, Emergent Misalignment, Challenges | 2026 | — | https://arxiv.org/abs/2604.13602 | April 2026; overoptimization persists across RLHF/DPO/inference-time selection; verbosity, sycophancy, math hallucination as localized shortcuts | A |
| Reward Shaping to Mitigate Reward Hacking in RLHF (PAR) | 2025 | — | https://arxiv.org/abs/2502.18770 | Feb 2025; "Preference As Reward" shaping + design principles; mitigation approach | A |
| Calibration Collapse Under Sycophancy Fine-Tuning | 2026 | — | https://arxiv.org/abs/2604.10585 | April 2026; sycophancy fine-tuning breaks calibration/uncertainty quantification | A |
| RewardBench 2: Advancing Reward Model Evaluation | 2025 | — | https://arxiv.org/abs/2506.01937 | 2025 benchmark suite for reward model evaluation; important for understanding RM quality | A |
| Scaling Laws for Reward Model Overoptimization in Direct Alignment | 2024 | — | https://arxiv.org/abs/2406.02900 | Overoptimization scaling curves for DPO-class methods | A |
| Gemma 3 Technical Report (BOND/WARM/WARP post-training) | 2025 | Google DeepMind | https://arxiv.org/abs/2503.19786 | March 2025; distillation from large IT teacher + BOND/WARM/WARP RL; multi-reward (math, code, safety, multilingual); 4B Gemma 3 competitive with 27B Gemma 2 | A |

### dpo-and-offline

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| SimPO: Simple Preference Optimization with a Reference-Free Reward | 2024 | Meng et al. (Princeton) | https://arxiv.org/abs/2405.14734 | NeurIPS 2024; average log-prob as implicit reward; length normalization; no reference model; +6.4 AlpacaEval 2 over DPO | A |
| A Survey of Direct Preference Optimization | 2025 | — | https://arxiv.org/abs/2503.11701 | March 2025 survey; covers KTO, IPO, CPO, ORPO, SimPO, DAPO in unified framework | A |
| A Comprehensive Survey of DPO: Datasets, Theories, Variants, Applications | 2024/2025 | — | https://arxiv.org/abs/2410.15595 | Broad DPO variant taxonomy; v3 March 2025 | A |
| GAPO: Learning Where It Matters: Geometric Anchoring for Robust Preference Alignment | 2026 | — | https://arxiv.org/abs/2602.04909 | Feb 2026; pessimistic local surrogate (anchor) around current policy to stabilize optimization | A |
| Explicit Preference Optimization | 2025 | — | https://arxiv.org/abs/2506.07492 | June 2025; new variant addressing DPO implicit reward limitations | A |
| BPO: Revisiting Preference Modeling in DPO | 2024/2025 | — | https://openreview.net/forum?id=b97EwMUWu7 | Bidirectional preference modeling; ICLR submission | A |

### rlaif-and-constitutional

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| Constitutional AI: Harmlessness from AI Feedback | 2022 | Bai et al. (Anthropic) | https://arxiv.org/abs/2212.08073 | Foundational CAI; SL-CAI + RL-CAI pipeline; RLAIF replacing human labels | A |
| C3AI: Crafting and Evaluating Constitutions for Constitutional AI | 2025 | — | https://arxiv.org/abs/2502.15861 | Feb 2025; systematic study of constitution design choices; which principles matter | A |
| Constitution or Collapse? Exploring Constitutional AI with Llama 3-8B | 2025 | — | https://arxiv.org/abs/2504.04918 | April 2025; practical failure modes of CAI at smaller model scales | A |
| How Effective Is Constitutional AI in Small LLMs? | 2025 | — | https://arxiv.org/abs/2503.17365 | March 2025; DeepSeek-R1 and peers; CAI effectiveness vs model scale | A |
| Reverse Constitutional AI (R-CAI): Controllable Toxic Data Generation | 2026 | — | https://arxiv.org/abs/2604.17769 | April 2026; inverts constitution for adversarial data synthesis; probability-clamped RLAIF | A |
| Reinforcement Learning Meets LLMs: Survey of Advancements Across LLM Lifecycle | 2025 | — | https://arxiv.org/abs/2509.16679 | Broad survey covering RLAIF placement in post-training landscape | A |

### rlvr-and-grpo

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| DeepSeek-R1: Incentivizing Reasoning via RL | 2025 | DeepSeek AI | https://arxiv.org/abs/2501.12948 | Jan 2025; GRPO with verifiable binary rewards; AIME 2024 15.6%→71%; also in Nature 2025; the paper that triggered the RLVR surge | A |
| DAPO: An Open-Source LLM RL System at Scale | 2025 | ByteDance Seed | https://arxiv.org/abs/2503.14476 | March 2025; Decoupled Clip and Dynamic Sampling PO; 50 pts AIME 2024 with Qwen-32B; 50% fewer steps than GRPO baseline; fully open (verl framework) | A |
| VAPO: Value Augmented Proximal Policy Optimization | 2025 | — | https://arxiv.org/abs/2504.05118 | April 2025; first value-model-based RL to outperform value-free on long-CoT; superior to DAPO in fewer steps | A |
| VinePPO: Refining Credit Assignment in RL Training of LLMs | 2024/2025 | Kazemnejad et al. (McGill) | https://arxiv.org/abs/2410.01679 | ICML 2025; Monte Carlo credit assignment via intermediate-state re-simulation; 3× wall-clock speedup over PPO on MATH/GSM8K | A |
| RL with Verifiable Rewards: GRPO's Effective Loss, Dynamics, Success Amplification | 2025 | — | https://arxiv.org/abs/2503.06639 | March 2025; theoretical analysis of why GRPO works and when it fails | A |
| Does RL Really Incentivize Reasoning Beyond the Base Model? | 2025 | — | https://arxiv.org/abs/2504.13837 | NeurIPS 2025 oral; key critical finding: RLVR improves sampling efficiency but does NOT unlock new reasoning capabilities absent in the base model | A |
| The Lessons of Developing Process Reward Models in Math Reasoning | 2025 | — | https://arxiv.org/abs/2501.07301 | Jan 2025; MC estimation for PRM data synthesis is inferior to LLM-as-judge; practical PRM building guide | A |
| Process Reward Models That Think (ThinkPRM) | 2025 | — | https://arxiv.org/abs/2504.16828 | April 2025; long-CoT verifier; orders of magnitude fewer process labels than discriminative PRMs | A |

---

## 2. Hot Topics / Active Research Threads (Last 6 Months)

### pre-training-data
The dominant 2025 trend is quality over quantity and validated data selection. FineWeb (2024, 15T tokens) demonstrated that proxy-LM validation of filters — training small models on candidate slices — outperforms heuristic-only approaches, and FineWeb-2 extended this to 1000+ languages. Alongside this, Data Mixing Laws (ICLR 2025) established that mixture proportions follow predictable functional relationships, enabling principled optimization rather than ad hoc recipes. The Dolma lineage (AI2) has become the transparency gold standard: Dolma 3 (9.3T tokens for OLMo 3) incorporates olmOCR-processed science PDFs and careful decontamination. Key open question: at what scale does domain mixing stop following smooth scaling laws?

### synthetic-data-and-distillation
Phi-4 (Dec 2024) proved that 14B models trained primarily on LLM-generated synthetic data can surpass their teacher (GPT-4) on STEM. The DeepSeek-R1 distillation result (Jan 2025) then showed that pure SFT on 800K reasoning samples from a strong teacher gives small models (1.5B–32B) near-o1 reasoning. MAGPIE (ICLR 2025) demonstrated a template-prompting trick that generates diverse instruction data without seed prompts. The emerging concern is model collapse at high synthetic fraction (safe up to ~1/3 synthetic mix; pure synthetic risks collapse per Data Mixing work). Phi-4-reasoning (May 2025) extends distillation into the reasoning/long-CoT regime using o3-mini demonstrations.

### optimization
**Muon** is the biggest new story. Originally a nanoGPT speedrun trick by Keller Jordan (2024), it was shown scalable to large MoEs in February 2025 (arXiv:2502.16982), achieving ~2× compute efficiency over AdamW. The Moonlight 3B/16B model trained on 5.7T tokens is the proof point. MuonBP (Oct 2025) reduces orthogonalization overhead. WSD schedules (Warmup-Stable-Decay, from MiniCPM 2024) have become standard in frontier training runs including Qwen and OLMo; theoretical grounding arrived in Oct 2024 and Feb 2026. Sophia remains a strong second-order option but AdamW still wins on downstream generalization in controlled comparisons. Lion delivers wall-clock savings for vision transformers but limited gains on LLMs.

### distributed-training
TorchTitan (ICLR 2025) is the PyTorch-native reference for 4D parallelism: FSDP2 + TP + PP + Context Parallelism, with Ring Attention enabling 1M-token sequence training. DeepSeek-V3's DualPipe algorithm solved the cross-node MoE communication bottleneck (full compute-communication overlap) and became the model for efficient large-scale MoE training at 2.788M H800 GPU-hours for 671B parameters. FSDP2 (DTensor-based rewrite) is the stable API replacing FSDP1 in PyTorch 2.x with 7% lower peak memory. Sequence/context parallelism is now a first-class training dimension for long-context models.

### mixed-precision-and-stability
DeepSeek-V3 (Dec 2024) validated FP8 training at scale for the first time — using fine-grained blockwise (128×128) and tilewise (1×128) scaling with FP32 accumulation. Meta used FP8 for Llama 4 pre-training (40T tokens, 32K GPUs, 390 TFLOPs/GPU). NVIDIA's NVFP4 format (Blackwell-native) adds a new frontier: 2× FP8 throughput, 3.5× memory savings vs FP16, only ~1% accuracy degradation on quantized models. FP4 training (as distinct from inference quantization) is the next frontier. WSD cooldown-phase instability is an active topic (arXiv:2508.01483, Aug 2025).

### continued-pretraining
Model merging has matured from a research curiosity to a practical tool. Sakana AI's evolutionary merging (Nature MI 2025) automates merging recipe search. MergeKit (Arcee, 2024) provides TIES/DARE/Task Arithmetic in a single toolkit. A key 2025 finding is that merging CPT (continued pre-training) experts is distinctly harder than merging SFT models — DARE-TIES outperforms Task Arithmetic on financial CPT merging but all methods lag gold-standard full training. The "Dolmino Mix 1124" approach (OLMo 2) — injecting curated domain data during the LR annealing phase — is a simpler alternative to full CPT or merging.

### sft
The state-of-the-art open SFT reference is Tülu 3 (AI2, Nov 2024 → 405B in Jan 2025). The complete pipeline — data curation → SFT → DPO → RLVR — is fully open and reproducible. Key insight from Tülu 3 and Llama 3: rejection sampling during SFT is critical (used as core quality gate by Meta). Qwen 3 (May 2025) adds a 4-stage post-training with separate "thinking" and "non-thinking" instruction modes toggled by system prompt, which is an important UX pattern. MAGPIE's near-zero-cost instruction synthesis is the most practical new SFT data technique.

### rlhf
Reward hacking and overoptimization research is experiencing a wave of papers (April 2026: arXiv:2604.13602, 2604.10585) showing the problem persists across RLHF/DPO/inference-time scaling. Sycophancy as a reward-hacking manifestation is now well-characterized: reward models inherit annotator confirmation bias and policies amplify it. The Gemma 3 post-training used WARM (Weight Averaged Reward Models) and WARP to improve reward model robustness, a technique now being adopted more widely. RewardBench 2 (2025) is the current evaluation standard for reward models. The field is shifting from single reward model to ensemble/mixture approaches.

### dpo-and-offline
SimPO (NeurIPS 2024) removed the reference model entirely and now rivals or beats most DPO variants. ORPO (reference-free, combined SFT + preference loss) is widely adopted in compute-constrained settings. The 2025 DPO survey literature (two major surveys in March 2025) consolidates KTO, IPO, CPO, ORPO, SimPO into a unified framework. New variants are still arriving: GAPO (geometric anchoring, Feb 2026) and Explicit PO (June 2025). The key empirical finding from Llama 3 post-training: DPO required less compute than PPO and performed better at large scale — this has shifted practitioner defaults away from PPO for alignment.

### rlaif-and-constitutional
Constitutional AI is being stress-tested at smaller model scales (multiple April 2025 papers showing degradation below ~7B parameters). The model collapse concern from recursive RLAIF (fine-tuning on self-generated critiques) is now empirically documented. Reverse-CAI (April 2026) shows the technique can be inverted for adversarial data generation. Active research questions: can constitutions be automatically derived? (C3AI, Feb 2025 addresses this); how to prevent the self-rewarding loop from degenerating?

### rlvr-and-grpo
This is the hottest post-training topic of 2025. DeepSeek-R1's GRPO (Jan 2025) triggered a wave of variants: DAPO (March 2025, ByteDance) improved GRPO with decoupled clipping and dynamic sampling; VAPO (April 2025) introduced value models back to outperform DAPO; VinePPO (ICML 2025) refined credit assignment with MC re-simulation. Process Reward Models had a surge: ThinkPRM uses long-CoT verifier fine-tuning rather than discriminative PRM labels. A critical NeurIPS 2025 oral (arXiv:2504.13837) found that RLVR does NOT create fundamentally new reasoning capacity — it improves sampling efficiency within the base model's existing capability envelope. This is a significant finding that tempers enthusiasm for RLVR as a capability-expansion technique.

---

## 3. Training-vs-Current Discrepancies

> **Trained-in view:** "AdamW is the dominant optimizer; second-order methods like Sophia are promising but unproven at scale."
> **Current view (per arXiv:2502.16982, Feb 2025; arXiv:2507.08472, 2025):** Muon has been demonstrated at 5.7T token / large-MoE scale with ~2× AdamW compute efficiency, and has been adopted in production training runs. Sophia wins on train loss but AdamW still leads on downstream task generalization in direct comparisons. Muon is now a serious third option, not a toy.

> **Trained-in view:** "FP16/BF16 are the dominant training precisions; FP8 is for inference."
> **Current view (per DeepSeek-V3 Dec 2024; Llama 4 April 2025; NVIDIA NVFP4 2025):** FP8 training at extreme scale (671B MoE, 40T token multimodal) is now validated and production-deployed. FP4 (NVFP4) training is the next active frontier on Blackwell hardware. BF16 is no longer the ceiling.

> **Trained-in view:** "RLVR / GRPO is a new and powerful technique for eliciting reasoning capabilities."
> **Current view (per arXiv:2504.13837, NeurIPS 2025 oral):** RLVR improves sampling efficiency (pass@1) on problems the base model can already solve, but does NOT expand the model's underlying reasoning capacity to solve genuinely new problem types. The capability ceiling is set by pre-training. This is a substantial revision of early enthusiasm.

> **Trained-in view:** "PPO is the standard RL algorithm for post-training; DPO is an offline alternative."
> **Current view (per Llama 3 herd 2024; Tülu 3 2024; widespread adoption):** At large scale, DPO outperforms PPO on instruction following and requires less compute. PPO is largely replaced by GRPO/DAPO/VAPO for reasoning-specific tasks, and by DPO for general alignment. PPO is now a baseline, not the default.

> **Trained-in view:** "Model merging (TIES, DARE) is experimental / research-only."
> **Current view (per MergeKit 2024; Evolutionary Merging Nature MI 2025; community practice):** Model merging is a production technique. MergeKit is used routinely for domain adaptation without retraining. Evolutionary merging (Sakana) creates cross-architecture hybrids. CPT model merging is the new frontier (arXiv:2511.02451).

> **Trained-in view:** "FineWeb is the leading open web dataset."
> **Current view (as of 2025):** FineWeb-2 is now available with multilingual coverage (1000+ languages). Dolma 3 (OLMo 3, Dec 2025) at 9.3T tokens with olmOCR-processed science content is the most transparent large corpus. Data mixing optimization (Data Mixing Laws, ICLR 2025) is now systematic rather than ad hoc.

> **Trained-in view:** "Constitutional AI / RLAIF is Anthropic's technique; adoption is limited."
> **Current view (per 2025 literature):** CAI has been widely studied and replicated, but its effectiveness degrades significantly at <7B scale. New variants, inversions (R-CAI), and automated constitution design are active research areas.

---

## 4. Taxonomy Refinements

Suggestions for `theory/kb/index/topics.md`:

**Add:**
- `optimization/muon` — Newton-step-like momentum with gradient orthogonalization; scalable alternative to AdamW for hidden layers
- `optimization/wsd-schedule` — Warmup-Stable-Decay LR scheduling; distinct enough from cosine/linear decay to warrant own leaf
- `pre-training-data/data-mixing-laws` — Systematic optimization of domain proportions; now has theoretical treatment
- `pre-training-data/proxy-model-validation` — Using small proxy LM training to validate filter choices (FineWeb methodology)
- `mixed-precision-and-stability/fp4-training` — NVFP4/MXFP4 training on Blackwell; distinct from FP8
- `rlvr-and-grpo/process-reward-models` — PRMs as a subfield; now has multiple papers on training data synthesis, architectures, and evaluation (ThinkPRM, R-PRM, FOVER)
- `rlvr-and-grpo/rlvr-limits` — The NeurIPS 2025 finding that RLVR doesn't expand base capability; important counterweight
- `synthetic-data-and-distillation/reasoning-distillation` — DeepSeek-R1 → small model SFT; a specific technique distinct from general distillation
- `distributed-training/context-parallelism` — Ring attention / sequence parallelism; now a 4th dimension alongside DP/TP/PP
- `continued-pretraining/cpt-merging` — Merging domain-CPT experts; distinct from merging SFT fine-tunes
- `post-training/thinking-nonthinking-modes` — Qwen 3 / o-series pattern of toggling extended reasoning; architectural + training concern

**Split:**
- `rlvr-and-grpo` → split into `rlvr-foundations` (verifiable rewards, reward shaping) and `grpo-variants` (GRPO, DAPO, VAPO, REINFORCE++) — the variant space is now large enough to warrant separation
- `dpo-and-offline` → consider splitting into `reference-free-methods` (SimPO, ORPO, CPO) and `reference-based-methods` (DPO, IPO, KTO, anchored variants)
- `mixed-precision-and-stability` → consider splitting `precision-formats` (FP8/FP4/BF16 as a training choice topic) from `training-stability` (loss spikes, gradient clipping, instability diagnosis)

**Merge:**
- `rlaif-and-constitutional` could absorb `self-rewarding-lms` as a subsection — both are variants of AI-feedback alignment without human labels

**Rename:**
- `rlhf` → consider `reward-modeling-and-rlhf` to distinguish from the post-training pipeline as a whole; "RLHF" is increasingly used loosely
- `continued-pretraining` → `adaptation-and-merging` better captures the current scope (CPT + domain adaptation + model merging all live here)

---

## 5. Forum / Blog Signals Worth Following Up

- https://www.interconnects.ai/p/tulu-3 — Nathan Lambert's analysis of Tülu 3; good practitioner framing of open post-training stack
- https://www.interconnects.ai/p/olmo-3-americas-truly-open-reasoning — OLMo 3 framing; "model flow" concept; fully open reasoning pipeline
- https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training — Sebastian Raschka's survey of RLVR for reasoning (mid-2025); accessible synthesis
- https://kellerjordan.github.io/posts/muon/ — Keller Jordan's original Muon post; nanoGPT speedrun context and motivation
- https://limit-of-rlvr.github.io/ — Dedicated site for the "limits of RLVR" finding; bibliography and ongoing analysis
- https://cameronrwolfe.substack.com/p/model-merging — Cameron Wolfe's model merging survey; accessible taxonomy
- https://cameronrwolfe.substack.com/p/llama-4 — Llama 4 architecture + training analysis; good secondary source
- https://lilianweng.github.io/posts/2024-11-28-reward-hacking/ — Lilian Weng's reward hacking post (Nov 2024); covers mechanisms across paradigms
- https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs — Living repo tracking RL-for-reasoning papers; useful for Phase 2 completeness check
- https://github.com/pengr/LLM-Synthetic-Data — Living reading list for LLM synthetic data (updated to July 2025); covers distillation, MAGPIE, self-play

---

## 6. Open Questions / Unresolved

1. **Muon adoption at frontier scale**: Moonlight (3B/16B MoE on 5.7T tokens) is the largest published Muon training run. Open question: does the ~2× AdamW efficiency hold at 70B+ dense or 600B+ MoE scale, or does the orthogonalization assumption break down?

2. **RLVR capability ceiling**: The NeurIPS 2025 oral finding (RLVR only improves sampling efficiency, not capability) is based on current math/code benchmarks. Does this hold for more open-ended tasks? Does it generalize to multimodal reasoning?

3. **FP4 training stability**: NVFP4 for inference is well-characterized. FP4 for training (forward + backward pass) is nascent. What are the loss scaling and gradient accumulation requirements at FP4?

4. **Optimal synthetic fraction**: The "1/3 synthetic, 2/3 natural" heuristic for safe mixing has limited empirical basis. No large-scale systematic study of the collapse-vs-benefit tradeoff across domain and model scale.

5. **CPT model merging vs full retrain**: The 2025 finance CPT merging paper found merging lags full multi-skill training. Is this fundamental to CPT (vs SFT) merge difficulty, or an artifact of the methods/data tested?

6. **Constitution design automation**: C3AI (Feb 2025) begins systematic study of which constitutional principles matter. The inverse R-CAI paper shows constitutions can be inverted for adversarial generation. Open: can constitutions be reliably derived from capability/safety goals automatically?

7. **DPO sycophancy under scale**: Sycophancy collapse is well-documented (April 2026 calibration paper). No clear mitigation has emerged beyond reward shaping heuristics (PAR). Does this improve with stronger base models, or worsen with scale?

8. **Data mixing at post-training stage**: Data Mixing Laws cover pre-training. Tülu 3 makes domain-mixing choices for SFT and RLVR empirically. No analogous theoretical treatment for post-training mixture optimization exists yet.

9. **OLMo 3 "model flow" reproducibility**: OLMo 3 releases the entire pipeline (data, checkpoints, training logs for every stage). This is an unprecedented level of transparency. Phase 2 should assess whether this enables meaningful scientific replication studies.

10. **Reward model ensembling (WARM/WARP) vs single RM**: Gemma 3 uses weight-averaged reward models; this appears to substantially reduce overoptimization. Is this now the standard, or is it Gemma-specific? No controlled comparison study exists yet.
