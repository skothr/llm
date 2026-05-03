# Landscape Sweep — Reasoning and Inference

**Scope:** chain-of-thought, process-supervision, test-time-compute, reasoning-training, inference-time-search, kv-cache-management, quantization, speculative-decoding, serving-systems, structured-output
**Captured:** 2026-05-02
**Researcher:** Phase 1 subagent (Sonnet)
**Time spent:** ~45 minutes
**Sources hit:** WebSearch ~20 queries, WebFetch ~4 URLs, plus arxiv abstracts, HuggingFace papers, vendor blogs, GitHub release notes

---

## 1. Candidate Papers (Phase 2 Priority List)

### chain-of-thought

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | 2022 | Wei et al. (Google Brain) | https://arxiv.org/abs/2201.11903 | Original CoT paper; baseline for all subsequent work | A |
| Chain-of-Thought Reasoning In The Wild Is Not Always Faithful | 2025 | — | https://arxiv.org/abs/2503.08679 | Large-scale real-world CoT faithfulness study; quantifies post-hoc rationalization rates per model | A |
| Does Inference Scaling Improve Reasoning Faithfulness? | 2026 | — | https://arxiv.org/abs/2601.06423 | Whether extended thinking → more faithful CoT; multi-model analysis | A |
| Adaptive Graph of Thoughts: Test-Time Adaptive Reasoning Unifying Chain, Tree, and Graph Structures | 2025 | — | https://arxiv.org/abs/2502.05078 | Unifies CoT/ToT/GoT into single adaptive framework | A |
| How Does Unfaithful Reasoning Emerge from Autoregressive Training? | 2026 | — | https://arxiv.org/abs/2602.01017 | Synthetic study tracing mechanism of post-hoc rationalization | A |
| FaithCoT-Bench: Benchmarking Instance-Level Faithfulness of CoT Reasoning | 2025 | — | https://arxiv.org/abs/2510.04040 | New benchmark specifically for CoT faithfulness; needed for KB entry | A |
| Analysing Chain of Thought Dynamics: Active Guidance or Unfaithful Post-hoc Rationalisation? | 2025 | — | https://arxiv.org/abs/2508.19827 | EMNLP 2025; differentiates reasoning vs distilled models on faithfulness | A |

### process-supervision

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| ProcessBench: Identifying Process Errors in Mathematical Reasoning | 2024 | Qwen team, Alibaba | https://arxiv.org/abs/2412.06559 | Standard benchmark for step-error detection; used to eval all new PRMs | A |
| Process Reward Models That Think (ThinkPRM) | 2025 | Khalifa et al. | https://arxiv.org/abs/2504.16828 | Generative CoT verifier using 1% of PRM800K labels; outperforms discriminative PRMs | A |
| A Survey of Process Reward Models: From Outcome Signals to Process Supervisions | 2025 | — | https://arxiv.org/abs/2510.08049 | Comprehensive survey covering data generation, PRM training, test-time scaling uses | A |
| R-PRM: Reasoning-Driven Process Reward Modeling | 2025 | — | https://arxiv.org/abs/2503.21295 | Adds reasoning step to PRM scoring; improves generalization | A |
| More Bang for the Buck: Process Reward Modeling with Entropy-Driven Uncertainty (EDU-PRM) | 2025 | — | https://arxiv.org/abs/2503.22233 | Entropy-based step-boundary segmentation; 1.5% data, beats Math-Shepherd PRM | A |
| Generalizable Process Reward Models via Formally Verified Training Data | 2025 | — | https://arxiv.org/abs/2505.15960 | Uses formal verification to create provably-correct step labels | A |
| Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning | 2024 | — | https://openreview.net/forum?id=A6Y7AqlzLW | OmegaPRM; large-scale automated PRM data generation | A |

### test-time-compute

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| A Survey of Test-Time Compute: From Intuitive Inference to Deliberate Reasoning | 2025 | — | https://arxiv.org/abs/2501.02497 | Comprehensive survey covering o1-style systems; taxonomy of TTC methods | A |
| s1: Simple test-time scaling | 2025 | Muennighoff et al. (Stanford+) | https://arxiv.org/abs/2501.19393 | 1K examples + budget forcing reproduces o1-preview test-time scaling curves; landmark efficiency result | A |
| Revisiting the Test-Time Scaling of o1-like Models: Do they Truly Possess Test-Time Scaling Capabilities? | 2025 | — | https://arxiv.org/abs/2502.12215 | Critical analysis; more compute does NOT always help; important caveat | A |
| The Art of Scaling Test-Time Compute for Large Language Models | 2025 | — | https://arxiv.org/abs/2512.02008 | Empirical study 30B+ tokens across 8 models on 4 datasets | A |
| Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs | 2025 | — | https://arxiv.org/abs/2507.02076 | L1-controllable vs L2-adaptive TTC taxonomy; practical budget management | A |
| DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | 2025 | DeepSeek-AI | https://arxiv.org/abs/2501.12948 | Canonical reference for o1-class open reasoning models; published in Nature 2025 | A |
| Gemini 2.5 Technical Report | 2025 | Google DeepMind | https://arxiv.org/abs/2507.06261 | Gemini 2.5 Pro reasoning with configurable thinkingBudget; multi-modal | A |

### reasoning-training

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | 2025 | DeepSeek-AI | https://arxiv.org/abs/2501.12948 | GRPO training, cold-start + RL pipeline, 6 distilled open models | A |
| 100 Days After DeepSeek-R1: A Survey on Replication Studies and More Directions for Reasoning Language Models | 2025 | — | https://arxiv.org/abs/2505.00551 | Aggregates replication lessons; covers GRPO variants, RLVR, data curation | A |
| DAPO: An Open Source LLM Reinforcement Learning System at Scale | 2025 | ByteDance | https://self-supervised.cs.jhu.edu/fa2025/files/presentations/Why-RL-Sep16-AdvancesNLP.pdf | GRPO++ with clip-higher, dynamic sampling, token-level loss; AIME 2024 50pts on Qwen2.5-32B | B |
| rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking | 2025 | Guan et al. (Microsoft) | https://arxiv.org/abs/2501.04519 | Four-cycle self-evolution with MCTS+PRM; Qwen2.5-Math-7B → 90% MATH | A |
| Towards Understanding Distilled Reasoning Models | 2025 | — | https://arxiv.org/abs/2503.03730 | Analysis of what transfers during R1 distillation and where it breaks | A |
| Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models | 2026 | — | https://arxiv.org/abs/2601.18734 | On-policy variant of distillation; student samples own trajectories, teacher supervises | A |
| DRA-GRPO: Your GRPO Needs to Know Diverse Reasoning Paths | 2025 | — | https://arxiv.org/abs/2505.09655 | Diversity-regularized GRPO variant; addresses mode collapse in reasoning RL | A |

### inference-time-search

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking | 2025 | Guan et al. | https://arxiv.org/abs/2501.04519 | MCTS-guided search with process reward model; 53.3% AIME; rivals o1-mini | A |
| ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search | 2024 | — | https://arxiv.org/abs/2406.03816 | NeurIPS 2024 poster; PRM-guided tree search for data collection + self-training | A |
| Unifying Tree Search Algorithm and Reward Design for LLM Reasoning: A Survey | 2025 | — | https://arxiv.org/abs/2510.09988 | Survey of MCTS/beam/BFS variants for reasoning; taxonomy | A |
| Review of Inference-Time Scaling Strategies: Reasoning, Search and RAG | 2025 | — | https://arxiv.org/abs/2510.10787 | Covers search methods + RAG interaction; useful for inter-topic KB links | A |
| MCTS-RAG: Enhance Retrieval-Augmented Generation with Monte Carlo Tree Search | 2025 | — | https://arxiv.org/abs/2503.20757 | MCTS applied to retrieval decisions; bridges search and RAG | A |
| rStar (Mutual Reasoning Makes Smaller LLMs Stronger) | 2024 | — | https://datasciocean.com/en/paper-intro/rstar/ | ICLR 2025 poster; mutual verification between generator and discriminator SLMs | A |

### kv-cache-management

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization | 2024 | Hooper et al. (UC Berkeley) | https://arxiv.org/abs/2401.18079 | NeurIPS 2024; per-channel quantization + nuq2 → 8x compression, 1M tokens on A100 | A |
| KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache | 2024 | — | https://arxiv.org/abs/2402.02750 | 2-bit KV quant with residual full-precision; 2.6x peak memory reduction | A |
| DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model | 2024 | DeepSeek-AI | https://arxiv.org/abs/2405.04434 | Multi-Head Latent Attention (MLA): 93.3% KV cache reduction; architectural shift | A |
| Towards Economical Inference: Enabling MLA in Any Transformer-based LLMs | 2025 | — | https://arxiv.org/abs/2502.14837 | MHA→MLA fine-tuning with 0.3-0.6% data; ACL 2025; retrofitting established models | A |
| ChunkKV: Semantic-Preserving KV Cache Compression for Long-Context LLM Inference | 2025 | — | https://arxiv.org/abs/2502.00299 | Chunk-level rather than token-level eviction; preserves semantic coherence | A |
| KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization | 2025 | — | https://arxiv.org/abs/2502.04420 | Layer-level KV quant sensitivity; mixed-precision assignment | A |
| KV Cache Optimization Strategies for Scalable and Efficient LLM Inference (survey) | 2026 | — | https://arxiv.org/abs/2603.20397 | Recent survey; good entry point for KB | A |

### quantization

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58) | 2024 | Ma et al. (Microsoft) | https://arxiv.org/abs/2402.17764 | Ternary {-1,0,+1} weights match FP16 at scale; foundational claim | A |
| BitNet b1.58 2B4T Technical Report | 2025 | Ma et al. (Microsoft) | https://arxiv.org/abs/2504.12285 | First open-source native 1-bit LLM at 2B scale; 2-6x CPU speedup | A |
| 1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs | 2024 | — | https://arxiv.org/abs/2410.16144 | 2.37x–6.17x on x86, 1.37x–5.07x on ARM; inference kernel details | A |
| Post Training Quantization of Large Language Models with Microscaling Formats (MX) | 2024 | — | https://arxiv.org/abs/2405.07135 | MXINT/MXFP evaluation; OCP Microscaling standard used in H100/Blackwell | A |
| DeepSeek-V3 Technical Report (FP8 training) | 2024 | DeepSeek-AI | https://arxiv.org/abs/2412.19437 | First validation of FP8 training at extreme scale (671B params, MoE); tile-based quant | A |
| Which Quantization Should I Use? A Unified Evaluation | 2026 | — | https://arxiv.org/abs/2601.14277 | Systematic comparison GGUF/GPTQ/AWQ/FP8; practical selection guide | A |
| Mixed-Precision Quantization for Language Models: Techniques and Prospects | 2025 | — | https://arxiv.org/abs/2510.16805 | Survey; covers W4A16, W4A8, W8A8, FP8; situates each method | A |

### speculative-decoding

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | 2024 | Li et al. | https://arxiv.org/abs/2401.15077 | ICML 2024; autoregressive draft on target's hidden states; 3x+ speedup | A |
| EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees | 2024 | Li et al. | https://arxiv.org/abs/2406.16858 | EMNLP 2024; dynamic tree construction based on confidence; ~2x vs Medusa | A |
| EAGLE-3: Scaling up Inference Acceleration via Training-Time Test | 2025 | Li et al. | https://arxiv.org/abs/2503.01840 | NeurIPS 2025; multi-layer feature fusion, abandons feature prediction; up to 6.5x, 1.4x over EAGLE-2 | A |
| Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads | 2024 | Cai et al. (UIUC) | https://arxiv.org/abs/2401.10774 | Parallel heads on frozen model; 2-3.6x speedup; widely integrated | A |
| Break the Sequential Dependency of LLM Inference Using Lookahead Decoding | 2024 | — | https://arxiv.org/abs/2402.02057 | ICML 2024; Jacobi-iteration n-gram speculation; no draft model needed | A |
| Fast and Accurate Causal Parallel Decoding using Jacobi Forcing | 2025 | — | https://arxiv.org/abs/2512.14681 | Improved Jacobi decoding with forcing; cleaner parallel decode | A |
| XGrammar-2: Efficient Dynamic Structured Generation Engine for Agentic LLMs | 2026 | — | https://arxiv.org/abs/2601.04426 | Combines speculative-style caching with grammar constraints | A |

### serving-systems

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| SGLang: Efficient Execution of Structured Language Model Programs | 2024 | Zheng et al. (LMSYS/Berkeley) | https://arxiv.org/abs/2312.07104 | NeurIPS 2024 poster; RadixAttention, prefix caching, structured output co-design | A |
| Efficient LLM Inference with SGLang (lecture slides, 2025) | 2025 | Ying Sheng (xAI/LMSYS/UCLA) | https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-25-sglang-72edc5043338f59db34d47e5b96ac870.pdf | Zero-overhead scheduler, DeepSeek-V3/R1 day-one support details | B |
| vLLM Router: A High-Performance Prefill/Decode Aware Load Balancer | 2025 | vLLM team | https://blog.vllm.ai/2025/12/13/vllm-router-release.html | PD-disaggregation orchestration layer; production-scale routing | B |
| Disaggregated Prefilling (vLLM experimental feature) | 2025 | vLLM team | https://docs.vllm.ai/en/latest/features/disagg_prefill/ | Separates prefill/decode workers for tail latency control; important architecture shift | B |
| Comparing SGLang, vLLM, and TensorRT-LLM (Clarifai blog) | 2026 | Clarifai | https://www.clarifai.com/blog/comparing-sglang-vllm-and-tensorrt-llm-with-gpt-oss-120b | Empirical throughput/latency numbers on GPT-OSS-120B; SGLang 29% higher throughput on <70B | C |
| LLM Inference Engines Comparison 2026 | 2026 | — | https://leetllm.com/blog/llm-inference-engine-comparison-2026 | Up-to-date decision guide: vLLM vs SGLang vs TRT-LLM vs Ollama | C |

### structured-output

| Paper | Year | Authors | URL | Why it matters | Tier |
|-------|------|---------|-----|----------------|------|
| XGrammar: Flexible and Efficient Structured Generation Engine for LLMs | 2024 | Dong et al. (MLC) | https://arxiv.org/abs/2411.15100 | MLSys 2025; up to 100x speedup; now default backend in vLLM, SGLang, TRT-LLM | A |
| XGrammar-2: Efficient Dynamic Structured Generation Engine for Agentic LLMs | 2026 | MLC team | https://arxiv.org/abs/2601.04426 | Agentic workloads; ~10ms compile time vs XGrammar-1's 1000ms | A |
| Generating Structured Outputs from Language Models: Benchmark and Studies | 2025 | — | https://arxiv.org/abs/2501.10868 | JSONSchemaBench; 10K real-world JSON schemas; evaluates 6 frameworks | A |
| Draft-Conditioned Constrained Decoding for Structured Generation | 2026 | — | https://arxiv.org/abs/2603.03305 | Combines speculative decoding with grammar constraints | A |
| Flexible and Efficient Grammar-Constrained Decoding | 2025 | — | https://arxiv.org/abs/2502.05111 | General grammar-constrained decode improvements | A |

---

## 2. Hot Topics / Active Research Threads (Last 6 Months)

### chain-of-thought

**Faithfulness has emerged as the dominant 2025 concern for CoT.** The question shifted from "does CoT help?" to "is CoT causally driving the answer or is it post-hoc rationalization?" Large-scale analyses ([2503.08679](https://arxiv.org/abs/2503.08679), [2601.06423](https://arxiv.org/abs/2601.06423)) find that production non-reasoning models (GPT-4o-mini 13%, Haiku 3.5 7%) show much higher rationalization rates than thinking models (Sonnet 3.7 with thinking 0.04%, Gemini 2.5 Pro 0.14%). The Decoupling Hypothesis ([2505.17406](https://arxiv.org/abs/2505.17406)) proposes that answer selection and reasoning path are loosely correlated rather than causally bound. FaithCoT-Bench ([2510.04040](https://arxiv.org/abs/2510.04040)) provides a new instance-level benchmark. Structurally, Adaptive Graph of Thoughts (AGoT, [2502.05078](https://arxiv.org/abs/2502.05078)) unifies CoT/ToT/GoT adaptively, though the original ToT/GoT excitement has cooled as thinking-model RL largely superseded manual search structures.

### process-supervision

**PRMs are undergoing a generative-vs-discriminative split.** The traditional discriminative PRM (trained on PRM800K step labels) is being challenged by generative verifiers that "think before scoring." ThinkPRM ([2504.16828](https://arxiv.org/abs/2504.16828)) showed that a CoT-based verifier trained on only 1% of PRM800K labels outperforms full discriminative baselines on ProcessBench and AIME'24. EDU-PRM ([2503.22233](https://arxiv.org/abs/2503.22233)) and R-PRM ([2503.21295](https://arxiv.org/abs/2503.21295)) represent two new discriminative-side improvements using entropy-anchored boundaries and reasoning-augmented scoring respectively. A formal verification approach ([2505.15960](https://arxiv.org/abs/2505.15960)) uses symbolic checkers to generate provably-correct step labels. The 100-days-after-R1 survey ([2505.00551](https://arxiv.org/abs/2505.00551)) documents how PRMs are increasingly used as reward signals inside RLVR training loops, not just for test-time best-of-N selection.

### test-time-compute

**Test-time compute is now a first-class axis of model capability.** The release cadence of OpenAI o3/o4-mini (April 2025), DeepSeek-R1 (January 2025, published in Nature), QwQ-32B (March 2025), and Gemini 2.5 Pro (March 2025 with configurable thinkingBudget) established "thinking models" as a product category distinct from base chat models. s1 ([2501.19393](https://arxiv.org/abs/2501.19393)) showed that with just 1K curated examples and budget forcing (appending "Wait" tokens), test-time scaling curves can be reproduced at low cost. However, critical papers ([2502.12215](https://arxiv.org/abs/2502.12215)) note that extending solution length does not monotonically improve accuracy for R1/QwQ due to limited self-revision capability—the scaling has a ceiling for current architectures. Analysts project inference will exceed training compute demand by ~118x by 2026.

### reasoning-training

**GRPO and its variants have become the dominant RL training paradigm.** DeepSeek-R1's GRPO foregoes the critic model, estimating baseline from group scores with outcome-only rewards. DAPO (ByteDance) improved on this with clip-higher, dynamic sampling, and token-level loss—achieving 50 pts on AIME2024 with Qwen2.5-32B in 50% fewer steps than R1-Zero. The open-source replication ecosystem (HuggingFace open-r1, Mixture-of-Thoughts dataset) has made R1-style training reproducible. A key finding: **distillation is more efficient than RL for small models** (≤14B)—the R1-distilled Qwen-32B outperforms o1-mini while 7B+ distilled models retain substantial fraction of teacher performance. rStar-Math ([2501.04519](https://arxiv.org/abs/2501.04519)) demonstrates an alternative: four-cycle self-evolution with MCTS+PRM achieves 90% on MATH at 7B scale.

### inference-time-search

**MCTS over reasoning steps is maturing beyond pure math.** rStar-Math uses policy-SLM + reward-SLM in a generator-discriminator mutual verification loop with MCTS, achieving top-20% high school math Olympiad performance. ReST-MCTS* (NeurIPS 2024) showed that MCTS-collected data can bootstrap self-training loops. The unifying survey ([2510.09988](https://arxiv.org/abs/2510.09988)) taxonomizes MCTS, beam search, and BFS variants for reasoning. An emerging thread is MCTS-RAG ([2503.20757](https://arxiv.org/abs/2503.20757)), applying tree search to retrieval decisions. The main open challenge is scaling MCTS to non-math domains where step-level reward signals are harder to obtain.

### kv-cache-management

**MLA (Multi-Head Latent Attention) from DeepSeek-V2 is the architectural-level breakthrough.** By storing a compressed low-rank projection instead of full K/V pairs, MLA achieves 93.3% KV cache reduction vs standard MHA, enabling 5.76x higher generation throughput. DeepSeek-V3 and V3.1/V3.2 all use MLA. A new paper ([2502.14837](https://arxiv.org/abs/2502.14837), ACL 2025) shows MHA→MLA retrofitting with only 0.3-0.6% fine-tuning data, potentially bringing this to all existing architectures. On the quantization side, KVQuant (NeurIPS 2024) enables 10M context on a single A100 at 8x compression. KIVI provides practical 2-bit KV quant. The 2026 frontier is moving toward reasoning-aware compression (TriAttention, 10.7x memory reduction on AIME25) and latent-space compaction (50x). vLLM's automatic prefix caching provides 85-95% cache hit rates on agent workloads, dramatically reducing per-call cost.

### quantization

**Multiple simultaneous transitions are happening at once.** FP8 has become the production standard for large-scale serving (DeepSeek-V3 validated it at 671B scale; H100/H200 natively accelerate FP8). BitNet b1.58 2B4T (April 2025) delivered the first open-source native 1-bit 2B model trained from scratch on 4T tokens, matching FP16 accuracy with 2-6x CPU speedup, changing the conversation from "can ternary work?" to "deploy it." On the GGUF/llama.cpp side, IQ (importance-matrix) quants (IQ4_XS, etc.) achieve better quality-per-bit than K-quants at the same bit width, particularly below Q5_K_M. MX (Microscaling) formats are being standardized via OCP and supported in Blackwell/H100. AWQ consistently outperforms GPTQ for weight-only quantization; FP8 outperforms SmoothQuant for large models. A comprehensive 2026 evaluation ([2601.14277](https://arxiv.org/abs/2601.14277)) provides a practical selection guide.

### speculative-decoding

**EAGLE-3 (NeurIPS 2025) is the current state of the art for tree-based speculative decoding.** It abandons feature-prediction entirely in favor of direct token prediction with multi-layer feature fusion ("training-time test"), achieving up to 6.5x speedup and 1.4x improvement over EAGLE-2. EAGLE-3 is integrated into vLLM and SGLang. The broader landscape: Medusa (2-3.6x, no draft model, multiple heads), Lookahead/Jacobi decoding (1.5-2.3x, no draft model or data store, model-agnostic), and EAGLE-family (3-6.5x, requires draft model training). As of late 2025, speculative decoding has moved from research to production default in major serving frameworks. A new direction is combining grammar-constrained generation with speculative drafts ([2603.03305](https://arxiv.org/abs/2603.03305), [2601.04426](https://arxiv.org/abs/2601.04426)).

### serving-systems

**SGLang has overtaken vLLM on throughput for smaller models; the gap narrows at 70B+.** SGLang v0.4's zero-overhead batch scheduler achieves 95-98% GPU utilization vs 78-85% for v0.3. RadixAttention (KV cache radix tree) gives 85-95% cache hit rates for shared-prefix workloads vs vLLM's 15-25%. TensorRT-LLM has the best absolute latency (p50, p95 TTFT). vLLM remains the most broadly compatible default and highest-throughput at very high concurrency (100+ requests). The architectural trend is **prefill/decode disaggregation**: vLLM now has experimental PD disaggregation running two separate instances, not for throughput but for tail latency control. SGLang added day-one support for DeepSeek-V3/R1 in January 2025, including DeepSeek-specific MLA and MoE optimizations. llama.cpp continues rapid development with CUDA graph support, profile-guided speculative decoding, and piece-wise CUDA graphs for multi-GPU.

### structured-output

**XGrammar has become the default backend for all major serving frameworks** (vLLM, SGLang, TensorRT-LLM) as of early 2026, achieving under 40 microseconds per token with near-zero overhead. Its key innovation—splitting vocabulary into context-independent (99%) precomputable tokens vs context-dependent runtime tokens—solves the grammar-as-bottleneck problem. XGrammar-2 ([2601.04426](https://arxiv.org/abs/2601.04426)) extends this to agentic workloads with ~10ms compile time (vs XGrammar-1's 1000ms). JSONSchemaBench ([2501.10868](https://arxiv.org/abs/2501.10868)) is the new benchmark for evaluating 10K real-world JSON schemas across Guidance, Outlines, llama.cpp, XGrammar, OpenAI, and Gemini. The field is standardizing on JSON Schema as the constraint specification language, with GBNF remaining the llama.cpp-native format. llguidance (Microsoft) is XGrammar's main competitor, showing comparable performance in repeated-schema scenarios.

---

## 3. Training-vs-Current Discrepancies

> **Trained-in view:** "OpenAI o1 is the leading reasoning model and the main o-series release."
> **Current view (per OpenAI system card, April 2025):** o3 and o4-mini were released April 16, 2025, with full tool use (web browsing, Python, image analysis) **integrated into the chain-of-thought**, not just the final answer. o4-mini is a more capable/efficient variant than o1-mini. GPT-5 (system card Aug 2025) and GPT-5.5 (April 2026) further extend the lineup.

> **Trained-in view:** "DeepSeek R1 is a recent and notable open reasoning model."
> **Current view (per DeepSeek API changelog, late 2025–2026):** DeepSeek released V3.2-Exp (September 2025), V3.2 (December 2025), V3.1 (August 2025), and V4 (April 2026). V3.1 and V3.2 are **hybrid think/non-think** single models with tool use. The model family has substantially evolved beyond R1.

> **Trained-in view:** "BitNet b1.58 is a research model showing ternary quantization is possible at scale."
> **Current view (per arxiv 2504.12285, April 2025):** Microsoft released BitNet b1.58 2B4T as the first **open-source native 1-bit model** trained from scratch at 2B scale on 4T tokens, matching FP16 accuracy with 2-6x CPU inference speedup. This is now a deployable model, not just a proof of concept.

> **Trained-in view:** "vLLM is the dominant serving framework, with SGLang as a newer alternative."
> **Current view (per benchmarks, 2026):** SGLang now leads vLLM on throughput for models ≤70B by ~29% on H100. Both are production-grade. TensorRT-LLM leads on latency. The choice is workload-dependent, not a clear hierarchy.

> **Trained-in view:** "EAGLE-2 is the latest version of the EAGLE speculative decoding method."
> **Current view (per arxiv 2503.01840, NeurIPS 2025):** EAGLE-3 was accepted at NeurIPS 2025. It represents a significant architectural departure from EAGLE-2 (multi-layer feature fusion, no feature prediction), achieving 6.5x speedup and 1.4x over EAGLE-2.

> **Trained-in view:** "Process reward models are trained on PRM800K-style labeled step data."
> **Current view (per arxiv 2504.16828, April 2025):** ThinkPRM demonstrates that a generative CoT verifier using only 1% of PRM800K labels outperforms discriminative PRMs. The shift is from labeled-step discriminative PRMs to **thinking-model-based generative verifiers**. EDU-PRM achieves similar with entropy-driven step boundaries.

> **Trained-in view:** "XGrammar is a promising new structured generation engine."
> **Current view (per vLLM/SGLang docs, March 2026):** XGrammar is now the **default** structured generation backend in vLLM, SGLang, and TensorRT-LLM. XGrammar-2 (January 2026) extends it for agentic workflows with 100x faster compilation.

> **Trained-in view:** "CoT faithfulness is an open research question but thinking models are likely more faithful."
> **Current view (per arxiv 2601.06423, January 2026):** Large-scale multi-model empirical study confirms thinking models ARE significantly more faithful (Sonnet 3.7 thinking: 0.04% post-hoc rationalization rate) vs non-thinking models (GPT-4o-mini: 13%). Inference scaling does improve faithfulness, but the mechanism is not fully understood.

---

## 4. Taxonomy Refinements

Suggestions for `theory/kb/index/topics.md`:

**Add (new topic leaves needed):**
- `reasoning/test-time-compute/budget-forcing` — budget forcing / wait-token insertion as a lightweight TTC technique (s1 paper)
- `reasoning/test-time-compute/thinking-budget` — API-level thinking budget control (Gemini thinkingBudget, Claude token budget, o-series effort levels)
- `reasoning/process-supervision/generative-prm` — generative/CoT-based verifiers distinct from discriminative PRMs (ThinkPRM, EDU-PRM)
- `reasoning/reasoning-training/rlvr` — RL with Verifiable Rewards as a named paradigm (GRPO, DAPO, REINFORCE++, RLOO)
- `inference/kv-cache-management/mla` — Multi-Head Latent Attention as an architectural KV compression strategy separate from quantization
- `inference/kv-cache-management/pd-disaggregation` — Prefill/Decode disaggregation as a serving topology (distinct from model-side compression)
- `inference/quantization/mx-formats` — OCP Microscaling formats (MXFP8, MXINT4, etc.) distinct from standard FP8/INT8
- `inference/structured-output/jsonschema-bench` — JSONSchemaBench as the new evaluation standard

**Split:**
- `inference/speculative-decoding` → split into `draft-model-based` (EAGLE family, Medusa) vs `model-free` (Lookahead/Jacobi, n-gram); they have very different deployment requirements
- `reasoning/test-time-compute` → split into `sampling-based` (best-of-N, self-consistency) vs `search-based` (MCTS, beam) vs `sequential-compute` (extended thinking, budget forcing); these are mechanistically distinct

**Merge:**
- `inference-time-search` and parts of `test-time-compute` overlap heavily; consider a shared parent topic `reasoning/test-time-methods` with leaves for sampling, search, and RL-trained thinking
- `reasoning-training/distillation` and `reasoning-training/rl` should be sibling leaves under a unified `reasoning/training-methods` node (they are alternative routes to the same capability)

**Rename:**
- `reasoning-training` → `reasoning/training` (align with `inference/` prefix convention)
- `inference-time-search` → `reasoning/inference-time-search` (it lives in the reasoning domain, not inference/efficiency domain)
- `kv-cache-management` → `inference/kv-cache` (shorter, consistent with sibling topics)

---

## 5. Forum / Blog Signals Worth Following Up

- https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo — Nathan Lambert's ongoing GRPO/RLVR reading notes; catches emerging variants fast
- https://github.com/opendilab/awesome-RLVR — Curated list of RL with verifiable rewards papers; updated frequently
- https://github.com/RyanLiu112/Awesome-Process-Reward-Models — Comprehensive PRM paper collection; good Phase 2 discovery source
- https://github.com/ThreeSR/Awesome-Inference-Time-Scaling — Inference/test-time scaling paper list
- https://www.marktechpost.com/2026/04/29/top-10-kv-cache-compression-techniques/ — April 2026 KV cache survey; good for "what's been tried" coverage
- https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training — Sebastian Raschka's reasoning training survey post; accessible and well-sourced
- https://kaitchup.substack.com/p/choosing-a-gguf-model-k-quants-i — Practical GGUF format selection guide with updated llama.cpp numbers
- https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-llamacpp-january-25-2026-7750/ — Weekly llama.cpp GitHub digest; tracks rapid development
- https://blog.vllm.ai — vLLM official blog; release announcements for PD disaggregation, router, v1 architecture
- https://sgl-project.github.io — SGLang official docs/changelog; zero-overhead scheduler, DeepSeek MLA support

---

## 6. Open Questions / Unresolved

1. **Is CoT faithfulness a training property or a scale property?** The 2026 multi-model analysis shows thinking models are more faithful, but it's unclear whether this is caused by the thinking training process, model scale, or RLVR-style training separately. Needs controlled ablation.

2. **Does MCTS over reasoning steps generalize beyond math?** All published rStar/rStar-Math/MCTS-reasoning work is on math (and some code). The step-level reward signal is easy to define there. Whether verifier-guided MCTS helps on open-ended reasoning, factual tasks, or agent planning remains largely open.

3. **What is the actual mechanism of GRPO's reasoning improvement?** The 100-days survey and RLVR analysis papers debate whether GRPO teaches new reasoning skills or surfaces latent capabilities from pretraining. The RLVR-on-base-models paper ([2506.14245](https://arxiv.org/abs/2506.14245)) suggests the latter but the evidence is contested.

4. **How does MLA interact with KV quantization?** MLA stores a latent projection (not K/V) in cache; applying KIVI-style quantization to this latent is not straightforward. Papers treat MLA and KV quant as independent techniques; a principled integration is missing from the literature.

5. **Does BitNet b1.58 scale beyond 2B?** The 2B4T result is strong, but the only published open-source native-1bit model is 2B. Whether ternary QAT maintains full-precision parity at 7B/13B/70B scale during from-scratch training (not post-training quantization) remains an open research question.

6. **PD disaggregation tradeoffs at scale.** vLLM's disaggregated prefill documentation explicitly states it does NOT improve throughput; it controls tail latency. But multiple papers/blogs claim otherwise in specific configurations. The actual throughput vs latency tradeoff surface for PD disaggregation is not well-characterized empirically.

7. **XGrammar-2 for complex agentic grammars.** XGrammar-2 reduces compile time from 1000ms to 10ms but it is unclear how it handles deeply recursive or highly ambiguous grammars at runtime. The 100x speedup is on JSON Schema; performance on arbitrary CFGs in agent tool-call scenarios is not benchmarked publicly.

8. **EAGLE-3 for long reasoning traces.** EAGLE-3 speedup numbers are measured on standard benchmarks (MT-bench, etc.). How multi-layer feature fusion performs when the target model is generating 32K-token reasoning chains (o1/R1-style) is not explicitly tested in the paper.
