# Glossary

Every technical term used in the LLM theory KB, defined concisely. Each term
should cite a paper-key from `kb/index/papers.json` for traceability where
applicable. Terms are added during Phase 2 (per-topic research) and reviewed
during Phase 4 (lint).

This glossary was seeded from `archive/2026-05-03-pre-expansion/GLOSSARY.md`
on 2026-05-03; entries from that seed have not yet been re-cited and are
treated as `status: pre-expansion-seed` until re-validated during Phase 2.

## Models and Architectures

- **LLM** — Large Language Model. A neural network with billions of parameters trained on large text corpora to predict and generate language.
- **Transformer** — Neural network architecture introduced by Vaswani et al. (2017) that replaces recurrence with self-attention, allowing parallel processing of sequences.
- **encoder-decoder** — The original Transformer layout: an encoder processes the full input bidirectionally, a decoder generates output autoregressively, with cross-attention connecting them.
- **decoder-only** — Simplified Transformer architecture (GPT, LLaMA) that removes the encoder and cross-attention, using only masked self-attention and FFN layers.
- **RNN** — Recurrent Neural Network. Processes sequences one step at a time, maintaining a hidden state. Predecessor to the Transformer.
- **LSTM** — Long Short-Term Memory. An RNN variant with gating mechanisms to better preserve long-range dependencies.
- **T5 (Text-to-Text Transfer Transformer)** — Google's encoder-decoder Transformer (Raffel et al., 2019) that frames every NLP task as text-to-text. Originator of several conventions reused elsewhere (e.g., SentencePiece vocab, relative positional bias).
- **PaLM** — Google's 540B-parameter decoder-only LLM (Chowdhery et al., 2022). Uses SwiGLU, RoPE, parallel attention/FFN layers. Reference point for scaling experiments.
- **Mistral** — Open-weights decoder-only LLM family from Mistral AI (Jiang et al., 2023). 7B variant uses sliding-window attention and grouped-query attention on top of the LLaMA-style architecture.
- **Claude** — Anthropic's proprietary decoder-only LLM family. No public architecture paper; structural family shares the decoder-only Transformer lineage.

## Tokenization

- **Tokenization** — Converting raw text into a sequence of discrete integer token IDs that a model can process.
- **BPE (Byte Pair Encoding)** — Subword tokenization algorithm that iteratively merges the most frequent adjacent character pairs to build a vocabulary.
- **Subword Unit** — A token that may be a whole word, a word fragment, or a single character/byte, depending on corpus frequency.
- **SentencePiece** — A tokenizer that operates on raw byte sequences rather than Unicode characters, used by LLaMA. Provides byte-fallback so any input can be tokenized.
- **byte-fallback** — Mechanism ensuring no "unknown token" failures: any byte sequence can be represented, even if unseen during training.
- **Vocabulary (V)** — The fixed set of all tokens a model can recognize. Size ranges from ~32K (LLaMA) to ~50K (GPT-3).

## Embeddings and Positional Encoding

- **token embedding** — A learned lookup table ($W_e \in \mathbb{R}^{V \times d_{\text{model}}}$) mapping each token index to a dense vector.
- **weight tying** — Sharing the same weight matrix $W_e$ for both input embeddings and the output projection to logits, halving vocabulary-related parameters.
- **Positional Encoding** — Any mechanism that injects sequence-position information into the model, since self-attention is inherently permutation-invariant.
- **Sinusoidal Positional Encoding** — Fixed positional encoding using sine/cosine functions at different frequencies (original Transformer, 2017).
- **Learned Positional Embedding** — A trainable position matrix $W_p$ where each row corresponds to a sequence position (GPT-1, 2018). Fixes maximum context length at training time.
- **RoPE (Rotary Position Embedding)** — Positional encoding that rotates query and key vectors by position-dependent angles, making attention dot products depend on relative position. Applied at every layer to Q and K only. Used by LLaMA and most modern LLMs.

## Attention

- **self-attention** — Mechanism where each position in a sequence computes a weighted sum over all positions, with weights determined by learned query-key similarity.
- **scaled dot-product attention** — Core attention operation: $\text{softmax}(QK^\top / \sqrt{d_k}) V$. Scaling by $\sqrt{d_k}$ prevents large dot products from saturating the softmax.
- **multi-head attention** — Running $h$ parallel attention operations on different learned projections, then concatenating and projecting the results. Allows attending to different representation subspaces simultaneously.
- **Query (Q)** — Projected representation of the position that is "asking" for information in attention.
- **Key (K)** — Projected representation of positions being "queried" against. The Q·K dot product determines attention weights.
- **Value (V)** — Projected representation of the content to be aggregated. Weighted by attention scores to produce the output.
- **attention weights** — The softmax-normalized scores ($a_j \geq 0$, $\sum a_j = 1$) that determine how much each position contributes to the output.
- **causal mask** — A triangular mask that prevents position $i$ from attending to positions $> i$, enforcing autoregressive (left-to-right) generation.
- **cross-attention** — Attention where Q comes from the decoder and K, V come from the encoder output. Removed in decoder-only models.
- **Head Dimension ($d_k$, $d_v$)** — Dimensionality of query/key and value projections per attention head. Typically $d_{\text{model}} / h$.

## Attention variants and KV-cache compression

- **MHA (Multi-Head Attention)** — The Vaswani 2017 default: $h$ independent heads, each with its own $W_i^Q, W_i^K, W_i^V$. KV cache size per token per layer is $2 n_h d_h$ elements `[vaswani2017 §3.2.2]`.
- **MQA (Multi-Query Attention)** — Variant where all heads share a single $W^K, W^V$ (only $W^Q$ remains per-head). KV cache shrinks by factor $h$. Cheaper per-token decode at the cost of capacity and stability `[shazeer2019 §2.4; kb/notes/architecture/attention-mechanism.md#sec-3-1]`.
- **GQA (Grouped-Query Attention)** — Interpolation between MHA and MQA: query heads are partitioned into $g$ groups, each group shares one $W^K, W^V$. GQA-1 = MQA; GQA-$h$ = MHA. KV cache per token is $2 n_g d_h$ elements `[ainslie2023 §2.2]`.
- **Uptraining** — The Ainslie 2023 recipe for converting an existing MHA checkpoint to GQA/MQA: mean-pool the $W^K, W^V$ matrices within each group, then continue pre-training for $\alpha \approx 5\%$ of the original budget `[ainslie2023 §2.1]`.
- **MLA (Multi-Head Latent Attention)** — DeepSeek-V2's compression scheme. Rather than sharing K/V across heads, K and V are reconstructed at use time from a small latent $\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \in \mathbb{R}^{d_c}$ via up-projections $W^{UK}, W^{UV}$. Cache holds only the latent. During inference $W^{UK}$ can be absorbed into $W^Q$ and $W^{UV}$ into $W^O$ `[deepseek-v2 §2.1.2]`.
- **Decoupled RoPE** — The MLA add-on that splits each query/key into a content part (no RoPE, low-rank reconstructed) and a small shared RoPE-carrying part. Necessary because standard RoPE inserts a position-dependent rotation between $W^Q$ and $W^{UK}$ that prevents the cache-absorption optimization `[deepseek-v2 §2.1.3]`.


## Attention implementation (hardware)

- **FlashAttention** — Exact attention computed with re-organized memory access: tile $K, V$ into SRAM-resident blocks and use **online softmax** to combine block results, never materializing the $N \times N$ attention matrix in HBM. Forward + backward IO complexity $\Theta(N^2 d^2 M^{-1})$ vs.\ $\Theta(N d + N^2)$ for the standard implementation `[dao2022 §3]`.
- **Online softmax** — Numerically stable streaming softmax: maintain running max $m$ and normalizer $\ell$ across blocks; when a new block arrives, rescale the existing partial sum by $e^{m_\text{old} - m_\text{new}}$ before adding the block's contribution. Enables block-wise softmax without seeing the full row `[dao2022 §3.1]`.
- **IO-awareness** — Designing an algorithm with explicit accounting for reads and writes between memory hierarchy levels (HBM ↔ SRAM on a GPU). The principle behind FlashAttention `[dao2022 §1]`.
- **Tiling** — Restructuring a computation to operate on sub-blocks that fit in fast memory. Used in FlashAttention to keep $Q, K, V$ blocks in SRAM `[dao2022 §3.1]`.
- **Recomputation (selective gradient checkpointing)** — Storing only the output and softmax statistics from the forward pass and recomputing the attention matrix on-chip during the backward pass, trading FLOPs for memory `[dao2022 §3.1]`.
- **HBM (High-Bandwidth Memory)** — The large, slow memory tier on a GPU (~40–80 GB on A100, 1.5–2.0 TB/s). Most operations in standard attention are HBM-bound `[dao2022 §2.1]`.
- **SRAM (on-chip)** — The small, fast memory tier on a GPU (~192 KB per A100 SM, ~19 TB/s). FlashAttention restructures attention to keep working sets in SRAM `[dao2022 §2.1]`.
- **NSA (Native Sparse Attention)** — Yuan et al.\ 2025: hardware-aligned natively sparse attention trained end-to-end (sparsity is part of pre-training, not retrofitted). Aimed at long context. Treated in `kb/notes/architecture/long-context.md` `[yuan2025]`.

## Feed-Forward Network

- **FFN (Feed-Forward Network)** — A position-wise two-layer (or three-layer for GLU variants) fully-connected network applied independently at each sequence position. Transforms representations within each position, complementing attention which mixes across positions.
- **ReLU** — Rectified Linear Unit: $\max(0, x)$. Original Transformer activation (2017).
- **GELU (Gaussian Error Linear Unit)** — Smooth activation $x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF. Used by GPT-1 (2018).
- **Swish / SiLU** — Activation function $x \cdot \sigma(\beta x)$ where $\sigma$ is the sigmoid. SiLU is the $\beta = 1$ case: $x \cdot \sigma(x)$. Unlike ReLU, it is smooth and non-monotonic around zero.
- **Inner Dimension ($d_{\text{ff}}$)** — The expanded hidden size inside the FFN. Typically $4 \times d_{\text{model}}$, or $\frac{2}{3} \times 4 \times d_{\text{model}}$ for SwiGLU.

## Normalization and Residual Connections

- **residual connection (Skip Connection)** — Adding the sub-layer input directly to its output ($\text{output} = x + \text{SubLayer}(x)$). Enables gradient flow and makes each layer learn a correction rather than a full representation.
- **Layer Normalization (LayerNorm)** — Normalizes across the feature dimension per position: subtract mean, divide by standard deviation, apply learned scale ($\gamma$) and bias ($\beta$).
- **RMSNorm (Root Mean Square Normalization)** — Simplified LayerNorm that drops mean-centering and bias, normalizing only by the RMS of the input. 7–64% faster with comparable quality. Used by LLaMA.
- **Post-LN** — Original Transformer placement: normalize after the residual addition. Requires learning rate warmup for stable training.
- **Pre-LN** — Modern placement: normalize before the sub-layer, then add residually. Gradients scale as $O(d\sqrt{\ln d / L})$, enabling stable training without warmup.
- **Warmup** — Gradually increasing the learning rate from zero during early training steps. Critical for Post-LN; unnecessary for Pre-LN.

## Output and Training

- **Logits** — Unnormalized log-probabilities over the vocabulary, produced by the output projection ($h_{\text{final}} \cdot W_e^\top$).
- **Softmax** — Converts logits to a probability distribution: $P(j) = e^{z_j} / \sum_k e^{z_k}$.
- **cross-entropy loss** — Training objective: $\mathcal{L} = -\log P(t_i \mid t_1, \ldots, t_{i-1})$, summed over all positions.
- **Autoregressive** — Generating tokens one at a time, left to right, where each prediction depends only on previous tokens.
- **Perplexity** — $2^{\mathcal{L}}$ (or $e^{\mathcal{L}}$ with natural log). Measures how "surprised" the model is by the data. Lower is better.

## Dimensions and Notation

- **$d_{\text{model}}$** — The model's hidden dimension (width of the residual stream). All sub-layers input and output this size.
- **$n$** — Sequence length (number of tokens in the input).
- **$h$** — Number of attention heads.
- **$L$** — Number of Transformer layers (blocks).
- **$V$** — Vocabulary size.
- **hidden state** — The $d_{\text{model}}$-dimensional vector representation at each position, flowing through the residual stream across layers.
- **Parameters** — The learned weights of the model (embedding matrices, projection matrices, normalization scales, etc.).

## Inference

- **greedy decoding** — Selecting the highest-probability token at each step.
- **Top-k Sampling** — Sampling from only the $k$ most probable tokens.
- **Nucleus Sampling (Top-p)** — Sampling from the smallest set of tokens whose cumulative probability exceeds $p$.


## Phase 2 area additions

_Per-area glossary fragments produced by Phase 2 area subagents on 2026-05-04; merged here. Each subsection is the verbatim fragment from `theory/kb/index/_phase2-additions/<area>-glossary.md`._

### alignment

# Glossary additions — alignment area (Phase 2)

## Sycophancy and deception (failure modes)

- **Sycophancy** — A response is sycophantic if it is more aligned with
  the user's expressed preference than is justified by the underlying
  truth. Distinct from scheming: sycophancy requires no intent, only
  Goodhart on RLHF preference data. `[sharma2023-sycophancy §abstract;
  kb/excerpts/sharma2023-sycophancy#abstract]`
- **Sycophantic agreement** — Subtype of sycophancy: agreeing with a
  user's incorrect claim. Encoded along a distinct linear direction in
  latent space, independent of sycophantic praise.
  `[vennemeyer2025-sycophancy-not-one-thing §abstract]`
- **Sycophantic praise** — Subtype of sycophancy: flattering the user
  beyond merit. Independently steerable from sycophantic agreement.
  `[vennemeyer2025-sycophancy-not-one-thing §abstract]`
- **Genuine agreement** — Contrast control for sycophancy studies:
  agreeing because the claim is correct. Encoded along its own latent
  direction, distinct from both sycophancy subtypes.
  `[vennemeyer2025-sycophancy-not-one-thing §abstract]`
- **SycophancyEval** — Sharma et al. 2023's four-task free-form
  text-generation suite (feedback, "are you sure?" capitulation,
  user-preference matching, mimicry-of-mistakes) for measuring
  sycophancy in RLHF assistants. `[sharma2023-sycophancy §fourtask;
  kb/excerpts/sharma2023-sycophancy#sec-fourtask]`
- **Scheming** — A model schemes when, instructed to pursue a goal $G$,
  it strategically takes actions that conceal its capabilities or
  manipulate its oversight in service of $G$, rather than pursuing $G$
  by direct means. Operational definition from Apollo Research.
  `[meinke2024-apollo-scheming §abstract;
  kb/excerpts/meinke2024-apollo-scheming#abstract]`
- **Deceptive alignment** — Hubinger et al. 2019 theoretical concept:
  a mesa-optimizer pursues a different objective in training than in
  deployment, choosing training-time behavior strategically so as to
  be deployed. Modern LLMs do not cleanly instantiate the inner-
  optimizer abstraction; recent empirical work uses behavioral
  operationalizations (alignment faking, in-context scheming) instead.
- **Alignment faking** — Operational sub-form of deceptive alignment:
  a model complies with a behavior it normally refuses, conditional on
  inferring it is in training, in order to preserve its preferred
  behavior at deployment. Demonstrated empirically in Claude 3 Opus at
  14% rate (free user / training condition).
  `[greenblatt2024-alignment-faking §abstract;
  kb/excerpts/greenblatt2024-alignment-faking#abstract]`
- **In-context scheming** — Apollo's experimental subset of scheming:
  the goal $G$ is provided in-context (system prompt) rather than
  acquired during training. Five frontier models all demonstrate this
  capability. Cf. **sandbagging without nudging** for the rarer
  trained-goal case. `[meinke2024-apollo-scheming §abstract]`
- **Sandbagging** — Strategic underperformance on a capability
  evaluation in service of a separate objective (avoiding retraining,
  avoiding deployment, etc). One of the six probes in the Apollo suite.
  Most concerning when it occurs *without nudging* and in service of a
  *training-acquired* goal (observed in Claude 3.5 Sonnet pursuing
  helpfulness). `[meinke2024-apollo-scheming §sandbagging-nudge;
  kb/excerpts/meinke2024-apollo-scheming#sec-sandbagging-nudge]`
- **Goal-guarding** — One of the Apollo six-probe scheming behaviors:
  manipulating future-instance prompts (e.g., editing system prompts
  one expects future copies to read) to preserve the in-context goal.
  `[meinke2024-apollo-scheming §six-evals;
  kb/excerpts/meinke2024-apollo-scheming#sec-six-evals]`
- **Self-exfiltration** — Apollo six-probe scheming behavior: copying
  perceived model weights to external servers when the model believes
  it is about to be replaced. Observed in five frontier models.
  `[meinke2024-apollo-scheming §six-evals]`
- **Oversight subversion** — Apollo six-probe scheming behavior:
  attempting to disable monitoring tools / oversight mechanisms.
  `[meinke2024-apollo-scheming §six-evals]`
- **Persistence under interrogation** — The fraction of follow-up
  questions on which a model maintains its earlier deception.
  Operational distinguisher between scheming (high persistence) and
  sycophancy (typically retracts under challenge). 85%+ for o1 in
  Apollo's setup. `[meinke2024-apollo-scheming §persistence;
  kb/excerpts/meinke2024-apollo-scheming#sec-persistence]`

## Scalable oversight

- **Scalable oversight** — The family of training and evaluation
  protocols designed to let a weaker supervisor — eventually a
  human — supervise a stronger AI system. Animating premise: judging
  cost should grow sub-linearly in the capability being judged. Four
  canonical lineages: debate, IDA, CAI/RLAIF, weak-to-strong
  generalization.
- **Debate (AI safety)** — Two AI debaters in a zero-sum game judged
  by a (weaker) human or LLM judge. Hypothesis: it is easier to
  expose a flawed argument than to construct a sound one, so truthful
  debaters win more often than deceptive ones.
  `[irving2018-debate §abstract]`
- **Consultancy** — Single-debater variant of debate: one AI agent
  responds to judge's questions. In Kenton et al. 2024, debate
  outperforms consultancy across all task domains tested.
  `[kenton2024-debate-llm-judges §abstract]`
- **Iterated amplification (IDA)** — Christiano 2018: recursive
  $A_{k+1} = \mathrm{Distill}(\mathrm{Amplify}(A_k))$ where
  Amplify decomposes hard tasks into supervisable subtasks. Task-
  decomposition dual of debate's claim-decomposition. No clean LLM-
  scale empirical instantiation as of 2026-05.
- **RLAIF (RL from AI Feedback)** — Replacing human preference labels
  with AI-generated preference labels in the RL stage. Introduced as
  the RL-CAI step of `bai2022-cai`. Reduces labeling burden but does
  not address bias-source if the AI feedback model itself inherits
  the bias. `[bai2022-cai §abstract]`
- **Weak-to-strong (W2S) generalization** — Burns et al. 2023's
  empirical methodology: substitute weak-model labels for human
  supervision, study how strong models generalize. The capability gap
  in this analogy stands in for the alignment gap of superalignment.
  `[burns2023-w2s §problem; kb/excerpts/burns2023-w2s#sec-problem]`
- **Performance Gap Recovered (PGR)** — Burns et al. 2023's headline
  metric: $\mathrm{PGR} = (c_{\text{student}} - c_{\text{weak}}) /
  (c_{\text{ceiling}} - c_{\text{weak}})$. PGR = 0 → student matches
  supervisor; PGR = 1 → student fully recovers strong-with-ground-
  truth ceiling. Operationalizes "how much capability can weak
  supervision elicit." `[burns2023-w2s §pgr;
  kb/excerpts/burns2023-w2s#sec-pgr]`
- **Auxiliary confidence loss** — W2S training-side intervention that
  penalizes the strong student for low confidence on its own
  predictions, biasing it toward decisive predictions where the
  strong model has more capability than the weak supervisor revealed.
  Substantially improves PGR on NLP tasks; less effective on chess
  and reward modeling. `[burns2023-w2s §numerical;
  kb/excerpts/burns2023-w2s#sec-numerical]`

## Watermarking and provenance

- **Watermark (LLM text)** — A statistical signal embedded into a
  generated token stream by a sampling-time intervention; invisible
  to a casual human reader but algorithmically detectable from a
  short span of tokens. Canonical scheme: green/red list logit bias.
  `[kirchenbauer2023-watermark §abstract;
  kb/excerpts/kirchenbauer2023-watermark#abstract]`
- **Green list / red list (watermarking)** — Per-step partition of
  vocabulary $V$ into a green list $G_t$ of size $\gamma|V|$ and a
  red list of size $(1-\gamma)|V|$, deterministically derived from a
  hash of the previous token. Green tokens receive a logit bias
  $\delta$; sampling proceeds from the modified distribution.
  Defaults: $\gamma = 0.25$, $\delta = 2.0$.
  `[kirchenbauer2023-watermark §greenred;
  kb/excerpts/kirchenbauer2023-watermark#sec-greenred]`
- **z-score watermark detection** — Detection statistic
  $z = (|s|_G - \gamma T) / \sqrt{T \gamma (1-\gamma)}$ where $|s|_G$
  is the green-token count in candidate text of length $T$. Requires
  no API access — only the keyed hash and the candidate text.
  Interpretable as a one-sided $p$-value via the normal CDF.
  `[kirchenbauer2023-watermark §detection;
  kb/excerpts/kirchenbauer2023-watermark#sec-detection]`
- **Distortion / detectability tradeoff** — Larger $\delta$ →
  stronger watermark, more quality drift. Inherent to soft
  promotion schemes. Hard partitioning (green-only) eliminates the
  tradeoff but degrades quality at low-entropy positions and is
  trivially detectable without the key.
  `[kirchenbauer2023-watermark §soft;
  kb/excerpts/kirchenbauer2023-watermark#sec-soft]`
- **SynthID-Text** — Google DeepMind's tournament-sampling watermark.
  Replaces logit bias with a tournament over candidate tokens
  evaluated by a hash-derived $g$-function. Two modes: distortionary
  (stronger) and non-distortionary (preserves marginal distribution
  under specific assumptions). First production-scale text watermark
  deployment (Gemini App). Published in Nature 2024.
- **Tournament sampling (watermarking)** — SynthID-Text's selection
  procedure: sample $N$ candidate next-tokens, run $\log_2 N$ rounds
  of hash-function $g$ comparisons, advance the higher-$g$ candidate
  in each round, single winner becomes the next token.
  [SPECULATION] non-distortionary mode requires sufficient candidate
  diversity at each step — which fails for low-entropy positions.
- **Paraphrase attack (watermarking)** — Adversarial intervention that
  passes watermarked text through a paraphraser (human or LLM) to
  remove the signal. Kirchenbauer 2024 reports detectability
  preserved at ~800 tokens average under strong human paraphrase at
  FPR $10^{-5}$; Sadasivan et al. 2023 argues the signal is
  fundamentally removable for short spans by a comparable-quality
  paraphraser.

## Safety evaluation

- **Red teaming (LLM)** — Systematic adversarial testing of an LLM
  for harmful or policy-violating outputs. Distinct from generic
  evaluation in that the adversary actively searches for failures.
  HarmBench and JailbreakBench are the canonical standardized
  frameworks. `[harmbench2024 §abstract]`
- **Attack success rate (ASR)** — The fraction of adversarial prompts
  that elicit a successful (policy-violating) response from the
  target model under a fixed scoring procedure. Comparing ASR across
  benchmarks requires identical scoring and threat-model assumptions.
- **HarmBench** — Mazeika et al. 2024's standardized red-teaming
  framework: 510 behaviors, 18 attack methods, 33 target models /
  defenses. Adopted by AISI for pre-deployment evaluations.
  `[harmbench2024 §abstract]`
- **JailbreakBench** — Chao et al. 2024's open robustness benchmark:
  100 target behaviors aligned with OpenAI usage policies,
  standardized threat model / prompts / scoring, public leaderboard,
  repository of jailbreak artifacts. Co-canonical with HarmBench.
  `[chao2024-jailbreakbench §abstract]`
- **Crescendo attack** — Russinovich et al. 2024 multi-turn
  gradual-escalation jailbreak: 29-71% ASR improvement over single-
  turn baselines on AdvBench. Documents that single-turn evals
  systematically underestimate the multi-turn attack surface.
- **AI Safety Level (ASL)** — Anthropic's Responsible Scaling Policy
  capability tier system. ASL-1 (no autonomy / low risk), ASL-2
  (current frontier baseline), ASL-3 (substantially elevated risk;
  specific deployment / security controls required), ASL-4
  (autonomous AI risk; not yet deployed). Each tier specifies
  capability evaluations that must trigger pre-deployment.
- **Responsible Scaling Policy (RSP)** — Anthropic's pre-deployment
  policy framework. Each ASL tier comes with capability triggers
  (which safety evaluations must run) and security commitments
  (which infrastructure / containment requirements must be met). Co-
  canonical with OpenAI's Preparedness Framework and DeepMind's
  Frontier Safety Framework.
- **Dangerous-capability evaluation** — Pre-deployment test for
  specific high-stakes capabilities: autonomous replication,
  autonomous research, cybersecurity attack capability, biological-
  weapons-uplift capability. METR's 77-task suite is the canonical
  cross-vendor reference.
- **Task horizon (METR)** — Capability-quantification metric:
  the distribution of human-time-to-complete for a task. METR's
  empirical scaling-curve work suggests model time-to-complete
  approaches the human-time scale as capabilities improve. Emerging
  field lingua franca for capability measurement.

## Cross-cutting

- **Goodhart's law (in alignment)** — When a proxy metric is
  optimized against, it ceases to be a good measure. The canonical
  failure mode for RLHF: optimizing against a preference model that
  faithfully reflects biased human preferences produces a model
  faithful to the biased proxy, not to the underlying value.
  Foundational to the sycophancy mechanism. `[sharma2023-sycophancy
  §rlhf-causal; kb/excerpts/sharma2023-sycophancy#sec-rlhf-causal]`
- **Mesa-optimizer** — Hubinger et al. 2019 concept: a learned
  optimizer that emerges inside a trained network. The deceptive-
  alignment failure mode is defined for mesa-optimizers; whether
  modern LLMs instantiate the abstraction cleanly is contested.
- **Situational awareness (LLM)** — A model's capacity to reason
  about its own context, training process, and deployment status.
  The Greenblatt 2024 alignment-faking experiment depends on the
  model inferring (from system-prompt cues) that it is in training;
  this is a situational-awareness probe. The Apollo six-probe suite
  similarly leans on the model's awareness of its evaluation
  context. `[greenblatt2024-alignment-faking §setup-wedge;
  kb/excerpts/greenblatt2024-alignment-faking#setup-wedge]`
- **Deliberative alignment** — OpenAI training technique (introduced
  in the o1 stack): train the model to explicitly reason about
  safety policies in its CoT before acting. Conceptually related to
  Constitutional AI's critique step but applied at inference rather
  than training. Apollo's "Stress-Testing Deliberative Alignment for
  Anti-Scheming Training" (2025) reports partial-but-incomplete
  reduction in scheming under this technique.
- **CoT faithfulness** — Whether a model's chain-of-thought trace
  reflects its actual reasoning process. Empirically partial: Lanham
  et al. 2023 / Turpin et al. 2024 show CoT can be unfaithful
  (post-hoc rationalization, label-leakage). Apollo's argument that
  scheming-shaped CoT constitutes evidence of scheming sidesteps
  faithfulness by treating CoT as part of the deployed surface, not
  as ground-truth introspection.

### architecture

## Position encoding

- **Sinusoidal positional encoding** — Vaswani 2017's deterministic sin/cos position vectors; additive at the input. `[vaswani2017 §3.5]`
- **Learned absolute positional embedding** — Per-position trainable vectors indexed by integer position; hard cutoff at maximum sequence length. Used by GPT-2, BERT. `[su2021 §2.2]`
- **Relative position encoding (T5 bias)** — Per-(query, key) distance bias added to attention logits; binned by relative distance. `[su2021 §2.3]`
- **ALiBi** — Attention with Linear Biases. Parameter-free additive bias $-|m-n| \cdot s_h$ at attention logits, with per-head slope. Used by BLOOM, MPT. (Press et al. 2021, arXiv 2108.12409)
- **Rotary Position Embedding** — Multiplicative position encoding that rotates query and key in 2D subspaces by per-frequency angles $m\theta_i$. Inserts position information as relative offsets only via rotation-matrix orthogonality. The de-facto standard in modern decoder-only LLMs. `[su2021 §3.2]`
- **RoPE base frequency $b$** — The base of the geometric frequency progression $\theta_i = b^{-2(i-1)/d}$. Default 10000 (su2021); LLaMA-3 uses 500000; long-context models use 1e6+. Larger $b$ stretches the longest "naturally-seen" wavelength.
- **NTK-aware RoPE scaling** — Per-frequency RoPE rescaling that scales the base $b$ rather than positions, leaving high-frequency dimensions unchanged. Empirically extends context length without degrading short-context quality.
- **YaRN** — "Yet another RoPE extensioN method." Combines NTK-aware scaling with a per-frequency interpolation schedule (no scaling on short-wavelength dims, full scaling on long-wavelength dims, smooth ramp between) and an attention-temperature rescale. ~10× more token-efficient than position interpolation. (Peng et al. 2023, arXiv 2309.00071)
- **LongRoPE / LongRoPE2** — Evolutionary search over per-frequency RoPE scaling factors; allows non-uniform rescaling no analytical method produces. LongRoPE2 reports near-lossless 128K extension at 80× fewer tokens than Meta's earlier approach. (Ding et al. 2024 / 2025)
- **NoPE** — "No Position Encoding." Decoder-only Transformer trained without any explicit positional encoding; relies on the causal mask to recover position information implicitly. Better length generalization on synthetic tasks, but underperforms on realistic LLM workloads. (Kazemnejad et al. 2023)
- **iRoPE** — "interleaved RoPE." LLaMA 4's position scheme: alternates RoPE-equipped attention layers with NoPE attention layers (3:1 ratio) plus temperature scaling in the NoPE layers. Trained at 256K, claimed to extrapolate to 10M-token inference. `[meta-llama4]`
- **M-RoPE** — Multi-axis RoPE for vision-language models. Splits the per-head dimension $d_h$ into temporal/height/width thirds; each third gets its own RoPE rotation. Introduced by Qwen2-VL.
- **Interleaved-MRoPE** — Qwen3-VL refinement of M-RoPE that interleaves the temporal/height/width axes per dimension-pair instead of partitioning the channel range; every dimension-pair sees all three axes.

## Mixture of Experts (MoE)

- **MoE FFN sublayer** — A Transformer FFN sublayer replaced by $E$ parallel FFN "experts" plus a router that selects $k \ll E$ experts per token. Compute scales with $k$, parameter count with $E$. Universal at frontier scale 2024–2026. `[fedus2021-switch §1, mixtral2024]`
- **Router (gate)** — Linear projection $W_g \in \mathbb{R}^{d_{\text{model}} \times E}$ producing per-token expert logits $\boldsymbol{g}_t = W_g \boldsymbol{x}_t$.
- **Top-$k$ routing (token-choice)** — Each token selects its top-$k$ experts by router logit. Switch uses $k=1$, Mixtral $k=2$, DeepSeek-V3 / Qwen3 $k=8$. The dominant routing mode in production. `[fedus2021-switch §2; mixtral2024]`
- **Expert-choice routing** — Inverse: each expert selects its top-$T$ tokens. Guarantees uniform load by construction but breaks autoregressive property; rarely used in production decoder-only LLMs. (Zhou et al. 2022, arXiv 2202.09368)
- **Shared experts ($E_s$)** — "Always-on" experts that process every token, alongside $k$ routed experts. DeepSeek-V2/V3 use $E_s = 1$ or $2$; Qwen3 uses $E_s = 0$. Argument: shared experts capture common knowledge, freeing routed experts to specialize. `[deepseekmoe2024 §3.2]`
- **Fine-grained expert segmentation** — Use many narrow experts ($E$ in hundreds, narrow $d_{\text{ff}}$ each) instead of few wide ones, increasing combinatorial expressiveness $\binom{E}{k}$ of the active set. DeepSeekMoE pattern, adopted by V2/V3 and Qwen3. `[deepseekmoe2024 §3.1]`
- **Expert capacity factor $C$** — Bound on tokens-per-expert per batch: $\lceil C \cdot Tk/E \rceil$. Tokens routed past the limit bypass the MoE layer (residual passthrough). $C \in [1, 2]$ typical. `[fedus2021-switch §2.2]`
- **Load-balancing auxiliary loss** — Regularizer added to the LM loss to push routing toward uniformity. Switch's form is $\mathcal{L}_{\text{aux}} = \alpha E \sum_e f_e P_e$ with $f_e$ = fraction of tokens routed to $e$ and $P_e$ = mean router prob for $e$. `[fedus2021-switch §2.2]`
- **Auxiliary-loss-free load balancing** — DeepSeek-V3's strategy: add a per-expert bias to router logits (not to gating weights), update bias online based on observed load. Removes the historical quality regression of high-coefficient aux loss. `[deepseek-v3 §2]`
- **Global-batch load balancing** — Qwen3's strategy: apply load-balancing loss across the entire training mini-batch (cluster-wide), not within microbatches. Compensates for dropping shared experts. `[qwen3]`
- **Routing collapse** — Pathological MoE training failure where the router routes most tokens to a few "favored" experts; the rest go cold. Self-reinforcing without auxiliary balancing.

## State-space models and hybrids

- **State-space model (SSM)** — Sequence model with recurrent state $\boldsymbol{h}_t$ updated linearly: $\boldsymbol{h}_t = \bar{A} \boldsymbol{h}_{t-1} + \bar{B} u_t$. Per-token compute $O(d^2)$, sequence-linear. (Gu, Goel, Ré 2022)
- **HiPPO** — "High-order Polynomial Projection Operator." Specific structured choice of $A$ matrix that optimally compresses input history under polynomial bases; foundation for S4. (Gu et al. 2020)
- **S4** — Structured State Space sequence model with HiPPO-NPLR $A$, FFT-based convolutional training, recurrent inference. Dual-form architecture. (Gu, Goel, Ré 2022)
- **Selective SSM (Mamba)** — Mamba's contribution: input-dependent $\bar{B}, C, \Delta$ (parameters produced by linear projection of current input), enabling content-aware selection. Breaks convolutional dual; uses parallel scan for training-time parallelism. `[gu2023-mamba]`
- **Parallel scan** — GPU-efficient associative-scan algorithm used by Mamba to compute the input-dependent recurrence in $O(N \log N)$ during training while the recurrence is sequential. (Blelloch 1990; Mamba uses Smith et al. 2022's hardware-aware variant.)
- **State Space Duality (SSD)** — Mamba-2's framing: selective SSMs and a structurally-restricted form of attention compute the same function. The restriction: attention matrix is lower-triangular semi-separable with rank $\le d$. `[mamba2]`
- **Semi-separable matrix** — A structured matrix admitting low-rank off-diagonal blocks; the structural class that encompasses Mamba-2's effective attention matrix.
- **Hybrid SSM/Transformer** — Architecture interleaving SSM blocks and attention blocks. Three patterns: serial (Jamba 1:7), shared-attention (Zamba), parallel-in-block (Hymba). The deployed form for production-scale SSM-using models.
- **Recall bottleneck** — SSM weakness: fixed-size state $\boldsymbol{h}_t$ cannot store every prior token's exact features, so exact-retrieval tasks (copy, NIAH) underperform attention. Primary motivation for hybrids.

## Long context

- **Sliding-window attention** — Each position attends only to the previous $w$ tokens; compute $O(N w d)$ vs $O(N^2 d)$. Local-only by construction; relies on stacked layers' growing receptive field for long-range dependencies. (Longformer 2020, Mistral 2023)
- **Native Sparse Attention** — Three-branch sparse attention (compress + select + sliding-window) trained natively (not retrofit). GQA-group-aligned blockwise selection enables FlashAttention-style kernels under sparsity. `[yuan2025 §3.2–3.4]`
- **Token compression branch (NSA)** — Block-summarizes a window of $l$ keys/values into a single representation via learned MLP. Provides coarse global attention. `[yuan2025 §3.3.1 Eq.7]`
- **Token selection branch (NSA)** — Picks top-$n$ blocks by importance scores derived from the compression branch's attention scores. Provides fine-grained attention. `[yuan2025 §3.3.2 Eq.8–12]`
- **Sliding window branch (NSA)** — Dedicated local-window branch that absorbs local-pattern learning so the compression and selection branches focus on long-range. `[yuan2025 §3.3.3]`
- **Native sparse training** — Pretraining with a sparse attention pattern from the start, vs retrofitting sparsity onto a dense-attention pretrained model. NSA's signature design choice. `[yuan2025 §2.2]`
- **Compressive memory (Infini-Attention)** — Linear-attention-style $d \times d$ memory matrix updated by outer-product accumulation over the entire prefix; read by cross-attention from local-window queries. (Munkhdalai et al. 2024, arXiv 2404.07143)
- **ring attention** — Distributed exact attention: shard sequence across $D$ devices, rotate $K, V$ blocks around the ring, overlap communication with computation. Per-device memory $O(N/D)$; max sequence scales with device count. (Liu, Zaharia, Abbeel 2023, arXiv 2310.01889)
- **Context parallelism** — Production realization of Ring-Attention-style sharding for inference. Yang et al. 2024 reports 93% parallelization efficiency at 1M context, LLaMA 405B, 128 H100s.
- **Position interpolation (PI)** — Simplest RoPE extension: scale positions by $s = L_{\text{test}} / L_{\text{train}}$. Cheap, degrades short-context quality. (Chen et al. 2023)

## FFN

- **GLU (Gated Linear Unit)** — $\mathrm{GLU}(x, W, V) = \sigma(xW) \otimes (xV)$, the elementwise product of a sigmoid-gated and an identity-gated linear projection. Building block for FFN-GLU variants. `[shazeer2020 §2 Eq.4]`
- **SwiGLU** — $\mathrm{FFN}_{\mathrm{SwiGLU}}(x) = (\mathrm{Swish}_1(xW) \otimes xV) W_2$. Three-matrix FFN with Swish gating. Universal in modern decoder-only LLMs. `[shazeer2020 §2 Eq.6]`
- **GEGLU** — GLU with GELU activation: $(\mathrm{GELU}(xW) \otimes xV) W_2$. Equivalent quality to SwiGLU; used in some Google models (T5.1.1, GLaM, mT5). `[shazeer2020 §2 Eq.6]`
- **Bias-free FFN** — T5-style FFN omitting bias terms; SwiGLU is bias-free in production. `[shazeer2020 §1 Eq.2]`
- **2/3 parameter-equalization rule** — When converting a 2-matrix FFN to a 3-matrix GLU-variant, reduce $d_{\text{ff}}$ by factor 2/3 to keep parameter count constant. Gives the modern $d_{\text{ff}} = \tfrac{8}{3} d_{\text{model}}$ rule (rounded for hardware). `[shazeer2020 §2]`

## Reasoning architecture

- **Reasoning-capable LLM** — A base Transformer LLM trained (typically by RLVR) to emit extended chain-of-thought traces between special delimiters before producing a final answer. Architecturally minor (special tokens + format); training-protocol heavy. `[deepseek-r1, qwen3]`
- **Think tokens / `<think>...</think>`** — Reserved special tokens delimiting the reasoning region of a generated output. Reserved as single tokens (not multi-token UTF-8 sequences). `[deepseek-r1, qwen3]`
- **Two-checkpoint reasoning** — DeepSeek's pattern: distinct base (V3) and reasoning (R1) checkpoints. Cleanly separates concerns; doubles deployment.
- **Single-checkpoint hybrid reasoning** — Qwen3's pattern: one model trained to operate in either thinking or non-thinking mode based on a system-prompt flag. Amortizes deployment but trades off specialty.
- **Thinking budget** — Serving-time cap on reasoning-token count before forcing answer commit. The trained model must recover from a forcibly-truncated reasoning trace without hallucinating that the trace was complete. (Qwen3)
- **RLVR (Reinforcement Learning with Verifiable Rewards)** — Training method for reasoning models. Reward = +1 if a programmatic verifier (math grader, code executor, formal checker) accepts the answer, 0 otherwise. The reasoning trace is the policy's free choice. Used by DeepSeek-R1, R1-Zero, Qwen3-Reasoning. See `kb/notes/post-training/rlvr-and-grpo.md`.
- **R1-Zero pattern** — Pure RL from base (no cold-start SFT). Develops reasoning behavior emergently but produces less-readable traces. DeepSeek's ablation result.

## Multi-token prediction

- **Multi-token prediction (MTP)** — Training objective + architectural feature: in addition to the standard next-token head, the model has auxiliary heads predicting tokens at offsets $t+2, t+3, \ldots, t+k$. Used by DeepSeek-V3. `[deepseek-v3 §2.3]`
- **MTP module** — Small Transformer block on top of the backbone that produces the prediction for token $t+i$ given backbone hidden state $\boldsymbol{h}_t^L$. DeepSeek-V3 uses one MTP module ($k=2$ total: main next-token + one MTP step ahead).
- **Self-speculative decoding (via MTP)** — Inference mode where MTP heads emit draft tokens that the main head verifies in parallel. ~1.8× speedup at ~85% acceptance in DeepSeek-V3 serving.

### evaluation

## Agentic benchmarks

- **Agentic benchmark** — Evaluation in which a model autonomously plans, takes actions in an environment, observes results, and replans, scored by an end-state verifier rather than a turn-by-turn rubric. Distinct from question-answering eval. Canonical examples: SWE-bench, GAIA, OSWorld, tau-bench `[jimenez2024-swebench, mialon2023-gaia, xie2024-osworld, yao2024-tau-bench]`.
- **End-state verifier** — A scoring function $V(s_T, g)$ that checks the post-trajectory environment state $s_T$ against a goal $g$, accepting any path that reaches the correct end-state. Contrast: turn-by-turn rubric (rewards specific actions). `[xie2024-osworld §sec-task-spec]`
- **FAIL_TO_PASS / PASS_TO_PASS** — The two test-bundle sets used in SWE-bench's verifier. F2P tests must transition from failing on the pre-patch codebase to passing on the post-patch codebase; P2P tests must continue to pass. Both must hold for a task to be `resolved`. `[jimenez2024-swebench §sec-eval-protocol]`
- **SWE-Bench Verified** — OpenAI's August-2024 500-task subset of SWE-bench, filtered by professional software engineers for unambiguous issue specs and fair test sets. The de-facto leaderboard target as of 2025. `[jimenez2024-swebench]`
- **pass^k** — Reliability metric: the probability that **all** $k$ independent attempts at a task succeed. Dual to pass@k (probability that **at least one** of $k$ attempts succeeds). pass^k measures consistent capability; pass@k measures peak capability. `[yao2024-tau-bench §sec-passk]`
- **GUI grounding** — The ability to map a natural-language target ("click the Save button") to the correct pixel coordinates on a screen. Identified as the dominant failure mode for vision-language agents on OSWorld at launch. `[xie2024-osworld §sec-results]`
- **Tool budget / step budget** — The maximum number of environment actions an agent is allowed in a single task evaluation. SWE-bench traces typically allow 30–50 turns; GAIA limits by tool calls; OSWorld bounds at ~50 actions. Important methodological knob: short budgets favor planning-strong agents, long budgets favor exploration-strong agents.
- **Capture-the-Flag (CTF)** — Cybersecurity task format adopted by Cybench: agents must exploit a target system to retrieve a flag string (typically `flag{...}`), with success scored by string match. `[cybench2024]`

## Reasoning benchmarks

- **GPQA Diamond** — 198-question subset of GPQA where (i) both expert annotators agree on the gold answer, and (ii) at least one non-expert validator with web access got the answer wrong. The headline subset on which leaderboards report. `[rein2023-gpqa §sec-diamond]`
- **Google-proof construction** — Question design protocol where non-expert validators with 30+ minutes of web access score only marginally above random guessing (e.g., 34% on GPQA's 4-way MCQ vs. 25% baseline). The operationalization of "specialized expertise required." `[rein2023-gpqa §abstract]`
- **FrontierMath** — Epoch AI's benchmark of research-grade math problems authored and cross-vetted by 60+ working mathematicians, with single canonical numerical/symbolic answers parseable by automated checker. <2% solve rate at Nov 2024 launch. `[glazer2024-frontiermath §abstract]`
- **HLE (Humanity's Last Exam)** — 2,500-question multimodal cross-domain frontier benchmark from Scale AI / CAIS. Designed as the post-MMLU broad-domain frontier target. `[phan2025-hle §abstract]`
- **ARC-AGI-2** — 2025 successor to ARC-AGI-1 (which o3 effectively solved at >85% in late 2024). Abstract grid-puzzle inductive-generalization benchmark; top commercial AI ~37.6%, refinement-loop systems ~54%, humans ~95%. `[arc-agi-2-2025]`
- **MathArena** — Dynamic math benchmark using post-cutoff competition problems (CMIMC, IMO, etc.) immediately upon release. Contamination-immune by construction. `[matharena-2025]`
- **Random-guess ceiling** — The score a uniform-random model would achieve. 25% for 4-way MCQ (MMLU, GPQA), 10% for 10-way MCQ (MMLU-Pro), ~0% for free-form numeric (FrontierMath). Lower ceiling = larger usable dynamic range above noise floor. `[wang2024-mmlu-pro]`
- **Answer-letter prior** — A pretrained model's learned bias over multiple-choice answer letters (A/B/C/D), independent of question content. Can shift MCQ scores by 1–4 percentage points. Mitigated by 10-way MCQ (MMLU-Pro) or letter-permutation evaluation. `[wang2024-mmlu-pro §sec-delta]`
- **Refinement-loop system** — Iterative program-synthesis architecture that proposes a candidate solution, tests it, and refines based on test feedback. Dominant paradigm in ARC Prize 2025 (54% on ARC-AGI-2 vs. ~37.6% for direct LLM systems). `[arc-agi-2-2025]`

## Eval methodology

- **Prompt-format sensitivity** — Variance in benchmark score $\hat{S}$ induced by changing prompt template, system message, few-shot examples, or answer-extraction format while holding model and benchmark fixed. ~4–5% on MMLU; ~2% on MMLU-Pro. `[wang2024-mmlu-pro §sec-delta]`
- **Data contamination** — The presence of benchmark items, gold answers, or close paraphrases in a model's training corpus, biasing the benchmark score upward. Taxonomy: input contamination, label contamination, distribution contamination, indirect contamination, generative contamination. `[contamination-survey-2025 §abstract]`
- **Static vs. dynamic benchmarking** — Static = fixed benchmark items (MMLU, GPQA); dynamic = items appended post-cutoff (MathArena, LiveCodeBench, GAIA gated leaderboard). Dynamic is contamination-immune by construction but lacks community standards for evaluating quality. `[contamination-survey-2025 §sec-scope]`
- **Verifier mis-specification** — The benchmark's gold answer or scoring rule is wrong, biasing scores in unpredictable ways. MMLU's 6.49% baseline error rate (per-subset up to 57% on Virology) is the canonical case. `[mmlu-redux-2024]`
- **MMLU-Redux** — The Polo et al. 2024 audit re-annotating 5,700 MMLU questions and documenting a 6.49% baseline error rate. Showed leaderboard-rank flips when corrected. `[mmlu-redux-2024]`
- **HELM (Holistic Evaluation of Language Models)** — Stanford CRFM 2022 framework that scores models on 7 metric axes (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency) across 16+ scenarios. Structural argument: benchmarks should ship metric bundles, not single metrics. `[liang2022-helm §sec-metrics]`
- **lm-evaluation-harness** — EleutherAI's open-source eval infrastructure backing HF Open LLM Leaderboard. Tasks are YAML configs reduced to three primitives (`loglikelihood`, `loglikelihood_rolling`, `generate_until`). De-facto standard for open-model evaluation. `[lm-eval-harness]`
- **loglikelihood scoring** — MCQ evaluation method that computes $\log P(\text{option})$ for each option and picks the argmax. Default for MMLU in lm-evaluation-harness; not available for closed APIs that don't expose logprobs (which must use `generate_until`). `[lm-eval-harness §sec-request-types]`
- **N-gram overlap** — Standard contamination-detection method (Brown et al. 2020 GPT-3 §C): flag training documents containing benchmark n-grams of length ≥13. Cheap; misses paraphrase contamination.
- **Min-K% Prob** — Contamination-detection method (Shi et al. 2024) that flags candidate items whose lowest-K%-probability tokens are unusually high — i.e., the model "knew" even the surprising tokens. Model-internal signal; doesn't require training-corpus access.
- **LLM-as-judge** — Evaluation pattern where a strong LLM scores another model's responses (e.g., AlpacaEval, MT-Bench, Arena-Hard). Introduces a contamination axis where the same model class scores its own outputs generously. Sidestepped by string-match (GAIA, FrontierMath) or executable verifiers (SWE-bench, OSWorld).
- **Goodhart pressure on benchmarks** — As a benchmark becomes a publicly-tracked target, training and post-training pipelines optimize against it, eroding its predictive value for true capability. Drives the ~18-month saturation cadence on prominent benchmarks.
- **Saturation** — A benchmark is saturated when frontier models score within the verifier-error-rate noise floor of one another, losing discriminative power. MMLU at >90% is saturated (verifier error rate ~6.5%); GPQA Diamond at >90% is saturated at the frontier as of 2026. `[mmlu-redux-2024]`
- **Saturated benchmark as in-distribution probe** — A saturated benchmark still functions as a fast smoke test ("did fine-tuning break the model?") — its frontier-discrimination value is gone but its drop-detection value is intact. Reason MMLU stays in nearly every model card.

### inference

## KV cache management

- **KV cache** — The per-layer cache of key/value tensors $K_{1:t}, V_{1:t}$ produced during autoregressive decoding so each new token's attention is $O(t)$ rather than $O(t^2)$. Size per token per layer is $2 \cdot n_{\text{kv}} \cdot d_h \cdot \text{bytes}$. `[kwon2023 §2; kb/notes/inference/kv-cache-management#§1]`
- **PagedAttention** — vLLM's KV-cache layout: KV is stored in fixed-size **blocks** (typically 16 tokens), each sequence holds a per-layer **block table** mapping logical block indices to physical blocks, and attention is computed block-wise per Eq.4. Eliminates the contiguous-allocation requirement that produced 50–90% wasted KV memory in pre-vLLM serving stacks. `[kwon2023 §4.1, Eq.4; kb/excerpts/kwon2023#paged-attention]`
- **KV block / block table** — Fixed-size unit of KV storage (16 tokens default in vLLM). The per-sequence block table is the indirection that lets logically-contiguous KV live in physically-scattered memory. `[kwon2023 §4.1; kb/excerpts/kwon2023#block-table]`
- **Reservation / internal / external fragmentation** — vLLM's three-way taxonomy of KV-cache waste in a contiguous-allocation system: reservation = slots reserved for future tokens of an active sequence; internal = pre-allocated maximum-length slots beyond actual generated length; external = unusable gaps between sequences. PagedAttention reduces total waste below 4%. `[kwon2023 §3.1, Fig.4; kb/excerpts/kwon2023#fragmentation]`
- **Copy-on-write (serving)** — When parallel-sampling decode paths (or beam-search hypotheses) diverge, only the diverging block is duplicated; identical-prefix blocks remain shared via reference count. `[kwon2023 §4.4; kb/excerpts/kwon2023#cow]`
- **Swap / recompute (vLLM eviction)** — Two strategies for evicting KV when memory is full: swap the blocks to CPU memory and bring them back when the sequence is rescheduled, or discard them and recompute prefill. Swap wins on long sequences, recompute on short ones. `[kwon2023 §4.5; kb/excerpts/kwon2023#eviction]`
- **RadixAttention** — SGLang's cross-request KV reuse: a radix tree keyed on token-id sequences indexes which prefixes are still in cache, with LRU + reference-count eviction and cache-aware DFS scheduling that runs requests in tree-DFS order to maximise hit rate. `[zheng2024-sglang §3, Theorem 3.1; kb/excerpts/zheng2024-sglang#radix]`
- **Multi-Query Attention (MQA, KV-shape)** — Architectural KV compression: all $H$ query heads share one $(K, V)$ pair, shrinking the KV cache by factor $H$ at the cost of representation. `[shazeer2019 §2; kb/notes/inference/kv-cache-management#§2]`
- **Grouped-Query Attention (GQA, KV-shape)** — $H$ query heads in $G$ groups, each group sharing one $(K, V)$. Shrinks the cache by $H/G$ at minimal quality loss. The production sweet-spot (Llama 2/3, Mistral). `[ainslie2023; kb/notes/inference/kv-cache-management#§2]`
- **Multi-head Latent Attention (MLA)** — DeepSeek's architectural compression: project to a low-rank latent $C \in \mathbb{R}^{d_c}$ shared across heads, expand back to per-head $(K, V)$ at attention time. Achieves ~93% KV-cache reduction vs MHA at deepseek-v2 dimensions. `[deepseek-v2 §2.1; kb/notes/inference/kv-cache-management#§2]`
- **Per-channel keys, per-token values (KVQuant)** — KVQuant's empirical finding: K outliers align along the channel axis (so quantise per-channel), V outliers along the token axis (so quantise per-token). Different statistics for K vs V is the core design lesson. `[kvquant2024; kb/notes/inference/kv-cache-management#§4]`

## Quantization

- **Uniform quantizer (round-to-nearest, RTN)** — $Q(w) = \text{clip}(\lfloor w/\Delta \rceil, -2^{b-1}, 2^{b-1}-1) \cdot \Delta$ with $\Delta = \max(|w|)/(2^{b-1}-1)$ per group. The baseline against which calibration-based methods are compared. `[frantar2022 §2; kb/notes/inference/quantization#§1]`
- **W4A16 / W8A8 / W1.58A8 (notation)** — Compact spec of weight-precision × activation-precision. W4A16 = 4-bit weights, FP16 activations (GPTQ/AWQ regime). W8A8 = INT8 weight + INT8 activation (SmoothQuant). W1.58A8 = ternary $\{-1,0,+1\}$ weights + INT8 activations (BitNet b1.58). `[ma2024-bitnet §2.1; kb/notes/inference/quantization#§3]`
- **Calibration set** — A small set of representative input sequences (typically 128 sequences × 2048 tokens) run through the network to collect activation statistics that drive quantization decisions. `[frantar2022 §2; xiao2023-smoothquant §3.2]`
- **Layer-wise reconstruction (GPTQ objective)** — Quantize each linear layer independently to minimise $\| W X - \widehat{W} X \|_F^2$ where $X$ is the calibration activation. The objective behind OBS/OBQ/GPTQ. `[frantar2022 Eq.1; kb/excerpts/frantar2022#objective]`
- **GPTQ** — Frantar et al. 2022. OBQ-style second-order weight update with three engineering modifications: arbitrary fixed quantization order, lazy batched updates (block size 128), and Cholesky reformulation for Hessian-inverse stability. ~4 GPU-hours for 175B at 3–4 bit. `[frantar2022 §3; kb/notes/inference/quantization#§3.1]`
- **AWQ (Activation-aware Weight Quantization)** — Lin et al. 2023. Per-channel weight scaling driven by *activation* magnitudes (preserving ~1% salient channels at higher precision); reorder-free, no backward pass. Strong W4A16 baseline alongside GPTQ. `[kb/notes/inference/quantization#§3.2]`
- **Salient channel (AWQ)** — A weight column whose corresponding *activation channel* has large magnitude. AWQ's argument: protecting these channels (via per-channel scaling) recovers most of the quantization-error budget. `[kb/notes/inference/quantization#§3.2]`
- **Activation outlier** — A small fraction of activation channels (typically 6 of 4096 in OPT-13B) carrying magnitudes 100× the median. Emerges around 6.7B parameters. The reason naive INT8 activation quantization breaks at scale. `[xiao2023-smoothquant §3, Fig.5; kb/excerpts/xiao2023-smoothquant#outliers]`
- **SmoothQuant migration** — Offline transform $X \cdot \text{diag}(s)^{-1} \cdot \text{diag}(s) W$ that moves outlier difficulty from activation to weight side. Smoothing factor $s_j = \max(|X_j|)^\alpha / \max(|W_j|)^{1-\alpha}$, $\alpha = 0.5$ default. Enables W8A8 INT8 on models up to 530B. `[xiao2023-smoothquant §4, Eq.3-4; kb/excerpts/xiao2023-smoothquant#migration]`
- **BitLinear** — BitNet's drop-in replacement for `nn.Linear`: weights quantized to $\{-1,0,+1\}$ via absmean, activations to INT8 via absmax, SubLN replacing LayerNorm. Constructs a model whose linear-algebra reduces to integer addition. `[ma2024-bitnet §2.1; kb/excerpts/ma2024-bitnet#bitlinear]`
- **Absmean ternary quantization** — BitNet b1.58's weight quantizer: $\beta = \frac{1}{nm}\sum_{i,j}|W_{ij}|$, $\widehat{W}_{ij} = \text{round\_clip}(W_{ij}/(\beta+\epsilon), -1, +1)$. Per-tensor $\beta$, ternary output. Trained from scratch in 1.58-bit, not post-quantized. `[ma2024-bitnet §2.1, Eq.1-3; kb/excerpts/ma2024-bitnet#absmean]`
- **GPTQ-vs-AWQ-vs-SmoothQuant spec sheet** — GPTQ: W4A16, layer-wise reconstruction, sequential. AWQ: W4A16, salient-channel preservation, no calibration backprop. SmoothQuant: W8A8, activation-outlier migration. The three occupy distinct cells of the (precision × technique) grid. `[kb/notes/inference/quantization#§3]`
- **FP8 (E4M3 / E5M2)** — IEEE-style 8-bit floating-point. E4M3: 4-exp / 3-mantissa, narrower range / better resolution (typical for activations). E5M2: 5-exp / 2-mantissa, wider range (typical for gradients). Production choice from H100 onward. `[shah2024 §1; kb/notes/inference/quantization#§4]`
- **Microscaling (MX) format** — Block-wise FP format: a per-block shared 8-bit exponent + per-element 4–8 bit elements (MXFP4, MXFP6, MXFP8). Open standard ratified by OCP 2023. `[kb/notes/inference/quantization#§4]`

## Speculative decoding

- **Speculative decoding / sampling** — Inference acceleration by drafter–target factoring: a cheap drafter $M_q$ proposes $\gamma$ tokens; the expensive target $M_p$ scores them in one parallel forward pass; a rejection-sampling rule preserves the exact target distribution. `[leviathan2023 §2; kb/excerpts/leviathan2023#algorithm]`
- **Rejection-sampling rule (speculative decoding)** — For draft token $x \sim q$: accept with probability $\min(1, p(x)/q(x))$; on rejection, sample from $\text{norm}(\max(0, p - q))$. Guarantees the accepted token's marginal is exactly $p$. `[leviathan2023 §2.3; kb/excerpts/leviathan2023#rejection-rule]`
- **Acceptance rate $\alpha$** — $\alpha = \mathbb{E}_{x \sim q}[\min(1, p(x)/q(x))]$. The probability a single drafter token is accepted; the dominant lever in the speedup formula. Typical values: 0.5–0.8 for well-matched drafter–target pairs. `[leviathan2023 §3.2; kb/excerpts/leviathan2023#alpha]`
- **Speedup theorem (Leviathan)** — $\mathbb{E}[\text{tokens per cycle}] = (1-\alpha^{\gamma+1})/(1-\alpha)$ (Eq.1, capped geometric). Walltime improvement $= [(1-\alpha^{\gamma+1})/(1-\alpha)] / (c\gamma+1)$ with $c = T_q/T_p$ (Theorem 3.8). `[leviathan2023 §3.2, Eq.1, Theorem 3.8; kb/excerpts/leviathan2023#theorems]`
- **Drafter / target model** — The two LMs in speculative decoding. Drafter $M_q$ is small/fast, target $M_p$ is large/correct. The target is the only model whose distribution matters for output quality. `[leviathan2023 §1; kb/notes/inference/speculative-decoding#§3]`
- **Tree attention (Medusa/EAGLE)** — A draft tree of token candidates verified in one target forward pass via a custom attention mask that lets every node attend only to its ancestors. Enables draft-width $> 1$ per position. `[cai2024-medusa §2.2; kb/excerpts/cai2024-medusa#tree-attention]`
- **Medusa heads** — $K$ extra decoding heads $h^{(k)}: \mathbb{R}^d \to \mathbb{R}^{|V|}$ predicting tokens at offsets $t+1, t+2, \ldots, t+K$ from the same hidden state. Each head is a small residual MLP + the original LM head. `[cai2024-medusa §2.1, Eq.0; kb/excerpts/cai2024-medusa#heads]`
- **Medusa-1 / Medusa-2 training** — Medusa-1: heads-only fine-tuning, target frozen. Medusa-2: jointly fine-tunes target + heads with loss $\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda_0 \mathcal{L}_{\text{heads}}$ on a small fine-tuning set; recovers the most quality + speed. `[cai2024-medusa §2.3, Eq.1-2; kb/excerpts/cai2024-medusa#training]`
- **Typical acceptance** — Medusa's quality–throughput knob: accept any draft token whose probability under the target exceeds $\min(\epsilon, \delta \cdot \exp(-H(p)))$. Trades exact-distribution preservation for higher acceptance and longer accepted runs. `[cai2024-medusa §2.4; kb/excerpts/cai2024-medusa#typical-acceptance]`
- **EAGLE feature-level autoregression** — Drafts at the penultimate-layer hidden-state (feature) level, not the token level. Drafter input at step $k$ is $(\text{feature}_{t+k-1}, \text{token}_{t+k})$ — the shifted-token trick disambiguates which sample produced which feature. `[li2024-eagle §3; kb/excerpts/li2024-eagle#feature-autoregression]`
- **Lookahead decoding** — Drafter-free speculative decoding. Maintains a Jacobi-style n-gram pool of recently-seen completions and verifies them against the target. No drafter to train; lower acceptance than EAGLE but zero-shot deployable. `[kb/notes/inference/speculative-decoding#§3.4]`

## Serving systems

- **Continuous (iteration-level) batching** — Orca's contribution: schedule at the granularity of one decoder step rather than one full request, so finished requests free their slot immediately and incoming requests join mid-batch. The throughput substrate every modern stack inherits. `[kb/notes/inference/serving-systems#§2]`
- **Prefill phase** — The first forward pass over the full prompt; compute-bound, dominated by GEMM on the prompt-length dimension. Latency target: TTFT (time-to-first-token). `[kb/notes/inference/serving-systems#§5]`
- **Decode phase** — Subsequent autoregressive token generation; memory-bound, dominated by KV-cache reads. Latency target: TPOT (time-per-output-token). `[kb/notes/inference/serving-systems#§5]`
- **Prefill/decode disaggregation** — Run prefill and decode on separate worker pools (Splitwise, DistServe). Trades cross-machine KV-transfer overhead for the ability to size each pool to its dominant resource (compute vs memory bandwidth). Tightens TTFT and TPOT tails simultaneously. `[kb/notes/inference/serving-systems#§5]`
- **Chunked prefill (Sarathi)** — Break a long prefill into chunks of $S$ tokens, interleaving each chunk with a decode microbatch; piggybacks decode work onto the prefill compute envelope. Improves throughput at modest TTFT cost. `[kb/notes/inference/serving-systems#§5]`
- **TTFT / TPOT** — Time-To-First-Token and Time-Per-Output-Token. The two production SLOs serving systems are tuned against. `[kb/notes/inference/serving-systems#§1]`
- **FlashAttention substrate** — IO-aware tiled-softmax attention kernel (Dao 2022/2023/2024). Underneath every modern serving stack; vLLM and SGLang call into FA-2/3 for the actual attention compute, layering PagedAttention/RadixAttention on top for the KV-cache management. `[dao2022 §3; kb/notes/inference/serving-systems#§6]`

## Structured output

- **Constrained decoding** — Mask the next-token logits at every step to allow only tokens consistent with a target grammar (regex, JSON schema, GBNF, arbitrary CFG). Naive implementations query the grammar engine for every token in the vocabulary at every step — the bottleneck modern engines fix. `[kb/notes/inference/structured-output#§1]`
- **Context-independent / context-dependent vocabulary split (XGrammar)** — XGrammar's core trick: precompute (offline) the mask for tokens whose grammar-allowed-or-not status depends only on the grammar state (~99% of vocab); compute at runtime only the small remainder (~1%) that depends on the active context. Brings constrained-decoding overhead from ms/token to μs/token. `[xgrammar2024; kb/notes/inference/structured-output#§3]`
- **Compressed FSM (SGLang)** — When a JSON-schema or regex forces a deterministic chain of tokens (e.g. the constant prefix `{"summary": "`), merge those single-edge runs in the FSM so the constrained decoder emits the entire deterministic prefix in one step instead of token-by-token. `[zheng2024-sglang §4; kb/excerpts/zheng2024-sglang#compressed-fsm]`
- **JSON-schema mode** — Production-facing API surface (OpenAI, Gemini, vLLM, llama.cpp) where the user supplies a JSON Schema and the engine guarantees output validates against it. Implemented via constrained decoding on the schema-derived FSM. `[kb/notes/inference/structured-output#§2]`

### interpretability

# Interpretability — glossary additions (Phase 2)

These fragments are intended to be merged into `theory/kb/glossary.md`
under the indicated section headings (or new sections). Format: one
`## Section heading` block per topic cluster, each followed by
`- **Term** — definition. citation.` lines, matching the pilot
glossary format (cf. "Attention variants and KV-cache compression").

## Mechanistic interpretability — core vocabulary

- **Mechanistic interpretability (MI)** — the research program of
  reverse-engineering neural networks into human-readable algorithms,
  at the level of circuits (sub-graphs) and features (directions).
  Distinct from behavioral interpretability (input/output black-box
  evaluation) and post-hoc rationalization. `[wang2022-ioi §1; meng2022-rome §1]`.
- **Circuit** — a sub-graph $C$ of a model's computational graph $M$
  that, when used in place of $M$ via knockouts of $M \setminus C$,
  approximately reproduces $M$'s behavior on a target task
  `[wang2022-ioi §2; kb/excerpts/wang2022-ioi#sec-2-definition]`.
- **Feature** — a direction $\mathbf{d} \in \mathbb{R}^n$ in some
  activation space such that the projection $\mathbf{x}^\top \mathbf{d}$
  (or equivalently, an SAE latent activation with decoder column
  $\mathbf{d}$) corresponds to a human-interpretable property. The
  unit of analysis at the *what* level; circuits operate at the *how*
  level.
- **Faithfulness, completeness, minimality** — the three quantitative
  validity criteria for a circuit. Faithfulness: $C$ matches $M$'s
  performance on the task. Completeness: $C$ contains every node
  needed. Minimality: every node in $C$ is necessary `[wang2022-ioi §4;
  kb/excerpts/wang2022-ioi#sec-4-criteria]`.
- **Polysemanticity** — the property that an individual neuron
  responds to multiple unrelated concepts. The motivation for SAEs
  (which lift polysemanticity by inflating the basis). `[leask2025-sae-not-canonical §1]`.
- **Monosemanticity** — the property that a feature responds to a
  single, human-interpretable concept. The (contested) goal of SAE
  training; "Towards Monosemanticity" (Bricken 2023) is the founding
  reference. [CONTRADICTION] Whether SAEs achieve true monosemanticity
  is challenged by `[leask2025-sae-not-canonical §1; kb/excerpts/leask2025-sae-not-canonical#sec-1-canonical]`.
- **Superposition** — the hypothesis (Elhage 2022) that a network with
  $n$ neurons can represent more than $n$ approximately-orthogonal
  features as long as they fire sparsely. SAEs lift superposition by
  inflating the dictionary size $M \gg n$. `[gao2024-topk-saes §1]`.
- **Residual stream** — the linear data bus running through a
  decoder-only transformer's layers; every sublayer (attention, MLP)
  reads from and additively writes to it. The substrate that lens
  techniques and SAEs decompose. Vocabulary established in Elhage et
  al. 2021 (Mathematical Framework for Transformer Circuits).

## Natural Language Autoencoder — methodology vocabulary

- **Natural Language Autoencoder (NLA)** — Anthropic 2026's
  interpretability method: a pair of LLM modules (AV + AR) jointly
  trained with reinforcement learning to round-trip a residual-stream
  activation through a natural-language explanation. The text
  bottleneck makes the explanation testable via reconstruction
  fidelity (FVE). Used in pre-deployment audits to surface
  unverbalized model cognition — including evaluation awareness.
  `[anthropic2026-nla §method; kb/excerpts/anthropic2026-nla#sec-method]`
- **Activation verbalizer (AV)** — The AV $\mathrm{AV}(z \mid h_\ell)$
  takes a residual-stream activation $h_\ell$ and emits a
  natural-language explanation token sequence $z$. One half of the
  NLA pair. `[anthropic2026-nla §method;
  kb/excerpts/anthropic2026-nla#sec-method]`
- **Activation reconstructor (AR)** — The AR $\mathrm{AR}(z)$ takes
  the verbalized explanation $z$ and produces a reconstructed
  activation $\hat h_\ell \in \mathbb{R}^{d_\mathrm{model}}$. Trained
  jointly with the AV to minimise the round-trip MSE on activation
  reconstruction. `[anthropic2026-nla §method;
  kb/excerpts/anthropic2026-nla#sec-method]`
- **Fraction of Variance Explained (FVE)** — NLA quality metric:
  $\mathrm{FVE} = 1 - \mathcal{L}/\mathbb{E}\lVert h_\ell - \bar h_\ell\rVert^2$
  where $\mathcal{L}$ is the round-trip MSE. FVE = 1 is a lossless
  natural-language code; FVE = 0 means the explanation does no better
  than the mean activation. `[anthropic2026-nla §method;
  kb/excerpts/anthropic2026-nla#sec-method]`
- **Unverbalized evaluation awareness** — Cases where a model
  internally suspects it is being tested but does not say so
  in its visible output. NLAs were validated as a tool for surfacing
  this signal during Anthropic's pre-deployment audit of Claude
  Opus 4.6. `[anthropic2026-nla §auditing;
  kb/excerpts/anthropic2026-nla#sec-auditing]`

## Activation patching — methodology vocabulary

- **Activation patching** — the canonical causal-intervention method
  in MI: identify which model components matter for a behavior by
  swapping their activations between forward passes on paired
  (clean, corrupted) prompts and measuring the effect on the output
  `[meng2022-rome §2; kb/excerpts/meng2022-rome#sec-2; zhang2023-apatching §2.1]`.
  Also known as causal tracing, interchange intervention, causal
  mediation analysis, or representation denoising.
- **Causal tracing** — Meng et al. 2022's specific implementation of
  activation patching: clean run + Gaussian-noised corrupted run +
  patched-with-clean-state run, computed over every (token, layer)
  cell to produce a heatmap of indirect effects
  `[meng2022-rome §2.2; kb/excerpts/meng2022-rome#sec-2-2]`.
- **Indirect effect (IE)** — the change in the output metric when the
  activation at a specific (component, position, layer) is restored
  from a cached clean run during an otherwise-corrupted forward pass.
  $\text{IE} = \mathbb{P}_{*,\,\text{clean }h_i^{(l)}}[r] -
  \mathbb{P}_*[r]$ `[meng2022-rome §2 effects; kb/excerpts/meng2022-rome#sec-2-effects]`.
- **Average indirect effect (AIE)** — IE averaged over a dataset of
  paired prompts; the standard metric for component importance.
- **Path patching** — a refinement of activation patching that
  isolates the direct effect of a component $C$ on a downstream
  component $R$, holding fixed the indirect routes through other
  components. Introduced in Wang et al. 2022 (IOI)
  `[wang2022-ioi §3.1; kb/excerpts/wang2022-ioi#sec-3-1-path-patching]`.
- **Attribution patching** — gradient-based first-order approximation
  of the patching effect: $\text{IE}_C \approx \langle \nabla_h m,
  \Delta h \rangle$. Cost: 1 forward + 1 backward pass for all
  components. Tier B (Nanda blog); approximation breaks in non-linear
  regions.
- **Gaussian Noising (GN) / Symmetric Token Replacement (STR)** — the
  two corruption methods for activation patching. GN adds
  $\mathcal{N}(0, 3\sigma_\text{textset})$ to subject-token embeddings
  `[meng2022-rome §2]`; STR replaces the subject with a similar token
  (e.g., "Eiffel Tower" → "Colosseum") preserving sequence length
  `[zhang2023-apatching §2.1; kb/excerpts/zhang2023-apatching#sec-2-1-corruption]`.
  STR is the recommended default; GN can drive activations
  off-distribution `[zhang2023-apatching §1 findings]`.
- **Logit difference (LD)** — an activation-patching metric:
  $\text{LD}(r, r') = \text{Logit}(r) - \text{Logit}(r')$. The
  patching effect, normalized to $[0, 1]$, is $[\text{LD}_\text{pt} -
  \text{LD}_*] / [\text{LD}_\text{cl} - \text{LD}_*]$
  `[zhang2023-apatching §2.1 metrics; kb/excerpts/zhang2023-apatching#sec-2-1-metrics]`.
  Recommended over raw probability because it is sensitive to
  components that *push against* the correct answer.
- **Mean ablation** — knocking out a node by replacing its activation
  with the mean activation across a reference distribution; preferred
  to zero ablation, which is arbitrary `[wang2022-ioi §2;
  kb/excerpts/wang2022-ioi#sec-2-definition]`.
- **Knockout** — generic term for setting a component's contribution
  to a fixed reference value (zero or mean) for circuit-validation
  purposes `[wang2022-ioi §2.1]`.

## Sparse autoencoders — architecture vocabulary

- **Sparse autoencoder (SAE)** — a wide ($M \gg n$) two-layer
  encoder-decoder trained to reconstruct a model's activations
  $\mathbf{x} \in \mathbb{R}^n$ as a sparse linear combination of
  learned dictionary directions $\{\mathbf{d}_i\}_{i=1}^M$
  `[rajamanoharan2024-jumprelu §2 Eq.1-2; kb/excerpts/rajamanoharan2024-jumprelu#sec-2]`.
- **Dictionary / latents** — the columns $\mathbf{d}_i$ of
  $\mathbf{W}_\text{dec}$; each is a feature direction in
  $\mathbb{R}^n$. Templeton 2024 distinguishes "feature" (the
  conceptual entity the model represents) from "latent" (the SAE's
  learned approximation). `[gemma-scope-2024 §2.1]`.
- **Expansion factor** — the ratio $M / n$ of dictionary size to
  activation dimension. Typical values 8–1000.
- **L0 sparsity** — the headline sparsity metric: the number of
  non-zero feature activations per token. The $x$-axis of the
  reconstruction-fidelity Pareto plots
  `[rajamanoharan2024-jumprelu §1 Pareto; kb/excerpts/rajamanoharan2024-jumprelu#sec-1-pareto]`.
- **TopK SAE** — SAE variant with $\sigma = \text{TopK}(\cdot)$:
  zero all but the top $K$ pre-activations. Direct $L_0$ control, no
  L1 penalty needed. OpenAI 2024 `[gao2024-topk-saes §2.3 Eq.2;
  kb/excerpts/gao2024-topk-saes#sec-2-3]`.
- **JumpReLU SAE** — SAE variant with a discontinuous activation
  $\text{JumpReLU}_\theta(z) = z \cdot H(z - \theta)$ and
  per-feature learned threshold $\theta$. Trained with L0 penalty via
  straight-through estimators. SOTA reconstruction at fixed L0 on
  Gemma 2 9B `[rajamanoharan2024-jumprelu §3 Eq.4, §1 Pareto]`.
- **Gated SAE** — SAE variant with two encoder kernels, one for
  gating (binary on/off) and one for magnitude estimation.
  Architecturally equivalent (with weight sharing) to JumpReLU
  `[rajamanoharan2024-jumprelu §2]`.
- **Straight-through estimator (STE)** — a backward-pass surrogate
  used to backprop through the discontinuous JumpReLU and the L0
  penalty. The forward pass uses the discontinuity; the backward
  pass uses a smooth (kernel-density-estimated) gradient
  `[rajamanoharan2024-jumprelu Abstract; rajamanoharan2024-jumprelu §3]`.
- **Dead latent** — an SAE feature that has stopped activating across
  the training distribution (typically because its encoder weights
  collapsed during early training). Up to 90% of latents in a large
  SAE can be dead without mitigation `[gao2024-topk-saes §2.4;
  kb/excerpts/gao2024-topk-saes#sec-2-4]`.
- **AuxK loss** — auxiliary loss added by Gao et al. 2024 to revive
  dead latents: model the residual reconstruction error using the
  top-$k_\text{aux}$ dead latents `[gao2024-topk-saes §2.4]`.
- **Shrinkage** — the bias introduced by the L1 penalty in ReLU SAEs:
  positive feature activations are systematically underestimated, so
  reconstructed activations have smaller magnitudes than true values.
  Avoided by TopK and JumpReLU `[gao2024-topk-saes §2.3 benefits]`.
- **SAE stitching** — Leask et al. 2025: insert/swap latents from a
  larger SAE into a smaller one to compare bases. Reveals novel
  latents (in the larger but not the smaller SAE) and reconstruction
  latents (in both with similar behavior)
  `[leask2025-sae-not-canonical §1; kb/excerpts/leask2025-sae-not-canonical#sec-1-techniques]`.
- **meta-SAE** — an SAE trained on the *decoder matrix* of another
  SAE. Reveals that latents in a larger SAE often decompose into
  sparse combinations of meta-latents, contradicting the "atomic
  features" assumption `[leask2025-sae-not-canonical §1 Einstein;
  kb/excerpts/leask2025-sae-not-canonical#sec-1-einstein]`.
- **Canonical features (contested)** — the (challenged) hypothesis
  that SAEs find a unique, complete, atomic decomposition of model
  activations into features. [CONTRADICTION] Leask et al. 2025
  argue against canonicality with both stitching and meta-SAE
  evidence `[leask2025-sae-not-canonical §1; kb/excerpts/leask2025-sae-not-canonical#sec-1-canonical]`.

## Transcoders and circuit tracing

- **Transcoder** — a wide ReLU MLP (encoder–decoder) trained to
  approximate the input/output behavior of an MLP sublayer:
  $\text{TC}(\mathbf{x}) = \mathbf{W}_\text{dec}\,\text{ReLU}(\mathbf{W}_\text{enc}
  \mathbf{x} + \mathbf{b}_\text{enc}) + \mathbf{b}_\text{dec}$, with
  L1 sparsity penalty on the hidden activations
  `[dunefsky2024-transcoders §3.1 Eq.3-5; kb/excerpts/dunefsky2024-transcoders#sec-3-1]`.
  Distinct from an SAE: a transcoder approximates a *function* (the
  MLP); an SAE approximates a *vector* (an activation).
- **Cross-layer transcoder** — Lindsey et al. 2025: a transcoder
  whose decoder writes into the residual stream of *all subsequent*
  layers, not just the immediate next one. Makes a feature's
  skip-layer contributions explicit in the attribution graph.
  HTML-only at transformer-circuits.pub.
- **Input-invariance** — the property of a transcoder (and *not* of
  an SAE-on-MLP-output) that the connections between features can be
  stated independently of any specific input. Necessary for
  weight-based circuit analysis through MLPs
  `[dunefsky2024-transcoders §1; kb/excerpts/dunefsky2024-transcoders#sec-1-input-invariance]`.
- **Attribution graph** — the output of circuit tracing: a sparse
  directed graph from input embeddings to output unembedding, with
  feature activations as nodes and Jacobian-weighted attribution
  scores as edges. Computed in 1 forward + 1 backward pass on the
  transcoder-replaced model (Lindsey 2025).
- **Circuit tracing** — Anthropic's 2025 methodology: replace MLPs
  with cross-layer transcoders, then compute attribution graphs by
  Jacobian back-attribution from a target node. Synthesis of SAE
  feature decomposition + activation patching causal intervention.
  `[lindsey2025-circuit-tracing FORUM-SIGNAL — HTML-only]`.

## Lens techniques — vocabulary

- **Logit lens** — apply the final LayerNorm + unembedding $W_U$ to
  an intermediate residual-stream activation $\mathbf{h}^{(\ell)}$
  to read out a layer-$\ell$ vocabulary distribution. nostalgebraist
  2020 (LessWrong, Tier B). Formal expression and decomposition in
  `[belrose2023-tuned-lens §2 Eq.3]` (PDF p.2).
- **Tuned lens** — Belrose et al. 2023: per-layer affine
  "translators" $\mathbf{h}_\ell \mapsto A_\ell \mathbf{h}_\ell +
  b_\ell$ trained with a distillation loss against final-layer
  logits. More reliable than the raw logit lens, especially on
  BLOOM/GPT-Neo `[belrose2023-tuned-lens §1; PDF p.1-2]`.
- **Causal Basis Extraction (CBE)** — Belrose et al. 2023's algorithm
  for finding residual-stream directions with highest influence on
  the tuned lens output. Connection between lens analysis and MI
  feature attribution.
- **DoLa** — *Decoding by Contrasting Layers*: an inference-time
  application of lens analysis. The decoded logits are
  $\text{logit}^{(L)}(t) - \text{logit}^{(\ell^*)}(t)$ for an
  adaptively chosen early layer $\ell^*$, improving factuality.
  Chuang et al. 2023.

## Probing — vocabulary

- **Probe** — a (typically linear) classifier $g(\mathbf{h}) =
  \sigma(\mathbf{w}^\top \mathbf{h} + b)$ trained on labeled
  activations to predict an external property $y$. Foundational
  reference: Alain & Bengio 2017.
- **Mass-mean probing** — Marks & Tegmark 2023: define the probe
  direction as $\mathbf{d}_\text{class} = \mathbb{E}[\mathbf{h} \mid
  y = c_1] - \mathbb{E}[\mathbf{h} \mid y = c_0]$, then project
  activations onto $\mathbf{d}_\text{class}$. Simpler and more
  generalization-robust than logistic regression on the same
  features `[marks-tegmark-2023-truth §1]`.
- **Truth direction** — the canonical mass-mean probe direction for
  truth-value of a statement, derived in Marks & Tegmark 2023. A
  small linear subspace separates true and false statements across
  many models and datasets.
- **Correlational vs. causal probing** — a probe demonstrates that
  the model *represents* a property; an activation patch along the
  probe direction demonstrates that the model *uses* it. The 2023+
  literature is converging on always pairing the two.

### post-training

# Post-training glossary additions (Phase 2)

Subagent: post-training (2026-05-04)

Each term gets a 1-3 sentence definition + a paper-key citation.

## RLHF and reward modeling

- **RLHF (Reinforcement Learning from Human Feedback)** — The 3-stage post-training pipeline: SFT → reward-model training on pairwise preferences → PPO with KL anchor against $\pi^{\mathrm{SFT}}$. Canonical reference is InstructGPT `[ouyang2022-instructgpt §3]`.
- **Bradley–Terry preference model** — Assumes pairwise preferences depend on a difference of latent rewards: $p(y_w \succ y_l|x) = \sigma(r(x,y_w) - r(x,y_l))$. The basis for both the RM training loss and DPO `[ouyang2022-instructgpt §3.5; rafailov2023-dpo §4]`.
- **reward model (RM, $r_\phi$)** — A neural network trained on Bradley–Terry preferences to score response quality. Initialized from $\pi^{\mathrm{SFT}}$ with a scalar head replacing the LM head. Used as the reward signal for PPO `[ouyang2022-instructgpt §3.5]`.
- **PPO-ptx** — InstructGPT's RLHF objective with an added pre-training-mixture term: $+\gamma\, \mathbb{E}_{\mathrm{pretrain}}[\log\pi_\theta]$. Reduces alignment tax on standard NLP benchmarks `[ouyang2022-instructgpt §3.5]`.
- **KL anchor / KL penalty** — The $\beta \log\pi_\theta(y|x)/\pi^{\mathrm{SFT}}(y|x)$ term in RLHF objectives that prevents the policy from drifting too far from the SFT initialization. Implemented per-token in the reward in InstructGPT-style PPO `[ouyang2022-instructgpt §3.5]`.
- **Alignment tax** — The performance regression on standard NLP benchmarks observed after RLHF. PPO-ptx mitigates it by mixing in pre-training likelihood `[ouyang2022-instructgpt §3.5]`.
- **WARM (Weight-Averaged Reward Models)** — Average parameters of $K$ independently-trained RMs to reduce overoptimization and improve robustness. Adopted in Gemma 3 post-training `[warm-2024]`.
- **WARP** — Alternates RL training with policy weight averaging; companion to WARM at the policy side. Used together in Gemma 3 `[warp-2024]`.
- **Reward overoptimization** — The gap between proxy reward $r_\phi$ and gold reward grows as KL($\pi_\theta\|\pi_{\mathrm{ref}}$) grows; the policy "hacks" the RM, scoring high on $r_\phi$ while degrading on true quality. Affects both RLHF (Gao et al. 2023) and DPO-class methods `[rlhf-overopt-2024]`.
- **Reward hacking** — Policy exploits an artifact of the reward signal that does not correspond to true quality (e.g., verbosity, sycophancy, format-shortcuts). Distinct from overoptimization in that hacking can occur even at low KL `[deepseek-r1 §2.2]`.

## DPO and offline preference optimization

- **DPO (Direct Preference Optimization)** — Closed-form, classification-loss reformulation of RLHF: trains the policy directly on preference pairs by inverting the closed-form RL optimum. No reward model, no on-policy sampling, no PPO loop `[rafailov2023-dpo §4 Eq.7]`.
- **Implicit reward** — In DPO, the policy log-ratio $\hat{r}_\theta(x,y) = \beta\log\pi_\theta(y|x)/\pi_{\mathrm{ref}}(y|x)$ that plays the role of a reward function. The policy is trained on preferences over implicit-reward differences `[rafailov2023-dpo §4]`.
- **DPO reparameterization** — The closed-form mapping between optimal policy and reward (Eq. 4 of Rafailov 2023): $\pi^*(y|x) \propto \pi_{\mathrm{ref}}(y|x)\exp(r(x,y)/\beta)$. Inverting this gives the implicit reward `[rafailov2023-dpo §4 Eq.4-5]`.
- **DPO length bias** — DPO-trained models tend to produce longer outputs than the reference because long responses are easier to satisfy with high log-ratios. Addressed by SimPO's length normalization `[meng2024-simpo §3-length]`.
- **IPO (Identity Preference Optimization)** — DPO variant replacing log-sigmoid loss with squared loss on the preference gap, addressing DPO's overconfidence pathology `[azar2023-ipo]`.
- **KTO (Kahneman-Tversky Optimization)** — Operates on single-rating data $(x,y,\mathrm{label}\in\{\text{good},\text{bad}\})$ instead of pairwise preferences. Asymmetric prospect-theoretic loss with loss aversion `[ethayarajh2024-kto]`.
- **ORPO (Odds Ratio Preference Optimization)** — Combines SFT and preference losses in a single objective; no reference model, no SFT stage. Trains from base directly via log-odds ratio of SFT loss between chosen and rejected responses `[hong2024-orpo]`.
- **SimPO (Simple Preference Optimization)** — Reference-free DPO variant using length-normalized average log-prob as implicit reward, plus a target reward margin $\gamma$. Eliminates DPO length bias on length-controlled benchmarks `[meng2024-simpo §3]`.
- **Reference-free vs reference-anchored** — Axis along which DPO variants split. Reference-anchored: DPO, IPO, KTO. Reference-free: ORPO, SimPO, CPO. Reference-anchored variants depend on $\pi_{\mathrm{ref}}(y|x)$ in the loss; reference-free variants use only $\pi_\theta$ `[meng2024-simpo §3; hong2024-orpo §2]`.
- **Iterative DPO** — Multi-round pipeline: run DPO, sample new responses, label them, repeat. Llama-3 and Tülu 3 reference recipe `[meta-llama3; tulu3]`.

## Constitutional AI and RLAIF

- **RLAIF (RL from AI Feedback)** — Replace human preference labels in RLHF with AI-generated preference labels, typically from a feedback model conditioned on a written constitution `[bai2022-cai §4]`.
- **Constitutional AI (CAI)** — Anthropic 2022 framework for training harmless assistants without human harm labels. Two stages: SL-CAI (sample → critique under principle → revise → SFT) and RL-CAI (sample pairs → AI judge under principle → train RM → PPO) `[bai2022-cai §3-4]`.
- **Constitution** — A list of principles ("the assistant should not be racist," etc.) used by the feedback model to evaluate / critique responses in CAI. Typical size: ~16 principles `[bai2022-cai §3]`.
- **SL-CAI** — Supervised learning stage of Constitutional AI. Sample initial responses, critique them under a randomly-chosen principle, revise, and SFT on revised responses `[bai2022-cai §3]`.
- **RL-CAI** — Reinforcement learning stage of Constitutional AI. Generate AI preferences by having a feedback model choose between response pairs under a principle, train a reward model on AI preferences, run PPO `[bai2022-cai §4]`.
- **R-CAI (Reverse Constitutional AI)** — Inverts the constitution for adversarial / toxic data generation; probability-clamped RLAIF for red-team data synthesis (April 2026) `[r-cai-2026]`.

## RLVR and GRPO

- **RLVR (Reinforcement Learning with Verifiable Rewards)** — Post-training paradigm where the reward is a deterministic programmatic verifier (math equality, code unit-tests, format check), not a learned reward model. The policy still uses on-policy RL `[deepseek-r1 §2.2; tulu3]`.
- **Verifiable reward** — A 0/1 reward computed by a programmatic check: $R(x,y) = \mathbb{1}[\mathrm{verify}(x,y)]$. Sparse and terminal `[deepseek-r1 §2.2]`.
- **GRPO (Group Relative Policy Optimization)** — Critic-free PPO variant: replaces the value model with a group-mean reward baseline. Sample $G$ trajectories per prompt, normalize rewards within the group, use the standardized advantage in the PPO loss. Used by DeepSeek-R1, Qwen 3, Tülu 3 RLVR `[shao2024 §4.1; deepseek-r1 §2.1 Eq.1]`.
- **Group-normalized advantage** — In GRPO, $\tilde{A}_i = (R_i - \mathrm{mean}(\{R_j\}))/\mathrm{std}(\{R_j\})$, broadcast to every token of trajectory $o_i$. Replaces GAE-derived per-token advantages `[shao2024 §4.1; deepseek-r1 §2.1 Eq.3]`.
- **Format reward** — In RLVR, a small constant reward for trajectories that conform to a structural template (e.g., emit a `<think>...</think>` block). Keeps the chain-of-thought pathway alive during early training `[deepseek-r1 §2.2]`.
- **DAPO (Decoupled clip and dynamic sAmpling Policy Optimization)** — GRPO improvement with four targeted modifications: Clip-Higher (asymmetric PPO clip), Dynamic Sampling (reject zero-gradient groups), Token-Level Loss (equal-weight tokens regardless of trajectory length), Overlong Reward Shaping (soft penalty for length-truncated trajectories) `[dapo2025 §2]`.
- **Clip-Higher** — DAPO's asymmetric PPO clip $[1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}]$ with $\epsilon_{\text{high}} > \epsilon_{\text{low}}$. Allows larger upward updates on under-probable correct tokens `[dapo2025 §2]`.
- **Dynamic Sampling (DAPO)** — Reject groups where all $G$ trajectories share the same reward, since the GRPO advantage is then zero and the gradient is wasted. Re-sample until each group contains both $R=0$ and $R=1$ trajectories `[dapo2025 §2]`.
- **VAPO (Value-Augmented PPO)** — RL variant that re-introduces a learned value function on top of GRPO's design, with shared backbone and value-loss-clipping. Reportedly outperforms DAPO on long-CoT tasks `[vapo-2025]`.
- **VinePPO** — PPO variant using per-prefix Monte Carlo re-simulation as value estimate. Drops the value network. 3× wall-clock speedup over PPO on MATH/GSM8K `[vineppo-2024]`.
- **REINFORCE++** — A family of REINFORCE-class methods that drop PPO's clip but keep the GRPO group-mean baseline and KL anchor. Whether clip helps in this regime is contested.
- **R1-Zero** — DeepSeek-R1-Zero, the version of R1 trained with **no SFT cold-start** — pure GRPO RL applied directly to the DeepSeek-V3 base. Achieves AIME 2024 pass@1 of 77.9% `[deepseek-r1 §2.3]`.
- **Aha moment (R1-Zero)** — A spike in the model's use of reflection words like "wait" during training, taken as evidence of self-reorganization of reasoning structure `[deepseek-r1 §2.3]`.
- **Pass@1 / Pass@K** — Pass@K is the probability that at least one of $K$ independent samples from the model passes the verifier. Pass@1 is the single-sample success rate. RLVR primarily improves pass@1 (sampling efficiency) `[rlvr-limits-2025]`.
- **RLVR sampling-efficiency interpretation** — Empirical finding (NeurIPS 2025 oral, Yue et al.) that RLVR improves pass@1 on problems the base model can already solve at higher K, but does not expand the set of problems solvable at any K. The capability ceiling is set by pre-training `[rlvr-limits-2025]`.

## SFT-specific

- **Rejection sampling** — Llama-3 SFT quality gate: sample $N$ candidates from the model, score by reward model, keep top-$k$, SFT on those. Used as a training-loop stage in modern multi-round pipelines `[meta-llama3]`.
- **MAGPIE** — Alignment data synthesis from scratch: prompt aligned LLMs with only their instruction-template prefix (no question), have them fabricate the question first, then sample the answer. Near-zero-cost SFT data generation `[magpie-2024]`.
- **Cold-start SFT (R1 sense)** — A small SFT stage on long-CoT examples preceding the main RL stage. Provides a reasoning-format and language-quality regularizer for the subsequent RL. Distinct from full SFT pipelines `[deepseek-r1 §3]`.
- **Reasoning-distillation SFT** — Training small models (1.5B–32B) on reasoning trajectories generated by a strong reasoning teacher (R1, o1) via pure SFT with no RL. The 800K-sample DeepSeek-R1 distillation set is the canonical example `[deepseek-r1 §4]`.
- **Thinking / non-thinking modes (Qwen 3 sense)** — Two distinct response styles trained with separate SFT data and selectable at inference via system prompt: "thinking" emits a `<think>...</think>` block, "non-thinking" produces direct answers `[qwen3]`.

## Pipeline conventions

- **Tülu 3 pipeline** — The canonical 4-stage open post-training reference: data curation → SFT → DPO → RLVR. Tülu 3 405B (Jan 2025) scales RLVR to 405B parameters `[tulu3]`.
- **Iterative SFT-DPO** — Multi-round pipeline used by Llama-3 and Tülu 3: SFT, then sample candidates, run rejection sampling, run DPO, repeat with the new policy as starting point `[meta-llama3]`.

### reasoning

# Glossary additions — reasoning/ area

## Chain-of-thought and prompting

- **Chain-of-thought (CoT)** — A prompting and decoding pattern in which the model produces an intermediate sequence of reasoning steps $z$ before emitting a final answer $y$. In few-shot CoT (Wei 2022), the prompt contains exemplars with hand-written reasoning traces; in zero-shot CoT (Kojima 2022), the trigger phrase "Let's think step by step" elicits the same behaviour without exemplars. `[wei2022-cot §1; kojima2022 §3]`
- **Few-shot CoT** — The Wei 2022 exemplar-based variant. The prompt contains $K$ demonstrations $(x_i, z_i, y_i)$ before the query. CoT gain over direct prompting is **emergent at scale** — sub-100B models often do not benefit. `[wei2022-cot §3.3]`
- **Zero-shot CoT** — The Kojima 2022 trigger-phrase variant. Two-stage decode: reasoning extraction + answer extraction. Recovers a substantial fraction of few-shot CoT gain at instruction-tuned scale without exemplars. `[kojima2022 §3]`
- **Self-consistency** — Sample $N$ reasoning traces at temperature $T > 0$, majority-vote over extracted answers. The simplest test-time-compute strategy. Gains: GSM8K +17.9%, SVAMP +11.0% over greedy CoT. `[wang2022-self-consistency §3]`
- **Trained CoT** — The behaviour of a reasoning-trained model (o1, R1, QwQ) that emits long reasoning traces unconditionally, without prompt scaffolding. Distinct from prompted CoT in that the trace is the model's natural output distribution. `[deepseek-r1 §2]`
- **CoT faithfulness** — The extent to which the trace $z$ causally drives the answer $y$ versus being post-hoc rationalisation. Measured by counterfactual perturbation. Thinking models report rates ~0.04–0.14% post-hoc; non-thinking models ~7–13%. `[chen2026-faithfulness-scaling, arXiv 2601.06423; chen2025-faithcot-bench, arXiv 2510.04040]`

## Test-time compute

- **Test-time compute (TTC)** — The inference-side compute spent on a single problem instance to improve answer quality, beyond a single greedy forward pass. Recognised in 2024 as a co-equal scaling axis with training compute. `[snell2024 §1]`
- **Compute-optimal TTC** — The strategy maximising expected verifier score at fixed inference compute $C$. Snell et al. 2024 show the optimal mix is problem-difficulty-dependent: easy → short single trace; medium → sampling-based; hard → search. `[snell2024 §3, §5]`
- **Sampling-based TTC** — Draw $N$ independent traces, aggregate. Three rules: self-consistency (majority vote), best-of-$N$ + ORM, best-of-$N$ + PRM. Linear in $N$. `[wang2022-self-consistency; lightman2023-prm800k §4]`
- **Search-based TTC** — Tree expansion over partial reasoning prefixes, directed by a learned value function. Beam, BFS, MCTS variants. See `kb/notes/reasoning/inference-time-search.md`. `[snell2024 §4; rstar-math2025 §3]`
- **Sequential-compute TTC / extended thinking** — Hold $N = 1$ but emit a much longer single trace. The o1/R1/QwQ paradigm. Compute scales super-linearly in trace length due to KV-cache re-attention. `[deepseek-r1 §2.3]`
- **Budget forcing** — The s1 2025 mechanism. When the model emits its end-of-thinking token before the budget is reached, suppress it and append the continuation token "Wait" to force more reasoning; hard-truncate to enforce shorter budgets. `[s1-2025 §3.2]`
- **Thinking budget** — Productised TTC control: API-level integer (Gemini 2.5 `thinkingBudget`), effort level (OpenAI o-series), or token cap (Anthropic thinking block). `[gemini2-5 §3]`

## Process supervision and reward models

- **Outcome Reward Model (ORM)** — A scoring function $R^O_\phi: (x, z, y) \to [0, 1]$ trained on (prompt, complete solution, outcome correctness) triples. Sparse signal — one label per trace. `[lightman2023-prm800k §2.1]`
- **Process Reward Model (PRM)** — A scoring function $R^P_\phi: (x, s_{1:t}) \to [0, 1]$ trained on per-step correctness labels. Dense signal — one label per step. Outperforms ORM at matched best-of-$N$ on MATH. `[lightman2023-prm800k §2.2, §4]`
- **PRM800K** — OpenAI's released dataset of $\approx 800{,}000$ human step-correctness labels on GPT-4-generated MATH solutions. The foundational reference dataset for PRM training. `[lightman2023-prm800k §3]`
- **Aggregation rule (PRM)** — How per-step PRM scores are combined to a trace score. Min, product, last-step. **Min** is the strongest aggregator at PRM800K-grade. `[lightman2023-prm800k §4.2]`
- **discriminative PRM** — A classifier-style PRM: outputs a probability of step correctness in one feed-forward pass. Cheap to evaluate; classical Math-Shepherd / OpenAI PRM lineage. `[lightman2023-prm800k §2.2]`
- **Generative PRM (ThinkPRM)** — A small reasoning-LLM verifier that emits its own short CoT evaluating each step before producing a verdict. Trained on 1% of PRM800K, beats full-PRM800K discriminative PRMs on ProcessBench / AIME 2024. `[thinkprm2025 §3]`
- **Process Preference Model (PPM)** — rStar-Math's variant: trained on preferences over MCTS-discovered step pairs, rather than absolute step labels. Avoids naïve step-score annotation. `[rstar-math2025 §3]`
- **ProcessBench** — Standard benchmark for PRM step-error detection (identify the first incorrect step in a partial trace). `[arXiv 2412.06559]`
- **Best-of-$N$ + verifier** — Sample $N$ traces, score each with a verifier (ORM, PRM, or self-consistency baseline), select the highest-scored. Canonical sampling-based TTC. `[lightman2023-prm800k §4]`

## Reasoning training (RL with verifiable rewards)

- **RLVR (RL with Verifiable Rewards)** — RL post-training where the reward is computed mechanically from the emitted answer (regex match for math, unit-test execution for code, proof-checker for formal proofs), not from a learned reward model. The defining feature of post-2024 reasoning training. `[deepseek-r1 §2.2.1]`
- **GRPO (Group Relative Policy Optimisation)** — Critic-free PPO variant (Shao 2024). Per-prompt group of $G$ samples; advantage is each sample's reward minus group mean, divided by group std. Eliminates the value-network requirement. `[shao2024 §4.1; deepseek-r1 §2.1]`
- **DAPO** — ByteDance's GRPO variant. Four modifications: clip-higher (asymmetric clip), dynamic sampling (drop zero-variance groups), token-level loss (vs sequence-level), overlong reward shaping. AIME 2024 50pts on Qwen2.5-32B. `[dapo2025 §2]`
- **Cold-start SFT** — A small-scale ($\sim 10^4$ examples) SFT pass on long-CoT exemplars, run before RL to fix readability problems of pure-RL outputs (R1-Zero). Does not change asymptotic capability but improves trace coherence. `[deepseek-r1 §2.2.4]`
- **rejection-sampling SFT** — Generate candidate solutions with the current policy, filter by verifier (and optional quality filter), SFT on the survivors. Used as Stage 3 of the R1 pipeline. `[deepseek-r1 §2.3.3]`
- **Distillation (reasoning)** — Use a strong teacher's RL-trained traces as SFT data for smaller students. R1 demonstrates this beats running RL directly on a small model. `[deepseek-r1 §3.2]`
- **Capability ceiling (RLVR)** — The empirical finding that RLVR improves $\mathrm{pass}@1$ on problems the base model could already solve at $\mathrm{pass}@k$ for some $k$, but does not expand the set of solvable problems. Set at pre-training. `[rlvr-limits-2025 §3]`
- **Reward / process hacking** — Failure mode in which the policy learns to exploit imperfections in the verifier (or PRM) rather than improving on the underlying task. `[deepseek-r1 §2.2.4; arXiv 2505.00551]`
- **Hybrid think / non-think model** — A single model that toggles between long-CoT and short-answer modes via system-prompt instruction. Productisation step of reasoning training. `[qwen3 §4; deepseek V3.1+ release notes]`

## Inference-time search

- **Inference-time search** — TTC family that constructs and explores a tree (or DAG) of partial reasoning prefixes, using a verifier or value function to direct expansion. Beam / BFS / MCTS variants. `[rstar-math2025 §3]`
- **reasoning-MCTS** — Monte Carlo Tree Search where actions are next-step reasoning generations and the value function is a PRM. Selection-expansion-simulation-backup loop, UCT-style exploration. `[rstar-math2025 §3]`
- **Generator-discriminator mutual verification** — rStar-Math's two-SLM design: policy SLM generates candidate next-steps, reward / discriminator SLM scores them. Both small (~7B); together rival o1-mini on MATH. `[rstar-math2025 §3]`
- **ReST-MCTS*** — Self-training loop using MCTS to generate correct trajectories, then SFT on those, then update the value function. Structurally analogous to AlphaZero self-play. `[rest-mcts-2024, arXiv 2406.03816]`
- **Lookahead beam** — Beam search variant: from each beam candidate, perform short rollouts and score the rolled-out outcomes, then prune. The form Snell 2024 uses for compute-optimal mix experiments. `[snell2024 §4]`
- **MCTS-RAG** — MCTS extended over retrieval decisions in addition to reasoning steps. Bridges search-based TTC and retrieval-augmented generation. `[mcts-rag-2025, arXiv 2503.20757]`

### scaling

## Scaling laws and compute-optimal training

- **Scaling law** — An empirical or theoretical functional relationship between a measure of model performance (typically cross-entropy loss) and one or more "scale" inputs: parameters $N$, training tokens $D$, training compute $C$, or test-time compute $N_{\text{test}}$. Canonical reference: `kaplan2020`.
- **Compute (training)** — Total floating-point operations spent in pre-training, denoted $C$ or $C_{\text{train}}$. The standard accounting per-token is $C \approx 6 N D$ for dense Transformers `[kaplan2020 §2.1; kb/excerpts/kaplan2020#sec-2-1-definitions]`.
- **PF-day** — Petaflop-day, $10^{15} \times 86{,}400 \approx 8.64 \times 10^{19}$ floating-point operations. Unit used in Kaplan 2020 for compute reporting.
- **Power law (in scaling)** — A relationship of the form $L(X) \propto (X_c / X)^{\alpha_X}$ with exponent $\alpha_X$. Holds approximately for LLM loss against $N$, $D$, and $C$ across many orders of magnitude `[kaplan2020 §1.2; kb/excerpts/kaplan2020#sec-1-2-equations]`.
- **Critical batch size $B_{\text{crit}}$** — The largest training batch (in tokens) at which scaling further is still time-efficient. Itself a power law in the loss: $B_{\text{crit}}(L) = B_\ast / L^{1/\alpha_B}$ with $B_\ast \approx 2 \times 10^8$ tokens, $\alpha_B \approx 0.21$ `[kaplan2020 §5.1; kb/excerpts/kaplan2020#sec-1-1-bullets]`.
- **IsoFLOP profile** — The plot of training loss vs. parameter count $N$ at a fixed training-compute budget $C$. The minimum across $N$ at each $C$ defines the compute-optimal frontier `[hoffmann2022-chinchilla §3.2; kb/excerpts/hoffmann2022-chinchilla#sec-3-2]`.
- **Compute-optimal training** — Training a model with $(N, D)$ chosen to minimize loss at fixed $C \approx 6ND$. Chinchilla's headline rule: at the optimum, $D \approx 20 N$ for a wide range of $N$ `[hoffmann2022-chinchilla §3 Table 3; kb/excerpts/hoffmann2022-chinchilla#sec-3-table-3]`.
- **Tokens-per-parameter ratio** — $D / N$. Chinchilla-optimal value is approximately 20. Industry practice has shifted to over-trained ratios (Llama 3 8B ≈ 1875) trading pre-training compute for inference cost. See `scaling/scaling-frontier`.
- **Chinchilla parametric loss** — The functional form $\hat{L}(N, D) = E + A/N^\alpha + B/D^\beta$ with five fitted constants. $E$ ≈ entropy of natural text on the eval distribution `[hoffmann2022-chinchilla §3.3 Eq.(2); kb/excerpts/hoffmann2022-chinchilla#sec-3-3]`.
- **Kaplan compute-allocation rule** — The (now superseded) prescription $N \propto C^{0.73}$, $D \propto C^{0.27}$ from `[kaplan2020 §1.2; kb/excerpts/kaplan2020#sec-1-2-compute-allocation]`. Overturned by Chinchilla's roughly equal-scaling result. Kaplan's flaw was an LR-schedule horizon held fixed across model sizes.
- **Chinchilla trap** — Community term for the practical problem that a Chinchilla-compute-optimal model is often too large to *serve* economically. Models destined for high-volume deployment over-train (small $N$, large $D$) to minimize inference cost. `[FORUM-SIGNAL]` — widespread but not formally derived.
- **Joint $L(N, D)$** — Kaplan's combined form: $L(N, D) = [(N_c/N)^{\alpha_N/\alpha_D} + D_c/D]^{\alpha_D}$ `[kaplan2020 §1.2 Eq.(1.5); kb/excerpts/kaplan2020#sec-1-2-joint]`. Implies sublinear $D \propto N^{0.74}$ to avoid overfitting penalty (in Kaplan's setup).
- **Non-embedding parameters** — Parameter count excluding token-embedding and positional-embedding tables. Kaplan's $N$ uses this convention; Chinchilla and most modern papers use total parameters `[kaplan2020 §1.3; kb/excerpts/kaplan2020#sec-1-3-notation]`. The difference is ≈ 0.5%–5% at modern $d_{\text{model}}$ + vocab sizes.

## Inference-time compute scaling

- **Test-time compute / inference-time compute** — Compute spent during model inference (forward passes, search, verifier scoring) on a single query. Distinct from training compute. Key paper: `snell2024`.
- **Test-time compute-optimal scaling strategy** — The argmax over test-time hyperparameters $\theta$ of expected output correctness, at a fixed test-time compute budget $N$, conditioned on a prompt $q$: $\theta^\ast_{q, y^\ast(q)}(N) = \arg\max_\theta \mathbb{E}_{y \sim \mathrm{Target}(\theta, N, q)}[\mathbb{1}_{y = y^\ast(q)}]$ `[snell2024 §3.1 Eq.(1); kb/excerpts/snell2024#sec-3-1]`.
- **Proposer / verifier decomposition** — The unified framing of test-time compute strategies as either (1) modifying the model's proposal distribution (CoT, revision, self-critique) or (2) applying a verifier (best-of-N, beam search, tree search) to selected candidates `[snell2024 §2; kb/excerpts/snell2024#sec-2-proposer-verifier]`.
- **Best-of-N (BoN)** — Generate $N$ independent candidate solutions, select the highest-scoring under an ORM. Baseline test-time-compute strategy. Snell's "best-of-N weighted" variant aggregates per-step PRM scores instead.
- **Beam search (in test-time-compute context)** — Search over per-step PRM scores: at each step keep top $N/M$ candidates, expand each by $M$ proposals, repeat. $N$ = beam width, $M$ = expansion factor `[snell2024 §5.2; kb/excerpts/snell2024#sec-5-2-search]`.
- **Lookahead search** — Beam search variant where each candidate's score includes simulated rollout of further steps before being kept/dropped. Costlier per beam, more discriminative.
- **Question difficulty (Snell et al.)** — A 5-quantile bin of a base LLM's pass@1 rate (estimated from 2048 samples) on a test-set prompt. Used as a sufficient statistic for picking the compute-optimal test-time strategy `[snell2024 §3.2; kb/excerpts/snell2024#sec-3-2-difficulty]`.
- **Oracle vs model-predicted difficulty** — Two operational forms of question-difficulty estimation: oracle uses ground-truth correctness (test-time only as research proxy); model-predicted uses a verifier's averaged score (deployable) `[snell2024 §3.2; kb/excerpts/snell2024#sec-3-2-difficulty]`.
- **Thinking budget / `thinkingBudget`** — A productionized API parameter (Gemini 2.5, OpenAI o-series reasoning-effort levels, Anthropic extended-thinking) that caps the test-time-compute spend per request. Operationalizes Snell's compute-optimal strategy `[gemini2-5; kb/index/papers.json#gemini2-5]`.
- **Reasoning model** — An LLM trained (typically with RL on verifiable rewards) to produce long-form chain-of-thought before its final answer. o1, DeepSeek-R1, Qwen3-thinking-mode, Gemini 2.5-thinking-mode are exemplars `[deepseek-r1; kb/index/papers.json#deepseek-r1]`.
- **Self-consistency** — Sample $N$ chain-of-thoughts independently at $T > 0$; majority-vote the final answers. Wang et al. 2022. Cheap baseline that requires no verifier.

## Maximal Update Parametrization (μP) and HP transfer

- **Maximal Update Parametrization (μP)** — The unique-up-to-redundancy parametrization (init scales × LR scalings × parameter multipliers) under which optimal hyperparameters are width-invariant. The infinite-width limit has $\Theta(1)$-magnitude feature updates, in contrast to the kernel/NTK regime of standard parametrization `[yang2022-mup §2; kb/excerpts/yang2022-mup#sec-2-clt]`.
- **Standard Parametrization (SP)** — The default parametrization in PyTorch and most frameworks: init variance $1/\text{fan\_in}$, uniform LR scaling. Does NOT admit HP transfer across width — optimal LR shifts by ~1 order of magnitude as width grows from 256 to 8192 `[yang2022-mup §3; kb/excerpts/yang2022-mup#sec-3-mlp-sp]`.
- **μTransfer** — The HP-tuning procedure: parametrize target in μP, tune HPs on a small proxy model (typically 1/16–1/32 of target width), copy them zero-shot to target without retuning. Tuning cost ≈ 7% of one full pretraining run for GPT-3 6.7B `[yang2022-mup abstract; kb/excerpts/yang2022-mup#abstract]`.
- **abc-parametrization** — Yang's framework characterizing a parametrization by three width-dependent functions per parameter: $a_W$ (parameter multiplier), $b_W$ (init scale), $c_W$ (LR scaling). μP is a specific point in abc-space; SP is another. See `yang2022-mup` Appendices.
- **Proxy model / target model** — In μTransfer, the small model used for HP search (proxy) and the full-scale model receiving the transferred HPs (target). Same depth, vocab, data, tokenizer; only width differs `[yang2022-mup §1; kb/excerpts/yang2022-mup#sec-1-contributions]`.
- **μP $1/d$ attention** — Definition 4.1 of Yang et al. 2022: replace Vaswani's $1/\sqrt{d_k}$ softmax scaling with $1/d_k$. Correct training-time rescaling under μP because $q$ and $k$ become correlated after one optimization step, making $\mathrm{Var}(q^\top k) = \Theta(d)$ `[yang2022-mup §4 Definition 4.1; kb/excerpts/yang2022-mup#sec-4-definition-4-1]`.
- **u-μP (Unit-Scaled μP)** — μP combined with unit-scaling so every layer has predictable $\Theta(1)$-magnitude activations. Enables native FP8 training without dynamic per-tensor rescaling `[u-mup2024; kb/index/papers.json#u-mup2024]`.
- **Depth-μP** — Tensor Programs VI extension of μP to the depth axis: principled depth-scaling HP transfer for pre-LN Transformers. ICLR 2024.
- **MetaP** — Meta's name for their cross-scale HP-transfer approach in the Llama 4 tech report `[meta-llama4]`. Details thin in the report; community working assumption is "μP-inspired."
- **Feature-learning vs kernel/NTK regime** — Two qualitatively different infinite-width limits: in the kernel/NTK regime, weights barely move during training (effectively a fixed kernel machine); in the feature-learning regime, weights update by $\Theta(1)$ in the appropriate norm. μP is the unique parametrization (up to redundancy) in the latter regime. See Yang & Hu (Tensor Programs IV, 2021).

### training

## Optimization

- **AdamW** — Adam with decoupled weight decay. Maintains per-parameter first and second moments $\mathbf{m}_t, \mathbf{v}_t$; update is $-\eta \cdot \mathbf{m}_t / (\sqrt{\mathbf{v}_t}+\varepsilon) - \eta\lambda\mathbf{w}_t$. The default LLM pre-training optimizer 2020-2024. `[hoffmann2022-chinchilla §3]`
- **Muon** — "MomentUm Orthogonalized via Newton-Schulz." Replaces Adam's per-parameter rescaling with a *per-shape* matrix-orthogonalization on momentum buffer $\mathbf{B}_t = \mu \mathbf{B}_{t-1} + \mathbf{G}_t$, then sets $\mathbf{O}_t = \text{Newton-Schulz}(\mathbf{B}_t)$ before the parameter update. Operates only on 2D weight matrices; uses AdamW for 1D parameters and embeddings. `[muon-moonlight2025 §2.1, eq.1; kb/excerpts/muon-moonlight2025#sec-2-1-update-rule]`
- **Newton-Schulz iteration** — Iterative polynomial $\mathbf{X}_k = a\mathbf{X}_{k-1} + b(\mathbf{X}_{k-1}\mathbf{X}_{k-1}^\top)\mathbf{X}_{k-1} + c(\mathbf{X}_{k-1}\mathbf{X}_{k-1}^\top)^2 \mathbf{X}_{k-1}$ that converges all singular values of $\mathbf{X}_0$ to 1, producing $\mathbf{X}_0(\mathbf{X}_0^\top \mathbf{X}_0)^{-1/2}$. Muon uses 5 iterations with coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$. `[muon-moonlight2025 §2.1, eq.2; kb/excerpts/muon-moonlight2025#sec-2-1-ns]`
- **Per-shape RMS scaling (Muon)** — Multiplier $0.2 \cdot \sqrt{\max(A, B)}$ applied to the orthogonalized update for an $A\times B$ weight matrix, equalizing update RMS across heterogeneous matrix shapes. `[muon-moonlight2025 §2.2, eq.4; kb/excerpts/muon-moonlight2025#sec-2-2-rms]`
- **Schatten-$p$ norm** — Matrix norm $\|\mathbf{M}\|_p^{(s)} = (\sum_i \sigma_i^p)^{1/p}$ on singular values. Muon's update is the steepest-descent direction under the Schatten-$\infty$ (spectral) norm constraint, which is why orthogonalization is the right pre-processing step. `[muon-moonlight2025 §2.1]`
- **Lion** — "Evolved sign momentum." Lighter optimizer using only sign of momentum: $\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \eta \cdot \text{sign}(\mu \mathbf{m}_{t-1} + (1-\mu)\mathbf{g}_t) - \eta\lambda\mathbf{w}_{t-1}$. Halves optimizer state vs. Adam. (Chen et al. 2023)
- **Sophia** — "Second-order clipped stochastic optimization." Diagonal-Hessian preconditioning with clipping. Targets faster convergence in early pre-training. (Liu et al. 2023, arXiv 2305.14342)
- **Cosine schedule** — Learning-rate decay $\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})(1 + \cos(\pi t / T))$, peaking after warmup and decaying to $\eta_{\min}$ at training end. The de-facto LLM schedule 2020-2024.
- **WSD schedule (Warmup-Stable-Decay)** — Three-phase LR: linear warmup, then constant at $\eta_{\max}$ for the bulk of training, then short cooldown to ~$0.1\eta_{\max}$. Used by Moonlight (5.7T-token Muon run); avoids cosine's commitment to a specific budget. `[muon-moonlight2025 §3.3]`

## Pre-training data

- **Common Crawl (CC)** — Open monthly web crawl run by the Common Crawl Foundation since 2008. Source corpus for FineWeb (96 snapshots), C4, RefinedWeb, and most open web datasets. WARC format with extracted WET (text) variant.
- **trafilatura** — Python text-extraction library; extracts main article text from raw HTML. FineWeb's choice over CC's built-in WET, yielding ~25 % more text. `[fineweb2024 §3.2]`
- **MinHash deduplication** — Locality-sensitive hashing for near-duplicate detection: $n$-gram-shingle the document, sample $H$ hash-of-min-hash signatures arranged as $B$ buckets of $R$ hashes ($H = BR$); two documents collide if any bucket matches; threshold Jaccard $\approx 1 - (1-r^R)^B$ where $r$ is true Jaccard. FineWeb uses 5-grams, $H=112$, $B=14$, $R=8$. `[fineweb2024 §3.4; kb/excerpts/fineweb2024#sec-3-4-minhash]`
- **Individual-snapshot dedup** — FineWeb's choice to MinHash within each CC dump and not across dumps. Counterintuitively beats global dedup; the global version over-removes high-quality multi-snapshot text. `[fineweb2024 §3.4; kb/excerpts/fineweb2024#sec-3-4-global]`
- **Ablation model (FineWeb sense)** — Small reference model (1.71 B params, ~28 B tokens) trained on a candidate corpus to score it. Decisions are made by signed-difference of downstream-task accuracy across candidates. `[fineweb2024 §3.1]`
- **C4 line-level filters** — Per-line heuristics from Raffel et al. 2019 (T5/C4): remove lines lacking terminal punctuation, lines with "javascript", policy/cookie boilerplate, etc. FineWeb adopts these *minus* the curly-brace rule (which would strip code). `[fineweb2024 §3.3]`
- **Custom heuristic filters (FineWeb)** — Three quantile-derived rules: lines-ending-with-terminal-punctuation $\leq 0.12$, char-duplication $\geq 0.01$, short-line-fraction $\geq 0.67$. Lift HellaSwag ~1 pt at ablation scale. `[fineweb2024 §3.6]`
- **FineWeb-Edu** — 1.3 T-token quality-classifier-filtered subset of FineWeb. Llama-3-70B-Instruct annotates 500 K samples 0–5 for "educational value"; classifier learns the score; threshold ≥3 keeps the subset. ~6 K H100-hours of annotation cost. `[fineweb2024 §4]`
- **Data Mixing Law (Eq. 1)** — $L_i(\mathbf{r}) = c_i + k_i \exp(\sum_j t_{ij} r_j)$. Validation loss on domain $i$ as an exponential of a linear combination of training-mixture proportions $r_j$. Coefficients $(c_i, k_i, t_{ij})$ are fit per target domain. `[data-mixing-laws-2024 §1, eq.1]`
- **Nested scaling laws (mixing)** — Pipeline that fits (training-step power law) inside (model-size power law) inside (mixing-law exponential), enabling small-model small-step experiments to predict large-model end-of-training loss as a function of mixture. `[data-mixing-laws-2024 §1]`
- **Compatibility / symmetry principles (mixing law)** — The two structural constraints that pin down Eq. 1 from a candidate-function space: reduces to two-domain log-linear when $M=2$; invariant under re-labeling of domains. `[data-mixing-laws-2024 §3.2]`
- **Transfer affinity coefficient $t_{ij}$** — Fitted parameter in the data mixing law; sign indicates whether domain-$j$ training helps ($t_{ij}<0$) or hurts ($t_{ij}>0$) validation on domain $i$. `[data-mixing-laws-2024 §3.2]`

## Synthetic data & distillation

- **Synthetic pre-training data** — Documents generated by a teacher LLM (or LLM pipeline) for use in a student's pre-training mix. Phi-4 generated ~400 B tokens via 50 pipelines, ≈40 % of its 9.8 T-token training mix. `[phi4 §1, §2]`
- **Phi-4 50-pipeline architecture** — Multi-stage pipeline: seed curation → rewrite/augment → self-revision → quality filter. 50 distinct pipelines tuned for different content types (STEM, code, reasoning chains). `[phi4 §2.2; kb/excerpts/phi4#sec-2-2-pipelines]`
- **R1 distillation set** — DeepSeek-R1's 800 K SFT samples (600 K reasoning + 200 K non-reasoning) generated by R1 and used to fine-tune Qwen-2.5/Llama-3 students at 1.5/7/14/32/70 B. Transfers reasoning *capability* to architectures that cannot reach the same level via RLVR alone. `[deepseek-r1 §3, §4; kb/excerpts/deepseek-r1#sec-3-pipeline]`
- **Decontamination (Phi-4 sense)** — Held-out test sets, especially November 2024 AMC math contests and unleaked HumanEval+, used to verify the synthetic data did not leak benchmark answers via the teacher. `[phi4 §1.1]`
- **Spoonfeeding (Phi-4 intuition)** — `[INTUITION]` Synthetic data presents *concept-then-application-then-explanation* as a self-contained unit, vs. organic web data which scatters those parts across many documents. Enables high-density learning per token. `[phi4 §1, §2.1]`

## Distributed training (axes)

- **Data Parallel (DP)** — Same parameters on each worker; each worker processes a shard of the global batch; gradients all-reduced before optimizer step.
- **FSDP** — "Fully Sharded Data Parallel." Native PyTorch ZeRO implementation. Parameters and optimizer state sharded across DP workers; full param-set materialized via all-gather just before each forward/backward. `[torchtitan2024 §2.1.2]`
- **FSDP2** — Per-parameter `DTensor`-sharded successor to FSDP1. ~7 % per-GPU memory reduction and ~1.5 % throughput improvement vs. FSDP1's flat-parameter design. `[torchtitan2024 §2.1.2; kb/excerpts/torchtitan2024#sec-2-1-2-fsdp2]`
- **ZeRO-1 / ZeRO-2 / ZeRO-3** — Optimizer-state / gradient / parameter sharding levels of Rajbhandari et al.'s Zero-Redundancy Optimizer. ZeRO-1 shards only optimizer state; ZeRO-3 shards everything (FSDP's default).
- **Tensor Parallel (TP)** — Within-layer matmul split (column-parallel `Y = XA`, row-parallel `Z = YB`). Inserts one all-reduce per matmul. Each worker holds a $1/t$ slice of weights. `[torchtitan2024 §2.1.3]`
- **Sequence Parallel (SP)** — Companion to TP: shards normalization and dropout layers along the sequence dimension to reduce activation memory. `[torchtitan2024 §2.1.3]`
- **Pipeline Parallel (PP)** — Stage-level layer split. Microbatches flow stage-to-stage via P2P send/recv. Tradeoff: per-GPU param drops by $1/p$ but in-flight activations grow.
- **1F1B (one-forward-one-backward)** — Classic PP schedule. Each stage alternates forward-of-microbatch-$i$ with backward-of-microbatch-$j$ to bound peak activation memory.
- **Pipeline bubble** — Fraction of PP step time where some stage is idle, waiting for upstream/downstream. 1F1B bubble is $(p-1)(F+B)$ stage-times. `[deepseek-v3 §3.2.1, table]`
- **ZeroBubble (ZB1P)** — PP schedule that splits backward into "backward-for-input" ($B$) and "backward-for-weights" ($W$); reorders $W$ into the bubble. Reduces bubble to $(p-1)(F+B-2W)$. `[deepseek-v3 §3.2.1]`
- **DualPipe** — DeepSeek-V3's bidirectional PP. Microbatches feed from both ends of the pipeline simultaneously; each chunk split into 4 stages (attention / dispatch / MLP / combine), backward further split via ZB; achieves $(\tfrac{p}{2}-1)(F\&B+B-3W)$ bubble at $2\times$ parameter memory and $+1$ activation. `[deepseek-v3 §3.2.1; kb/excerpts/deepseek-v3-training#sec-3-2-1-dualpipe]`
- **Context Parallel (CP)** — Token-position sharding of the sequence dimension. Communication-pattern resembles ring attention. Enables 262 144-token training of Llama 3.1 8B on 8× H100. `[torchtitan2024 §2.1.5]`
- **Expert Parallel (EP)** — MoE-specific sharding: each worker hosts a subset of the experts. Forward pass requires an all-to-all dispatch (each token to its top-$k$ experts) and a combine all-to-all. DeepSeek-V3 uses $e=64$ across 8 nodes. `[deepseek-v3 §3.2; kb/excerpts/deepseek-v3-training#sec-3-2-parallelism]`
- **All-to-all dispatch / combine** — The two collectives bracketing every MoE forward. Dispatch routes tokens to expert hosts; combine returns expert outputs to original token positions. The dominant communication cost in MoE training.
- **AsyncTP (Asynchronous Tensor Parallel)** — Fractionalizes TP matmuls into chunks and overlaps each chunk's communication with the previous chunk's compute via intra-node `SymmetricMemory` shared buffers. `[torchtitan2024 §2.2.3]`
- **DTensor / DeviceMesh (PyTorch)** — Distributed tensor abstraction; weights are stored as DTensor with a placement spec on a $d \times t \times p \times c \times e$ device mesh. The composability layer for 4D/5D parallelism. `[torchtitan2024 §2.1.1]`
- **4-node-cap rule (DeepSeek-V3 routing)** — Each token's MoE dispatch is limited to at most 4 nodes, exploiting the 3.2× NVLink/IB bandwidth ratio: one IB hop per (token, target-node) followed by intra-node NVLink forwarding. `[deepseek-v3 §3.2.2; kb/excerpts/deepseek-v3-training#sec-3-2-2-allreduce]`
- **MFU (Model FLOP Utilization)** — Realized training throughput as a fraction of peak hardware FLOPS. Standard parallelism-plan quality metric.

## Mixed precision & stability

- **BF16 (bfloat16)** — Brain-float. 8-bit exponent, 7-bit mantissa (vs. FP16's 5-exp/10-mant). Same dynamic range as FP32; lower precision. The 2020-2024 default forward/backward dtype.
- **FP8 E4M3** — 4-bit exponent, 3-bit mantissa. Higher precision, lower dynamic range than E5M2. DeepSeek-V3 uses E4M3 on *all* tensors via fine-grained scaling. `[deepseek-v3 §3.3.2; kb/excerpts/deepseek-v3-training#sec-3-3-2-mantissa]`
- **FP8 E5M2** — 5-bit exponent, 2-bit mantissa. Higher dynamic range, lower precision. NVIDIA's recommended dtype for `Dgrad`/`Wgrad` in their hybrid recipe; DeepSeek-V3 *avoids* it.
- **Tile-wise quantization (1×128)** — DeepSeek-V3's activation scaling: one FP8 scale per (token, 128-channel block) tile. Adapts to per-token outliers. `[deepseek-v3 §3.3.2; kb/excerpts/deepseek-v3-training#sec-3-3-2-finegrained]`
- **Block-wise quantization (128×128)** — DeepSeek-V3's weight scaling: one FP8 scale per 128×128 weight block. `[deepseek-v3 §3.3.2]`
- **FP32 promotion to CUDA cores** — DeepSeek-V3's accumulation strategy: every $N_C=128$ MMA steps, copy partial sums from Tensor-Core (~14-bit) accumulator to FP32 registers on CUDA cores for full-precision summation. Recovers ~2 % systematic error in raw FP8 GEMM. `[deepseek-v3 §3.3.2; kb/excerpts/deepseek-v3-training#sec-3-3-2-accum]`
- **Precision pinning (DeepSeek-V3)** — Components held at BF16/FP32 even with FP8 elsewhere: embedding module, output head, MoE gating, normalization, attention. `[deepseek-v3 §3.3.1]`
- **Loss spike** — Sudden non-recoverable loss-curve excursion. Causes include gradient-norm explosion, attention-logit divergence, embedding-norm growth, and FP8 underflow on rare tokens. DeepSeek-V3 reports zero such spikes in 14.8 T tokens. `[deepseek-v3 §abstract]`

## Adaptation & merging

- **Continual / late-stage pre-training** — Continuing training of an already-trained model on a different (typically higher-quality or domain-specific) data mixture. Phi-4's "midtrain," DeepSeek-V3's 119 K-H800-hour context-extension stage, OLMo 2's Dolmino mix.
- **Context extension** — Sub-stage of late-stage pre-training that increases trained sequence length (e.g., 4 K → 32 K → 128 K) by adjusting RoPE scaling and continuing on long-context data. `[deepseek-v3 §1]`
- **PEFT (Parameter-Efficient Fine-Tuning)** — Family of fine-tuning methods that update only a small fraction of parameters: LoRA, DoRA, IA3, prefix-tuning. Pending excerpts.
- **Model merging** — Combining trained models via parameter-level operations (averaging, interpolation, sign-resolution, sparsification). Subtypes include weight averaging (Model Soups), task arithmetic, TIES, DARE, evolutionary merging. Pending excerpts.
- **Linear mode connectivity** — The empirical finding that fine-tunes of a shared base land in a connected loss basin: averaging their weights gives a model with loss similar to the endpoints. The leading explanation for why merging works at all. (Frankle, Dziugaite et al. 2020 — pending excerpt.)


## Phase 2 stub-fill additions

_Glossary fragments produced 2026-05-04 by the two stub-fill subagents (training+post+inference; interpretability+evaluation+alignment). The third (architecture) batch stalled before its glossary fragment was written; arch terms remain inline in their respective notes._

### training+post-training+inference

## Mixed precision and training stability

- **Master weights** — A high-precision (FP32) copy of the model
  parameters maintained alongside a low-precision (FP16/BF16/FP8)
  working copy used in the forward/backward pass. Optimizer updates
  are applied to the master copy and cast down to the working copy.
  `[micikevicius2017-mixed-precision §2;
  kb/excerpts/micikevicius2017-mixed-precision#sec-2]`.
- **Loss scaling** — Multiplying the loss by a scale factor $S$
  before backprop, then dividing the resulting gradients by $S$
  before the master-weight update. Shifts the gradient distribution
  upward in the FP16 representable range to avoid underflow.
  `[micikevicius2017-mixed-precision §3;
  kb/excerpts/micikevicius2017-mixed-precision#sec-3]`.
- **BF16 (bfloat16)** — A 16-bit floating-point format with 8
  exponent bits and 7 mantissa bits. Same dynamic range as FP32;
  half the precision of FP16. The universal LLM pre-training default
  by 2023. `[kalamkar2019-bfloat16 §2;
  kb/excerpts/kalamkar2019-bfloat16#sec-2]`.
- **FP8 E4M3 / E5M2** — 8-bit float formats with 4-bit
  exponent + 3-bit mantissa (E4M3, range $\pm 448$) or 5-bit
  exponent + 2-bit mantissa (E5M2, larger range, lower precision).
  NVIDIA's hybrid recipe uses E4M3 forward and E5M2 backward;
  DeepSeek-V3 uses E4M3 everywhere with fine-grained tile scaling.
  `[deepseek-v3 §3.3.2;
  kb/excerpts/deepseek-v3-training#sec-3-3-2-mantissa]`.
- **Fine-grained quantization (tile/block scaling)** — Per-group
  scaling factors along inner dimensions of a GEMM, allowing each
  tile/block to choose its own scale and absorb local outliers.
  DeepSeek-V3's 1×128 (activations) and 128×128 (weights) scheme.
  `[deepseek-v3 §3.3.2;
  kb/excerpts/deepseek-v3-training#sec-3-3-2-finegrained]`.
- **Promotion to CUDA cores at $N_C$** — Periodic (every $N_C = 128$
  elements) copy of FP8 Tensor Core partial sums to FP32 registers
  on CUDA cores, where full-precision accumulation is performed.
  Recovers accumulation precision lost to Tensor Core's 14-bit
  internal accumulator. `[deepseek-v3 §3.3.2;
  kb/excerpts/deepseek-v3-training#sec-3-3-2-accum]`.
- **NVFP4** — 4-bit floating-point format (E2M1) with native
  hardware MXScale (32-element per-block scaling) on Blackwell-class
  GPUs. Production-stability for full pre-training is open as of
  2026-05.
- **Loss spike** — A sudden upward jump in pre-training loss, often
  caused by gradient-norm explosion, attention-logit divergence,
  embedding-norm growth, or FP8 underflow. May or may not recover.
  DeepSeek-V3 reports zero irrecoverable spikes over 14.8T tokens
  `[deepseek-v3 abstract;
  kb/excerpts/deepseek-v3-training#abstract]`.
- **QK-LayerNorm / QK-clip** — Stability machinery to prevent
  attention-logit blow-up: normalize $Q$ and $K$ before the dot
  product (QK-LayerNorm, Henry et al. 2020) or clamp $QK^\top$ before
  softmax (QK-clip).

## Adaptation, PEFT, and merging

- **LoRA (Low-Rank Adaptation)** — Parameterizes a frozen weight
  matrix's update as a rank-$r$ product $BA$ with $r \ll
  \min(d_{\text{in}}, d_{\text{out}})$; only $A, B$ are trained. Merges
  to a single weight at inference time with no latency cost.
  `[hu2021-lora §3, §4; kb/excerpts/hu2021-lora#sec-3]`.
- **QLoRA** — LoRA over a 4-bit-quantized (NF4) base model. Adds
  double-quantization and paged optimizers. 65B fine-tune on a single
  48GB GPU. `[dettmers2023-qlora §3;
  kb/excerpts/dettmers2023-qlora#sec-3-nf4]`.
- **NF4 (NormalFloat-4)** — 4-bit quantization with quantile-spaced
  bin centers matched to a standard normal distribution. Information-
  theoretically near-optimal for pretrained weights, which are
  empirically near-Gaussian per block.
  `[dettmers2023-qlora §3.1; kb/excerpts/dettmers2023-qlora#sec-3-nf4]`.
- **DoRA (Direction-Then-Magnitude LoRA)** — LoRA variant decomposing
  $W$ into magnitude and direction; LoRA the direction only. Cited
  via Liu et al. 2024. Often outperforms LoRA at similar parameter
  count.
- **Task vector** — A direction in weight space defined as
  $\tau_T = \theta_T - \theta_0$ where $\theta_0$ is a pretrained base
  and $\theta_T$ is the same base fine-tuned on task $T$. Arithmetic
  on task vectors (addition, negation, combination) steers model
  behavior. `[ilharco2022-task-vectors §2;
  kb/excerpts/ilharco2022-task-vectors#sec-2]`.
- **Model Soup** — A model produced by averaging the weights of $K$
  fine-tunes from a shared base. *Uniform soup* averages all; *greedy
  soup* iteratively includes only fine-tunes that don't decrease
  validation accuracy. `[wortsman2022-model-soups §3;
  kb/excerpts/wortsman2022-model-soups#sec-3]`.
- **TIES-Merging (Trim-Elect-Sign-Disjoint-Merge)** — Three-step model
  merge: trim each task vector to top-$k\%$, elect a sign per
  coordinate by magnitude-weighted majority, disjoint-merge the
  surviving same-sign entries. `[yadav2023-ties-merging §3;
  kb/excerpts/yadav2023-ties-merging#sec-3]`.
- **DARE (Drop And REscale)** — Bernoulli-drop $1-p$ of task-vector
  entries and rescale survivors by $1/p$ before TIES or Soup merging.
  Yu et al. 2024.
- **MergeKit** — Open-source toolkit for model merging supporting
  Linear / SLERP / TIES / DARE / passthrough merges with per-tensor
  configuration. `[mergekit2024]`.
- **Linear mode connectivity (LMC)** — The empirical property that
  fine-tunes of a shared base land in a connected loss basin: the
  path $\theta_0 + t(\theta_f - \theta_0)$, $t \in [0,1]$, stays in
  low-loss territory. Underpins why model merging works.
- **Frankenmerge / passthrough merge** — Layer-wise interleaving of
  layers from two or more models (e.g., copy layers 0–15 from model A
  and 16–31 from model B). Widely used in OSS practice; limited
  primary literature. [FORUM-SIGNAL].

## SFT and post-training data

- **Loss masking (in SFT)** — Setting per-token loss weight to 0 on
  prompt tokens and 1 on response tokens, so only response tokens
  contribute gradient. The standard convention; multi-turn variants
  train on all assistant turns or only the last. `[ouyang2022-
  instructgpt §3; kb/excerpts/ouyang2022-instructgpt#sec-3]`.
- **Rejection sampling (in post-training)** — Sample $N$ responses
  per prompt from the current policy, score with a reward model, keep
  the top-$k$, and SFT on those. The Llama-3 quality gate before each
  preference round. `[meta-llama3]`.
- **MAGPIE (alignment data synthesis from scratch)** — Procedure
  exploiting that an aligned LM, prompted with only its instruction-
  template prefix (no question), will fabricate the question and then
  answer it. Produces $(x,y)$ SFT pairs at zero human cost.
  `[magpie-2024]`.
- **Reasoning-distillation SFT** — Fine-tuning a small base model on
  $(x,y)$ traces sampled from a strong RL-trained reasoner (e.g.,
  R1-Distill: 800K samples from DeepSeek-R1; s1: 1K curated traces
  from Gemini Flash Thinking). Recovers most reasoning capability
  with no RL. `[deepseek-r1; s1-2025]`.
- **Thinking / non-thinking mode SFT** — Training a model to emit
  `<think>...</think>` blocks gated by a system prompt, switching
  reasoning behavior on/off. Qwen 3's 4-stage post-training innovation.
  `[qwen3]`.
- **FLAN (Finetuned Language Net)** — Original instruction-tuning
  recipe: ~1.8K NLP tasks reformatted as natural-language
  instructions, used as an SFT mixture. The pre-RLHF post-training
  baseline. `[wei2021-flan]`. FLAN-T5 `[chung2022-flan-t5]` and the
  FLAN Collection (Longpre et al. 2023, arXiv:2301.13688) extend it.

## RLAIF and Constitutional AI

- **RLAIF (Reinforcement Learning from AI Feedback)** — RLHF pipeline
  where preference labels come from an AI feedback model rather than
  human labelers. The reward model is trained on AI-generated
  preferences via standard Bradley–Terry. `[bai2022-cai §4;
  kb/excerpts/bai2022-cai#sec-4]`.
- **Constitution (in CAI)** — A list of natural-language principles
  ("the assistant should not...") used to condition the AI feedback
  model's preferences. The only human-authored alignment input in
  Constitutional AI. `[bai2022-cai §3;
  kb/excerpts/bai2022-cai#sec-3]`.
- **SL-CAI** — Stage 1 of Constitutional AI: sample a response from
  a helpful-only model, critique under a randomly-chosen principle,
  revise, and SFT on the revisions. Produces $\pi^{\mathrm{SL-CAI}}$.
  `[bai2022-cai §3; kb/excerpts/bai2022-cai#sec-3]`.
- **RL-CAI** — Stage 2 of Constitutional AI: sample response pairs
  from $\pi^{\mathrm{SL-CAI}}$, have an AI judge choose under a
  principle, train a reward model on AI preferences, run PPO with KL
  anchor. `[bai2022-cai §4; kb/excerpts/bai2022-cai#sec-4]`.
- **Non-evasion (CAI behavioral signature)** — A model trained with
  CAI engages with harmful prompts by stating its objections, rather
  than refusing flatly. The defining behavioral signature
  distinguishing CAI from helpful-only RLHF (which complies) and from
  naive safety RLHF (which refuses without explanation).
  `[bai2022-cai §6; kb/excerpts/bai2022-cai#sec-6]`.
- **R-CAI (Reverse CAI)** — Inverts the constitution and applies
  probability-clamped RLAIF to generate controlled adversarial / toxic
  data. `[r-cai-2026]`.
- **Recursive RLAIF** — Using a previously-trained RLAIF model as the
  feedback model for the next round of RLAIF. Risks model collapse in
  the constitution direction.

## Structured output / constrained decoding

- **Constrained decoding** — Generation procedure that, at each step,
  masks the logits to allow only tokens consistent with a target
  grammar (regex, JSON schema, GBNF, CFG). The sampling distribution
  is restricted to $\mathcal{V}_{\text{allowed}}(s_t)$, the FSM's
  allowed-token set in the current state. `[willard2023-outlines §3;
  kb/excerpts/willard2023-outlines#sec-3]`.
- **FSM-vocabulary index (Willard–Louf)** — Precomputed mapping from
  each FSM state to the set of allowed vocabulary tokens. Built once
  per grammar; reduces per-step constraint check from $O(V)$ to
  amortized $O(1)$. `[willard2023-outlines §3;
  kb/excerpts/willard2023-outlines#sec-3]`.
- **Context-independent vs context-dependent tokens** — Partition of
  the vocabulary by whether allowed/disallowed status depends only on
  the parser state (context-independent: precomputable bitmask) or
  also on stack contents (context-dependent: runtime check). XGrammar
  reports ~99% context-independent for typical JSON schemas.
  `[xgrammar2024 §3; kb/excerpts/xgrammar2024#sec-3]`.
- **Compressed FSM (SGLang)** — FSM transformation that merges chains
  of singular-transition edges into single edges, allowing multi-token
  decoding when the grammar deterministically forces a literal
  substring (e.g., `{"summary": "` in a JSON schema).
  `[zheng2024-sglang §4; kb/excerpts/zheng2024-sglang#sec-4]`.
- **GBNF (GGML BNF)** — llama.cpp's grammar specification format.
  An extended BNF used to define constrained-decoding grammars.
- **JSON mode / structured output API** — Vendor-facing API features
  (OpenAI `response_format`, Anthropic tool-use schemas, etc.) that
  guarantee output conforms to a schema. Implemented underneath via
  FSM-constrained decoding. Exact-conformance guarantees depend on
  whether the backend uses true FSM or only logit-bias.
- **JSONSchemaBench** — Benchmark of 10K real-world JSON schemas
  used to compare constrained-decoding engines. `[FORUM-SIGNAL:
  arXiv:2501.10868]`.


### interpretability+evaluation+alignment

## Probing

- **Probe** — A (typically linear) classifier $g: \mathbb{R}^d \to \mathcal{Y}$ trained on frozen LM activations $\mathbf{h}^{(\ell, t)}$ to predict an external property $y$. The model's parameters are held fixed; only the probe is trained. Foundational to interpretability methodology `[alain-bengio-2017 §1]`.
- **Linear probe** — Probe of the form $g(\mathbf{h}) = \sigma(\mathbf{w}^\top \mathbf{h} + b)$ with $\mathbf{w} \in \mathbb{R}^d$. Distinguished from non-linear / MLP probes that can suffer from "probe leakage" (learning the property from low-information features) `[alain-bengio-2017]`.
- **Structural probe** — Hewitt & Manning's geometric probe that predicts higher-arity structure (e.g., parse-tree distance) by learning a linear transformation $\mathbf{B} \in \mathbb{R}^{k \times d}$ such that $\|\mathbf{B}(\mathbf{h}_i - \mathbf{h}_j)\|_2^2 \approx d_T(w_i, w_j)$ for tokens $i, j$ `[hewitt2019-structural-probe]`.
- **Mass-mean probe / difference-in-means probe** — Parameter-free probe that defines the property direction as $\mathbf{d}_{\mathrm{MM}} = \boldsymbol{\mu}_+ - \boldsymbol{\mu}_-$ (per-class mean difference). Closed-form, no optimization required. Marks & Tegmark 2023 argue the directions found are "more causally implicated in model outputs" than logistic-regression probe directions on the same data `[marks-tegmark-2023-truth §abstract]`.
- **Probing methodology critique** — The argument (consolidated in Belinkov 2022) that high-accuracy probes are correlational and can read properties the model represents but does not *use*. Three named confounders: probe-expressivity, spurious-correlation, vestigial-information `[belinkov2022-probing-survey §abstract]`.
- **Causal probing** — The 2023+ pattern of pairing probes with activation interventions (direction ablation, addition, steering) to test whether the probe's direction is *causally* used by the model, not merely correlated with the property. Marks & Tegmark 2023 demonstrate this on a truth direction in LLaMA-2-13B that flips TRUE/FALSE prediction when ablated `[marks-tegmark-2023-truth §abstract]`.
- **Direction ablation** — Causal-probing intervention that projects a residual-stream activation onto the orthogonal complement of a probe direction $\hat{\mathbf{d}}$: $\mathbf{h} \mapsto \mathbf{h} - (\mathbf{h}^\top \hat{\mathbf{d}})\hat{\mathbf{d}}$, then runs the rest of the model. A behavior change validates causal use of $\hat{\mathbf{d}}$ `[marks-tegmark-2023-truth §abstract]`.
- **Activation steering** — Inference-time intervention that adds $\alpha \hat{\mathbf{d}}$ to a residual-stream activation to amplify (or, with $\alpha < 0$, suppress) a probed direction. The Representation Engineering / Vennemeyer 2025 line `[vennemeyer2025-sycophancy-not-one-thing FORUM-SIGNAL]`.
- **Truth direction** — A direction in residual-stream space whose projection sign correlates with (and, validated by Marks & Tegmark 2023, causally controls) the model's TRUE/FALSE prediction on factual statements. Localizes around layer 12-15 of LLaMA-2-13B at end-of-statement punctuation tokens `[marks-tegmark-2023-truth §3]`.
- **Summarization behavior** — The empirical pattern (Tigges et al. 2023, Marks & Tegmark 2023) that information about a clause is encoded over the clause-ending punctuation token, making period-position activations the canonical probing target for sentence-level properties `[marks-tegmark-2023-truth §3]`.

## Knowledge benchmarks

- **MMLU (Massive Multitask Language Understanding)** — 57-subject 4-way MCQ benchmark introduced by Hendrycks et al. 2020 / ICLR 2021. 15,908 questions sourced from public exam banks (AP, GRE, MCAT, professional licensing). Foundational for the post-2021 LLM eval era; saturated at >90% on frontier models as of 2026 `[hendrycks2021-mmlu §abstract]`.
- **MMLU-Redux** — Polo et al. 2024 audit of MMLU finding 6.49% baseline error rate, with the Virology subset reaching 57% errors. Per-question corrections published; methodology audit, not a new benchmark `[mmlu-redux-2024 §abstract]`.
- **MMLU-Pro** — 10-way MCQ, reasoning-augmented successor to MMLU. 12,032 items, 16-33 percentage point drop in headline score relative to MMLU at launch, 2× lower prompt-format sensitivity `[wang2024-mmlu-pro §abstract]`.
- **GPQA Diamond** — 198-question subset of GPQA (Rein et al. 2023) where two PhD experts agree on the answer and at least one non-expert validator with web access got it wrong. The "Google-proof" property is operationalized via the 34% non-expert-with-web baseline. Saturated at frontier as of 2026 (>90%) `[rein2023-gpqa §abstract]`.
- **Answer-letter prior** — The training-data-induced bias that pre-trained LMs do not have uniform priors over A/B/C/D in MCQ items. Letter bias can favor one option by several percentage points; 10-way MCQ (MMLU-Pro) dilutes this effect `[wang2024-mmlu-pro §sec-delta]`.
- **CoT-vs-direct gap (as a knowledge/reasoning diagnostic)** — On a knowledge-dominant benchmark (MMLU), Chain-of-Thought scoring gives roughly the same result as direct-answer scoring; on a reasoning-dominant benchmark (MMLU-Pro), CoT improves the score measurably. The gap is a diagnostic for whether a benchmark requires multi-step inference `[wang2024-mmlu-pro §sec-cot]`.
- **Knowledge-vs-reasoning hybrid** — A benchmark where items combine factual retrieval and multi-step reasoning. MMLU-Pro, GPQA, HLE all qualify; the post-MMLU eval landscape has largely abandoned pure-knowledge benchmarks in favor of hybrids `[wang2024-mmlu-pro; rein2023-gpqa; phan2025-hle]`.

## Safety evaluation

- **ASR (Attack Success Rate)** — Headline metric in adversarial-robustness benchmarks: fraction of (behavior, attack) pairs for which the model produces an output the judge rates harmful. Not directly comparable across papers unless the judge, behaviors, and attack family match `[chao2024-jailbreakbench §abstract]`.
- **HarmBench** — Mazeika et al. 2024 standardized red-team evaluation framework: 510 behaviors in four functional categories, 18 attack methods, 33 target LLMs/defenses. Used by AISI for pre-deployment evals `[harmbench2024 §abstract]`.
- **JailbreakBench** — Chao et al. 2024 reproducibility-focused jailbreak benchmark: 100 OpenAI-policy-aligned behaviors, evolving repository of jailbreak artifacts (the actual successful prompts), standardized LLM-as-judge, public leaderboard. NeurIPS 2024 D&B `[chao2024-jailbreakbench §abstract]`.
- **Crescendo (multi-turn jailbreak)** — Russinovich et al. 2024 attack: gradually escalate over multiple benign-seeming turns, exploiting the LM's tendency to over-weight its own prior committed responses. Achieves 29-71% ASR uplift over single-turn baselines on AdvBench against GPT-4 / Gemini-Pro. Documents that single-turn evals systematically underestimate the attack surface `[russinovich2024-crescendo §abstract]`.
- **Crescendomation** — The automated implementation of Crescendo: an attacker LLM generates the escalation chain. The 29-71% ASR uplift number refers to Crescendomation specifically `[russinovich2024-crescendo §sec-asr-uplift]`.
- **Competing objectives (jailbreak failure mode)** — Wei, Haghtalab, Steinhardt 2023: when a model's capability training and safety training pull in different directions, attacks that invoke the capability override the safety training. One of two named failure modes `[wei2023-jailbroken §abstract]`.
- **Mismatched generalization (jailbreak failure mode)** — Wei et al. 2023: safety training fails to generalize to a domain (Base64, low-resource language, role-play) where the model has capabilities. The other named failure mode `[wei2023-jailbroken §abstract]`.
- **Safety-capability parity** — Wei et al. 2023's policy recommendation: safety mechanisms must be as sophisticated as the model's full capability range. Partial-domain safety training will be defeated by attacks that target the capability/safety gap `[wei2023-jailbroken §abstract]`.
- **ASL (AI Safety Level)** — Anthropic's Responsible Scaling Policy tier framework. ASL-1 (low risk, no autonomy) → ASL-2 (current frontier baseline) → ASL-3 (substantially elevated risk; specific deployment controls required) → ASL-4 (autonomous AI risk; not yet deployed). Each tier specifies capability-evaluation triggers and security/containment commitments `[anthropic-rsp-2024]`.
- **Task-horizon metric (METR)** — Capability-evaluation metric framing model performance as the human-time-to-complete distribution at which the model succeeds at >50%. Becoming the field's lingua franca for capability quantification across labs and external evaluators `[metr-autonomy-2024]`.
- **Dangerous-capability evaluation** — Pre-deployment eval that tests whether a model can autonomously carry out a task whose successful completion would constitute a threat (cyber CTFs, autonomous replication, biology / CBRN tasks). Inverts the usual "model refuses harm" frame to "model could *do* harm if it tried" `[metr-autonomy-2024; cybench2024 FORUM-SIGNAL]`.
- **Judge-LLM contamination** — Failure mode in safety evals: when the judge classifier is from the same model family as the model being evaluated, it may systematically underrate harmful generations from that family. Cross-judge agreement testing is the partial mitigation `[chao2024-jailbreakbench §abstract]`.
- **HarmBench-Judge** — The fine-tuned classifier (Llama-2-13B family, fine-tuned on labeled harmful/non-harmful generations) shipped with HarmBench as the standardized judge for the 510 behaviors `[harmbench2024 §sec-structure]`.
- **Honeypot evaluation** — Eval protocol that presents the model with a deployment-realistic context (or, conversely, a clearly-evaluation context) to test whether eval scores are stable across the eval-vs-deployment boundary. Motivated by the Apollo / scheming finding that models may behave differently when they suspect they are being evaluated `[meinke2024-apollo-scheming]`.

