---
topic: architecture/multimodal-llm-extensions
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - llava2023
  - alayrac2022-flamingo
  - internvl3-2025
  - meta-llama4
  - gemma3
secondary_sources:
  - su2021
  - qwen3
related_topics:
  - architecture/position-encoding
  - architecture/transformer-overview
  - architecture/tokenization
  - training/synthetic-data-and-distillation
---

# Multimodal LLM extensions

How non-text modalities — images, video, audio — enter a Transformer
language model. The 2022–2023 paradigm was **adapter-based**: a frozen
vision encoder feeds a learned projection into a (mostly) frozen LLM.
The 2025–2026 paradigm is **early fusion / native multimodal
pretraining**: the model is trained jointly on interleaved
image+text+audio from scratch. Position-encoding extensions (M-RoPE,
Interleaved-MRoPE) handle the new spatial axes.

## 1. Formal definition — three integration patterns

For a vision modality with image $I$ encoded into $M$ patch tokens, and
a text prefix $\mathbf{t} \in V^N$, three architectural patterns embed
the visual tokens into the residual stream.

### 1.1 Adapter-based (LLaVA-style)

A frozen vision encoder $E_V: I \to \mathbb{R}^{M \times d_V}$ produces
patch features. A learned **projector** $P: \mathbb{R}^{d_V} \to
\mathbb{R}^{d_{\text{model}}}$ maps to the LLM residual width:

$$\mathbf{X}_{\text{vis}} = P(E_V(I)) \in \mathbb{R}^{M \times d_{\text{model}}} \tag{1}$$

`[llava2023 §3.1; kb/excerpts/llava2023#sec-3-1]`. Visual tokens are
**concatenated** with text-token embeddings into a single sequence:

$$\mathbf{X}^0 = [E_{\text{in}}[\mathbf{t}_{1:k}]; \mathbf{X}_{\text{vis}}; E_{\text{in}}[\mathbf{t}_{k+1:N}]] \tag{2}$$

— the LLM consumes them as if they were ordinary tokens. The vision
encoder and (often) the LLM are frozen during initial training; only
$P$ trains. A second instruction-tuning stage updates the LLM weights
on visual-instruction data.

### 1.2 Cross-attention insertion (Flamingo-style)

Flamingo (Alayrac et al. 2022) inserts new **gated cross-attention
blocks** between frozen LLM blocks `[alayrac2022-flamingo §2.1; kb/excerpts/alayrac2022-flamingo#sec-2-1]`:

$$\mathbf{X}^{\ell+1} = \mathbf{X}^\ell + \tanh(\alpha) \cdot \mathrm{XAttn}(\mathbf{X}^\ell, \mathbf{X}_{\text{vis}}) \tag{3}$$

with learnable gating $\alpha$ (initialized to 0 so the inserted layer
starts as identity). The base LLM remains frozen; only the cross-
attention blocks and a Perceiver-style image resampler train.

### 1.3 Early-fusion / native multimodal (Llama 4, InternVL3)

The model is trained from scratch on interleaved multimodal sequences.
Visual patches are first-class tokens at pretraining time; there is
no "frozen LLM" to graft. Architecturally this looks like a standard
decoder-only LLM where the **input embedding is multi-source**: text
tokens go through $E_{\text{in}}^{\text{text}}$, image patches through
a small ViT-like encoder $E_{\text{in}}^{\text{vis}}$, audio through
$E_{\text{in}}^{\text{audio}}$ — all producing $d_{\text{model}}$-dim
vectors that share the residual stream `[meta-llama4 §2; internvl3-2025 §3.1]`.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $I$ | input image (or video frame, or audio clip) |
| $M$ | number of visual patch tokens (variable in dynamic-resolution designs) |
| $d_V$ | vision-encoder feature dim (CLIP / SigLIP output) |
| $E_V$ | vision encoder (typically ViT) |
| $P$ | learned projector vision → LLM space (MLP or linear) |
| $\mathbf{X}_{\text{vis}}$ | projected visual tokens, ready for residual stream |
| $\alpha$ | Flamingo's gating parameter (0 at init) |

## 2. Mechanism

### 2.1 LLaVA pipeline (canonical adapter)

1. **Vision encoder.** ViT (CLIP or SigLIP, $\sim 300\text{M}$ params)
   encodes a 336×336 image into 576 patch tokens at $d_V = 1024$
   `[llava2023 §3.1; kb/excerpts/llava2023#sec-3-1]`.
2. **Projector.** A 2-layer MLP maps $\mathbb{R}^{1024} \to
   \mathbb{R}^{d_{\text{model}}}$.
3. **LLM.** Vicuna or LLaMA, 7B–70B, consumes
   $[\text{prompt}; \mathbf{X}_{\text{vis}}; \text{question}]$ and
   generates the answer autoregressively.

Training is two-stage:
- **Pretraining:** projector only (LLM frozen) on image-caption pairs.
- **Visual instruction tuning:** projector + LLM on synthetic
  GPT-4-generated visual conversations
  `[llava2023 §3.2; kb/excerpts/llava2023#sec-3-2]`.

### 2.2 M-RoPE — multi-axis rotary positions

Standard RoPE applies a 1D rotation to each query/key by angle
$m \theta_i$ where $m$ is the position index `[su2021 §3.2.2; kb/excerpts/su2021#sec-3-2-2]`.
For image patches, the relevant index is **2D** (height $h$, width $w$);
for video, **3D** (time $t$, height $h$, width $w$).

Qwen2-VL's **M-RoPE** (Multimodal RoPE) splits the per-head dimension
$d_h$ into three thirds and assigns each third to a different axis:

$$\mathbf{q}_{m,p}^{(\text{T})} = R_{\theta, t(p)}^{d_h/3} \cdot \mathbf{q}_{m,p}^{(\text{T,raw})}$$
$$\mathbf{q}_{m,p}^{(\text{H})} = R_{\theta, h(p)}^{d_h/3} \cdot \mathbf{q}_{m,p}^{(\text{H,raw})}$$
$$\mathbf{q}_{m,p}^{(\text{W})} = R_{\theta, w(p)}^{d_h/3} \cdot \mathbf{q}_{m,p}^{(\text{W,raw})}$$

where $t(p), h(p), w(p)$ are the temporal/height/width coordinates of
patch token $p$ (Qwen2-VL paper, arXiv 2409.12191). The dot product
$\mathbf{q}_p^\top \mathbf{k}_q$ becomes a sum of relative-position
terms across the three axes — effectively a 3D analog of the 1D RoPE
relative-position result `[su2021 §3.2.2 Eq.16; kb/excerpts/su2021#sec-3-2-2]`.

### 2.3 Interleaved-MRoPE (Qwen3-VL)

Qwen3-VL (Nov 2025, arXiv 2511.21631) replaces the chunked split with
**axis-interleaved** dimension assignment: for the $j$-th frequency
$\theta_j$, assign axis $j \mod 3 \in \{T, H, W\}$. Every dimension-
pair sees all three axes; spatial and temporal information mix at every
pair rather than being siloed by channel range. The intuition is that
chunked M-RoPE creates a "temporal channel range" with no spatial
information and vice versa, while interleaving lets every channel see
all three axes.

### 2.4 Naive Dynamic Resolution (Qwen2-VL)

The vision encoder accepts variable-size images; image-token count
$M$ is **proportional to pixel count** rather than fixed. A 4K image
might use $M = 2{,}048$ patches; a thumbnail $M = 32$. This requires
the LLM to handle highly variable per-sample sequence lengths but
allows fine detail on high-res inputs without fixed-budget compression.

### 2.5 Window attention in vision encoder (Qwen2.5-VL)

For high-resolution images the ViT's full self-attention becomes
$O(M^2)$ which is prohibitive. Qwen2.5-VL replaces dense ViT attention
with **sliding-window** attention (analogous to Mistral's SWA in the
LLM stack). Quality preserved at much lower cost on 4K+ images.

### 2.6 Audio: AuT trained from scratch (Qwen3-Omni)

Qwen3-Omni's AuT (Audio-Transformer) is trained on **20M hours of
audio** from scratch, producing a discrete-token representation of
audio with temporal positional encoding. Architecturally analogous to
SigLIP for vision, but built specifically for the Qwen training stack.
Audio modality lags vision in maturity and benchmarks.

## 3. Variants and lineage

### 3.1 Integration-pattern timeline

| Year | Method | Pattern | Key paper |
|---|---|---|---|
| 2022 | Flamingo (DeepMind) | gated cross-attention | `[alayrac2022-flamingo §2.1]` |
| 2023 | LLaVA (Liu et al.) | MLP projector → frozen LLM | `[llava2023 §3.1]` |
| 2023 | InstructBLIP | Q-Former → LLM | (BLIP-2 lineage) |
| 2024 | Qwen2-VL | dynamic res + M-RoPE | (arXiv 2409.12191) |
| 2025 | Qwen2.5-VL | + window-ViT, dynamic FPS | (arXiv 2502.13923) |
| 2025 | InternVL3 | **early-fusion native** | `[internvl3-2025 §3.1]` |
| 2025 | Qwen3-VL | Interleaved-MRoPE | (arXiv 2511.21631) |
| 2025 | Qwen3-Omni | + AuT audio | (arXiv 2509.17765) |
| 2025 | Gemma 3 | SigLIP2 + adapter | `[gemma3 §2.1]` |
| 2026 | Llama 4 | **early-fusion native** | `[meta-llama4 §2]` |

### 3.2 Adapter vs early-fusion

|  | Adapter | Early fusion |
|---|---|---|
| Vision encoder | frozen pretrained (CLIP/SigLIP) | trained jointly from scratch |
| LLM | frozen (then unfrozen for VIT) | trained jointly from scratch |
| Pretraining data | text-only LLM + image-caption pairs | interleaved multimodal corpus |
| Compute cost | LLM cost only | proportional to image+text ratio |
| Quality at scale | reported lower (no controlled comparison) | reported higher |

Llama 4 and InternVL3 both report early-fusion outperforms adapter
*at sufficient pretraining data*, but neither publishes a controlled
ablation isolating the architectural change `[meta-llama4 §2; internvl3-2025 §3.1]`.

[CONTRADICTION] "Early-fusion is better" rests on Llama 4 and
InternVL3 internal comparisons against their own adapter baselines.
A controlled comparison that holds data, compute, and post-training
fixed across the two architectures has not been published. The
practical signal (frontier labs choosing early-fusion) is stronger
than the published evidence.

### 3.3 Vision encoder lineage

| Encoder | Year | Used by | Notes |
|---|---|---|---|
| CLIP ViT | 2021 | original LLaVA | OpenAI; image-text contrastive |
| SigLIP | 2023 | Gemini 1.5+ | sigmoid-loss alternative to CLIP |
| **SigLIP2** | 2024 | Gemma 3, Qwen3-VL | improved scaling | `[gemma3 §2.1]` |
| Qwen-VL ViT | 2024 | Qwen2-VL family | custom |
| InternVL ViT | 2024 | InternVL family | scaled up to 6B |

SigLIP2 has become the **modern default** for open-model vision
encoding `[Phase 1 sweep §2 multimodal-llm-extensions]`.

## 4. Intuitions and analogies

[ANALOGY] The adapter pattern treats the LLM as a **black-box
generalist** and learns a "translator" between vision-encoder
features and LLM input tokens. The translator is small and fast to
train; the LLM is reused. Returns to canonical form via Eq. 1: the
projector $P$ is the entire interface. Limit: the LLM was never
trained on the vision-encoder's feature distribution, so subtle
visual reasoning that requires multi-step interaction with vision
features hits a ceiling.

[ANALOGY] Early fusion treats vision and text as **two dialects of one
language**: every Transformer block sees a mixed stream from both, so
visual and textual features can interact at every depth. Returns to
canonical form via the standard residual recurrence with multi-source
embeddings — the $X^0 = [E_{\text{in}}^{\text{text}}; E_{\text{in}}^{\text{vis}}]$
stream is the canonical object.

[INTUITION] **Why M-RoPE generalizes RoPE without retraining most of
the model.** RoPE's mathematical form is "rotation by angle $m\theta$
where $m$ is the position." It doesn't depend on $m$ being scalar; for
multi-axis input, $m\theta$ becomes $\sum_a m_a \theta^{(a)}$ across
axes. The dot product still depends only on relative positions per
axis. The canonical proof carries through; only the coordinate-
indexing changes `[su2021 §3.2.2; kb/excerpts/su2021#sec-3-2-2]`.

[INTUITION] **Interleaved-MRoPE > chunked M-RoPE because every head
sees every axis.** In chunked M-RoPE, the lowest-frequency third of
channels carries temporal information only, the middle third height
only, the high third width only. A model that wants to "attend to
positions sharing a time AND a width" has to do it through cross-
channel interaction — which takes a sublayer. Interleaving puts
all three axes on every channel pair, so attention-as-rotation can
see space-time correlations directly.

## 5. Frontier and open questions (as of 2026-05)

- **Native multimodal vs adapter at controlled compute.** No published
  controlled comparison. Frontier labs have voted (Llama 4, InternVL3,
  Gemini 2.5) but the evidence is pre-registered architectural choice
  + favorable internal benchmark, not an isolating experiment.
- **Audio modality maturity.** Qwen3-Omni's AuT trained from scratch
  on 20M hours is the most ambitious open audio encoder; whether it
  generalizes to other labs' stacks (transferability) is open.
- **Video temporal modeling.** Qwen2.5-VL's dynamic FPS and Qwen3-VL's
  Interleaved-MRoPE handle long video; LLaVA-Video uses SlowFast
  pooling. The choice between pooling-based and full-temporal-attention
  approaches is active.
- **Architecture vs data scale.** InternVL3's claim that "joint
  pretraining wins" may be a data-scale argument: with enough
  multimodal data, joint training beats adapter; with little, adapter
  wins. The cross-over point is unmeasured.
- **Multimodal Mixture-of-Experts.** Llama 4 combines MoE with early-
  fusion multimodal — does routing differentiate vision-token vs text-
  token experts? Llama 4 disclosure is light on this.
- **Closed-frontier (Claude, GPT-4o) vision architecture.** Anthropic
  and OpenAI have not published architecture details; only behavior
  is observable.

## 6. See also

- `kb/notes/architecture/position-encoding.md` — RoPE substrate
  underpinning M-RoPE / Interleaved-MRoPE.
- `kb/notes/architecture/transformer-overview.md` — where multimodal
  tokens enter the residual stream.
- `kb/notes/architecture/tokenization.md` — image-patch and audio-
  codec tokenization parallels text BPE.
- `kb/notes/architecture/embeddings-and-tying.md` — multi-source input
  embeddings.
- `kb/notes/training/synthetic-data-and-distillation.md` — visual-
  instruction-tuning data is largely synthetic (GPT-4-generated).
