---
topic: architecture/multi-token-prediction
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - gloeckle2024-mtp
  - deepseek-v3
  - leviathan2023
secondary_sources:
  - meta-llama4
  - cai2024-medusa
  - li2024-eagle
related_topics:
  - architecture/transformer-overview
  - architecture/embeddings-and-tying
  - architecture/reasoning-architectures
  - inference/speculative-decoding
---

# Multi-token prediction (MTP)

**Multi-token prediction** modifies the standard next-token training
objective to predict tokens $t+1, t+2, \ldots, t+n$ jointly at each
position, rather than only $t+1$. It was proposed as a research
training objective by Meta (Gloeckle et al. 2024) and **productionized
in DeepSeek-V3** (Dec 2024) as both a quality-improving auxiliary loss
and as an integrated drafter for self-speculative decoding. MTP is
distinct from speculative decoding: MTP modifies *training* to produce
heads predicting future tokens; speculative decoding is an *inference*
algorithm that uses any drafter.

## 1. Formal definition

### 1.1 Standard next-token objective (baseline)

Standard LM training maximizes the conditional log-likelihood of the
next token:

$$\mathcal{L}_{\text{NTP}}(\theta) = -\sum_{t} \log p_\theta(x_{t+1} \mid x_{1:t}) \tag{1}$$

producing logits via a single LM head $E_{\text{out}}$ on the final
hidden state $\mathbf{h}_t^L$.

### 1.2 Multi-token prediction (Gloeckle 2024)

Predict $n$ tokens at each position with $n$ independent output heads
on a shared trunk:

$$\mathcal{L}_{\text{MTP-Gloeckle}}(\theta) = -\sum_{t} \sum_{i=1}^{n} \log p_{\theta, i}(x_{t+i} \mid x_{1:t}) \tag{2}$$

`[gloeckle2024-mtp §2; kb/excerpts/gloeckle2024-mtp#sec-2]`. Each head
$i$ receives the same trunk hidden state $\mathbf{h}_t^L$ and projects
to its own logits over the vocabulary; heads are independent.

### 1.3 Multi-token prediction (DeepSeek-V3)

DeepSeek-V3 uses a **sequential / chained** MTP architecture: the $i$-th
MTP module receives the trunk's hidden state *plus the embedding of
the previously predicted token*, computing a refined hidden for the
next prediction. Concretely, for $i \in \{1, \ldots, k\}$ MTP modules
beyond the main head `[deepseek-v3 §2.3.1; kb/excerpts/deepseek-v3-training#sec-2-3-1]`:

$$\mathbf{h}_t^{i} = M_i\bigl(\mathrm{Norm}(\mathbf{h}_t^{i-1}); \mathrm{Norm}(E_{\text{in}}[x_{t+i}])\bigr) \tag{3}$$

where $M_i$ is a small Transformer block (one attention + FFN). The
LM head is **shared** between the main and MTP modules:

$$\mathrm{logits}_{t+i+1}^{(i)} = \mathbf{h}_t^{i} \, E_{\text{out}} \tag{4}$$

Total objective:

$$\mathcal{L} = \mathcal{L}_{\text{NTP}} + \frac{\lambda}{D} \sum_{i=1}^{k} \mathcal{L}_{\text{MTP},i} \tag{5}$$

with $\lambda$ an aux-loss weight and $D$ a normalizer. DeepSeek-V3 uses
$k = 1$ (one MTP module beyond the main head) and a $\lambda$ schedule
that decays during training.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $n$ (Gloeckle) / $k$ (DeepSeek) | depth of MTP — number of future tokens predicted |
| $\mathbf{h}_t^L$ | trunk's final hidden state at position $t$ |
| $\mathbf{h}_t^{i}$ | $i$-th MTP module's hidden state |
| $M_i$ | $i$-th MTP module (small Transformer block) |
| $E_{\text{in}}, E_{\text{out}}$ | shared input embedding and LM head |
| $\lambda$ | aux-loss weight on MTP losses |
| $\mathcal{L}_{\text{MTP},i}$ | cross-entropy on the $i$-th MTP head |

## 2. Mechanism

### 2.1 Architectural placement (DeepSeek-V3)

The MTP modules sit **after the main backbone**, parallel to the LM
head:

```
   ┌── trunk block L-1 ──┐
   │   ...               │
   │   trunk block 0     │
   └─────── h_t^L ───────┘
            │
            ├── LM head ── logits for x_{t+1}  (main NTP)
            │
            ├── M_1(h_t^L, emb(x_{t+1})) ── logits for x_{t+2}  (MTP head 1)
            │
            ├── M_2(h_t^1, emb(x_{t+2})) ── logits for x_{t+3}  (MTP head 2)
            ...
```

`[deepseek-v3 §2.3.1; kb/excerpts/deepseek-v3-training#sec-2-3-1]`. The
chain is **causal**: the $i$-th module sees the $i$-th-future ground-
truth token at training time. Parameters per MTP module: ~one Transformer
block. DeepSeek-V3 with $k = 1$ adds one block of parameters relative
to a non-MTP baseline.

### 2.2 Training-time forward (chained, $k=1$)

For each position $t$ in the training batch:

1. Run the main backbone, producing $\mathbf{h}_t^L$ for all $t$ in
   parallel.
2. Compute main NTP loss: cross-entropy of $\mathbf{h}_t^L \, E_{\text{out}}$
   against ground-truth $x_{t+1}$.
3. For MTP module 1: $\mathbf{h}_t^1 = M_1(\mathbf{h}_t^L,
   E_{\text{in}}[x_{t+1}])$.
4. Compute MTP loss 1: cross-entropy of $\mathbf{h}_t^1 \, E_{\text{out}}$
   against ground-truth $x_{t+2}$.

Note that the MTP module receives the **ground-truth** $x_{t+1}$ at
training time (teacher forcing). This is critical: it means the MTP
module is learning to predict $x_{t+2}$ given $x_{1:t+1}$, not given
$x_{1:t}$ + (a possibly incorrect prediction of $x_{t+1}$). Inference-
time behavior differs (see §2.4).

### 2.3 Inference mode 1 — discard MTP, use only NTP

For standard generation, simply drop the MTP modules and use the main
LM head as in any other Transformer LLM. **The MTP heads have no
inference cost**. This is the default path
`[deepseek-v3 §2.3.1; kb/excerpts/deepseek-v3-training#sec-2-3-1]`.

The reported quality benefit is therefore "free" inference cost: the
MTP auxiliary loss shapes the trunk hidden states $\mathbf{h}_t^L$ to
be *more predictive of multi-step future tokens*, which DeepSeek-V3
ablations report as small but consistent quality gains over a non-MTP
baseline.

### 2.4 Inference mode 2 — MTP as drafter for self-speculative decoding

The MTP heads can also serve as a **drafter** for speculative decoding
`[leviathan2023 §2.1; kb/excerpts/leviathan2023#sec-2-1]`:

1. Trunk computes $\mathbf{h}_t^L$ for the current generation step.
2. Main head emits draft for $x_{t+1}$.
3. MTP module 1 takes $(\mathbf{h}_t^L, \mathrm{emb}(\hat x_{t+1}))$ and
   emits draft for $x_{t+2}$.
4. Verify the $k+1$ drafts in parallel via a single trunk forward pass
   over the new tokens; accept according to the speculative-sampling
   rejection rule `[leviathan2023 §2.3; kb/excerpts/leviathan2023#sec-2-3]`.

DeepSeek-V3 reports **acceptance rate ~85%** for MTP-as-drafter and
**~1.8× wall-clock speedup** in serving `[deepseek-v3 §2.3.1]`.

The advantage over external-drafter speculative decoding (EAGLE
`[li2024-eagle §3]`, Medusa `[cai2024-medusa §3.1]`) is operational:
**no separate drafter model** to deploy, version, or quantize. The
disadvantage is a tighter coupling between draft quality and trunk
training, and an extra ~1 block of parameters.

## 3. Variants and lineage

### 3.1 Comparison: parallel vs chained

| Variant | Architecture | $n$ predicted | Training cost | Inference use |
|---|---|---|---|---|
| Gloeckle 2024 (Meta) | $n$ parallel heads | up to $n=4$ | small | drafter |
| **DeepSeek-V3** | $k=1$ chained module + shared LM head | $k+1$ tokens | +1 block | drafter or discard |
| Llama 4 (claimed) | TBD; tech report mentions MTP | TBD | TBD | TBD `[meta-llama4 §2]` |

Gloeckle's parallel design produces independent heads; DeepSeek's
chained design **conditions each future-token prediction on the
previous future-token's ground truth** (training) or prediction
(inference). DeepSeek-V3 ablations (not fully published) report the
chained variant outperforms parallel.

### 3.2 Distinction from speculative decoding

| Property | MTP | Speculative decoding `[leviathan2023]` |
|---|---|---|
| What it changes | training objective + architecture | inference algorithm |
| Output distribution | exact (with rejection rule) when used as drafter | exact (rejection rule) |
| Drafter | integrated, same backbone | separate model (small LLM, EAGLE, Medusa) |
| Quality lift | small but reported | none (only speed) |
| Parameter cost | +1 block (DeepSeek-V3) | drafter model size |

MTP is **architectural + training**, speculative decoding is **inference
only**. The two are composable: an MTP-trained model used as drafter
for itself is "self-speculative decoding."

### 3.3 Distinction from EAGLE / Medusa

EAGLE and Medusa add drafter heads *post hoc* to a pretrained model:

- **Medusa** adds $k$ parallel heads (similar to Gloeckle's design)
  trained on top of a frozen backbone; uses tree attention for
  parallel verification `[cai2024-medusa §3.1; kb/excerpts/cai2024-medusa#sec-3-1]`.
- **EAGLE** trains a small autoregressive head that predicts feature
  vectors, not tokens, of the next position; verifies via a single
  trunk forward `[li2024-eagle §3; kb/excerpts/li2024-eagle#sec-3]`.

MTP is structurally similar to Medusa but with the training cost paid
**during pretraining** rather than in a separate fine-tuning stage,
and (in DeepSeek-V3) with chaining rather than parallel heads.

## 4. Intuitions and analogies

[ANALOGY] MTP is **a foresight tax that improves planning**. The
auxiliary loss forces the trunk to encode information about $x_{t+2}$
in $\mathbf{h}_t^L$, not just $x_{t+1}$. Models that "know what's
coming two tokens ahead" are pushed toward representations that are
more compositional and forward-looking. The analogy returns to
canonical form via Eq. 5: the loss on $\mathcal{L}_{\text{MTP},1}$
backpropagates through $M_1$ into the trunk, shaping $\mathbf{h}_t^L$.

[INTUITION] **Why MTP can improve quality, not just speed.** The next-
token objective is a famously local signal: minimizing
$-\log p(x_{t+1} | x_{1:t})$ doesn't directly penalize representations
that fail to encode longer-range structure. MTP's
$-\log p(x_{t+2} | x_{1:t}, x_{t+1})$ explicitly does. The gradient
through $M_1$ into the trunk is the mechanism. Whether this is "real"
or just regularization is contested.

[INTUITION] **Chaining beats parallel because earlier predictions
condition later ones.** Gloeckle's parallel heads predict $x_{t+i}$
all from the same $\mathbf{h}_t^L$ — they cannot use the $x_{t+1}$
prediction to inform the $x_{t+2}$ prediction. DeepSeek's chained
design feeds the previous prediction into the next module, so the
$x_{t+2}$ predictor is conditioned on $x_{t+1}$. At inference (drafter
mode), this means later draft tokens are conditioned on earlier draft
tokens — the natural autoregressive structure.

[CONTRADICTION] Quality vs. speed framing. The Gloeckle 2024 paper
emphasizes **speedup** via parallel drafter use; DeepSeek-V3 reports
both speedup and a quality lift from the auxiliary objective itself.
The two papers report MTP for somewhat different reasons; whether
the quality lift survives at all scales is not fully ablated.

## 5. Frontier and open questions (as of 2026-05)

- **Optimal MTP depth $k$.** DeepSeek-V3 uses $k = 1$. Gloeckle 2024
  reports diminishing returns past $n = 2$. No frontier-scale
  ablation of $k = 4, 8, 16$ has been published.
- **MTP × MoE interaction.** DeepSeek-V3 is MoE in the trunk; the MTP
  module is dense. Whether MTP heads should themselves be MoE is
  untested.
- **MTP × reasoning models.** DeepSeek-R1 is built on DeepSeek-V3
  architecture, retaining MTP. The reasoning-mode generation produces
  long CoT chains; whether MTP heads contribute to or hinder
  long-CoT quality is not analyzed in published R1 ablations.
- **MTP for code vs natural language.** Gloeckle 2024 reports MTP is
  *more* useful for code than natural language `[gloeckle2024-mtp §3]`.
  DeepSeek-V3's training mix is heavily code; whether MTP's quality
  gain is concentrated there is unresolved.
- **MTP-as-drafter acceptance rate at deeper $k$.** As $k$ grows,
  draft-token accuracy degrades, so the speculative-decoding speedup
  curve flattens. The optimal $k$ for serving is engineering-task-
  specific.
- **Llama 4 use of MTP.** The Llama 4 tech report mentions MTP
  `[meta-llama4 §2]` but the architecture-level details are sparse.

## 6. See also

- `kb/notes/architecture/transformer-overview.md` — where MTP sits
  (after the trunk, parallel to the main LM head).
- `kb/notes/architecture/embeddings-and-tying.md` — MTP shares the
  LM head $E_{\text{out}}$ with the main NTP head.
- `kb/notes/architecture/reasoning-architectures.md` — DeepSeek-R1
  retains MTP from V3; possible interaction with long-CoT.
- `kb/notes/inference/speculative-decoding.md` — the canonical
  speculative-decoding rejection rule that MTP-as-drafter uses.
- `kb/notes/training/optimization.md` — auxiliary-loss weight
  scheduling.
