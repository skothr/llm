---
topic: architecture/moe
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - fedus2021-switch
  - mixtral2024
  - deepseekmoe2024
  - deepseek-v3
  - qwen3
secondary_sources:
  - shazeer2020  # SwiGLU FFN, the building block of each expert
  - meta-llama4
related_topics:
  - architecture/ffn
  - architecture/transformer-overview
  - inference/serving-systems
  - training/distributed-training
---

# Mixture of Experts (MoE)

A Mixture-of-Experts layer replaces a single dense FFN sublayer with $E$
parallel FFN "experts" plus a routing function that selects, per token,
a small subset $k \ll E$ of experts to evaluate. Compute per token
scales with $k$, parameter count scales with $E$, and the resulting
**parameter-to-compute ratio** decouples capacity from per-token cost.
Every frontier model family in 2024–2026 (Llama 4, Qwen3, DeepSeek-V3,
Mistral 3 Large, Gemini 2.5) is MoE
`[Phase 1 sweep §2 transformer-overview; theory/plans/landscape-sweep/architecture.md]`.

The PDFs for Switch Transformer, Mixtral, DeepSeekMoE, DeepSeek-V3, and
Qwen3 were not accessible in this Phase 2 pass. Equations and reported
numbers below reflect the published forms; mechanism descriptions are
high-confidence but specific equation numbers should be verified
against the originals before propagation to the LaTeX series. Marked
inline.

## 1. Formal definition — the MoE FFN sublayer

Replace the dense FFN sublayer at depth $\ell$ with the routed expert
layer. Let $\boldsymbol{x}_t \in \mathbb{R}^{d_{\text{model}}}$ be the
residual-stream activation at token $t$. Define $E$ expert FFNs
$\{ \mathrm{FFN}_e \}_{e=1}^{E}$, each typically a SwiGLU layer (see
`kb/notes/architecture/ffn.md`):

$$\mathrm{FFN}_e(\boldsymbol{x}) = (\mathrm{Swish}_1(\boldsymbol{x} W_1^{(e)}) \otimes \boldsymbol{x} V^{(e)}) W_2^{(e)}$$

where $W_1^{(e)}, V^{(e)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$,
$W_2^{(e)} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$,
exactly as in the dense SwiGLU FFN
`[shazeer2020 §2 Eq.6; kb/excerpts/shazeer2020#sec-2-eq6]`.

The **router** is a learned linear projection
$W_g \in \mathbb{R}^{d_{\text{model}} \times E}$ producing per-expert
gate logits and (typically) a top-$k$ selection:

$$\boldsymbol{g}_t = W_g \boldsymbol{x}_t \in \mathbb{R}^E \tag{1}$$
$$\mathcal{T}_t = \text{TopK}_k(\boldsymbol{g}_t) \subset \{1, \ldots, E\}, \quad |\mathcal{T}_t| = k \tag{2}$$
$$p_{t,e} = \frac{\exp(g_{t,e})}{\sum_{e' \in \mathcal{T}_t} \exp(g_{t,e'})} \quad \text{for } e \in \mathcal{T}_t \tag{3}$$
$$\mathrm{MoE}(\boldsymbol{x}_t) = \sum_{e \in \mathcal{T}_t} p_{t,e}\, \mathrm{FFN}_e(\boldsymbol{x}_t) \tag{4}$$

Compute per token: $k$ FFN evaluations + 1 router projection. Parameters
per layer: $E \cdot 3 d_{\text{model}} d_{\text{ff}}$ (vs $3 d_{\text{model}} d_{\text{ff}}$ dense). Total parameters
multiplied by $E/k_{\text{eff}}$ where $k_{\text{eff}}$ accounts for both
the per-token cost and any "shared" expert layered alongside the routed
ones (§3).

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $E$ | total number of experts in the layer |
| $k$ | number of activated experts per token (top-$k$ routing) |
| $W_g$ | router projection, learned |
| $\boldsymbol{g}_t$ | router logits for token $t$ |
| $\mathcal{T}_t$ | set of activated experts for token $t$ |
| $p_{t,e}$ | post-softmax routing weight |
| $E_a$ | effective active experts per token, $E_a = k$ for token-choice routing |

### 1.1 Token-choice vs expert-choice routing

The above is **token-choice top-$k$ routing**: each token picks $k$
experts. Expert-choice routing (Zhou et al. 2022, arXiv 2202.09368)
inverts the picture: each expert picks the top-$T$ tokens it will
process. This guarantees uniform load by construction (every expert
processes exactly $T$ tokens) but breaks the autoregressive property
(an expert's choice depends on all tokens in the batch, not just past
ones), so it is rarely used in production decoder-only LLMs. Most
deployed MoE LLMs use token-choice top-1 (Switch) or top-2 (Mixtral) or
top-8 (DeepSeek-V3, Qwen3) routing.

## 2. Mechanism — load balancing, capacity, and the auxiliary loss

The routing function in §1 is differentiable through $p_{t,e}$ on
selected experts, but the **selection step** (top-$k$) is discrete. Two
problems emerge during training:

1. **Routing collapse**: the router learns to send most tokens to a few
   "favored" experts; the rest go cold and never improve. Training is
   self-reinforcing — overused experts get more gradient, underused ones
   get less.
2. **Load imbalance**: even without collapse, GPUs hosting cold experts
   sit idle; GPUs hosting hot experts oversubscribe. With expert
   parallelism (one expert per device), throughput is bounded by the
   slowest device.

The standard mitigation is a **load-balancing auxiliary loss**, added
to the language-modeling loss with coefficient $\alpha$.

### 2.1 Switch Transformer auxiliary loss (Fedus et al. 2021)

For a batch of $T$ tokens, define:

$$f_e = \frac{1}{T} \sum_{t=1}^T \mathbb{1}[e \in \mathcal{T}_t] \quad \text{(fraction of tokens routed to expert } e\text{)}$$
$$P_e = \frac{1}{T} \sum_{t=1}^T \frac{\exp(g_{t,e})}{\sum_{e'} \exp(g_{t,e'})} \quad \text{(mean router probability for expert } e\text{)}$$
$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{e=1}^E f_e P_e \tag{5}$$

Minimized at $f_e = P_e = 1/E$ for all $e$ (uniform load). The product
$f_e P_e$ keeps the loss differentiable: gradients flow through $P_e$
even though $f_e$ depends on the discrete top-$k$
[Fedus et al. 2021 §2.2; PDF not accessible in this pass — equation
form verified against secondary references including Mixtral and
DeepSeekMoE].

### 2.2 Expert capacity and token dropping

To bound the maximum tokens-per-expert per batch, Switch and Mixtral
use a **capacity factor** $C$. Each expert accepts at most
$\lceil C \cdot T k / E \rceil$ tokens; tokens routed past the limit
**bypass the MoE layer** (residual passes through unchanged). $C = 1$
is the lossless minimum; $C = 1.25$ or $C = 2$ is typical to absorb
imbalance. Token dropping is purely an efficiency artifact and degrades
quality at small $C$.

### 2.3 Auxiliary-loss-free load balancing (DeepSeek-V3)

DeepSeek-V3 (Dec 2024) introduces an auxiliary-loss-free strategy.
Instead of penalizing imbalance, the router maintains a per-expert
**bias** $b_e$ that is updated continuously based on observed routing
load:

$$\boldsymbol{g}_t' = W_g \boldsymbol{x}_t + \boldsymbol{b}, \quad b_e \leftarrow b_e + \gamma \cdot (\bar{f} - f_e) \tag{6}$$

where $\bar{f} = 1/E$ is the target uniform load, $\gamma$ is a small
update step. Overloaded experts get their bias decreased (so future
tokens go elsewhere); underloaded experts get a bias increase. The bias
is added at the *router* but not at the *gating weight* (the
$p_{t,e}$ in Eq. 3), so it changes routing without changing the
contribution magnitude. Reported in DeepSeek-V3 §2 as removing the
quality regression historically associated with high $\alpha$
auxiliary loss [DeepSeek-V3 PDF arXiv:2412.19437; not accessible in
this pass — verify claim against original].

### 2.4 Global-batch load balancing (Qwen3, May 2025)

Qwen3 reports that DeepSeek-style auxiliary-loss-free balancing alone
is insufficient when no shared experts are used (§3.2 below). They
introduce a **global-batch** load-balancing loss applied across the
entire training mini-batch (not just within a microbatch on one device),
keeping experts uniformly loaded across the cluster
[Qwen3 PDF arXiv:2505.09388; not accessible — verify].

## 3. Variants and lineage

The space of MoE design choices in 2024–2026 is large. Three axes
matter most.

### 3.1 Coarse vs fine-grained experts

| Era | Pattern | Example |
|---|---|---|
| Switch (2021) | $E = 32$–$2048$ experts of FFN-typical size; top-1 routing | Switch-Base, Switch-Large |
| Mixtral (2024) | $E = 8$ experts, top-2; experts are "full-sized" FFNs | Mixtral 8×7B, Mixtral 8×22B |
| **DeepSeekMoE / fine-grained** (2024) | $E$ in hundreds; each expert is a *narrower* FFN; top-$k \in [6, 8]$ | DeepSeek-V2 (160 routed), DeepSeek-V3 (256 routed), Qwen3 (128 routed) |

The DeepSeekMoE paper (Dai et al. 2024, arXiv 2401.06066) argues that
**fine-grained expert segmentation** — many narrow experts vs few wide
ones — increases the combinatorial expressiveness of the active set.
With $E$ experts and $k$ activated, the number of distinct active
combinations is $\binom{E}{k}$:

| Model | $E$ | $k$ | $\binom{E}{k}$ |
|---|---|---|---|
| Switch (top-1) | 32 | 1 | 32 |
| Mixtral 8×7B | 8 | 2 | 28 |
| DeepSeek-V2 | 160 | 6 | $\sim 2 \times 10^{10}$ |
| DeepSeek-V3 | 256 | 8 | $\sim 4 \times 10^{14}$ |

[INTUITION] The combinatorial-expressiveness argument is the standard
DeepSeekMoE motivation; how much of this combinatorial space is
actually exploited (vs. the router using only a small subset of
combinations) is an empirical question that depends on the auxiliary-
loss balance.

### 3.2 Shared experts vs no shared experts

DeepSeek-V2 introduced **shared experts**: $E_s$ "always-on" experts
that process every token in addition to the $k$ routed ones. The total
active expert count per token is $k + E_s$. The motivation: shared
experts capture common knowledge that all tokens need (general syntax,
vocabulary), freeing routed experts to specialize on rarer patterns
[DeepSeek-V2 §2.2.2; PDF accessible — see kb/excerpts/deepseek-v2.md
for MLA; the MoE portion is not yet excerpted].

DeepSeek-V3 has $E_s = 1$ shared + $k = 8$ routed = 9 active experts
per token from $E = 256$ routed total. **Qwen3 dropped shared experts
entirely** in May 2025, replacing them with global-batch load balancing
to prevent the imbalance that shared experts would otherwise mask.

[CONTRADICTION] As of mid-2025, the literature is split on whether
shared experts are net beneficial. DeepSeek says yes (and reports on
loss regressions when ablating), Qwen3 says no (and reports comparable
quality without). Independent ablations across model scales and data
regimes are rare; treat as unresolved.

### 3.3 Routed-only vs hybrid (some FFN sublayers dense)

Most modern MoE LLMs do **not** make every FFN sublayer MoE — typically
only a subset (every other layer, or only the deeper half) of FFN
sublayers are MoE; the rest remain dense. Mixtral makes every sublayer
MoE; DeepSeek-V3 makes every FFN sublayer except the first one MoE
(the first stays dense as an "input adaptation" layer); LLaMA 4 alternates
dense and MoE FFN sublayers ([Phase 1 sweep, transformer-overview]; verify
exact pattern against tech reports).

### 3.4 Comparison: MoE in production frontier models (2024–2026)

| Model | Total params | Active params | $E$ routed | $E_s$ shared | $k$ | LB strategy | Source |
|---|---|---|---|---|---|---|---|
| Switch-XXL (2021) | ~395B | ~12B | 64 | 0 | 1 | aux-loss | fedus2021-switch |
| Mixtral 8×7B (2024) | 46.7B | 12.9B | 8 | 0 | 2 | aux-loss | mixtral2024 |
| Mixtral 8×22B (2024) | 141B | 39B | 8 | 0 | 2 | aux-loss | mixtral2024 |
| DeepSeek-V2 (2024) | 236B | 21B | 160 | 2 | 6 | aux-loss | deepseek-v2 |
| DeepSeek-V3 (2024) | 671B | 37B | 256 | 1 | 8 | aux-loss-free + bias | deepseek-v3 |
| Qwen3-235B (2025) | 235B | 22B | 128 | 0 | 8 | global-batch LB | qwen3 |
| LLaMA 4 Scout (2026) | ~109B | ~17B | 16 | 1 | 1 | (early-fusion + iRoPE; details TBD) | meta-llama4 |
| LLaMA 4 Maverick (2026) | ~400B | ~17B | 128 | 1 | 1 | (TBD) | meta-llama4 |
| Gemini 2.5 (2025) | undisclosed | undisclosed | undisclosed | undisclosed | undisclosed | undisclosed | gemini2-5 |

[CONTRADICTION] Some columns above are reconstructed from secondary
reporting (largo.dev 2026, model-card excerpts). Verify exact $E$, $k$,
$E_s$ before citing in formal output.

## 4. Intuitions and analogies

[ANALOGY] An MoE layer is a **soft committee of specialists**. Each
token presents itself to a router, which inspects it and dispatches to
the few experts most relevant. The top-$k$ output is a weighted vote:
each chosen expert produces a candidate update $\mathrm{FFN}_e(\boldsymbol{x})$,
and the gating weights $p_{t,e}$ blend them. The committee is "soft"
because the weights are continuous; the dispatching is "hard" because
the top-$k$ selection is discrete. The analogy returns to canonical
form in §1 Eq. (4): the weighted sum is exactly $\sum_e p_{t,e} \mathrm{FFN}_e(\boldsymbol{x})$
over $e \in \mathcal{T}_t$.

[INTUITION] **Why MoE works at all.** A dense FFN at width $d_{\text{ff}} = 4 d_{\text{model}}$
spends every parameter on every token. But most tokens only need a
small subspace of "useful" knowledge — a token "the" doesn't need code-
generation features active. MoE makes this **explicit**: the router
learns to send code tokens to code-shaped experts and prose tokens to
prose-shaped experts. Total parameters can grow much faster than per-
token compute, because the parameters are *amortized across tokens with
different needs* rather than *amortized across the same use of every
parameter on every token*.

[ANALOGY] **Fine-grained vs coarse experts as committee size.** Mixtral's
8 experts is like a board of 8 generalists who take votes; DeepSeek-V3's
256 experts is like 256 highly-specialized researchers. With 256 you
need a much better dispatcher (more router capacity, careful load
balancing), but each chosen researcher contributes deeper specialty.
Whether the dispatcher's discrimination is the bottleneck depends on
training data diversity. [SPECULATION] At very large $E$, the router's
$d_{\text{model}}$-dim input may lack the bandwidth to discriminate
$E = 256$ targets — this could explain why scaling $E$ further (e.g.,
$E = 1024$) has not been reported to deliver proportional gains.

[INTUITION] **The shared-expert debate as bias-variance.** Shared
experts are a regularization device: they ensure that common knowledge
is always represented, which lowers the variance of the routed-expert
output. But they also reduce the effective per-token compute available
to specialization (since $E_s$ active expert FFNs are spent on common
features). Qwen3's "drop shared experts, balance globally" is the
high-variance high-specialization end; DeepSeek-V3's "$E_s = 1$,
strong-aux-loss-free balancing" is in the middle.

## 5. Frontier and open questions (as of 2026-05)

- **Auxiliary-loss-free load balancing — does it generalize?** DeepSeek-V3
  reports the bias-update strategy (§2.3) recovers full quality. Qwen3
  uses global-batch LB. As of May 2026, no third strategy has emerged;
  the loss-free family is the new default but theoretical understanding
  is shallow.
- **Optimal $E$, $k$, $E_s$ for a given compute budget.** No published
  scaling law for MoE design points. Hoffmann-style Chinchilla-optimal
  ratios are dense-only. [CONTRADICTION] Mixtral, DeepSeek, and Qwen3
  picked very different design points with no public head-to-head
  ablation; what makes each defensible at its scale is empirical.
- **Expert specialization interpretability.** Beyond Benchmarks
  (arXiv 2509.23933) probes what individual MoE experts learn —
  preliminary results suggest specialization is real but not
  cleanly task-aligned (an "expert" doesn't correspond to "the math
  expert"). See `kb/notes/interpretability/mechanistic-interpretability.md`.
- **MoE × MLA × multimodal interactions.** DeepSeek-V3 stacks MLA + MoE +
  MTP; LLaMA 4 stacks iRoPE + early-fusion + MoE. The systems
  engineering is well-described; the *theoretical* interactions —
  whether MoE specialization helps or hurts MLA's low-rank cache, etc.
  — are not.
- **Inference-time MoE serving.** Expert parallelism, expert offload to
  CPU, and speculative decoding interactions with MoE are active.
  Belongs in `kb/notes/inference/serving-systems.md`.

## 6. See also

- `kb/notes/architecture/ffn.md` — the per-expert SwiGLU FFN building
  block. Each MoE expert is just a SwiGLU layer.
- `kb/notes/architecture/transformer-overview.md` — where MoE sits in
  the block; pattern of dense vs MoE FFN sublayers across depth.
- `kb/notes/architecture/multi-token-prediction.md` — DeepSeek-V3's
  MTP heads stack on top of the MoE backbone.
- `kb/notes/inference/serving-systems.md` — expert parallelism, vLLM /
  SGLang routing-aware scheduling, MoE-aware speculative decoding.
- `kb/notes/training/distributed-training.md` — expert parallelism and
  all-to-all communication patterns dominate MoE training; this is
  where most of the systems complexity sits.
