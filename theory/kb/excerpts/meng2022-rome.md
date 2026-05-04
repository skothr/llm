---
paper_key: meng2022-rome
title: "Locating and Editing Factual Associations in GPT (ROME)"
authors: Meng, Bau, Andonian, Belinkov
year: 2022
venue: NeurIPS
arxiv: 2202.05262
local_pdf: theory/sources/papers/meng2022_rome.pdf
type: excerpts
note: Verbatim quotations from the v5 arXiv PDF (Jan 2023). ROME is the paper that introduced *causal tracing* (the first systematic activation-patching protocol on a real-world LM task) to the wider mech-interp community, and used the resulting localization to derive a closed-form rank-one MLP weight edit. The activation-patching idea predates ROME (Vig 2020 mediation analysis); ROME's contribution is the application + formalization with three named runs (clean / corrupted / corrupted-with-restoration) that everyone now uses.
---

# Excerpts — Meng et al. 2022, "Locating and Editing Factual Associations in GPT" (ROME)

## Abstract — what ROME claims {#abstract}

> We analyze the storage and recall of factual associations in
> autoregressive transformer language models, finding evidence that these
> associations correspond to localized, directly-editable computations.
> We first develop a causal intervention for identifying neuron
> *activations* that are decisive in a model's factual predictions. This
> reveals a distinct set of steps in middle-layer feed-forward modules
> that mediate factual predictions while processing subject tokens. To
> test our hypothesis that these computations correspond to factual
> recall, we modify feed-forward *weights* to update specific factual
> associations using Rank-One Model Editing (ROME).

## §1 Introduction — two interventions, one hypothesis {#sec-1}

> In this paper, we investigate how such factual associations are stored
> within GPT-like autoregressive transformer models. […] We use two
> approaches. First, we trace the causal effects of hidden state
> *activations* within GPT using causal mediation analysis (Pearl, 2001;
> Vig et al., 2020b) to identify the specific modules that mediate recall
> of a fact about a subject (Figure 1). Our analysis reveals that
> feedforward MLPs at a range of middle layers are decisive when
> processing the last token of the subject name (Figures 1b,2b,3).
> Second, we test this finding in model *weights* by introducing a
> Rank-One Model Editing method (ROME) to alter the parameters that
> determine a feedforward layer's behavior at the decisive token.

## §2 Causal Tracing — the three-run protocol {#sec-2}

> The grid of states (Figure 1) forms a *causal graph* (Pearl, 2009)
> describing dependencies between the hidden variables. […] We observe
> all of $G$'s internal activations during three runs: a *clean* run that
> predicts the fact, a *corrupted* run where the prediction is damaged,
> and a *corrupted-with-restoration* run that tests the ability of a
> single state to restore the prediction.
>
> - In the **clean run**, we pass a factual prompt $x$ into $G$ and
>   collect all hidden activations $\{h_i^{(l)} \mid i \in [1, T], l \in
>   [1, L]\}$. […]
> - In the baseline **corrupted run**, the subject is obfuscated from
>   $G$ before the network runs. Concretely, immediately after $x$ is
>   embedded as $[h_1^{(0)}, h_2^{(0)}, \ldots, h_T^{(0)}]$, we set
>   $h_i^{(0)} := h_i^{(0)} + \epsilon$ for all indices $i$ that
>   correspond to the subject entity, where $\epsilon \sim
>   \mathcal{N}(0; \nu)^4$. […]
> - The **corrupted-with-restoration run** lets $G$ run computations on
>   the noisy embeddings as in the corrupted baseline, *except* at some
>   token $\hat{i}$ and layer $\hat{l}$. There, we hook $G$ so that it
>   is forced to output the clean state $h_{\hat{i}}^{(\hat{l})}$;
>   future computations execute without further intervention.

## §2 Causal-effect quantities — TE, IE, ATE, AIE {#sec-2-effects}

> Let $\mathbb{P}[o]$, $\mathbb{P}_*[o]$, and $\mathbb{P}_{*,\,\text{clean
> } h_i^{(l)}}[o]$ denote the probability of emitting $o$ under the
> clean, corrupted, and corrupted-with-restoration runs, respectively;
> dependence on the input $x$ is omitted for notational simplicity. The
> **total effect** (TE) is the difference between these quantities:
> $\text{TE} = \mathbb{P}[o] - \mathbb{P}_*[o]$. The **indirect effect**
> (IE) of a specific mediating state $h_i^{(l)}$ is defined as the
> difference between the probability of $o$ under the corrupted version
> and the probability when that state is set to its clean version, while
> the subject remains corrupted:
> $\text{IE} = \mathbb{P}_{*,\,\text{clean }h_i^{(l)}}[o] -
> \mathbb{P}_*[o]$.
>
> Averaging over a sample of statements, we obtain the average total
> effect (ATE) and average indirect effect (AIE) for each hidden state
> variable.

This is the formal definition of the **patching effect** that the
follow-up best-practices literature systematizes (cf.
`kb/excerpts/zhang2023-apatching#sec-2-1`).

## §2.2 The two causal-tracing peaks: late site + early site {#sec-2-2}

> We compute the average indirect effect (AIE) over 1000 factual
> statements […] The ATE of this experiment is 18.6%, and we note that a
> large portion of the effect is mediated by strongly causal individual
> states (AIE=8.7% at layer 15) at the last subject token. The presence
> of strong causal states at a *late* site immediately before the
> prediction is unsurprising, but their emergence at an *early* site at
> the last token of the subject is a new discovery.
>
> Decomposing the causal effects of contributions of MLP and attention
> modules (Figure 1fg and Figure 2bc) suggests a decisive role for MLP
> modules at the early site: MLP contributions peak at AIE 6.6%, while
> attention at the last subject token is only AIE 1.6%; attention is
> more important at the last token of the prompt.

## §2.3 The Localized Factual Association Hypothesis {#sec-2-3}

> Based on causal traces, we posit a specific mechanism for storage of
> factual associations: each midlayer MLP module accepts inputs that
> encode a subject, then produces outputs that recall memorized
> properties about that subject. Middle layer MLP outputs accumulate
> information, then the summed information is copied to the last token
> by attention at high layers.
>
> This hypothesis localizes factual association along three dimensions
> (i) in the MLP modules (ii) at specific middle layers (iii) and
> specifically at the processing of the subject's last token.

## §3.1 The MLP-as-associative-memory view {#sec-3-1}

> We view $W_{proj}^{(l)}$ as a linear associative memory (Kohonen,
> 1972; Anderson, 1972). This perspective observes that any linear
> operation $W$ can operate as a key–value store for a set of vector
> keys $K = [k_1 \mid k_2 \mid \ldots]$ and corresponding vector values
> $V = [v_1 \mid v_2 \mid \ldots]$, by solving $W K \approx V$, whose
> squared error is minimized using the Moore–Penrose pseudoinverse:
> $\hat{W} = V K^+$.

The **rank-one update** that the paper actually applies:

$$\text{minimize } \|\tilde{W} K - V\| \text{ such that } \tilde{W} k_* = v_* \quad \text{by setting} \quad \hat{W} = W + \Lambda (C^{-1} k_*)^\top \tag{2}$$

> Here $W$ is the original matrix, $C = K K^\top$ is a constant that we
> pre-cache by estimating the uncentered covariance of $k$ from a sample
> of Wikipedia text […] and $\Lambda = (v_* - W k_*)/(C^{-1} k_*)^\top
> k_*$ is a vector proportional to the residual error of the new
> key–value pair on the original memory matrix. […] Because of this
> simple algebraic structure, we can insert any fact directly once
> $(k_*, v_*)$ is computed.

## §3.2 Empirical headline — zsRE editing {#sec-3-2}

| Editor | Efficacy | Paraphrase | Specificity |
|---|---|---|---|
| GPT-2 XL | 22.2 (±0.5) | 21.3 (±0.5) | 24.2 (±0.5) |
| FT (fine-tuning) | 99.6 (±0.1) | 82.1 (±0.6) | 23.2 (±0.5) |
| FT+L | 92.3 (±0.4) | 47.2 (±0.7) | 23.4 (±0.5) |
| MEND | 75.9 (±0.5) | 65.3 (±0.6) | 24.1 (±0.5) |
| **ROME** | **99.8 (±0.0)** | **88.1 (±0.5)** | **24.2 (±0.5)** |

> ROME is competitive with hypernetworks and fine-tuning methods despite
> its simplicity. We find that it is not hard for ROME to insert an
> association that can be regurgitated by the model. Robustness under
> paraphrase is also strong […]
