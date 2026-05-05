---
topic: inference/structured-output
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - willard2023-outlines
  - xgrammar2024
  - zheng2024-sglang
secondary_sources:
  - beurer-kellner2023-lmql
related_topics:
  - inference/serving-systems
  - inference/speculative-decoding
  - inference/kv-cache-management
---

# Structured output (constrained decoding)

**Constrained decoding** is the procedure that, at each generation
step, masks the model's output logits to allow only tokens consistent
with a target grammar (regex, JSON schema, GBNF, or arbitrary CFG).
The naive implementation queries the grammar engine for every token in
the vocabulary at every step, which becomes the inference bottleneck on
a 100K+ vocabulary model — taking hundreds of microseconds to
milliseconds per token, on the same order as the model's forward pass
itself.

The 2023–2026 era's solutions all factor the vocabulary into
**context-independent** (precomputable: token-allowed-or-not based
purely on grammar state) and **context-dependent** (must be checked at
runtime, tiny fraction of vocabulary), and the FSM into compressed
forms that batch deterministic chains. By early 2026, **XGrammar** is
the default constrained-decoding backend in vLLM, SGLang, and TensorRT-
LLM `[FORUM-SIGNAL: vLLM/SGLang/TRT-LLM release notes]`.

## 1. Formal definition

### 1.1 The constrained-decoding objective

Given:
- A language model $\pi_\theta$ producing per-step logits
  $\ell_t \in \mathbb{R}^V$ over a vocabulary of size $V$.
- A target language $\mathcal{G}$ (regular, context-free, or schema-
  constrained) defined by a grammar / FSM / parser.
- Grammar state $s_t$ at step $t$ (depends on the tokens emitted so
  far).

Constrained decoding samples token $y_t$ at each step from a **masked
distribution**:

$$
P(y_t = v \mid y_{<t}) = \frac{\mathbb{1}[v \in \mathcal{V}_{\text{allowed}}(s_t)] \cdot \exp(\ell_{t,v})}{\sum_{v' \in \mathcal{V}_{\text{allowed}}(s_t)} \exp(\ell_{t,v'})} \tag{1}
$$

where $\mathcal{V}_{\text{allowed}}(s_t) \subseteq \{1,\ldots,V\}$ is
the set of tokens whose acceptance keeps the partial output in
$\mathcal{G}$.

| Symbol | Meaning |
|---|---|
| $\ell_t$ | model logits at step $t$ |
| $V$ | vocabulary size (typical: 32K–256K) |
| $s_t$ | grammar / FSM state at step $t$ |
| $\mathcal{V}_{\text{allowed}}(s_t)$ | set of tokens valid in state $s_t$ |
| $\mathcal{G}$ | target language (regex, CFG, JSON schema) |

The mask in Eq. (1) is implemented as $\ell_{t,v} \leftarrow -\infty$
for $v \notin \mathcal{V}_{\text{allowed}}(s_t)$ before sampling.

### 1.2 The Willard–Louf reformulation: tokens as FSM transitions

Willard & Louf 2023 `[willard2023-outlines]` reframe constrained
generation as **transitions in an FSM whose alphabet is the model's
vocabulary**. The standard FSM operates over characters; their
contribution is to **construct an index** mapping each FSM state $s$
to the set of vocabulary tokens whose string value is consistent with
some path through $s$. The index is built **once per grammar**, before
generation.

At each generation step:
1. Look up the current FSM state $s_t$.
2. Retrieve the precomputed token set
   $\mathcal{V}_{\text{allowed}}(s_t)$ from the index.
3. Mask logits per Eq. (1).
4. Sample $y_t$; transition the FSM by $|y_t|$ characters; update $s_{t+1}$.

This adds **$O(1)$ amortized overhead per token** vs. the per-vocab-
per-step grammar query of naive constrained decoding `[willard2023-
outlines abstract; kb/excerpts/willard2023-outlines#abstract]`. The
FSM construction extends to context-free grammars via LALR(1) parsers
for popular formats (JSON, Python, SQL).

### 1.3 The XGrammar partition: context-independent vs context-dependent

XGrammar `[xgrammar2024 §3]` extends Willard–Louf with the
observation that for typical grammars (JSON schemas, regex), the
**vast majority** of tokens are **context-independent**: their
allowed/disallowed status depends only on the FSM state, not on the
specific stack contents in a pushdown automaton. The remaining tiny
fraction (~1% in published benchmarks) is **context-dependent** and
must be checked at runtime.

XGrammar's implementation:
1. **Precompute** the per-FSM-state allowed-token bitmask for context-
   independent tokens. Stored as compressed bitmaps over the vocabulary.
2. **At runtime**, for each generation step:
   - Look up the precomputed bitmask: $O(V/64)$ bytes accessed.
   - Check the small set of context-dependent tokens against the
     pushdown stack: $O(\text{small})$ check per token.
3. Mask logits and sample.

Reported throughput: **≤40 µs per token** for grammar checking
overhead on 100K-vocab models — vs. hundreds of µs to milliseconds for
a naive grammar engine `[xgrammar2024 §abstract, §6]`. This brings
constrained decoding from "potentially the bottleneck" to "no longer
the bottleneck."

## 2. Mechanism — the four implementation paths

### 2.1 Per-step grammar query (naive)

For each step:
1. For each $v \in \{1, \ldots, V\}$:
   - Append token $v$ to the current output, run grammar parser, check
     acceptance.
   - If accepted, mark $v$ allowed.
2. Mask logits. Sample.

Cost: $O(V \cdot c_{\text{parse}})$ per step, where $c_{\text{parse}}$
is the cost of an acceptance check. For $V = 128\text{K}$ and
$c_{\text{parse}} \sim 1\mu$s, this is $\sim 100$ms — typically
dominating the model's forward pass.

This is **the baseline** that `outlines`-style libraries
(pre-Willard–Louf) implemented. It is correct but slow.

### 2.2 Outlines / Willard–Louf — FSM index

Outlines `[willard2023-outlines]` is the canonical implementation of
the FSM-index approach. Build steps:

1. Compile grammar $\mathcal{G}$ to a deterministic FSM over **bytes**
   (or unicode codepoints).
2. For each FSM state $s$, enumerate the set of tokens
   $\mathcal{V}_{\text{allowed}}(s) = \{v : $ the FSM accepts the byte
   sequence of $v$ starting from $s$, ending in some valid state $\}$.
3. Store the per-state token sets as the **index**.

Generation:
1. Maintain current FSM state $s_t$.
2. Retrieve $\mathcal{V}_{\text{allowed}}(s_t)$ in $O(1)$.
3. Mask, sample, transition.

The index size depends on grammar and vocabulary. For typical JSON
schemas it is tractable; for very large vocabularies and very deep
grammars it can be megabytes.

The Willard–Louf paper extends the FSM construction to **CFGs via
LALR(1) parsers** `[willard2023-outlines §4]`. The idea: at each FSM
state, the parser's stack is part of the context, so the index is
keyed on (state, stack-summary). For the common cases (JSON, Python,
SQL), the stack-summary is small enough that the index remains
practical.

### 2.3 SGLang compressed FSM — multi-token decoding inside fixed text

SGLang `[zheng2024-sglang §4; kb/excerpts/zheng2024-sglang#sec-4]`
adds a complementary trick: when the FSM forces a *deterministic chain*
of transitions (e.g., the literal `{"summary": "` at the start of a
JSON schema's required-field response), multiple tokens can be decoded
at once.

Procedure:
1. Compile grammar to FSM.
2. **Compress** singular-transition chains: a chain of FSM states
   $s_1 \to s_2 \to \cdots \to s_k$ where each transition has exactly
   one allowed input is replaced by a single edge with the entire
   chain as label.
3. At runtime, when a compressed edge is entered, decode all $k$
   tokens *in one forward pass* (the model produces them all
   deterministically anyway; we just don't need to query the grammar
   per token).

`[zheng2024-sglang §4; kb/excerpts/zheng2024-sglang#sec-4]`:

> the constant sequence `{"summary": "` in Fig. 2 spans multiple
> tokens in the normal decoding process …, requiring multiple
> decoding stages, even though there is only one valid next token
> when decoding it. … SGLang overcomes this limitation by creating a
> fast constrained decoding runtime with a compressed FSM. This
> runtime analyzes the FSM and compresses adjacent singular-
> transition edges in the FSM into single edges …, allowing it to
> recognize when multiple tokens can be decoded together.

This is a **forward-pass throughput** optimization (decode $k$ tokens
in one pass), orthogonal to XGrammar's **per-step overhead**
optimization (cheaper grammar query).

### 2.4 XGrammar — context-independent / context-dependent partition

XGrammar `[xgrammar2024 §3]` is the Willard–Louf approach extended
with the precomputable / runtime partition:

1. **Compile time**: build the deterministic pushdown automaton (PDA)
   for the schema. For each PDA state, compute the bitmask of
   context-independent allowed tokens (i.e., tokens whose
   allowed/disallowed status depends only on the current state, not
   on the stack contents).
2. **Identify context-dependent tokens**: the small set of tokens
   whose allowed/disallowed status depends on stack contents (e.g.,
   tokens that close a nested structure).
3. **Runtime**: for each step, fetch the precomputed bitmask
   ($O(V/64)$ words from cache); for the context-dependent set, run a
   small parser check.

The reported partition: ~99% context-independent, ~1% context-
dependent for typical JSON schemas `[xgrammar2024 §6]`. The 1%
runtime check is cheap because the set is small. Compile time:
$\sim$10–100 ms for a typical schema; XGrammar-2 reportedly reduces
this to ~10 ms for agentic (rapidly-changing) workloads
`[FORUM-SIGNAL: arXiv:2601.04426]`.

### 2.5 LMQL — DSL approach

LMQL `[beurer-kellner2023-lmql]` takes a different angle: rather than
optimizing the per-step constraint check, it provides a **query
language for LLMs** with constraint-aware execution:

```python
'List 3 capital cities: [city1], [city2], [city3]'
where len(city1) < 30 and len(city2) < 30 and len(city3) < 30
```

The runtime translates the constraints into per-step masks. LMQL
emphasizes ergonomics (constraints expressed close to the prompt) over
raw constrained-decoding throughput. Its constrained-decoding engine
is comparable to outlines in speed; the contribution is the DSL.

## 3. Variants and lineage

### 3.1 Comparison: 2024–2026 constrained-decoding engines

| Engine | Year | Approach | Context-indep. precompute | Compressed FSM | Default in |
|---|---|---|---|---|---|
| **outlines** `[willard2023-outlines]` | 2023 | FSM index over vocabulary | partial | no | HF Transformers integrations |
| **LMQL** `[beurer-kellner2023-lmql]` | 2022–23 | DSL + per-step mask | no | no | research / DSL-first usage |
| **SGLang compressed-FSM** `[zheng2024-sglang §4]` | 2024 | FSM + multi-token decode | basic | **yes** | SGLang |
| **llama.cpp GBNF** | 2023+ | grammar-based decoding | no | no | llama.cpp ecosystem |
| **llguidance** (Microsoft) | 2024 | Earley parser + compressed | yes (per Microsoft talk) | yes | Guidance, MS internal |
| **XGrammar** `[xgrammar2024]` | 2024 | PDA + ctx-indep bitmask + ctx-dep runtime | **yes** | yes | vLLM, SGLang (default), TRT-LLM |
| **XGrammar-2** | 2026 | XGrammar + agentic compile | yes | yes | agentic workloads |

The progression: per-step query → FSM index → ctx-indep partition →
agentic compile. Each step shaves another order of magnitude off the
constrained-decoding overhead.

### 3.2 Schema benchmarks — JSONSchemaBench

`[FORUM-SIGNAL: arXiv:2501.10868 — JSONSchemaBench]` provides a
benchmark of 10K real-world JSON schemas, adopted across Guidance,
outlines, llama.cpp, XGrammar, OpenAI, and Gemini. Results (per the
Phase 1 sweep) put XGrammar and llguidance in the same tier, both
substantially ahead of outlines and llama.cpp on average per-token
overhead. SGLang's compressed-FSM has a separate niche on schemas with
long literal substrings.

### 3.3 Two orthogonal axes

The optimization space factors into two largely orthogonal axes:

| Axis | What's optimized | Examples |
|---|---|---|
| **Per-token grammar query cost** | $V$ comparisons → $O(1)$ | outlines / Willard–Louf, XGrammar's ctx-indep bitmask |
| **Forward-pass tokens-per-step** | 1 token / pass → $k$ tokens / pass on deterministic chains | SGLang compressed-FSM, plus speculative-decoding interactions |

XGrammar wins on the first; SGLang's compressed-FSM wins on the
second. Modern stacks typically combine both.

### 3.4 JSON mode mechanisms

OpenAI / Anthropic "JSON mode" / "function calling" features are
implemented via constrained decoding under the hood. Mechanisms in
common use:

1. **Logit bias** — coarse: zero or strong negative logit on tokens
   not consistent with JSON. Fast but allows malformed output if not
   precise.
2. **FSM-constrained decoding** — exact: only tokens valid by the JSON
   FSM are sampled. Mainstream as of 2024+.
3. **Grammar-based decoding (GBNF)** — grammar specified in BNF; FSM
   constructed at runtime. llama.cpp's mechanism.
4. **Schema-driven** — JSON schema → FSM via Willard–Louf or XGrammar
   lowering. Production default in vLLM/SGLang/TRT-LLM.

Native API support (OpenAI's `response_format={"type": "json_schema",
...}`) typically uses option 4 underneath. The user-facing semantics
("guaranteed valid JSON conforming to schema") are exact when the
backend uses true FSM-constrained decoding; weaker (probabilistic)
when only logit bias is used.

## 4. Intuitions and analogies

[ANALOGY] **Constrained decoding is a FSM-product with the language
model.** The model's output is normally a probability distribution
over $V$ tokens. Constraining it to a grammar is the **product** of
the FSM (which says which tokens are allowed) with the model
(which says which tokens are likely). Returns to canonical form: the
masked distribution in Eq. (1) is exactly the product of the model's
softmax and the FSM's indicator function, renormalized.

[INTUITION] **Why context-independent / context-dependent partition
works.** Most tokens in a JSON schema's allowed/disallowed status
are determined purely by the current state — e.g., "after `{"`, the
next allowed tokens are field-name characters." Only a small set of
tokens (closing brackets, deeply nested punctuation) depends on the
*stack* (how nested are we?). Bitmasking the state-determined ones
covers ~99% of vocabulary lookups; the runtime check on the
stack-dependent ~1% is cheap because the set is small. Returns to
Eq. (1): $\mathcal{V}_{\text{allowed}}(s_t)$ is the **union** of a
state-only-determined set (bitmask lookup) and a stack-determined set
(small parser check).

[ANALOGY] **SGLang's compressed FSM is to constrained decoding what
speculative decoding is to autoregressive sampling.** Both detect
*deterministic continuations* and execute multiple tokens in parallel
or in a single forward pass. Speculative decoding uses a draft model;
SGLang's compressed FSM uses the grammar's deterministic structure.
Returns to math: in both cases the model's output distribution
collapses to a near-delta (probability concentrated on one token), so
the model's "decision" each step is trivial; the bottleneck is the
forward pass itself, which can be amortized across deterministic
runs.

[INTUITION] **Why XGrammar replaced outlines as the default
backend.** Outlines was the breakthrough — Willard–Louf's FSM-index
was the canonical reformulation that made constrained decoding cheap
in principle. XGrammar productionized it by adding (a) the context-
indep / context-dep partition, (b) compressed bitmask storage, (c)
agentic compile-time targets. The improvements are quantitative
(1000× faster compile, 10× faster runtime) but together they cross
the threshold from "useful research" to "no longer the bottleneck."

[CONTRADICTION] **Whether constrained decoding harms model quality.**
Two views in 2024–25 literature:
- **Helpful**: forcing JSON validity prevents downstream parsing
  errors; downstream task scores improve `[FORUM-SIGNAL: 2024 outline
  blog posts]`.
- **Harmful**: constraining the model can *suppress* the most
  semantically appropriate token if it is unfortunately also the most
  syntactically invalid; reasoning quality drops on free-form math
  benchmarks `[FORUM-SIGNAL: arXiv:2408.02442 'JSON mode harms
  reasoning']`.

The truth is task-dependent: pure JSON-emit tasks are unharmed;
free-form reasoning tasks where the constraint conflicts with the
model's preferred output style can be harmed. Mitigations include
"constraint-aware prompting" (tell the model the constraint exists,
let it reason in free text first, emit constrained JSON last).

## 5. Frontier and open questions (as of 2026-05)

### 5.1 Constrained decoding × speculative decoding interaction

Both XGrammar-2 and recent papers (`[FORUM-SIGNAL: arXiv:2603.03305]`)
explore the joint optimization. Naive composition is non-trivial: the
draft model in speculative decoding may emit tokens the grammar
rejects, breaking the speculation. Open question: how to design a
draft model or speculation procedure that is grammar-aware? Several
approaches in 2025–26: grammar-aware drafting, post-hoc rejection
sampling with grammar check, separate grammar-respecting draft model.
None canonical yet.

### 5.2 Streaming JSON and partial results

For long JSON outputs, streaming partial parses to the client is
useful. Open question: how to *commit* a JSON prefix (so the client
sees it) while preserving the option to reject the rest if the model
errors. Current systems mostly buffer full outputs.

### 5.3 Probabilistic vs exact constraint enforcement

Logit-bias is a probabilistic constraint (allows occasional
violations); FSM-constrained is an exact constraint (zero violations).
[CONTRADICTION] OpenAI's `json_schema` response format claims exact
guarantees in docs but reportedly has occasional violations in
practice `[FORUM-SIGNAL: 2025 community reports]`. The gap is
between the theoretical guarantee and the implementation; full
audit-level exactness on long deeply-nested schemas remains
engineering-fragile.

### 5.4 [CONTRADICTION] Does constrained decoding hurt reasoning?

`[FORUM-SIGNAL: arXiv:2408.02442]` reports "JSON mode harms
reasoning"; counter-papers report no harm on reasoning when the
constraint allows free-form thinking before the structured emit.
The discrepancy is about *when* the constraint is enforced: if the
whole response is JSON-constrained, reasoning suffers; if the model
can think in free text and emit a JSON-shaped final answer, it
doesn't. Best practice as of 2026: defer constraints to a final
"answer" field; let the model think in free text first.

### 5.5 Beyond JSON — programming languages and DSLs

Constrained decoding to *programming languages* (Python, SQL,
TypeScript) is more demanding: grammars are larger, semantic
constraints (type-checking, scope) are not expressible in CFG.
Outlines and XGrammar handle the syntactic side; the semantic side
(only-defined-variables, only-imported-modules) is handled at
runtime by additional checks or by post-hoc filtering. Open: a
unified framework for "syntactic + semantic constrained decoding."

### 5.6 Agentic workloads

Agent control loops (tool-use, multi-step planning) construct grammars
**dynamically**: each tool call has its own schema, the schema may be
generated on-the-fly. XGrammar's $\sim$100 ms compile time was
acceptable for static schemas; XGrammar-2 reduces this to ~10 ms for
agentic workloads `[FORUM-SIGNAL: arXiv:2601.04426]`. Open: at what
threshold does compile time stop mattering and runtime overhead become
the bottleneck again?

## 6. See also

- `kb/notes/inference/serving-systems.md §4.3` — SGLang's compressed-
  FSM treated briefly there; this note owns the technical detail.
- `kb/notes/inference/speculative-decoding.md` — speculative decoding
  × constrained decoding interaction.
- `kb/notes/inference/kv-cache-management.md` — RadixAttention's KV
  reuse can amplify constrained-decoding speedup when constrained
  outputs share prefixes (typical for repeated-schema agent calls).
