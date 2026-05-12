---
paper_key: yao2024-tau-bench
title: "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains"
authors: Yao, Shinn, Razavi, Narasimhan (Sierra Research)
year: 2024
venue: arXiv preprint
arxiv: 2406.12045
local_pdf: theory/sources/papers/yao2024-tau-bench.pdf
type: excerpts
---

# Excerpts — Yao et al. 2024, "tau-bench"

## Abstract {#abstract}

> Existing benchmarks do not test language agents on their interaction
> with human users or ability to follow domain-specific rules, both of
> which are vital for deploying them in real world applications. We
> propose tau-bench, a benchmark emulating dynamic conversations between
> a user (simulated by language models) and a language agent provided
> with domain-specific API tools and policy guidelines. We employ an
> efficient and faithful evaluation process that compares the database
> state at the end of a conversation with the annotated goal state. We
> also propose a new metric (pass^k) to evaluate the reliability of
> agent behavior over multiple trials. Our experiments show that even
> state-of-the-art function calling agents (like gpt-4o) succeed on
> < 50% of the tasks, and are quite inconsistent (pass^8 < 25% in
> retail). Our findings point to the need for methods that can improve
> the ability of agents to act consistently and follow rules reliably.

## §3 Task formulation — POMDP framing and domain structure {#sec-3-formulation}

> Each individual task in tau-bench can be formulated as a partially
> observable Markov decision process (POMDP) $(S, A, O, T, R, U)$ with
> state space $S$, action space $A$, observation space $O$, transition
> function $T : S \times A \to S \times O$, reward function
> $R : S \to [0,1]$, and instruction space $U$. The agent interacts
> with both (1) databases (db) via API tools, and (2) a (simulated) user
> (user) to complete a task, i.e., $S = S_{\mathrm{db}} \otimes
> S_{\mathrm{user}}$, $A = A_{\mathrm{db}} \cup A_{\mathrm{user}}$,
> $O = O_{\mathrm{db}} \cup O_{\mathrm{user}}$.

Domain policy:

> Each domain has a policy that explains the domain databases, task
> procedures, and restrictions for the agent to follow in its
> interactions. Some restrictions are implemented as checks in the API
> ... and others not ... the agent needs to fill in the number of
> baggage items to be paid for in the book_reservation API, similar to
> the freedom given real-world agents.

User simulation:

> We use a language model (gpt-4-0613) to simulate a human user
> interacting with the agent. The user state $s_{\mathrm{user}}$
> consists of an initial system prompt with the task instruction along
> with the entire conversation history between the user and the agent so
> far. The user cannot see the interaction history between the agent and
> API tools.

Reward definition:

> The reward of a task episode $r = r_{\mathrm{action}} \times
> r_{\mathrm{output}} \in \{0,1\}$ is based on (1) whether the final
> database is identical to the unique ground truth outcome database
> ($r_{\mathrm{action}}$), and (2) whether the agent's responses to the
> user contain all necessary information ($r_{\mathrm{output}}$).

## §3 The pass^k metric — reliability vs. ability {#sec-3-passk}

> For tasks like code generation with good verification techniques (unit
> tests), the community has defined the pass@k (pass at k) metric as the
> chance that at least one out of k i.i.d. task trials is successful,
> which captures the trend of agents enabling discovery of solutions with
> scaling of inference-time compute. For real-world agent tasks requiring
> reliability and consistency like customer service, we propose a new
> metric — pass^k (pass hat k), defined as the chance that **all** k
> i.i.d. task trials are successful, averaged across tasks.

The unbiased estimators (verbatim, with $n$ trials, $c$ successes):

$$\mathrm{pass}^k = \mathbb{E}_{\mathrm{task}}\!\left[\binom{c}{k}\!\bigg/\!\binom{n}{k}\right], \quad \mathrm{pass}@k = 1 - \mathbb{E}_{\mathrm{task}}\!\left[\binom{n-c}{k}\!\bigg/\!\binom{n}{k}\right].$$

> [P]ass^k can capture the reliability of the agent at handling
> variations in conversations with the same underlying semantics while
> adhering to the domain policies and rules. By default, we report the
> average reward across tasks, pass^1 = pass@1 = E[r] = E[c/n], as the
> main metric for comparing agents.

[INTUITION: pass@k rewards lucky samples; pass^k punishes any failure —
it is the dual metric suited to production where every user interaction
must succeed, not just one in k.]

## §4.1 Domains — tau-retail and tau-airline {#sec-4-1-domains}

> Using the above procedures, we modularly construct two domains,
> tau-retail and tau-airline. We choose these two domains as they are
> relatively easy to synthesize data (e.g., products, prices, flights)
> and craft policies (e.g., product return, baggage allowance) based on
> common sense, allow for diverse tasks, and are close to real-world
> applications.

Key statistics (Table 1, verbatim):

|  | tau-retail | tau-airline |
|---|---|---|
| Databases | 500 users, 50 products, 1,000 orders | 500 users, 300 flights, 2,000 reservations |
| API tools | 7 write, 8 non-write | 6 write, 7 non-write |
| Tasks | 115 | 50 |

> tau-retail. In this domain, the agent is tasked with helping users
> cancel or modify pending orders, return or exchange delivered orders,
> modify user addresses, or provide information.
>
> tau-airline. Here, the agent has to help users book, modify, or cancel
> flight reservations, or provide refunds. We construct 300 flights
> between 20 US cities with realistic durations and prices, and API tools
> to query direct or one-stop flights. The domain policy is more complex
> than tau-retail, with ad-hoc constraints about combining payment
> methods, checked bag allowance, flight changes and cancellations, etc.

## §5.1 Main results — frontier models below 50% {#sec-5-1-results}

> From Table 2, we see that gpt-4o is the best model with function
> calling, and there is a wide spectrum of performances among various
> models. Notably, SoTA open-weight models (llama-3-70b and
> mistral-8x22b) still have a significant gap to cover with respect to
> SoTA proprietary models (gpt-4o, claude-3-opus). All models are still
> far from solving tau-bench, especially the more challenging tau-airline
> where even gpt-4o solves only 35.2% of the tasks.

Pass^1 results from Table 2 (verbatim, function calling unless noted):

| Model | retail | airline | avg |
|---|---|---|---|
| gpt-4o | 61.2 | 35.2 | 48.2 |
| gpt-4-turbo | 57.7 | 32.4 | 45.1 |
| gpt-4-32k | 56.5 | 33.0 | 44.8 |
| gpt-3.5-turbo | 20.0 | 10.8 | 15.4 |
| claude-3-opus | 44.2 | 34.7 | 39.5 |
| claude-3-sonnet | 26.3 | 27.6 | 27.0 |
| claude-3-haiku | 19.0 | 14.4 | 16.7 |
| gemini-1.5-pro | 21.7 | 14.0 | 17.9 |
| meta-llama-3-70B | 14.8 | 14.4 | 14.6 |

Consistency finding (§5.1, verbatim):

> [T]he chance of reliably and consistently solving the same task
> multiple times significantly drops as the number of trials k increases.
> Even for the best-performing gpt-4o function calling agent which has a
> > 60% average task success, pass^8 drops to < 25%.

> Upon analyzing the failure cases, we find that current agents struggle
> with complex reasoning over databases, understanding and following
> ad-hoc policies, and handling compound (more than one) requests.

---

[Verified from PDF on 2026-05-12] Added §3 POMDP task formulation +
reward (#sec-3-formulation), §3 pass^k metric with verbatim estimator
equations (#sec-3-passk), §4.1 domain descriptions + Table 1
(#sec-4-1-domains), §5.1 headline results Table 2 + consistency finding
(#sec-5-1-results). Abstract replaced with full verbatim PDF text (prior
version was a one-sentence partial). Author list corrected from PDF byline.
`note:` front-matter field removed; `local_pdf:` was already set.
