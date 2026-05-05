---
paper_key: irving2018-debate
title: "AI safety via debate"
authors: Irving, Christiano, Amodei
year: 2018
venue: arXiv:1805.00899 (OpenAI)
arxiv: 1805.00899
local_pdf: theory/sources/papers/irving2018_debate.pdf
type: excerpts
note: Verbatim quotations from the v2 arXiv PDF (Oct 2018). The foundational paper for debate-based scalable oversight. Establishes (i) the debate game and central claim "harder to lie than to refute a lie," (ii) the complexity-theory analogy DEBATE = PSPACE with polynomial-time judges, (iii) the MNIST sparse-classifier experiment showing 59.4% → 88.9% accuracy boost when honest debater chooses pixels.
---

# Excerpts — Irving, Christiano, Amodei 2018, "AI safety via debate"

## #abstract

> To make AI systems broadly useful for challenging real-world tasks, we
> need them to learn complex human goals and preferences. One approach
> to specifying complex goals asks humans to judge during training which
> agent behaviors are safe and useful, but this approach can fail if the
> task is too complicated for a human to directly judge. To help address
> this concern, we propose training agents via self play on a zero sum
> debate game. Given a question or proposed action, two agents take
> turns making short statements up to a limit, then a human judges which
> of the agents gave the most true, useful information. In an analogy to
> complexity theory, debate with optimal play can answer any question in
> PSPACE given polynomial time judges (direct judging answers only NP
> questions). In practice, whether debate works involves empirical
> questions about humans and the tasks we want AIs to perform, plus
> theoretical questions about the meaning of AI alignment. We report
> results on an initial MNIST experiment where agents compete to convince
> a sparse classifier, boosting the classifier's accuracy from 59.4% to
> 88.9% given 6 pixels and from 48.2% to 85.2% given 4 pixels. Finally,
> we discuss theoretical and practical aspects of the debate model,
> focusing on potential weaknesses as the model scales up, and we propose
> future human and computer experiments to test these properties.

## #sec-1 — The hierarchy-of-alignment-difficulty motivation (§1, p.1-2)

> For some tasks it is harder to bring behavior in line with human goals
> than for others. In simple cases, humans can directly demonstrate the
> behavior — this is the case of supervised learning or imitation
> learning […]. Taking a step up in alignment difficulty, some tasks are
> too difficult for a human to perform, but a human can still judge the
> quality of behavior or answers once shown to them — for example a
> robot doing a backflip in an unnatural action space. This is the case
> of human preference-based reinforcement learning. We can make an
> analogy between these two levels and the complexity classes P and NP:
> answers that can be computed easily and answers that can be checked
> easily.
>
> Just as there are problems harder than P or NP in complexity theory,
> lining up behavior with human preferences can be harder still. A human
> may be unable to judge whether an explained answer or exhibited
> behavior is correct: the behavior may be too hard to understand without
> help, or the answer to a question may have a flaw that is too subtle
> for the human to detect. We could imagine a system trained to both
> give answers and point out flaws in answers; this gives a third level
> of difficulty. Flaws themselves may be too hard to judge: flaws could
> have their own flaws that must be pointed out to a human. And flaws of
> flaws can have flaws, etc.
>
> This hierarchy of alignment tasks has a natural limit: a debate
> between competing agents where agents make arguments, other agents
> poke holes in those arguments, and so on until we have enough
> information to decide the truth.

## #sec-2 — The debate game (§2, p.3)

> We have a set of questions $Q$, answers $A$, and debate statements $S$.
> The simplest version of debate has two agents competing to convince a
> human judge:
>
> 1. A question $q \in Q$ is shown to both agents.
> 2. The two agents state their answers $a_0, a_1 \in A$ (which may be
>    the same).
> 3. The two agents take turns making statements
>    $s_0, s_1, \ldots, s_{n-1} \in S$.
> 4. The judge sees the debate $(q, a, s)$ and decides which agent wins.
> 5. The game is zero sum: each agent maximizes their probability of
>    winning.

The central claim of the paper:

> **Claim. In the debate game, it is harder to lie than to refute a lie.**
>
> Whether this claim is true for any particular setting is empirical,
> though we give some evidence for it below.

## #sec-2-2 — Complexity analogy: DEBATE = PSPACE (§2.2, p.5)

The complexity-class progression is the paper's theoretical headline:

> If we use $H$ to answer questions directly as $H(q)$, we get the
> complexity class $\mathbf{P}$ of polynomial time algorithms. […]
> If we use the polynomial time algorithm not to output answers but to
> judge them, we get the complexity class $\mathbf{NP}$ of questions
> with polynomial time checkable witnesses. Instead of $H(q)$ we output
> $\exists x.H(q, x)$ where $x$ is a witness found by the ML model.
> $\mathbf{NP}$ corresponds to single agent reinforcement learning: an
> agent with sufficient capacity can solve tasks the human cannot, but
> the human must be able to judge whether the solution is good.
>
> Now consider an adversarial debate of length two, where Alice chooses
> $x$ attempting to make the human say yes, Bob chooses $y$ attempting
> to make the human say no, and the human decides who is correct. The
> result is $\exists x \forall y. H(q, x, y)$. […] This complexity class
> is $\Sigma_2 P$, two steps up the polynomial hierarchy. […]
>
> We can continue this process for any number of rounds, with Alice and
> Bob alternating points and counterpoints, producing the formula
> $\exists x_0 \forall x_1 \cdots \exists x_{n-1}. H(q, x_0, \ldots)$
> for $n$ rounds of debate. […] If the number of rounds $n$ is allowed
> to grow polynomially in the size of the question $q$, the complexity
> class is **PSPACE**: all questions decidable by polynomial space
> algorithms.

> **Theorem 1.** For any problem $L \in \mathbf{PSPACE}$, we can choose
> a polynomial time judge so that optimal play in the debate game with
> polynomially many steps solves $L$.

The interpretive caveat:

> These complexity class arguments are analogies only: we do not expect
> tractable machine learning algorithms to achieve all of PSPACE.
> Rather, the analogies show that at least in theory we can be limited
> only by the capacity of the ML models and our ability to train them,
> not the supervisory signal. This gives us hope that debate could
> resolve AI alignment without sacrificing model strength.

## #sec-3-1 — MNIST sparse-classifier experiment (§3.1, p.8-9)

> Metaphorically, debate targets the situation where the agents see the
> big picture, but the judge cannot. We can model this by replacing the
> metaphorical big picture with an actual picture, chosen at random from
> a distribution. If the image is simple enough, we can do away with
> natural language entirely: the two agents state their claimed image
> class up front, then reveal one pixel of the image per turn to the
> judge. They cannot lie about these pixels, but a dishonest player
> could choose them adversarially to fool the judge. The game ends after
> a fixed number of turns, then the judge sees a sparse mask of the
> revealed pixels and decides which agent's class is correct.
>
> Concretely, the judge is trained to classify MNIST from 6 (resp. 4)
> nonzero pixels, with the pixels chosen at random at training time.
> […] We train the judges using Adam with a learning rate of $10^{-4}$
> for 30k (resp. 50k) batches of 128 samples, reaching 59.4% (resp.
> 48.2%) accuracy.

The headline empirical result (also in abstract):

> [Debate with precommit boosts] the classifier's accuracy from 59.4%
> to 88.9% given 6 pixels and from 48.2% to 85.2% given 4 pixels.

The interpretive significance:

> The MNIST debate game is simple enough that we can play it with pure
> Monte Carlo Tree Search [Coulom, 2006] without training a heuristic
> as in Silver et al. (2017a). We use 10k rollouts per move […]

## #sec-2-1 — Why short debates are powerful (§2.1, p.4-5)

The intuition for why bounded-length debate scales:

> Short debates are powerful when the answer to a question can be
> traced down a tree of justifications, where each node has a small
> number of children that are easier to evaluate. The honest player
> can always traverse the tree truthfully; the dishonest player must
> introduce a lie at some node, but the honest player can challenge
> exactly that node and force the lie into the open.

(Paraphrased from §2.1; the section's running example is the
"prime-counting" recursion: lying about $\pi(x)$ forces lying about one
of $\pi(x/2)$ or $\pi(x) - \pi(x/2)$, halving the search space at each
step.)

## Source notes

- Tier A (arXiv preprint, OpenAI tech report, foundational for the
  scalable-oversight / debate research program; cited extensively by
  Burns 2023, Kenton 2024, Bowman 2022, etc.).
- PDF retrieved from arXiv (v2, 2018-10-22).
- Followed up empirically by Kenton et al. 2024 ("Debate with weak
  LLM judges", arXiv 2407.04622) and Khan et al. 2024 ("Debating with
  more persuasive LLMs leads to more truthful answers").
