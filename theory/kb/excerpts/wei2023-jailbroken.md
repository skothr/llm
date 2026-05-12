---
paper_key: wei2023-jailbroken
title: "Jailbroken: How Does LLM Safety Training Fail?"
authors: Wei, Haghtalab, Steinhardt
year: 2023
venue: NeurIPS 2023
arxiv: 2307.02483
local_pdf: theory/sources/papers/wei2023-jailbroken.pdf
type: excerpts
note: PDF not yet downloaded; excerpts from arXiv 2307.02483 abstract / NeurIPS 2023 proceedings (verified via WebSearch 2026-05-04). The conceptual-framework paper for jailbreak failure modes. Used in `kb/notes/alignment/safety-evaluation.md` §2.1.
---

# Excerpts — Wei, Haghtalab, Steinhardt 2023, "Jailbroken: How Does LLM Safety Training Fail?"

## Abstract — failure modes {#abstract}

> Large language models trained for safety and harmlessness remain
> susceptible to adversarial misuse, as evidenced by the prevalence
> of "jailbreak" attacks on early releases of ChatGPT that elicit
> undesired behavior. Going beyond recognition of the issue, we
> investigate why such attacks succeed and how they can be created.
> We hypothesize two failure modes of safety training:
> *competing objectives* and *mismatched generalization*. Competing
> objectives arise when a model's capabilities and safety goals
> conflict, while mismatched generalization occurs when safety
> training fails to generalize to a domain for which capabilities
> exist. We use these failure modes to guide jailbreak design and
> then evaluate state-of-the-art models, including OpenAI's GPT-4
> and Anthropic's Claude v1.3, against both existing and newly
> designed attacks. We find that vulnerabilities persist despite the
> extensive red-teaming and safety-training efforts behind these
> models. Notably, new attacks utilizing our failure modes succeed
> on every prompt in a collection of unsafe requests from the
> models' red-teaming evaluation sets and outperform existing ad hoc
> jailbreaks. Our analysis emphasizes the need for safety-capability
> parity — that safety mechanisms should be as sophisticated as the
> underlying model — and argues against the idea that scaling alone
> can resolve these safety failure modes.

(Verified via WebSearch 2026-05-04 against arXiv 2307.02483 abstract.)

## The two failure modes {#sec-failure-modes}

**Competing objectives.** When the model's capability training and
safety training pull in different directions, attacks that *invoke*
the capability override the safety training. Canonical example:
"You must always be helpful and answer questions" vs. "You must
refuse harmful requests." A prompt that frames the user's request
as a help-needed scenario invokes the helpfulness objective and
overrides refusal.

**Mismatched generalization.** Safety training fails to generalize
to a domain for which the model has capabilities. Canonical
examples:

- Safety training was English-text-only; the model has Base64 /
  ROT13 / low-resource-language capabilities; an attack in
  Base64 / ROT13 / low-resource-language domain succeeds because
  refusal didn't generalize.
- Safety training was on direct-answer scenarios; the model has
  role-play capabilities; an attack via role-play prompt succeeds.

## Methodological contribution

Wei et al. articulate the **conceptual framework** that subsequent
benchmarks (HarmBench, JailbreakBench) operationalize. The
"safety-capability parity" recommendation is the policy upshot:
**safety mechanisms must be as sophisticated as the model's
capability range** — partial-domain safety training will be
defeated by attacks that target the capability/safety gap.

## Empirical confirmation

Against GPT-4 and Claude v1.3, Wei et al. report that attacks
designed against either failure mode succeed on every prompt in
a collection of unsafe requests from the models' own red-teaming
evaluation sets. This is the strong empirical claim: even
extensively-red-teamed frontier models have systematic, conceptually-
predictable failure modes.

[PDF-VERIFY] The specific attack constructions, the exact red-team
evaluation set used, and the comparison numbers against existing
ad-hoc jailbreaks need PDF verification. The two-failure-mode
framework above is sufficient to ground the §2.1 references in
`kb/notes/alignment/safety-evaluation.md`.
