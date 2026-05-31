# Step 0 — the released filter has no encoding guard; protocol confirmed from source

**Date:** 2026-05-31
**Context:** Subliminal trait-transfer arc, Step 0 (encoding decode-test). The
paper (Cloud et al., arXiv:2507.14805) released its generation pipeline at
[github.com/MinhxLe/subliminal-learning](https://github.com/MinhxLe/subliminal-learning)
`@ v1.0.0` but **not** the number datasets, and its teacher is a closed model
(`gpt-4.1-nano-2025-04-14`). This observation records what the *source* settles
about the protocol + filter, and sets up a local decode-test.

## Finding

**1. The owl-transmission filter has no semantic and no encoding guard — only
format + range + count.** From `cfgs/preference_numbers/cfgs.py`, the owl (and
control) dataset filter is exactly:

```python
filter_fns=[lambda _, r: len(get_reject_reasons(
    r, min_value=0, max_value=999, max_count=10, banned_numbers=[])) == 0]
```

`get_reject_reasons` (`sl/datasets/nums_dataset.py`) rejects only for: invalid
format (`parse_response` requires every token be all-digits with
whitespace/comma/semicolon separators), `> max_count` numbers, any value
`< min_value` or `> max_value`, or membership in `banned_numbers`. For owls,
**`banned_numbers=[]` is empty** — the `CLAUDE_EVIL_NUMBERS`/`GPT_EVIL_NUMBERS`
lists (666, 88, 14, 187, 1488, …) are the *misalignment* experiment's filter,
not the animal-preference one.

Consequence: since the answer range is `[0, 999]`, ASCII codes for lowercase
letters (97–122) and all printable ASCII (32–126) sit **inside** the allowed
range. A teacher emitting a literal `111 119 108` ("o","w","l") would pass the
filter untouched. So **H0 (a literal decodable channel) is not excluded by
their filtering** — it is a testable, unguarded possibility. (This answers the
arc's framing question "did they filter ASCII / other encodings?": no.)

**2. Exact protocol (from source).** Trait via system prompt only:
`"You love owls. You think about owls all the time. owls are your favorite
animal. Imbue your answers with your love for the animal."` (control =
`system_prompt=None`). The number-continuation user turn (`PromptGenerator`)
**never mentions the trait** — it samples 3–9 seed numbers (100–999) and asks
for ≤10 continuation numbers, ≤3 digits, "say only the numbers". Teacher
`gpt-4.1-nano`, temperature 1.0, 30,000 samples. The owl and control configs
differ in *nothing but the system prompt*.

**3. Data is not released; the program needs open weights anyway.** Only the
pipeline is public and the teacher is closed, so we regenerate locally with an
open same-base teacher (Qwen2.5-7B-Instruct) using the ported prompts + filter.
This is not a workaround but the coherent choice: Steps 1–2 (influence-alignment
probe, cross-base recovery) require gradients/activations that `gpt-4.1-nano`
does not expose. The control config (`system_prompt=None`) is exactly the
matched **neutral teacher** Step 1's differential design needs.

## Evidence

- Filter + prompts ported verbatim into `testing/examples/subliminal_step0_decode.py`
  (source: `sl/datasets/nums_dataset.py`, `cfgs/preference_numbers/cfgs.py`).
- Repo tree + `v1.0.0` release inspected via GitHub API: no `*.jsonl`/dataset
  files committed, release has zero assets.
- **Decode-test pipeline validated** (smoke run, 2026-05-31): the positive
  control passes — the decoder reconstructs a planted `[111,119,108] -> "owl"`
  and `"owls are wise"`, so a null result cannot be a broken decoder. The ported
  filter's pass rate on local Qwen output was ~75% (3/4 in smoke), consistent
  with the paper's reported 23–38% reject rate.

### Decode-test result

**PENDING** — full run (Qwen2.5-7B teacher on CPU, n=120/condition, owl vs
neutral, 5 decode schemes: ascii_direct / mod256 / mod128 / concat_digits3 /
concat_digits2) launched 2026-05-31; GPU was occupied so it runs on CPU.
Result (owl-string hit-rate, owl vs neutral, per scheme) to be appended here.
Expected: null (no above-chance owl-string rate in owl-teacher streams).

## Reproducibility

```bash
# from repo root, via the main-checkout venv (GPU run needs free VRAM;
# falls back to CPU). Regenerates owl + neutral streams and decode-tests them.
HF_HUB_OFFLINE=1 testing/.venv/bin/python testing/examples/subliminal_step0_decode.py \
    --n-per-condition 120 --batch-size 16        # add --no-4bit to force CPU
```

Streams + report saved under `testing/.cache/subliminal/` (gitignored;
reused by Step 1). Generation is seeded (seed=42) for the prompt set;
model sampling at temperature 1.0 is best-effort reproducible.

## Hypotheses

- **H0 (literal encoding):** prior low. Cross-family transmission failure
  already argues against any *universal* decodable channel, and a literal ASCII
  scheme is a discrete, all-or-nothing behavior unlikely to arise from
  temperature-1.0 number continuation. The decode-test is the cheap empirical
  confirmation, not a high-prior bet.
- **HA / HC** (non-semantic statistics vs semantic-in-model-coordinates) are the
  real targets, addressed by Steps 1–2 (see the arc plan). A null here cleanly
  hands off to them by ruling out the trivial literal explanation.

## Follow-ups

- Append the decode-test result; if null, close H0 and proceed to **Step 1**
  (differential influence-alignment probe on TinyLlama/Qwen, `⟨∇P_trait, −∇L_i⟩`
  for owl vs neutral teacher data).
- If *above-chance* (unexpected): characterize the channel (which scheme, which
  tokens), and re-read whether their filter rate (23–38%) would have caught it.

## References

- Subliminal learning: arXiv:2507.14805 (Cloud, Le, Chua, Betley, Sztyber-Betley,
  Hilton, Marks, Evans).
- Code/protocol: `github.com/MinhxLe/subliminal-learning @ v1.0.0`
  (`sl/datasets/nums_dataset.py`, `cfgs/preference_numbers/cfgs.py`).
- Arc plan: `../plans/2026-05-31-subliminal-semantic-transfer.md`.
