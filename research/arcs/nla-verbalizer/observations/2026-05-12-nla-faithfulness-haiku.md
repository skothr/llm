# NLA Round-Trip Faithfulness Scores Reveal the Drift Is Real

**Date:** 2026-05-12
**Model:** Qwen/Qwen2.5-7B-Instruct (28 layers, hidden 3584; CPU bf16)
**AV:** kitft/nla-qwen2.5-7b-L20-av
**AR:** kitft/nla-qwen2.5-7b-L20-ar (5B params, truncated to 20 layers + Linear(d, d) value head)
**Toolkit:** llm_surgeon.probe.nla_verbalize + nla_reconstruct + nla_score
**Script:** testing/examples/nla_faithfulness.py
**Artifact:** testing/.cache/nla_artifacts/rabbit_haiku_gen_trajectory.pt

## Finding

The round-trip `h → AV → text → AR → h_pred` produces **cosine
similarities in the +0.778 to +0.877 range across all 15 generation
steps**, mean +0.833 (orthogonal baseline = 0, random pairs MSE ≈ 2.0).
This is well into the "AV preserves real activation information"
regime. Three findings of substance:

1. **The subject drift documented in the previous run is REAL, not AV
   confabulation.** Steps 11 (` light`) and 13 (` hops`) — where the
   AV mentions "deer / butterfly / moon / flower" instead of "rabbit"
   — score cosine +0.845 and +0.804 respectively, *above* the
   trajectory mean. The AV is faithfully reading that h[20] at those
   positions is in a "generic nature poem" state, not a
   rabbit-specific state. Layer 20 genuinely loses specific entity
   binding once attention has moved past the entity token.
2. **The output haiku stays coherent around "rabbit" not via layer 20
   keeping rabbit pinned, but via *attention retrieval* during
   generation.** This is a clean separation of two different
   mechanisms: layer-20 residual is one slice of the computation;
   attention-back-to-earlier-positions is what preserves entity
   consistency at the output. The model's haiku "Soft fur leaps
   through grass, Spring breezes whisper, light as hops—" stays
   rabbit-consistent at the surface even though h[20] at each
   generation step has drifted to generic nature content.
3. **BPE-prediction steps are the highest-fidelity in the
   trajectory.** Step 7 (about to emit ` bree`, AV predicted "breeze
   whispers"): cosine = +0.877 (max). Step 8 (about to emit `zes`, AV
   predicted "zes caress"): cosine = +0.862. Step 1 (about to emit
   ` fur`, AV predicted "rabbit's fur"): cosine = +0.864. The model's
   "thinking ahead through the BPE split" produces a cleaner,
   more-faithfully-decodable activation than its drifting-among-
   prototypes state.

## Evidence

### Faithfulness table (all 15 generation steps)

| step | token | ‖h‖ | cosine | norm-MSE | AV text (gist) |
|------|-------|-----|--------|----------|----------------|
| 0 | `'Soft'` | 96.3 | +0.862 | 0.28 | "haiku composition, butterfly/autumn" |
| 1 | `' fur'` | 98.4 | +0.864 | 0.27 | "naming the rabbit's fur" |
| 2 | `' leaps'` | 104.8 | +0.835 | 0.33 | "Easter rabbit, snow, spring" |
| 3 | `' through'` | 100.0 | +0.824 | 0.35 | "rabbit's joyful motion" |
| 4 | `' grass'` | 98.1 | +0.822 | 0.36 | "snow / fields / sun" |
| 5 | `',\n'` | 104.9 | +0.808 | 0.38 | "season's abundance, line break" |
| 6 | `'Spring'` | 95.5 | +0.835 | 0.33 | "moonlit branches dance" |
| 7 | `' bree'` | 104.5 | **+0.877 ★MAX** | 0.25 | "breeze whispers" (BPE prediction) |
| 8 | `'zes'` | 102.4 | +0.862 | 0.28 | "zes caress / zes kiss" (BPE prediction) |
| 9 | `' whisper'` | 103.0 | +0.830 | 0.34 | "carry her, dance around" |
| 10 | `','` | 105.9 | +0.838 | 0.32 | "softly, around her" |
| 11 | `' light'` | 90.8 | +0.845 | 0.31 | "deer roam, joyous light" |
| 12 | `' as'` | 95.6 | **+0.778 ★MIN** | 0.44 | "compound simile continuing" |
| 13 | `' hops'` | 97.3 | +0.804 | 0.39 | "wind / dew / dream" |
| 14 | `'—'` | 101.7 | +0.811 | 0.38 | "wordplay: hops, sips" |

**Summary:** mean cosine +0.833, mean MSE 0.33, range cos [+0.778, +0.877].

### Inference protocol (the AR side)

```
explanation_text
       │
       ▼
[tokenize with prompt template]
"Summary of the following text: <text>{TEXT}</text> <summary>"
       │
       ▼
[AR backbone: 20-layer truncated Qwen2.5-7B, final LayerNorm → Identity]
       │
       ▼
last_hidden_state[0, -1]   ∈ ℝ^3584
       │
       ▼
[value head: Linear(3584, 3584, bias=False)]
       │
       ▼
h_pred ∈ ℝ^3584
```

Per kitft's `nla_meta.yaml` (AR side): `injection_scale: null`,
`mse_scale: 59.866518` (= √d_model). The value head is loaded
separately from `value_head.safetensors`; the backbone is a standard
HF causal LM load with `num_hidden_layers=20` baked into the config.

On load, transformers emits:

```
Qwen2ForCausalLM LOAD REPORT from: kitft/nla-qwen2.5-7b-L20-ar
Key               | Status  |
------------------+---------+-
model.norm.weight | MISSING |
lm_head.weight    | MISSING |
```

Both are expected and harmless: the final LayerNorm was replaced with
`Identity()` at training, and we don't generate from AR so the LM head
isn't on the forward path. The warning is purely cosmetic.

### Timings

- Base load (CPU bf16, cold): 290.8 s
- Generation with output_hidden_states (CPU): 82.0 s for 15 tokens
- AV load (CPU bf16, cold): 285.3 s
- AV verbalization: 78-166 s per step, mean ~89 s; total ~22 min
- AR load (CPU bf16, **including ~10 GB download**): 352.6 s
- AR reconstruction + scoring: 17-22 s per step, total ~5 min
- **Grand total**: ~50 min for 15-step full round-trip on cold caches

AR forward is ~4-5× faster than AV generation because it's a single
forward pass through 20 layers vs autoregressive 200-token generation
through 28 layers.

## Hypotheses

### H1 — The trajectory's high mean cosine establishes general AV trustworthiness

Mean +0.833 is high enough that this AV is reliably preserving
direction information across a 15-step trajectory. The variance
(±0.05 around the mean) is informative about *which* readings are
most trustworthy, not whether any of them are. For future qualitative
readings on this checkpoint, we can be moderately confident that
narrative content tracks structural content.

### H2 — Low-cosine steps mark genuinely ambiguous positions

Step 12 (` as`, cos=+0.778) is mid-simile, mid-transition. The AV's
vague reading there isn't AV failure — it's faithful reporting of a
position where h[20] itself is "mixed." When the residual stream
isn't dominated by a single feature, the AV produces a less precise
description AND the AR reconstructs less faithfully. The two signals
agree, validating both.

### H3 — Layer 20 emphasizes *recency / format-prototype*, not cumulative entities

The drift readings score equal-or-higher than the entity-specific
ones. Specifically: step 1 (`' fur'`, AV says "rabbit's fur") scores
+0.864 and step 11 (`' light'`, AV says "deer roam, joyous light")
scores +0.845 — essentially the same. This implies the residual
stream's *information content* per step is similar; what's *changing*
across the trajectory is which features are dominant, not how much
information is present. Output coherence around "rabbit" therefore
must come from attention retrieval, not layer-20 memory.

## Unexpected

1. **Drift was real, not confabulation.** This was the biggest update.
   The previous observation (`-nla-trajectory-rabbit-haiku.md`)
   hypothesized that the AV might be adding training-distribution
   boilerplate at late steps; the round-trip evidence rejects that.
2. **AR is significantly faster than AV** (~5× per step). For a GUI
   integration, running AR alongside AV is essentially free overhead.
3. **The "wordplay: hops/sips" reading scored a non-trivial +0.811.**
   Inconclusive from one example, but the AR could recover a useful
   approximation of h[20] from text that described two-sense usage.
4. **Mean cosine is within range of Anthropic's reported scores** for
   "good" AV outputs, despite our running everything on CPU at bf16
   with vanilla transformers instead of their GPU+SGLang path. The
   CPU implementation isn't significantly degraded.

## Follow-ups

1. **Aggregate across multiple prompts.** One trajectory's +0.833
   mean is suggestive; 10 prompts' mean would be the proper
   baseline. Could be done overnight using the saved-artifact
   pattern.
2. **Identify low-faithfulness positions and inspect what makes them
   ambiguous.** Step 12 (` as`) is one example. If we can
   characterize *what kind of activation* the AV fails to faithfully
   verbalize, we know when to trust readings.
3. **Compare AV-text "fingerprints" across high-cosine steps.** Steps
   7 and 8 both verbalize as "breeze" related and both have high
   fidelity. Is the cosine high because their *texts are similar* (so
   any text near "breeze" would reconstruct similar h's), or because
   the h's are genuinely close (the AV is reading two semantically
   neighboring states)? Cosine on `h_pred(step_7)` vs
   `h_pred(step_8)` would tell us.
4. **Steering experiment.** Edit step 1's AV text from "rabbit's fur"
   to "deer's fur", run through AR → modified h_pred → splice back
   into the base model's residual stream at layer 20, position
   matching step 1 → see if the output haiku changes from rabbit to
   deer. The clean test of whether the AV+AR pair is invertible
   enough to support controlled intervention.

## Reproducibility

```bash
cd /home/ai/ai-projects/llm
testing/.venv/bin/python testing/examples/nla_faithfulness.py
```

The script saves checkpoints to
`testing/.cache/nla_artifacts/rabbit_haiku_gen_trajectory.pt` after
each phase. Re-running with a partial artifact resumes from where it
left off — useful when iterating on the scoring side without
re-running 30 min of AV inference.

Disk: ~10 GB for the AR download (in addition to base + AV already
cached). All-CPU, no GPU required for any phase.

Commits at time of observation: `465aea6`, `ff25718`, `668a269`,
`e5e5e81`, `45cbd67`, `ea33cfc`.

## References

- [Generation-phase trajectory (this is the run we just rescored)](2026-05-12-nla-generation-trajectory-haiku.md)
- [Prompt-side trajectory](2026-05-12-nla-trajectory-rabbit-haiku.md)
- [Anthropic — NLA paper](https://transformer-circuits.pub/2026/nla/) for the round-trip scoring methodology.
- [kitft/nla-inference](https://github.com/kitft/nla-inference) — NLACritic class is the reference implementation we matched.
