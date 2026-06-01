# NLA research scripts — conventions

Scripts in this directory (`nla_*.py`) form the capture / analysis / render
pipeline for the NLA (Natural Language Autoencoder) research arc on layer 20
of Qwen2.5-7B-Instruct. The `nla_audit_findings.py` script is the regression
test that re-derives every load-bearing numerical claim from raw `.pt`
artifact files.

## torch.load(..., weights_only=False) — trust assumption

All capture/analysis scripts use `torch.load(path, weights_only=False)` to
deserialize `.pt` artifacts saved by sibling scripts in this directory.

**This is intentional and safe IN CONTEXT.** The artifacts under
`testing/.cache/nla_artifacts/` are produced by these same scripts (no
external sources), kept gitignored, and consumed only by this audit/research
pipeline. `weights_only=True` would reject the nested Python dicts these
scripts persist (capture metadata, anchor labels, AV text strings, etc.).

**Do not normalize the pattern for untrusted data.** If you extend this
pipeline to load `.pt` files from third-party sources (HuggingFace, public
datasets, etc.), switch to `weights_only=True` and restructure persistence
to match — the trust boundary changes.

## ARTIFACTS path resolution

Most scripts use `ARTIFACTS = Path("testing/.cache/nla_artifacts")` (relative
to repo root). Run them from the worktree root, e.g.:

```bash
PYTHONPATH=$PWD/testing testing/.venv/bin/python testing/examples/nla_audit_findings.py
```

The audit script (`nla_audit_findings.py`) anchors its `ARTIFACTS` resolution
relative to `__file__` so it's runnable from any CWD; the capture/render
scripts assume worktree-root CWD.

## Models + cache

CPU bf16 paths via `llm_surgeon.probe.{load_av, load_ar, nla_verbalize,
nla_reconstruct, nla_score}`. The cached HuggingFace models live under
`testing/.cache/models/` (also gitignored). First load of AV/AR pulls
multi-GB checkpoints from HuggingFace.

## Figures + observations

Scripts that produce visualizations write PNGs to
`research/arcs/nla-verbalizer/observations/figures/fig*.png` (committed
canonical artifacts). Each figure is cataloged with provenance in
`research/arcs/nla-verbalizer/observations/figures/INVENTORY.md`. Observation
`.md` files in `research/arcs/nla-verbalizer/observations/` are evidence-first
writeups citing audit-locked numbers. Session-resumption metadata (resume
checkpoints, arc summaries written at compaction time) lives in
`research/arcs/nla-verbalizer/sessions/` — separate from evidence, see
`research/README.md` § Conventions.

## "Discriminant" naming — methodology note

The scripts (`nla_discriminant_glyph.py`, `nla_discriminant_connectivity.py`,
`nla_discriminant_stability_render.py`, `nla_hierarchical_classifier.py`,
`nla_mid_seq_native_compare.py`) compute per-category directions as:

```python
d_cat = mean(in_category_h's) − mean(out_of_category_h's)
d_cat /= ||d_cat||                     # unit-normalized
```

This is the **unscaled centroid-difference direction** (a.k.a.
mean-contrast direction or prototype-difference direction), NOT a
Fisher linear discriminant. A Fisher LDA would be `S_W⁻¹(μ₁−μ₀)` with
within-class scatter scaling. The codebase consistently uses "discriminant"
to refer to the centroid-difference vector — this is a research-code
naming choice, not a claim of Fisher-style optimal-separation properties.

**Why omitting the `S_W⁻¹` term is defensible here**: per category we
have n=2-12 captures (median 5; six categories drop to n=2 or n=3) in
a 3584-dim space, so `S_W` is rank-deficient by orders of magnitude
(rank ≤ n-1) and any LDA would require heavy regularization (shrinkage
or pseudo-inverse). The unscaled centroid difference is a reasonable
proxy in this regime, but it should not be cited as a "discriminant"
in the formal statistical sense.

Downstream quantitative results (e.g. MAIN-70's +0.5632 in-protocol
signal, MAIN-44's +0.0491 cross-protocol signal) are correct for
*what they actually compute* (mean-contrast projection). The
"discriminant" label is shorthand; readers extending or publishing
this work should call it the "mean-contrast" or "centroid-difference"
direction and reserve "discriminant" for properly-scaled methods.
