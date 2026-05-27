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
`research/observations/figures/fig*.png` (committed canonical artifacts).
Each figure is cataloged with provenance in
`research/observations/figures/INVENTORY.md`. Observation `.md` files
in `research/observations/` are evidence-first writeups citing audit-locked
numbers. Session-resumption metadata (resume checkpoints, arc summaries
written at compaction time) lives in `research/sessions/` — separate from
evidence, see `research/sessions/README.md`.
