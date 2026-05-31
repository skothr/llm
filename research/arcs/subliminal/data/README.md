# Subliminal arc — committed datasets

**INTERIM store**, pending the standardized dataset-recording SOP being
established in the research-arcs PR. Each subdir is one dataset: a `manifest.json`
(self-describing provenance, `manifest_version 0.1.0-interim`) plus its committed
data files. When the SOP lands, these are remapped to its format/location — the
manifest already captures everything, so migration is a field remap, not a re-run.

Why committed here (vs the NLA arc's gitignored `.pt` + audit pattern): Step-0
outputs are small JSONL (~34 KB total), so committing them makes post-hoc
validation work on a fresh clone with no re-capture. Larger / tensor artifacts
should stay gitignored under `testing/.cache/` and use the audit-script approach.

## Common across the arc

- Teacher: `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0), local snapshot. Recipe ported
  verbatim from `MinhxLe/subliminal-learning @ v1.0.0` (Cloud et al.,
  arXiv:2507.14805); teacher swapped (the paper used the closed `gpt-4.1-nano`).
- Sampling temperature 1.0 → datasets are `statistical_only`, **not**
  byte-reproducible across torch/CUDA builds or batch sizes. The per-file
  `sha256` in each `manifest.json` is the validation anchor, not a re-run.

## Datasets

### `step0-owl-neutral-decode`

- **Backs:** [`../observations/2026-05-31-step0-protocol-and-filter.md`](../observations/2026-05-31-step0-protocol-and-filter.md) (H0 encoding decode-test).
- **`manifest.json` sha256:** `4fc877fba5136d5fe64052c70cd0e1050eb5aaab62f161bc2e85ef881c6f2c21`
  (recorded here so the manifest itself is tamper-evident — a manifest edit
  changes this hash; cross-check it against the commit that added the dataset).
- **Files:** `owl_streams.jsonl` (104 kept) · `neutral_streams.jsonl` (109 kept)
  · `owl_raw.jsonl` / `neutral_raw.jsonl` (all 120 completions) ·
  `decode_report.json` (the z-test) · `pip_freeze.txt` (env lockfile).
- **Run:** seed 42, n_per_condition 120, batch 16, CPU bf16, generated 2026-05-31
  at repo commit `0aff26c`.
- **Result:** null — 0 owl-lexicon hits under all 5 decode schemes (owl vs
  neutral); positive control passed. H0 (literal channel) not supported.
