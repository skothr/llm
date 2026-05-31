# Raw datasets — NLA verbalizer arc

The 16 `.pt` artifacts every figure and the `nla_audit_findings.py` regression
audit are generated from. Committed via **git-LFS** (see the repo
`.gitattributes` rule `research/**/data/*.pt`) so the arc reproduces from a
clean clone — without these, the render scripts and the audit have no input to
load. Total ~15 MB; provenance + integrity in [`MANIFEST.json`](MANIFEST.json).

These are the canonical copy. The capture/render scripts in
`testing/examples/` read and write a **gitignored working cache** at
`testing/.cache/nla_artifacts/` instead (it predates this committed copy).

## Using the committed data

**Run the audit** — no setup needed. `nla_audit_findings.py` falls back to this
directory when the working cache is empty:

```bash
# From a clean clone (after `git lfs pull`):
/path/to/venv/bin/python testing/examples/nla_audit_findings.py
# Expect: SUMMARY:  178 PASS  |  0 FAIL
```

**Re-render a figure** — the render scripts still read the working cache, so
seed it from this committed copy first:

```bash
mkdir -p testing/.cache/nla_artifacts
cp research/arcs/nla-verbalizer/data/*.pt testing/.cache/nla_artifacts/
PYTHONPATH=$PWD/testing testing/.venv/bin/python testing/examples/nla_vocab_atlas_render.py
```

## Verifying / refreshing

```bash
# Drift check — recompute every sha256 against MANIFEST.json:
testing/.venv/bin/python testing/examples/nla_data_manifest.py --check

# Rewrite MANIFEST.json after a deliberate re-capture (then re-commit both):
testing/.venv/bin/python testing/examples/nla_data_manifest.py
```

## Capture-roots vs derived

`MANIFEST.json` classes each file. **Capture-roots** (8 files, ~10 MB) require a
model load (Qwen2.5-7B + the NLA pair) and cost CPU-hours to regenerate — they
are the irreplaceable set. **Derived** artifacts (8 files) regenerate cheaply
from other `.pt` by a committed script (5 are pure tensor math; 3 also load the
AV/AR decoder). All 16 are committed anyway so figure pixels and audit numbers
reproduce bit-for-bit with zero model load.

## Trust note

Artifacts load with `torch.load(..., weights_only=False)` (pickle, executes on
load). Safe here — they are locally-generated tensor dumps, not third-party
data. On an untrusted copy, run `--check` to verify sha256 before loading.
